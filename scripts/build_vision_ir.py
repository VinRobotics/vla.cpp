#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a model's vision encoder as a native OpenVINO IR straight from its GGUF,
for scripts/npu_vision_worker.py (one camera view on the Intel NPU).

    python3 scripts/build_vision_ir.py smolvla-libero.gguf smolvla_vision.xml

Same arithmetic as the ggml graph (src/models/smolvla.cpp): conv patch embed +
bias + positions, pre-LN SigLIP blocks (tanh GELU), post-LN, the pixel shuffle
and the connector. Input: one preprocessed CHW image [1, 3, S, S] (the floats
preprocess_image_chw produces); output: [n_img_tokens, hidden] f32, the rows the
model splices into its prefix. Weights are kept f16 (the NPU computes in f16).

The ggml exporter (GGML_OPENVINO_DUMP_IR) produces an equivalent graph, but
the NPU compiler is ~10x slower on SmolVLA's translated tower than on this one.
"""
import argparse
import struct
import sys

import numpy as np
import openvino as ov
import openvino.opset13 as ops


class GGUF:
    """Minimal GGUF reader: metadata and tensors as numpy f32 (row-major shapes)."""
    _T = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    _S = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def __init__(self, path):
        f = self.f = open(path, 'rb')
        if f.read(4) != b'GGUF':
            raise ValueError(f"{path}: not a GGUF file")
        _, nt, nkv = struct.unpack('<IQQ', f.read(20))
        self.kv = {}
        for _ in range(nkv):
            k = self._str()
            (t,) = struct.unpack('<I', f.read(4))
            self.kv[k] = self._val(t)
        self.tensors = {}
        for _ in range(nt):
            name = self._str()
            (nd,) = struct.unpack('<I', f.read(4))
            ne = list(struct.unpack('<' + 'Q' * nd, f.read(8 * nd)))
            tt, off = struct.unpack('<IQ', f.read(12))
            self.tensors[name] = (ne, tt, off)
        align = self.kv.get('general.alignment', 32)
        self.data0 = (f.tell() + align - 1) // align * align

    def _str(self):
        (n,) = struct.unpack('<Q', self.f.read(8))
        return self.f.read(n).decode('utf-8', 'replace')

    def _val(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            et, n = struct.unpack('<IQ', self.f.read(12))
            if et == 8:
                return [self._str() for _ in range(n)]
            if et == 9:
                return [self._val(9) for _ in range(n)]
            return list(struct.unpack('<' + self._T[et] * n, self.f.read(self._S[et] * n)))
        return struct.unpack('<' + self._T[t], self.f.read(self._S[t]))[0]

    def tensor(self, name):
        ne, tt, off = self.tensors[name]
        n = int(np.prod(ne))
        shape = tuple(reversed(ne))  # ggml ne[0] is the fastest axis
        self.f.seek(self.data0 + off)
        if tt == 0:
            return np.frombuffer(self.f.read(4 * n), dtype=np.float32).reshape(shape)
        if tt == 1:
            return np.frombuffer(self.f.read(2 * n), dtype=np.float16).astype(np.float32).reshape(shape)
        if tt == 30:  # BF16
            u = np.frombuffer(self.f.read(2 * n), dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if tt == 8:  # Q8_0: blocks of 32 = f16 scale + 32 int8
            b = np.frombuffer(self.f.read(n // 32 * 34), dtype=np.dtype([('d', '<f2'), ('q', 'i1', (32,))]))
            return (b['d'].astype(np.float32)[:, None] * b['q'].astype(np.float32)).reshape(shape)
        raise NotImplementedError(f"{name}: ggml type {tt}")


def i64(*v):
    return ops.constant(np.array(v, dtype=np.int64))


def f16w(a):
    """A weight stored f16, computed in f32 (the plugin keeps it f16 on the NPU)."""
    return ops.convert(ops.constant(np.ascontiguousarray(a.astype(np.float16))), ov.Type.f32)


def f32c(a):
    return ops.constant(np.ascontiguousarray(a.astype(np.float32)))


def build_smolvla(g):
    kv = g.kv
    D, L, H = kv['smolvla.vit_hidden'], kv['smolvla.vit_layers'], kv['smolvla.vit_heads']
    S, P, s = kv['smolvla.image_size'], kv['smolvla.patch_size'], kv['smolvla.vit_pixel_shuffle']
    eps = kv.get('smolvla.vit_ln_eps', 1e-6)
    grid = S // P
    T, hd, gb = grid * grid, D // H, grid // s
    t = g.tensor

    def lin(x, w, b=None):
        y = ops.matmul(x, f16w(t(w)), False, True)
        return ops.add(y, f32c(t(b))) if b else y

    def ln(x, w, b):
        m = ops.mvn(x, i64(-1), True, eps, 'inside_sqrt')
        return ops.add(ops.multiply(m, f32c(t(w))), f32c(t(b)))

    px = ops.parameter([1, 3, S, S], ov.Type.f32, name='image')
    conv = ops.convolution(px, f32c(t('vit.patch_embd.weight')), [P, P], [0, 0], [0, 0], [1, 1])
    x = ops.transpose(ops.reshape(conv, i64(1, D, T), False), i64(0, 2, 1))     # [1, T, D], t = y*grid + x
    x = ops.add(ops.add(x, f32c(t('vit.patch_embd.bias'))), f32c(t('vit.pos_embd')))
    for i in range(L):
        p = f'vit.blk.{i}'
        h = ln(x, f'{p}.ln1.weight', f'{p}.ln1.bias')

        def heads(z):
            return ops.transpose(ops.reshape(z, i64(1, T, H, hd), False), i64(0, 2, 1, 3))
        a = ops.scaled_dot_product_attention(heads(lin(h, f'{p}.attn_q.weight', f'{p}.attn_q.bias')),
                                             heads(lin(h, f'{p}.attn_k.weight', f'{p}.attn_k.bias')),
                                             heads(lin(h, f'{p}.attn_v.weight', f'{p}.attn_v.bias')), causal=False)
        a = ops.reshape(ops.transpose(a, i64(0, 2, 1, 3)), i64(1, T, D), False)
        x = ops.add(x, lin(a, f'{p}.attn_o.weight', f'{p}.attn_o.bias'))
        h = ops.gelu(lin(ln(x, f'{p}.ln2.weight', f'{p}.ln2.bias'), f'{p}.fc1.weight', f'{p}.fc1.bias'), 'tanh')
        x = ops.add(x, lin(h, f'{p}.fc2.weight', f'{p}.fc2.bias'))
    x = ln(x, 'vit.post_ln.weight', 'vit.post_ln.bias')
    # Pixel shuffle, as smolvla.cpp's two reshape/permute steps: each s x s block of
    # patches becomes one token whose features are (row in block, column in block, channel).
    x = ops.reshape(x, i64(gb, s, gb, s, D), False)                              # (by, iy, bx, ix, d)
    x = ops.reshape(ops.transpose(x, i64(0, 2, 1, 3, 4)), i64(gb * gb, s * s * D), False)
    y = ops.matmul(x, f16w(t('mm.fc.weight')), False, True)                      # [n_img_tokens, hidden]
    return ov.Model([ops.result(y)], [px], 'smolvla_vision')


BUILDERS = {'smolvla': build_smolvla}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gguf')
    ap.add_argument('out', help='output .xml (the .bin is written next to it)')
    a = ap.parse_args()
    g = GGUF(a.gguf)
    arch = g.kv.get('general.architecture')
    if arch not in BUILDERS:
        sys.exit(f"build_vision_ir: no builder for '{arch}' (have: {', '.join(BUILDERS)})")
    model = BUILDERS[arch](g)
    ov.save_model(model, a.out, compress_to_fp16=False)
    print(f"build_vision_ir: {arch} -> {a.out} ({len(model.get_ops())} ops, "
          f"in {model.inputs[0].get_partial_shape()}, out {model.outputs[0].get_partial_shape()})")


if __name__ == '__main__':
    main()
