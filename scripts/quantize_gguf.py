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

"""Requantize a vla.cpp GGUF: pack large weight matrices to Q8_0/Q4_0 and copy
everything else unchanged. The loader keeps quantized weights packed and ggml
runs them as int8 dot products, so a Q8_0 file is about half the size of the
bf16 one with near-identical actions. Embeddings, the output head, norms, conv
patch embeddings and position tables stay float (row-fetch and small tensors do
not benefit and can lose accuracy).

    python scripts/quantize_gguf.py --in model-bf16.gguf --out model-q8_0.gguf --type Q8_0

This is the stock ggml repack: block-32 absmax weights, float activations,
dequantized inside ggml_mul_mat, runs on every backend. It is not FoldQuant.
A FoldQuant GGUF (docs/QUANTIZATION.md: INT8/INT4 codes + sidecar scales executed
by the in-tree integer kernels on CUDA/CPU, read back as float weights elsewhere) is produced from a FoldQuantVLA
quantized model by scripts/convert_quantized_model_to_gguf.py or, uncalibrated,
by scripts/foldquant_fake_export.py; this script refuses such a file.
"""

import argparse
import re
import numpy as np
import gguf

from gguf_common import copy_kv, copy_tensor

# Substrings that keep a tensor at its source precision. Embeddings, the output
# head, norms, conv, position tables and the action expert stay float. The vision
# tower stays float too by default; add --vision to pack it as well.
SKIP = (
    "token_embd",
    "tok_embd",
    "output.weight",
    "patch_embd",
    "norm",
    "pos",
    "embed",
    "cls",
    "action",
    "state",
    "aex.",
    "ah.",
    "act.",
    "octo.head",
    "dit",
    "adaln",
    "ada_",
    "time"
)
SKIP_VISION = ("vit", "vision", "vis.d.", "vis.s.", "octo.obs.")
# With --dit, the DiT's weight matrices are packed too; these entries of SKIP
# are what kept them float.
DIT_UNSKIP = ("dit", "aex.", "ah.")

# Block size per row (ne0 must divide this). Only the types the gguf writer can
# pack are offered; Q8_0 is near-lossless, Q4_0/Q4_1 are 4-bit.
QK = {"Q8_0": 32, "Q4_0": 32, "Q5_0": 32, "Q4_1": 32, "Q5_1": 32}


def bf16_to_f32(u8: np.ndarray) -> np.ndarray:
    u16 = u8.view(np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


def to_f32(t) -> np.ndarray:
    # C-order array with the ggml ne axes reversed (ne0 is the last axis).
    shape = tuple(int(x) for x in t.shape[::-1])
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return t.data.astype(np.float32, copy=False).reshape(shape)
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        return bf16_to_f32(np.ascontiguousarray(t.data).reshape(-1)).reshape(shape)
    return None  # already packed / unsupported source


def eligible(name: str, shape, qtype: str, skip, dit: bool = False, keep=None) -> bool:
    if len(shape) != 2 or not name.endswith(".weight"):
        return False
    if keep is not None and keep.search(name):
        return False
    if dit and ".dit." in name:
        skip = tuple(s for s in skip if s not in DIT_UNSKIP)
    if any(s in name for s in skip):
        return False
    return int(shape[0]) % QK[qtype] == 0  # ne0 is the quantization axis


def ffn_pads(r) -> dict:
    """FFN blocks whose width is not a multiple of 32: prefix -> (width, padded)."""
    pads = {}
    for t in r.tensors:
        if t.name.endswith(".fc1.weight") and len(t.shape) == 2 and int(t.shape[1]) % 32:
            width = int(t.shape[1])
            pads[t.name[:-len(".fc1.weight")]] = (width, (width + 31) // 32 * 32)
    return pads


def pad_ffn(t, old: int, new: int) -> np.ndarray:
    """fc1 rows and bias, or fc2 columns, zero-padded from old to new."""
    arr = to_f32(t)
    if t.name.endswith(".fc1.weight"):
        return np.concatenate([arr, np.zeros((new - old, arr.shape[1]), np.float32)], axis=0)
    if t.name.endswith(".fc1.bias"):
        return np.concatenate([arr, np.zeros((new - old,), np.float32)])
    return np.concatenate([arr, np.zeros((arr.shape[0], new - old), np.float32)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--type", default="Q8_0", choices=sorted(QK))
    ap.add_argument("--vision", action="store_true", help="also pack the vision tower")
    ap.add_argument("--dit", action="store_true",
                    help="also pack the DiT's weight matrices (adaLN, the time embedding, "
                         "norms and the state/action encoders stay float)")
    ap.add_argument("--keep-float", metavar="REGEX",
                    help="keep the weights whose name matches REGEX at their source precision, "
                         "e.g. '^vlm\\.blk\\.(0|1)\\.' for an LM's first two blocks")
    ap.add_argument("--pad-m32", action="store_true",
                    help="zero-pad FFN widths that are not a multiple of 32 (fc1 rows and bias, "
                         "fc2 columns): Hexagon's matrix unit takes no other width for a packed "
                         "weight. The output is unchanged")
    args = ap.parse_args()
    keep = re.compile(args.keep_float) if args.keep_float else None
    skip = SKIP if args.vision else SKIP + SKIP_VISION

    r = gguf.GGUFReader(args.src)
    arch = r.fields["general.architecture"].contents()
    if f"{arch}.quant.method" in r.fields:
        raise SystemExit(f"{args.src} is a FoldQuant GGUF; its INT8 sites are not ggml block types "
                         "and cannot be repacked (see docs/QUANTIZATION.md)")
    w = gguf.GGUFWriter(args.dst, arch)

    copy_kv(r, w)

    qtype = getattr(gguf.GGMLQuantizationType, args.type)
    pads = ffn_pads(r) if args.pad_m32 else {}
    if pads:
        old, new = next(iter(pads.values()))
        print(f"--pad-m32: {len(pads)} FFN blocks, e.g. {old} -> {new}")
    padded = (".fc1.weight", ".fc1.bias", ".fc2.weight")
    n_q = 0
    bytes_in = bytes_out = 0
    for t in r.tensors:
        src_bytes = int(t.data.nbytes)
        bytes_in += src_bytes
        blk = t.name.rsplit(".", 2)[0] if t.name.endswith(padded) else None
        if blk in pads and to_f32(t) is not None:
            arr = pad_ffn(t, *pads[blk])
            if eligible(t.name, arr.shape[::-1], args.type, skip, args.dit, keep):
                packed = gguf.quants.quantize(arr, qtype)
                w.add_tensor(t.name, packed, raw_dtype=qtype)
                bytes_out += int(packed.nbytes)
                n_q += 1
            else:
                w.add_tensor(t.name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
                bytes_out += int(arr.nbytes)
            continue
        f32 = to_f32(t) if eligible(t.name, t.shape, args.type, skip, args.dit, keep) else None
        if f32 is not None:
            packed = gguf.quants.quantize(f32, qtype)  # rows are byte-sized; shape inferred
            w.add_tensor(t.name, packed, raw_dtype=qtype)
            bytes_out += int(packed.nbytes)
            n_q += 1
        else:
            copy_tensor(w, t)
            bytes_out += src_bytes

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.type}: quantized {n_q}/{len(r.tensors)} tensors, "
          f"weights {bytes_in/1e9:.2f} GB -> {bytes_out/1e9:.2f} GB "
          f"({100*bytes_out/bytes_in:.0f}%)")


if __name__ == "__main__":
    main()
