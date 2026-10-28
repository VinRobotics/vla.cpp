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
"""Turn FoldQuant sites of a GGUF back into BF16 float weights (a debugging aid).

A FoldQuant site computes ``y = Wq . q(H (x / s))`` with ``Wq`` the INT8 codes
times the per-row scale, ``H`` the block Hadamard (symmetric, orthonormal) and
``s`` the optional SmoothQuant vector. Dropping the activation quantizer gives
the float weight ``W_eff = Wq . H . diag(1/s)`` that reproduces the site up to
quantization noise, with every fold (SQ in the norm gammas, rotation) kept.
Running the result on a float path isolates the FoldQuant kernels from the
rest of the graph: a bisection over ``--modules`` says which module's
integer path deviates from the reference.

    python scripts/foldquant_dequant.py --in int8.gguf --out deq.gguf [--modules llm,action]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf  # noqa: E402

import foldquant_ref as fq  # noqa: E402
from quantize_gguf import to_f32  # noqa: E402

LLM_SITE = re.compile(r"^vlm\.blk\.\d+\.(attn_q|attn_k|attn_v|attn_o|ffn_gate|ffn_up|ffn_down)\.weight$")
DIT_SITE = re.compile(r"^aex\.dit\.\d+\.(attn_q|attn_k|attn_v|attn_o|ff0|ff2)\.weight$")


def _bf16(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bit pattern (uint16)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((128, 256)).astype(np.float32)
    s = rng.uniform(0.5, 2.0, 256).astype(np.float32)
    folded = fq.fold_weight(w * s[None, :], 64)   # what the exporter stores (SQ then rotate)
    back = fq.fwht_rows(folded, 64) / s[None, :]
    assert np.allclose(back, w, atol=1e-4), "fold/dequant are not inverses; check the Hadamard convention"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--modules", default="llm,action", help="comma list of llm, action: which sites to dequantize")
    args = ap.parse_args()
    modules = {m.strip() for m in args.modules.split(",") if m.strip()}
    _selfcheck()

    r = gguf.GGUFReader(str(args.src))
    arch = r.fields["general.architecture"].contents()
    qp = f"{arch}.quant."
    if not any(k.startswith(qp) for k in r.fields):
        raise SystemExit(f"{args.src} is not a FoldQuant GGUF")
    tensors = {t.name: t for t in r.tensors}
    rot = {
        "llm": int(r.fields[qp + "llm_rot_block_size"].contents()),
        "action": int(r.fields[qp + "action_rot_block_size"].contents()),
    }
    fold_before = str(r.fields[qp + "action_fold_order"].contents()) == "before"

    w = gguf.GGUFWriter(str(args.dst), arch)
    meta = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for name, f in r.fields.items():
        if name in meta:
            continue
        val = f.contents()
        if name == qp + "provenance":
            val = f"{val}; dequantized {sorted(modules)} by scripts/foldquant_dequant.py"
        if f.types and f.types[0] == gguf.GGUFValueType.ARRAY:
            w.add_array(name, val)
        else:
            w.add_key_value(name, val, f.types[0])

    n_deq = 0
    skip: set[str] = set()
    for t in r.tensors:
        if t.name in skip:
            continue
        mod = "llm" if LLM_SITE.match(t.name) else ("action" if DIT_SITE.match(t.name) else None)
        if mod in modules and t.tensor_type == gguf.GGMLQuantizationType.I8:
            base = t.name[: -len(".weight")]
            codes = np.ascontiguousarray(t.data).view(np.int8)
            n_out, kpack = int(t.shape[1]), int(t.shape[0])   # reader shape is ggml ne order
            if int(r.fields[qp + f"{mod}_weight_bits"].contents()) == 4:
                codes = fq.unpack_nibbles(codes.reshape(n_out, kpack))
            k_in = codes.shape[1] if codes.ndim == 2 else kpack
            codes = codes.reshape(n_out, k_in).astype(np.float32)
            wscale = np.asarray(tensors[base + ".wscale"].data, dtype=np.float32).reshape(n_out)
            wq = codes * wscale[:, None]
            bs = fq.rotation_block_for(k_in, rot[mod])
            w_eff = fq.fwht_rows(wq, bs) if bs > 1 else wq
            asc_t = tensors.get(base + ".ascale")
            if asc_t is not None:
                s = np.asarray(asc_t.data, dtype=np.float32).reshape(k_in)
                # fold-before: y = Wq H (x/s)  -> W_eff = (Wq H) / s
                # fold-after : y = Wq ((H x)/s) -> W_eff = (Wq diag(1/s)) H
                w_eff = w_eff / s[None, :] if fold_before else (fq.fwht_rows(wq / s[None, :], bs) if bs > 1 else wq / s[None, :])
                skip.add(base + ".ascale")
            skip.add(base + ".wscale")
            w.add_tensor(t.name, _bf16(w_eff), raw_shape=[n_out, k_in], raw_dtype=gguf.GGMLQuantizationType.BF16)
            n_deq += 1
            continue
        data = np.ascontiguousarray(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            data = data.view(np.uint16)
        elif t.tensor_type == gguf.GGMLQuantizationType.F32:
            data = data.astype(np.float32, copy=False)
        w.add_tensor(t.name, data, raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"dequantized {n_deq} sites ({sorted(modules)}) -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
