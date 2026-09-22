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

"""Turn a float GR00T GGUF into a contract-conformant FoldQuant GGUF without
calibration: block-Hadamard rotation of every site's weight plus per-row
symmetric INT8 (or nibble-packed INT4) rounding, no SmoothQuant (so no
.ascale and unchanged norms). It is the in-repo producer of the format in
docs/QUANTIZATION.md, for kernel bring-up, benchmarks and CI without VLA-OPT;
the calibrated export that carries the SmoothQuant folds is `vla-opt build
--target vlacpp`.

    python scripts/foldquant_fake_export.py --in n17-bf16.gguf --out n17-fq.gguf
        [--rot-block 64] [--wbits 8|4] [--modules llm,action]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foldquant_ref as fq          # noqa: E402
from quantize_gguf import to_f32    # noqa: E402

LLM_SITE = re.compile(r"^vlm\.blk\.\d+\.(attn_q|attn_k|attn_v|attn_o|ffn_gate|ffn_up|ffn_down)\.weight$")
DIT_SITE = re.compile(r"^aex\.dit\.\d+\.(attn_q|attn_k|attn_v|attn_o|ff0|ff2)\.weight$")
# VLA-OPT's FoldQuant keys (one per weight width). This file is the un-smoothed,
# uncalibrated realization of that arm: the key names the fold family, not a
# calibration.
SCHEME = {8: "vlaopt_foldq_w8a8", 4: "vlaopt_foldq_w4a4"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--rot-block", type=int, default=64, help="nominal Hadamard block (narrowed per site)")
    ap.add_argument("--wbits", type=int, default=8, choices=(8, 4))
    ap.add_argument("--abits", type=int, default=8, choices=(8, 4), help="activation width recorded in the metadata")
    ap.add_argument("--modules", default="llm,action", help="comma list of llm, action")
    ap.add_argument("--sites", default=None, help="regex; only matching site names are quantized (bisection aid)")
    args = ap.parse_args()
    modules = {m.strip() for m in args.modules.split(",") if m.strip()}
    site_filter = re.compile(args.sites) if args.sites else None

    r = gguf.GGUFReader(str(args.src))
    arch = r.fields["general.architecture"].contents()
    quant_prefix = f"{arch}.quant."
    if any(k.startswith(quant_prefix) for k in r.fields):
        raise SystemExit(f"{args.src} is already a FoldQuant GGUF")

    w = gguf.GGUFWriter(str(args.dst), arch)
    meta = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for name, f in r.fields.items():
        if name in meta:
            continue
        if f.types and f.types[0] == gguf.GGUFValueType.ARRAY:
            w.add_array(name, f.contents())
        else:
            w.add_key_value(name, f.contents(), f.types[0])

    kv = lambda k: quant_prefix + k   # noqa: E731
    w.add_string(kv("method"), "foldquant")
    w.add_string(kv("applied_at"), "foldquant_fake_export")
    w.add_string(kv("provenance"), f"scripts/foldquant_fake_export.py rot_block={args.rot_block} wbits={args.wbits} (no calibration, no SmoothQuant)")
    for mod in ("llm", "action"):
        on = mod in modules
        w.add_string(kv(f"scheme_{mod}"), SCHEME[args.wbits] if on else "float")
        w.add_uint32(kv(f"{mod}_weight_bits"), args.wbits if on else 16)
        w.add_uint32(kv(f"{mod}_act_bits"), args.abits if on else 32)
        w.add_uint32(kv(f"{mod}_rot_block_size"), args.rot_block if on else 0)
    w.add_string(kv("action_fold_order"), "before")
    w.add_float32(kv("act_clip_ratio"), 1.0)
    w.add_string(kv("site_bits"), "")

    n_sites = skipped = 0
    bytes_in = bytes_out = 0
    for t in r.tensors:
        src_bytes = int(t.data.nbytes)
        bytes_in += src_bytes
        is_llm = bool(LLM_SITE.match(t.name)) and "llm" in modules
        is_dit = bool(DIT_SITE.match(t.name)) and "action" in modules
        if site_filter is not None and not site_filter.search(t.name):
            is_llm = is_dit = False
        f32 = to_f32(t) if (is_llm or is_dit) else None
        if f32 is not None and f32.ndim == 2 and f32.shape[0] % 64 == 0 and f32.shape[1] % 64 == 0:
            n_out, k_in = f32.shape
            bs = fq.rotation_block_for(k_in, args.rot_block)
            folded = fq.fold_weight(f32, bs)
            codes, wscale = fq.weight_quant_per_row(folded, args.wbits)
            body = fq.pack_nibbles(codes) if args.wbits == 4 else codes.view(np.uint8)
            w.add_tensor(t.name, np.ascontiguousarray(body), raw_shape=[n_out, body.shape[1]],
                         raw_dtype=gguf.GGMLQuantizationType.I8)
            base = t.name[: -len(".weight")]
            w.add_tensor(base + ".wscale", np.ascontiguousarray(wscale, dtype=np.float32),
                         raw_shape=[n_out], raw_dtype=gguf.GGMLQuantizationType.F32)
            bytes_out += int(body.nbytes) + int(wscale.nbytes)
            n_sites += 1
            continue
        if is_llm or is_dit:
            skipped += 1
            print(f"  note: {t.name} shape {tuple(int(s) for s in t.shape)} not a multiple of 64, kept float")
        data = np.ascontiguousarray(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            data = data.view(np.uint16)
        elif t.tensor_type == gguf.GGMLQuantizationType.F32:
            data = data.astype(np.float32, copy=False)
        w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
        bytes_out += src_bytes

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"foldquant W{args.wbits}: {n_sites} sites quantized ({skipped} kept float), "
          f"weights {bytes_in/1e9:.2f} GB -> {bytes_out/1e9:.2f} GB ({100*bytes_out/max(bytes_in,1):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
