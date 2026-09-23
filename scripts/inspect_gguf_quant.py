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

"""Print a GGUF's FoldQuant metadata and per-site layout, and check it against
the contract in docs/QUANTIZATION.md: every INT8 weight has an F32 .wscale of
N entries, any .ascale has K entries, K and N are multiples of 64, and fused
groups (q/k/v, k/v) agree on their .ascale. Exit status 1 on a violation, so
an exporter can use it as its conformance gate.

    python scripts/inspect_gguf_quant.py model.gguf [--quiet]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foldquant_ref as fq   # noqa: E402

GROUP = re.compile(r"^(.*\.\d+)\.attn_([qkv])$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--quiet", action="store_true", help="only report violations and the summary")
    args = ap.parse_args()

    r = gguf.GGUFReader(str(args.gguf))
    arch = r.fields["general.architecture"].contents()
    prefix = f"{arch}.quant."
    quant = {k[len(prefix):]: f.contents() for k, f in r.fields.items() if k.startswith(prefix)}
    tensors = {t.name: t for t in r.tensors}

    if not quant:
        print(f"{args.gguf}: no {prefix}* keys (not a FoldQuant GGUF)")
        return 0
    print(f"{args.gguf}: arch={arch}")
    for k in sorted(quant):
        print(f"  {prefix}{k} = {quant[k]!r}")
    if quant.get("method") != "foldquant":
        print("VIOLATION: quant.method must be 'foldquant'")
        return 1

    problems = 0
    sites = []
    for name, t in tensors.items():
        if t.tensor_type != gguf.GGMLQuantizationType.I8 or not name.endswith(".weight"):
            continue
        base = name[: -len(".weight")]
        mod = "llm" if base.startswith("vlm.") else "action"
        wbits = int(quant.get(f"{mod}_weight_bits", 8))
        shape = [int(s) for s in t.shape]          # ggml ne order: [K_pack, N]
        if len(shape) != 2:
            print(f"VIOLATION: {name} is {len(shape)}-D"); problems += 1; continue
        kpack, n_out = shape
        k_in = kpack * 2 if wbits == 4 else kpack
        ws = tensors.get(base + ".wscale")
        asc = tensors.get(base + ".ascale")
        ok = True
        if ws is None or ws.tensor_type != gguf.GGMLQuantizationType.F32 or [int(s) for s in ws.shape] != [n_out]:
            print(f"VIOLATION: {base}.wscale missing or not F32[{n_out}]"); ok = False
        if asc is not None and (asc.tensor_type != gguf.GGMLQuantizationType.F32 or [int(s) for s in asc.shape] != [k_in]):
            print(f"VIOLATION: {base}.ascale is not F32[{k_in}]"); ok = False
        if k_in % 64 or n_out % 64:
            print(f"VIOLATION: {name} K={k_in} N={n_out} must be multiples of 64"); ok = False
        problems += not ok
        bs = fq.rotation_block_for(k_in, int(quant.get(f"{mod}_rot_block_size", 64)))
        sites.append((base, mod, k_in, n_out, wbits, asc is not None, bs))
        if not args.quiet:
            print(f"  site {base:40s} W{wbits} K={k_in:<5d} N={n_out:<5d} rot={bs:<3d} ascale={'yes' if asc is not None else 'no'}")

    # Fused groups share one activation transform. Which projections are fused
    # follows the input they read: k and v always read the same tensor, and q
    # joins them only on a self-attention site. The file carries no self/cross
    # flag, so the split is inferred from K: on a cross-attention block q reads
    # the hidden state and k/v the encoder, and their K differ (GR00T: 1536 vs
    # 2048); when q's K equals k's, all three must agree, as vla.cpp's loader
    # (fq_declare_fused) then fuses them.
    groups = defaultdict(dict)
    for base, _mod, k_in, *_ in sites:
        m = GROUP.match(base)
        if m:
            groups[m.group(1)][m.group(2)] = (k_in, tensors.get(base + ".ascale"))
    for g, members in groups.items():
        have = [v is not None for _, v in members.values()]
        if any(have) and not all(have):
            print(f"VIOLATION: {g}: .ascale present on some of q/k/v but not all"); problems += 1
            continue
        if not all(have) or len(members) < 2:
            continue
        kq = members["q"][0] if "q" in members else None
        fused = [n for n in ("q", "k", "v") if n in members and (n != "q" or kq == members.get("k", (kq,))[0])]
        ref = np.asarray(members[fused[0]][1].data, dtype=np.float32)
        for n in fused[1:]:
            if not np.array_equal(ref, np.asarray(members[n][1].data, dtype=np.float32)):
                print(f"VIOLATION: {g}: fused {'/'.join(fused)} .ascale vectors differ"); problems += 1; break

    n_llm = sum(1 for s in sites if s[1] == "llm")
    n_act = len(sites) - n_llm
    print(f"summary: {len(sites)} FoldQuant sites ({n_llm} llm, {n_act} action), {problems} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
