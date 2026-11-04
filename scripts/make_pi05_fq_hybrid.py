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

"""A pi0.5 FoldQuant GGUF whose action expert is float again.

Copies every key and tensor of a FoldQuant pi0.5 GGUF (e.g. FoldQuant W4A4) and
replaces the action expert's quantized projections (aex.blk.*.{attn_q,attn_k,
attn_v,attn_o,ffn_gate,ffn_up,ffn_down}: integer codes + wscale + ascale) with
the float weights of the same base checkpoint's plain GGUF. The prefix (VLM)
keeps its FoldQuant sites; the expert can then take the load-time W8A8 / W2A8
weights (VLA_PI05_W8A8=expert VLA_PI05_W2=expert).

The two files must come from the same base checkpoint: every tensor they share
with the same type is checked byte for byte before anything is written.

    python scripts/make_pi05_fq_hybrid.py --fq pi05-libero-w4a4.gguf --float pi05-openpi-bf16.gguf \
        --out pi05-libero-w4a4-aexf.gguf
"""
import argparse
import re
import sys

import numpy as np
import gguf

SITES = ("attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down")
SITE_RE = re.compile(r"^aex\.blk\.\d+\.(%s)\.(weight|wscale|ascale)$" % "|".join(SITES))


def copy_fields(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name in ("general.architecture",):
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            sub = field.types[-1]
            if sub == gguf.GGUFValueType.STRING:
                val = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
            else:
                val = [field.parts[i].tolist()[0] for i in field.data]
            writer.add_array(name, val)
        elif vtype == gguf.GGUFValueType.STRING:
            writer.add_string(name, bytes(field.parts[field.data[0]]).decode("utf-8"))
        else:
            writer.add_key_value(name, field.parts[field.data[0]].tolist()[0], vtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fq", required=True, help="FoldQuant pi0.5 GGUF (prefix sites kept)")
    ap.add_argument("--float", required=True, help="plain pi0.5 GGUF of the same base checkpoint (expert weights)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fq = gguf.GGUFReader(args.fq)
    fl = gguf.GGUFReader(args.float)
    tfq = {t.name: t for t in fq.tensors}
    tfl = {t.name: t for t in fl.tensors}

    shared = [n for n in tfq if n in tfl and tfq[n].tensor_type == tfl[n].tensor_type]
    bad = [n for n in shared if not np.array_equal(np.asarray(tfq[n].data), np.asarray(tfl[n].data))]
    if bad:
        print(f"{len(bad)} of {len(shared)} shared tensors differ (e.g. {bad[:3]}): not the same base checkpoint")
        return 1
    print(f"{len(shared)} shared tensors byte-identical")

    arch = bytes(fq.fields["general.architecture"].parts[fq.fields["general.architecture"].data[0]]).decode()
    w = gguf.GGUFWriter(args.out, arch)
    copy_fields(fq, w)
    w.add_string("pi05.fq_expert_float_from", args.float.split("/")[-1])

    dropped = replaced = kept = 0
    names = []
    for t in fq.tensors:
        m = SITE_RE.match(t.name)
        if not m:
            names.append(("fq", t.name))
            kept += 1
            continue
        if m.group(2) == "weight":
            if t.name not in tfl:
                print(f"missing float tensor {t.name}")
                return 1
            names.append(("float", t.name))
            replaced += 1
        else:
            dropped += 1
    for src, name in names:
        t = (tfq if src == "fq" else tfl)[name]
        # the reader hands quantized and BF16 data as bytes; the writer derives the element shape from them
        w.add_tensor(name, np.asarray(t.data), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {args.out}: {kept} tensors from the FoldQuant file, {replaced} expert weights from the float file, "
          f"{dropped} expert scales dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
