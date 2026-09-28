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

"""Copy a vla.cpp GGUF and embed the SentencePiece tokenizer its arch was trained
with as <arch>.tokenizer.spm_model, so vla-cli --text tokenizes in-process with
no Python and no HF login.

    python scripts/add_tokenizer_to_gguf.py --in pi0.gguf --out pi0-tok.gguf

pi05 puts the robot state in its prompt, so its observation.state q01/q99 go in
too as pi05.state.q01/q99 (--stats, default: the LIBERO meta/stats.json the eval
client uses).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import gguf

from gguf_common import copy_kv, copy_tensor

TOKENIZERS = {
    "pi0":         "google/paligemma-3b-pt-224",
    "pi05":        "google/paligemma-3b-pt-224",
    "openvla_oft": "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--tokenizer", help="HF repo holding tokenizer.model, or a local .model file "
                                        "(default: the one the arch was trained with)")
    ap.add_argument("--stats", help="pi05: stats.json with observation.state q01/q99")
    args = ap.parse_args()

    r = gguf.GGUFReader(args.src)
    arch = r.fields["general.architecture"].contents()
    if arch not in TOKENIZERS:
        raise SystemExit(f"vla-cli has no in-process --text prompt for arch {arch}; "
                         f"supported: {', '.join(TOKENIZERS)}")
    tok = args.tokenizer or TOKENIZERS[arch]
    if Path(tok).is_file():
        spm = Path(tok)
    else:
        from huggingface_hub import hf_hub_download
        spm = Path(hf_hub_download(tok, "tokenizer.model"))
    kv = {f"{arch}.tokenizer.spm_model": spm.read_bytes()}

    if arch == "pi05":
        stats = args.stats
        if not stats:
            from huggingface_hub import hf_hub_download
            stats = hf_hub_download("lerobot/libero", "meta/stats.json", repo_type="dataset")
        st = json.loads(Path(stats).read_text())["observation.state"]
        for q in ("q01", "q99"):
            kv[f"pi05.state.{q}"] = np.asarray(st[q], dtype=np.float32).reshape(-1).tolist()

    w = gguf.GGUFWriter(args.dst, arch)
    copy_kv(r, w, skip=kv)
    for k, v in kv.items():
        w.add_array(k, v)
    for t in r.tensors:
        copy_tensor(w, t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.dst}: added {', '.join(kv)} from {spm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
