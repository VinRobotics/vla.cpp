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

"""Print the token ids for an instruction, using the tokenizer an arch was trained with.

    tokenize_prompt.py --arch smolvla --text "pick up the black bowl"
    -> 1,4842,731,254,2482,7681,2

vla-cli --text calls this so the quickstart does not need raw ids. The eval
client keeps its own richer prompt handling; this only covers the plain case.
"""

import argparse
import sys

# Same tokenizers the eval client uses (eval/client/vla_cpp_client.py).
TOKENIZERS = {
    "smolvla":     "HuggingFaceTB/SmolVLM2-500M-Instruct",
    "pi0":         "google/paligemma-3b-pt-224",
    "pi05":        "google/paligemma-3b-pt-224",
    "evo1":        "OpenGVLab/InternVL3-1B",
    "bitvla":      "hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16",
    "vla_adapter": "VLA-Adapter/LIBERO-Object-Pro",
    "openvla_oft": "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    "vla_jepa":    "Qwen/Qwen3-VL-2B-Instruct",
    "gr00t_n1_5":  "lerobot/eagle2hg-processor-groot-n1p5",
    "gr00t_n1_7":  "nvidia/Cosmos-Reason2-2B",
}
TRUST_REMOTE_CODE = {"evo1", "gr00t_n1_5"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=sorted(TOKENIZERS))
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokenizer", help="override the HuggingFace tokenizer id")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is not installed: pip install -e \".[client]\"", file=sys.stderr)
        return 1

    name = args.tokenizer or TOKENIZERS[args.arch]
    tok = AutoTokenizer.from_pretrained(
        name, trust_remote_code=args.arch in TRUST_REMOTE_CODE)
    ids = tok(args.text)["input_ids"]
    print(",".join(str(int(i)) for i in ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
