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

"""Print the token ids for an instruction, using the tokenizer and prompt an arch was trained with.

    tokenize_prompt.py --arch smolvla --text "pick up the black bowl"
    -> 18188,614,260,2632,12505,198

vla-cli --text calls this so the quickstart does not need raw ids. The prompt
templates are the ones eval/client/vla_cpp_client.py sends, so the ids match
the eval client token for token. pi05 puts the robot state in the prompt and
needs --state; --stats defaults to the LIBERO stats the eval client uses.
--views is the number of camera images for the archs whose prompt has image
slots.
"""

import argparse
import json
import re
import sys
from pathlib import Path

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
    "gr00t_n1_6":  "vrfai/gr00tn1d6-libero-gguf",
    "gr00t_n1_7":  "nvidia/Cosmos-Reason2-2B",
    "turbovla":    "bert-base-uncased",
}
TRUST_REMOTE_CODE = {"evo1", "gr00t_n1_5"}
MAX_LENGTH = {"smolvla": 48, "pi0": 48, "pi05": 200, "turbovla": 21}
VIEWS = {"evo1": 3, "bitvla": 2, "vla_jepa": 2, "gr00t_n1_5": 2, "gr00t_n1_6": 2, "gr00t_n1_7": 2}

BITVLA_PROMPT = "What action should the robot take to {}?"
VLA_ADAPTER_PROMPT = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
    "assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to "
    "{}?<|im_end|>\n<|im_start|>assistant\n"
)
OPENVLA_OFT_PROMPT = "In: What action should the robot take to {}?\nOut:"
OPENVLA_OFT_EMPTY_TOKEN = 29871
VLA_JEPA_PROMPT = (
    "Your task is {}. Infer the temporal dynamics from frames "
    + "".join(f"<|action_{i}|>" * 8 for i in range(3))
    + " and produce the corresponding policy actions "
    + "<|embodied_action|>" * 32
    + "."
)
GR00T_N1_7_IMAGE_PAD = 151655


def fail(msg: str) -> int:
    print(f"tokenize_prompt: {msg}", file=sys.stderr)
    return 1


def pi05_prompt(text: str, state: str, stats: str) -> str:
    import numpy as np
    if not stats:
        from huggingface_hub import hf_hub_download
        stats = hf_hub_download("lerobot/libero", "meta/stats.json", repo_type="dataset")
    st = json.loads(Path(stats).read_text())["observation.state"]
    q01 = np.asarray(st["q01"], dtype=np.float32).reshape(-1)
    q99 = np.asarray(st["q99"], dtype=np.float32).reshape(-1)
    s = np.asarray([float(x) for x in state.replace(",", " ").split()], dtype=np.float32)
    if s.size < q01.size:
        raise ValueError(f"--state has {s.size} values, the stats describe {q01.size}")
    normed = 2.0 * (s[:q01.size] - q01) / (q99 - q01) - 1.0
    disc = np.digitize(normed, bins=np.linspace(-1.0, 1.0, 256 + 1)[:-1]) - 1
    cleaned = text.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned}, State: {' '.join(map(str, disc.tolist()))};\nAction: "


def token_ids(arch: str, text: str, tok, args) -> list:
    views = args.views
    if arch in ("smolvla", "pi0"):
        text = text if text.endswith("\n") else text + "\n"
    if arch == "pi05":
        text = pi05_prompt(text, args.state, args.stats)
    if arch in MAX_LENGTH:
        return tok(text, truncation=True, max_length=MAX_LENGTH[arch])["input_ids"]

    if arch == "evo1":
        block = "<img>" + "<IMG_CONTEXT>" * 256 + "</img>"
        prompt = "".join(f"Image-{i+1}: {block}\n" for i in range(views)) + text.strip()
        enc = tok(prompt, padding="max_length", truncation=True, max_length=1024)
        return enc["input_ids"][:sum(enc["attention_mask"])]
    if arch == "bitvla":
        content = "<|image_pad|>" * (views * 256) + "<proprio_pad>" + BITVLA_PROMPT.format(text.lower())
        prompt = tok.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
        return tok(prompt, add_special_tokens=True)["input_ids"]
    if arch == "vla_adapter":
        return tok(VLA_ADAPTER_PROMPT.format(text.lower()), add_special_tokens=False)["input_ids"]
    if arch == "openvla_oft":
        ids = tok(OPENVLA_OFT_PROMPT.format(text.lower()), add_special_tokens=True)["input_ids"]
        return ids if ids and ids[-1] == OPENVLA_OFT_EMPTY_TOKEN else ids + [OPENVLA_OFT_EMPTY_TOKEN]
    if arch == "vla_jepa":
        from PIL import Image
        content = [{"type": "image", "image": Image.new("RGB", (224, 224))} for _ in range(views)]
        content.append({"type": "text", "text": VLA_JEPA_PROMPT.format(text)})
        enc = tok.apply_chat_template([[{"role": "user", "content": content}]], tokenize=True,
                                      add_generation_prompt=True, return_dict=True,
                                      processor_kwargs={"padding": True, "return_tensors": "pt"})
        return enc["input_ids"][0].tolist()
    if arch == "gr00t_n1_5":
        images = "".join(f"<image {i+1}><img>" + "<IMG_CONTEXT>" * 256 + "</img>" for i in range(views))
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                  + images + str([text]) + "<|im_end|>\n<|im_start|>assistant\n")
        return tok(prompt, add_special_tokens=False)["input_ids"]

    language = re.sub(r"[^\w\s]", "", text.lower())
    if arch == "gr00t_n1_6":
        content = language + "".join(f"<image {i+1}><img>" + "<IMG_CONTEXT>" * 64 + "</img>"
                                     for i in range(views))
    else:
        content = [{"type": "image"} for _ in range(views)] + [{"type": "text", "text": language}]
    prompt = tok.apply_chat_template([{"role": "user", "content": content}],
                                     tokenize=False, add_generation_prompt=False)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if arch == "gr00t_n1_6":
        return ids
    return [t for i in ids for t in ([i] * 64 if i == GR00T_N1_7_IMAGE_PAD else [i])]


def load_tokenizer(arch: str, name: str):
    if arch == "vla_jepa":
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(name)
        proc.tokenizer.add_tokens([f"<|action_{i}|>" for i in range(28)], special_tokens=True)
        proc.tokenizer.add_tokens(["<|embodied_action|>"], special_tokens=True)
        return proc
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=arch in TRUST_REMOTE_CODE,
                                        use_fast=arch != "evo1")
    if arch == "gr00t_n1_6":
        path = Path(name) / "chat_template.json"
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(name, "chat_template.json"))
        tok.chat_template = json.loads(path.read_text())["chat_template"]
    return tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(TOKENIZERS))
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokenizer", help="override the HuggingFace tokenizer id or a local dir")
    ap.add_argument("--views", type=int, help="camera images the prompt has slots for")
    ap.add_argument("--state", help="pi05: robot state, comma-separated floats (--state=-0.1,... if it starts with -)")
    ap.add_argument("--stats", help="pi05: stats.json with observation.state q01/q99 "
                                    "(default: lerobot/libero meta/stats.json)")
    args = ap.parse_args()

    if args.arch == "pi05" and not args.state:
        return fail("pi05 puts the robot state in the prompt, so it needs --state f,f,...")
    if args.views is None:
        args.views = VIEWS.get(args.arch, 1)
    if args.views < 1:
        return fail("--views must be at least 1")

    try:
        tok = load_tokenizer(args.arch, args.tokenizer or TOKENIZERS[args.arch])
        ids = token_ids(args.arch, args.text, tok, args)
    except ImportError as e:
        return fail(f"{e}. Install the client extras: pip install -e \".[client]\"")
    except (ValueError, KeyError, OSError) as e:
        return fail(str(e))
    print(",".join(str(int(i)) for i in ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
