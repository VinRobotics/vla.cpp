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

"""Reference data for tests/cosmos3_edge_prompt_check.cpp, made with lerobot and Hugging Face's tokenizer.

    python tests/py/make_cosmos3_edge_fixtures.py --policy-dir <Cosmos3-Edge-Policy-LIBERO checkpoint dir> \\
        --lerobot-src <lerobot/src> --out tests/data

  cosmos3_edge_prompts.tsv   hex(task) <TAB> hex(caption) <TAB> ids    every LIBERO task + edge cases: the caption of
                             Cosmos3PromptProcessorStep and the ids Cosmos3TokenizerProcessorStep gives
  cosmos3_edge_bpe.tsv       hex(text) <TAB> ids                   random ASCII and word mixes with special tokens
  cosmos3_edge_rot6d.tsv     10 floats <TAB> 7 floats              Cosmos3LiberoActionProcessorStep
"""
import argparse, glob, json, random, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--policy-dir", type=Path, required=True)
ap.add_argument("--lerobot-src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
sys.path.insert(0, str(args.lerobot_src))

import torch
from transformers import AutoTokenizer
from lerobot.policies.cosmos3.processor_cosmos3 import (
    START_OF_GENERATION, Cosmos3LiberoActionProcessorStep, Cosmos3PromptProcessorStep, make_cosmos3_pre_post_processors)

cfg = json.loads((args.policy_dir / "config.json").read_text())
tok = AutoTokenizer.from_pretrained(str(args.policy_dir / "tokenizer"))
step = Cosmos3PromptProcessorStep(view_description=cfg["view_description"], fps=cfg["fps"],
                                  chunk_size=cfg["chunk_size"], use_state=cfg["use_state"])
args.out.mkdir(parents=True, exist_ok=True)

def chat(prompt):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + tok.eos_token + START_OF_GENERATION

# --- prompts: every LIBERO task ------------------------------------------------------------------------------
from libero.libero import benchmark
tasks = []
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    b = benchmark.get_benchmark_dict()[suite]()
    tasks += [b.get_task(i).language for i in range(b.get_num_tasks())]
tasks += ["", "  Turn on the stove  ", "open the drawer!", "what?", 'say "hi" and\tgo']  # edge cases of the caption
with open(args.out / "cosmos3_edge_prompts.tsv", "w") as f:
    for t in tasks:
        cap = step.caption(t, 192, 320, 0, cfg["fps"])
        ids = tok(chat(cap), add_special_tokens=False)["input_ids"]
        f.write(t.encode().hex() + "\t" + cap.encode().hex() + "\t" + " ".join(map(str, ids)) + "\n")
print(len(tasks), "prompts")

# --- BPE stress -------------------------------------------------------------------------------------------------
rnd = random.Random(0)
alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n.,;:!?'\"-_/\\()[]{}<>@#$%^&*+=~`|"
words = ["Hello", "world's", "can't", "I've", "they'll", "ROBOT", "camelCase", "snake_case", "x86_64", "3.14159", "1234567",
         "end.\n\n", "  spaced  out  ", "a\r\nb", "<|im_start|>", "<|im_end|>", "<think>", "[INST]", "https://x.org/a?b=c&d=e",
         "</think>", "<SPECIAL_22>", "<|vision_start|>", "\x0b\x0c", "a  \n  b"]
texts = ["".join(rnd.choice(alpha) for _ in range(rnd.randint(1, 80))) for _ in range(400)]
texts += [" ".join(rnd.choice(words) for _ in range(rnd.randint(1, 12))) for _ in range(150)]
with open(args.out / "cosmos3_edge_bpe.tsv", "w") as f:
    for t in texts:
        f.write(t.encode().hex() + "\t" + " ".join(map(str, tok(t, add_special_tokens=False)["input_ids"])) + "\n")
print(len(texts), "bpe texts")

# --- rot6d ------------------------------------------------------------------------------------------------------
act = Cosmos3LiberoActionProcessorStep()
g = torch.Generator().manual_seed(0)
rows = []
for k in range(60):
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    r6 = torch.cat([q[:, 0], q[:, 1]])
    noise = [0.0, 0.01, 0.1, 0.3][k % 4]
    r6 = r6 * (1.0 + 0.3 * torch.randn(1, generator=g)) + noise * torch.randn(6, generator=g)
    rows.append(torch.cat([torch.randn(3, generator=g) * 0.3, r6, torch.rand(1, generator=g) * 1.4 - 0.2]))
rows.append(torch.tensor([0.1, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 1.0]))   # identity rotation
rows.append(torch.tensor([0.0, 0.0, 0.0, 0, 1, 0, 1, 0, 0, 0.0]))    # a half turn about (1,1,0)/sqrt2 minus rounding
x = torch.stack(rows).float()
y = act.action(x)
with open(args.out / "cosmos3_edge_rot6d.tsv", "w") as f:
    for a, b in zip(x.tolist(), y.tolist()):
        f.write(" ".join(f"{v:.9g}" for v in a) + "\t" + " ".join(f"{v:.9g}" for v in b) + "\n")
print(len(rows), "rot6d rows")
