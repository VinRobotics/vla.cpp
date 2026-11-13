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

"""MolmoAct2's prompt text, as its modeling_molmoact2.py builds it in predict_action.

The task is normalized (_normalize_question_text), the state normalized with the norm_tag statistics and
binned into <state_k> tokens, the setup and control text wrapped in their tokens, and one
"Image k<|image|>" put in front per view (vla-server expands each into the image tokens). Tokenize the
result with the checkpoint's tokenizer. numpy only: eval/client/vla_cpp_client.py and
scripts/tokenize_prompt.py both use it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_MOLMOACT2_PUNCT = ".,!?;:,\u2026"
_MOLMOACT2_CLOSERS = "\"'\u201d\u2019)]}"
_MOLMOACT2_DELIMS = "\"'`\u201c\u201d\u2018\u2019[](){}"
_MOLMOACT2_PREFIXES = (r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
                       r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+")
_MOLMOACT2_NORM_KEYS = {"q01_q99": ("q01", "q99"), "q10_q90": ("q10", "q90"), "min_max": ("min", "max"),
                        "mean_std": ("mean", "std")}


def _molmoact2_task(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    prev = None
    while t and t != prev:
        prev = t
        t = t.strip().strip(_MOLMOACT2_DELIMS).strip()
        for pat in _MOLMOACT2_PREFIXES:
            t = re.sub(pat, "", t, count=1, flags=re.IGNORECASE).strip()
        t = t.rstrip(_MOLMOACT2_PUNCT).rstrip().rstrip(_MOLMOACT2_CLOSERS).rstrip().rstrip(_MOLMOACT2_PUNCT).rstrip()
    chunks = [c.strip() for c in re.split(r"[.!?]+", t) if c.strip()]
    if len(chunks) > 1:
        t = "; ".join(chunks)
    return t.lower()


class Molmoact2Prompt:
    """Builds MolmoAct2 prompt text from the checkpoint's config.json and norm_stats.json payloads."""

    def __init__(self, config: dict, norm_stats: dict, norm_tag: str = "libero"):
        if norm_tag not in norm_stats.get("metadata_by_tag", {}):
            raise ValueError(f"norm tag {norm_tag!r} not in norm_stats.json "
                             f"({sorted(norm_stats.get('metadata_by_tag', {}))})")
        meta = norm_stats["metadata_by_tag"][norm_tag]
        self.mode = str(norm_stats.get("norm_mode", "min_max"))
        st = meta["state_stats"]
        if self.mode != "none":
            lo, hi = _MOLMOACT2_NORM_KEYS[self.mode]
            self.lo = np.asarray(st[lo], dtype=np.float32).reshape(-1)
            self.hi = np.asarray(st[hi], dtype=np.float32).reshape(-1)
        dim = len(st.get("names") or st.get("mean") or st.get("q01") or st["min"])
        self.mask = np.broadcast_to(np.asarray(st.get("mask", [True] * dim), dtype=bool), (dim,))
        self.zero = None
        if self.mode != "mean_std" and "min" in st and "max" in st:
            self.zero = np.asarray(st["min"], dtype=np.float32) == np.asarray(st["max"], dtype=np.float32)
        self.state_dim = dim
        self.n_bins = int(config["num_state_tokens"])
        wrap = lambda text, a, b, on: text if not text or not on or (text.startswith(a) and text.endswith(b)) \
            else f"{a}{text}{b}"
        self.setup = wrap(str(meta.get("setup_type", "") or ""), "<setup_start>", "<setup_end>",
                          bool(config.get("add_setup_tokens", False)))
        self.control = wrap(str(meta.get("control_mode", "") or ""), "<control_start>", "<control_end>",
                            bool(config.get("add_control_tokens", False)))

    def state_string(self, state) -> str:
        x = np.asarray(state, dtype=np.float32).reshape(-1)[: self.state_dim]
        eps = np.float32(1e-6)
        if self.mode == "mean_std":
            n = (x - self.lo) / np.maximum(self.hi, eps)
        elif self.mode == "none":
            n = x
        else:
            n = np.float32(2.0) * (x - self.lo) / np.maximum(self.hi - self.lo, eps) - np.float32(1.0)
            n = np.clip(n, -1.0, 1.0)
        n = np.where(self.mask, n, x)
        if self.zero is not None:
            n = np.where(self.zero, 0.0, n)
        n = np.clip(np.nan_to_num(n.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        bins = np.clip(np.rint((n + 1.0) / 2.0 * float(self.n_bins - 1)).astype(np.int64), 0, self.n_bins - 1)
        return "<state_start>" + "".join(f"<state_{int(b)}>" for b in bins) + "<state_end>"

    def text(self, task: str, state, n_views: int) -> str:
        prompt = (f"The task is to {_molmoact2_task(task)}. The setup is {self.setup}. The current state of the "
                  f"robot is {self.state_string(state)}. The expected control mode is {self.control}. Given these, "
                  "what action should the robot take to complete the task?")
        images = "<|image|>" if n_views == 1 else "".join(f"Image {i + 1}<|image|>" for i in range(n_views))
        return f"{images}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<action_output>"

    def ids(self, tok, task: str, state, n_views: int) -> list[int]:
        """The processor's input_ids before image expansion: the text tokenized, BOS (<|im_end|>) in front."""
        ids = [int(t) for t in tok(self.text(task, state, n_views))["input_ids"]]
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        return ids if ids and ids[0] == bos else [int(bos)] + ids

    @staticmethod
    def load(repo: str, stats_json: str | Path | None = None, norm_tag: str = "libero") -> "Molmoact2Prompt":
        def fetch(name: str) -> Path:
            local = Path(repo) / name
            if local.exists():
                return local
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(repo, name))
        cfg = json.loads(fetch("config.json").read_text())
        stats = json.loads((Path(stats_json) if stats_json else fetch(cfg.get("norm_stats_filename",
                                                                             "norm_stats.json"))).read_text())
        return Molmoact2Prompt(cfg, stats, norm_tag)
