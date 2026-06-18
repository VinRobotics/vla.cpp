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

from __future__ import annotations

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from PIL import Image
from transformers import AutoTokenizer

ARCH_PRESETS = {
    "smolvla": {"image_size": 512, "tokenizer": "HuggingFaceTB/SmolVLM2-500M-Instruct", "max_state_dim": 32},
    "pi0":     {"image_size": 224, "tokenizer": "google/paligemma-3b-pt-224",           "max_state_dim": 32},
    "pi05":    {"image_size": 224, "tokenizer": "google/paligemma-3b-pt-224",           "max_state_dim": 32},

    "evo1":    {"image_size": 448, "tokenizer": "OpenGVLab/InternVL3-1B", "max_state_dim": 24,
                "trust_remote_code": True, "use_fast_tokenizer": False},

    "bitvla":  {"image_size": 224, "tokenizer": "hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16", "max_state_dim": 32},
    "vla_adapter": {"image_size": 224,
                    "tokenizer": "VLA-Adapter/LIBERO-Object-Pro",
                    "max_state_dim": 8},
    "openvla_oft": {"image_size": 224,
                    "tokenizer": "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
                    "max_state_dim": 8},

    "gr00t_n1_7": {"image_size": 256, "tokenizer": "nvidia/Cosmos-Reason2-2B", "max_state_dim": 132},

    "gr00t_n1_5": {"image_size": 224, "tokenizer": "lerobot/eagle2hg-processor-groot-n1p5",
                   "max_state_dim": 64, "trust_remote_code": True},

    "gr00t_n1_6": {"image_size": 224, "tokenizer": None, "max_state_dim": 128, "trust_remote_code": True},
}

BITVLA_N_PATCHES_PER_VIEW = 256
BITVLA_N_VIEWS            = 2
BITVLA_IMAGE_PAD_TOKEN    = "<|image_pad|>"
BITVLA_PROPRIO_PAD_TOKEN  = "<proprio_pad>"
BITVLA_USER_PROMPT_TPL    = "What action should the robot take to {instruction}?"
VLA_ADAPTER_N_VIEWS       = 2
VLA_ADAPTER_PROMPT_TPL    = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
    "assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to "
    "{instruction}?<|im_end|>\n<|im_start|>assistant\n"
)
OPENVLA_OFT_N_VIEWS    = 2
OPENVLA_OFT_PROMPT_TPL = "In: What action should the robot take to {instruction}?\nOut:"
OPENVLA_OFT_EMPTY_TOKEN = 29871

def _load_pb():

    if "vla_pb2" in sys.modules:
        return sys.modules["vla_pb2"]
    proto_file = Path(os.environ.get(
        "VLA_CPP_PROTO",
        Path(__file__).resolve().parents[2] / "src" / "serving" / "vla.proto",
    ))
    if not proto_file.exists():
        raise FileNotFoundError(
            f"vla.proto not found at {proto_file}. "
            f"Set VLA_CPP_PROTO to override the expected "
            f"<vla.cpp>/src/serving/vla.proto location.")
    tmpdir = Path(tempfile.mkdtemp(prefix="vla-pb-"))
    subprocess.check_call([
        "protoc",
        f"--proto_path={proto_file.parent}",
        f"--python_out={tmpdir}",
        str(proto_file),
    ])
    sys.path.insert(0, str(tmpdir))
    import vla_pb2
    return vla_pb2

def _resize_with_pad(img_chw: np.ndarray, target_h: int, target_w: int,
                     pad_value: float = 0.0) -> np.ndarray:

    t = torch.from_numpy(img_chw).unsqueeze(0)
    cur_h, cur_w = t.shape[2:]
    ratio = max(cur_w / target_w, cur_h / target_h)
    rh = int(cur_h / ratio)
    rw = int(cur_w / ratio)
    t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
    pad_h = max(0, target_h - rh)
    pad_w = max(0, target_w - rw)
    t = F.pad(t, (pad_w, 0, pad_h, 0), value=pad_value)
    return t.squeeze(0).numpy()

class VlaCppClient:

    DEFAULT_RECV_TIMEOUT_MS = 30_000

    def __init__(
        self,
        vla_addr: str = "tcp://localhost:5555",
        *,
        arch: str = "smolvla",
        tokenizer_name: str | None = None,
        image_size: int | None = None,
        max_state_dim: int | None = None,
        real_action_dim: int = 7,
        image_keys: Sequence[str] = (
            "observation.images.image",
            "observation.images.image2",
        ),
        max_length: int = 48,
        recv_timeout_ms: int = DEFAULT_RECV_TIMEOUT_MS,
        n_action_steps: int = 1,

        stats_json: str | Path | None = None,
        bitvla_unnorm_key: str | None = None,
    ):
        if arch not in ARCH_PRESETS:
            raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCH_PRESETS)}")
        preset = ARCH_PRESETS[arch]
        tokenizer_name = tokenizer_name if tokenizer_name is not None else preset["tokenizer"]
        image_size     = image_size     if image_size     is not None else preset["image_size"]
        max_state_dim  = max_state_dim  if max_state_dim  is not None else preset.get("max_state_dim", 32)
        if tokenizer_name is None:
            raise ValueError(
                f"arch={arch} has no default tokenizer; pass --tokenizer "
                f"(an HF id or a local ckpt dir).")
        self.arch = arch

        self.pb = _load_pb()
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.sock.connect(vla_addr)
        print(f"vla-cpp-direct[arch={arch}]: connected to {vla_addr}", flush=True)

        trust_remote = bool(preset.get("trust_remote_code", False))
        use_fast    = bool(preset.get("use_fast_tokenizer", True))
        print(f"vla-cpp-direct: loading tokenizer {tokenizer_name}"
              f"{' (trust_remote_code=True)' if trust_remote else ''}"
              f"{' (use_fast=False)' if not use_fast else ''}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote, use_fast=use_fast)

        if arch == "gr00t_n1_6":
            ct_path = Path(tokenizer_name) / "chat_template.json"
            if ct_path.exists():
                self.tok.chat_template = json.loads(ct_path.read_text())["chat_template"]
        self.image_size = image_size
        self.max_state_dim = max_state_dim
        self.real_action_dim = real_action_dim
        self.image_keys = list(image_keys)
        self.max_length = max_length
        self._step = 0
        self._last_response = None

        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        self.n_action_steps = n_action_steps
        self._action_queue: deque = deque(maxlen=n_action_steps)

        self._bitvla_proprio_norm = None
        self._bitvla_unnorm_key   = None
        if arch == "bitvla":
            if stats_json:
                stats_path = Path(stats_json)
            elif (Path(tokenizer_name) / "dataset_statistics.json").exists():

                stats_path = Path(tokenizer_name) / "dataset_statistics.json"
            else:

                from huggingface_hub import hf_hub_download
                stats_path = Path(hf_hub_download(tokenizer_name, "dataset_statistics.json"))
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"BitVLA dataset_statistics.json not found at {stats_path}. "
                    f"Pass --stats-json or point --tokenizer at a ckpt dir that has it.")
            blob = json.loads(stats_path.read_text())
            key = bitvla_unnorm_key
            if key is None:
                if len(blob) != 1:
                    raise ValueError(
                        f"{stats_path} has multiple keys {list(blob)}; pass "
                        f"--bitvla-unnorm-key explicitly.")
                key = next(iter(blob.keys()))
            self._bitvla_unnorm_key = key
            p = blob[key]["proprio"]
            q01 = np.asarray(p["q01"], dtype=np.float32)
            q99 = np.asarray(p["q99"], dtype=np.float32)
            mask = (np.ones_like(q01, dtype=bool)
                    if p.get("mask", None) is None
                    else np.asarray(p["mask"], dtype=bool))
            def _norm(x: np.ndarray, q01=q01, q99=q99, mask=mask) -> np.ndarray:
                y = x.astype(np.float32)
                out = np.where(mask, 2.0 * (y - q01) / (q99 - q01 + 1e-8) - 1.0, y)
                return np.clip(out, -1.0, 1.0).astype(np.float32)
            self._bitvla_proprio_norm = _norm
            print(f"vla-cpp-direct[arch=bitvla]: proprio normalizer "
                  f"BOUNDS_Q99 via {stats_path}::{key}.proprio", flush=True)

        self._oft_proprio_norm = None
        if arch == "openvla_oft":
            if stats_json:
                stats_path = Path(stats_json)
            else:
                stats_path = Path(tokenizer_name) / "dataset_statistics.json"
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"OpenVLA-OFT dataset_statistics.json not found at {stats_path}. "
                    f"Pass --stats-json or point --tokenizer at the ckpt dir.")
            blob = json.loads(stats_path.read_text())
            key = os.environ.get("VLA_OPENVLA_OFT_UNNORM_KEY", "libero_object_no_noops")
            if key not in blob:
                raise ValueError(f"{stats_path} has no suite {key!r}; available {list(blob)}")
            p = blob[key]["proprio"]
            q01 = np.asarray(p["q01"], dtype=np.float32)
            q99 = np.asarray(p["q99"], dtype=np.float32)
            mask = (np.ones_like(q01, dtype=bool)
                    if p.get("mask", None) is None else np.asarray(p["mask"], dtype=bool))
            def _oft_norm(x, q01=q01, q99=q99, mask=mask):
                y = x.astype(np.float32)
                out = np.where(mask, 2.0 * (y - q01) / (q99 - q01 + 1e-8) - 1.0, y)
                return np.clip(out, -1.0, 1.0).astype(np.float32)
            self._oft_proprio_norm = _oft_norm
            print(f"vla-cpp-direct[arch=openvla_oft]: proprio BOUNDS_Q99 via "
                  f"{stats_path}::{key}.proprio", flush=True)

        self._pi05_state_q01 = None
        self._pi05_state_q99 = None
        if arch == "pi05":
            if stats_json:
                stats_path = Path(stats_json)
            else:
                from huggingface_hub import hf_hub_download
                stats_path = Path(hf_hub_download(
                    repo_id="lerobot/libero", filename="meta/stats.json",
                    repo_type="dataset"))
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"π0.5 state stats not found at {stats_path}. Pass --stats-json "
                    f"<LIBERO meta/stats.json>.")
            blob = json.loads(stats_path.read_text())
            st = blob["observation.state"]
            if "q01" not in st or "q99" not in st:
                raise ValueError(
                    f"{stats_path}::observation.state lacks q01/q99 (π0.5 uses QUANTILES). "
                    f"Use a meta/stats.json with quantile stats.")
            self._pi05_state_q01 = np.asarray(st["q01"], dtype=np.float32).reshape(-1)
            self._pi05_state_q99 = np.asarray(st["q99"], dtype=np.float32).reshape(-1)
            print(f"vla-cpp-direct[arch=pi05]: state QUANTILES via "
                  f"{stats_path}::observation.state ({self._pi05_state_q01.shape[0]}-D)",
                  flush=True)


        self._gr00t_action_unnorm = None
        self._gr00t_state_norm = None
        if arch == "gr00t_n1_7" and stats_json is not None:
            stats_path = Path(stats_json)
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"--gr00t stats JSON not found at {stats_path}")
            blob = json.loads(stats_path.read_text())
            key = bitvla_unnorm_key
            if key is None:

                if "libero_sim" in blob:
                    key = "libero_sim"
                elif len(blob) == 1:
                    key = next(iter(blob.keys()))
                else:
                    raise ValueError(
                        f"{stats_path} has top-level keys {list(blob)}; pass "
                        f"--bitvla-unnorm-key explicitly for arch=gr00t_n1_7.")
            if key not in blob:
                raise KeyError(f"embodiment {key!r} not in {stats_path}; have {list(blob)}")
            modalities = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            action_stats = blob[key]["action"]
            q01 = np.array([action_stats[m]["q01"][0] for m in modalities], dtype=np.float32)
            q99 = np.array([action_stats[m]["q99"][0] for m in modalities], dtype=np.float32)
            rng = (q99 - q01).astype(np.float32)
            def _unnorm(chunk_132: np.ndarray, q01=q01, q99=q99, rng=rng) -> np.ndarray:

                norm = chunk_132[..., :7].astype(np.float32)

                norm = np.clip(norm, -1.0, 1.0)

                raw = (norm + 1.0) * 0.5 * rng[None, :] + q01[None, :]

                return raw.astype(np.float32)
            self._gr00t_action_unnorm = _unnorm
            print(f"vla-cpp-direct[arch=gr00t_n1_7]: action unnormalizer "
                  f"(q01/q99 + clip + gripper flip) via {stats_path}::{key}.action "
                  f"[modalities={modalities}, q01={q01.tolist()}, q99={q99.tolist()}]",
                  flush=True)

            state_stats = blob[key]["state"]
            s_q01_parts, s_q99_parts = [], []
            for m, dim in zip(self._GR00T_STATE_KEYS, self._GR00T_STATE_DIMS):
                if m not in state_stats:
                    raise KeyError(f"state modality {m!r} not in {stats_path}::{key}.state")
                q01_m = np.asarray(state_stats[m]["q01"], dtype=np.float32)
                q99_m = np.asarray(state_stats[m]["q99"], dtype=np.float32)
                if q01_m.size != dim or q99_m.size != dim:
                    raise ValueError(f"state.{m}: stats dim {q01_m.size}/{q99_m.size} != expected {dim}")
                s_q01_parts.append(q01_m); s_q99_parts.append(q99_m)
            s_q01 = np.concatenate(s_q01_parts)
            s_q99 = np.concatenate(s_q99_parts)
            s_rng = (s_q99 - s_q01).astype(np.float32)
            def _state_norm(state_8d: np.ndarray, q01=s_q01, q99=s_q99, rng=s_rng) -> np.ndarray:

                norm = 2.0 * (state_8d - q01) / np.where(rng > 1e-8, rng, 1.0) - 1.0
                return np.clip(norm, -1.0, 1.0).astype(np.float32)
            self._gr00t_state_norm = _state_norm
            print(f"vla-cpp-direct[arch=gr00t_n1_7]: state normalizer "
                  f"(q01/q99 + clip) via {stats_path}::{key}.state "
                  f"[q01={s_q01.tolist()}, q99={s_q99.tolist()}]", flush=True)

        if arch == "gr00t_n1_6" and stats_json is not None:
            stats_path = Path(stats_json)
            if not stats_path.exists():
                raise FileNotFoundError(f"--gr00t stats JSON not found at {stats_path}")
            blob = json.loads(stats_path.read_text())
            key = bitvla_unnorm_key
            if key is None:
                for cand in ("libero_panda", "libero_sim"):
                    if cand in blob: key = cand; break
                if key is None and len(blob) == 1:
                    key = next(iter(blob.keys()))
                if key is None:
                    raise ValueError(
                        f"{stats_path} has top-level keys {list(blob)}; pass "
                        f"--bitvla-unnorm-key explicitly for arch=gr00t_n1_6.")
            if key not in blob:
                raise KeyError(f"embodiment {key!r} not in {stats_path}; have {list(blob)}")

            modalities = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            action_stats = blob[key]["action"]
            a_min = np.array([action_stats[m]["min"][0] for m in modalities], dtype=np.float32)
            a_max = np.array([action_stats[m]["max"][0] for m in modalities], dtype=np.float32)
            a_rng = (a_max - a_min).astype(np.float32)
            def _unnorm_n16(chunk_full: np.ndarray, mn=a_min, mx=a_max, rng=a_rng) -> np.ndarray:

                norm = chunk_full[..., :7].astype(np.float32)

                norm = np.clip(norm, -1.0, 1.0)
                raw = (norm + 1.0) * 0.5 * rng[None, :] + mn[None, :]

                return raw[:16].astype(np.float32)
            self._gr00t_action_unnorm = _unnorm_n16
            print(f"vla-cpp-direct[arch=gr00t_n1_6]: action unnormalizer "
                  f"(min/max + clip) via {stats_path}::{key}.action "
                  f"[modalities={modalities}, min={a_min.tolist()}, max={a_max.tolist()}]",
                  flush=True)

            state_stats = blob[key]["state"]
            s_min_parts, s_max_parts = [], []
            for m, dim in zip(self._GR00T_STATE_KEYS, self._GR00T_STATE_DIMS):
                if m not in state_stats:
                    raise KeyError(f"state modality {m!r} not in {stats_path}::{key}.state")
                mn = np.asarray(state_stats[m]["min"], dtype=np.float32)
                mx = np.asarray(state_stats[m]["max"], dtype=np.float32)
                if mn.size != dim or mx.size != dim:
                    raise ValueError(f"state.{m}: stats dim {mn.size}/{mx.size} != expected {dim}")
                s_min_parts.append(mn); s_max_parts.append(mx)
            s_min = np.concatenate(s_min_parts)
            s_max = np.concatenate(s_max_parts)
            s_rng = (s_max - s_min).astype(np.float32)
            def _state_norm_n16(state_8d: np.ndarray, mn=s_min, mx=s_max, rng=s_rng) -> np.ndarray:
                norm = 2.0 * (state_8d - mn) / np.where(rng > 1e-8, rng, 1.0) - 1.0
                return np.clip(norm, -1.0, 1.0).astype(np.float32)
            self._gr00t_state_norm = _state_norm_n16
            print(f"vla-cpp-direct[arch=gr00t_n1_6]: state normalizer "
                  f"(min/max + clip) via {stats_path}::{key}.state "
                  f"[min={s_min.tolist()}, max={s_max.tolist()}]", flush=True)

        if arch == "gr00t_n1_5" and stats_json is not None:
            stats_path = Path(stats_json)
            if not stats_path.exists():
                raise FileNotFoundError(f"--gr00t stats JSON not found at {stats_path}")
            blob = json.loads(stats_path.read_text())
            key = bitvla_unnorm_key
            if key is None:
                for cand in ("new_embodiment", "libero_sim", "libero_panda"):
                    if cand in blob:
                        key = cand
                        break
                if key is None and len(blob) == 1:
                    key = next(iter(blob.keys()))
                if key is None:
                    raise ValueError(
                        f"{stats_path} has top-level keys {list(blob)}; pass "
                        f"--bitvla-unnorm-key explicitly for arch=gr00t_n1_5.")
            if key not in blob:
                raise KeyError(f"embodiment {key!r} not in {stats_path}; have {list(blob)}")
            a_min = np.asarray(blob[key]["action"]["min"], dtype=np.float32)
            a_max = np.asarray(blob[key]["action"]["max"], dtype=np.float32)
            a_rng = (a_max - a_min).astype(np.float32)
            def _unnorm_n15(chunk_full, mn=a_min, mx=a_max, rng=a_rng):

                norm = chunk_full[..., : mn.size].astype(np.float32)
                raw = (norm + 1.0) * 0.5 * rng[None, :] + mn[None, :]
                return raw[:16].astype(np.float32)
            self._gr00t_action_unnorm = _unnorm_n15
            print(f"vla-cpp-direct[arch=gr00t_n1_5]: action unnormalizer "
                  f"(flat min/max, no clip/flip) via {stats_path}::{key}.action "
                  f"[min={a_min.tolist()}, max={a_max.tolist()}]", flush=True)
            s_min = np.asarray(blob[key]["state"]["min"], dtype=np.float32)
            s_max = np.asarray(blob[key]["state"]["max"], dtype=np.float32)
            s_rng = (s_max - s_min).astype(np.float32)
            def _state_norm_n15(state_vec, mn=s_min, mx=s_max, rng=s_rng):

                return (2.0 * (state_vec - mn) / np.where(rng > 1e-8, rng, 1.0) - 1.0).astype(np.float32)
            self._gr00t_state_norm = _state_norm_n15
            print(f"vla-cpp-direct[arch=gr00t_n1_5]: state normalizer "
                  f"(flat min/max, no clip) via {stats_path}::{key}.state "
                  f"[min={s_min.tolist()}, max={s_max.tolist()}]", flush=True)

    def ping(self) -> bool:

        return True

    def reset(self) -> None:

        self._action_queue.clear()

    def get_action(self, observations: dict[str, Any]) -> np.ndarray:

        if not self._action_queue:
            if self.arch == "bitvla":
                chunk = self._predict_chunk_bitvla(observations)
            elif self.arch == "vla_adapter":
                chunk = self._predict_chunk_vla_adapter(observations)
            elif self.arch == "openvla_oft":
                chunk = self._predict_chunk_openvla_oft(observations)
            else:
                chunk = self._predict_chunk(observations)
            for row in chunk[: self.n_action_steps, : self.real_action_dim]:
                self._action_queue.append(np.ascontiguousarray(row, dtype=np.float32))
        return self._action_queue.popleft()

    def _predict_chunk(self, observations: dict[str, Any]) -> np.ndarray:

        if self.arch == "evo1":
            return self._predict_chunk_evo1(observations)
        if self.arch == "gr00t_n1_7":
            return self._predict_chunk_gr00t_n1_7(observations)
        if self.arch == "gr00t_n1_5":
            return self._predict_chunk_gr00t_n1_5(observations)
        if self.arch == "gr00t_n1_6":
            return self._predict_chunk_gr00t_n1_6(observations)
        if self.arch == "pi05":
            return self._predict_chunk_pi05(observations)

        images_f32: list[np.ndarray] = []
        for key in self.image_keys:
            if key not in observations:
                raise KeyError(
                    f"image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(
                    f"{key}: expected CHW float32 [3, H, W], got {img.shape}")
            img = _resize_with_pad(img, self.image_size, self.image_size, pad_value=0.0)
            img_hwc = np.transpose(img, (1, 2, 0))
            images_f32.append(np.ascontiguousarray(img_hwc, dtype=np.float32))

        s = observations["observation.state"]
        if isinstance(s, torch.Tensor):
            s = s.numpy()
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[:s.shape[0]] = s

        task = observations.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        if not task.endswith("\n"):
            task = task + "\n"
        toks = self.tok(task, padding=False, truncation=True,
                        max_length=self.max_length, return_tensors="np")
        lang = toks["input_ids"][0].astype(np.int32)

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_f32:
            ip = req.images.add()
            ip.encoding = self.pb.Image.F32_RGB_01
            ip.height = img.shape[0]
            ip.width  = img.shape[1]
            ip.data   = img.tobytes()
        req.lang_tokens.extend(lang.tolist())
        req.state.extend(state_padded.tolist())

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")

        self._last_response = resp

        return (np.array(resp.action_chunk, dtype=np.float32)
                  .reshape(resp.chunk_size, resp.action_dim))

    def _predict_chunk_pi05(self, observations: dict[str, Any]) -> np.ndarray:
        if self._pi05_state_q01 is None:
            raise RuntimeError("arch=pi05 needs state stats; pass --stats-json "
                               "<LIBERO meta/stats.json> (or allow the lerobot/libero fetch).")
        images_f32: list[np.ndarray] = []
        for key in self.image_keys:
            if key not in observations:
                raise KeyError(
                    f"image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"{key}: expected CHW float32 [3, H, W], got {img.shape}")
            img = _resize_with_pad(img, self.image_size, self.image_size, pad_value=0.0)
            img_hwc = np.transpose(img, (1, 2, 0))
            images_f32.append(np.ascontiguousarray(img_hwc, dtype=np.float32))

        s = observations["observation.state"]
        if isinstance(s, torch.Tensor):
            s = s.numpy()
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        d = self._pi05_state_q01.shape[0]
        normed = 2.0 * (s[:d] - self._pi05_state_q01) / (self._pi05_state_q99 - self._pi05_state_q01) - 1.0
        bins = np.linspace(-1.0, 1.0, 256 + 1)[:-1]
        disc = np.digitize(normed, bins=bins) - 1

        task = observations.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, disc.tolist()))
        prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
        toks = self.tok(prompt, padding=False, truncation=True,
                        max_length=self.max_length, return_tensors="np")
        lang = toks["input_ids"][0].astype(np.int32)

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_f32:
            ip = req.images.add()
            ip.encoding = self.pb.Image.F32_RGB_01
            ip.height = img.shape[0]
            ip.width  = img.shape[1]
            ip.data   = img.tobytes()
        req.lang_tokens.extend(lang.tolist())
        req.state.extend([0.0] * self.max_state_dim)

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp
        return (np.array(resp.action_chunk, dtype=np.float32)
                  .reshape(resp.chunk_size, resp.action_dim))

    _EVO1_IMG_START         = "<img>"
    _EVO1_IMG_END           = "</img>"
    _EVO1_IMG_CTX           = "<IMG_CONTEXT>"
    _EVO1_NUM_IMAGE_TOKEN   = 256
    _EVO1_MAX_TEXT_LENGTH   = 1024

    def _predict_chunk_evo1(self, observations: dict[str, Any]) -> np.ndarray:

        import cv2 as _cv2
        images_raw = observations["image"]
        image_mask = observations.get("image_mask", [1] * len(images_raw))
        if len(image_mask) != len(images_raw):
            raise ValueError(
                f"evo1: image_mask len {len(image_mask)} != images len {len(images_raw)}")
        images_u8: list[np.ndarray] = []
        for i, im in enumerate(images_raw):
            arr = im.numpy() if isinstance(im, torch.Tensor) else np.asarray(im)
            arr = np.asarray(arr, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(
                    f"evo1 image[{i}]: expected HWC u8 with C=3, got {arr.shape} dtype={arr.dtype}")
            arr = _cv2.resize(arr, (self.image_size, self.image_size))
            arr = _cv2.cvtColor(arr, _cv2.COLOR_BGR2RGB)
            images_u8.append(np.ascontiguousarray(arr))

        s = observations["state"]
        if isinstance(s, torch.Tensor):
            s = s.numpy()
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        if s.size > self.max_state_dim:
            raise ValueError(
                f"evo1: state has {s.size} dims, exceeds max_state_dim={self.max_state_dim}")
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[:s.size] = s

        instr = observations.get("prompt", "") or ""
        if isinstance(instr, bytes):
            instr = instr.decode()
        n_views = len(images_u8)
        image_block = (self._EVO1_IMG_START
                       + self._EVO1_IMG_CTX * self._EVO1_NUM_IMAGE_TOKEN
                       + self._EVO1_IMG_END)
        prompt = "".join(f"Image-{i+1}: {image_block}\n" for i in range(n_views)) + instr.strip()

        model_inputs = self.tok(prompt, return_tensors="np",
                                padding="max_length", truncation=True,
                                max_length=self._EVO1_MAX_TEXT_LENGTH)
        input_ids_full = model_inputs["input_ids"][0].astype(np.int32)
        attn_mask      = model_inputs["attention_mask"][0].astype(np.int32).copy()
        n_real = int((attn_mask > 0).sum())

        img_ctx_id = self.tok.convert_tokens_to_ids(self._EVO1_IMG_CTX)
        if img_ctx_id is None or img_ctx_id == self.tok.unk_token_id:
            raise RuntimeError(
                f"evo1: tokenizer has no '{self._EVO1_IMG_CTX}' token - "
                f"check that you're loading OpenGVLab/InternVL3-1B with trust_remote_code=True")
        ctx_positions = np.where(input_ids_full == img_ctx_id)[0]
        expected = n_views * self._EVO1_NUM_IMAGE_TOKEN
        if ctx_positions.size != expected:
            raise RuntimeError(
                f"evo1: expected {expected} IMG_CTX positions in input_ids, "
                f"got {ctx_positions.size}. Prompt was truncated to max_length={self._EVO1_MAX_TEXT_LENGTH}? "
                f"instruction len={len(instr)}")
        cursor = 0
        for i in range(n_views):
            slot = ctx_positions[cursor:cursor + self._EVO1_NUM_IMAGE_TOKEN]
            if not image_mask[i]:
                attn_mask[slot] = 0
            cursor += self._EVO1_NUM_IMAGE_TOKEN

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_u8:
            ip = req.images.add()
            ip.encoding = self.pb.Image.RGB_U8
            ip.height = img.shape[0]
            ip.width  = img.shape[1]
            ip.data   = img.tobytes()

        req.lang_tokens.extend(input_ids_full[:n_real].tolist())
        req.state.extend(state_padded.tolist())
        req.attention_mask.extend(attn_mask.tolist())

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp
        return (np.array(resp.action_chunk, dtype=np.float32)
                  .reshape(resp.chunk_size, resp.action_dim))

    def _predict_chunk_bitvla(self, observations: dict[str, Any]) -> np.ndarray:

        images_u8: list[np.ndarray] = []
        for key in self.image_keys[:BITVLA_N_VIEWS]:
            if key not in observations:
                raise KeyError(f"image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"{key}: expected CHW float [3, H, W], got {img.shape}")
            img_u8 = np.clip(np.transpose(img, (1, 2, 0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)

            if img_u8.shape[0] != self.image_size or img_u8.shape[1] != self.image_size:
                img_u8 = np.array(Image.fromarray(img_u8, mode="RGB").resize(
                    (self.image_size, self.image_size), resample=Image.LANCZOS),
                    dtype=np.uint8)

            h, w = img_u8.shape[:2]
            s = 0.9 ** 0.5
            new_h, new_w = int(round(h * s)), int(round(w * s))
            off_h, off_w = (h - new_h) // 2, (w - new_w) // 2
            cropped = img_u8[off_h:off_h + new_h, off_w:off_w + new_w]
            img_u8 = np.array(Image.fromarray(cropped, mode="RGB").resize(
                (w, h), resample=Image.BILINEAR), dtype=np.uint8)
            images_u8.append(np.ascontiguousarray(img_u8, dtype=np.uint8))

        s = observations["observation.state"]
        if isinstance(s, torch.Tensor):
            s = s.numpy()
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        state_norm = self._bitvla_proprio_norm(s)

        task = observations.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        user_content = (BITVLA_IMAGE_PAD_TOKEN * (BITVLA_N_VIEWS * BITVLA_N_PATCHES_PER_VIEW)
                        + BITVLA_PROPRIO_PAD_TOKEN
                        + BITVLA_USER_PROMPT_TPL.format(instruction=task.lower()))
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True)
        lang_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_u8:
            ip = req.images.add()
            ip.encoding = self.pb.Image.RGB_U8
            ip.height = img.shape[0]
            ip.width  = img.shape[1]
            ip.data   = img.tobytes()
        req.lang_tokens.extend(int(t) for t in lang_ids)
        req.state.extend(float(x) for x in state_norm)

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp

        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))

        chunk[..., -1] = np.sign(2.0 * chunk[..., -1] - 1.0)
        chunk[..., -1] *= -1.0
        return chunk

    def _predict_chunk_vla_adapter(self, observations: dict[str, Any]) -> np.ndarray:
        images_u8: list[np.ndarray] = []
        for key in self.image_keys[:VLA_ADAPTER_N_VIEWS]:
            if key not in observations:
                raise KeyError(f"image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"{key}: expected CHW float [3, H, W], got {img.shape}")
            img_u8 = np.clip(np.transpose(img, (1, 2, 0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            if img_u8.shape[0] != self.image_size or img_u8.shape[1] != self.image_size:
                img_u8 = np.array(Image.fromarray(img_u8, mode="RGB").resize(
                    (self.image_size, self.image_size), resample=Image.LANCZOS), dtype=np.uint8)
            h, w = img_u8.shape[:2]
            s = 0.9 ** 0.5
            new_h, new_w = int(round(h * s)), int(round(w * s))
            off_h, off_w = (h - new_h) // 2, (w - new_w) // 2
            cropped = img_u8[off_h:off_h + new_h, off_w:off_w + new_w]
            img_u8 = np.array(Image.fromarray(cropped, mode="RGB").resize(
                (w, h), resample=Image.BILINEAR), dtype=np.uint8)
            images_u8.append(np.ascontiguousarray(img_u8, dtype=np.uint8))

        st = observations["observation.state"]
        if isinstance(st, torch.Tensor):
            st = st.numpy()
        st = np.asarray(st, dtype=np.float32).reshape(-1)[:8]

        task = observations.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        prompt = VLA_ADAPTER_PROMPT_TPL.format(instruction=task.lower())
        lang_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_u8:
            ip = req.images.add()
            ip.encoding = self.pb.Image.RGB_U8
            ip.height = img.shape[0]
            ip.width = img.shape[1]
            ip.data = img.tobytes()
        req.lang_tokens.extend(int(t) for t in lang_ids)
        req.state.extend(float(x) for x in st)

        self.sock.send(req.SerializeToString())
        resp = self.pb.PredictResponse()
        resp.ParseFromString(self.sock.recv())
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp

        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))
        chunk[..., -1] = np.sign(2.0 * chunk[..., -1] - 1.0)
        chunk[..., -1] *= -1.0
        return chunk

    def _predict_chunk_openvla_oft(self, observations: dict[str, Any]) -> np.ndarray:
        images_u8: list[np.ndarray] = []
        for key in self.image_keys[:OPENVLA_OFT_N_VIEWS]:
            if key not in observations:
                raise KeyError(f"image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"{key}: expected CHW float [3, H, W], got {img.shape}")
            img_u8 = np.clip(np.transpose(img, (1, 2, 0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            if img_u8.shape[0] != self.image_size or img_u8.shape[1] != self.image_size:
                img_u8 = np.array(Image.fromarray(img_u8, mode="RGB").resize(
                    (self.image_size, self.image_size), resample=Image.LANCZOS), dtype=np.uint8)
            h, w = img_u8.shape[:2]
            s = 0.9 ** 0.5
            new_h, new_w = int(round(h * s)), int(round(w * s))
            off_h, off_w = (h - new_h) // 2, (w - new_w) // 2
            cropped = img_u8[off_h:off_h + new_h, off_w:off_w + new_w]
            img_u8 = np.array(Image.fromarray(cropped, mode="RGB").resize(
                (w, h), resample=Image.BILINEAR), dtype=np.uint8)
            images_u8.append(np.ascontiguousarray(img_u8, dtype=np.uint8))

        st = observations["observation.state"]
        if isinstance(st, torch.Tensor):
            st = st.numpy()
        st = np.asarray(st, dtype=np.float32).reshape(-1)[:8]
        st = self._oft_proprio_norm(st)

        task = observations.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        prompt = OPENVLA_OFT_PROMPT_TPL.format(instruction=task.lower())
        lang_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        if not lang_ids or lang_ids[-1] != OPENVLA_OFT_EMPTY_TOKEN:
            lang_ids = list(lang_ids) + [OPENVLA_OFT_EMPTY_TOKEN]

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_u8:
            ip = req.images.add()
            ip.encoding = self.pb.Image.RGB_U8
            ip.height = img.shape[0]
            ip.width = img.shape[1]
            ip.data = img.tobytes()
        req.lang_tokens.extend(int(t) for t in lang_ids)
        req.state.extend(float(x) for x in st)

        self.sock.send(req.SerializeToString())
        resp = self.pb.PredictResponse()
        resp.ParseFromString(self.sock.recv())
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp

        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))
        chunk[..., -1] = np.sign(2.0 * chunk[..., -1] - 1.0)
        chunk[..., -1] *= -1.0
        return chunk


    _GR00T_IMG_PAD_TOKEN  = "<|image_pad|>"
    _GR00T_IMG_PAD_ID     = 151655
    _GR00T_N_TOK_PER_VIEW = 64
    _GR00T_TARGET_SIZE    = 256
    _GR00T_CROP_FRACTION  = 0.95
    _GR00T_SHORTEST_EDGE  = 256

    @staticmethod
    def _gr00t_eval_image_transform(img_u8_hwc: np.ndarray, target_size: int,
                                    shortest_edge: int, crop_fraction: float) -> np.ndarray:

        import cv2
        img = img_u8_hwc
        h, w = img.shape[:2]

        if h != w:
            side = max(h, w)
            pad = np.zeros((side, side, 3), dtype=img.dtype)
            pad[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
            img = pad; h, w = side, side

        short = min(h, w)
        if short > shortest_edge:
            scale = shortest_edge / short
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        nh, nw = int(round(h * crop_fraction)), int(round(w * crop_fraction))
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        img = img[y0:y0 + nh, x0:x0 + nw]; h, w = img.shape[:2]

        short = min(h, w)
        if short != shortest_edge:
            scale = shortest_edge / short
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)

        if img.shape[:2] != (target_size, target_size):
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img, dtype=np.uint8)

    _GR00T_STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    _GR00T_STATE_DIMS = (1,   1,   1,   1,      1,       1,    2)

    def _predict_chunk_gr00t_n1_7(self, observations: dict[str, Any]) -> np.ndarray:

        import re

        images_f32: list[np.ndarray] = []
        for key in ("video.image", "video.wrist_image"):
            if key not in observations:
                raise KeyError(
                    f"gr00t_n1_7 image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.uint8)

            while img.ndim > 3:
                img = img[0]
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"{key}: expected HWC u8 [H, W, 3], got {img.shape} dtype={img.dtype}")
            img_u8 = self._gr00t_eval_image_transform(
                img, self._GR00T_TARGET_SIZE, self._GR00T_SHORTEST_EDGE, self._GR00T_CROP_FRACTION)

            img_f32 = np.ascontiguousarray(img_u8.astype(np.float32) / 255.0)
            images_f32.append(img_f32)

        state_chunks = []
        for key, dim in zip(self._GR00T_STATE_KEYS, self._GR00T_STATE_DIMS):
            mk = f"state.{key}"
            if mk not in observations:
                raise KeyError(f"gr00t_n1_7 state key '{mk}' missing; got {list(observations.keys())}")
            v = observations[mk]
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size != dim:

                raise ValueError(f"gr00t_n1_7 state '{mk}': expected {dim}-d, got {v.size}-d")
            state_chunks.append(v)
        state_raw = np.concatenate(state_chunks, axis=0).astype(np.float32)
        if self._gr00t_state_norm is not None:
            state_raw = self._gr00t_state_norm(state_raw)
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[: state_raw.size] = state_raw

        task_field = observations.get("task", "")
        if isinstance(task_field, tuple):
            task_field = task_field[0] if task_field else ""
        if isinstance(task_field, bytes):
            task_field = task_field.decode()
        task = task_field
        language = re.sub(r"[^\w\s]", "", task.lower())
        n_views = len(images_f32)
        conv = [{"role": "user", "content": [
            *[{"type": "image"} for _ in range(n_views)],
            {"type": "text", "text": language},
        ]}]
        text = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        expanded: list[int] = []
        for tid in ids:
            if tid == self._GR00T_IMG_PAD_ID:
                expanded.extend([self._GR00T_IMG_PAD_ID] * self._GR00T_N_TOK_PER_VIEW)
            else:
                expanded.append(tid)
        lang = np.array(expanded, dtype=np.int32)

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_f32:
            ip = req.images.add()
            ip.encoding = self.pb.Image.F32_RGB_01
            ip.height = img.shape[0]
            ip.width  = img.shape[1]
            ip.data   = img.tobytes()
        req.lang_tokens.extend(int(t) for t in lang)
        req.state.extend(float(x) for x in state_padded)

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp
        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))
        if self._gr00t_action_unnorm is not None:

            chunk = self._gr00t_action_unnorm(chunk)
        return chunk

    _GR00T_N16_IMG_PAD_ID     = 151669
    _GR00T_N16_IMG_START_ID   = 151670
    _GR00T_N16_IMG_END_ID     = 151671
    _GR00T_N16_N_TOK_PER_VIEW = 64

    _GR00T_N15_N_TOK_PER_VIEW = 256
    _GR00T_N15_IMG_CTX_ID     = 151669
    _GR00T_N16_ACTION_HORIZON = 16

    @staticmethod
    def _gr00t_n16_image_transform(img_u8_hwc: np.ndarray, target_size: int,
                                   crop_fraction: float) -> np.ndarray:

        import cv2
        img = img_u8_hwc
        h, w = img.shape[:2]
        if h != w:
            side = max(h, w)
            pad = np.zeros((side, side, 3), dtype=img.dtype)
            pad[(side - h) // 2:(side - h) // 2 + h,
                (side - w) // 2:(side - w) // 2 + w] = img
            img = pad; h, w = side, side
        nh, nw = int(round(h * crop_fraction)), int(round(w * crop_fraction))
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        img = img[y0:y0 + nh, x0:x0 + nw]
        if img.shape[:2] != (target_size, target_size):
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img, dtype=np.uint8)

    def _predict_chunk_gr00t_n1_6(self, observations: dict[str, Any]) -> np.ndarray:

        import re

        images_f32: list[np.ndarray] = []
        for key in ("video.image", "video.wrist_image"):
            if key not in observations:
                raise KeyError(
                    f"gr00t_n1_6 image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.uint8)
            while img.ndim > 3:
                img = img[0]
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"{key}: expected HWC u8 [H, W, 3], got {img.shape} dtype={img.dtype}")

            img_u8 = self._gr00t_n16_image_transform(img, self.image_size, self._GR00T_CROP_FRACTION)
            images_f32.append(np.ascontiguousarray(img_u8.astype(np.float32) / 255.0))

        state_chunks = []
        for key, dim in zip(self._GR00T_STATE_KEYS, self._GR00T_STATE_DIMS):
            mk = f"state.{key}"
            if mk not in observations:
                raise KeyError(f"gr00t_n1_6 state key '{mk}' missing; got {list(observations.keys())}")
            v = observations[mk]
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size != dim:
                raise ValueError(f"gr00t_n1_6 state '{mk}': expected {dim}-d, got {v.size}-d")
            state_chunks.append(v)
        state_raw = np.concatenate(state_chunks, axis=0).astype(np.float32)
        if self._gr00t_state_norm is not None:
            state_raw = self._gr00t_state_norm(state_raw)
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[: state_raw.size] = state_raw

        task_field = observations.get("task", "")
        if isinstance(task_field, tuple):
            task_field = task_field[0] if task_field else ""
        if isinstance(task_field, bytes):
            task_field = task_field.decode()

        language = re.sub(r"[^\w\s]", "", task_field.lower())

        n_views = len(images_f32)
        img_block = (lambda n:
            f"<image {n}><img>"
            + "<IMG_CONTEXT>" * self._GR00T_N16_N_TOK_PER_VIEW
            + "</img>")
        user_content = language + "".join(img_block(i + 1) for i in range(n_views))
        conv = [{"role": "user", "content": user_content}]
        text = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]

        n_slots = sum(1 for t in ids if t == self._GR00T_N16_IMG_PAD_ID)
        if n_slots != n_views * self._GR00T_N16_N_TOK_PER_VIEW:
            raise RuntimeError(
                f"gr00t_n1_6: expected {n_views * self._GR00T_N16_N_TOK_PER_VIEW} "
                f"<IMG_CONTEXT> slots in tokenized prompt, got {n_slots}. "
                f"Check that <IMG_CONTEXT> / <img> / </img> are special tokens in the tokenizer.")
        lang = np.array(ids, dtype=np.int32)

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_f32:
            ip = req.images.add()
            ip.encoding = self.pb.Image.F32_RGB_01
            ip.height = img.shape[0]
            ip.width  = img.shape[1]
            ip.data   = img.tobytes()
        req.lang_tokens.extend(int(t) for t in lang)
        req.state.extend(float(x) for x in state_padded)

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp
        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))
        if self._gr00t_action_unnorm is not None:

            chunk = self._gr00t_action_unnorm(chunk)
        return chunk

    def _predict_chunk_gr00t_n1_5(self, observations: dict[str, Any]) -> np.ndarray:

        import cv2 as _cv2

        images_u8: list[np.ndarray] = []
        for key in ("video.image", "video.wrist_image"):
            if key not in observations:
                raise KeyError(
                    f"gr00t_n1_5 image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.uint8)
            while img.ndim > 3:
                img = img[0]
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"{key}: expected HWC u8 [H, W, 3], got {img.shape}")
            img = _cv2.resize(img, (self.image_size, self.image_size), interpolation=_cv2.INTER_AREA)
            images_u8.append(np.ascontiguousarray(img))

        state_chunks = []
        for key, dim in zip(self._GR00T_STATE_KEYS, self._GR00T_STATE_DIMS):
            mk = f"state.{key}"
            if mk not in observations:
                raise KeyError(f"gr00t_n1_5 state key '{mk}' missing; got {list(observations.keys())}")
            v = observations[mk]
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size != dim:
                raise ValueError(f"gr00t_n1_5 state '{mk}': expected {dim}-d, got {v.size}-d")
            state_chunks.append(v)
        state_raw = np.concatenate(state_chunks, axis=0).astype(np.float32)
        if self._gr00t_state_norm is not None:
            state_raw = self._gr00t_state_norm(state_raw)
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[: state_raw.size] = state_raw

        task_field = observations.get("task", "")
        if isinstance(task_field, tuple):
            task_field = task_field[0] if task_field else ""
        if isinstance(task_field, bytes):
            task_field = task_field.decode()
        n_views = len(images_u8)

        sys_turn = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        img_part = "".join(
            f"<image {i + 1}><img>" + "<IMG_CONTEXT>" * self._GR00T_N15_N_TOK_PER_VIEW + "</img>"
            for i in range(n_views))
        text = (sys_turn + "<|im_start|>user\n" + img_part + str([task_field])
                + "<|im_end|>\n<|im_start|>assistant\n")
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        n_ctx = sum(1 for t in ids if t == self._GR00T_N15_IMG_CTX_ID)
        if n_ctx != n_views * self._GR00T_N15_N_TOK_PER_VIEW:
            raise RuntimeError(
                f"gr00t_n1_5: expected {n_views * self._GR00T_N15_N_TOK_PER_VIEW} "
                f"<IMG_CONTEXT> in tokenized prompt, got {n_ctx}.")

        lang = np.array(ids, dtype=np.int32)

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img in images_u8:
            ip = req.images.add()
            ip.encoding = self.pb.Image.RGB_U8
            ip.height = img.shape[0]
            ip.width  = img.shape[1]
            ip.data   = img.tobytes()
        req.lang_tokens.extend(int(t) for t in lang)
        req.state.extend(float(x) for x in state_padded)

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        self._last_response = resp
        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))
        if self._gr00t_action_unnorm is not None:
            chunk = self._gr00t_action_unnorm(chunk)
        return chunk

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

class VlaCppSimplerGr00tClient:

    arch_type = "gr00t"

    _STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")
    _ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    _ACTION_MEANSTD_KEYS = frozenset({"x", "y", "z", "roll", "pitch", "yaw"})
    _VIDEO_KEY = "image_0"
    _LANG_KEY = "annotation.human.action.task_description"

    _DECODE_HORIZON = 8

    def __init__(
        self,
        vla_addr: str,
        stats_json: str,
        embodiment: str = "oxe_widowx",
        n_action_steps: int = 1,
        recv_timeout_ms: int = 120_000,
        tokenizer: str | None = None,
        image_size: int | None = None,
    ):

        self._c = VlaCppClient(
            vla_addr=vla_addr, arch="gr00t_n1_6", tokenizer_name=tokenizer,
            image_size=image_size, max_state_dim=None, real_action_dim=7,
            image_keys=[], max_length=48, recv_timeout_ms=recv_timeout_ms,
            n_action_steps=1, stats_json=None, bitvla_unnorm_key=None,
        )
        blob = json.loads(Path(stats_json).read_text())
        if embodiment not in blob:
            raise KeyError(f"embodiment {embodiment!r} not in {stats_json}; have {list(blob)}")
        st = blob[embodiment]
        if "state" not in st or "action" not in st:
            raise KeyError(f"{stats_json}::{embodiment} missing state/action stats")
        sk = st["state"]
        self._s_min = np.array([sk[k]["min"][0] for k in self._STATE_KEYS], dtype=np.float32)
        self._s_max = np.array([sk[k]["max"][0] for k in self._STATE_KEYS], dtype=np.float32)
        ak = st["action"]
        self._a_mean = np.array([ak[k]["mean"][0] for k in self._ACTION_KEYS], dtype=np.float32)
        self._a_std = np.array([ak[k]["std"][0] for k in self._ACTION_KEYS], dtype=np.float32)
        self._a_min = np.array([ak[k]["min"][0] for k in self._ACTION_KEYS], dtype=np.float32)
        self._a_max = np.array([ak[k]["max"][0] for k in self._ACTION_KEYS], dtype=np.float32)
        self._a_meanstd = np.array(
            [k in self._ACTION_MEANSTD_KEYS for k in self._ACTION_KEYS], dtype=bool)
        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        self.n_action_steps = n_action_steps
        self._queue: deque = deque(maxlen=n_action_steps)

        _PATCH, _PSHUF = 14, 2
        self._n_tok_per_view = (self._c.image_size // _PATCH // _PSHUF) ** 2
        print(f"vla-cpp-direct[arch=gr00t_n1_6/simpler]: embodiment={embodiment} "
              f"image_size={self._c.image_size} tokens/view={self._n_tok_per_view} "
              f"state(min/max+clip, 8-D) action(mean-std EEF + min/max gripper, 7-D) "
              f"n_action_steps={n_action_steps} via {stats_json}", flush=True)

    def get_arch(self) -> str:
        return self.arch_type

    def reset(self) -> None:
        self._queue.clear()

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:

        if not self._queue:
            chunk = self._predict_chunk(observation)
            for row in chunk[: self.n_action_steps]:
                self._queue.append(np.ascontiguousarray(row, dtype=np.float32))
        row = self._queue.popleft()
        return {k: row[i].reshape(1, 1, 1).astype(np.float32)
                for i, k in enumerate(self._ACTION_KEYS)}

    def _state_norm(self, sv: np.ndarray) -> np.ndarray:

        mn, mx = self._s_min, self._s_max
        rng = mx - mn
        mask = ~np.isclose(mx, mn)
        out = np.zeros_like(sv)
        out[mask] = 2.0 * (sv[mask] - mn[mask]) / rng[mask] - 1.0
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _decode_action(self, chunk: np.ndarray) -> np.ndarray:

        norm = chunk[: self._DECODE_HORIZON, : len(self._ACTION_KEYS)].astype(np.float32)
        raw = np.empty_like(norm)
        for j in range(len(self._ACTION_KEYS)):
            col = norm[:, j]
            if self._a_meanstd[j]:

                raw[:, j] = col * self._a_std[j] + self._a_mean[j] if self._a_std[j] != 0 else col
            else:

                raw[:, j] = ((np.clip(col, -1.0, 1.0) + 1.0) * 0.5
                             * (self._a_max[j] - self._a_min[j]) + self._a_min[j])
        return raw

    def _predict_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        import re
        c = self._c

        img = observation["video"][self._VIDEO_KEY]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.asarray(img, dtype=np.uint8)
        while img.ndim > 3:
            img = img[0]
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"oxe_widowx image: expected HWC u8 [H,W,3], got {img.shape}")
        img_u8 = VlaCppClient._gr00t_n16_image_transform(
            img, c.image_size, VlaCppClient._GR00T_CROP_FRACTION)
        img_f32 = np.ascontiguousarray(img_u8.astype(np.float32) / 255.0)

        sv = np.array(
            [float(np.asarray(observation["state"][k]).reshape(-1)[0]) for k in self._STATE_KEYS],
            dtype=np.float32)
        sv = self._state_norm(sv)
        state_padded = np.zeros(c.max_state_dim, dtype=np.float32)
        state_padded[: sv.size] = sv

        lang_field = observation["language"][self._LANG_KEY]
        text = lang_field[0][0] if isinstance(lang_field, (list, tuple)) else str(lang_field)
        if isinstance(text, bytes):
            text = text.decode()
        language = re.sub(r"[^\w\s]", "", text.lower())
        img_block = ("<image 1><img>"
                     + "<IMG_CONTEXT>" * self._n_tok_per_view
                     + "</img>")
        conv = [{"role": "user", "content": language + img_block}]
        prompt = c.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        ids = c.tok(prompt, add_special_tokens=False)["input_ids"]
        n_slots = sum(1 for t in ids if t == VlaCppClient._GR00T_N16_IMG_PAD_ID)
        if n_slots != self._n_tok_per_view:
            raise RuntimeError(
                f"gr00t_n1_6/simpler: expected {self._n_tok_per_view} "
                f"<IMG_CONTEXT> slots, got {n_slots}. Check the tokenizer's special tokens.")
        lang = np.array(ids, dtype=np.int32)

        req = c.pb.PredictRequest()
        req.request_id = c._step
        c._step += 1
        ip = req.images.add()
        ip.encoding = c.pb.Image.F32_RGB_01
        ip.height = img_f32.shape[0]
        ip.width = img_f32.shape[1]
        ip.data = img_f32.tobytes()
        req.lang_tokens.extend(int(t) for t in lang)
        req.state.extend(float(x) for x in state_padded)
        c.sock.send(req.SerializeToString())
        resp = c.pb.PredictResponse()
        resp.ParseFromString(c.sock.recv())
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")
        c._last_response = resp
        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))
        return self._decode_action(chunk)

    def close(self):
        self._c.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
