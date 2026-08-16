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

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open

from gguf_blocks import (
    norm_eps,
    probe_paligemma_vision,
    write_decoder_blocks,
    write_paligemma_vision,
    write_pi_kv
)
from gguf_common import (
    add,
    add_array,
    arg_parser,
    finish,
    kv_prefix,
    max_layer,
    open_writer,
    read_json,
    require,
    resolve_out
)

ARCH = "pi05"
KV = kv_prefix(ARCH)

PFX_VLM      = "paligemma_with_expert.paligemma.model.language_model"
PFX_VLM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
PFX_AEX      = "paligemma_with_expert.gemma_expert.model"

PFX_VIS_CANDIDATES = [
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model",
    "paligemma_with_expert.paligemma.vision_tower.vision_model",
]
PFX_MMP_CANDIDATES = [
    "paligemma_with_expert.paligemma.model.multi_modal_projector",
    "paligemma_with_expert.paligemma.multi_modal_projector",
]

GEMMA_2B   = dict(hidden=2048, n_q_heads=8, n_kv_heads=1, head_dim=256, intermediate=16384)
GEMMA_300M = dict(expert_h=1024, expert_inter=4096)

ROPE_THETA   = 10000.0
RMS_NORM_EPS = 1e-6

AEX_MAP = [
    ("input_layernorm.dense.weight",          "attn_norm.weight"),
    ("input_layernorm.dense.bias",            "attn_norm.bias"),
    ("post_attention_layernorm.dense.weight", "ffn_norm.weight"),
    ("post_attention_layernorm.dense.bias",   "ffn_norm.bias"),
    ("self_attn.q_proj.weight",               "attn_q.weight"),
    ("self_attn.k_proj.weight",               "attn_k.weight"),
    ("self_attn.v_proj.weight",               "attn_v.weight"),
    ("self_attn.o_proj.weight",               "attn_o.weight"),
    ("mlp.gate_proj.weight",                  "ffn_gate.weight"),
    ("mlp.up_proj.weight",                    "ffn_up.weight"),
    ("mlp.down_proj.weight",                  "ffn_down.weight"),
]

PROJ_SUFFIXES = [
    "action_in_proj.weight",
    "action_in_proj.bias",
    "time_mlp_in.weight",
    "time_mlp_in.bias",
    "time_mlp_out.weight",
    "time_mlp_out.bias",
    "action_out_proj.weight",
    "action_out_proj.bias",
]

def _write_adarms_blocks(writer, sf, n_layers: int) -> None:

    for i in range(n_layers):
        for src_suf, dst_suf in AEX_MAP:
            add(writer, f"aex.blk.{i}.{dst_suf}", sf.get_tensor(f"{PFX_AEX}.layers.{i}.{src_suf}"))

def _load_dataset_stats(
    stats_json: Optional[Path],
    dataset_repo: Optional[str],
    state_dim: int,
    action_dim: int
) -> dict[str, np.ndarray]:

    def _from_json(path: Path) -> dict[str, np.ndarray]:
        d = json.loads(path.read_text())

        def grab(feat, dim):
            s = d[feat]
            mean = np.asarray(s["mean"], dtype=np.float32).reshape(-1)
            std  = np.asarray(s["std"],  dtype=np.float32).reshape(-1)
            if mean.size < dim or std.size < dim:
                raise SystemExit(f"stats {feat} dim {mean.size} < expected {dim}")
            out = {"mean": mean[:dim], "std": std[:dim]}

            for q in ("q01", "q99"):
                if q not in s:
                    raise SystemExit(
                        f"stats {feat} missing {q} (QUANTILES normalization needs it). "
                        f"Use a meta/stats.json with quantile stats.")
                out[q] = np.asarray(s[q], dtype=np.float32).reshape(-1)[:dim]
            return out

        st = grab("observation.state", state_dim)
        ac = grab("action", action_dim)
        return {
            "state_mean":  st["mean"],
            "state_std":   st["std"],
            "state_q01":   st["q01"],
            "state_q99":   st["q99"],
            "action_mean": ac["mean"],
            "action_std":  ac["std"],
            "action_q01":  ac["q01"],
            "action_q99":  ac["q99"]
        }

    if stats_json is not None:
        print(f"  stats: loading from {stats_json}")
        return _from_json(stats_json)

    if dataset_repo is not None:
        from huggingface_hub import hf_hub_download
        print(f"  stats: fetching meta/stats.json from dataset repo {dataset_repo}")
        p = hf_hub_download(repo_id=dataset_repo, filename="meta/stats.json", repo_type="dataset")
        return _from_json(Path(p))

    raise SystemExit(
        "no stats source: π0.5 checkpoints carry no normalizer stats — pass "
        "--dataset-stats <meta/stats.json> or --dataset-repo lerobot/libero. "
        "Refusing to bake identity stats (would make the policy miss; skill footgun #1).")

def main() -> int:
    ap = arg_parser(ARCH, "lerobot π0.5 checkpoint dir (model.safetensors + config.json + policy_*processor.json)")
    ap.add_argument(
        "--dataset-stats",
        type=Path,
        default=None,
        help="Path to a LIBERO dataset meta/stats.json for MEAN_STD norm stats"
    )
    ap.add_argument(
        "--dataset-repo",
        type=str,
        default=None,
        help="HF dataset repo to fetch meta/stats.json from (e.g. lerobot/libero)"
    )
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    sf_path = ckpt / "model.safetensors"
    require(sf_path)

    cfg_json = read_json(ckpt / "config.json")
    if cfg_json.get("type") != ARCH:
        raise SystemExit(f"config.json type is {cfg_json.get('type')!r}, expected 'pi05'")

    cfg = dict(GEMMA_2B, **GEMMA_300M)
    cfg["paligemma_variant"]     = str(cfg_json.get("paligemma_variant", "gemma_2b"))
    cfg["action_expert_variant"] = str(cfg_json.get("action_expert_variant", "gemma_300m"))
    cfg["chunk_size"]            = int(cfg_json["chunk_size"])
    cfg["num_steps"]             = int(cfg_json["num_inference_steps"])
    cfg["n_action_steps"]        = int(cfg_json["n_action_steps"])
    cfg["max_state_dim"]         = int(cfg_json["max_state_dim"])
    cfg["max_action_dim"]        = int(cfg_json["max_action_dim"])
    cfg["min_period"]            = float(cfg_json["min_period"])
    cfg["max_period"]            = float(cfg_json["max_period"])
    cfg["tokenizer_max_length"]  = int(cfg_json["tokenizer_max_length"])
    cfg["real_state_dim"]        = int(cfg_json["input_features"]["observation.state"]["shape"][0])
    cfg["real_action_dim"]       = int(cfg_json["output_features"]["action"]["shape"][0])
    cfg["rope_theta"]            = ROPE_THETA
    cfg["rms_norm_eps"]          = RMS_NORM_EPS
    cfg["norm_eps"]              = norm_eps(ckpt)

    print(f"opening {sf_path}")
    sf = safe_open(sf_path, framework="pt")
    keys = set(sf.keys())

    n_layers_vlm = max_layer(keys, f"{PFX_VLM}.layers.")
    n_layers_aex = max_layer(keys, f"{PFX_AEX}.layers.")
    if n_layers_vlm <= 0:
        raise SystemExit("cannot find PaliGemma language-model layers in checkpoint")
    if n_layers_aex != n_layers_vlm:
        raise SystemExit(f"layer count mismatch: VLM={n_layers_vlm} expert={n_layers_aex}")
    cfg["n_layers"] = n_layers_vlm

    q0    = sf.get_slice(f"{PFX_VLM}.layers.0.self_attn.q_proj.weight").get_shape()
    kv0   = sf.get_slice(f"{PFX_VLM}.layers.0.self_attn.k_proj.weight").get_shape()
    gate0 = sf.get_slice(f"{PFX_VLM}.layers.0.mlp.gate_proj.weight").get_shape()
    if q0[1] != cfg["hidden"]:
        raise SystemExit(f"hidden mismatch: cfg={cfg['hidden']} ckpt={q0[1]}")
    if q0[0] != cfg["n_q_heads"] * cfg["head_dim"]:
        raise SystemExit(f"q_proj rows {q0[0]} != n_q_heads*head_dim")
    if kv0[0] != cfg["n_kv_heads"] * cfg["head_dim"]:
        raise SystemExit(f"k_proj rows {kv0[0]} != n_kv_heads*head_dim")
    if gate0[0] != cfg["intermediate"]:
        raise SystemExit(f"intermediate mismatch: cfg={cfg['intermediate']} ckpt={gate0[0]}")

    ada0 = sf.get_slice(f"{PFX_AEX}.layers.0.input_layernorm.dense.weight").get_shape()
    if ada0 != [3 * cfg["expert_h"], cfg["expert_h"]]:
        raise SystemExit(f"expert adaRMS dense shape {ada0} != [3*expert_h, expert_h] "
                         f"{[3*cfg['expert_h'], cfg['expert_h']]}")
    aex_o0 = sf.get_slice(f"{PFX_AEX}.layers.0.self_attn.o_proj.weight").get_shape()
    if aex_o0 != [cfg["expert_h"], cfg["n_q_heads"] * cfg["head_dim"]]:
        raise SystemExit(f"expert o_proj shape {aex_o0} unexpected")

    cfg["vocab_size"] = int(sf.get_slice(PFX_VLM_HEAD).get_shape()[0])

    print(f"resolved cfg: hidden={cfg['hidden']} n_layers={cfg['n_layers']} "
          f"expert_h={cfg['expert_h']} vocab={cfg['vocab_size']} chunk={cfg['chunk_size']} "
          f"steps={cfg['num_steps']} real_state={cfg['real_state_dim']} "
          f"real_action={cfg['real_action_dim']} max_len={cfg['tokenizer_max_length']} "
          f"norm_eps={cfg['norm_eps']:g}")

    cfg["vit"] = probe_paligemma_vision(sf, keys, cfg_json, PFX_VIS_CANDIDATES, PFX_MMP_CANDIDATES)
    v = cfg["vit"]
    print(f"vision: SigLIP hidden={v['vit_hidden']} layers={v['vit_layers']} "
          f"heads={v['vit_heads']} image={v['image_size']} patch={v['patch_size']} "
          f"tokens={v['n_img_tokens']} ln_eps={v['vit_ln_eps']:g}")

    print("loading dataset normalizer stats...")
    stats = _load_dataset_stats(
        args.dataset_stats,
        args.dataset_repo,
        cfg["real_state_dim"],
        cfg["real_action_dim"]
    )
    print(f"  state_q01[:3]={stats['state_q01'][:3]}  state_q99[:3]={stats['state_q99'][:3]}")
    print(f"  action_q01[:3]={stats['action_q01'][:3]}  action_q99[:3]={stats['action_q99'][:3]}  (QUANTILES)")

    writer = open_writer(out, ARCH)
    write_pi_kv(writer, KV, cfg, adarms=True)

    add(writer, "token_embd.weight",      sf.get_tensor(PFX_VLM_HEAD))
    add(writer, "vlm.output_norm.weight", sf.get_tensor(f"{PFX_VLM}.norm.weight"))
    write_decoder_blocks(writer, sf.get_tensor, PFX_VLM, "vlm", cfg["n_layers"])

    add(writer, "aex.output_norm.weight", sf.get_tensor(f"{PFX_AEX}.norm.dense.weight"))
    add(writer, "aex.output_norm.bias",   sf.get_tensor(f"{PFX_AEX}.norm.dense.bias"))
    _write_adarms_blocks(writer, sf, cfg["n_layers"])

    for suf in PROJ_SUFFIXES:
        add(writer, suf, sf.get_tensor(suf))

    write_paligemma_vision(writer, sf, cfg["vit"])

    for name, vec in stats.items():
        add_array(writer, name, vec)

    rc = finish(writer, out)
    print("note: self-contained GGUF — SigLIP vision tower + PaliGemma projector are baked in; "
          "no separate mmproj is needed.")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
