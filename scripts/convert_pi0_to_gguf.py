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

from pathlib import Path

import numpy as np
from safetensors import safe_open

from gguf_blocks import (
    identity_stats,
    load_processor_stats,
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

ARCH = "pi0"
KV = kv_prefix(ARCH)

PFX_VLM      = "model.paligemma_with_expert.paligemma.model.language_model"
PFX_VLM_HEAD = "model.paligemma_with_expert.paligemma.lm_head.weight"
PFX_AEX      = "model.paligemma_with_expert.gemma_expert.model"
PFX_PROJ     = "model"

PFX_VIS_CANDIDATES = [
    "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model",
    "model.paligemma_with_expert.paligemma.vision_tower.vision_model",
]
PFX_MMP_CANDIDATES = [
    "model.paligemma_with_expert.paligemma.model.multi_modal_projector",
    "model.paligemma_with_expert.paligemma.multi_modal_projector",
]

GEMMA_2B   = dict(hidden=2048, n_q_heads=8, n_kv_heads=1, head_dim=256, intermediate=16384)
GEMMA_300M = dict(expert_h=1024, expert_inter=4096)

ROPE_THETA   = 10000.0
RMS_NORM_EPS = 1e-6

PROJ_SUFFIXES = [
    "state_proj.weight",
    "state_proj.bias",
    "action_in_proj.weight",
    "action_in_proj.bias",
    "action_time_mlp_in.weight",
    "action_time_mlp_in.bias",
    "action_time_mlp_out.weight",
    "action_time_mlp_out.bias",
    "action_out_proj.weight",
    "action_out_proj.bias",
]

def _load_stats(sf, ckpt: Path, state_dim: int, action_dim: int) -> dict[str, np.ndarray]:

    out = identity_stats(state_dim, action_dim)

    got_state = load_processor_stats(
        ckpt,
        "policy_preprocessor.json",
        "normalizer_processor",
        "observation.state",
        state_dim
    )
    got_action = load_processor_stats(
        ckpt,
        "policy_postprocessor.json",
        "unnormalizer_processor",
        "action",
        action_dim
    )

    keys = set(sf.keys())

    def _legacy(mk: str, sk: str, mean_dst: str, std_dst: str, dim: int) -> None:
        if mk not in keys or sk not in keys:
            print(f"  stats: legacy {mk} / {sk} missing - using identity for {mean_dst[:-5]}")
            return
        mean = sf.get_tensor(mk).float().numpy().reshape(-1)
        std  = sf.get_tensor(sk).float().numpy().reshape(-1)
        if mean.size != dim or std.size != dim:
            print(f"  stats: legacy {mk} dim mismatch ({mean.size} vs {dim}) - using identity")
            return
        out[mean_dst] = mean.astype(np.float32, copy=False)
        out[std_dst]  = std .astype(np.float32, copy=False)
        print(f"  stats: loaded {mean_dst[:-5]} from model.safetensors ({mk}/{sk}) [legacy]")

    if got_state is not None:
        out["state_mean"], out["state_std"] = got_state
    else:
        _legacy(
            "normalize_inputs.buffer_observation_state.mean",
            "normalize_inputs.buffer_observation_state.std",
            "state_mean",
            "state_std",
            state_dim
        )

    if got_action is not None:
        out["action_mean"], out["action_std"] = got_action
    elif "unnormalize_outputs.buffer_action.mean" in keys:
        _legacy(
            "unnormalize_outputs.buffer_action.mean",
            "unnormalize_outputs.buffer_action.std",
            "action_mean",
            "action_std",
            action_dim
        )
    else:
        _legacy(
            "normalize_targets.buffer_action.mean",
            "normalize_targets.buffer_action.std",
            "action_mean",
            "action_std",
            action_dim
        )
    return out

def main() -> int:
    ap = arg_parser(ARCH, "lerobot π₀ checkpoint dir (model.safetensors + config.json + policy_*processor.json)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    sf_path = ckpt / "model.safetensors"
    require(sf_path)

    cfg_json = read_json(ckpt / "config.json")
    if cfg_json.get("type") != ARCH:
        raise SystemExit(f"config.json type is {cfg_json.get('type')!r}, expected 'pi0' "
                         f"(π0.5 / other variants are not handled by this converter)")

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
        raise SystemExit(f"layer count mismatch: VLM={n_layers_vlm} expert={n_layers_aex} "
                         f"(π0 expects them equal)")
    cfg["n_layers"] = n_layers_vlm

    q0    = sf.get_slice(f"{PFX_VLM}.layers.0.self_attn.q_proj.weight").get_shape()
    kv0   = sf.get_slice(f"{PFX_VLM}.layers.0.self_attn.k_proj.weight").get_shape()
    gate0 = sf.get_slice(f"{PFX_VLM}.layers.0.mlp.gate_proj.weight").get_shape()
    if q0[1] != cfg["hidden"]:
        raise SystemExit(f"hidden mismatch: cfg={cfg['hidden']} ckpt={q0[1]}")
    if q0[0] != cfg["n_q_heads"] * cfg["head_dim"]:
        raise SystemExit(f"q_proj rows {q0[0]} != n_q_heads*head_dim {cfg['n_q_heads']*cfg['head_dim']}")
    if kv0[0] != cfg["n_kv_heads"] * cfg["head_dim"]:
        raise SystemExit(f"k_proj rows {kv0[0]} != n_kv_heads*head_dim {cfg['n_kv_heads']*cfg['head_dim']}")
    if gate0[0] != cfg["intermediate"]:
        raise SystemExit(f"intermediate mismatch: cfg={cfg['intermediate']} ckpt={gate0[0]}")

    aex_gate0 = sf.get_slice(f"{PFX_AEX}.layers.0.mlp.gate_proj.weight").get_shape()
    if aex_gate0[1] != cfg["expert_h"]:
        raise SystemExit(f"expert_h mismatch: cfg={cfg['expert_h']} ckpt={aex_gate0[1]}")
    if aex_gate0[0] != cfg["expert_inter"]:
        raise SystemExit(f"expert_inter mismatch: cfg={cfg['expert_inter']} ckpt={aex_gate0[0]}")
    aex_o0 = sf.get_slice(f"{PFX_AEX}.layers.0.self_attn.o_proj.weight").get_shape()
    if aex_o0 != [cfg["expert_h"], cfg["n_q_heads"] * cfg["head_dim"]]:
        raise SystemExit(f"expert o_proj shape {aex_o0} != [expert_h, n_q*head_dim] "
                         f"{[cfg['expert_h'], cfg['n_q_heads']*cfg['head_dim']]}")

    head_w = sf.get_slice(PFX_VLM_HEAD).get_shape()
    if head_w[1] != cfg["hidden"]:
        raise SystemExit(f"lm_head hidden mismatch: cfg={cfg['hidden']} ckpt={head_w[1]}")
    cfg["vocab_size"] = int(head_w[0])

    print(f"resolved cfg: hidden={cfg['hidden']} n_layers={cfg['n_layers']} "
          f"inter={cfg['intermediate']} heads={cfg['n_q_heads']}q/{cfg['n_kv_heads']}kv×{cfg['head_dim']} "
          f"expert_h={cfg['expert_h']} expert_inter={cfg['expert_inter']} vocab={cfg['vocab_size']} "
          f"chunk={cfg['chunk_size']} steps={cfg['num_steps']} "
          f"real_state={cfg['real_state_dim']} real_action={cfg['real_action_dim']} "
          f"norm_eps={cfg['norm_eps']:g}")

    print("loading normalizer stats...")
    stats = _load_stats(sf, ckpt, cfg["real_state_dim"], cfg["real_action_dim"])

    cfg["vit"] = probe_paligemma_vision(sf, keys, cfg_json, PFX_VIS_CANDIDATES, PFX_MMP_CANDIDATES)
    v = cfg["vit"]
    print(f"vision: SigLIP hidden={v['vit_hidden']} layers={v['vit_layers']} "
          f"heads={v['vit_heads']} image={v['image_size']} patch={v['patch_size']} "
          f"tokens={v['n_img_tokens']} ln_eps={v['vit_ln_eps']:g}")

    writer = open_writer(out, ARCH)
    write_pi_kv(writer, KV, cfg)

    add(writer, "token_embd.weight",      sf.get_tensor(PFX_VLM_HEAD))
    add(writer, "vlm.output_norm.weight", sf.get_tensor(f"{PFX_VLM}.norm.weight"))
    write_decoder_blocks(writer, sf.get_tensor, PFX_VLM, "vlm", cfg["n_layers"])

    add(writer, "aex.output_norm.weight", sf.get_tensor(f"{PFX_AEX}.norm.weight"))
    write_decoder_blocks(writer, sf.get_tensor, PFX_AEX, "aex", cfg["n_layers"])

    for suf in PROJ_SUFFIXES:
        add(writer, suf, sf.get_tensor(f"{PFX_PROJ}.{suf}"))

    write_paligemma_vision(writer, sf, cfg["vit"])

    for name, vec in stats.items():
        add_array(writer, name, vec)

    rc = finish(writer, out)
    print("self-contained: SigLIP vision tower + projector are bundled; no separate mmproj needed.")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
