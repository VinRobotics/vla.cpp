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

from safetensors import safe_open

from gguf_blocks import (
    lerobot_stats,
    norm_eps,
    pi_root,
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
    if cfg_json.get("adapt_to_pi_aloha"):
        raise SystemExit("adapt_to_pi_aloha=true is not supported")

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
    root = pi_root(keys)
    vlm, head, aex = root + PFX_VLM, root + PFX_VLM_HEAD, root + PFX_AEX

    n_layers_vlm = max_layer(keys, f"{vlm}.layers.")
    n_layers_aex = max_layer(keys, f"{aex}.layers.")
    if n_layers_vlm <= 0:
        raise SystemExit("cannot find PaliGemma language-model layers in checkpoint")
    if n_layers_aex != n_layers_vlm:
        raise SystemExit(f"layer count mismatch: VLM={n_layers_vlm} expert={n_layers_aex} "
                         f"(π0 expects them equal)")
    cfg["n_layers"] = n_layers_vlm

    q0    = sf.get_slice(f"{vlm}.layers.0.self_attn.q_proj.weight").get_shape()
    kv0   = sf.get_slice(f"{vlm}.layers.0.self_attn.k_proj.weight").get_shape()
    gate0 = sf.get_slice(f"{vlm}.layers.0.mlp.gate_proj.weight").get_shape()
    if q0[1] != cfg["hidden"]:
        raise SystemExit(f"hidden mismatch: cfg={cfg['hidden']} ckpt={q0[1]}")
    if q0[0] != cfg["n_q_heads"] * cfg["head_dim"]:
        raise SystemExit(f"q_proj rows {q0[0]} != n_q_heads*head_dim {cfg['n_q_heads']*cfg['head_dim']}")
    if kv0[0] != cfg["n_kv_heads"] * cfg["head_dim"]:
        raise SystemExit(f"k_proj rows {kv0[0]} != n_kv_heads*head_dim {cfg['n_kv_heads']*cfg['head_dim']}")
    if gate0[0] != cfg["intermediate"]:
        raise SystemExit(f"intermediate mismatch: cfg={cfg['intermediate']} ckpt={gate0[0]}")

    aex_gate0 = sf.get_slice(f"{aex}.layers.0.mlp.gate_proj.weight").get_shape()
    if aex_gate0[1] != cfg["expert_h"]:
        raise SystemExit(f"expert_h mismatch: cfg={cfg['expert_h']} ckpt={aex_gate0[1]}")
    if aex_gate0[0] != cfg["expert_inter"]:
        raise SystemExit(f"expert_inter mismatch: cfg={cfg['expert_inter']} ckpt={aex_gate0[0]}")
    aex_o0 = sf.get_slice(f"{aex}.layers.0.self_attn.o_proj.weight").get_shape()
    if aex_o0 != [cfg["expert_h"], cfg["n_q_heads"] * cfg["head_dim"]]:
        raise SystemExit(f"expert o_proj shape {aex_o0} != [expert_h, n_q*head_dim] "
                         f"{[cfg['expert_h'], cfg['n_q_heads']*cfg['head_dim']]}")

    head_w = sf.get_slice(head).get_shape()
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
    stats = lerobot_stats(sf, ckpt, cfg["real_state_dim"], cfg["real_action_dim"],
                          cfg_json.get("normalization_mapping") or {})

    cfg["vit"] = probe_paligemma_vision(sf, keys, cfg_json, [root + p for p in PFX_VIS_CANDIDATES],
                                        [root + p for p in PFX_MMP_CANDIDATES])
    v = cfg["vit"]
    print(f"vision: SigLIP hidden={v['vit_hidden']} layers={v['vit_layers']} "
          f"heads={v['vit_heads']} image={v['image_size']} patch={v['patch_size']} "
          f"tokens={v['n_img_tokens']} ln_eps={v['vit_ln_eps']:g}")

    writer = open_writer(out, ARCH)
    write_pi_kv(writer, KV, cfg)

    add(writer, "token_embd.weight",      sf.get_tensor(head))
    add(writer, "vlm.output_norm.weight", sf.get_tensor(f"{vlm}.norm.weight"))
    write_decoder_blocks(writer, sf.get_tensor, vlm, "vlm", cfg["n_layers"])

    add(writer, "aex.output_norm.weight", sf.get_tensor(f"{aex}.norm.weight"))
    write_decoder_blocks(writer, sf.get_tensor, aex, "aex", cfg["n_layers"])

    for suf in PROJ_SUFFIXES:
        add(writer, suf, sf.get_tensor(root + suf))

    write_paligemma_vision(writer, sf, cfg["vit"])

    for name, vec in stats.items():
        add_array(writer, name, vec)

    rc = finish(writer, out)
    print("self-contained: SigLIP vision tower + projector are bundled; no separate mmproj needed.")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
