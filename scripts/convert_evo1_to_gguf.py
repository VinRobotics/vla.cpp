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

import numpy as np
import torch

from gguf_common import (
    add,
    add_array,
    arg_parser,
    check_layers,
    finish,
    kv_f32,
    kv_prefix,
    kv_u32,
    max_layer,
    open_writer,
    read_json,
    require,
    resolve_out
)

ARCH = "evo1"
KV = kv_prefix(ARCH)

VIT = dict(
    vit_hidden=1024,
    vit_layers=24,
    vit_heads=16,
    vit_inter=4096,
    image_size=448,
    patch_size=14,
    vit_ln_eps=1e-6
)

QWEN2 = dict(
    lm_hidden=896,
    lm_q_heads=14,
    lm_kv_heads=2,
    lm_head_dim=64,
    lm_inter=4864,
    lm_rope_theta=1000000.0,
    lm_rms_eps=1e-6
)
LM_LAYERS_USED = 14

PROJ_LN_EPS = 1e-5
DIT_HEADS   = 8
NUM_INFERENCE_TIMESTEPS = 32
STATE_PAD = 24

VIT_ROOT = "embedder.model.vision_model"
LM_ROOT  = "embedder.model.language_model.model"
AHK      = "action_head"

U32_KEYS = (
    "vit_hidden",
    "vit_layers",
    "vit_heads",
    "vit_inter",
    "image_size",
    "patch_size",
    "num_image_token",
    "lm_hidden",
    "lm_layers_used",
    "lm_q_heads",
    "lm_kv_heads",
    "lm_head_dim",
    "lm_inter",
    "vocab_size",
    "embed_dim",
    "dit_layers",
    "dit_heads",
    "mlp_head_hidden",
    "horizon",
    "per_action_dim",
    "state_dim",
    "action_dim",
    "num_inference_timesteps",
    "real_state_dim",
    "real_action_dim",
    "max_text_length",
    "n_images",
    "img_context_token_id",
    "img_start_token_id",
    "img_end_token_id",
    "pad_token_id"
)

def _pad24(x) -> np.ndarray:

    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size > STATE_PAD:
        raise SystemExit(f"norm-stats vector of length {a.size} exceeds {STATE_PAD}")
    out = np.zeros(STATE_PAD, dtype=np.float32)
    out[:a.size] = a
    return out

def main() -> int:
    ap = arg_parser(ARCH, "Evo-1 checkpoint dir (mp_rank_00_model_states.pt + config.json + norm_stats.json)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    pt_path = ckpt / "mp_rank_00_model_states.pt"
    ns_path = ckpt / "norm_stats.json"
    require(pt_path, ns_path)

    cfg_json = read_json(ckpt / "config.json")
    if str(cfg_json.get("action_head", "")).lower() != "flowmatching":
        raise SystemExit(f"config.json action_head is {cfg_json.get('action_head')!r}, expected 'flowmatching'")

    cfg = dict(VIT, **QWEN2)
    cfg["lm_layers_used"]  = LM_LAYERS_USED
    cfg["horizon"]         = int(cfg_json["horizon"])
    cfg["per_action_dim"]  = int(cfg_json["per_action_dim"])
    cfg["state_dim"]       = int(cfg_json["state_dim"])
    cfg["action_dim"]      = int(cfg_json["action_dim"])
    cfg["dit_layers"]      = int(cfg_json.get("num_layers", 8))
    cfg["embed_dim"]       = int(cfg_json.get("embed_dim", 896))
    cfg["mlp_head_hidden"] = int(cfg_json.get("hidden_dim", 1024))
    cfg["num_inference_timesteps"] = int(cfg_json.get("num_inference_timesteps", NUM_INFERENCE_TIMESTEPS))
    cfg["image_size"]      = int(cfg_json.get("image_size", VIT["image_size"]))
    cfg["dit_heads"]       = DIT_HEADS
    cfg["proj_ln_eps"]     = PROJ_LN_EPS
    if cfg["action_dim"] != cfg["horizon"] * cfg["per_action_dim"]:
        raise SystemExit(f"action_dim {cfg['action_dim']} != horizon*per_action_dim {cfg['horizon']*cfg['per_action_dim']}")

    print(f"loading {pt_path} ...")
    module = torch.load(pt_path, map_location="cpu", weights_only=False)["module"]
    keys = set(module.keys())
    print(f"  {len(module)} tensors")

    check_layers(max_layer(keys, f"{LM_ROOT}.layers."), LM_LAYERS_USED, "LM layers (Evo-1 truncates to layers[:14])")
    check_layers(max_layer(keys, f"{VIT_ROOT}.encoder.layers."), VIT["vit_layers"], "ViT layers")
    check_layers(max_layer(keys, f"{AHK}.transformer_blocks."), cfg["dit_layers"], "DiT blocks")

    grid = cfg["image_size"] // cfg["patch_size"]
    cfg["num_image_token"] = (grid // 2) ** 2
    cfg["vocab_size"] = int(module[f"{LM_ROOT}.embed_tokens.weight"].shape[0])

    cfg["img_context_token_id"] = 151667
    cfg["img_start_token_id"]   = 151665
    cfg["img_end_token_id"]     = 151666
    cfg["pad_token_id"]         = 151643
    cfg["max_text_length"]      = 1024
    cfg["n_images"]             = 3

    ns = json.loads(ns_path.read_text())
    if len(ns) != 1:
        raise SystemExit(f"norm_stats.json should have exactly one robot key; got {list(ns)}")
    robot = next(iter(ns.values()))
    stats = {
        "state_min":  _pad24(robot["observation.state"]["min"]),
        "state_max":  _pad24(robot["observation.state"]["max"]),
        "action_min": _pad24(robot["action"]["min"]),
        "action_max": _pad24(robot["action"]["max"]),
    }
    cfg["real_state_dim"]  = int(len(robot["observation.state"]["min"]))
    cfg["real_action_dim"] = int(len(robot["action"]["min"]))
    cfg["norm_eps"]        = 1e-8

    print(f"resolved cfg: vit={cfg['vit_hidden']}d×{cfg['vit_layers']}L  lm={cfg['lm_hidden']}d×{cfg['lm_layers_used']}L "
          f"({cfg['lm_q_heads']}q/{cfg['lm_kv_heads']}kv×{cfg['lm_head_dim']})  vocab={cfg['vocab_size']}  "
          f"embed={cfg['embed_dim']}  dit={cfg['dit_layers']}L×{cfg['dit_heads']}h  horizon={cfg['horizon']} "
          f"per_a={cfg['per_action_dim']}  N_steps={cfg['num_inference_timesteps']}  "
          f"img_tok={cfg['num_image_token']}  n_img={cfg['n_images']}  real_state={cfg['real_state_dim']} "
          f"real_action={cfg['real_action_dim']}")

    writer = open_writer(out, ARCH)
    kv_u32(writer, KV, {k: cfg[k] for k in U32_KEYS})
    writer.add_float64(KV("lm_rope_theta"), float(cfg["lm_rope_theta"]))
    kv_f32(writer, KV, {k: cfg[k] for k in ("lm_rms_eps", "vit_ln_eps", "proj_ln_eps", "norm_eps")})

    g = module.__getitem__

    VE = f"{VIT_ROOT}.embeddings."
    add(writer, "vit.patch_embd.weight", g(VE + "patch_embedding.weight"))
    add(writer, "vit.patch_embd.bias",   g(VE + "patch_embedding.bias"))
    add(writer, "vit.class_embd",        g(VE + "class_embedding"))
    add(writer, "vit.pos_embd",          g(VE + "position_embedding"))
    for i in range(cfg["vit_layers"]):
        VL = f"{VIT_ROOT}.encoder.layers.{i}."
        add(writer, f"vit.blk.{i}.norm1.weight", g(VL + "norm1.weight")); add(writer, f"vit.blk.{i}.norm1.bias", g(VL + "norm1.bias"))
        add(writer, f"vit.blk.{i}.norm2.weight", g(VL + "norm2.weight")); add(writer, f"vit.blk.{i}.norm2.bias", g(VL + "norm2.bias"))
        add(writer, f"vit.blk.{i}.ls1", g(VL + "ls1")); add(writer, f"vit.blk.{i}.ls2", g(VL + "ls2"))
        add(writer, f"vit.blk.{i}.attn_qkv.weight", g(VL + "attn.qkv.weight")); add(writer, f"vit.blk.{i}.attn_qkv.bias", g(VL + "attn.qkv.bias"))
        add(writer, f"vit.blk.{i}.attn_proj.weight", g(VL + "attn.proj.weight")); add(writer, f"vit.blk.{i}.attn_proj.bias", g(VL + "attn.proj.bias"))
        add(writer, f"vit.blk.{i}.fc1.weight", g(VL + "mlp.fc1.weight")); add(writer, f"vit.blk.{i}.fc1.bias", g(VL + "mlp.fc1.bias"))
        add(writer, f"vit.blk.{i}.fc2.weight", g(VL + "mlp.fc2.weight")); add(writer, f"vit.blk.{i}.fc2.bias", g(VL + "mlp.fc2.bias"))

    add(writer, "mm.ln.weight",  g("embedder.model.mlp1.0.weight")); add(writer, "mm.ln.bias",  g("embedder.model.mlp1.0.bias"))
    add(writer, "mm.fc1.weight", g("embedder.model.mlp1.1.weight")); add(writer, "mm.fc1.bias", g("embedder.model.mlp1.1.bias"))
    add(writer, "mm.fc2.weight", g("embedder.model.mlp1.3.weight")); add(writer, "mm.fc2.bias", g("embedder.model.mlp1.3.bias"))

    add(writer, "token_embd.weight",      g(f"{LM_ROOT}.embed_tokens.weight"))
    add(writer, "vlm.output_norm.weight", g(f"{LM_ROOT}.norm.weight"))
    for i in range(cfg["lm_layers_used"]):
        LL = f"{LM_ROOT}.layers.{i}."
        add(writer, f"vlm.blk.{i}.attn_norm.weight", g(LL + "input_layernorm.weight"))
        for q in ("q", "k", "v"):
            add(writer, f"vlm.blk.{i}.attn_{q}.weight", g(LL + f"self_attn.{q}_proj.weight")); add(writer, f"vlm.blk.{i}.attn_{q}.bias", g(LL + f"self_attn.{q}_proj.bias"))
        add(writer, f"vlm.blk.{i}.attn_o.weight", g(LL + "self_attn.o_proj.weight"))
        add(writer, f"vlm.blk.{i}.ffn_norm.weight", g(LL + "post_attention_layernorm.weight"))
        add(writer, f"vlm.blk.{i}.ffn_gate.weight", g(LL + "mlp.gate_proj.weight"))
        add(writer, f"vlm.blk.{i}.ffn_up.weight",   g(LL + "mlp.up_proj.weight"))
        add(writer, f"vlm.blk.{i}.ffn_down.weight", g(LL + "mlp.down_proj.weight"))

    AE = f"{AHK}.action_encoder."
    for w in ("W1", "W2", "W3"):
        add(writer, f"aex.ae.{w}.weight", g(AE + f"{w}.linear.weight")); add(writer, f"aex.ae.{w}.bias", g(AE + f"{w}.linear.bias"))
    add(writer, "aex.ae.pos_enc", g(AE + "pos_encoding.pe"))
    for i in range(cfg["dit_layers"]):
        TB = f"{AHK}.transformer_blocks.{i}."
        add(writer, f"aex.blk.{i}.norm1.weight", g(TB + "norm1.weight")); add(writer, f"aex.blk.{i}.norm1.bias", g(TB + "norm1.bias"))
        add(writer, f"aex.blk.{i}.norm2.weight", g(TB + "norm2.weight")); add(writer, f"aex.blk.{i}.norm2.bias", g(TB + "norm2.bias"))
        add(writer, f"aex.blk.{i}.attn_in.weight", g(TB + "attn.in_proj_weight")); add(writer, f"aex.blk.{i}.attn_in.bias", g(TB + "attn.in_proj_bias"))
        add(writer, f"aex.blk.{i}.attn_out.weight", g(TB + "attn.out_proj.weight")); add(writer, f"aex.blk.{i}.attn_out.bias", g(TB + "attn.out_proj.bias"))
        add(writer, f"aex.blk.{i}.ff1.weight", g(TB + "ff.0.weight")); add(writer, f"aex.blk.{i}.ff1.bias", g(TB + "ff.0.bias"))
        add(writer, f"aex.blk.{i}.ff2.weight", g(TB + "ff.2.weight")); add(writer, f"aex.blk.{i}.ff2.bias", g(TB + "ff.2.bias"))
    add(writer, "aex.norm_out.weight", g(f"{AHK}.norm_out.weight")); add(writer, "aex.norm_out.bias", g(f"{AHK}.norm_out.bias"))
    add(writer, "aex.seq_pool.weight", g(f"{AHK}.seq_pool_proj.weight")); add(writer, "aex.seq_pool.bias", g(f"{AHK}.seq_pool_proj.bias"))
    add(writer, "aex.head.fc1.weight", g(f"{AHK}.mlp_head.fc1.linear.weight")); add(writer, "aex.head.fc1.bias", g(f"{AHK}.mlp_head.fc1.linear.bias"))
    add(writer, "aex.head.fc2.weight", g(f"{AHK}.mlp_head.fc2.linear.weight")); add(writer, "aex.head.fc2.bias", g(f"{AHK}.mlp_head.fc2.linear.bias"))
    add(writer, "aex.time_pos_enc", g(f"{AHK}.time_pos_enc.pe"))
    add(writer, "aex.state_enc.fc1.weight", g(f"{AHK}.state_encoder.fc1.linear.weight")); add(writer, "aex.state_enc.fc1.bias", g(f"{AHK}.state_encoder.fc1.linear.bias"))
    add(writer, "aex.state_enc.fc2.weight", g(f"{AHK}.state_encoder.fc2.linear.weight")); add(writer, "aex.state_enc.fc2.bias", g(f"{AHK}.state_encoder.fc2.linear.bias"))

    for name, vec in stats.items():
        add_array(writer, name, vec)

    return finish(writer, out, "  - combined GGUF (vision + LM + action head + stats + cfg)")

if __name__ == "__main__":
    raise SystemExit(main())
