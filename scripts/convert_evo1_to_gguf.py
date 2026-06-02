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

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import gguf

ARCH = "evo1"
KV = lambda name: f"{ARCH}.{name}"

VIT = dict(vit_hidden=1024, vit_layers=24, vit_heads=16, vit_inter=4096,
           image_size=448, patch_size=14, vit_ln_eps=1e-6)

QWEN2 = dict(lm_hidden=896, lm_q_heads=14, lm_kv_heads=2, lm_head_dim=64,
             lm_inter=4864, lm_rope_theta=1000000.0, lm_rms_eps=1e-6)
LM_LAYERS_USED = 14

PROJ_LN_EPS = 1e-5
DIT_HEADS   = 8
NUM_INFERENCE_TIMESTEPS = 32
STATE_PAD = 24

def _bf16_u16(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        print(f"  warn: casting non-BF16 tensor (dtype={t.dtype}) to BF16 for storage")
        t = t.to(torch.bfloat16)
    return t.contiguous().view(torch.uint16).cpu().numpy()

def _add(writer: gguf.GGUFWriter, name: str, t: torch.Tensor) -> None:
    if t.dtype == torch.float32:
        writer.add_tensor(name, t.contiguous().cpu().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
    elif t.dtype == torch.bfloat16:
        writer.add_tensor(name, _bf16_u16(t), raw_shape=list(t.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        raise NotImplementedError(f"unsupported dtype {t.dtype} for {name}")

def _pad24(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size > STATE_PAD:
        raise SystemExit(f"norm-stats vector of length {a.size} exceeds {STATE_PAD}")
    out = np.zeros(STATE_PAD, dtype=np.float32)
    out[:a.size] = a
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Evo-1 checkpoint dir (mp_rank_00_model_states.pt + config.json + norm_stats.json)")
    ap.add_argument("--out", type=Path, default=None, help="output GGUF path (default: <ckpt>/evo1.gguf)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out = (args.out or ckpt / "evo1.gguf").resolve()
    pt_path = ckpt / "mp_rank_00_model_states.pt"
    cfg_path = ckpt / "config.json"
    ns_path = ckpt / "norm_stats.json"
    for p in (pt_path, cfg_path, ns_path):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    cfg_json = json.loads(cfg_path.read_text())
    if str(cfg_json.get("action_head", "")).lower() != "flowmatching":
        raise SystemExit(f"config.json action_head is {cfg_json.get('action_head')!r}, expected 'flowmatching'")

    cfg = dict(VIT, **QWEN2)
    cfg["lm_layers_used"]   = LM_LAYERS_USED
    cfg["horizon"]          = int(cfg_json["horizon"])
    cfg["per_action_dim"]   = int(cfg_json["per_action_dim"])
    cfg["state_dim"]        = int(cfg_json["state_dim"])
    cfg["action_dim"]       = int(cfg_json["action_dim"])
    cfg["dit_layers"]       = int(cfg_json.get("num_layers", 8))
    cfg["embed_dim"]        = int(cfg_json.get("embed_dim", 896))
    cfg["mlp_head_hidden"]  = int(cfg_json.get("hidden_dim", 1024))
    cfg["num_inference_timesteps"] = int(cfg_json.get("num_inference_timesteps", NUM_INFERENCE_TIMESTEPS))
    cfg["image_size"]       = int(cfg_json.get("image_size", VIT["image_size"]))
    cfg["dit_heads"]        = DIT_HEADS
    cfg["proj_ln_eps"]      = PROJ_LN_EPS
    if cfg["action_dim"] != cfg["horizon"] * cfg["per_action_dim"]:
        raise SystemExit(f"action_dim {cfg['action_dim']} != horizon*per_action_dim {cfg['horizon']*cfg['per_action_dim']}")

    print(f"loading {pt_path} ...")
    module = torch.load(pt_path, map_location="cpu", weights_only=False)["module"]
    keys = set(module.keys())
    print(f"  {len(module)} tensors")

    def _maxlayer(pfx):
        m = -1
        for k in keys:
            if k.startswith(pfx):
                try: m = max(m, int(k[len(pfx):].split(".", 1)[0]))
                except ValueError: pass
        return m + 1
    n_lm = _maxlayer("embedder.model.language_model.model.layers.")
    n_vit = _maxlayer("embedder.model.vision_model.encoder.layers.")
    n_dit = _maxlayer("action_head.transformer_blocks.")
    if n_lm != LM_LAYERS_USED:
        raise SystemExit(f"checkpoint has {n_lm} LM layers, expected {LM_LAYERS_USED} (Evo-1 truncates to layers[:14])")
    if n_vit != VIT["vit_layers"]:
        raise SystemExit(f"checkpoint has {n_vit} ViT layers, expected {VIT['vit_layers']}")
    if n_dit != cfg["dit_layers"]:
        raise SystemExit(f"checkpoint has {n_dit} DiT blocks, expected {cfg['dit_layers']}")

    grid = cfg["image_size"] // cfg["patch_size"]
    cfg["num_image_token"] = (grid // 2) ** 2

    cfg["vocab_size"] = int(module["embedder.model.language_model.model.embed_tokens.weight"].shape[0])

    cfg["img_context_token_id"] = 151667
    cfg["img_start_token_id"]   = 151665
    cfg["img_end_token_id"]     = 151666
    cfg["pad_token_id"]         = 151643
    cfg["max_text_length"]      = 1024
    cfg["n_images"]             = int(cfg_json.get("empty_cameras", 1)) if False else 3

    ns = json.loads(ns_path.read_text())
    if len(ns) != 1:
        raise SystemExit(f"norm_stats.json should have exactly one robot key; got {list(ns)}")
    robot = next(iter(ns.values()))
    state_min = _pad24(robot["observation.state"]["min"])
    state_max = _pad24(robot["observation.state"]["max"])
    action_min = _pad24(robot["action"]["min"])
    action_max = _pad24(robot["action"]["max"])
    cfg["real_state_dim"]  = int(len(robot["observation.state"]["min"]))
    cfg["real_action_dim"] = int(len(robot["action"]["min"]))
    cfg["norm_eps"]        = 1e-8

    print(f"resolved cfg: vit={cfg['vit_hidden']}d×{cfg['vit_layers']}L  lm={cfg['lm_hidden']}d×{cfg['lm_layers_used']}L "
          f"({cfg['lm_q_heads']}q/{cfg['lm_kv_heads']}kv×{cfg['lm_head_dim']})  vocab={cfg['vocab_size']}  "
          f"embed={cfg['embed_dim']}  dit={cfg['dit_layers']}L×{cfg['dit_heads']}h  horizon={cfg['horizon']} "
          f"per_a={cfg['per_action_dim']}  N_steps={cfg['num_inference_timesteps']}  "
          f"img_tok={cfg['num_image_token']}  n_img={cfg['n_images']}  real_state={cfg['real_state_dim']} "
          f"real_action={cfg['real_action_dim']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    writer.add_string(KV("architecture"), ARCH)
    for k in ("vit_hidden", "vit_layers", "vit_heads", "vit_inter", "image_size", "patch_size",
              "num_image_token", "lm_hidden", "lm_layers_used", "lm_q_heads", "lm_kv_heads",
              "lm_head_dim", "lm_inter", "vocab_size", "embed_dim", "dit_layers", "dit_heads",
              "mlp_head_hidden", "horizon", "per_action_dim", "state_dim", "action_dim",
              "num_inference_timesteps", "real_state_dim", "real_action_dim", "max_text_length",
              "n_images", "img_context_token_id", "img_start_token_id", "img_end_token_id",
              "pad_token_id"):
        writer.add_uint32(KV(k), int(cfg[k]))
    writer.add_float64(KV("lm_rope_theta"), float(cfg["lm_rope_theta"]))
    writer.add_float32(KV("lm_rms_eps"),    float(cfg["lm_rms_eps"]))
    writer.add_float32(KV("vit_ln_eps"),    float(cfg["vit_ln_eps"]))
    writer.add_float32(KV("proj_ln_eps"),   float(cfg["proj_ln_eps"]))
    writer.add_float32(KV("norm_eps"),      float(cfg["norm_eps"]))

    g = lambda name: module[name]

    VE = "embedder.model.vision_model.embeddings."
    _add(writer, "vit.patch_embd.weight", g(VE + "patch_embedding.weight"))
    _add(writer, "vit.patch_embd.bias",   g(VE + "patch_embedding.bias"))
    _add(writer, "vit.class_embd",         g(VE + "class_embedding"))
    _add(writer, "vit.pos_embd",           g(VE + "position_embedding"))
    for i in range(cfg["vit_layers"]):
        VL = f"embedder.model.vision_model.encoder.layers.{i}."
        _add(writer, f"vit.blk.{i}.norm1.weight", g(VL + "norm1.weight")); _add(writer, f"vit.blk.{i}.norm1.bias", g(VL + "norm1.bias"))
        _add(writer, f"vit.blk.{i}.norm2.weight", g(VL + "norm2.weight")); _add(writer, f"vit.blk.{i}.norm2.bias", g(VL + "norm2.bias"))
        _add(writer, f"vit.blk.{i}.ls1", g(VL + "ls1")); _add(writer, f"vit.blk.{i}.ls2", g(VL + "ls2"))
        _add(writer, f"vit.blk.{i}.attn_qkv.weight", g(VL + "attn.qkv.weight")); _add(writer, f"vit.blk.{i}.attn_qkv.bias", g(VL + "attn.qkv.bias"))
        _add(writer, f"vit.blk.{i}.attn_proj.weight", g(VL + "attn.proj.weight")); _add(writer, f"vit.blk.{i}.attn_proj.bias", g(VL + "attn.proj.bias"))
        _add(writer, f"vit.blk.{i}.fc1.weight", g(VL + "mlp.fc1.weight")); _add(writer, f"vit.blk.{i}.fc1.bias", g(VL + "mlp.fc1.bias"))
        _add(writer, f"vit.blk.{i}.fc2.weight", g(VL + "mlp.fc2.weight")); _add(writer, f"vit.blk.{i}.fc2.bias", g(VL + "mlp.fc2.bias"))

    _add(writer, "mm.ln.weight",  g("embedder.model.mlp1.0.weight")); _add(writer, "mm.ln.bias",  g("embedder.model.mlp1.0.bias"))
    _add(writer, "mm.fc1.weight", g("embedder.model.mlp1.1.weight")); _add(writer, "mm.fc1.bias", g("embedder.model.mlp1.1.bias"))
    _add(writer, "mm.fc2.weight", g("embedder.model.mlp1.3.weight")); _add(writer, "mm.fc2.bias", g("embedder.model.mlp1.3.bias"))

    _add(writer, "token_embd.weight",      g("embedder.model.language_model.model.embed_tokens.weight"))
    _add(writer, "vlm.output_norm.weight", g("embedder.model.language_model.model.norm.weight"))
    for i in range(cfg["lm_layers_used"]):
        LL = f"embedder.model.language_model.model.layers.{i}."
        _add(writer, f"vlm.blk.{i}.attn_norm.weight", g(LL + "input_layernorm.weight"))
        _add(writer, f"vlm.blk.{i}.attn_q.weight", g(LL + "self_attn.q_proj.weight")); _add(writer, f"vlm.blk.{i}.attn_q.bias", g(LL + "self_attn.q_proj.bias"))
        _add(writer, f"vlm.blk.{i}.attn_k.weight", g(LL + "self_attn.k_proj.weight")); _add(writer, f"vlm.blk.{i}.attn_k.bias", g(LL + "self_attn.k_proj.bias"))
        _add(writer, f"vlm.blk.{i}.attn_v.weight", g(LL + "self_attn.v_proj.weight")); _add(writer, f"vlm.blk.{i}.attn_v.bias", g(LL + "self_attn.v_proj.bias"))
        _add(writer, f"vlm.blk.{i}.attn_o.weight", g(LL + "self_attn.o_proj.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_norm.weight", g(LL + "post_attention_layernorm.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_gate.weight", g(LL + "mlp.gate_proj.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_up.weight",   g(LL + "mlp.up_proj.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_down.weight", g(LL + "mlp.down_proj.weight"))

    AE = "action_head.action_encoder."
    for w in ("W1", "W2", "W3"):
        _add(writer, f"aex.ae.{w}.weight", g(AE + f"{w}.linear.weight")); _add(writer, f"aex.ae.{w}.bias", g(AE + f"{w}.linear.bias"))
    _add(writer, "aex.ae.pos_enc", g(AE + "pos_encoding.pe"))
    for i in range(cfg["dit_layers"]):
        TB = f"action_head.transformer_blocks.{i}."
        _add(writer, f"aex.blk.{i}.norm1.weight", g(TB + "norm1.weight")); _add(writer, f"aex.blk.{i}.norm1.bias", g(TB + "norm1.bias"))
        _add(writer, f"aex.blk.{i}.norm2.weight", g(TB + "norm2.weight")); _add(writer, f"aex.blk.{i}.norm2.bias", g(TB + "norm2.bias"))
        _add(writer, f"aex.blk.{i}.attn_in.weight", g(TB + "attn.in_proj_weight")); _add(writer, f"aex.blk.{i}.attn_in.bias", g(TB + "attn.in_proj_bias"))
        _add(writer, f"aex.blk.{i}.attn_out.weight", g(TB + "attn.out_proj.weight")); _add(writer, f"aex.blk.{i}.attn_out.bias", g(TB + "attn.out_proj.bias"))
        _add(writer, f"aex.blk.{i}.ff1.weight", g(TB + "ff.0.weight")); _add(writer, f"aex.blk.{i}.ff1.bias", g(TB + "ff.0.bias"))
        _add(writer, f"aex.blk.{i}.ff2.weight", g(TB + "ff.2.weight")); _add(writer, f"aex.blk.{i}.ff2.bias", g(TB + "ff.2.bias"))
    _add(writer, "aex.norm_out.weight", g("action_head.norm_out.weight")); _add(writer, "aex.norm_out.bias", g("action_head.norm_out.bias"))
    _add(writer, "aex.seq_pool.weight", g("action_head.seq_pool_proj.weight")); _add(writer, "aex.seq_pool.bias", g("action_head.seq_pool_proj.bias"))
    _add(writer, "aex.head.fc1.weight", g("action_head.mlp_head.fc1.linear.weight")); _add(writer, "aex.head.fc1.bias", g("action_head.mlp_head.fc1.linear.bias"))
    _add(writer, "aex.head.fc2.weight", g("action_head.mlp_head.fc2.linear.weight")); _add(writer, "aex.head.fc2.bias", g("action_head.mlp_head.fc2.linear.bias"))
    _add(writer, "aex.time_pos_enc", g("action_head.time_pos_enc.pe"))
    _add(writer, "aex.state_enc.fc1.weight", g("action_head.state_encoder.fc1.linear.weight")); _add(writer, "aex.state_enc.fc1.bias", g("action_head.state_encoder.fc1.linear.bias"))
    _add(writer, "aex.state_enc.fc2.weight", g("action_head.state_encoder.fc2.linear.weight")); _add(writer, "aex.state_enc.fc2.bias", g("action_head.state_encoder.fc2.linear.bias"))

    writer.add_tensor("state_min",  state_min,  raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("state_max",  state_max,  raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("action_min", action_min, raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("action_max", action_max, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB)  - combined GGUF (vision + LM + action head + stats + cfg)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
