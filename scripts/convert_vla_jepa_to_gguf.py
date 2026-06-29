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
from safetensors import safe_open

import gguf

ARCH = "vla_jepa"
KV = lambda name: f"{ARCH}.{name}"

VIT = dict(vit_hidden=1024, vit_layers=24, vit_heads=16, vit_inter=4096,
           patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
           vit_num_position_embeddings=2304, vit_ln_eps=1e-6)
DEEPSTACK_IDXS = [5, 11, 17]

QWEN3 = dict(lm_hidden=2048, lm_q_heads=16, lm_kv_heads=8, lm_head_dim=128,
             lm_inter=6144, lm_rope_theta=5000000.0, lm_rms_eps=1e-6)
LM_LAYERS = 28

AH = dict(dit_hidden=768, dit_heads=12, dit_head_dim=64, dit_layers=16,
          cross_dim=2048, output_dim=1024,
          action_dim=7, state_dim=8, action_horizon=7, num_future_tokens=32,
          time_proj_dim=256, num_inference_timesteps=4, num_timestep_buckets=1000,
          ln_eps=1e-5, norm_out_eps=1e-6)

IMAGE_TOKEN_INDEX = 151655
EMBODIED_ACTION_TOKEN_ID = 151697
ACTION_TOKEN_ID_0 = 151669

def _bf16_u16(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous().view(torch.uint16).cpu().numpy()

def _add(writer: gguf.GGUFWriter, name: str, t: torch.Tensor) -> None:
    if t.dtype == torch.float32:
        writer.add_tensor(name, t.contiguous().cpu().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
    elif t.dtype == torch.bfloat16:
        writer.add_tensor(name, _bf16_u16(t), raw_shape=list(t.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        raise NotImplementedError(f"unsupported dtype {t.dtype} for {name}")

def _load_sharded(ckpt: Path) -> dict[str, torch.Tensor]:
    idx = ckpt / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
    else:
        one = ckpt / "model.safetensors"
        if not one.exists():
            raise SystemExit(f"no model.safetensors[.index.json] under {ckpt}")
        with safe_open(str(one), framework="pt") as f:
            weight_map = {k: one.name for k in f.keys()}
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)
    out: dict[str, torch.Tensor] = {}
    KEEP = ("model.qwen.", "model.action_model.")
    for shard in sorted(by_shard):
        with safe_open(str(ckpt / shard), framework="pt") as f:
            for k in by_shard[shard]:
                if k.startswith(KEEP):
                    out[k] = f.get_tensor(k)
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="VLA-JEPA-LIBERO checkpoint dir")
    ap.add_argument("--out", type=Path, default=None, help="output GGUF path (default: <ckpt>/vla_jepa.gguf)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out = (args.out or ckpt / f"{ARCH}.gguf").resolve()
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")
    cfg_json = json.loads(cfg_path.read_text())
    if str(cfg_json.get("type", "")) != "vla_jepa":
        raise SystemExit(f"config.json type is {cfg_json.get('type')!r}, expected 'vla_jepa'")
    AH["action_dim"] = int(cfg_json.get("action_dim", AH["action_dim"]))
    AH["state_dim"] = int(cfg_json.get("state_dim", AH["state_dim"]))
    AH["action_horizon"] = int(cfg_json.get("chunk_size", AH["action_horizon"]))
    AH["num_future_tokens"] = int(cfg_json.get("num_embodied_action_tokens_per_instruction", AH["num_future_tokens"]))
    AH["num_inference_timesteps"] = int(cfg_json.get("num_inference_timesteps", AH["num_inference_timesteps"]))
    AH["num_timestep_buckets"] = int(cfg_json.get("action_num_timestep_buckets", AH["num_timestep_buckets"]))
    AH["output_dim"] = int(cfg_json.get("action_hidden_size", AH["output_dim"]))
    n_layers_cfg = int(cfg_json.get("action_num_layers", AH["dit_layers"]))
    if n_layers_cfg != AH["dit_layers"]:
        AH["dit_layers"] = n_layers_cfg

    print(f"loading safetensors from {ckpt} (dropping world model) ...")
    W = _load_sharded(ckpt)
    keys = set(W.keys())
    print(f"  {len(W)} kept tensors")

    def _maxlayer(pfx):
        m = -1
        for k in keys:
            if k.startswith(pfx):
                try: m = max(m, int(k[len(pfx):].split(".", 1)[0]))
                except ValueError: pass
        return m + 1
    VS = "model.qwen.model.model.visual."
    LM = "model.qwen.model.model.language_model."
    AHK = "model.action_model."
    n_vit = _maxlayer(VS + "blocks.")
    n_lm = _maxlayer(LM + "layers.")
    n_dit = _maxlayer(AHK + "model.transformer_blocks.")
    n_ds = _maxlayer(VS + "deepstack_merger_list.")
    if n_vit != VIT["vit_layers"]: raise SystemExit(f"checkpoint has {n_vit} ViT layers, expected {VIT['vit_layers']}")
    if n_lm != LM_LAYERS:          raise SystemExit(f"checkpoint has {n_lm} LM layers, expected {LM_LAYERS}")
    if n_dit != AH["dit_layers"]:  raise SystemExit(f"checkpoint has {n_dit} DiT blocks, expected {AH['dit_layers']}")
    if n_ds != len(DEEPSTACK_IDXS):raise SystemExit(f"checkpoint has {n_ds} deepstack mergers, expected {len(DEEPSTACK_IDXS)}")

    vocab = int(W[LM + "embed_tokens.weight"].shape[0])
    pe_w = W[VS + "patch_embed.proj.weight"]
    assert pe_w.shape == (VIT["vit_hidden"], 3, VIT["temporal_patch_size"], VIT["patch_size"], VIT["patch_size"]), pe_w.shape
    patch_flat = 3 * VIT["temporal_patch_size"] * VIT["patch_size"] ** 2
    c_merged = VIT["vit_hidden"] * VIT["spatial_merge_size"] ** 2
    assert W[VS + "merger.norm.weight"].shape == (VIT["vit_hidden"],), W[VS + "merger.norm.weight"].shape
    assert W[VS + "deepstack_merger_list.0.norm.weight"].shape == (c_merged,)
    assert W[AHK + "model.transformer_blocks.0.attn1.to_k.weight"].shape == (AH["dit_hidden"], AH["cross_dim"])
    assert W[AHK + "model.transformer_blocks.1.attn1.to_k.weight"].shape == (AH["dit_hidden"], AH["dit_hidden"])

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"resolved: vit=Qwen3-VL {VIT['vit_hidden']}d×{VIT['vit_layers']}L (deepstack@{DEEPSTACK_IDXS}, merge÷{VIT['spatial_merge_size']})  "
          f"lm=Qwen3-VL {QWEN3['lm_hidden']}d×{LM_LAYERS}L ({QWEN3['lm_q_heads']}q/{QWEN3['lm_kv_heads']}kv×{QWEN3['lm_head_dim']}, θ={QWEN3['lm_rope_theta']:g})  vocab={vocab}  "
          f"dit-B {AH['dit_layers']}L×{AH['dit_heads']}h×{AH['dit_head_dim']}(inner {AH['dit_hidden']}, cross {AH['cross_dim']}, out {AH['output_dim']})  "
          f"horizon={AH['action_horizon']} action_dim={AH['action_dim']} state_dim={AH['state_dim']} future={AH['num_future_tokens']} N_steps={AH['num_inference_timesteps']}  "
          f"img_tok={IMAGE_TOKEN_INDEX} emb_tok={EMBODIED_ACTION_TOKEN_ID}")
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    writer.add_string(KV("architecture"), ARCH)
    u32 = dict(
        vit_hidden=VIT["vit_hidden"], vit_layers=VIT["vit_layers"], vit_heads=VIT["vit_heads"], vit_inter=VIT["vit_inter"],
        patch_size=VIT["patch_size"], temporal_patch_size=VIT["temporal_patch_size"], spatial_merge_size=VIT["spatial_merge_size"],
        vit_num_position_embeddings=VIT["vit_num_position_embeddings"], vit_patch_flat=patch_flat, vit_merged_dim=c_merged,
        image_target_size=256,
        deepstack_idx_0=DEEPSTACK_IDXS[0], deepstack_idx_1=DEEPSTACK_IDXS[1], deepstack_idx_2=DEEPSTACK_IDXS[2],
        lm_hidden=QWEN3["lm_hidden"], lm_layers=LM_LAYERS, lm_q_heads=QWEN3["lm_q_heads"], lm_kv_heads=QWEN3["lm_kv_heads"],
        lm_head_dim=QWEN3["lm_head_dim"], lm_inter=QWEN3["lm_inter"],
        vocab_size=vocab, image_token_index=IMAGE_TOKEN_INDEX, embodied_action_token_id=EMBODIED_ACTION_TOKEN_ID,
        action_token_id_0=ACTION_TOKEN_ID_0,
        dit_hidden=AH["dit_hidden"], dit_heads=AH["dit_heads"], dit_head_dim=AH["dit_head_dim"], dit_layers=AH["dit_layers"],
        cross_dim=AH["cross_dim"], output_dim=AH["output_dim"], time_proj_dim=AH["time_proj_dim"],
        action_dim=AH["action_dim"], state_dim=AH["state_dim"], action_horizon=AH["action_horizon"],
        num_future_tokens=AH["num_future_tokens"], num_inference_timesteps=AH["num_inference_timesteps"],
        num_timestep_buckets=AH["num_timestep_buckets"],
    )
    for k, v in u32.items():
        writer.add_uint32(KV(k), int(v))
    writer.add_float64(KV("lm_rope_theta"),  float(QWEN3["lm_rope_theta"]))
    writer.add_float32(KV("lm_rms_eps"),     float(QWEN3["lm_rms_eps"]))
    writer.add_float32(KV("vit_ln_eps"),     float(VIT["vit_ln_eps"]))
    writer.add_float32(KV("vit_rope_theta"), 10000.0)
    writer.add_float32(KV("connector_ln_eps"), 1e-6)
    writer.add_float32(KV("dit_ln_eps"),     float(AH["ln_eps"]))
    writer.add_float32(KV("dit_norm_out_eps"), float(AH["norm_out_eps"]))

    g = lambda name: W[name]

    _add(writer, "vit.patch_embd.weight", pe_w.reshape(VIT["vit_hidden"], patch_flat))
    _add(writer, "vit.patch_embd.bias",   g(VS + "patch_embed.proj.bias"))
    _add(writer, "vit.pos_embd",          g(VS + "pos_embed.weight"))
    for i in range(VIT["vit_layers"]):
        VL = f"{VS}blocks.{i}."
        _add(writer, f"vit.blk.{i}.ln1.weight", g(VL + "norm1.weight")); _add(writer, f"vit.blk.{i}.ln1.bias", g(VL + "norm1.bias"))
        _add(writer, f"vit.blk.{i}.ln2.weight", g(VL + "norm2.weight")); _add(writer, f"vit.blk.{i}.ln2.bias", g(VL + "norm2.bias"))
        _add(writer, f"vit.blk.{i}.attn_qkv.weight", g(VL + "attn.qkv.weight")); _add(writer, f"vit.blk.{i}.attn_qkv.bias", g(VL + "attn.qkv.bias"))
        _add(writer, f"vit.blk.{i}.attn_o.weight", g(VL + "attn.proj.weight")); _add(writer, f"vit.blk.{i}.attn_o.bias", g(VL + "attn.proj.bias"))
        _add(writer, f"vit.blk.{i}.fc1.weight", g(VL + "mlp.linear_fc1.weight")); _add(writer, f"vit.blk.{i}.fc1.bias", g(VL + "mlp.linear_fc1.bias"))
        _add(writer, f"vit.blk.{i}.fc2.weight", g(VL + "mlp.linear_fc2.weight")); _add(writer, f"vit.blk.{i}.fc2.bias", g(VL + "mlp.linear_fc2.bias"))
    for j in range(len(DEEPSTACK_IDXS)):
        DM = f"{VS}deepstack_merger_list.{j}."
        _add(writer, f"vit.deepstack.{j}.norm.weight", g(DM + "norm.weight")); _add(writer, f"vit.deepstack.{j}.norm.bias", g(DM + "norm.bias"))
        _add(writer, f"vit.deepstack.{j}.fc1.weight", g(DM + "linear_fc1.weight")); _add(writer, f"vit.deepstack.{j}.fc1.bias", g(DM + "linear_fc1.bias"))
        _add(writer, f"vit.deepstack.{j}.fc2.weight", g(DM + "linear_fc2.weight")); _add(writer, f"vit.deepstack.{j}.fc2.bias", g(DM + "linear_fc2.bias"))
    MG = f"{VS}merger."
    _add(writer, "vit.merger.norm.weight", g(MG + "norm.weight")); _add(writer, "vit.merger.norm.bias", g(MG + "norm.bias"))
    _add(writer, "vit.merger.fc1.weight", g(MG + "linear_fc1.weight")); _add(writer, "vit.merger.fc1.bias", g(MG + "linear_fc1.bias"))
    _add(writer, "vit.merger.fc2.weight", g(MG + "linear_fc2.weight")); _add(writer, "vit.merger.fc2.bias", g(MG + "linear_fc2.bias"))

    _add(writer, "token_embd.weight",      g(LM + "embed_tokens.weight"))
    _add(writer, "vlm.output_norm.weight", g(LM + "norm.weight"))
    for i in range(LM_LAYERS):
        LL = f"{LM}layers.{i}."
        _add(writer, f"vlm.blk.{i}.attn_norm.weight", g(LL + "input_layernorm.weight"))
        for q in ("q", "k", "v"):
            _add(writer, f"vlm.blk.{i}.attn_{q}.weight", g(LL + f"self_attn.{q}_proj.weight"))
        _add(writer, f"vlm.blk.{i}.attn_o.weight", g(LL + "self_attn.o_proj.weight"))
        _add(writer, f"vlm.blk.{i}.attn_q_norm.weight", g(LL + "self_attn.q_norm.weight"))
        _add(writer, f"vlm.blk.{i}.attn_k_norm.weight", g(LL + "self_attn.k_norm.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_norm.weight", g(LL + "post_attention_layernorm.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_gate.weight", g(LL + "mlp.gate_proj.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_up.weight",   g(LL + "mlp.up_proj.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_down.weight", g(LL + "mlp.down_proj.weight"))

    for src, dst in (("action_encoder.layer1", "ah.act_enc.l1"), ("action_encoder.layer2", "ah.act_enc.l2"),
                     ("action_encoder.layer3", "ah.act_enc.l3"),
                     ("state_encoder.layer1", "ah.state_enc.l1"), ("state_encoder.layer2", "ah.state_enc.l2"),
                     ("action_decoder.layer1", "ah.act_dec.l1"), ("action_decoder.layer2", "ah.act_dec.l2"),
                     ("model.timestep_encoder.timestep_embedder.linear_1", "ah.time_emb.l1"),
                     ("model.timestep_encoder.timestep_embedder.linear_2", "ah.time_emb.l2"),
                     ("model.proj_out_1", "ah.proj_out1"), ("model.proj_out_2", "ah.proj_out2")):
        _add(writer, f"{dst}.weight", g(AHK + src + ".weight")); _add(writer, f"{dst}.bias", g(AHK + src + ".bias"))
    _add(writer, "ah.future_tokens", g(AHK + "future_tokens.weight"))
    _add(writer, "ah.pos_embd",      g(AHK + "position_embedding.weight"))
    for i in range(AH["dit_layers"]):
        TB = f"{AHK}model.transformer_blocks.{i}."
        _add(writer, f"ah.dit.{i}.adaln.weight", g(TB + "norm1.linear.weight")); _add(writer, f"ah.dit.{i}.adaln.bias", g(TB + "norm1.linear.bias"))
        for q in ("q", "k", "v"):
            _add(writer, f"ah.dit.{i}.attn_{q}.weight", g(TB + f"attn1.to_{q}.weight")); _add(writer, f"ah.dit.{i}.attn_{q}.bias", g(TB + f"attn1.to_{q}.bias"))
        _add(writer, f"ah.dit.{i}.attn_o.weight", g(TB + "attn1.to_out.0.weight")); _add(writer, f"ah.dit.{i}.attn_o.bias", g(TB + "attn1.to_out.0.bias"))
        _add(writer, f"ah.dit.{i}.ff0.weight", g(TB + "ff.net.0.proj.weight")); _add(writer, f"ah.dit.{i}.ff0.bias", g(TB + "ff.net.0.proj.bias"))
        _add(writer, f"ah.dit.{i}.ff2.weight", g(TB + "ff.net.2.weight")); _add(writer, f"ah.dit.{i}.ff2.bias", g(TB + "ff.net.2.bias"))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB) - combined GGUF "
          f"(Qwen3-VL backbone + deepstack + DiT-B action head + cfg)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
