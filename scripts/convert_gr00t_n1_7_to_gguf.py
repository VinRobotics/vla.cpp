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

ARCH = "gr00t_n1_7"
KV = lambda name: f"{ARCH}.{name}"

VIT = dict(vit_hidden=1024, vit_layers=24, vit_heads=16, vit_inter=4096,
           patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
           vit_num_position_embeddings=2304, vit_ln_eps=1e-6)
DEEPSTACK_IDXS = [5, 11, 17]

QWEN3 = dict(lm_hidden=2048, lm_q_heads=16, lm_kv_heads=8, lm_head_dim=128,
             lm_inter=6144, lm_rope_theta=5000000.0, lm_rms_eps=1e-6)
LM_LAYERS_USED = 16

AH = dict(backbone_embedding_dim=2048, input_embedding_dim=1536,
          dit_hidden=1536, dit_heads=32, dit_head_dim=48, dit_layers=32, dit_interleave=1,
          attend_text_every_n_blocks=2,
          vlsa_layers=4, vlsa_heads=32, vlsa_head_dim=64,
          action_horizon=40, action_dim=132, max_state_dim=132,
          num_inference_timesteps=4, num_timestep_buckets=1000, max_num_embodiments=32,
          max_seq_len=1024, ln_eps=1e-5, norm_out_eps=1e-6, vlln_eps=1e-5, vlsa_ln_eps=1e-5)
IMAGE_TOKEN_INDEX = 151655

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
    for shard in sorted(by_shard):
        with safe_open(str(ckpt / shard), framework="pt") as f:
            for k in by_shard[shard]:
                out[k] = f.get_tensor(k)
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="GR00T-N1.7-3B snapshot dir")
    ap.add_argument("--out", type=Path, default=None, help="output GGUF path (default: <ckpt>/gr00t_n1_7.gguf)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out = (args.out or ckpt / f"{ARCH}.gguf").resolve()
    cfg_path = ckpt / "config.json"
    for p in (cfg_path, ckpt / "model.safetensors.index.json"):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    cfg_json = json.loads(cfg_path.read_text())
    if str(cfg_json.get("model_type", "")) != "Gr00tN1d7":
        raise SystemExit(f"config.json model_type is {cfg_json.get('model_type')!r}, expected 'Gr00tN1d7'")
    if int(cfg_json.get("select_layer", LM_LAYERS_USED)) != LM_LAYERS_USED:
        raise SystemExit(f"select_layer = {cfg_json.get('select_layer')}, expected {LM_LAYERS_USED}")
    AH["action_horizon"] = int(cfg_json.get("action_horizon", AH["action_horizon"]))
    AH["action_dim"] = int(cfg_json.get("max_action_dim", AH["action_dim"]))
    AH["max_state_dim"] = int(cfg_json.get("max_state_dim", AH["max_state_dim"]))
    AH["num_inference_timesteps"] = int(cfg_json.get("num_inference_timesteps", AH["num_inference_timesteps"]))
    AH["num_timestep_buckets"] = int(cfg_json.get("num_timestep_buckets", AH["num_timestep_buckets"]))
    AH["max_num_embodiments"] = int(cfg_json.get("max_num_embodiments", AH["max_num_embodiments"]))
    AH["max_seq_len"] = int(cfg_json.get("max_seq_len", AH["max_seq_len"]))
    AH["attend_text_every_n_blocks"] = int(cfg_json.get("attend_text_every_n_blocks", AH["attend_text_every_n_blocks"]))
    AH["input_embedding_dim"] = int(cfg_json.get("input_embedding_dim", AH["input_embedding_dim"]))
    AH["backbone_embedding_dim"] = int(cfg_json.get("backbone_embedding_dim", AH["backbone_embedding_dim"]))
    dmc = cfg_json.get("diffusion_model_cfg", {})
    AH["dit_layers"] = int(dmc.get("num_layers", AH["dit_layers"]))
    AH["dit_heads"] = int(dmc.get("num_attention_heads", AH["dit_heads"]))
    AH["dit_head_dim"] = int(dmc.get("attention_head_dim", AH["dit_head_dim"]))
    AH["dit_hidden"] = AH["dit_heads"] * AH["dit_head_dim"]
    AH["dit_interleave"] = int(bool(dmc.get("interleave_self_attention", True)))
    vsac = cfg_json.get("vl_self_attention_cfg", {})
    AH["vlsa_layers"] = int(vsac.get("num_layers", AH["vlsa_layers"]))
    AH["vlsa_heads"] = int(vsac.get("num_attention_heads", AH["vlsa_heads"]))
    AH["vlsa_head_dim"] = int(vsac.get("attention_head_dim", AH["vlsa_head_dim"]))
    SHORTEST_EDGE = int(cfg_json.get("shortest_image_edge", 256) or 256)
    CROP_FRACTION = float(cfg_json.get("crop_fraction", 0.95) or 0.95)
    ICS = cfg_json.get("image_crop_size", [230, 230]) or [230, 230]
    ITS = cfg_json.get("image_target_size", [256, 256]) or [256, 256]
    USE_RELATIVE_ACTION = bool(cfg_json.get("use_relative_action", False))
    APPLY_SINCOS_STATE = bool(cfg_json.get("apply_sincos_state_encoding", False))

    print(f"loading sharded safetensors from {ckpt} ...")
    W = _load_sharded(ckpt)
    keys = set(W.keys())
    print(f"  {len(W)} tensors")

    def _maxlayer(pfx):
        m = -1
        for k in keys:
            if k.startswith(pfx):
                try: m = max(m, int(k[len(pfx):].split(".", 1)[0]))
                except ValueError: pass
        return m + 1
    VS = "backbone.model.model.visual."
    LM = "backbone.model.model.language_model."
    n_vit = _maxlayer(VS + "blocks.")
    n_lm  = _maxlayer(LM + "layers.")
    n_dit = _maxlayer("action_head.model.transformer_blocks.")
    n_vsa = _maxlayer("action_head.vl_self_attention.transformer_blocks.")
    n_ds  = _maxlayer(VS + "deepstack_merger_list.")
    if n_vit != VIT["vit_layers"]:   raise SystemExit(f"checkpoint has {n_vit} Qwen3-VL ViT layers, expected {VIT['vit_layers']}")
    if n_lm  != LM_LAYERS_USED:      raise SystemExit(f"checkpoint has {n_lm} Qwen3-VL text layers, expected {LM_LAYERS_USED}")
    if n_dit != AH["dit_layers"]:    raise SystemExit(f"checkpoint has {n_dit} DiT blocks, expected {AH['dit_layers']}")
    if n_vsa != AH["vlsa_layers"]:   raise SystemExit(f"checkpoint has {n_vsa} vl_self_attention blocks, expected {AH['vlsa_layers']}")
    if n_ds  != len(DEEPSTACK_IDXS): raise SystemExit(f"checkpoint has {n_ds} deepstack mergers, expected {len(DEEPSTACK_IDXS)}")

    vocab = int(W[LM + "embed_tokens.weight"].shape[0])
    pe_w = W[VS + "patch_embed.proj.weight"]
    assert pe_w.shape == (VIT["vit_hidden"], 3, VIT["temporal_patch_size"], VIT["patch_size"], VIT["patch_size"]), pe_w.shape
    patch_flat = 3 * VIT["temporal_patch_size"] * VIT["patch_size"] ** 2
    c_merged = VIT["vit_hidden"] * VIT["spatial_merge_size"] ** 2
    assert W[VS + "merger.linear_fc1.weight"].shape == (c_merged, c_merged)
    assert W[VS + "merger.linear_fc2.weight"].shape == (AH["backbone_embedding_dim"], c_merged)
    assert W[VS + "merger.norm.weight"].shape == (VIT["vit_hidden"],), W[VS + "merger.norm.weight"].shape
    assert W[VS + "deepstack_merger_list.0.norm.weight"].shape == (c_merged,)
    vlsa_ff_inner = 4 * AH["backbone_embedding_dim"]
    assert W["action_head.vl_self_attention.transformer_blocks.0.ff.net.0.proj.weight"].shape == (vlsa_ff_inner, AH["backbone_embedding_dim"])

    statistics_json = (ckpt / "statistics.json").read_text() if (ckpt / "statistics.json").exists() else "{}"
    processor_json = (ckpt / "processor_config.json").read_text() if (ckpt / "processor_config.json").exists() else "{}"
    embodiment_id_json = (ckpt / "embodiment_id.json").read_text() if (ckpt / "embodiment_id.json").exists() else "{}"
    proc_kwargs = json.loads(processor_json).get("processor_kwargs", {}) if processor_json != "{}" else {}
    USE_PERCENTILES = bool(proc_kwargs.get("use_percentiles", True))
    CLIP_OUTLIERS = bool(proc_kwargs.get("clip_outliers", True))

    print(f"resolved cfg: vit=Qwen3-VL {VIT['vit_hidden']}d×{VIT['vit_layers']}L×{VIT['vit_heads']}h (Conv3d patch {VIT['patch_size']}², temporal {VIT['temporal_patch_size']}, "
          f"learned pos {VIT['vit_num_position_embeddings']}=48² + 2D rope; deepstack@{DEEPSTACK_IDXS}; merger LN={VIT['vit_hidden']} pre-merge / deepstack LN={c_merged} post-merge ⇒ "
          f"merge÷{VIT['spatial_merge_size']})  lm=Qwen3-VL {QWEN3['lm_hidden']}d×{LM_LAYERS_USED}L ({QWEN3['lm_q_heads']}q/{QWEN3['lm_kv_heads']}kv×{QWEN3['lm_head_dim']}, θ={QWEN3['lm_rope_theta']:g})  "
          f"vocab={vocab} img_tok={IMAGE_TOKEN_INDEX}  vlsa={AH['vlsa_layers']}L×{AH['vlsa_heads']}h×{AH['vlsa_head_dim']} ff{vlsa_ff_inner}  "
          f"dit=AlternateVLDiT {AH['dit_layers']}L×{AH['dit_heads']}h×{AH['dit_head_dim']}(inner {AH['dit_hidden']}) attend_text_every_n={AH['attend_text_every_n_blocks']}  "
          f"in_emb={AH['input_embedding_dim']}  horizon={AH['action_horizon']} action_dim={AH['action_dim']} max_state={AH['max_state_dim']}  N_steps={AH['num_inference_timesteps']}  "
          f"embodiments={AH['max_num_embodiments']}  relative={USE_RELATIVE_ACTION} percentiles={USE_PERCENTILES} clip={CLIP_OUTLIERS} sincos={APPLY_SINCOS_STATE}  "
          f"img: shortest_edge={SHORTEST_EDGE} crop_frac={CROP_FRACTION} crop_size={ICS} target_size={ITS}  stats={len(statistics_json)}c proc={len(processor_json)}c emb_id={embodiment_id_json.strip()}")

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    writer.add_string(KV("architecture"), ARCH)
    u32 = dict(
        vit_hidden=VIT["vit_hidden"], vit_layers=VIT["vit_layers"], vit_heads=VIT["vit_heads"], vit_inter=VIT["vit_inter"],
        patch_size=VIT["patch_size"], temporal_patch_size=VIT["temporal_patch_size"], spatial_merge_size=VIT["spatial_merge_size"],
        vit_num_position_embeddings=VIT["vit_num_position_embeddings"], vit_patch_flat=patch_flat, vit_merged_dim=c_merged,
        n_img_tokens_per_view=64,
        shortest_image_edge=SHORTEST_EDGE, image_crop_size=int(ICS[0]), image_target_size=int(ITS[0]),
        deepstack_idx_0=DEEPSTACK_IDXS[0], deepstack_idx_1=DEEPSTACK_IDXS[1], deepstack_idx_2=DEEPSTACK_IDXS[2],
        lm_hidden=QWEN3["lm_hidden"], lm_layers_used=LM_LAYERS_USED, lm_q_heads=QWEN3["lm_q_heads"],
        lm_kv_heads=QWEN3["lm_kv_heads"], lm_head_dim=QWEN3["lm_head_dim"], lm_inter=QWEN3["lm_inter"],
        vocab_size=vocab, image_token_index=IMAGE_TOKEN_INDEX,
        vlsa_layers=AH["vlsa_layers"], vlsa_heads=AH["vlsa_heads"], vlsa_head_dim=AH["vlsa_head_dim"], vlsa_ff_inner=vlsa_ff_inner,
        backbone_embedding_dim=AH["backbone_embedding_dim"], input_embedding_dim=AH["input_embedding_dim"],
        dit_hidden=AH["dit_hidden"], dit_heads=AH["dit_heads"], dit_head_dim=AH["dit_head_dim"], dit_layers=AH["dit_layers"],
        dit_interleave=AH["dit_interleave"], attend_text_every_n_blocks=AH["attend_text_every_n_blocks"],
        action_horizon=AH["action_horizon"], action_dim=AH["action_dim"], max_state_dim=AH["max_state_dim"],
        num_inference_timesteps=AH["num_inference_timesteps"], num_timestep_buckets=AH["num_timestep_buckets"],
        max_num_embodiments=AH["max_num_embodiments"], max_seq_len=AH["max_seq_len"],
        use_relative_action=int(USE_RELATIVE_ACTION), use_percentiles=int(USE_PERCENTILES),
        clip_outliers=int(CLIP_OUTLIERS), apply_sincos_state_encoding=int(APPLY_SINCOS_STATE),
    )
    for k, v in u32.items():
        writer.add_uint32(KV(k), int(v))
    writer.add_float64(KV("lm_rope_theta"),  float(QWEN3["lm_rope_theta"]))
    writer.add_float32(KV("lm_rms_eps"),     float(QWEN3["lm_rms_eps"]))
    writer.add_float32(KV("vit_ln_eps"),     float(VIT["vit_ln_eps"]))
    writer.add_float32(KV("vit_rope_theta"), 10000.0)
    writer.add_float32(KV("ln_eps"),         float(AH["ln_eps"]))
    writer.add_float32(KV("norm_out_eps"),   float(AH["norm_out_eps"]))
    writer.add_float32(KV("vlln_eps"),       float(AH["vlln_eps"]))
    writer.add_float32(KV("vlsa_ln_eps"),    float(AH["vlsa_ln_eps"]))
    writer.add_float32(KV("connector_ln_eps"), 1e-6)
    writer.add_float32(KV("crop_fraction"),  float(CROP_FRACTION))
    writer.add_string(KV("embodiment_id_mapping"), embodiment_id_json.strip())
    writer.add_string(KV("statistics_json"), statistics_json)
    writer.add_string(KV("processor_config_json"), processor_json)

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
    for i in range(LM_LAYERS_USED):
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

    AHK = "action_head."
    _add(writer, "aex.vlln.weight", g(AHK + "vlln.weight")); _add(writer, "aex.vlln.bias", g(AHK + "vlln.bias"))

    for i in range(AH["vlsa_layers"]):
        SB = f"{AHK}vl_self_attention.transformer_blocks.{i}."
        _add(writer, f"aex.vlsa.{i}.norm1.weight", g(SB + "norm1.weight")); _add(writer, f"aex.vlsa.{i}.norm1.bias", g(SB + "norm1.bias"))
        _add(writer, f"aex.vlsa.{i}.norm3.weight", g(SB + "norm3.weight")); _add(writer, f"aex.vlsa.{i}.norm3.bias", g(SB + "norm3.bias"))
        for q in ("q", "k", "v"):
            _add(writer, f"aex.vlsa.{i}.attn_{q}.weight", g(SB + f"attn1.to_{q}.weight")); _add(writer, f"aex.vlsa.{i}.attn_{q}.bias", g(SB + f"attn1.to_{q}.bias"))
        _add(writer, f"aex.vlsa.{i}.attn_o.weight", g(SB + "attn1.to_out.0.weight")); _add(writer, f"aex.vlsa.{i}.attn_o.bias", g(SB + "attn1.to_out.0.bias"))
        _add(writer, f"aex.vlsa.{i}.ff0.weight", g(SB + "ff.net.0.proj.weight")); _add(writer, f"aex.vlsa.{i}.ff0.bias", g(SB + "ff.net.0.proj.bias"))
        _add(writer, f"aex.vlsa.{i}.ff2.weight", g(SB + "ff.net.2.weight")); _add(writer, f"aex.vlsa.{i}.ff2.bias", g(SB + "ff.net.2.bias"))
    for src, dst in (("state_encoder.layer1", "aex.state_enc.l1"), ("state_encoder.layer2", "aex.state_enc.l2"),
                     ("action_encoder.W1", "aex.act_enc.W1"), ("action_encoder.W2", "aex.act_enc.W2"), ("action_encoder.W3", "aex.act_enc.W3"),
                     ("action_decoder.layer1", "aex.act_dec.l1"), ("action_decoder.layer2", "aex.act_dec.l2")):
        _add(writer, f"{dst}.W", g(AHK + src + ".W")); _add(writer, f"{dst}.b", g(AHK + src + ".b"))
    _add(writer, "aex.pos_embd", g(AHK + "position_embedding.weight"))

    _add(writer, "aex.dit.time_emb.l1.weight", g(AHK + "model.timestep_encoder.timestep_embedder.linear_1.weight")); _add(writer, "aex.dit.time_emb.l1.bias", g(AHK + "model.timestep_encoder.timestep_embedder.linear_1.bias"))
    _add(writer, "aex.dit.time_emb.l2.weight", g(AHK + "model.timestep_encoder.timestep_embedder.linear_2.weight")); _add(writer, "aex.dit.time_emb.l2.bias", g(AHK + "model.timestep_encoder.timestep_embedder.linear_2.bias"))
    for i in range(AH["dit_layers"]):
        TB = f"{AHK}model.transformer_blocks.{i}."
        _add(writer, f"aex.dit.{i}.adaln.weight", g(TB + "norm1.linear.weight")); _add(writer, f"aex.dit.{i}.adaln.bias", g(TB + "norm1.linear.bias"))
        for q in ("q", "k", "v"):
            _add(writer, f"aex.dit.{i}.attn_{q}.weight", g(TB + f"attn1.to_{q}.weight")); _add(writer, f"aex.dit.{i}.attn_{q}.bias", g(TB + f"attn1.to_{q}.bias"))
        _add(writer, f"aex.dit.{i}.attn_o.weight", g(TB + "attn1.to_out.0.weight")); _add(writer, f"aex.dit.{i}.attn_o.bias", g(TB + "attn1.to_out.0.bias"))
        _add(writer, f"aex.dit.{i}.ff0.weight", g(TB + "ff.net.0.proj.weight")); _add(writer, f"aex.dit.{i}.ff0.bias", g(TB + "ff.net.0.proj.bias"))
        _add(writer, f"aex.dit.{i}.ff2.weight", g(TB + "ff.net.2.weight")); _add(writer, f"aex.dit.{i}.ff2.bias", g(TB + "ff.net.2.bias"))
    _add(writer, "aex.dit.proj_out1.weight", g(AHK + "model.proj_out_1.weight")); _add(writer, "aex.dit.proj_out1.bias", g(AHK + "model.proj_out_1.bias"))
    _add(writer, "aex.dit.proj_out2.weight", g(AHK + "model.proj_out_2.weight")); _add(writer, "aex.dit.proj_out2.bias", g(AHK + "model.proj_out_2.bias"))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB)  - combined GGUF (Qwen3-VL backbone + deepstack + vl_self_attention + AlternateVLDiT action head + cfg + sidecars)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
