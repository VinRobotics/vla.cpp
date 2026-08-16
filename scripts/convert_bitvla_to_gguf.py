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

import re
from pathlib import Path

import numpy as np
import torch

import gguf
from gguf_common import (
    add,
    arg_parser,
    finish,
    kv_f32,
    kv_prefix,
    kv_u32,
    load_pt_module,
    load_safetensors,
    open_writer,
    read_json,
    read_text,
    require,
    resolve_out
)

ARCH = "bitvla"
KV = kv_prefix(ARCH)

VIT = dict(
    vit_hidden=1152,
    vit_layers=26,
    vit_heads=16,
    vit_inter=4304,
    image_size=224,
    patch_size=14,
    vit_ln_eps=1e-6,
    vit_weight_bits=1,
    vit_act_bits=8
)
N_PATCHES = (VIT["image_size"] // VIT["patch_size"]) ** 2
VIT_HEAD_DIM = VIT["vit_hidden"] // VIT["vit_heads"]

LM = dict(
    lm_hidden=2560,
    lm_layers=30,
    lm_q_heads=20,
    lm_kv_heads=5,
    lm_head_dim=128,
    lm_inter=6912,
    lm_rope_theta=500000.0,
    lm_rms_eps=1e-5,
    lm_max_pos=4096,
    lm_weight_bits=1,
    lm_act_bits=8
)

PROJ = dict(mm_in=1152, mm_out=2560)
ACTION = dict(num_actions_chunk=8, action_dim=7, proprio_dim=8, ln_eps=1e-5)

TOKENS = dict(
    image_token_id=128010,
    proprio_pad_id=128011,
    action_begin_id=128012,
    stop_id=128001,
    vocab_size=128264
)

PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

# Ladder int2 layout: the CUDA kernel reads 16 ternary values per 4-byte slot,
# addressed by (n_block, k_iter, wmma tile, half, sub-k). Mirrored by
# src/models/bitvla.cpp.
N_BLOCK_SIZE    = 16
K_BLOCK_SIZE    = 8
K_PER_LOOP      = 16
WMMA_K          = 32
K_PER_ITER      = K_PER_LOOP * K_BLOCK_SIZE
BYTES_PER_KITER = 512

def weight_quant_to_ternary(W: torch.Tensor):

    Wf = W.float()
    absmean = Wf.abs().double().mean().clamp_(min=1e-5).float()
    s = 1.0 / absmean
    Wq = (Wf * s).round().clamp(-1, 1)
    return Wq.to(torch.int8).cpu().numpy(), float(absmean.item())

def pack_ladder_int2(W_ternary: np.ndarray) -> np.ndarray:

    assert W_ternary.dtype == np.int8, W_ternary.dtype
    N, K = W_ternary.shape
    assert N % N_BLOCK_SIZE == 0, f"N={N} not divisible by {N_BLOCK_SIZE}"
    assert K % K_PER_ITER == 0,    f"K={K} not divisible by {K_PER_ITER}"
    assert np.all((W_ternary >= -1) & (W_ternary <= 1)), "values outside {-1,0,1}"

    W_enc = (W_ternary + 2).astype(np.uint8)
    assert W_enc.shape == (N, K)

    n_slots = N * K // 16
    slot_addr = np.arange(n_slots, dtype=np.int64)

    slots_per_block = (N_BLOCK_SIZE * K) // 16
    n_block         = slot_addr // slots_per_block
    in_block        = slot_addr  % slots_per_block

    slots_per_k0    = BYTES_PER_KITER // 4
    k_0             = in_block // slots_per_k0
    in_k0           = in_block  % slots_per_k0

    slots_per_major = 128 // 4
    major_k         = in_k0 // slots_per_major
    in_major        = in_k0  % slots_per_major

    slots_per_yhalf = 64 // 4
    y_half          = in_major // slots_per_yhalf
    in_yhalf        = in_major  % slots_per_yhalf

    slots_per_subk  = 32 // 4
    sub_k           = in_yhalf // slots_per_subk
    y_in_half       = in_yhalf  % slots_per_subk

    n_global    = n_block * N_BLOCK_SIZE + y_half * 8 + y_in_half
    k_sub_start = k_0 * K_PER_ITER + major_k * WMMA_K + sub_k * K_PER_LOOP

    k_idx = k_sub_start[:, None] + np.arange(16, dtype=np.int64)[None, :]
    n_idx = np.broadcast_to(n_global[:, None], k_idx.shape)
    enc_slots = W_enc[n_idx, k_idx]

    out = np.zeros((n_slots, 4), dtype=np.uint8)
    for byte_i in range(4):
        b = np.zeros(n_slots, dtype=np.uint8)
        for j in range(4):
            t_idx = byte_i + 4 * j
            b |= (enc_slots[:, t_idx] & 0x3) << (2 * j)
        out[:, byte_i] = b

    return out.reshape(-1)

def pack_fused_projection(weights: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:

    ternaries = []
    scales = []
    for W in weights:
        Wq, sc = weight_quant_to_ternary(W)
        ternaries.append(Wq)
        scales.append(sc)
    Wt = np.concatenate(ternaries, axis=0).astype(np.int8)
    return pack_ladder_int2(Wt), np.array(scales, dtype=np.float32)

def _add_packed(writer, base: str, packed: np.ndarray, scales: np.ndarray) -> None:

    writer.add_tensor(
        base + ".weight",
        np.ascontiguousarray(packed, dtype=np.uint8),
        raw_shape=[int(packed.size)],
        raw_dtype=gguf.GGMLQuantizationType.I8
    )
    writer.add_tensor(
        base + ".scale",
        np.ascontiguousarray(scales, dtype=np.float32),
        raw_shape=[int(scales.size)],
        raw_dtype=gguf.GGMLQuantizationType.F32
    )

def _add_bit(writer, base: str, W: torch.Tensor, ffn_pad: int | None = None) -> None:

    tern, scale = weight_quant_to_ternary(W)
    if ffn_pad is not None and tern.shape[1] < ffn_pad:
        padded = np.zeros((tern.shape[0], ffn_pad), dtype=np.int8)
        padded[:, :tern.shape[1]] = tern
        tern = padded
    _add_packed(
        writer,
        base,
        pack_ladder_int2(np.ascontiguousarray(tern)),
        np.array([scale], dtype=np.float32)
    )

def _add_bit_fused(writer, base: str, Ws: list[torch.Tensor]) -> None:

    packed, scales = pack_fused_projection(Ws)
    _add_packed(writer, base, packed, scales)

def _find_sidecar(ckpt: Path, stem: str) -> Path | None:

    cands = sorted(
        ckpt.glob(f"{stem}--*_checkpoint.pt"),
        key=lambda p: int(m.group(1)) if (m := re.search(r"--(\d+)_checkpoint\.pt$", p.name)) else -1,
    )
    return cands[-1] if cands else None

def _add_kv(writer, statistics_json: str, processor_json: str, preproc_json: str) -> None:

    kv_u32(
        writer,
        KV,
        {
            "vit.hidden":     VIT["vit_hidden"],
            "vit.layers":     VIT["vit_layers"],
            "vit.heads":      VIT["vit_heads"],
            "vit.head_dim":   VIT_HEAD_DIM,
            "vit.inter":      VIT["vit_inter"],
            "vit.image_size": VIT["image_size"],
            "vit.patch_size": VIT["patch_size"],
            "vit.n_patches":  N_PATCHES
        }
    )
    writer.add_float32(KV("vit.ln_eps"), VIT["vit_ln_eps"])

    kv_u32(
        writer,
        KV,
        {
            "lm.hidden":   LM["lm_hidden"],
            "lm.layers":   LM["lm_layers"],
            "lm.q_heads":  LM["lm_q_heads"],
            "lm.kv_heads": LM["lm_kv_heads"],
            "lm.head_dim": LM["lm_head_dim"],
            "lm.inter":    LM["lm_inter"]
        }
    )
    kv_f32(writer, KV, {"lm.rope_theta": LM["lm_rope_theta"], "lm.rms_eps": LM["lm_rms_eps"]})
    kv_u32(writer, KV, {"lm.max_pos": LM["lm_max_pos"], "lm.vocab_size": TOKENS["vocab_size"]})

    kv_u32(
        writer,
        KV,
        {
            "action.num_actions_chunk": ACTION["num_actions_chunk"],
            "action.action_dim":        ACTION["action_dim"],
            "action.proprio_dim":       ACTION["proprio_dim"]
        }
    )
    writer.add_float32(KV("action.ln_eps"), ACTION["ln_eps"])

    kv_u32(
        writer,
        KV,
        {
            "quant.vit_weight_bits": VIT["vit_weight_bits"],
            "quant.vit_act_bits":    VIT["vit_act_bits"],
            "quant.lm_weight_bits":  LM["lm_weight_bits"],
            "quant.lm_act_bits":     LM["lm_act_bits"]
        }
    )
    writer.add_string(KV("quant.method"),     "absmean_ternary+per_token_int8")
    writer.add_string(KV("quant.applied_at"), "convert")
    writer.add_uint32(KV("quant.int2_packed"), 1)

    kv_u32(
        writer,
        KV,
        {
            "tokens.image_id":        TOKENS["image_token_id"],
            "tokens.proprio_id":      TOKENS["proprio_pad_id"],
            "tokens.action_begin_id": TOKENS["action_begin_id"],
            "tokens.stop_id":         TOKENS["stop_id"]
        }
    )

    writer.add_string(KV("statistics_json"),          statistics_json)
    writer.add_string(KV("processor_config_json"),    processor_json)
    writer.add_string(KV("preprocessor_config_json"), preproc_json)
    writer.add_string(KV("prompt_template"),          PROMPT_TEMPLATE)

def main() -> int:
    ap = arg_parser(ARCH, "BitVLA libero_* finetune snapshot dir")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    ffn_pad = ((VIT["vit_inter"] + 127) // 128) * 128

    cfg_json = read_json(ckpt / "config.json")
    if str(cfg_json.get("model_type", "")) != "openvla":
        raise SystemExit(f"config.json model_type is {cfg_json.get('model_type')!r}, expected 'openvla' (BitVLA's HF model_type)")

    print(f"loading main shards from {ckpt} ...")
    W = load_safetensors(ckpt)
    print(f"  {len(W)} main tensors")

    ah_path = _find_sidecar(ckpt, "action_head")
    pp_path = _find_sidecar(ckpt, "proprio_projector")
    if ah_path is None or pp_path is None:
        raise SystemExit(f"missing action_head/proprio_projector sidecars in {ckpt}")
    print(f"  sidecars: {ah_path.name}, {pp_path.name}")
    AH = load_pt_module(ah_path)
    PP = load_pt_module(pp_path)
    print(f"  +{len(AH)} action_head tensors, +{len(PP)} proprio_projector tensors")

    stats_path = ckpt / "dataset_statistics.json"
    require(stats_path)
    statistics_json = read_text(stats_path)
    processor_json  = read_text(ckpt / "processor_config.json")
    preproc_json    = read_text(ckpt / "preprocessor_config.json")

    assert W["language_model.model.embed_tokens.weight"].shape == (TOKENS["vocab_size"], LM["lm_hidden"])
    assert W["language_model.model.norm.weight"].shape == (LM["lm_hidden"],)
    n_vit = sum(1 for k in W if k.startswith("vision_tower.vision_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"))
    n_lm  = sum(1 for k in W if k.startswith("language_model.model.layers.") and k.endswith(".self_attn.q_proj.weight"))
    if n_vit != VIT["vit_layers"]:
        raise SystemExit(f"checkpoint has {n_vit} vision layers, expected {VIT['vit_layers']}")
    if n_lm != LM["lm_layers"]:
        raise SystemExit(f"checkpoint has {n_lm} LM layers, expected {LM['lm_layers']}")

    print(f"resolved cfg: vit={VIT['vit_hidden']}d×{VIT['vit_layers']}L×{VIT['vit_heads']}h@224  "
          f"⇒ {N_PATCHES} patches  mm={PROJ['mm_in']}→{PROJ['mm_out']}  "
          f"lm=BitNet {LM['lm_hidden']}d×{LM['lm_layers']}L ({LM['lm_q_heads']}q/{LM['lm_kv_heads']}kv×{LM['lm_head_dim']})  "
          f"chunk×dim={ACTION['num_actions_chunk']}×{ACTION['action_dim']}  vocab={TOKENS['vocab_size']}")

    writer = open_writer(out, ARCH)
    _add_kv(writer, statistics_json, processor_json, preproc_json)

    print("  writing vision tower (patch_embd + 26 layers)")
    add(writer, "vit.patch_embd.weight", W["vision_tower.vision_model.embeddings.patch_embedding.weight"].reshape(VIT["vit_hidden"], -1).contiguous())
    add(writer, "vit.patch_embd.bias",   W["vision_tower.vision_model.embeddings.patch_embedding.bias"])
    add(writer, "vit.pos_embd.weight",   W["vision_tower.vision_model.embeddings.position_embedding.weight"])
    for L in range(VIT["vit_layers"]):
        P = f"vision_tower.vision_model.encoder.layers.{L}."
        add(writer, f"vit.blk.{L}.ln1.weight", W[P + "layer_norm1.weight"])
        add(writer, f"vit.blk.{L}.ln1.bias",   W[P + "layer_norm1.bias"])
        add(writer, f"vit.blk.{L}.ln2.weight", W[P + "layer_norm2.weight"])
        add(writer, f"vit.blk.{L}.ln2.bias",   W[P + "layer_norm2.bias"])

        for src, dst in (
            ("q_proj",   "attn_q"),
            ("k_proj",   "attn_k"),
            ("v_proj",   "attn_v"),
            ("out_proj", "attn_o")
        ):
            _add_bit(writer, f"vit.blk.{L}.{dst}", W[P + f"self_attn.{src}.weight"])
            add(writer, f"vit.blk.{L}.{dst}.bias", W[P + f"self_attn.{src}.bias"])

        _add_bit(writer, f"vit.blk.{L}.fc1", W[P + "mlp.fc1.weight"])
        add(writer, f"vit.blk.{L}.fc1.bias", W[P + "mlp.fc1.bias"])
        _add_bit(writer, f"vit.blk.{L}.fc2", W[P + "mlp.fc2.weight"], ffn_pad=ffn_pad)
        add(writer, f"vit.blk.{L}.fc2.bias", W[P + "mlp.fc2.bias"])

    print("  writing multi-modal projector")
    add(writer, "mm.linear_1.weight", W["multi_modal_projector.linear_1.weight"])
    add(writer, "mm.linear_1.bias",   W["multi_modal_projector.linear_1.bias"])
    add(writer, "mm.linear_2.weight", W["multi_modal_projector.linear_2.weight"])
    add(writer, "mm.linear_2.bias",   W["multi_modal_projector.linear_2.bias"])

    print("  writing proprio projector")
    add(writer, "aex.proprio.fc1.weight", PP["fc1.weight"])
    add(writer, "aex.proprio.fc1.bias",   PP["fc1.bias"])
    add(writer, "aex.proprio.fc2.weight", PP["fc2.weight"])
    add(writer, "aex.proprio.fc2.bias",   PP["fc2.bias"])

    print("  writing LM (embed_tokens + 30 layers + output_norm)")
    add(writer, "token_embd.weight",     W["language_model.model.embed_tokens.weight"])
    add(writer, "lm.output_norm.weight", W["language_model.model.norm.weight"])
    for L in range(LM["lm_layers"]):
        P = f"language_model.model.layers.{L}."
        add(writer, f"lm.blk.{L}.attn_norm.weight", W[P + "input_layernorm.weight"])
        _add_bit(writer, f"lm.blk.{L}.attn_q", W[P + "self_attn.q_proj.weight"])
        _add_bit(writer, f"lm.blk.{L}.attn_k", W[P + "self_attn.k_proj.weight"])
        _add_bit(writer, f"lm.blk.{L}.attn_v", W[P + "self_attn.v_proj.weight"])
        add(writer, f"lm.blk.{L}.attn_sub_norm.weight", W[P + "self_attn.attn_sub_norm.weight"])
        _add_bit(writer, f"lm.blk.{L}.attn_o", W[P + "self_attn.o_proj.weight"])
        add(writer, f"lm.blk.{L}.ffn_norm.weight", W[P + "post_attention_layernorm.weight"])
        _add_bit_fused(
            writer,
            f"lm.blk.{L}.ffn_gate_up",
            [W[P + "mlp.gate_proj.weight"], W[P + "mlp.up_proj.weight"]]
        )
        add(writer, f"lm.blk.{L}.ffn_sub_norm.weight", W[P + "mlp.ffn_sub_norm.weight"])
        _add_bit(writer, f"lm.blk.{L}.ffn_down", W[P + "mlp.down_proj.weight"])

    print("  writing action head")
    add(writer, "aex.head.ln1.weight", AH["model.layer_norm1.weight"])
    add(writer, "aex.head.ln1.bias",   AH["model.layer_norm1.bias"])
    add(writer, "aex.head.fc1.weight", AH["model.fc1.weight"])
    add(writer, "aex.head.fc1.bias",   AH["model.fc1.bias"])
    for b in (0, 1):
        add(writer, f"aex.head.blk.{b}.ln.weight", AH[f"model.mlp_resnet_blocks.{b}.ffn.0.weight"])
        add(writer, f"aex.head.blk.{b}.ln.bias",   AH[f"model.mlp_resnet_blocks.{b}.ffn.0.bias"])
        add(writer, f"aex.head.blk.{b}.fc.weight", AH[f"model.mlp_resnet_blocks.{b}.ffn.1.weight"])
        add(writer, f"aex.head.blk.{b}.fc.bias",   AH[f"model.mlp_resnet_blocks.{b}.ffn.1.bias"])
    add(writer, "aex.head.ln2.weight", AH["model.layer_norm2.weight"])
    add(writer, "aex.head.ln2.bias",   AH["model.layer_norm2.bias"])
    add(writer, "aex.head.fc2.weight", AH["model.fc2.weight"])
    add(writer, "aex.head.fc2.bias",   AH["model.fc2.bias"])

    return finish(writer, out, "  - int2-packed BitLinear weights + F32 absmean scales (CUDA-only)")

if __name__ == "__main__":
    raise SystemExit(main())
