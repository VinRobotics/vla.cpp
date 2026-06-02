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
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitvla_int2_pack import (
    weight_quant_to_ternary,
    pack_ladder_int2,
    pack_fused_projection,
)

ARCH = "bitvla"
KV = lambda name: f"{ARCH}.{name}"

VIT = dict(vit_hidden=1152, vit_layers=26, vit_heads=16, vit_inter=4304,
           image_size=224, patch_size=14, vit_ln_eps=1e-6,
           vit_weight_bits=1, vit_act_bits=8)
N_PATCHES = (VIT["image_size"] // VIT["patch_size"]) ** 2
VIT_HEAD_DIM = VIT["vit_hidden"] // VIT["vit_heads"]

LM = dict(lm_hidden=2560, lm_layers=30, lm_q_heads=20, lm_kv_heads=5,
          lm_head_dim=128, lm_inter=6912, lm_rope_theta=500000.0,
          lm_rms_eps=1e-5, lm_max_pos=4096,
          lm_weight_bits=1, lm_act_bits=8)

PROJ = dict(mm_in=1152, mm_out=2560)
PROPRIO = dict(in_dim=8, hid_dim=2560)
ACTION = dict(num_actions_chunk=8, action_dim=7, proprio_dim=8,
              ln_eps=1e-5)

TOKENS = dict(image_token_id=128010,
              proprio_pad_id=128011,
              action_begin_id=128012,
              stop_id=128001,
              vocab_size=128264)

PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

def weight_quant(W: torch.Tensor) -> torch.Tensor:

    dtype = W.dtype
    Wf = W.float()
    absmean = Wf.abs().double().mean().clamp_(min=1e-5).float()
    s = 1.0 / absmean
    Wq = (Wf * s).round().clamp(-1, 1) / s
    return Wq.to(dtype)

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

def _add_bf16(writer: gguf.GGUFWriter, name: str, t: torch.Tensor) -> None:

    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    _add(writer, name, t)

def _add_packed(writer: gguf.GGUFWriter, base: str, packed: np.ndarray, scales: np.ndarray) -> None:
    writer.add_tensor(base + ".weight", np.ascontiguousarray(packed, dtype=np.uint8),
                      raw_shape=[int(packed.size)], raw_dtype=gguf.GGMLQuantizationType.I8)
    writer.add_tensor(base + ".scale", np.ascontiguousarray(scales, dtype=np.float32),
                      raw_shape=[int(scales.size)], raw_dtype=gguf.GGMLQuantizationType.F32)

def _add_bit(writer: gguf.GGUFWriter, base: str, W: torch.Tensor, pack: bool,
             ffn_pad: int | None = None) -> None:

    if not pack:
        _add(writer, base + ".weight", weight_quant(W))
        return
    tern, scale = weight_quant_to_ternary(W)
    if ffn_pad is not None and tern.shape[1] < ffn_pad:
        padded = np.zeros((tern.shape[0], ffn_pad), dtype=np.int8)
        padded[:, :tern.shape[1]] = tern
        tern = padded
    _add_packed(writer, base, pack_ladder_int2(np.ascontiguousarray(tern)),
                np.array([scale], dtype=np.float32))

def _add_bit_fused(writer: gguf.GGUFWriter, base: str, Ws: list[torch.Tensor]) -> None:

    packed, scales = pack_fused_projection(Ws)
    _add_packed(writer, base, packed, scales)

def _load_sharded(ckpt: Path) -> dict[str, torch.Tensor]:

    idx = ckpt / "model.safetensors.index.json"
    weight_map = json.loads(idx.read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(by_shard):
        with safe_open(str(ckpt / shard), framework="pt") as f:
            for k in by_shard[shard]:
                out[k] = f.get_tensor(k)
    return out

def _load_pt_sidecar(path: Path) -> dict[str, torch.Tensor]:

    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v.contiguous()
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="BitVLA libero_* finetune snapshot dir")
    ap.add_argument("--out", type=Path, default=None, help="output GGUF path (default: <ckpt>/bitvla[-int2].gguf)")
    ap.add_argument("--pack-int2", action="store_true",
                    help="store BitLinear weights as int2 ladder bytes + F32 scales (CUDA-only; "
                         "~4x smaller on disk, no fp32 host load-time spike). Omit for the bf16 GGUF "
                         "that the CPU/ggml fallback path can also run.")
    args = ap.parse_args()

    pack = args.pack_int2
    ckpt = args.ckpt.resolve()
    default_name = f"{ARCH}-int2.gguf" if pack else f"{ARCH}.gguf"
    out  = (args.out or ckpt / default_name).resolve()

    ffn_pad = ((VIT["vit_inter"] + 127) // 128) * 128
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")
    cfg_json = json.loads(cfg_path.read_text())
    if str(cfg_json.get("model_type", "")) != "openvla":
        raise SystemExit(f"config.json model_type is {cfg_json.get('model_type')!r}, expected 'openvla' (BitVLA's HF model_type)")

    print(f"loading main shards from {ckpt} ...")
    W = _load_sharded(ckpt)
    print(f"  {len(W)} main tensors")

    ah_path = ckpt / "action_head--10000_checkpoint.pt"
    pp_path = ckpt / "proprio_projector--10000_checkpoint.pt"
    if not ah_path.exists() or not pp_path.exists():
        raise SystemExit(f"missing action_head/proprio_projector sidecars in {ckpt}")
    AH = _load_pt_sidecar(ah_path)
    PP = _load_pt_sidecar(pp_path)
    print(f"  +{len(AH)} action_head tensors, +{len(PP)} proprio_projector tensors")

    statistics_json = (ckpt / "dataset_statistics.json").read_text()
    processor_json  = (ckpt / "processor_config.json").read_text()      if (ckpt / "processor_config.json").exists()      else "{}"
    preproc_json    = (ckpt / "preprocessor_config.json").read_text()   if (ckpt / "preprocessor_config.json").exists()   else "{}"

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

    print(f"writing GGUF to {out} ...")
    writer = gguf.GGUFWriter(str(out), ARCH)
    writer.add_string  ("general.architecture",                    ARCH)
    writer.add_string  (KV("architecture"),                        ARCH)

    writer.add_uint32  (KV("vit.hidden"),                          VIT["vit_hidden"])
    writer.add_uint32  (KV("vit.layers"),                          VIT["vit_layers"])
    writer.add_uint32  (KV("vit.heads"),                           VIT["vit_heads"])
    writer.add_uint32  (KV("vit.head_dim"),                        VIT_HEAD_DIM)
    writer.add_uint32  (KV("vit.inter"),                           VIT["vit_inter"])
    writer.add_uint32  (KV("vit.image_size"),                      VIT["image_size"])
    writer.add_uint32  (KV("vit.patch_size"),                      VIT["patch_size"])
    writer.add_uint32  (KV("vit.n_patches"),                       N_PATCHES)
    writer.add_float32 (KV("vit.ln_eps"),                          VIT["vit_ln_eps"])

    writer.add_uint32  (KV("lm.hidden"),                           LM["lm_hidden"])
    writer.add_uint32  (KV("lm.layers"),                           LM["lm_layers"])
    writer.add_uint32  (KV("lm.q_heads"),                          LM["lm_q_heads"])
    writer.add_uint32  (KV("lm.kv_heads"),                         LM["lm_kv_heads"])
    writer.add_uint32  (KV("lm.head_dim"),                         LM["lm_head_dim"])
    writer.add_uint32  (KV("lm.inter"),                            LM["lm_inter"])
    writer.add_float32 (KV("lm.rope_theta"),                       LM["lm_rope_theta"])
    writer.add_float32 (KV("lm.rms_eps"),                          LM["lm_rms_eps"])
    writer.add_uint32  (KV("lm.max_pos"),                          LM["lm_max_pos"])
    writer.add_uint32  (KV("lm.vocab_size"),                       TOKENS["vocab_size"])

    writer.add_uint32  (KV("action.num_actions_chunk"),            ACTION["num_actions_chunk"])
    writer.add_uint32  (KV("action.action_dim"),                   ACTION["action_dim"])
    writer.add_uint32  (KV("action.proprio_dim"),                  ACTION["proprio_dim"])
    writer.add_float32 (KV("action.ln_eps"),                       ACTION["ln_eps"])

    writer.add_uint32  (KV("quant.vit_weight_bits"),               VIT["vit_weight_bits"])
    writer.add_uint32  (KV("quant.vit_act_bits"),                  VIT["vit_act_bits"])
    writer.add_uint32  (KV("quant.lm_weight_bits"),                LM["lm_weight_bits"])
    writer.add_uint32  (KV("quant.lm_act_bits"),                   LM["lm_act_bits"])
    writer.add_string  (KV("quant.method"),                        "absmean_ternary+per_token_int8")
    writer.add_string  (KV("quant.applied_at"),                    "convert")
    writer.add_uint32  (KV("quant.int2_packed"),                   1 if pack else 0)

    writer.add_uint32  (KV("tokens.image_id"),                     TOKENS["image_token_id"])
    writer.add_uint32  (KV("tokens.proprio_id"),                   TOKENS["proprio_pad_id"])
    writer.add_uint32  (KV("tokens.action_begin_id"),              TOKENS["action_begin_id"])
    writer.add_uint32  (KV("tokens.stop_id"),                      TOKENS["stop_id"])

    writer.add_string  (KV("statistics_json"),                     statistics_json)
    writer.add_string  (KV("processor_config_json"),               processor_json)
    writer.add_string  (KV("preprocessor_config_json"),            preproc_json)
    writer.add_string  (KV("prompt_template"),                     PROMPT_TEMPLATE)

    print("  writing vision tower (patch_embd + 26 layers)")
    pe_w = W["vision_tower.vision_model.embeddings.patch_embedding.weight"]

    pe_w = pe_w.reshape(VIT["vit_hidden"], -1).contiguous()
    _add(writer, "vit.patch_embd.weight",   pe_w)
    _add(writer, "vit.patch_embd.bias",     W["vision_tower.vision_model.embeddings.patch_embedding.bias"])
    _add(writer, "vit.pos_embd.weight",     W["vision_tower.vision_model.embeddings.position_embedding.weight"])
    for L in range(VIT["vit_layers"]):
        P = f"vision_tower.vision_model.encoder.layers.{L}."
        _add(writer, f"vit.blk.{L}.ln1.weight", W[P + "layer_norm1.weight"])
        _add(writer, f"vit.blk.{L}.ln1.bias",   W[P + "layer_norm1.bias"])
        _add(writer, f"vit.blk.{L}.ln2.weight", W[P + "layer_norm2.weight"])
        _add(writer, f"vit.blk.{L}.ln2.bias",   W[P + "layer_norm2.bias"])

        for src, dst in [("q_proj",   "attn_q"),
                          ("k_proj",   "attn_k"),
                          ("v_proj",   "attn_v"),
                          ("out_proj", "attn_o")]:
            _add_bit(writer, f"vit.blk.{L}.{dst}", W[P + f"self_attn.{src}.weight"], pack)
            _add(writer, f"vit.blk.{L}.{dst}.bias", W[P + f"self_attn.{src}.bias"])

        _add_bit(writer, f"vit.blk.{L}.fc1", W[P + "mlp.fc1.weight"], pack)
        _add(writer, f"vit.blk.{L}.fc1.bias", W[P + "mlp.fc1.bias"])
        _add_bit(writer, f"vit.blk.{L}.fc2", W[P + "mlp.fc2.weight"], pack, ffn_pad=ffn_pad)
        _add(writer, f"vit.blk.{L}.fc2.bias", W[P + "mlp.fc2.bias"])

    print("  writing multi-modal projector")
    _add(writer, "mm.linear_1.weight", W["multi_modal_projector.linear_1.weight"])
    _add(writer, "mm.linear_1.bias",   W["multi_modal_projector.linear_1.bias"])
    _add(writer, "mm.linear_2.weight", W["multi_modal_projector.linear_2.weight"])
    _add(writer, "mm.linear_2.bias",   W["multi_modal_projector.linear_2.bias"])

    print("  writing proprio projector")
    _add(writer, "aex.proprio.fc1.weight", PP["fc1.weight"])
    _add(writer, "aex.proprio.fc1.bias",   PP["fc1.bias"])
    _add(writer, "aex.proprio.fc2.weight", PP["fc2.weight"])
    _add(writer, "aex.proprio.fc2.bias",   PP["fc2.bias"])

    print("  writing LM (embed_tokens + 30 layers + output_norm)")
    _add(writer, "token_embd.weight", W["language_model.model.embed_tokens.weight"])
    _add(writer, "lm.output_norm.weight", W["language_model.model.norm.weight"])
    for L in range(LM["lm_layers"]):
        P = f"language_model.model.layers.{L}."
        _add(writer, f"lm.blk.{L}.attn_norm.weight",     W[P + "input_layernorm.weight"])
        _add_bit(writer, f"lm.blk.{L}.attn_q", W[P + "self_attn.q_proj.weight"], pack)
        _add_bit(writer, f"lm.blk.{L}.attn_k", W[P + "self_attn.k_proj.weight"], pack)
        _add_bit(writer, f"lm.blk.{L}.attn_v", W[P + "self_attn.v_proj.weight"], pack)
        _add(writer, f"lm.blk.{L}.attn_sub_norm.weight", W[P + "self_attn.attn_sub_norm.weight"])
        _add_bit(writer, f"lm.blk.{L}.attn_o", W[P + "self_attn.o_proj.weight"], pack)
        _add(writer, f"lm.blk.{L}.ffn_norm.weight",      W[P + "post_attention_layernorm.weight"])
        if pack:

            _add_bit_fused(writer, f"lm.blk.{L}.ffn_gate_up",
                           [W[P + "mlp.gate_proj.weight"], W[P + "mlp.up_proj.weight"]])
        else:
            _add(writer, f"lm.blk.{L}.ffn_gate.weight",  weight_quant(W[P + "mlp.gate_proj.weight"]))
            _add(writer, f"lm.blk.{L}.ffn_up.weight",    weight_quant(W[P + "mlp.up_proj.weight"]))
        _add(writer, f"lm.blk.{L}.ffn_sub_norm.weight",  W[P + "mlp.ffn_sub_norm.weight"])
        _add_bit(writer, f"lm.blk.{L}.ffn_down", W[P + "mlp.down_proj.weight"], pack)

    print("  writing action head")
    _add(writer, "aex.head.ln1.weight",        AH["model.layer_norm1.weight"])
    _add(writer, "aex.head.ln1.bias",          AH["model.layer_norm1.bias"])
    _add(writer, "aex.head.fc1.weight",        AH["model.fc1.weight"])
    _add(writer, "aex.head.fc1.bias",          AH["model.fc1.bias"])
    for b in (0, 1):
        _add(writer, f"aex.head.blk.{b}.ln.weight", AH[f"model.mlp_resnet_blocks.{b}.ffn.0.weight"])
        _add(writer, f"aex.head.blk.{b}.ln.bias",   AH[f"model.mlp_resnet_blocks.{b}.ffn.0.bias"])
        _add(writer, f"aex.head.blk.{b}.fc.weight", AH[f"model.mlp_resnet_blocks.{b}.ffn.1.weight"])
        _add(writer, f"aex.head.blk.{b}.fc.bias",   AH[f"model.mlp_resnet_blocks.{b}.ffn.1.bias"])
    _add(writer, "aex.head.ln2.weight",        AH["model.layer_norm2.weight"])
    _add(writer, "aex.head.ln2.bias",          AH["model.layer_norm2.bias"])
    _add(writer, "aex.head.fc2.weight",        AH["model.fc2.weight"])
    _add(writer, "aex.head.fc2.bias",          AH["model.fc2.bias"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done → {out}  ({out.stat().st_size / 2**30:.2f} GiB, "
          f"{'int2-packed (CUDA-only)' if pack else 'bf16'})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
