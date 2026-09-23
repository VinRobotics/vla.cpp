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

"""Convert TurboVLA (LeRobot format) checkpoint to GGUF.

Supports any TurboVLA variant with auto-detection of dimensions from checkpoint.
Architecture must be compatible with the LeRobot TurboVLA format.

Usage:
    python convert_turbovla_to_gguf.py /path/to/checkpoint [--out output.gguf] [--verify]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

F32 = gguf.GGMLQuantizationType.F32
BF16 = gguf.GGMLQuantizationType.BF16

ARCH = "turbovla"


# =============================================================================
# TENSOR KEY PREFIXES
# =============================================================================

# Prefixes for official TurboVLA checkpoint (no "model." prefix)
PREFIX_VIT = "vision_encoder.backbone"
PREFIX_TEXT = "text_encoder.bert"
PREFIX_VIT_PROJ = "vision_projection"
PREFIX_FUSION = "vision_language_interaction.fusion_layers"
PREFIX_VL_TEXT = "vision_language_interaction.text_layers"
PREFIX_ACT_DEC = "action_head.decoder"
PREFIX_STATE_PROJ = "action_head.state_projection"
KEY_VIEW_EMB = "view_embedding"
KEY_TEXT_PROJ = "text_encoder.text_projection"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def kv_prefix(name: str) -> str:
    return f"{ARCH}.{name}"


def bf16_u16(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.uint16).cpu().numpy()


def add_tensor(writer: gguf.GGUFWriter, name: str, t: torch.Tensor) -> None:
    """Add tensor with preserved dtype."""
    if t.dtype == torch.float32:
        writer.add_tensor(name, t.contiguous().cpu().numpy(), raw_dtype=F32)
    elif t.dtype == torch.bfloat16:
        writer.add_tensor(name, bf16_u16(t), raw_shape=list(t.shape), raw_dtype=BF16)
    elif t.dtype == torch.float16:
        writer.add_tensor(name, t.contiguous().cpu().numpy().astype(np.float32), raw_dtype=F32)
    else:
        raise NotImplementedError(f"unsupported dtype {t.dtype} for {name}")


def max_layer(keys: set[str], pfx: str) -> int:
    """Count number of layers with given prefix."""
    m = -1
    for k in keys:
        if k.startswith(pfx):
            try:
                m = max(m, int(k[len(pfx):].split(".", 1)[0]))
            except ValueError:
                pass
    return m + 1


def load_safetensors(ckpt: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from safetensors file."""
    tensors = {}
    safetensors_path = ckpt / "model.safetensors"
    if not safetensors_path.exists():
        raise SystemExit(f"model.safetensors not found in {ckpt}")

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        for key in f:
            tensors[key] = f.get_tensor(key)
    return tensors


# =============================================================================
# DIMENSION INFERENCE
# =============================================================================

class TurboVLADims:
    """Auto-detect all TurboVLA dimensions from checkpoint tensor shapes."""

    def __init__(self, tensors: dict[str, torch.Tensor], keys: set[str], cfg_json: dict | None = None,
                 dinov3_cfg: dict | None = None):
        self.tensors = tensors
        self.keys = keys
        self.cfg = cfg_json or {}
        self.dinov3_cfg = dinov3_cfg or {}

        # Vision encoder (DINOv3 ViT)
        q0 = self._get(f"{PREFIX_VIT}.layer.0.attention.q_proj.weight")
        self.vit_dim = int(q0.shape[0])
        self.vit_layers = max_layer(keys, f"{PREFIX_VIT}.layer.")

        # Text encoder (BERT)
        q0 = self._get(f"{PREFIX_TEXT}.encoder.layer.0.attention.self.query.weight")
        self.text_dim = int(q0.shape[0])
        self.text_layers = max_layer(keys, f"{PREFIX_TEXT}.encoder.layer.")

        # VL Fusion layers
        self.num_fusion_layers = max_layer(keys, f"{PREFIX_FUSION}.")
        self.num_text_layers = max_layer(keys, f"{PREFIX_VL_TEXT}.")

        # BiMultiHeadAttention projects to interaction.enhancer_inner_dim, whereas
        # the text enhancer and ACT decoder operate at interaction.hidden_dim.
        interaction_cfg = self.cfg.get("interaction", {})
        self.hidden_dim = int(interaction_cfg.get("hidden_dim", 256))
        raw_nheads = int(interaction_cfg.get("nheads", 8))
        self.fusion_heads = max(1, raw_nheads // 2)
        self.fusion_dim = int(interaction_cfg.get("enhancer_inner_dim", 0))
        if self.fusion_dim <= 0 or self.fusion_dim % self.fusion_heads:
            raise SystemExit("interaction.enhancer_inner_dim must be divisible by fusion_heads")
        self.fusion_head_dim = self.fusion_dim // self.fusion_heads

        # Text enhancer heads: same split as fusion
        self.text_enhancer_heads = max(1, raw_nheads // 2)  # = 4
        self.text_enhancer_head_dim = self.hidden_dim // self.text_enhancer_heads  # = 64

        # TurboVLA constructs the ACT decoder with interaction.nheads.
        self.action_heads = raw_nheads
        self.action_head_dim = self.hidden_dim // self.action_heads  # = 32

        # State projection config
        self.num_state_tokens = int(self.cfg.get("action", {}).get("num_state_tokens", 2))

        # Action decoder
        self.num_action_decoder_layers = max_layer(keys, f"{PREFIX_ACT_DEC}.decoder.layers.")
        action_q = self._get(f"{PREFIX_ACT_DEC}.action_queries.weight")
        self.action_horizon = int(action_q.shape[0])
        act_proj_2 = self._get(f"{PREFIX_ACT_DEC}.action_projection.layers.2.weight")
        self.action_dim = int(act_proj_2.shape[0])

        # State projection
        state_weight = self._get(f"{PREFIX_STATE_PROJ}.net.1.weight")
        self.state_dim = int(state_weight.shape[1])

        # Image specs - infer from config or tensor shapes
        view_emb = self._get(KEY_VIEW_EMB)
        if view_emb.ndim == 3:
            if view_emb.shape[0] != 1:
                raise SystemExit(f"Unexpected view_embedding shape: {tuple(view_emb.shape)}")
            self.num_views = int(view_emb.shape[1])
        elif view_emb.ndim == 2:
            self.num_views = int(view_emb.shape[0])
        else:
            raise SystemExit(f"Unexpected view_embedding shape: {tuple(view_emb.shape)}")

        # image_size: config first, then infer
        cfg_image_size = self.cfg.get("vision", {}).get("image_size", self.cfg.get("image_size"))
        if cfg_image_size is not None:
            if isinstance(cfg_image_size, (list, tuple)):
                if not cfg_image_size:
                    raise SystemExit("empty image_size in config")
                self.image_size = int(cfg_image_size[0])
            else:
                self.image_size = int(cfg_image_size)
        else:
            inferred = self._infer_image_size()
            if inferred is None:
                raise SystemExit(
                    "Cannot determine image_size. "
                    "Checkpoint does not encode enough information "
                    "and config.json has no image_size."
                )
            self.image_size = inferred

        # Patch size from vision encoder
        patch_weight = self._get(f"{PREFIX_VIT}.embeddings.patch_embeddings.weight")
        if len(patch_weight.shape) == 4:
            self.patch_size = patch_weight.shape[2]
        else:
            raise SystemExit("Cannot determine patch_size from patch_embeddings")

        # Register tokens (optional)
        reg_key = f"{PREFIX_VIT}.embeddings.register_tokens"
        if reg_key in tensors:
            reg = tensors[reg_key]
            self.num_register_tokens = int(reg.shape[1]) if reg.ndim == 3 else 0
        else:
            self.num_register_tokens = 0

        # DINOv3 RoPE is architectural data, not a shape-derived default.  The
        # checkpoint's TurboVLA config names the backbone, while the matching
        # DINO config supplies rope_theta and coordinate normalization.
        if "rope_theta" not in self.dinov3_cfg:
            raise SystemExit("DINOv3 config with rope_theta is required; pass --dinov3-config")
        self.rope_theta = float(self.dinov3_cfg["rope_theta"])
        self.rope_normalize_coords = str(self.dinov3_cfg.get("rope_normalize_coords", "separate"))
        if self.rope_normalize_coords != "separate":
            raise SystemExit(
                f"unsupported DINOv3 rope coordinate normalization: {self.rope_normalize_coords!r}"
            )
        if int(self.dinov3_cfg.get("num_register_tokens", self.num_register_tokens)) != self.num_register_tokens:
            raise SystemExit("DINOv3 config num_register_tokens disagrees with checkpoint weights")

        # Infer vocab_size from word embeddings
        word_emb = self._get(f"{PREFIX_TEXT}.embeddings.word_embeddings.weight")
        self.vocab_size = int(word_emb.shape[0])

        # dropout: only from config, no fallback
        self.dropout = float(self.cfg.get("vision", {}).get("dropout", 0.0))

    def _get(self, key: str) -> torch.Tensor:
        if key not in self.tensors:
            raise KeyError(f"Required tensor not found: {key}")
        return self.tensors[key]

    def _infer_image_size(self) -> int | None:
        """Infer image size from tensor shapes if config not available.

        Note: DINOv3 uses RoPE (rotary position embeddings) instead of learned
        position embeddings, so position_embeddings.weight may not exist or may
        have different semantics. Use patch_size * grid_size from config instead.
        """
        # Try the validated DINOv3 config first.
        if self.dinov3_cfg:
            image_size = self.dinov3_cfg.get("image_size")
            if image_size:
                return int(image_size)

        # Fallback: try to infer from position embeddings
        # DINOv2 with position embeddings
        pos_key = f"{PREFIX_VIT}.embeddings.position_embeddings.weight"
        if pos_key in self.tensors:
            pos = self.tensors[pos_key]
            # Position embeddings format: [seq_len, dim] or [1, seq_len, dim]
            seq_len = pos.shape[0] if pos.ndim == 2 else pos.shape[1]
            num_patches = seq_len - 1  # exclude CLS token
            patch = self.tensors.get(f"{PREFIX_VIT}.embeddings.patch_embeddings.weight")
            if patch is not None and len(patch.shape) == 4:
                patch_size = patch.shape[2]
                grid_size = int(num_patches ** 0.5)
                if grid_size * grid_size == num_patches:
                    return grid_size * patch_size

        return None

    def __repr__(self):
        return (
            f"TurboVLADims(vit={self.vit_dim}d×{self.vit_layers}L, "
            f"text={self.text_dim}d×{self.text_layers}L, "
            f"hidden={self.hidden_dim}, "
            f"fusion={self.num_fusion_layers}L, "
            f"decoder={self.num_action_decoder_layers}L, "
            f"action={self.action_dim}d×{self.action_horizon}horizon, "
            f"state={self.state_dim}d, views={self.num_views})"
        )


# =============================================================================
# TENSOR WRITERS
# =============================================================================

def write_vision_encoder(writer: gguf.GGUFWriter, tensors: dict, dims: TurboVLADims) -> None:
    """Write DINOv3 ViT vision encoder.

    Tensor naming follows TurboVLA naming convention with attn_q/k/v/o.
    LayerScale is baked into weights.
    """
    root = PREFIX_VIT

    # embeddings
    # PyTorch patch_embedding weight: [H, IC, KH, KW] = [768, 3, 16, 16]
    # For matmul-based patch embedding: flatten weight to [H, IC*KH*KW] = [768, 768]
    patch_weight = tensors[f"{root}.embeddings.patch_embeddings.weight"]
    patch_weight = patch_weight.reshape(dims.vit_dim, -1)  # [768, 768]
    add_tensor(writer, "vit.cls_token", tensors[f"{root}.embeddings.cls_token"].squeeze(0))
    add_tensor(writer, "vit.patch_embed.weight", patch_weight)
    add_tensor(writer, "vit.patch_embed.bias", tensors[f"{root}.embeddings.patch_embeddings.bias"])

    # register tokens (optional - DINOv3 uses them)
    reg_key = f"{root}.embeddings.register_tokens"
    if reg_key in tensors:
        add_tensor(writer, "vit.register_tokens", tensors[reg_key].squeeze(0))

    # transformer layers
    for i in range(dims.vit_layers):
        lr = f"{root}.layer.{i}"

        # Get LayerScale values for baking into weights
        ls1 = tensors[f"{lr}.layer_scale1.lambda1"].clone()
        ls2 = tensors[f"{lr}.layer_scale2.lambda1"].clone()

        # attention projections - bake LayerScale into o_proj
        w_q = tensors[f"{lr}.attention.q_proj.weight"]
        b_q = tensors[f"{lr}.attention.q_proj.bias"]
        w_k = tensors[f"{lr}.attention.k_proj.weight"]
        k_bias_key = f"{lr}.attention.k_proj.bias"
        b_k = tensors[k_bias_key] if k_bias_key in tensors else torch.zeros(dims.vit_dim, dtype=torch.float32)
        w_v = tensors[f"{lr}.attention.v_proj.weight"]
        b_v = tensors[f"{lr}.attention.v_proj.bias"]
        # Bake ls1 into o_proj
        w_o = tensors[f"{lr}.attention.o_proj.weight"] * ls1.view(-1, 1)
        b_o = tensors[f"{lr}.attention.o_proj.bias"] * ls1

        add_tensor(writer, f"vit.blk.{i}.attn_q.weight", w_q)
        add_tensor(writer, f"vit.blk.{i}.attn_q.bias", b_q)
        add_tensor(writer, f"vit.blk.{i}.attn_k.weight", w_k)
        add_tensor(writer, f"vit.blk.{i}.attn_k.bias", b_k)
        add_tensor(writer, f"vit.blk.{i}.attn_v.weight", w_v)
        add_tensor(writer, f"vit.blk.{i}.attn_v.bias", b_v)
        add_tensor(writer, f"vit.blk.{i}.attn_o.weight", w_o)
        add_tensor(writer, f"vit.blk.{i}.attn_o.bias", b_o)

        # layer norms
        add_tensor(writer, f"vit.blk.{i}.ln1.weight", tensors[f"{lr}.norm1.weight"])
        add_tensor(writer, f"vit.blk.{i}.ln1.bias", tensors[f"{lr}.norm1.bias"])
        add_tensor(writer, f"vit.blk.{i}.ln2.weight", tensors[f"{lr}.norm2.weight"])
        add_tensor(writer, f"vit.blk.{i}.ln2.bias", tensors[f"{lr}.norm2.bias"])

        # MLP - bake ls2 into down_proj
        w_fc1 = tensors[f"{lr}.mlp.up_proj.weight"]
        b_fc1 = tensors[f"{lr}.mlp.up_proj.bias"]
        w_fc2 = tensors[f"{lr}.mlp.down_proj.weight"] * ls2.view(-1, 1)
        b_fc2 = tensors[f"{lr}.mlp.down_proj.bias"] * ls2

        add_tensor(writer, f"vit.blk.{i}.fc1.weight", w_fc1)
        add_tensor(writer, f"vit.blk.{i}.fc1.bias", b_fc1)
        add_tensor(writer, f"vit.blk.{i}.fc2.weight", w_fc2)
        add_tensor(writer, f"vit.blk.{i}.fc2.bias", b_fc2)

    # final norm
    add_tensor(writer, "vit.final_norm.weight", tensors[f"{root}.norm.weight"])
    add_tensor(writer, "vit.final_norm.bias", tensors[f"{root}.norm.bias"])


def write_text_encoder(writer: gguf.GGUFWriter, tensors: dict, dims: TurboVLADims) -> None:
    """Write BERT text encoder."""
    root = PREFIX_TEXT

    # embeddings
    add_tensor(writer, "text.embed.word_embeddings", tensors[f"{root}.embeddings.word_embeddings.weight"])
    add_tensor(writer, "text.embed.position_embeddings", tensors[f"{root}.embeddings.position_embeddings.weight"])
    add_tensor(writer, "text.embed.token_type_embeddings", tensors[f"{root}.embeddings.token_type_embeddings.weight"])
    add_tensor(writer, "text.embed.LayerNorm.weight", tensors[f"{root}.embeddings.LayerNorm.weight"])
    add_tensor(writer, "text.embed.LayerNorm.bias", tensors[f"{root}.embeddings.LayerNorm.bias"])

    # encoder layers
    for i in range(dims.text_layers):
        lr = f"{root}.encoder.layer.{i}"

        add_tensor(writer, f"text.encoder.layer.{i}.attention.self.query.weight", tensors[f"{lr}.attention.self.query.weight"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.self.query.bias", tensors[f"{lr}.attention.self.query.bias"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.self.key.weight", tensors[f"{lr}.attention.self.key.weight"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.self.key.bias", tensors[f"{lr}.attention.self.key.bias"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.self.value.weight", tensors[f"{lr}.attention.self.value.weight"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.self.value.bias", tensors[f"{lr}.attention.self.value.bias"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.output.dense.weight", tensors[f"{lr}.attention.output.dense.weight"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.output.dense.bias", tensors[f"{lr}.attention.output.dense.bias"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.output.LayerNorm.weight", tensors[f"{lr}.attention.output.LayerNorm.weight"])
        add_tensor(writer, f"text.encoder.layer.{i}.attention.output.LayerNorm.bias", tensors[f"{lr}.attention.output.LayerNorm.bias"])
        add_tensor(writer, f"text.encoder.layer.{i}.intermediate.dense.weight", tensors[f"{lr}.intermediate.dense.weight"])
        add_tensor(writer, f"text.encoder.layer.{i}.intermediate.dense.bias", tensors[f"{lr}.intermediate.dense.bias"])
        add_tensor(writer, f"text.encoder.layer.{i}.output.dense.weight", tensors[f"{lr}.output.dense.weight"])
        add_tensor(writer, f"text.encoder.layer.{i}.output.dense.bias", tensors[f"{lr}.output.dense.bias"])
        add_tensor(writer, f"text.encoder.layer.{i}.output.LayerNorm.weight", tensors[f"{lr}.output.LayerNorm.weight"])
        add_tensor(writer, f"text.encoder.layer.{i}.output.LayerNorm.bias", tensors[f"{lr}.output.LayerNorm.bias"])

    # pooler
    add_tensor(writer, "text.pooler.dense.weight", tensors[f"{root}.pooler.dense.weight"])
    add_tensor(writer, "text.pooler.dense.bias", tensors[f"{root}.pooler.dense.bias"])

    # text projection
    add_tensor(writer, "text_proj.weight", tensors[KEY_TEXT_PROJ + ".weight"])
    add_tensor(writer, "text_proj.bias", tensors[KEY_TEXT_PROJ + ".bias"])


def write_vision_projection(writer: gguf.GGUFWriter, tensors: dict) -> None:
    """Write vision projection MLP."""
    root = PREFIX_VIT_PROJ

    add_tensor(writer, "vit_proj.input_norm.weight", tensors[f"{root}.input_norm.weight"])
    add_tensor(writer, "vit_proj.input_norm.bias", tensors[f"{root}.input_norm.bias"])
    add_tensor(writer, "vit_proj.mlp.0.weight", tensors[f"{root}.mlp.0.weight"])
    add_tensor(writer, "vit_proj.mlp.0.bias", tensors[f"{root}.mlp.0.bias"])
    add_tensor(writer, "vit_proj.mlp.3.weight", tensors[f"{root}.mlp.3.weight"])
    add_tensor(writer, "vit_proj.mlp.3.bias", tensors[f"{root}.mlp.3.bias"])
    add_tensor(writer, "vit_proj.skip.weight", tensors[f"{root}.skip.weight"])
    add_tensor(writer, "vit_proj.output_norm.weight", tensors[f"{root}.output_norm.weight"])
    add_tensor(writer, "vit_proj.output_norm.bias", tensors[f"{root}.output_norm.bias"])


def write_vision_language_interaction(writer: gguf.GGUFWriter, tensors: dict, dims: TurboVLADims) -> None:
    """Write VL fusion and text self-attention layers."""
    fusion_root = PREFIX_FUSION
    text_root = PREFIX_VL_TEXT

    # VL Fusion layers
    for i in range(dims.num_fusion_layers):
        lr = f"{fusion_root}.{i}"

        add_tensor(writer, f"vl_fusion.{i}.v_proj.weight", tensors[f"{lr}.attn.v_proj.weight"])
        add_tensor(writer, f"vl_fusion.{i}.v_proj.bias", tensors[f"{lr}.attn.v_proj.bias"])
        add_tensor(writer, f"vl_fusion.{i}.l_proj.weight", tensors[f"{lr}.attn.l_proj.weight"])
        add_tensor(writer, f"vl_fusion.{i}.l_proj.bias", tensors[f"{lr}.attn.l_proj.bias"])
        add_tensor(writer, f"vl_fusion.{i}.values_v.weight", tensors[f"{lr}.attn.values_v_proj.weight"])
        add_tensor(writer, f"vl_fusion.{i}.values_v.bias", tensors[f"{lr}.attn.values_v_proj.bias"])
        add_tensor(writer, f"vl_fusion.{i}.values_l.weight", tensors[f"{lr}.attn.values_l_proj.weight"])
        add_tensor(writer, f"vl_fusion.{i}.values_l.bias", tensors[f"{lr}.attn.values_l_proj.bias"])
        add_tensor(writer, f"vl_fusion.{i}.out_v.weight", tensors[f"{lr}.attn.out_v_proj.weight"])
        add_tensor(writer, f"vl_fusion.{i}.out_v.bias", tensors[f"{lr}.attn.out_v_proj.bias"])
        add_tensor(writer, f"vl_fusion.{i}.out_l.weight", tensors[f"{lr}.attn.out_l_proj.weight"])
        add_tensor(writer, f"vl_fusion.{i}.out_l.bias", tensors[f"{lr}.attn.out_l_proj.bias"])

        add_tensor(writer, f"vl_fusion.{i}.norm_v.weight", tensors[f"{lr}.layer_norm_v.weight"])
        add_tensor(writer, f"vl_fusion.{i}.norm_v.bias", tensors[f"{lr}.layer_norm_v.bias"])
        add_tensor(writer, f"vl_fusion.{i}.norm_l.weight", tensors[f"{lr}.layer_norm_l.weight"])
        add_tensor(writer, f"vl_fusion.{i}.norm_l.bias", tensors[f"{lr}.layer_norm_l.bias"])

        add_tensor(writer, f"vl_fusion.{i}.gamma_v", tensors[f"{lr}.gamma_v"])
        add_tensor(writer, f"vl_fusion.{i}.gamma_l", tensors[f"{lr}.gamma_l"])

    # VL Text self-attention layers
    for i in range(dims.num_text_layers):
        lr = f"{text_root}.{i}"

        add_tensor(writer, f"vl_text.{i}.attn_qkv.weight", tensors[f"{lr}.self_attn.in_proj_weight"])
        add_tensor(writer, f"vl_text.{i}.attn_qkv.bias", tensors[f"{lr}.self_attn.in_proj_bias"])
        add_tensor(writer, f"vl_text.{i}.attn_o.weight", tensors[f"{lr}.self_attn.out_proj.weight"])
        add_tensor(writer, f"vl_text.{i}.attn_o.bias", tensors[f"{lr}.self_attn.out_proj.bias"])

        add_tensor(writer, f"vl_text.{i}.ln1.weight", tensors[f"{lr}.norm1.weight"])
        add_tensor(writer, f"vl_text.{i}.ln1.bias", tensors[f"{lr}.norm1.bias"])

        add_tensor(writer, f"vl_text.{i}.fc1.weight", tensors[f"{lr}.linear1.weight"])
        add_tensor(writer, f"vl_text.{i}.fc1.bias", tensors[f"{lr}.linear1.bias"])
        add_tensor(writer, f"vl_text.{i}.fc2.weight", tensors[f"{lr}.linear2.weight"])
        add_tensor(writer, f"vl_text.{i}.fc2.bias", tensors[f"{lr}.linear2.bias"])
        add_tensor(writer, f"vl_text.{i}.ln2.weight", tensors[f"{lr}.norm2.weight"])
        add_tensor(writer, f"vl_text.{i}.ln2.bias", tensors[f"{lr}.norm2.bias"])


def write_action_decoder(writer: gguf.GGUFWriter, tensors: dict, dims: TurboVLADims) -> None:
    """Write ACT-style action decoder."""
    root = PREFIX_ACT_DEC

    add_tensor(writer, "act.q.weight", tensors[f"{root}.action_queries.weight"])

    for i in range(dims.num_action_decoder_layers):
        lr = f"{root}.decoder.layers.{i}"

        add_tensor(writer, f"act.dec.{i}.self_qkv.weight", tensors[f"{lr}.self_attn.in_proj_weight"])
        add_tensor(writer, f"act.dec.{i}.self_qkv.bias", tensors[f"{lr}.self_attn.in_proj_bias"])
        add_tensor(writer, f"act.dec.{i}.self_out.weight", tensors[f"{lr}.self_attn.out_proj.weight"])
        add_tensor(writer, f"act.dec.{i}.self_out.bias", tensors[f"{lr}.self_attn.out_proj.bias"])

        add_tensor(writer, f"act.dec.{i}.cross_qkv.weight", tensors[f"{lr}.multihead_attn.in_proj_weight"])
        add_tensor(writer, f"act.dec.{i}.cross_qkv.bias", tensors[f"{lr}.multihead_attn.in_proj_bias"])
        add_tensor(writer, f"act.dec.{i}.cross_out.weight", tensors[f"{lr}.multihead_attn.out_proj.weight"])
        add_tensor(writer, f"act.dec.{i}.cross_out.bias", tensors[f"{lr}.multihead_attn.out_proj.bias"])

        add_tensor(writer, f"act.dec.{i}.ln1.weight", tensors[f"{lr}.norm1.weight"])
        add_tensor(writer, f"act.dec.{i}.ln1.bias", tensors[f"{lr}.norm1.bias"])
        add_tensor(writer, f"act.dec.{i}.ln2.weight", tensors[f"{lr}.norm2.weight"])
        add_tensor(writer, f"act.dec.{i}.ln2.bias", tensors[f"{lr}.norm2.bias"])
        add_tensor(writer, f"act.dec.{i}.ln3.weight", tensors[f"{lr}.norm3.weight"])
        add_tensor(writer, f"act.dec.{i}.ln3.bias", tensors[f"{lr}.norm3.bias"])

        add_tensor(writer, f"act.dec.{i}.fc1.weight", tensors[f"{lr}.linear1.weight"])
        add_tensor(writer, f"act.dec.{i}.fc1.bias", tensors[f"{lr}.linear1.bias"])
        add_tensor(writer, f"act.dec.{i}.fc2.weight", tensors[f"{lr}.linear2.weight"])
        add_tensor(writer, f"act.dec.{i}.fc2.bias", tensors[f"{lr}.linear2.bias"])

    act_proj_root = f"{root}.action_projection"
    add_tensor(writer, "act.proj.0.weight", tensors[f"{act_proj_root}.layers.0.weight"])
    add_tensor(writer, "act.proj.0.bias", tensors[f"{act_proj_root}.layers.0.bias"])
    add_tensor(writer, "act.proj.1.weight", tensors[f"{act_proj_root}.layers.1.weight"])
    add_tensor(writer, "act.proj.1.bias", tensors[f"{act_proj_root}.layers.1.bias"])
    add_tensor(writer, "act.proj.2.weight", tensors[f"{act_proj_root}.layers.2.weight"])
    add_tensor(writer, "act.proj.2.bias", tensors[f"{act_proj_root}.layers.2.bias"])


def write_state_projection(writer: gguf.GGUFWriter, tensors: dict) -> None:
    """Write state projection MLP."""
    root = PREFIX_STATE_PROJ

    add_tensor(writer, "state.proj.0.weight", tensors[f"{root}.net.0.weight"])
    add_tensor(writer, "state.proj.0.bias", tensors[f"{root}.net.0.bias"])
    add_tensor(writer, "state.proj.1.weight", tensors[f"{root}.net.1.weight"])
    add_tensor(writer, "state.proj.1.bias", tensors[f"{root}.net.1.bias"])
    add_tensor(writer, "state.proj.4.weight", tensors[f"{root}.net.4.weight"])
    add_tensor(writer, "state.proj.4.bias", tensors[f"{root}.net.4.bias"])

    add_tensor(writer, "state.proj.output_norm.weight", tensors[f"{root}.output_norm.weight"])
    add_tensor(writer, "state.proj.output_norm.bias", tensors[f"{root}.output_norm.bias"])
    add_tensor(writer, "state.proj.position", tensors[f"{root}.position"])


def write_view_embeddings(writer: gguf.GGUFWriter, tensors: dict) -> None:
    """Write view embeddings."""
    add_tensor(writer, "view_emb", tensors[KEY_VIEW_EMB])


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_consumed_tensors(source_keys: set[str], dims: TurboVLADims) -> None:
    """Verify all source tensors are consumed by the converter."""
    consumed = set()

    root_vit = PREFIX_VIT
    consumed.add(f"{root_vit}.embeddings.cls_token")
    consumed.add(f"{root_vit}.embeddings.patch_embeddings.weight")
    consumed.add(f"{root_vit}.embeddings.patch_embeddings.bias")
    reg_key = f"{root_vit}.embeddings.register_tokens"
    if reg_key in source_keys:
        consumed.add(reg_key)

    for i in range(dims.vit_layers):
        lr = f"{root_vit}.layer.{i}"
        consumed.add(f"{lr}.attention.q_proj.weight")
        consumed.add(f"{lr}.attention.q_proj.bias")
        consumed.add(f"{lr}.attention.k_proj.weight")
        kb = f"{lr}.attention.k_proj.bias"
        if kb in source_keys:
            consumed.add(kb)
        consumed.add(f"{lr}.attention.v_proj.weight")
        consumed.add(f"{lr}.attention.v_proj.bias")
        consumed.add(f"{lr}.attention.o_proj.weight")
        consumed.add(f"{lr}.attention.o_proj.bias")
        consumed.add(f"{lr}.norm1.weight")
        consumed.add(f"{lr}.norm1.bias")
        consumed.add(f"{lr}.norm2.weight")
        consumed.add(f"{lr}.norm2.bias")
        consumed.add(f"{lr}.mlp.up_proj.weight")
        consumed.add(f"{lr}.mlp.up_proj.bias")
        consumed.add(f"{lr}.mlp.down_proj.weight")
        consumed.add(f"{lr}.mlp.down_proj.bias")
        consumed.add(f"{lr}.layer_scale1.lambda1")
        consumed.add(f"{lr}.layer_scale2.lambda1")

    consumed.add(f"{root_vit}.norm.weight")
    consumed.add(f"{root_vit}.norm.bias")

    root_text = PREFIX_TEXT
    consumed.add(f"{root_text}.embeddings.word_embeddings.weight")
    consumed.add(f"{root_text}.embeddings.position_embeddings.weight")
    consumed.add(f"{root_text}.embeddings.token_type_embeddings.weight")
    consumed.add(f"{root_text}.embeddings.LayerNorm.weight")
    consumed.add(f"{root_text}.embeddings.LayerNorm.bias")
    consumed.add(f"{root_text}.pooler.dense.weight")
    consumed.add(f"{root_text}.pooler.dense.bias")

    for i in range(dims.text_layers):
        lr = f"{root_text}.encoder.layer.{i}"
        consumed.add(f"{lr}.attention.self.query.weight")
        consumed.add(f"{lr}.attention.self.query.bias")
        consumed.add(f"{lr}.attention.self.key.weight")
        consumed.add(f"{lr}.attention.self.key.bias")
        consumed.add(f"{lr}.attention.self.value.weight")
        consumed.add(f"{lr}.attention.self.value.bias")
        consumed.add(f"{lr}.attention.output.dense.weight")
        consumed.add(f"{lr}.attention.output.dense.bias")
        consumed.add(f"{lr}.attention.output.LayerNorm.weight")
        consumed.add(f"{lr}.attention.output.LayerNorm.bias")
        consumed.add(f"{lr}.intermediate.dense.weight")
        consumed.add(f"{lr}.intermediate.dense.bias")
        consumed.add(f"{lr}.output.dense.weight")
        consumed.add(f"{lr}.output.dense.bias")
        consumed.add(f"{lr}.output.LayerNorm.weight")
        consumed.add(f"{lr}.output.LayerNorm.bias")

    consumed.add(KEY_TEXT_PROJ + ".weight")
    consumed.add(KEY_TEXT_PROJ + ".bias")

    vp = PREFIX_VIT_PROJ
    consumed.add(f"{vp}.input_norm.weight")
    consumed.add(f"{vp}.input_norm.bias")
    consumed.add(f"{vp}.mlp.0.weight")
    consumed.add(f"{vp}.mlp.0.bias")
    consumed.add(f"{vp}.mlp.3.weight")
    consumed.add(f"{vp}.mlp.3.bias")
    consumed.add(f"{vp}.skip.weight")
    consumed.add(f"{vp}.output_norm.weight")
    consumed.add(f"{vp}.output_norm.bias")

    for i in range(dims.num_fusion_layers):
        lr = f"{PREFIX_FUSION}.{i}"
        consumed.add(f"{lr}.attn.v_proj.weight")
        consumed.add(f"{lr}.attn.v_proj.bias")
        consumed.add(f"{lr}.attn.l_proj.weight")
        consumed.add(f"{lr}.attn.l_proj.bias")
        consumed.add(f"{lr}.attn.values_v_proj.weight")
        consumed.add(f"{lr}.attn.values_v_proj.bias")
        consumed.add(f"{lr}.attn.values_l_proj.weight")
        consumed.add(f"{lr}.attn.values_l_proj.bias")
        consumed.add(f"{lr}.attn.out_v_proj.weight")
        consumed.add(f"{lr}.attn.out_v_proj.bias")
        consumed.add(f"{lr}.attn.out_l_proj.weight")
        consumed.add(f"{lr}.attn.out_l_proj.bias")
        consumed.add(f"{lr}.layer_norm_v.weight")
        consumed.add(f"{lr}.layer_norm_v.bias")
        consumed.add(f"{lr}.layer_norm_l.weight")
        consumed.add(f"{lr}.layer_norm_l.bias")
        consumed.add(f"{lr}.gamma_v")
        consumed.add(f"{lr}.gamma_l")

    for i in range(dims.num_text_layers):
        lr = f"{PREFIX_VL_TEXT}.{i}"
        consumed.add(f"{lr}.self_attn.in_proj_weight")
        consumed.add(f"{lr}.self_attn.in_proj_bias")
        consumed.add(f"{lr}.self_attn.out_proj.weight")
        consumed.add(f"{lr}.self_attn.out_proj.bias")
        consumed.add(f"{lr}.norm1.weight")
        consumed.add(f"{lr}.norm1.bias")
        consumed.add(f"{lr}.linear1.weight")
        consumed.add(f"{lr}.linear1.bias")
        consumed.add(f"{lr}.linear2.weight")
        consumed.add(f"{lr}.linear2.bias")
        consumed.add(f"{lr}.norm2.weight")
        consumed.add(f"{lr}.norm2.bias")

    consumed.add(f"{PREFIX_ACT_DEC}.action_queries.weight")
    for i in range(dims.num_action_decoder_layers):
        lr = f"{PREFIX_ACT_DEC}.decoder.layers.{i}"
        consumed.add(f"{lr}.self_attn.in_proj_weight")
        consumed.add(f"{lr}.self_attn.in_proj_bias")
        consumed.add(f"{lr}.self_attn.out_proj.weight")
        consumed.add(f"{lr}.self_attn.out_proj.bias")
        consumed.add(f"{lr}.multihead_attn.in_proj_weight")
        consumed.add(f"{lr}.multihead_attn.in_proj_bias")
        consumed.add(f"{lr}.multihead_attn.out_proj.weight")
        consumed.add(f"{lr}.multihead_attn.out_proj.bias")
        consumed.add(f"{lr}.norm1.weight")
        consumed.add(f"{lr}.norm1.bias")
        consumed.add(f"{lr}.norm2.weight")
        consumed.add(f"{lr}.norm2.bias")
        consumed.add(f"{lr}.norm3.weight")
        consumed.add(f"{lr}.norm3.bias")
        consumed.add(f"{lr}.linear1.weight")
        consumed.add(f"{lr}.linear1.bias")
        consumed.add(f"{lr}.linear2.weight")
        consumed.add(f"{lr}.linear2.bias")

    ap = f"{PREFIX_ACT_DEC}.action_projection"
    consumed.add(f"{ap}.layers.0.weight")
    consumed.add(f"{ap}.layers.0.bias")
    consumed.add(f"{ap}.layers.1.weight")
    consumed.add(f"{ap}.layers.1.bias")
    consumed.add(f"{ap}.layers.2.weight")
    consumed.add(f"{ap}.layers.2.bias")

    sp = PREFIX_STATE_PROJ
    consumed.add(f"{sp}.net.0.weight")
    consumed.add(f"{sp}.net.0.bias")
    consumed.add(f"{sp}.net.1.weight")
    consumed.add(f"{sp}.net.1.bias")
    consumed.add(f"{sp}.net.4.weight")
    consumed.add(f"{sp}.net.4.bias")
    consumed.add(f"{sp}.output_norm.weight")
    consumed.add(f"{sp}.output_norm.bias")
    consumed.add(f"{sp}.position")

    consumed.add(KEY_VIEW_EMB)

    missing = source_keys - consumed
    unexpected_missing = [
        k for k in missing
        if not any(x in k for x in ["_optim", "_avg", "mask", "running_", ".nbytes"])
        and not k.endswith(".idx")
    ]

    if unexpected_missing:
        print(f"\n  WARNING: {len(unexpected_missing)} keys not consumed:")
        for k in sorted(unexpected_missing)[:20]:
            print(f"    {k}")
        if len(unexpected_missing) > 20:
            print(f"    ... and {len(unexpected_missing) - 20} more")
        print(f"\n  Consumed: {len(consumed)} / {len(source_keys)} keys (missing {len(unexpected_missing)})")
    else:
        print(f"\n  All {len(consumed)} keys consumed successfully!")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description=f"Convert TurboVLA checkpoint to GGUF ({ARCH})"
    )
    parser.add_argument("ckpt", type=Path, help="Path to TurboVLA checkpoint directory")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output GGUF path")
    parser.add_argument("--dinov3-config", type=Path, required=True,
                        help="Validated config.json for the exact DINOv3 backbone")
    parser.add_argument("--verify", action="store_true", help="Verify all tensor keys are consumed")
    args = parser.parse_args()

    ckpt = args.ckpt.resolve()
    if not (ckpt / "model.safetensors").exists():
        raise SystemExit(f"model.safetensors not found in {ckpt}")

    out = args.out or (ckpt / f"{ARCH}.gguf")

    print(f"Loading checkpoint from {ckpt}...")
    tensors = load_safetensors(ckpt)
    keys = set(tensors.keys())
    print(f"  Loaded {len(tensors)} tensors")

    cfg_json = {}
    cfg_path = ckpt / "config.json"
    if cfg_path.exists():
        cfg_json = json.loads(cfg_path.read_text())
        print(f"  Config: {cfg_path}")

    dinov3_cfg_path = args.dinov3_config.resolve()
    if not dinov3_cfg_path.is_file():
        raise SystemExit(f"DINOv3 config not found: {dinov3_cfg_path}")
    dinov3_cfg = json.loads(dinov3_cfg_path.read_text())
    print(f"  DINOv3 config: {dinov3_cfg_path}")

    print("  Inferring dimensions from checkpoint...")
    dims = TurboVLADims(tensors, keys, cfg_json, dinov3_cfg)
    print(f"  Detected: {dims}")

    print(f"Writing GGUF to {out}...")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out), ARCH)

    kv = kv_prefix
    writer.add_string(kv("architecture"), ARCH)
    writer.add_uint32(kv("hidden"), dims.hidden_dim)
    writer.add_uint32(kv("vit_dim"), dims.vit_dim)
    writer.add_uint32(kv("vit_layers"), dims.vit_layers)
    writer.add_uint32(kv("vit_head_dim"), 64)
    writer.add_uint32(kv("vit_heads"), 12)
    writer.add_uint32(kv("text_dim"), dims.text_dim)
    writer.add_uint32(kv("text_layers"), dims.text_layers)
    writer.add_uint32(kv("text_head_dim"), 64)
    writer.add_uint32(kv("text_heads"), 12)
    writer.add_uint32(kv("num_fusion_layers"), dims.num_fusion_layers)
    # CORRECT: fusion_heads = nheads // 2, NOT nheads
    writer.add_uint32(kv("fusion_heads"), dims.fusion_heads)
    # CORRECT: fusion_head_dim = hidden / fusion_heads
    writer.add_uint32(kv("fusion_head_dim"), dims.fusion_head_dim)
    writer.add_uint32(kv("num_text_layers"), dims.num_text_layers)
    # Text enhancer: same split as fusion
    writer.add_uint32(kv("text_enhancer_heads"), dims.text_enhancer_heads)
    writer.add_uint32(kv("text_enhancer_head_dim"), dims.text_enhancer_head_dim)
    # Action transformer config
    writer.add_uint32(kv("action_heads"), dims.action_heads)
    writer.add_uint32(kv("action_head_dim"), dims.action_head_dim)
    writer.add_uint32(kv("num_action_decoder_layers"), dims.num_action_decoder_layers)
    writer.add_uint32(kv("action_dim"), dims.action_dim)
    writer.add_uint32(kv("state_dim"), dims.state_dim)
    writer.add_uint32(kv("num_state_tokens"), dims.num_state_tokens)
    writer.add_uint32(kv("action_horizon"), dims.action_horizon)
    writer.add_uint32(kv("image_size"), dims.image_size)
    writer.add_uint32(kv("num_views"), dims.num_views)
    writer.add_uint32(kv("patch_size"), dims.patch_size)
    writer.add_uint32(kv("num_register_tokens"), dims.num_register_tokens)
    writer.add_float32(kv("rope_theta"), float(dims.rope_theta))
    writer.add_uint32(kv("vocab_size"), dims.vocab_size)
    writer.add_uint32(kv("max_text_length"), int(cfg_json.get("text", {}).get("padding_length", 256)))
    # TurboVLA's BERT wrapper uses these exact punctuation IDs when it creates
    # sub-sentence attention masks. Persist them so the GGUF runtime does not
    # silently depend on a tokenizer installation.
    text_cfg = cfg_json.get("text", {})
    model_name = text_cfg.get("model_name_or_path", "bert-base-uncased")
    if model_name not in ("bert-base-uncased", "google-bert/bert-base-uncased"):
        raise SystemExit(
            "unsupported TurboVLA text tokenizer for deterministic mask conversion: "
            f"{model_name!r}"
        )
    writer.add_uint32(kv("pad_token_id"), 0)
    writer.add_uint32(kv("cls_token_id"), 101)
    writer.add_uint32(kv("sep_token_id"), 102)
    writer.add_uint32(kv("period_token_id"), 1012)
    writer.add_uint32(kv("question_token_id"), 1029)
    writer.add_float32(kv("dropout"), dims.dropout)

    print("  Writing vision encoder...")
    write_vision_encoder(writer, tensors, dims)

    print("  Writing text encoder...")
    write_text_encoder(writer, tensors, dims)

    print("  Writing vision projection...")
    write_vision_projection(writer, tensors)

    print("  Writing VL interaction...")
    write_vision_language_interaction(writer, tensors, dims)

    print("  Writing action decoder...")
    write_action_decoder(writer, tensors, dims)

    print("  Writing state projection...")
    write_state_projection(writer, tensors)

    print("  Writing view embeddings...")
    write_view_embeddings(writer, tensors)

    if cfg_json:
        writer.add_string(kv("config_json"), json.dumps(cfg_json))

    if args.verify:
        print("  Verifying tensor consumption...")
        verify_consumed_tensors(keys, dims)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  Done! Output: {out} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
