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

"""Convert a TurboVLA checkpoint (H-EmbodVis/TurboVLA) to GGUF.

Reads the released .pth ({model_state_dict, model_config}, e.g.
checkpoints/libero/turbovla_libero.pth) or a directory holding
model.safetensors and config.json. Dimensions come from tensor shapes.

Usage:
    python convert_turbovla_to_gguf.py turbovla_libero.pth -o turbovla-libero.gguf [--verify]
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


# Prefixes used by official TurboVLA checkpoints (without a "model." prefix).
PREFIX_VIT = "vision_encoder.backbone"
PREFIX_TEXT = "text_encoder.bert"
PREFIX_VIT_PROJ = "vision_projection"
PREFIX_FUSION = "vision_language_interaction.fusion_layers"
PREFIX_VL_TEXT = "vision_language_interaction.text_layers"
PREFIX_ACT_DEC = "action_head.decoder"
PREFIX_STATE_PROJ = "action_head.state_projection"
KEY_VIEW_EMB = "view_embedding"
KEY_TEXT_PROJ = "text_encoder.text_projection"


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


# facebook/dinov3-vitb16-pretrain-lvd1689m is gated, but only these architecture
# values are needed: the fine-tuned weights ship inside the TurboVLA checkpoint.
DINOV3_VITB16 = {"rope_theta": 100.0, "num_register_tokens": 4}

# Tensors the runtime never reads: DINOv3's final norm (TurboVLA taps
# hidden_states[-1], before it), BERT's pooler, and the MAE mask token.
UNUSED = (f"{PREFIX_VIT}.norm.", f"{PREFIX_TEXT}.pooler.", f"{PREFIX_VIT}.embeddings.mask_token")


class TrackedTensors(dict):
    """A state dict that records which keys the writers read."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def load_checkpoint(ckpt: Path) -> tuple[TrackedTensors, dict]:
    """Return (state dict, TurboVLA model_config) from a .pth or a directory."""
    if ckpt.is_file():
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "model_state_dict" not in blob:
            raise SystemExit(f"{ckpt} is not a TurboVLA checkpoint (no model_state_dict)")
        state = blob["model_state_dict"]
        cfg = blob.get("model_config") or {}
    else:
        st_path = ckpt / "model.safetensors"
        if not st_path.exists():
            raise SystemExit(f"model.safetensors not found in {ckpt}")
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            state = {k: f.get_tensor(k) for k in f.keys()}
        cfg_path = ckpt / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    state = {k.removeprefix("module."): v for k, v in state.items()}
    return TrackedTensors(state), cfg


class TurboVLADims:
    """Auto-detect all TurboVLA dimensions from checkpoint tensor shapes."""

    def __init__(self, tensors: dict[str, torch.Tensor], keys: set[str], cfg_json: dict | None = None,
                 dinov3_cfg: dict | None = None):
        self.tensors = tensors
        self.keys = keys
        self.cfg = cfg_json or {}
        self.dinov3_cfg = dinov3_cfg or {}

        q0 = self._get(f"{PREFIX_VIT}.layer.0.attention.q_proj.weight")
        self.vit_dim = int(q0.shape[0])
        self.vit_layers = max_layer(keys, f"{PREFIX_VIT}.layer.")

        q0 = self._get(f"{PREFIX_TEXT}.encoder.layer.0.attention.self.query.weight")
        self.text_dim = int(q0.shape[0])
        self.text_layers = max_layer(keys, f"{PREFIX_TEXT}.encoder.layer.")

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

        self.text_enhancer_heads = max(1, raw_nheads // 2)
        self.text_enhancer_head_dim = self.hidden_dim // self.text_enhancer_heads

        # TurboVLA constructs the ACT decoder with interaction.nheads.
        self.action_heads = raw_nheads
        self.action_head_dim = self.hidden_dim // self.action_heads

        self.num_state_tokens = int(self.cfg.get("action", {}).get("num_state_tokens", 2))

        self.num_action_decoder_layers = max_layer(keys, f"{PREFIX_ACT_DEC}.decoder.layers.")
        action_q = self._get(f"{PREFIX_ACT_DEC}.action_queries.weight")
        self.action_horizon = int(action_q.shape[0])
        act_proj_2 = self._get(f"{PREFIX_ACT_DEC}.action_projection.layers.2.weight")
        self.action_dim = int(act_proj_2.shape[0])

        state_weight = self._get(f"{PREFIX_STATE_PROJ}.net.1.weight")
        self.state_dim = int(state_weight.shape[1])

        view_emb = self._get(KEY_VIEW_EMB)
        if view_emb.ndim == 3:
            if view_emb.shape[0] != 1:
                raise SystemExit(f"Unexpected view_embedding shape: {tuple(view_emb.shape)}")
            self.num_views = int(view_emb.shape[1])
        elif view_emb.ndim == 2:
            self.num_views = int(view_emb.shape[0])
        else:
            raise SystemExit(f"Unexpected view_embedding shape: {tuple(view_emb.shape)}")

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

        patch_weight = self._get(f"{PREFIX_VIT}.embeddings.patch_embeddings.weight")
        if len(patch_weight.shape) == 4:
            self.patch_size = patch_weight.shape[2]
        else:
            raise SystemExit("Cannot determine patch_size from patch_embeddings")

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
            backbone = self.cfg.get("vision", {}).get("model_name_or_path", "")
            if "dinov3-vitb16" not in backbone:
                raise SystemExit(f"unknown DINOv3 backbone {backbone!r}; pass --dinov3-config")
            self.dinov3_cfg = DINOV3_VITB16
        self.rope_theta = float(self.dinov3_cfg["rope_theta"])
        self.rope_normalize_coords = str(self.dinov3_cfg.get("rope_normalize_coords", "separate"))
        if self.rope_normalize_coords != "separate":
            raise SystemExit(
                f"unsupported DINOv3 rope coordinate normalization: {self.rope_normalize_coords!r}"
            )
        if int(self.dinov3_cfg.get("num_register_tokens", self.num_register_tokens)) != self.num_register_tokens:
            raise SystemExit("DINOv3 config num_register_tokens disagrees with checkpoint weights")

        word_emb = self._get(f"{PREFIX_TEXT}.embeddings.word_embeddings.weight")
        self.vocab_size = int(word_emb.shape[0])


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
        if self.dinov3_cfg:
            image_size = self.dinov3_cfg.get("image_size")
            if image_size:
                return int(image_size)

        pos_key = f"{PREFIX_VIT}.embeddings.position_embeddings.weight"
        if pos_key in self.tensors:
            pos = self.tensors[pos_key]
            seq_len = pos.shape[0] if pos.ndim == 2 else pos.shape[1]
            num_patches = seq_len - 1
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


def write_vision_encoder(writer: gguf.GGUFWriter, tensors: dict, dims: TurboVLADims) -> None:
    """Write DINOv3 ViT vision encoder.

    Tensor naming follows TurboVLA naming convention with attn_q/k/v/o.
    LayerScale is baked into weights.
    """
    root = PREFIX_VIT

    # PyTorch patch_embedding weight: [H, IC, KH, KW] = [768, 3, 16, 16]
    # For matmul-based patch embedding: flatten weight to [H, IC*KH*KW] = [768, 768]
    patch_weight = tensors[f"{root}.embeddings.patch_embeddings.weight"]
    patch_weight = patch_weight.reshape(dims.vit_dim, -1)
    add_tensor(writer, "vit.cls_token", tensors[f"{root}.embeddings.cls_token"].squeeze(0))
    add_tensor(writer, "vit.patch_embed.weight", patch_weight)
    add_tensor(writer, "vit.patch_embed.bias", tensors[f"{root}.embeddings.patch_embeddings.bias"])

    reg_key = f"{root}.embeddings.register_tokens"
    if reg_key in tensors:
        add_tensor(writer, "vit.register_tokens", tensors[reg_key].squeeze(0))

    for i in range(dims.vit_layers):
        lr = f"{root}.layer.{i}"

        ls1 = tensors[f"{lr}.layer_scale1.lambda1"].clone()
        ls2 = tensors[f"{lr}.layer_scale2.lambda1"].clone()

        w_q = tensors[f"{lr}.attention.q_proj.weight"]
        b_q = tensors[f"{lr}.attention.q_proj.bias"]
        w_k = tensors[f"{lr}.attention.k_proj.weight"]
        k_bias_key = f"{lr}.attention.k_proj.bias"
        b_k = tensors[k_bias_key] if k_bias_key in tensors else torch.zeros(dims.vit_dim, dtype=torch.float32)
        w_v = tensors[f"{lr}.attention.v_proj.weight"]
        b_v = tensors[f"{lr}.attention.v_proj.bias"]
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

        add_tensor(writer, f"vit.blk.{i}.ln1.weight", tensors[f"{lr}.norm1.weight"])
        add_tensor(writer, f"vit.blk.{i}.ln1.bias", tensors[f"{lr}.norm1.bias"])
        add_tensor(writer, f"vit.blk.{i}.ln2.weight", tensors[f"{lr}.norm2.weight"])
        add_tensor(writer, f"vit.blk.{i}.ln2.bias", tensors[f"{lr}.norm2.bias"])

        w_fc1 = tensors[f"{lr}.mlp.up_proj.weight"]
        b_fc1 = tensors[f"{lr}.mlp.up_proj.bias"]
        w_fc2 = tensors[f"{lr}.mlp.down_proj.weight"] * ls2.view(-1, 1)
        b_fc2 = tensors[f"{lr}.mlp.down_proj.bias"] * ls2

        add_tensor(writer, f"vit.blk.{i}.fc1.weight", w_fc1)
        add_tensor(writer, f"vit.blk.{i}.fc1.bias", b_fc1)
        add_tensor(writer, f"vit.blk.{i}.fc2.weight", w_fc2)
        add_tensor(writer, f"vit.blk.{i}.fc2.bias", b_fc2)


def write_text_encoder(writer: gguf.GGUFWriter, tensors: dict, dims: TurboVLADims) -> None:
    """Write BERT text encoder."""
    root = PREFIX_TEXT

    add_tensor(writer, "text.embed.word_embeddings", tensors[f"{root}.embeddings.word_embeddings.weight"])
    add_tensor(writer, "text.embed.position_embeddings", tensors[f"{root}.embeddings.position_embeddings.weight"])
    add_tensor(writer, "text.embed.token_type_embeddings", tensors[f"{root}.embeddings.token_type_embeddings.weight"])
    add_tensor(writer, "text.embed.LayerNorm.weight", tensors[f"{root}.embeddings.LayerNorm.weight"])
    add_tensor(writer, "text.embed.LayerNorm.bias", tensors[f"{root}.embeddings.LayerNorm.bias"])

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


def write_text_groups(writer: gguf.GGUFWriter, text_cfg: dict) -> None:
    """Store each training instruction's BERT tokens and padded length.

    The ACT decoder attends the padded text rows too, so the padded length is
    part of the model's output. TurboVLA pads each instruction to the length
    its training batch used (padding_length_by_instruction); the runtime only
    sees token ids, so the table is keyed by them.
    """
    groups = text_cfg.get("padding_length_by_instruction") or {}
    writer.add_uint32(kv_prefix("text_groups.count"), len(groups))
    if not groups:
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(text_cfg.get("model_name_or_path", "bert-base-uncased"))
    ids, lengths, pad_to = [], [], []
    for instruction, length in sorted(groups.items()):
        seq = tok(str(instruction), truncation=True, max_length=int(length))["input_ids"]
        ids += [int(t) for t in seq]
        lengths.append(len(seq))
        pad_to.append(int(length))
    writer.add_array(kv_prefix("text_groups.tokens"), ids)
    writer.add_array(kv_prefix("text_groups.lengths"), lengths)
    writer.add_array(kv_prefix("text_groups.pad_to"), pad_to)


def verify_consumed_tensors(tensors: TrackedTensors) -> None:
    """Fail on any checkpoint tensor that no writer read."""
    left = sorted(k for k in tensors if k not in tensors.read and not k.startswith(UNUSED))
    if left:
        raise SystemExit(f"{len(left)} checkpoint tensors not converted: {left[:20]}")
    print(f"  All {len(tensors.read)} used tensors consumed.")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description=f"Convert TurboVLA checkpoint to GGUF ({ARCH})"
    )
    parser.add_argument("ckpt", type=Path,
                        help="TurboVLA .pth, or a directory with model.safetensors + config.json")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output GGUF path")
    parser.add_argument("--dinov3-config", type=Path, default=None,
                        help="config.json of the DINOv3 backbone (default: built-in ViT-B/16 values)")
    parser.add_argument("--verify", action="store_true", help="Fail if a checkpoint tensor is left unconverted")
    args = parser.parse_args()

    ckpt = args.ckpt.resolve()
    out = args.out or (ckpt.with_suffix(".gguf") if ckpt.is_file() else ckpt / f"{ARCH}.gguf")

    print(f"Loading checkpoint from {ckpt}...")
    tensors, cfg_json = load_checkpoint(ckpt)
    keys = set(tensors.keys())
    print(f"  Loaded {len(tensors)} tensors")

    dinov3_cfg = {}
    if args.dinov3_config:
        dinov3_cfg = json.loads(args.dinov3_config.read_text())
        print(f"  DINOv3 config: {args.dinov3_config}")

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
    # Fusion attention splits the configured head count in half.
    writer.add_uint32(kv("fusion_heads"), dims.fusion_heads)
    writer.add_uint32(kv("fusion_head_dim"), dims.fusion_head_dim)
    writer.add_uint32(kv("num_text_layers"), dims.num_text_layers)
    writer.add_uint32(kv("text_enhancer_heads"), dims.text_enhancer_heads)
    writer.add_uint32(kv("text_enhancer_head_dim"), dims.text_enhancer_head_dim)
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
    # The runtime pads to a fixed length; a checkpoint without text.padding_length
    # pads each batch to its longest instruction, which it does not implement.
    padding_length = cfg_json.get("text", {}).get("padding_length")
    if padding_length is None:
        raise SystemExit("checkpoint config has no text.padding_length; padding to the longest "
                         "instruction is not supported")
    writer.add_uint32(kv("max_text_length"), int(padding_length))
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
    write_text_groups(writer, text_cfg)

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
        verify_consumed_tensors(tensors)

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
