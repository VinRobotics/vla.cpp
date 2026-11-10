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

"""Convert an Ai2 MolmoAct2 checkpoint (allenai/MolmoAct2-LIBERO and its siblings) to one GGUF.

    python scripts/convert_molmoact2_to_gguf.py --ckpt <snapshot dir> --out molmoact2-libero-bf16.gguf

The snapshot is the Hub repo as downloaded: config.json, norm_stats.json, tokenizer.json and the sharded
model-*.safetensors. Tensors are streamed shard by shard, so the 22 GB fp32 checkpoint converts in a few GB
of RAM.

What goes in, under vla.cpp's names (src/models/molmoact2.cpp reads them):
  vit.*      the SigLIP-so400m/14 tower at 378 px, only the 25 layers the adapter reads
  mm.*       the 2x2 attention pooling and the SwiGLU projector into the LM
  token_embd the LM vocabulary with the 128 added rows appended
  vlm.*      the 36-layer Qwen3-style LM; the fused q|k|v and up|gate projections are split into the
             Qwen3LM names (ff_proj is [up; gate], the reference applies silu to the second half)
  aex.*      the flow-matching action expert (36 adaLN blocks, one per LM layer)
  molmoact2.action_* / state_*   the normalization of one norm_tag (default libero)
The LM head is left out: continuous actions never decode a token.

--outtype bf16 (default) stores the GEMM weights in bf16 and norms, biases and the position table in f32;
--outtype f32 keeps every tensor f32.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import gguf
from gguf_common import BF16, F32, arg_parser, finish, kv_f32, kv_prefix, kv_u32, open_writer, read_json, require, resolve_out

try:
    import torch
    from safetensors import safe_open
except ImportError:
    pass

ARCH = "molmoact2"
kv = kv_prefix(ARCH)

# The processor's image tokens (processing_molmoact2.IMAGE_TOKENS); the LM attends bidirectionally among them.
IMAGE_TOKENS = ("<im_patch>", "<im_col>", "<im_start>", "<low_res_im_start>", "<frame_start>", "<im_end>",
                "<frame_end>", "<im_low>")
IMAGE_PROMPT = "<|image|>"
NORM_MODES = {"q01_q99": ("q01", "q99"), "q10_q90": ("q10", "q90"), "min_max": ("min", "max"),
              "mean_std": ("mean", "std")}


class Shards:
    """Tensor getter over a sharded safetensors snapshot, one open handle per shard."""

    def __init__(self, ckpt: Path):
        idx = ckpt / "model.safetensors.index.json"
        if idx.exists():
            self.weight_map = json.loads(idx.read_text())["weight_map"]
        else:
            one = ckpt / "model.safetensors"
            require(one)
            with safe_open(str(one), framework="pt") as f:
                self.weight_map = {k: one.name for k in f.keys()}
        self.ckpt = ckpt
        self.handles = {}

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def _f(self, name: str):
        if name not in self.weight_map:
            raise SystemExit(f"checkpoint has no tensor {name}")
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.ckpt / shard), framework="pt")
        return self.handles[shard]

    def shape(self, name: str) -> list[int]:
        return list(self._f(name).get_slice(name).get_shape())

    def get(self, name: str) -> "torch.Tensor":
        return self._f(name).get_tensor(name)


class Lazy:
    """What GGUFWriter needs of a tensor (shape, dtype, nbytes, tofile), materialized only at write time."""

    def __init__(self, shape: list[int], bf16: bool, make):
        self.shape = tuple(int(x) for x in shape)
        self.nbytes = math.prod(self.shape) * (2 if bf16 else 4)
        self.bf16 = bf16
        self.make = make

    @property
    def dtype(self):
        return np.dtype(np.uint16 if self.bf16 else np.float32)

    def tofile(self, fout) -> None:
        t = self.make().float().contiguous()
        if tuple(t.shape) != self.shape:
            raise SystemExit(f"tensor shape {tuple(t.shape)} != declared {self.shape}")
        if self.bf16:
            t.to(torch.bfloat16).view(torch.uint16).numpy().tofile(fout)
        else:
            t.numpy().tofile(fout)


class Emitter:
    def __init__(self, writer: gguf.GGUFWriter, sh: Shards, gemm_bf16: bool):
        self.w, self.sh, self.gemm_bf16 = writer, sh, gemm_bf16
        self.n = 0

    def _add(self, dst: str, shape: list[int], bf16: bool, make) -> None:
        self.w.add_tensor(dst, Lazy(shape, bf16, make), raw_shape=shape, raw_dtype=BF16 if bf16 else F32)
        self.n += 1

    def gemm(self, dst: str, src: str, rows: tuple[int, int] | None = None) -> None:
        shape = self.sh.shape(src)
        if rows is not None:
            shape = [rows[1] - rows[0], *shape[1:]]
            self._add(dst, shape, self.gemm_bf16, lambda: self.sh.get(src)[rows[0]:rows[1]])
        else:
            self._add(dst, shape, self.gemm_bf16, lambda: self.sh.get(src))

    def f32(self, dst: str, src: str) -> None:
        self._add(dst, self.sh.shape(src), False, lambda: self.sh.get(src))

    def linear(self, dst: str, src: str, bias: bool = True) -> None:
        self.gemm(f"{dst}.weight", f"{src}.weight")
        if bias:
            self.f32(f"{dst}.bias", f"{src}.bias")

    def array(self, dst: str, a: np.ndarray) -> None:
        a = np.ascontiguousarray(a, dtype=np.float32)
        self.w.add_tensor(dst, a, raw_dtype=F32)
        self.n += 1


def emit_weights(e: Emitter, sh: Shards, tx: dict, ae: dict, vit_used: int, emb_rows: int) -> None:
    """Every weight, under the names src/models/molmoact2.cpp declares."""
    lm_inter = int(tx["intermediate_size"])
    hq = int(tx["num_attention_heads"]) * int(tx["head_dim"])
    hkv = int(tx["num_key_value_heads"]) * int(tx["head_dim"])
    V = "model.vision_backbone."
    e.linear("vit.patch_embd", V + "image_vit.patch_embedding")
    e.f32("vit.pos_embd", V + "image_vit.positional_embedding")
    for i in range(vit_used):
        S, D = f"{V}image_vit.transformer.resblocks.{i}.", f"vit.blk.{i}."
        e.f32(D + "ln1.weight", S + "attention_norm.weight"); e.f32(D + "ln1.bias", S + "attention_norm.bias")
        e.f32(D + "ln2.weight", S + "ffn_norm.weight");       e.f32(D + "ln2.bias", S + "ffn_norm.bias")
        for q in "qkvo":
            e.linear(D + f"attn_{q}", S + f"attention.w{q}")
        e.linear(D + "fc1", S + "feed_forward.w1")
        e.linear(D + "fc2", S + "feed_forward.w2")
    for q in "qkvo":
        e.linear(f"mm.pool.attn_{q}", V + f"image_pooling_2d.w{q}")
    e.linear("mm.proj.ffn_gate", V + "image_projector.w1", bias=False)
    e.linear("mm.proj.ffn_down", V + "image_projector.w2", bias=False)
    e.linear("mm.proj.ffn_up", V + "image_projector.w3", bias=False)

    T = "model.transformer."
    hidden = int(tx["hidden_size"])
    e._add("token_embd.weight", [emb_rows, hidden], e.gemm_bf16,
           lambda: torch.cat([sh.get(T + "wte.embedding"), sh.get(T + "wte.new_embedding")], 0))
    e.f32("vlm.output_norm.weight", T + "ln_f.weight")
    for i in range(int(tx["num_hidden_layers"])):
        S, D = f"{T}blocks.{i}.", f"vlm.blk.{i}."
        e.f32(D + "attn_norm.weight", S + "attn_norm.weight")
        e.gemm(D + "attn_q.weight", S + "self_attn.att_proj.weight", (0, hq))
        e.gemm(D + "attn_k.weight", S + "self_attn.att_proj.weight", (hq, hq + hkv))
        e.gemm(D + "attn_v.weight", S + "self_attn.att_proj.weight", (hq + hkv, hq + 2 * hkv))
        e.gemm(D + "attn_o.weight", S + "self_attn.attn_out.weight")
        e.f32(D + "attn_q_norm.weight", S + "self_attn.q_norm.weight")
        e.f32(D + "attn_k_norm.weight", S + "self_attn.k_norm.weight")
        e.f32(D + "ffn_norm.weight", S + "ff_norm.weight")
        e.gemm(D + "ffn_up.weight", S + "mlp.ff_proj.weight", (0, lm_inter))
        e.gemm(D + "ffn_gate.weight", S + "mlp.ff_proj.weight", (lm_inter, 2 * lm_inter))
        e.gemm(D + "ffn_down.weight", S + "mlp.ff_out.weight")

    A = "model.action_expert."
    e.linear("aex.time_in", A + "time_embed.1")
    e.linear("aex.time_out", A + "time_embed.3")
    e.linear("aex.action_in", A + "action_embed")
    e.linear("aex.ctx_k", A + "context_k_proj", bias=False)
    e.linear("aex.ctx_v", A + "context_v_proj", bias=False)
    for i in range(int(ae["num_layers"])):
        S, D = f"{A}blocks.{i}.", f"aex.blk.{i}."
        e.linear(D + "attn_qkv", S + "self_attn.qkv")
        e.linear(D + "attn_o", S + "self_attn.out_proj")
        e.linear(D + "cross_q", S + "cross_attn.q_proj")
        e.linear(D + "cross_o", S + "cross_attn.out_proj")
        e.linear(D + "ffn_gate", S + "mlp.gate_proj")
        e.linear(D + "ffn_up", S + "mlp.up_proj")
        e.linear(D + "ffn_down", S + "mlp.down_proj")
        e.linear(D + "mod", S + "modulation.linear")
    e.linear("aex.final_mod", A + "final_layer.modulation.linear")
    e.linear("aex.action_out", A + "final_layer.linear")


def token_id(tok_json: dict, content: str) -> int:
    for t in tok_json["added_tokens"]:
        if t["content"] == content:
            return int(t["id"])
    raise SystemExit(f"tokenizer.json has no added token {content!r}")


def norm_arrays(stats: dict, mode: str, what: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(low, high, mask, zero) of one feature for the reference's _FeatureNormalizer."""
    if mode == "none":
        raise SystemExit(f"norm_mode 'none' is not supported ({what})")
    lo_key, hi_key = NORM_MODES[mode]
    if lo_key not in stats or hi_key not in stats:
        raise SystemExit(f"{what} stats lack {lo_key}/{hi_key} for norm_mode {mode}")
    lo = np.asarray(stats[lo_key], dtype=np.float32).reshape(-1)
    hi = np.asarray(stats[hi_key], dtype=np.float32).reshape(-1)
    mask = np.asarray(stats.get("mask", [True] * lo.size), dtype=bool).reshape(-1)
    if mask.size != lo.size:
        mask = np.broadcast_to(mask, lo.shape)
    zero = np.zeros(lo.size, dtype=bool)
    if mode != "mean_std" and "min" in stats and "max" in stats:
        zero = np.asarray(stats["min"], dtype=np.float32).reshape(-1) == np.asarray(stats["max"], dtype=np.float32).reshape(-1)
    return lo, hi, mask.astype(np.float32), zero.astype(np.float32)


def main() -> int:
    ap = arg_parser(ARCH, "MolmoAct2 snapshot dir (config.json, norm_stats.json, tokenizer.json, model-*.safetensors)",
                    __doc__)
    ap.add_argument("--outtype", choices=("bf16", "f32"), default="bf16",
                    help="storage of the GEMM weights (norms, biases and tables are always f32)")
    ap.add_argument("--norm-tag", default=None,
                    help="norm_stats.json tag to bake in (default: the only one, or libero)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out = resolve_out(args, ckpt, ARCH)
    cfg = read_json(ckpt / "config.json")
    norm = read_json(ckpt / cfg.get("norm_stats_filename", "norm_stats.json"))
    tok_json = read_json(ckpt / "tokenizer.json")
    proc = read_json(ckpt / "processor_config.json") if (ckpt / "processor_config.json").exists() else {}

    vit, ad, tx, ae = cfg["vit_config"], cfg["adapter_config"], cfg["text_config"], cfg["action_expert_config"]
    if not cfg.get("add_action_expert", False):
        raise SystemExit("checkpoint has no action expert (add_action_expert=false)")
    if cfg.get("state_format", "discrete") != "discrete":
        raise SystemExit(f"state_format {cfg.get('state_format')!r} is not supported (discrete state tokens only)")
    if cfg.get("action_expert_depth_gate", False) or cfg.get("enable_depth_reasoning", False):
        raise SystemExit("depth gating / depth reasoning checkpoints are not supported")
    if ae.get("causal_attn", False) or not ae.get("context_layer_norm", True) or not ae.get("qk_norm", True) \
            or not ae.get("rope", True):
        raise SystemExit("action expert variant not supported (needs causal_attn=false, context_layer_norm, "
                         "qk_norm and rope)")
    if tx.get("norm_after", False) or tx.get("qkv_bias", False) or tx.get("qk_norm_type", "qwen3") != "qwen3" \
            or tx.get("rope_scaling_layers") is not None:
        raise SystemExit("LM variant not supported (needs pre-norm, no qkv bias, qwen3 qk-norm, plain RoPE)")
    if int(ae["num_layers"]) != int(tx["num_hidden_layers"]):
        raise SystemExit("action expert must have one block per LM layer")
    ip = proc.get("image_processor", {})
    if ip.get("crop_mode", "resize") != "resize":
        raise SystemExit(f"image crop_mode {ip.get('crop_mode')!r} is not supported (resize only)")
    if proc.get("use_single_crop_col_tokens", False):
        raise SystemExit("use_single_crop_col_tokens=true is not supported")
    pool_h, pool_w = (int(x) for x in ip.get("pooling_size", [2, 2]))
    size = ip.get("size", {"height": vit["image_default_input_size"][0], "width": vit["image_default_input_size"][1]})
    image_size = int(size["height"])
    if int(size["width"]) != image_size or image_size % int(vit["image_patch_size"]) != 0:
        raise SystemExit(f"image size {size} is not a square multiple of the patch")
    if [float(x) for x in ip.get("image_mean", [0.5] * 3)] != [0.5] * 3 or \
            [float(x) for x in ip.get("image_std", [0.5] * 3)] != [0.5] * 3:
        raise SystemExit("image mean/std other than 0.5 are not supported")

    n_vit = int(vit["num_hidden_layers"])
    feat_layers = [int(l) if int(l) >= 0 else int(l) + n_vit for l in ad["vit_layers"]]
    vit_used = max(feat_layers) + 1

    tags = sorted(norm.get("metadata_by_tag", {}))
    tag = args.norm_tag or (tags[0] if len(tags) == 1 else "libero")
    if tag not in tags:
        raise SystemExit(f"norm tag {tag!r} not in norm_stats.json (have {tags})")
    meta = norm["metadata_by_tag"][tag]
    mode = str(norm.get("norm_mode", "min_max"))
    if mode not in NORM_MODES:
        raise SystemExit(f"norm_mode {mode!r} is not supported")
    a_lo, a_hi, a_mask, a_zero = norm_arrays(meta["action_stats"], mode, "action")
    s_lo, s_hi, s_mask, s_zero = norm_arrays(meta["state_stats"], mode, "state")
    action_dim = a_lo.size
    horizon = int(meta.get("action_horizon", cfg["max_action_horizon"]))
    n_action_steps = int(meta.get("n_action_steps", horizon))
    if horizon > int(cfg["max_action_horizon"]) or action_dim > int(cfg["max_action_dim"]):
        raise SystemExit(f"tag {tag}: horizon {horizon} / action dim {action_dim} exceed the checkpoint's")

    sh = Shards(ckpt)
    lm_inter = int(tx["intermediate_size"])
    hq = int(tx["num_attention_heads"]) * int(tx["head_dim"])
    hkv = int(tx["num_key_value_heads"]) * int(tx["head_dim"])
    if sh.shape("model.transformer.blocks.0.self_attn.att_proj.weight")[0] != hq + 2 * hkv:
        raise SystemExit("att_proj is not [q; k; v]")
    if sh.shape("model.transformer.blocks.0.mlp.ff_proj.weight")[0] != 2 * lm_inter:
        raise SystemExit("ff_proj is not [up; gate]")
    for i in range(vit_used):
        if f"model.vision_backbone.image_vit.transformer.resblocks.{i}.attention.wq.weight" not in sh:
            raise SystemExit(f"checkpoint lacks ViT layer {i}, the adapter reads layers {feat_layers}")
    emb_rows = sh.shape("model.transformer.wte.embedding")[0] + sh.shape("model.transformer.wte.new_embedding")[0]
    inner = int(np.ceil(int(ae["hidden_size"]) * float(ae["mlp_ratio"]) / int(ae["ffn_multiple_of"]))) * int(ae["ffn_multiple_of"])

    writer = open_writer(out, ARCH)
    kv_u32(writer, kv, {
        "image_size": image_size, "patch_size": vit["image_patch_size"], "pool_h": pool_h, "pool_w": pool_w,
        "vit_hidden": vit["hidden_size"], "vit_heads": vit["num_attention_heads"], "vit_head_dim": vit["head_dim"],
        "vit_inter": vit["intermediate_size"], "vit_layers": vit_used, "vit_n_feature_layers": len(feat_layers),
        "adapter_hidden": ad["hidden_size"], "adapter_heads": ad["num_attention_heads"],
        "adapter_head_dim": ad["head_dim"], "adapter_inter": ad["intermediate_size"],
        "lm_hidden": tx["hidden_size"], "lm_layers": tx["num_hidden_layers"], "lm_q_heads": tx["num_attention_heads"],
        "lm_kv_heads": tx["num_key_value_heads"], "lm_head_dim": tx["head_dim"], "lm_inter": lm_inter,
        "vocab_size": emb_rows,
        "aex_hidden": ae["hidden_size"], "aex_heads": ae["num_heads"], "aex_layers": ae["num_layers"],
        "aex_inter": inner, "aex_time_dim": ae["timestep_embed_dim"],
        "max_action_dim": cfg["max_action_dim"], "max_action_horizon": cfg["max_action_horizon"],
        "num_steps": cfg.get("flow_matching_num_steps", 10),
        "action_dim": action_dim, "action_horizon": horizon, "n_action_steps": n_action_steps,
        "state_dim": s_lo.size, "num_state_tokens": cfg["num_state_tokens"],
        "state_token_start_id": cfg["state_token_start_id"],
        "image_prompt_id": token_id(tok_json, IMAGE_PROMPT),
        "image_start_id": cfg["image_start_token_id"], "image_end_id": cfg["image_end_token_id"],
        "image_patch_id": cfg["image_patch_id"],
        "bos_id": cfg["bos_token_id"], "eos_id": cfg["eos_token_id"],
        "action_start_id": cfg["action_start_token_id"], "action_end_id": cfg["action_end_token_id"],
    })
    for i, l in enumerate(feat_layers):
        writer.add_uint32(kv(f"vit_feature_layer.{i}"), l)
    writer.add_array(kv("image_token_ids"), [token_id(tok_json, t) for t in IMAGE_TOKENS])
    kv_f32(writer, kv, {"vit_ln_eps": vit["layer_norm_eps"], "lm_rms_eps": tx["layer_norm_eps"],
                        "aex_qk_norm_eps": ae.get("qk_norm_eps", 1e-6)})
    writer.add_float64(kv("lm_rope_theta"), float(tx.get("rope_theta", tx.get("rope_parameters", {}).get("rope_theta"))))
    writer.add_string(kv("norm_tag"), tag)
    writer.add_string(kv("norm_mode"), mode)
    writer.add_string(kv("setup_type"), str(meta.get("setup_type", "")))
    writer.add_string(kv("control_mode"), str(meta.get("control_mode", "")))
    writer.add_bool(kv("add_setup_tokens"), bool(cfg.get("add_setup_tokens", False)))
    writer.add_bool(kv("add_control_tokens"), bool(cfg.get("add_control_tokens", False)))
    writer.add_bool(kv("mask_action_dim_padding"), bool(cfg.get("mask_action_dim_padding", True)))
    # _get_encoder_attention_mask hides <|im_end|> and discrete action spans from the expert only in this mode.
    writer.add_bool(kv("encoder_mask_eos"), cfg.get("action_mode") == "both")
    writer.add_string(kv("norm_stats_json"), json.dumps(norm))

    e = Emitter(writer, sh, args.outtype == "bf16")
    e.array("molmoact2.action_low", a_lo)
    e.array("molmoact2.action_high", a_hi)
    e.array("molmoact2.action_mask", a_mask)
    e.array("molmoact2.state_low", s_lo)
    e.array("molmoact2.state_high", s_hi)
    e.array("molmoact2.state_mask", s_mask)
    e.array("molmoact2.state_zero", s_zero)

    emit_weights(e, sh, tx, ae, vit_used, emb_rows)

    print(f"  {e.n} tensors, norm tag {tag} ({mode}), action dim {action_dim} x horizon {horizon}, "
          f"ViT layers {vit_used} (features from {feat_layers}), {args.outtype} GEMM weights")
    return finish(writer, out, f"  - MolmoAct2 (SigLIP {vit_used}L + Qwen3 {tx['num_hidden_layers']}L + "
                               f"flow expert {ae['num_layers']}L)")


if __name__ == "__main__":
    raise SystemExit(main())
