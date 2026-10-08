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

"""Convert a picovla (fast_smolvla) LeRobot checkpoint to GGUF.

Reads a directory holding model.safetensors, config.json and the LeRobot
policy_{pre,post}processor normalizer sidecars, e.g. a download of
khanhnd61/picovla-libero-pretrained-gguf. The latent head (the offline-RL
actor) is dropped: the runtime implements the supervised MeanFlow policy only.

Usage:
    python convert_picovla_to_gguf.py --ckpt <dir> --out picovla-libero-f32.gguf
"""

from __future__ import annotations

import numpy as np
from safetensors import safe_open

from gguf_blocks import lerobot_stats, norm_eps, write_decoder_blocks
from gguf_common import add, add_array, arg_parser, finish, kv_prefix, max_layer, open_writer, read_json, resolve_out

ARCH = "picovla"
KV = kv_prefix(ARCH)

PFX_BASE = "base_model"
PFX_LM   = "base_model.backbone.context_model"
PFX_AEX  = "base_model.expert_head"
PFX_VIS  = "base_model.backbone.vision_encoder.vision_model"
PFX_CONN = "base_model.backbone.vision_encoder.connector"

# Read from the bundled configs/{context_model,vision_model}/config.json of the
# picovla package; the checkpoint carries neither. RoPE is policy_utils.apply_rope,
# whose max_wavelength is 10000: the context config's rope_theta is never read.
LLAMA = dict(n_heads=8, n_kv_heads=4, head_dim=48, rms_eps=1e-5, rope_base=10000.0, pad_id=2)
CONVNEXT = dict(image_size=448, eps=1e-6, stem=4)

PROJ = [
    ("state_proj.weight",        "state_proj.weight"),
    ("state_proj.bias",          "state_proj.bias"),
    ("action_in_proj.weight",    "action_in_proj.weight"),
    ("action_in_proj.bias",      "action_in_proj.bias"),
    ("action_out_proj.0.weight", "action_out_proj.0.weight"),
    ("action_out_proj.0.bias",   "action_out_proj.0.bias"),
    ("action_out_proj.2.weight", "action_out_proj.2.weight"),
    ("action_out_proj.2.bias",   "action_out_proj.2.bias"),
    ("time_emb_proj.0.weight",   "time_mlp_in.weight"),
    ("time_emb_proj.0.bias",     "time_mlp_in.bias"),
    ("time_emb_proj.2.weight",   "time_mlp_out.weight"),
    ("time_emb_proj.2.bias",     "time_mlp_out.bias"),
]

# Expert tensors beyond the plain decoder map: the cross projections that read
# the backbone's cached prefix K/V, and the AdaRMSNorm modulations.
EXPERT_EXTRA = [
    ("self_attn.cross_k_proj.weight",      "cross_k.weight"),
    ("self_attn.cross_v_proj.weight",      "cross_v.weight"),
    ("input_adanorm.linear.weight",        "ada_attn.weight"),
    ("input_adanorm.linear.bias",          "ada_attn.bias"),
    ("post_attention_adanorm.linear.weight", "ada_ffn.weight"),
    ("post_attention_adanorm.linear.bias",   "ada_ffn.bias"),
]


def s2d_weight(w):
    """[OC, IC, k, k] conv weight as a [OC, k*k*IC] GEMM over (ky, kx, ic).

    The runtime keeps the ConvNeXt channels-last, so a stride-k k x k conv is a
    space-to-depth view of k x k pixels (channel fastest, then kx, then ky)
    followed by one matmul.
    """
    return w.permute(0, 2, 3, 1).reshape(w.shape[0], -1).contiguous()


def dw_weight(w):
    """[C, 1, 7, 7] depthwise weight as [ky, kx, C]: ggml's channels-last
    depthwise kernel reads the taps channel-fastest."""
    return w[:, 0].permute(1, 2, 0).contiguous()


def write_vision(writer, g, depths: list[int]) -> None:
    add(writer, "vis.stem.weight", s2d_weight(g(f"{PFX_VIS}.model.stages.0.downsample_layers.0.weight")))
    add(writer, "vis.stem.bias", g(f"{PFX_VIS}.model.stages.0.downsample_layers.0.bias"))
    add(writer, "vis.stem_norm.weight", g(f"{PFX_VIS}.model.stages.0.downsample_layers.1.weight"))
    add(writer, "vis.stem_norm.bias", g(f"{PFX_VIS}.model.stages.0.downsample_layers.1.bias"))
    for s, depth in enumerate(depths):
        st = f"{PFX_VIS}.model.stages.{s}"
        if s > 0:
            add(writer, f"vis.down.{s}.norm.weight", g(f"{st}.downsample_layers.0.weight"))
            add(writer, f"vis.down.{s}.norm.bias", g(f"{st}.downsample_layers.0.bias"))
            add(writer, f"vis.down.{s}.weight", s2d_weight(g(f"{st}.downsample_layers.1.weight")))
            add(writer, f"vis.down.{s}.bias", g(f"{st}.downsample_layers.1.bias"))
        for i in range(depth):
            lr, dst = f"{st}.layers.{i}", f"vis.blk.{s}.{i}"
            gamma = g(f"{lr}.gamma")
            add(writer, f"{dst}.dw.weight", dw_weight(g(f"{lr}.depthwise_conv.weight")))
            add(writer, f"{dst}.dw.bias", g(f"{lr}.depthwise_conv.bias"))
            add(writer, f"{dst}.ln.weight", g(f"{lr}.layer_norm.weight"))
            add(writer, f"{dst}.ln.bias", g(f"{lr}.layer_norm.bias"))
            add(writer, f"{dst}.fc1.weight", g(f"{lr}.pointwise_conv1.weight"))
            add(writer, f"{dst}.fc1.bias", g(f"{lr}.pointwise_conv1.bias"))
            # Layer scale folds into the second pointwise conv.
            add(writer, f"{dst}.fc2.weight", g(f"{lr}.pointwise_conv2.weight") * gamma.view(-1, 1))
            add(writer, f"{dst}.fc2.bias", g(f"{lr}.pointwise_conv2.bias") * gamma)
    add(writer, "vis.norm.weight", g(f"{PFX_VIS}.layer_norm.weight"))
    add(writer, "vis.norm.bias", g(f"{PFX_VIS}.layer_norm.bias"))
    add(writer, "conn.task_proj.weight", g(f"{PFX_CONN}.task_proj.weight"))
    add(writer, "conn.vision_proj.weight", g(f"{PFX_CONN}.vision_proj.weight"))
    add(writer, "conn.proj.weight", g(f"{PFX_CONN}.proj.weight"))


def write_expert(writer, g, n_layers: int) -> None:
    add(writer, "aex.cond_proj.weight", g(f"{PFX_AEX}.cond_proj.weight"))
    add(writer, "aex.cond_proj.bias", g(f"{PFX_AEX}.cond_proj.bias"))
    write_decoder_blocks(writer, g, f"{PFX_AEX}.model", "aex", n_layers)
    for i in range(n_layers):
        for src, dst in EXPERT_EXTRA:
            add(writer, f"aex.blk.{i}.{dst}", g(f"{PFX_AEX}.model.layers.{i}.{src}"))
    add(writer, "aex.output_ada.weight", g(f"{PFX_AEX}.model.norm.linear.weight"))
    add(writer, "aex.output_ada.bias", g(f"{PFX_AEX}.model.norm.linear.bias"))


def compact_vocab(used_mask: np.ndarray, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """CompactTaskEmbedding.compact() (policy/base_modules.py): keep the tokens
    seen in training plus padding, and send every other id to padding.

    Returns (rows of the original table to keep, original id -> compact id)."""
    used = used_mask.astype(bool)
    used[pad_id] = True
    rows = np.flatnonzero(used)
    remap = np.full(used.shape[0], -1, dtype=np.int64)
    remap[rows] = np.arange(rows.size)
    remap[remap < 0] = remap[pad_id]
    return rows, remap.astype(np.int32)


def main() -> int:
    ap = arg_parser(ARCH, "picovla checkpoint directory (model.safetensors + config.json)", __doc__)
    args = ap.parse_args()
    ckpt = args.ckpt.resolve()
    out = resolve_out(args, ckpt, ARCH)

    cfg_json = read_json(ckpt / "config.json")
    if cfg_json.get("type") != "fast_smolvla":
        raise SystemExit(f"{ckpt} is not a fast_smolvla checkpoint (type={cfg_json.get('type')!r})")
    if cfg_json.get("use_latent_head"):
        raise SystemExit("use_latent_head=true is not supported: the runtime samples Gaussian noise")
    for k, want in (("empty_cameras", 0), ("prefix_length", -1), ("num_language_layers", 1), ("n_obs_steps", 1)):
        if cfg_json.get(k, want) != want:
            raise SystemExit(f"{k}={cfg_json[k]} is not supported (expected {want})")
    images = [k for k, f in cfg_json["input_features"].items() if f["type"] == "VISUAL"]

    sf = safe_open(str(ckpt / "model.safetensors"), framework="pt")
    keys = set(sf.keys())
    g = sf.get_tensor

    n_layers = max_layer(keys, f"{PFX_LM}.layers.") - 1
    n_expert = max_layer(keys, f"{PFX_AEX}.model.layers.")
    if n_layers < 1 or n_expert != n_layers:
        raise SystemExit(f"backbone has {n_layers} joint layers but the expert has {n_expert}")
    depths = [max_layer(keys, f"{PFX_VIS}.model.stages.{s}.layers.") for s in range(4)]
    dims = [int(sf.get_slice(f"{PFX_VIS}.model.stages.{s}.layers.0.depthwise_conv.weight").get_shape()[0])
            for s in range(4)]

    hidden = int(sf.get_slice(f"{PFX_LM}.norm.weight").get_shape()[0])
    inter = int(sf.get_slice(f"{PFX_LM}.layers.0.mlp.gate_proj.weight").get_shape()[0])
    expert_inter, expert_h = sf.get_slice(f"{PFX_AEX}.model.layers.0.mlp.gate_proj.weight").get_shape()
    n_record = int(sf.get_slice(f"{PFX_BASE}.record_tokens").get_shape()[1])
    q_dim = int(sf.get_slice(f"{PFX_LM}.layers.0.self_attn.q_proj.weight").get_shape()[0])
    if q_dim != LLAMA["n_heads"] * LLAMA["head_dim"]:
        raise SystemExit(f"q_proj width {q_dim} != {LLAMA['n_heads']}x{LLAMA['head_dim']}")

    real_state = int(cfg_json["input_features"]["observation.state"]["shape"][0])
    real_action = int(cfg_json["output_features"]["action"]["shape"][0])
    lo, hi = (float(v) for v in cfg_json["transport_range"])

    print(f"resolved: hidden={hidden} layers=1+{n_layers} expert={expert_h}/{expert_inter} "
          f"convnext depths={depths} dims={dims} views={len(images)} records={n_record}")

    emb = g(f"{PFX_LM}.embed_tokens.token_embedding.weight")
    rows, remap = compact_vocab(g(f"{PFX_LM}.embed_tokens.token_id_map").numpy(), LLAMA["pad_id"])
    print(f"vocabulary: {rows.size} of {emb.shape[0]} tokens seen in training")

    print("loading normalizer stats...")
    stats = lerobot_stats(sf, ckpt, real_state, real_action, cfg_json.get("normalization_mapping") or {})

    writer = open_writer(out, ARCH)
    for k, v in dict(hidden=hidden, intermediate=inter, n_layers=n_layers, n_heads=LLAMA["n_heads"],
                     n_kv_heads=LLAMA["n_kv_heads"], head_dim=LLAMA["head_dim"], expert_hidden=int(expert_h),
                     expert_intermediate=int(expert_inter), n_record=n_record,
                     chunk_size=int(cfg_json["chunk_size"]), num_steps=int(cfg_json["n_denoise_steps"]),
                     max_state_dim=int(cfg_json["max_state_dim"]), max_action_dim=int(cfg_json["max_action_dim"]),
                     real_state_dim=real_state, real_action_dim=real_action, num_views=len(images),
                     image_size=CONVNEXT["image_size"], stem_patch=CONVNEXT["stem"],
                     tokenizer_max_length=int(cfg_json["tokenizer_max_length"]), vocab_size=int(emb.shape[0])).items():
        writer.add_uint32(KV(k), int(v))
    for k, v in dict(rms_eps=LLAMA["rms_eps"], rope_base=LLAMA["rope_base"], vis_eps=CONVNEXT["eps"],
                     transport_low=lo, transport_high=hi, min_period=float(cfg_json["min_period"]),
                     max_period=float(cfg_json["max_period"]), norm_eps=norm_eps(ckpt)).items():
        writer.add_float32(KV(k), float(v))
    writer.add_array(KV("vis_depths"), depths)
    writer.add_array(KV("vis_dims"), dims)
    writer.add_array(KV("token_map"), remap.tolist())

    add(writer, "token_embd.weight", emb[rows].contiguous())
    add(writer, "vlm.output_norm.weight", g(f"{PFX_LM}.norm.weight"))
    write_decoder_blocks(writer, g, PFX_LM, "vlm", 1 + n_layers)
    add(writer, "record_tokens", g(f"{PFX_BASE}.record_tokens")[0])
    for src, dst in PROJ:
        add(writer, dst, g(f"{PFX_BASE}.{src}"))
    write_expert(writer, g, n_layers)
    write_vision(writer, g, depths)

    for name, vec in stats.items():
        add_array(writer, name, vec)
    return finish(writer, out)


if __name__ == "__main__":
    raise SystemExit(main())
