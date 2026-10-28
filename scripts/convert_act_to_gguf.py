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

"""Convert a LeRobot ACT checkpoint (policies/act) to GGUF.

Reads a pretrained_model directory: config.json, model.safetensors and, for
checkpoints saved by LeRobot >= 0.4, the normalizer processor files. Older
checkpoints (lerobot/act_aloha_sim_transfer_cube_human) keep their stats in
model.safetensors, which is read as a fallback.

At inference the VAE encoder is unused (the latent is zeros), so it is not
written. The ResNet's frozen batch norms are folded into the convolutions.

Usage:
    python convert_act_to_gguf.py --ckpt <pretrained_model dir> [--out act.gguf]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from gguf_blocks import load_processor_stats
from gguf_common import (
    add_array,
    add_f32,
    arg_parser,
    finish,
    kv_prefix,
    kv_u32,
    open_writer,
    read_json,
    resolve_out,
)

ARCH = "act"
KV = kv_prefix(ARCH)

# torchvision BasicBlock counts per stage; ACT's backbone option is a torchvision name.
RESNET_BLOCKS = {"resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3)}
BN_EPS = 1e-5  # torchvision FrozenBatchNorm2d default

# Read only at training time.
UNUSED = ("model.vae_encoder", "normalize_targets.", "model.backbone.fc.")


class TrackedTensors(dict):
    """A state dict that records which keys the writers read."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def fold_bn(conv_w: torch.Tensor, bn: str, t: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Conv (no bias) followed by FrozenBatchNorm2d, as one conv with a bias."""
    scale = t[f"{bn}.weight"].double() * torch.rsqrt(t[f"{bn}.running_var"].double() + BN_EPS)
    w = conv_w.double() * scale.view(-1, 1, 1, 1)
    b = t[f"{bn}.bias"].double() - t[f"{bn}.running_mean"].double() * scale
    return w.float(), b.float()


def write_conv_bn(writer, t: dict, conv: str, bn: str, dst: str) -> None:
    w, b = fold_bn(t[f"{conv}.weight"], bn, t)
    add_f32(writer, f"{dst}.weight", w)
    add_f32(writer, f"{dst}.bias", b)


def write_backbone(writer, t: dict, blocks: tuple[int, ...]) -> None:
    pfx = "model.backbone"
    write_conv_bn(writer, t, f"{pfx}.conv1", f"{pfx}.bn1", "bb.conv1")
    for li, n in enumerate(blocks, start=1):
        for bi in range(n):
            src, dst = f"{pfx}.layer{li}.{bi}", f"bb.layer{li}.{bi}"
            write_conv_bn(writer, t, f"{src}.conv1", f"{src}.bn1", f"{dst}.conv1")
            write_conv_bn(writer, t, f"{src}.conv2", f"{src}.bn2", f"{dst}.conv2")
            if f"{src}.downsample.0.weight" in t:
                write_conv_bn(writer, t, f"{src}.downsample.0", f"{src}.downsample.1", f"{dst}.down")


def write_attn(writer, t: dict, src: str, dst: str, dim: int) -> None:
    """nn.MultiheadAttention: in_proj is [W_q; W_k; W_v]. Split, since ACT adds
    the position embedding to the query and key inputs but not to the value."""
    w, b = t[f"{src}.in_proj_weight"], t[f"{src}.in_proj_bias"]
    for i, part in enumerate("qkv"):
        add_f32(writer, f"{dst}_{part}.weight", w[i*dim:(i + 1)*dim])
        add_f32(writer, f"{dst}_{part}.bias", b[i*dim:(i + 1)*dim])
    add_f32(writer, f"{dst}_o.weight", t[f"{src}.out_proj.weight"])
    add_f32(writer, f"{dst}_o.bias", t[f"{src}.out_proj.bias"])


def write_ffn_norms(writer, t: dict, src: str, dst: str, n_norms: int) -> None:
    for name, part in (("fc1", "linear1"), ("fc2", "linear2")):
        add_f32(writer, f"{dst}.{name}.weight", t[f"{src}.{part}.weight"])
        add_f32(writer, f"{dst}.{name}.bias", t[f"{src}.{part}.bias"])
    for i in range(1, n_norms + 1):
        add_f32(writer, f"{dst}.ln{i}.weight", t[f"{src}.norm{i}.weight"])
        add_f32(writer, f"{dst}.ln{i}.bias", t[f"{src}.norm{i}.bias"])


def write_transformer(writer, t: dict, n_enc: int, n_dec: int, dim: int, pre_norm: bool) -> None:
    for i in range(n_enc):
        src, dst = f"model.encoder.layers.{i}", f"enc.blk.{i}"
        write_attn(writer, t, f"{src}.self_attn", f"{dst}.attn", dim)
        write_ffn_norms(writer, t, src, dst, 2)
    if pre_norm:
        add_f32(writer, "enc.norm.weight", t["model.encoder.norm.weight"])
        add_f32(writer, "enc.norm.bias", t["model.encoder.norm.bias"])
    for i in range(n_dec):
        src, dst = f"model.decoder.layers.{i}", f"dec.blk.{i}"
        write_attn(writer, t, f"{src}.self_attn", f"{dst}.attn", dim)
        write_attn(writer, t, f"{src}.multihead_attn", f"{dst}.cross", dim)
        write_ffn_norms(writer, t, src, dst, 3)
    add_f32(writer, "dec.norm.weight", t["model.decoder.norm.weight"])
    add_f32(writer, "dec.norm.bias", t["model.decoder.norm.bias"])


def write_io(writer, t: dict, has_state: bool) -> None:
    # The latent is zeros at inference, so its projection is just the bias.
    add_f32(writer, "latent_tok", t["model.encoder_latent_input_proj.bias"])
    t["model.encoder_latent_input_proj.weight"]  # noqa: B018 (read for --verify)
    if has_state:
        add_f32(writer, "state_proj.weight", t["model.encoder_robot_state_input_proj.weight"])
        add_f32(writer, "state_proj.bias", t["model.encoder_robot_state_input_proj.bias"])
    add_f32(writer, "pos_1d", t["model.encoder_1d_feature_pos_embed.weight"])
    w = t["model.encoder_img_feat_input_proj.weight"]
    add_f32(writer, "img_proj.weight", w.reshape(w.shape[0], w.shape[1]))
    add_f32(writer, "img_proj.bias", t["model.encoder_img_feat_input_proj.bias"])
    add_f32(writer, "dec.pos", t["model.decoder_pos_embed.weight"])
    add_f32(writer, "action_head.weight", t["model.action_head.weight"])
    add_f32(writer, "action_head.bias", t["model.action_head.bias"])


def legacy_key(feature: str) -> str:
    return feature.replace(".", "_")


def read_stats(ckpt: Path, t: dict, feature: str, dim: int, legacy: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """mean/std of one feature: the processor files first, then model.safetensors buffers."""
    meta, registry = ("policy_postprocessor.json", "unnormalizer_processor") if feature == "action" \
        else ("policy_preprocessor.json", "normalizer_processor")
    got = load_processor_stats(ckpt, meta, registry, feature, dim)
    for pfx in legacy:
        if got is None and f"{pfx}.mean" in t:
            got = (t[f"{pfx}.mean"].float().numpy().reshape(-1), t[f"{pfx}.std"].float().numpy().reshape(-1))
            print(f"  stats: loaded {feature} from model.safetensors ({pfx})")
    if got is None or got[0].size != dim:
        raise SystemExit(f"no mean/std of {feature} (dim {dim}) in the processor files or model.safetensors")
    return got


def main() -> int:
    ap = arg_parser(ARCH, "LeRobot ACT pretrained_model directory (config.json + model.safetensors)",
                    description=__doc__)
    ap.add_argument("--verify", action="store_true", help="fail if a checkpoint tensor is left unconverted")
    args = ap.parse_args()
    ckpt = args.ckpt.resolve()
    out = resolve_out(args, ckpt, ARCH)

    cfg = read_json(ckpt / "config.json")
    if cfg.get("type") != "act":
        raise SystemExit(f"{ckpt}/config.json is not an ACT config (type={cfg.get('type')!r})")
    feats = cfg.get("input_features") or {}
    images = [k for k, v in feats.items() if v.get("type") == "VISUAL"]
    has_state = "observation.state" in feats
    if any(v.get("type") == "ENV" for v in feats.values()):
        raise SystemExit("observation.environment_state is not supported")
    if not images:
        raise SystemExit("ACT without cameras is not supported")
    shapes = {tuple(feats[k]["shape"]) for k in images}
    if len(shapes) != 1:
        raise SystemExit(f"cameras of different sizes are not supported: {sorted(shapes)}")
    _, img_h, img_w = shapes.pop()
    backbone = cfg.get("vision_backbone", "resnet18")
    if backbone not in RESNET_BLOCKS:
        raise SystemExit(f"vision_backbone {backbone!r} is not supported ({', '.join(RESNET_BLOCKS)})")
    if cfg.get("replace_final_stride_with_dilation"):
        raise SystemExit("replace_final_stride_with_dilation is not supported")
    act_fn = cfg.get("feedforward_activation", "relu")
    if act_fn not in ("relu", "gelu"):
        raise SystemExit(f"feedforward_activation {act_fn!r} is not supported (relu, gelu)")
    if cfg.get("temporal_ensemble_coeff") is not None:
        print("  note: temporal_ensemble_coeff is a client-side policy; the GGUF predicts the full chunk")
    norm_map = cfg.get("normalization_mapping") or {}
    for ftype in ("VISUAL", "STATE", "ACTION"):
        if norm_map.get(ftype, "MEAN_STD") != "MEAN_STD":
            raise SystemExit(f"normalization_mapping {ftype}={norm_map[ftype]} is not supported (MEAN_STD only)")

    with safe_open(str(ckpt / "model.safetensors"), framework="pt") as f:
        t = TrackedTensors({k: f.get_tensor(k) for k in f.keys()})

    dim = int(cfg["dim_model"])
    pre_norm = bool(cfg.get("pre_norm", False))
    n_enc, n_dec = int(cfg["n_encoder_layers"]), int(cfg["n_decoder_layers"])
    state_dim = int(feats["observation.state"]["shape"][0]) if has_state else 0
    action_dim = int(cfg["output_features"]["action"]["shape"][0])
    chunk = int(cfg["chunk_size"])
    if t["model.decoder_pos_embed.weight"].shape != (chunk, dim) or t["model.action_head.weight"].shape != (action_dim, dim):
        raise SystemExit("config.json disagrees with the checkpoint's tensor shapes")

    writer = open_writer(out, ARCH)
    kv_u32(writer, KV, {
        "dim_model": dim, "n_heads": cfg["n_heads"], "dim_feedforward": cfg["dim_feedforward"],
        "n_encoder_layers": n_enc, "n_decoder_layers": n_dec, "pre_norm": int(pre_norm),
        "gelu": int(act_fn == "gelu"), "chunk_size": chunk, "n_action_steps": cfg.get("n_action_steps", chunk),
        "state_dim": state_dim, "action_dim": action_dim, "num_views": len(images),
        "image_height": img_h, "image_width": img_w,
    })
    writer.add_array(KV("backbone_blocks"), list(RESNET_BLOCKS[backbone]))
    writer.add_array(KV("cameras"), images)

    print(f"  {backbone}, {len(images)} camera(s) at {img_h}x{img_w}, state {state_dim}, action {action_dim}, "
          f"chunk {chunk}, {n_enc}+{n_dec} layers, dim {dim}")
    write_backbone(writer, t, RESNET_BLOCKS[backbone])
    write_transformer(writer, t, n_enc, n_dec, dim, pre_norm)
    write_io(writer, t, has_state)

    img_mean = np.zeros((len(images), 3), dtype=np.float32)
    img_std = np.ones((len(images), 3), dtype=np.float32)
    for i, k in enumerate(images):
        img_mean[i], img_std[i] = read_stats(ckpt, t, k, 3, (f"normalize_inputs.buffer_{legacy_key(k)}",))
    add_array(writer, "image_mean", img_mean)
    add_array(writer, "image_std", img_std)
    if has_state:
        for name, vec in zip(("state_mean", "state_std"),
                             read_stats(ckpt, t, "observation.state", state_dim,
                                        ("normalize_inputs.buffer_observation_state",))):
            add_array(writer, name, vec)
    for name, vec in zip(("action_mean", "action_std"),
                         read_stats(ckpt, t, "action", action_dim, ("unnormalize_outputs.buffer_action",))):
        add_array(writer, name, vec)
    writer.add_string(KV("config_json"), json.dumps(cfg))

    if args.verify:
        left = sorted(k for k in t if k not in t.read and not k.startswith(UNUSED)
                      and not k.startswith("normalize_inputs.") and not k.startswith("unnormalize_outputs."))
        if left:
            raise SystemExit(f"{len(left)} checkpoint tensors not converted: {left[:20]}")
        print(f"  all {len(t.read)} used tensors consumed")

    return finish(writer, out)


if __name__ == "__main__":
    raise SystemExit(main())
