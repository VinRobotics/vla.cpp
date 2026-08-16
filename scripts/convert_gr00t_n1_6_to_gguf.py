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

import json

import torch
import torch.nn.functional as F

from gguf_blocks import (
    write_dit_blocks,
    write_gr00t_proj_out,
    write_gr00t_projectors,
    write_gr00t_time_embed,
    write_qwen3_lm,
    write_siglip_tower
)
from gguf_common import (
    add,
    arg_parser,
    check_layers,
    finish,
    kv_f32,
    kv_prefix,
    kv_u32,
    load_safetensors,
    max_layer,
    open_writer,
    read_json,
    read_text,
    require,
    resolve_out
)

ARCH = "gr00t_n1_6"
KV = kv_prefix(ARCH)

VIT = dict(
    vit_hidden=1152,
    vit_layers=27,
    vit_heads=16,
    vit_inter=4304,
    image_size=224,
    patch_size=14,
    vit_ln_eps=1e-6
)

QWEN3 = dict(
    lm_hidden=2048,
    lm_q_heads=16,
    lm_kv_heads=8,
    lm_head_dim=128,
    lm_inter=6144,
    lm_rope_theta=1000000.0,
    lm_rms_eps=1e-6
)
LM_LAYERS_USED = 16

AH = dict(
    backbone_embedding_dim=2048,
    input_embedding_dim=1536,
    dit_hidden=1536,
    dit_heads=32,
    dit_head_dim=48,
    dit_layers=32,
    dit_interleave=1,
    attend_text_every_n_blocks=2,
    action_horizon=50,
    action_dim=128,
    max_state_dim=128,
    num_inference_timesteps=4,
    num_timestep_buckets=1000,
    max_num_embodiments=32,
    max_seq_len=1024,
    ln_eps=1e-5,
    norm_out_eps=1e-6,
    vlln_eps=1e-5
)
IMAGE_TOKEN_INDEX = 151669

VIT_ROOT = "backbone.model.vision_model.vision_model"
LM_ROOT  = "backbone.model.language_model.model"
AHK      = "action_head"

def _resize_pos_embd(pos: torch.Tensor, grid: int) -> torch.Tensor:

    native = int(round(pos.shape[0] ** 0.5))
    if native == grid:
        return pos
    src_dtype = pos.dtype
    p = pos.to(torch.float32).reshape(native, native, -1).permute(2, 0, 1).unsqueeze(0)
    p = F.interpolate(p, size=(grid, grid), mode="bilinear", align_corners=False, antialias=True)
    print(f"  interpolated vit.pos_embd {native}×{native} → {grid}×{grid} "
          f"({native * native}→{grid * grid}) bilinear+antialias")
    return p.reshape(pos.shape[-1], grid * grid).transpose(0, 1).contiguous().to(src_dtype)

def main() -> int:
    ap = arg_parser(ARCH, "GR00T-N1.6-3B snapshot dir")
    ap.add_argument(
        "--vision-size",
        type=int,
        default=None,
        help="Override the SigLIP2 vision-tower input resolution (default: native 224). "
             "Set 252 to match the reference processor's smart_resize(factor=28) of a "
             "256px image (252 = 18×14 patches ⇒ 324 patches ⇒ 81 tokens after "
             "pixel_shuffle÷2). When != 224 the `vit.pos_embd` is bilinear-antialias "
             "interpolated from the native 16×16 grid to the new grid, exactly mirroring "
             "SiglipVisionEmbeddings.resize_positional_embeddings (F.interpolate "
             "mode=bilinear, align_corners=False, antialias=True, float32). The runtime "
             "is otherwise resolution-agnostic (grid = image_size/patch_size)."
    )
    args = ap.parse_args()
    if args.vision_size is not None:
        if args.vision_size % VIT["patch_size"] != 0:
            raise SystemExit(f"--vision-size {args.vision_size} not divisible by patch_size {VIT['patch_size']}")
        VIT["image_size"] = int(args.vision_size)

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    require(ckpt / "model.safetensors.index.json")
    cfg_json = read_json(ckpt / "config.json")
    if str(cfg_json.get("model_type", "")) != "Gr00tN1d6":
        raise SystemExit(f"config.json model_type is {cfg_json.get('model_type')!r}, expected 'Gr00tN1d6'")
    if int(cfg_json.get("select_layer", LM_LAYERS_USED)) != LM_LAYERS_USED:
        raise SystemExit(f"select_layer = {cfg_json.get('select_layer')}, expected {LM_LAYERS_USED}")

    AH["action_horizon"]             = int(cfg_json.get("action_horizon", AH["action_horizon"]))
    AH["action_dim"]                 = int(cfg_json.get("max_action_dim", AH["action_dim"]))
    AH["max_state_dim"]              = int(cfg_json.get("max_state_dim", AH["max_state_dim"]))
    AH["num_inference_timesteps"]    = int(cfg_json.get("num_inference_timesteps", AH["num_inference_timesteps"]))
    AH["num_timestep_buckets"]       = int(cfg_json.get("num_timestep_buckets", AH["num_timestep_buckets"]))
    AH["max_num_embodiments"]        = int(cfg_json.get("max_num_embodiments", AH["max_num_embodiments"]))
    AH["max_seq_len"]                = int(cfg_json.get("max_seq_len", AH["max_seq_len"]))
    AH["attend_text_every_n_blocks"] = int(cfg_json.get("attend_text_every_n_blocks", AH["attend_text_every_n_blocks"]))
    AH["input_embedding_dim"]        = int(cfg_json.get("input_embedding_dim", AH["input_embedding_dim"]))
    AH["backbone_embedding_dim"]     = int(cfg_json.get("backbone_embedding_dim", AH["backbone_embedding_dim"]))
    dmc = cfg_json.get("diffusion_model_cfg", {})
    AH["dit_layers"]     = int(dmc.get("num_layers", AH["dit_layers"]))
    AH["dit_heads"]      = int(dmc.get("num_attention_heads", AH["dit_heads"]))
    AH["dit_head_dim"]   = int(dmc.get("attention_head_dim", AH["dit_head_dim"]))
    AH["dit_hidden"]     = AH["dit_heads"] * AH["dit_head_dim"]
    AH["dit_interleave"] = int(bool(dmc.get("interleave_self_attention", True)))
    SHORTEST_EDGE       = int(cfg_json.get("shortest_image_edge", 256) or 256)
    CROP_FRACTION       = float(cfg_json.get("crop_fraction", 0.95) or 0.95)
    USE_RELATIVE_ACTION = bool(cfg_json.get("use_relative_action", False))
    APPLY_SINCOS_STATE  = bool(cfg_json.get("apply_sincos_state_encoding", False))

    print(f"loading sharded safetensors from {ckpt} ...")
    W = load_safetensors(ckpt)
    keys = set(W.keys())
    print(f"  {len(W)} tensors")

    check_layers(max_layer(keys, f"{VIT_ROOT}.encoder.layers."), VIT["vit_layers"], "SigLIP2 layers")
    check_layers(max_layer(keys, f"{LM_ROOT}.layers."), LM_LAYERS_USED, "Qwen3 layers")
    check_layers(max_layer(keys, f"{AHK}.model.transformer_blocks."), AH["dit_layers"], "DiT blocks")

    vocab = int(W[f"{LM_ROOT}.embed_tokens.weight"].shape[0])
    grid = VIT["image_size"] // VIT["patch_size"]
    n_patches = grid * grid

    c4 = int(W["backbone.model.mlp1.0.weight"].shape[0])
    downscale = int(round((c4 / VIT["vit_hidden"]) ** 0.5))
    n_img_tokens = (grid // downscale) * (grid // downscale)
    assert W["backbone.model.mlp1.1.weight"].shape == (QWEN3["lm_hidden"], c4), W["backbone.model.mlp1.1.weight"].shape
    assert W["backbone.model.mlp1.3.weight"].shape == (QWEN3["lm_hidden"], QWEN3["lm_hidden"])
    assert W[f"{VIT_ROOT}.embeddings.patch_embedding.weight"].shape == (VIT["vit_hidden"], 3 * VIT["patch_size"] ** 2)

    statistics_json    = read_text(ckpt / "statistics.json")
    processor_json     = read_text(ckpt / "processor_config.json")
    embodiment_id_json = read_text(ckpt / "embodiment_id.json")
    proc_kwargs = json.loads(processor_json).get("processor_kwargs", {}) if processor_json != "{}" else {}
    USE_PERCENTILES = bool(proc_kwargs.get("use_percentiles", True))
    CLIP_OUTLIERS   = bool(proc_kwargs.get("clip_outliers", True))

    print(f"resolved cfg: vit={VIT['vit_hidden']}d×{VIT['vit_layers']}L×{VIT['vit_heads']}h (Linear patch embed)  "
          f"pixel_shuffle ÷{downscale} ⇒ n_img_tok={n_img_tokens}  mlp1=LN({c4})→Linear({c4}→{QWEN3['lm_hidden']})→GELU→Linear({QWEN3['lm_hidden']}→{QWEN3['lm_hidden']})  "
          f"lm=Qwen3 {QWEN3['lm_hidden']}d×{LM_LAYERS_USED}L ({QWEN3['lm_q_heads']}q/{QWEN3['lm_kv_heads']}kv×{QWEN3['lm_head_dim']})  vocab={vocab}  "
          f"dit=AlternateVLDiT {AH['dit_layers']}L×{AH['dit_heads']}h×{AH['dit_head_dim']}(inner {AH['dit_hidden']}) attend_text_every_n={AH['attend_text_every_n_blocks']}  "
          f"in_emb={AH['input_embedding_dim']}  horizon={AH['action_horizon']} action_dim={AH['action_dim']} max_state={AH['max_state_dim']}  N_steps={AH['num_inference_timesteps']}  "
          f"embodiments={AH['max_num_embodiments']}  relative_action={USE_RELATIVE_ACTION} percentiles={USE_PERCENTILES} clip_outliers={CLIP_OUTLIERS} sincos_state={APPLY_SINCOS_STATE}  "
          f"stats={len(statistics_json)}c proc={len(processor_json)}c emb_id={embodiment_id_json.strip()}")

    writer = open_writer(out, ARCH)
    kv_u32(
        writer,
        KV,
        dict(
            **{k: VIT[k] for k in ("vit_hidden", "vit_layers", "vit_heads", "vit_inter", "image_size", "patch_size")},
            vit_num_patches=n_patches,
            n_img_tokens=n_img_tokens,
            vit_pixel_shuffle=downscale,
            mlp_connector_inner=c4,
            mlp_connector_layers=2,
            shortest_image_edge=SHORTEST_EDGE,
            lm_hidden=QWEN3["lm_hidden"],
            lm_layers_used=LM_LAYERS_USED,
            lm_q_heads=QWEN3["lm_q_heads"],
            lm_kv_heads=QWEN3["lm_kv_heads"],
            lm_head_dim=QWEN3["lm_head_dim"],
            lm_inter=QWEN3["lm_inter"],
            vocab_size=vocab,
            image_token_index=IMAGE_TOKEN_INDEX,
            **{k: AH[k] for k in (
                "backbone_embedding_dim",
                "input_embedding_dim",
                "dit_hidden",
                "dit_heads",
                "dit_head_dim",
                "dit_layers",
                "dit_interleave",
                "attend_text_every_n_blocks",
                "action_horizon",
                "action_dim",
                "max_state_dim",
                "num_inference_timesteps",
                "num_timestep_buckets",
                "max_num_embodiments",
                "max_seq_len"
            )},
            use_relative_action=int(USE_RELATIVE_ACTION),
            use_percentiles=int(USE_PERCENTILES),
            clip_outliers=int(CLIP_OUTLIERS),
            apply_sincos_state_encoding=int(APPLY_SINCOS_STATE),
        )
    )
    writer.add_float64(KV("lm_rope_theta"), float(QWEN3["lm_rope_theta"]))
    kv_f32(
        writer,
        KV,
        dict(
            lm_rms_eps=QWEN3["lm_rms_eps"],
            vit_ln_eps=VIT["vit_ln_eps"],
            ln_eps=AH["ln_eps"],
            norm_out_eps=AH["norm_out_eps"],
            vlln_eps=AH["vlln_eps"],
            connector_ln_eps=1e-5,
            crop_fraction=CROP_FRACTION
        )
    )
    writer.add_string(KV("embodiment_id_mapping"), embodiment_id_json.strip())
    writer.add_string(KV("statistics_json"), statistics_json)
    writer.add_string(KV("processor_config_json"), processor_json)

    g = W.__getitem__

    pos_embd = _resize_pos_embd(g(f"{VIT_ROOT}.embeddings.position_embedding.weight"), grid)
    write_siglip_tower(writer, g, VIT_ROOT, VIT["vit_layers"], pos_embd=pos_embd)

    add(writer, "mm.ln.weight",  g("backbone.model.mlp1.0.weight")); add(writer, "mm.ln.bias",  g("backbone.model.mlp1.0.bias"))
    add(writer, "mm.fc1.weight", g("backbone.model.mlp1.1.weight")); add(writer, "mm.fc1.bias", g("backbone.model.mlp1.1.bias"))
    add(writer, "mm.fc2.weight", g("backbone.model.mlp1.3.weight")); add(writer, "mm.fc2.bias", g("backbone.model.mlp1.3.bias"))

    write_qwen3_lm(writer, g, LM_ROOT, LM_LAYERS_USED)

    add(writer, "aex.vlln.weight", g(f"{AHK}.vlln.weight")); add(writer, "aex.vlln.bias", g(f"{AHK}.vlln.bias"))
    write_gr00t_projectors(writer, g, AHK)
    add(writer, "aex.pos_embd", g(f"{AHK}.position_embedding.weight"))

    write_gr00t_time_embed(writer, g, AHK, "aex.dit")
    write_dit_blocks(writer, g, f"{AHK}.model.transformer_blocks", "aex.dit", AH["dit_layers"])
    write_gr00t_proj_out(writer, g, AHK, "aex.dit")

    return finish(writer, out, "  - combined GGUF (Eagle-3-VL + AlternateVLDiT action head + cfg + sidecars)")

if __name__ == "__main__":
    raise SystemExit(main())
