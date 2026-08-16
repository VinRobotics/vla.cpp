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

from gguf_blocks import (
    write_dit_blocks,
    write_gr00t_proj_out,
    write_gr00t_projectors,
    write_gr00t_time_embed,
    write_qwen3_lm,
    write_qwen3vl_vit,
    write_vlsa_blocks
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

ARCH = "gr00t_n1_7"
KV = kv_prefix(ARCH)

VIT = dict(
    vit_hidden=1024,
    vit_layers=24,
    vit_heads=16,
    vit_inter=4096,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    vit_num_position_embeddings=2304,
    vit_ln_eps=1e-6
)
DEEPSTACK_IDXS = [5, 11, 17]

QWEN3 = dict(
    lm_hidden=2048,
    lm_q_heads=16,
    lm_kv_heads=8,
    lm_head_dim=128,
    lm_inter=6144,
    lm_rope_theta=5000000.0,
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
    vlsa_layers=4,
    vlsa_heads=32,
    vlsa_head_dim=64,
    action_horizon=40,
    action_dim=132,
    max_state_dim=132,
    num_inference_timesteps=4,
    num_timestep_buckets=1000,
    max_num_embodiments=32,
    max_seq_len=1024,
    ln_eps=1e-5,
    norm_out_eps=1e-6,
    vlln_eps=1e-5,
    vlsa_ln_eps=1e-5
)
IMAGE_TOKEN_INDEX = 151655

VIT_ROOT = "backbone.model.model.visual"
LM_ROOT  = "backbone.model.model.language_model"
AHK      = "action_head"

def main() -> int:
    ap = arg_parser(ARCH, "GR00T-N1.7-3B snapshot dir")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    require(ckpt / "model.safetensors.index.json")
    cfg_json = read_json(ckpt / "config.json")
    if str(cfg_json.get("model_type", "")) != "Gr00tN1d7":
        raise SystemExit(f"config.json model_type is {cfg_json.get('model_type')!r}, expected 'Gr00tN1d7'")
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
    vsac = cfg_json.get("vl_self_attention_cfg", {})
    AH["vlsa_layers"]   = int(vsac.get("num_layers", AH["vlsa_layers"]))
    AH["vlsa_heads"]    = int(vsac.get("num_attention_heads", AH["vlsa_heads"]))
    AH["vlsa_head_dim"] = int(vsac.get("attention_head_dim", AH["vlsa_head_dim"]))
    SHORTEST_EDGE       = int(cfg_json.get("shortest_image_edge", 256) or 256)
    CROP_FRACTION       = float(cfg_json.get("crop_fraction", 0.95) or 0.95)
    ICS                 = cfg_json.get("image_crop_size", [230, 230]) or [230, 230]
    ITS                 = cfg_json.get("image_target_size", [256, 256]) or [256, 256]
    USE_RELATIVE_ACTION = bool(cfg_json.get("use_relative_action", False))
    APPLY_SINCOS_STATE  = bool(cfg_json.get("apply_sincos_state_encoding", False))

    print(f"loading sharded safetensors from {ckpt} ...")
    W = load_safetensors(ckpt)
    keys = set(W.keys())
    print(f"  {len(W)} tensors")

    check_layers(max_layer(keys, f"{VIT_ROOT}.blocks."), VIT["vit_layers"], "Qwen3-VL ViT layers")
    check_layers(max_layer(keys, f"{LM_ROOT}.layers."), LM_LAYERS_USED, "Qwen3-VL text layers")
    check_layers(max_layer(keys, f"{AHK}.model.transformer_blocks."), AH["dit_layers"], "DiT blocks")
    check_layers(
        max_layer(keys, f"{AHK}.vl_self_attention.transformer_blocks."),
        AH["vlsa_layers"],
        "vl_self_attention blocks"
    )
    check_layers(max_layer(keys, f"{VIT_ROOT}.deepstack_merger_list."), len(DEEPSTACK_IDXS), "deepstack mergers")

    vocab = int(W[f"{LM_ROOT}.embed_tokens.weight"].shape[0])
    pe_w = W[f"{VIT_ROOT}.patch_embed.proj.weight"]
    assert pe_w.shape == (VIT["vit_hidden"], 3, VIT["temporal_patch_size"], VIT["patch_size"], VIT["patch_size"]), pe_w.shape
    patch_flat = 3 * VIT["temporal_patch_size"] * VIT["patch_size"] ** 2
    c_merged = VIT["vit_hidden"] * VIT["spatial_merge_size"] ** 2
    assert W[f"{VIT_ROOT}.merger.linear_fc1.weight"].shape == (c_merged, c_merged)
    assert W[f"{VIT_ROOT}.merger.linear_fc2.weight"].shape == (AH["backbone_embedding_dim"], c_merged)
    assert W[f"{VIT_ROOT}.merger.norm.weight"].shape == (VIT["vit_hidden"],), W[f"{VIT_ROOT}.merger.norm.weight"].shape
    assert W[f"{VIT_ROOT}.deepstack_merger_list.0.norm.weight"].shape == (c_merged,)
    vlsa_ff_inner = 4 * AH["backbone_embedding_dim"]
    assert W[f"{AHK}.vl_self_attention.transformer_blocks.0.ff.net.0.proj.weight"].shape == (vlsa_ff_inner, AH["backbone_embedding_dim"])

    statistics_json    = read_text(ckpt / "statistics.json")
    processor_json     = read_text(ckpt / "processor_config.json")
    embodiment_id_json = read_text(ckpt / "embodiment_id.json")
    proc_kwargs = json.loads(processor_json).get("processor_kwargs", {}) if processor_json != "{}" else {}
    USE_PERCENTILES = bool(proc_kwargs.get("use_percentiles", True))
    CLIP_OUTLIERS   = bool(proc_kwargs.get("clip_outliers", True))

    print(f"resolved cfg: vit=Qwen3-VL {VIT['vit_hidden']}d×{VIT['vit_layers']}L×{VIT['vit_heads']}h (Conv3d patch {VIT['patch_size']}², temporal {VIT['temporal_patch_size']}, "
          f"learned pos {VIT['vit_num_position_embeddings']}=48² + 2D rope; deepstack@{DEEPSTACK_IDXS}; merger LN={VIT['vit_hidden']} pre-merge / deepstack LN={c_merged} post-merge ⇒ "
          f"merge÷{VIT['spatial_merge_size']})  lm=Qwen3-VL {QWEN3['lm_hidden']}d×{LM_LAYERS_USED}L ({QWEN3['lm_q_heads']}q/{QWEN3['lm_kv_heads']}kv×{QWEN3['lm_head_dim']}, θ={QWEN3['lm_rope_theta']:g})  "
          f"vocab={vocab} img_tok={IMAGE_TOKEN_INDEX}  vlsa={AH['vlsa_layers']}L×{AH['vlsa_heads']}h×{AH['vlsa_head_dim']} ff{vlsa_ff_inner}  "
          f"dit=AlternateVLDiT {AH['dit_layers']}L×{AH['dit_heads']}h×{AH['dit_head_dim']}(inner {AH['dit_hidden']}) attend_text_every_n={AH['attend_text_every_n_blocks']}  "
          f"in_emb={AH['input_embedding_dim']}  horizon={AH['action_horizon']} action_dim={AH['action_dim']} max_state={AH['max_state_dim']}  N_steps={AH['num_inference_timesteps']}  "
          f"embodiments={AH['max_num_embodiments']}  relative={USE_RELATIVE_ACTION} percentiles={USE_PERCENTILES} clip={CLIP_OUTLIERS} sincos={APPLY_SINCOS_STATE}  "
          f"img: shortest_edge={SHORTEST_EDGE} crop_frac={CROP_FRACTION} crop_size={ICS} target_size={ITS}  stats={len(statistics_json)}c proc={len(processor_json)}c emb_id={embodiment_id_json.strip()}")

    writer = open_writer(out, ARCH)
    kv_u32(
        writer,
        KV,
        dict(
            vit_hidden=VIT["vit_hidden"],
            vit_layers=VIT["vit_layers"],
            vit_heads=VIT["vit_heads"],
            vit_inter=VIT["vit_inter"],
            patch_size=VIT["patch_size"],
            temporal_patch_size=VIT["temporal_patch_size"],
            spatial_merge_size=VIT["spatial_merge_size"],
            vit_num_position_embeddings=VIT["vit_num_position_embeddings"],
            vit_patch_flat=patch_flat,
            vit_merged_dim=c_merged,
            n_img_tokens_per_view=64,
            shortest_image_edge=SHORTEST_EDGE,
            image_crop_size=int(ICS[0]),
            image_target_size=int(ITS[0]),
            deepstack_idx_0=DEEPSTACK_IDXS[0],
            deepstack_idx_1=DEEPSTACK_IDXS[1],
            deepstack_idx_2=DEEPSTACK_IDXS[2],
            lm_hidden=QWEN3["lm_hidden"],
            lm_layers_used=LM_LAYERS_USED,
            lm_q_heads=QWEN3["lm_q_heads"],
            lm_kv_heads=QWEN3["lm_kv_heads"],
            lm_head_dim=QWEN3["lm_head_dim"],
            lm_inter=QWEN3["lm_inter"],
            vocab_size=vocab,
            image_token_index=IMAGE_TOKEN_INDEX,
            vlsa_layers=AH["vlsa_layers"],
            vlsa_heads=AH["vlsa_heads"],
            vlsa_head_dim=AH["vlsa_head_dim"],
            vlsa_ff_inner=vlsa_ff_inner,
            backbone_embedding_dim=AH["backbone_embedding_dim"],
            input_embedding_dim=AH["input_embedding_dim"],
            dit_hidden=AH["dit_hidden"],
            dit_heads=AH["dit_heads"],
            dit_head_dim=AH["dit_head_dim"],
            dit_layers=AH["dit_layers"],
            dit_interleave=AH["dit_interleave"],
            attend_text_every_n_blocks=AH["attend_text_every_n_blocks"],
            action_horizon=AH["action_horizon"],
            action_dim=AH["action_dim"],
            max_state_dim=AH["max_state_dim"],
            num_inference_timesteps=AH["num_inference_timesteps"],
            num_timestep_buckets=AH["num_timestep_buckets"],
            max_num_embodiments=AH["max_num_embodiments"],
            max_seq_len=AH["max_seq_len"],
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
            vit_rope_theta=10000.0,
            ln_eps=AH["ln_eps"],
            norm_out_eps=AH["norm_out_eps"],
            vlln_eps=AH["vlln_eps"],
            vlsa_ln_eps=AH["vlsa_ln_eps"],
            connector_ln_eps=1e-6,
            crop_fraction=CROP_FRACTION
        )
    )
    writer.add_string(KV("embodiment_id_mapping"), embodiment_id_json.strip())
    writer.add_string(KV("statistics_json"), statistics_json)
    writer.add_string(KV("processor_config_json"), processor_json)

    g = W.__getitem__

    write_qwen3vl_vit(
        writer,
        g,
        VIT_ROOT,
        VIT["vit_layers"],
        len(DEEPSTACK_IDXS),
        VIT["vit_hidden"],
        patch_flat
    )
    write_qwen3_lm(writer, g, LM_ROOT, LM_LAYERS_USED)

    add(writer, "aex.vlln.weight", g(f"{AHK}.vlln.weight")); add(writer, "aex.vlln.bias", g(f"{AHK}.vlln.bias"))
    write_vlsa_blocks(writer, g, f"{AHK}.vl_self_attention.transformer_blocks", "aex.vlsa", AH["vlsa_layers"])
    write_gr00t_projectors(writer, g, AHK)
    add(writer, "aex.pos_embd", g(f"{AHK}.position_embedding.weight"))

    write_gr00t_time_embed(writer, g, AHK, "aex.dit")
    write_dit_blocks(writer, g, f"{AHK}.model.transformer_blocks", "aex.dit", AH["dit_layers"])
    write_gr00t_proj_out(writer, g, AHK, "aex.dit")

    return finish(
        writer,
        out,
        "  - combined GGUF (Qwen3-VL backbone + deepstack + vl_self_attention "
        "+ AlternateVLDiT action head + cfg + sidecars)"
    )

if __name__ == "__main__":
    raise SystemExit(main())
