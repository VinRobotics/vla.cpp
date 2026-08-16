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

from gguf_blocks import write_dit_blocks, write_qwen3_lm, write_qwen3vl_vit
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
    resolve_out
)

ARCH = "vla_jepa"
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
LM_LAYERS = 28

AH = dict(
    dit_hidden=768,
    dit_heads=12,
    dit_head_dim=64,
    dit_layers=16,
    cross_dim=2048,
    output_dim=1024,
    action_dim=7,
    state_dim=8,
    action_horizon=7,
    num_future_tokens=32,
    time_proj_dim=256,
    num_inference_timesteps=4,
    num_timestep_buckets=1000,
    ln_eps=1e-5,
    norm_out_eps=1e-6
)

IMAGE_TOKEN_INDEX = 151655
EMBODIED_ACTION_TOKEN_ID = 151697
ACTION_TOKEN_ID_0 = 151669

VIT_ROOT = "model.qwen.model.model.visual"
LM_ROOT  = "model.qwen.model.model.language_model"
AHK      = "model.action_model"

AH_PROJECTORS = [
    ("action_encoder.layer1", "ah.act_enc.l1"),
    ("action_encoder.layer2", "ah.act_enc.l2"),
    ("action_encoder.layer3", "ah.act_enc.l3"),
    ("state_encoder.layer1",  "ah.state_enc.l1"),
    ("state_encoder.layer2",  "ah.state_enc.l2"),
    ("action_decoder.layer1", "ah.act_dec.l1"),
    ("action_decoder.layer2", "ah.act_dec.l2"),
    ("model.timestep_encoder.timestep_embedder.linear_1", "ah.time_emb.l1"),
    ("model.timestep_encoder.timestep_embedder.linear_2", "ah.time_emb.l2"),
    ("model.proj_out_1", "ah.proj_out1"),
    ("model.proj_out_2", "ah.proj_out2"),
]

def main() -> int:
    ap = arg_parser(ARCH, "VLA-JEPA-LIBERO checkpoint dir")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    cfg_json = read_json(ckpt / "config.json")
    if str(cfg_json.get("type", "")) != ARCH:
        raise SystemExit(f"config.json type is {cfg_json.get('type')!r}, expected 'vla_jepa'")

    AH["action_dim"]              = int(cfg_json.get("action_dim", AH["action_dim"]))
    AH["state_dim"]               = int(cfg_json.get("state_dim", AH["state_dim"]))
    AH["action_horizon"]          = int(cfg_json.get("chunk_size", AH["action_horizon"]))
    AH["num_future_tokens"]       = int(cfg_json.get("num_embodied_action_tokens_per_instruction", AH["num_future_tokens"]))
    AH["num_inference_timesteps"] = int(cfg_json.get("num_inference_timesteps", AH["num_inference_timesteps"]))
    AH["num_timestep_buckets"]    = int(cfg_json.get("action_num_timestep_buckets", AH["num_timestep_buckets"]))
    AH["output_dim"]              = int(cfg_json.get("action_hidden_size", AH["output_dim"]))
    AH["dit_layers"]              = int(cfg_json.get("action_num_layers", AH["dit_layers"]))

    print(f"loading safetensors from {ckpt} (dropping world model) ...")
    W = load_safetensors(ckpt, keep=("model.qwen.", "model.action_model."))
    keys = set(W.keys())
    print(f"  {len(W)} kept tensors")

    check_layers(max_layer(keys, f"{VIT_ROOT}.blocks."), VIT["vit_layers"], "ViT layers")
    check_layers(max_layer(keys, f"{LM_ROOT}.layers."), LM_LAYERS, "LM layers")
    check_layers(max_layer(keys, f"{AHK}.model.transformer_blocks."), AH["dit_layers"], "DiT blocks")
    check_layers(max_layer(keys, f"{VIT_ROOT}.deepstack_merger_list."), len(DEEPSTACK_IDXS), "deepstack mergers")

    vocab = int(W[f"{LM_ROOT}.embed_tokens.weight"].shape[0])
    pe_w = W[f"{VIT_ROOT}.patch_embed.proj.weight"]
    assert pe_w.shape == (VIT["vit_hidden"], 3, VIT["temporal_patch_size"], VIT["patch_size"], VIT["patch_size"]), pe_w.shape
    patch_flat = 3 * VIT["temporal_patch_size"] * VIT["patch_size"] ** 2
    c_merged = VIT["vit_hidden"] * VIT["spatial_merge_size"] ** 2
    assert W[f"{VIT_ROOT}.merger.norm.weight"].shape == (VIT["vit_hidden"],), W[f"{VIT_ROOT}.merger.norm.weight"].shape
    assert W[f"{VIT_ROOT}.deepstack_merger_list.0.norm.weight"].shape == (c_merged,)
    assert W[f"{AHK}.model.transformer_blocks.0.attn1.to_k.weight"].shape == (AH["dit_hidden"], AH["cross_dim"])
    assert W[f"{AHK}.model.transformer_blocks.1.attn1.to_k.weight"].shape == (AH["dit_hidden"], AH["dit_hidden"])

    print(f"resolved: vit=Qwen3-VL {VIT['vit_hidden']}d×{VIT['vit_layers']}L (deepstack@{DEEPSTACK_IDXS}, merge÷{VIT['spatial_merge_size']})  "
          f"lm=Qwen3-VL {QWEN3['lm_hidden']}d×{LM_LAYERS}L ({QWEN3['lm_q_heads']}q/{QWEN3['lm_kv_heads']}kv×{QWEN3['lm_head_dim']}, θ={QWEN3['lm_rope_theta']:g})  vocab={vocab}  "
          f"dit-B {AH['dit_layers']}L×{AH['dit_heads']}h×{AH['dit_head_dim']}(inner {AH['dit_hidden']}, cross {AH['cross_dim']}, out {AH['output_dim']})  "
          f"horizon={AH['action_horizon']} action_dim={AH['action_dim']} state_dim={AH['state_dim']} future={AH['num_future_tokens']} N_steps={AH['num_inference_timesteps']}  "
          f"img_tok={IMAGE_TOKEN_INDEX} emb_tok={EMBODIED_ACTION_TOKEN_ID}")

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
            image_target_size=256,
            deepstack_idx_0=DEEPSTACK_IDXS[0],
            deepstack_idx_1=DEEPSTACK_IDXS[1],
            deepstack_idx_2=DEEPSTACK_IDXS[2],
            lm_hidden=QWEN3["lm_hidden"],
            lm_layers=LM_LAYERS,
            lm_q_heads=QWEN3["lm_q_heads"],
            lm_kv_heads=QWEN3["lm_kv_heads"],
            lm_head_dim=QWEN3["lm_head_dim"],
            lm_inter=QWEN3["lm_inter"],
            vocab_size=vocab,
            image_token_index=IMAGE_TOKEN_INDEX,
            embodied_action_token_id=EMBODIED_ACTION_TOKEN_ID,
            action_token_id_0=ACTION_TOKEN_ID_0,
            dit_hidden=AH["dit_hidden"],
            dit_heads=AH["dit_heads"],
            dit_head_dim=AH["dit_head_dim"],
            dit_layers=AH["dit_layers"],
            cross_dim=AH["cross_dim"],
            output_dim=AH["output_dim"],
            time_proj_dim=AH["time_proj_dim"],
            action_dim=AH["action_dim"],
            state_dim=AH["state_dim"],
            action_horizon=AH["action_horizon"],
            num_future_tokens=AH["num_future_tokens"],
            num_inference_timesteps=AH["num_inference_timesteps"],
            num_timestep_buckets=AH["num_timestep_buckets"],
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
            connector_ln_eps=1e-6,
            dit_ln_eps=AH["ln_eps"],
            dit_norm_out_eps=AH["norm_out_eps"]
        )
    )

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
    write_qwen3_lm(writer, g, LM_ROOT, LM_LAYERS)

    for src, dst in AH_PROJECTORS:
        add(writer, f"{dst}.weight", g(f"{AHK}.{src}.weight")); add(writer, f"{dst}.bias", g(f"{AHK}.{src}.bias"))
    add(writer, "ah.future_tokens", g(f"{AHK}.future_tokens.weight"))
    add(writer, "ah.pos_embd",      g(f"{AHK}.position_embedding.weight"))
    write_dit_blocks(writer, g, f"{AHK}.model.transformer_blocks", "ah.dit", AH["dit_layers"])

    return finish(writer, out, "  - combined GGUF (Qwen3-VL backbone + deepstack + DiT-B action head + cfg)")

if __name__ == "__main__":
    raise SystemExit(main())
