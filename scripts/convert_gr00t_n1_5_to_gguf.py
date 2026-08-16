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
from pathlib import Path

from safetensors import safe_open

from gguf_blocks import (
    write_dit_blocks,
    write_gr00t_proj_out,
    write_gr00t_projectors,
    write_gr00t_time_embed,
    write_qwen3_lm,
    write_siglip_tower,
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
    resolve_out
)

ARCH = "gr00t_n1_5"
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
LM_LAYERS_USED = 12

AH = dict(
    backbone_embedding_dim=2048,
    input_embedding_dim=1536,
    dit_hidden=1536,
    dit_heads=32,
    dit_head_dim=48,
    dit_layers=16,
    dit_interleave=1,
    vlsa_layers=4,
    vlsa_heads=32,
    vlsa_head_dim=64,
    vlsa_inter=8192,
    num_target_vision_tokens=32,
    action_horizon=16,
    action_dim=32,
    max_state_dim=64,
    num_inference_timesteps=4,
    num_timestep_buckets=1000,
    max_num_embodiments=32,
    max_seq_len=1024,
    ln_eps=1e-5,
    norm_out_eps=1e-6,
    vlln_eps=1e-5
)
IMAGE_TOKEN_INDEX = 151669
EMBODIMENT_TAG_MAPPING = {"new_embodiment": 31, "oxe_droid": 17, "agibot_genie1": 26, "gr1": 24}

VIT_ROOT = "backbone.eagle_model.vision_model.vision_model"
LM_ROOT  = "backbone.eagle_model.language_model.model"
AHK      = "action_head"

def _read_st_vec(path: Path, key: str) -> list[float] | None:

    with safe_open(str(path), framework="pt") as f:
        if key not in f.keys():
            return None
        return f.get_tensor(key).float().reshape(-1).cpu().numpy().tolist()

def _write_lerobot_stats(ckpt: Path, out_path: Path, emb_key: str = "new_embodiment") -> None:

    pre = next(ckpt.glob("policy_preprocessor_step_*groot_pack_inputs*.safetensors"), None)
    post = next(ckpt.glob("policy_postprocessor_step_*unnormalize*.safetensors"), None)
    if pre is None or post is None:
        raise SystemExit(f"lerobot stats files not found under {ckpt} "
                         f"(need policy_preprocessor_step_*groot_pack_inputs* and "
                         f"policy_postprocessor_step_*unnormalize*)")
    s_min = _read_st_vec(pre, "observation.state.min")
    s_max = _read_st_vec(pre, "observation.state.max")
    a_min = _read_st_vec(post, "action.min")
    a_max = _read_st_vec(post, "action.max")
    if None in (s_min, s_max, a_min, a_max):
        raise SystemExit(f"missing observation.state.{{min,max}} / action.{{min,max}} in {pre.name} / {post.name}")
    blob = {
        emb_key: {
            "state":  {"min": s_min, "max": s_max},
            "action": {"min": a_min, "max": a_max}
        }
    }
    out_path.write_text(json.dumps(blob, indent=2))
    print(f"wrote {out_path}  (embodiment {emb_key!r}: state[{len(s_min)}] + action[{len(a_min)}] min/max)")

def main() -> int:
    ap = arg_parser(
        ARCH,
        "GR00T-N1.5-3B checkpoint dir. Two layouts are auto-detected: "
        "(a) NVIDIA Isaac snapshot (sharded safetensors, un-prefixed tensor names, "
        "config.json model_type=gr00t_n1_5, experiment_cfg/metadata.json); "
        "(b) lerobot finetune (single model.safetensors with `_groot_model.` prefix, "
        "config.json type=groot, policy_*processor_step_*.safetensors min/max stats)."
    )
    ap.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="[lerobot ckpt] where to write the bridge's dataset_statistics.json "
             "(default: <out dir>/dataset_statistics.json). state+action min/max are "
             "read from the lerobot processor safetensors; the eval bridge consumes it "
             "via --stats-json (the un-normalize is a host-side affine, not a ggml concern)."
    )
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    cfg_json = read_json(ckpt / "config.json")
    if not (ckpt / "model.safetensors.index.json").exists() and not (ckpt / "model.safetensors").exists():
        raise SystemExit(f"no model.safetensors[.index.json] under {ckpt}")

    model_type = str(cfg_json.get("model_type", "") or cfg_json.get("type", ""))
    is_lerobot = model_type != ARCH
    if is_lerobot:
        print(f"detected lerobot finetune layout (config type={model_type!r}); arch hparams from "
              f"built-in GR00T-N1.5 defaults, weights de-prefixed `_groot_model.`, stats from processors")

    md_path = ckpt / "experiment_cfg" / "metadata.json"
    bcfg  = cfg_json.get("backbone_cfg", {})
    ahcfg = cfg_json.get("action_head_cfg", {})
    if int(bcfg.get("select_layer", LM_LAYERS_USED)) != LM_LAYERS_USED:
        raise SystemExit(f"backbone_cfg.select_layer = {bcfg.get('select_layer')}, expected {LM_LAYERS_USED}")
    if int(ahcfg.get("action_horizon", AH["action_horizon"])) != AH["action_horizon"]:
        AH["action_horizon"] = int(ahcfg["action_horizon"])
    if int(ahcfg.get("action_dim", AH["action_dim"])) != AH["action_dim"]:
        AH["action_dim"] = int(ahcfg["action_dim"])
    AH["num_inference_timesteps"]  = int(ahcfg.get("num_inference_timesteps", AH["num_inference_timesteps"]))
    AH["num_target_vision_tokens"] = int(ahcfg.get("num_target_vision_tokens", AH["num_target_vision_tokens"]))
    AH["max_state_dim"]            = int(ahcfg.get("max_state_dim", AH["max_state_dim"]))
    dmc = ahcfg.get("diffusion_model_cfg", {})
    AH["dit_layers"]     = int(dmc.get("num_layers", AH["dit_layers"]))
    AH["dit_heads"]      = int(dmc.get("num_attention_heads", AH["dit_heads"]))
    AH["dit_head_dim"]   = int(dmc.get("attention_head_dim", AH["dit_head_dim"]))
    AH["dit_hidden"]     = AH["dit_heads"] * AH["dit_head_dim"]
    AH["dit_interleave"] = int(bool(dmc.get("interleave_self_attention", True)))
    vcfg = ahcfg.get("vl_self_attention_cfg", {})
    AH["vlsa_layers"]   = int(vcfg.get("num_layers", AH["vlsa_layers"]))
    AH["vlsa_heads"]    = int(vcfg.get("num_attention_heads", AH["vlsa_heads"]))
    AH["vlsa_head_dim"] = int(vcfg.get("attention_head_dim", AH["vlsa_head_dim"]))

    print(f"loading sharded safetensors from {ckpt} ...")
    W = load_safetensors(ckpt)
    if is_lerobot:
        pfx = "_groot_model."
        W = {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in W.items()}
    keys = set(W.keys())
    print(f"  {len(W)} tensors")

    check_layers(max_layer(keys, f"{VIT_ROOT}.encoder.layers."), VIT["vit_layers"], "SigLIP layers")
    check_layers(max_layer(keys, f"{LM_ROOT}.layers."), LM_LAYERS_USED, "Qwen3 layers")
    check_layers(max_layer(keys, f"{AHK}.model.transformer_blocks."), AH["dit_layers"], "DiT blocks")
    check_layers(
        max_layer(keys, f"{AHK}.vl_self_attention.transformer_blocks."),
        AH["vlsa_layers"],
        "vl_self_attention blocks"
    )

    emb_key = f"{LM_ROOT}.embed_tokens.weight"
    lmh_key = "backbone.eagle_model.language_model.lm_head.weight"
    tok_embd_key = emb_key if emb_key in keys else lmh_key
    if tok_embd_key not in keys:
        raise SystemExit("checkpoint has neither embed_tokens.weight nor lm_head.weight")
    vocab = int(W[tok_embd_key].shape[0])
    grid = VIT["image_size"] // VIT["patch_size"]
    n_img_tokens = grid * grid
    conn_in, conn_out = W["backbone.eagle_model.mlp1.0.weight"].shape[1], W["backbone.eagle_model.mlp1.0.weight"].shape[0]
    assert conn_in == VIT["vit_hidden"] and conn_out == QWEN3["lm_hidden"], (conn_in, conn_out)

    metadata_json = read_text(md_path)

    print(f"resolved cfg: vit={VIT['vit_hidden']}d×{VIT['vit_layers']}L×{VIT['vit_heads']}h  n_img_tok={n_img_tokens}  "
          f"mlp1=Linear({conn_in}→{conn_out})  lm=Qwen3 {QWEN3['lm_hidden']}d×{LM_LAYERS_USED}L "
          f"({QWEN3['lm_q_heads']}q/{QWEN3['lm_kv_heads']}kv×{QWEN3['lm_head_dim']}, q/k_norm)  vocab={vocab}  "
          f"dit={AH['dit_layers']}L×{AH['dit_heads']}h×{AH['dit_head_dim']}(inner {AH['dit_hidden']}) interleave={AH['dit_interleave']}  "
          f"vlsa={AH['vlsa_layers']}L×{AH['vlsa_heads']}h×{AH['vlsa_head_dim']}  in_emb={AH['input_embedding_dim']}  "
          f"horizon={AH['action_horizon']} action_dim={AH['action_dim']} max_state={AH['max_state_dim']}  N_steps={AH['num_inference_timesteps']}  "
          f"future_tok={AH['num_target_vision_tokens']}  embodiments={AH['max_num_embodiments']}  metadata.json={len(metadata_json)} chars")

    writer = open_writer(out, ARCH)
    kv_u32(
        writer,
        KV,
        dict(
            **{k: VIT[k] for k in ("vit_hidden", "vit_layers", "vit_heads", "vit_inter", "image_size", "patch_size")},
            n_img_tokens=n_img_tokens,
            vit_pixel_shuffle=0,
            mlp_connector_layers=1,
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
                "vlsa_layers",
                "vlsa_heads",
                "vlsa_head_dim",
                "vlsa_inter",
                "num_target_vision_tokens",
                "action_horizon",
                "action_dim",
                "max_state_dim",
                "num_inference_timesteps",
                "num_timestep_buckets",
                "max_num_embodiments",
                "max_seq_len"
            )},
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
            vlln_eps=AH["vlln_eps"]
        )
    )
    writer.add_string(KV("embodiment_tag_mapping"), json.dumps(EMBODIMENT_TAG_MAPPING))
    writer.add_string(KV("metadata_json"), metadata_json)

    g = W.__getitem__

    write_siglip_tower(writer, g, VIT_ROOT, VIT["vit_layers"])
    add(writer, "mm.fc.weight", g("backbone.eagle_model.mlp1.0.weight")); add(writer, "mm.fc.bias", g("backbone.eagle_model.mlp1.0.bias"))

    write_qwen3_lm(writer, g, LM_ROOT, LM_LAYERS_USED, embd_key=tok_embd_key)

    add(writer, "aex.vlln.weight", g(f"{AHK}.vlln.weight")); add(writer, "aex.vlln.bias", g(f"{AHK}.vlln.bias"))
    write_vlsa_blocks(writer, g, f"{AHK}.vl_self_attention.transformer_blocks", "aex.vlsa", AH["vlsa_layers"])
    write_gr00t_projectors(writer, g, AHK)
    add(writer, "aex.future_tokens", g(f"{AHK}.future_tokens.weight"))
    add(writer, "aex.pos_embd",      g(f"{AHK}.position_embedding.weight"))

    write_gr00t_time_embed(writer, g, AHK, "aex.dit")
    write_dit_blocks(writer, g, f"{AHK}.model.transformer_blocks", "aex.dit", AH["dit_layers"])
    write_gr00t_proj_out(writer, g, AHK, "aex.dit")

    rc = finish(writer, out, "  - combined GGUF (Eagle-2.5-VL + action head + cfg + metadata.json)")

    if is_lerobot:
        stats_out = (args.stats_out or out.parent / "dataset_statistics.json").resolve()
        _write_lerobot_stats(ckpt, stats_out)
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
