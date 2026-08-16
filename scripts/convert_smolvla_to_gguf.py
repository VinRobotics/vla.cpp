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

from pathlib import Path

import numpy as np
from safetensors import safe_open

from gguf_blocks import (
    identity_stats,
    load_processor_stats,
    norm_eps,
    probe_siglip,
    write_decoder_blocks,
    write_siglip_tower
)
from gguf_common import (
    add,
    add_array,
    arg_parser,
    finish,
    kv_prefix,
    max_layer,
    open_writer,
    read_json,
    require,
    resolve_out
)

ARCH = "smolvla"
KV = kv_prefix(ARCH)

PFX_TXT  = "model.vlm_with_expert.vlm.model.text_model"
PFX_AEX  = "model.vlm_with_expert.lm_expert"
PFX_PROJ = "model"
PFX_VIS  = "model.vlm_with_expert.vlm.model.vision_model"
PFX_CONN = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"

SMOLLM2_500M = dict(hidden=960, n_q_heads=15, n_kv_heads=5, head_dim=64, intermediate=2560)

PROJ_SUFFIXES = [
    "state_proj.weight",
    "state_proj.bias",
    "action_in_proj.weight",
    "action_in_proj.bias",
    "action_time_mlp_in.weight",
    "action_time_mlp_in.bias",
    "action_time_mlp_out.weight",
    "action_time_mlp_out.bias",
    "action_out_proj.weight",
    "action_out_proj.bias",
]

def _probe_vision(sf, keys, cfg_json: dict) -> dict:

    if PFX_CONN not in keys:
        raise SystemExit(f"vision_model / connector not found under {PFX_VIS}")
    vcfg = cfg_json.get("vision_config") or {}
    v = probe_siglip(sf, keys, PFX_VIS, vcfg, 12)
    scale = int(cfg_json.get("scale_factor", vcfg.get("scale_factor", 4)))
    grid  = v["image_size"] // v["patch_size"]
    v["vit_pixel_shuffle"] = scale
    v["n_img_tokens"]      = (grid // scale) ** 2
    return v

def _load_stats(ckpt: Path, state_dim: int, action_dim: int) -> dict[str, np.ndarray]:

    out = identity_stats(state_dim, action_dim)
    got_state = load_processor_stats(
        ckpt,
        "policy_preprocessor.json",
        "normalizer_processor",
        "observation.state",
        state_dim
    )
    got_action = load_processor_stats(
        ckpt,
        "policy_postprocessor.json",
        "unnormalizer_processor",
        "action",
        action_dim
    )
    if got_state is not None:
        out["state_mean"], out["state_std"] = got_state
    if got_action is not None:
        out["action_mean"], out["action_std"] = got_action
    return out

def _add_kv(writer, cfg: dict) -> None:

    writer.add_uint32  (KV("hidden"),                      cfg["hidden"])
    writer.add_uint32  (KV("intermediate"),                cfg["intermediate"])
    writer.add_uint32  (KV("n_q_heads"),                   cfg["n_q_heads"])
    writer.add_uint32  (KV("n_kv_heads"),                  cfg["n_kv_heads"])
    writer.add_uint32  (KV("head_dim"),                    cfg["head_dim"])
    writer.add_uint32  (KV("n_layers"),                    cfg["n_layers"])
    writer.add_uint32  (KV("vocab_size"),                  cfg["vocab_size"])
    writer.add_uint32  (KV("expert_h"),                    cfg["expert_h"])
    writer.add_uint32  (KV("expert_inter"),                cfg["expert_inter"])
    writer.add_uint32  (KV("chunk_size"),                  cfg["chunk_size"])
    writer.add_uint32  (KV("num_steps"),                   cfg["num_steps"])
    writer.add_uint32  (KV("max_state_dim"),               cfg["max_state_dim"])
    writer.add_uint32  (KV("max_action_dim"),              cfg["max_action_dim"])
    writer.add_uint32  (KV("real_state_dim"),              cfg["real_state_dim"])
    writer.add_uint32  (KV("real_action_dim"),             cfg["real_action_dim"])
    writer.add_uint32  (KV("self_attn_every_n_layers"),    cfg["self_attn_every_n_layers"])
    writer.add_uint32  (KV("tokenizer_max_length"),        cfg["tokenizer_max_length"])
    writer.add_float64 (KV("min_period"),                  cfg["min_period"])
    writer.add_float64 (KV("max_period"),                  cfg["max_period"])
    writer.add_float64 (KV("expert_width_multiplier"),     cfg["expert_width_multiplier"])
    writer.add_float32 (KV("norm_eps"),                    cfg["norm_eps"])

    v = cfg["vit"]
    writer.add_uint32  (KV("vit_hidden"),                  v["vit_hidden"])
    writer.add_uint32  (KV("vit_layers"),                  v["vit_layers"])
    writer.add_uint32  (KV("vit_heads"),                   v["vit_heads"])
    writer.add_uint32  (KV("vit_inter"),                   v["vit_inter"])
    writer.add_uint32  (KV("image_size"),                  v["image_size"])
    writer.add_uint32  (KV("patch_size"),                  v["patch_size"])
    writer.add_uint32  (KV("vit_pixel_shuffle"),           v["vit_pixel_shuffle"])
    writer.add_uint32  (KV("n_img_tokens"),                v["n_img_tokens"])
    writer.add_float32 (KV("vit_ln_eps"),                  v["vit_ln_eps"])

def main() -> int:
    ap = arg_parser(ARCH, "HuggingFace SmolVLA checkpoint dir (with model.safetensors + config.json + policy_*processor.json)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    sf_path = ckpt / "model.safetensors"
    require(sf_path)

    cfg_json = read_json(ckpt / "config.json")

    cfg = dict(SMOLLM2_500M)
    cfg["chunk_size"]               = int(cfg_json["chunk_size"])
    cfg["num_steps"]                = int(cfg_json["num_steps"])
    cfg["max_state_dim"]            = int(cfg_json["max_state_dim"])
    cfg["max_action_dim"]           = int(cfg_json["max_action_dim"])
    cfg["min_period"]               = float(cfg_json["min_period"])
    cfg["max_period"]               = float(cfg_json["max_period"])
    cfg["self_attn_every_n_layers"] = int(cfg_json["self_attn_every_n_layers"])
    cfg["tokenizer_max_length"]     = int(cfg_json["tokenizer_max_length"])
    cfg["expert_width_multiplier"]  = float(cfg_json["expert_width_multiplier"])
    cfg["expert_h"]                 = int(round(cfg["hidden"] * cfg["expert_width_multiplier"]))
    cfg["real_state_dim"]           = int(cfg_json["input_features"]["observation.state"]["shape"][0])
    cfg["real_action_dim"]          = int(cfg_json["output_features"]["action"]["shape"][0])
    cfg["norm_eps"]                 = norm_eps(ckpt)

    print(f"opening {sf_path}")
    sf = safe_open(sf_path, framework="pt")
    keys = set(sf.keys())

    n_layers_hint = int(cfg_json.get("num_vlm_layers", 0))
    n_layers_ckpt = max_layer(keys, f"{PFX_TXT}.layers.")
    if n_layers_hint > 0:
        cfg["n_layers"] = n_layers_hint
    elif n_layers_ckpt > 0:
        cfg["n_layers"] = n_layers_ckpt
    else:
        raise SystemExit("cannot infer n_layers from checkpoint")

    expert_inter, expert_h_observed = sf.get_slice(f"{PFX_AEX}.layers.0.mlp.gate_proj.weight").get_shape()
    if expert_h_observed != cfg["expert_h"]:
        raise SystemExit(f"expert_h mismatch: cfg={cfg['expert_h']} ckpt={expert_h_observed}")
    cfg["expert_inter"] = int(expert_inter)

    vocab_observed, hidden_observed = sf.get_slice(f"{PFX_TXT}.embed_tokens.weight").get_shape()
    if hidden_observed != cfg["hidden"]:
        raise SystemExit(f"hidden mismatch: cfg={cfg['hidden']} ckpt={hidden_observed}")
    cfg["vocab_size"] = int(vocab_observed)

    print(f"resolved cfg: hidden={cfg['hidden']} n_layers={cfg['n_layers']} "
          f"expert_h={cfg['expert_h']} expert_inter={cfg['expert_inter']} "
          f"vocab={cfg['vocab_size']} chunk={cfg['chunk_size']}")

    print("loading normalizer stats...")
    stats = _load_stats(ckpt, cfg["real_state_dim"], cfg["real_action_dim"])

    cfg["vit"] = _probe_vision(sf, keys, cfg_json)
    v = cfg["vit"]
    print(f"vision: SigLIP hidden={v['vit_hidden']} layers={v['vit_layers']} "
          f"heads={v['vit_heads']} inter={v['vit_inter']} image={v['image_size']} "
          f"patch={v['patch_size']} shuffle={v['vit_pixel_shuffle']} tokens={v['n_img_tokens']}")

    writer = open_writer(out, ARCH)
    _add_kv(writer, cfg)

    add(writer, "token_embd.weight",      sf.get_tensor(f"{PFX_TXT}.embed_tokens.weight"))
    add(writer, "vlm.output_norm.weight", sf.get_tensor(f"{PFX_TXT}.norm.weight"))
    write_decoder_blocks(writer, sf.get_tensor, PFX_TXT, "vlm", cfg["n_layers"])

    add(writer, "aex.output_norm.weight", sf.get_tensor(f"{PFX_AEX}.norm.weight"))
    write_decoder_blocks(writer, sf.get_tensor, PFX_AEX, "aex", cfg["n_layers"])

    for suf in PROJ_SUFFIXES:
        add(writer, suf, sf.get_tensor(f"{PFX_PROJ}.{suf}"))

    write_siglip_tower(writer, sf.get_tensor, PFX_VIS, v["vit_layers"], f32_norms=True)
    add(writer, "mm.fc.weight", sf.get_tensor(PFX_CONN))

    for name, vec in stats.items():
        add_array(writer, name, vec)

    return finish(writer, out)

if __name__ == "__main__":
    raise SystemExit(main())
