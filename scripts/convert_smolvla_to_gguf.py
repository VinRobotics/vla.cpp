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

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

import gguf

ARCH = "smolvla"
KV = lambda name: f"{ARCH}.{name}"

PFX_TXT  = "model.vlm_with_expert.vlm.model.text_model"
PFX_AEX  = "model.vlm_with_expert.lm_expert"
PFX_PROJ = "model"

def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        print(f"  warn: casting non-BF16 tensor (dtype={t.dtype}) to BF16 for storage")
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).contiguous().cpu().numpy()

def _f32_np(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.float32, t.dtype
    return t.contiguous().cpu().numpy()

def _read_norm_eps(meta_path: Path) -> Optional[float]:
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    for step in meta.get("steps", []):
        if step.get("registry_name") in ("normalizer_processor", "unnormalizer_processor"):
            cfg = step.get("config", {})
            if "eps" in cfg:
                return float(cfg["eps"])
    return None

def _try_load_stats(model_dir: Path,
                    real_state_dim: int,
                    real_action_dim: int) -> dict[str, np.ndarray]:

    out = {
        "state_mean":  np.zeros(real_state_dim,  dtype=np.float32),
        "state_std":   np.ones (real_state_dim,  dtype=np.float32),
        "action_mean": np.zeros(real_action_dim, dtype=np.float32),
        "action_std":  np.ones (real_action_dim, dtype=np.float32),
    }

    def _load(meta_json: str, registry: str, key_prefix: str,
              mean_dst: str, std_dst: str, dim: int):
        meta_path = model_dir / meta_json
        if not meta_path.exists():
            print(f"  stats: {meta_path.name} missing - using identity for {key_prefix}")
            return
        meta = json.loads(meta_path.read_text())
        state_file = None
        for step in meta.get("steps", []):
            if step.get("registry_name") == registry:
                state_file = step.get("state_file")
                break
        if not state_file:
            print(f"  stats: no {registry} step in {meta_path.name} - using identity")
            return
        sf = model_dir / state_file
        if not sf.exists():
            print(f"  stats: state_file {sf.name} missing - using identity")
            return
        with safe_open(sf, framework="pt") as f:
            keys = set(f.keys())
            mk, sk = f"{key_prefix}.mean", f"{key_prefix}.std"
            if mk not in keys or sk not in keys:
                print(f"  stats: {sf.name} lacks {mk}/{sk} - using identity")
                return
            mean = f.get_tensor(mk).float().numpy().reshape(-1)
            std  = f.get_tensor(sk).float().numpy().reshape(-1)
        if mean.size != dim or std.size != dim:
            print(f"  stats: {sf.name} dim mismatch ({mean.size} vs expected {dim}) - using identity")
            return
        out[mean_dst] = mean.astype(np.float32, copy=False)
        out[std_dst]  = std .astype(np.float32, copy=False)
        print(f"  stats: loaded {key_prefix} from {sf.name}")

    _load("policy_preprocessor.json",  "normalizer_processor",
          "observation.state",  "state_mean",  "state_std",  real_state_dim)
    _load("policy_postprocessor.json", "unnormalizer_processor",
          "action",             "action_mean", "action_std", real_action_dim)
    return out

PFX_VIS  = "model.vlm_with_expert.vlm.model.vision_model"
PFX_CONN = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"


def _probe_vision(sf, keys, cfg_json) -> dict:
    if f"{PFX_VIS}.embeddings.patch_embedding.weight" not in keys or PFX_CONN not in keys:
        raise SystemExit(f"vision_model / connector not found under {PFX_VIS}")
    conv = sf.get_slice(f"{PFX_VIS}.embeddings.patch_embedding.weight").get_shape()  # [OC, IC, KH, KW]
    pos  = sf.get_slice(f"{PFX_VIS}.embeddings.position_embedding.weight").get_shape()  # [n_patches, OC]
    fc1  = sf.get_slice(f"{PFX_VIS}.encoder.layers.0.mlp.fc1.weight").get_shape()  # [inter, OC]
    n_vit = 0
    while f"{PFX_VIS}.encoder.layers.{n_vit}.layer_norm1.weight" in keys:
        n_vit += 1
    vcfg = cfg_json.get("vision_config") or {}
    scale = int(cfg_json.get("scale_factor", vcfg.get("scale_factor", 4)))
    patch = int(conv[2]); grid = int(round(int(pos[0]) ** 0.5))
    return dict(vit_hidden=int(conv[0]), vit_layers=n_vit,
                vit_heads=int(vcfg.get("num_attention_heads", 12)), vit_inter=int(fc1[0]),
                image_size=grid * patch, patch_size=patch, vit_pixel_shuffle=scale,
                n_img_tokens=(grid // scale) ** 2,
                vit_ln_eps=float(vcfg.get("layer_norm_eps", 1e-6)))


def _add_vision_tensors(writer: gguf.GGUFWriter, sf) -> None:
    af32  = lambda dst, src: _add_one_tensor(writer, dst, sf.get_tensor(src).float())
    akeep = lambda dst, src: _add_one_tensor(writer, dst, sf.get_tensor(src))
    af32("vit.patch_embd.weight", f"{PFX_VIS}.embeddings.patch_embedding.weight")  # 4-D conv, as-is
    af32("vit.patch_embd.bias",   f"{PFX_VIS}.embeddings.patch_embedding.bias")
    af32("vit.pos_embd",          f"{PFX_VIS}.embeddings.position_embedding.weight")
    i = 0
    while True:
        L = f"{PFX_VIS}.encoder.layers.{i}."
        try:
            sf.get_slice(L + "layer_norm1.weight")
        except Exception:
            break
        af32(f"vit.blk.{i}.ln1.weight", L + "layer_norm1.weight"); af32(f"vit.blk.{i}.ln1.bias", L + "layer_norm1.bias")
        af32(f"vit.blk.{i}.ln2.weight", L + "layer_norm2.weight"); af32(f"vit.blk.{i}.ln2.bias", L + "layer_norm2.bias")
        for q in ("q", "k", "v"):
            akeep(f"vit.blk.{i}.attn_{q}.weight", L + f"self_attn.{q}_proj.weight")
            af32 (f"vit.blk.{i}.attn_{q}.bias",   L + f"self_attn.{q}_proj.bias")
        akeep(f"vit.blk.{i}.attn_o.weight", L + "self_attn.out_proj.weight"); af32(f"vit.blk.{i}.attn_o.bias", L + "self_attn.out_proj.bias")
        akeep(f"vit.blk.{i}.fc1.weight", L + "mlp.fc1.weight"); af32(f"vit.blk.{i}.fc1.bias", L + "mlp.fc1.bias")
        akeep(f"vit.blk.{i}.fc2.weight", L + "mlp.fc2.weight"); af32(f"vit.blk.{i}.fc2.bias", L + "mlp.fc2.bias")
        i += 1
    af32("vit.post_ln.weight", f"{PFX_VIS}.post_layernorm.weight"); af32("vit.post_ln.bias", f"{PFX_VIS}.post_layernorm.bias")
    akeep("mm.fc.weight", PFX_CONN)   # single bias-free pixel-shuffle connector linear


def _add_kv(writer: gguf.GGUFWriter, cfg: dict) -> None:

    writer.add_string  (KV("architecture"),                ARCH)
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
    if "vit" in cfg:
        v = cfg["vit"]
        writer.add_uint32 (KV("vit_hidden"),        v["vit_hidden"])
        writer.add_uint32 (KV("vit_layers"),        v["vit_layers"])
        writer.add_uint32 (KV("vit_heads"),         v["vit_heads"])
        writer.add_uint32 (KV("vit_inter"),         v["vit_inter"])
        writer.add_uint32 (KV("image_size"),        v["image_size"])
        writer.add_uint32 (KV("patch_size"),        v["patch_size"])
        writer.add_uint32 (KV("vit_pixel_shuffle"), v["vit_pixel_shuffle"])
        writer.add_uint32 (KV("n_img_tokens"),      v["n_img_tokens"])
        writer.add_float32(KV("vit_ln_eps"),        v["vit_ln_eps"])

def _add_one_tensor(writer: gguf.GGUFWriter, dst_name: str, t: torch.Tensor) -> None:

    if t.dtype == torch.float32:
        writer.add_tensor(dst_name, _f32_np(t),
                          raw_dtype=gguf.GGMLQuantizationType.F32)
    elif t.dtype == torch.bfloat16:
        writer.add_tensor(dst_name, _bf16_to_u16_bytes(t),
                          raw_shape=list(t.shape),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        raise NotImplementedError(f"unsupported dtype {t.dtype} for {dst_name}")

def _stream_block(writer: gguf.GGUFWriter, sf, src_pfx: str, dst_pfx: str, n_layers: int) -> None:

    suffix_map = [
        ("input_layernorm.weight",          "attn_norm.weight"),
        ("self_attn.q_proj.weight",         "attn_q.weight"),
        ("self_attn.k_proj.weight",         "attn_k.weight"),
        ("self_attn.v_proj.weight",         "attn_v.weight"),
        ("self_attn.o_proj.weight",         "attn_o.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("mlp.gate_proj.weight",            "ffn_gate.weight"),
        ("mlp.up_proj.weight",              "ffn_up.weight"),
        ("mlp.down_proj.weight",            "ffn_down.weight"),
    ]
    for i in range(n_layers):
        for src_suf, dst_suf in suffix_map:
            src_name = f"{src_pfx}.layers.{i}.{src_suf}"
            dst_name = f"{dst_pfx}.blk.{i}.{dst_suf}"
            t = sf.get_tensor(src_name)
            _add_one_tensor(writer, dst_name, t)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True,
        help="HuggingFace SmolVLA checkpoint dir (with model.safetensors + config.json + policy_*processor.json)")
    ap.add_argument("--out", type=Path, default=None,
        help="Output GGUF path (default: <ckpt>/smolvla.gguf)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = (args.out or ckpt / "smolvla.gguf").resolve()
    sf_path  = ckpt / "model.safetensors"
    cfg_path = ckpt / "config.json"
    if not sf_path.exists():
        raise SystemExit(f"missing {sf_path}")
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")

    cfg_json = json.loads(cfg_path.read_text())

    SMOLLM2_500M = dict(hidden=960, n_q_heads=15, n_kv_heads=5, head_dim=64, intermediate=2560)

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

    norm_eps = _read_norm_eps(ckpt / "policy_preprocessor.json")
    if norm_eps is None:
        norm_eps = _read_norm_eps(ckpt / "policy_postprocessor.json")
    cfg["norm_eps"] = float(norm_eps if norm_eps is not None else 1e-8)

    print(f"opening {sf_path}")
    sf = safe_open(sf_path, framework="pt")

    layer_re_pfx = f"{PFX_TXT}.layers."
    max_layer = -1
    for k in sf.keys():
        if k.startswith(layer_re_pfx):
            try:
                idx = int(k[len(layer_re_pfx):].split(".", 1)[0])
                max_layer = max(max_layer, idx)
            except ValueError:
                pass
    cfg_n_layers_hint = int(cfg_json.get("num_vlm_layers", 0))
    if cfg_n_layers_hint > 0:
        cfg["n_layers"] = cfg_n_layers_hint
    elif max_layer >= 0:
        cfg["n_layers"] = max_layer + 1
    else:
        raise SystemExit("cannot infer n_layers from checkpoint")

    aex_gate = sf.get_slice(f"{PFX_AEX}.layers.0.mlp.gate_proj.weight")
    expert_inter, expert_h_observed = aex_gate.get_shape()
    if expert_h_observed != cfg["expert_h"]:
        raise SystemExit(f"expert_h mismatch: cfg={cfg['expert_h']} ckpt={expert_h_observed}")
    cfg["expert_inter"] = int(expert_inter)

    embed_slice = sf.get_slice(f"{PFX_TXT}.embed_tokens.weight")
    vocab_observed, hidden_observed = embed_slice.get_shape()
    if hidden_observed != cfg["hidden"]:
        raise SystemExit(f"hidden mismatch: cfg={cfg['hidden']} ckpt={hidden_observed}")
    cfg["vocab_size"] = int(vocab_observed)

    print(f"resolved cfg: hidden={cfg['hidden']} n_layers={cfg['n_layers']} "
          f"expert_h={cfg['expert_h']} expert_inter={cfg['expert_inter']} "
          f"vocab={cfg['vocab_size']} chunk={cfg['chunk_size']}")

    print("loading normalizer stats...")
    stats = _try_load_stats(ckpt, cfg["real_state_dim"], cfg["real_action_dim"])

    cfg["vit"] = _probe_vision(sf, set(sf.keys()), cfg_json)
    print(f"vision: SigLIP hidden={cfg['vit']['vit_hidden']} layers={cfg['vit']['vit_layers']} "
          f"heads={cfg['vit']['vit_heads']} inter={cfg['vit']['vit_inter']} image={cfg['vit']['image_size']} "
          f"patch={cfg['vit']['patch_size']} shuffle={cfg['vit']['vit_pixel_shuffle']} tokens={cfg['vit']['n_img_tokens']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    _add_kv(writer, cfg)

    _add_one_tensor(writer, "token_embd.weight",
                    sf.get_tensor(f"{PFX_TXT}.embed_tokens.weight"))
    _add_one_tensor(writer, "vlm.output_norm.weight",
                    sf.get_tensor(f"{PFX_TXT}.norm.weight"))

    _stream_block(writer, sf, PFX_TXT, "vlm", cfg["n_layers"])

    _add_one_tensor(writer, "aex.output_norm.weight",
                    sf.get_tensor(f"{PFX_AEX}.norm.weight"))
    _stream_block(writer, sf, PFX_AEX, "aex", cfg["n_layers"])

    for src_suf, dst_name in [
        ("state_proj.weight",          "state_proj.weight"),
        ("state_proj.bias",            "state_proj.bias"),
        ("action_in_proj.weight",      "action_in_proj.weight"),
        ("action_in_proj.bias",        "action_in_proj.bias"),
        ("action_time_mlp_in.weight",  "action_time_mlp_in.weight"),
        ("action_time_mlp_in.bias",    "action_time_mlp_in.bias"),
        ("action_time_mlp_out.weight", "action_time_mlp_out.weight"),
        ("action_time_mlp_out.bias",   "action_time_mlp_out.bias"),
        ("action_out_proj.weight",     "action_out_proj.weight"),
        ("action_out_proj.bias",       "action_out_proj.bias"),
    ]:
        _add_one_tensor(writer, dst_name, sf.get_tensor(f"{PFX_PROJ}.{src_suf}"))

    _add_vision_tensors(writer, sf)

    for k, v in stats.items():
        writer.add_tensor(k, v, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
