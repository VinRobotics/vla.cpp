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

ARCH = "pi0"
KV = lambda name: f"{ARCH}.{name}"

PFX_VLM  = "model.paligemma_with_expert.paligemma.model.language_model"
PFX_VLM_HEAD = "model.paligemma_with_expert.paligemma.lm_head.weight"
PFX_AEX  = "model.paligemma_with_expert.gemma_expert.model"
PFX_PROJ = "model"

# The SigLIP tower + PaliGemma projector are bundled in the ckpt GGUF and built
# in-tree by src/models/pi0.cpp (no llama.cpp clip.cpp). HF nests them under one
# of these prefixes depending on the transformers version.
PFX_VIS_CANDIDATES = [
    "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model",
    "model.paligemma_with_expert.paligemma.vision_tower.vision_model",
]
PFX_MMP_CANDIDATES = [
    "model.paligemma_with_expert.paligemma.model.multi_modal_projector",
    "model.paligemma_with_expert.paligemma.multi_modal_projector",
]

GEMMA_2B   = dict(hidden=2048, n_q_heads=8, n_kv_heads=1, head_dim=256, intermediate=16384)
GEMMA_300M = dict(expert_h=1024, expert_inter=4096)

ROPE_THETA   = 10000.0
RMS_NORM_EPS = 1e-6

def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        print(f"  warn: casting non-BF16 tensor (dtype={t.dtype}) to BF16 for storage")
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).contiguous().cpu().numpy()

def _f32_np(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.float32, t.dtype
    return t.contiguous().cpu().numpy()

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
            t = sf.get_tensor(f"{src_pfx}.layers.{i}.{src_suf}")
            _add_one_tensor(writer, f"{dst_pfx}.blk.{i}.{dst_suf}", t)

def _probe_vision(sf, keys, cfg_json) -> Optional[dict]:
    vis = next((p for p in PFX_VIS_CANDIDATES
                if f"{p}.embeddings.patch_embedding.weight" in keys), None)
    mmp = next((p for p in PFX_MMP_CANDIDATES if f"{p}.linear.weight" in keys), None)
    if vis is None or mmp is None:
        raise SystemExit("vision_tower / multi_modal_projector not found in checkpoint; "
                         f"tried {PFX_VIS_CANDIDATES} and {PFX_MMP_CANDIDATES}")
    conv = sf.get_slice(f"{vis}.embeddings.patch_embedding.weight").get_shape()  # [OC, IC, KH, KW]
    pos  = sf.get_slice(f"{vis}.embeddings.position_embedding.weight").get_shape()  # [n_tok, OC]
    n_vit = 0
    while f"{vis}.encoder.layers.{n_vit}.layer_norm1.weight" in keys:
        n_vit += 1
    vcfg = cfg_json.get("vision_config") or {}
    patch = int(conv[2]); ntok = int(pos[0]); grid = int(round(ntok ** 0.5))
    return dict(prefix=vis, mmp=mmp, vit_hidden=int(conv[0]), vit_layers=n_vit,
                vit_heads=int(vcfg.get("num_attention_heads", 16)),
                image_size=grid * patch, patch_size=patch, n_img_tokens=ntok,
                vit_ln_eps=float(vcfg.get("layer_norm_eps", 1e-6)))


def _add_vision_tensors(writer: gguf.GGUFWriter, sf, v: dict) -> None:
    vis, mmp = v["prefix"], v["mmp"]
    af32  = lambda dst, src: _add_one_tensor(writer, dst, sf.get_tensor(src).float())
    akeep = lambda dst, src: _add_one_tensor(writer, dst, sf.get_tensor(src))
    af32("vit.patch_embd.weight", f"{vis}.embeddings.patch_embedding.weight")  # 4-D conv, as-is
    af32("vit.patch_embd.bias",   f"{vis}.embeddings.patch_embedding.bias")
    af32("vit.pos_embd",          f"{vis}.embeddings.position_embedding.weight")
    for i in range(v["vit_layers"]):
        L = f"{vis}.encoder.layers.{i}."
        af32(f"vit.blk.{i}.ln1.weight", L + "layer_norm1.weight"); af32(f"vit.blk.{i}.ln1.bias", L + "layer_norm1.bias")
        af32(f"vit.blk.{i}.ln2.weight", L + "layer_norm2.weight"); af32(f"vit.blk.{i}.ln2.bias", L + "layer_norm2.bias")
        for q in ("q", "k", "v"):
            akeep(f"vit.blk.{i}.attn_{q}.weight", L + f"self_attn.{q}_proj.weight")
            af32 (f"vit.blk.{i}.attn_{q}.bias",   L + f"self_attn.{q}_proj.bias")
        akeep(f"vit.blk.{i}.attn_o.weight", L + "self_attn.out_proj.weight"); af32(f"vit.blk.{i}.attn_o.bias", L + "self_attn.out_proj.bias")
        akeep(f"vit.blk.{i}.fc1.weight", L + "mlp.fc1.weight"); af32(f"vit.blk.{i}.fc1.bias", L + "mlp.fc1.bias")
        akeep(f"vit.blk.{i}.fc2.weight", L + "mlp.fc2.weight"); af32(f"vit.blk.{i}.fc2.bias", L + "mlp.fc2.bias")
    af32("vit.post_ln.weight", f"{vis}.post_layernorm.weight"); af32("vit.post_ln.bias", f"{vis}.post_layernorm.bias")
    akeep("mm.proj.weight", f"{mmp}.linear.weight")            # PaliGemma projector (1/sqrt scale applied in pi0.cpp)
    af32 ("mm.proj.bias",   f"{mmp}.linear.bias")


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

def _load_stats(sf, ckpt_dir: Path,
                real_state_dim: int, real_action_dim: int) -> dict[str, np.ndarray]:

    out = {
        "state_mean":  np.zeros(real_state_dim,  dtype=np.float32),
        "state_std":   np.ones (real_state_dim,  dtype=np.float32),
        "action_mean": np.zeros(real_action_dim, dtype=np.float32),
        "action_std":  np.ones (real_action_dim, dtype=np.float32),
    }

    def _load_from_processor_sf(meta_json: str, registry: str, key_prefix: str,
                                mean_dst: str, std_dst: str, dim: int) -> bool:
        meta_path = ckpt_dir / meta_json
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"  stats: {meta_json} parse failed ({e}) - trying legacy")
            return False
        state_file = None
        for step in meta.get("steps", []):
            if step.get("registry_name") == registry:
                state_file = step.get("state_file")
                break
        if not state_file:
            return False
        sf_path = ckpt_dir / state_file
        if not sf_path.is_file():
            print(f"  stats: {sf_path.name} referenced by {meta_json} but missing")
            return False
        with safe_open(str(sf_path), framework="pt") as f:
            keys = set(f.keys())
            mk, sk = f"{key_prefix}.mean", f"{key_prefix}.std"
            if mk not in keys or sk not in keys:
                return False
            mean = f.get_tensor(mk).float().numpy().reshape(-1)
            std  = f.get_tensor(sk).float().numpy().reshape(-1)
        if mean.size != dim or std.size != dim:
            print(f"  stats: {mk} dim mismatch ({mean.size} vs {dim}) in {sf_path.name}")
            return False
        out[mean_dst] = mean.astype(np.float32, copy=False)
        out[std_dst]  = std .astype(np.float32, copy=False)
        print(f"  stats: loaded {mean_dst[:-5]} from {sf_path.name} ({mk}/{sk})")
        return True

    got_state  = _load_from_processor_sf("policy_preprocessor.json",  "normalizer_processor",
                                         "observation.state", "state_mean", "state_std",
                                         real_state_dim)
    got_action = _load_from_processor_sf("policy_postprocessor.json", "unnormalizer_processor",
                                         "action", "action_mean", "action_std",
                                         real_action_dim)

    keys = set(sf.keys())
    def _try_legacy(mk: str, sk: str, mdst: str, sdst: str, dim: int):
        if mk not in keys or sk not in keys:
            print(f"  stats: legacy {mk} / {sk} missing - using identity for {mdst[:-5]}")
            return
        mean = sf.get_tensor(mk).float().numpy().reshape(-1)
        std  = sf.get_tensor(sk).float().numpy().reshape(-1)
        if mean.size != dim or std.size != dim:
            print(f"  stats: legacy {mk} dim mismatch ({mean.size} vs {dim}) - using identity")
            return
        out[mdst] = mean.astype(np.float32, copy=False)
        out[sdst] = std .astype(np.float32, copy=False)
        print(f"  stats: loaded {mdst[:-5]} from model.safetensors ({mk}/{sk}) [legacy]")

    if not got_state:
        _try_legacy("normalize_inputs.buffer_observation_state.mean",
                    "normalize_inputs.buffer_observation_state.std",
                    "state_mean", "state_std", real_state_dim)
    if not got_action:
        if "unnormalize_outputs.buffer_action.mean" in keys:
            _try_legacy("unnormalize_outputs.buffer_action.mean",
                        "unnormalize_outputs.buffer_action.std",
                        "action_mean", "action_std", real_action_dim)
        else:
            _try_legacy("normalize_targets.buffer_action.mean",
                        "normalize_targets.buffer_action.std",
                        "action_mean", "action_std", real_action_dim)
    return out

def _add_kv(writer: gguf.GGUFWriter, cfg: dict) -> None:
    writer.add_string  (KV("architecture"),             ARCH)
    writer.add_string  (KV("paligemma_variant"),        cfg["paligemma_variant"])
    writer.add_string  (KV("action_expert_variant"),    cfg["action_expert_variant"])
    writer.add_uint32  (KV("hidden"),                   cfg["hidden"])
    writer.add_uint32  (KV("intermediate"),             cfg["intermediate"])
    writer.add_uint32  (KV("n_q_heads"),                cfg["n_q_heads"])
    writer.add_uint32  (KV("n_kv_heads"),               cfg["n_kv_heads"])
    writer.add_uint32  (KV("head_dim"),                 cfg["head_dim"])
    writer.add_uint32  (KV("n_layers"),                 cfg["n_layers"])
    writer.add_uint32  (KV("vocab_size"),               cfg["vocab_size"])
    writer.add_uint32  (KV("expert_h"),                 cfg["expert_h"])
    writer.add_uint32  (KV("expert_inter"),             cfg["expert_inter"])
    writer.add_uint32  (KV("chunk_size"),               cfg["chunk_size"])
    writer.add_uint32  (KV("num_steps"),                cfg["num_steps"])
    writer.add_uint32  (KV("n_action_steps"),           cfg["n_action_steps"])
    writer.add_uint32  (KV("max_state_dim"),            cfg["max_state_dim"])
    writer.add_uint32  (KV("max_action_dim"),           cfg["max_action_dim"])
    writer.add_uint32  (KV("real_state_dim"),           cfg["real_state_dim"])
    writer.add_uint32  (KV("real_action_dim"),          cfg["real_action_dim"])
    writer.add_uint32  (KV("tokenizer_max_length"),     cfg["tokenizer_max_length"])
    writer.add_float64 (KV("min_period"),               cfg["min_period"])
    writer.add_float64 (KV("max_period"),               cfg["max_period"])
    writer.add_float64 (KV("rope_theta"),               cfg["rope_theta"])
    writer.add_float32 (KV("rms_norm_eps"),             cfg["rms_norm_eps"])
    writer.add_float32 (KV("norm_eps"),                 cfg["norm_eps"])
    if "vit" in cfg:
        v = cfg["vit"]
        writer.add_uint32 (KV("vit_hidden"),   v["vit_hidden"])
        writer.add_uint32 (KV("vit_layers"),   v["vit_layers"])
        writer.add_uint32 (KV("vit_heads"),    v["vit_heads"])
        writer.add_uint32 (KV("image_size"),   v["image_size"])
        writer.add_uint32 (KV("patch_size"),   v["patch_size"])
        writer.add_uint32 (KV("n_img_tokens"), v["n_img_tokens"])
        writer.add_float32(KV("vit_ln_eps"),   v["vit_ln_eps"])

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True,
        help="lerobot π₀ checkpoint dir (model.safetensors + config.json + policy_*processor.json)")
    ap.add_argument("--out", type=Path, default=None,
        help="Output GGUF path (default: <ckpt>/pi0.gguf)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = (args.out or ckpt / "pi0.gguf").resolve()
    sf_path  = ckpt / "model.safetensors"
    cfg_path = ckpt / "config.json"
    if not sf_path.exists():
        raise SystemExit(f"missing {sf_path}")
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")

    cfg_json = json.loads(cfg_path.read_text())
    if cfg_json.get("type") != "pi0":
        raise SystemExit(f"config.json type is {cfg_json.get('type')!r}, expected 'pi0' "
                         f"(π0.5 / other variants are not handled by this converter)")

    cfg = dict(GEMMA_2B, **GEMMA_300M)
    cfg["paligemma_variant"]     = str(cfg_json.get("paligemma_variant", "gemma_2b"))
    cfg["action_expert_variant"] = str(cfg_json.get("action_expert_variant", "gemma_300m"))
    cfg["chunk_size"]            = int(cfg_json["chunk_size"])
    cfg["num_steps"]             = int(cfg_json["num_inference_steps"])
    cfg["n_action_steps"]        = int(cfg_json["n_action_steps"])
    cfg["max_state_dim"]         = int(cfg_json["max_state_dim"])
    cfg["max_action_dim"]        = int(cfg_json["max_action_dim"])
    cfg["min_period"]            = float(cfg_json["min_period"])
    cfg["max_period"]            = float(cfg_json["max_period"])
    cfg["tokenizer_max_length"]  = int(cfg_json["tokenizer_max_length"])
    cfg["real_state_dim"]        = int(cfg_json["input_features"]["observation.state"]["shape"][0])
    cfg["real_action_dim"]       = int(cfg_json["output_features"]["action"]["shape"][0])
    cfg["rope_theta"]            = ROPE_THETA
    cfg["rms_norm_eps"]          = RMS_NORM_EPS

    norm_eps = _read_norm_eps(ckpt / "policy_preprocessor.json")
    if norm_eps is None:
        norm_eps = _read_norm_eps(ckpt / "policy_postprocessor.json")
    cfg["norm_eps"] = float(norm_eps if norm_eps is not None else 1e-8)

    print(f"opening {sf_path}")
    sf = safe_open(sf_path, framework="pt")
    keys = set(sf.keys())

    def _maxlayer(pfx: str) -> int:
        m = -1
        for k in keys:
            if k.startswith(pfx):
                try:
                    m = max(m, int(k[len(pfx):].split(".", 1)[0]))
                except ValueError:
                    pass
        return m + 1

    n_layers_vlm = _maxlayer(f"{PFX_VLM}.layers.")
    n_layers_aex = _maxlayer(f"{PFX_AEX}.layers.")
    if n_layers_vlm <= 0:
        raise SystemExit("cannot find PaliGemma language-model layers in checkpoint")
    if n_layers_aex != n_layers_vlm:
        raise SystemExit(f"layer count mismatch: VLM={n_layers_vlm} expert={n_layers_aex} "
                         f"(π0 expects them equal)")
    cfg["n_layers"] = n_layers_vlm

    q0  = sf.get_slice(f"{PFX_VLM}.layers.0.self_attn.q_proj.weight").get_shape()
    kv0 = sf.get_slice(f"{PFX_VLM}.layers.0.self_attn.k_proj.weight").get_shape()
    gate0 = sf.get_slice(f"{PFX_VLM}.layers.0.mlp.gate_proj.weight").get_shape()
    if q0[1] != cfg["hidden"]:
        raise SystemExit(f"hidden mismatch: cfg={cfg['hidden']} ckpt={q0[1]}")
    if q0[0] != cfg["n_q_heads"] * cfg["head_dim"]:
        raise SystemExit(f"q_proj rows {q0[0]} != n_q_heads*head_dim {cfg['n_q_heads']*cfg['head_dim']}")
    if kv0[0] != cfg["n_kv_heads"] * cfg["head_dim"]:
        raise SystemExit(f"k_proj rows {kv0[0]} != n_kv_heads*head_dim {cfg['n_kv_heads']*cfg['head_dim']}")
    if gate0[0] != cfg["intermediate"]:
        raise SystemExit(f"intermediate mismatch: cfg={cfg['intermediate']} ckpt={gate0[0]}")

    aex_gate0 = sf.get_slice(f"{PFX_AEX}.layers.0.mlp.gate_proj.weight").get_shape()
    if aex_gate0[1] != cfg["expert_h"]:
        raise SystemExit(f"expert_h mismatch: cfg={cfg['expert_h']} ckpt={aex_gate0[1]}")
    if aex_gate0[0] != cfg["expert_inter"]:
        raise SystemExit(f"expert_inter mismatch: cfg={cfg['expert_inter']} ckpt={aex_gate0[0]}")
    aex_o0 = sf.get_slice(f"{PFX_AEX}.layers.0.self_attn.o_proj.weight").get_shape()
    if aex_o0 != [cfg["expert_h"], cfg["n_q_heads"] * cfg["head_dim"]]:
        raise SystemExit(f"expert o_proj shape {aex_o0} != [expert_h, n_q*head_dim] "
                         f"{[cfg['expert_h'], cfg['n_q_heads']*cfg['head_dim']]}")

    head_w = sf.get_slice(PFX_VLM_HEAD).get_shape()
    if head_w[1] != cfg["hidden"]:
        raise SystemExit(f"lm_head hidden mismatch: cfg={cfg['hidden']} ckpt={head_w[1]}")
    cfg["vocab_size"] = int(head_w[0])

    print(f"resolved cfg: hidden={cfg['hidden']} n_layers={cfg['n_layers']} "
          f"inter={cfg['intermediate']} heads={cfg['n_q_heads']}q/{cfg['n_kv_heads']}kv×{cfg['head_dim']} "
          f"expert_h={cfg['expert_h']} expert_inter={cfg['expert_inter']} vocab={cfg['vocab_size']} "
          f"chunk={cfg['chunk_size']} steps={cfg['num_steps']} "
          f"real_state={cfg['real_state_dim']} real_action={cfg['real_action_dim']} "
          f"norm_eps={cfg['norm_eps']:g}")

    print("loading normalizer stats...")
    stats = _load_stats(sf, ckpt, cfg["real_state_dim"], cfg["real_action_dim"])

    vit = _probe_vision(sf, keys, cfg_json)
    cfg["vit"] = vit
    print(f"vision: SigLIP hidden={vit['vit_hidden']} layers={vit['vit_layers']} "
          f"heads={vit['vit_heads']} image={vit['image_size']} patch={vit['patch_size']} "
          f"tokens={vit['n_img_tokens']} ln_eps={vit['vit_ln_eps']:g}")

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    _add_kv(writer, cfg)

    _add_one_tensor(writer, "token_embd.weight",      sf.get_tensor(PFX_VLM_HEAD))
    _add_one_tensor(writer, "vlm.output_norm.weight", sf.get_tensor(f"{PFX_VLM}.norm.weight"))
    _stream_block(writer, sf, PFX_VLM, "vlm", cfg["n_layers"])

    _add_one_tensor(writer, "aex.output_norm.weight", sf.get_tensor(f"{PFX_AEX}.norm.weight"))
    _stream_block(writer, sf, PFX_AEX, "aex", cfg["n_layers"])

    for suf in [
        "state_proj.weight",          "state_proj.bias",
        "action_in_proj.weight",      "action_in_proj.bias",
        "action_time_mlp_in.weight",  "action_time_mlp_in.bias",
        "action_time_mlp_out.weight", "action_time_mlp_out.bias",
        "action_out_proj.weight",     "action_out_proj.bias",
    ]:
        _add_one_tensor(writer, suf, sf.get_tensor(f"{PFX_PROJ}.{suf}"))

    _add_vision_tensors(writer, sf, vit)

    for k, v in stats.items():
        writer.add_tensor(k, v, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB)")
    print("self-contained: SigLIP vision tower + projector are bundled; no separate mmproj needed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
