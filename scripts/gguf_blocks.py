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

"""Tensor groups and config blocks more than one architecture emits: the SigLIP
and Qwen3-VL towers, the Qwen3 / Gemma decoder stacks, the DiT and
VL-self-attention blocks, the Prismatic dual tower, and the lerobot normalizer
sidecars.

Every writer takes a getter `g(name) -> torch.Tensor`, so a checkpoint held in a
dict and one streamed from a safetensors handle look the same from here. The
emitted name and order are what the src/models/ loaders expect: changing either
changes the GGUF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from gguf_common import add, add_bf16, add_f32

def probe_siglip(sf, keys, root: str, vcfg: dict, default_heads: int) -> dict:

    if f"{root}.embeddings.patch_embedding.weight" not in keys:
        raise SystemExit(f"SigLIP vision tower not found under {root}")
    conv = sf.get_slice(f"{root}.embeddings.patch_embedding.weight").get_shape()
    pos  = sf.get_slice(f"{root}.embeddings.position_embedding.weight").get_shape()
    fc1  = sf.get_slice(f"{root}.encoder.layers.0.mlp.fc1.weight").get_shape()
    n_vit = 0
    while f"{root}.encoder.layers.{n_vit}.layer_norm1.weight" in keys:
        n_vit += 1

    patch = int(conv[2])
    ntok  = int(pos[0])
    grid  = int(round(ntok ** 0.5))
    return dict(
        vit_hidden=int(conv[0]),
        vit_layers=n_vit,
        vit_heads=int(vcfg.get("num_attention_heads", default_heads)),
        vit_inter=int(fc1[0]),
        image_size=grid*patch,
        patch_size=patch,
        n_img_tokens=ntok,
        vit_ln_eps=float(vcfg.get("layer_norm_eps", 1e-6))
    )

def write_siglip_tower(
    writer,
    g,
    root: str,
    n_layers: int,
    f32_norms: bool = False,
    pos_embd=None
) -> None:

    add_n = add_f32 if f32_norms else add
    E = f"{root}.embeddings."
    add_n(writer, "vit.patch_embd.weight", g(E + "patch_embedding.weight"))
    add_n(writer, "vit.patch_embd.bias",   g(E + "patch_embedding.bias"))
    add_n(writer, "vit.pos_embd", pos_embd if pos_embd is not None else g(E + "position_embedding.weight"))

    for i in range(n_layers):
        L = f"{root}.encoder.layers.{i}."
        add_n(writer, f"vit.blk.{i}.ln1.weight", g(L + "layer_norm1.weight")); add_n(writer, f"vit.blk.{i}.ln1.bias", g(L + "layer_norm1.bias"))
        add_n(writer, f"vit.blk.{i}.ln2.weight", g(L + "layer_norm2.weight")); add_n(writer, f"vit.blk.{i}.ln2.bias", g(L + "layer_norm2.bias"))
        for q in ("q", "k", "v"):
            add  (writer, f"vit.blk.{i}.attn_{q}.weight", g(L + f"self_attn.{q}_proj.weight"))
            add_n(writer, f"vit.blk.{i}.attn_{q}.bias",   g(L + f"self_attn.{q}_proj.bias"))
        add(writer, f"vit.blk.{i}.attn_o.weight", g(L + "self_attn.out_proj.weight")); add_n(writer, f"vit.blk.{i}.attn_o.bias", g(L + "self_attn.out_proj.bias"))
        add(writer, f"vit.blk.{i}.fc1.weight", g(L + "mlp.fc1.weight")); add_n(writer, f"vit.blk.{i}.fc1.bias", g(L + "mlp.fc1.bias"))
        add(writer, f"vit.blk.{i}.fc2.weight", g(L + "mlp.fc2.weight")); add_n(writer, f"vit.blk.{i}.fc2.bias", g(L + "mlp.fc2.bias"))

    add_n(writer, "vit.post_ln.weight", g(f"{root}.post_layernorm.weight")); add_n(writer, "vit.post_ln.bias", g(f"{root}.post_layernorm.bias"))

def probe_paligemma_vision(sf, keys, cfg_json: dict, vis_candidates, mmp_candidates) -> dict:

    vis = next((p for p in vis_candidates if f"{p}.embeddings.patch_embedding.weight" in keys), None)
    mmp = next((p for p in mmp_candidates if f"{p}.linear.weight" in keys), None)
    if vis is None or mmp is None:
        raise SystemExit("vision_tower / multi_modal_projector not found in checkpoint; "
                         f"tried {vis_candidates} and {mmp_candidates}")
    v = probe_siglip(sf, keys, vis, cfg_json.get("vision_config") or {}, 16)
    return dict(v, prefix=vis, mmp=mmp)

def write_paligemma_vision(writer, sf, v: dict) -> None:

    write_siglip_tower(writer, sf.get_tensor, v["prefix"], v["vit_layers"], f32_norms=True)
    add    (writer, "mm.proj.weight", sf.get_tensor(f"{v['mmp']}.linear.weight"))
    add_f32(writer, "mm.proj.bias",   sf.get_tensor(f"{v['mmp']}.linear.bias"))

def write_pi_kv(writer, kv, cfg: dict, adarms: bool = False) -> None:

    writer.add_string  (kv("paligemma_variant"),        cfg["paligemma_variant"])
    writer.add_string  (kv("action_expert_variant"),    cfg["action_expert_variant"])
    writer.add_uint32  (kv("hidden"),                   cfg["hidden"])
    writer.add_uint32  (kv("intermediate"),             cfg["intermediate"])
    writer.add_uint32  (kv("n_q_heads"),                cfg["n_q_heads"])
    writer.add_uint32  (kv("n_kv_heads"),               cfg["n_kv_heads"])
    writer.add_uint32  (kv("head_dim"),                 cfg["head_dim"])
    writer.add_uint32  (kv("n_layers"),                 cfg["n_layers"])
    writer.add_uint32  (kv("vocab_size"),               cfg["vocab_size"])
    writer.add_uint32  (kv("expert_h"),                 cfg["expert_h"])
    writer.add_uint32  (kv("expert_inter"),             cfg["expert_inter"])
    writer.add_uint32  (kv("chunk_size"),               cfg["chunk_size"])
    writer.add_uint32  (kv("num_steps"),                cfg["num_steps"])
    writer.add_uint32  (kv("n_action_steps"),           cfg["n_action_steps"])
    writer.add_uint32  (kv("max_state_dim"),            cfg["max_state_dim"])
    writer.add_uint32  (kv("max_action_dim"),           cfg["max_action_dim"])
    writer.add_uint32  (kv("real_state_dim"),           cfg["real_state_dim"])
    writer.add_uint32  (kv("real_action_dim"),          cfg["real_action_dim"])
    writer.add_uint32  (kv("tokenizer_max_length"),     cfg["tokenizer_max_length"])
    if adarms:
        writer.add_bool  (kv("use_adarms_expert"),      True)
        writer.add_uint32(kv("adarms_cond_dim"),        cfg["expert_h"])
        writer.add_string(kv("norm_mode"),              "quantiles")
    writer.add_float64 (kv("min_period"),               cfg["min_period"])
    writer.add_float64 (kv("max_period"),               cfg["max_period"])
    writer.add_float64 (kv("rope_theta"),               cfg["rope_theta"])
    writer.add_float32 (kv("rms_norm_eps"),             cfg["rms_norm_eps"])
    writer.add_float32 (kv("norm_eps"),                 cfg["norm_eps"])

    v = cfg["vit"]
    writer.add_uint32  (kv("vit_hidden"),               v["vit_hidden"])
    writer.add_uint32  (kv("vit_layers"),               v["vit_layers"])
    writer.add_uint32  (kv("vit_heads"),                v["vit_heads"])
    writer.add_uint32  (kv("image_size"),               v["image_size"])
    writer.add_uint32  (kv("patch_size"),               v["patch_size"])
    writer.add_uint32  (kv("n_img_tokens"),             v["n_img_tokens"])
    writer.add_float32 (kv("vit_ln_eps"),               v["vit_ln_eps"])

def write_qwen3vl_vit(
    writer,
    g,
    root: str,
    n_layers: int,
    n_deepstack: int,
    hidden: int,
    patch_flat: int
) -> None:

    add(writer, "vit.patch_embd.weight", g(f"{root}.patch_embed.proj.weight").reshape(hidden, patch_flat))
    add(writer, "vit.patch_embd.bias",   g(f"{root}.patch_embed.proj.bias"))
    add(writer, "vit.pos_embd",          g(f"{root}.pos_embed.weight"))

    for i in range(n_layers):
        VL = f"{root}.blocks.{i}."
        add(writer, f"vit.blk.{i}.ln1.weight", g(VL + "norm1.weight")); add(writer, f"vit.blk.{i}.ln1.bias", g(VL + "norm1.bias"))
        add(writer, f"vit.blk.{i}.ln2.weight", g(VL + "norm2.weight")); add(writer, f"vit.blk.{i}.ln2.bias", g(VL + "norm2.bias"))
        add(writer, f"vit.blk.{i}.attn_qkv.weight", g(VL + "attn.qkv.weight")); add(writer, f"vit.blk.{i}.attn_qkv.bias", g(VL + "attn.qkv.bias"))
        add(writer, f"vit.blk.{i}.attn_o.weight", g(VL + "attn.proj.weight")); add(writer, f"vit.blk.{i}.attn_o.bias", g(VL + "attn.proj.bias"))
        add(writer, f"vit.blk.{i}.fc1.weight", g(VL + "mlp.linear_fc1.weight")); add(writer, f"vit.blk.{i}.fc1.bias", g(VL + "mlp.linear_fc1.bias"))
        add(writer, f"vit.blk.{i}.fc2.weight", g(VL + "mlp.linear_fc2.weight")); add(writer, f"vit.blk.{i}.fc2.bias", g(VL + "mlp.linear_fc2.bias"))

    for j in range(n_deepstack):
        DM = f"{root}.deepstack_merger_list.{j}."
        add(writer, f"vit.deepstack.{j}.norm.weight", g(DM + "norm.weight")); add(writer, f"vit.deepstack.{j}.norm.bias", g(DM + "norm.bias"))
        add(writer, f"vit.deepstack.{j}.fc1.weight", g(DM + "linear_fc1.weight")); add(writer, f"vit.deepstack.{j}.fc1.bias", g(DM + "linear_fc1.bias"))
        add(writer, f"vit.deepstack.{j}.fc2.weight", g(DM + "linear_fc2.weight")); add(writer, f"vit.deepstack.{j}.fc2.bias", g(DM + "linear_fc2.bias"))

    MG = f"{root}.merger."
    add(writer, "vit.merger.norm.weight", g(MG + "norm.weight")); add(writer, "vit.merger.norm.bias", g(MG + "norm.bias"))
    add(writer, "vit.merger.fc1.weight", g(MG + "linear_fc1.weight")); add(writer, "vit.merger.fc1.bias", g(MG + "linear_fc1.bias"))
    add(writer, "vit.merger.fc2.weight", g(MG + "linear_fc2.weight")); add(writer, "vit.merger.fc2.bias", g(MG + "linear_fc2.bias"))

def write_qwen3_lm(writer, g, root: str, n_layers: int, embd_key: str | None = None) -> None:

    add(writer, "token_embd.weight",      g(embd_key or f"{root}.embed_tokens.weight"))
    add(writer, "vlm.output_norm.weight", g(f"{root}.norm.weight"))
    for i in range(n_layers):
        LL = f"{root}.layers.{i}."
        add(writer, f"vlm.blk.{i}.attn_norm.weight", g(LL + "input_layernorm.weight"))
        for q in ("q", "k", "v"):
            add(writer, f"vlm.blk.{i}.attn_{q}.weight", g(LL + f"self_attn.{q}_proj.weight"))
        add(writer, f"vlm.blk.{i}.attn_o.weight", g(LL + "self_attn.o_proj.weight"))
        add(writer, f"vlm.blk.{i}.attn_q_norm.weight", g(LL + "self_attn.q_norm.weight"))
        add(writer, f"vlm.blk.{i}.attn_k_norm.weight", g(LL + "self_attn.k_norm.weight"))
        add(writer, f"vlm.blk.{i}.ffn_norm.weight", g(LL + "post_attention_layernorm.weight"))
        add(writer, f"vlm.blk.{i}.ffn_gate.weight", g(LL + "mlp.gate_proj.weight"))
        add(writer, f"vlm.blk.{i}.ffn_up.weight",   g(LL + "mlp.up_proj.weight"))
        add(writer, f"vlm.blk.{i}.ffn_down.weight", g(LL + "mlp.down_proj.weight"))

DECODER_MAP = [
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

def write_decoder_blocks(writer, g, root: str, dst: str, n_layers: int) -> None:

    for i in range(n_layers):
        for src_suf, dst_suf in DECODER_MAP:
            add(writer, f"{dst}.blk.{i}.{dst_suf}", g(f"{root}.layers.{i}.{src_suf}"))

def write_dit_blocks(writer, g, root: str, dst: str, n_layers: int) -> None:

    for i in range(n_layers):
        TB = f"{root}.{i}."
        Q  = f"{dst}.{i}."
        add(writer, Q + "adaln.weight", g(TB + "norm1.linear.weight")); add(writer, Q + "adaln.bias", g(TB + "norm1.linear.bias"))
        for q in ("q", "k", "v"):
            add(writer, Q + f"attn_{q}.weight", g(TB + f"attn1.to_{q}.weight")); add(writer, Q + f"attn_{q}.bias", g(TB + f"attn1.to_{q}.bias"))
        add(writer, Q + "attn_o.weight", g(TB + "attn1.to_out.0.weight")); add(writer, Q + "attn_o.bias", g(TB + "attn1.to_out.0.bias"))
        add(writer, Q + "ff0.weight", g(TB + "ff.net.0.proj.weight")); add(writer, Q + "ff0.bias", g(TB + "ff.net.0.proj.bias"))
        add(writer, Q + "ff2.weight", g(TB + "ff.net.2.weight")); add(writer, Q + "ff2.bias", g(TB + "ff.net.2.bias"))

def write_vlsa_blocks(writer, g, root: str, dst: str, n_layers: int) -> None:

    for i in range(n_layers):
        TB = f"{root}.{i}."
        Q  = f"{dst}.{i}."
        add(writer, Q + "norm1.weight", g(TB + "norm1.weight")); add(writer, Q + "norm1.bias", g(TB + "norm1.bias"))
        add(writer, Q + "norm3.weight", g(TB + "norm3.weight")); add(writer, Q + "norm3.bias", g(TB + "norm3.bias"))
        for q in ("q", "k", "v"):
            add(writer, Q + f"attn_{q}.weight", g(TB + f"attn1.to_{q}.weight")); add(writer, Q + f"attn_{q}.bias", g(TB + f"attn1.to_{q}.bias"))
        add(writer, Q + "attn_o.weight", g(TB + "attn1.to_out.0.weight")); add(writer, Q + "attn_o.bias", g(TB + "attn1.to_out.0.bias"))
        add(writer, Q + "ff0.weight", g(TB + "ff.net.0.proj.weight")); add(writer, Q + "ff0.bias", g(TB + "ff.net.0.proj.bias"))
        add(writer, Q + "ff2.weight", g(TB + "ff.net.2.weight")); add(writer, Q + "ff2.bias", g(TB + "ff.net.2.bias"))

GR00T_PROJECTORS = [
    ("state_encoder.layer1",  "aex.state_enc.l1"),
    ("state_encoder.layer2",  "aex.state_enc.l2"),
    ("action_encoder.W1",     "aex.act_enc.W1"),
    ("action_encoder.W2",     "aex.act_enc.W2"),
    ("action_encoder.W3",     "aex.act_enc.W3"),
    ("action_decoder.layer1", "aex.act_dec.l1"),
    ("action_decoder.layer2", "aex.act_dec.l2"),
]

def write_gr00t_projectors(writer, g, root: str) -> None:

    for src, dst in GR00T_PROJECTORS:
        add(writer, f"{dst}.W", g(f"{root}.{src}.W"))
        add(writer, f"{dst}.b", g(f"{root}.{src}.b"))

def write_gr00t_time_embed(writer, g, root: str, dst: str) -> None:

    TE = f"{root}.model.timestep_encoder.timestep_embedder.linear_"
    add(writer, f"{dst}.time_emb.l1.weight", g(TE + "1.weight")); add(writer, f"{dst}.time_emb.l1.bias", g(TE + "1.bias"))
    add(writer, f"{dst}.time_emb.l2.weight", g(TE + "2.weight")); add(writer, f"{dst}.time_emb.l2.bias", g(TE + "2.bias"))

def write_gr00t_proj_out(writer, g, root: str, dst: str) -> None:

    add(writer, f"{dst}.proj_out1.weight", g(f"{root}.model.proj_out_1.weight")); add(writer, f"{dst}.proj_out1.bias", g(f"{root}.model.proj_out_1.bias"))
    add(writer, f"{dst}.proj_out2.weight", g(f"{root}.model.proj_out_2.weight")); add(writer, f"{dst}.proj_out2.bias", g(f"{root}.model.proj_out_2.bias"))

def write_prismatic_tower(writer, g, dino_layers: int, sig_layers: int) -> None:

    PD = "vision_backbone.featurizer."
    add_bf16(writer, "vis.d.patch.weight", g(PD + "patch_embed.proj.weight"))
    add_bf16(writer, "vis.d.patch.bias",   g(PD + "patch_embed.proj.bias"))
    add_bf16(writer, "vis.d.cls", g(PD + "cls_token"))
    add_bf16(writer, "vis.d.reg", g(PD + "reg_token"))
    add_bf16(writer, "vis.d.pos", g(PD + "pos_embed"))
    for i in range(dino_layers):
        P, Q = f"{PD}blocks.{i}.", f"vis.d.blk.{i}."
        add_bf16(writer, Q + "ln1.weight", g(P + "norm1.weight")); add_bf16(writer, Q + "ln1.bias", g(P + "norm1.bias"))
        add_bf16(writer, Q + "ln2.weight", g(P + "norm2.weight")); add_bf16(writer, Q + "ln2.bias", g(P + "norm2.bias"))
        add_bf16(writer, Q + "ls1", g(P + "ls1.scale_factor")); add_bf16(writer, Q + "ls2", g(P + "ls2.scale_factor"))
        add_bf16(writer, Q + "qkv.weight", g(P + "attn.qkv.weight")); add_bf16(writer, Q + "qkv.bias", g(P + "attn.qkv.bias"))
        add_bf16(writer, Q + "proj.weight", g(P + "attn.proj.weight")); add_bf16(writer, Q + "proj.bias", g(P + "attn.proj.bias"))
        add_bf16(writer, Q + "fc1.weight", g(P + "mlp.fc1.weight")); add_bf16(writer, Q + "fc1.bias", g(P + "mlp.fc1.bias"))
        add_bf16(writer, Q + "fc2.weight", g(P + "mlp.fc2.weight")); add_bf16(writer, Q + "fc2.bias", g(P + "mlp.fc2.bias"))

    PS = "vision_backbone.fused_featurizer."
    add_bf16(writer, "vis.s.patch.weight", g(PS + "patch_embed.proj.weight"))
    add_bf16(writer, "vis.s.patch.bias",   g(PS + "patch_embed.proj.bias"))
    add_bf16(writer, "vis.s.pos", g(PS + "pos_embed"))
    for i in range(sig_layers):
        P, Q = f"{PS}blocks.{i}.", f"vis.s.blk.{i}."
        add_bf16(writer, Q + "ln1.weight", g(P + "norm1.weight")); add_bf16(writer, Q + "ln1.bias", g(P + "norm1.bias"))
        add_bf16(writer, Q + "ln2.weight", g(P + "norm2.weight")); add_bf16(writer, Q + "ln2.bias", g(P + "norm2.bias"))
        add_bf16(writer, Q + "qkv.weight", g(P + "attn.qkv.weight")); add_bf16(writer, Q + "qkv.bias", g(P + "attn.qkv.bias"))
        add_bf16(writer, Q + "proj.weight", g(P + "attn.proj.weight")); add_bf16(writer, Q + "proj.bias", g(P + "attn.proj.bias"))
        add_bf16(writer, Q + "fc1.weight", g(P + "mlp.fc1.weight")); add_bf16(writer, Q + "fc1.bias", g(P + "mlp.fc1.bias"))
        add_bf16(writer, Q + "fc2.weight", g(P + "mlp.fc2.weight")); add_bf16(writer, Q + "fc2.bias", g(P + "mlp.fc2.bias"))

    for fc in ("fc1", "fc2", "fc3"):
        add_bf16(writer, f"vis.proj.{fc}.weight", g(f"projector.{fc}.weight"))
        add_bf16(writer, f"vis.proj.{fc}.bias",   g(f"projector.{fc}.bias"))

def write_prismatic_lm(writer, g, n_layers: int, qkv_bias: bool = False) -> None:

    for i in range(n_layers):
        P, Q = f"language_model.model.layers.{i}.", f"lm.blk.{i}."
        add_bf16(writer, Q + "attn_norm.weight", g(P + "input_layernorm.weight"))
        add_bf16(writer, Q + "ffn_norm.weight",  g(P + "post_attention_layernorm.weight"))
        for src, dst in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v")):
            add_bf16(writer, Q + dst + ".weight", g(P + f"self_attn.{src}_proj.weight"))
            if qkv_bias:
                add_bf16(writer, Q + dst + ".bias", g(P + f"self_attn.{src}_proj.bias"))
        add_bf16(writer, Q + "attn_o.weight", g(P + "self_attn.o_proj.weight"))
        add_bf16(writer, Q + "ffn_gate.weight", g(P + "mlp.gate_proj.weight"))
        add_bf16(writer, Q + "ffn_up.weight",   g(P + "mlp.up_proj.weight"))
        add_bf16(writer, Q + "ffn_down.weight", g(P + "mlp.down_proj.weight"))
    add_bf16(writer, "lm.output_norm.weight", g("language_model.model.norm.weight"))

def norm_eps(ckpt: Path, default: float = 1e-8) -> float:

    def _eps(meta_path: Path):
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        for step in meta.get("steps", []):
            if step.get("registry_name") in ("normalizer_processor", "unnormalizer_processor"):
                cfg = step.get("config", {})
                if "eps" in cfg:
                    return float(cfg["eps"])
        return None

    eps = _eps(ckpt / "policy_preprocessor.json")
    if eps is None:
        eps = _eps(ckpt / "policy_postprocessor.json")
    return float(eps if eps is not None else default)

def load_processor_stats(ckpt: Path, meta_json: str, registry: str, key: str, dim: int):

    meta_path = ckpt / meta_json
    if not meta_path.exists():
        print(f"  stats: {meta_json} missing - using identity for {key}")
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"  stats: {meta_json} parse failed ({e}) - using identity for {key}")
        return None

    state_file = None
    for step in meta.get("steps", []):
        if step.get("registry_name") == registry:
            state_file = step.get("state_file")
            break
    if not state_file:
        print(f"  stats: no {registry} step in {meta_json} - using identity for {key}")
        return None

    sf_path = ckpt / state_file
    if not sf_path.is_file():
        print(f"  stats: {sf_path.name} referenced by {meta_json} but missing - using identity for {key}")
        return None

    with safe_open(str(sf_path), framework="pt") as f:
        keys = set(f.keys())
        mk, sk = f"{key}.mean", f"{key}.std"
        if mk not in keys or sk not in keys:
            print(f"  stats: {sf_path.name} lacks {mk}/{sk} - using identity for {key}")
            return None
        mean = f.get_tensor(mk).float().numpy().reshape(-1)
        std  = f.get_tensor(sk).float().numpy().reshape(-1)

    if mean.size != dim or std.size != dim:
        print(f"  stats: {mk} dim mismatch ({mean.size} vs {dim}) in {sf_path.name} - using identity")
        return None
    print(f"  stats: loaded {key} from {sf_path.name} ({mk}/{sk})")
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)

def identity_stats(state_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    return {
        "state_mean":  np.zeros(state_dim,  dtype=np.float32),
        "state_std":   np.ones (state_dim,  dtype=np.float32),
        "action_mean": np.zeros(action_dim, dtype=np.float32),
        "action_std":  np.ones (action_dim, dtype=np.float32),
    }
