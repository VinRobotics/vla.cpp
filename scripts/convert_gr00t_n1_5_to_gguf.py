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

import numpy as np
import torch
from safetensors import safe_open

import gguf

ARCH = "gr00t_n1_5"
KV = lambda name: f"{ARCH}.{name}"

VIT = dict(vit_hidden=1152, vit_layers=27, vit_heads=16, vit_inter=4304,
           image_size=224, patch_size=14, vit_ln_eps=1e-6)

QWEN3 = dict(lm_hidden=2048, lm_q_heads=16, lm_kv_heads=8, lm_head_dim=128,
             lm_inter=6144, lm_rope_theta=1000000.0, lm_rms_eps=1e-6)
LM_LAYERS_USED = 12

AH = dict(backbone_embedding_dim=2048, input_embedding_dim=1536,
          dit_hidden=1536, dit_heads=32, dit_head_dim=48, dit_layers=16, dit_interleave=1,
          vlsa_layers=4, vlsa_heads=32, vlsa_head_dim=64, vlsa_inter=8192,
          num_target_vision_tokens=32, action_horizon=16, action_dim=32, max_state_dim=64,
          num_inference_timesteps=4, num_timestep_buckets=1000, max_num_embodiments=32,
          max_seq_len=1024, ln_eps=1e-5, norm_out_eps=1e-6, vlln_eps=1e-5)
IMAGE_TOKEN_INDEX = 151669
EMBODIMENT_TAG_MAPPING = {"new_embodiment": 31, "oxe_droid": 17, "agibot_genie1": 26, "gr1": 24}

def _bf16_u16(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous().view(torch.uint16).cpu().numpy()

def _add(writer: gguf.GGUFWriter, name: str, t: torch.Tensor) -> None:
    if t.dtype == torch.float32:
        writer.add_tensor(name, t.contiguous().cpu().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
    elif t.dtype == torch.bfloat16:
        writer.add_tensor(name, _bf16_u16(t), raw_shape=list(t.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        raise NotImplementedError(f"unsupported dtype {t.dtype} for {name}")

def _load_sharded(ckpt: Path) -> dict[str, torch.Tensor]:

    idx = ckpt / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
    else:
        one = ckpt / "model.safetensors"
        if not one.exists():
            raise SystemExit(f"no model.safetensors[.index.json] under {ckpt}")
        with safe_open(str(one), framework="pt") as f:
            weight_map = {k: one.name for k in f.keys()}
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(by_shard):
        with safe_open(str(ckpt / shard), framework="pt") as f:
            for k in by_shard[shard]:
                out[k] = f.get_tensor(k)
    return out

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
    blob = {emb_key: {"state":  {"min": s_min, "max": s_max},
                      "action": {"min": a_min, "max": a_max}}}
    out_path.write_text(json.dumps(blob, indent=2))
    print(f"wrote {out_path}  (embodiment {emb_key!r}: state[{len(s_min)}] + action[{len(a_min)}] min/max)")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="GR00T-N1.5-3B checkpoint dir. Two layouts are auto-detected: "
                         "(a) NVIDIA Isaac snapshot (sharded safetensors, un-prefixed tensor names, "
                         "config.json model_type=gr00t_n1_5, experiment_cfg/metadata.json); "
                         "(b) lerobot finetune (single model.safetensors with `_groot_model.` prefix, "
                         "config.json type=groot, policy_*processor_step_*.safetensors min/max stats).")
    ap.add_argument("--out", type=Path, default=None, help="output GGUF path (default: <ckpt>/gr00t_n1_5.gguf)")
    ap.add_argument("--stats-out", type=Path, default=None,
                    help="[lerobot ckpt] where to write the bridge's dataset_statistics.json "
                         "(default: <out dir>/dataset_statistics.json). state+action min/max are "
                         "read from the lerobot processor safetensors; the eval bridge consumes it "
                         "via --stats-json (the un-normalize is a host-side affine, not a ggml concern).")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out = (args.out or ckpt / f"{ARCH}.gguf").resolve()
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")
    if not (ckpt / "model.safetensors.index.json").exists() and not (ckpt / "model.safetensors").exists():
        raise SystemExit(f"no model.safetensors[.index.json] under {ckpt}")

    cfg_json = json.loads(cfg_path.read_text())
    model_type = str(cfg_json.get("model_type", "") or cfg_json.get("type", ""))

    is_lerobot = model_type != "gr00t_n1_5"
    if is_lerobot:
        print(f"detected lerobot finetune layout (config type={model_type!r}); arch hparams from "
              f"built-in GR00T-N1.5 defaults, weights de-prefixed `_groot_model.`, stats from processors")
    md_path = ckpt / "experiment_cfg" / "metadata.json"
    bcfg = cfg_json.get("backbone_cfg", {})
    ahcfg = cfg_json.get("action_head_cfg", {})
    if int(bcfg.get("select_layer", LM_LAYERS_USED)) != LM_LAYERS_USED:
        raise SystemExit(f"backbone_cfg.select_layer = {bcfg.get('select_layer')}, expected {LM_LAYERS_USED}")
    if int(ahcfg.get("action_horizon", AH["action_horizon"])) != AH["action_horizon"]:
        AH["action_horizon"] = int(ahcfg["action_horizon"])
    if int(ahcfg.get("action_dim", AH["action_dim"])) != AH["action_dim"]:
        AH["action_dim"] = int(ahcfg["action_dim"])
    AH["num_inference_timesteps"] = int(ahcfg.get("num_inference_timesteps", AH["num_inference_timesteps"]))
    AH["num_target_vision_tokens"] = int(ahcfg.get("num_target_vision_tokens", AH["num_target_vision_tokens"]))
    AH["max_state_dim"] = int(ahcfg.get("max_state_dim", AH["max_state_dim"]))
    dmc = ahcfg.get("diffusion_model_cfg", {})
    AH["dit_layers"] = int(dmc.get("num_layers", AH["dit_layers"]))
    AH["dit_heads"] = int(dmc.get("num_attention_heads", AH["dit_heads"]))
    AH["dit_head_dim"] = int(dmc.get("attention_head_dim", AH["dit_head_dim"]))
    AH["dit_hidden"] = AH["dit_heads"] * AH["dit_head_dim"]
    AH["dit_interleave"] = int(bool(dmc.get("interleave_self_attention", True)))
    vcfg = ahcfg.get("vl_self_attention_cfg", {})
    AH["vlsa_layers"] = int(vcfg.get("num_layers", AH["vlsa_layers"]))
    AH["vlsa_heads"] = int(vcfg.get("num_attention_heads", AH["vlsa_heads"]))
    AH["vlsa_head_dim"] = int(vcfg.get("attention_head_dim", AH["vlsa_head_dim"]))

    print(f"loading sharded safetensors from {ckpt} ...")
    W = _load_sharded(ckpt)
    if is_lerobot:

        pfx = "_groot_model."
        W = {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in W.items()}
    keys = set(W.keys())
    print(f"  {len(W)} tensors")

    def _maxlayer(pfx):
        m = -1
        for k in keys:
            if k.startswith(pfx):
                try: m = max(m, int(k[len(pfx):].split(".", 1)[0]))
                except ValueError: pass
        return m + 1
    n_vit = _maxlayer("backbone.eagle_model.vision_model.vision_model.encoder.layers.")
    n_lm  = _maxlayer("backbone.eagle_model.language_model.model.layers.")
    n_dit = _maxlayer("action_head.model.transformer_blocks.")
    n_vlsa = _maxlayer("action_head.vl_self_attention.transformer_blocks.")
    if n_vit != VIT["vit_layers"]:   raise SystemExit(f"checkpoint has {n_vit} SigLIP layers, expected {VIT['vit_layers']}")
    if n_lm  != LM_LAYERS_USED:      raise SystemExit(f"checkpoint has {n_lm} Qwen3 layers, expected {LM_LAYERS_USED}")
    if n_dit != AH["dit_layers"]:    raise SystemExit(f"checkpoint has {n_dit} DiT blocks, expected {AH['dit_layers']}")
    if n_vlsa != AH["vlsa_layers"]:  raise SystemExit(f"checkpoint has {n_vlsa} vl_self_attention blocks, expected {AH['vlsa_layers']}")

    _emb_key = "backbone.eagle_model.language_model.model.embed_tokens.weight"
    _lmh_key = "backbone.eagle_model.language_model.lm_head.weight"
    tok_embd_key = _emb_key if _emb_key in keys else _lmh_key
    if tok_embd_key not in keys:
        raise SystemExit("checkpoint has neither embed_tokens.weight nor lm_head.weight")
    vocab = int(W[tok_embd_key].shape[0])
    grid = VIT["image_size"] // VIT["patch_size"]
    n_img_tokens = grid * grid
    conn_in, conn_out = W["backbone.eagle_model.mlp1.0.weight"].shape[1], W["backbone.eagle_model.mlp1.0.weight"].shape[0]
    assert conn_in == VIT["vit_hidden"] and conn_out == QWEN3["lm_hidden"], (conn_in, conn_out)

    metadata_json = md_path.read_text() if md_path.exists() else "{}"

    print(f"resolved cfg: vit={VIT['vit_hidden']}d×{VIT['vit_layers']}L×{VIT['vit_heads']}h  n_img_tok={n_img_tokens}  "
          f"mlp1=Linear({conn_in}→{conn_out})  lm=Qwen3 {QWEN3['lm_hidden']}d×{LM_LAYERS_USED}L "
          f"({QWEN3['lm_q_heads']}q/{QWEN3['lm_kv_heads']}kv×{QWEN3['lm_head_dim']}, q/k_norm)  vocab={vocab}  "
          f"dit={AH['dit_layers']}L×{AH['dit_heads']}h×{AH['dit_head_dim']}(inner {AH['dit_hidden']}) interleave={AH['dit_interleave']}  "
          f"vlsa={AH['vlsa_layers']}L×{AH['vlsa_heads']}h×{AH['vlsa_head_dim']}  in_emb={AH['input_embedding_dim']}  "
          f"horizon={AH['action_horizon']} action_dim={AH['action_dim']} max_state={AH['max_state_dim']}  N_steps={AH['num_inference_timesteps']}  "
          f"future_tok={AH['num_target_vision_tokens']}  embodiments={AH['max_num_embodiments']}  metadata.json={len(metadata_json)} chars")

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    writer.add_string(KV("architecture"), ARCH)

    u32 = dict(
        **{k: VIT[k] for k in ("vit_hidden", "vit_layers", "vit_heads", "vit_inter", "image_size", "patch_size")},
        n_img_tokens=n_img_tokens, vit_pixel_shuffle=0, mlp_connector_layers=1,
        lm_hidden=QWEN3["lm_hidden"], lm_layers_used=LM_LAYERS_USED, lm_q_heads=QWEN3["lm_q_heads"],
        lm_kv_heads=QWEN3["lm_kv_heads"], lm_head_dim=QWEN3["lm_head_dim"], lm_inter=QWEN3["lm_inter"],
        vocab_size=vocab, image_token_index=IMAGE_TOKEN_INDEX,
        **{k: AH[k] for k in ("backbone_embedding_dim", "input_embedding_dim", "dit_hidden", "dit_heads",
                              "dit_head_dim", "dit_layers", "dit_interleave", "vlsa_layers", "vlsa_heads",
                              "vlsa_head_dim", "vlsa_inter", "num_target_vision_tokens", "action_horizon",
                              "action_dim", "max_state_dim", "num_inference_timesteps", "num_timestep_buckets",
                              "max_num_embodiments", "max_seq_len")},
    )
    for k, v in u32.items():
        writer.add_uint32(KV(k), int(v))
    writer.add_float64(KV("lm_rope_theta"), float(QWEN3["lm_rope_theta"]))
    writer.add_float32(KV("lm_rms_eps"),    float(QWEN3["lm_rms_eps"]))
    writer.add_float32(KV("vit_ln_eps"),    float(VIT["vit_ln_eps"]))
    writer.add_float32(KV("ln_eps"),        float(AH["ln_eps"]))
    writer.add_float32(KV("norm_out_eps"),  float(AH["norm_out_eps"]))
    writer.add_float32(KV("vlln_eps"),      float(AH["vlln_eps"]))
    writer.add_string(KV("embodiment_tag_mapping"), json.dumps(EMBODIMENT_TAG_MAPPING))
    writer.add_string(KV("metadata_json"), metadata_json)

    g = lambda name: W[name]

    VE = "backbone.eagle_model.vision_model.vision_model.embeddings."
    _add(writer, "vit.patch_embd.weight", g(VE + "patch_embedding.weight"))
    _add(writer, "vit.patch_embd.bias",   g(VE + "patch_embedding.bias"))
    _add(writer, "vit.pos_embd",           g(VE + "position_embedding.weight"))
    for i in range(VIT["vit_layers"]):
        VL = f"backbone.eagle_model.vision_model.vision_model.encoder.layers.{i}."
        _add(writer, f"vit.blk.{i}.ln1.weight", g(VL + "layer_norm1.weight")); _add(writer, f"vit.blk.{i}.ln1.bias", g(VL + "layer_norm1.bias"))
        _add(writer, f"vit.blk.{i}.ln2.weight", g(VL + "layer_norm2.weight")); _add(writer, f"vit.blk.{i}.ln2.bias", g(VL + "layer_norm2.bias"))
        for q in ("q", "k", "v"):
            _add(writer, f"vit.blk.{i}.attn_{q}.weight", g(VL + f"self_attn.{q}_proj.weight")); _add(writer, f"vit.blk.{i}.attn_{q}.bias", g(VL + f"self_attn.{q}_proj.bias"))
        _add(writer, f"vit.blk.{i}.attn_o.weight", g(VL + "self_attn.out_proj.weight")); _add(writer, f"vit.blk.{i}.attn_o.bias", g(VL + "self_attn.out_proj.bias"))
        _add(writer, f"vit.blk.{i}.fc1.weight", g(VL + "mlp.fc1.weight")); _add(writer, f"vit.blk.{i}.fc1.bias", g(VL + "mlp.fc1.bias"))
        _add(writer, f"vit.blk.{i}.fc2.weight", g(VL + "mlp.fc2.weight")); _add(writer, f"vit.blk.{i}.fc2.bias", g(VL + "mlp.fc2.bias"))
    _add(writer, "vit.post_ln.weight", g("backbone.eagle_model.vision_model.vision_model.post_layernorm.weight"))
    _add(writer, "vit.post_ln.bias",   g("backbone.eagle_model.vision_model.vision_model.post_layernorm.bias"))

    _add(writer, "mm.fc.weight", g("backbone.eagle_model.mlp1.0.weight")); _add(writer, "mm.fc.bias", g("backbone.eagle_model.mlp1.0.bias"))

    _add(writer, "token_embd.weight",      g(tok_embd_key))
    _add(writer, "vlm.output_norm.weight", g("backbone.eagle_model.language_model.model.norm.weight"))
    for i in range(LM_LAYERS_USED):
        LL = f"backbone.eagle_model.language_model.model.layers.{i}."
        _add(writer, f"vlm.blk.{i}.attn_norm.weight", g(LL + "input_layernorm.weight"))
        for q in ("q", "k", "v"):
            _add(writer, f"vlm.blk.{i}.attn_{q}.weight", g(LL + f"self_attn.{q}_proj.weight"))
        _add(writer, f"vlm.blk.{i}.attn_o.weight", g(LL + "self_attn.o_proj.weight"))
        _add(writer, f"vlm.blk.{i}.attn_q_norm.weight", g(LL + "self_attn.q_norm.weight"))
        _add(writer, f"vlm.blk.{i}.attn_k_norm.weight", g(LL + "self_attn.k_norm.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_norm.weight", g(LL + "post_attention_layernorm.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_gate.weight", g(LL + "mlp.gate_proj.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_up.weight",   g(LL + "mlp.up_proj.weight"))
        _add(writer, f"vlm.blk.{i}.ffn_down.weight", g(LL + "mlp.down_proj.weight"))

    AHK = "action_head."
    _add(writer, "aex.vlln.weight", g(AHK + "vlln.weight")); _add(writer, "aex.vlln.bias", g(AHK + "vlln.bias"))
    for i in range(AH["vlsa_layers"]):
        TB = f"{AHK}vl_self_attention.transformer_blocks.{i}."
        _add(writer, f"aex.vlsa.{i}.norm1.weight", g(TB + "norm1.weight")); _add(writer, f"aex.vlsa.{i}.norm1.bias", g(TB + "norm1.bias"))
        _add(writer, f"aex.vlsa.{i}.norm3.weight", g(TB + "norm3.weight")); _add(writer, f"aex.vlsa.{i}.norm3.bias", g(TB + "norm3.bias"))
        for q in ("q", "k", "v"):
            _add(writer, f"aex.vlsa.{i}.attn_{q}.weight", g(TB + f"attn1.to_{q}.weight")); _add(writer, f"aex.vlsa.{i}.attn_{q}.bias", g(TB + f"attn1.to_{q}.bias"))
        _add(writer, f"aex.vlsa.{i}.attn_o.weight", g(TB + "attn1.to_out.0.weight")); _add(writer, f"aex.vlsa.{i}.attn_o.bias", g(TB + "attn1.to_out.0.bias"))
        _add(writer, f"aex.vlsa.{i}.ff0.weight", g(TB + "ff.net.0.proj.weight")); _add(writer, f"aex.vlsa.{i}.ff0.bias", g(TB + "ff.net.0.proj.bias"))
        _add(writer, f"aex.vlsa.{i}.ff2.weight", g(TB + "ff.net.2.weight")); _add(writer, f"aex.vlsa.{i}.ff2.bias", g(TB + "ff.net.2.bias"))

    for src, dst in (("state_encoder.layer1", "aex.state_enc.l1"), ("state_encoder.layer2", "aex.state_enc.l2"),
                     ("action_encoder.W1", "aex.act_enc.W1"), ("action_encoder.W2", "aex.act_enc.W2"), ("action_encoder.W3", "aex.act_enc.W3"),
                     ("action_decoder.layer1", "aex.act_dec.l1"), ("action_decoder.layer2", "aex.act_dec.l2")):
        _add(writer, f"{dst}.W", g(AHK + src + ".W")); _add(writer, f"{dst}.b", g(AHK + src + ".b"))
    _add(writer, "aex.future_tokens", g(AHK + "future_tokens.weight"))
    _add(writer, "aex.pos_embd",      g(AHK + "position_embedding.weight"))

    _add(writer, "aex.dit.time_emb.l1.weight", g(AHK + "model.timestep_encoder.timestep_embedder.linear_1.weight")); _add(writer, "aex.dit.time_emb.l1.bias", g(AHK + "model.timestep_encoder.timestep_embedder.linear_1.bias"))
    _add(writer, "aex.dit.time_emb.l2.weight", g(AHK + "model.timestep_encoder.timestep_embedder.linear_2.weight")); _add(writer, "aex.dit.time_emb.l2.bias", g(AHK + "model.timestep_encoder.timestep_embedder.linear_2.bias"))
    for i in range(AH["dit_layers"]):
        TB = f"{AHK}model.transformer_blocks.{i}."
        _add(writer, f"aex.dit.{i}.adaln.weight", g(TB + "norm1.linear.weight")); _add(writer, f"aex.dit.{i}.adaln.bias", g(TB + "norm1.linear.bias"))
        for q in ("q", "k", "v"):
            _add(writer, f"aex.dit.{i}.attn_{q}.weight", g(TB + f"attn1.to_{q}.weight")); _add(writer, f"aex.dit.{i}.attn_{q}.bias", g(TB + f"attn1.to_{q}.bias"))
        _add(writer, f"aex.dit.{i}.attn_o.weight", g(TB + "attn1.to_out.0.weight")); _add(writer, f"aex.dit.{i}.attn_o.bias", g(TB + "attn1.to_out.0.bias"))
        _add(writer, f"aex.dit.{i}.ff0.weight", g(TB + "ff.net.0.proj.weight")); _add(writer, f"aex.dit.{i}.ff0.bias", g(TB + "ff.net.0.proj.bias"))
        _add(writer, f"aex.dit.{i}.ff2.weight", g(TB + "ff.net.2.weight")); _add(writer, f"aex.dit.{i}.ff2.bias", g(TB + "ff.net.2.bias"))
    _add(writer, "aex.dit.proj_out1.weight", g(AHK + "model.proj_out_1.weight")); _add(writer, "aex.dit.proj_out1.bias", g(AHK + "model.proj_out_1.bias"))
    _add(writer, "aex.dit.proj_out2.weight", g(AHK + "model.proj_out_2.weight")); _add(writer, "aex.dit.proj_out2.bias", g(AHK + "model.proj_out_2.bias"))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB)  — combined GGUF (Eagle-2.5-VL + action head + cfg + metadata.json)")

    if is_lerobot:
        stats_out = (args.stats_out or out.parent / "dataset_statistics.json").resolve()
        _write_lerobot_stats(ckpt, stats_out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
