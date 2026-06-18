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

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

ARCH = "openvla_oft"

def KV(name: str) -> str:
    return f"{ARCH}.{name}"

DINO = dict(hidden=1024, layers=23, heads=16, head_dim=64, inter=4096)
SIG  = dict(hidden=1152, layers=26, heads=16, head_dim=72, inter=4304)
IMG, PATCH, NPATCH = 224, 14, 256
VIT_LN_EPS = 1e-6

LM = dict(hidden=4096, layers=32, q_heads=32, kv_heads=32, head_dim=128, inter=11008,
          rope_theta=10000.0, rms_eps=1e-6, vocab=32064)
ACT = dict(chunk=8, action_dim=7, proprio_dim=8, head_hidden=4096, head_blocks=2,
           head_ln_eps=1e-5)
NUM_IMAGES = 2
STOP_TOKEN_ID = 2
EMPTY_TOKEN_ID = 29871
VDIM = DINO["hidden"] + SIG["hidden"]
PROJ_MID = 4 * VDIM

def _bf16_u16(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)

def _add(writer, name: str, t: torch.Tensor) -> None:
    writer.add_tensor(name, _bf16_u16(t), raw_shape=list(t.shape),
                      raw_dtype=gguf.GGMLQuantizationType.BF16)

def _load_sharded_safetensors(ckpt: Path) -> dict:

    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    M = {}
    for shard in shards:
        with safe_open(str(ckpt / shard), "pt") as f:
            for k in f.keys():
                M[k] = f.get_tensor(k)
    return M

def _load_pt_module(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--action-head", type=str, default=None,
                    help="action_head .pt (default: action_head--*_checkpoint.pt in ckpt)")
    ap.add_argument("--proprio", type=str, default=None,
                    help="proprio_projector .pt (default: proprio_projector--*_checkpoint.pt in ckpt)")
    args = ap.parse_args()

    ckpt = args.ckpt
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ah_path = Path(args.action_head) if args.action_head else next(ckpt.glob("action_head--*checkpoint.pt"))
    pp_path = Path(args.proprio) if args.proprio else next(ckpt.glob("proprio_projector--*checkpoint.pt"))

    print(f"loading merged safetensors from {ckpt} ...")
    M = _load_sharded_safetensors(ckpt)
    AH = _load_pt_module(ah_path)
    PP = _load_pt_module(pp_path)
    statistics_json = (ckpt / "dataset_statistics.json").read_text()
    print(f"  action_head: {ah_path.name}   proprio: {pp_path.name}")

    writer = gguf.GGUFWriter(str(args.out), ARCH)
    writer.add_string(KV("architecture"), ARCH)

    for tag, D in (("dino", DINO), ("sig", SIG)):
        for k in ("hidden", "layers", "heads", "head_dim", "inter"):
            writer.add_uint32(KV(f"vit.{tag}.{k}"), D[k])
    writer.add_uint32(KV("vit.image_size"), IMG)
    writer.add_uint32(KV("vit.patch_size"), PATCH)
    writer.add_uint32(KV("vit.n_patches"), NPATCH)
    writer.add_float32(KV("vit.ln_eps"), VIT_LN_EPS)
    writer.add_uint32(KV("vit.proj_mid"), PROJ_MID)
    writer.add_uint32(KV("vit.vdim"), VDIM)
    writer.add_uint32(KV("vit.num_images"), NUM_IMAGES)

    for k in ("hidden", "layers", "q_heads", "kv_heads", "head_dim", "inter", "vocab"):
        writer.add_uint32(KV(f"lm.{k}"), LM[k])
    writer.add_float32(KV("lm.rope_theta"), LM["rope_theta"])
    writer.add_float32(KV("lm.rms_eps"), LM["rms_eps"])

    for k in ("chunk", "action_dim", "proprio_dim", "head_hidden", "head_blocks"):
        writer.add_uint32(KV(f"action.{k}"), ACT[k])
    writer.add_float32(KV("action.head_ln_eps"), ACT["head_ln_eps"])
    writer.add_uint32(KV("tokens.stop_id"), STOP_TOKEN_ID)
    writer.add_uint32(KV("tokens.empty_id"), EMPTY_TOKEN_ID)
    writer.add_string(KV("statistics_json"), statistics_json)

    PD = "vision_backbone.featurizer."
    _add(writer, "vis.d.patch.weight", M[PD + "patch_embed.proj.weight"])
    _add(writer, "vis.d.patch.bias",   M[PD + "patch_embed.proj.bias"])
    _add(writer, "vis.d.cls", M[PD + "cls_token"])
    _add(writer, "vis.d.reg", M[PD + "reg_token"])
    _add(writer, "vis.d.pos", M[PD + "pos_embed"])
    for i in range(DINO["layers"]):
        P, Q = f"{PD}blocks.{i}.", f"vis.d.blk.{i}."
        _add(writer, Q + "ln1.weight", M[P + "norm1.weight"]); _add(writer, Q + "ln1.bias", M[P + "norm1.bias"])
        _add(writer, Q + "ln2.weight", M[P + "norm2.weight"]); _add(writer, Q + "ln2.bias", M[P + "norm2.bias"])
        _add(writer, Q + "ls1", M[P + "ls1.scale_factor"]); _add(writer, Q + "ls2", M[P + "ls2.scale_factor"])
        _add(writer, Q + "qkv.weight", M[P + "attn.qkv.weight"]); _add(writer, Q + "qkv.bias", M[P + "attn.qkv.bias"])
        _add(writer, Q + "proj.weight", M[P + "attn.proj.weight"]); _add(writer, Q + "proj.bias", M[P + "attn.proj.bias"])
        _add(writer, Q + "fc1.weight", M[P + "mlp.fc1.weight"]); _add(writer, Q + "fc1.bias", M[P + "mlp.fc1.bias"])
        _add(writer, Q + "fc2.weight", M[P + "mlp.fc2.weight"]); _add(writer, Q + "fc2.bias", M[P + "mlp.fc2.bias"])

    PS = "vision_backbone.fused_featurizer."
    _add(writer, "vis.s.patch.weight", M[PS + "patch_embed.proj.weight"])
    _add(writer, "vis.s.patch.bias",   M[PS + "patch_embed.proj.bias"])
    _add(writer, "vis.s.pos", M[PS + "pos_embed"])
    for i in range(SIG["layers"]):
        P, Q = f"{PS}blocks.{i}.", f"vis.s.blk.{i}."
        _add(writer, Q + "ln1.weight", M[P + "norm1.weight"]); _add(writer, Q + "ln1.bias", M[P + "norm1.bias"])
        _add(writer, Q + "ln2.weight", M[P + "norm2.weight"]); _add(writer, Q + "ln2.bias", M[P + "norm2.bias"])
        _add(writer, Q + "qkv.weight", M[P + "attn.qkv.weight"]); _add(writer, Q + "qkv.bias", M[P + "attn.qkv.bias"])
        _add(writer, Q + "proj.weight", M[P + "attn.proj.weight"]); _add(writer, Q + "proj.bias", M[P + "attn.proj.bias"])
        _add(writer, Q + "fc1.weight", M[P + "mlp.fc1.weight"]); _add(writer, Q + "fc1.bias", M[P + "mlp.fc1.bias"])
        _add(writer, Q + "fc2.weight", M[P + "mlp.fc2.weight"]); _add(writer, Q + "fc2.bias", M[P + "mlp.fc2.bias"])

    for n in ("fc1", "fc2", "fc3"):
        _add(writer, f"vis.proj.{n}.weight", M[f"projector.{n}.weight"])
        _add(writer, f"vis.proj.{n}.bias",   M[f"projector.{n}.bias"])

    _add(writer, "token_embd.weight", M["language_model.model.embed_tokens.weight"])

    for i in range(LM["layers"]):
        P, Q = f"language_model.model.layers.{i}.", f"lm.blk.{i}."
        _add(writer, Q + "attn_norm.weight", M[P + "input_layernorm.weight"])
        _add(writer, Q + "ffn_norm.weight",  M[P + "post_attention_layernorm.weight"])
        _add(writer, Q + "attn_q.weight", M[P + "self_attn.q_proj.weight"])
        _add(writer, Q + "attn_k.weight", M[P + "self_attn.k_proj.weight"])
        _add(writer, Q + "attn_v.weight", M[P + "self_attn.v_proj.weight"])
        _add(writer, Q + "attn_o.weight", M[P + "self_attn.o_proj.weight"])
        _add(writer, Q + "ffn_gate.weight", M[P + "mlp.gate_proj.weight"])
        _add(writer, Q + "ffn_up.weight",   M[P + "mlp.up_proj.weight"])
        _add(writer, Q + "ffn_down.weight", M[P + "mlp.down_proj.weight"])
    _add(writer, "lm.output_norm.weight", M["language_model.model.norm.weight"])

    _add(writer, "aex.proprio.fc1.weight", PP["fc1.weight"]); _add(writer, "aex.proprio.fc1.bias", PP["fc1.bias"])
    _add(writer, "aex.proprio.fc2.weight", PP["fc2.weight"]); _add(writer, "aex.proprio.fc2.bias", PP["fc2.bias"])

    _add(writer, "aex.head.ln1.weight", AH["model.layer_norm1.weight"]); _add(writer, "aex.head.ln1.bias", AH["model.layer_norm1.bias"])
    _add(writer, "aex.head.fc1.weight", AH["model.fc1.weight"]);         _add(writer, "aex.head.fc1.bias", AH["model.fc1.bias"])
    for i in range(ACT["head_blocks"]):
        P, Q = f"model.mlp_resnet_blocks.{i}.", f"aex.head.blk.{i}."
        _add(writer, Q + "ln.weight",  AH[P + "ffn.0.weight"]); _add(writer, Q + "ln.bias",  AH[P + "ffn.0.bias"])
        _add(writer, Q + "lin.weight", AH[P + "ffn.1.weight"]); _add(writer, Q + "lin.bias", AH[P + "ffn.1.bias"])
    _add(writer, "aex.head.ln2.weight", AH["model.layer_norm2.weight"]); _add(writer, "aex.head.ln2.bias", AH["model.layer_norm2.bias"])
    _add(writer, "aex.head.fc2.weight", AH["model.fc2.weight"]);         _add(writer, "aex.head.fc2.bias", AH["model.fc2.bias"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
