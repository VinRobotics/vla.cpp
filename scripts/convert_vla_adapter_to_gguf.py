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

from gguf_blocks import write_prismatic_lm, write_prismatic_tower
from gguf_common import (
    add_bf16,
    arg_parser,
    finish,
    kv_prefix,
    kv_u32,
    load_pt_module,
    load_safetensors,
    open_writer,
    read_text,
    require,
    resolve_out
)

ARCH = "vla_adapter"
KV = kv_prefix(ARCH)

DINO = dict(hidden=1024, layers=23, heads=16, head_dim=64, inter=4096)
SIG  = dict(hidden=1152, layers=26, heads=16, head_dim=72, inter=4304)
IMG, PATCH, NPATCH = 224, 14, 256
VIT_LN_EPS = 1e-6

LM = dict(
    hidden=896,
    layers=24,
    q_heads=14,
    kv_heads=2,
    head_dim=64,
    inter=4864,
    rope_theta=1_000_000.0,
    rms_eps=1e-6,
    vocab=151936
)
ACT = dict(
    chunk=8,
    action_dim=7,
    proprio_dim=8,
    num_tokens=64,
    head_blocks=24,
    head_heads=8,
    head_dim=112,
    head_rope_base=10000.0,
    ln_eps=1e-5
)
STOP_TOKEN_ID = 2
VDIM = DINO["hidden"] + SIG["hidden"]
PROJ_MID = 4 * VDIM

HEAD_PROJECTIONS = ("q_proj", "k_self", "v_self", "k_adapter", "v_adapter", "k_task", "v_task", "o_proj")

def _add_kv(writer, statistics_json: str) -> None:

    for tag, D in (("dino", DINO), ("sig", SIG)):
        kv_u32(writer, KV, {f"vit.{tag}.{k}": D[k] for k in ("hidden", "layers", "heads", "head_dim", "inter")})
    kv_u32(writer, KV, {"vit.image_size": IMG, "vit.patch_size": PATCH, "vit.n_patches": NPATCH})
    writer.add_float32(KV("vit.ln_eps"), VIT_LN_EPS)
    kv_u32(writer, KV, {"vit.proj_mid": PROJ_MID, "vit.vdim": VDIM})

    kv_u32(writer, KV, {f"lm.{k}": LM[k] for k in ("hidden", "layers", "q_heads", "kv_heads", "head_dim", "inter", "vocab")})
    writer.add_float32(KV("lm.rope_theta"), LM["rope_theta"])
    writer.add_float32(KV("lm.rms_eps"),    LM["rms_eps"])

    kv_u32(
        writer,
        KV,
        {f"action.{k}": ACT[k] for k in (
            "chunk",
            "action_dim",
            "proprio_dim",
            "num_tokens",
            "head_blocks",
            "head_heads",
            "head_dim"
        )}
    )
    writer.add_float32(KV("action.head_rope_base"), ACT["head_rope_base"])
    writer.add_float32(KV("action.ln_eps"),         ACT["ln_eps"])
    kv_u32(writer, KV, {"tokens.stop_id": STOP_TOKEN_ID})
    writer.add_string(KV("statistics_json"), statistics_json)

def main() -> int:
    ap = arg_parser(ARCH, "VLA-Adapter finetune dir (model.safetensors + action_head/proprio .pt sidecars)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)
    stats_path = ckpt / "dataset_statistics.json"
    require(stats_path)

    print(f"loading safetensors from {ckpt} ...")
    M = load_safetensors(ckpt)
    AH = load_pt_module(ckpt / "action_head--checkpoint.pt")
    PP = load_pt_module(ckpt / "proprio_projector--checkpoint.pt")
    statistics_json = read_text(stats_path)

    writer = open_writer(out, ARCH)
    _add_kv(writer, statistics_json)

    g = M.__getitem__
    write_prismatic_tower(writer, g, DINO["layers"], SIG["layers"])

    add_bf16(writer, "token_embd.weight",     g("language_model.model.embed_tokens.weight"))
    add_bf16(writer, "action_queries.weight", g("action_queries.weight"))
    write_prismatic_lm(writer, g, LM["layers"], qkv_bias=True)

    add_bf16(writer, "aex.head.ln1.weight", AH["model.layer_norm1.weight"]); add_bf16(writer, "aex.head.ln1.bias", AH["model.layer_norm1.bias"])
    add_bf16(writer, "aex.head.fc1.weight", AH["model.fc1.weight"]);         add_bf16(writer, "aex.head.fc1.bias", AH["model.fc1.bias"])
    add_bf16(writer, "aex.head.ln2.weight", AH["model.layer_norm2.weight"]); add_bf16(writer, "aex.head.ln2.bias", AH["model.layer_norm2.bias"])
    add_bf16(writer, "aex.head.fc2.weight", AH["model.fc2.weight"]);         add_bf16(writer, "aex.head.fc2.bias", AH["model.fc2.bias"])
    for i in range(ACT["head_blocks"]):
        P, Q = f"model.mlp_resnet_blocks.{i}.", f"aex.head.blk.{i}."
        for nm in HEAD_PROJECTIONS:
            add_bf16(writer, Q + nm + ".weight", AH[P + nm + ".weight"]); add_bf16(writer, Q + nm + ".bias", AH[P + nm + ".bias"])
        add_bf16(writer, Q + "ffn_ln.weight",  AH[P + "ffn.0.weight"]); add_bf16(writer, Q + "ffn_ln.bias",  AH[P + "ffn.0.bias"])
        add_bf16(writer, Q + "ffn_lin.weight", AH[P + "ffn.1.weight"]); add_bf16(writer, Q + "ffn_lin.bias", AH[P + "ffn.1.bias"])
        add_bf16(writer, Q + "gating", AH[P + "gating_factor"])

    add_bf16(writer, "aex.proprio.fc1.weight", PP["fc1.weight"]); add_bf16(writer, "aex.proprio.fc1.bias", PP["fc1.bias"])
    add_bf16(writer, "aex.proprio.fc2.weight", PP["fc2.weight"]); add_bf16(writer, "aex.proprio.fc2.bias", PP["fc2.bias"])

    return finish(writer, out, "  - combined GGUF (DINOv2 + SigLIP dual tower + Qwen2.5 0.5B + policy head)")

if __name__ == "__main__":
    raise SystemExit(main())
