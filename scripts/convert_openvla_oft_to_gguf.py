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

ARCH = "openvla_oft"
KV = kv_prefix(ARCH)

DINO = dict(hidden=1024, layers=23, heads=16, head_dim=64, inter=4096)
SIG  = dict(hidden=1152, layers=26, heads=16, head_dim=72, inter=4304)
IMG, PATCH, NPATCH = 224, 14, 256
VIT_LN_EPS = 1e-6

LM = dict(
    hidden=4096,
    layers=32,
    q_heads=32,
    kv_heads=32,
    head_dim=128,
    inter=11008,
    rope_theta=10000.0,
    rms_eps=1e-6,
    vocab=32064
)
ACT = dict(
    chunk=8,
    action_dim=7,
    proprio_dim=8,
    head_hidden=4096,
    head_blocks=2,
    head_ln_eps=1e-5
)
NUM_IMAGES = 2
STOP_TOKEN_ID = 2
EMPTY_TOKEN_ID = 29871
VDIM = DINO["hidden"] + SIG["hidden"]
PROJ_MID = 4 * VDIM

def _add_kv(writer, statistics_json: str) -> None:

    for tag, D in (("dino", DINO), ("sig", SIG)):
        kv_u32(writer, KV, {f"vit.{tag}.{k}": D[k] for k in ("hidden", "layers", "heads", "head_dim", "inter")})
    kv_u32(writer, KV, {"vit.image_size": IMG, "vit.patch_size": PATCH, "vit.n_patches": NPATCH})
    writer.add_float32(KV("vit.ln_eps"), VIT_LN_EPS)
    kv_u32(writer, KV, {"vit.proj_mid": PROJ_MID, "vit.vdim": VDIM, "vit.num_images": NUM_IMAGES})

    kv_u32(writer, KV, {f"lm.{k}": LM[k] for k in ("hidden", "layers", "q_heads", "kv_heads", "head_dim", "inter", "vocab")})
    writer.add_float32(KV("lm.rope_theta"), LM["rope_theta"])
    writer.add_float32(KV("lm.rms_eps"),    LM["rms_eps"])

    kv_u32(writer, KV, {f"action.{k}": ACT[k] for k in ("chunk", "action_dim", "proprio_dim", "head_hidden", "head_blocks")})
    writer.add_float32(KV("action.head_ln_eps"), ACT["head_ln_eps"])
    kv_u32(writer, KV, {"tokens.stop_id": STOP_TOKEN_ID, "tokens.empty_id": EMPTY_TOKEN_ID})
    writer.add_string(KV("statistics_json"), statistics_json)

def main() -> int:
    ap = arg_parser(ARCH, "OpenVLA-OFT finetune snapshot dir (sharded safetensors + action_head/proprio .pt sidecars)")
    ap.add_argument(
        "--action-head",
        type=Path,
        default=None,
        help="action_head .pt (default: action_head--*_checkpoint.pt in ckpt)"
    )
    ap.add_argument(
        "--proprio",
        type=Path,
        default=None,
        help="proprio_projector .pt (default: proprio_projector--*_checkpoint.pt in ckpt)"
    )
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = resolve_out(args, ckpt, ARCH)

    ah_path = args.action_head or next(ckpt.glob("action_head--*checkpoint.pt"))
    pp_path = args.proprio     or next(ckpt.glob("proprio_projector--*checkpoint.pt"))
    stats_path = ckpt / "dataset_statistics.json"
    require(stats_path)

    print(f"loading merged safetensors from {ckpt} ...")
    M = load_safetensors(ckpt)
    AH = load_pt_module(ah_path)
    PP = load_pt_module(pp_path)
    statistics_json = read_text(stats_path)
    print(f"  action_head: {ah_path.name}   proprio: {pp_path.name}")

    writer = open_writer(out, ARCH)
    _add_kv(writer, statistics_json)

    g = M.__getitem__
    write_prismatic_tower(writer, g, DINO["layers"], SIG["layers"])

    add_bf16(writer, "token_embd.weight", g("language_model.model.embed_tokens.weight"))
    write_prismatic_lm(writer, g, LM["layers"])

    add_bf16(writer, "aex.proprio.fc1.weight", PP["fc1.weight"]); add_bf16(writer, "aex.proprio.fc1.bias", PP["fc1.bias"])
    add_bf16(writer, "aex.proprio.fc2.weight", PP["fc2.weight"]); add_bf16(writer, "aex.proprio.fc2.bias", PP["fc2.bias"])

    add_bf16(writer, "aex.head.ln1.weight", AH["model.layer_norm1.weight"]); add_bf16(writer, "aex.head.ln1.bias", AH["model.layer_norm1.bias"])
    add_bf16(writer, "aex.head.fc1.weight", AH["model.fc1.weight"]);         add_bf16(writer, "aex.head.fc1.bias", AH["model.fc1.bias"])
    for i in range(ACT["head_blocks"]):
        P, Q = f"model.mlp_resnet_blocks.{i}.", f"aex.head.blk.{i}."
        add_bf16(writer, Q + "ln.weight",  AH[P + "ffn.0.weight"]); add_bf16(writer, Q + "ln.bias",  AH[P + "ffn.0.bias"])
        add_bf16(writer, Q + "lin.weight", AH[P + "ffn.1.weight"]); add_bf16(writer, Q + "lin.bias", AH[P + "ffn.1.bias"])
    add_bf16(writer, "aex.head.ln2.weight", AH["model.layer_norm2.weight"]); add_bf16(writer, "aex.head.ln2.bias", AH["model.layer_norm2.bias"])
    add_bf16(writer, "aex.head.fc2.weight", AH["model.fc2.weight"]);         add_bf16(writer, "aex.head.fc2.bias", AH["model.fc2.bias"])

    return finish(writer, out, "  - combined GGUF (DINOv2 + SigLIP dual tower + Llama-2 7B + L1 action head)")

if __name__ == "__main__":
    raise SystemExit(main())
