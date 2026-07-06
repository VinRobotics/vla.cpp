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
"""Fold a PaliGemma mmproj GGUF into a pi05 checkpoint GGUF, producing one
self-contained file whose SigLIP tower is built in-tree by src/models/pi05.cpp
(no llama.cpp clip.cpp). Use this when you only have the released pi05 + mmproj
GGUFs; convert_pi05_to_gguf.py bakes the same tensors straight from safetensors.

The clip tensor names (v.blk.*, mm.input_projection.*) are remapped to the
in-tree vit.* / mm.proj.* scheme, and the SigLIP geometry is copied into pi05.* KV.
"""
import argparse
import sys

import gguf

# clip.cpp mmproj tensor name -> in-tree pi05 name
FIXED = {
    "v.patch_embd.weight": "vit.patch_embd.weight",
    "v.patch_embd.bias":   "vit.patch_embd.bias",
    "v.position_embd.weight": "vit.pos_embd",
    "v.post_ln.weight": "vit.post_ln.weight",
    "v.post_ln.bias":   "vit.post_ln.bias",
    "mm.input_projection.weight": "mm.proj.weight",
    "mm.input_projection.bias":   "mm.proj.bias",
}


def remap(name: str):
    if name in FIXED:
        return FIXED[name]
    if name.startswith("v.blk."):
        idx, sub = name[len("v.blk."):].split(".", 1)
        sub = sub.replace("attn_out", "attn_o").replace("ffn_up", "fc1").replace("ffn_down", "fc2")
        return f"vit.blk.{idx}.{sub}"
    return None  # audio / unrelated -> drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="pi05-*.gguf produced by convert_pi05_to_gguf.py")
    ap.add_argument("mmproj", help="mmproj-*.gguf (PaliGemma SigLIP tower)")
    ap.add_argument("out", help="output combined .gguf")
    args = ap.parse_args()

    ck = gguf.GGUFReader(args.ckpt)
    mm = gguf.GGUFReader(args.mmproj)

    if not ck.get_field("pi05.architecture"):
        sys.exit(f"{args.ckpt} is not a pi05 GGUF (pi05.architecture missing)")

    w = gguf.GGUFWriter(args.out, arch="pi05")

    # copy every ckpt KV verbatim (skip virtual + the arch GGUFWriter re-adds)
    for f in ck.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        w.add_key_value(f.name, f.contents(), f.types[0],
                        sub_type=f.types[1] if len(f.types) > 1 else None)

    # SigLIP geometry from the mmproj -> pi05.vit_* KV
    g = lambda k: mm.get_field(k).contents()
    vh, vl, vhd = g("clip.vision.embedding_length"), g("clip.vision.block_count"), g("clip.vision.attention.head_count")
    img, patch = g("clip.vision.image_size"), g("clip.vision.patch_size")
    eps = g("clip.vision.attention.layer_norm_epsilon")
    ntok = (img // patch) ** 2
    w.add_uint32("pi05.vit_hidden", int(vh))
    w.add_uint32("pi05.vit_layers", int(vl))
    w.add_uint32("pi05.vit_heads", int(vhd))
    w.add_uint32("pi05.image_size", int(img))
    w.add_uint32("pi05.patch_size", int(patch))
    w.add_uint32("pi05.n_img_tokens", int(ntok))
    w.add_float32("pi05.vit_ln_eps", float(eps))
    print(f"vision: hidden={vh} layers={vl} heads={vhd} image={img} patch={patch} tokens={ntok} eps={eps}")

    # ckpt tensors as-is
    for t in ck.tensors:
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    # remapped vision tensors
    n_vis = 0
    for t in mm.tensors:
        nn = remap(t.name)
        if nn is None:
            continue
        w.add_tensor(nn, t.data, raw_dtype=t.tensor_type)
        n_vis += 1
    print(f"copied {len(ck.tensors)} ckpt tensors + {n_vis} vision tensors")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
