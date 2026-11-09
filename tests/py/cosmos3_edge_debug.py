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

"""Stage-wise comparison of the Cosmos3-Edge engine with lerobot's policy, for when the actions disagree.

`record` runs the policy on a CASE.npz of tests/py/cosmos3_edge_reference.py up to the first network call and
saves what the engine dumps under VLA_COSMOS3_DUMP_DIR: layer 0's and the last layer's prompt keys (normed and
rotated for the generation queries) and values, the generation stream's input embeddings, layer 0's rotated
queries and output, and the velocities of the first step. `compare` reads both and prints, per tensor, the
largest difference against the reference's magnitude. A tensor that matches to the reference's precision
(about 1e-2 scaled with a bf16 reference) and a later one that does not point at the stage in between.

    python tests/py/cosmos3_edge_debug.py record --ckpt <checkpoint dir> --lerobot-src <lerobot/src> \\
        --case CASE.npz --out REF.npz [--dtype bf16|f32]
    VLA_COSMOS3_DUMP_DIR=dump VLA_LIBRARY=build/libvla.so python tests/py/cosmos3_edge_parity.py --gguf M.gguf --case CASE.npz
    python tests/py/cosmos3_edge_debug.py compare --ref REF.npz --dump dump
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
rec_ap = sub.add_parser("record")
rec_ap.add_argument("--ckpt", type=Path, required=True)
rec_ap.add_argument("--lerobot-src", type=Path, default=None)
rec_ap.add_argument("--case", type=Path, required=True)
rec_ap.add_argument("--out", type=Path, required=True)
rec_ap.add_argument("--device", default="cuda")
rec_ap.add_argument("--dtype", default="bf16", choices=("bf16", "f32"))
cmp_ap = sub.add_parser("compare")
cmp_ap.add_argument("--ref", type=Path, required=True)
cmp_ap.add_argument("--dump", type=Path, required=True)
args = ap.parse_args()


def record():
    if args.lerobot_src is not None:
        sys.path.insert(0, str(args.lerobot_src))
        pkg = types.ModuleType("lerobot.policies")  # see cosmos3_edge_reference.py
        pkg.__path__ = [str(args.lerobot_src / "lerobot" / "policies")]
        sys.modules["lerobot.policies"] = pkg
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3EdgePolicy, FlowState, MoTAttention, MoTDecoderLayer

    device = torch.device(args.device)
    cfg = PreTrainedConfig.from_pretrained(str(args.ckpt))
    cfg.device, cfg.dtype = "cpu", torch.bfloat16 if args.dtype == "bf16" else torch.float32
    cfg.gradient_checkpointing = cfg.compile_model = False
    policy = Cosmos3EdgePolicy.from_pretrained(str(args.ckpt), config=cfg)
    policy.eval()
    net = policy.model.network
    del net.mot_backbone.model.visual, net.mot_backbone.model.projector, net.mot_backbone.lm_head, net.vae.decoder
    policy.to(device)

    case = np.load(args.case)
    ids = torch.from_numpy(case["input_ids"]).to(device)
    mask = torch.from_numpy(case["attention_mask"]).to(device)
    pv = (torch.from_numpy(case["canvas"]).float().to(device) / 127.5 - 1.0)[:, :, None]  # [1, 3, 1, H, W]
    frames = 1 + cfg.chunk_size // net.vae.config.scale_factor_temporal
    lat_h, lat_w = case["canvas"].shape[2] * 0 + 10, 20  # the content grid of the 192x320 canvas
    grid = torch.tensor([[frames, lat_h, lat_w]], device=device)
    spg = torch.full((1,), net.vae.config.scale_factor_temporal / cfg.fps, device=device)
    patch, z_dim = cfg.latent_patch_size, net.vae.config.z_dim
    ppf = (lat_h // patch) * (lat_w // patch)

    rec = {}
    layers = net.mot_backbone.model.language_model.layers
    last = len(layers) - 1
    orig_attn = MoTAttention.two_way_attention

    def two_way(self, query, key, value, query_gen, key_gen, value_gen, am, amg, key_und_for_gen=None, **kw):
        if "done" not in rec and self.layer_idx in (0, last):
            rec[f"l{self.layer_idx}_key_und_for_gen"] = key_und_for_gen.float().cpu().numpy()
            rec[f"l{self.layer_idx}_value"] = value.float().cpu().numpy()
            if self.layer_idx == 0:
                rec["l0_query_gen"] = query_gen.float().cpu().numpy()
        return orig_attn(self, query, key, value, query_gen, key_gen, value_gen, am, amg, key_und_for_gen=key_und_for_gen, **kw)

    orig_layer = MoTDecoderLayer.forward

    def layer_fwd(self, h, hg, pe, peg, **kw):
        if "done" not in rec and self is layers[0]:
            rec["gen_emb"] = hg.float().cpu().numpy()
        out = orig_layer(self, h, hg, pe, peg, **kw)
        if "done" not in rec and self is layers[0]:
            rec["l0_hidden_gen"] = out[1].float().cpu().numpy()
        return out

    MoTAttention.two_way_attention = two_way
    MoTDecoderLayer.forward = layer_fwd

    model = policy.model
    with torch.no_grad():
        patches = torch.zeros(1, frames * ppf, patch * patch * z_dim)
        patches[0, ppf:] = torch.from_numpy(case["video_noise"])
        vn = net.unpatchify(patches, torch.Size((1, z_dim, frames, lat_h, lat_w))).to(device)
        an = torch.zeros(1, cfg.chunk_size, cfg.max_action_dim)
        an[0, :, : cfg.original_action_dim] = torch.from_numpy(case["action_noise"])
        plan = model.build_sequence_plan(grid, cfg.chunk_size)
        pos = net.get_rope_index(ids, grid, spg, num_actions=cfg.chunk_size,
                                 action_start_frame_offset=plan.action_start_frame_offset, attention_mask=mask,
                                 mm_token_type_ids=torch.zeros_like(ids, dtype=torch.int))
        clean = model.encode_clean(pv, grid, torch.zeros(1, cfg.chunk_size, cfg.max_action_dim, device=device), plan)
        ones = torch.ones(1, device=device)
        noisy = model.add_noise(clean, FlowState(vn, an.to(device)), ones, ones, plan)
        t0 = model.make_scheduler(device).timesteps[0]
        vel = model.mask_velocity(model.predict_velocity(noisy, t0.expand(1), t0.expand(1), plan, ids, pos, attention_mask=mask), plan)
        rec["step0_vv"] = net.patchify(vel.video_latents)[0].reshape(frames * ppf, -1).float().cpu().numpy()
        rec["step0_va"] = vel.actions[0, :, : cfg.original_action_dim].float().cpu().numpy()
        rec["done"] = np.array(1)
    np.savez(args.out, **rec)
    print("saved", {k: v.shape for k, v in rec.items()})


def compare():
    ref = np.load(args.ref)
    n = ref["l0_key_und_for_gen"].shape[2]
    last = max(int(k[1:].split("_")[0]) for k in ref.files if k.startswith("l") and k[1].isdigit())
    n_gen, hidden = ref["gen_emb"].shape[1:]
    n_heads, _, head_dim = ref["l0_query_gen"].shape[1:]
    n_kv = ref["l0_value"].shape[1]

    def eng(name, shape):
        return np.fromfile(args.dump / f"{name}.f32", dtype=np.float32).reshape(shape)

    def cmp(label, a, b, rows=None):
        if rows is not None:
            a, b = a[rows], b[rows]
        a, b = a.astype(np.float64), b.astype(np.float64)
        d = np.abs(a - b).max()
        print(f"{label:30s} max|d| {d:.3e}   scaled {d / max(1e-9, np.abs(b).max()):.3e}")

    # ggml stores [head_dim, n, heads]: numpy [heads, n, head_dim], like torch's [1, heads, n, head_dim].
    cmp("prompt keys, layer 0", eng("kv_k0", (n_kv, n, head_dim)), ref["l0_key_und_for_gen"][0])
    cmp("prompt values, layer 0", eng("kv_v0", (n_kv, n, head_dim)), ref["l0_value"][0])
    cmp(f"prompt keys, layer {last}", eng("kv_k_last", (n_kv, n, head_dim)), ref[f"l{last}_key_und_for_gen"][0])
    cmp(f"prompt values, layer {last}", eng("kv_v_last", (n_kv, n, head_dim)), ref[f"l{last}_value"][0])
    cmp("generation embeddings", eng("step0_emb", (n_gen, hidden)), ref["gen_emb"][0])
    cmp("rotated queries, layer 0", eng("step0_q0", (n_heads, n_gen, head_dim)), ref["l0_query_gen"][0])
    cmp("hidden after layer 0", eng("step0_h0", (n_gen, hidden)), ref["l0_hidden_gen"][0])
    vv_ref = ref["step0_vv"]
    ppf = int(np.flatnonzero(np.abs(vv_ref).max(axis=1))[0])  # the condition frame's rows are zero in the reference
    cmp("video velocity, step 0 (noisy frames)", eng("step0_vv", vv_ref.shape), vv_ref, rows=slice(ppf, None))
    cmp("action velocity, step 0", eng("step0_va", ref["step0_va"].shape), ref["step0_va"])


record() if args.cmd == "record" else compare()
