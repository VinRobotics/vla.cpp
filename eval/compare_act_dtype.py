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

"""Action-for-action comparison of two vla-server configurations.

Used to check what a numerics change (BF16 activations, flash attention, ...)
actually does to the predicted actions, independently of whether it changes a
LIBERO episode's outcome.

The two servers must see byte-identical inputs, which a live environment cannot
guarantee: the second server's actions steer the sim somewhere else and every
later observation diverges for reasons that have nothing to do with the kernel.
So `record` drives the env once and dumps the observation stream, and `replay`
feeds that fixed stream to any server. VLA_FIXED_NOISE_SEED pins the
flow-matching noise on top (see VlaCppClient._maybe_add_fixed_noise), leaving
the arithmetic as the only thing that differs.

  record: python compare_act_dtype.py record --addr ... --out ref.npz
  replay: python compare_act_dtype.py replay --addr ... --obs ref.npz --out b.npz
  diff:   python compare_act_dtype.py diff a.npz b.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "client"))


def _client(addr: str, arch: str, tokenizer: str | None):
    from client.vla_cpp_client import VlaCppClient
    kw = {"arch": arch, "n_action_steps": 1}
    if tokenizer:
        kw["tokenizer_name"] = tokenizer
    return VlaCppClient(addr, **kw)


def _adapter(arch: str, client):
    from client.adapters import Evo1PipelineAdapter
    if arch != "evo1":
        raise SystemExit(f"only evo1 is wired up here, got {arch!r}")
    return Evo1PipelineAdapter(client=client)


def _obs_key(obs: dict) -> dict:
    """Keep only what the evo1 request path reads, as plain arrays."""
    return {
        "image": np.stack([np.asarray(im, dtype=np.uint8) for im in obs["image"]]),
        "image_mask": np.asarray(obs.get("image_mask", [1] * len(obs["image"])), dtype=np.int32),
        "state": np.asarray(obs["state"], dtype=np.float32),
        "prompt": str(obs.get("prompt", "")),
    }


def cmd_record(args) -> int:
    import gymnasium as gym
    import sim.libero  # noqa: F401  registers the envs

    client = _client(args.addr, args.arch, args.tokenizer)
    adapter = _adapter(args.arch, client)
    env = gym.make(f"{args.task}/task_{args.task_id}", video_fps=30,
                   output_video_dir=str(args.out.parent / "_cmp_videos"),
                   video_view_mode="single-view")
    obs, _ = env.reset(seed=args.seed)

    frames, actions = [], []
    for i in range(args.n_steps):
        # record the model-space observation, which is what replay must reproduce
        parsed = adapter.parse_observation(obs)
        frames.append(_obs_key(parsed))
        chunk = client.get_action(parsed)
        actions.append(np.asarray(chunk, dtype=np.float32))
        try:
            obs, _r, done, trunc, _ = env.step(adapter.parse_action(chunk))
        except ValueError:
            obs, _ = env.reset(seed=args.seed)
            continue
        if done or trunc:
            obs, _ = env.reset(seed=args.seed)

    np.savez_compressed(
        args.out,
        images=np.stack([f["image"] for f in frames]),
        image_masks=np.stack([f["image_mask"] for f in frames]),
        states=np.stack([f["state"] for f in frames]),
        prompts=np.array([f["prompt"] for f in frames]),
        actions=np.stack(actions),
    )
    print(f"recorded {len(frames)} steps -> {args.out}")
    return 0


def cmd_replay(args) -> int:
    d = np.load(args.obs, allow_pickle=True)
    client = _client(args.addr, args.arch, args.tokenizer)

    actions = []
    for i in range(len(d["images"])):
        obs = {
            "image": list(d["images"][i]),
            "image_mask": list(d["image_masks"][i]),
            "state": d["states"][i],
            "prompt": str(d["prompts"][i]),
        }
        actions.append(np.asarray(client.get_action(obs), dtype=np.float32))

    np.savez_compressed(args.out, actions=np.stack(actions))
    print(f"replayed {len(actions)} steps -> {args.out}")
    return 0


def cmd_diff(args) -> int:
    a = np.load(args.a)["actions"].astype(np.float64)
    b = np.load(args.b)["actions"].astype(np.float64)
    if a.shape != b.shape:
        print(f"shape mismatch: {a.shape} vs {b.shape}")
        return 1
    d = np.abs(a - b)
    scale = np.abs(a).mean()
    print(f"steps        : {a.shape[0]}  action dim {a.shape[1:]}")
    print(f"mean |a|     : {scale:.6f}")
    print(f"max  |diff|  : {d.max():.6e}")
    print(f"mean |diff|  : {d.mean():.6e}")
    print(f"p99  |diff|  : {np.percentile(d, 99):.6e}")
    print(f"rel mean     : {d.mean() / scale:.3e}")
    print(f"exact equal  : {bool(np.array_equal(a, b))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("record", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--addr", required=True)
        p.add_argument("--arch", default="evo1")
        p.add_argument("--tokenizer", default=None)
        p.add_argument("--out", type=Path, required=True)
        if name == "record":
            p.add_argument("--task", default="libero_object")
            p.add_argument("--task-id", type=int, default=0)
            p.add_argument("--n-steps", type=int, default=20)
            p.add_argument("--seed", type=int, default=42)
        else:
            p.add_argument("--obs", type=Path, required=True)

    p = sub.add_parser("diff")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)

    args = ap.parse_args()
    return {"record": cmd_record, "replay": cmd_replay, "diff": cmd_diff}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
