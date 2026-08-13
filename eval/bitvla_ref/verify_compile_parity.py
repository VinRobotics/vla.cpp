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

"""Check that torch.compile does not change BitVLA's actions.

A 3x speedup on a model whose whole cost is online quantization is plausible —
inductor fuses the per-BitLinear absmean/absmax elementwise chains that eager
runs as dozens of separate kernels — but "fast" is only interesting if the
numbers still match. This loads the policy once, records the action chunk for
a fixed observation in eager, then compiles the same instance and records it
again, and reports the difference.

Same process and same weights on both sides, so any delta is the compiled
kernels, not a reload.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_bitvla_pytorch import apply_compile  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--task-suite-name", default="libero_object")
    p.add_argument(
        "--task-label",
        default="pick up the black bowl between the plate and the ramekin and place it on the plate",
    )
    p.add_argument("--compile", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile-target", default="modules", choices=["modules", "predict_action"])
    p.add_argument("--n-obs", type=int, default=4, help="distinct observations to compare")
    p.add_argument(
        "--real-obs",
        action="store_true",
        help=(
            "draw observations from a real LIBERO episode instead of uniform "
            "noise; noise is out of distribution and exaggerates the delta"
        ),
    )
    p.add_argument("--task-id", type=int, default=0, help="[--real-obs] LIBERO task")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def real_observations(cfg, n: int, task_id: int, log=print):
    """Step a LIBERO episode and keep every Nth observation.

    Uses upstream's own prepare_observation, so the images have been through
    the same TF resize the policy sees at eval time.
    """
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from experiments.robot.libero.run_libero_eval_bitnet import prepare_observation
    from experiments.robot.robot_utils import get_image_resize_size

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    env.reset()
    obs = env.set_init_state(task_suite.get_task_init_states(task_id)[0])

    resize_size = get_image_resize_size(cfg)
    out = []
    # num_steps_wait dummy actions first, same as the eval driver, so the
    # objects have settled before anything is captured.
    for t in range(cfg.num_steps_wait + n * 3):
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            continue
        if (t - cfg.num_steps_wait) % 3 == 0:
            observation, _ = prepare_observation(obs, resize_size)
            out.append(observation)
            if len(out) == n:
                break
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    env.close()
    log(f"[verify] collected {len(out)} real observations from '{task_description}'")
    return out, task_description


def main() -> int:
    args = parse_args()

    oft_root = Path(__file__).resolve().parents[2] / "third_party" / "BitVLA" / "openvla-oft"
    os.chdir(oft_root)
    sys.path.insert(0, str(oft_root / "bitvla"))

    from experiments.robot.libero.run_libero_eval_bitnet import GenerateConfig, initialize_model
    from experiments.robot.robot_utils import get_action, set_seed_everywhere
    from bitvla.constants import (
        BITNET_ACTION_TOKEN_BEGIN_IDX,
        BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        BITNET_IGNORE_INDEX,
        BITNET_PROPRIO_PAD_IDX,
        BITNET_STOP_INDEX,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    set_seed_everywhere(args.seed)
    cfg = GenerateConfig(
        model_family="bitnet",
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        use_l1_regression=True,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        use_wandb=False,
        seed=args.seed,
    )
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model.set_constant(
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
        ignore_idx=BITNET_IGNORE_INDEX,
        action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=BITNET_STOP_INDEX,
    )

    task_label = args.task_label
    if args.real_obs:
        observations, task_label = real_observations(cfg, args.n_obs, args.task_id)
    else:
        rng = np.random.default_rng(args.seed)
        observations = [
            {
                "full_image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
                "wrist_image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
                "state": rng.uniform(-1.0, 1.0, size=(8,)).astype(np.float64),
            }
            for _ in range(args.n_obs)
        ]

    def run(obs):
        # get_action normalizes obs["state"] in place.
        o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        return np.stack(
            get_action(
                cfg, model, o, task_label,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
        )

    eager = [run(o) for o in observations]
    print(f"[verify] eager done ({len(eager)} observations, chunk shape {eager[0].shape})")

    apply_compile(model, action_head, args.compile, args.compile_target)
    run(observations[0])  # trigger compilation, discard
    compiled = [run(o) for o in observations]
    print(f"[verify] compile-{args.compile} done")

    worst_abs = 0.0
    worst_rel = 0.0
    for i, (a, b) in enumerate(zip(eager, compiled)):
        abs_d = float(np.max(np.abs(a - b)))
        scale = float(np.max(np.abs(a))) or 1.0
        rel_d = abs_d / scale
        worst_abs = max(worst_abs, abs_d)
        worst_rel = max(worst_rel, rel_d)
        print(f"  obs {i}: max|Δ| = {abs_d:.3e}   rel = {rel_d:.3e}")

    print()
    print(f"[verify] observations: {'real LIBERO' if args.real_obs else 'uniform noise'}")
    print(f"[verify] worst max|Δ| over {len(eager)} observations: {worst_abs:.3e}")
    print(f"[verify] worst relative                             : {worst_rel:.3e}")
    # There is no defensible tolerance here to pass/fail against. BitVLA
    # ternarizes weights from an absmean scale computed inside the forward, so
    # a 1-ulp change in that reduction moves a weight across a rounding
    # boundary and flips it between -1/0/+1 -- a discrete change, not a drift.
    # torch.compile reassociates those reductions, so some flips are expected
    # and the per-action delta says little on its own. Success rate on the real
    # suite is the arbiter; see run_libero_bitvla_pytorch.py --compile.
    print("[verify] reported, not gated: see the report for why a tolerance would be arbitrary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
