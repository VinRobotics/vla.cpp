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

"""Run BitVLA's own LIBERO driver, instrumented for latency and memory.

This does NOT reimplement the rollout: it imports upstream's
``experiments/robot/libero/run_libero_eval_bitnet.py`` and calls its
``eval_libero``, so the episode protocol, preprocessing and action
post-processing are upstream's byte for byte. What it adds is

  * a timer around ``get_action`` (the full policy query) and one around
    ``model.predict_action`` (the forward alone),
  * CUDA allocator snapshots taken right after the weights land on the device
    and again at the end of the run,
  * a JSON report next to the upstream text log.

Note that the GPU numbers here include MuJoCo's offscreen renderer, which
shares the device with the policy. ``bench_bitvla_pytorch.py`` runs the same
model with no simulator attached, and the difference between the two is the
renderer's share.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instrument import (  # noqa: E402
    MemoryProbe,
    Timer,
    parameter_report,
    print_summary,
    write_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--task-suite-name", default="libero_object")
    p.add_argument(
        "--num-trials-per-task",
        type=int,
        default=1,
        help="episodes per task; the suite has 10 tasks, so 1 gives a 10-episode sanity run",
    )
    p.add_argument("--output", default="", help="path for the JSON report")
    p.add_argument("--log-dir", default="", help="upstream text-log directory")
    p.add_argument("--run-note", default="latency", help="upstream info_in_path tag")
    p.add_argument("--save-videos", action="store_true", help="keep the MP4 rollouts")
    p.add_argument(
        "--warmup-calls",
        type=int,
        default=0,
        help=(
            "policy queries to discard before the timed window "
            "(0 = 5 eager / 20 compiled; covers lazy CUDA init and inductor codegen)"
        ),
    )
    p.add_argument(
        "--compile",
        default="off",
        choices=["off", "default", "reduce-overhead", "max-autotune"],
        help="torch.compile inductor mode; 'off' runs eager",
    )
    p.add_argument(
        "--compile-target",
        default="modules",
        choices=["modules", "predict_action"],
        help="see bench_bitvla_pytorch.py",
    )
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve caller-relative paths BEFORE chdir, or they land under
    # third_party/BitVLA/openvla-oft instead of where the caller meant.
    args.output = str(Path(args.output).resolve()) if args.output else ""
    args.log_dir = str(Path(args.log_dir).resolve()) if args.log_dir else ""

    # check_model_logic_mismatch() resolves "./bitvla" against the CWD.
    repo_root = Path(__file__).resolve().parents[2]
    oft_root = repo_root / "third_party" / "BitVLA" / "openvla-oft"
    os.chdir(oft_root)
    # See bench_bitvla_pytorch.py: `configuration_bit_vla` is imported as a
    # top-level module by the BitVLA modeling file.
    sys.path.insert(0, str(oft_root / "bitvla"))

    import experiments.robot.libero.run_libero_eval_bitnet as ev
    from bench_bitvla_pytorch import apply_compile, dynamo_stats

    compiling = args.compile != "off"
    warmup_calls = args.warmup_calls if args.warmup_calls > 0 else (20 if compiling else 5)

    probe = MemoryProbe()
    probe.start()
    t_get_action = Timer("get_action")
    t_forward = Timer("predict_action")
    state = {
        "params": None,
        "warmup_left": warmup_calls,
        "reset_done": False,
        "compiled": [],
    }

    # eval_libero() builds the model itself, so hook the constructor to grab
    # the post-load allocator snapshot and to wrap predict_action.
    orig_initialize_model = ev.initialize_model

    def initialize_model(cfg):
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            orig_initialize_model(cfg)
        )
        probe.mark_weights_loaded()
        state["params"] = parameter_report(
            {"vla": model, "action_head": action_head, "proprio_projector": proprio_projector}
        )
        if compiling:
            state["compiled"] = apply_compile(
                model, action_head, args.compile, args.compile_target
            )
        # Wrap after compiling: apply_compile may replace predict_action itself.
        model.predict_action = t_forward.wrap(model.predict_action)
        return model, action_head, proprio_projector, noisy_action_projector, processor

    ev.initialize_model = initialize_model

    # The first few queries pay lazy cuBLAS/cuDNN init and kernel autotuning;
    # keeping them would put a multi-hundred-ms outlier in every percentile.
    orig_get_action = ev.get_action
    timed_get_action = t_get_action.wrap(orig_get_action)

    def get_action(*a, **kw):
        out = timed_get_action(*a, **kw)
        if state["warmup_left"] > 0:
            state["warmup_left"] -= 1
            if state["warmup_left"] == 0:
                dropped = (t_get_action.reset(), t_forward.reset())
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                state["reset_done"] = True
                print(f"[instrument] warmup over, dropped {dropped[0]} samples")
        return out

    ev.get_action = get_action

    if not args.save_videos:
        ev.save_rollout_video = lambda *a, **kw: None

    log_dir = args.log_dir or str(repo_root / "outputs" / "bitvla_pytorch" / "logs")

    # eval_libero is draccus-wrapped, so it parses sys.argv itself.
    sys.argv = [
        "run_libero_eval_bitnet.py",
        "--pretrained_checkpoint", str(args.pretrained_checkpoint),
        "--task_suite_name", args.task_suite_name,
        "--model_family", "bitnet",
        "--num_trials_per_task", str(args.num_trials_per_task),
        "--use_wandb", "False",
        "--local_log_dir", log_dir,
        "--info_in_path", args.run_note,
        "--seed", str(args.seed),
    ]

    success_rate = ev.eval_libero()
    probe.finish()

    payload = {
        "run": {
            "mode": "libero_rollout",
            "checkpoint": str(args.pretrained_checkpoint),
            "task_suite": args.task_suite_name,
            "num_trials_per_task": args.num_trials_per_task,
            "warmup_calls": warmup_calls,
            "warmup_applied": state["reset_done"],
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "dtype": "bfloat16",
            "variant": f"compile-{args.compile}" if compiling else "eager",
            "compile_target": args.compile_target if compiling else None,
            "compiled_modules": state["compiled"],
            "dynamo": dynamo_stats() if compiling else {},
        },
        "success_rate": success_rate,
        "latency": {
            "get_action": t_get_action.stats(),
            "predict_action": t_forward.stats(),
        },
        "memory": probe.report(),
        "parameters": state["params"] or {},
    }
    print(f"\n[instrument] LIBERO {args.task_suite_name} success rate: {success_rate * 100:.1f}%")
    print_summary(payload)
    if args.output:
        write_report(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
