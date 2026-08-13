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

"""Sim-free latency + memory benchmark for the upstream PyTorch BitVLA policy.

Loads the model exactly the way ``run_libero_eval_bitnet.py`` does (same
``initialize_model``, same action head and proprio projector, same processor),
then replays one fixed LIBERO-object observation through ``get_action`` N
times. Dropping the simulator removes MuJoCo's EGL rendering from the GPU
memory accounting and its stepping from the wall clock, so the numbers here
are the policy's alone.

The companion script ``run_libero_eval_bitnet_instrumented.py`` measures the
same two timers inside a real rollout; the two should agree on latency, and
differ on GPU memory by whatever the renderer holds.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
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
        "--task-label",
        default="pick up the black bowl between the plate and the ramekin and place it on the plate",
        help="Instruction string; only its token count matters for latency.",
    )
    p.add_argument("--n-steps", type=int, default=200, help="timed get_action calls")
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="untimed calls before the window (0 = 10 eager / 20 compiled)",
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
        help=(
            "'modules' compiles the vision tower and the BitNet LM separately, "
            "matching what the other reference policies in eval/pytorch_ref do; "
            "'predict_action' compiles the whole policy entry point"
        ),
    )
    p.add_argument("--output", default="", help="path for the JSON report")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def load_compile_helper():
    """Load eval/pytorch_ref/policies/torch_compile.py without its package.

    Importing it as ``policies.torch_compile`` would run the package __init__,
    which claims HUGGINGFACE_HUB_CACHE and creates a weights dir. The module
    itself only needs torch, so load it straight from the file and keep the two
    stacks on one definition of "compiled".
    """
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "pytorch_ref" / "policies" / "torch_compile.py"
    )
    spec = importlib.util.spec_from_file_location("vla_torch_compile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_compile(model, action_head, mode: str, target: str) -> list:
    """Wrap the compile targets in place; returns the labels that were wrapped."""
    helper = load_compile_helper()
    os.environ["VLA_TORCH_COMPILE"] = "1"
    os.environ["VLA_TORCH_COMPILE_MODE"] = mode

    if target == "predict_action":
        # predict_action ends in .cpu().numpy() and does .item() on token
        # counts, so this graph-breaks hard; kept as a variant because it is
        # the entry point a user would reach for first.
        model.predict_action = helper.maybe_compile(
            model.predict_action, tag="bitvla.predict_action"
        )
        return ["predict_action"]

    # BitVLA never routes pixels through LlavaForConditionalGeneration.forward
    # (predict_action calls get_image_features itself and passes pixel_values
    # =None downstream), so the tower and the LM have to be compiled
    # separately -- compiling the parent forward would miss the vision half.
    model.vision_tower = helper.maybe_compile(model.vision_tower, tag="bitvla.vision_tower")
    model.language_model = helper.maybe_compile(model.language_model, tag="bitvla.language_model")
    return ["vision_tower", "language_model"]


def dynamo_stats() -> dict:
    """How much of the model dynamo actually captured.

    maybe_compile() sets ``suppress_errors``, which turns an untraceable region
    into a graph break instead of a hard failure. That is the right default for
    measuring what a user gets, but it means "compiled" can quietly mean
    "partly compiled" -- so count the breaks and report them.
    """
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return {}
    breaks = dict(counters.get("graph_break", {}))
    return {
        "graph_breaks_total": sum(breaks.values()),
        "graph_breaks_distinct": len(breaks),
        "graph_break_reasons": dict(sorted(breaks.items(), key=lambda kv: -kv[1])[:10]),
        "frames_ok": counters.get("frames", {}).get("ok", 0),
        "frames_total": counters.get("frames", {}).get("total", 0),
    }


def main() -> int:
    args = parse_args()

    # Resolve caller-relative paths BEFORE chdir, or they land under
    # third_party/BitVLA/openvla-oft instead of where the caller meant.
    args.output = str(Path(args.output).resolve()) if args.output else ""

    # experiments.robot.* resolve through the editable openvla-oft install, but
    # check_model_logic_mismatch() reads "./bitvla" relative to the CWD, so the
    # loader only finds the reference modeling files from the repo root.
    oft_root = Path(__file__).resolve().parents[2] / "third_party" / "BitVLA" / "openvla-oft"
    os.chdir(oft_root)
    # bitvla/model/bitvla_for_action_prediction.py imports `configuration_bit_vla`
    # as a top-level module. Upstream's `pip install -e bitvla/` makes that work
    # by rooting the distribution at bitvla/; the modern editable finder maps
    # only the subpackages, so put the directory on the path explicitly.
    sys.path.insert(0, str(oft_root / "bitvla"))

    from experiments.robot.libero.run_libero_eval_bitnet import (
        GenerateConfig,
        initialize_model,
    )
    from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
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
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        use_wandb=False,
        seed=args.seed,
    )

    probe = MemoryProbe()
    probe.start()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model.set_constant(
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
        ignore_idx=BITNET_IGNORE_INDEX,
        action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=BITNET_STOP_INDEX,
    )
    probe.mark_weights_loaded()

    params = parameter_report(
        {
            "vla": model,
            "action_head": action_head,
            "proprio_projector": proprio_projector,
        }
    )

    compiled_targets = []
    if args.compile != "off":
        compiled_targets = apply_compile(model, action_head, args.compile, args.compile_target)

    # Time the forward on its own as well as the whole policy call: vla.cpp's
    # server-reported latency excludes host-side image preprocessing, and here
    # that preprocessing is a TensorFlow JPEG round-trip plus a lanczos3
    # resize, which is not a rounding error.
    t_get_action = Timer("get_action")
    t_forward = Timer("predict_action")
    model.predict_action = t_forward.wrap(model.predict_action)
    timed_get_action = t_get_action.wrap(get_action)

    resize_size = get_image_resize_size(cfg)
    rng = np.random.default_rng(args.seed)
    # env_img_res=256 in the LIBERO driver; resize_image_for_policy() takes it
    # down to 224, so feed the benchmark the same 256x256 the sim would emit.
    observation = {
        "full_image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
        "state": rng.uniform(-1.0, 1.0, size=(8,)).astype(np.float64),
    }
    print(f"[bench] resize_size={resize_size}, chunk={NUM_ACTIONS_CHUNK}, unnorm_key={cfg.unnorm_key}")

    def one_call():
        # get_action() mutates obs["state"] in place (normalize_proprio), so
        # hand it a fresh copy each call or the proprio drifts across steps.
        obs = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in observation.items()}
        return timed_get_action(
            cfg,
            model,
            obs,
            args.task_label,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )

    # Compilation happens on the first call and CUDA graphs need a few more to
    # settle, so a compiled variant gets a longer warmup. Every warmup sample is
    # discarded, which is what keeps the tens of seconds of inductor codegen out
    # of the measured window.
    warmup = args.warmup if args.warmup > 0 else (20 if compiled_targets else 10)
    t0_warm = time.perf_counter()
    for _ in range(warmup):
        actions = one_call()
    print(
        f"[bench] warmup done ({warmup} calls in {time.perf_counter() - t0_warm:.1f}s); "
        f"chunk length={len(actions)}"
    )
    t_get_action.reset()
    t_forward.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i in range(args.n_steps):
        one_call()
        if (i + 1) % 50 == 0:
            print(f"[bench] {i + 1}/{args.n_steps}")

    probe.finish()

    payload = {
        "run": {
            "mode": "sim_free",
            "checkpoint": str(args.pretrained_checkpoint),
            "task_suite": args.task_suite_name,
            "n_steps": args.n_steps,
            "warmup": warmup,
            "chunk": NUM_ACTIONS_CHUNK,
            "variant": f"compile-{args.compile}" if compiled_targets else "eager",
            "compile_target": args.compile_target if compiled_targets else None,
            "compiled_modules": compiled_targets,
            "dynamo": dynamo_stats() if compiled_targets else {},
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "dtype": "bfloat16",
        },
        "latency": {
            "get_action": t_get_action.stats(),
            "predict_action": t_forward.stats(),
        },
        "memory": probe.report(),
        "parameters": params,
    }
    print_summary(payload)
    if args.output:
        write_report(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
