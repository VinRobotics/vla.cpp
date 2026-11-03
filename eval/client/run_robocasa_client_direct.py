#!/usr/bin/env python3
# Copyright 2026 VinRobotics - Apache-2.0
"""Run RoboCasa365 episodes against vla-server; --check-env needs no model."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import time

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from client.vla_cpp_client import VlaCppClient
from utils.sim_adapters.robocasa import (
    RoboCasaSimAdapter,
    GR00TN16RoboCasaParser,
    SmolVLARoboCasaParser,
    load_smolvla_metadata, smolvla_client_options,
)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("smolvla", "gr00t_n1_6"), default="smolvla")
    parser.add_argument("--vla-addr", default="tcp://localhost:5555")
    parser.add_argument("--task", default="CloseFridge", help="Task name or comma-separated task names")
    parser.add_argument("--split", choices=("target", "pretrain", "all"), default="target")
    parser.add_argument("--task-prompt", choices=("task_id", "language"), default=None,
                        help="Policy instruction source: default task_id for SmolVLA (training labels), "
                             "language for GR00T (episode instruction)")
    parser.add_argument("--n-episodes", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=positive_int, default=None,
                        help="Override RoboCasa's registered task horizon")
    parser.add_argument("--n-action-steps", type=positive_int, default=None,
                        help="Default: SmolVLA 50, GR00T N1.6 8")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer ID or local tokenizer directory")
    parser.add_argument("--policy-path", type=Path, default=None,
                        help="Source checkpoint directory. SmolVLA: config.json and policy_preprocessor.json; "
                             "GR00T: statistics.json, processor_config.json, tokenizer/. "
                             "Default: pinned HF metadata for SmolVLA; weights/GR00T-N1.6 for GR00T")
    parser.add_argument("--stats-json", type=Path)
    parser.add_argument("--processor-config", type=Path)
    parser.add_argument("--image-size", type=positive_int, default=224,
                        help="GR00T vision side; must match the GGUF (224 or e.g. 252)")
    parser.add_argument("--camera-size", type=positive_int, default=None,
                        help="Simulator render side before conversion to 256 pixels; "
                             "default: SmolVLA 256, GR00T 512")
    parser.add_argument("--recv-timeout-ms", type=positive_int, default=120_000)
    parser.add_argument("--output-dir", type=Path, default=EVAL_ROOT.parent / "outputs" / "robocasa")
    parser.add_argument("--fps", type=positive_int, default=20)
    parser.add_argument("--view-mode", choices=("single-view", "multi-view"), default="multi-view")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--check-env", action="store_true", help="Check rendered reset and two zero-action steps")
    args = parser.parse_args(argv)
    if args.task_prompt is None:
        args.task_prompt = "task_id" if args.arch == "smolvla" else "language"
    args.tasks = [task.strip() for task in args.task.split(",") if task.strip()]
    if not args.tasks or any(not task.isidentifier() for task in args.tasks):
        parser.error("--task must contain RoboCasa class names, e.g. CloseFridge,OpenDrawer")
    args.tasks = list(dict.fromkeys(args.tasks))
    if args.camera_size is None:
        args.camera_size = 256 if args.arch == "smolvla" else 512
    if args.arch == "gr00t_n1_6" and not args.check_env:
        args.policy_path = args.policy_path or EVAL_ROOT.parent / "weights/GR00T-N1.6"
        args.stats_json = args.stats_json or args.policy_path / "statistics.json"
        args.processor_config = args.processor_config or args.policy_path / "processor_config.json"
        if args.tokenizer is None:
            directory = args.policy_path / "tokenizer"
            if not (directory / "tokenizer_config.json").is_file():
                parser.error("GR00T tokenizer missing; run python3 "
                             "eval/sim/robocasa/download_gr00t_tokenizer.py "
                             f"--output-dir {directory}, or pass --tokenizer")
            args.tokenizer = str(directory)
        for path in (args.stats_json, args.processor_config):
            if not path.is_file():
                parser.error(f"file not found: {path}")
    if args.n_action_steps is None:
        args.n_action_steps = 50 if args.arch == "smolvla" else 8
    max_replay = 50 if args.arch == "smolvla" else 16
    if args.n_action_steps > max_replay:
        parser.error(f"--n-action-steps must be <= {max_replay} for {args.arch}")
    return args


def run_episodes(env, client, args, output_dir):
    """Persist attempted episodes, including failures; never count aborts as success."""
    from utils.sim_adapters.robocasa import task_instruction
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    try:
        for episode in range(args.n_episodes):
            row = dict(episode=episode, seed=args.seed + episode, success=0, steps=0, aborted=0,
                       terminated=0, truncated=0, inference_seconds=0.0, inference_calls=0,
                       call_seconds=0.0, error="", prompt="")
            started = time.perf_counter()
            try:
                client.reset()
                obs, _ = env.reset(seed=row["seed"])
                row["prompt"] = task_instruction(obs, args.task_prompt)
                print(f"[episode {episode}] task_prompt={args.task_prompt} prompt={row['prompt']!r}", flush=True)
                while True:
                    before = client._step
                    t0 = time.perf_counter()
                    action = client.get_action(obs)
                    elapsed = time.perf_counter() - t0
                    row["inference_seconds"] += elapsed
                    if client._step != before:
                        row["inference_calls"] += 1
                        row["call_seconds"] += elapsed
                    obs, _, terminated, truncated, info = env.step(action)
                    row["steps"] += 1
                    if terminated or truncated:
                        row["success"] = int(bool(info["is_success"]))
                        row["terminated"], row["truncated"] = int(terminated), int(truncated)
                        break
            except Exception as exc:
                row["aborted"] = 1
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[episode {episode}] {row['error']}", file=sys.stderr, flush=True)
            row["duration_seconds"] = time.perf_counter() - started
            episodes.append(row)
            print(f"[episode {episode}] seed={row['seed']} steps={row['steps']} "
                  f"success={row['success']} aborted={row['aborted']}", flush=True)
            if row["aborted"]:
                break  # A broken simulator or server must not consume a whole sweep.
    finally:
        try:
            env.close()
        except Exception as exc:
            if not episodes:
                raise
            episodes[-1].update(success=0, aborted=1, error=f"Environment close failed: {exc}")
        summary = {
            "schema_version": 1, "arch": args.arch, "task": env.task, "split": args.split,
            "seed": args.seed, "requested_episodes": args.n_episodes,
            "task_prompt": args.task_prompt,
            "attempted_episodes": len(episodes), "n_action_steps": args.n_action_steps,
            "max_episode_steps": env.max_episode_steps,
            "camera_size": args.camera_size,
            "successes": sum(e["success"] for e in episodes),
            "aborted": sum(e["aborted"] for e in episodes),
            "steps": sum(e["steps"] for e in episodes),
            "inference_seconds": sum(e["inference_seconds"] for e in episodes),
            "inference_calls": sum(e["inference_calls"] for e in episodes),
            "call_seconds": sum(e["call_seconds"] for e in episodes),
            "tokenizer": args.tokenizer or "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "stats_json": str(args.stats_json) if args.stats_json else None,
            "processor_config": str(args.processor_config) if args.processor_config else None,
        }
        summary["versions"] = {}
        for package in ("robocasa", "robosuite", "mujoco", "gymnasium", "numpy", "torch", "transformers"):
            try:
                summary["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                summary["versions"][package] = "not installed"
        summary["complete"] = len(episodes) == args.n_episodes and not summary["aborted"]
        with (output_dir / "episodes.csv").open("w", newline="") as file:
            if episodes:
                writer = csv.DictWriter(file, fieldnames=list(episodes[0]))
                writer.writeheader()
                writer.writerows(episodes)
        temp = output_dir / "summary.json.tmp"
        temp.write_text(json.dumps(summary, indent=2) + "\n")
        temp.replace(output_dir / "summary.json")
        attempts = summary["attempted_episodes"]
        sr = summary["successes"] / attempts if attempts else 0.0
        avg_ms = 1000 * summary["inference_seconds"] / summary["steps"] if summary["steps"] else 0.0
        (output_dir / "summary.txt").write_text(
            f"Arch: {args.arch}\nTask: {env.task}\nSplit: {args.split}\n"
            f"n_action_steps: {args.n_action_steps}\n"
            f"Success rate: {sr:.2%} ({summary['successes']}/{attempts})\n"
            f"Aborted: {summary['aborted']}\nComplete: {summary['complete']}\n"
            f"Average inference time per step: {avg_ms:.2f} ms\n")
    return summary


def main(argv=None):
    args = parse_args(argv)
    # Imports are deliberately deferred so --help and CLI validation need no sim.
    from sim.robocasa.robocasa_env import RoboCasaEnv
    from utils.sim_adapters.robocasa import ACTION_DIMS, CAMERAS
    if args.check_env:
        import numpy as np
        for task in args.tasks:
            env = RoboCasaEnv(task, split=args.split, episode_length=args.max_episode_steps,
                             camera_size=args.camera_size)
            try:
                obs, _ = env.reset(seed=args.seed)
                assert env.observation_space.contains(obs), "Observation violates its declared space"
                for _ in range(2):
                    action = {f"action.{key}": np.zeros(dim, dtype=np.float32)
                              for key, dim in ACTION_DIMS.items()}
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                print(f"[check-env] {task}: reset/step/render OK; "
                      f"cameras={list(CAMERAS)} frame={env.render().shape}", flush=True)
            finally:
                env.close()
        return 0

    if args.arch == "smolvla":
        config, processors = load_smolvla_metadata(args.policy_path)
        client = VlaCppClient(args.vla_addr, **smolvla_client_options(
            config, processors, tokenizer=args.tokenizer, n_action_steps=args.n_action_steps,
            recv_timeout_ms=args.recv_timeout_ms))
        parser = SmolVLARoboCasaParser(config, processors, task_prompt=args.task_prompt)
    else:
        parser = GR00TN16RoboCasaParser(
            n_action_steps=args.n_action_steps, task_prompt=args.task_prompt,
            stats_json=args.stats_json, processor_config=args.processor_config,
            image_size=args.image_size)
        client = VlaCppClient(args.vla_addr, **parser.client_options(
            tokenizer=args.tokenizer, recv_timeout_ms=args.recv_timeout_ms))
    client = RoboCasaSimAdapter(client, parser)
    try:
        for task in args.tasks:
            output_dir = args.output_dir / args.arch / task
            env = RoboCasaEnv(task, split=args.split, episode_length=args.max_episode_steps,
                             camera_size=args.camera_size,
                             output_video_dir=None if args.no_video else output_dir,
                             video_fps=args.fps, view_mode=args.view_mode)
            summary = run_episodes(env, client, args, output_dir)
            print(f"[results] {output_dir}", flush=True)
            if not summary["complete"]:
                return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}\n"
              "Check RoboCasa assets, MUJOCO_GL/PYOPENGL_PLATFORM, and the server/model configuration.",
              file=sys.stderr)
        raise SystemExit(1)
