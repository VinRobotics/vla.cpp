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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym

import sim.libero  # noqa: F401  side-effect: registers gymnasium envs
from client.vla_cpp_client import VlaCppClient, ARCH_PRESETS
from client.adapters import (
    LeRobotPipelineAdapter,
    Evo1PipelineAdapter,
    Gr00tPipelineAdapter,
    Gr00tN15PipelineAdapter,
)

ARCH_CHOICES = sorted(ARCH_PRESETS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", type=str, default="libero_object",
        help="LIBERO suite: one of ['libero_10', 'libero_spatial', "
             "'libero_object', 'libero_goal', 'libero_90'].",
    )
    parser.add_argument("--task-id", type=int, default=0,
        help="Task variation id within the suite.")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--view-mode",
        choices=["single-view", "multi-view"], default="multi-view",
        help="single-view: write one camera key; multi-view: side-by-side front+wrist.",
    )
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for the LIBERO env reset/init-state rollout (default: 42).")

    parser.add_argument("--arch", choices=ARCH_CHOICES, default="smolvla",
        help="Preprocessing preset + pipeline adapter. Also namespaces the output dir.")
    parser.add_argument("--vla-addr", type=str, default="tcp://localhost:5555",
        help="ZMQ address of vla-server (the C++ inference daemon).")
    parser.add_argument("--tokenizer", type=str, default=None,
        help="Override the arch preset's tokenizer (HF id or local dir). "
             "For BitVLA: must be the local ckpt dir.")
    parser.add_argument("--image-size", type=int, default=None,
        help="Override the arch preset's vision input size.")
    parser.add_argument("--max-state-dim", type=int, default=None,
        help="Override the state vector length sent to vla-server "
             "(default: arch preset).")
    parser.add_argument("--real-action-dim", type=int, default=7)
    parser.add_argument("--image-keys", nargs="+",
        default=["observation.images.image", "observation.images.image2"])
    parser.add_argument("--max-length", type=int, default=None,
        help="prompt token budget; default = arch preset's max_length (pi05=200) or 48.")
    parser.add_argument("--recv-timeout-ms", type=int, default=120_000,
        help="ZMQ receive timeout. π0 CPU inference is slow (~5–10 s/step with 2 views).")
    parser.add_argument(
        "--n-action-steps", type=int, default=1,
        help="How many actions to replay from each predicted chunk before "
             "re-querying vla-server. Mirrors lerobot's PI0Policy._action_queue / "
             "SmolVLAPolicy._action_queue. Defaults to 1 (re-predict every step - "
             "historical SmolVLA path). For π0 set to the checkpoint's "
             "`n_action_steps` (pi0_libero_base=10, pi0_libero_finetuned_v044=50); "
             "for BitVLA pass 8 (= NUM_ACTIONS_CHUNK).",
    )

    parser.add_argument(
        "--stats-json", type=str, default=None,
        help="[bitvla/gr00t_n1_6/gr00t_n1_7] path to dataset_statistics.json. Default for bitvla: "
             "<tokenizer>/dataset_statistics.json. For gr00t_n1_{6,7}, pass the canonical "
             "<model-dir>/experiment_cfg/dataset_statistics.json explicitly. Without "
             "it, gr00t_n1_7 returns the raw normalized [40, 132] chunk and gr00t_n1_6 "
             "returns the raw [50, 128] chunk (won't drive LIBERO).",
    )
    parser.add_argument(
        "--bitvla-unnorm-key", type=str, default=None,
        help="[bitvla/gr00t_n1_6/gr00t_n1_7] embodiment key inside dataset_statistics.json "
             "(default: auto-detect - bitvla picks the sole top-level key; "
             "gr00t_n1_7 prefers 'libero_sim'; gr00t_n1_6 prefers 'libero_panda').",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.arch / args.task / f"task_{args.task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = VlaCppClient(
        vla_addr=args.vla_addr,
        arch=args.arch,
        tokenizer_name=args.tokenizer,
        image_size=args.image_size,
        max_state_dim=args.max_state_dim,
        real_action_dim=args.real_action_dim,
        image_keys=args.image_keys,
        max_length=args.max_length,
        recv_timeout_ms=args.recv_timeout_ms,
        n_action_steps=args.n_action_steps,
        stats_json=args.stats_json,
        bitvla_unnorm_key=args.bitvla_unnorm_key,
    )
    if args.arch == "evo1":
        client = Evo1PipelineAdapter(client=client)
    elif args.arch == "gr00t_n1_5":

        client = Gr00tN15PipelineAdapter(client=client)
    elif args.arch in ("gr00t_n1_6", "gr00t_n1_7"):

        client = Gr00tPipelineAdapter(client=client)
    else:
        client = LeRobotPipelineAdapter(client=client)

    env = gym.make(
        f"{args.task}/task_{args.task_id}",
        seed=args.seed,
        video_fps=args.fps,
        output_video_dir=output_dir,
        video_view_mode=args.view_mode,
    )

    success_count, inference_times = 0.0, []
    skipped = 0
    for episode in range(args.n_episodes):
        print(f"*** Episode {episode + 1}/{args.n_episodes}")

        client.reset()
        obs, info = env.reset()
        run_times, step_id = [], 0
        episode_aborted = False
        done = False
        truncated = False
        reward = 0.0

        while True:
            t0 = time.time()
            action = client.get_action(obs)
            run_times.append(time.time() - t0)

            try:
                obs, reward, done, truncated, info = env.step(action)
            except ValueError as e:

                if "terminated episode" not in str(e):
                    raise
                print(f"- Episode aborted (env reported terminated mid-step): {e}")
                episode_aborted = True
                break
            step_id += 1

            if done or truncated or episode_aborted:
                avg_t = sum(run_times) / len(run_times)
                inference_times.append(avg_t)
                success_count += info.get("is_success", 0.0)

                print(f"- Episode finished after {step_id} steps.")
                print(f"- Final reward: {reward:.2f}")
                print(f"- Episode Information:\n{info}")
                print(f"- Average inference time per step: {round(1000 * avg_t, 2)} ms")
                break

        if episode_aborted:
            skipped += 1

    env.close()
    counted = max(1, args.n_episodes - skipped)
    avg_inf_ms = (round(1000 * sum(inference_times) / len(inference_times), 2)
                  if inference_times else 0.0)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Arch: {args.arch}\n")
        f.write(f"Task: {args.task}/task_{args.task_id}\n")
        f.write(f"n_action_steps: {args.n_action_steps}\n")
        f.write(f"Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})\n")
        f.write(f"Skipped (terminated mid-step): {skipped}/{args.n_episodes}\n")
        f.write(f"Average inference time per step: {avg_inf_ms} ms\n")

    print("*** All episodes completed.")
    print(f"- Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})")
    print(f"- Skipped (terminated mid-step): {skipped}/{args.n_episodes}")
    print(f"- Saved videos to: {output_dir.resolve()}")
