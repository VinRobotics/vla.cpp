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

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym

import sim.simpler  # noqa: F401  side-effect: registers gymnasium envs
from utils.sim_adapters.simpler import SimplerSimAdapter
from client.vla_cpp_client import VlaCppSimplerGr00tClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arch", choices=["gr00t_n1_6"], default="gr00t_n1_6",
        help="Policy arch (only gr00t_n1_6 / oxe_widowx is wired for SIMPLER so far).")
    parser.add_argument(
        "--task-id", type=str, default="oxe_widowx/widowx_spoon_on_towel",
        help="SimplerEnv task id, e.g. 'oxe_widowx/widowx_spoon_on_towel'.")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20,
        help="FPS for the per-episode output video recording.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for the SimplerEnv reset/init-state rollout (default: 42).")

    parser.add_argument("--vla-addr", type=str, default="tcp://localhost:5566",
        help="ZMQ address of vla-server (the C++ inference daemon).")
    parser.add_argument("--stats-json", type=str, required=True,
        help="Path to the bridge ckpt's statistics.json (per-embodiment "
             "state/action min/max/mean/std). Required for the oxe_widowx decode.")
    parser.add_argument("--embodiment", type=str, default="oxe_widowx",
        help="Embodiment key inside statistics.json (default: oxe_widowx). Must "
             "match the server's VLA_GR00T_EMBODIMENT.")
    parser.add_argument("--tokenizer", type=str, default=None,
        help="Override the gr00t_n1_6 preset's tokenizer (HF id or local dir).")
    parser.add_argument("--image-size", type=int, default=None,
        help="Override the vision-tower input size (default: preset 224).")
    parser.add_argument(
        "--n-action-steps", type=int, default=1,
        help="Open-loop replay length from each predicted chunk before "
             "re-querying. Default 1 == the PyTorch reference (re-predict every "
             "env step, use chunk step 0).")
    parser.add_argument("--recv-timeout-ms", type=int, default=120_000)
    args = parser.parse_args()

    client = VlaCppSimplerGr00tClient(
        vla_addr=args.vla_addr,
        stats_json=args.stats_json,
        embodiment=args.embodiment,
        n_action_steps=args.n_action_steps,
        recv_timeout_ms=args.recv_timeout_ms,
        tokenizer=args.tokenizer,
        image_size=args.image_size,
    )
    client = SimplerSimAdapter(client)

    output_dir = Path(args.output_dir) / args.arch / args.task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.task_id, output_video_dir=output_dir, video_fps=args.fps)

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
                success_count += info.get("success", 0.0)

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
        f.write(f"Task: {args.task_id}\n")
        f.write(f"Embodiment: {args.embodiment}\n")
        f.write(f"n_action_steps: {args.n_action_steps}\n")
        f.write(f"Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})\n")
        f.write(f"Skipped (terminated mid-step): {skipped}/{args.n_episodes}\n")
        f.write(f"Average inference time per step: {avg_inf_ms} ms\n")

    print("*** All episodes completed.")
    print(f"- Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})")
    print(f"- Skipped (terminated mid-step): {skipped}/{args.n_episodes}")
    print(f"- Saved videos to: {output_dir.resolve()}")
