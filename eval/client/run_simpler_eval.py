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
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim.simpler  # noqa: F401  side-effect: registers gymnasium envs
from utils.service import RobotInferenceClient
from utils.sim_adapters.simpler import SimplerSimAdapter

import time
import argparse
import gymnasium as gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id", type=str, default="oxe_google/google_robot_move_near",
        help="The simpler environment task id to test on. Select one of the registered simpler env ids, e.g. 'oxe_google/google_robot_move_near'."
    )
    parser.add_argument(
        "--n-episodes", type=int, default=30,
        help="The number of episodes to run for evaluation"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="The frames per second (FPS) for the output video recording of each episode"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="The directory to save the output videos. Each episode will be saved as a separate video file in this directory."
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Host of the inference server (run_server.py)."
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="Port of the inference server (run_server.py)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the LIBERO environment reset/init-state rollout (default: 42)."
    )
    args = parser.parse_args()

    client = RobotInferenceClient(host=args.host, port=args.port, api_token=None)
    client = SimplerSimAdapter(client)

    output_dir = Path(args.output_dir) / client.arch / args.task_id
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
        f.write(f"Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})\n")
        f.write(f"Skipped (terminated mid-step): {skipped}/{args.n_episodes}\n")
        f.write(f"Average inference time per step: {avg_inf_ms} ms\n")

    print("*** All episodes completed.")
    print(f"- Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})")
    print(f"- Skipped (terminated mid-step): {skipped}/{args.n_episodes}")
    print(f"- Saved videos to: {output_dir.resolve()}")
