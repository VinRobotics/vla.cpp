import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim.libero # noqa: F401 to ensure the environments are registered with Gym
from utils.service import RobotInferenceClient
from utils.sim_adapters.libero import LIBEROSimAdapter

import time
import argparse
import gymnasium as gym


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="libero_object/task_0",
        help="The task to test on, select one of ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90'] "
            "with the corresponding task_id, e.g. 'libero_object/task_0'."
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
        "--view-mode",
        choices=["single-view", "multi-view"],
        default="multi-view",
        help="single-view: write one camera key, multi-view: side-by-side front+wrist views",
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
    parser.add_argument(
        "--out-name", type=str, default=None,
        help="Name of the per-model output subdir (default: the arch reported by "
             "the server). GR00T N1.6 and N1.7 both report arch 'gr00t', so pass "
             "this to keep their results apart."
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=None,
        help="Recorded in summary.txt for provenance only — the replay itself is "
             "server-side (see server/*.py --n-action-steps). Pass the same value "
             "you gave the server so the comparison report can verify the two "
             "stacks ran the same control cadence."
    )
    args = parser.parse_args()

    client = RobotInferenceClient(host=args.host, port=args.port, api_token=None)
    client = LIBEROSimAdapter(client=client)

    out_name = args.out_name or client.arch
    output_dir = Path(args.output_dir) / out_name / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # control_mode = "absolute" if client.arch == "gr00t" else "relative"  # GR00T-N1.6 is trained with absolute control, while the others are trained with relative control

    env = gym.make(
        args.task,
        seed=args.seed,
        video_fps=args.fps,
        output_video_dir=output_dir,
        video_view_mode=args.view_mode,
        # control_mode=control_mode, 
    )

    # Run Simulations
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
            # Get action from the policy
            t0 = time.time()
            action = client.get_action(obs)
            run_times.append(time.time() - t0)

            try:
                obs, reward, done, truncated, info = env.step(action)
            except ValueError as e:
                # robosuite raises this when the underlying env's `done` flag was
                # set on the previous step but the lerobot/LIBERO wrapper didn't
                # propagate it as terminated=True (so our auto-reset path didn't
                # fire). Skip this episode and move on.
                if "terminated episode" not in str(e):
                    raise
                print(f"- Episode aborted (env reported terminated mid-step): {e}")
                episode_aborted = True
                break
            #print(f"- Step {step_id}: reward={reward:.2f}, done={done}, truncated={truncated}, info={info}")
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
    # Same field layout as vla.cpp's eval/client/run_sim_client_direct.py, so
    # eval/collect_libero_results.py parses both stacks' outputs unchanged.
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Arch: {out_name}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"n_action_steps: {args.n_action_steps}\n")
        f.write("Task Description: " + env.task_description + "\n")
        f.write(f"Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})\n")
        f.write(f"Skipped (terminated mid-step): {skipped}/{args.n_episodes}\n")
        f.write(f"Average inference time per step: {avg_inf_ms} ms\n")

    print("*** All episodes completed.")
    print(f"- Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})")
    print(f"- Skipped (terminated mid-step): {skipped}/{args.n_episodes}")
    print(f"- Saved videos to: {output_dir.resolve()}")
