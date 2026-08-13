import sys
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policies.gr00t_n17 import GR00TN17PolicyPipeline
from utils.service import RobotInferenceServer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="nvidia/GR00T-N1.7-3B",
        help="HF model id / local checkpoint dir (default: the arch-specific preset)."
    )
    parser.add_argument(
        "--embodiment-tag", type=str, default="libero_sim",
        help="The embodiment tag to specify the input/output interface of the policy. "
             "For GR00T-N17, common values are 'libero_sim' (LIBERO Panda), 'simpler_env_google', "
             "'simpler_env_widowx', 'unitree_g1_full_body_with_waist_height_nav_cmd'. "
             "Make sure to use the same embodiment tag for both the server and the client to ensure compatibility."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="torch device, e.g. 'cuda', 'cuda:1', 'cpu' (default: cuda if available)."
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--n-action-steps", type=int, default=None,
        help="Actions replayed from each predicted chunk before re-querying the "
             "policy. Match vla.cpp's client-side --n-action-steps so the PyTorch "
             "reference and the GGUF port run the same control cadence."
    )
    args = parser.parse_args()


    policy = GR00TN17PolicyPipeline(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_id,
        device=args.device,
        n_action_steps=args.n_action_steps,
    )
    server = RobotInferenceServer(
        policy=policy,
        host=args.host,
        port=args.port,
        api_token=None
    )
    server.run()
