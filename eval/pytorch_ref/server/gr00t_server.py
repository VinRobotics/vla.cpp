import sys
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policies.gr00t_n16 import GR00TN16PolicyPipeline
from utils.service import RobotInferenceServer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="nvidia/GR00T-N1.6-fractal",
        help="HF model id / local checkpoint dir (default: the arch-specific preset)."
    )
    parser.add_argument(
        "--embodiment-tag", type=str, default="oxe_google",
        help="The embodiment tag to specify the input/output interface of the policy. "
             "For GR00T-N16, the default is 'oxe_google' which corresponds to the observation/action space used in the OXE paper. "
             "Make sure to use the same embodiment tag for both the server and the client (env adapter) to ensure compatibility."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="torch device, e.g. 'cuda', 'cuda:1', 'cpu' (default: cuda if available). "
             "Use 'cpu' for pi0 if the GPU is too small (~3B params)."
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


    policy = GR00TN16PolicyPipeline(
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
