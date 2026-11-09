"""Cosmos3-Edge through libvla against the PyTorch reference, one case.

CASE.npz comes from tests/py/cosmos3_edge_reference.py (lerobot's policy on the same views, prompt and noise).
The engine gets the views, the token ids and the action noise through libvla, the video noise through
VLA_COSMOS3_VIDEO_NOISE, and dumps its canvas and latent for the stage checks.

    VLA_LIBRARY=build/libvla.so python tests/py/cosmos3_edge_parity.py --gguf M.gguf --case CASE.npz
"""
import argparse, os, sys, tempfile
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "bindings", "python"))

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--case", required=True, help="CASE.npz of tests/py/cosmos3_edge_reference.py")
ap.add_argument("--save", help="write the engine's actions (both runs) to this .npz")
args = ap.parse_args()

tmp = tempfile.mkdtemp()
case = np.load(args.case)
vn_path, canvas_path, z0_path = (os.path.join(tmp, f) for f in ("vn.f32", "canvas.f32", "z0.f32"))
np.ascontiguousarray(case["video_noise"], dtype=np.float32).tofile(vn_path)
os.environ["VLA_COSMOS3_VIDEO_NOISE"] = vn_path
os.environ["VLA_COSMOS3_DUMP_CANVAS"] = canvas_path
os.environ["VLA_COSMOS3_DUMP_Z0"] = z0_path

import vla_cpp

n = int(case["attention_mask"].sum())
tokens = case["input_ids"][0, :n].tolist()
noise = case["action_noise"].astype(np.float32).ravel()
views = [np.ascontiguousarray(v, dtype=np.float32) for v in case["views"]]

model = vla_cpp.load(args.gguf)
actions = model.predict(views, tokens, noise=noise, pixel_format=vla_cpp.PIXEL_F32_RGB_01, timing=1)
st = model.last_stats()
print(f"actions {actions.shape}  total {st.ms_total:.0f} ms (vae {st.ms_vision:.0f}, prefill {st.ms_prefill:.0f}, "
      f"denoise {st.ms_denoise:.0f})")


def err(a, b):
    """Max absolute difference, scaled by the reference's magnitude (at least 1)."""
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


canvas = np.fromfile(canvas_path, dtype=np.float32).reshape(case["canvas"].shape)
diff = canvas != case["canvas"]
print(f"canvas: {int(diff.sum())} of {diff.size} pixels differ from lerobot's"
      + (f", max |d| {np.abs(canvas - case['canvas']).max():.0f} levels" if diff.any() else " (identical)"))
z0 = np.fromfile(z0_path, dtype=np.float32).reshape(case["z0"].shape)
print(f"latent z0 (on the engine's canvas): scaled error {err(z0, case['z0']):.3e}")

ref = case["actions"].reshape(actions.shape)
print(f"[two views, canvas built here] vs PyTorch: scaled error {err(actions, ref):.3e}, "
      f"max |d| {np.abs(actions - ref).max():.3e}")

# The same model on the canvas lerobot's processor made (one canvas-sized image): the engine alone.
lerobot_canvas = np.ascontiguousarray(np.transpose(case["canvas"][0], (1, 2, 0)).astype(np.uint8))
actions2 = model.predict([lerobot_canvas], tokens, noise=noise, pixel_format=vla_cpp.PIXEL_U8)
z0b = np.fromfile(z0_path, dtype=np.float32).reshape(case["z0"].shape)
print(f"latent z0 (on lerobot's canvas): scaled error {err(z0b, case['z0']):.3e}")
print(f"[lerobot canvas passed in] vs PyTorch: scaled error {err(actions2, ref):.3e}, "
      f"max |d| {np.abs(actions2 - ref).max():.3e}")
if args.save:
    np.savez(args.save, actions=actions, actions_lerobot_canvas=actions2, z0=z0, z0_lerobot_canvas=z0b)
