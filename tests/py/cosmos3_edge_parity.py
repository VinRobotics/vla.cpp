"""Cosmos3-Edge through libvla against the ggml-compiler runtime and the PyTorch reference, one case.

Inputs come from ggml/check_stages.py (the cosmos3_edge_wam repo): `--out CASE.npz` writes CASE.npz (the f32 PyTorch
outputs) and CASE.npz.inputs.npz (canvas, input_ids, attention_mask, video_noise, action_noise). The two camera
views are the first two torch.rand draws of the generator seeded with --seed, as check_stages.py makes them.

    VLA_LIBRARY=build/libvla.so python tests/py/cosmos3_edge_parity.py --gguf M.gguf --inputs CASE.npz.inputs.npz \\
        --torch CASE.npz --seed 1 [--compiler-out REF.npz]
"""
import argparse, os, sys, tempfile
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "bindings", "python"))

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--inputs", required=True)
ap.add_argument("--torch", required=True, help="CASE.npz, the f32 PyTorch outputs")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--compiler-out", help="`ggml-compiler run` outputs of the same GGUF's source on the same inputs")
args = ap.parse_args()

import torch  # only to draw the same views as check_stages.py
tmp = tempfile.mkdtemp()
inp = np.load(args.inputs)
vn_path, canvas_path = os.path.join(tmp, "vn.f32"), os.path.join(tmp, "canvas.f32")
np.ascontiguousarray(inp["video_noise"], dtype=np.float32).tofile(vn_path)
os.environ["VLA_COSMOS3_VIDEO_NOISE"] = vn_path
os.environ["VLA_COSMOS3_DUMP_CANVAS"] = canvas_path

import vla_cpp

g = torch.Generator().manual_seed(args.seed)
views = [torch.rand(3, 256, 256, generator=g).permute(1, 2, 0).contiguous().numpy() for _ in range(2)]
n = int(inp["attention_mask"].sum())
tokens = inp["input_ids"][0, :n].tolist()

model = vla_cpp.load(args.gguf)
actions = model.predict(views, tokens, noise=inp["action_noise"].astype(np.float32).ravel(),
                        pixel_format=vla_cpp.PIXEL_F32_RGB_01, timing=1)
st = model.last_stats()
print(f"actions {actions.shape}  total {st.ms_total:.0f} ms (vae {st.ms_vision:.0f}, prefill {st.ms_prefill:.0f}, denoise {st.ms_denoise:.0f})")

canvas = np.fromfile(canvas_path, dtype=np.float32).reshape(inp["canvas"].shape)
diff = canvas != inp["canvas"]
print(f"canvas: {int(diff.sum())} of {diff.size} pixels differ from lerobot's"
      + (f", max |d| {np.abs(canvas - inp['canvas']).max():.0f} levels" if diff.any() else " (identical)"))

def err(a, b):
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))

ref = np.load(args.torch)["actions"].reshape(actions.shape)
comp = np.load(args.compiler_out)["actions"].reshape(actions.shape) if args.compiler_out else None

def report(label, a):
    line = f"[{label}] vs PyTorch f32: scaled error {err(a, ref):.3e}"
    if comp is not None:
        line += f";  vs ggml-compiler runtime: max|d| {np.abs(a - comp).max():.3e}, bit-identical {np.array_equal(a, comp)}"
    print(line)

report("two views, canvas built here", actions)

# The same model on the canvas lerobot's processor made (one canvas-sized image): the engine alone.
lerobot_canvas = np.ascontiguousarray(np.transpose(inp["canvas"][0], (1, 2, 0)).astype(np.uint8))
actions2 = model.predict([lerobot_canvas], tokens, noise=inp["action_noise"].astype(np.float32).ravel(),
                         pixel_format=vla_cpp.PIXEL_U8)
report("lerobot canvas passed in", actions2)
