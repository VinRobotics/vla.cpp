"""Verify the self-contained `model.VLAAdapter` against the gold reference
(tests/vla_adapter/runtime_parity/_workdir/runtime_ref.bin — produced by the REAL
upstream OpenVLAForActionPrediction.predict_action). Same deterministic inputs.

    python3 verify_ref.py [ckpt_dir] [runtime_ref.bin]
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # eval/ on path
from policies.vla_adapter.model import VLAAdapter

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/khanhnd61/data/VLA-Adapter/LIBERO-Object-Pro"
REF = sys.argv[2] if len(sys.argv) > 2 else str(
    Path(__file__).resolve().parents[2] / ".." / "tests" / "vla_adapter" / "_workdir" / "runtime_ref.bin")
IMG = 224
LCG_MUL, LCG_ADD = 6364136223846793005, 1442695040888963407


def lcg_f32(seed, n):
    s = seed & ((1 << 64) - 1); o = np.empty(n, np.float32)
    for i in range(n):
        s = (s * LCG_MUL + LCG_ADD) & ((1 << 64) - 1); o[i] = ((s >> 32) & 0xFFFFFFFF) / 2147483648.0 - 1.0
    return o


def lcg_ids(seed, n, vocab=151000):
    s = seed & ((1 << 64) - 1); o = np.empty(n, np.int64)
    for i in range(n):
        s = (s * LCG_MUL + LCG_ADD) & ((1 << 64) - 1); o[i] = ((s >> 32) & 0xFFFFFFFF) % vocab
    return o


def load_dump(path):
    t = {}
    with open(path, "rb") as f:
        while True:
            nl = struct.unpack("<I", f.read(4))[0]
            if nl == 0:
                break
            nm = f.read(nl).decode(); nd = struct.unpack("<I", f.read(4))[0]
            sh = struct.unpack(f"<{nd}q", f.read(8 * nd)); ns = tuple(reversed(sh)); n = int(np.prod(ns))
            t[nm] = np.frombuffer(f.read(4 * n), np.float32, n).reshape(ns).copy()
    return t


def main():
    ref = load_dump(REF)
    model = VLAAdapter.from_checkpoint(CKPT, device="cpu", dtype=torch.float32)

    input_ids = torch.from_numpy(lcg_ids(0x1D000001, 28)).unsqueeze(0)
    pixel_values = torch.from_numpy(lcg_f32(0x1D000002, 12 * IMG * IMG).reshape(1, 12, IMG, IMG))
    proprio = lcg_f32(0x1D000003, 8)

    actions, norm_actions = model.predict_action(input_ids, pixel_values, proprio,
                                                 unnorm_key="libero_object_no_noops")

    for name, got in (("norm_actions", norm_actions), ("actions", actions)):
        r = ref[name]
        dd = np.abs(r - got).max()
        rel = dd / max(1e-9, np.abs(r).max())
        ok = dd <= 1e-3
        print(f"  {name:13s} max|Δ| = {dd:.3e}  rel = {rel:.2e}{'' if ok else '  <-- FAIL'}")
    print(f"\nactions[0] reinvented = {actions[0]}")
    print(f"actions[0] gold       = {ref['actions'][0]}")


if __name__ == "__main__":
    main()
