#!/usr/bin/env python3
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

"""Generate GR00T relative-action statistics for the ALOHA right-arm datasets.

Mirrors gr00t/data/stats.py::load_relative_actions for ActionType.NON_EEF:
for every frame i, the reference is state[i] and the chunk is action[i .. i+H-1];
the relative action is a plain element-wise subtraction (JointPose.__sub__, see
JointActionChunk.relative_chunking).  Stats are per chunk-step and per joint,
matching the [H, D] layout of meta/relative_stats.json.
"""
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

H = 16                      # len(modality action delta_indices)
SLICE = slice(0, 6)         # modality.json: action/state single_arm = [0:6]
KEY = "single_arm"          # only RELATIVE modalities get relative stats

def deltas_for_dataset(root: Path) -> np.ndarray:
    out = []
    for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        df = pd.read_parquet(f, columns=["action", "observation.state"])
        act = np.stack(df["action"].values).astype(np.float32)[:, SLICE]
        st  = np.stack(df["observation.state"].values).astype(np.float32)[:, SLICE]
        usable = len(df) - (H - 1)
        if usable <= 0:
            continue
        idx = np.arange(usable)[:, None] + np.arange(H)[None, :]      # (usable, H)
        out.append(act[idx] - st[:usable][:, None, :])                 # (usable, H, 6)
    return np.concatenate(out, axis=0)

def stats_of(d: np.ndarray) -> dict:
    return {"max":  d.max(axis=0).tolist(),
            "min":  d.min(axis=0).tolist(),
            "q01":  np.quantile(d, 0.01, axis=0).tolist(),
            "q99":  np.quantile(d, 0.99, axis=0).tolist(),
            "mean": d.mean(axis=0).tolist(),
            "std":  d.std(axis=0).tolist()}

def main(roots, agg_out):
    pooled = []
    for r in roots:
        r = Path(r)
        d = deltas_for_dataset(r)
        pooled.append(d)
        p = r / "meta" / "relative_stats.json"
        p.write_text(json.dumps({KEY: stats_of(d)}, indent=4))
        print(f"  {r.name}: {d.shape[0]} chunks -> {p}")
    allp = np.concatenate(pooled, axis=0)
    Path(agg_out).write_text(json.dumps({KEY: stats_of(allp)}, indent=4))
    print(f"  aggregate: {allp.shape[0]} chunks, shape {allp.shape[1:]} -> {agg_out}")
    print(f"  delta range over all steps: min {allp.min():+.4f}  max {allp.max():+.4f}  mean |d| {np.abs(allp).mean():.4f}")

if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
