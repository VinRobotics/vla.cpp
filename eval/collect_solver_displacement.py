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
"""Analyse the fixed-noise action chunks written by run_solver_step_displacement.sh.

For each T, reports the action chunk's displacement against the T=4 reference
chunk (the checkpoint default): max absolute deviation, relative RMS, and
cosine similarity, plus the measured per-call latency.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "outputs" / "solver_sweep" / "displacement"

BENCH_RE = re.compile(r"predict\(\) over (\d+) iters: min=([\d.]+) ms\s+avg=([\d.]+) ms")
SPLIT_RE = re.compile(r"last split: vision=([\d.]+)\s+inference=([\d.]+)\s+total=([\d.]+) ms")
REF_T = 4


def read_actions(path: Path) -> list[float]:
    # model_load banners land on stdout ahead of the payload, so locate the
    # header rather than assuming it is the first line.
    lines = path.read_text().splitlines()
    idx = next((i for i, l in enumerate(lines) if l.startswith("action_len=")), None)
    if idx is None:
        raise ValueError(f"{path}: no action_len header")
    n = int(lines[idx].split("=", 1)[1])
    if n == 0:
        raise ValueError(f"{path}: empty action chunk (action_len=0) -- "
                         "usually an image-size mismatch; check the .timing.txt")
    vals = [float(x) for x in lines[idx + 1 : idx + 1 + n]]
    if len(vals) != n:
        raise ValueError(f"{path}: expected {n} values, got {len(vals)}")
    return vals


def read_timing(path: Path) -> dict:
    text = path.read_text() if path.is_file() else ""
    out = {}
    if (m := BENCH_RE.search(text)):
        out["min_ms"], out["avg_ms"] = float(m.group(2)), float(m.group(3))
    if (m := SPLIT_RE.search(text)):
        out["vision_ms"], out["inf_ms"] = float(m.group(1)), float(m.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out: Path = args.out
    if not out.is_dir():
        print(f"ERROR: no such directory: {out}", file=sys.stderr)
        return 1

    files = sorted(out.glob("T*.actions.txt"),
                   key=lambda p: int(p.name.split(".")[0][1:]))
    if not files:
        print(f"ERROR: no T*.actions.txt under {out}", file=sys.stderr)
        return 1

    chunks = {int(f.name.split(".")[0][1:]): read_actions(f) for f in files}
    if REF_T not in chunks:
        print(f"ERROR: reference T={REF_T} missing", file=sys.stderr)
        return 1
    ref = chunks[REF_T]
    ref_rms = math.sqrt(sum(v * v for v in ref) / len(ref))

    hdr = (f"{'T':>3}  {'max|d| vs T=4':>13}  {'rel RMS':>9}  {'cosine':>10}  "
           f"{'avg ms':>7}  {'vision':>7}  {'inf':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for t, ch in chunks.items():
        if len(ch) != len(ref):
            print(f"ERROR: T={t} chunk length {len(ch)} != ref {len(ref)}", file=sys.stderr)
            return 1
        diffs = [a - b for a, b in zip(ch, ref)]
        max_abs = max(abs(d) for d in diffs)
        rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        rel_rms = rms / ref_rms if ref_rms else float("nan")
        dot = sum(a * b for a, b in zip(ch, ref))
        na = math.sqrt(sum(a * a for a in ch))
        nb = math.sqrt(sum(b * b for b in ref))
        cos = dot / (na * nb) if na and nb else float("nan")
        tm = read_timing(out / f"T{t}.timing.txt")
        rows.append((t, max_abs, rel_rms, cos, tm))
        print(f"{t:>3}  {max_abs:>13.4e}  {rel_rms:>9.3e}  {cos:>10.8f}  "
              f"{tm.get('avg_ms', float('nan')):>7.2f}  "
              f"{tm.get('vision_ms', float('nan')):>7.2f}  "
              f"{tm.get('inf_ms', float('nan')):>7.2f}")

    md = [
        "# Solver-step displacement — GR00T-N1.7, fixed noise",
        "",
        "Action chunk at each T against the T=4 checkpoint default, from",
        "`vla_predict_check` (fixed images / language / state / noise).",
        f"Chunk is {len(ref)} values; `rel RMS` is RMS(delta) / RMS(reference chunk).",
        "",
        "| T | max abs dev vs T=4 | rel RMS | cosine | avg ms | vision ms | inference ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, ma, rr, cos, tm in rows:
        md.append(
            f"| {t} | {ma:.4e} | {rr:.3e} | {cos:.8f} | "
            f"{tm.get('avg_ms', float('nan')):.2f} | "
            f"{tm.get('vision_ms', float('nan')):.2f} | "
            f"{tm.get('inf_ms', float('nan')):.2f} |")
    report = out / "displacement.md"
    report.write_text("\n".join(md) + "\n")
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
