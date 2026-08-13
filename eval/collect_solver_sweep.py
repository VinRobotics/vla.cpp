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
"""Aggregate the solver-step (T) sweep written by run_solver_step_sweep.sh.

Reads the per-task summaries of both stacks at each T:

    <root>/T<t>/vla_cpp/gr00t_n1_7/gr00t_n1_7/libero_object/task_<id>/summary.txt
    <root>/T<t>/pytorch/gr00t_n1_7/gr00t_n1_7/libero_object/task_<id>/summary.txt

and reports, per T, each stack's success rate with a Wilson 95% interval, the
delta in percentage points with a two-proportion z-test, and the per-step
latency. Episodes the env terminated mid-step count as failures, matching
collect_sr_compare.py.

Prints a table and writes a markdown report.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "outputs" / "solver_sweep"

SUCCESS_RE = re.compile(r"Success rate:\s*[\d.]+%\s*\((\d+)/(\d+)\)")
SKIPPED_RE = re.compile(r"Skipped \(terminated mid-step\):\s*(\d+)/(\d+)")
INF_RE = re.compile(r"Average inference time per step:\s*([\d.]+)\s*ms")
NACT_RE = re.compile(r"n_action_steps:\s*(\d+)")

ARCH = "gr00t_n1_7"
SUITE = "libero_object"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half))


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value for H0: p1 == p2, pooled-variance z-test."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    # Two-sided normal tail via erfc.
    return math.erfc(abs(z) / math.sqrt(2.0))


def read_cell(stack_dir: Path) -> dict | None:
    """Sum the 10 per-task summaries under one <T>/<stack> directory."""
    suite_dir = stack_dir / ARCH / ARCH / SUITE
    if not suite_dir.is_dir():
        return None
    succ = total = skipped = 0
    inf_ms: list[float] = []
    n_act: set[int] = set()
    tasks = 0
    for task_dir in sorted(suite_dir.glob("task_*")):
        summary = task_dir / "summary.txt"
        if not summary.is_file():
            continue
        text = summary.read_text()
        m = SUCCESS_RE.search(text)
        if not m:
            continue
        tasks += 1
        succ += int(m.group(1))
        total += int(m.group(2))
        if (s := SKIPPED_RE.search(text)):
            skipped += int(s.group(1))
        if (i := INF_RE.search(text)):
            inf_ms.append(float(i.group(1)))
        if (a := NACT_RE.search(text)):
            n_act.add(int(a.group(1)))
    if total == 0:
        return None
    return {
        "succ": succ,
        "total": total,
        "skipped": skipped,
        "tasks": tasks,
        "step_ms": sum(inf_ms) / len(inf_ms) if inf_ms else float("nan"),
        "n_act": sorted(n_act),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--root", type=Path, default=DEFAULT_ROOT,
                    help=f"sweep results root (default: {DEFAULT_ROOT})")
    ap.add_argument("-r", "--report", type=Path, default=None,
                    help="markdown report path (default: <root>/solver_sweep.md)")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"ERROR: no such directory: {root}", file=sys.stderr)
        return 1

    t_dirs = sorted(root.glob("T*"), key=lambda p: int(p.name[1:]))
    if not t_dirs:
        print(f"ERROR: no T<n> directories under {root}", file=sys.stderr)
        return 1

    rows = []
    for t_dir in t_dirs:
        t = int(t_dir.name[1:])
        cpp = read_cell(t_dir / "vla_cpp")
        pt = read_cell(t_dir / "pytorch")
        rows.append((t, cpp, pt))

    hdr = (f"{'T':>3}  {'vla.cpp SR':>22}  {'PyTorch SR':>22}  "
           f"{'delta':>8}  {'p':>6}  {'cpp ms/step':>11}  {'pt ms/step':>10}")
    print(hdr)
    print("-" * len(hdr))

    lines_md = [
        "# Solver-step sweep — GR00T-N1.7, libero_object",
        "",
        f"- Generated: {datetime.now():%Y-%m-%d}",
        f"- Root: `{root}`",
        "- `T` is the flow-matching solver-step count (checkpoint default 4),",
        "  set on both stacks by `VLA_NUM_STEPS`.",
        "- `n_action_steps` = 16 on both stacks at every T, so chunk-replay",
        "  cadence is held fixed and T is the only variable.",
        "- Intervals are Wilson 95%; `p` is a two-proportion test, vla.cpp vs PyTorch.",
        "",
        "| T | vla.cpp SR (Wilson 95%) | PyTorch SR (Wilson 95%) | delta (pp) | p | vla.cpp ms/step | PyTorch ms/step |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]

    for t, cpp, pt in rows:
        def fmt(c):
            if c is None:
                return "—", "—"
            lo, hi = wilson(c["succ"], c["total"])
            return (f"{100.0 * c['succ'] / c['total']:.1f}% "
                    f"({c['succ']}/{c['total']}) [{lo:.1f}, {hi:.1f}]",
                    f"{c['step_ms']:.2f}")
        cpp_sr, cpp_ms = fmt(cpp)
        pt_sr, pt_ms = fmt(pt)
        if cpp and pt:
            delta = 100.0 * (cpp["succ"] / cpp["total"] - pt["succ"] / pt["total"])
            p = two_proportion_p(cpp["succ"], cpp["total"], pt["succ"], pt["total"])
            delta_s, p_s = f"{delta:+.1f}", f"{p:.2f}"
        else:
            delta_s, p_s = "—", "—"
        print(f"{t:>3}  {cpp_sr:>22}  {pt_sr:>22}  {delta_s:>8}  {p_s:>6}  "
              f"{cpp_ms:>11}  {pt_ms:>10}")
        lines_md.append(
            f"| {t} | {cpp_sr} | {pt_sr} | {delta_s} | {p_s} | {cpp_ms} | {pt_ms} |")

    # Terminated-episode audit: a nonzero count anywhere invalidates the cell.
    lines_md += ["", "## Terminated-episode audit", "",
                 "| T | vla.cpp skipped | PyTorch skipped | vla.cpp n_act | PyTorch n_act |",
                 "|---:|---:|---:|---|---|"]
    for t, cpp, pt in rows:
        cs = str(cpp["skipped"]) if cpp else "—"
        ps = str(pt["skipped"]) if pt else "—"
        cn = ",".join(map(str, cpp["n_act"])) if cpp else "—"
        pn = ",".join(map(str, pt["n_act"])) if pt else "—"
        lines_md.append(f"| {t} | {cs} | {ps} | {cn} | {pn} |")

    report = args.report or (root / "solver_sweep.md")
    report.write_text("\n".join(lines_md) + "\n")
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
