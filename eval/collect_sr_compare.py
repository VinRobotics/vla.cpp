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
"""Aggregate the side-by-side LIBERO sweep written by run_libero_compare.sh.

Reads both phases' per-task `summary.txt` files:

    <root>/pytorch/<model>/<model>/<suite>/task_<id>/summary.txt
    <root>/vla_cpp/<arch>/<arch>/<suite>/task_<id>/summary.txt

and reports, per model, the PyTorch reference success rate next to the vla.cpp
GGUF success rate plus the delta in percentage points. Episodes the env
terminated mid-step ("skipped") count as failures, matching
collect_libero_results.py.

Prints a table and writes a markdown report.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "outputs" / "sr_compare"

TASK_RE = re.compile(r"Task:\s*(\S+?)/task_(\d+)")
SUCCESS_RE = re.compile(r"Success rate:\s*[\d.]+%\s*\((\d+)/(\d+)\)")
SKIPPED_RE = re.compile(r"Skipped \(terminated mid-step\):\s*(\d+)/(\d+)")
INF_RE = re.compile(r"Average inference time per step:\s*([\d.]+)\s*ms")
NACT_RE = re.compile(r"n_action_steps:\s*(\d+)")

# Models with no PyTorch wrapper in eval/pytorch_ref get their reference number
# from the upstream paper instead of a local run.
PAPER_SR = {
    "bitvla": (99.6, "BitVLA paper, LIBERO-Object, 50 ep x 10 tasks"),
}

# Display order; anything else found is appended alphabetically.
MODEL_ORDER = [
    "smolvla", "pi0", "evo1", "bitvla",
    "gr00t_n1_5", "gr00t_n1_6", "gr00t_n1_7",
]


def parse_summary(path: Path) -> dict | None:
    text = path.read_text()
    m_s = SUCCESS_RE.search(text)
    m_k = SKIPPED_RE.search(text)
    if not (m_s and m_k):
        print(f"WARN: unparsable summary, skipping: {path}", file=sys.stderr)
        return None
    successes, counted = int(m_s.group(1)), int(m_s.group(2))
    skipped, n_episodes = int(m_k.group(1)), int(m_k.group(2))
    m_t = TASK_RE.search(text)
    m_i = INF_RE.search(text)
    m_n = NACT_RE.search(text)
    return {
        "successes": successes,
        "counted": counted,
        "skipped": skipped,
        "n_episodes": n_episodes,
        "suite": m_t.group(1) if m_t else None,
        "task_id": int(m_t.group(2)) if m_t else int(path.parent.name.split("_")[-1]),
        "inference_ms": float(m_i.group(1)) if m_i else None,
        "n_action_steps": int(m_n.group(1)) if m_n else None,
    }


def collect_phase(phase_dir: Path) -> dict[str, dict]:
    """model -> {tasks: {id: rec}, successes, episodes, inference_ms, n_action_steps}"""
    out: dict[str, dict] = {}
    if not phase_dir.is_dir():
        return out
    for summary in sorted(phase_dir.glob("*/*/*/task_*/summary.txt")):
        model = summary.parents[3].name
        rec = parse_summary(summary)
        if rec is None:
            continue
        entry = out.setdefault(
            model, {"tasks": {}, "successes": 0, "episodes": 0,
                    "inf": [], "n_action_steps": set(), "suites": set()}
        )
        if rec["task_id"] in entry["tasks"]:
            print(f"WARN: duplicate task_{rec['task_id']} for {model}: {summary}",
                  file=sys.stderr)
        entry["tasks"][rec["task_id"]] = rec
        entry["successes"] += rec["successes"]
        entry["episodes"] += rec["n_episodes"]
        if rec["inference_ms"] is not None:
            entry["inf"].append(rec["inference_ms"])
        if rec["n_action_steps"] is not None:
            entry["n_action_steps"].add(rec["n_action_steps"])
        if rec["suite"]:
            entry["suites"].add(rec["suite"])
    return out


def sr(entry: dict) -> float | None:
    return 100.0 * entry["successes"] / entry["episodes"] if entry["episodes"] else None


def fmt_sr(entry: dict | None) -> str:
    if entry is None or not entry["episodes"]:
        return "—"
    return f"{sr(entry):.1f}% ({entry['successes']}/{entry['episodes']})"


def fmt_nact(entry: dict | None) -> str:
    if entry is None or not entry["n_action_steps"]:
        return "—"
    return "/".join(str(n) for n in sorted(entry["n_action_steps"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--root", type=Path, default=DEFAULT_ROOT,
                    help=f"sweep root written by run_libero_compare.sh (default: {DEFAULT_ROOT})")
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown report path (default: <root>/sr_compare.md)")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"ERROR: sweep root not found: {root}", file=sys.stderr)
        return 1

    pt = collect_phase(root / "pytorch")
    cpp = collect_phase(root / "vla_cpp")
    if not pt and not cpp:
        print(f"ERROR: no summary.txt found under {root}", file=sys.stderr)
        return 1

    models = [m for m in MODEL_ORDER if m in pt or m in cpp]
    models += sorted(set(pt) | set(cpp) - set(models) - set(MODEL_ORDER))
    seen, ordered = set(), []
    for m in models:
        if m not in seen:
            seen.add(m)
            ordered.append(m)

    suites = {s for e in list(pt.values()) + list(cpp.values()) for s in e["suites"]}
    suite = ", ".join(sorted(suites)) if suites else "libero_object"

    lines: list[str] = []
    lines.append("# LIBERO success rate — PyTorch reference vs vla.cpp GGUF")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Suite: `{suite}`, tasks 0–9")
    lines.append(f"- Sweep root: `{root}`")
    lines.append("")
    lines.append("Both stacks ran the same LIBERO env (`eval/sim/libero/libero_env.py`), "
                 "seed 42, 500-step cap, and the same per-arch action-chunk replay "
                 "(`n_act` column). Episodes the env terminated mid-step count as failures.")
    lines.append("")
    lines.append("| Model | n_act | PyTorch SR | vla.cpp SR | Δ (pp) | PyTorch ms/step | vla.cpp ms/step |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for m in ordered:
        p, c = pt.get(m), cpp.get(m)
        if p is None and m in PAPER_SR:
            paper, _ = PAPER_SR[m]
            pt_cell = f"{paper:.1f}%¹"
            delta = f"{sr(c) - paper:+.1f}" if c and c["episodes"] else "—"
        else:
            pt_cell = fmt_sr(p)
            delta = (f"{sr(c) - sr(p):+.1f}"
                     if p and c and p["episodes"] and c["episodes"] else "—")
        # Prefer the vla.cpp value, but fall back to PyTorch's while only one
        # phase has run — fmt_nact returns "—", which is truthy, so `or` here
        # would never reach the fallback.
        nact = fmt_nact(c) if (c and c["n_action_steps"]) else fmt_nact(p)
        p_ms = f"{sum(p['inf']) / len(p['inf']):.1f}" if p and p["inf"] else "—"
        c_ms = f"{sum(c['inf']) / len(c['inf']):.1f}" if c and c["inf"] else "—"
        lines.append(f"| `{m}` | {nact} | {pt_cell} | {fmt_sr(c)} | {delta} | {p_ms} | {c_ms} |")

    if any(m in PAPER_SR for m in ordered if m not in pt):
        lines.append("")
        for m in ordered:
            if m not in pt and m in PAPER_SR:
                lines.append(f"¹ `{m}`: no PyTorch wrapper in `eval/pytorch_ref`; "
                             f"reference is the published number ({PAPER_SR[m][1]}), "
                             f"not a local run — the episode protocol differs.")

    # Per-task breakdown
    lines.append("")
    lines.append("## Per-task success (successes / episodes)")
    lines.append("")
    header = "| Model | Stack | " + " | ".join(f"t{i}" for i in range(10)) + " | total |"
    lines.append(header)
    lines.append("|---|---|" + "---:|" * 11)
    for m in ordered:
        for label, entry in (("PyTorch", pt.get(m)), ("vla.cpp", cpp.get(m))):
            if entry is None:
                continue
            cells = []
            for i in range(10):
                r = entry["tasks"].get(i)
                cells.append(f"{r['successes']}/{r['n_episodes']}" if r else "—")
            cells.append(f"**{entry['successes']}/{entry['episodes']}**")
            lines.append(f"| `{m}` | {label} | " + " | ".join(cells) + " |")

    missing = [m for m in ordered if m in pt and m in cpp
               and pt[m]["episodes"] != cpp[m]["episodes"]]
    if missing:
        lines.append("")
        lines.append("> **Uneven episode counts** (one stack ran fewer episodes — the "
                     "delta above is not like-for-like): "
                     + ", ".join(f"`{m}`" for m in missing))

    report = "\n".join(lines) + "\n"
    out_path = args.out or (root / "sr_compare.md")
    out_path.write_text(report)
    print(report)
    print(f"[wrote] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
