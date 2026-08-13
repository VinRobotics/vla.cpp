#!/usr/bin/env python3
"""Aggregate the BitVLA multi-suite LIBERO sweep produced by run_libero_bitvla.sh.

Unlike collect_libero_results.py (many models, ONE suite), this sweep is ONE
model (bitvla) across SEVERAL suites — each suite under its own subdir:

    <sweep>/bitvla_<label>/bitvla/<suite>/task_<id>/summary.txt
    <sweep>/_server_logs/bitvla_<label>.log
    <sweep>/_server_logs/bitvla_<label>.mem.json

e.g. bitvla_spatial → libero_spatial, bitvla_long → libero_10, etc. The suite
name is read from each summary.txt's `Task:` line rather than hardcoded, so the
libero_long→libero_10 rename and any partial run are handled automatically.

For each suite it reports per-task and aggregate success rate (terminated
episodes counted as failures), client-side inference time, server-side timing
breakdown, and peak memory. Prints a table and writes a markdown report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP = REPO_ROOT / "outputs" / "libero_bitvla_sweep"

TASK_RE = re.compile(r"Task:\s*(\S+)/task_(\d+)")
SUCCESS_RE = re.compile(r"Success rate:\s*[\d.]+%\s*\((\d+)/(\d+)\)")
SKIPPED_RE = re.compile(r"Skipped \(terminated mid-step\):\s*(\d+)/(\d+)")
INF_RE = re.compile(r"Average inference time per step:\s*([\d.]+)\s*ms")
NACT_RE = re.compile(r"n_action_steps:\s*(\d+)")

# vla-server logs every Nth request as:
#   vla-server: rid=40  served=41  total=480.3 ms  vision=343.8  inf=129.3  other=7.2
SRV_RE = re.compile(
    r"vla-server:\s*rid=\d+\s+served=\d+\s+"
    r"total=([\d.]+)\s*ms\s+vision=([\d.]+)\s+inf=([\d.]+)\s+other=([\d.]+)"
)


def parse_summary(path: Path) -> dict:
    text = path.read_text()
    m_t = TASK_RE.search(text)
    m_s = SUCCESS_RE.search(text)
    m_k = SKIPPED_RE.search(text)
    m_i = INF_RE.search(text)
    if not (m_t and m_s and m_k and m_i):
        raise ValueError(f"could not parse {path}")
    successes, counted = int(m_s.group(1)), int(m_s.group(2))
    skipped, n_episodes = int(m_k.group(1)), int(m_k.group(2))
    if counted + skipped != n_episodes:
        raise ValueError(
            f"{path}: counted({counted}) + skipped({skipped}) != n_episodes({n_episodes})"
        )
    m_n = NACT_RE.search(text)
    return {
        "suite": m_t.group(1),
        "task_id": int(m_t.group(2)),
        "successes": successes,
        "n_episodes": n_episodes,
        "skipped": skipped,
        "inf_ms": float(m_i.group(1)),
        "n_action_steps": int(m_n.group(1)) if m_n else None,
    }


def parse_mem_json(mem_path: Path) -> dict | None:
    if not mem_path.is_file():
        return None
    try:
        return json.loads(mem_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: failed to parse {mem_path}: {e}", file=sys.stderr)
        return None


def parse_server_log(log_path: Path) -> dict | None:
    """Average server-side timings; drops the first (warmup) sample."""
    if not log_path.is_file():
        return None
    totals, visions, infs, others = [], [], [], []
    seen = 0
    with log_path.open() as f:
        for line in f:
            m = SRV_RE.search(line)
            if not m:
                continue
            seen += 1
            if seen == 1:  # rid=0 includes one-time warmup (graph capture, KV alloc)
                continue
            totals.append(float(m.group(1)))
            visions.append(float(m.group(2)))
            infs.append(float(m.group(3)))
            others.append(float(m.group(4)))
    if not totals:
        return None
    n = len(totals)
    return {
        "n_samples": n,
        "total": sum(totals) / n,
        "vision": sum(visions) / n,
        "inf": sum(infs) / n,
        "other": sum(others) / n,
    }


def collect_suite(suite_dir: Path) -> dict | None:
    """Parse every task summary under one bitvla_<label>/ subdir."""
    per_task: dict[int, dict] = {}
    for sp in sorted(suite_dir.glob("**/task_*/summary.txt")):
        try:
            rec = parse_summary(sp)
        except ValueError as e:
            print(f"warning: {e}", file=sys.stderr)
            continue
        if rec["task_id"] in per_task:
            print(f"warning: duplicate task_{rec['task_id']} under {suite_dir.name}: {sp}",
                  file=sys.stderr)
        per_task[rec["task_id"]] = rec
    if not per_task:
        return None
    suites = {t["suite"] for t in per_task.values()}
    if len(suites) > 1:
        print(f"warning: {suite_dir.name} mixes suites {suites}", file=sys.stderr)
    return {"suite": sorted(suites)[0], "per_task": per_task}


def aggregate(per_task: dict) -> dict:
    total_succ = sum(t["successes"] for t in per_task.values())
    total_eps = sum(t["n_episodes"] for t in per_task.values())
    total_skip = sum(t["skipped"] for t in per_task.values())
    sr = total_succ / total_eps if total_eps else 0.0
    weighted_inf = (
        sum(t["inf_ms"] * t["n_episodes"] for t in per_task.values()) / total_eps
        if total_eps else 0.0
    )
    n_acts = [t["n_action_steps"] for t in per_task.values() if t["n_action_steps"] is not None]
    n_act = max(set(n_acts), key=n_acts.count) if n_acts else None
    if n_acts and len(set(n_acts)) > 1:
        print(f"warning: n_action_steps disagrees across tasks: {sorted(set(n_acts))}",
              file=sys.stderr)
    return {
        "n_tasks": len(per_task),
        "total_succ": total_succ,
        "total_eps": total_eps,
        "total_skip": total_skip,
        "sr": sr,
        "avg_inf_ms": weighted_inf,
        "n_action_steps": n_act,
    }


def render_markdown(sweep: Path, rows: list[dict]) -> str:
    L: list[str] = []
    L.append(f"# BitVLA LIBERO sweep report — `{sweep.name}`")
    L.append("")
    L.append(f"- Sweep root: `{sweep}`")
    L.append(f"- Model: `bitvla` across {len(rows)} suite(s)")
    L.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    L.append("")
    L.append("## Success rate & inference time")
    L.append("")
    L.append("- **SR** counts terminated episodes as failures: `successes / n_episodes`.")
    L.append("- **client/step** — wall-time per env step (amortized over chunk replay; "
             "the `Average inference time per step` in each `summary.txt`).")
    L.append("- **client/call** = `client/step × n_act` — per actual `vla-server` call "
             "(client pre/post + ZMQ loopback + server compute).")
    L.append("")
    L.append("| Suite | n_act | Tasks | Successes | Terminated | SR | client/step (ms) | client/call (ms) | server total (ms) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        a = r["agg"]
        n_act = a["n_action_steps"]
        n_act_str = str(n_act) if n_act is not None else "?"
        per_call = f"{a['avg_inf_ms'] * n_act:.2f}" if n_act is not None else "?"
        srv = r.get("server")
        srv_total = f"{srv['total']:.2f}" if srv else "—"
        L.append(
            f"| `{r['suite']}` | {n_act_str} | {a['n_tasks']} | "
            f"{a['total_succ']}/{a['total_eps']} | {a['total_skip']}/{a['total_eps']} | "
            f"{a['sr']:.2%} | {a['avg_inf_ms']:.2f} | {per_call} | {srv_total} |"
        )
    # Overall
    tot_succ = sum(r["agg"]["total_succ"] for r in rows)
    tot_eps = sum(r["agg"]["total_eps"] for r in rows)
    tot_skip = sum(r["agg"]["total_skip"] for r in rows)
    if tot_eps:
        L.append(
            f"| **all** | | {sum(r['agg']['n_tasks'] for r in rows)} | "
            f"{tot_succ}/{tot_eps} | {tot_skip}/{tot_eps} | "
            f"**{tot_succ / tot_eps:.2%}** | | | |"
        )
    L.append("")

    if any(r.get("server") for r in rows):
        L.append("## Server-side inference breakdown")
        L.append("")
        L.append("Parsed from `_server_logs/bitvla_<label>.log` "
                 "(`total = vision + inf + other`; first/warmup sample dropped):")
        L.append("")
        L.append("| Suite | Samples | total (ms) | vision | inf | other |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            s = r.get("server")
            if not s:
                L.append(f"| `{r['suite']}` | (no log) | — | — | — | — |")
                continue
            L.append(f"| `{r['suite']}` | {s['n_samples']} | {s['total']:.2f} | "
                     f"{s['vision']:.2f} | {s['inf']:.2f} | {s['other']:.2f} |")
        L.append("")

    if any(r.get("mem") for r in rows):
        L.append("## Peak memory")
        L.append("")
        L.append("Sampled by `mem_sampler` in "
                 "[`eval/run_libero_bitvla.sh`](run_libero_bitvla.sh) while `vla-server` "
                 "was alive. **VRAM** = per-PID `nvidia-smi` (`(no GPU)` on Tegra); "
                 "**RAM** = `VmHWM` (host only); **sys RAM / sys Δ** = system-wide used "
                 "RAM and its rise over baseline (the only signal that captures iGPU "
                 "unified-memory weights on Tegra).")
        L.append("")
        L.append("| Suite | Peak VRAM (MiB) | Peak RAM (MiB) | Peak sys RAM (MiB) | sys Δ (MiB) | Samples |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            m = r.get("mem")
            if not m:
                L.append(f"| `{r['suite']}` | n/a | n/a | n/a | n/a | n/a |")
                continue
            vram = m.get("peak_vram_mib")
            vram_str = f"{vram}" if isinstance(vram, int) else "(no GPU)"
            sp = m.get("peak_sys_used_mib")
            sd = m.get("sys_used_delta_mib")
            sp_str = f"{sp:.1f}" if isinstance(sp, (int, float)) else "n/a"
            sd_str = f"{sd:.1f}" if isinstance(sd, (int, float)) else "n/a"
            L.append(f"| `{r['suite']}` | {vram_str} | {m.get('peak_rss_mib', 0):.1f} | "
                     f"{sp_str} | {sd_str} | {m.get('samples', 0)} |")
        L.append("")

    L.append("## Per-task breakdown")
    L.append("")
    for r in rows:
        L.append(f"<details><summary><code>{r['suite']}</code></summary>")
        L.append("")
        L.append("| Task | Successes | Terminated | SR | client/step (ms) |")
        L.append("|---|---:|---:|---:|---:|")
        for tid in sorted(r["per_task"]):
            t = r["per_task"][tid]
            sr = t["successes"] / t["n_episodes"] if t["n_episodes"] else 0.0
            L.append(f"| task_{tid} | {t['successes']}/{t['n_episodes']} | "
                     f"{t['skipped']}/{t['n_episodes']} | {sr:.2%} | {t['inf_ms']:.2f} |")
        L.append("")
        L.append("</details>")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP,
                    help=f"sweep root (default: {DEFAULT_SWEEP})")
    ap.add_argument("--md", type=Path, default=None, metavar="PATH",
                    help="markdown report path (default: <sweep>/report.md)")
    ap.add_argument("--no-md", action="store_true", help="do not write a markdown report")
    args = ap.parse_args()

    if not args.sweep.is_dir():
        print(f"ERROR: sweep dir not found: {args.sweep}", file=sys.stderr)
        return 1

    server_logs_dir = args.sweep / "_server_logs"
    suite_dirs = sorted(p for p in args.sweep.iterdir()
                        if p.is_dir() and not p.name.startswith(("_", ".")))

    rows: list[dict] = []
    for d in suite_dirs:
        c = collect_suite(d)
        if c is None:
            print(f"warning: no summaries under {d}", file=sys.stderr)
            continue
        agg = aggregate(c["per_task"])
        rows.append({
            "label": d.name,
            "suite": c["suite"],
            "per_task": c["per_task"],
            "agg": agg,
            "server": parse_server_log(server_logs_dir / f"{d.name}.log"),
            "mem": parse_mem_json(server_logs_dir / f"{d.name}.mem.json"),
        })

    if not rows:
        print("No results found.", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r["suite"])

    print(f"Sweep: {args.sweep}")
    print(f"Model: bitvla  ({len(rows)} suite(s))")
    print()
    hdr = (f"{'suite':<16} {'n_act':>6} {'tasks':>6} {'success':>10} {'terminated':>12} "
           f"{'SR':>8} {'client/step':>12} {'client/call':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        a = r["agg"]
        n_act = a["n_action_steps"]
        n_act_str = str(n_act) if n_act is not None else "?"
        per_call = f"{a['avg_inf_ms'] * n_act:>12.2f}" if n_act is not None else f"{'?':>12}"
        print(f"{r['suite']:<16} {n_act_str:>6} {a['n_tasks']:>6} "
              f"{a['total_succ']:>4}/{a['total_eps']:<5} "
              f"{a['total_skip']:>5}/{a['total_eps']:<6} "
              f"{a['sr']:>7.2%} {a['avg_inf_ms']:>12.2f} {per_call}")
    tot_succ = sum(r["agg"]["total_succ"] for r in rows)
    tot_eps = sum(r["agg"]["total_eps"] for r in rows)
    if tot_eps:
        print("-" * len(hdr))
        print(f"{'ALL':<16} {'':>6} {sum(r['agg']['n_tasks'] for r in rows):>6} "
              f"{tot_succ:>4}/{tot_eps:<5} {'':>12} {tot_succ / tot_eps:>7.2%}")

    if not args.no_md:
        md_path = args.md if args.md is not None else (args.sweep / "report.md")
        md_path.write_text(render_markdown(args.sweep, rows), encoding="utf-8")
        print()
        print(f"Markdown report written to: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
