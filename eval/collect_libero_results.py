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

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP = REPO_ROOT / "outputs" / "libero_object_sweep"
TASK_SUITE = "libero_object"
N_TASKS = 10

SUCCESS_RE = re.compile(r"Success rate:\s*[\d.]+%\s*\((\d+)/(\d+)\)")
SKIPPED_RE = re.compile(r"Skipped \(terminated mid-step\):\s*(\d+)/(\d+)")
INF_RE = re.compile(r"Average inference time per step:\s*([\d.]+)\s*ms")
NACT_RE = re.compile(r"n_action_steps:\s*(\d+)")

SRV_RE = re.compile(
    r"vla-server:\s*rid=\d+\s+served=\d+\s+"
    r"total=([\d.]+)\s*ms\s+"
    r"vision=([\d.]+)\s+"
    r"inf=([\d.]+)\s+"
    r"other=([\d.]+)"
)

def parse_summary(path: Path) -> dict:
    text = path.read_text()
    m_s = SUCCESS_RE.search(text)
    m_k = SKIPPED_RE.search(text)
    m_i = INF_RE.search(text)
    if not (m_s and m_k and m_i):
        raise ValueError(f"could not parse {path}")
    successes = int(m_s.group(1))
    counted = int(m_s.group(2))
    skipped = int(m_k.group(1))
    n_episodes = int(m_k.group(2))
    inf_ms = float(m_i.group(1))
    if counted + skipped != n_episodes:
        raise ValueError(
            f"{path}: counted({counted}) + skipped({skipped}) != n_episodes({n_episodes})"
        )
    m_n = NACT_RE.search(text)
    n_act = int(m_n.group(1)) if m_n else None
    return {
        "successes": successes,
        "n_episodes": n_episodes,
        "skipped": skipped,
        "inf_ms": inf_ms,
        "n_action_steps": n_act,
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
            if seen == 1:
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

def collect_model(model_dir: Path) -> dict | None:

    per_task: dict[int, dict] = {}
    for task_id in range(N_TASKS):
        hits = list(model_dir.glob(f"**/{TASK_SUITE}/task_{task_id}/summary.txt"))
        if not hits:
            continue
        if len(hits) > 1:
            print(f"warning: multiple summaries for {model_dir.name} task_{task_id}: {hits}",
                  file=sys.stderr)
        per_task[task_id] = parse_summary(hits[0])
    if not per_task:
        return None
    return per_task

def fmt_row(model: str, per_task: dict) -> tuple[str, dict]:
    total_succ = sum(t["successes"] for t in per_task.values())
    total_eps = sum(t["n_episodes"] for t in per_task.values())
    total_skip = sum(t["skipped"] for t in per_task.values())
    sr = (total_succ / total_eps) if total_eps else 0.0

    weighted_inf = (
        sum(t["inf_ms"] * t["n_episodes"] for t in per_task.values()) / total_eps
        if total_eps else 0.0
    )

    n_acts = [t["n_action_steps"] for t in per_task.values()
              if t.get("n_action_steps") is not None]
    if n_acts:
        if len(set(n_acts)) > 1:
            print(f"warning: {model}: n_action_steps disagrees across tasks: "
                  f"{sorted(set(n_acts))}", file=sys.stderr)
        n_act = max(set(n_acts), key=n_acts.count)
        n_act_source = "summary"
    else:
        n_act = None
        n_act_source = "unknown"
    return model, {
        "n_tasks": len(per_task),
        "total_succ": total_succ,
        "total_eps": total_eps,
        "total_skip": total_skip,
        "sr": sr,
        "avg_inf_ms": weighted_inf,
        "n_action_steps": n_act,
        "n_act_source": n_act_source,
    }

def reproducibility_block(gguf_paths: list[Path], libero_venv: Path | None) -> str | None:

    script = REPO_ROOT / "scripts" / "print_versions.sh"
    if not script.is_file():
        print(f"warning: {script} not found - skipping ## Reproducibility block",
              file=sys.stderr)
        return None
    env = os.environ.copy()
    if libero_venv is not None and libero_venv.is_dir():
        env["VLA_LIBERO_VENV"] = str(libero_venv)
    cmd = ["bash", str(script), *(str(p) for p in gguf_paths)]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        stderr = (getattr(e, "stderr", "") or "").strip()
        msg = f"warning: {script.name} failed ({e}) - skipping ## Reproducibility block"
        print(msg + (f"\n{stderr}" if stderr else ""), file=sys.stderr)
        return None
    return proc.stdout.strip() or None

def render_markdown(
    sweep: Path,
    rows: list[tuple[str, dict]],
    server_stats: dict[str, dict],
    mem_stats: dict[str, dict],
    per_task_all: dict[str, dict],
    repro_block: str | None = None,
) -> str:
    from datetime import datetime
    lines: list[str] = []
    lines.append(f"# LIBERO sweep report - `{sweep.name}`")
    lines.append("")
    lines.append(f"- Sweep root: `{sweep}`")
    lines.append(f"- Suite: `{TASK_SUITE}` (tasks 0..{N_TASKS - 1})")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    if repro_block:
        lines.append(repro_block)
        lines.append("")

    lines.append("## Success rate & client-side inference time")
    lines.append("")
    lines.append("- **SR** counts terminated episodes as failures: `successes / n_episodes`.")
    lines.append("- **client/step** - wall-time per env step (amortized over chunk replay; "
                 "matches `Average inference time per step` in each `summary.txt`).")
    lines.append("- **client/call** = `client/step × n_action_steps` - wall-time per actual "
                 "`vla-server` call. Includes client pre/post + ZMQ transport (TCP loopback) "
                 "+ server compute.")
    lines.append("")
    lines.append("| Model | n_act | Tasks | Successes | Terminated | SR | client/step (ms) | client/call (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in rows:
        n_act = r["n_action_steps"]
        n_act_str = str(n_act) if n_act is not None else "?"
        per_call = r["avg_inf_ms"] * n_act if n_act is not None else None
        per_call_str = f"{per_call:.2f}" if per_call is not None else "?"
        lines.append(
            f"| `{name}` | {n_act_str} | {r['n_tasks']} | "
            f"{r['total_succ']}/{r['total_eps']} | "
            f"{r['total_skip']}/{r['total_eps']} | "
            f"{r['sr']:.2%} | "
            f"{r['avg_inf_ms']:.2f} | "
            f"{per_call_str} |"
        )
    lines.append("")

    if server_stats:
        lines.append("## Server-side inference breakdown")
        lines.append("")
        lines.append("Parsed from `_server_logs/<arch>.log` lines:")
        lines.append("")
        lines.append("```")
        lines.append("vla-server: rid=…  served=…  total=… ms  vision=…  inf=…  other=…")
        lines.append("```")
        lines.append("")
        lines.append("These are server-side measurements only - they exclude ZMQ transport "
                     "and client pre/post. `total = vision + inf + other`.")
        lines.append("")
        lines.append("| Model | Samples | total (ms) | vision | inf | other |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, _ in rows:
            s = server_stats.get(name)
            if s is None:
                lines.append(f"| `{name}` | (no log) | - | - | - | - |")
                continue
            lines.append(
                f"| `{name}` | {s['n_samples']} | "
                f"{s['total']:.2f} | {s['vision']:.2f} | {s['inf']:.2f} | {s['other']:.2f} |"
            )
        lines.append("")

        lines.append("### Transport + client overhead")
        lines.append("")
        lines.append("`overhead = client/call − server total` - time spent outside vla-server "
                     "(ZMQ over loopback + client preprocessing + protobuf round-trip).")
        lines.append("")
        lines.append("| Model | client/call (ms) | server total (ms) | overhead (ms) |")
        lines.append("|---|---:|---:|---:|")
        for name, r in rows:
            s = server_stats.get(name)
            n_act = r["n_action_steps"]
            if s is None or n_act is None:
                continue
            client_call = r["avg_inf_ms"] * n_act
            lines.append(
                f"| `{name}` | {client_call:.2f} | {s['total']:.2f} | "
                f"{client_call - s['total']:.2f} |"
            )
        lines.append("")

    lines.append("## Peak memory")
    lines.append("")
    lines.append("Sampled by the inline `mem_sampler` function in "
                 "[`eval/run_libero.sh`](../../eval/run_libero.sh) while "
                 "`vla-server` was alive:")
    lines.append("")
    lines.append("- **Peak VRAM** - max of per-PID `used_memory` from "
                 "`nvidia-smi --query-compute-apps`, polled every 1s. `(no GPU)` on "
                 "Tegra/Jetson, which doesn't support that query.")
    lines.append("- **Peak RAM** - `VmHWM` from `/proc/<pid>/status` (kernel-tracked "
                 "high-water mark of resident memory). Host only - does **not** include "
                 "the iGPU's unified-memory allocations.")
    lines.append("- **Peak sys RAM** / **sys Δ** - peak system-wide used RAM "
                 "(`MemTotal - MemAvailable`) and its rise over the sampler-start "
                 "baseline. On Tegra (unified memory) this is the only metric that "
                 "captures the iGPU weights VRAM/VmHWM miss; the Δ is an upper bound on "
                 "the server's footprint (a co-resident client/sim is included).")
    lines.append("")
    if not mem_stats:
        lines.append("_No `<arch>.mem.json` files found - these runs predate the sampler. "
                     "Re-run `eval/run_libero.sh` to capture peak memory._")
        lines.append("")
    else:
        lines.append("| Model | Peak VRAM (MiB) | Peak RAM (MiB) | Peak sys RAM (MiB) | sys Δ (MiB) | Samples |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, _ in rows:
            m = mem_stats.get(name)
            if m is None:
                lines.append(f"| `{name}` | n/a | n/a | n/a | n/a | n/a |")
                continue
            vram = m.get("peak_vram_mib")
            vram_str = f"{vram}" if isinstance(vram, int) else "(no GPU)"
            sys_peak = m.get("peak_sys_used_mib")
            sys_delta = m.get("sys_used_delta_mib")
            sys_peak_str = f"{sys_peak:.1f}" if isinstance(sys_peak, (int, float)) else "n/a"
            sys_delta_str = f"{sys_delta:.1f}" if isinstance(sys_delta, (int, float)) else "n/a"
            lines.append(
                f"| `{name}` | {vram_str} | "
                f"{m.get('peak_rss_mib', 0):.1f} | "
                f"{sys_peak_str} | {sys_delta_str} | "
                f"{m.get('samples', 0)} |"
            )
        lines.append("")

    lines.append("## Per-task breakdown")
    lines.append("")
    for name, per_task in per_task_all.items():
        lines.append(f"<details><summary><code>{name}</code></summary>")
        lines.append("")
        lines.append("| Task | Successes | Terminated | SR | client/step (ms) |")
        lines.append("|---|---:|---:|---:|---:|")
        for task_id in sorted(per_task):
            t = per_task[task_id]
            sr = t["successes"] / t["n_episodes"] if t["n_episodes"] else 0.0
            lines.append(
                f"| task_{task_id} | {t['successes']}/{t['n_episodes']} | "
                f"{t['skipped']}/{t['n_episodes']} | {sr:.2%} | {t['inf_ms']:.2f} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP,
                    help=f"sweep root (default: {DEFAULT_SWEEP})")
    ap.add_argument("--per-task", action="store_true",
                    help="also print per-task breakdown for each model to stdout")
    ap.add_argument("--md", type=Path, default=None, metavar="PATH",
                    help="write a markdown report to PATH (default: <sweep>/report.md). "
                         "Pass --no-md to skip.")
    ap.add_argument("--no-md", action="store_true",
                    help="do not write a markdown report")
    ap.add_argument("--gguf", type=Path, nargs="*", default=[], metavar="PATH",
                    help="GGUF files to checksum in the ## Reproducibility block "
                         "(combined ckpt + mmproj where applicable)")
    ap.add_argument("--libero-venv", type=Path,
                    default=REPO_ROOT / "eval" / "sim" / "libero" / "libero_uv" / ".venv",
                    help="LIBERO uv venv whose torch/transformers/lerobot/numpy pins are "
                         "recorded in the ## Reproducibility block")
    ap.add_argument("--no-reproducibility", action="store_true",
                    help="omit the ## Reproducibility version block from the markdown report")
    args = ap.parse_args()

    if not args.sweep.is_dir():
        print(f"ERROR: sweep dir not found: {args.sweep}", file=sys.stderr)
        return 1

    models = sorted(
        p for p in args.sweep.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    )

    server_logs_dir = args.sweep / "_server_logs"

    rows: list[tuple[str, dict]] = []
    per_task_all: dict[str, dict] = {}
    server_stats: dict[str, dict] = {}
    mem_stats: dict[str, dict] = {}
    for m in models:
        per_task = collect_model(m)
        if per_task is None:
            print(f"warning: no summaries under {m}", file=sys.stderr)
            continue
        name, row = fmt_row(m.name, per_task)
        rows.append((name, row))
        per_task_all[name] = per_task
        srv = parse_server_log(server_logs_dir / f"{name}.log")
        if srv is None:
            print(f"warning: no server timings parsed from {server_logs_dir / f'{name}.log'}",
                  file=sys.stderr)
        else:
            server_stats[name] = srv
        mem = parse_mem_json(server_logs_dir / f"{name}.mem.json")
        if mem is not None:
            mem_stats[name] = mem

    if not rows:
        print("No results found.", file=sys.stderr)
        return 1

    print(f"Sweep: {args.sweep}")
    print(f"Suite: {TASK_SUITE}  (tasks 0..{N_TASKS - 1})")
    print()
    print("Success / inference summary")
    print("  client/step = wall-time per env step, client-side (amortized over chunk replay).")
    print("  client/call = wall-time per actual vla-server call = client/step * n_action_steps;")
    print("                includes client pre/post + ZMQ transport (TCP loopback) + server compute.")
    print()
    sr_header = (
        f"{'model':<10} {'n_act':>6} {'tasks':>6} {'success':>10} {'terminated':>12} "
        f"{'SR':>8} {'client/step (ms)':>18} {'client/call (ms)':>18}"
    )
    print(sr_header)
    print("-" * len(sr_header))
    for name, r in rows:
        n_act = r["n_action_steps"]
        n_act_str = str(n_act) if n_act is not None else "?"
        per_call = r["avg_inf_ms"] * n_act if n_act is not None else None
        per_call_str = f"{per_call:>18.2f}" if per_call is not None else f"{'?':>18}"
        print(
            f"{name:<10} {n_act_str:>6} {r['n_tasks']:>6} "
            f"{r['total_succ']:>4}/{r['total_eps']:<5} "
            f"{r['total_skip']:>5}/{r['total_eps']:<6} "
            f"{r['sr']:>7.2%} "
            f"{r['avg_inf_ms']:>18.2f} "
            f"{per_call_str}"
        )

    if server_stats:
        print()
        print("Server-side inference breakdown (parsed from _server_logs/<arch>.log)")
        print("  Excludes ZMQ transport and client pre/post. total = vision + inf + other.")
        srv_header = (
            f"{'model':<10} {'samples':>8} {'total (ms)':>12} "
            f"{'vision':>10} {'inf':>10} {'other':>10}"
        )
        print(srv_header)
        print("-" * len(srv_header))
        for name, _ in rows:
            s = server_stats.get(name)
            if s is None:
                print(f"{name:<10} {'(no log)':>8}")
                continue
            print(
                f"{name:<10} {s['n_samples']:>8} "
                f"{s['total']:>12.2f} "
                f"{s['vision']:>10.2f} "
                f"{s['inf']:>10.2f} "
                f"{s['other']:>10.2f}"
            )

        print()
        print("Transport + client overhead (client/call - server total), per arch:")
        for name, r in rows:
            s = server_stats.get(name)
            n_act = r["n_action_steps"]
            if s is None or n_act is None:
                continue
            client_call = next(r["avg_inf_ms"] * n_act for n, r in rows if n == name)
            print(f"  {name:<10} client/call={client_call:8.2f}  server total={s['total']:8.2f}  "
                  f"overhead={client_call - s['total']:7.2f} ms")

    print()
    print("Peak memory (sampled while vla-server was alive; from _server_logs/<arch>.mem.json)")
    if not mem_stats:
        print("  No <arch>.mem.json files found - legacy runs predate the sampler. "
              "Re-run eval/run_libero.sh to capture peak memory.")
    else:
        mem_header = (f"{'model':<10} {'VRAM (MiB)':>12} {'RAM (MiB)':>12} "
                      f"{'sysRAM (MiB)':>13} {'sysDelta(MiB)':>13} {'samples':>9}")
        print(mem_header)
        print("-" * len(mem_header))
        for name, _ in rows:
            m = mem_stats.get(name)
            if m is None:
                print(f"{name:<10} {'n/a':>12} {'n/a':>12} {'n/a':>13} {'n/a':>13} {'n/a':>9}")
                continue
            vram = m.get("peak_vram_mib")
            vram_str = f"{vram:>12d}" if isinstance(vram, int) else f"{'(no GPU)':>12}"
            sys_peak = m.get("peak_sys_used_mib")
            sys_delta = m.get("sys_used_delta_mib")
            sys_peak_str = f"{sys_peak:>13.1f}" if isinstance(sys_peak, (int, float)) else f"{'n/a':>13}"
            sys_delta_str = f"{sys_delta:>13.1f}" if isinstance(sys_delta, (int, float)) else f"{'n/a':>13}"
            print(
                f"{name:<10} {vram_str} "
                f"{m.get('peak_rss_mib', 0):>12.1f} "
                f"{sys_peak_str} {sys_delta_str} "
                f"{m.get('samples', 0):>9d}"
            )

    if not args.no_md:
        md_path = args.md if args.md is not None else (args.sweep / "report.md")
        repro_block = (None if args.no_reproducibility
                       else reproducibility_block(args.gguf, args.libero_venv))
        md_path.write_text(
            render_markdown(args.sweep, rows, server_stats, mem_stats, per_task_all,
                            repro_block),
            encoding="utf-8",
        )
        print()
        print(f"Markdown report written to: {md_path}")

    if args.per_task:
        for name, per_task in per_task_all.items():
            print()
            print(f"[{name}] per-task (terminated counted as failures)")
            print(f"  {'task':<6} {'success':>10} {'terminated':>12} {'SR':>8} {'inf (ms)':>10}")
            for task_id in sorted(per_task):
                t = per_task[task_id]
                sr = t["successes"] / t["n_episodes"] if t["n_episodes"] else 0.0
                print(
                    f"  task_{task_id:<2} "
                    f"{t['successes']:>4}/{t['n_episodes']:<5} "
                    f"{t['skipped']:>5}/{t['n_episodes']:<6} "
                    f"{sr:>7.2%} "
                    f"{t['inf_ms']:>10.2f}"
                )

    return 0

if __name__ == "__main__":
    sys.exit(main())
