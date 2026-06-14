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
"""CI gate for a vla.cpp sweep.

Reads a sweep directory produced by the CI runners (per-task ``summary.txt``,
``_server_logs/<arch>.<suite>.log``, ``_server_logs/<arch>.<suite>.mem.json``),
compares each model against the committed baseline for the platform, and emits a
verdict.

Gates (all must pass for exit 0):
  * SR > 0                          per (model, suite)   - "must be a positive number"
  * server latency <= tol*baseline  per model            - mean `total` ms / call
  * server memory   <= tol*baseline per model            - platform mem metric
                                                           (skipped where the
                                                           baseline mem_metric is
                                                           null, e.g. the M4).

Latency/memory gate per *model* against one committed baseline, but per-suite
models (bitvla, gr00t_n1_7) run a separate server per suite, so latency is the
sample-weighted mean across those runs and memory is the peak across them. SR is
per (model, suite). "In range (or better)" => actual <= tol*baseline; lower
always passes.

The per-file parsers are reused from eval/collect_libero_results.py so this stays
in lock-step with how the reports are generated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from collect_libero_results import (  # noqa: E402  (path injected above)
    parse_mem_json,
    parse_server_log,
    parse_summary,
)

N_TASKS = 10
SUITES_ALL = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def discover_suites(model_dir: Path) -> list[str]:
    """Suites that actually produced output under this model dir, ordered."""
    found = []
    for suite in SUITES_ALL:
        if list(model_dir.glob(f"**/{suite}/task_*/summary.txt")):
            found.append(suite)
    return found


def _server_logs(logs_dir: Path, name: str) -> list[Path]:
    """A model's server logs. Per-suite models emit one ``<arch>.<suite>.log``
    per suite (a fresh process serves each suite's weights); single-suite models
    emit one. A legacy flat ``<arch>.log`` is honoured for old sweeps."""
    legacy = logs_dir / f"{name}.log"
    suite_logs = sorted(p for p in logs_dir.glob(f"{name}.*.log")
                        if not p.name.endswith(".launch.log"))
    return ([legacy] if legacy.is_file() else []) + suite_logs


def aggregate_server_latency(logs_dir: Path, name: str) -> dict | None:
    """Sample-weighted mean ``total`` ms across all of a model's server logs."""
    weighted, n = 0.0, 0
    for lp in _server_logs(logs_dir, name):
        srv = parse_server_log(lp)
        if srv:
            weighted += srv["total"] * srv["n_samples"]
            n += srv["n_samples"]
    return {"total": weighted / n, "n_samples": n} if n else None


def aggregate_server_mem(logs_dir: Path, name: str) -> list[dict]:
    """Parsed mem.json for every server run of a model (one per suite)."""
    legacy = logs_dir / f"{name}.mem.json"
    paths = ([legacy] if legacy.is_file() else []) + sorted(
        logs_dir.glob(f"{name}.*.mem.json"))
    return [m for m in (parse_mem_json(p) for p in paths) if m is not None]


def suite_sr(model_dir: Path, suite: str) -> tuple[int, int, list[int]]:
    """(successes, episodes, task_ids_seen) aggregated over a suite's tasks."""
    succ = eps = 0
    seen: list[int] = []
    for task_id in range(N_TASKS):
        hits = list(model_dir.glob(f"**/{suite}/task_{task_id}/summary.txt"))
        if not hits:
            continue
        s = parse_summary(hits[0])
        succ += s["successes"]
        eps += s["n_episodes"]
        seen.append(task_id)
    return succ, eps, seen


def check_model(name: str, model_dir: Path, logs_dir: Path, base: dict,
                latency_metric: str, mem_metric: str | None, tol: float) -> dict:
    res: dict = {"model": name, "suites": {}, "checks": [], "ok": True}

    def gate(ok: bool, label: str, detail: str):
        res["checks"].append({"ok": bool(ok), "label": label, "detail": detail})
        if not ok:
            res["ok"] = False

    # ---- SR per suite (gate: > 0) -----------------------------------------
    suites = discover_suites(model_dir)
    if not suites:
        gate(False, "outputs", f"no summary.txt found under {model_dir}")
        return res
    for suite in suites:
        succ, eps, seen = suite_sr(model_dir, suite)
        sr = succ / eps if eps else 0.0
        res["suites"][suite] = {"successes": succ, "episodes": eps,
                                "tasks": len(seen), "sr": sr}
        gate(succ > 0, f"SR>0 [{suite}]",
             f"{succ}/{eps} success ({sr:.1%}) over {len(seen)} tasks")

    # ---- server latency (gate: <= tol * baseline) -------------------------
    # Per-suite models run several server processes; aggregate (sample-weighted)
    # so the gate stays per-model against the single committed baseline.
    srv = aggregate_server_latency(logs_dir, name)
    base_lat = base.get(latency_metric)
    if srv is None:
        gate(False, "latency", f"no parsable server log for {name} under {logs_dir}")
    else:
        res["server_total_ms"] = round(srv["total"], 2)
        res["server_samples"] = srv["n_samples"]
        if base_lat is None:
            res["checks"].append({"ok": True, "label": "latency",
                                  "detail": f"{srv['total']:.2f} ms (no baseline - recorded only)"})
        else:
            limit = base_lat * tol
            gate(srv["total"] <= limit, "latency",
                 f"{srv['total']:.2f} ms vs {base_lat:.2f}*{tol:g}={limit:.2f} ms baseline")

    # ---- server memory (gate: <= tol * baseline; skip if no metric) -------
    # Peak across all of a model's server runs (one per suite for per-suite models).
    mems = aggregate_server_mem(logs_dir, name)
    if mem_metric is None:
        if mems:
            res["mem"] = {k: max((m.get(k) for m in mems if m.get(k) is not None),
                                 default=None)
                          for k in ("peak_rss_mib", "peak_sys_used_mib", "sys_used_delta_mib")}
        res["checks"].append({"ok": True, "label": "memory",
                              "detail": "not gated on this platform (no baseline)"})
    elif not mems:
        gate(False, "memory", f"no mem.json for {name} under {logs_dir}")
    else:
        vals = [m.get(mem_metric) for m in mems if m.get(mem_metric) is not None]
        actual = max(vals) if vals else None
        res["mem"] = {mem_metric: actual}
        base_mem = base.get(mem_metric)
        if actual is None:
            gate(False, "memory", f"{mem_metric} missing/null in mem.json")
        elif base_mem is None:
            res["checks"].append({"ok": True, "label": "memory",
                                  "detail": f"{actual} MiB (no baseline - recorded only)"})
        else:
            limit = base_mem * tol
            gate(actual <= limit, "memory",
                 f"{actual} MiB ({mem_metric}) vs {base_mem}*{tol:g}={limit:.1f} MiB baseline")
    return res


def render_md(platform: str, tol: float, results: list[dict]) -> str:
    overall = all(r["ok"] for r in results)
    out = [f"# CI gate - `{platform}`  {'PASS ✅' if overall else 'FAIL ❌'}",
           "", f"Tolerance: actual ≤ {tol:g}× reported baseline. SR gate: > 0.", ""]
    for r in results:
        out.append(f"## `{r['model']}`  {'✅' if r['ok'] else '❌'}")
        for c in r["checks"]:
            out.append(f"- {'✅' if c['ok'] else '❌'} **{c['label']}** - {c['detail']}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--sweep", required=True, type=Path,
                    help="sweep root (per-model dirs + _server_logs/)")
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these model dirs (default: all in baseline)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write verdict.json + verdict.md here (default: <sweep>)")
    args = ap.parse_args()

    spec = json.loads(args.baseline.read_text())
    tol = float(spec.get("tolerance", 1.10))
    lat_metric = spec.get("latency_metric", "server_total_ms")
    mem_metric = spec.get("mem_metric")
    base_models = spec["models"]

    logs_dir = args.sweep / "_server_logs"
    want = args.models or list(base_models.keys())

    results: list[dict] = []
    for name in want:
        if name not in base_models:
            results.append({"model": name, "ok": False,
                            "checks": [{"ok": False, "label": "baseline",
                                        "detail": f"no baseline entry for {name}"}],
                            "suites": {}})
            continue
        model_dir = args.sweep / name
        if not model_dir.is_dir():
            results.append({"model": name, "ok": False,
                            "checks": [{"ok": False, "label": "outputs",
                                        "detail": f"missing model dir {model_dir}"}],
                            "suites": {}})
            continue
        results.append(check_model(name, model_dir, logs_dir, base_models[name],
                                   lat_metric, mem_metric, tol))

    overall = bool(results) and all(r["ok"] for r in results)
    verdict = {"platform": args.platform, "tolerance": tol, "ok": overall,
               "results": results}

    out_dir = args.out or args.sweep
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    md = render_md(args.platform, tol, results)
    (out_dir / "verdict.md").write_text(md)
    print(md)
    print(f"\n[gate] {'PASS' if overall else 'FAIL'} - wrote {out_dir / 'verdict.json'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
