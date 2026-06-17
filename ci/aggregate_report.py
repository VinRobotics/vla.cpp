#!/usr/bin/env python3
# Copyright 2026 VinRobotics - Apache-2.0
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
"""Aggregate per-platform CI verdicts into one cross-platform report.

After `orchestrate.sh all`, every platform's sweep has been gated by
check_thresholds.py into <root>/<platform>/verdict.json, which already carries the
server-side metrics the orchestrator obtained from each server after the sim eval
(latency breakdown total/vision/inf/other, peak memory) plus per-suite SR. This
merges them into <root>/report.{md,json}: the commit under test, each platform's
pass/fail, and a per-model table of those server-side numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="CI_OUTPUT_ROOT (per-platform dirs)")
    ap.add_argument("--platforms", required=True, help="space-separated platform keys")
    ap.add_argument("--commit", default="")
    args = ap.parse_args()

    report = {"commit": args.commit, "ok": True, "platforms": {}}
    body: list[str] = []

    for p in args.platforms.split():
        v = load(args.root / p / "verdict.json")
        ctxt = args.root / p / "commit.txt"
        deployed = ctxt.read_text().strip() if ctxt.is_file() else ""
        if v is None:
            report["ok"] = False
            report["platforms"][p] = {"ok": False, "error": "no verdict.json"}
            body += [f"## `{p}` - ❌ no verdict.json (sweep did not complete)", ""]
            continue
        report["platforms"][p] = {"ok": v.get("ok", False), "commit": deployed,
                                  "results": v.get("results", [])}
        if not v.get("ok"):
            report["ok"] = False
        # Server's own commit (check_commits.sh already verified the tested
        # machines agree; it may differ from the orchestrator's - that's a warning,
        # not a failure).
        cnote = f"  · server commit `{deployed[:12]}`" if deployed else ""
        body += [f"## `{p}` - {'PASS ✅' if v.get('ok') else 'FAIL ❌'}{cnote}", "",
                 "| model | per-suite SR | server ms total (vision/inf/other) | peak mem |",
                 "|---|---|---|---|"]
        for r in v.get("results", []):
            sr = ", ".join(f"{s.replace('libero_','')}:{d['successes']}/{d['episodes']}"
                           for s, d in r.get("suites", {}).items()) or "-"
            b = r.get("server_breakdown_ms")
            lat = (f"{b['total']} ({b['vision']}/{b['inf']}/{b['other']})" if b
                   else str(r.get("server_total_ms", "-")))
            mem = r.get("mem", {}) or {}
            mems = ", ".join(f"{k}={val}" for k, val in mem.items() if val is not None) or "-"
            body += [f"| `{r['model']}` | {sr} | {lat} | {mems} |"]
        body += [""]

    header = [f"# vla-ci cross-platform report - {'PASS ✅' if report['ok'] else 'FAIL ❌'}",
              "", f"Orchestrator commit: `{args.commit or 'unknown'}` "
              "(servers self-manage their own checkout; their commits are shown per platform "
              "and were verified consistent across tested machines before the run).",
              "", "Server-side latency (ms/call) and memory are measured on each platform's "
              "own machine and fetched back after the sim eval; SR is client-side.", ""]
    text = "\n".join(header + body)

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "report.json").write_text(json.dumps(report, indent=2))
    (args.root / "report.md").write_text(text)
    print(text)
    print(f"[aggregate] wrote {args.root / 'report.md'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
