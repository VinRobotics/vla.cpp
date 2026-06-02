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
from pathlib import Path

def _fmt(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

def _ratio(a, b):
    if a is None or b is None or a == 0:
        return "-"
    return f"{b / a:+.2%}"

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="First benchmark JSON (baseline / left column)")
    ap.add_argument("b", type=Path, help="Second benchmark JSON (right column)")
    args = ap.parse_args()

    a = json.loads(args.a.read_text())
    b = json.loads(args.b.read_text())

    if a.get("task") != b.get("task") or a.get("task_id") != b.get("task_id"):
        print(f"warning: task mismatch ({a.get('task')}/{a.get('task_id')} vs "
              f"{b.get('task')}/{b.get('task_id')})")
    if a.get("n_steps") != b.get("n_steps"):
        print(f"warning: n_steps mismatch ({a.get('n_steps')} vs {b.get('n_steps')})")

    name_a = a["backend"]
    name_b = b["backend"]

    print()
    print(f"{'metric':<24}  {name_a:>14}  {name_b:>14}  {'b vs a':>10}")
    print("-" * 70)

    sa = a["step_ms"]; sb = b["step_ms"]
    rows = [
        ("step ms (mean)",   sa["mean"],   sb["mean"]),
        ("step ms (median)", sa["median"], sb["median"]),
        ("step ms (p95)",    sa["p95"],    sb["p95"]),
        ("step ms (p99)",    sa["p99"],    sb["p99"]),
        ("step ms (max)",    sa["max"],    sb["max"]),
    ]
    for name, va, vb in rows:
        print(f"{name:<24}  {_fmt(va):>14}  {_fmt(vb):>14}  {_ratio(va, vb):>10}")

    print()
    va = a["vram_mib"]; vb = b["vram_mib"]
    rows = [
        ("VRAM peak (MiB)",  va["peak"], vb["peak"]),
        ("VRAM mean (MiB)",  va["mean"], vb["mean"]),
    ]
    for name, x, y in rows:
        print(f"{name:<24}  {_fmt(x):>14}  {_fmt(y):>14}  {_ratio(x, y):>10}")

    print()
    print(f"wall (s)                 "
          f"{_fmt(a.get('wall_time_s')):>14}  "
          f"{_fmt(b.get('wall_time_s')):>14}")
    print(f"steps                    "
          f"{_fmt(sa.get('n')):>14}  {_fmt(sb.get('n')):>14}")

    for label, x in (("a", a), ("b", b)):
        sb_ = x.get("server_latency_breakdown") or []
        if sb_:
            n = len(sb_)
            tm = sum(s["total"]     for s in sb_) / n
            vm = sum(s["vision"]    for s in sb_) / n
            im = sum(s["inference"] for s in sb_) / n
            print(f"{x['backend']} server-internal mean: "
                  f"total={tm:.1f}  vision={vm:.1f}  inference={im:.1f}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
