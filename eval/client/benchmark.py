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
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym

import sim.libero  # noqa: F401  side-effect: registers gymnasium envs


def _find_server_pid(addr: str) -> int | None:

    m = re.search(r"tcp://[^:]+:(\d+)$", addr)
    if not m:
        return None
    port = m.group(1)
    try:
        out = subprocess.check_output(["ss", "-tlnp"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if f":{port} " not in line:
            continue

        m2 = re.search(r"pid=(\d+)", line)
        if m2:
            return int(m2.group(1))
    return None

def _sample_vram_for_pid(pid: int) -> int | None:

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None

class VramSampler(threading.Thread):

    def __init__(self, pid: int, interval_s: float = 0.25):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval_s = interval_s
        self.samples_mib: list[int] = []

        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            v = _sample_vram_for_pid(self.pid)
            if v is not None:
                self.samples_mib.append(v)
            self._stop_evt.wait(self.interval_s)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)

def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[idx]

def _make_client(backend: str, addr: str):

    if backend == "lerobot":
        from utils.service import RobotInferenceClient
        return RobotInferenceClient(host=_host_of(addr), port=_port_of(addr))
    elif backend == "vla-cpp":
        from client.vla_cpp_client import VlaCppClient
        client = VlaCppClient(vla_addr=addr)
        return client
    else:
        raise ValueError(f"unknown backend: {backend}")

def _host_of(addr: str) -> str:
    m = re.match(r"tcp://([^:]+):(\d+)$", addr)
    return m.group(1) if m else "localhost"

def _port_of(addr: str) -> int:
    m = re.match(r"tcp://([^:]+):(\d+)$", addr)
    return int(m.group(2)) if m else 5555

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["lerobot", "vla-cpp"], required=True)
    ap.add_argument("--addr", required=True,
        help="ZMQ address of the running server (tcp://host:port)")
    ap.add_argument("--server-pid", type=int, default=None,
        help="Server PID for VRAM sampling. If omitted, derived from --addr.")
    ap.add_argument("--task", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=200,
        help="Total inference calls. Episodes restart if they terminate early.")
    ap.add_argument("--warmup-steps", type=int, default=5,
        help="Steps run before timing starts (lets caches / cuDNN settle).")
    ap.add_argument("--vram-interval-s", type=float, default=0.25)
    ap.add_argument("--output", type=Path, required=True,
        help="Path to write the stats JSON.")
    args = ap.parse_args()

    pid = args.server_pid or _find_server_pid(args.addr)
    if pid is None:
        print(f"warning: could not derive server PID from {args.addr}; "
              f"VRAM stats will be empty", flush=True)
    else:
        print(f"server PID = {pid}", flush=True)

        v = _sample_vram_for_pid(pid)
        print(f"VRAM (pre-warmup) = {v} MiB", flush=True)

    print(f"connecting to {args.addr} as backend={args.backend} ...", flush=True)
    client = _make_client(args.backend, args.addr)

    output_dir = args.output.parent / "_bench_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        f"{args.task}/task_{args.task_id}",
        video_fps=30,
        output_video_dir=output_dir,
        video_view_mode="single-view",
    )

    print(f"warmup ({args.warmup_steps} steps) ...", flush=True)
    obs, _info = env.reset()
    for _ in range(args.warmup_steps):
        action = client.get_action(obs)
        try:
            obs, _r, done, trunc, _info = env.step(action)
        except ValueError:
            obs, _info = env.reset()
            continue
        if done or trunc:
            obs, _info = env.reset()

    sampler = None
    if pid is not None:
        sampler = VramSampler(pid, interval_s=args.vram_interval_s)
        sampler.start()

    print(f"benchmarking {args.n_steps} steps ...", flush=True)
    step_latencies_ms: list[float] = []
    server_latencies: list[dict] = []
    n_inference_calls    = 0   # = len(step_latencies_ms); loop terminates on this
    n_env_step_ok        = 0   # env.step accepted the action
    n_env_step_failed    = 0   # env.step raised ValueError (action rejected)
    n_episodes_terminated = 0  # env.step returned done or trunc
    t_run0 = time.time()
    while n_inference_calls < args.n_steps:
        t0 = time.time()
        action = client.get_action(obs)
        t1 = time.time()
        step_latencies_ms.append(1000.0 * (t1 - t0))
        n_inference_calls += 1

        if args.backend == "vla-cpp" and hasattr(client, "_last_response"):
            r = client._last_response
            if r is not None:
                server_latencies.append({
                    "total":     r.latency_ms_total,
                    "vision":    r.latency_ms_vision,
                    "inference": r.latency_ms_inference,
                    "prefill":   r.latency_ms_prefill,
                    "denoise":   r.latency_ms_denoise,
                })

        try:
            obs, _r, done, trunc, _info = env.step(action)
        except ValueError:
            n_env_step_failed += 1
            obs, _info = env.reset()
            continue
        n_env_step_ok += 1
        if done or trunc:
            n_episodes_terminated += 1
            obs, _info = env.reset()
    t_run1 = time.time()
    env.close()

    if sampler is not None:
        sampler.stop()

    vram = sampler.samples_mib if sampler else []
    stats = {
        "backend":      args.backend,
        "addr":         args.addr,
        "task":         args.task,
        "task_id":      args.task_id,
        "n_steps":      args.n_steps,
        "warmup_steps": args.warmup_steps,
        "wall_time_s":  round(t_run1 - t_run0, 3),
        "counters": {
            "inference_calls":     n_inference_calls,
            "env_step_ok":         n_env_step_ok,
            "env_step_failed":     n_env_step_failed,
            "episodes_terminated": n_episodes_terminated,
        },
        "step_ms": {
            "n":      len(step_latencies_ms),
            "mean":   round(statistics.fmean(step_latencies_ms), 3),
            "median": round(statistics.median(step_latencies_ms), 3),
            "p95":    round(_percentile(step_latencies_ms, 0.95), 3),
            "p99":    round(_percentile(step_latencies_ms, 0.99), 3),
            "min":    round(min(step_latencies_ms), 3),
            "max":    round(max(step_latencies_ms), 3),
        },
        "vram_mib": {
            "n_samples": len(vram),
            "peak":      max(vram) if vram else None,
            "mean":      round(statistics.fmean(vram), 1) if vram else None,
        },
        "server_latency_breakdown": server_latencies,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2))

    print(f"\n=== {args.backend} on {args.task}/{args.task_id} ===")
    print(f"counters        : inference_calls={n_inference_calls}  "
          f"env_step_ok={n_env_step_ok}  "
          f"env_step_failed={n_env_step_failed}  "
          f"episodes_terminated={n_episodes_terminated}")
    print(f"steps           : {stats['step_ms']['n']}  ({stats['wall_time_s']} s wall)")
    print(f"step ms (client): mean={stats['step_ms']['mean']}  "
          f"med={stats['step_ms']['median']}  "
          f"p95={stats['step_ms']['p95']}  "
          f"p99={stats['step_ms']['p99']}  "
          f"max={stats['step_ms']['max']}")
    if vram:
        print(f"VRAM MiB        : peak={stats['vram_mib']['peak']}  "
              f"mean={stats['vram_mib']['mean']}  ({len(vram)} samples)")
    if server_latencies:
        ms = [s["total"] for s in server_latencies]
        v  = [s["vision"] for s in server_latencies]
        i  = [s["inference"] for s in server_latencies]
        print(f"server-internal : total={statistics.fmean(ms):.1f}  "
              f"vision={statistics.fmean(v):.1f}  "
              f"inference={statistics.fmean(i):.1f}  (means)")
    print(f"\nwrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
