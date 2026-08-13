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

"""Latency and memory instrumentation for the upstream PyTorch BitVLA stack.

Two timers, because they answer different questions:

  * ``get_action``      — everything the control loop pays per query: the TF
    JPEG round-trip + lanczos3 resize, the center crop, tokenization, and the
    forward. This is BitVLA's analogue of ``select_action`` in the other
    PyTorch reference servers under ``eval/pytorch_ref``.
  * ``predict_action``  — the model forward alone. vla.cpp's server-reported
    latency excludes host-side image preprocessing, so this is the number that
    lines up with it.

CUDA kernel launches are asynchronous, so both timers synchronize on each side
of the call; without that they would measure launch overhead, not compute.

Memory is reported four ways because they are not interchangeable:

  * ``weights_bytes``   — allocator bytes resident right after the model, the
    action head and the proprio projector are on the device. This is what the
    checkpoint costs, and it is the number the BitVLA paper's "Memory Usage"
    column is about — except the released checkpoint holds BF16 *master*
    weights and quantizes online, so the measured value is the BF16 cost, not
    the 1.58-bit one.
  * ``peak_allocated``  — allocator high-water mark during the timed window
    (weights + activations + KV cache).
  * ``peak_reserved``   — what the caching allocator held from the driver.
  * ``peak_vram``       — what the driver charges the process: reserved plus
    the CUDA context, cuBLAS/cuDNN workspaces and any EGL rendering surfaces.
    This is the number ``nvidia-smi`` shows, and the only one that reflects
    what the GPU cannot give to anything else. Sampled on a background thread
    and reported as a maximum, which is what ``mem_sampler_linux`` in
    ``ci/lib/common.sh`` does for the vla.cpp server — so ``peak_vram_mib``
    here and in the ``*.mem.json`` files mean the same thing.
"""

from __future__ import annotations

import json
import os
import resource
import statistics
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

import torch


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Accumulates per-call wall times (ms) around a wrapped callable."""

    def __init__(self, name: str):
        self.name = name
        self.samples: List[float] = []

    def wrap(self, fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            _cuda_sync()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            _cuda_sync()
            self.samples.append(1000.0 * (time.perf_counter() - t0))
            return out

        return wrapper

    def reset(self) -> int:
        """Drop samples collected so far; returns how many were discarded."""
        n = len(self.samples)
        self.samples.clear()
        return n

    def stats(self) -> Dict[str, float]:
        s = sorted(self.samples)
        if not s:
            return {"n": 0}

        def pct(p: float) -> float:
            if len(s) == 1:
                return s[0]
            idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
            return s[idx]

        return {
            "n": len(s),
            "mean_ms": statistics.fmean(s),
            "median_ms": statistics.median(s),
            "min_ms": s[0],
            "max_ms": s[-1],
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "std_ms": statistics.pstdev(s) if len(s) > 1 else 0.0,
        }


def nvidia_smi_process_mib(device_index: int = 0) -> Optional[float]:
    """GPU memory the driver charges to this PID, in MiB.

    Falls back to ``None`` when nvidia-smi is unavailable or the PID has not
    shown up in the compute-apps table yet.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    pid = str(os.getpid())
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == pid:
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def parameter_report(modules: Dict[str, torch.nn.Module]) -> Dict[str, object]:
    """Per-module parameter counts and bytes, split by dtype.

    BitVLA ships BF16 master weights and ternarizes on the fly, so the byte
    total here is the deployed cost of the released checkpoint — not the
    1.58-bit figure the paper quotes.
    """
    per_module = {}
    total_params = 0
    total_bytes = 0
    dtype_params: Dict[str, int] = {}
    for name, mod in modules.items():
        if mod is None:
            continue
        n_params = 0
        n_bytes = 0
        for p in mod.parameters():
            n_params += p.numel()
            n_bytes += p.numel() * p.element_size()
            key = str(p.dtype).replace("torch.", "")
            dtype_params[key] = dtype_params.get(key, 0) + p.numel()
        for b in mod.buffers():
            n_bytes += b.numel() * b.element_size()
        per_module[name] = {"params": n_params, "bytes": n_bytes}
        total_params += n_params
        total_bytes += n_bytes
    return {
        "per_module": per_module,
        "total_params": total_params,
        "total_bytes": total_bytes,
        "params_by_dtype": dtype_params,
    }


class MemoryProbe:
    """Snapshots CUDA allocator state and samples driver-side VRAM."""

    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.2):
        self.device_index = device_index
        self.sample_interval_s = sample_interval_s
        self.weights_bytes: Optional[int] = None
        self.peak_allocated: Optional[int] = None
        self.peak_reserved: Optional[int] = None
        self.peak_vram_mib: Optional[float] = None
        self.vram_samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample_loop(self) -> None:
        # A single reading at the end would miss the high-water mark: the
        # caching allocator does not return memory to the driver, but MuJoCo's
        # renderer and cuBLAS workspaces come and go.
        while not self._stop.is_set():
            mib = nvidia_smi_process_mib(self.device_index)
            if mib is not None:
                self.vram_samples += 1
                if self.peak_vram_mib is None or mib > self.peak_vram_mib:
                    self.peak_vram_mib = mib
            self._stop.wait(self.sample_interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def mark_weights_loaded(self) -> None:
        """Record resident bytes with the model on device but nothing run yet."""
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        self.weights_bytes = torch.cuda.memory_allocated(self.device_index)
        torch.cuda.reset_peak_memory_stats(self.device_index)

    def finish(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_allocated = torch.cuda.max_memory_allocated(self.device_index)
            self.peak_reserved = torch.cuda.max_memory_reserved(self.device_index)
        # One last reading before the sampler stops, so a short run that never
        # completed a sampling tick still reports something.
        mib = nvidia_smi_process_mib(self.device_index)
        if mib is not None and (self.peak_vram_mib is None or mib > self.peak_vram_mib):
            self.peak_vram_mib = mib
            self.vram_samples += 1
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def report(self) -> Dict[str, object]:
        mib = 1024.0 * 1024.0
        out: Dict[str, object] = {}
        if self.weights_bytes is not None:
            out["weights_mib"] = self.weights_bytes / mib
        if self.peak_allocated is not None:
            out["peak_allocated_mib"] = self.peak_allocated / mib
        if self.peak_reserved is not None:
            out["peak_reserved_mib"] = self.peak_reserved / mib
        if self.peak_vram_mib is not None:
            out["peak_vram_mib"] = self.peak_vram_mib
            out["vram_samples"] = self.vram_samples
        out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        return out


def write_report(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[instrument] wrote {path}")


def print_summary(payload: Dict[str, object]) -> None:
    print("\n" + "=" * 72)
    print("BitVLA (PyTorch) — latency and memory")
    print("=" * 72)
    for key in ("get_action", "predict_action"):
        st = payload.get("latency", {}).get(key)
        if not st or not st.get("n"):
            continue
        print(
            f"{key:<16} n={st['n']:<5d} mean {st['mean_ms']:7.2f} ms   "
            f"median {st['median_ms']:7.2f}   p95 {st['p95_ms']:7.2f}   "
            f"min {st['min_ms']:7.2f}   max {st['max_ms']:7.2f}"
        )
    mem = payload.get("memory", {})
    for key, label in (
        ("weights_mib", "weights on device"),
        ("peak_allocated_mib", "peak allocated"),
        ("peak_reserved_mib", "peak reserved"),
        ("peak_vram_mib", "peak VRAM (nvidia-smi)"),
        ("peak_rss_mib", "peak host RSS"),
    ):
        if key in mem:
            print(f"{label:<22} {mem[key]:9.1f} MiB")
    params = payload.get("parameters", {})
    if params:
        print(
            f"{'parameters':<22} {params.get('total_params', 0) / 1e9:9.3f} B   "
            f"({params.get('total_bytes', 0) / (1024 ** 3):.2f} GiB on device)"
        )
        for dt, n in sorted(params.get("params_by_dtype", {}).items()):
            print(f"  {dt:<20} {n / 1e9:9.3f} B params")
    print("=" * 72 + "\n")
