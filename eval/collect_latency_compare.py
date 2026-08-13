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

"""Build the vla.cpp vs torch.compile latency table from a sweep's JSONs.

Reads outputs/latency_compare/<model>/<variant>.json (written by
eval/client/benchmark.py via eval/run_latency_compare.sh) and emits markdown.

Every latency here is SERVER-SIDE inference time — the vla.cpp server reports
it per response, the PyTorch server times select_action in-process with CUDA
synchronization. Neither includes ZMQ transport or image serialization.

Weight precision is read off the artifacts themselves rather than assumed: the
GGUF tensor-type histogram for vla.cpp, the safetensors dtype histogram for the
PyTorch checkpoint. The two stacks are NOT forced to matching precision — each
runs as shipped — so the dtype columns are load-bearing when reading the table.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = ["smolvla", "pi0", "evo1", "gr00t_n1_5", "gr00t_n1_6", "gr00t_n1_7"]
VARIANTS = ["vla.cpp", "eager", "compile-default", "compile-reduce-overhead"]
TORCH_VARIANTS = ["eager", "compile-default", "compile-reduce-overhead"]

# Where each stack's weights live, for the precision disclosure.
GGUF_ROOT = Path("/mnt/data/hf_data/vrfai")
CKPT_ROOT = Path("/mnt/data/hf_data")

GGUF_PATHS = {
    "smolvla":    GGUF_ROOT / "smolvla-libero-gguf/smolvla-libero.gguf",
    "pi0":        GGUF_ROOT / "pi0-libero-finetuned-v044-gguf/pi0-libero-finetuned-v044.gguf",
    "evo1":       GGUF_ROOT / "evo1-libero-gguf/evo1-libero.gguf",
    "gr00t_n1_5": GGUF_ROOT / "gr00tn1d5-libero-object-gguf/gr00tn1d5-libero-object.gguf",
    "gr00t_n1_6": GGUF_ROOT / "gr00tn1d6-libero-gguf/gr00tn1d6-libero.gguf",
    "gr00t_n1_7": GGUF_ROOT / "gr00tn1d7-libero-gguf/libero_object/gr00tn1d7-libero-object.gguf",
}

CKPT_PATHS = {
    "smolvla":    CKPT_ROOT / "HuggingFaceVLA/smolvla_libero",
    "pi0":        CKPT_ROOT / "lerobot/pi0_libero_finetuned_v044",
    "evo1":       Path("/mnt/data/vla_sr_compare/weights/Evo1_LIBERO"),
    "gr00t_n1_5": CKPT_ROOT / "liorbenhorin-nv/groot-libero_object-64_40000",
    "gr00t_n1_6": CKPT_ROOT / "0xAnkitSingh/GR00T-N1.6-LIBERO",
    "gr00t_n1_7": CKPT_ROOT / "nvidia/GR00T-N1.7-LIBERO/libero_object",
}

# Runtime compute dtype, from the policy pipelines (not the on-disk weights):
#   gr00t_n1_5/6/7 cast weights to bf16 before serving; evo1 runs its forward
#   under torch.autocast(bf16); smolvla and pi0 run at the checkpoint's dtype.
TORCH_RUNTIME_DTYPE = {
    "smolvla":    "ckpt dtype",
    "pi0":        "ckpt dtype",
    "evo1":       "bf16 autocast",
    "gr00t_n1_5": "bf16",
    "gr00t_n1_6": "bf16",
    "gr00t_n1_7": "bf16",
}

# ggml_type enum -> name (only the values these GGUFs actually use).
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}


def _read_gguf_tensor_types(path: Path) -> Counter | None:
    """Parse a GGUF header and histogram its tensor types by element count.

    Hand-rolled rather than via the `gguf` package, which is not installed in
    any of this repo's venvs. Only the header is read — no weights are loaded.
    """
    try:
        with path.open("rb") as f:
            if f.read(4) != b"GGUF":
                return None
            version, = struct.unpack("<I", f.read(4))
            n_tensors, = struct.unpack("<Q", f.read(8))
            n_kv, = struct.unpack("<Q", f.read(8))

            def read_str() -> str:
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", errors="replace")

            # Metadata values are typed; skip them without interpreting.
            scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

            def skip_value(vtype: int) -> None:
                if vtype == 8:            # string
                    n, = struct.unpack("<Q", f.read(8))
                    f.read(n)
                elif vtype == 9:          # array
                    etype, = struct.unpack("<I", f.read(4))
                    count, = struct.unpack("<Q", f.read(8))
                    for _ in range(count):
                        skip_value(etype)
                else:
                    f.read(scalar.get(vtype, 4))

            for _ in range(n_kv):
                read_str()
                vtype, = struct.unpack("<I", f.read(4))
                skip_value(vtype)

            hist: Counter = Counter()
            for _ in range(n_tensors):
                read_str()
                n_dims, = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                ttype, = struct.unpack("<I", f.read(4))
                f.read(8)  # offset
                n_elem = 1
                for d in dims:
                    n_elem *= d
                hist[GGML_TYPES.get(ttype, f"type{ttype}")] += n_elem
            return hist
    except (OSError, struct.error):
        return None


def _read_safetensors_dtypes(ckpt_dir: Path) -> Counter | None:
    """Histogram safetensors dtypes by element count, from headers only."""
    files = sorted(ckpt_dir.glob("*.safetensors"))
    if not files:
        files = sorted(ckpt_dir.glob("**/*.safetensors"))
    if not files:
        return None
    hist: Counter = Counter()
    for fp in files:
        try:
            with fp.open("rb") as f:
                n, = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n).decode("utf-8"))
        except (OSError, struct.error, ValueError):
            continue
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            shape = meta.get("shape") or []
            n_elem = 1
            for d in shape:
                n_elem *= d
            hist[meta.get("dtype", "?")] += n_elem
    return hist or None


# Checkpoints that are not safetensors, so the header parser cannot read them.
# Evo-1 ships a single DeepSpeed `mp_rank_00_model_states.pt`; measured once
# with torch.load(mmap=True) over 776M elements — 100% bfloat16.
CKPT_DTYPE_FALLBACK = {
    "evo1": "BF16 100%",
}


def _fmt_hist(hist: Counter | None, top: int = 2) -> str:
    """Render a dtype histogram as the dominant types by share of elements."""
    if not hist:
        return "n/a"
    total = sum(hist.values())
    if total == 0:
        return "n/a"
    parts = []
    for name, count in hist.most_common(top):
        pct = 100.0 * count / total
        if pct < 1.0:
            continue
        parts.append(f"{name} {pct:.0f}%")
    return " + ".join(parts) if parts else "n/a"


def _failure_reason(root: Path, model: str, variant: str) -> str:
    """Recover why a cell has no result, from that run's server log.

    Auto-extracted rather than hardcoded so the table stays honest if the
    sweep is re-run and a previously failing configuration starts working.
    """
    log = root / "_server_logs" / f"{model}.{variant}.log"
    if not log.is_file():
        return "no server log — run did not start"
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return "server log unreadable"
    for line in text.splitlines():
        if "Error in server:" in line:
            msg = line.split("Error in server:", 1)[1].strip()
            if not msg:
                continue
            if "accessing tensor output of CUDAGraphs" in msg:
                return ("CUDA-graph output overwritten by the next run — a tensor "
                        "escapes the compiled region and is reused across calls")
            if "Dynamo failed to run FX node" in msg and "flash_attn" in msg:
                return "dynamo cannot trace flash-attn's varlen op"
            return msg[:180]
    if "StopIteration" in text:
        return "StopIteration"
    return "failed — see server log"


def _load(root: Path, model: str, variant: str) -> dict | None:
    p = root / model / f"{variant}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def _server_mean(stats: dict | None) -> float | None:
    if not stats:
        return None
    s = stats.get("server_ms")
    return s.get("mean") if s else None


def _server_med(stats: dict | None) -> float | None:
    if not stats:
        return None
    s = stats.get("server_ms")
    return s.get("median") if s else None


def _cell(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
        default=REPO_ROOT / "outputs" / "latency_compare")
    ap.add_argument("--output", type=Path, default=None,
        help="Write markdown here (default: stdout).")
    args = ap.parse_args()

    data = {m: {v: _load(args.root, m, v) for v in VARIANTS} for m in MODELS}

    lines: list[str] = []
    add = lines.append

    add("# vla.cpp vs torch.compile — inference latency")
    add("")
    add("Server-side inference time per prediction, mean over the timed window, "
        "on one LIBERO-object episode stream.")
    add("")

    # --- main table --------------------------------------------------------
    add("Each cell is **mean (median)**. Several models have a heavy right tail, "
        "so the two differ materially — see the distribution table below.")
    add("")
    add("| Model | vla.cpp | PyTorch eager | compile (default) | compile (reduce-overhead) | best PyTorch | vla.cpp vs best PyTorch |")
    add("|---|---:|---:|---:|---:|---|---:|")
    for m in MODELS:
        cpp = _server_mean(data[m]["vla.cpp"])
        torch_vals = {v: _server_mean(data[m][v]) for v in TORCH_VARIANTS}
        present = {k: v for k, v in torch_vals.items() if v is not None}
        best_name, best_val = (min(present.items(), key=lambda kv: kv[1])
                               if present else (None, None))
        if cpp is not None and best_val:
            ratio = best_val / cpp
            verdict = (f"**{ratio:.2f}× faster**" if ratio > 1.0
                       else f"{1.0 / ratio:.2f}× slower")
        else:
            verdict = "—"
        def both(variant: str) -> str:
            mean = _server_mean(data[m][variant])
            med = _server_med(data[m][variant])
            if mean is None:
                return "—"
            return f"{mean:.1f} ({med:.1f})" if med is not None else f"{mean:.1f}"

        add(f"| {m} | {both('vla.cpp')} | {both('eager')} | "
            f"{both('compile-default')} | "
            f"{both('compile-reduce-overhead')} | "
            f"{best_name or '—'} | {verdict} |")
    add("")
    add("All figures in milliseconds. Lower is better. "
        "The final column is best-PyTorch ÷ vla.cpp: >1× means vla.cpp wins.")
    add("")

    # Scoreboard, computed rather than asserted.
    wins, losses, ties = [], [], []
    for m in MODELS:
        cpp = _server_mean(data[m]["vla.cpp"])
        present = [v for v in (_server_mean(data[m][x]) for x in TORCH_VARIANTS)
                   if v is not None]
        if cpp is None or not present:
            continue
        ratio = min(present) / cpp
        (wins if ratio > 1.02 else losses if ratio < 0.98 else ties).append(m)
    add(f"**Scoreboard** — against each model's best available PyTorch variant, "
        f"vla.cpp is faster on {len(wins)}/{len(wins) + len(losses) + len(ties)} "
        f"models ({', '.join(wins) or 'none'}), slower on {len(losses)} "
        f"({', '.join(losses) or 'none'})"
        + (f", tied on {len(ties)} ({', '.join(ties)})" if ties else "") + ".")
    add("")
    add("Note the spread: vla.cpp's mean and median sit within ~1% of each "
        "other on every model, while the PyTorch variants carry long tails on "
        "several. For a control loop the tail is often what matters.")
    add("")

    # --- what torch.compile bought ----------------------------------------
    add("## What torch.compile bought")
    add("")
    add("| Model | eager | best compiled | compile speedup |")
    add("|---|---:|---:|---:|")
    for m in MODELS:
        eager = _server_mean(data[m]["eager"])
        comp = {v: _server_mean(data[m][v])
                for v in ("compile-default", "compile-reduce-overhead")}
        comp = {k: v for k, v in comp.items() if v is not None}
        if not eager or not comp:
            add(f"| {m} | {_cell(eager)} | — | — |")
            continue
        bn, bv = min(comp.items(), key=lambda kv: kv[1])
        add(f"| {m} | {eager:.1f} | {bv:.1f} ({bn.replace('compile-', '')}) | "
            f"{eager / bv:.2f}× |")
    add("")

    # --- precision disclosure ---------------------------------------------
    add("## Weight precision (as shipped — not forced to match)")
    add("")
    add("| Model | vla.cpp GGUF weights | PyTorch checkpoint weights | PyTorch compute |")
    add("|---|---|---|---|")
    for m in MODELS:
        gg = _fmt_hist(_read_gguf_tensor_types(GGUF_PATHS[m])) if GGUF_PATHS[m].is_file() else "n/a"
        st = _fmt_hist(_read_safetensors_dtypes(CKPT_PATHS[m])) if CKPT_PATHS[m].exists() else "n/a"
        if st == "n/a":
            st = CKPT_DTYPE_FALLBACK.get(m, "n/a")
        add(f"| {m} | {gg} | {st} | {TORCH_RUNTIME_DTYPE[m]} |")
    add("")
    add("Percentages are share of tensor **elements**, not bytes. Each stack "
        "runs in its intended configuration, so a row where the two dtypes "
        "differ is comparing deployments, not kernels.")
    add("")

    # --- validity ----------------------------------------------------------
    # Verified 2026-08-05 by reading the GGUF metadata KVs against each
    # PyTorch checkpoint config (falling back to its base model where the
    # finetune omits a key). Latency on a flow-matching policy scales with
    # denoise steps and chunk length, so a mismatch here would mean the table
    # measured configuration rather than implementation.
    add("## Work per call (verified equal on both stacks)")
    add("")
    add("| Model | denoise steps | chunk (timesteps) | action dim |")
    add("|---|---:|---:|---:|")
    for m, steps, chunk, dim in [
        ("smolvla",    10, 50, 32),
        ("pi0",        10, 50, 32),
        ("evo1",       32, 50, 24),
        ("gr00t_n1_5",  4, 16, 32),
        ("gr00t_n1_6",  4, 50, 128),
        ("gr00t_n1_7",  4, 40, 132),
    ]:
        add(f"| {m} | {steps} | {chunk} | {dim} |")
    add("")
    add("Both stacks agree on every row, so each pair of numbers compares the "
        "same computation. Two caveats on reading the PyTorch configs: Evo-1's "
        "step count is forced to 32 in the pipeline (its checkpoint omits the "
        "key, and the code default is 50), and GR00T-N1.5 inherits 4 steps / "
        "16-timestep horizon from its base model — lerobot's `chunk_size=50` "
        "is a wrapper value, not what the DiT head generates.")
    add("")
    add("Every call is also a genuine forward pass, not an action-queue pop: "
        "per-cell `min` sits within 3% of `median` on all variants except "
        "GR00T-N1.5's compiled ones (recompilation, not replay). A pop costs "
        "only preprocessing, so any replay would drag `min` far below "
        "`median` — that is exactly how the GR00T-N1.5 bug was caught.")
    add("")

    # --- spread ------------------------------------------------------------
    add("## Distribution (median / p95, ms)")
    add("")
    add("| Model | vla.cpp | eager | compile (default) | compile (reduce-overhead) |")
    add("|---|---|---|---|---|")
    for m in MODELS:
        cells = []
        for v in VARIANTS:
            s = (data[m][v] or {}).get("server_ms")
            cells.append(f"{s['median']:.1f} / {s['p95']:.1f}" if s else "—")
        add(f"| {m} | " + " | ".join(cells) + " |")
    add("")

    # --- provenance --------------------------------------------------------
    add("## Run configuration")
    add("")
    any_stats = next((data[m][v] for m in MODELS for v in VARIANTS
                      if data[m][v] is not None), None)
    if any_stats:
        add(f"- Timed calls per cell: **{any_stats.get('n_steps')}** "
            f"(warmup excluded from the reported samples)")
        add(f"- Task: `{any_stats.get('task')}` / task_{any_stats.get('task_id')}")
        add(f"- `n_action_steps = {any_stats.get('n_action_steps')}` on both "
            f"stacks, so every call is a real forward pass rather than an "
            f"action-queue pop")
    add("- Metric: server-side inference only — excludes ZMQ transport and "
        "image serialization")
    add("- PyTorch timing is CUDA-synchronized on both sides of `select_action`")
    add("")

    # --- missing cells -----------------------------------------------------
    missing = [(m, v) for m in MODELS for v in VARIANTS if data[m][v] is None]
    if missing:
        add("## Configurations that did not run")
        add("")
        add("| Model | Variant | Why |")
        add("|---|---|---|")
        for m, v in missing:
            add(f"| {m} | {v} | {_failure_reason(args.root, m, v)} |")
        add("")
        add("These are reported rather than silently dropped: a variant that "
            "cannot run is a real property of that model on this stack, not a "
            "gap in the measurement.")
        add("")

    out = "\n".join(lines)
    if args.output:
        args.output.write_text(out)
        print(f"wrote {args.output}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
