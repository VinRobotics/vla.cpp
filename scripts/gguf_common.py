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

"""Checkpoint loading, dtype-preserving tensor writes and the CLI/writer
lifecycle every scripts/convert_*_to_gguf.py repeats. Lowest of the two
converter levels: nothing here knows about an architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

F32  = gguf.GGMLQuantizationType.F32
BF16 = gguf.GGMLQuantizationType.BF16

def kv_prefix(arch: str):
    return lambda name: f"{arch}.{name}"

def bf16_u16(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.uint16).cpu().numpy()

def add(writer: gguf.GGUFWriter, name: str, t: torch.Tensor) -> None:
    if t.dtype == torch.float32:
        writer.add_tensor(name, t.contiguous().cpu().numpy(), raw_dtype=F32)
    elif t.dtype == torch.bfloat16:
        writer.add_tensor(name, bf16_u16(t), raw_shape=list(t.shape), raw_dtype=BF16)
    else:
        raise NotImplementedError(f"unsupported dtype {t.dtype} for {name}")

def add_f32(writer: gguf.GGUFWriter, name: str, t: torch.Tensor) -> None:
    add(writer, name, t.float())

def add_bf16(writer: gguf.GGUFWriter, name: str, t: torch.Tensor) -> None:
    add(writer, name, t.to(torch.bfloat16))

def add_array(writer: gguf.GGUFWriter, name: str, a: np.ndarray) -> None:
    writer.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32), raw_dtype=F32)

def kv_u32(writer: gguf.GGUFWriter, kv, values: dict) -> None:
    for k, v in values.items():
        writer.add_uint32(kv(k), int(v))

def kv_f32(writer: gguf.GGUFWriter, kv, values: dict) -> None:
    for k, v in values.items():
        writer.add_float32(kv(k), float(v))

def load_safetensors(ckpt: Path, keep: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:

    idx = ckpt / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
    else:
        one = ckpt / "model.safetensors"
        if not one.exists():
            raise SystemExit(f"no model.safetensors[.index.json] under {ckpt}")
        with safe_open(str(one), framework="pt") as f:
            weight_map = {k: one.name for k in f.keys()}
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)

    out: dict[str, torch.Tensor] = {}
    for shard in sorted(by_shard):
        with safe_open(str(ckpt / shard), framework="pt") as f:
            for k in by_shard[shard]:
                if keep is None or k.startswith(keep):
                    out[k] = f.get_tensor(k)
    return out

def load_pt_module(path: Path) -> dict[str, torch.Tensor]:

    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    pfx = "module."
    return {(k[len(pfx):] if k.startswith(pfx) else k): v.contiguous() for k, v in sd.items()}

def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())

def read_text(path: Path, default: str = "{}") -> str:
    return path.read_text() if path.exists() else default

def require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing {p}")

def max_layer(keys, pfx: str) -> int:

    m = -1
    for k in keys:
        if k.startswith(pfx):
            try:
                m = max(m, int(k[len(pfx):].split(".", 1)[0]))
            except ValueError:
                pass
    return m + 1

def check_layers(got: int, want: int, what: str) -> None:
    if got != want:
        raise SystemExit(f"checkpoint has {got} {what}, expected {want}")

def arg_parser(arch: str, ckpt_help: str, description: str | None = None) -> argparse.ArgumentParser:

    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help=ckpt_help)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output GGUF path (default: <ckpt>/{arch}.gguf)"
    )
    return ap

def resolve_out(args: argparse.Namespace, ckpt: Path, arch: str) -> Path:
    return (args.out or ckpt / f"{arch}.gguf").resolve()

def open_writer(out: Path, arch: str) -> gguf.GGUFWriter:

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=arch)
    writer.add_string(f"{arch}.architecture", arch)
    return writer

def finish(writer: gguf.GGUFWriter, out: Path, note: str = "") -> int:

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB){note}")
    return 0
