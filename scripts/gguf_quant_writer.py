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

"""A gguf.GGUFWriter that writes FoldQuant sites (docs/QUANTIZATION.md) as a converter adds them.

The model converters (`convert_<arch>_to_gguf.py::convert`) take a
`writer_factory(out, arch)`; hand them `quantizing_writer_factory(...)` and each
tensor named in `sites` is written as integer codes under its own `.weight`
name plus the `.wscale` / `.ascale` sidecars, each name in `overrides` is
replaced by the given array (the SmoothQuant-folded norm gains), and the
`<arch>.quant.*` metadata is added. Everything else is the converter's file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import gguf

SITE_ALIGNMENT = 64


@dataclass(frozen=True)
class Site:
    """One FoldQuant site as the GGUF carries it.

    codes:  int8 (N, K), or uint8 (N, K/2) nibble-packed (low nibble = even column).
    wscale: float32 (N,) per-output-row weight scale.
    ascale: float32 (K,) SmoothQuant vector divided in before the rotation, or None.
    bits:   weight width, 4 or 8.
    """

    codes: np.ndarray
    wscale: np.ndarray
    ascale: Optional[np.ndarray]
    bits: int

    @property
    def n_out(self) -> int:
        return int(self.codes.shape[0])

    @property
    def k_in(self) -> int:
        return int(self.codes.shape[1]) * (2 if self.bits == 4 else 1)


#: `<arch>.quant.<key>` -> ("str" | "u32" | "f32", value)
QuantKV = Dict[str, Tuple[str, Any]]


class QuantizingGGUFWriter(gguf.GGUFWriter):
    def __init__(self, path: Any, arch: str, *, sites: Dict[str, Site], overrides: Dict[str, np.ndarray],
                 quant_kv: QuantKV, f32_overrides: bool = False) -> None:
        super().__init__(str(path), arch=arch)
        self.add_string(f"{arch}.architecture", arch)   # the same first key open_writer() writes
        self._sites = dict(sites)
        self._overrides = dict(overrides)
        self._f32_overrides = f32_overrides
        self._consumed: set = set()
        for key, (kind, value) in quant_kv.items():
            full = f"{arch}.quant.{key}"
            if kind == "str":
                self.add_string(full, str(value))
            elif kind == "u32":
                self.add_uint32(full, int(value))
            elif kind == "f32":
                self.add_float32(full, float(value))
            else:
                raise ValueError(f"unknown quant KV kind {kind!r} for {full}")

    def add_tensor(self, name: str, tensor: Any, raw_shape: Any = None, raw_dtype: Any = None) -> None:
        site = self._sites.get(name)
        if site is not None:
            shape = list(raw_shape) if raw_shape is not None else list(tensor.shape)
            if len(shape) != 2 or int(shape[0]) != site.n_out or int(shape[1]) != site.k_in:
                raise ValueError(f"{name}: the converter writes {shape}, the site is (N={site.n_out}, K={site.k_in})")
            if site.k_in % SITE_ALIGNMENT or site.n_out % SITE_ALIGNMENT:
                raise ValueError(f"{name}: K={site.k_in} N={site.n_out} must be multiples of {SITE_ALIGNMENT}")
            base = name[: -len(".weight")]
            body = np.ascontiguousarray(site.codes).view(np.uint8)
            super().add_tensor(name, body, raw_shape=[site.n_out, int(site.codes.shape[1])],
                               raw_dtype=gguf.GGMLQuantizationType.I8)
            super().add_tensor(base + ".wscale", np.ascontiguousarray(site.wscale, dtype=np.float32),
                               raw_shape=[site.n_out], raw_dtype=gguf.GGMLQuantizationType.F32)
            if site.ascale is not None:
                if int(site.ascale.shape[0]) != site.k_in:
                    raise ValueError(f"{name}: ascale has {site.ascale.shape[0]} entries, K={site.k_in}")
                super().add_tensor(base + ".ascale", np.ascontiguousarray(site.ascale, dtype=np.float32),
                                   raw_shape=[site.k_in], raw_dtype=gguf.GGMLQuantizationType.F32)
            self._consumed.add(name)
            return
        override = self._overrides.get(name)
        if override is not None:
            self._consumed.add(name)
            ov = np.ascontiguousarray(override, dtype=np.float32)
            if raw_dtype == gguf.GGMLQuantizationType.BF16 and not self._f32_overrides:
                import torch

                data = torch.from_numpy(ov).to(torch.bfloat16).view(torch.uint16).numpy()
                super().add_tensor(name, data, raw_shape=list(ov.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
            else:
                super().add_tensor(name, ov, raw_dtype=gguf.GGMLQuantizationType.F32)
            return
        super().add_tensor(name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype)

    def unconsumed(self) -> list:
        """Sites / overrides the converter never wrote: a name mismatch, so an error."""
        return sorted((set(self._sites) | set(self._overrides)) - self._consumed)


def quantizing_writer_factory(sites: Dict[str, Site], overrides: Dict[str, np.ndarray], quant_kv: QuantKV, *,
                              f32_overrides: bool = False) -> Tuple[Callable[[Path, str], QuantizingGGUFWriter], list]:
    """(factory, holder): pass `factory` as a converter's writer_factory; `holder[0]` is the writer afterwards."""
    holder: list = []

    def factory(out: Path, arch: str) -> QuantizingGGUFWriter:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        print(f"writing {out} ({len(sites)} FoldQuant sites)")
        w = QuantizingGGUFWriter(out, arch, sites=sites, overrides=overrides, quant_kv=quant_kv,
                                 f32_overrides=f32_overrides)
        holder.append(w)
        return w

    return factory, holder
