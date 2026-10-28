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

"""Convert a FoldQuant quantized model to a vla.cpp FoldQuant GGUF.

A quantized model is a quantized policy saved as data instead of as an engine:
the base checkpoint plus the calibration result and the integer weight codes of
every quantized site, exactly as the TensorRT plugins would carry them.
FoldQuantVLA writes one in either of two formats, and this converter reads both:

* FoldQuantVLA's quantized checkpoint, `foldquant.quantized_checkpoint`
  (format "foldquant-quantized-checkpoint"): the base checkpoint with every
  quantized projection's `.weight` replaced by `.qweight` (int8, or int4
  nibble-packed) + `.weight_scale`, the SmoothQuant vectors as
  `foldquant.<module>.sq.<site>` tensors, and `foldquant_quant.json` naming the
  checkpoint keys of every site. The family converter reads the directory
  through a base view in which each missing `.weight` is qweight x scale; the
  sites are then written as codes, and the one quantized projection vla.cpp
  keeps in float, the DiT adaLN of a W4A4 arm, from that dequantized weight.
* FoldQuantVLA's earlier fake-quant state, `foldquant.fakequant` (format
  "foldquant-quant-state"): the base checkpoint files next to
  `foldquant_quant.json` and `quant_state.safetensors`, sites by name, INT8 DiT
  packs by call order.

  For both, the LLM's SmoothQuant fold into the norm gains is recomputed here
  the way FoldQuant's emitter computes it.

Nothing is recalibrated or re-rounded: the recorded codes go into the GGUF as
they are (FoldQuant's INT4 layout is vla.cpp's: row-major, low nibble = even
column), each action site ships its SmoothQuant vector as `.ascale`, and the
rest of the file is the family's own converter output.

    python scripts/convert_quantized_model_to_gguf.py --quantized-model foldquant_model --out model.gguf
    python scripts/convert_quantized_model_to_gguf.py --quantized-model foldquant_model --out model.gguf --check-onnx exports/arm/onnx

`--check-onnx` byte-compares every site and folded gain
against the plugin ONNX graphs the TensorRT engines were built from and fails
on any difference.

Supported families: GR00T N1.5 / N1.6 / N1.7 (Qwen3 LLM + DiT) and pi0.5
(PaliGemma prefix + Gemma expert). Supported schemes are the ones vla.cpp has
kernels for: `w8a8_sr` / `w4a4_srg` LLM (with `site_bits`) and `w8a8_sh` /
`w4a4_sh` / `w4a4_shg` action arms. Action arms with a dense learned rotation
(`*_sr`) are refused. See docs/QUANTIZATION.md.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_quant_writer import QuantKV, Site, quantizing_writer_factory  # noqa: E402

# FoldQuantVLA: both formats keep their manifest under this name; `format` tells them apart.
FQ_MANIFEST = "foldquant_quant.json"
FQ_TENSORS = "quant_state.safetensors"
FQ_FORMAT = "foldquant-quant-state"                 # base checkpoint + a separate quant state
FQC_FORMAT = "foldquant-quantized-checkpoint"      # the base checkpoint with .qweight/.weight_scale in place
FQC_MAX_VERSION = 1

LLM_SCHEMES = {"w8a8_sr", "w4a4_srg"}
ACTION_SCHEMES = {"w8a8_sh", "w4a4_sh", "w4a4_shg"}

LLM_SITES = ("qkv", "o", "gateup", "down")
LLM_NODES = {"qkv": "L{i}_rmsnorm_qkv", "o": "L{i}_o_proj_res", "gateup": "L{i}_rmsnorm_gateup", "down": "L{i}_down_proj_res"}
EXPERT_SITES = ("qkv", "o", "gu", "dn")


@dataclass
class Rec:
    """One recorded site: codes int8 (N, K) or uint8 (N, K/2), per-row scale (N,), SmoothQuant vector (K,) or None."""

    codes: np.ndarray
    scale: np.ndarray
    ascale: Optional[np.ndarray] = None

    @property
    def bits(self) -> int:
        return 4 if self.codes.dtype == np.uint8 else 8

    @property
    def n_out(self) -> int:
        return int(self.codes.shape[0])

    def split(self, rows: List[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
        if sum(rows) != self.n_out:
            raise SystemExit(f"recorded codes have {self.n_out} rows, the projections {rows}")
        out, r0 = [], 0
        for n in rows:
            out.append((np.ascontiguousarray(self.codes[r0:r0 + n]), np.ascontiguousarray(self.scale[r0:r0 + n])))
            r0 += n
        return out


def _as_codes(buf: bytes, n: int, k: int, bits: int) -> np.ndarray:
    if bits == 4:
        if len(buf) != n * k // 2:
            raise SystemExit(f"INT4 weight blob has {len(buf)} bytes, expected {n}x{k}/2")
        return np.frombuffer(buf, np.uint8).reshape(n, k // 2).copy()
    if len(buf) != n * k:
        raise SystemExit(f"INT8 weight blob has {len(buf)} bytes, expected {n}x{k}")
    return np.frombuffer(buf, np.int8).reshape(n, k).copy()


def _fold_gain(gain: torch.Tensor, s: np.ndarray, *, gemma: bool) -> torch.Tensor:
    """FoldQuant's SmoothQuant fold of a norm gain, as its plugin emitter writes it: (w [+ 1]) / s in fp32,
    rounded once to the gain's own dtype (byte-checked against the TensorRT graphs' `gamma`)."""
    dt = gain.dtype
    g32 = gain if dt in (torch.float32, torch.float64) else gain.float()
    if gemma:
        g32 = g32 + 1.0                         # Gemma applies (1 + w); the 1 is added in fp32, not in bf16
    return (g32 / torch.from_numpy(np.asarray(s, np.float32))).to(dt)


def _gamma_from_bytes(buf: bytes, k: int) -> torch.Tensor:
    if len(buf) == 2 * k:
        return torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).clone()
    if len(buf) == 4 * k:
        return torch.frombuffer(bytearray(buf), dtype=torch.float32).clone()
    raise SystemExit(f"gamma blob has {len(buf)} bytes for K={k}: neither bf16 nor f32")


# ---------------------------------------------------------------------------
# base checkpoint
# ---------------------------------------------------------------------------


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 (N, K/2), low nibble = even column, two's complement -> int8 (N, K)."""
    p = packed.to(torch.int16)
    both = torch.stack([p & 0xF, (p >> 4) & 0xF], dim=-1).reshape(p.shape[0], p.shape[1] * 2)
    return torch.where(both >= 8, both - 16, both).to(torch.int8)


def _is_adaln(site: str) -> bool:
    return site.endswith("_adaln")


class Checkpoint:
    """The base checkpoint's safetensors (one file or a sharded index).

    A FoldQuantVLA quantized checkpoint (FQC_FORMAT) is the base checkpoint with every quantized
    projection's `<p>.weight` replaced by `<p>.qweight` + `<p>.weight_scale`, plus `foldquant.*` site
    tensors. Opened on one, this class presents the base view the family converters expect: the
    `.qweight` / `.weight_scale` / `foldquant.*` tensors are hidden and each `<p>.weight` is rebuilt as
    qweight x weight_scale (bf16). A site's rebuilt weight is only read for its shape (the quantizing
    writer replaces it with the codes); the one quantized projection vla.cpp keeps in float, the DiT
    adaLN of a W4A4 arm, is written from it, with the scale rounded to bf16 first as the TensorRT
    AdaLNModInt4 plugin bakes it.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        index = self.root / "model.safetensors.index.json"
        single = self.root / "model.safetensors"
        self._files: Dict[str, Path] = {}
        if index.is_file():
            for key, fname in json.loads(index.read_text())["weight_map"].items():
                self._files[key] = self.root / fname
        elif single.is_file():
            with safe_open(str(single), framework="pt") as f:
                for key in f.keys():
                    self._files[key] = single
        else:
            raise SystemExit(f"{self.root}: no model.safetensors or model.safetensors.index.json; pass --ckpt <base checkpoint>")
        self._open: Dict[Path, Any] = {}
        # "<p>.weight" -> (bits, is_adaln) for a quantized checkpoint's replaced projections
        self.quantized: Dict[str, Tuple[int, bool]] = {}
        manifest = self.root / FQ_MANIFEST
        if manifest.is_file():
            m = json.loads(manifest.read_text())
            if m.get("format") == FQC_FORMAT:
                for module in (m.get("modules") or {}).values():
                    for key, info in (module.get("sites") or {}).items():
                        for group in info["params"]:
                            for ck in group:
                                self.quantized[ck] = (int(info["bits"]), _is_adaln(key))
        hidden = set()
        for ck in self.quantized:
            head = ck[: -len(".weight")]
            for suffix in (".qweight", ".weight_scale"):
                if head + suffix not in self._files:
                    raise SystemExit(f"{self.root}: quantized projection {head} has no {head}{suffix}")
                hidden.add(head + suffix)
        hidden |= {k for k in self._files if k.startswith("foldquant.")}
        self._hidden = hidden

    @property
    def is_quantized(self) -> bool:
        return bool(self.quantized)

    @property
    def weight_files(self) -> List[Path]:
        return sorted(set(self._files.values()))

    def _f(self, key: str) -> Any:
        if key not in self._files:
            raise SystemExit(f"{self.root}: no tensor {key!r} in the base checkpoint")
        p = self._files[key]
        if p not in self._open:
            self._open[p] = safe_open(str(p), framework="pt")
        return self._open[p]

    def raw(self, key: str) -> torch.Tensor:
        """A tensor as stored, hidden ones included."""
        return self._f(key).get_tensor(key)

    def keys(self) -> List[str]:
        """The base view's tensor names."""
        return sorted((set(self._files) - self._hidden) | set(self.quantized))

    def shape(self, key: str) -> Tuple[int, ...]:
        if key in self.quantized:
            head, (bits, _) = key[: -len(".weight")], self.quantized[key]
            n, cols = self._f(head + ".qweight").get_slice(head + ".qweight").get_shape()
            return (int(n), int(cols) * (2 if bits == 4 else 1))
        if key in self._hidden:
            raise SystemExit(f"{self.root}: {key!r} is part of a quantized projection, not a base tensor")
        return tuple(self._f(key).get_slice(key).get_shape())

    def tensor(self, key: str) -> torch.Tensor:
        if key in self.quantized:
            head, (bits, adaln) = key[: -len(".weight")], self.quantized[key]
            codes = self.raw(head + ".qweight")
            codes = _unpack_int4(codes) if bits == 4 else codes.to(torch.int8)
            scale = self.raw(head + ".weight_scale").float()
            if adaln:
                scale = scale.to(torch.bfloat16).float()
            return (codes.float() * scale[:, None]).to(torch.bfloat16)
        if key in self._hidden:
            raise SystemExit(f"{self.root}: {key!r} is part of a quantized projection, not a base tensor")
        return self._f(key).get_tensor(key)

    # -- what the family converters read ------------------------------------------------

    def mapping(self, keep: Optional[Tuple[str, ...]] = None) -> "_TensorMap":
        return _TensorMap(self, keep)

    def handle(self) -> "_Handle":
        return _Handle(self)


class _TensorMap(Mapping):
    """`gguf_common.load_safetensors(ckpt)` over the base view, read lazily."""

    def __init__(self, ckpt: Checkpoint, keep: Optional[Tuple[str, ...]]) -> None:
        self._ckpt = ckpt
        self._keys = [k for k in ckpt.keys() if keep is None or k.startswith(keep)]
        self._set = set(self._keys)

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._set:
            raise KeyError(key)
        return self._ckpt.tensor(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


class _Slice:
    def __init__(self, shape: Tuple[int, ...]) -> None:
        self._shape = list(shape)

    def get_shape(self) -> List[int]:
        return list(self._shape)


class _Handle:
    """`safe_open(<ckpt>/model.safetensors)` over the base view (keys, get_tensor, get_slice().get_shape())."""

    def __init__(self, ckpt: Checkpoint) -> None:
        self._ckpt = ckpt

    def keys(self) -> List[str]:
        return self._ckpt.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._ckpt.tensor(key)

    def get_slice(self, key: str) -> _Slice:
        return _Slice(self._ckpt.shape(key))

    def metadata(self) -> Dict[str, str]:
        return {"format": "pt"}

    def __enter__(self) -> "_Handle":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


@contextlib.contextmanager
def base_view(conv: Any, ckpt: Checkpoint) -> Iterator[None]:
    """Point a family converter's checkpoint reads at `ckpt`'s base view while it converts.

    The GR00T converters read through `load_safetensors` and the pi0.5 converter through `safe_open`,
    both imported into the converter module by name; reads of any other path pass through.
    """
    if not ckpt.is_quantized:
        yield
        return
    files = {p.resolve() for p in ckpt.weight_files}
    saved: Dict[str, Any] = {}
    if hasattr(conv, "load_safetensors"):
        real_load = saved["load_safetensors"] = conv.load_safetensors

        def load_safetensors(path: Any, keep: Optional[Tuple[str, ...]] = None) -> Any:
            if Path(path).resolve() == ckpt.root:
                return ckpt.mapping(keep)
            return real_load(path, keep)

        conv.load_safetensors = load_safetensors
    if hasattr(conv, "safe_open"):
        real_open = saved["safe_open"] = conv.safe_open

        def open_(path: Any, framework: str = "pt", device: str = "cpu") -> Any:
            if Path(path).resolve() in files:
                if len(files) != 1:
                    raise SystemExit(f"{ckpt.root}: a sharded quantized checkpoint read through safe_open() by the converter")
                return ckpt.handle()
            return real_open(path, framework=framework, device=device)

        conv.safe_open = open_
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(conv, name, value)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


class Source:
    """What a quantized-model producer recorded: schemes, sites, folded gains."""

    producer: str
    family: str
    root: Path
    base_checkpoint: Optional[Path]

    def scheme(self, module: str) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def llm_layers(self) -> int:
        raise NotImplementedError

    def llm_site(self, i: int, key: str) -> Rec:
        raise NotImplementedError

    def llm_gamma(self, i: int, key: str, ckpt_gain: torch.Tensor, *, gemma: bool) -> torch.Tensor:
        """The SmoothQuant-folded norm gain feeding site `key` (qkv -> attn norm, gateup -> ffn norm)."""
        raise NotImplementedError

    def llm_rot_block(self) -> int:
        raise NotImplementedError

    def dit_blocks(self) -> int:
        raise NotImplementedError

    def dit_block(self, i: int) -> Dict[str, Rec]:
        """Sites of DiT block i: {qkv, o, ffn0, ffn2} (self-attention) or {q, kv, o, ffn0, ffn2} (cross)."""
        raise NotImplementedError

    def dit_rot_block(self) -> int:
        return 64

    def expert_layers(self) -> int:
        raise NotImplementedError

    def expert_site(self, i: int, key: str) -> Rec:
        raise NotImplementedError

    def provenance(self) -> str:
        raise NotImplementedError


class FoldQuantState(Source):
    """FoldQuantVLA's quantized model: named site records + SmoothQuant vectors."""

    producer = "foldquant"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest = json.loads((self.root / FQ_MANIFEST).read_text())
        if self.manifest.get("format") != FQ_FORMAT:
            raise SystemExit(f"{self.root / FQ_MANIFEST}: format {self.manifest.get('format')!r}, expected {FQ_FORMAT!r}")
        self.family = str(self.manifest["family"])
        self.base_checkpoint = self.root          # the state is saved next to (or bundled with) the base files
        self._st = safe_open(str(self.root / FQ_TENSORS), framework="np")
        self.keys = set(self._st.keys())

    def scheme(self, module: str) -> Tuple[str, Dict[str, Any]]:
        mod = self.manifest["modules"].get(module)
        if mod is None:
            raise SystemExit(f"{self.root / FQ_MANIFEST}: no module {module!r} (has {sorted(self.manifest['modules'])})")
        return str(mod["scheme"]), dict(mod.get("config") or {})

    def tensor(self, key: str) -> np.ndarray:
        if key not in self.keys:
            raise SystemExit(f"{self.root / FQ_TENSORS}: no tensor {key!r}")
        return self._st.get_tensor(key)

    def _pack(self, module: str, group: str, index: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ck = f"{module}/g/{group}/{index}/codes"
        if ck not in self.keys:
            return None
        return self.tensor(ck), self.tensor(f"{module}/g/{group}/{index}/scale").astype(np.float32)

    def site(self, module: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """(codes, scale) of a site recorded under its own name (GPTQ, or site-scoped round-to-nearest)."""
        for group in (f"{module}.{key}", f"{module}.rtn.{key}"):
            rec = self._pack(module, group)
            if rec is not None:
                if f"{module}/g/{group}/1/codes" in self.keys:
                    raise SystemExit(f"{group}: more than one recorded pack; not a single-site record")
                return rec
        raise SystemExit(f"{self.root / FQ_TENSORS}: site {module}.{key} has no recorded codes "
                         f"(recorded by call order only, or not quantized)")

    # LLM
    def llm_layers(self) -> int:
        n = 0
        while f"llm/t/sq/L{n}_qkv" in self.keys:
            n += 1
        return n

    def llm_site(self, i: int, key: str) -> Rec:
        codes, scale = self.site("llm", f"L{i}_{key}")
        return Rec(codes, scale)

    @property
    def model_dtype(self) -> torch.dtype:
        """The dtype FoldQuant loaded the model in (its export precision; bf16 unless recorded otherwise)."""
        prec = str(self.export_metadata().get("precision", "bf16")).lower()
        return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
                "fp32": torch.float32, "float32": torch.float32}.get(prec, torch.bfloat16)

    def llm_gamma(self, i: int, key: str, ckpt_gain: torch.Tensor, *, gemma: bool) -> torch.Tensor:
        # The emitter folds the gain the model holds, i.e. the checkpoint value cast to the export dtype
        # (some GR00T checkpoints store fp32 norm gains; the exported model is bf16).
        return _fold_gain(ckpt_gain.to(self.model_dtype), self.tensor(f"llm/t/sq/L{i}_{key}"), gemma=gemma)

    def llm_rot_block(self) -> int:
        return int(self.scheme("llm")[1].get("rot_bs", 64))

    # DiT (GR00T)
    def dit_blocks(self) -> int:
        n = 0
        while f"dit/t/sq/block{n}_o" in self.keys:
            n += 1
        return n

    def _dit_is_self(self, i: int) -> bool:
        return f"dit/t/sq/block{i}_qkv" in self.keys

    # The INT8 DiT emitter packs round-to-nearest codes in this order per block (FoldQuant records
    # them as `dit.rtn/{n}` by call order); the GPTQ INT4 emitter records each site by name instead.
    _RTN_ORDER_CROSS = ("q", "o", "kv", "ffn0", "ffn2")
    _RTN_ORDER_SELF = ("qkv", "o", "ffn0", "ffn2")

    def dit_block(self, i: int) -> Dict[str, Rec]:
        def sq(key: str) -> np.ndarray:
            return self.tensor(f"dit/t/sq/{key}").astype(np.float32)

        is_self = self._dit_is_self(i)
        names = self._RTN_ORDER_SELF if is_self else self._RTN_ORDER_CROSS
        vec = {k: sq("encoder" if k == "kv" else f"block{i}_{k}") for k in names}
        if self._pack("dit", f"dit.block{i}_o") is not None:
            # GPTQ: every site by name; the cross-attention KV packs share the encoder input and
            # are recorded under `dit.encoder` in block order (the j-th record = the j-th cross block).
            out: Dict[str, Rec] = {}
            for k in names:
                if k == "kv":
                    j = sum(1 for b in range(i) if not self._dit_is_self(b))
                    kv = self._pack("dit", "dit.encoder", j)
                    if kv is None:
                        raise SystemExit(f"{self.root / FQ_TENSORS}: no dit.encoder pack #{j} for cross block {i}")
                    out[k] = Rec(kv[0], kv[1], vec[k])
                else:
                    codes, scale = self.site("dit", f"block{i}_{k}")
                    out[k] = Rec(codes, scale, vec[k])
            return out
        # Round-to-nearest: walk `dit.rtn/{n}` in emitter order, checking every shape against the vectors.
        start = sum(len(self._RTN_ORDER_SELF if self._dit_is_self(b) else self._RTN_ORDER_CROSS) for b in range(i))
        d = int(vec["o"].shape[0])
        want_n = {"q": d, "qkv": 3 * d, "kv": 2 * d, "o": int(vec["qkv" if is_self else "q"].shape[0]),
                  "ffn0": int(vec["ffn2"].shape[0]), "ffn2": int(vec["ffn0"].shape[0])}
        out = {}
        for n, k in enumerate(names):
            pack = self._pack("dit", "dit.rtn", start + n)
            if pack is None:
                raise SystemExit(f"{self.root / FQ_TENSORS}: block {i} site {k}: no dit.rtn pack #{start + n} "
                                 "(the state records neither named DiT sites nor the expected call-order packs)")
            rec = Rec(pack[0], pack[1], vec[k])
            k_in = int(pack[0].shape[1]) * (2 if rec.bits == 4 else 1)
            if rec.n_out != want_n[k] or k_in != int(vec[k].shape[0]):
                raise SystemExit(f"{self.root / FQ_TENSORS}: dit.rtn pack #{start + n} is {rec.n_out}x{k_in}, block {i} "
                                 f"site {k} expects {want_n[k]}x{int(vec[k].shape[0])}; the emitter order changed")
            out[k] = rec
        return out

    # Gemma expert (pi0.5)
    def expert_layers(self) -> int:
        n = 0
        while f"expert/t/sq/G{n}_qkv" in self.keys:
            n += 1
        return n

    def expert_site(self, i: int, key: str) -> Rec:
        codes, scale = self.site("expert", f"G{i}_{key}")
        return Rec(codes, scale, self.tensor(f"expert/t/sq/G{i}_{key}").astype(np.float32))

    def provenance(self) -> str:
        base = self.manifest.get("base") or {}
        return f"foldquant quantized model {self.root.name}; base={base.get('model_id')}; digest={str(base.get('digest', ''))[:16]}"

    def export_metadata(self) -> Dict[str, Any]:
        return dict((self.manifest.get("extra_files") or {}).get("export_metadata.json") or {})


_SITE_RE = {
    "llm": re.compile(r"^L(\d+)_(qkv|o|gateup|down)$"),
    "expert": re.compile(r"^G(\d+)_(qkv|o|gu|dn)$"),
    "dit": re.compile(r"^(?:block(\d+)_(qkv|q|kv|o|ffn0|ffn2|adaln)|encoder)$"),
}


class FoldQuantCheckpoint(FoldQuantState):
    """FoldQuantVLA's quantized checkpoint (FQC_FORMAT, `foldquant.quantized_checkpoint`).

    One directory that loads like the base checkpoint: each quantized projection's `.weight` is
    replaced by `.qweight` (int8 (N, K), or int4 nibble-packed (N, K/2), low nibble = even column,
    which is vla.cpp's layout) and `.weight_scale` (fp32 (N,)); the manifest's `sites` name, per
    recorded pack, the checkpoint keys it covers in row order; the SmoothQuant vectors are
    `foldquant.<module>.sq.<site>` tensors. The base checkpoint is the directory itself.
    """

    producer = "foldquant"

    def __init__(self, root: Path) -> None:  # noqa: D107 (FoldQuantState's reader does not apply)
        self.root = root.resolve()
        self.manifest = json.loads((self.root / FQ_MANIFEST).read_text())
        if self.manifest.get("format") != FQC_FORMAT:
            raise SystemExit(f"{self.root / FQ_MANIFEST}: format {self.manifest.get('format')!r}, expected {FQC_FORMAT!r}")
        version = int(self.manifest.get("format_version", 0))
        if version > FQC_MAX_VERSION:
            raise SystemExit(f"{self.root}: quantized checkpoint format_version {version} is newer than this converter "
                             f"reads ({FQC_MAX_VERSION}); update vla.cpp")
        self.family = str(self.manifest["family"])
        self.base_checkpoint = self.root
        self.ckpt = Checkpoint(self.root)
        # module -> site name -> (bits, [[checkpoint keys], ...])
        self.sites: Dict[str, Dict[str, Tuple[int, List[List[str]]]]] = {}
        for module, meta in (self.manifest.get("modules") or {}).items():
            table: Dict[str, Tuple[int, List[List[str]]]] = {}
            for key, info in (meta.get("sites") or {}).items():
                name = key[len(module) + 1:] if key.startswith(module + ".") else key
                name = name[len("rtn."):] if name.startswith("rtn.") else name
                pattern = _SITE_RE.get(module)
                if pattern is None or not pattern.match(name):
                    raise SystemExit(f"{self.root}: module {module!r} records site {key!r}, which vla.cpp has no mapping for")
                if name in table:
                    raise SystemExit(f"{self.root}: site {module}.{name} is recorded twice")
                table[name] = (int(info["bits"]), [list(g) for g in info["params"]])
            self.sites[module] = table
        self.consumed: set = set()          # checkpoint keys that went into a GGUF site
        self.keys = set()                   # FoldQuantState's key set is not used here

    def tensor(self, key: str) -> np.ndarray:    # "llm/t/sq/L0_qkv" -> foldquant.llm.sq.L0_qkv
        module, _, rest = key.partition("/t/")
        return self.ckpt.raw(f"foldquant.{module}.{rest.replace('/', '.')}").float().numpy()

    def _has_sq(self, module: str, site: str) -> bool:
        return f"foldquant.{module}.sq.{site}" in self.ckpt._files

    def _rec(self, module: str, site: str, group: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        entry = self.sites.get(module, {}).get(site)
        if entry is None:
            raise SystemExit(f"{self.root}: no quantized site {module}.{site}")
        bits, groups = entry
        if group >= len(groups):
            raise SystemExit(f"{self.root}: site {module}.{site} has {len(groups)} group(s), asked for #{group}")
        codes, scales = [], []
        for ck in groups[group]:
            head = ck[: -len(".weight")]
            c = self.ckpt.raw(head + ".qweight")
            want = torch.uint8 if bits == 4 else torch.int8
            if c.dtype != want:
                raise SystemExit(f"{self.root}: {head}.qweight is {c.dtype}, a {bits}-bit site stores {want}")
            codes.append(c.numpy())
            scales.append(self.ckpt.raw(head + ".weight_scale").float().numpy())
            self.consumed.add(ck)
        return np.ascontiguousarray(np.concatenate(codes)), np.ascontiguousarray(np.concatenate(scales).astype(np.float32))

    def site(self, module: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
        return self._rec(module, key)

    def scheme(self, module: str) -> Tuple[str, Dict[str, Any]]:
        mod = (self.manifest.get("modules") or {}).get(module)
        if mod is None:
            raise SystemExit(f"{self.root / FQ_MANIFEST}: no module {module!r} (has {sorted(self.manifest.get('modules') or {})})")
        return str(mod["scheme"]), dict(mod.get("config") or {})

    # LLM
    def llm_layers(self) -> int:
        return sum(1 for s in self.sites.get("llm", {}) if s.endswith("_qkv"))

    # DiT
    def dit_blocks(self) -> int:
        return sum(1 for s in self.sites.get("dit", {}) if s.startswith("block") and s.endswith("_o"))

    def _dit_is_self(self, i: int) -> bool:
        return f"block{i}_qkv" in self.sites.get("dit", {})

    def dit_block(self, i: int) -> Dict[str, Rec]:
        def rec(site: str, sq: str, group: int = 0) -> Rec:
            codes, scale = self._rec("dit", site, group)
            return Rec(codes, scale, self.tensor(f"dit/t/sq/{sq}").astype(np.float32))

        out = {k: rec(f"block{i}_{k}", f"block{i}_{k}") for k in ("o", "ffn0", "ffn2")}
        if self._dit_is_self(i):
            out["qkv"] = rec(f"block{i}_qkv", f"block{i}_qkv")
            return out
        out["q"] = rec(f"block{i}_q", f"block{i}_q")
        # Cross-attention K/V read the encoder, so they share its SmoothQuant vector. Round-to-nearest
        # records them per block (`block{i}_kv`); GPTQ under `encoder`, one group per cross block in order.
        if f"block{i}_kv" in self.sites.get("dit", {}):
            out["kv"] = rec(f"block{i}_kv", "encoder")
        else:
            j = sum(1 for b in range(i) if not self._dit_is_self(b))
            out["kv"] = rec("encoder", "encoder", j)
        return out

    # Gemma expert
    def expert_layers(self) -> int:
        return sum(1 for s in self.sites.get("expert", {}) if s.endswith("_qkv"))

    def float_projections(self) -> List[str]:
        """Quantized projections vla.cpp runs in float, from their dequantized weight: the DiT adaLN."""
        return sorted(ck for site, (_, groups) in self.sites.get("dit", {}).items() if _is_adaln(site) for g in groups for ck in g)

    def check_accounting(self) -> None:
        """Every quantized projection is a GGUF site or an allowed float projection; nothing is dropped."""
        left = set(self.ckpt.quantized) - self.consumed - set(self.float_projections())
        if left:
            raise SystemExit(f"{self.root}: {len(left)} quantized projection(s) are neither a vla.cpp site nor an allowed "
                             f"float projection: {sorted(left)[:4]}")

    def provenance(self) -> str:
        base = self.manifest.get("base") or {}
        adaln = " adaLN dequantized from INT4;" if self.float_projections() else ""
        return (f"foldquant quantized checkpoint {self.root.name} (format {FQC_FORMAT} v{self.manifest.get('format_version')});"
                f"{adaln} base={base.get('model_id')}; digest={str(base.get('digest', ''))[:16]}")


class PluginGraphs(Source):
    """Sites read from FoldQuant plugin ONNX graphs (the nodes the TensorRT engines are built from).

    `files` maps a module (`llm`, `dit`, `expert`) to its graph. Used by `--check-onnx`.
    """

    producer = "onnx"

    def __init__(self, files: Dict[str, Path], family: str = "",
                 schemes: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None) -> None:
        self.files = {m: Path(p) for m, p in files.items()}
        self.family = family
        self.root = next(iter(self.files.values())).parent if self.files else Path(".")
        self.base_checkpoint = None
        self._schemes = schemes or {}
        self._nodes: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def find(directory: Path, modules: Tuple[str, ...] = ("llm", "dit", "expert")) -> Dict[str, Path]:
        """`{module: graph}` for the `<module>*.onnx` files under `directory`."""
        found: Dict[str, Path] = {}
        for m in modules:
            hits = sorted(directory.glob(f"{m}*.onnx"))
            if len(hits) > 1:
                raise SystemExit(f"{directory}: several graphs for module {m!r}: {[h.name for h in hits]}")
            if hits:
                found[m] = hits[0]
        if not found:
            raise SystemExit(f"{directory}: no llm*.onnx / dit*.onnx / expert*.onnx plugin graphs")
        return found

    def nodes(self, module: str) -> Dict[str, Dict[str, Any]]:
        if module not in self._nodes:
            import onnx

            if module not in self.files:
                raise SystemExit(f"no plugin graph for module {module!r} (have {sorted(self.files)})")
            model = onnx.load(str(self.files[module]), load_external_data=True)
            plug: Dict[str, Dict[str, Any]] = {}
            for node in model.graph.node:
                if node.domain:
                    plug[node.name] = {a.name: self._attr(a) for a in node.attribute}
                    plug[node.name]["__op__"] = node.op_type
            self._nodes[module] = plug
        return self._nodes[module]

    @staticmethod
    def _attr(a: Any) -> Any:
        import onnx

        if a.type == onnx.AttributeProto.STRING:
            return a.s
        if a.type == onnx.AttributeProto.INT:
            return int(a.i)
        if a.type == onnx.AttributeProto.FLOAT:
            return float(a.f)
        if a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
        return None

    def node(self, module: str, name: str) -> Dict[str, Any]:
        n = self.nodes(module).get(name)
        if n is None:
            raise SystemExit(f"{self.files[module]}: no plugin node {name!r}")
        return n

    @staticmethod
    def _weight(a: Dict[str, Any], stem: str, n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        i4, i8 = f"{stem}_i4", f"{stem}_i8"
        if i4 in a:
            codes = _as_codes(a[i4], n, k, 4)
        elif i8 in a:
            codes = _as_codes(a[i8], n, k, 8)
        else:
            raise SystemExit(f"plugin node has neither {i4} nor {i8}")
        scale = np.frombuffer(a[f"{stem}_scale"], np.float32)
        if scale.shape[0] != n:
            raise SystemExit(f"{stem}_scale has {scale.shape[0]} entries, N={n}")
        return codes, scale.copy()

    @staticmethod
    def _vec(a: Dict[str, Any], key: str, k: int) -> np.ndarray:
        v = np.frombuffer(a.get(key, b""), np.float32)
        if v.shape[0] != k:
            raise SystemExit(f"{key} has {v.shape[0]} entries, K={k} (vla.cpp runs the fold-before SmoothQuant vector only)")
        return v.copy()

    def scheme(self, module: str) -> Tuple[str, Dict[str, Any]]:
        if module not in self._schemes:
            raise SystemExit(f"no scheme recorded for module {module!r}")
        return self._schemes[module]

    # LLM
    def llm_layers(self) -> int:
        return sum(1 for n in self.nodes("llm") if n.startswith("L") and n.endswith("_rmsnorm_qkv"))

    def llm_site(self, i: int, key: str) -> Rec:
        a = self.node("llm", LLM_NODES[key].format(i=i))
        codes, scale = self._weight(a, "weight", int(a["N"]), int(a["K"]))
        return Rec(codes, scale)

    def llm_gamma(self, i: int, key: str, ckpt_gain: torch.Tensor, *, gemma: bool) -> torch.Tensor:  # noqa: ARG002
        a = self.node("llm", LLM_NODES[key].format(i=i))
        return _gamma_from_bytes(a["gamma"], int(a["K"]))

    def llm_rot_block(self) -> int:
        return int(self.node("llm", LLM_NODES["qkv"].format(i=0)).get("rot_block_size", 64))

    # DiT
    def _dit_node(self, i: int, kind: str) -> Optional[Dict[str, Any]]:
        nodes = self.nodes("dit")
        for suffix in ("_int4", "_full", ""):
            n = nodes.get(f"block{i}_{kind}{suffix}")
            if n is not None:
                return n
        return None

    def dit_blocks(self) -> int:
        n = 0
        while self._dit_node(n, "ffn") is not None:
            n += 1
        return n

    def dit_block(self, i: int) -> Dict[str, Rec]:
        out: Dict[str, Rec] = {}
        ffn = self._dit_node(i, "ffn")
        if ffn is None:
            raise SystemExit(f"{self.files['dit']}: no ffn node for block {i}")
        k, inner = int(ffn["K"]), int(ffn["inner_dim"])
        c, s = self._weight(ffn, "weight_proj0", inner, k)
        out["ffn0"] = Rec(c, s, self._vec(ffn, "act_scale_pre0", k))
        c, s = self._weight(ffn, "weight_proj2", k, inner)
        out["ffn2"] = Rec(c, s, self._vec(ffn, "act_scale_pre2", inner))
        sa, ca = self._dit_node(i, "selfattn"), self._dit_node(i, "crossattn")
        if sa is not None:
            d = int(sa["inner_dim"])
            c, s = self._weight(sa, "weight_qkv", 3 * d, k)
            out["qkv"] = Rec(c, s, self._vec(sa, "act_scale_pre_in", k))
            c, s = self._weight(sa, "weight_o", k, d)
            out["o"] = Rec(c, s, self._vec(sa, "act_scale_pre_o", d))
        elif ca is not None:
            d, k_enc = int(ca["inner_dim"]), int(ca["K_enc"])
            c, s = self._weight(ca, "weight_q", d, k)
            out["q"] = Rec(c, s, self._vec(ca, "act_scale_pre_in", k))
            nodes = self.nodes("dit")
            enc = nodes.get("encoder_prequant_int4") or nodes.get("encoder_prequant")
            if enc is None:
                raise SystemExit(f"{self.files['dit']}: no encoder_prequant node")
            c, s = self._weight(ca, "weight_kv", 2 * d, k_enc)
            out["kv"] = Rec(c, s, self._vec(enc, "act_scale_pre_enc", k_enc))
            c, s = self._weight(ca, "weight_o", k, d)
            out["o"] = Rec(c, s, self._vec(ca, "act_scale_pre_o", d))
        else:
            raise SystemExit(f"{self.files['dit']}: block {i} has neither a selfattn nor a crossattn node")
        return out

    def dit_rot_block(self) -> int:
        a = self._dit_node(0, "ffn") or {}
        return int(a.get("rot_block_size", a.get("block_size", 64)))

    # Gemma expert
    def expert_layers(self) -> int:
        return sum(1 for n in self.nodes("expert") if n.startswith("G") and "_qkv_plr" in n)

    def expert_site(self, i: int, key: str) -> Rec:
        nodes = self.nodes("expert")
        a = nodes.get(f"G{i}_{key}_plr4") or nodes.get(f"G{i}_{key}_plr")
        if a is None:
            raise SystemExit(f"{self.files['expert']}: no plugin node G{i}_{key}_plr4 / _plr")
        n, k = int(a["N"]), int(a["K"])
        codes, scale = self._weight(a, "weight", n, k)
        return Rec(codes, scale, self._vec(a, "act_scale_pre", k))

    def provenance(self) -> str:
        return f"plugin graphs {self.root}"


def open_source(root: Path) -> Source:
    if root.is_file() and root.suffix == ".safetensors":
        root = root.parent
    if (root / FQ_MANIFEST).is_file():
        # Both FoldQuantVLA formats keep their manifest under the same name; `format` tells them apart.
        fmt = json.loads((root / FQ_MANIFEST).read_text()).get("format")
        if fmt == FQC_FORMAT:
            return FoldQuantCheckpoint(root)
        if fmt == FQ_FORMAT:
            return FoldQuantState(root)
        raise SystemExit(f"{root / FQ_MANIFEST}: FoldQuant format {fmt!r} is not one this converter reads "
                         f"({FQC_FORMAT!r}, {FQ_FORMAT!r})")
    raise SystemExit(f"{root}: not a FoldQuantVLA quantized model (no {FQ_MANIFEST})")


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    arch: str
    converter: str            # vla.cpp converter module with convert(ckpt, out, *, writer_factory, **kwargs)
    llm_prefix: str           # GGUF block prefix of the LLM
    action_module: str        # "dit" (GR00T) or "expert" (pi0.5)
    action_prefix: str        # GGUF block prefix of the action module
    gemma: bool               # norm gains are applied as 1 + w (pi0.5); the GGUF stores the folded gain - 1
    f32_overrides: bool       # folded gains written as F32 (else in the checkpoint's dtype)
    converter_kwargs: Dict[str, Any]


FAMILIES: Dict[str, Family] = {
    "groot_n1_5": Family("gr00t_n1_5", "convert_gr00t_n1_5_to_gguf", "vlm.blk", "dit", "aex.dit", False, False, {}),
    # N1.6's tower sees 252 px frames (see convert_gr00t_n1_6_to_gguf.py --vision-size).
    "groot_n1_6": Family("gr00t_n1_6", "convert_gr00t_n1_6_to_gguf", "vlm.blk", "dit", "aex.dit", False, False, {"vision_size": 252}),
    "groot_n1_7": Family("gr00t_n1_7", "convert_gr00t_n1_7_to_gguf", "vlm.blk", "dit", "aex.dit", False, False, {}),
    "pi05": Family("pi05", "convert_pi05_to_gguf", "vlm.blk", "expert", "aex.blk", True, True, {}),
}

LLM_PROJ = {"qkv": ("attn_q", "attn_k", "attn_v"), "o": ("attn_o",), "gateup": ("ffn_gate", "ffn_up"), "down": ("ffn_down",)}
EXPERT_PROJ = {"qkv": ("attn_q", "attn_k", "attn_v"), "o": ("attn_o",), "gu": ("ffn_gate", "ffn_up"), "dn": ("ffn_down",)}


def _check_schemes(src: Source, action_module: str) -> Tuple[str, str]:
    ls, _ = src.scheme("llm")
    as_, _ = src.scheme(action_module)
    if ls not in LLM_SCHEMES:
        raise SystemExit(f"LLM scheme {ls!r} has no vla.cpp kernel (supported: {sorted(LLM_SCHEMES)})")
    if as_ not in ACTION_SCHEMES:
        raise SystemExit(f"action scheme {as_!r} has no vla.cpp kernel (supported: {sorted(ACTION_SCHEMES)}; "
                         "`*_sr` action schemes use a dense learned rotation, vla.cpp runs the fixed butterfly)")
    return ls, as_


def _rows(n: int, kv_dim: int, key: str) -> List[int]:
    if key == "qkv":
        return [n - 2 * kv_dim, kv_dim, kv_dim]
    if key in ("gateup", "gu", "kv"):
        return [n // 2, n // 2]
    return [n]


def build_sites(src: Source, fam: Family, ckpt: Checkpoint, conv: Any) -> Tuple[Dict[str, Site], Dict[str, np.ndarray], QuantKV, Dict[str, Any]]:
    """GGUF sites, folded-gain overrides and `<arch>.quant.*` metadata for one quantized model."""
    llm_scheme, act_scheme = _check_schemes(src, fam.action_module)
    _, llm_cfg = src.scheme("llm")
    _, act_cfg = src.scheme(fam.action_module)
    sites: Dict[str, Site] = {}
    overrides: Dict[str, np.ndarray] = {}
    gains: Dict[str, torch.Tensor] = {}

    # -- LLM --------------------------------------------------------------------------
    lm_root = conv.PFX_VLM if fam.gemma else conv.LM_ROOT
    n_layers = src.llm_layers()
    if n_layers == 0:
        raise SystemExit("the quantized model records no LLM layers")
    kv_dim = ckpt.shape(f"{lm_root}.layers.0.self_attn.k_proj.weight")[0]
    widths: Dict[str, set] = {k: set() for k in LLM_SITES}
    for i in range(n_layers):
        for key in LLM_SITES:
            rec = src.llm_site(i, key)
            widths[key].add(rec.bits)
            for name, (c, s) in zip(LLM_PROJ[key], rec.split(_rows(rec.n_out, kv_dim, key))):
                sites[f"{fam.llm_prefix}.{i}.{name}.weight"] = Site(codes=c, wscale=s, ascale=None, bits=rec.bits)
        for key, norm, gguf_norm in (("qkv", "input_layernorm", "attn_norm"), ("gateup", "post_attention_layernorm", "ffn_norm")):
            gain = ckpt.tensor(f"{lm_root}.layers.{i}.{norm}.weight")
            g = src.llm_gamma(i, key, gain, gemma=fam.gemma)
            if g.numel() != gain.numel():
                raise SystemExit(f"L{i}_{key}: folded gain has {g.numel()} entries, the checkpoint's {norm} {gain.numel()}")
            gains[f"L{i}_{key}"] = g
            # vla.cpp loads Gemma norms as 1 + w, so a Gemma file gets gain - 1 (exact in fp32 from bf16 values).
            overrides[f"{fam.llm_prefix}.{i}.{gguf_norm}.weight"] = ((g.double() - 1.0).float().numpy() if fam.gemma
                                                                      else g.float().numpy())
    for key, ws in widths.items():
        if len(ws) != 1:
            raise SystemExit(f"LLM site {key!r} has widths {sorted(ws)} across layers; the GGUF metadata carries one width per site type")
    wbits = int(llm_cfg.get("bits", next(iter(widths["qkv"]))))
    site_bits = {k: next(iter(ws)) for k, ws in widths.items() if next(iter(ws)) != wbits}
    act_bits = int(llm_cfg.get("act_bits") or wbits)
    clip = llm_cfg.get("act_clip", llm_cfg.get("act_clip_ratio", 1.0))
    if isinstance(clip, dict):
        raise SystemExit("per-site learned activation clips are not representable in the GGUF metadata yet")
    rot_bs = int(llm_cfg.get("rot_bs", llm_cfg.get("rot_block_size", src.llm_rot_block())))

    # -- action module -------------------------------------------------------------------
    act_widths: set = set()
    if fam.action_module == "dit":
        n_blocks = src.dit_blocks()
        if n_blocks == 0:
            raise SystemExit("the quantized model records no DiT blocks")
        for i in range(n_blocks):
            base = f"{fam.action_prefix}.{i}"
            d = ckpt.shape(f"{conv.AHK}.model.transformer_blocks.{i}.attn1.to_k.weight")[0]
            recs = src.dit_block(i)
            if "qkv" in recs:
                r = recs["qkv"]
                for name, (c, s) in zip(("attn_q", "attn_k", "attn_v"), r.split([r.n_out - 2 * d, d, d])):
                    sites[f"{base}.{name}.weight"] = Site(codes=c, wscale=s, ascale=r.ascale, bits=r.bits)
                act_widths.add(r.bits)
            else:
                r = recs["q"]
                sites[f"{base}.attn_q.weight"] = Site(codes=r.codes, wscale=r.scale, ascale=r.ascale, bits=r.bits)
                act_widths.add(r.bits)
                r = recs["kv"]
                for name, (c, s) in zip(("attn_k", "attn_v"), r.split([d, d])):
                    sites[f"{base}.{name}.weight"] = Site(codes=c, wscale=s, ascale=r.ascale, bits=r.bits)
                act_widths.add(r.bits)
            for key, name in (("o", "attn_o"), ("ffn0", "ff0"), ("ffn2", "ff2")):
                r = recs[key]
                sites[f"{base}.{name}.weight"] = Site(codes=r.codes, wscale=r.scale, ascale=r.ascale, bits=r.bits)
                act_widths.add(r.bits)
        act_rot = int(act_cfg.get("rot_bs", act_cfg.get("rot_block_size", src.dit_rot_block())))
    else:
        n_exp = src.expert_layers()
        if n_exp == 0:
            raise SystemExit("the quantized model records no expert layers")
        exp_kv = ckpt.shape(f"{conv.PFX_AEX}.layers.0.self_attn.k_proj.weight")[0]
        for i in range(n_exp):
            for key in EXPERT_SITES:
                r = src.expert_site(i, key)
                for name, (c, s) in zip(EXPERT_PROJ[key], r.split(_rows(r.n_out, exp_kv, key))):
                    sites[f"{fam.action_prefix}.{i}.{name}.weight"] = Site(codes=c, wscale=s, ascale=r.ascale, bits=r.bits)
                act_widths.add(r.bits)
        act_rot = int(act_cfg.get("rot_bs", act_cfg.get("rot_block_size", 64)))
    if len(act_widths) != 1:
        raise SystemExit(f"action sites have mixed widths {sorted(act_widths)}; vla.cpp runs one width per action module")
    abits = act_widths.pop()
    fold_order = str(act_cfg.get("sq_fold_order", (act_cfg.get("params") or {}).get("sq_fold_order", "before")))
    if fold_order != "before":
        raise SystemExit(f"action sq_fold_order {fold_order!r}: vla.cpp divides the SmoothQuant vector in before the rotation only")

    kv: QuantKV = {
        "method": ("str", "foldquant"),
        "applied_at": ("str", src.producer),
        "provenance": ("str", src.provenance()),
        "scheme_llm": ("str", llm_scheme),
        "llm_weight_bits": ("u32", wbits),
        "llm_act_bits": ("u32", act_bits),
        "llm_rot_block_size": ("u32", rot_bs),
        "act_clip_ratio": ("f32", float(clip)),
        "site_bits": ("str", ",".join(f"{k}:{v}" for k, v in site_bits.items())),
        "scheme_action": ("str", act_scheme),
        "action_weight_bits": ("u32", abits),
        "action_act_bits": ("u32", abits),
        "action_rot_block_size": ("u32", act_rot),
        "action_fold_order": ("str", fold_order),
    }
    return sites, overrides, kv, {"gains": gains}


def check_against_graphs(src: Source, fam: Family, ckpt: Checkpoint, conv: Any, sites: Dict[str, Site],
                         gains: Dict[str, torch.Tensor], onnx_dir: Path) -> int:
    """Byte-compare the sites and folded gains against the plugin graphs in `onnx_dir`. Returns the site count."""
    graphs = PluginGraphs(PluginGraphs.find(onnx_dir), family=src.family,
                          schemes={m: src.scheme(m) for m in ("llm", fam.action_module)})
    ref_sites, _, _, ref_extra = build_sites(graphs, fam, ckpt, conv)
    bad: List[str] = []
    if set(ref_sites) != set(sites):
        bad.append(f"site sets differ ({len(sites)} vs {len(ref_sites)} in the graphs)")
    for name in sorted(set(ref_sites) & set(sites)):
        a, b = sites[name], ref_sites[name]
        if a.bits != b.bits or a.codes.tobytes() != b.codes.tobytes():
            bad.append(f"{name} codes")
        if a.wscale.astype(np.float32).tobytes() != b.wscale.astype(np.float32).tobytes():
            bad.append(f"{name} wscale")
        if (a.ascale is None) != (b.ascale is None) or (a.ascale is not None and a.ascale.tobytes() != b.ascale.tobytes()):
            bad.append(f"{name} ascale")
    for key, g in gains.items():
        want = ref_extra["gains"].get(key)
        if want is None or g.dtype != want.dtype or not torch.equal(g, want):
            bad.append(f"{key} gamma")
    if bad:
        raise SystemExit(f"--check-onnx: {len(bad)} mismatch(es) against {onnx_dir}: {bad[:8]}")
    return len(ref_sites)


# ---------------------------------------------------------------------------
# converter arguments per family
# ---------------------------------------------------------------------------


def converter_kwargs(src: Source, fam: Family, ckpt: Checkpoint, args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(fam.converter_kwargs)
    if fam.arch != "pi05":
        return kwargs
    stats = args.dataset_stats
    if stats is None:
        hits = sorted(ckpt.root.glob("assets/*/*/norm_stats.json"))
        if len(hits) != 1:
            raise SystemExit(f"{ckpt.root}: expected one assets/*/*/norm_stats.json, found {len(hits)}; pass --dataset-stats")
        stats = hits[0]
    kwargs["dataset_stats"] = Path(stats)
    cfg_path = ckpt.root / "config.json"
    cfg_json = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    if cfg_json.get("type") != "pi05":
        if args.config_json:
            kwargs["config"] = json.loads(Path(args.config_json).read_text())
        else:
            meta = src.export_metadata() if isinstance(src, FoldQuantState) else {}  # also FoldQuantCheckpoint
            norm = json.loads(Path(stats).read_text()).get("norm_stats", {})
            horizon = int(cfg_json.get("action_horizon", meta.get("action_horizon", 50)))
            dim = int(cfg_json.get("action_dim", meta.get("action_dim", 32)))
            kwargs["config"] = {
                "chunk_size": horizon, "n_action_steps": horizon,
                "num_inference_steps": int(meta.get("num_steps", 10)),
                "max_state_dim": dim, "max_action_dim": dim,
                "min_period": 4e-3, "max_period": 4.0,
                "tokenizer_max_length": int(cfg_json.get("max_token_len", 200)),
                "input_features": {"observation.state": {"shape": [len(norm["state"]["mean"])]}},
                "output_features": {"action": {"shape": [len(norm["actions"]["mean"])]}},
            }
    return kwargs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantized-model", type=Path, required=True,
                    help="FoldQuantVLA quantized model dir (foldquant_quant.json): a quantized checkpoint or an earlier fake-quant state")
    ap.add_argument("--out", type=Path, required=True, help="output GGUF path")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="[fake-quant state] base checkpoint dir (default: the state dir itself); a quantized checkpoint is its own base")
    ap.add_argument("--check-onnx", type=Path, default=None,
                    help="FoldQuantVLA plugin ONNX dir (llm_bf16.onnx, dit_bf16.onnx / expert_bf16.onnx) to byte-compare every site against")
    ap.add_argument("--dataset-stats", type=Path, default=None, help="[pi05] normalizer stats (default: the checkpoint's assets/*/*/norm_stats.json)")
    ap.add_argument("--config-json", type=Path, default=None, help="[pi05] lerobot policy fields for an OpenPI checkpoint (default: from the export metadata)")
    args = ap.parse_args()

    src = open_source(args.quantized_model)
    fam = FAMILIES.get(src.family)
    if fam is None:
        raise SystemExit(f"family {src.family!r} has no vla.cpp FoldQuant sites (supported: {sorted(FAMILIES)})")
    if isinstance(src, FoldQuantCheckpoint) and args.ckpt and Path(args.ckpt).resolve() != src.root:
        raise SystemExit("a FoldQuantVLA quantized checkpoint is its own base checkpoint (configs, assets and every "
                         "unquantized weight are in it); drop --ckpt")
    ckpt_root = args.ckpt or src.base_checkpoint
    if ckpt_root is None:
        raise SystemExit("the quantized model records no checkpoint path; pass --ckpt <base checkpoint>")
    ckpt = Checkpoint(Path(ckpt_root))
    conv = importlib.import_module(fam.converter)

    sites, overrides, kv, extra = build_sites(src, fam, ckpt, conv)
    if isinstance(src, FoldQuantCheckpoint):
        src.check_accounting()
        if src.float_projections():
            print(f"{len(src.float_projections())} DiT adaLN projection(s) are INT4 in the checkpoint; vla.cpp runs them in "
                  "bf16 from the dequantized weight (codes x bf16 scale, as the TensorRT AdaLNModInt4 plugin uses it)")
    print(f"{src.producer} {src.family}: {len(sites)} sites, {len(overrides)} folded norm gains, "
          f"llm={kv['scheme_llm'][1]} W{kv['llm_weight_bits'][1]}A{kv['llm_act_bits'][1]}"
          f"{' site_bits=' + kv['site_bits'][1] if kv['site_bits'][1] else ''} "
          f"action={kv['scheme_action'][1]} W{kv['action_weight_bits'][1]}A{kv['action_act_bits'][1]}")
    if args.check_onnx:
        n = check_against_graphs(src, fam, ckpt, conv, sites, extra["gains"], args.check_onnx)
        print(f"--check-onnx: {n} sites and every folded gain byte-identical to {args.check_onnx}")

    factory, holder = quantizing_writer_factory(sites, overrides, kv, f32_overrides=fam.f32_overrides)
    with base_view(conv, ckpt):
        out = conv.convert(ckpt.root, args.out, writer_factory=factory, **converter_kwargs(src, fam, ckpt, args))
    missing = holder[0].unconsumed() if holder else ["<no writer>"]
    if missing:
        args.out.unlink(missing_ok=True)
        raise SystemExit(f"{len(missing)} site(s)/override(s) were never written (name mismatch): {missing[:8]}")
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
