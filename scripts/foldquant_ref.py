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

"""numpy reference for the FoldQuant contract (docs/QUANTIZATION.md).

Mirrors src/foldquant_ref.h operation for operation in float32 so that, for the
sites without a fused RMSNorm, the INT8 codes it produces are bit-identical to
the runtime's (the norm's sum of squares is the one reduction whose order numpy
cannot reproduce; there the runtime may differ by one code at a rounding
boundary). tests/py/test_foldquant_ref.py pins this against the C++ test's
golden checksum. numpy only, so it also serves scripts/foldquant_fake_export.py.
"""

from __future__ import annotations

import numpy as np

QMAX = {8: 127.0, 4: 7.0}
ACT_TAIL = 16          # bytes after the codes in an activation-blob row
SCALE_FLOOR = 1e-12


def lcg(seed: int, n: int) -> np.ndarray:
    """Same generator as tests/test_foldquant_cpu_op.cpp: n floats in [-1, 1]."""
    # Every operation in float32, like the C++: a float64 division rounded to
    # float32 afterwards differs by an ulp often enough to flip a code.
    out = np.empty(n, dtype=np.float32)
    s = seed & 0xFFFFFFFF
    two, one, denom = np.float32(2.0), np.float32(1.0), np.float32(65535.0)
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (np.float32((s >> 8) & 0xFFFF) / denom) * two - one
    return out


def rotation_block_for(k_in: int, nominal: int) -> int:
    """Largest power of two <= nominal dividing k_in; 1 means no rotation."""
    bs = nominal
    while bs > 1 and k_in % bs:
        bs //= 2
    return bs if bs >= 2 else 1


def inv_sqrt_block(bs: int) -> np.float32:
    return np.float32(1.0 / np.sqrt(float(bs)))


def hadamard(bs: int) -> np.ndarray:
    """Normalised natural-order Sylvester Hadamard, float64."""
    h = np.array([[1.0]])
    while h.shape[0] < bs:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(bs)


def fwht_rows(x: np.ndarray, bs: int) -> np.ndarray:
    """In-place-order butterfly on every bs-block of the last axis, float32,
    stages h = 1, 2, 4, ... with (a+b, a-b), then * 1/sqrt(bs)."""
    y = np.ascontiguousarray(x, dtype=np.float32).copy()
    if bs <= 1:
        return y
    lead = y.shape[:-1]
    k = y.shape[-1]
    v = y.reshape(*lead, k // bs, bs)
    h = 1
    while h < bs:
        v = v.reshape(*lead, k // bs, bs // (2 * h), 2, h)
        a = v[..., 0, :].copy()
        b = v[..., 1, :].copy()
        v[..., 0, :] = a + b
        v[..., 1, :] = a - b
        v = v.reshape(*lead, k // bs, bs)
        h *= 2
    v = v * inv_sqrt_block(bs)
    return v.reshape(*lead, k).astype(np.float32)


def act_quant(x: np.ndarray, *, bits: int = 8, rot_block: int = 64, ascale: np.ndarray | None = None,
              fold_before: bool = False, clip: float = 1.0, gamma: np.ndarray | None = None,
              eps: float = 1e-6):
    """x [T, K] float32 -> (codes int8 [T, K], scales float32 [T]).

    Runtime order: [RMSNorm(gamma)] -> [/ascale if before] -> FWHT -> [/ascale
    if after] -> per-token symmetric quant, scale = max(clip*amax/qmax, 1e-12),
    q = rint(y * (1/scale)) half to even, clamp +-qmax."""
    y = np.ascontiguousarray(x, dtype=np.float32)
    if gamma is not None:
        rstd = np.float32(1.0) / np.sqrt((y * y).sum(axis=-1, keepdims=True, dtype=np.float32) / np.float32(y.shape[-1]) + np.float32(eps))
        y = (y * rstd) * gamma.astype(np.float32)
    if ascale is not None and fold_before:
        y = y / ascale.astype(np.float32)
    y = fwht_rows(y, rot_block)
    if ascale is not None and not fold_before:
        y = y / ascale.astype(np.float32)
    amax = np.abs(y).max(axis=-1, keepdims=True).astype(np.float32)
    qmax = np.float32(QMAX[bits])
    scale = (np.float32(clip) * amax) / qmax
    scale = np.maximum(scale, np.float32(SCALE_FLOOR)).astype(np.float32)
    inv = (np.float32(1.0) / scale).astype(np.float32)   # reciprocal, like the TensorRT kernels
    q = np.clip(np.rint(y * inv), -qmax, qmax).astype(np.int8)
    return q, scale.reshape(-1)


def weight_quant_per_row(w: np.ndarray, bits: int = 8):
    """w [N, K] float32 -> (codes int8 [N, K], wscale float32 [N]); symmetric
    per output row, scale = amax/qmax (floored), clamp +-qmax."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    qmax = np.float32(QMAX[bits])
    amax = np.abs(w).max(axis=1, keepdims=True).astype(np.float32)
    scale = np.maximum(amax / qmax, np.float32(SCALE_FLOOR)).astype(np.float32)
    codes = np.clip(np.rint(w / scale), -qmax, qmax).astype(np.int8)
    return codes, scale.reshape(-1)


def fold_weight(w: np.ndarray, rot_block: int) -> np.ndarray:
    """W' = W . H_block^T along K (H is symmetric): the offline rotation whose
    inverse the runtime butterfly applies to the activation."""
    if rot_block <= 1:
        return np.ascontiguousarray(w, dtype=np.float32)
    n, k = w.shape
    h = hadamard(rot_block)
    v = np.asarray(w, dtype=np.float64).reshape(n, k // rot_block, rot_block) @ h.T
    return v.reshape(n, k).astype(np.float32)


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """[.., K] int8 in [-7, 7] -> [.., K/2] uint8, low nibble = even column."""
    c = np.asarray(codes, dtype=np.int8)
    lo = (c[..., 0::2].astype(np.uint8)) & 0xF
    hi = (c[..., 1::2].astype(np.uint8)) & 0xF
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    lo = ((p << 4).astype(np.int8) >> 4).astype(np.int8)
    hi = (p.astype(np.int8) >> 4).astype(np.int8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.int8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def act_blob(codes: np.ndarray, scales: np.ndarray, bits: int = 8) -> np.ndarray:
    """The runtime's activation blob: per row, codes (nibble-packed for 4-bit)
    then the float32 scale, in a row of K_pack + ACT_TAIL bytes."""
    t, k = codes.shape
    body = pack_nibbles(codes) if bits == 4 else codes.astype(np.int8).view(np.uint8)
    kp = body.shape[1]
    blob = np.zeros((t, kp + ACT_TAIL), dtype=np.uint8)
    blob[:, :kp] = body
    blob[:, kp:kp + 4] = np.asarray(scales, dtype=np.float32).reshape(t, 1).view(np.uint8)
    return blob


def gemm_ref(w_codes: np.ndarray, wscale: np.ndarray, x_codes: np.ndarray, xscale: np.ndarray,
             bias: np.ndarray | None = None) -> np.ndarray:
    """y [T, N] = ((int32 acc) * xscale[t]) * wscale[n] (+ bias[n]) in float32."""
    acc = x_codes.astype(np.int32) @ w_codes.astype(np.int32).T
    y = (acc.astype(np.float32) * xscale.astype(np.float32).reshape(-1, 1)) * wscale.astype(np.float32).reshape(1, -1)
    if bias is not None:
        y = y + bias.astype(np.float32).reshape(1, -1)
    return y


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
