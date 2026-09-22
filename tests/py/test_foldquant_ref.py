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

"""Pins scripts/foldquant_ref.py to the C++ reference (src/foldquant_ref.h):
the golden checksum of the codes for the fixed LCG input is the same constant
tests/test_foldquant_cpu_op.cpp asserts, so the two references cannot drift
apart. Also pins the nibble order, the blob layout and rotation_block_for."""

import pathlib
import re
import sys

import pytest

np = pytest.importorskip("numpy")

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import foldquant_ref as fq  # noqa: E402

K, T = 128, 5


def _golden_from_cpp() -> int:
    src = (pathlib.Path(__file__).resolve().parents[1] / "test_foldquant_cpu_op.cpp").read_text()
    m = re.search(r"FQ_GOLDEN_FNV\s*=\s*0x([0-9a-fA-F]+)", src)
    assert m, "golden constant missing from test_foldquant_cpu_op.cpp"
    return int(m.group(1), 16)


def test_golden_checksum_matches_cpp():
    x = fq.lcg(0x5EED1234, K * T).reshape(T, K) * np.float32(4.0)
    codes, _ = fq.act_quant(x, bits=8, rot_block=64)
    assert fq.fnv1a(codes.tobytes()) == _golden_from_cpp()


def test_fwht_matches_dense_hadamard():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 256)).astype(np.float32)
    h = fq.hadamard(64)
    dense = (x.reshape(3, 4, 64).astype(np.float64) @ h.T).reshape(3, 256)
    assert np.allclose(fq.fwht_rows(x, 64), dense, atol=1e-5)


def test_fold_then_rotate_is_identity_up_to_quant():
    # W' x' with x' = H x and W' = W H^T equals W x (H orthogonal).
    rng = np.random.default_rng(1)
    w = rng.standard_normal((16, 128)).astype(np.float32)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    wf = fq.fold_weight(w, 64)
    xr = fq.fwht_rows(x, 64)
    assert np.allclose(xr @ wf.T, x @ w.T, atol=1e-4)


def test_nibble_round_trip_and_order():
    codes = np.array([[-7, 7, 0, -1, 3, -8 + 1, 1, 2]], dtype=np.int8)
    packed = fq.pack_nibbles(codes)
    assert packed[0, 0] == ((-7 & 0xF) | ((7 & 0xF) << 4))   # low nibble = even column
    assert np.array_equal(fq.unpack_nibbles(packed), codes)


def test_blob_layout():
    codes = np.zeros((2, 64), dtype=np.int8)
    scales = np.array([1.5, -2.0], dtype=np.float32)
    blob = fq.act_blob(codes, scales, bits=8)
    assert blob.shape == (2, 64 + fq.ACT_TAIL)
    assert blob[:, 64:68].copy().view(np.float32).reshape(-1).tolist() == [1.5, -2.0]
    blob4 = fq.act_blob(codes, scales, bits=4)
    assert blob4.shape == (2, 32 + fq.ACT_TAIL)


def test_rotation_block_for():
    assert fq.rotation_block_for(2048, 64) == 64
    assert fq.rotation_block_for(480, 64) == 32
    assert fq.rotation_block_for(100, 64) == 4
    assert fq.rotation_block_for(7, 64) == 1


def test_gemm_ref_epilogue():
    wc = np.array([[1, 2], [3, 4]], dtype=np.int8)
    xc = np.array([[5, 6]], dtype=np.int8)
    y = fq.gemm_ref(wc, np.array([0.5, 0.25], np.float32), xc, np.array([2.0], np.float32), np.array([1.0, 0.0], np.float32))
    assert y.tolist() == [[(5 + 12) * 2 * 0.5 + 1.0, (15 + 24) * 2 * 0.25]]
