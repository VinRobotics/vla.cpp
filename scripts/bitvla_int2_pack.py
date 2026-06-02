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

import numpy as np
import torch

N_BLOCK_SIZE  = 16
K_BLOCK_SIZE  = 8
K_PER_LOOP    = 16
WMMA_K        = 32
WMMA_N        = 16
K_PER_ITER    = K_PER_LOOP * K_BLOCK_SIZE
BYTES_PER_KITER = 512

def weight_quant_to_ternary(W: torch.Tensor):

    Wf = W.float()
    absmean = Wf.abs().double().mean().clamp_(min=1e-5).float()
    s = 1.0 / absmean
    Wq = (Wf * s).round().clamp(-1, 1)
    return Wq.to(torch.int8).cpu().numpy(), float(absmean.item())

def pack_ladder_int2(W_ternary: np.ndarray) -> np.ndarray:

    assert W_ternary.dtype == np.int8, W_ternary.dtype
    N, K = W_ternary.shape
    assert N % N_BLOCK_SIZE == 0, f"N={N} not divisible by {N_BLOCK_SIZE}"
    assert K % K_PER_ITER == 0,    f"K={K} not divisible by {K_PER_ITER}"
    assert np.all((W_ternary >= -1) & (W_ternary <= 1)), "values outside {-1,0,1}"

    W_enc = (W_ternary + 2).astype(np.uint8)
    assert W_enc.shape == (N, K)

    n_slots = N * K // 16
    slot_addr = np.arange(n_slots, dtype=np.int64)

    slots_per_block = (N_BLOCK_SIZE * K) // 16
    n_block         = slot_addr // slots_per_block
    in_block        = slot_addr  % slots_per_block

    slots_per_k0    = BYTES_PER_KITER // 4
    k_0             = in_block // slots_per_k0
    in_k0           = in_block  % slots_per_k0

    slots_per_major = 128 // 4
    major_k         = in_k0 // slots_per_major
    in_major        = in_k0  % slots_per_major

    slots_per_yhalf = 64 // 4
    y_half          = in_major // slots_per_yhalf
    in_yhalf        = in_major  % slots_per_yhalf

    slots_per_subk  = 32 // 4
    sub_k           = in_yhalf // slots_per_subk
    y_in_half       = in_yhalf  % slots_per_subk

    n_global    = n_block * N_BLOCK_SIZE + y_half * 8 + y_in_half
    k_sub_start = k_0 * K_PER_ITER + major_k * WMMA_K + sub_k * K_PER_LOOP

    k_idx = k_sub_start[:, None] + np.arange(16, dtype=np.int64)[None, :]
    n_idx = np.broadcast_to(n_global[:, None], k_idx.shape)
    enc_slots = W_enc[n_idx, k_idx]

    out = np.zeros((n_slots, 4), dtype=np.uint8)
    for byte_i in range(4):
        b = np.zeros(n_slots, dtype=np.uint8)
        for j in range(4):
            t_idx = byte_i + 4 * j
            b |= (enc_slots[:, t_idx] & 0x3) << (2 * j)
        out[:, byte_i] = b

    return out.reshape(-1)

def unpack_ladder_int2_reference(packed: np.ndarray, N: int, K: int) -> np.ndarray:

    assert packed.dtype == np.uint8 and packed.size == N * K // 4
    W_ternary = np.zeros((N, K), dtype=np.int8)
    n_slots = N * K // 16

    for s in range(n_slots):

        slots_per_block = (N_BLOCK_SIZE * K) // 16
        n_block     = s // slots_per_block
        in_block    = s  % slots_per_block
        k_0         = in_block // 128
        in_k0       = in_block  % 128
        major_k     = in_k0 // 32
        in_major    = in_k0  % 32
        y_half      = in_major // 16
        in_yhalf    = in_major  % 16
        sub_k       = in_yhalf // 8
        y_in_half   = in_yhalf  % 8
        n_global    = n_block * N_BLOCK_SIZE + y_half * 8 + y_in_half
        k_sub_start = k_0 * K_PER_ITER + major_k * WMMA_K + sub_k * K_PER_LOOP

        bytes4 = packed[s*4:(s+1)*4]

        for byte_i in range(4):
            b = int(bytes4[byte_i])
            for j in range(4):
                enc = (b >> (2*j)) & 0x3
                ternary = enc - 2
                W_ternary[n_global, k_sub_start + 4*j + byte_i] = ternary
    return W_ternary

def pack_fused_projection(weights: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:

    ternaries = []
    scales = []
    for W in weights:
        Wq, sc = weight_quant_to_ternary(W)
        ternaries.append(Wq)
        scales.append(sc)
    Wt = np.concatenate(ternaries, axis=0).astype(np.int8)
    packed = pack_ladder_int2(Wt)
    return packed, np.array(scales, dtype=np.float32)

def _self_test():
    rng = np.random.default_rng(0xB1701B17)
    N, K = 32, 256
    W = rng.integers(-1, 2, size=(N, K), dtype=np.int8)
    assert ((W >= -1) & (W <= 1)).all()

    packed = pack_ladder_int2(W)
    assert packed.shape == (N * K // 4,)
    assert packed.dtype == np.uint8

    W_rt = unpack_ladder_int2_reference(packed, N, K)
    diff = (W != W_rt).sum()
    print(f"self-test: N={N} K={K}, mismatches = {diff} / {N*K}")
    assert diff == 0, "packing roundtrip failed"

    enc_max = (packed & 0x3).max()
    enc_min = (packed & 0x3).min()
    print(f"per-2bit encoded range: [{enc_min}, {enc_max}] (expect [0,3]; 0 only if many -1 vals)")
    print("OK")

if __name__ == "__main__":
    _self_test()
