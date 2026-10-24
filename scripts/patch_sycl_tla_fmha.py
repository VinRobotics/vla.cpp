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

"""Fix sycl-tla's flash-attention forward for key counts that are not a whole tile.

src/sycl/vla_sycl_fmha.cpp runs sycl-tla's FMHA kernel (applications/
flash_attention_v2) on pi0.5's attention, whose key count (prefix tokens,
prefix + action tokens) is rarely a multiple of the 64-key tile. Two bugs make
the kernel wrong there (tests/fmha_sycl_check shows both):

  1. kernel/xe_fmha_fwd_kernel.hpp: without a causal mask or a KV cache, the
     number of key tiles is seq_len / tile, rounded down, so the keys of the last
     partial tile are never visited.
  2. collective/xe_fmha_fwd_mainloop.hpp: the remainder mask takes each score's
     key index from the K *copy* partition (tKgK), whose per-thread offsets are
     not the score layout's, so some keys past the end stay in the softmax. It
     now takes them from the MMA's C partition, as the causal mask does.

Usage: scripts/patch_sycl_tla_fmha.py <sycl-tla dir>   (idempotent)
"""

import pathlib
import sys

MARKER = "vla.cpp: sycl-tla FMHA remainder fix"

KBLOCKS = (
    "applications/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp",
    """        k_blocks = static_cast<int>(static_cast<unsigned>(seq_len) / static_cast<unsigned>(get<1>(TileShapeQK{})));""",
    """        // vla.cpp: sycl-tla FMHA remainder fix -- was rounded down, dropping the last partial key tile.
        k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));""",
)

MASK = (
    "applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp",
    """          FragSRow k_rem_mask;
          // Use logical tile index to compute k_val, so the mask is correct even
          // when PagedKV is enabled (k_idx is physical in that case).
          int logical_k_tile = is_cache ? K : (K - kblocks_cache);
          int k_val = get<0>(tKgK_cur(0,0,0,logical_k_tile,0));
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_bound) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }""",
    """          // vla.cpp: sycl-tla FMHA remainder fix -- each score's key index from the
          // MMA's C partition (as the causal mask does), not the K copy partition.
          int logical_k_tile = is_cache ? K : (K - kblocks_cache);
          Tensor cMask = make_identity_tensor(make_shape(get<0>(TileShapeQK{}), (logical_k_tile + 1) * get<1>(TileShapeQK{})));
          Tensor gMask = local_tile(cMask, take<0,2>(TileShapeQK{}), make_coord(0, logical_k_tile));
          auto cMask_thread = thr_mma_qk.partition_C(gMask);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            if (get<1>(cMask_thread(i)) >= seq_bound) tSrS(i) = ElementS(-INFINITY);
          }""",
)


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    for rel, old, new in (KBLOCKS, MASK):
        path = root / rel
        if not path.exists():
            raise SystemExit(f"not a sycl-tla checkout: {path} is missing")
        text = path.read_text()
        if MARKER in text:
            continue  # idempotent: re-configure over an already-patched tree
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"{path}: anchor found {n} times, expected 1. sycl-tla probably moved; "
                f"re-check this anchor against the new revision.\n---\n{old[:400]}\n---"
            )
        path.write_text(text.replace(old, new))


if __name__ == "__main__":
    main()
