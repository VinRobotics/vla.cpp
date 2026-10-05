// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The FoldQuant reference: what the CPU backend runs, and what the CUDA kernels
// must reproduce bit for bit. Every float operation is written in the order the
// kernels perform it (per-thread strided partial sums, a fixed reduction tree,
// IEEE sqrt/division, round-half-even), and the translation units that include
// this header are compiled with FP contraction off, so a+b*c never becomes an
// FMA on one side only.
//
// The bodies live in foldquant_ref.cpp, the one translation unit compiled with
// -ffp-contract=off: as inline functions in a header they were compiled once per
// including file, most of them with the compiler's default FMA contraction, and
// the linker kept an arbitrary copy - a reference that rounded differently from
// the kernels on every bias add and every sum of squares.

#pragma once

#include "foldquant.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vla {
namespace fqref {

// "Threads" per row in the kernel: lane l owns the 64-element chunks
// c = l, l+32, ... of the row and accumulates its partial sum of squares over
// them in element order; the 32 partials are combined by an xor butterfly.
constexpr int NT    = 32;
constexpr int CHUNK = 64;

// Kernel-order reduction of the 32 lane partials. Clobbers p.
float block_sum(float * p);

// In-place natural-order Sylvester-Hadamard butterfly on every bs-block of a
// row, normalised by inv_sqrt_bs = 1/sqrt(bs) afterwards.
void  fwht_row(float * y, int64_t K, int bs, float inv_sqrt_bs);
float inv_sqrt_block(int bs);
float qmax_for(int bits);

// One activation row: x[K] -> codes + scale. tmp and partial are scratch
// (K floats and NT floats).
void act_row(const float * x, const float * ascale, const float * gamma, const FqActSpec & s,
             float * tmp, float * partial, int8_t * codes_out, float * scale_out);

// Unpack a nibble row (low nibble = even column, two's complement) to int8.
void unpack_nibbles(const int8_t * packed, int64_t K, int8_t * out);

// y for one (token, output row): exact int32 accumulation, then the kernel's
// epilogue order ((float)acc * xs) * ws + bias.
float gemm_dot(const int8_t * w_row, const int8_t * x_row, int64_t K, float xs, float ws, float bias);

}  // namespace fqref

// GGML_OP_CUSTOM entry points. Sources are packed without holes:
//   fq_act : src[0]=x F32 [K, T...] contiguous, then gamma F32[K] if has_gamma,
//            then ascale F32[K] if has_ascale; dst I8 [row_bytes, T]
//   fq_gemm: src[0]=w I8 [K_pack, N], src[1]=xq blob, src[2]=wscale F32[N], src[3]=bias F32[N] or null
//            dst F32 [N, T]
inline void fq_act_srcs(const ggml_tensor * dst, const FqActSpec & s,
                        const ggml_tensor ** gamma, const ggml_tensor ** ascale) {
    int i = 1;
    *gamma  = s.has_gamma  ? dst->src[i++] : nullptr;
    *ascale = s.has_ascale ? dst->src[i++] : nullptr;
}

void fq_act_cpu (ggml_tensor * dst, int ith, int nth, void * userdata);
void fq_gemm_cpu(ggml_tensor * dst, int ith, int nth, void * userdata);

}  // namespace vla
