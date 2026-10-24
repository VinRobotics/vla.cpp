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

// INT4 x INT4 (and few-row INT8 x INT8) GEMMs on Xe DPAS (src/sycl/vla_sycl_s4gemm.cpp).

#pragma once

#include <cstdint>

namespace vla {

// out [M][N] int32 = A x W^T. A: M rows of K nibble-packed codes, row_bytes apart
// (a FoldQuant A4 act blob); W: [N][K/2] nibble-packed. K % 128 == 0, N % 256 == 0.
bool sycl_s4_gemm_s32(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, const int8_t * w,
                      int64_t N, int32_t * out);

// The same GEMM with launch_epilogue's F32 epilogue in its registers: out [M][N] =
// ((acc * xs_row) * ws (+ bias)) (+ res, or res + v * gate), xs_row the float at
// byte xs_off of each blob row; res [M][N] and gate [N] optional. Bit-identical
// to the s32 sums followed by that epilogue. Returns false for K >= 8192 above
// 16 rows on small Xe parts (that tile spills with the epilogue).
bool sycl_s4_gemm_f32(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, int64_t xs_off,
                      const int8_t * w, int64_t N, const float * ws, const float * bias, const float * res,
                      const float * gate, float * out);

// INT8 x INT8 with the same epilogue, for M <= 16 (the action expert's rows):
// A rows of K int8 codes (a FoldQuant A8 blob), W [N][K] int8. K % 128 == 0,
// N % 64 == 0. Returns false outside that shape (the caller uses oneDNN).
bool sycl_s8_gemm_f32(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, int64_t xs_off,
                      const int8_t * w, int64_t N, const float * ws, const float * bias, const float * res,
                      const float * gate, float * out);

// The fused gate/up site: W rows [gate (Nh) | up (Nh)], ws/bias over 2*Nh;
// out [M][Nh] BF16 bits = gelu_tanh(g) * u with g = (acc * xs_row) * ws (+ bias),
// xs_row the float at byte xs_off of each blob row. K % 128 == 0, Nh % 256 == 0
// (Nh % 64 == 0 on small Xe parts).
bool sycl_s4_gemm_glu(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, int64_t xs_off,
                      const int8_t * w, const float * ws, const float * bias, int64_t Nh, uint16_t * out_bf16);

}  // namespace vla
