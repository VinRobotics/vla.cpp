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

// FoldQuant CUDA kernels: the activation prologue ([RMSNorm] -> [/s] -> block
// FWHT -> [/s] -> per-token INT8) and the INT8 x INT8 -> INT32 GEMM with the
// dequant epilogue. Pure CUDA, no ggml: the glue that decodes graph nodes lives
// in src/cuda/vla_cuda_foldquant.cu.
//
// Numerics are pinned to src/foldquant_ref.h: this archive is compiled with
// -fmad=false and without --use_fast_math so every float op rounds exactly
// where the reference does. All launches run on the caller's stream, allocate
// nothing and never synchronize (CUDA-graph capture safe).

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vla {
namespace fq {

struct ActArgs {
    const float * x;          // [M][K] F32, contiguous
    const float * ascale;     // [K] or null
    const float * gamma;      // [K] or null (fused RMSNorm)
    int8_t *      blob;       // [M][row_bytes]: codes then the float scale at K_pack
    int64_t       row_bytes;
    int64_t       M, K;
    int           abits;      // 8 | 4
    int           rot_block;  // 1 = none
    bool          fold_before;
    float         clip, eps;
    float         inv_sqrt_bs;
};

struct GemmArgs {
    const int8_t * w;         // [N][K_pack] INT8 codes (nibbles when wbits == 4)
    const int8_t * blob;      // activation blob from the prologue
    const float *  wscale;    // [N]
    const float *  bias;      // [N] or null
    float *        y;         // [M][N] F32
    int64_t        M, N, K;
    int64_t        row_bytes;
    int            wbits, abits;
};

// Both return cudaSuccess or the launch error. Shapes the kernels do not cover
// (W4/A4 until phase 2/3) return cudaErrorNotSupported without launching.
cudaError_t launch_act (const ActArgs & a, cudaStream_t stream);
cudaError_t launch_gemm(const GemmArgs & g, cudaStream_t stream);

}  // namespace fq
}  // namespace vla
