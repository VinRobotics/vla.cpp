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
    const float * x;          // [M][K] F32; rows contiguous, x_stride floats apart (0 = K)
    int64_t       x_stride = 0;
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
    const float *  res = nullptr;   // [M][N] F32 residual added in the epilogue (y = ... + res), or null
    const int8_t * pf = nullptr;    // next site's weights to prefetch into L2 (pf_bytes of them), or null
    int64_t        pf_bytes = 0;
    // Head layout of y (0 = plain [M][N]); see FqGemmSpec / fq_out_index.
    int            head_dim = 0, heads = 0;
    uint32_t       vmask = 0;
    int64_t        M, N, K;
    int64_t        row_bytes;
    int            wbits, abits;
    // Split-K only (set by launch_gemm): int32 [M][N] partial-sum workspace and
    // one arrival counter per output tile, both zero between launches.
    int *          ws       = nullptr;
    int *          counters = nullptr;
};

#ifdef __CUDACC__
// Column n of a head-laid-out output maps to y[off + m * stride]: this splits
// the column part so the epilogue can hoist it out of its row loop. 32-bit
// math: M * N < 2^31 for every site.
__device__ __forceinline__ void fq_out_column(const GemmArgs & g, int n, int & off, int & stride) {
    const int M = (int) g.M, dim = g.head_dim * g.heads;
    const int p = n / dim, r = n - p * dim, h = r / g.head_dim, d = r - h * g.head_dim;
    const int base = p * dim * M;
    if ((g.vmask >> p) & 1) { off = base + (h * g.head_dim + d) * M; stride = 1; }
    else                    { off = base + h * M * g.head_dim + d;   stride = g.head_dim; }
}
__device__ __forceinline__ int64_t fq_out_index(const GemmArgs & g, int64_t m, int64_t n) {
    if (!g.heads) return m * g.N + n;
    int off, stride;
    fq_out_column(g, (int) n, off, stride);
    return (int64_t) off + m * (int64_t) stride;
}
#endif

// Both return cudaSuccess or the launch error. Shapes the kernels do not cover
// (W4/A4 until phase 2/3) return cudaErrorNotSupported without launching.
cudaError_t launch_act (const ActArgs & a, cudaStream_t stream);
cudaError_t launch_gemm(const GemmArgs & g, cudaStream_t stream);
// mma.sync/ldmatrix kernel (fq_gemm_mma.cu); W8A8 and W4A4. variant -1 = default.
cudaError_t launch_gemm_mma(const GemmArgs & g, int variant, cudaStream_t stream);

}  // namespace fq
}  // namespace vla
