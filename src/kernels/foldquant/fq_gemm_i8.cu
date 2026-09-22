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

// INT8 x INT8 -> INT32 GEMM on the integer tensor cores (wmma s8, 16x16x16),
// K streamed through a multi-stage cp.async pipeline so that while one chunk
// of the weight stripe is being multiplied the next ones are in flight: at
// the small M a VLA runs (a few hundred tokens, 41 action tokens) the GEMM is
// bound by streaming W, and the pipeline is what keeps the DRAM busy.
//
// Three tilings, picked per shape by launch_gemm (see there): the CTA count
// is what decides the achieved bandwidth on a 16-SM Orin, so narrow-N sites
// get a 32-wide N tile and large-M sites a 128-row M tile that halves how
// often each weight stripe is re-read.
//
// A = the activation blob (row m at m*row_bytes, codes then the float scale),
// B = W[N][K] with K contiguous, which is exactly a col_major K x N operand.
// Fused dequant epilogue y = ((float)acc * act_scale[m]) * wscale[n] (+ bias[n])
// in the order foldquant_ref.h uses; the accumulation is integer so any tiling
// gives the same result.
//
// M tails are zero-filled; N and K are multiples of 64 (checked at load).

#include "fq_kernels.h"

#include <mma.h>

namespace vla {
namespace fq {

namespace {

using namespace nvcuda;

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool pred) {
    const unsigned s  = (unsigned) __cvta_generic_to_shared(smem);
    const int      sz = pred ? 16 : 0;   // src-size 0 zero-fills the 16 bytes
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

template <int BM, int BN, int BK, int STAGES, int WM, int WN>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int LDS     = BK + 16;   // smem row stride in bytes: 16-byte aligned rows, fewer bank conflicts
    static constexpr int LDC     = BN + 4;
    static constexpr int STAGE_BYTES = (BM + BN) * LDS;
    static constexpr int C_BYTES = BM * LDC * 4;
    // The int32 C staging reuses the drained pipeline buffers.
    static constexpr int SMEM_BYTES = STAGES * STAGE_BYTES > C_BYTES ? STAGES * STAGE_BYTES : C_BYTES;
    static constexpr int CHUNKS  = (BM + BN) * (BK / 16);      // 16-byte copies per stage
    static constexpr int PER_THR = CHUNKS / THREADS;
    static constexpr int FM = BM / WM / 16, FN = BN / WN / 16;  // 16x16 fragments per warp
    static_assert(CHUNKS % THREADS == 0, "tile copies must split evenly over the threads");
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0, "warp tile must be a multiple of 16");
};

template <int BM, int BN, int BK, int STAGES, int WM, int WN, bool HAS_BIAS>
__global__ void __launch_bounds__(WM * WN * 32) gemm_i8_kernel(const GemmArgs g) {
    using T = Tile<BM, BN, BK, STAGES, WM, WN>;
    // Dynamic: the wide tiles need more than the 48 KB static limit (Orin
    // allows 164 KB per SM, opted into once per instantiation in launch_tile).
    extern __shared__ __align__(128) int8_t smem[];
    int * Cs = (int *) smem;

    const int     tid  = threadIdx.x;
    const int     warp = tid >> 5;
    const int     wm   = warp / WN, wn = warp % WN;
    // M tiles are the fastest grid axis: the CTAs that share a weight stripe
    // run back to back and the stripe is served from L2 after the first.
    const int64_t m0   = (int64_t) blockIdx.x * BM;
    const int64_t n0   = (int64_t) blockIdx.y * BN;
    const int     KT   = (int) (g.K / BK);

    // Per-thread copy list for one stage: global source (at k0 = 0), smem
    // offset and the zero-fill predicate for A rows past M.
    const int8_t * src[T::PER_THR];
    int            dst[T::PER_THR];
    bool           pred[T::PER_THR];
    #pragma unroll
    for (int i = 0; i < T::PER_THR; ++i) {
        const int c = tid + i * T::THREADS;
        const int r = c / (BK / 16), c16 = c % (BK / 16);
        if (r < BM) {
            const int64_t m = m0 + r;
            pred[i] = m < g.M;
            src[i]  = g.blob + (pred[i] ? m : 0) * g.row_bytes + c16 * 16;
            dst[i]  = r * T::LDS + c16 * 16;
        } else {
            const int rb = r - BM;
            pred[i] = true;
            src[i]  = g.w + (n0 + rb) * g.K + c16 * 16;
            dst[i]  = BM * T::LDS + rb * T::LDS + c16 * 16;
        }
    }
    auto load_tile = [&](int stage, int kt) {
        int8_t * base = smem + stage * T::STAGE_BYTES;
        const int64_t k0 = (int64_t) kt * BK;
        #pragma unroll
        for (int i = 0; i < T::PER_THR; ++i) cp_async16(base + dst[i], src[i] + k0, pred[i]);
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[T::FM][T::FN];
    #pragma unroll
    for (int i = 0; i < T::FM; ++i)
        #pragma unroll
        for (int j = 0; j < T::FN; ++j)
            wmma::fill_fragment(acc[i][j], 0);

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();   // stage kt%STAGES landed for every thread; stage (kt-1)%STAGES is free
        const int stage = kt % STAGES;
        const int nk = kt + STAGES - 1;
        if (nk < KT) load_tile(nk % STAGES, nk);
        cp_async_commit();

        const int8_t * as = smem + stage * T::STAGE_BYTES;
        const int8_t * bs = as + BM * T::LDS;
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af[T::FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf[T::FN];
            #pragma unroll
            for (int i = 0; i < T::FM; ++i)
                wmma::load_matrix_sync(af[i], as + (wm * T::FM * 16 + i * 16) * T::LDS + kk, T::LDS);
            #pragma unroll
            for (int j = 0; j < T::FN; ++j)
                wmma::load_matrix_sync(bf[j], bs + (wn * T::FN * 16 + j * 16) * T::LDS + kk, T::LDS);
            #pragma unroll
            for (int i = 0; i < T::FM; ++i)
                #pragma unroll
                for (int j = 0; j < T::FN; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < T::FM; ++i)
        #pragma unroll
        for (int j = 0; j < T::FN; ++j)
            wmma::store_matrix_sync(Cs + (wm * T::FM * 16 + i * 16) * T::LDC + wn * T::FN * 16 + j * 16, acc[i][j],
                                    T::LDC, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM * BN; idx += T::THREADS) {
        const int r = idx / BN, c = idx - r * BN;
        const int64_t m = m0 + r, n = n0 + c;
        if (m >= g.M) continue;
        const float xs = *(const float *) (g.blob + m * g.row_bytes + g.K);
        float v = ((float) Cs[r * T::LDC + c] * xs) * g.wscale[n];
        if (HAS_BIAS) v = v + g.bias[n];
        g.y[m * g.N + n] = v;
    }
}

template <int BM, int BN, int BK, int STAGES, int WM, int WN>
cudaError_t launch_tile(const GemmArgs & g, cudaStream_t stream) {
    using T = Tile<BM, BN, BK, STAGES, WM, WN>;
    // Opt into the dynamic shared memory once per instantiation. Not a stream
    // operation, so it is safe under CUDA-graph capture.
    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(gemm_i8_kernel<BM, BN, BK, STAGES, WM, WN, true>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_BYTES);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute(gemm_i8_kernel<BM, BN, BK, STAGES, WM, WN, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_BYTES);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }
    const dim3 grid((unsigned) ((g.M + BM - 1) / BM), (unsigned) (g.N / BN));
    if (g.bias) gemm_i8_kernel<BM, BN, BK, STAGES, WM, WN, true ><<<grid, T::THREADS, T::SMEM_BYTES, stream>>>(g);
    else        gemm_i8_kernel<BM, BN, BK, STAGES, WM, WN, false><<<grid, T::THREADS, T::SMEM_BYTES, stream>>>(g);
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_gemm(const GemmArgs & g, cudaStream_t stream) {
    if (g.M <= 0) return cudaSuccess;
    if (g.wbits != 8 || g.abits != 8) return cudaErrorNotSupported;   // W4/A4: phase 2/3
    if (g.K % 128 != 0 && g.K % 64 != 0) return cudaErrorNotSupported;
    if (g.N % 64 != 0) return cudaErrorNotSupported;
    // More than one 64-row tile: a 256-row tile streams each weight stripe
    // once for a prefix of up to 256 tokens (four times, not sixteen, at 1024).
    if (g.M > 64)
        return launch_tile<256, 64, 64, 3, 4, 2>(g, stream);
    // Narrow N (o_proj, down, ff2): a 32-wide N tile doubles the CTA count,
    // which is what keeps 16 SMs' worth of loads in flight.
    if (g.N <= 3072 && g.K % 128 == 0)
        return launch_tile<64, 32, 128, 3, 2, 2>(g, stream);
    return launch_tile<64, 64, 64, 4, 2, 2>(g, stream);
}

}  // namespace fq
}  // namespace vla
