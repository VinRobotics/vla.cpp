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
// Four tilings, picked per shape by launch_gemm (see there) from a measured
// sweep on a 16-SM Orin: narrow-N sites at M <= 64 get a 32-wide N tile (CTA
// count), the LLM prefill a 192-row M tile that holds ~160 tokens in one tile,
// and the widest / longest sites a 128-row BK=128 tile. VLA_FQ_TILE=<n>
// forces one of the twelve candidates for re-tuning on another GPU.
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

#include <cstdlib>
#include <string>

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

// SPLITK > 1: gridDim.z CTAs each accumulate one K slice of the tile into
// registers, add their int32 partials into a zeroed workspace with integer
// atomics (exact and associative, so the sum does not depend on arrival
// order) and the last CTA to arrive - counted per tile - applies the float
// epilogue and re-zeroes the workspace for the next launch. Streams each
// weight byte once, like SPLITK == 1, but with gridDim.z times the CTAs in
// flight, which is what a DRAM-bound M <= 64 site needs.
template <int BM, int BN, int BK, int STAGES, int WM, int WN, bool HAS_BIAS, int SPLITK>
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
    const int     KT_all = (int) (g.K / BK);
    const int     KT     = SPLITK > 1 ? KT_all / SPLITK : KT_all;   // this CTA's k-steps
    const int     kt_base = SPLITK > 1 ? (int) blockIdx.z * KT : 0;

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
        const int64_t k0 = (int64_t) (kt_base + kt) * BK;
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

    if (SPLITK > 1) {
        // Partial tile -> workspace (int32 [M][N]); the last CTA of this tile finishes.
        __shared__ int s_last;
        for (int idx = tid; idx < BM * BN; idx += T::THREADS) {
            const int r = idx / BN, c = idx - r * BN;
            const int64_t m = m0 + r, n = n0 + c;
            if (m >= g.M) continue;
            atomicAdd(g.ws + m * g.N + n, Cs[r * T::LDC + c]);
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int tile = (int) (blockIdx.y * gridDim.x + blockIdx.x);
            s_last = (atomicAdd(g.counters + tile, 1) == SPLITK - 1);
        }
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        for (int idx = tid; idx < BM * BN; idx += T::THREADS) {
            const int r = idx / BN, c = idx - r * BN;
            const int64_t m = m0 + r, n = n0 + c;
            if (m >= g.M) continue;
            int * wp = g.ws + m * g.N + n;
            const int acc = __ldcg(wp);
            *wp = 0;                                            // ready for the next launch
            const float xs = *(const float *) (g.blob + m * g.row_bytes + g.K);
            float v = ((float) acc * xs) * g.wscale[n];
            if (HAS_BIAS) v = v + g.bias[n];
            if (g.res) v = v + g.res[m * g.N + n];
            g.y[fq_out_index(g, m, n)] = v;
        }
        if (tid == 0) g.counters[(int) (blockIdx.y * gridDim.x + blockIdx.x)] = 0;
        return;
    }

    for (int idx = tid; idx < BM * BN; idx += T::THREADS) {
        const int r = idx / BN, c = idx - r * BN;
        const int64_t m = m0 + r, n = n0 + c;
        if (m >= g.M) continue;
        const float xs = *(const float *) (g.blob + m * g.row_bytes + g.K);
        float v = ((float) Cs[r * T::LDC + c] * xs) * g.wscale[n];
        if (HAS_BIAS) v = v + g.bias[n];
        if (g.res) v = v + g.res[m * g.N + n];
        g.y[fq_out_index(g, m, n)] = v;
    }
}

// Split-K workspace: one int32 [M][N] accumulator plus one counter per tile,
// zero between launches (the finishing CTA re-zeroes what it used). GEMMs on a
// stream run in order, so one buffer serves them all; it grows on demand,
// never inside CUDA-graph capture (the caller then gets cudaErrorNotSupported
// and falls back to the unsplit tile).
struct SplitWs {
    int *  ws       = nullptr;
    int *  counters = nullptr;
    size_t ws_elems = 0, n_counters = 0;
};
static SplitWs & split_ws() { static SplitWs w; return w; }

static cudaError_t ensure_split_ws(size_t elems, size_t counters, cudaStream_t stream) {
    SplitWs & w = split_ws();
    if (w.ws_elems >= elems && w.n_counters >= counters) return cudaSuccess;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone)
        return cudaErrorNotSupported;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return cudaErrorUnknown;   // old buffer may be in use
    if (w.ws)       cudaFree(w.ws);
    if (w.counters) cudaFree(w.counters);
    w.ws = nullptr; w.counters = nullptr; w.ws_elems = 0; w.n_counters = 0;
    const size_t e = elems > w.ws_elems ? elems : w.ws_elems, c = counters > 4096 ? counters : 4096;
    if (cudaMalloc(&w.ws, e * sizeof(int)) != cudaSuccess) return cudaErrorMemoryAllocation;
    if (cudaMalloc(&w.counters, c * sizeof(int)) != cudaSuccess) return cudaErrorMemoryAllocation;
    // Zeroed on the caller's stream: ggml's streams are non-blocking, so a
    // legacy-stream memset would not be ordered before the first split launch.
    if (cudaMemsetAsync(w.ws, 0, e * sizeof(int), stream) != cudaSuccess ||
        cudaMemsetAsync(w.counters, 0, c * sizeof(int), stream) != cudaSuccess)
        return cudaErrorUnknown;
    w.ws_elems = e; w.n_counters = c;
    return cudaSuccess;
}

template <int BM, int BN, int BK, int STAGES, int WM, int WN, int SPLITK = 1>
cudaError_t launch_tile(const GemmArgs & g_in, cudaStream_t stream) {
    using T = Tile<BM, BN, BK, STAGES, WM, WN>;
    // Opt into the dynamic shared memory once per instantiation. Not a stream
    // operation, so it is safe under CUDA-graph capture.
    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(gemm_i8_kernel<BM, BN, BK, STAGES, WM, WN, true, SPLITK>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_BYTES);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute(gemm_i8_kernel<BM, BN, BK, STAGES, WM, WN, false, SPLITK>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_BYTES);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }
    GemmArgs g = g_in;
    const dim3 grid((unsigned) ((g.M + BM - 1) / BM), (unsigned) (g.N / BN), (unsigned) SPLITK);
    if (SPLITK > 1) {
        if ((g.K / BK) % SPLITK != 0) return cudaErrorNotSupported;
        const cudaError_t e = ensure_split_ws((size_t) g.M * (size_t) g.N, (size_t) grid.x * grid.y, stream);
        if (e != cudaSuccess) return e;
        g.ws = split_ws().ws; g.counters = split_ws().counters;
    }
    if (g.bias) gemm_i8_kernel<BM, BN, BK, STAGES, WM, WN, true,  SPLITK><<<grid, T::THREADS, T::SMEM_BYTES, stream>>>(g);
    else        gemm_i8_kernel<BM, BN, BK, STAGES, WM, WN, false, SPLITK><<<grid, T::THREADS, T::SMEM_BYTES, stream>>>(g);
    return cudaGetLastError();
}

}  // namespace

// VLA_FQ_TILE=<n>: force one tiling for every shape (tuning aid; see the
// table in foldquant_gemm_check). Unset or -1 = the per-shape dispatch below.
static int forced_tile() {
    static const int v = [] {
        const char * e = std::getenv("VLA_FQ_TILE");
        return e && *e ? std::atoi(e) : -1;
    }();
    return v;
}

static cudaError_t launch_forced(int t, const GemmArgs & g, cudaStream_t stream) {
    switch (t) {
        case 0:  return launch_tile<256, 64,  64,  3, 4, 2>(g, stream);
        case 1:  return launch_tile<64,  32,  128, 3, 2, 2>(g, stream);
        case 2:  return launch_tile<64,  64,  64,  4, 2, 2>(g, stream);
        case 3:  return launch_tile<128, 64,  64,  4, 4, 2>(g, stream);
        case 4:  return launch_tile<128, 64,  64,  3, 2, 2>(g, stream);
        case 5:  return launch_tile<128, 128, 64,  3, 4, 2>(g, stream);
        case 6:  return launch_tile<192, 64,  64,  3, 4, 2>(g, stream);
        case 7:  return launch_tile<64,  128, 64,  4, 2, 4>(g, stream);
        case 8:  return launch_tile<128, 64,  128, 2, 4, 2>(g, stream);
        case 9:  return launch_tile<64,  64,  128, 3, 2, 2>(g, stream);
        case 10: return launch_tile<32,  64,  128, 4, 1, 2>(g, stream);
        case 11: return launch_tile<64,  64,  64,  6, 2, 2>(g, stream);
        case 12: return launch_tile<64,  32,  128, 6, 2, 2>(g, stream);
        case 13: return launch_tile<64,  32,  256, 4, 2, 2>(g, stream);
        case 14: return launch_tile<64,  64,  128, 6, 2, 2>(g, stream);
        case 15: return launch_tile<64,  32,  128, 8, 2, 2>(g, stream);
        case 16: return launch_tile<64,  32,  256, 6, 2, 2>(g, stream);
        case 20: return launch_tile<64,  32,  128, 3, 2, 2, 2>(g, stream);
        case 21: return launch_tile<64,  32,  128, 3, 2, 2, 3>(g, stream);
        case 22: return launch_tile<64,  32,  128, 3, 2, 2, 4>(g, stream);
        case 23: return launch_tile<64,  64,  64,  4, 2, 2, 2>(g, stream);
        case 24: return launch_tile<64,  64,  64,  4, 2, 2, 4>(g, stream);
        case 25: return launch_tile<64,  32,  128, 3, 2, 2, 6>(g, stream);
        default: return cudaErrorInvalidValue;
    }
}

static int mma_variant() {
    static const int v = [] {
        const char * e = std::getenv("VLA_FQ_MMA");
        return e && *e ? std::atoi(e) : -1;
    }();
    return v;
}
static bool use_wmma() {
    static const bool w = [] { const char * e = std::getenv("VLA_FQ_GEMM"); return e && std::string(e) == "wmma"; }();
    return w;
}

cudaError_t launch_gemm(const GemmArgs & g, cudaStream_t stream) {
    if (g.M <= 0) return cudaSuccess;
    // The mma.sync kernel serves W8A8 and W4A4 (fq_gemm_mma.cu); the wmma
    // tiles below stay as the VLA_FQ_GEMM=wmma fallback and for the tile sweep.
    if (!use_wmma() && forced_tile() < 0) {
        const cudaError_t e = launch_gemm_mma(g, mma_variant(), stream);
        if (e != cudaErrorNotSupported) return e;
    }
    if (g.wbits != 8 || g.abits != 8) return cudaErrorNotSupported;   // W4A8 / mixed: CPU reference
    if (g.K % 128 != 0 && g.K % 64 != 0) return cudaErrorNotSupported;
    if (g.N % 64 != 0) return cudaErrorNotSupported;
    // VLA_FQ_TILE_SMALL_M=1 restricts the override to the M <= 64 shapes (the
    // DiT sites), so a model run times one DiT tiling with the LLM dispatch intact.
    if (forced_tile() >= 0 && !(std::getenv("VLA_FQ_TILE_SMALL_M") && g.M > 64)) {
        const int t = forced_tile();
        if (g.K % 128 != 0 && (t == 1 || t == 8 || t == 9 || t == 10 || t == 12 || t == 14 || t == 15)) return cudaErrorNotSupported;
        if (g.K % 256 != 0 && (t == 13 || t == 16)) return cudaErrorNotSupported;
        return launch_forced(t, g, stream);
    }
    // LLM prefill (M > 64, ~160 tokens on GR00T): measured on Orin (VLA_FQ_TILE
    // sweep, tests/foldquant_gemm_check). The wide gate+up site (N = 12288) and
    // long prefixes want the 128-row, BK=128 two-stage tile (M=160: 719 -> 426
    // us; M=1024 qkv 1081 -> 689 us); the other sites a 192-row tile that holds
    // 160 tokens in one M tile with 17% zero rows instead of 256's 37%
    // (qkv 173 -> 131, o 122 -> 69, down 339 -> 207 us).
    if (g.M > 64) {
        if ((g.N >= 8192 || g.M > 256) && g.K % 128 == 0)
            return launch_tile<128, 64, 128, 2, 4, 2>(g, stream);
        return launch_tile<192, 64, 64, 3, 4, 2>(g, stream);
    }
    // M <= 64 (the DiT's 41 action tokens): narrow N (o_proj, ff2, cross kv)
    // gets a 32-wide N tile for the CTA count. Split-K over the same tiles
    // (VLA_FQ_TILE=20..25, exact int32 atomics) was measured 5-8% slower end
    // to end on Orin - the atomics and the extra L2 round trip cost more than
    // the added CTAs buy - so it stays an opt-in candidate for other GPUs.
    if (g.N <= 3072 && g.K % 128 == 0)
        return launch_tile<64, 32, 128, 3, 2, 2>(g, stream);
    return launch_tile<64, 64, 64, 4, 2, 2>(g, stream);
}

}  // namespace fq
}  // namespace vla
