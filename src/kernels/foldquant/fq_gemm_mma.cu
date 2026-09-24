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

// Integer tensor-core GEMM on mma.sync + ldmatrix, the successor of the wmma
// kernel in fq_gemm_i8.cu for the shapes a VLA runs: a few dozen to a few
// hundred activation rows against a weight matrix streamed once from DRAM.
//
// Two operand widths share one kernel body:
//   INT8: mma.m16n8k32.s8  - smem rows hold BK bytes (one byte per k)
//   INT4: mma.m16n8k64.s4  - smem rows hold BK/2 bytes (two k per byte, the
//         low nibble the even k, exactly the file and blob packing)
// and the same ldmatrix fragment mapping serves both, since ldmatrix moves
// 16-byte row segments and mma reads its A/B registers as packed k runs.
//
// Layout: A = the activation blob (row m at m*row_bytes, codes then the float
// scale), B = W[N][K_pack] with K contiguous. Both tiles land in shared memory
// through a cp.async pipeline with the 16-byte chunks XOR-swizzled by (row & 7),
// so the eight rows an ldmatrix touches hit eight different bank groups.
// Accumulation is int32 and the epilogue y = ((float)acc * act_scale[m]) *
// wscale[n] (+ bias[n]) is applied per element in that order, so the result is
// bit-identical to the wmma kernel and the CPU reference for any tiling.

#include "fq_kernels.h"

#include <cstdlib>

namespace vla {
namespace fq {

namespace {

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool pred) {
    const unsigned s  = (unsigned) __cvta_generic_to_shared(smem);
    const int      sz = pred ? 16 : 0;   // src-size 0 zero-fills the 16 bytes
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned & r0, unsigned & r1, unsigned & r2, unsigned & r3, const void * smem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}

// D = A * B + D on one m16n8 tile; K is 32 (s8) or 64 (s4) packed values.
template <int KPACK>
__device__ __forceinline__ void mma_tile(int * c, const unsigned * a, const unsigned * b) {
    if (KPACK == 1) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

// BM x BN CTA tile, BKB bytes of K per stage (32 or 64 k-values per mma step,
// i.e. 32 bytes for both widths), WM x WN warps. Rows in smem are BKB bytes,
// a multiple of 128 so the swizzle covers the 8 chunks a bank group spans.
template <int BM, int BN, int BKB, int STAGES, int WM, int WN, int KPACK>
struct MTile {
    static constexpr int THREADS  = WM * WN * 32;
    static constexpr int CHUNKS_R = BKB / 16;                   // 16-byte chunks per row
    static constexpr int STAGE_BYTES = (BM + BN) * BKB;
    static constexpr int SMEM_BYTES  = STAGES * STAGE_BYTES;
    static constexpr int CHUNKS  = (BM + BN) * CHUNKS_R;
    static constexpr int PER_THR = CHUNKS / THREADS;
    static constexpr int MT = BM / WM / 16;                     // m16 tiles per warp
    static constexpr int NT = BN / WN / 8;                      // n8 tiles per warp
    static constexpr int KSTEP_B = 32;                          // bytes of k per mma step
    static constexpr int KSTEPS  = BKB / KSTEP_B;
    static_assert(CHUNKS % THREADS == 0, "tile copies must split evenly over the threads");
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0, "warp tile: multiples of m16 / n16");
    static_assert(BKB % 128 == 0, "smem rows must be a multiple of 128 bytes for the swizzle");
    static_assert(NT % 2 == 0, "B fragments load two n8 tiles per ldmatrix.x4");
};

// Physical byte offset of 16-byte chunk c of row r within a tile whose rows are BKB bytes.
template <int BKB>
__device__ __forceinline__ int swz(int r, int c) { return r * BKB + ((c ^ (r & 7)) << 4); }

template <int BM, int BN, int BKB, int STAGES, int WM, int WN, int KPACK, bool HAS_BIAS>
__global__ void __launch_bounds__(WM * WN * 32) gemm_mma_kernel(const GemmArgs g) {
    using T = MTile<BM, BN, BKB, STAGES, WM, WN, KPACK>;
    extern __shared__ __align__(128) int8_t smem[];

    const int     tid  = threadIdx.x;
    const int     warp = tid >> 5, lane = tid & 31;
    const int     wm   = warp / WN, wn = warp % WN;
    const int64_t m0   = (int64_t) blockIdx.x * BM;
    const int64_t n0   = (int64_t) blockIdx.y * BN;
    const int64_t KB   = g.K / KPACK;                           // packed bytes per row
    const int     KT   = (int) (KB / BKB);

    // Per-thread copy list for one stage (sources at k0 = 0).
    const int8_t * src[T::PER_THR];
    int            dst[T::PER_THR];
    bool           pred[T::PER_THR];
    #pragma unroll
    for (int i = 0; i < T::PER_THR; ++i) {
        const int c = tid + i * T::THREADS;
        const int r = c / T::CHUNKS_R, c16 = c % T::CHUNKS_R;
        if (r < BM) {
            const int64_t m = m0 + r;
            pred[i] = m < g.M;
            src[i]  = g.blob + (pred[i] ? m : 0) * g.row_bytes + c16 * 16;
            dst[i]  = swz<BKB>(r, c16);
        } else {
            const int rb = r - BM;
            pred[i] = true;
            src[i]  = g.w + (n0 + rb) * KB + c16 * 16;
            dst[i]  = BM * BKB + swz<BKB>(rb, c16);
        }
    }
    auto load_tile = [&](int stage, int kt) {
        int8_t * base = smem + stage * T::STAGE_BYTES;
        const int64_t k0 = (int64_t) kt * BKB;
        #pragma unroll
        for (int i = 0; i < T::PER_THR; ++i) cp_async16(base + dst[i], src[i] + k0, pred[i]);
    };

    int acc[T::MT][T::NT][4];
    #pragma unroll
    for (int i = 0; i < T::MT; ++i)
        #pragma unroll
        for (int j = 0; j < T::NT; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }

    // ldmatrix row addressing: lane l provides the row (l & 7) of matrix (l >> 3).
    // A matrices per m16 x 32B step: (rows 0-7, chunk 0), (rows 8-15, chunk 0),
    // (rows 0-7, chunk 1), (rows 8-15, chunk 1) -> a0..a3.
    // B matrices per n16 x 32B step: (n 0-7, chunk 0), (n 0-7, chunk 1),
    // (n 8-15, chunk 0), (n 8-15, chunk 1) -> b0,b1 of tile n, b0,b1 of tile n+8.
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_chk = lane >> 4;
    const int b_row = (lane & 7) + (lane >> 4) * 8,        b_chk = (lane >> 3) & 1;

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int stage = kt % STAGES;
        const int nk = kt + STAGES - 1;
        if (nk < KT) load_tile(nk % STAGES, nk);
        cp_async_commit();

        const int8_t * as = smem + stage * T::STAGE_BYTES;
        const int8_t * bs = as + BM * BKB;
        #pragma unroll
        for (int ks = 0; ks < T::KSTEPS; ++ks) {
            unsigned af[T::MT][4], bf[T::NT][2];
            #pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (T::MT * 16) + i * 16 + a_row;
                ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], as + swz<BKB>(r, ks * 2 + a_chk));
            }
            #pragma unroll
            for (int j = 0; j < T::NT; j += 2) {
                const int r = wn * (T::NT * 8) + j * 8 + b_row;
                ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], bs + swz<BKB>(r, ks * 2 + b_chk));
            }
            #pragma unroll
            for (int i = 0; i < T::MT; ++i)
                #pragma unroll
                for (int j = 0; j < T::NT; ++j)
                    mma_tile<KPACK>(acc[i][j], af[i], bf[j]);
        }
    }
    cp_async_wait<0>();

    // Next site's weights: every CTA prefetches an equal slice of the head of
    // that matrix into L2 now that its own loads are in, so the next GEMM's
    // first stages hit L2 instead of DRAM. Fire-and-forget; no data dependence.
    if (g.pf) {
        const int64_t lines  = g.pf_bytes >> 7;
        const int64_t ctas   = (int64_t) gridDim.x * gridDim.y;
        const int64_t cta    = (int64_t) blockIdx.y * gridDim.x + blockIdx.x;
        const int64_t per    = (lines + ctas - 1) / ctas;
        const int64_t l0     = cta * per, l1 = l0 + per < lines ? l0 + per : lines;
        for (int64_t l = l0 + tid; l < l1; l += T::THREADS)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(g.pf + (l << 7)));
    }

    // Fragment layout of a m16n8 int32 tile: c0,c1 at (row g, cols 2q, 2q+1),
    // c2,c3 at (row g+8, same cols), g = lane/4, q = lane%4.
    const int grp = lane >> 2, q = lane & 3;
    // The per-column factors are shared by every row this thread writes: load
    // them once per n8 tile instead of once per (row, tile).
    float2 ws[T::NT], bs2[T::NT];
    int ho[T::NT], hs[T::NT];   // head layout: y[ho + m*hs] for column n, column n+1 at +1 (Q/K) or +M (V)
    #pragma unroll
    for (int j = 0; j < T::NT; ++j) {
        const int64_t n = n0 + wn * (T::NT * 8) + j * 8 + 2 * q;
        ws[j] = *(const float2 *) (g.wscale + n);
        if (HAS_BIAS) bs2[j] = *(const float2 *) (g.bias + n);
        if (g.heads) fq_out_column(g, (int) n, ho[j], hs[j]);
    }
    #pragma unroll
    for (int i = 0; i < T::MT; ++i) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t m = m0 + wm * (T::MT * 16) + i * 16 + grp + h * 8;
            if (m >= g.M) continue;
            const float xs = *(const float *) (g.blob + m * g.row_bytes + KB);
            #pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int64_t n = n0 + wn * (T::NT * 8) + j * 8 + 2 * q;
                float v0 = ((float) acc[i][j][2 * h]     * xs) * ws[j].x;
                float v1 = ((float) acc[i][j][2 * h + 1] * xs) * ws[j].y;
                if (HAS_BIAS) { v0 = v0 + bs2[j].x; v1 = v1 + bs2[j].y; }
                if (g.res) {   // the residual add that followed the GEMM: one float add, same result
                    const float2 r = *(const float2 *) (g.res + m * g.N + n);
                    v0 = v0 + r.x; v1 = v1 + r.y;
                }
                if (g.heads) {   // head layout (hd even, n even: a Q/K pair stays adjacent and 8-B aligned)
                    float * yp = g.y + ho[j] + m * hs[j];
                    if (hs[j] == 1) { yp[0] = v0; yp[(int) g.M] = v1; }
                    else            { *(float2 *) yp = make_float2(v0, v1); }
                } else {
                    *(float2 *) (g.y + m * g.N + n) = make_float2(v0, v1);
                }
            }
        }
    }
}

template <int BM, int BN, int BKB, int STAGES, int WM, int WN, int KPACK>
cudaError_t launch_mma(const GemmArgs & g, cudaStream_t stream) {
    using T = MTile<BM, BN, BKB, STAGES, WM, WN, KPACK>;
    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(gemm_mma_kernel<BM, BN, BKB, STAGES, WM, WN, KPACK, true>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_BYTES);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute(gemm_mma_kernel<BM, BN, BKB, STAGES, WM, WN, KPACK, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_BYTES);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }
    if ((g.K / KPACK) % BKB != 0 || g.N % BN != 0) return cudaErrorNotSupported;
    const dim3 grid((unsigned) ((g.M + BM - 1) / BM), (unsigned) (g.N / BN));
    if (g.bias) gemm_mma_kernel<BM, BN, BKB, STAGES, WM, WN, KPACK, true ><<<grid, T::THREADS, T::SMEM_BYTES, stream>>>(g);
    else        gemm_mma_kernel<BM, BN, BKB, STAGES, WM, WN, KPACK, false><<<grid, T::THREADS, T::SMEM_BYTES, stream>>>(g);
    return cudaGetLastError();
}

}  // namespace

// variant: -1 = the per-shape default, otherwise one of the candidates below
// (VLA_FQ_MMA=<n> from launch_gemm), for tuning on another GPU.
cudaError_t launch_gemm_mma(const GemmArgs & g, int variant, cudaStream_t stream) {
    const bool w4 = g.wbits == 4;
    if ((g.wbits != 8 && g.wbits != 4) || g.abits != g.wbits) return cudaErrorNotSupported;   // W8A8 or W4A4
    if (variant < 0) {
        if (g.M > 64) {
            // 128x64x128, 3 stages (72 KB smem, 2 CTAs/SM) for every prefill site: on
            // Orin it beat the 192-row tile end to end (ALOHA shape p50 174 vs 205 ms).
            variant = 3;
        } else {
            variant = g.N <= 3072 ? 0 : 1;
        }
    }
    if (!w4) {
        switch (variant) {
            case 0: return launch_mma<64,  32,  128, 3, 2, 2, 1>(g, stream);
            case 1: return launch_mma<64,  64,  128, 3, 2, 2, 1>(g, stream);
            case 2: return launch_mma<192, 64,  128, 3, 4, 2, 1>(g, stream);
            case 3: return launch_mma<128, 64,  128, 3, 4, 2, 1>(g, stream);
            case 4: return launch_mma<64,  32,  256, 3, 2, 2, 1>(g, stream);
            case 5: return launch_mma<128, 32,  128, 3, 4, 2, 1>(g, stream);
            case 6: return launch_mma<64,  64,  128, 4, 2, 2, 1>(g, stream);
            case 7: return launch_mma<256, 64,  128, 3, 4, 2, 1>(g, stream);
            case 8: return launch_mma<192, 64,  128, 2, 4, 2, 1>(g, stream);   // 64 KB smem: 2 CTAs/SM
            case 9: return launch_mma<128, 64,  128, 2, 4, 2, 1>(g, stream);
            case 10: return launch_mma<96, 64,  128, 3, 2, 2, 1>(g, stream);   // 3 m16 tiles per warp
            // Small-M candidates (DiT, M = 41): fewer padded rows, more CTAs, deeper
            // pipeline. Cold micro-bench on Orin: 11/13 within 2% of the default, the
            // rest 5-45% slower (the sites run at 130-145 GB/s, near the DRAM ceiling).
            case 11: return launch_mma<48,  32,  128, 3, 1, 2, 1>(g, stream);   // 2 warps, 30 KB smem
            case 12: return launch_mma<48,  32,  128, 6, 1, 2, 1>(g, stream);
            case 13: return launch_mma<48,  64,  128, 3, 1, 4, 1>(g, stream);
            case 14: return launch_mma<32,  32,  128, 3, 1, 2, 1>(g, stream);   // two m-tiles per site
            case 15: return launch_mma<64,  32,  128, 6, 2, 2, 1>(g, stream);
            case 16: return launch_mma<48,  32,  256, 3, 1, 2, 1>(g, stream);
            default: return cudaErrorInvalidValue;
        }
    }
    switch (variant) {   // INT4: rows hold K/2 bytes, so BKB=128 covers 256 k
        case 0: return launch_mma<64,  32,  128, 3, 2, 2, 2>(g, stream);
        case 1: return launch_mma<64,  64,  128, 3, 2, 2, 2>(g, stream);
        case 2: return launch_mma<192, 64,  128, 3, 4, 2, 2>(g, stream);
        case 11: return launch_mma<48,  32,  128, 3, 1, 2, 2>(g, stream);
        case 12: return launch_mma<48,  32,  128, 6, 1, 2, 2>(g, stream);
        case 13: return launch_mma<48,  64,  128, 3, 1, 4, 2>(g, stream);
        case 14: return launch_mma<32,  32,  128, 3, 1, 2, 2>(g, stream);
        case 15: return launch_mma<64,  32,  128, 6, 2, 2, 2>(g, stream);
        case 16: return launch_mma<48,  32,  256, 3, 1, 2, 2>(g, stream);
        case 3: return launch_mma<128, 64,  128, 3, 4, 2, 2>(g, stream);
        default: return launch_mma<64,  32,  128, 3, 2, 2, 2>(g, stream);
    }
}

}  // namespace fq
}  // namespace vla
