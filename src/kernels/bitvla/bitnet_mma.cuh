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

/**
 * @file bitnet_mma.cuh
 * @brief int8 x int2 GEMM on mma.sync.m16n8k32 (header-only).
 *
 * Gives the same bits as the ladder kernels in bitnet_kernels.h: products are
 * summed in int32 (exact in any order) and the epilogue is the same
 * expression, bf16(((float)acc)/s[m]*ws[group]).
 *
 * - A and the weight are staged with cp.async (STAGES deep); A is read with
 *   ldmatrix from an XOR-swizzled tile.
 * - The weight stays 2-bit in shared memory. Lanes decode unsigned codes
 *   u = w+2 in 0..3 and the MMA runs s8 x u8; the accumulator then gets
 *   -2*rowsum(A), with the row sums taken from the A fragments by dp4a.
 * - M tiles run fastest in the grid, so CTAs sharing a weight tile run
 *   together.
 * - Up to three GEMMs sharing A (q/k/v) run as one launch.
 *
 * Ladder pack: 16-column tiles of K/4 bytes, each 128-wide k chunk 512
 * bytes. The word for column ty, k = 16*tx .. 16*tx+15 is at byte
 * (tx>>1)*128 + (ty>>3)*64 + (tx&1)*32 + (ty&7)*4; its decode word i holds
 * k = 16*tx+4i .. +3, one code per byte at bits 2i.
 *
 * Fragment layout (repack_frag): same 512-byte blocks. Lane l of n8 half nb
 * reads 8 bytes at (32*nb+l)*8: two words holding its eight B registers for
 * the chunk (word w, register r: k step 2*w+r/2, k half r%2), codes at bits
 * 8*j+2*r.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace bitmma {

enum Act { ACT_NONE = 0, ACT_GELU_TANH = 1 };

// Epilogue ops of a launch. A template parameter: run-time branches here slow
// the main loop.
enum Epi { EPI_BIAS = 1, EPI_GELU = 2, EPI_RES = 4, EPI_HEADS = 8 };

// One GEMM of a launch and its epilogue ops, applied in the order and with
// the bf16 roundings of the original separate kernels:
//   y = bf16(acc/s*ws); y = bf16(y+bias); y = act(y); y = bf16(res+y)
struct Seg {
    const int8_t * B;
    __nv_bfloat16 * out;
    const float * ws;
    int N;        ///< output columns, a multiple of 16
    int ws_div;   ///< N/ws_num: columns per weight-scale group
    int tile0;    ///< first BN-wide column tile of this GEMM in the launch
    const __nv_bfloat16 * bias = nullptr;  ///< add_bias_bf16
    const __nv_bfloat16 * res  = nullptr;  ///< add_bf16(res, y); may alias out
    int act = ACT_NONE;
    int hd  = 0;  ///< >0: store head-major [N/hd][M][hd] (transpose_sHhd_to_HShd)
    int rep = 1;  ///< with hd: store each head to rep consecutive heads (repeat_kv)
};

struct Segs {
    Seg s[3];
    int n;
    bool frag = false;   ///< weights in the fragment layout (repack_frag)
};

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool pred) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void * smem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}

// Codes at bits 2i of each byte of w (weight = code-2, see decode_i2s_to_i8s).
__device__ __forceinline__ unsigned decode_codes(unsigned w, int i) {
    return (w >> (2*i)) & 0x03030303u;
}

__device__ __forceinline__ void mma_s8u8(int (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int BK = 128;   // the pack's k chunk

// gelu_tanh_bf16_kernel's arithmetic.
__device__ __forceinline__ float gelu_tanh(float x) {
    const float kAlpha = 0.7978845608028654f;
    const float kBeta  = 0.044715f;
    float u = kAlpha * (x+kBeta * x * x * x);
    float t = tanhf(u);
    return 0.5f * x * (1.0f+t);
}

template <int EPI>
__device__ __forceinline__ __nv_bfloat16 epilogue(const Seg & sg, int acc, float sm, float wsv, int n, size_t res_idx) {
    __nv_bfloat16 y = __float2bfloat16(((float) acc)/sm*wsv);
    if constexpr ((EPI & EPI_BIAS) != 0) y = __float2bfloat16(__bfloat162float(y)+__bfloat162float(sg.bias[n]));
    if constexpr ((EPI & EPI_GELU) != 0) y = __float2bfloat16(gelu_tanh(__bfloat162float(y)));
    if constexpr ((EPI & EPI_RES) != 0)  y = __float2bfloat16(__bfloat162float(sg.res[res_idx])+__bfloat162float(y));
    return y;
}

// The Epi flags a segment asks for.
inline int epi_of(const Seg & sg) {
    return (sg.bias ? EPI_BIAS : 0) | (sg.act == ACT_GELU_TANH ? EPI_GELU : 0) | (sg.res ? EPI_RES : 0) | (sg.hd > 0 ? EPI_HEADS : 0);
}

/**
 * @tparam BM,BN  CTA tile; BN a multiple of 16.
 * @tparam WM,WN  warp tile; multiples of 16 and 8.
 * @tparam STAGES cp.async pipeline depth.
 * @tparam EPI    Epi flags of every segment of the launch.
 * @param A  int8 activations, M x K row-major.
 * @param s  per-row activation scale (length M).
 */
template <int BM, int BN, int WM, int WN, int STAGES, int EPI, bool FRAG>
__global__ void __launch_bounds__((BM/WM)*(BN/WN)*32)
k_bitnet_mma(const int8_t * __restrict__ A, const float * __restrict__ s, Segs segs, int M, int K) {
    constexpr int WARPS_N = BN/WN, NTH = (BM/WM)*WARPS_N*32;
    constexpr int MT = WM/16, NT = WN/8;
    constexpr int A_BYTES = BM*BK, B_BYTES = BN*BK/4;
    extern __shared__ __align__(128) int8_t smem[];

    // Which GEMM of the launch, and which column tile of it.
    const int ty = (int) blockIdx.y;
    int si = 0;
    #pragma unroll
    for (int i = 1; i < 3; ++i)
        if (i < segs.n && ty >= segs.s[i].tile0) si = i;
    const Seg sg = segs.s[si];
    const int n_base = (ty - sg.tile0)*BN;
    const int m_base = (int) blockIdx.x*BM;
    const int n16 = sg.N/16;

    const int tid = (int) threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp/WARPS_N)*WM, wn = (warp%WARPS_N)*WN;
    const int KT = K/BK;

    auto load = [&](int stage, int kt) {
        int8_t * sa = smem + stage*(A_BYTES+B_BYTES);
        int8_t * sb = sa + A_BYTES;
        #pragma unroll
        for (int i = tid; i < BM*8; i += NTH) {
            const int r = i >> 3, c = i & 7;
            const int m = m_base + r;
            cp_async16(sa + r*BK + ((c ^ (r & 7)) << 4), A + (size_t) (m < M ? m : 0)*K + kt*BK + c*16, m < M);
        }
        #pragma unroll
        for (int i = tid; i < BN*2; i += NTH) {
            const int t = i >> 5, c = i & 31;
            const int tile = n_base/16 + t;
            if (tile < n16)
                cp_async16(sb + t*512 + c*16, sg.B + (size_t) tile*16*K/4 + (size_t) kt*512 + c*16, true);
        }
    };

    int acc[MT][NT][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;

    #pragma unroll
    for (int st = 0; st < STAGES-1; ++st) {
        if (st < KT) load(st, st);
        cp_async_commit();
    }

    int rs[MT][2] = {};   // partial row sums of A: rows lane/4 and lane/4+8 of each m16 tile
    // B: lane's column within an n8 tile and decode word.
    const int bn = lane >> 2, bw = lane & 3;
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES-2>();
        __syncthreads();
        if (kt+STAGES-1 < KT) load((kt+STAGES-1) % STAGES, kt+STAGES-1);
        cp_async_commit();

        const int8_t * sa = smem + (kt % STAGES)*(A_BYTES+B_BYTES);
        const int8_t * sb = sa + A_BYTES;
        uint2 bq[FRAG ? NT : 1];
        if constexpr (FRAG) {
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int n = wn + j*8;   // n8 tile within the CTA tile
                bq[j] = *(const uint2 *) (sb + (n >> 4)*512 + (((n >> 3) & 1)*32 + lane)*8);
            }
        }
        #pragma unroll
        for (int ks = 0; ks < BK/32; ++ks) {
            unsigned a[MT][4];
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm + i*16 + (lane & 15);
                const int c = ks*2 + (lane >> 4);
                ldmatrix_x4(a[i], sa + r*BK + ((c ^ (r & 7)) << 4));
                rs[i][0] = __dp4a((int) a[i][0], 0x01010101, __dp4a((int) a[i][2], 0x01010101, rs[i][0]));
                rs[i][1] = __dp4a((int) a[i][1], 0x01010101, __dp4a((int) a[i][3], 0x01010101, rs[i][1]));
            }
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                unsigned b0, b1;
                if constexpr (FRAG) {
                    const unsigned w = ks < 2 ? bq[j].x : bq[j].y;
                    b0 = decode_codes(w, (ks & 1)*2);
                    b1 = decode_codes(w, (ks & 1)*2 + 1);
                } else {
                    const int n = wn + j*8 + bn;   // column within the CTA tile
                    const int8_t * w = sb + (n >> 4)*512 + ks*128 + ((n & 15) >> 3)*64 + (n & 7)*4;
                    b0 = decode_codes(*(const unsigned *) w, bw);
                    b1 = decode_codes(*(const unsigned *) (w + 32), bw);
                }
                #pragma unroll
                for (int i = 0; i < MT; ++i)
                    mma_s8u8(acc[i][j], a[i], b0, b1);
            }
        }
    }
    cp_async_wait<0>();
    // Row sums: the four lanes of a quad hold different k of the same rows.
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            int v = rs[i][h];
            v += __shfl_xor_sync(0xffffffff, v, 1);
            v += __shfl_xor_sync(0xffffffff, v, 2);
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[i][j][2*h+0] -= 2*v;
                acc[i][j][2*h+1] -= 2*v;
            }
        }

    // Column-major order: the head index is computed once per column pair.
    float sm[MT][2];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = m_base + wm + i*16 + (lane >> 2) + h*8;
            sm[i][h] = m < M ? s[m] : 1.0f;
        }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int n = n_base + wn + j*8 + (lane & 3)*2;
        if (n >= sg.N) continue;
        const float wsv = sg.ws[n/sg.ws_div];
        int head = 0, d = 0;
        if constexpr ((EPI & EPI_HEADS) != 0) { head = n/sg.hd; d = n - head*sg.hd; }
        #pragma unroll
        for (int i = 0; i < MT; ++i) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m_base + wm + i*16 + (lane >> 2) + h*8;
                if (m >= M) continue;
                const size_t idx = (size_t) m*sg.N + n;
                __nv_bfloat162 v;
                v.x = epilogue<EPI>(sg, acc[i][j][2*h+0], sm[i][h], wsv, n+0, idx+0);
                v.y = epilogue<EPI>(sg, acc[i][j][2*h+1], sm[i][h], wsv, n+1, idx+1);
                if constexpr ((EPI & EPI_HEADS) == 0) {
                    *(__nv_bfloat162 *) (sg.out + idx) = v;
                } else {
                    __nv_bfloat16 * o = sg.out + ((size_t) head*sg.rep*M + m)*sg.hd + d;
                    for (int r = 0; r < sg.rep; ++r)
                        *(__nv_bfloat162 *) (o + (size_t) r*M*sg.hd) = v;
                }
            }
        }
    }
}

template <int BM, int BN, int WM, int WN, int STAGES, int EPI, bool FRAG>
inline void launch_epi(const int8_t * A, const float * s, const Segs & segs, int tiles, int M, int K, cudaStream_t st) {
    constexpr int NTH = (BM/WM)*(BN/WN)*32;
    constexpr int SMEM = STAGES*(BM*BK + BN*BK/4);
    auto kern = k_bitnet_mma<BM, BN, WM, WN, STAGES, EPI, FRAG>;
    if constexpr (SMEM > 48*1024) {
        static const bool attr = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
        (void) attr;
    }
    kern<<<dim3((M+BM-1)/BM, tiles), NTH, SMEM, st>>>(A, s, segs, M, K);
}

// All segments of a launch must ask for the same Epi flags.
template <int BM, int BN, int WM, int WN, int STAGES>
inline cudaError_t launch(const int8_t * A, const float * s, Segs segs, int M, int K, cudaStream_t st) {
    int tiles = 0;
    const int epi = epi_of(segs.s[0]);
    for (int i = 0; i < segs.n; ++i) {
        if (epi_of(segs.s[i]) != epi) return cudaErrorInvalidValue;
        segs.s[i].tile0 = tiles;
        tiles += (segs.s[i].N + BN-1)/BN;
    }
    if (!segs.frag) {   // ladder pack: tests only, plain epilogue
        if (epi != 0) return cudaErrorInvalidValue;
        launch_epi<BM, BN, WM, WN, STAGES, 0, false>(A, s, segs, tiles, M, K, st);
        return cudaGetLastError();
    }
#define BITMMA_EPI(E) case E: launch_epi<BM, BN, WM, WN, STAGES, E, true>(A, s, segs, tiles, M, K, st); break;
    switch (epi) {
        BITMMA_EPI(0)
        BITMMA_EPI(EPI_HEADS)
        BITMMA_EPI(EPI_RES)
        BITMMA_EPI(EPI_BIAS | EPI_HEADS)
        BITMMA_EPI(EPI_BIAS | EPI_RES)
        BITMMA_EPI(EPI_BIAS | EPI_GELU)
        default: return cudaErrorInvalidValue;
    }
#undef BITMMA_EPI
    return cudaGetLastError();
}


// Ladder pack -> fragment layout, one output word per thread; N x K weight.
static __global__ void k_repack_frag(const uint8_t * __restrict__ src, uint32_t * __restrict__ dst, int n_words) {
    const int wi = (int) (blockIdx.x*blockDim.x + threadIdx.x);
    if (wi >= n_words) return;
    const int blk = wi >> 7, in = wi & 127;            // 512-byte block (tile, chunk), word within it
    const int word = in & 1, l = (in >> 1) & 31, nb = in >> 6;
    const int n = nb*8 + (l >> 2), q = l & 3;           // column in the tile, lane's word in the quad
    const uint8_t * sblk = src + (size_t) blk*512;
    uint32_t out = 0;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int ks = word*2 + (r >> 1), half = r & 1;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int kk = ks*32 + half*16 + q*4 + j;   // k within the 128-wide chunk
            const int tx = kk >> 4, e = kk & 15;
            const uint32_t w = *(const uint32_t *) (sblk + (tx >> 1)*128 + (n >> 3)*64 + (tx & 1)*32 + (n & 7)*4);
            const uint32_t code = (w >> (8*(e & 3) + 2*(e >> 2))) & 3u;
            out |= code << (8*j + 2*r);
        }
    }
    dst[wi] = out;
}

// Repacks an N x K ladder-packed weight (N*K/4 bytes) into the fragment layout, in place.
inline cudaError_t repack_frag(int8_t * w, int N, int K, cudaStream_t st) {
    const size_t bytes = (size_t) N*K/4;
    void * tmp = nullptr;
    cudaError_t e = cudaMalloc(&tmp, bytes);
    if (e != cudaSuccess) return e;
    const int n_words = (int) (bytes/4);
    k_repack_frag<<<(n_words+255)/256, 256, 0, st>>>((const uint8_t *) w, (uint32_t *) tmp, n_words);
    e = cudaMemcpyAsync(w, tmp, bytes, cudaMemcpyDeviceToDevice, st);
    if (e == cudaSuccess) e = cudaStreamSynchronize(st);
    cudaFree(tmp);
    return e;
}

}  // namespace bitmma
