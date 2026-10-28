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
 * @file bitvla_rowops.cuh
 * @brief Row kernels for BitVLA with the same bits as the originals in
 *        bitvla_lm_cuda.cu.
 *
 * The originals reduce a row with one 256-thread block: thread t sums
 * elements t, t+256, ... in order, each warp folds with a shfl_down tree, and
 * warp 0 folds the 8 warp sums the same way. Float sums depend on that order,
 * so these kernels keep it (a single warp can play all 8 warps, lane l holding
 * thread l+32*w in register w). Maxima are order-free.
 *
 * Element-wise code matches the originals' SASS: where it has an FFMA, the
 * code here spells __fmaf_rn, because values held in registers would
 * otherwise be shared and not contracted.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace bitvla_rowops {

constexpr int VW = 8;   // warps of the originals' 256-thread block

// The originals' block sum, with v[w] = thread lane+32*w. Result in every lane.
__device__ __forceinline__ float block256_sum(const float (&v)[VW]) {
    float ws[VW];
    #pragma unroll
    for (int w = 0; w < VW; ++w) {
        float x = v[w];
        for (int off = 16; off > 0; off >>= 1)
            x += __shfl_down_sync(0xffffffff, x, off);
        ws[w] = x;   // the warp's sum, valid in lane 0
    }
    const int lane = threadIdx.x & 31;
    float x = 0.0f;
    #pragma unroll
    for (int w = 0; w < VW; ++w) {
        const float b = __shfl_sync(0xffffffff, ws[w], 0);
        if (lane == w) x = b;
    }
    for (int off = 16; off > 0; off >>= 1)
        x += __shfl_down_sync(0xffffffff, x, off);
    return __shfl_sync(0xffffffff, x, 0);
}

/**
 * softmax_scaled_bf16_kernel<256> with one warp per row, for S <= 256*Q.
 * In place on bf16 rows of length S.
 */
template <int Q>
__global__ void __launch_bounds__(128) softmax_scaled_warp(__nv_bfloat16 * __restrict__ inout, float scale, int S, int rows) {
    const int row = (int) blockIdx.x*4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    __nv_bfloat16 * r = inout + (size_t) row*S;

    float x[Q][VW];
    float mx = -INFINITY;
    #pragma unroll
    for (int q = 0; q < Q; ++q)
        #pragma unroll
        for (int w = 0; w < VW; ++w) {
            const int i = q*256 + w*32 + lane;
            x[q][w] = i < S ? __bfloat162float(r[i]) : 0.0f;
            if (i < S) {
                float v = x[q][w]*scale;
                if (v > mx)
                    mx = v;
            }
        }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, mx, off);
        if (other > mx)
            mx = other;
    }
    const float max_v = mx;

    float s[VW];
    #pragma unroll
    for (int w = 0; w < VW; ++w) {
        float s_sum = 0.0f;
        #pragma unroll
        for (int q = 0; q < Q; ++q)
            if (q*256 + w*32 + lane < S)
                s_sum += expf(__fmaf_rn(x[q][w], scale, -max_v));
        s[w] = s_sum;
    }
    const float inv_sum = 1.0f/block256_sum(s);

    #pragma unroll
    for (int q = 0; q < Q; ++q)
        #pragma unroll
        for (int w = 0; w < VW; ++w) {
            const int i = q*256 + w*32 + lane;
            if (i < S) {
                float v = expf(__fmaf_rn(x[q][w], scale, -max_v))*inv_sum;
                r[i] = __float2bfloat16(v);
            }
        }
}

// Launches softmax_scaled_warp; false when S is too long for it.
inline bool softmax_scaled(__nv_bfloat16 * inout, float scale, int n_rows, int S, cudaStream_t st) {
    const dim3 grid((n_rows+3)/4), block(128);
    if (S <= 256)      softmax_scaled_warp<1><<<grid, block, 0, st>>>(inout, scale, S, n_rows);
    else if (S <= 512) softmax_scaled_warp<2><<<grid, block, 0, st>>>(inout, scale, S, n_rows);
    else return false;
    return true;
}


// ---------------------------------------------------------------------------
// Norm + act_quant: rmsnorm_bf16_kernel or layernorm_bias_bf16_kernel (bf16
// out), then act_quant_kernel, in one 256-thread block per row.

enum In { IN_ROW = 0, IN_HEADS = 1, IN_SQRELU = 2 };
enum Norm { NORM_NONE = 0, NORM_RMS = 1, NORM_LN = 2 };

struct NormQuant {
    const __nv_bfloat16 * x;   ///< IN_ROW [M][K]; IN_HEADS [K/hd][M][hd]; IN_SQRELU [M][2K] (gate | up)
    const __nv_bfloat16 * w;
    const __nv_bfloat16 * b;
    float eps;
    int K, hd, M;
    int8_t * q;                ///< [M][ld_q]
    int ld_q;
    float * scales;            ///< [M]
};

// The originals' 256-thread block sum (warp shfl_down trees, warp 0 folds).
__device__ __forceinline__ float block_sum256(float v, float * smem) {
    const int tid = threadIdx.x;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31) == 0)
        smem[tid >> 5] = v;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float x = (tid < 8) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            x += __shfl_down_sync(0xffffffff, x, off);
        if (tid == 0)
            smem[0] = x;
    }
    __syncthreads();
    const float r = smem[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_max256(float v, float * smem) {
    const int tid = threadIdx.x;
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    if ((tid & 31) == 0)
        smem[tid >> 5] = v;
    __syncthreads();
    float r = smem[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i)
        r = fmaxf(r, smem[i]);
    __syncthreads();
    return r;
}

// Converts 8 bf16 (one 16-byte load) to float.
__device__ __forceinline__ void bf16x8(const __nv_bfloat16 * src, float (&f)[8]) {
    const uint4 u = *(const uint4 *) src;
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2*i+0] = __uint_as_float(w[i] << 16);
        f[2*i+1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// gate_up_fused_sqrelu_mul_bf16_kernel's value for one element.
__device__ __forceinline__ float sqrelu_mul(float g, float u) {
    if (g < 0.0f)
        g = 0.0f;
    return __bfloat162float(__float2bfloat16(g * g * u));
}

// Loads the row into row[0..K). VEC: 16-byte loads (K, hd multiples of 8).
template <int IN, bool VEC>
__device__ __forceinline__ void stage_row(const NormQuant & p, int m, float * row) {
    const int tid = threadIdx.x, K = p.K;
    if constexpr (VEC) {
        for (int v = tid; v < K/8; v += 256) {
            const int k = v*8;
            float f[8];
            if constexpr (IN == IN_ROW) {
                bf16x8(p.x + (size_t) m*K + k, f);
            } else if constexpr (IN == IN_HEADS) {
                const int h = k/p.hd, d = k - h*p.hd;
                bf16x8(p.x + ((size_t) h*p.M + m)*p.hd + d, f);
            } else {
                float g[8];
                bf16x8(p.x + (size_t) m*2*K + k, g);
                bf16x8(p.x + (size_t) m*2*K + K + k, f);
                #pragma unroll
                for (int i = 0; i < 8; ++i) f[i] = sqrelu_mul(g[i], f[i]);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) row[k+i] = f[i];
        }
    } else {
        for (int k = tid; k < K; k += 256) {
            float v;
            if constexpr (IN == IN_ROW) {
                v = __bfloat162float(p.x[(size_t) m*K + k]);
            } else if constexpr (IN == IN_HEADS) {
                const int h = k/p.hd, d = k - h*p.hd;
                v = __bfloat162float(p.x[((size_t) h*p.M + m)*p.hd + d]);
            } else {
                v = sqrelu_mul(__bfloat162float(p.x[(size_t) m*2*K + k]), __bfloat162float(p.x[(size_t) m*2*K + K + k]));
            }
            row[k] = v;
        }
    }
    __syncthreads();
}

template <int NPT, int IN, int NORM, bool VEC>
__global__ void __launch_bounds__(256) norm_quant(NormQuant p) {
    const int m = blockIdx.x, tid = threadIdx.x, K = p.K;
    __shared__ float smem[32];
    extern __shared__ float row[];   // K floats, then K int8 for the output
    stage_row<IN, VEC>(p, m, row);
    float x[NPT];
    #pragma unroll
    for (int i = 0; i < NPT; ++i) {
        const int k = tid + i*256;
        x[i] = k < K ? row[k] : 0.0f;
    }

    // Normalized row, rounded to bf16 as the norm kernel stores it.
    if constexpr (NORM == NORM_RMS) {
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < NPT; ++i)
            if (tid + i*256 < K) ss = __fmaf_rn(x[i], x[i], ss);
        const float mean  = block_sum256(ss, smem)/(float)K;
        const float scale = rsqrtf(mean+p.eps);
        #pragma unroll
        for (int i = 0; i < NPT; ++i) {
            const int k = tid + i*256;
            if (k < K) {
                float v  = x[i]*scale;
                float wv = __bfloat162float(p.w[k]);
                x[i] = __bfloat162float(__float2bfloat16(v * wv));
            }
        }
    } else if constexpr (NORM == NORM_LN) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < NPT; ++i)
            if (tid + i*256 < K) sum += x[i];
        const float mean = block_sum256(sum, smem)/(float)K;
        float vsum = 0.0f;
        #pragma unroll
        for (int i = 0; i < NPT; ++i)
            if (tid + i*256 < K) {
                float v = x[i]-mean;
                vsum = __fmaf_rn(v, v, vsum);
            }
        const float inv_std = rsqrtf(block_sum256(vsum, smem)/(float)K+p.eps);
        #pragma unroll
        for (int i = 0; i < NPT; ++i) {
            const int k = tid + i*256;
            if (k < K) {
                float v = (x[i]-mean)*inv_std;
                float wv = __bfloat162float(p.w[k]);
                float bv = __bfloat162float(p.b[k]);
                x[i] = __bfloat162float(__float2bfloat16(__fmaf_rn(v, wv, bv)));
            }
        }
    }

    // act_quant_kernel
    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < NPT; ++i)
        if (tid + i*256 < K) local_max = fmaxf(local_max, fabsf(x[i]));
    const float mx = block_max256(local_max, smem);
    const float amax  = mx < 1e-5f ? 1e-5f : mx;
    const float scale = 127.0f/amax;
    if (tid == 0)
        p.scales[m] = scale;
    int8_t * row_out = p.q + (size_t) m*p.ld_q;
    int8_t * qs = (int8_t *) (row + K);
    #pragma unroll
    for (int i = 0; i < NPT; ++i) {
        const int k = tid + i*256;
        if (k < K) {
            float v = x[i]*scale;
            float q = nearbyintf(v);
            if (q >  127.0f)
                q =  127.0f;
            if (q < -128.0f)
                q = -128.0f;
            if constexpr (VEC) qs[k] = (int8_t)q;
            else row_out[k] = (int8_t)q;
        }
    }
    if constexpr (VEC) {
        __syncthreads();
        for (int v = tid; v < K/16; v += 256)
            *(uint4 *) (row_out + v*16) = *(const uint4 *) (qs + v*16);
    }
}

template <int IN, int NORM>
inline bool launch_norm_quant(const NormQuant & p, cudaStream_t st) {
    const int npt = (p.K + 255)/256;
    const bool vec = p.K % 16 == 0 && p.ld_q % 16 == 0 && (IN != IN_HEADS || p.hd % 8 == 0);
    const size_t smem = (size_t) p.K*4 + p.K;
#define NQ(N) { if (vec) norm_quant<N, IN, NORM, true><<<p.M, 256, smem, st>>>(p); else norm_quant<N, IN, NORM, false><<<p.M, 256, smem, st>>>(p); }
    if      (npt <= 5)  NQ(5)
    else if (npt <= 10) NQ(10)
    else if (npt <= 18) NQ(18)
    else if (npt <= 27) NQ(27)
    else return false;
#undef NQ
    return true;
}


// ---------------------------------------------------------------------------
// rope_neox_bf16_kernel on q (in place) and on k, k written to its repeated
// heads (repeat_kv_bf16_kernel). One warp per (head, position).

// As in the original's SASS: fma(a, c, -(b*si)) and fma(a, si, b*c).
__device__ __forceinline__ void rope_pair(float a, float b, float c, float si, float & o0, float & o1) {
    o0 = __fmaf_rn(a, c, -__fmul_rn(b, si));
    o1 = __fmaf_rn(a, si, __fmul_rn(b, c));
}

static __global__ void __launch_bounds__(128) rope_qk_repeat(__nv_bfloat16 * __restrict__ q, const __nv_bfloat16 * __restrict__ k,
                                                      __nv_bfloat16 * __restrict__ k_rep, const float * __restrict__ cos_tab,
                                                      const float * __restrict__ sin_tab, int n_q, int n_kv, int S, int D) {
    const int row = (int) blockIdx.x*4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int h = row / S, s = row - h*S;
    if (h >= n_q + n_kv) return;
    const int half = D/2;
    const float * c_row = cos_tab + (size_t) s*half;
    const float * s_row = sin_tab + (size_t) s*half;
    const __nv_bfloat16 * src = h < n_q ? q + ((size_t) h*S + s)*D : k + ((size_t) (h-n_q)*S + s)*D;
    const int rep = n_q/n_kv;
    for (int i = lane*2; i < half; i += 64) {
        const __nv_bfloat162 a2 = *(const __nv_bfloat162 *) (src + i);
        const __nv_bfloat162 b2 = *(const __nv_bfloat162 *) (src + i + half);
        float o[4];
        rope_pair(__bfloat162float(a2.x), __bfloat162float(b2.x), c_row[i],   s_row[i],   o[0], o[2]);
        rope_pair(__bfloat162float(a2.y), __bfloat162float(b2.y), c_row[i+1], s_row[i+1], o[1], o[3]);
        const __nv_bfloat162 lo = __floats2bfloat162_rn(o[0], o[1]), hi = __floats2bfloat162_rn(o[2], o[3]);
        if (h < n_q) {
            __nv_bfloat16 * dst = q + ((size_t) h*S + s)*D;
            *(__nv_bfloat162 *) (dst + i) = lo;
            *(__nv_bfloat162 *) (dst + i + half) = hi;
        } else {
            for (int r = 0; r < rep; ++r) {
                __nv_bfloat16 * dst = k_rep + ((size_t) ((h-n_q)*rep + r)*S + s)*D;
                *(__nv_bfloat162 *) (dst + i) = lo;
                *(__nv_bfloat162 *) (dst + i + half) = hi;
            }
        }
    }
}

// q: [n_q][S][D] in place; k: [n_kv][S][D] read only; k_rep: [n_q][S][D]. D/2 even.
inline bool rope_qk_repeat_launch(__nv_bfloat16 * q, const __nv_bfloat16 * k, __nv_bfloat16 * k_rep, const float * cos_tab,
                                  const float * sin_tab, int n_q, int n_kv, int S, int D, cudaStream_t st) {
    if (D % 4 != 0 || n_q % n_kv != 0) return false;
    const int rows = (n_q + n_kv)*S;
    rope_qk_repeat<<<(rows+3)/4, 128, 0, st>>>(q, k, k_rep, cos_tab, sin_tab, n_q, n_kv, S, D);
    return true;
}

}  // namespace bitvla_rowops
