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

// One warp per token row, four rows per CTA. Lane l owns the 64-element chunks
// c = l, l+32, ... of the row, so a chunk's butterfly (block 64 or smaller)
// runs entirely in that lane's registers: no shared memory, no barriers, and
// the two row-wide reductions (sum of squares, amax) are warp shuffles. The
// reduction tree - per-lane partial in element order, xor butterfly 16..1 -
// is the one foldquant_ref.h mirrors, so rstd is bit-identical between
// backends. amax is exact in any order.
//
// The row is read twice (three times with a fused RMSNorm): once per pass the
// value must be known for. It is L1/L2 resident, so the passes cost ALU, not
// bandwidth, which is what matters at M of a few hundred rows.

#include "fq_kernels.h"

#include <cstdlib>

namespace vla {
namespace fq {

namespace {

constexpr int CHUNK   = 64;
constexpr int WARPS   = 4;
constexpr int THREADS = WARPS * 32;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = v + __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Load a 64-float chunk, apply the pre-quant chain, return it rotated. ROT is
// a compile-time constant so every index into v[] is static and the chunk
// stays in registers (a runtime loop bound would spill it to local memory).
template <int ROT>
__device__ __forceinline__ void load_chunk(const ActArgs & a, const float * x, int64_t c0, float rstd, float * v) {
    const float4 * src = (const float4 *) (x + c0);
    #pragma unroll
    for (int i = 0; i < CHUNK / 4; ++i) {
        const float4 q = src[i];
        v[4*i] = q.x; v[4*i+1] = q.y; v[4*i+2] = q.z; v[4*i+3] = q.w;
    }
    if (a.gamma) {
        const float4 * g = (const float4 *) (a.gamma + c0);
        #pragma unroll
        for (int i = 0; i < CHUNK / 4; ++i) {
            const float4 q = g[i];
            v[4*i]   = (v[4*i]   * rstd) * q.x;
            v[4*i+1] = (v[4*i+1] * rstd) * q.y;
            v[4*i+2] = (v[4*i+2] * rstd) * q.z;
            v[4*i+3] = (v[4*i+3] * rstd) * q.w;
        }
    }
    if (a.ascale && a.fold_before) {
        const float4 * s = (const float4 *) (a.ascale + c0);
        #pragma unroll
        for (int i = 0; i < CHUNK / 4; ++i) {
            const float4 q = s[i];
            v[4*i] = v[4*i] / q.x; v[4*i+1] = v[4*i+1] / q.y; v[4*i+2] = v[4*i+2] / q.z; v[4*i+3] = v[4*i+3] / q.w;
        }
    }
    if (ROT > 1) {
        // Natural-order Sylvester butterfly on every ROT sub-block of the chunk.
        #pragma unroll
        for (int h = 1; h < ROT; h <<= 1) {
            #pragma unroll
            for (int p = 0; p < CHUNK / 2; ++p) {
                // pairs (j, j+h) with j = (p / h) * 2h + (p % h)
                const int j = ((p / h) * 2 * h) + (p % h);
                const float u = v[j], w = v[j + h];
                v[j]     = u + w;
                v[j + h] = u - w;
            }
        }
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) v[i] = v[i] * a.inv_sqrt_bs;
    }
    if (a.ascale && !a.fold_before) {
        const float4 * s = (const float4 *) (a.ascale + c0);
        #pragma unroll
        for (int i = 0; i < CHUNK / 4; ++i) {
            const float4 q = s[i];
            v[4*i] = v[4*i] / q.x; v[4*i+1] = v[4*i+1] / q.y; v[4*i+2] = v[4*i+2] / q.z; v[4*i+3] = v[4*i+3] / q.w;
        }
    }
}

template <int ABITS, int ROT>
__global__ void __launch_bounds__(THREADS) act_kernel(const ActArgs a) {
    const int     lane = threadIdx.x & 31;
    const int64_t m    = (int64_t) blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (m >= a.M) return;
    const int64_t K       = a.K;
    const int     nchunks = (int) (K / CHUNK);
    const float * x       = a.x + m * (a.x_stride ? a.x_stride : K);

    float rstd = 0.0f;
    if (a.gamma) {
        float p = 0.0f;
        for (int c = lane; c < nchunks; c += 32) {
            const float4 * src = (const float4 *) (x + (int64_t) c * CHUNK);
            #pragma unroll
            for (int i = 0; i < CHUNK / 4; ++i) {
                const float4 q = src[i];
                p = p + q.x * q.x; p = p + q.y * q.y; p = p + q.z * q.z; p = p + q.w * q.w;
            }
        }
        const float sumsq = warp_sum(p);
        rstd = 1.0f / sqrtf(sumsq / (float) K + a.eps);
    }

    float v[CHUNK];
    float mx = 0.0f;
    for (int c = lane; c < nchunks; c += 32) {
        load_chunk<ROT>(a, x, (int64_t) c * CHUNK, rstd, v);
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) mx = fmaxf(mx, fabsf(v[i]));
    }
    mx = warp_max(mx);

    const float qmax  = ABITS == 4 ? 7.0f : 127.0f;
    float       scale = (a.clip * mx) / qmax;
    if (scale < 1e-12f) scale = 1e-12f;
    // Quantize with the reciprocal, as VLA-OPT's TensorRT kernels do
    // (rmsnorm_per_row_quant_cuda.cu) and as foldquant_ref.h mirrors.
    const float inv = 1.0f / scale;

    int8_t * row = a.blob + m * a.row_bytes;
    for (int c = lane; c < nchunks; c += 32) {
        // A lane that owns a single chunk still holds it rotated in v[].
        if (nchunks > 32) load_chunk<ROT>(a, x, (int64_t) c * CHUNK, rstd, v);
        if (ABITS == 8) {
            uint32_t * dst = (uint32_t *) (row + (int64_t) c * CHUNK);
            #pragma unroll
            for (int i = 0; i < CHUNK; i += 4) {
                uint32_t packed = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float q = rintf(v[i + j] * inv);
                    q = fminf(qmax, fmaxf(-qmax, q));
                    packed |= ((uint32_t) (int) q & 0xFFu) << (8 * j);
                }
                dst[i / 4] = packed;
            }
        } else {
            uint32_t * dst = (uint32_t *) (row + (int64_t) c * (CHUNK / 2));
            #pragma unroll
            for (int i = 0; i < CHUNK; i += 8) {
                uint32_t packed = 0;
                #pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    float q0 = rintf(v[i + j] * inv), q1 = rintf(v[i + j + 1] * inv);
                    q0 = fminf(qmax, fmaxf(-qmax, q0));
                    q1 = fminf(qmax, fmaxf(-qmax, q1));
                    const uint32_t b = ((uint32_t) (int) q0 & 0xFu) | (((uint32_t) (int) q1 & 0xFu) << 4);
                    packed |= b << (4 * j);
                }
                dst[i / 8] = packed;
            }
        }
    }
    if (lane == 0) *(float *) (row + (ABITS == 8 ? K : K / 2)) = scale;
}

// CTA-per-row variant for the short rows a VLA quantizes (41 action tokens,
// a few hundred LLM tokens): one warp per row leaves a 16-SM GPU nearly idle
// and each launch latency-bound at ~50 us. Here RW warps share a row; a warp
// takes chunks c = warp, warp + RW, ... and its lanes own the element pair
// (2*lane, 2*lane + 1) of each chunk, so loads and stores are coalesced and
// the per-row chain is a handful of shuffles. The 64-wide butterfly runs
// stage h = 1 inside the lane and the other stages across lanes: every stage
// forms exactly the (a + b, a - b) pairs fwht_row forms, so the values are
// bit-identical to the reference. The RMSNorm sum of squares keeps the
// reference's lane-partial order (warp 0 runs the warp-per-row loop).
constexpr int RW   = 8;          // warps per row
constexpr int RT   = RW * 32;
constexpr int MAXC = 16;         // chunks per warp held in registers: K <= RW * MAXC * 64

template <int ROT>
__device__ __forceinline__ void rot_pair(float & v0, float & v1, int lane, float inv_sqrt_bs) {
    if (ROT > 1) {
        { const float u = v0, w = v1; v0 = u + w; v1 = u - w; }   // h = 1: pair (2l, 2l+1)
        #pragma unroll
        for (int h = 2; h < ROT; h <<= 1) {
            const int   half  = h >> 1;                               // partner lane distance
            const float o0    = __shfl_xor_sync(0xffffffffu, v0, half);
            const float o1    = __shfl_xor_sync(0xffffffffu, v1, half);
            const bool  upper = (lane & half) != 0;                   // this lane holds j + h
            v0 = upper ? (o0 - v0) : (v0 + o0);
            v1 = upper ? (o1 - v1) : (v1 + o1);
        }
        v0 = v0 * inv_sqrt_bs;
        v1 = v1 * inv_sqrt_bs;
    }
}

template <int ABITS, int ROT>
__global__ void __launch_bounds__(RT) act_row_kernel(const ActArgs a) {
    __shared__ float red[RW];
    __shared__ float s_rstd;
    const int     tid     = threadIdx.x;
    const int     warp    = tid >> 5, lane = tid & 31;
    const int64_t m       = blockIdx.x;
    const int64_t K       = a.K;
    const int     nchunks = (int) (K / CHUNK);
    const float * x       = a.x + m * (a.x_stride ? a.x_stride : K);

    if (a.gamma) {
        if (warp == 0) {
            float p = 0.0f;
            for (int c = lane; c < nchunks; c += 32) {
                const float4 * src = (const float4 *) (x + (int64_t) c * CHUNK);
                #pragma unroll
                for (int i = 0; i < CHUNK / 4; ++i) {
                    const float4 q = src[i];
                    p = p + q.x * q.x; p = p + q.y * q.y; p = p + q.z * q.z; p = p + q.w * q.w;
                }
            }
            const float sumsq = warp_sum(p);
            if (lane == 0) s_rstd = 1.0f / sqrtf(sumsq / (float) K + a.eps);
        }
        __syncthreads();
    }
    const float rstd = a.gamma ? s_rstd : 0.0f;

    float v0[MAXC], v1[MAXC];
    float mx = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = warp + i * RW;
        if (c < nchunks) {
            const int64_t k0 = (int64_t) c * CHUNK + 2 * lane;
            const float2  q  = *(const float2 *) (x + k0);
            float e0 = q.x, e1 = q.y;
            if (a.gamma) {
                const float2 g = *(const float2 *) (a.gamma + k0);
                e0 = (e0 * rstd) * g.x;
                e1 = (e1 * rstd) * g.y;
            }
            if (a.ascale && a.fold_before) {
                const float2 s = *(const float2 *) (a.ascale + k0);
                e0 = e0 / s.x; e1 = e1 / s.y;
            }
            rot_pair<ROT>(e0, e1, lane, a.inv_sqrt_bs);
            if (a.ascale && !a.fold_before) {
                const float2 s = *(const float2 *) (a.ascale + k0);
                e0 = e0 / s.x; e1 = e1 / s.y;
            }
            v0[i] = e0; v1[i] = e1;
            mx = fmaxf(mx, fmaxf(fabsf(e0), fabsf(e1)));
        }
    }
    mx = warp_max(mx);
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    mx = red[0];
    #pragma unroll
    for (int w = 1; w < RW; ++w) mx = fmaxf(mx, red[w]);

    const float qmax  = ABITS == 4 ? 7.0f : 127.0f;
    float       scale = (a.clip * mx) / qmax;
    if (scale < 1e-12f) scale = 1e-12f;
    const float inv = 1.0f / scale;
    int8_t * row = a.blob + m * a.row_bytes;
    #pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = warp + i * RW;
        if (c < nchunks) {
            float q0 = rintf(v0[i] * inv), q1 = rintf(v1[i] * inv);
            q0 = fminf(qmax, fmaxf(-qmax, q0));
            q1 = fminf(qmax, fmaxf(-qmax, q1));
            if (ABITS == 8) {
                const uint16_t packed = (uint16_t) (((uint32_t) (int) q0 & 0xFFu) | (((uint32_t) (int) q1 & 0xFFu) << 8));
                *(uint16_t *) (row + (int64_t) c * CHUNK + 2 * lane) = packed;
            } else {
                row[(int64_t) c * (CHUNK / 2) + lane] = (int8_t) (((uint32_t) (int) q0 & 0xFu) | (((uint32_t) (int) q1 & 0xFu) << 4));
            }
        }
    }
    if (tid == 0) *(float *) (row + (ABITS == 8 ? K : K / 2)) = scale;
}

}  // namespace

template <int ABITS>
static cudaError_t launch_rot_row(const ActArgs & a, cudaStream_t stream) {
    const unsigned grid = (unsigned) a.M;
    switch (a.rot_block) {
        case 64: act_row_kernel<ABITS, 64><<<grid, RT, 0, stream>>>(a); break;
        case 32: act_row_kernel<ABITS, 32><<<grid, RT, 0, stream>>>(a); break;
        case 16: act_row_kernel<ABITS, 16><<<grid, RT, 0, stream>>>(a); break;
        case 8:  act_row_kernel<ABITS, 8 ><<<grid, RT, 0, stream>>>(a); break;
        case 4:  act_row_kernel<ABITS, 4 ><<<grid, RT, 0, stream>>>(a); break;
        case 2:  act_row_kernel<ABITS, 2 ><<<grid, RT, 0, stream>>>(a); break;
        case 1:
        case 0:  act_row_kernel<ABITS, 1 ><<<grid, RT, 0, stream>>>(a); break;
        default: return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

template <int ABITS>
static cudaError_t launch_rot(const ActArgs & a, unsigned grid, cudaStream_t stream) {
    switch (a.rot_block) {
        case 64: act_kernel<ABITS, 64><<<grid, THREADS, 0, stream>>>(a); break;
        case 32: act_kernel<ABITS, 32><<<grid, THREADS, 0, stream>>>(a); break;
        case 16: act_kernel<ABITS, 16><<<grid, THREADS, 0, stream>>>(a); break;
        case 8:  act_kernel<ABITS, 8 ><<<grid, THREADS, 0, stream>>>(a); break;
        case 4:  act_kernel<ABITS, 4 ><<<grid, THREADS, 0, stream>>>(a); break;
        case 2:  act_kernel<ABITS, 2 ><<<grid, THREADS, 0, stream>>>(a); break;
        case 1:
        case 0:  act_kernel<ABITS, 1 ><<<grid, THREADS, 0, stream>>>(a); break;
        default: return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

cudaError_t launch_act(const ActArgs & a, cudaStream_t stream) {
    if (a.M <= 0) return cudaSuccess;
    if (a.K % CHUNK != 0 || a.rot_block > CHUNK) return cudaErrorNotSupported;
    if (a.x_stride && (a.x_stride % 4 != 0)) return cudaErrorNotSupported;   // float4 loads need 16-byte rows
    // VLA_FQ_ACT_WARP=1 keeps the warp-per-row kernel (A/B and fallback).
    static const bool warp_only = [] { const char * e = std::getenv("VLA_FQ_ACT_WARP"); return e && *e && *e != '0'; }();
    if (!warp_only && a.K / CHUNK <= RW * MAXC) {
        if (a.abits == 8) return launch_rot_row<8>(a, stream);
        if (a.abits == 4) return launch_rot_row<4>(a, stream);
    }
    const unsigned grid = (unsigned) ((a.M + WARPS - 1) / WARPS);
    if (a.abits == 8) return launch_rot<8>(a, grid, stream);
    if (a.abits == 4) return launch_rot<4>(a, grid, stream);
    return cudaErrorNotSupported;
}

}  // namespace fq
}  // namespace vla
