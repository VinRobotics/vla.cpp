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

// CUDA kernels for the fused nodes of layers/fused_ops.h.
//
// Bit-exactness against the unfused ggml chain is the contract:
//  - every place ggml rounds to F32 between two ops is rounded here too, with
//    __fmul_rn / __fadd_rn so the compiler cannot contract it into an FMA;
//  - the norm reproduces ggml-cuda's norm_f32 (same block size, same strided
//    per-thread sums, same xor-shuffle and shared-memory reduction tree);
//  - GELU is ggml-cuda's tanh expression, compiled with the same default
//    FMA contraction as ggml-cuda itself.

#include "ggml.h"
#include "cuda/vla_cuda_ext.h"
#include "layers/fused_ops.h"
#include "cuda/gemm_skinny.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int WARP = 32;

struct custom_op_params_mirror {
    void * fun;
    int    n_tasks;
    void * userdata;
};

const vla::FusedSpec * spec_of(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_CUSTOM) return nullptr;
    custom_op_params_mirror p;
    std::memcpy(&p, dst->op_params, sizeof(p));
    if (!p.userdata) return nullptr;
    uint32_t m = 0;
    std::memcpy(&m, p.userdata, sizeof(m));
    return m == vla::FUSED_MAGIC ? (const vla::FusedSpec *) p.userdata : nullptr;
}

__device__ __forceinline__ float gelu_ggml(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ void st_out(float * o, int64_t i, float v)         { o[i] = v; }
__device__ __forceinline__ void st_out(__nv_bfloat16 * o, int64_t i, float v) { o[i] = __float2bfloat16(v); }
__device__ __forceinline__ void st_out(half * o, int64_t i, float v)          { o[i] = __float2half(v); }

// Packed output slot of input part p for a node writing part_mask (0: all).
__device__ __forceinline__ int packed_part(int part_mask, int p) {
    return part_mask ? __popc(part_mask & ((1 << p)-1)) : p;
}

// One block per token row; threads walk the row's columns, so reads are
// coalesced and the Q/K (and untransposed V) writes are contiguous per head.
// Parts written transposed are skipped here and done by k_qkv_vt.
template <typename T>
__global__ void k_qkv_rows(const float * __restrict__ y, const float * __restrict__ b,
                           const float * __restrict__ cs, const float * __restrict__ sn, T * __restrict__ out,
                           int n_parts, int hd, int heads, int S, int vt_mask, int rope_mask, int part_mask) {
    const int D = hd*heads, W = D*n_parts, half = hd/2;
    const int tok = blockIdx.x, t = tok%S, v = tok/S;
    const int64_t part = (int64_t) D*gridDim.x;   // D x every token
    const float * yr = y+(int64_t) tok*W;
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        const int p = c/D;
        if (((vt_mask >> p) & 1) || (part_mask && !((part_mask >> p) & 1))) continue;
        const int hc = c-p*D, h = hc/hd, d = hc-h*hd;
        float val = b ? __fadd_rn(yr[c], b[c]) : yr[c];
        if ((rope_mask >> p) & 1) {
            const int   pc = d < half ? c+half : c-half;
            const float pv = b ? __fadd_rn(yr[pc], b[pc]) : yr[pc];
            const float rot = d < half ? -pv : pv;
            val = __fadd_rn(__fmul_rn(val, cs[d+hd*t]), __fmul_rn(rot, sn[d+hd*t]));
        }
        st_out(out, (int64_t) packed_part(part_mask, p)*part+d+(int64_t) hd*(t+(int64_t) S*(h+(int64_t) heads*v)), val);
    }
}

// Transposed part p (no RoPE): a 32x32 tile of (token, column) through shared
// memory, so both the read along columns and the write along tokens coalesce.
template <typename T>
__global__ void k_qkv_vt(const float * __restrict__ y, const float * __restrict__ b, T * __restrict__ out,
                         int n_parts, int p, int hd, int heads, int S, int n_tok, int slot) {
    __shared__ float tile[32][33];
    const int D = hd*heads, W = D*n_parts;
    const int64_t part = (int64_t) D*n_tok;
    const int c0 = blockIdx.x*32, t0 = blockIdx.y*32;
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        const int tok = t0+k, c = c0+threadIdx.x;
        if (tok < n_tok && c < D) {
            const int col = p*D+c;
            tile[k][threadIdx.x] = b ? __fadd_rn(y[(int64_t) tok*W+col], b[col]) : y[(int64_t) tok*W+col];
        }
    }
    __syncthreads();
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        const int c = c0+k, tok = t0+threadIdx.x;
        if (tok < n_tok && c < D) {
            const int h = c/hd, d = c-h*hd, t = tok%S, v = tok/S;
            st_out(out, (int64_t) slot*part+t+(int64_t) S*(d+(int64_t) hd*(h+(int64_t) heads*v)), tile[threadIdx.x][k]);
        }
    }
}

__global__ void k_bias_res(const float * __restrict__ y, const float * __restrict__ b, const float * __restrict__ r,
                           float * __restrict__ out, int64_t n, int N) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = __fadd_rn(r[i], __fadd_rn(y[i], b[i%N]));
}


template <typename T>
__global__ void k_bias_gelu(const float * __restrict__ y, const float * __restrict__ b, T * __restrict__ out,
                            int64_t n, int N) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) st_out(out, i, gelu_ggml(__fadd_rn(y[i], b[i%N])));
}

__device__ __forceinline__ float2 warp_sum2(float2 a) {
#pragma unroll
    for (int off = WARP/2; off > 0; off >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, off, WARP);
        a.y += __shfl_xor_sync(0xffffffff, a.y, off, WARP);
    }
    return a;
}
__device__ __forceinline__ float  warp_sum(float a) {
#pragma unroll
    for (int off = WARP/2; off > 0; off >>= 1)
        a += __shfl_xor_sync(0xffffffff, a, off, WARP);
    return a;
}

__device__ __forceinline__ float  shfl_xor(float a, int off)  { return __shfl_xor_sync(0xffffffff, a, off, WARP); }
__device__ __forceinline__ float  shfl_idx(float a, int src)  { return __shfl_sync(0xffffffff, a, src, WARP); }
__device__ __forceinline__ float  add_rn(float a, float b)    { return a + b; }

// NW xor butterflies warp_sum(p[0..NW-1]) at once (NW a power of two <= 32),
// with the same pairwise additions: 31 shuffles instead of 5*NW. Returns
// warp_sum(p[lane]) in lanes < NW, zero elsewhere (the input of ggml's second
// reduction stage).
template <int NW, typename T>
__device__ __forceinline__ T warp_sums_t(T (&p)[NW], int lane, T zero) {
    static_assert(NW >= 1 && NW <= WARP && (NW & (NW-1)) == 0, "NW: a power of two <= 32");
#pragma unroll
    for (int st = 0; st < 5; ++st) {
        const int off = (WARP/2) >> st, n = NW >> st;   // n: vectors held before this step
        if (n > 1) {
            const bool hi = lane & off;
#pragma unroll
            for (int i = 0; i < n/2; ++i) {
                const T keep = hi ? p[i+n/2] : p[i];
                const T send = hi ? p[i] : p[i+n/2];
                p[i] = add_rn(keep, shfl_xor(send, off));
            }
        } else {
            p[0] = add_rn(p[0], shfl_xor(p[0], off));
        }
    }
    // lane l now holds vector l/(32/NW)
    const T r = shfl_idx(p[0], (lane%NW)*(WARP/NW));
    return lane < NW ? r : zero;
}

// ggml-cuda norm_f32, then the affine (w, b), the modulation (sc, sh), or nothing.
enum LnMode { LN_AFFINE, LN_MOD, LN_PLAIN };

// ggml's norm block reduction (BS threads) with one warp per row: lane l holds
// the strided partial sum of thread 32*w + l for every warp w, then the same
// two butterfly stages run. op adds one element as ggml's statement does.
template <int BS, typename T, typename Op>
__device__ __forceinline__ T emul_block_sum(int ncols, int lane, T zero, Op op) {
    if (BS == WARP) {
        T acc = zero;
        for (int col = lane; col < ncols; col += WARP) op(acc, col);
        return warp_sum(acc);
    }
    T mine = zero;
    for (int w = 0; w < BS/WARP; ++w) {
        T acc = zero;
        for (int col = w*WARP+lane; col < ncols; col += BS) op(acc, col);
        acc = warp_sum(acc);
        if (lane == w) mine = acc;
    }
    return warp_sum(mine);
}

// The epilogue of an emulated rms_norm row: MODE 0 is ggml rms_norm_f32<BS,
// do_multiply> (scale * x * w); MODE 1 is a plain rms_norm followed by ggml's
// adaRMS ops, n + n*sc then + sh, each rounded as its own kernel would.
template <int MODE>
__device__ __forceinline__ float rms_epilogue(float scale, float xi, const float * p1, const float * p2, int col) {
    if (MODE == 0) return scale * xi * p1[col];
    const float n = scale * xi;
    return __fadd_rn(__fadd_rn(n, __fmul_rn(n, p1[col])), p2[col]);
}

// ggml rms_norm_f32<BS, ...> with one warp per row, then rms_epilogue.
template <int BS, typename T, int MODE = 0>
__global__ void k_rms_mul_warp(const float * __restrict__ x, const float * __restrict__ w, T * __restrict__ dst,
                               int ncols, int nrows, float eps, const float * __restrict__ p2 = nullptr) {
    const int row = blockIdx.x*(blockDim.x/WARP)+threadIdx.x/WARP, lane = threadIdx.x%WARP;
    if (row >= nrows) return;
    x   += (int64_t) row*ncols;
    dst += (int64_t) row*ncols;
    const float tmp = emul_block_sum<BS>(ncols, lane, 0.0f, [&](float & acc, int col) {
        const float xi = x[col];
        acc += xi * xi;
    });
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = lane; col < ncols; col += WARP)
        st_out(dst, col, rms_epilogue<MODE>(scale, x[col], w, p2, col));
}

// Same arithmetic as the host patchify: u8/255 and (x - mean)/std, IEEE division.
template <typename P>
__global__ void k_patchify(const P * __restrict__ img, float * __restrict__ out, int side, int ps, int tps, int m,
                           float m0, float m1, float m2, float s0, float s1, float s2, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    const int gw = side/ps, S = gw*gw, pf = 3*tps*ps*ps, pp = ps*ps;
    const int     k  = (int) (i%pf);
    const int64_t sv = i/pf;
    const int si = (int) (sv%S), v = (int) (sv/S);
    const int ch = k/(tps*pp), rem = k%pp, ph = rem/ps, pw = rem%ps;
    int t = si; const int wj = t%m; t /= m; const int wi = t%m; t /= m;
    const int bc = t%(gw/m), br = t/(gw/m), row = br*m+wi, col = bc*m+wj;
    const int64_t px = (((int64_t) v*side+row*ps+ph)*side+col*ps+pw)*3+ch;
    float raw;
    if constexpr (sizeof(P) == 1) raw = __fdiv_rn((float) (uint8_t) img[px], 255.0f);
    else                          raw = img[px];
    const float mean = ch == 0 ? m0 : ch == 1 ? m1 : m2, sd = ch == 0 ? s0 : ch == 1 ? s1 : s2;
    out[i] = __fdiv_rn(__fsub_rn(raw, mean), sd);
}

// ggml rms_norm_f32<256, mul> per (token, head) row of hd, then rope_multi
// (IMROPE, no YaRN) on the registers, written in the attention layout.
// One warp per row; lane l owns the rotated pairs (l, l+hd/2), (l+32, l+32+hd/2), ...
template <typename T>
__global__ void k_qk_norm_rope(const float * __restrict__ x, const float * __restrict__ w, const int32_t * __restrict__ pos,
                               T * __restrict__ dst, int hd, int heads, int ntok, float eps, float theta_scale,
                               int s0, int s1, int s2, int s3) {
    const int row = blockIdx.x*(blockDim.x/WARP)+threadIdx.x/WARP, lane = threadIdx.x%WARP;
    if (row >= heads*ntok) return;
    const int h = row%heads, t = row/heads, half = hd/2;
    const float * xr = x+(int64_t) row*hd;
    const float tmp = emul_block_sum<256>(hd, lane, 0.0f, [&](float & acc, int col) {
        const float xi = xr[col];
        acc += xi * xi;
    });
    const float mean = tmp / hd;
    const float scale = rsqrtf(mean + eps);
    const int sect_dims = s0+s1+s2+s3;
    T * o = dst+(int64_t) hd*(t+(int64_t) ntok*h);
    for (int j = lane; j < half; j += WARP) {
        const float x0 = scale * xr[j] * w[j];
        const float x1 = scale * xr[j+half] * w[j+half];
        const int iw = 2*j, sector = (iw / 2) % sect_dims;
        float theta_base = 0.0;
        if (sector % 3 == 1 && sector < 3 * s1) {
            theta_base = pos[t + ntok * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * s2) {
            theta_base = pos[t + ntok * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * s0) {
            theta_base = pos[t] * powf(theta_scale, iw / 2.0f);
        } else {
            theta_base = pos[t + ntok * 3] * powf(theta_scale, iw / 2.0f);
        }
        // rope_yarn with freq_scale 1, ext_factor 0, mscale 1, freq_factor 1
        const float theta = 1.0f * (theta_base / 1.0f);
        const float cos_theta = cosf(theta) * 1.0f;
        const float sin_theta = sinf(theta) * 1.0f;
        // The contraction ggml-cuda's rope_multi compiles to (checked in its SASS):
        // fma(x0, cos, -(x1*sin)) and fma(x0, sin, x1*cos).
        st_out(o, j,      __fmaf_rn(x0, cos_theta, -__fmul_rn(x1, sin_theta)));
        st_out(o, j+half, __fmaf_rn(x0, sin_theta,  __fmul_rn(x1, cos_theta)));
    }
}

// k_rms_mul_warp with the row in registers (see k_ln_reg); BS is the block
// size ggml launches (1024 for ncols >= 1024, else 256).
template <int NPL, int BS, typename T, int MODE = 0>
__global__ void k_rms_mul_reg(const float * __restrict__ x, const float * __restrict__ w, T * __restrict__ dst,
                              int ncols, int nrows, float eps, const float * __restrict__ p2 = nullptr) {
    const int row = blockIdx.x*(blockDim.x/WARP)+threadIdx.x/WARP, lane = threadIdx.x%WARP;
    if (row >= nrows) return;
    x   += (int64_t) row*ncols;
    dst += (int64_t) row*ncols;
    float v[NPL];
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int col = j*WARP+lane;
        v[j] = col < ncols ? x[col] : 0.0f;
    }
    constexpr int NW = BS/WARP;
    float p[NW];
#pragma unroll
    for (int vw = 0; vw < NW; ++vw) {
        float tmp = 0.0f;
#pragma unroll
        for (int j = vw; j < NPL; j += NW) {
            if (j*WARP+lane < ncols) {
                const float xi = v[j];
                tmp += xi * xi;
            }
        }
        p[vw] = tmp;
    }
    const float sum = BS > WARP ? warp_sum(warp_sums_t<NW>(p, lane, 0.0f)) : warp_sum(p[0]);
    const float mean = sum / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int col = j*WARP+lane;
        if (col < ncols) st_out(dst, col, rms_epilogue<MODE>(scale, v[j], w, p2, col));
    }
}

__device__ __forceinline__ float silu_ggml(float x) {
    return x / (1.0f + expf(-x));
}

template <typename T>
__global__ void k_silu_mul(const float * __restrict__ g, const float * __restrict__ u, T * __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) st_out(out, i, __fmul_rn(silu_ggml(g[i]), u[i]));
}

// ggml unary_gated_op_kernel with op_gelu: gelu(g) * u.
template <typename T>
__global__ void k_gelu_mul(const float * __restrict__ g, const float * __restrict__ u, T * __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) st_out(out, i, __fmul_rn(gelu_ggml(g[i]), u[i]));
}

// IEEE single ops without flush-to-zero. This file builds with -use_fast_math
// (ftz) to match ggml, while the FoldQuant act chain rounds like
// src/kernels/foldquant (-fmad=false, no ftz), so its ops are spelled out.
__device__ __forceinline__ float p_add(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float p_sub(float a, float b) { float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float p_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float p_div(float a, float b) { float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float p_max(float a, float b) { float r; asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float p_min(float a, float b) { float r; asm("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float p_abs(float a) { float r; asm("abs.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float p_rint(float a) { float r; asm("cvt.rni.f32.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }

// The activation as k_gelu_mul / k_silu_mul / ggml_gelu compute it (GLU: a
// FusedGlu), then fq_prologue.cu's act_row_kernel on it (no RMSNorm): one CTA
// of NW warps per row, lane pairs (2l, 2l+1) of each 64-wide chunk, the same
// butterfly pairs and the same amax and rounding.
template <int GLU, int ABITS, int ROT, int NW>
__global__ void __launch_bounds__(NW * 32) k_glu_act(const float * __restrict__ g, const float * __restrict__ u,
                                                          const float * __restrict__ ascale, int8_t * __restrict__ blob,
                                                          int K, int64_t row_bytes, bool fold_before, float clip,
                                                          float inv_sqrt_bs) {
    constexpr int MAXC = 16;
    __shared__ float red[NW];
    const int     warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t m    = blockIdx.x;
    const int     nchunks = K / 64;
    float v0[MAXC], v1[MAXC];
    float mx = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = warp + i * NW;
        if (c < nchunks) {
            const int64_t k0 = (int64_t) c * 64 + 2 * lane;
            const float2  gg = *(const float2 *) (g + m * K + k0);
            float e0, e1;
            if (GLU == (int) vla::FusedGlu::Gelu) {
                e0 = gelu_ggml(gg.x);
                e1 = gelu_ggml(gg.y);
            } else {
                const float2 uu = *(const float2 *) (u + m * K + k0);
                const bool   si = GLU == (int) vla::FusedGlu::SiluMul;
                e0 = __fmul_rn(si ? silu_ggml(gg.x) : gelu_ggml(gg.x), uu.x);
                e1 = __fmul_rn(si ? silu_ggml(gg.y) : gelu_ggml(gg.y), uu.y);
            }
            float2 s = make_float2(1.0f, 1.0f);
            if (ascale) s = *(const float2 *) (ascale + k0);
            if (ascale && fold_before) { e0 = p_div(e0, s.x); e1 = p_div(e1, s.y); }
            if (ROT > 1) {
                { const float a = e0, b = e1; e0 = p_add(a, b); e1 = p_sub(a, b); }
                #pragma unroll
                for (int h = 2; h < ROT; h <<= 1) {
                    const int   half  = h >> 1;
                    const float o0    = __shfl_xor_sync(0xffffffffu, e0, half);
                    const float o1    = __shfl_xor_sync(0xffffffffu, e1, half);
                    const bool  upper = (lane & half) != 0;
                    e0 = upper ? p_sub(o0, e0) : p_add(e0, o0);
                    e1 = upper ? p_sub(o1, e1) : p_add(e1, o1);
                }
                e0 = p_mul(e0, inv_sqrt_bs);
                e1 = p_mul(e1, inv_sqrt_bs);
            }
            if (ascale && !fold_before) { e0 = p_div(e0, s.x); e1 = p_div(e1, s.y); }
            v0[i] = e0; v1[i] = e1;
            mx = p_max(mx, p_max(p_abs(e0), p_abs(e1)));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = p_max(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    mx = red[0];
    #pragma unroll
    for (int w = 1; w < NW; ++w) mx = p_max(mx, red[w]);

    const float qmax  = ABITS == 4 ? 7.0f : 127.0f;
    float       scale = p_div(p_mul(clip, mx), qmax);
    if (scale < 1e-12f) scale = 1e-12f;
    const float inv = p_div(1.0f, scale);
    int8_t * row = blob + m * row_bytes;
    #pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = warp + i * NW;
        if (c < nchunks) {
            const float q0 = p_min(qmax, p_max(-qmax, p_rint(p_mul(v0[i], inv))));
            const float q1 = p_min(qmax, p_max(-qmax, p_rint(p_mul(v1[i], inv))));
            if (ABITS == 8) {
                const uint16_t packed = (uint16_t) (((uint32_t) (int) q0 & 0xFFu) | (((uint32_t) (int) q1 & 0xFFu) << 8));
                *(uint16_t *) (row + (int64_t) c * 64 + 2 * lane) = packed;
            } else {
                row[(int64_t) c * 32 + lane] = (int8_t) (((uint32_t) (int) q0 & 0xFu) | (((uint32_t) (int) q1 & 0xFu) << 4));
            }
        }
    }
    if (threadIdx.x == 0) *(float *) (row + (ABITS == 8 ? K : K / 2)) = scale;
}

template <int GLU, int ABITS, int NW>
bool launch_glu_act_nw(const vla::FusedSpec & s, const ggml_tensor * dst, cudaStream_t st) {
    const ggml_tensor * g = dst->src[0];
    const float * u  = dst->src[1] ? (const float *) dst->src[1]->data : nullptr;
    const float * as = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;
    const int K = (int) g->ne[0];
    const unsigned rows = (unsigned) dst->ne[1];
    auto go = [&](auto kern) {
        kern<<<rows, NW * 32, 0, st>>>((const float *) g->data, u, as,
                                      (int8_t *) dst->data, K, (int64_t) dst->ne[0], s.fold_before != 0, s.clip,
                                      s.inv_sqrt_bs);
        return true;
    };
    switch (s.rot_block) {
        case 64: return go(k_glu_act<GLU, ABITS, 64, NW>);
        case 32: return go(k_glu_act<GLU, ABITS, 32, NW>);
        case 16: return go(k_glu_act<GLU, ABITS, 16, NW>);
        case 8:  return go(k_glu_act<GLU, ABITS, 8,  NW>);
        case 4:  return go(k_glu_act<GLU, ABITS, 4,  NW>);
        case 2:  return go(k_glu_act<GLU, ABITS, 2,  NW>);
        case 1:
        case 0:  return go(k_glu_act<GLU, ABITS, 1,  NW>);
        default: return false;
    }
}

template <int GLU>
bool launch_glu_act_g(const vla::FusedSpec & s, const ggml_tensor * dst, cudaStream_t st) {
    const bool wide = dst->src[0]->ne[0] > 8 * 16 * 64;
    if (s.abits == 8) return wide ? launch_glu_act_nw<GLU, 8, 16>(s, dst, st) : launch_glu_act_nw<GLU, 8, 8>(s, dst, st);
    return wide ? launch_glu_act_nw<GLU, 4, 16>(s, dst, st) : launch_glu_act_nw<GLU, 4, 8>(s, dst, st);
}

bool launch_glu_act(const vla::FusedSpec & s, const ggml_tensor * dst, cudaStream_t st) {
    const ggml_tensor * g = dst->src[0], * u = dst->src[1];
    const int64_t K = g->ne[0];
    if (dst->type != GGML_TYPE_I8 || K % 64 != 0 || K > 16 * 16 * 64 || (s.abits != 8 && s.abits != 4) ||
        (s.glu == vla::FusedGlu::Gelu) != !u || (u && !ggml_are_same_shape(g, u)) || ggml_nrows(g) != dst->ne[1] ||
        (dst->src[2] && ggml_nelements(dst->src[2]) != K))
        return false;
    switch (s.glu) {
        case vla::FusedGlu::GeluMul: return launch_glu_act_g<(int) vla::FusedGlu::GeluMul>(s, dst, st);
        case vla::FusedGlu::SiluMul: return launch_glu_act_g<(int) vla::FusedGlu::SiluMul>(s, dst, st);
        case vla::FusedGlu::Gelu:    return launch_glu_act_g<(int) vla::FusedGlu::Gelu>(s, dst, st);
    }
    return false;
}

// k_glu_act's act chain on the values load(k0, e0, e1) gives for columns k0
// and k0 + 1. k_glu_act keeps its own copy: built on this function it ran its
// 528 x 16384 rows 40% slower on Orin. test_fused_cuda_ops checks both
// against fq_act.
template <int ABITS, int ROT, int NW, typename Load>
__device__ __forceinline__ void act_row_pairs(Load load, const float * __restrict__ ascale, int8_t * __restrict__ row,
                                              int K, bool fold_before, float clip, float inv_sqrt_bs) {
    constexpr int MAXC = 16;
    __shared__ float red[NW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nchunks = K / 64;
    float v0[MAXC], v1[MAXC];
    float mx = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = warp + i * NW;
        if (c < nchunks) {
            const int64_t k0 = (int64_t) c * 64 + 2 * lane;
            float e0, e1;
            load(k0, e0, e1);
            float2 s = make_float2(1.0f, 1.0f);
            if (ascale) s = *(const float2 *) (ascale + k0);
            if (ascale && fold_before) { e0 = p_div(e0, s.x); e1 = p_div(e1, s.y); }
            if (ROT > 1) {
                { const float a = e0, b = e1; e0 = p_add(a, b); e1 = p_sub(a, b); }
                #pragma unroll
                for (int h = 2; h < ROT; h <<= 1) {
                    const int   half  = h >> 1;
                    const float o0    = __shfl_xor_sync(0xffffffffu, e0, half);
                    const float o1    = __shfl_xor_sync(0xffffffffu, e1, half);
                    const bool  upper = (lane & half) != 0;
                    e0 = upper ? p_sub(o0, e0) : p_add(e0, o0);
                    e1 = upper ? p_sub(o1, e1) : p_add(e1, o1);
                }
                e0 = p_mul(e0, inv_sqrt_bs);
                e1 = p_mul(e1, inv_sqrt_bs);
            }
            if (ascale && !fold_before) { e0 = p_div(e0, s.x); e1 = p_div(e1, s.y); }
            v0[i] = e0; v1[i] = e1;
            mx = p_max(mx, p_max(p_abs(e0), p_abs(e1)));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = p_max(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    mx = red[0];
    #pragma unroll
    for (int w = 1; w < NW; ++w) mx = p_max(mx, red[w]);

    const float qmax  = ABITS == 4 ? 7.0f : 127.0f;
    float       scale = p_div(p_mul(clip, mx), qmax);
    if (scale < 1e-12f) scale = 1e-12f;
    const float inv = p_div(1.0f, scale);
    #pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = warp + i * NW;
        if (c < nchunks) {
            const float q0 = p_min(qmax, p_max(-qmax, p_rint(p_mul(v0[i], inv))));
            const float q1 = p_min(qmax, p_max(-qmax, p_rint(p_mul(v1[i], inv))));
            if (ABITS == 8) {
                const uint16_t packed = (uint16_t) (((uint32_t) (int) q0 & 0xFFu) | (((uint32_t) (int) q1 & 0xFFu) << 8));
                *(uint16_t *) (row + (int64_t) c * 64 + 2 * lane) = packed;
            } else {
                row[(int64_t) c * 32 + lane] = (int8_t) (((uint32_t) (int) q0 & 0xFu) | (((uint32_t) (int) q1 & 0xFu) << 4));
            }
        }
    }
    if (threadIdx.x == 0) *(float *) (row + (ABITS == 8 ? K : K / 2)) = scale;
}

// A norm as k_ln_reg / k_rms_mul_reg compute it for rows of 1024 columns or
// more (ggml's 1024-thread block), then act_row_pairs on the normed values.
// ggml thread t sums columns t, t + 1024, ...; warp w here reduces ggml warps
// w, w + NW, ... with the same butterfly, and warp 0 runs the second stage
// over the 32 warp sums.
template <vla::FusedKind NORM, int ABITS, int ROT, int NW>
__global__ void __launch_bounds__(NW * 32) k_norm_act(const float * __restrict__ x, const float * __restrict__ p1,
                                                       const float * __restrict__ p2, const float * __restrict__ ascale,
                                                       int8_t * __restrict__ blob, int K, int64_t row_bytes, float eps,
                                                       bool fold_before, float clip, float inv_sqrt_bs) {
    constexpr bool RMS = NORM == vla::FusedKind::RmsModulate;
    __shared__ float2 part[WARP];
    __shared__ float2 stat;
    const int     warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t m    = blockIdx.x;
    x += m * K;
    for (int w = warp; w < WARP; w += NW) {
        float2 acc = make_float2(0.0f, 0.0f);
        for (int col = w*WARP+lane; col < K; col += WARP*WARP) {
            const float xi = x[col];
            if (RMS) {
                acc.x += xi * xi;
            } else {
                acc.x += xi;
                acc.y += xi * xi;
            }
        }
        if (RMS) acc.x = warp_sum(acc.x);
        else     acc   = warp_sum2(acc);
        if (lane == 0) part[w] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        if (RMS) {
            const float sum  = warp_sum(part[lane].x);
            const float mean = sum / K;
            if (lane == 0) stat = make_float2(rsqrtf(mean + eps), 0.0f);
        } else {
            const float2 total = warp_sum2(part[lane]);
            const float  mean  = total.x / K;
            const float  var   = total.y / K - mean * mean;
            if (lane == 0) stat = make_float2(mean, rsqrtf(var + eps));
        }
    }
    __syncthreads();
    const float2 st = stat;
    act_row_pairs<ABITS, ROT, NW>([&](int64_t k0, float & e0, float & e1) {
        const float2 xx = *(const float2 *) (x + k0);
        if (RMS) {
            e0 = rms_epilogue<1>(st.x, xx.x, p1, p2, (int) k0);
            e1 = rms_epilogue<1>(st.x, xx.y, p1, p2, (int) k0 + 1);
            return;
        }
        const float n0 = (xx.x - st.x) * st.y, n1 = (xx.y - st.x) * st.y;
        if (NORM == vla::FusedKind::LnModulate) {
            e0 = __fadd_rn(__fadd_rn(n0, __fmul_rn(n0, p1[k0])), p2[k0]);
            e1 = __fadd_rn(__fadd_rn(n1, __fmul_rn(n1, p1[k0+1])), p2[k0+1]);
        } else {
            e0 = n0;
            e1 = n1;
        }
    }, ascale, blob + m * row_bytes, K, fold_before, clip, inv_sqrt_bs);
}

template <vla::FusedKind NORM, int ABITS>
bool launch_norm_act_n(const vla::FusedSpec & s, const ggml_tensor * dst, cudaStream_t st) {
    constexpr int NW = 8;
    const ggml_tensor * x = dst->src[0];
    auto f = [](const ggml_tensor * t) { return t ? (const float *) t->data : nullptr; };
    auto go = [&](auto kern) {
        kern<<<(unsigned) dst->ne[1], NW * 32, 0, st>>>((const float *) x->data, f(dst->src[1]), f(dst->src[2]),
                                                       f(dst->src[3]), (int8_t *) dst->data, (int) x->ne[0],
                                                       (int64_t) dst->ne[0], s.eps, s.fold_before != 0, s.clip,
                                                       s.inv_sqrt_bs);
        return true;
    };
    switch (s.rot_block) {
        case 64: return go(k_norm_act<NORM, ABITS, 64, NW>);
        case 32: return go(k_norm_act<NORM, ABITS, 32, NW>);
        case 16: return go(k_norm_act<NORM, ABITS, 16, NW>);
        case 8:  return go(k_norm_act<NORM, ABITS, 8,  NW>);
        case 4:  return go(k_norm_act<NORM, ABITS, 4,  NW>);
        case 2:  return go(k_norm_act<NORM, ABITS, 2,  NW>);
        case 1:
        case 0:  return go(k_norm_act<NORM, ABITS, 1,  NW>);
        default: return false;
    }
}

template <vla::FusedKind NORM>
bool launch_norm_act_k(const vla::FusedSpec & s, const ggml_tensor * dst, cudaStream_t st) {
    return s.abits == 8 ? launch_norm_act_n<NORM, 8>(s, dst, st) : launch_norm_act_n<NORM, 4>(s, dst, st);
}

bool launch_norm_act(const vla::FusedSpec & s, const ggml_tensor * dst, cudaStream_t st) {
    const ggml_tensor * x = dst->src[0];
    const int64_t K = x->ne[0];
    const bool mod = s.norm != vla::FusedKind::LnPlain;
    if (dst->type != GGML_TYPE_I8 || K % 64 != 0 || K < 1024 || K > 8 * 16 * 64 || (s.abits != 8 && s.abits != 4) ||
        ggml_nrows(x) != dst->ne[1] || mod != (dst->src[1] && dst->src[2]) ||
        (dst->src[3] && ggml_nelements(dst->src[3]) != K))
        return false;
    switch (s.norm) {
        case vla::FusedKind::LnModulate:  return launch_norm_act_k<vla::FusedKind::LnModulate>(s, dst, st);
        case vla::FusedKind::LnPlain:     return launch_norm_act_k<vla::FusedKind::LnPlain>(s, dst, st);
        case vla::FusedKind::RmsModulate: return launch_norm_act_k<vla::FusedKind::RmsModulate>(s, dst, st);
        default:                          return false;
    }
}

__device__ __forceinline__ float gelu_erf_ggml(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

// ggml_gelu_erf(ggml_add(y, b)).
template <typename T>
__global__ void k_bias_gelu_erf(const float * __restrict__ y, const float * __restrict__ b, T * __restrict__ out,
                                int64_t n, int N) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) st_out(out, i, gelu_erf_ggml(__fadd_rn(y[i], b[i%N])));
}

// ggml_add(r, ggml_mul(ggml_add(y, b), g)), b and g broadcast over rows.
__global__ void k_bias_gate_res(const float * __restrict__ y, const float * __restrict__ b, const float * __restrict__ g,
                                const float * __restrict__ r, float * __restrict__ out, int64_t n, int N) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = __fadd_rn(r[i], __fmul_rn(__fadd_rn(y[i], b[i%N]), g[i%N]));
}

// ggml_add(r, ggml_mul(y, g)), g broadcast over rows.
__global__ void k_gate_res(const float * __restrict__ y, const float * __restrict__ g, const float * __restrict__ r,
                           float * __restrict__ out, int64_t n, int N) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = __fadd_rn(r[i], __fmul_rn(y[i], g[i%N]));
}

// fused_heads_rope, untransposed: [hd, Tp+T, H]. One thread per rotated pair
// (j, j+hd/2) of a (token, head); ggml rope_neox's theta and the FMA forms its
// SASS shows: fma(x0, cos, -(x1*sin)), fma(x0, sin, x1*cos).
__global__ void k_heads_rope(const float * __restrict__ x, const int32_t * __restrict__ pos, const float * __restrict__ P,
                             float * __restrict__ out, int hd, int H, int T, int Tp, float theta_scale,
                             const float * __restrict__ b) {
    const int half = hd/2, Tt = Tp+T;
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= (int64_t) half*H*Tt) return;
    const int j = (int) (i%half), h = (int) ((i/half)%H), t = (int) (i/((int64_t) half*H));
    float * o = out+(int64_t) hd*(t+(int64_t) Tt*h);
    if (t < Tp) {
        const float * pr = P+(int64_t) hd*(h+(int64_t) H*t);
        o[j] = pr[j];
        o[j+half] = pr[j+half];
        return;
    }
    const float * xr = x+(int64_t) hd*(h+(int64_t) H*(t-Tp));
    const float x0 = b ? __fadd_rn(xr[j], b[h*hd+j]) : xr[j], x1 = b ? __fadd_rn(xr[j+half], b[h*hd+j+half]) : xr[j+half];
    if (!pos) { o[j] = x0; o[j+half] = x1; return; }
    const float theta_base = pos[t-Tp]*powf(theta_scale, (2*j)/2.0f);
    // rope_yarn with freq_scale 1, ext_factor 0, mscale 1, freq_factor 1
    const float theta = 1.0f * (theta_base / 1.0f);
    const float cos_theta = cosf(theta) * 1.0f;
    const float sin_theta = sinf(theta) * 1.0f;
    o[j]      = __fmaf_rn(x0, cos_theta, -__fmul_rn(x1, sin_theta));
    o[j+half] = __fmaf_rn(x0, sin_theta,  __fmul_rn(x1, cos_theta));
}

// fused_heads_rope, transposed (no RoPE): [Tp+T, hd, H] through 32x32 tiles of
// (token, channel), so the reads along channels and writes along tokens coalesce.
__global__ void k_heads_vt(const float * __restrict__ x, const float * __restrict__ P, float * __restrict__ out,
                           int hd, int H, int T, int Tp, const float * __restrict__ b) {
    __shared__ float tile[32][33];
    const int Tt = Tp+T, D = hd*H;
    const int c0 = blockIdx.x*32, t0 = blockIdx.y*32;
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        const int t = t0+k, c = c0+threadIdx.x;
        if (t < Tt && c < D)
            tile[k][threadIdx.x] = t < Tp ? P[(int64_t) t*D+c] : b ? __fadd_rn(x[(int64_t) (t-Tp)*D+c], b[c]) : x[(int64_t) (t-Tp)*D+c];
    }
    __syncthreads();
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        const int c = c0+k, t = t0+threadIdx.x;
        if (t < Tt && c < D) {
            const int h = c/hd, d = c-h*hd;
            out[t+(int64_t) Tt*(d+(int64_t) hd*h)] = tile[threadIdx.x][k];
        }
    }
}

// The same sums with one warp per row (ncols >= 1024: 32 warps in ggml's
// block). Lane l holds thread 32*w + l's partial sum for every warp w; the
// additions are the same, without shared memory or barriers.
template <int mode, typename T>
__global__ void k_ln_warp(const float * __restrict__ x, const float * __restrict__ p1, const float * __restrict__ p2,
                          T * __restrict__ dst, int ncols, int nrows, float eps) {
    const int row = blockIdx.x*(blockDim.x/WARP)+threadIdx.x/WARP, lane = threadIdx.x%WARP;
    if (row >= nrows) return;
    x   += (int64_t) row*ncols;
    dst += (int64_t) row*ncols;

    float2 total = make_float2(0.0f, 0.0f);
    if (ncols < 1024) {
        float2 mean_var = make_float2(0.0f, 0.0f);
        for (int col = lane; col < ncols; col += WARP) {
            const float xi = x[col];
            mean_var.x += xi;
            mean_var.y += xi * xi;
        }
        total = warp_sum2(mean_var);
    } else {
        float2 mine = make_float2(0.0f, 0.0f);
        for (int w = 0; w < WARP; ++w) {
            float2 mean_var = make_float2(0.0f, 0.0f);
            for (int col = w*WARP+lane; col < ncols; col += 1024) {
                const float xi = x[col];
                mean_var.x += xi;
                mean_var.y += xi * xi;
            }
            mean_var = warp_sum2(mean_var);
            if (lane == w) mine = mean_var;
        }
        total = warp_sum2(mine);
    }

    const float mean = total.x / ncols;
    const float var = total.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = lane; col < ncols; col += WARP) {
        const float n = (x[col] - mean) * inv_std;
        st_out(dst, col, mode == LN_MOD    ? __fadd_rn(__fadd_rn(n, __fmul_rn(n, p1[col])), p2[col])
                       : mode == LN_AFFINE ? __fadd_rn(__fmul_rn(n, p1[col]), p2[col]) : n);
    }
}

// k_ln_warp with the row in registers: lane l loads column 32*j + l into v[j],
// so thread 32*w + l of ggml's block owns v[w], v[w + BS/32], ... in ggml's
// order. The output pass reuses v.
template <int NPL, int mode, typename T>
__global__ void k_ln_reg(const float * __restrict__ x, const float * __restrict__ p1, const float * __restrict__ p2,
                         T * __restrict__ dst, int ncols, int nrows, float eps) {
    // The per-column parameters are shared by all rows: staged in shared memory
    // up front, so their loads overlap the row loads.
    __shared__ float s_p[mode == LN_PLAIN ? 1 : 2*NPL*WARP];
    if (mode != LN_PLAIN) {
        for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
            s_p[c]          = p1[c];
            s_p[NPL*WARP+c] = p2[c];
        }
    }
    const int row = blockIdx.x*(blockDim.x/WARP)+threadIdx.x/WARP, lane = threadIdx.x%WARP;
    const bool active = row < nrows;
    x   += (int64_t) (active ? row : 0)*ncols;
    dst += (int64_t) (active ? row : 0)*ncols;
    float v[NPL];
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int col = j*WARP+lane;
        v[j] = active && col < ncols ? x[col] : 0.0f;
    }
    if (mode != LN_PLAIN)
        __syncthreads();
    if (!active) return;
    float2 total;
    if (ncols < 1024) {
        float2 mean_var = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int j = 0; j < NPL; ++j) {
            if (j*WARP+lane < ncols) {
                const float xi = v[j];
                mean_var.x += xi;
                mean_var.y += xi * xi;
            }
        }
        total = warp_sum2(mean_var);
    } else {
        float2 mine = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int w = 0; w < WARP; ++w) {
            float2 mean_var = make_float2(0.0f, 0.0f);
#pragma unroll
            for (int j = w; j < NPL; j += WARP) {
                if (j*WARP+lane < ncols) {
                    const float xi = v[j];
                    mean_var.x += xi;
                    mean_var.y += xi * xi;
                }
            }
            mean_var = warp_sum2(mean_var);
            if (lane == w) mine = mean_var;
        }
        total = warp_sum2(mine);
    }
    const float mean = total.x / ncols;
    const float var = total.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int col = j*WARP+lane;
        if (col < ncols) {
            const float n = (v[j] - mean) * inv_std;
            const float a = mode == LN_PLAIN ? 0.0f : s_p[col], c = mode == LN_PLAIN ? 0.0f : s_p[NPL*WARP+col];
            st_out(dst, col, mode == LN_MOD    ? __fadd_rn(__fadd_rn(n, __fmul_rn(n, a)), c)
                           : mode == LN_AFFINE ? __fadd_rn(__fmul_rn(n, a), c) : n);
        }
    }
}

template <int mode, typename T>
void launch_ln(const float * x, const float * p1, const float * p2, T * dst, int ncols, int nrows, float eps,
               cudaStream_t st) {
    constexpr int rows_per_block = 4;
    const dim3 grid((nrows+rows_per_block-1)/rows_per_block), block(rows_per_block*WARP);
    const int npl = (ncols+WARP-1)/WARP;
    if      (npl <= 32) k_ln_reg<32, mode><<<grid, block, 0, st>>>(x, p1, p2, dst, ncols, nrows, eps);
    else if (npl <= 48) k_ln_reg<48, mode><<<grid, block, 0, st>>>(x, p1, p2, dst, ncols, nrows, eps);
    else if (npl <= 64) k_ln_reg<64, mode><<<grid, block, 0, st>>>(x, p1, p2, dst, ncols, nrows, eps);
    else                k_ln_warp<mode><<<grid, block, 0, st>>>(x, p1, p2, dst, ncols, nrows, eps);
}

template <int mode>
void launch_ln_t(const ggml_tensor * dst, const float * x, const float * p1, const float * p2, int ncols, int nrows,
                 float eps, cudaStream_t st) {
    if (dst->type == GGML_TYPE_BF16) launch_ln<mode>(x, p1, p2, (__nv_bfloat16 *) dst->data, ncols, nrows, eps, st);
    else                             launch_ln<mode>(x, p1, p2, (float *) dst->data, ncols, nrows, eps, st);
}

enum ScratchSlot { SCR_KQ, SCR_KQV, SCR_PTRS, SCR_BF16, SCR_LT_WS, SCR_FLAG, SCR_COUNT };

int trace_level();
cublasLtHandle_t lt_handle();

// What the GEMM and attention nodes keep per (device, stream): the device, a
// cuBLAS handle configured like ggml-cuda's, the cuBLASLt handle, and the
// scratch slots. Found once per thread and stream, so the per-node path takes
// no lock and no cudaGetDevice.
struct StreamCtx {
    struct Slot { void * p = nullptr; size_t size = 0; std::vector<void *> old; };
    int dev = 0;
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t lt = nullptr;
    std::array<Slot, SCR_COUNT> slots;
};

std::mutex & stream_ctx_mu() {
    static std::mutex mu;
    return mu;
}

StreamCtx & stream_ctx(cudaStream_t st) {
    thread_local cudaStream_t last_st = nullptr;
    thread_local StreamCtx * last = nullptr;
    if (last && st == last_st) return *last;
    static std::map<std::pair<int, cudaStream_t>, StreamCtx> ctxs;
    int dev = 0;
    cudaGetDevice(&dev);
    const cublasLtHandle_t lt = lt_handle();
    std::lock_guard<std::mutex> lk(stream_ctx_mu());
    StreamCtx & c = ctxs[{dev, st}];
    c.dev = dev;
    c.lt = lt;
    last_st = st;
    last = &c;
    return c;
}

// The stream's scratch for one slot, grown on a shape's first (eager) use.
// Nodes on one stream run in order, so they share it. A slot that grows keeps
// its old block, which a CUDA graph captured earlier may still read; with
// doubling the total stays under four times the largest request.
void * scratch(cudaStream_t st, int slot, size_t bytes) {
    StreamCtx & c = stream_ctx(st);
    StreamCtx::Slot & s = c.slots[slot];
    if (bytes <= s.size) return s.p;
    std::lock_guard<std::mutex> lk(stream_ctx_mu());
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cap);
    if (cap != cudaStreamCaptureStatusNone) return nullptr;
    const size_t want = std::max(bytes, 2*s.size);
    void * b = nullptr;
    if (cudaMalloc(&b, want) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    if (s.p) s.old.push_back(s.p);
    s.p = b;
    s.size = want;
    if (trace_level() >= 2)
        std::fprintf(stderr, "vla(scratch): device %d stream %p slot %d -> %.1f MiB\n", c.dev, (void *) st, slot, want/1048576.0);
    return b;
}

// A cuBLAS handle configured like ggml-cuda's (TF32 tensor-op math, an
// explicit 4 MiB workspace on pre-Hopper parts, same stream), so a GEMM issued
// here gets the algorithm ggml's own call would get. Created on first use,
// which ggml runs eagerly before it captures a CUDA graph.
cublasHandle_t ggml_like_handle(cudaStream_t st) {
    StreamCtx & c = stream_ctx(st);
    if (c.cublas) return c.cublas;
    std::lock_guard<std::mutex> lk(stream_ctx_mu());
    if (c.cublas) return c.cublas;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, c.dev);
    const size_t ws_size = prop.major >= 9 ? 32u*1024*1024 : 4u*1024*1024;
    void * ws = nullptr;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS || cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS || cudaMalloc(&ws, ws_size) != cudaSuccess ||
        cublasSetWorkspace(h, ws, ws_size) != CUBLAS_STATUS_SUCCESS)
        return nullptr;
    c.cublas = h;
    return h;
}

// ggml_cuda_mul_mat_cublas_impl<BF16> with prefer_f32_output (every pre-RDNA3,
// non-CDNA device) for one view.
cublasStatus_t gemm_one(cublasHandle_t h, const void * w, const __nv_bfloat16 * xb, float * o, int N, int S, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, S, K,
                        &alpha, w,  CUDA_R_16BF, K,
                                xb, CUDA_R_16BF, K,
                        &beta,  o,  CUDA_R_32F,  N,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// All views in one cuBLASLt call, with an algorithm that reproduces
// cublasGemmEx bit for bit. Chosen once per shape on random data, outside
// graph capture; with no exact match, per-view cublasGemmEx.
struct LtPlan {
    bool ok = false;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t ws_size = 4u*1024*1024;       // from the stream's scratch
    float t_ref = 0.0f, t_pick = 0.0f;   // us, at selection
    skinny::LaunchFn sk = nullptr;       // a gemm_skinny configuration, when that is the pick
    const char * sk_name = nullptr;
};

cublasLtHandle_t lt_handle() {
    static std::mutex mu;
    static std::map<int, cublasLtHandle_t> handles;
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(mu);
    cublasLtHandle_t & lt = handles[dev];
    if (!lt && cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) lt = nullptr;
    return lt;
}

bool lt_layouts(LtPlan & p, int N, int S, int K, int nv) {
    const int64_t sa = 0, sb = (int64_t) S*K, sc = (int64_t) S*N;
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    if (cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, S, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32F,  N, S, N) != CUBLAS_STATUS_SUCCESS) return false;
    for (auto [l, stride] : {std::make_pair(p.a, sa), std::make_pair(p.b, sb), std::make_pair(p.c, sc)}) {
        cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &nv, sizeof(nv));
        cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
    }
    return true;
}

bool lt_run(const LtPlan & p, const cublasLtMatmulAlgo_t & algo, const void * w, const void * x, float * o, cudaStream_t st) {
    const float alpha = 1.0f, beta = 0.0f;
    void * ws = scratch(st, SCR_LT_WS, p.ws_size);
    return ws && cublasLtMatmul(stream_ctx(st).lt, p.op, &alpha, w, p.a, x, p.b, &beta, o, p.c, o, p.c, &algo, ws, p.ws_size, st)
                 == CUBLAS_STATUS_SUCCESS;
}

// Mean time per call in us, over 10 calls after 3 warmups.
float time_us(cudaStream_t st, const std::function<void()> & f) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; ++i) f();
    cudaEventRecord(e0, st);
    for (int i = 0; i < 10; ++i) f();
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms*100.0f;
}

// VLA_EXACT_TRACE: 1 prints each shape's GEMM and attention choice, 2 also
// every candidate.
int trace_level() {
    static const int lvl = [] { const char * t = std::getenv("VLA_EXACT_TRACE"); return t ? std::atoi(t) : 0; }();
    return lvl;
}

// The first (eager) predict often finds the GPU clocks low: run f for ~50 ms
// so the timings that follow are taken at the speed the model will run at.
void warm_clocks(cudaStream_t st, const std::function<void()> & f) {
    float spent = 0.0f;
    for (int i = 0; i < 1000 && spent < 50e3f; ++i) spent += time_us(st, f)*10.0f;
}

// Per-device cache of cuBLASLt configurations (VLA_GEMM_CACHE=file): one line
// per configuration, "<cublasLt version> N S K id tile stages swizzle custom".
// VLA_GEMM_SEARCH=1 enumerates every configuration of a shape (slow) and
// appends the fastest bit-exact ones. Entries of any S for the same N and K
// are candidates (prompt lengths vary); they are checked and timed like the
// heuristic's.
struct LtConfig { int id, tile, stages, swz, custom; };

std::vector<LtConfig> lt_cache_read(int N, int K) {
    std::vector<LtConfig> v;
    const char * path = std::getenv("VLA_GEMM_CACHE");
    FILE * f = path ? std::fopen(path, "r") : nullptr;
    if (!f) return v;
    size_t ver = 0; int n, s_, k; LtConfig c;
    while (std::fscanf(f, "%zu %d %d %d %d %d %d %d %d", &ver, &n, &s_, &k, &c.id, &c.tile, &c.stages, &c.swz, &c.custom) == 9)
        if (ver == cublasLtGetVersion() && n == N && k == K &&
            std::none_of(v.begin(), v.end(), [&](const LtConfig & o) { return std::memcmp(&o, &c, sizeof c) == 0; }))
            v.push_back(c);
    std::fclose(f);
    return v;
}

void lt_cache_append(int N, int S, int K, const std::vector<LtConfig> & v) {
    const char * path = std::getenv("VLA_GEMM_CACHE");
    FILE * f = path ? std::fopen(path, "a") : nullptr;
    if (!f) return;
    for (const LtConfig & c : v)
        std::fprintf(f, "%zu %d %d %d %d %d %d %d %d\n", cublasLtGetVersion(), N, S, K, c.id, c.tile, c.stages, c.swz, c.custom);
    std::fclose(f);
}

bool lt_algo(const LtConfig & c, cublasLtMatmulAlgo_t & algo) {
    if (cublasLtMatmulAlgoInit(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F,
                               c.id, &algo) != CUBLAS_STATUS_SUCCESS) return false;
    const int one = 1, red = CUBLASLT_REDUCTION_SCHEME_NONE;
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &c.tile, sizeof(c.tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &c.stages, sizeof(c.stages));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof(one));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &c.swz, sizeof(c.swz));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &c.custom, sizeof(c.custom));
    return true;
}

// Every split-K-free configuration cuBLASLt offers for the layouts.
std::vector<LtConfig> lt_enumerate(const LtPlan & p) {
    std::vector<LtConfig> v;
    int ids[256], n_ids = 0;
    cublasLtMatmulAlgoGetIds(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F,
                             256, ids, &n_ids);
    for (int i = 0; i < n_ids; ++i) {
        cublasLtMatmulAlgo_t algo;
        if (!lt_algo({ids[i], 0, 0, 0, 0}, algo)) continue;
        size_t sz = 0;
        auto cap_list = [&](cublasLtMatmulAlgoCapAttributes_t a) {
            cublasLtMatmulAlgoCapGetAttribute(&algo, a, nullptr, 0, &sz);
            std::vector<int> l(std::max<size_t>(1, sz/sizeof(int)), 0);
            if (sz) cublasLtMatmulAlgoCapGetAttribute(&algo, a, l.data(), sz, &sz);
            return l;
        };
        const std::vector<int> tiles = cap_list(CUBLASLT_ALGO_CAP_TILE_IDS), stages = cap_list(CUBLASLT_ALGO_CAP_STAGES_IDS);
        int swz = 0, cust = 0;
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swz, sizeof(swz), &sz);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &cust, sizeof(cust), &sz);
        for (int t : tiles) for (int sg : stages) for (int w = 0; w <= swz; ++w) for (int c = 0; c <= cust; ++c) {
            const LtConfig cfg{ids[i], t, sg, w, c};
            cublasLtMatmulAlgo_t a;
            cublasLtMatmulHeuristicResult_t chk{};
            if (lt_algo(cfg, a) && cublasLtMatmulAlgoCheck(lt_handle(), p.op, p.a, p.b, p.c, p.c, &a, &chk) == CUBLAS_STATUS_SUCCESS &&
                chk.workspaceSize <= p.ws_size)
                v.push_back(cfg);
        }
    }
    return v;
}

// Random BF16 test operands: draw skip+i+1 of the LCG x -> 1664525 x + 1013904223
// from `seed`, as ((x >> 8) / 2^24) * 2 - 1 times `scale`. Each thread jumps
// ahead in log2 steps, so the values are those of the sequential host loop.
__global__ void k_lcg_bf16(__nv_bfloat16 * __restrict__ out, int64_t n, uint32_t seed, int64_t skip, float scale) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    uint32_t A = 1u, C = 0u, a = 1664525u, c = 1013904223u;
    for (uint64_t k = (uint64_t) (skip+i+1); k; k >>= 1) {
        if (k & 1) { A = a*A; C = a*C+c; }
        c = a*c+c; a = a*a;
    }
    const uint32_t x = A*seed+C;
    const float v = __fadd_rn(__fmul_rn(__fmul_rn((float) (x >> 8), 1.0f/16777216.0f), 2.0f), -1.0f);
    out[i] = __float2bfloat16(__fmul_rn(scale, v));
}

void lcg_bf16(void * out, size_t n, size_t skip, float scale, cudaStream_t st) {
    k_lcg_bf16<<<(unsigned) ((n+255)/256), 256, 0, st>>>((__nv_bfloat16 *) out, (int64_t) n, 12345u, (int64_t) skip, scale);
}

__global__ void k_bits_differ(const uint32_t * __restrict__ a, const uint32_t * __restrict__ b, int64_t n, int * flag) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x)
        if (a[i] != b[i]) *flag = 1;
}

// The n floats at a and b have the same bits (a memcmp on the device).
bool same_bits(const float * a, const float * b, size_t n, cudaStream_t st) {
    int * flag = (int *) scratch(st, SCR_FLAG, sizeof(int));
    if (!flag || cudaMemsetAsync(flag, 0, sizeof(int), st) != cudaSuccess) return false;
    k_bits_differ<<<256, 256, 0, st>>>((const uint32_t *) a, (const uint32_t *) b, (int64_t) n, flag);
    int h = 1;
    return cudaMemcpyAsync(&h, flag, sizeof(int), cudaMemcpyDeviceToHost, st) == cudaSuccess &&
           cudaStreamSynchronize(st) == cudaSuccess && h == 0;
}

LtPlan * lt_plan(cublasHandle_t h, int N, int S, int K, int nv, cudaStream_t st) {
    static std::mutex mu;
    static std::map<std::tuple<int, int, int, int, int>, LtPlan *> plans;
    const int dev = stream_ctx(st).dev;
    std::lock_guard<std::mutex> lk(mu);
    auto it = plans.find({dev, N, S, K, nv});
    if (it != plans.end()) return it->second;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cap);
    LtPlan * p = new LtPlan();
    plans[{dev, N, S, K, nv}] = p;
    if (cap != cudaStreamCaptureStatusNone || !vla::env_flag("VLA_GEMM_LT", true) || !lt_handle()) return p;
    const auto t_start = std::chrono::steady_clock::now();
    auto ms_since = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t_start).count(); };

    // A prompt that changes length every call (pi0.5 writes the state into it)
    // brings a new S each time. A cuBLASLt or gemm_skinny pick made for the
    // same N, K and views within 64 rows is tried first: kept without timing
    // the candidates again, which is slow on large shapes, if it is bit-exact
    // here too.
    const LtPlan * near = nullptr;
    int near_s = 0;
    if (vla::env_flag("VLA_GEMM_REUSE", true) && !vla::env_flag("VLA_GEMM_SEARCH"))
        for (const auto & [key, q] : plans) {
            const int s_ = std::get<2>(key);
            if ((q->ok || q->sk) && std::get<0>(key) == dev && std::get<1>(key) == N && std::get<3>(key) == K && std::get<4>(key) == nv &&
                std::abs(s_-S) <= 64 && (!near || std::abs(s_-S) < std::abs(near_s-S))) {
                near = q;
                near_s = s_;
            }
        }
    int reused_s = -1;

    // Random BF16 operands (W, then X, from one LCG stream) and the per-view reference.
    const size_t nw = (size_t) K*N, nx = (size_t) K*S*nv, no = (size_t) N*S*nv;
    void * dw = nullptr, * dx = nullptr; float * dref = nullptr, * dout = nullptr;
    bool good = cudaMalloc(&dw, nw*2) == cudaSuccess && cudaMalloc(&dx, nx*2) == cudaSuccess &&
                cudaMalloc(&dref, no*4) == cudaSuccess && cudaMalloc(&dout, no*4) == cudaSuccess;
    if (good) {
        lcg_bf16(dw, nw, 0, 1.0f, st);
        lcg_bf16(dx, nx, nw, 1.0f, st);
    }
    for (int v = 0; good && v < nv; ++v)
        good = gemm_one(h, dw, (const __nv_bfloat16 *) dx+(size_t) v*S*K, dref+(size_t) v*S*N, N, S, K) == CUBLAS_STATUS_SUCCESS;

    LtPlan one;   // single-view layouts to identify cublasGemmEx's algorithm
    good = good && lt_layouts(one, N, S, K, 1) && lt_layouts(*p, N, S, K, nv);
    if (good) {
        // Second operand set (wider range, the stream's next draws) for the checks below.
        void * dx2 = nullptr; float * dref2 = nullptr;
        bool good2 = cudaMalloc(&dx2, nx*2) == cudaSuccess && cudaMalloc(&dref2, no*4) == cudaSuccess;
        if (good2) lcg_bf16(dx2, nx, nw+nx, 8.0f, st);
        for (int v = 0; good2 && v < nv; ++v)
            good2 = gemm_one(h, dw, (const __nv_bfloat16 *) dx2+(size_t) v*S*K, dref2+(size_t) v*S*N, N, S, K) == CUBLAS_STATUS_SUCCESS;
        // Bit-exact against cublasGemmEx: one view on the first set, all views on both.
        auto is_exact = [&](const cublasLtMatmulAlgo_t & algo) {
            return good2 &&
                   lt_run(one, algo, dw, dx, dout, st) && same_bits(dout, dref, (size_t) N*S, st) &&
                   lt_run(*p, algo, dw, dx2, dout, st) && same_bits(dout, dref2, no, st) &&
                   lt_run(*p, algo, dw, dx, dout, st) && same_bits(dout, dref, no, st);
        };
        auto sk_exact = [&](skinny::LaunchFn fn) {
            bool same = good2;
            for (int set = 0; set < 2 && same; ++set)
                same = fn((const __nv_bfloat16 *) dw, (const __nv_bfloat16 *) (set ? dx2 : dx), dout, N, S, K, nv, st) &&
                       same_bits(dout, set ? dref2 : dref, no, st);
            cudaGetLastError();
            return same;
        };
        if (near && near->ok && is_exact(near->algo)) {
            p->algo = near->algo;
            p->ok = true;
            reused_s = near_s;
        } else if (near && near->sk && sk_exact(near->sk)) {
            p->sk = near->sk;
            p->sk_name = near->sk_name;
            reused_s = near_s;
        } else {
            cublasLtMatmulPreference_t pref = nullptr;
            cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p->ws_size, sizeof(p->ws_size));
            cublasLtMatmulHeuristicResult_t res[32];
            int n_res = 0;
            cublasLtMatmulAlgoGetHeuristic(lt_handle(), one.op, one.a, one.b, one.c, one.c, pref, 32, res, &n_res);
            cublasLtMatmulPreferenceDestroy(pref);
            // Candidates: the heuristic's split-K-free picks, cached configurations,
            // and with VLA_GEMM_SEARCH=1 every configuration (cfg.id < 0: heuristic).
            std::vector<cublasLtMatmulAlgo_t> algos;
            std::vector<LtConfig> cfgs;
            for (int i = 0; i < n_res; ++i) {
                uint32_t splitk = 1, red = 0; size_t sz = 0;
                cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &sz);
                cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red), &sz);
                if (splitk > 1 || red != CUBLASLT_REDUCTION_SCHEME_NONE) continue;
                algos.push_back(res[i].algo);
                cfgs.push_back({-1, 0, 0, 0, 0});
            }
            const bool search = vla::env_flag("VLA_GEMM_SEARCH");
            for (const LtConfig & c : search ? lt_enumerate(one) : lt_cache_read(N, K)) {
                cublasLtMatmulAlgo_t a;
                if (lt_algo(c, a)) { algos.push_back(a); cfgs.push_back(c); }
            }
            std::vector<cublasLtMatmulAlgo_t> exact;
            std::vector<LtConfig> exact_cfg;
            for (size_t i = 0; i < algos.size(); ++i)
                if (is_exact(algos[i])) {
                    exact.push_back(algos[i]);
                    exact_cfg.push_back(cfgs[i]);
                }
            cudaGetLastError();
            // gemm_skinny configurations that reproduce cublasGemmEx on both operand sets.
            std::vector<skinny::Cand> sk_ok;
            if (vla::env_flag("VLA_GEMM_SKINNY", true))
                for (const skinny::Cand & c : skinny::cands(S)) {
                    const bool same = sk_exact(c.fn);
                    if (same) sk_ok.push_back(c);
                    if (trace_level() >= 2)
                        std::fprintf(stderr, "vla(gemm_exact):   N=%d S=%d K=%d gemm_skinny %s: %s\n", N, S, K, c.name,
                                     same ? "bit-exact" : "differs");
                }
            // Fastest of the bit-exact algorithms, against cublasGemmEx itself:
            // interleaved rounds, minimum per candidate (clocks drift under DVFS).
            auto ref_call = [&] {
                for (int v = 0; v < nv; ++v)
                    gemm_one(h, dw, (const __nv_bfloat16 *) dx+(size_t) v*S*K, dout+(size_t) v*S*N, N, S, K);
            };
            const size_t n_cand = exact.size()+sk_ok.size();
            if (n_cand > 0) warm_clocks(st, ref_call);
            std::vector<float> best(n_cand+1, 1e30f);
            for (int round = 0; round < 4 && n_cand > 0; ++round) {
                best[0] = std::min(best[0], time_us(st, ref_call));
                for (size_t i = 0; i < exact.size(); ++i)
                    best[i+1] = std::min(best[i+1], time_us(st, [&] { lt_run(*p, exact[i], dw, dx, dout, st); }));
                for (size_t i = 0; i < sk_ok.size(); ++i)
                    best[exact.size()+i+1] = std::min(best[exact.size()+i+1], time_us(st, [&] {
                        sk_ok[i].fn((const __nv_bfloat16 *) dw, (const __nv_bfloat16 *) dx, dout, N, S, K, nv, st); }));
            }
            size_t pick = 0;
            for (size_t i = 1; i < best.size(); ++i)
                if (best[i] < best[pick]) pick = i;
            if (trace_level() >= 2)
                for (size_t i = 0; i < best.size(); ++i)
                    std::fprintf(stderr, "vla(gemm_exact):   N=%d S=%d K=%d %s: %.1f us\n", N, S, K,
                                 i == 0 ? "cublasGemmEx" : i <= exact.size() ? "cuBLASLt" : sk_ok[i-exact.size()-1].name, best[i]);
            if (search) {   // the three fastest enumerated configurations go to the cache
                std::vector<size_t> idx;
                for (size_t i = 0; i < exact.size(); ++i)
                    if (exact_cfg[i].id >= 0) idx.push_back(i);
                std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return best[a+1] < best[b+1]; });
                std::vector<LtConfig> keep;
                for (size_t i = 0; i < idx.size() && i < 3; ++i) keep.push_back(exact_cfg[idx[i]]);
                lt_cache_append(N, S, K, keep);
            }
            if (pick > exact.size()) {
                p->sk = sk_ok[pick-exact.size()-1].fn;
                p->sk_name = sk_ok[pick-exact.size()-1].name;
            } else if (pick > 0) {
                p->algo = exact[pick-1];
                p->ok = true;
            }
            p->t_ref = n_cand == 0 ? 0.0f : best[0]; p->t_pick = n_cand == 0 ? 0.0f : best[pick];
        }
        cudaGetLastError();
        cudaFree(dx2); cudaFree(dref2);
    }
    if (trace_level() >= 1 && reused_s >= 0)
        std::fprintf(stderr, "vla(gemm_exact): N=%d S=%d K=%d views=%d -> %s, bit-exact (choice of S=%d), %.0f ms\n",
                     N, S, K, nv, p->sk ? "gemm_skinny" : "cuBLASLt", reused_s, ms_since());
    else if (trace_level() >= 1)
        std::fprintf(stderr, "vla(gemm_exact): N=%d S=%d K=%d views=%d -> %s%s (%.1f -> %.1f us), %.0f ms\n", N, S, K, nv,
                     p->sk ? "gemm_skinny, bit-exact: " : p->ok ? "cuBLASLt, bit-exact" : "cublasGemmEx",
                     p->sk ? p->sk_name : "", p->t_ref, p->t_pick, ms_since());
    cudaFree(dw); cudaFree(dx); cudaFree(dref); cudaFree(dout);
    cudaGetLastError();
    return p;
}

// ---------------------------------------------------------------------------
// attn_exact
// ---------------------------------------------------------------------------

// ggml soft_max_f32 for one row of ncols (no mask, no sinks), block of nth
// threads with thread t owning columns t, t + nth, ... (one column while
// nth >= ncols; nth stops at 1024): x*scale + 0, max, __expf, block sum,
// 1/sum, multiply.
template <int NTH>
__device__ __forceinline__ float softmax_sum_emul(const float * e, int ncols, int lane) {
    if (NTH <= WARP) {
        float t = 0.0f;
        for (int col = lane; col < ncols; col += NTH) t += e[col];
        return warp_sum(t);
    }
    float mine = 0.0f;
#pragma unroll
    for (int w = 0; w < NTH/WARP; ++w) {
        float t = 0.0f;
        for (int col = w*WARP+lane; col < ncols; col += NTH) t += e[col];
        t = warp_sum(t);
        if (lane == w) mine = t;
    }
    return warp_sum(mine);
}

// Path A: a block per (head, group of 8 queries), one warp per query. K, V^T
// and the group's Q rows are staged in shared memory (K rows padded to hd+1
// floats so the lanes' reads of different keys do not share banks). QK^T and
// PV are sequential FMA chains over the reduction index - cuBLAS's FFMA sgemm
// order - and the softmax is soft_max_f32's, as emulated above.
constexpr int ATTN_QPB = 8;

template <int NTH, typename T>
__global__ void k_attn_exact(const float * __restrict__ Q, const float * __restrict__ K, const float * __restrict__ Vt,
                             T * __restrict__ out, int hd, int H, int Tq, int Tkv, float scale, int nv) {
    extern __shared__ float sh[];
    const int h = blockIdx.x, q0 = blockIdx.y*ATTN_QPB;
    const int wib = threadIdx.x/WARP, lane = threadIdx.x%WARP;
    const int ks = hd+1;
    float * sK = sh;                       // [Tkv][hd+1]
    float * sV = sK+Tkv*ks;                // [hd][Tkv]
    float * sQ = sV+hd*Tkv;                // [QPB][hd]
    float * sE = sQ+ATTN_QPB*hd;           // [QPB][NTH]
    const float * Kh = K+(int64_t) hd*Tkv*h, * Vh = Vt+(int64_t) Tkv*hd*h;
    for (int i = threadIdx.x; i < hd*Tkv; i += blockDim.x) {
        sK[(i/hd)*ks+i%hd] = Kh[i];
        sV[i] = Vh[i];
    }
    const int nq = min(ATTN_QPB, Tq-q0);
    for (int i = threadIdx.x; i < nq*hd; i += blockDim.x)
        sQ[i] = Q[(int64_t) hd*(q0+(int64_t) Tq*h)+i];
    __syncthreads();
    if (wib >= nq) return;
    const int q = q0+wib;
    const float * qv = sQ+wib*hd;
    float * e = sE+wib*NTH;
    float mx = -INFINITY;
    for (int k = lane; k < Tkv; k += WARP) {
        const float * kv = sK+k*ks;
        float acc = 0.0f;
        for (int i = 0; i < hd; ++i) acc = fmaf(kv[i], qv[i], acc);
        const float val = acc*scale + 0.0f;
        e[k] = val;
        mx = fmaxf(mx, val);
    }
#pragma unroll
    for (int off = WARP/2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, WARP));
    __syncwarp();
    for (int k = lane; k < Tkv; k += WARP) e[k] = expf(e[k] - mx);
    __syncwarp();
    const float sum = softmax_sum_emul<NTH>(e, Tkv, lane);
    const float inv_sum = 1.0f / sum;
    for (int k = lane; k < Tkv; k += WARP) e[k] = e[k] * inv_sum;
    __syncwarp();
    const int hpv = H/nv;
    T * o = out+(int64_t) hd*(h%hpv)+(int64_t) hd*hpv*(q+(int64_t) Tq*(h/hpv));
    for (int i = lane; i < hd; i += WARP) {
        const float * vr = sV+i*Tkv;
        float acc = 0.0f;
        for (int k = 0; k < Tkv; ++k) acc = fmaf(vr[k], e[k], acc);
        st_out(o, i, acc);
    }
}

// Path B's softmax (in place) and output permute, for the shapes path A does not
// reproduce. mask (optional, F32 or F16, row q%Tq at stride mstride elements)
// enters as soft_max_f32's x*scale + slope*mask with slope 1.
template <int NTH, typename M>
__global__ void k_softmax_rows_loop(float * __restrict__ x, int nrows, int ncols, float scale,
                                    const M * __restrict__ mask = nullptr, int Tq = 1, int64_t mstride = 0) {
    const int row = blockIdx.x*(blockDim.x/WARP)+threadIdx.x/WARP, lane = threadIdx.x%WARP;
    if (row >= nrows) return;
    float * r = x+(int64_t) row*ncols;
    const M * mr = mask ? mask+(int64_t) (row%Tq)*mstride : nullptr;
    float mx = -INFINITY;
    for (int k = lane; k < ncols; k += WARP) {
        const float v = mr ? __fmaf_rn(r[k], scale, (float) mr[k]) : r[k]*scale + 0.0f;
        r[k] = v;
        mx = fmaxf(mx, v);
    }
#pragma unroll
    for (int off = WARP/2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, WARP));
    __syncwarp();
    for (int k = lane; k < ncols; k += WARP) r[k] = expf(r[k] - mx);
    __syncwarp();
    const float inv_sum = 1.0f / softmax_sum_emul<NTH>(r, ncols, lane);
    __syncwarp();
    for (int k = lane; k < ncols; k += WARP) r[k] = r[k] * inv_sum;
}

template <int NTH, typename M>
__global__ void k_softmax_rows_reg(float * __restrict__ x, int nrows, int ncols, float scale,
                                    const M * __restrict__ mask = nullptr, int Tq = 1, int64_t mstride = 0) {
    // Lane l holds columns 32*j + l (j < NTH/32): virtual thread 32*j + l of
    // ggml's NTH-thread block owns exactly that column (NTH >= ncols).
    constexpr int NPL = NTH > WARP ? NTH/WARP : 1;
    const int row = blockIdx.x*(blockDim.x/WARP)+threadIdx.x/WARP, lane = threadIdx.x%WARP;
    if (row >= nrows) return;
    float * r = x+(int64_t) row*ncols;
    const M * mr = mask ? mask+(int64_t) (row%Tq)*mstride : nullptr;
    float v[NPL];
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int k = j*WARP+lane;
        v[j] = k < ncols ? r[k] : 0.0f;
    }
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int k = j*WARP+lane;
        if (k < ncols) {
            v[j] = mr ? __fmaf_rn(v[j], scale, (float) mr[k]) : v[j]*scale + 0.0f;
            mx = fmaxf(mx, v[j]);
        }
    }
#pragma unroll
    for (int off = WARP/2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, WARP));
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int k = j*WARP+lane;
        v[j] = k < ncols ? expf(v[j] - mx) : 0.0f;
    }
    float sum;
    if (NTH <= WARP) {
        sum = warp_sum(lane < ncols ? 0.0f + v[0] : 0.0f);
    } else {
        float p[NPL];
#pragma unroll
        for (int j = 0; j < NPL; ++j) p[j] = j*WARP+lane < ncols ? 0.0f + v[j] : 0.0f;
        sum = warp_sum(warp_sums_t<NPL>(p, lane, 0.0f));
    }
    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int k = j*WARP+lane;
        if (k < ncols) r[k] = v[j] * inv_sum;
    }
}

// Path B's softmax (in place). Rows of up to 256 columns are held in
// registers; for the 1024-thread block the plain loop is faster (measured).
template <int NTH, typename M>
void softmax_rows_emul(float * x, int nrows, int ncols, float scale, const M * mask, int Tq, int64_t mstride, cudaStream_t st) {
    if (NTH <= 256) k_softmax_rows_reg<NTH, M><<<(nrows+3)/4, 128, 0, st>>>(x, nrows, ncols, scale, mask, Tq, mstride);
    else            k_softmax_rows_loop<NTH, M><<<(nrows+3)/4, 128, 0, st>>>(x, nrows, ncols, scale, mask, Tq, mstride);
}

// ggml's k_compute_batched_ptrs for the two attention GEMMs with K/V broadcast
// over r query heads: [K_h/r, Q_h, kq_h] and [Vt_h/r, kq_h, kqv_h].
__global__ void k_attn_ptrs(const void ** ptrs, const float * Q, const float * K, const float * Vt, float * kq, float * kqv,
                            int hd, int H, int Tq, int Tkv, int r) {
    const int h = blockIdx.x*blockDim.x+threadIdx.x;
    if (h >= H) return;
    ptrs[0*H+h] = K+(int64_t) hd*Tkv*(h/r);
    ptrs[1*H+h] = Q+(int64_t) hd*Tq*h;
    ptrs[2*H+h] = kq+(int64_t) Tkv*Tq*h;
    ptrs[3*H+h] = Vt+(int64_t) Tkv*hd*(h/r);
    ptrs[4*H+h] = kq+(int64_t) Tkv*Tq*h;
    ptrs[5*H+h] = kqv+(int64_t) hd*Tq*h;
}

template <typename T>
__global__ void k_heads_to_rows(const float * __restrict__ kqv, T * __restrict__ out, int hd, int Tq, int H, int nv) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x+threadIdx.x, n = (int64_t) hd*Tq*H;
    if (i >= n) return;
    const int d = (int) (i%hd), q = (int) ((i/hd)%Tq), h = (int) (i/((int64_t) hd*Tq)), hpv = H/nv;
    st_out(out, d+(int64_t) hd*(h%hpv)+(int64_t) hd*hpv*(q+(int64_t) Tq*(h/hpv)), kqv[i]);
}

int softmax_nth(int ncols) {
    int nth = WARP;
    while (nth < ncols && nth < 1024) nth *= 2;   // ggml's CUDA_SOFT_MAX_BLOCK_SIZE is 1024
    return nth;
}

template <typename F>
bool dispatch_nth(int nth, F && f) {
    switch (nth) {
        case 32:   f(std::integral_constant<int, 32>{});   return true;
        case 64:   f(std::integral_constant<int, 64>{});   return true;
        case 128:  f(std::integral_constant<int, 128>{});  return true;
        case 256:  f(std::integral_constant<int, 256>{});  return true;
        case 512:  f(std::integral_constant<int, 512>{});  return true;
        case 1024: f(std::integral_constant<int, 1024>{}); return true;
        default:   return false;
    }
}

struct AttnPlan {
    bool fused = false;          // path A reproduces path B for this shape
    int c_warps = 0;             // path C reproduces path B and is faster, with this many warps per CTA
};

// What path B needs beyond the shape: K/V heads (Hkv < H: broadcast) and the mask.
struct AttnExtra {
    int Hkv = 0;
    const void * mask = nullptr;
    bool mask_f16 = false;
    int64_t mstride = 0;
};

// Path B: ggml's own sequence for this attention (cuBLAS F32 GEMMs on a
// ggml-like handle - strided batched, or with K/V broadcast over query heads
// pointer-array batched, as ggml_cuda_mul_mat_cublas picks - then
// soft_max_f32 and the permute).
template <typename T>
bool attn_path_b(cublasHandle_t hd_, const float * Q, const float * K, const float * Vt, T * out,
                 int hd, int H, int Tq, int Tkv, float scale, cudaStream_t st, int nv = 1, const AttnExtra & x = {}) {
    const float alpha = 1.0f, beta = 0.0f;
    const int Hkv = x.Hkv > 0 ? x.Hkv : H;
    float * kq  = (float *) scratch(st, SCR_KQ,  (size_t) Tkv*Tq*H*sizeof(float));
    float * kqv = (float *) scratch(st, SCR_KQV, (size_t) hd*Tq*H*sizeof(float));
    const void ** ptrs = Hkv != H ? (const void **) scratch(st, SCR_PTRS, (size_t) 6*H*sizeof(void *)) : nullptr;
    if (!kq || !kqv || (Hkv != H && !ptrs)) return false;
    if (Hkv != H) {
        k_attn_ptrs<<<(H+31)/32, 32, 0, st>>>(ptrs, Q, K, Vt, kq, kqv, hd, H, Tq, Tkv, H/Hkv);
        if (cublasGemmBatchedEx(hd_, CUBLAS_OP_T, CUBLAS_OP_N, Tkv, Tq, hd,
                &alpha, ptrs+0*H, CUDA_R_32F, hd, ptrs+1*H, CUDA_R_32F, hd,
                &beta, (void **) (ptrs+2*H), CUDA_R_32F, Tkv, H,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
    } else if (cublasGemmStridedBatchedEx(hd_, CUBLAS_OP_T, CUBLAS_OP_N, Tkv, Tq, hd,
            &alpha, K, CUDA_R_32F, hd, (long long) hd*Tkv,
                    Q, CUDA_R_32F, hd, (long long) hd*Tq,
            &beta,  kq, CUDA_R_32F, Tkv, (long long) Tkv*Tq, H,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
    const int rows = Tq*H;
    if (!dispatch_nth(softmax_nth(Tkv), [&](auto N) {
            constexpr int NT = decltype(N)::value;
            if (x.mask && x.mask_f16)
                softmax_rows_emul<NT, half>(kq, rows, Tkv, scale, (const half *) x.mask, Tq, x.mstride, st);
            else
                softmax_rows_emul<NT, float>(kq, rows, Tkv, scale, (const float *) x.mask, Tq, x.mstride, st);
        }))
        return false;
    if (Hkv != H) {
        if (cublasGemmBatchedEx(hd_, CUBLAS_OP_T, CUBLAS_OP_N, hd, Tq, Tkv,
                &alpha, ptrs+3*H, CUDA_R_32F, Tkv, ptrs+4*H, CUDA_R_32F, Tkv,
                &beta, (void **) (ptrs+5*H), CUDA_R_32F, hd, H,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
    } else if (cublasGemmStridedBatchedEx(hd_, CUBLAS_OP_T, CUBLAS_OP_N, hd, Tq, Tkv,
            &alpha, Vt, CUDA_R_32F, Tkv, (long long) Tkv*hd,
                    kq, CUDA_R_32F, Tkv, (long long) Tkv*Tq,
            &beta,  kqv, CUDA_R_32F, hd, (long long) hd*Tq, H,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
    const int64_t n = (int64_t) hd*Tq*H;
    k_heads_to_rows<<<(unsigned) ((n+255)/256), 256, 0, st>>>(kqv, out, hd, Tq, H, nv);
    return true;
}

// Path C: the whole attention in one kernel. cuBLAS's TF32 GEMMs of path B are
// chains of mma.m16n8k8.tf32 in K order on cvt.rna operands; attn_plan checks
// path C against path B per shape. Score rows stay in shared memory.
__device__ __forceinline__ uint32_t tf32_rna(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One CTA of NW warps per (32 queries, head). Q [HD, Tq, H], K [HD, Tkv, Hkv],
// Vt [Tkv, HD, Hkv]; out in attention_rows' layout. mask: row q at
// q*mstride, or null. QK^T: a warp per 8-key tile. PV: a warp per (16
// queries, 8 dims), V^T staged in ATC_KC-key chunks (cp.async, 2 buffers).
constexpr int ATC_QB = 32, ATC_MT = ATC_QB/16, ATC_KC = 32;

__device__ __forceinline__ void cp_async4(void * smem, const void * gmem, bool valid) {
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sa), "l"(gmem), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void cp_async16z(void * smem, const void * gmem, int valid_bytes) {
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(gmem), "r"(valid_bytes));
}

template <int HD, int NW, int NTH, typename T, typename M>
__global__ void __launch_bounds__(NW*WARP)
k_attn_tf32(const float * __restrict__ Q, const float * __restrict__ K, const float * __restrict__ Vt, T * __restrict__ out,
            int H, int Hkv, int Tq, int Tkv, float scale, int nv, const M * __restrict__ mask, int64_t mstride) {
    static_assert(HD % 8 == 0 && NW % ATC_MT == 0, "");
    constexpr int LDQ = HD+4, NDT = HD/8, QREG = HD <= 72, LDV = ATC_KC+4;
    constexpr int PW3 = (ATC_MT*NDT+NW-1)/NW;
    extern __shared__ float sh[];
    const int Tk8 = (Tkv+7)/8*8, nck = (Tkv+ATC_KC-1)/ATC_KC, ldS = max(nck*ATC_KC, Tk8)+4;
    float * sS = sh;                              // [QB][ldS]: scores, then P as TF32 bits
    uint32_t * sQ = (uint32_t *) (sS+ATC_QB*ldS);  // [QB][LDQ] TF32 bits
    float * sV = (float *) (sQ+ATC_QB*LDQ);       // [2][HD][LDV]
    const int h = blockIdx.y, hk = h/(H/Hkv), q0 = blockIdx.x*ATC_QB;
    const int tid = threadIdx.x, warp = tid/WARP, lane = tid%WARP, g = lane/4, t = lane%4;
    const float * Qh = Q+(size_t) HD*Tq*h, * Kh = K+(size_t) HD*Tkv*hk, * Vh = Vt+(size_t) Tkv*HD*hk;

    const bool v16 = Tkv % 4 == 0;
    auto load_v = [&](int c, float * buf) {
        if (v16) {
            for (int i = tid; i < HD*(ATC_KC/4); i += blockDim.x) {
                const int d = i/(ATC_KC/4), kk = (i%(ATC_KC/4))*4, key = c*ATC_KC+kk;
                const int nb = key+4 <= Tkv ? 16 : key < Tkv ? (Tkv-key)*4 : 0;
                cp_async16z(buf+d*LDV+kk, Vh+(size_t) d*Tkv+(nb ? key : 0), nb);
            }
        } else {
            for (int i = tid; i < HD*ATC_KC; i += blockDim.x) {
                const int d = i/ATC_KC, kk = i%ATC_KC, key = c*ATC_KC+kk;
                cp_async4(buf+d*LDV+kk, Vh+(size_t) d*Tkv+(key < Tkv ? key : 0), key < Tkv);
            }
        }
    };
    // The first V^T chunk loads under QK^T and the softmax.
    load_v(0, sV);
    skinny::cp_commit();

    for (int i = tid; i < ATC_QB*HD; i += blockDim.x) {
        const int r = i/HD, d = i%HD;
        sQ[r*LDQ+d] = tf32_rna(q0+r < Tq ? Qh[(size_t) (q0+r)*HD+d] : 0.0f);
    }
    __syncthreads();
    // S = Q K^T: warp w takes tiles of 8 keys w, w + NW, ... for all 32 queries;
    // the head dim in K order.
    {
        uint32_t qf[QREG ? ATC_MT : 1][QREG ? NDT : 1][4];
        if (QREG)
#pragma unroll
            for (int mt = 0; mt < (QREG ? ATC_MT : 1); ++mt)
#pragma unroll
                for (int s8 = 0; s8 < (QREG ? NDT : 1); ++s8) {
                    const uint32_t * qa = sQ+(mt*16+g)*LDQ+s8*8, * qb = qa+8*LDQ;
                    qf[mt][s8][0] = qa[t]; qf[mt][s8][1] = qb[t]; qf[mt][s8][2] = qa[t+4]; qf[mt][s8][3] = qb[t+4];
                }
        for (int kt = warp; kt < Tk8/8; kt += NW) {
            const int key = kt*8+g;
            const bool kv = key < Tkv;
            const float * kr = Kh+(size_t) (kv ? key : 0)*HD;
            float acc[ATC_MT][4];
#pragma unroll
            for (int mt = 0; mt < ATC_MT; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.0f;
            float kb[NDT][2];
#pragma unroll
            for (int s8 = 0; s8 < NDT; ++s8) { kb[s8][0] = kv ? kr[s8*8+t] : 0.0f; kb[s8][1] = kv ? kr[s8*8+t+4] : 0.0f; }
#pragma unroll
            for (int s8 = 0; s8 < NDT; ++s8) {
                const uint32_t b0 = tf32_rna(kb[s8][0]), b1 = tf32_rna(kb[s8][1]);
#pragma unroll
                for (int mt = 0; mt < ATC_MT; ++mt) {
                    if (QREG) {
                        const uint32_t (&f)[4] = qf[QREG ? mt : 0][QREG ? s8 : 0];
                        mma_tf32(acc[mt], f[0], f[1], f[2], f[3], b0, b1);
                    } else {
                        const uint32_t * qa = sQ+(mt*16+g)*LDQ+s8*8, * qb = qa+8*LDQ;
                        mma_tf32(acc[mt], qa[t], qb[t], qa[t+4], qb[t+4], b0, b1);
                    }
                }
            }
#pragma unroll
            for (int mt = 0; mt < ATC_MT; ++mt) {
                float * s0 = sS+(mt*16+g)*ldS+kt*8+2*t, * s1 = s0+8*ldS;
                s0[0] = acc[mt][0]; s0[1] = acc[mt][1]; s1[0] = acc[mt][2]; s1[1] = acc[mt][3];
            }
        }
    }
    __syncthreads();
    // soft_max_f32 per query row; P stored as TF32 bits, zero past Tkv. Rows
    // of up to 256 columns in registers (lane l holds columns 32j + l, virtual
    // thread 32j + l of ggml's NTH-thread block).
    constexpr int NPL = NTH > WARP ? NTH/WARP : 1;
    if constexpr (NPL <= 8) {
    for (int r = warp; r < ATC_QB; r += NW) {
        float * row = sS+r*ldS;
        if (q0+r < Tq) {
            const M * mr = mask ? mask+(int64_t) (q0+r)*mstride : nullptr;
            float v[NPL];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < NPL; ++j) {
                const int k = j*WARP+lane;
                v[j] = 0.0f;
                if (k < Tkv) {
                    v[j] = mr ? __fmaf_rn(row[k], scale, (float) mr[k]) : row[k]*scale + 0.0f;
                    mx = fmaxf(mx, v[j]);
                }
            }
#pragma unroll
            for (int off = WARP/2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, WARP));
            float p[NPL];
#pragma unroll
            for (int j = 0; j < NPL; ++j) {
                const bool in = j*WARP+lane < Tkv;
                v[j] = in ? expf(v[j] - mx) : 0.0f;
                p[j] = in ? 0.0f + v[j] : 0.0f;
            }
            const float sum = NTH <= WARP ? warp_sum(p[0]) : warp_sum(warp_sums_t<NPL>(p, lane, 0.0f));
            const float inv_sum = 1.0f / sum;
#pragma unroll
            for (int j = 0; j < NPL; ++j) {
                const int k = j*WARP+lane;
                if (k < Tkv) ((uint32_t *) row)[k] = tf32_rna(v[j] * inv_sum);
            }
        } else {
            for (int k = lane; k < Tkv; k += WARP) ((uint32_t *) row)[k] = 0u;
        }
        for (int k = Tkv+lane; k < nck*ATC_KC; k += WARP) ((uint32_t *) row)[k] = 0u;
    }
    } else {
    // Longer rows: as k_softmax_rows_loop (measured faster there).
    for (int r = warp; r < ATC_QB; r += NW) {
        float * row = sS+r*ldS;
        if (q0+r < Tq) {
            const M * mr = mask ? mask+(int64_t) (q0+r)*mstride : nullptr;
            float mx = -INFINITY;
            for (int k = lane; k < Tkv; k += WARP) {
                const float v = mr ? __fmaf_rn(row[k], scale, (float) mr[k]) : row[k]*scale + 0.0f;
                row[k] = v;
                mx = fmaxf(mx, v);
            }
#pragma unroll
            for (int off = WARP/2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, WARP));
            __syncwarp();
            for (int k = lane; k < Tkv; k += WARP) row[k] = expf(row[k] - mx);
            __syncwarp();
            const float inv_sum = 1.0f / softmax_sum_emul<NTH>(row, Tkv, lane);
            __syncwarp();
            for (int k = lane; k < Tkv; k += WARP) ((uint32_t *) row)[k] = tf32_rna(row[k] * inv_sum);
        } else {
            for (int k = lane; k < Tkv; k += WARP) ((uint32_t *) row)[k] = 0u;
        }
        for (int k = Tkv+lane; k < nck*ATC_KC; k += WARP) ((uint32_t *) row)[k] = 0u;
    }
    }
    // O = P V: warp w takes (its 16 queries, tiles of 8 dims w/MT, w/MT + NW/MT, ...); keys in K order.
    const int mt = warp%ATC_MT;
    float o[PW3][4];
#pragma unroll
    for (int j = 0; j < PW3; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    const uint32_t * pa = (const uint32_t *) sS+(mt*16+g)*ldS, * pb = pa+8*ldS;
    for (int c = 0; c < nck; ++c) {
        if (c+1 < nck) load_v(c+1, sV+((c+1)%2)*HD*LDV);
        skinny::cp_commit();
        skinny::cp_wait<1>();
        __syncthreads();
        const float * buf = sV+(c%2)*HD*LDV;
#pragma unroll
        for (int kk = 0; kk < ATC_KC; kk += 8) {
            const int k = c*ATC_KC+kk;
            const uint32_t a0 = pa[k+t], a1 = pb[k+t], a2 = pa[k+t+4], a3 = pb[k+t+4];
#pragma unroll
            for (int j = 0; j < PW3; ++j) {
                const int dt = warp/ATC_MT+j*(NW/ATC_MT);
                if (dt < NDT) {
                    const float * vr = buf+(dt*8+g)*LDV+kk;
                    mma_tf32(o[j], a0, a1, a2, a3, tf32_rna(vr[t]), tf32_rna(vr[t+4]));
                }
            }
        }
        __syncthreads();
    }
    const int hpv = H/nv;
#pragma unroll
    for (int j = 0; j < PW3; ++j) {
        const int dt = warp/ATC_MT+j*(NW/ATC_MT);
        if (dt >= NDT) break;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int q = q0+mt*16+g+half*8;
            if (q >= Tq) continue;
            T * orow = out+(size_t) HD*(h%hpv)+(size_t) HD*hpv*(q+(size_t) Tq*(h/hpv));
            st_out(orow, dt*8+2*t,   o[j][2*half]);
            st_out(orow, dt*8+2*t+1, o[j][2*half+1]);
        }
    }
}

size_t attn_tf32_smem(int HD, int Tkv) {
    const int nck = (Tkv+ATC_KC-1)/ATC_KC;
    return ((size_t) ATC_QB*(std::max(nck*ATC_KC, (Tkv+7)/8*8)+4)+(size_t) ATC_QB*(HD+4)+(size_t) 2*HD*(ATC_KC+4))*sizeof(float);
}

template <int NW, typename T, typename M>
bool attn_path_c_nw(const float * Q, const float * K, const float * Vt, T * out, int hd, int H, int Hkv, int Tq, int Tkv,
                    float scale, cudaStream_t st, int nv, const M * mask, int64_t mstride) {
    const size_t shm = attn_tf32_smem(hd, Tkv);
    const dim3 grid((Tq+ATC_QB-1)/ATC_QB, H);
    bool ok = false;
    auto go = [&](auto HDc) {
        constexpr int HD = decltype(HDc)::value;
        ok = dispatch_nth(softmax_nth(Tkv), [&](auto N) {
            constexpr int NT = decltype(N)::value;
            auto kern = k_attn_tf32<HD, NW, NT, T, M>;
            if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm) != cudaSuccess) return;
            kern<<<grid, NW*WARP, shm, st>>>(Q, K, Vt, out, H, Hkv, Tq, Tkv, scale, nv, mask, mstride);
        }) && cudaGetLastError() == cudaSuccess;
    };
    switch (hd) {
        case 64:  go(std::integral_constant<int, 64>{});  break;
        case 72:  go(std::integral_constant<int, 72>{});  break;
        case 128: go(std::integral_constant<int, 128>{}); break;
        case 256: go(std::integral_constant<int, 256>{}); break;
        default:  return false;
    }
    return ok;
}

// nw: warps per CTA (8 or 16).
template <typename T>
bool attn_path_c(int nw, const float * Q, const float * K, const float * Vt, T * out, int hd, int H, int Tq, int Tkv,
                 float scale, cudaStream_t st, int nv, const AttnExtra & x) {
    const int Hkv = x.Hkv > 0 ? x.Hkv : H;
    if (Tkv > 1024 || H % Hkv) return false;
    auto run = [&](const auto * mask) {
        return nw == 8 ? attn_path_c_nw<8>(Q, K, Vt, out, hd, H, Hkv, Tq, Tkv, scale, st, nv, mask, x.mstride)
                       : attn_path_c_nw<16>(Q, K, Vt, out, hd, H, Hkv, Tq, Tkv, scale, st, nv, mask, x.mstride);
    };
    return x.mask && x.mask_f16 ? run((const half *) x.mask) : run((const float *) x.mask);
}

template <typename T>
bool attn_path_a(const float * Q, const float * K, const float * Vt, T * out, int hd, int H, int Tq, int Tkv, float scale,
                 cudaStream_t st, int nv = 1) {
    const int nth = softmax_nth(Tkv);
    const size_t shm = ((size_t) Tkv*(hd+1)+(size_t) hd*Tkv+(size_t) ATTN_QPB*hd+(size_t) ATTN_QPB*nth)*sizeof(float);
    if (shm > 48*1024 || Tkv > nth) return false;
    return dispatch_nth(nth, [&](auto N) {
        constexpr int NT = decltype(N)::value;
        k_attn_exact<NT><<<dim3(H, (Tq+ATTN_QPB-1)/ATTN_QPB), ATTN_QPB*WARP, shm, st>>>(Q, K, Vt, out, hd, H, Tq, Tkv, scale, nv);
    });
}

// Path A (no K/V broadcast, no mask) is used where it reproduces path B; path C
// where it reproduces path B and is faster. Both are checked on random inputs
// (and a random 0 / -inf mask), outside graph capture.
AttnPlan * attn_plan(cublasHandle_t h, int hd, int H, int Tq, int Tkv, float scale, cudaStream_t st, int Hkv = 0,
                     bool masked = false, bool mask_f16 = false) {
    static std::mutex mu;
    static std::map<std::tuple<int, int, int, int, int, float, int, bool, bool>, AttnPlan *> plans;
    const int dev = stream_ctx(st).dev;
    std::lock_guard<std::mutex> lk(mu);
    if (Hkv <= 0) Hkv = H;
    auto it = plans.find({dev, hd, H, Tq, Tkv, scale, Hkv, masked, mask_f16});
    if (it != plans.end()) return it->second;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cap);
    if (cap != cudaStreamCaptureStatusNone) return nullptr;   // first use must be eager
    // A prompt that changes length every call brings new Tq / Tkv. A path A or
    // C choice made for the nearest shape (same heads, scale and mask; Tq and
    // Tkv each within 64) is checked against path B below and kept without
    // timing.
    const AttnPlan * near = nullptr;
    int near_q = 0, near_kv = 0;
    if (vla::env_flag("VLA_GEMM_REUSE", true))
        for (const auto & [key, q] : plans) {
            const auto [d_, hd_, H_, tq, tkv, sc, hkv, mk, m16] = key;
            const int dist = std::abs(tq-Tq)+std::abs(tkv-Tkv), best = std::abs(near_q-Tq)+std::abs(near_kv-Tkv);
            if ((q->fused || q->c_warps) && d_ == dev && hd_ == hd && H_ == H && sc == scale && hkv == Hkv && mk == masked && m16 == mask_f16 &&
                std::abs(tq-Tq) <= 64 && std::abs(tkv-Tkv) <= 64 && (!near || dist < best)) {
                near = q;
                near_q = tq;
                near_kv = tkv;
            }
        }
    AttnPlan * p = new AttnPlan();
    plans[{dev, hd, H, Tq, Tkv, scale, Hkv, masked, mask_f16}] = p;
    const auto t_start = std::chrono::steady_clock::now();
    const size_t nq = (size_t) hd*Tq*H, nk = (size_t) hd*Tkv*Hkv, nm = (size_t) Tkv*Tq;
    auto trace = [&](const char * what) {
        if (trace_level() >= 1)
            std::fprintf(stderr, "vla(attn_exact): hd=%d H=%d Hkv=%d Tq=%d Tkv=%d%s -> %s, %.0f ms\n", hd, H, Hkv, Tq, Tkv,
                         masked ? " masked" : "", what,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t_start).count());
    };
    // Random inputs, path B as the reference, bitwise comparisons.
    std::vector<float> hq(nq), hk(nk), hv(nk), hm(masked ? nm : 0), a(nq), b(nq);
    uint32_t seed = 777u;
    auto rnd = [&] { seed = seed*1664525u+1013904223u; return ((seed >> 8)*(1.0f/16777216.0f))*4.0f-2.0f; };
    for (auto & v : hq) v = rnd();
    for (auto & v : hk) v = rnd();
    for (auto & v : hv) v = rnd();
    for (size_t i = 0; i < hm.size(); ++i) hm[i] = (i % Tkv) != 0 && rnd() < -1.0f ? -INFINITY : 0.0f;
    std::vector<half> hm16(hm.size());
    for (size_t i = 0; i < hm.size(); ++i) hm16[i] = __float2half(hm[i]);
    float * dq = nullptr, * dk = nullptr, * dv = nullptr, * da = nullptr, * db = nullptr; void * dm = nullptr;
    bool good = cudaMalloc(&dq, nq*4) == cudaSuccess && cudaMalloc(&dk, nk*4) == cudaSuccess &&
                cudaMalloc(&dv, nk*4) == cudaSuccess && cudaMalloc(&da, nq*4) == cudaSuccess && cudaMalloc(&db, nq*4) == cudaSuccess &&
                (!masked || cudaMalloc(&dm, nm*4) == cudaSuccess);
    good = good && cudaMemcpyAsync(dq, hq.data(), nq*4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
                   cudaMemcpyAsync(dk, hk.data(), nk*4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
                   cudaMemcpyAsync(dv, hv.data(), nk*4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
                   (!masked || (mask_f16 ? cudaMemcpyAsync(dm, hm16.data(), nm*2, cudaMemcpyHostToDevice, st)
                                         : cudaMemcpyAsync(dm, hm.data(), nm*4, cudaMemcpyHostToDevice, st)) == cudaSuccess);
    AttnExtra x;
    x.Hkv = Hkv; x.mask = masked ? dm : nullptr; x.mask_f16 = mask_f16; x.mstride = Tkv;
    good = good && attn_path_b(h, dq, dk, dv, db, hd, H, Tq, Tkv, scale, st, 1, x) &&
                   cudaMemcpyAsync(b.data(), db, nq*4, cudaMemcpyDeviceToHost, st) == cudaSuccess &&
                   cudaStreamSynchronize(st) == cudaSuccess;
    auto same_as_b = [&](bool ran) {
        return ran && cudaMemcpyAsync(a.data(), da, nq*4, cudaMemcpyDeviceToHost, st) == cudaSuccess &&
               cudaStreamSynchronize(st) == cudaSuccess && std::memcmp(a.data(), b.data(), nq*4) == 0;
    };
    bool reused = false;
    if (good && near) {
        if (near->fused)
            p->fused = same_as_b(attn_path_a(dq, dk, dv, da, hd, H, Tq, Tkv, scale, st));
        else if (near->c_warps)
            p->c_warps = same_as_b(attn_path_c(near->c_warps, dq, dk, dv, da, hd, H, Tq, Tkv, scale, st, 1, x)) ? near->c_warps : 0;
        reused = p->fused == near->fused && p->c_warps == near->c_warps;
        cudaGetLastError();
    }
    if (good && !reused && Hkv == H && !masked && vla::env_flag("VLA_ATTN_EXACT_FUSED", true))
        p->fused = same_as_b(attn_path_a(dq, dk, dv, da, hd, H, Tq, Tkv, scale, st));
    cudaGetLastError();
    // Path C: bit-exact configurations, then timed against path B.
    if (good && !reused && !p->fused && vla::env_flag("VLA_ATTN_TF32", true)) {
        std::vector<int> exact;
        for (int nw : {8, 16}) {
            if (same_as_b(attn_path_c(nw, dq, dk, dv, da, hd, H, Tq, Tkv, scale, st, 1, x))) exact.push_back(nw);
            cudaGetLastError();
        }
        if (!exact.empty()) {
            auto run_b = [&] { attn_path_b(h, dq, dk, dv, db, hd, H, Tq, Tkv, scale, st, 1, x); };
            warm_clocks(st, run_b);
            std::vector<float> best(exact.size()+1, 1e30f);
            for (int round = 0; round < 4; ++round) {
                best[0] = std::min(best[0], time_us(st, run_b));
                for (size_t i = 0; i < exact.size(); ++i)
                    best[i+1] = std::min(best[i+1], time_us(st, [&] {
                        attn_path_c(exact[i], dq, dk, dv, da, hd, H, Tq, Tkv, scale, st, 1, x); }));
            }
            size_t pick = 0;
            for (size_t i = 1; i < best.size(); ++i)
                if (best[i] < best[pick]) pick = i;
            if (pick > 0) p->c_warps = exact[pick-1];
            if (trace_level() >= 1)
                std::fprintf(stderr, "vla(attn_exact):   path B %.1f us, path C (%zu bit-exact) best %.1f us\n",
                             best[0], exact.size(), *std::min_element(best.begin()+1, best.end()));
        }
    }
    if (reused)
        trace("the choice of a nearby shape (bit-exact)");
    else
        trace(p->fused ? "fused kernel (bit-exact)" : p->c_warps ? "TF32 fused kernel (bit-exact)" : "cuBLAS chain");
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(da); cudaFree(db); cudaFree(dm);
    cudaGetLastError();
    return p;
}

bool attn_exact(const ggml_tensor * dst, const vla::FusedSpec & s, cudaStream_t st) {
    const ggml_tensor * Qt = dst->src[0], * Kt = dst->src[1], * Vtt = dst->src[2], * Mt = dst->src[3];
    cublasHandle_t h = ggml_like_handle(st);
    if (!h) return false;
    const int hd = s.hd, H = s.heads, Tq = s.S, Tkv = (int) Kt->ne[1], nv = s.nv > 0 ? s.nv : 1;
    AttnExtra x;
    x.Hkv = (int) Kt->ne[2];
    if (Mt) {
        x.mask = Mt->data;
        x.mask_f16 = Mt->type == GGML_TYPE_F16;
        x.mstride = (int64_t) (Mt->nb[1]/ggml_element_size(Mt));
    }
    const AttnPlan * p = attn_plan(h, hd, H, Tq, Tkv, s.eps, st, x.Hkv, Mt != nullptr, x.mask_f16);
    const float * Q = (const float *) Qt->data, * K = (const float *) Kt->data, * V = (const float *) Vtt->data;
    const bool bf = dst->type == GGML_TYPE_BF16;
    // No plan: this shape's first use is inside graph capture. Path B, ggml's
    // own chain, still runs if the stream's scratch is large enough.
    if (p && p->c_warps)
        return bf ? attn_path_c(p->c_warps, Q, K, V, (__nv_bfloat16 *) dst->data, hd, H, Tq, Tkv, s.eps, st, nv, x)
                  : attn_path_c(p->c_warps, Q, K, V, (float *) dst->data, hd, H, Tq, Tkv, s.eps, st, nv, x);
    if (p && p->fused)
        return bf ? attn_path_a(Q, K, V, (__nv_bfloat16 *) dst->data, hd, H, Tq, Tkv, s.eps, st, nv)
                  : attn_path_a(Q, K, V, (float *) dst->data, hd, H, Tq, Tkv, s.eps, st, nv);
    return bf ? attn_path_b(h, Q, K, V, (__nv_bfloat16 *) dst->data, hd, H, Tq, Tkv, s.eps, st, nv, x)
              : attn_path_b(h, Q, K, V, (float *) dst->data, hd, H, Tq, Tkv, s.eps, st, nv, x);
}

// ggml's F32 -> BF16 operand conversion (convert_unary_cont_vec4: round to
// nearest even via __float2bfloat16), vectorized by 4.
__global__ void k_f32_to_bf16(const float * __restrict__ x, __nv_bfloat16 * __restrict__ y, int64_t n) {
    const int64_t i = ((int64_t) blockIdx.x*blockDim.x+threadIdx.x)*4;
    if (i+3 < n) {
        const float4 v = *(const float4 *) (x+i);
        y[i] = __float2bfloat16(v.x); y[i+1] = __float2bfloat16(v.y);
        y[i+2] = __float2bfloat16(v.z); y[i+3] = __float2bfloat16(v.w);
    } else {
        for (int64_t j = i; j < n; ++j) y[j] = __float2bfloat16(x[j]);
    }
}

bool gemm_views(const ggml_tensor * dst, const vla::FusedSpec & s, cudaStream_t st) {
    const ggml_tensor * W = dst->src[0], * X = dst->src[1];
    if (W->type != GGML_TYPE_BF16 || (X->type != GGML_TYPE_BF16 && X->type != GGML_TYPE_F32)) return false;
    cublasHandle_t h = ggml_like_handle(st);
    if (!h) return false;
    const void * xdata = X->data;
    if (X->type == GGML_TYPE_F32) {
        // The stream's BF16 staging buffer.
        const int64_t n = ggml_nelements(X);
        __nv_bfloat16 * xb = (__nv_bfloat16 *) scratch(st, SCR_BF16, (size_t) n*sizeof(__nv_bfloat16));
        if (!xb) return false;
        k_f32_to_bf16<<<(unsigned) ((n/4+256)/256), 256, 0, st>>>((const float *) X->data, xb, n);
        xdata = xb;
    }
    const int K = (int) W->ne[0], N = (int) W->ne[1], S = s.S;
    const LtPlan * p = lt_plan(h, N, S, K, s.nv, st);
    if (p->sk)
        return p->sk((const __nv_bfloat16 *) W->data, (const __nv_bfloat16 *) xdata, (float *) dst->data, N, S, K, s.nv, st);
    if (p->ok)
        return lt_run(*p, p->algo, W->data, xdata, (float *) dst->data, st);
    const float alpha = 1.0f, beta = 0.0f;
    for (int v = 0; v < s.nv; ++v) {
        const __nv_bfloat16 * xb = (const __nv_bfloat16 *) xdata+(int64_t) v*S*K;
        float * o = (float *) dst->data+(int64_t) v*S*N;
        if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, S, K,
                         &alpha, W->data, CUDA_R_16BF, K,
                                 xb,      CUDA_R_16BF, K,
                         &beta,  o,       CUDA_R_32F,  N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
            return false;
    }
    return true;
}

const float * fptr(const ggml_tensor * t) {
    return t ? (const float *) t->data : nullptr;
}

int packed_part_host(int part_mask, int p) {
    return part_mask ? __builtin_popcount(part_mask & ((1 << p)-1)) : p;
}

template <typename T>
void launch_qkv(const vla::FusedSpec & s, const ggml_tensor * dst, T * out, cudaStream_t st) {
    const ggml_tensor * a = dst->src[0];
    const int n_tok = s.S*s.nv, D = s.hd*s.heads, all = (1 << s.n_parts)-1;
    const int wmask = s.part_mask ? s.part_mask : all;
    if (wmask & ~s.vt_mask)
        k_qkv_rows<<<n_tok, 256, 0, st>>>(fptr(a), fptr(dst->src[1]), fptr(dst->src[2]), fptr(dst->src[3]),
                                          out, s.n_parts, s.hd, s.heads, s.S, s.vt_mask, s.rope_mask, s.part_mask);
    for (int p = 0; p < s.n_parts; ++p)
        if (((wmask & s.vt_mask) >> p) & 1)
            k_qkv_vt<<<dim3((D+31)/32, (n_tok+31)/32), dim3(32, 8), 0, st>>>(fptr(a), fptr(dst->src[1]), out,
                                                                          s.n_parts, p, s.hd, s.heads, s.S, n_tok,
                                                                          packed_part_host(s.part_mask, p));
}

// RmsMul (MODE 0) / RmsModulate (MODE 1) with ggml rms_norm_f32's block size:
// 1024 threads from 1024 columns up, else 256.
template <int MODE>
void launch_rms(const ggml_tensor * dst, const vla::FusedSpec & s, cudaStream_t st) {
    const ggml_tensor * a = dst->src[0];
    const float * x = fptr(a), * p1 = fptr(dst->src[1]), * p2 = MODE ? fptr(dst->src[2]) : nullptr;
    const int ncols = (int) a->ne[0], nrows = (int) ggml_nrows(a), g = (nrows+3)/4;
    auto go = [&](auto * o) {
        if (ncols > 1024 && ncols <= 2048) k_rms_mul_reg<64, 1024, std::remove_pointer_t<decltype(o)>, MODE><<<g, 128, 0, st>>>(x, p1, o, ncols, nrows, s.eps, p2);
        else if (ncols == 1024)            k_rms_mul_reg<32, 1024, std::remove_pointer_t<decltype(o)>, MODE><<<g, 128, 0, st>>>(x, p1, o, ncols, nrows, s.eps, p2);
        else if (ncols < 1024)             k_rms_mul_warp<256, std::remove_pointer_t<decltype(o)>, MODE><<<g, 128, 0, st>>>(x, p1, o, ncols, nrows, s.eps, p2);
        else                               k_rms_mul_warp<1024, std::remove_pointer_t<decltype(o)>, MODE><<<g, 128, 0, st>>>(x, p1, o, ncols, nrows, s.eps, p2);
    };
    if (dst->type == GGML_TYPE_BF16) go((__nv_bfloat16 *) dst->data);
    else                             go((float *) dst->data);
}

bool contiguous_f32(const ggml_tensor * t) {
    return !t || (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
}

extern "C" bool vla_cuda_fused_forward(ggml_tensor * dst, void * stream_v) {
    const vla::FusedSpec * s = spec_of(dst);
    if (!s) return false;
    if (s->kind == vla::FusedKind::GemmViews)
        return gemm_views(dst, *s, (cudaStream_t) stream_v);
    if (s->kind == vla::FusedKind::AttnExact)
        return attn_exact(dst, *s, (cudaStream_t) stream_v);
    for (int i = 0; i < 4; ++i)
        if (!contiguous_f32(dst->src[i]) &&
            !(s->kind == vla::FusedKind::Patchify && i == 0 && dst->src[0]->type == GGML_TYPE_I8 && ggml_is_contiguous(dst->src[0])) &&
            !(s->kind == vla::FusedKind::QkNormRope && i == 2 && dst->src[2]->type == GGML_TYPE_I32 && ggml_is_contiguous(dst->src[2])) &&
            !(s->kind == vla::FusedKind::HeadsRope && i == 1 && dst->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(dst->src[1])))
            return false;
    if (s->kind == vla::FusedKind::QkvHeads && (s->rope_mask & s->vt_mask)) return false;   // not implemented
    cudaStream_t st = (cudaStream_t) stream_v;
    const ggml_tensor * a = dst->src[0];
    float * out = (float *) dst->data;
    constexpr int B = 256;
    switch (s->kind) {
        case vla::FusedKind::QkvHeads:
            if (dst->type == GGML_TYPE_F16) launch_qkv(*s, dst, (half *) dst->data, st);
            else                            launch_qkv(*s, dst, out, st);
            break;
        case vla::FusedKind::BiasRes: {
            const int64_t n = ggml_nelements(a);
            k_bias_res<<<(unsigned) ((n+B-1)/B), B, 0, st>>>(fptr(a), fptr(dst->src[1]), fptr(dst->src[2]), out, n, (int) a->ne[0]);
            break;
        }
        case vla::FusedKind::BiasGelu: {
            const int64_t n = ggml_nelements(a);
            const unsigned g = (unsigned) ((n+B-1)/B);
            if (dst->type == GGML_TYPE_BF16)
                k_bias_gelu<<<g, B, 0, st>>>(fptr(a), fptr(dst->src[1]), (__nv_bfloat16 *) dst->data, n, (int) a->ne[0]);
            else
                k_bias_gelu<<<g, B, 0, st>>>(fptr(a), fptr(dst->src[1]), out, n, (int) a->ne[0]);
            break;
        }
        case vla::FusedKind::LnAffine:
            launch_ln_t<LN_AFFINE>(dst, fptr(a), fptr(dst->src[1]), fptr(dst->src[2]), (int) a->ne[0], (int) ggml_nrows(a), s->eps, st);
            break;
        case vla::FusedKind::LnModulate:
            launch_ln_t<LN_MOD>(dst, fptr(a), fptr(dst->src[1]), fptr(dst->src[2]), (int) a->ne[0], (int) ggml_nrows(a), s->eps, st);
            break;
        case vla::FusedKind::LnPlain:
            launch_ln_t<LN_PLAIN>(dst, fptr(a), nullptr, nullptr, (int) a->ne[0], (int) ggml_nrows(a), s->eps, st);
            break;
        case vla::FusedKind::RmsMul:
            launch_rms<0>(dst, *s, st);
            break;
        case vla::FusedKind::RmsModulate:
            launch_rms<1>(dst, *s, st);
            break;
        case vla::FusedKind::GeluMul: {
            const int64_t n = ggml_nelements(a);
            const unsigned g = (unsigned) ((n+B-1)/B);
            if (dst->type == GGML_TYPE_BF16) k_gelu_mul<<<g, B, 0, st>>>(fptr(a), fptr(dst->src[1]), (__nv_bfloat16 *) dst->data, n);
            else                             k_gelu_mul<<<g, B, 0, st>>>(fptr(a), fptr(dst->src[1]), out, n);
            break;
        }
        case vla::FusedKind::GluAct:
            if (!launch_glu_act(*s, dst, st)) return false;
            break;
        case vla::FusedKind::NormAct:
            if (!launch_norm_act(*s, dst, st)) return false;
            break;
        case vla::FusedKind::BiasGeluErf: {
            const int64_t n = ggml_nelements(a);
            const unsigned g = (unsigned) ((n+B-1)/B);
            if (dst->type == GGML_TYPE_BF16)
                k_bias_gelu_erf<<<g, B, 0, st>>>(fptr(a), fptr(dst->src[1]), (__nv_bfloat16 *) dst->data, n, (int) a->ne[0]);
            else
                k_bias_gelu_erf<<<g, B, 0, st>>>(fptr(a), fptr(dst->src[1]), out, n, (int) a->ne[0]);
            break;
        }
        case vla::FusedKind::BiasGateRes: {
            const int64_t n = ggml_nelements(a);
            k_bias_gate_res<<<(unsigned) ((n+B-1)/B), B, 0, st>>>(fptr(a), fptr(dst->src[1]), fptr(dst->src[2]), fptr(dst->src[3]),
                                                                out, n, (int) a->ne[0]);
            break;
        }
        case vla::FusedKind::GateRes: {
            const int64_t n = ggml_nelements(a);
            k_gate_res<<<(unsigned) ((n+B-1)/B), B, 0, st>>>(fptr(a), fptr(dst->src[1]), fptr(dst->src[2]), out, n, (int) a->ne[0]);
            break;
        }
        case vla::FusedKind::HeadsRope: {
            if (dst->type != GGML_TYPE_F32) return false;
            const int hd = s->hd, H = s->heads, T = s->S;
            const int Tp = dst->src[2] ? (int) (ggml_nelements(dst->src[2])/((int64_t) hd*H)) : 0;
            const int32_t * pos = dst->src[1] ? (const int32_t *) dst->src[1]->data : nullptr;
            if (s->vt_mask) {
                if (pos) return false;   // RoPE on a transposed operand: not needed, not implemented
                k_heads_vt<<<dim3((hd*H+31)/32, (Tp+T+31)/32), dim3(32, 8), 0, st>>>(fptr(a), fptr(dst->src[2]), out, hd, H, T, Tp,
                                                                                    fptr(dst->src[3]));
            } else {
                // ggml computes theta_scale on the host the same way.
                const float theta_scale = powf(s->freq_base, -2.0f / hd);
                const int64_t n = (int64_t) (hd/2)*H*(Tp+T);
                k_heads_rope<<<(unsigned) ((n+B-1)/B), B, 0, st>>>(fptr(a), pos, fptr(dst->src[2]), out, hd, H, T, Tp, theta_scale,
                                                                    fptr(dst->src[3]));
            }
            break;
        }
        case vla::FusedKind::Patchify: {
            const int64_t n = ggml_nelements(dst);
            const unsigned g = (unsigned) ((n+B-1)/B);
            if (a->type == GGML_TYPE_I8)
                k_patchify<<<g, B, 0, st>>>((const int8_t *) a->data, out, s->side, s->patch, s->temporal, s->merge,
                                            s->mean[0], s->mean[1], s->mean[2], s->std[0], s->std[1], s->std[2], n);
            else
                k_patchify<<<g, B, 0, st>>>((const float *) a->data, out, s->side, s->patch, s->temporal, s->merge,
                                            s->mean[0], s->mean[1], s->mean[2], s->std[0], s->std[1], s->std[2], n);
            break;
        }
        case vla::FusedKind::QkNormRope: {
            // ggml computes theta_scale on the host the same way.
            const float theta_scale = powf(s->freq_base, -2.0f / s->hd);
            const int rows = s->heads*s->S, g = (rows+3)/4;
            if (dst->type == GGML_TYPE_F16)
                k_qk_norm_rope<<<g, 128, 0, st>>>(fptr(a), fptr(dst->src[1]), (const int32_t *) dst->src[2]->data, (half *) dst->data,
                                                  s->hd, s->heads, s->S, s->eps, theta_scale, s->sect[0], s->sect[1], s->sect[2], s->sect[3]);
            else
                k_qk_norm_rope<<<g, 128, 0, st>>>(fptr(a), fptr(dst->src[1]), (const int32_t *) dst->src[2]->data, out,
                                                  s->hd, s->heads, s->S, s->eps, theta_scale, s->sect[0], s->sect[1], s->sect[2], s->sect[3]);
            break;
        }
        case vla::FusedKind::SiluMul: {
            const int64_t n = ggml_nelements(a);
            const unsigned g = (unsigned) ((n+B-1)/B);
            if (dst->type == GGML_TYPE_BF16) k_silu_mul<<<g, B, 0, st>>>(fptr(a), fptr(dst->src[1]), (__nv_bfloat16 *) dst->data, n);
            else                             k_silu_mul<<<g, B, 0, st>>>(fptr(a), fptr(dst->src[1]), out, n);
            break;
        }
        default:
            return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

namespace vla {

void cuda_register_fused_ops() {
    cuda_ext_add_handler(vla_cuda_fused_forward);
}

}  // namespace vla
