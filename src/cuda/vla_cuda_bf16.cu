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

// BF16 activation kernels for the CUDA backend.
//
// ggml's CUDA backend implements these ops for F32 (and some for F16), but not
// for BF16: binbcast has no BF16 instantiation, and norm / rms_norm / scale
// assert F32 outright. An all-BF16 activation graph therefore has nowhere to
// run, and ggml offers no way to register kernels for built-in ops
// (GGML_OP_CUSTOM is CPU-only).
//
// So the fetched ggml carries one hook - a function pointer consulted at the
// top of ggml_cuda_compute_forward - and everything else lives here, in tree,
// as ordinary first-party code. See scripts/patch_ggml_cuda_ext_hook.py for the
// hook itself; it is the entire ggml modification.
//
// This file deliberately depends only on the PUBLIC ggml header. It never
// includes ggml-cuda internals (common.cuh and friends), so a llama.cpp bump
// cannot break it the way an anchored source patch would.
//
// Every entry point returns false for anything it does not handle, and ggml
// then runs the op exactly as it would have. Nothing here changes the F32 path.
//
// Accumulation is float throughout: only operand and result *storage* is BF16,
// never a reduction.

#include "ggml.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>

// Must match the typedef the hook patch inserts into ggml-cuda.cu.
extern "C" {
typedef bool (*ggml_cuda_ext_forward_t)(struct ggml_tensor * dst, void * stream);
extern ggml_cuda_ext_forward_t ggml_cuda_ext_forward;

typedef bool (*ggml_cuda_ext_fused_binbcast_t)(struct ggml_tensor * dst, int n_fuse, void * stream);
extern ggml_cuda_ext_fused_binbcast_t ggml_cuda_ext_fused_binbcast;
}

namespace {

constexpr int BLOCK = 256;

inline __device__ float bf2f(const __nv_bfloat16 v) { return __bfloat162float(v); }
inline __device__ __nv_bfloat16 f2bf(const float v) { return __float2bfloat16(v); }

// ---------------------------------------------------------------------------
// elementwise binary: dst = op(src0, src1), src1 broadcast per ggml_can_repeat
// ---------------------------------------------------------------------------
//
// ggml's rule is index-wise modulo on each dimension, so a [N,1,1,1] bias and a
// [N,M,1,1] activation combine the way the F32 path combines them. src1 may be
// F32 (a resident bias or norm weight) or BF16 (another activation).

enum class BinOp { Add, Mul };

template <BinOp op, typename S1>
__global__ void k_bin_bcast_bf16(
        const __nv_bfloat16 * __restrict__ src0, const S1 * __restrict__ src1,
        __nv_bfloat16 * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t s00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s10, const int64_t s11, const int64_t s12, const int64_t s13,
        const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3) {
    const int64_t total = ne0*ne1*ne2*ne3;
    for (int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t) gridDim.x*blockDim.x) {
        const int64_t i0 =  idx                  % ne0;
        const int64_t i1 = (idx / ne0)           % ne1;
        const int64_t i2 = (idx / (ne0*ne1))     % ne2;
        const int64_t i3 =  idx / (ne0*ne1*ne2);

        const float a = bf2f(src0[i0*s00 + i1*s01 + i2*s02 + i3*s03]);
        const float b = (float) src1[(i0 % ne10)*s10 + (i1 % ne11)*s11 +
                                     (i2 % ne12)*s12 + (i3 % ne13)*s13];

        dst[i0*d0 + i1*d1 + i2*d2 + i3*d3] = f2bf(op == BinOp::Add ? a + b : a * b);
    }
}

// element strides (ggml stores byte strides)
inline int64_t es(const ggml_tensor * t, int i) { return t->nb[i] / ggml_type_size(t->type); }

template <BinOp op>
bool bin_bcast(ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!src0 || !src1) return false;
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    if (src1->type != GGML_TYPE_BF16 && src1->type != GGML_TYPE_F32) return false;
    if (!ggml_are_same_shape(src0, dst)) return false;
    if (!ggml_can_repeat(src1, src0)) return false;

    const int64_t total = ggml_nelements(dst);
    const int64_t blocks = (total + BLOCK - 1) / BLOCK;
    const int grid = (int) (blocks < 65535 ? blocks : 65535);

    if (src1->type == GGML_TYPE_BF16) {
        k_bin_bcast_bf16<op, __nv_bfloat16><<<grid, BLOCK, 0, stream>>>(
            (const __nv_bfloat16 *) src0->data, (const __nv_bfloat16 *) src1->data,
            (__nv_bfloat16 *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            es(src0,0), es(src0,1), es(src0,2), es(src0,3),
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            es(src1,0), es(src1,1), es(src1,2), es(src1,3),
            es(dst,0), es(dst,1), es(dst,2), es(dst,3));
    } else {
        k_bin_bcast_bf16<op, float><<<grid, BLOCK, 0, stream>>>(
            (const __nv_bfloat16 *) src0->data, (const float *) src1->data,
            (__nv_bfloat16 *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            es(src0,0), es(src0,1), es(src0,2), es(src0,3),
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            es(src1,0), es(src1,1), es(src1,2), es(src1,3),
            es(dst,0), es(dst,1), es(dst,2), es(dst,3));
    }
    return true;
}

// ---------------------------------------------------------------------------
// fused elementwise binary: dst = (((src0 op src1) op src2) ... op src[n_fuse])
// ---------------------------------------------------------------------------
//
// ggml fuses runs of up to 8 ADD or MUL nodes in ggml_cuda_try_fuse and hands
// the run over as one synthetic node: src[0] is the base, src[1..n_fuse] are the
// addends, dst->data is the last node's output. The fusion check upstream only
// admits a run when every addend has the same layout, which is why one stride
// set covers all of them.
//
// Accumulation is in float with a single conversion at the end, matching ggml's
// own k_bin_bcast: `float result = (float) src0[..]; result = bin_op(result, ..);
// dst[i0] = (dst_t) result;`. Doing it any other way would make a fused run
// disagree with the unfused one.

constexpr int MAX_FUSE = 8;

template <typename S1>
struct SrcPtrs { const S1 * p[MAX_FUSE]; };

template <BinOp op, typename S1>
__global__ void k_fused_bin_bcast_bf16(
        const __nv_bfloat16 * __restrict__ src0, const SrcPtrs<S1> srcs, const int n_fuse,
        __nv_bfloat16 * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t s00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s10, const int64_t s11, const int64_t s12, const int64_t s13,
        const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3) {
    const int64_t total = ne0*ne1*ne2*ne3;
    for (int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t) gridDim.x*blockDim.x) {
        const int64_t i0 =  idx                  % ne0;
        const int64_t i1 = (idx / ne0)           % ne1;
        const int64_t i2 = (idx / (ne0*ne1))     % ne2;
        const int64_t i3 =  idx / (ne0*ne1*ne2);

        const int64_t j = (i0 % ne10)*s10 + (i1 % ne11)*s11 +
                          (i2 % ne12)*s12 + (i3 % ne13)*s13;

        float acc = bf2f(src0[i0*s00 + i1*s01 + i2*s02 + i3*s03]);
        for (int k = 0; k < n_fuse; ++k) {
            const float b = (float) srcs.p[k][j];
            acc = (op == BinOp::Add) ? acc + b : acc * b;
        }
        dst[i0*d0 + i1*d1 + i2*d2 + i3*d3] = f2bf(acc);
    }
}

template <BinOp op>
bool fused_bin_bcast(ggml_tensor * dst, int n_fuse, cudaStream_t stream) {
    if (n_fuse < 2 || n_fuse > MAX_FUSE) return false;

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) return false;
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    if (!ggml_are_same_shape(src0, dst)) return false;

    // src[1] fixes the layout and type every other addend must match; the
    // upstream fusion check guarantees it, and this re-checks rather than
    // trusting it, because getting it wrong reads out of bounds.
    const ggml_tensor * src1 = dst->src[1];
    if (!src1) return false;
    if (src1->type != GGML_TYPE_BF16 && src1->type != GGML_TYPE_F32) return false;
    if (!ggml_can_repeat(src1, src0)) return false;

    for (int k = 1; k < n_fuse; ++k) {
        const ggml_tensor * s = dst->src[k + 1];
        if (!s || s->type != src1->type) return false;
        if (!ggml_are_same_shape(s, src1)) return false;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (s->nb[d] != src1->nb[d]) return false;
        }
    }

    const int64_t total  = ggml_nelements(dst);
    const int64_t blocks = (total + BLOCK - 1) / BLOCK;
    const int     grid   = (int) (blocks < 65535 ? blocks : 65535);

#define VLA_LAUNCH_FUSED(TYPE)                                                    \
    do {                                                                          \
        SrcPtrs<TYPE> srcs{};                                                     \
        for (int k = 0; k < n_fuse; ++k) srcs.p[k] = (const TYPE *) dst->src[k + 1]->data; \
        k_fused_bin_bcast_bf16<op, TYPE><<<grid, BLOCK, 0, stream>>>(             \
            (const __nv_bfloat16 *) src0->data, srcs, n_fuse,                     \
            (__nv_bfloat16 *) dst->data,                                          \
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],                       \
            es(src0,0), es(src0,1), es(src0,2), es(src0,3),                       \
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],                   \
            es(src1,0), es(src1,1), es(src1,2), es(src1,3),                       \
            es(dst,0), es(dst,1), es(dst,2), es(dst,3));                          \
    } while (0)

    if (src1->type == GGML_TYPE_BF16) {
        VLA_LAUNCH_FUSED(__nv_bfloat16);
    } else {
        VLA_LAUNCH_FUSED(float);
    }
#undef VLA_LAUNCH_FUSED
    return true;
}

// ---------------------------------------------------------------------------
// unary
// ---------------------------------------------------------------------------

enum class UnOp { Silu, Relu, Gelu, GeluErf };

template <UnOp op>
inline __device__ float apply_unary(const float x) {
    if (op == UnOp::Silu)    return x / (1.0f + expf(-x));
    if (op == UnOp::Relu)    return x > 0.0f ? x : 0.0f;
    if (op == UnOp::GeluErf) return 0.5f*x*(1.0f + erff(x*0.70710678118654752440f));
    // tanh approximation, matching ggml's GGML_UNARY_OP_GELU
    const float c = 0.79788456080286535588f;  // sqrt(2/pi)
    return 0.5f*x*(1.0f + tanhf(c*(x + 0.044715f*x*x*x)));
}

template <UnOp op>
__global__ void k_unary_bf16(const __nv_bfloat16 * __restrict__ x,
                             __nv_bfloat16 * __restrict__ dst, const int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x*blockDim.x) {
        dst[i] = f2bf(apply_unary<op>(bf2f(x[i])));
    }
}

template <UnOp op>
bool unary(ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * src0 = dst->src[0];
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    // The elementwise index math above assumes a dense buffer.
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;

    const int64_t n = ggml_nelements(dst);
    const int64_t blocks = (n + BLOCK - 1) / BLOCK;
    const int grid = (int) (blocks < 65535 ? blocks : 65535);
    k_unary_bf16<op><<<grid, BLOCK, 0, stream>>>(
        (const __nv_bfloat16 *) src0->data, (__nv_bfloat16 *) dst->data, n);
    return true;
}

// ---------------------------------------------------------------------------
// scale: dst = x*scale + bias
// ---------------------------------------------------------------------------

__global__ void k_scale_bf16(const __nv_bfloat16 * __restrict__ x, __nv_bfloat16 * __restrict__ dst,
                             const float scale, const float bias, const int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x*blockDim.x) {
        dst[i] = f2bf(scale*bf2f(x[i]) + bias);
    }
}

bool scale(ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * src0 = dst->src[0];
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;

    float s = 1.0f, b = 0.0f;
    memcpy(&s, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) dst->op_params + 1, sizeof(float));

    const int64_t n = ggml_nelements(dst);
    const int64_t blocks = (n + BLOCK - 1) / BLOCK;
    const int grid = (int) (blocks < 65535 ? blocks : 65535);
    k_scale_bf16<<<grid, BLOCK, 0, stream>>>(
        (const __nv_bfloat16 *) src0->data, (__nv_bfloat16 *) dst->data, s, b, n);
    return true;
}

// ---------------------------------------------------------------------------
// norm / rms_norm - one block per row, float reduction
// ---------------------------------------------------------------------------

__device__ inline float block_sum(float v, float * shared) {
    const int tid = threadIdx.x;
    shared[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    return shared[0];
}

template <bool rms>
__global__ void k_norm_bf16(const __nv_bfloat16 * __restrict__ x, __nv_bfloat16 * __restrict__ dst,
                            const int64_t ncols, const int64_t sx1, const int64_t sd1,
                            const float eps) {
    __shared__ float shared[BLOCK];
    const int64_t row = blockIdx.x;
    const __nv_bfloat16 * xr = x   + row*sx1;
    __nv_bfloat16 *       dr = dst + row*sd1;

    float sum = 0.0f, sumsq = 0.0f;
    for (int64_t c = threadIdx.x; c < ncols; c += blockDim.x) {
        const float v = bf2f(xr[c]);
        sumsq += v*v;
        if (!rms) sum += v;
    }

    if (rms) {
        const float ms = block_sum(sumsq, shared) / (float) ncols;
        const float inv = rsqrtf(ms + eps);
        for (int64_t c = threadIdx.x; c < ncols; c += blockDim.x) dr[c] = f2bf(bf2f(xr[c])*inv);
    } else {
        const float mean = block_sum(sum, shared) / (float) ncols;
        __syncthreads();
        const float meansq = block_sum(sumsq, shared) / (float) ncols;
        const float inv = rsqrtf(meansq - mean*mean + eps);
        for (int64_t c = threadIdx.x; c < ncols; c += blockDim.x) dr[c] = f2bf((bf2f(xr[c]) - mean)*inv);
    }
}

template <bool rms>
bool norm(ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * src0 = dst->src[0];
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    // Rows must be dense; higher dims are handled by flattening into the row index.
    if (src0->nb[0] != ggml_type_size(src0->type)) return false;
    if (dst->nb[0]  != ggml_type_size(dst->type))  return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nelements(src0) / ncols;
    if (nrows > 2147483647) return false;

    k_norm_bf16<rms><<<(int) nrows, BLOCK, 0, stream>>>(
        (const __nv_bfloat16 *) src0->data, (__nv_bfloat16 *) dst->data,
        ncols, ncols, ncols, eps);
    return true;
}

// ---------------------------------------------------------------------------
// mul_mat: BF16 x BF16 -> BF16
// ---------------------------------------------------------------------------
//
// The stock path converts src1 F32->BF16 on the way into every cuBLAS GEMM
// because ggml_mul_mat's result is F32 by definition. When the caller asked for
// a BF16 result (vla::mul_mat_t) and both operands are already BF16 there is
// nothing to convert: hand the tensors straight to cuBLAS. Accumulation stays
// CUBLAS_COMPUTE_32F, matching the stock BF16 path.
//
// Note this cannot be left to ggml_cuda_mul_mat_cublas: that writes F32 into
// dst->data unconditionally, which for a BF16 dst is both wrong and twice the
// bytes the allocator reserved.

cublasHandle_t g_handle = nullptr;

bool mul_mat(ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!src0 || !src1) return false;
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_BF16) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // src0 is either shared across the whole batch or batched 1:1 with src1
    const bool batch_ok = (src0->ne[2] == 1           && src0->ne[3] == 1) ||
                          (src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]);
    if (!batch_ok) return false;

    if (!g_handle && cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return false;
    if (cublasSetStream(g_handle, stream) != CUBLAS_STATUS_SUCCESS) return false;

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne0  = dst->ne[0],  ne1  = dst->ne[1];

    const __nv_bfloat16 * a = (const __nv_bfloat16 *) src0->data;
    const __nv_bfloat16 * b = (const __nv_bfloat16 *) src1->data;
    __nv_bfloat16       * c = (__nv_bfloat16       *) dst->data;

    const float alpha = 1.0f, beta = 0.0f;
    const int64_t n_batch = ne12*ne13;

    cublasStatus_t st;
    if (n_batch == 1) {
        st = cublasGemmEx(g_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          (int) ne01, (int) ne11, (int) ne10,
                          &alpha, a, CUDA_R_16BF, (int) ne00,
                                  b, CUDA_R_16BF, (int) ne10,
                          &beta,  c, CUDA_R_16BF, (int) ne0,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        // stride_a == 0 broadcasts one weight matrix across the batch
        const long long stride_a = (src0->ne[2] == 1 && src0->ne[3] == 1) ? 0 : (long long) ne00*ne01;
        st = cublasGemmStridedBatchedEx(g_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          (int) ne01, (int) ne11, (int) ne10,
                          &alpha, a, CUDA_R_16BF, (int) ne00, stride_a,
                                  b, CUDA_R_16BF, (int) ne10, (long long) ne10*ne11,
                          &beta,  c, CUDA_R_16BF, (int) ne0,  (long long) ne0*ne1,
                          (int) n_batch,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    return st == CUBLAS_STATUS_SUCCESS;
}

}  // namespace

// ---------------------------------------------------------------------------
// hook entry point
// ---------------------------------------------------------------------------

extern "C" bool vla_cuda_bf16_fused_binbcast(ggml_tensor * dst, int n_fuse, void * stream_v) {
    if (!dst) return false;
    cudaStream_t stream = (cudaStream_t) stream_v;

    switch (dst->op) {
        case GGML_OP_ADD: return fused_bin_bcast<BinOp::Add>(dst, n_fuse, stream);
        case GGML_OP_MUL: return fused_bin_bcast<BinOp::Mul>(dst, n_fuse, stream);
        default: return false;
    }
}

extern "C" bool vla_cuda_bf16_forward(ggml_tensor * dst, void * stream_v) {
    if (!dst) return false;
    cudaStream_t stream = (cudaStream_t) stream_v;

    switch (dst->op) {
        case GGML_OP_MUL_MAT:  return mul_mat(dst, stream);
        case GGML_OP_ADD:      return bin_bcast<BinOp::Add>(dst, stream);
        case GGML_OP_MUL:      return bin_bcast<BinOp::Mul>(dst, stream);
        case GGML_OP_SCALE:    return scale(dst, stream);
        case GGML_OP_NORM:     return norm<false>(dst, stream);
        case GGML_OP_RMS_NORM: return norm<true>(dst, stream);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_SILU:     return unary<UnOp::Silu>(dst, stream);
                case GGML_UNARY_OP_RELU:     return unary<UnOp::Relu>(dst, stream);
                case GGML_UNARY_OP_GELU:     return unary<UnOp::Gelu>(dst, stream);
                case GGML_UNARY_OP_GELU_ERF: return unary<UnOp::GeluErf>(dst, stream);
                default: return false;
            }
        default: return false;
    }
}

namespace vla {

// Called once, after the CUDA backend is up. Idempotent.
void cuda_register_bf16_ops() {
    ggml_cuda_ext_forward = vla_cuda_bf16_forward;

    // Fusion runs in ggml_backend_cuda_graph_compute, upstream of
    // ggml_cuda_compute_forward, so the pointer above never sees a fused node.
    // Without this second one a fused BF16 add reaches a kernel that handles
    // F32/F16 only and GGML_ABORTs ("unsupported types for fusion: dst: bf16,
    // src0: bf16, src1: f32"), and declining fusion instead costs ~18 ms/call
    // on evo1 -- enough to make BF16 activations a net loss.
    ggml_cuda_ext_fused_binbcast = vla_cuda_bf16_fused_binbcast;
}

}  // namespace vla
