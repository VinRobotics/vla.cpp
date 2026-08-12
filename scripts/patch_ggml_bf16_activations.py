#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teach the fetched ggml how to carry BF16 activations (CUDA backend).

Why this exists
---------------
ggml_mul_mat's result is F32 by definition, so a BF16-resident weight meeting an
F32 activation makes ggml_cuda_op_mul_mat_cublas convert src1 F32->BF16 on the
way into every GEMM and the result BF16->F32 on the way out. An nsys trace of
the evo1 server put those convert_unary launches at 11.1% of GPU time, in
exactly balanced pairs (8,228 each direction). Carrying activations as BF16
removes both, and halves the bytes every bias-add, norm, activation and layout
copy has to move.

What it changes
---------------
  ggml.c / ggml.h   ggml_mul_mat_t(ctx, a, b, type) - mul_mat with an explicit
                    result type; ggml_mul_mat becomes a wrapper at F32.
  ggml-cuda.cu      a direct cuBLAS BF16xBF16->BF16 path, taken whenever a
                    caller asked for a BF16 matmul result.
  binbcast.cu       BF16 add/mul (activation x activation, activation x F32
                    bias/weight), plain and fused.
  unary.cu          BF16 gelu/silu/relu/...
  norm.cu           BF16 norm / rms_norm / fused rms_norm+mul, float reductions.
  scale.cu          BF16 scale.

concat.cu needs nothing: it already dispatches on ggml_type_size to a
width-generic kernel, so BF16 lands on the uint16_t instantiation.

Everything keeps float accumulation, so only operand and result *storage*
changes, never a reduction. All BF16 branches are additive: an F32 graph hits
exactly the code it hit before.

llama.cpp is pulled in by FetchContent (see the top-level CMakeLists), so the
tree lives under build/_deps/llama-src and a fresh configure re-clones it. Run
this after configuring and before building. It is idempotent.

Usage: scripts/patch_ggml_bf16_activations.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp: BF16 activation support"


def edit(path, subs):
    """Apply (old, new) pairs to `path`; every `old` must appear exactly once.

    Returns the new text rather than writing, so main() can apply every file's
    edits or none: a half-patched tree is worse than an unpatched one.
    """
    text = path.read_text()
    for old, new in subs:
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"{path}: anchor found {n} times, expected 1:\n---\n{old[:400]}\n---"
            )
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# ggml.c / ggml.h - mul_mat with an explicit result type
# ---------------------------------------------------------------------------
GGML_C = [(
    """struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}""",
    """// vla.cpp: BF16 activation support - mul_mat with an explicit result type.
//
// ggml_mul_mat always produces F32, which forces a BF16->F32 conversion out of
// every cuBLAS BF16 GEMM and an F32->BF16 one back in at the next matmul.
// Letting the caller ask for a BF16 result is what makes an end-to-end BF16
// activation graph expressible. Only CUDA implements a non-F32 result; see
// ggml_cuda_mul_mat_bf16.
struct ggml_tensor * ggml_mul_mat_t(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        enum   ggml_type      type) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, type, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_mul_mat_t(ctx, a, b, GGML_TYPE_F32);
}""",
)]

GGML_H = [(
    """    GGML_API struct ggml_tensor * ggml_mul_mat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // change the precision of a matrix multiplication""",
    """    GGML_API struct ggml_tensor * ggml_mul_mat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // vla.cpp: BF16 activation support - as ggml_mul_mat, but with an explicit
    // result type. Only GGML_TYPE_BF16 (CUDA, BF16 a and b) is implemented
    // beyond F32; it keeps a BF16 GEMM's output in BF16 so an all-BF16
    // activation graph does not round-trip through F32 at every matmul.
    GGML_API struct ggml_tensor * ggml_mul_mat_t(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            enum   ggml_type      type);

    // change the precision of a matrix multiplication""",
)]

# ---------------------------------------------------------------------------
# ggml-cuda.cu - direct BF16 x BF16 -> BF16 cuBLAS GEMM
# ---------------------------------------------------------------------------
GGML_CUDA = [(
    """static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
""",
    """// vla.cpp: BF16 activation support - BF16 x BF16 -> BF16 cuBLAS GEMM.
//
// The stock BF16 path (ggml_cuda_op_mul_mat_cublas) always converts src1 to
// BF16 on the way in and the BF16 GEMM result back to F32 on the way out,
// because ggml_mul_mat's result is F32 by definition. On an F32-activation
// graph that is one convert_unary launch per operand per matmul - ~11% of GPU
// time on evo1. When the caller asked for a BF16 result (ggml_mul_mat_t) and
// both operands are already BF16 there is nothing to convert: hand the tensors
// straight to cuBLAS.
//
// Accumulation stays CUBLAS_COMPUTE_32F, matching the stock BF16 path, so only
// operand and result storage changes, not the reduction.
static bool ggml_cuda_can_mul_mat_bf16(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_BF16 || dst->type != GGML_TYPE_BF16) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // src0 is either shared across the whole batch or batched 1:1 with src1
    const bool batch_ok = (src0->ne[2] == 1           && src0->ne[3] == 1) ||
                          (src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]);
    return batch_ok && bf16_mma_hardware_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

static void ggml_cuda_mul_mat_bf16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const nv_bfloat16 * a = (const nv_bfloat16 *) src0->data;
    const nv_bfloat16 * b = (const nv_bfloat16 *) src1->data;
    nv_bfloat16       * c = (nv_bfloat16       *) dst->data;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), ctx.stream()));

    const int64_t n_batch = ne12*ne13;
    if (n_batch == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    &alpha, a, CUDA_R_16BF, ne00,
                            b, CUDA_R_16BF, ne10,
                    &beta,  c, CUDA_R_16BF, ne0,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return;
    }

    // stride_a == 0 broadcasts one weight matrix across the batch
    const long long stride_a = (src0->ne[2] == 1 && src0->ne[3] == 1) ? 0 : (long long) ne00*ne01;
    CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                &alpha, a, CUDA_R_16BF, ne00, stride_a,
                        b, CUDA_R_16BF, ne10, (long long) ne10*ne11,
                &beta,  c, CUDA_R_16BF, ne0,  (long long) ne0*ne1,
                n_batch,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // vla.cpp: nothing but ggml_mul_mat_t(..., GGML_TYPE_BF16) produces a BF16
    // matmul. This has to intercept before the dst->type != F32 early-out below,
    // which would otherwise hand the BF16 result to ggml_cuda_mul_mat_cublas and
    // convert it straight back to F32. There is no split-buffer case to exclude:
    // CUDA split buffers were removed from ggml upstream.
    if (dst->type == GGML_TYPE_BF16) {
        if (!ggml_cuda_can_mul_mat_bf16(src0, src1, dst)) {
            // Name the offending operand rather than just asserting: the usual
            // cause is a graph asking for a BF16 result from a weight that is
            // resident F32 (or quantized), which mm_act() is meant to route
            // around. See src/models/act_dtype.h.
            for (const ggml_tensor * t : {src0, src1, (const ggml_tensor *) dst}) {
                fprintf(stderr, "BF16 mul_mat operand %-24s type=%-5s contiguous=%d ne=[%ld %ld %ld %ld]\\n",
                        t->name, ggml_type_name(t->type), (int) ggml_is_contiguous(t),
                        (long) t->ne[0], (long) t->ne[1], (long) t->ne[2], (long) t->ne[3]);
            }
            GGML_ABORT("BF16 mul_mat result requires contiguous BF16 operands on BF16-MMA hardware");
        }
        ggml_cuda_mul_mat_bf16(ctx, src0, src1, dst);
        return;
    }

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
""",
), (
    """        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }""",
    """        // vla.cpp: BF16 activation support. ggml_cuda_op_rms_norm_fused now takes
        // a BF16 activation with an F32 norm weight; the three-op variant that
        // also folds an add stays F32-only.
        const enum ggml_type rms_at = rms_norm->src[0]->type;
        if (rms_at != GGML_TYPE_F32 && rms_at != GGML_TYPE_BF16) {
            return false;
        }
        GGML_ASSERT(rms_norm->type == rms_at);

        const ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        if (mul_w->type != GGML_TYPE_F32 || mul->type != rms_at) {
            return false;
        }

        if (add && (rms_at != GGML_TYPE_F32 ||
            add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }""",
)]

# ---------------------------------------------------------------------------
# binbcast.cu - BF16 add / mul, plain and fused
# ---------------------------------------------------------------------------
BINBCAST = [
    (
        """    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);""",
        """    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    // vla.cpp: BF16 activation support. src1 is F32 for the model's bias and
    // norm-weight tensors (kept F32 in the GGUF) and BF16 for residual adds.
    if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()(src0, src1, dst, (const nv_bfloat16 *)src0_dd, (const nv_bfloat16 *)src1_dd, (nv_bfloat16 *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()(src0, src1, dst, (const nv_bfloat16 *)src0_dd, (const float *)src1_dd, (nv_bfloat16 *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()(src0, src1, dst, (const float *)src0_dd, (const nv_bfloat16 *)src1_dd, (nv_bfloat16 *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const nv_bfloat16 *)src0_dd, (const nv_bfloat16 *)src1_dd, (float *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);""",
    ),
    (
        """    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, float, float, float>(src0, src1, dst,
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});""",
        """    // vla.cpp: BF16 activation support. Fusion requires identical src1 layouts
    // (ggml_are_same_layout compares type), so a chain never mixes F32 bias and
    // BF16 residual operands.
    if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast_pack<op, nv_bfloat16, nv_bfloat16, nv_bfloat16>(src0, src1, dst,
            (const nv_bfloat16 *) src0->data, (const nv_bfloat16 *) src1->data, (nv_bfloat16 *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast_pack<op, nv_bfloat16, float, nv_bfloat16>(src0, src1, dst,
            (const nv_bfloat16 *) src0->data, (const float *) src1->data, (nv_bfloat16 *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, float, float, float>(src0, src1, dst,
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});""",
    ),
]

# ---------------------------------------------------------------------------
# unary.cu - BF16 elementwise activations
# ---------------------------------------------------------------------------
UNARY = [(
    """    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}""",
    """    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16 ||  dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        // vla.cpp: BF16 activation support. unary_op_kernel already evaluates in
        // float and casts back, so only the storage type changes.
        unary_cuda<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), stream);
    } else {
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}""",
)]

# ---------------------------------------------------------------------------
# norm.cu - BF16 norm / rms_norm, float reductions
# ---------------------------------------------------------------------------
NORM = [
    (
        """template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {""",
        """// vla.cpp: BF16 activation support. T is the activation storage type (float or
// nv_bfloat16). The mean/variance reduction and the normalisation stay in float
// regardless, so a BF16 graph gets the numerics PyTorch's bf16 LayerNorm gives
// (bf16 in/out, fp32 accumulate).
template <int block_size, typename T = float>
static __global__ void norm_f32(
        const T * x, T * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {""",
    ),
    (
        """    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }""",
        """    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float) x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }""",
    ),
    (
        """    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}""",
        """    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (T) (((float) x[col] - mean) * inv_std);
    }
}""",
    ),
    (
        """template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,""",
        """// vla.cpp: BF16 activation support. T is the activation storage type; `mul` and
// `add` stay F32 because the norm weight and bias are F32 in the GGUF.
template <int block_size, bool do_multiply = false, bool do_add = false, typename T = float>
static __global__ void rms_norm_f32(const T *     x,
                                    T *           dst,
                                    const int     ncols,""",
    ),
    (
        # disambiguated from the identical loop in l2_norm_f32 by the tail
        """    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;""",
        """    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float) x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;""",
    ),
    (
        """        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }""",
        """        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = (T) (scale * (float) x[col] * mul[mul_col] + add[add_col]);
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = (T) (scale * (float) x[col] * mul[mul_col]);
        } else {
            dst[col] = (T) (scale * (float) x[col]);
        }""",
    ),
    (
        """static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,""",
        """template <typename T>
static void norm_f32_cuda(
        const T * x, T * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,""",
    ),
    (
        """        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);""",
        """        norm_f32<WARP_SIZE, T><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);""",
    ),
    (
        """        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);""",
        """        norm_f32<1024, T><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);""",
    ),
    (
        """static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,""",
        """template <typename T>
static void rms_norm_f32_cuda(
        const T * x, T * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,""",
    ),
    (
        """ggml_cuda_kernel_launch(rms_norm_f32<256, false>,""",
        """ggml_cuda_kernel_launch(rms_norm_f32<256, false, false, T>,""",
    ),
    (
        """ggml_cuda_kernel_launch(rms_norm_f32<1024, false>,""",
        """ggml_cuda_kernel_launch(rms_norm_f32<1024, false, false, T>,""",
    ),
    (
        """static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,""",
        """template <typename T>
static void rms_norm_mul_f32_cuda(const T *      x,
                                  const float *  mul,
                                  const float *  add,
                                  T *            dst,""",
    ),
    (
        """ggml_cuda_kernel_launch(rms_norm_f32<256, true>,""",
        """ggml_cuda_kernel_launch(rms_norm_f32<256, true, false, T>,""",
    ),
    (
        """ggml_cuda_kernel_launch(rms_norm_f32<1024, true>,""",
        """ggml_cuda_kernel_launch(rms_norm_f32<1024, true, false, T>,""",
    ),
    (
        """ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>,""",
        """ggml_cuda_kernel_launch(rms_norm_f32<256, true, true, T>,""",
    ),
    (
        """ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>,""",
        """ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true, T>,""",
    ),
    (
        """void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);""",
        """void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // vla.cpp: BF16 activation support
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == src0->type);""",
    ),
    (
        """    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}""",
        """    if (src0->type == GGML_TYPE_BF16) {
        norm_f32_cuda((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data,
                      ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    } else {
        norm_f32_cuda((const float *) src0->data, (float *) dst->data,
                      ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    }
}""",
    ),
    (
        """void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);""",
        """void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // vla.cpp: BF16 activation support
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == src0->type);""",
    ),
    (
        """    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}""",
        """    if (src0->type == GGML_TYPE_BF16) {
        rms_norm_f32_cuda((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data,
                          ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    } else {
        rms_norm_f32_cuda((const float *) src0->data, (float *) dst->data,
                          ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    }
}""",
    ),
    (
        """    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);""",
        """    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    cudaStream_t stream = ctx.stream();

    // vla.cpp: BF16 activation support - activation may be BF16, norm weight stays F32
    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32 || rms_norm_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == rms_norm_src->type);
    GGML_ASSERT(mul_tensor->type == rms_norm_src->type);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);""",
    ),
    (
        """    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}""",
        """    if (rms_norm_src->type == GGML_TYPE_BF16) {
        rms_norm_mul_f32_cuda((const nv_bfloat16 *) rms_norm_src->data, mul_d, (const float *) nullptr,
                              (nv_bfloat16 *) mul_tensor->data,
                              ne00, ne01, ne02, ne03,
                              /*s00*/ s01, s02, s03,
                              /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                              mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                              /*add_s00*/ 0, 0, 0,
                              0, 0, 0, 0,
                              eps, stream);
    } else {
        rms_norm_mul_f32_cuda((const float *) rms_norm_src->data, mul_d, (const float *) nullptr,
                              (float *) mul_tensor->data,
                              ne00, ne01, ne02, ne03,
                              /*s00*/ s01, s02, s03,
                              /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                              mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                              /*add_s00*/ 0, 0, 0,
                              0, 0, 0, 0,
                              eps, stream);
    }
}""",
    ),
]

# ---------------------------------------------------------------------------
# scale.cu - BF16 scale
# ---------------------------------------------------------------------------
SCALE = [(
    """static __global__ void scale_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale * x[i] + bias;
    }
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32, launch_params, x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    scale_f32_cuda(src0_d, dst_d, scale, bias, ggml_nelements(src0), stream);
}""",
    """// vla.cpp: BF16 activation support. T is the activation storage type; the
// scale and bias apply in float.
template <typename T>
static __global__ void scale_f32(const T * x, T * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = (T) (scale * (float) x[i] + bias);
    }
}

template <typename T>
static void scale_f32_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32<T>, launch_params, x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == src0->type);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_BF16) {
        scale_f32_cuda((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data, scale, bias, ggml_nelements(src0), stream);
    } else {
        scale_f32_cuda((const float *) src0->data, (float *) dst->data, scale, bias, ggml_nelements(src0), stream);
    }
}""",
)]


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "build/_deps/llama-src"
    if not (src / "ggml/src/ggml.c").exists():
        raise SystemExit(f"not a llama.cpp source tree: {src}")

    if MARKER in (src / "ggml/src/ggml.c").read_text():
        print(f"already patched: {src}")
        return

    cuda = src / "ggml/src/ggml-cuda"
    pending = [
        (src / "ggml/src/ggml.c",     edit(src / "ggml/src/ggml.c", GGML_C)),
        (src / "ggml/include/ggml.h", edit(src / "ggml/include/ggml.h", GGML_H)),
        (cuda / "ggml-cuda.cu",       edit(cuda / "ggml-cuda.cu", GGML_CUDA)),
        (cuda / "binbcast.cu",        edit(cuda / "binbcast.cu", BINBCAST)),
        (cuda / "unary.cu",           edit(cuda / "unary.cu", UNARY)),
        (cuda / "norm.cu",            edit(cuda / "norm.cu", NORM)),
        (cuda / "scale.cu",           edit(cuda / "scale.cu", SCALE)),
    ]
    for path, text in pending:
        path.write_text(text)
    print(f"patched {len(pending)} files: {src}")


if __name__ == "__main__":
    main()
