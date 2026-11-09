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

// Copied unchanged from ggml-compiler (runtime/include/ggml_compiler_ops.h, Apache-2.0,
// https://github.com/khanhnd61-vr/ggml-compiler). cosmos3_edge_graphs.cpp was generated against exactly
// this header, which is what keeps its graphs node for node the ones ggml-compiler's runtime builds.
// If it changes upstream, regenerate the graphs and copy it again together.

// ggml_compiler_ops.h - ggml expansions of the IR ops that are more than one ggml call.
//
// Shared by the runtime's op dispatcher and by C++ generated with `ggml-compiler compile --emit cpp`, so a
// generated graph is the same ggml graph the runtime builds, node for node: that is what makes
// their outputs bit-identical. Edit with care; both sides change together.
#pragma once

#include "ggml.h"

namespace ggml_compiler::ops {

// Integer arithmetic of symbolic dims, with Python's floor semantics (as ir/sym.py and the
// runtime's Expr::eval use).
inline int64_t floordiv(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
inline int64_t mod(int64_t a, int64_t b) {
    const int64_t r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
inline int64_t max(int64_t a, int64_t b) { return a > b ? a : b; }
inline int64_t min(int64_t a, int64_t b) { return a < b ? a : b; }

// IR conv_2d: kernel [KW, KH, IC, OC], input [W, H, IC, N] -> [OW, OH, OC, N] (+ bias [OC]).
// f16 kernels on a GPU run as ggml_conv_2d_direct (implicit GEMM on tensor cores) when the output
// has enough positions to fill it (image backbones). Otherwise im2col in the kernel's float type
// and one matmul, with the kernel as the right operand so the result comes out channel-first.
// (ggml_conv_2d itself would build the patch buffer in f16.) A 1-D U-Net over 16 steps
// runs ~2x faster as im2col + matmul than as direct convolution on an RTX 3060.
constexpr int64_t kDirectConvMinPositions = 1024;

inline ggml_tensor * conv_2d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * bias, int s0, int s1,
                             int p0, int p1, int d0, int d1, bool gpu) {
    ggml_tensor * y;
    const int64_t ow = (x->ne[0] + 2 * p0 - d0 * (w->ne[0] - 1) - 1) / s0 + 1;
    const int64_t oh = (x->ne[1] + 2 * p1 - d1 * (w->ne[1] - 1) - 1) / s1 + 1;
    if (gpu && w->type == GGML_TYPE_F16 && ow * oh * x->ne[3] >= kDirectConvMinPositions) {
        y = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    } else {
        const ggml_type ct  = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor *   col = ggml_im2col(ctx, w, x, s0, s1, p0, p1, d0, d1, true, ct);  // [K, OW, OH, N]
        y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2] * col->ne[3]),
                         ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
        if (col->ne[3] == 1) {
            y = ggml_reshape_4d(ctx, y, col->ne[1], col->ne[2], w->ne[3], 1);
        } else {
            y = ggml_cont(ctx,
                          ggml_permute(ctx, ggml_reshape_4d(ctx, y, col->ne[1], col->ne[2], col->ne[3], w->ne[3]), 0, 1, 3, 2));
        }
    }
    if (bias) {
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1));
    }
    return y;
}

// IR gelu_tanh: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) in f32. GPU backends compute
// ggml_gelu with exactly that formula; the CPU one reads an fp16 table, so CPU spells it out.
inline ggml_tensor * gelu_tanh(ggml_context * ctx, ggml_tensor * x, bool gpu) {
    if (gpu) {
        return ggml_gelu(ctx, x);
    }
    ggml_tensor * inner = ggml_add(ctx, x, ggml_scale(ctx, ggml_mul(ctx, ggml_sqr(ctx, x), x), 0.044715f));
    ggml_tensor * t     = ggml_tanh(ctx, ggml_scale(ctx, inner, 0.7978845608f));  // sqrt(2/pi)
    return ggml_mul(ctx, x, ggml_scale_bias(ctx, t, 0.5f, 0.5f));
}

// IR view: a step-1 slice along ggml dim `dim` starting at `start`; ne is the result's shape.
// A size-1 dim's stride never moves an address, so it gets the packed value: ggml's layout checks
// (ggml_is_contiguous, ggml_is_padded_1d in ggml_scale) compare strides even across size-1 dims,
// and a slice of the outermost dim would otherwise keep its parent's.
inline ggml_tensor * slice(ggml_context * ctx, ggml_tensor * x, int dim, int64_t start, int64_t ne0, int64_t ne1,
                           int64_t ne2, int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    size_t        nb[4] = { x->nb[0], x->nb[1], x->nb[2], x->nb[3] };
    for (int k = 1; k < 4; k++) {
        if (ne[k] == 1) {
            nb[k] = nb[k - 1] * ne[k - 1];
        }
    }
    return ggml_view_4d(ctx, x, ne0, ne1, ne2, ne3, nb[1], nb[2], nb[3], start * x->nb[dim]);
}

// IR reshape. A contiguous input is ggml_reshape_4d. A strided one (a view) may only gain or lose
// size-1 dims: its non-unit dims keep their order and strides, which is a view with those strides
// (size-1 dims packed, as in slice). Returns nullptr if neither applies.
inline ggml_tensor * reshape(ggml_context * ctx, ggml_tensor * x, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    if (ggml_is_contiguous(x)) {
        return ggml_reshape_4d(ctx, x, ne0, ne1, ne2, ne3);
    }
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    size_t        nb[4] = { x->nb[0], 0, 0, 0 };
    int           k     = 0;  // next dim of x
    for (int i = 0; i < 4; i++) {
        if (ne[i] == 1) {
            if (i > 0) nb[i] = nb[i - 1] * ne[i - 1];
            continue;
        }
        while (k < 4 && x->ne[k] == 1) k++;
        if (k == 4 || x->ne[k] != ne[i]) return nullptr;
        nb[i] = x->nb[k++];
    }
    while (k < 4 && x->ne[k] == 1) k++;
    if (k != 4 || nb[0] != x->nb[0]) return nullptr;  // dims left over, or the innermost moved
    return ggml_view_4d(ctx, x, ne0, ne1, ne2, ne3, nb[1], nb[2], nb[3], 0);
}

// IR scale: x * s + b.
inline ggml_tensor * scale(ggml_context * ctx, ggml_tensor * x, float s, float b) {
    return b == 0.0f ? ggml_scale(ctx, x, s) : ggml_scale_bias(ctx, x, s, b);
}

// A matmul; acc_f32 asks the backend to accumulate in f32 (Stage(matmul_acc="f32")): CUDA's cuBLAS
// path otherwise accumulates 16-bit weights' products in f16.
inline ggml_tensor * mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, bool acc_f32) {
    ggml_tensor * r = ggml_mul_mat(ctx, a, b);
    if (acc_f32) {
        ggml_prec_set_acc(r, GGML_PREC_F32);
    }
    return r;
}

// Flash attention accumulating in f32, as llama.cpp asks for in every attention (its language
// models and its vision encoders): the default lets CUDA's kernels accumulate in f16, which Qwen3's
// large activations do not survive (Qwen3-VL-2B F16 KL 9.7e-3 against 1.1e-4 on CPU).
//
// `supported(t)` asks the backend whether it can run the flash attention node t. CUDA covers some
// head sizes only on some GPUs (512: Turing tensor cores and grouped-query heads); where it cannot,
// the same attention is built as Q·Kᵀ, softmax and P·V, both matmuls accumulating in f32 too. The
// runtime and generated C++ ask the same question, so they build the same graph.
template <class Supported>
inline ggml_tensor * flash_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                                float scale, Supported && supported) {
    ggml_tensor * r = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(r, GGML_PREC_F32);
    if (supported(r)) {
        return r;
    }
    // q [D, Sq, H, B], k and v [D, Sk, Hkv, B]: mul_mat broadcasts the K/V heads over the query
    // heads as flash attention does (query head h reads K/V head h / (H / Hkv)).
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);  // [Sk, Sq, H, B]
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), kq);  // [D, Sq, H, B]
    ggml_prec_set_acc(kqv, GGML_PREC_F32);
    return ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [D, H, Sq, B], flash attention's layout
}

}  // namespace ggml_compiler::ops
