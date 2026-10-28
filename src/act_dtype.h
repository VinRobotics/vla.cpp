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

#pragma once

// Activation-dtype helpers for the BF16 activation path.
//
// ggml_mul_mat's result is F32 by definition, so with BF16-resident weights the
// CUDA backend converts the F32 activation to BF16 on the way into every GEMM
// and the BF16 result back to F32 on the way out
// (ggml_cuda_op_mul_mat_cublas). An nsys trace of the evo1 server put those
// convert_unary launches at ~18 ms per request — in exactly balanced pairs.
// Carrying activations as BF16 removes both, and halves the bytes every
// bias-add, norm, activation and layout copy has to move.
//
// A model opts in by setting its `act_type` to GGML_TYPE_BF16 and routing its
// weight GEMMs through mm_act(). The precision split that goes with it follows
// torch.autocast(bfloat16), which is what these references run under:
//
//   BF16 - weight GEMMs, bias adds, residuals, elementwise multiplies, norms
//          (float reductions internally), and the activation functions.
//   F32  - attention scores, softmax and RoPE; patch-embed convolutions; and
//          any flow-matching Euler integrator, so N accumulations of dt = 1/N
//          do not collapse into 8 mantissa bits.
//
// The BF16 kernels themselves are in-tree (src/cuda/vla_cuda_bf16.cu); the only
// ggml change is the one-function-pointer hook that lets them run
// (scripts/patch_ggml_cuda_ext_hook.py).
//
// At act_type == GGML_TYPE_F32 both helpers below are the identity, so the
// default path builds exactly the graph it built before.

#include "ggml.h"
#include "layers/fused_ops.h"

#include <cstddef>

namespace vla {

// Cast only when the type actually differs, so the F32 path adds no nodes.
inline ggml_tensor * as_type(ggml_context * C, ggml_tensor * t, ggml_type ty) {
    return t->type == ty ? t : ggml_cast(C, t, ty);
}

// mul_mat with an explicit result type.
//
// ggml_mul_mat hard-codes GGML_TYPE_F32 for its result and there is no API to
// ask for another one - but ggml_tensor's op/src fields are public, so the node
// can simply be built here rather than patched into ggml.c. Shapes, op and
// operand order are identical to ggml_mul_mat; only the result type differs.
//
// Only CUDA executes a non-F32 result, via the in-tree kernels in
// src/cuda/vla_cuda_bf16.cu.
inline ggml_tensor * mul_mat_t(ggml_context * C, ggml_tensor * a, ggml_tensor * b, ggml_type type) {
    // ggml_can_mul_mat is internal to ggml; this is the same condition.
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(b->ne[2]%a->ne[2] == 0 && b->ne[3]%a->ne[3] == 0);
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    ggml_tensor * result = ggml_new_tensor(C, type, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// Weight-by-activation GEMM producing an activation-typed result.
//
// A BF16 result needs a BF16 weight. Not every weight is one: pi0 keeps its
// action-expert projections (state_proj, action_in_proj, action_time_mlp_*,
// action_out_proj) resident F32, and a quantized GGUF keeps its packed type.
// Those run the stock matmul at F32 and rejoin the activation stream after,
// which is exactly what they did before the BF16 path existed.
inline ggml_tensor * mm_act(ggml_context * C, ggml_tensor * w, ggml_tensor * x, ggml_type at) {
    // F32 activations with the CUDA fused ops on: the BF16-weight GEMM runs
    // through gemm_exact (bit-identical to ggml's cuBLAS call, often faster).
    if (at == GGML_TYPE_F32 && fused_ops_for(w))
        return mm_exact(C, w, x);
    if (w->type != at) {
        return as_type(C, ggml_mul_mat(C, w, as_type(C, x, GGML_TYPE_F32)), at);
    }
    return mul_mat_t(C, w, x, at);
}

// Give an op result another type before anything uses it. ggml's constructors
// fix the result type (RoPE: the input's; flash attention: F32), but the SYCL
// kernels can also write the other one (RoPE F32 -> F16, fused SDPA -> BF16),
// which saves the conversion that would follow.
inline ggml_tensor * retype(ggml_tensor * t, ggml_type type) {
    GGML_ASSERT(t->view_src == nullptr && t->data == nullptr);
    t->type  = type;
    t->nb[0] = ggml_type_size(type);
    for (int i = 1; i < GGML_MAX_DIMS; ++i)
        t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    return t;
}

// Make op result t a 1-D I8 tensor of `bytes` (a q8 blob, layers/fused_ew.h),
// for a kernel that writes its result quantized. Call before anything uses it.
inline ggml_tensor * retype_blob(ggml_tensor * t, int64_t bytes) {
    GGML_ASSERT(t->view_src == nullptr && t->data == nullptr);
    t->type = GGML_TYPE_I8;
    t->ne[0] = bytes;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) t->ne[i] = 1;
    t->nb[0] = 1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    return t;
}

// Make op result t ([hd, n_kv, seq], contiguous) write into rows [row0, row0+seq)
// of buf instead of its own memory, like ggml_cpy into a view but for any op.
inline ggml_tensor * write_rows(ggml_tensor * t, ggml_tensor * buf, int64_t row0) {
    GGML_ASSERT(t->type == buf->type && t->ne[0] == buf->ne[0] && t->ne[1] == buf->ne[1]);
    GGML_ASSERT(row0 + t->ne[2] <= buf->ne[2] && t->view_src == nullptr && t->data == nullptr);
    t->view_src  = buf;
    t->view_offs = (size_t) row0 * buf->nb[2];
    for (int i = 0; i < GGML_MAX_DIMS; ++i)
        t->nb[i] = buf->nb[i];
    return t;
}

// An ordering edge: node runs after dep. Kernels read only the srcs they know
// (flash attention: 0-4), so the spare slots carry it.
inline void add_dep(ggml_tensor * node, ggml_tensor * dep, int first_free = 5) {
    for (int i = first_free; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] == dep) return;
        if (!node->src[i]) { node->src[i] = dep; return; }
    }
    GGML_ABORT("add_dep: no free src slot");
}

// mul_mat with an F32 result whatever the activation type: a BF16 x BF16 GEMM
// writes F32 directly where a kernel takes that pairing (src/sycl/vla_sycl_bf16.cpp),
// rather than a BF16 result and a cast. Elsewhere, mm_act and a cast.
inline ggml_tensor * mm_f32(ggml_context * C, ggml_tensor * w, ggml_tensor * x) {
    if (w->type == GGML_TYPE_BF16 && x->type == GGML_TYPE_BF16)
        return ggml_mul_mat(C, w, x);
    return as_type(C, mm_act(C, w, x, x->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F32), GGML_TYPE_F32);
}

}  // namespace vla
