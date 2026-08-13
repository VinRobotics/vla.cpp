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
    GGML_ASSERT(b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0);
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
    if (w->type != at) {
        return as_type(C, ggml_mul_mat(C, w, as_type(C, x, GGML_TYPE_F32)), at);
    }
    return mul_mat_t(C, w, x, at);
}

}  // namespace vla
