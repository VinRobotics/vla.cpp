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

// FoldQuant linears as graph nodes. Two GGML_OP_CUSTOM nodes per site so the
// quantized activation is an ordinary gallocr intermediate (shared by q/k/v
// or gate/up, no backend workspace, capture-safe), and one encoding serves both
// backends: the CPU backend runs the custom function, the CUDA backend claims
// the node by the magic word in its userdata.

#pragma once

#include "foldquant.h"
#include "foldquant_ref.h"

#include "ggml.h"

namespace vla {

// x: F32 [K, T, ...] -> I8 blob [row_bytes, T*...]. With s.gamma set, x is the
// pre-norm hidden state and the RMSNorm is fused into this node.
inline ggml_tensor * fq_act(ggml_context * C, const FqLinear & s, ggml_tensor * x) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == s.act.K);
    ggml_tensor * xin = ggml_is_contiguous(x) ? x : ggml_cont(C, x);
    const int64_t rows = ggml_nelements(xin) / s.act.K;

    // Sources are packed without holes: src[1] is the gamma when the norm is
    // fused, then the ascale when the site ships one (fq_act_srcs() decodes).
    ggml_tensor * args[3] = { xin, nullptr, nullptr };
    int n_args = 1;
    if (s.gamma)  args[n_args++] = s.gamma;
    if (s.ascale) args[n_args++] = s.ascale;
    ggml_tensor * t = ggml_custom_4d(C, GGML_TYPE_I8, fq_act_row_bytes(s.act.K, s.act.abits), rows, 1, 1,
                                     args, n_args, fq_act_cpu, GGML_N_TASKS_MAX, (void *) &s.act);
    ggml_format_name(t, "%s.fq_act", ggml_get_name(s.w));
    return t;
}

// xq: blob from fq_act (any site sharing the same input transform) -> F32 [N, T].
inline ggml_tensor * fq_gemm(ggml_context * C, const FqLinear & s, ggml_tensor * xq) {
    GGML_ASSERT(xq->type == GGML_TYPE_I8 && xq->ne[0] == fq_act_row_bytes(s.act.K, s.act.abits));
    ggml_tensor * args[4] = { s.w, xq, s.wscale, s.bias };
    ggml_tensor * y = ggml_custom_4d(C, GGML_TYPE_F32, s.gemm.N, xq->ne[1], 1, 1,
                                     args, 4, fq_gemm_cpu, GGML_N_TASKS_MAX, (void *) &s.gemm);
    ggml_format_name(y, "%s.fq_gemm", ggml_get_name(s.w));
    return y;
}

inline ggml_tensor * fq_linear(ggml_context * C, const FqLinear & s, ggml_tensor * x) {
    return fq_gemm(C, s, fq_act(C, s, x));
}

}  // namespace vla
