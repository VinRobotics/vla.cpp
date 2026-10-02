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

#include <cstdlib>

namespace vla {

// x: F32 [K, T, ...] -> I8 blob [row_bytes, T*...]. With s.gamma set, x is the
// pre-norm hidden state and the RMSNorm is fused into this node.
inline ggml_tensor * fq_act(ggml_context * C, const FqLinear & s, ggml_tensor * x) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == s.act.K);
    // Rows contiguous and 16-byte aligned with the higher dims packed is enough
    // (the prologue takes a row stride); anything else is made contiguous.
    const bool rows_ok = x->nb[0] == sizeof(float) && x->nb[1] % 16 == 0 &&
                         x->nb[2] == x->nb[1] * (size_t) x->ne[1] && x->nb[3] == x->nb[2] * (size_t) x->ne[2];
    ggml_tensor * xin = rows_ok ? x : ggml_cont(C, x);
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
// residual: an F32 tensor shaped like the output that the model would add
// right after the GEMM; folded into the epilogue (one float add, so the
// result is the same as ggml_add would produce). VLA_FQ_NO_FUSE=1 keeps the
// separate add for A/B runs.
inline bool fq_fuse_residual() {
    static const bool off = [] { const char * e = std::getenv("VLA_FQ_NO_FUSE"); return e && *e && *e != '0'; }();
    return !off;
}

inline ggml_tensor * fq_gemm(ggml_context * C, const FqLinear & s, ggml_tensor * xq, ggml_tensor * residual = nullptr) {
    GGML_ASSERT(xq->type == GGML_TYPE_I8 && xq->ne[0] == fq_act_row_bytes(s.act.K, s.act.abits));
    const int64_t T = xq->ne[1];
    const bool fuse = residual && fq_fuse_residual() && !s.gemm.heads && residual->type == GGML_TYPE_F32 &&
                      ggml_is_contiguous(residual) && residual->ne[0] == s.gemm.N && ggml_nelements(residual) == s.gemm.N * T;
    ggml_tensor * args[5] = { s.w, xq, s.wscale, s.bias, fuse ? residual : nullptr };
    ggml_tensor * y = ggml_custom_4d(C, GGML_TYPE_F32, s.gemm.N, T, 1, 1,
                                     args, fuse ? 5 : 4, fq_gemm_cpu, GGML_N_TASKS_MAX, (void *) &s.gemm);
    ggml_format_name(y, "%s.fq_gemm", ggml_get_name(s.w));
    return (residual && !fuse) ? ggml_add(C, residual, y) : y;
}

// Part `part` of a head-laid-out GEMM output (fq_set_heads), as the tensor the
// attention takes: [hd, T, heads] for Q/K, [T, hd, heads] for a V part.
inline ggml_tensor * fq_head_view(ggml_context * C, ggml_tensor * y, const FqLinear & s, int part, int64_t T) {
    GGML_ASSERT(s.gemm.heads && y->ne[1] == T);
    const int64_t hd = s.gemm.head_dim, heads = s.gemm.heads;
    const size_t  off = (size_t) part * heads * hd * T * sizeof(float);
    ggml_tensor * v = ((s.gemm.vmask >> part) & 1)
        ? ggml_view_3d(C, y, T,  hd, heads, T  * sizeof(float), T * hd * sizeof(float), off)
        : ggml_view_3d(C, y, hd, T,  heads, hd * sizeof(float), hd * T * sizeof(float), off);
    ggml_format_name(v, "%s.part%d", ggml_get_name(y), part);
    return v;
}

// y = W x (+ bias) (+ residual)
inline ggml_tensor * fq_linear(ggml_context * C, const FqLinear & s, ggml_tensor * x, ggml_tensor * residual = nullptr) {
    return fq_gemm(C, s, fq_act(C, s, x), residual);
}

}  // namespace vla
