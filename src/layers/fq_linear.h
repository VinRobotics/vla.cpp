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
    ggml_tensor * t = ggml_custom_4d(C, GGML_TYPE_I8, fq_act_row_bytes(s.act.K, fq_act_store_bits(s.act)), rows, 1, 1,
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

// VLA_FQ_NO_GLU=1 declares gate and up as two sites (no GEGLU epilogue), for A/B runs.
inline bool fq_glu_enabled() {
    static const bool off = [] { const char * e = std::getenv("VLA_FQ_NO_GLU"); return e && *e && *e != '0'; }();
    return !off;
}

// gate: with a residual, the model's gated residual residual + y * gate[n]
// (pi0.5's action expert); fused on SYCL, a mul and an add elsewhere.
inline ggml_tensor * fq_gemm(ggml_context * C, const FqLinear & s, ggml_tensor * xq, ggml_tensor * residual = nullptr,
                             ggml_tensor * gate = nullptr) {
    GGML_ASSERT(xq->type == GGML_TYPE_I8 && xq->ne[0] == fq_act_row_bytes(s.act.K, fq_act_store_bits(s.act)));
    const int64_t T = xq->ne[1];
    const int64_t N_out = s.gemm.glu ? s.gemm.N / 2 : s.gemm.N;
    bool fuse = residual && fq_fuse_residual() && !s.gemm.heads && !s.gemm.glu && residual->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(residual) && residual->ne[0] == s.gemm.N && ggml_nelements(residual) == s.gemm.N * T;
#ifndef GGML_USE_SYCL
    if (gate) fuse = false;   // the gated epilogue is a SYCL kernel feature
#endif
    if (gate && (gate->type != GGML_TYPE_F32 || gate->ne[0] != s.gemm.N || !ggml_is_contiguous(gate))) fuse = false;
    ggml_tensor * args[6] = { s.w, xq, s.wscale, s.bias, fuse ? residual : nullptr, fuse ? gate : nullptr };
    ggml_tensor * y = ggml_custom_4d(C, GGML_TYPE_F32, N_out, T, 1, 1,
                                     args, fuse ? (gate ? 6 : 5) : 4, fq_gemm_cpu, GGML_N_TASKS_MAX, (void *) &s.gemm);
    ggml_format_name(y, "%s.fq_gemm", ggml_get_name(s.w));
    if (residual && !fuse)
        return ggml_add(C, residual, gate ? ggml_mul(C, y, gate) : y);
    return y;
}

// An act node with its own spec (outliving the graph) and per-node gamma and
// shift: pi0.5's adaRMS, gamma = 1 + scale, quantized in the same pass. x is
// the pre-norm hidden state; s supplies the site's ascale.
inline ggml_tensor * fq_act_mod(ggml_context * C, const FqActSpec * spec, const FqLinear & s, ggml_tensor * x,
                                ggml_tensor * gamma, ggml_tensor * shift) {
    GGML_ASSERT(spec->has_gamma && spec->has_shift && x->type == GGML_TYPE_F32 && x->ne[0] == spec->K);
    ggml_tensor * xin = ggml_is_contiguous(x) ? x : ggml_cont(C, x);
    ggml_tensor * args[4] = { xin, gamma, shift, s.ascale };
    ggml_tensor * t = ggml_custom_4d(C, GGML_TYPE_I8, fq_act_row_bytes(spec->K, fq_act_store_bits(*spec)),
                                     ggml_nelements(xin) / spec->K, 1, 1, args, s.ascale ? 4 : 3, fq_act_cpu,
                                     GGML_N_TASKS_MAX, (void *) spec);
    ggml_format_name(t, "%s.fq_act_mod", ggml_get_name(s.w));
    return t;
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
inline ggml_tensor * fq_linear(ggml_context * C, const FqLinear & s, ggml_tensor * x, ggml_tensor * residual = nullptr,
                               ggml_tensor * gate = nullptr) {
    return fq_gemm(C, s, fq_act(C, s, x), residual, gate);
}

}  // namespace vla
