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

// A Qwen2 / Llama decoder layer as fused nodes (layers/fused_ew.h), for a
// backend that claims them (SYCL) with view-taking flash attention:
//
//   RMSNorm into BF16, one q|k|v GEMM with its biases (F32 out), RoPE writing q
//   as F16 and k/v into F16 [hd, n_kv, >= seq] buffers, attention over them,
//   the o GEMM with the residual as its epilogue, RMSNorm, one gate|up GEMM,
//   SwiGLU, and the down GEMM with the residual.
//
// 8 nodes where the per-op graph runs ~20. The residual stream stays F32.

#pragma once

#include "act_dtype.h"
#include "layers/fused_ew.h"
#include "loader.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <string>

namespace vla {

struct FusedLmW {
    ggml_tensor *attn_norm = nullptr, *Wqkv = nullptr, *bqkv = nullptr, *Wo = nullptr;
    ggml_tensor *ffn_norm = nullptr, *Wgu = nullptr, *Wdown = nullptr;
    explicit operator bool() const { return Wqkv != nullptr; }
};

struct FusedLmDims {
    int64_t head_dim = 0, n_q = 0, n_kv = 0;
    float   rope_base = 10000.0f;  // NEOX RoPE, no frequency scaling
};

// The layer under GGUF prefix b (e.g. "lm.blk.3."), with q|k|v (and their
// biases, if qkv_bias) and gate|up concatenated along N at load.
inline FusedLmW fused_lm_declare(WeightLoader & L, const std::string & b, bool qkv_bias) {
    FusedLmW w;
    w.attn_norm = L.f32("%sattn_norm.weight", b.c_str());
    w.ffn_norm  = L.f32("%sffn_norm.weight", b.c_str());
    w.Wqkv = L.fuse_gemm((b + "attn_qkv.weight").c_str(), {b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"});
    if (qkv_bias)
        w.bqkv = L.fuse_f32((b + "attn_qkv.bias").c_str(), {b + "attn_q.bias", b + "attn_k.bias", b + "attn_v.bias"});
    w.Wo    = L.gemm("%sattn_o.weight", b.c_str());
    w.Wgu   = L.fuse_gemm((b + "ffn_gate_up.weight").c_str(), {b + "ffn_gate.weight", b + "ffn_up.weight"});
    w.Wdown = L.gemm("%sffn_down.weight", b.c_str());
    return w;
}

// h: F32 [hidden, seq]; positions I32 [seq]; mask_f16 F16 [seq, seq] or null;
// qmask F32 [1, seq] or null: zeroes the attention rows of masked queries (the
// o projection has no bias, so that is zeroing its output rows). kbuf / vbuf:
// F16 [hd, n_kv, >= seq], rewritten here; a caller may share one pair across
// layers, since each layer's RoPE node runs after the previous attention.
// Returns F32 [hidden, seq].
inline ggml_tensor * fused_lm_layer(ggml_context * C, const FusedLmW & w, const FusedLmDims & d,
                                    const FusedRmsSpec & rms, ggml_tensor * h, ggml_tensor * positions,
                                    ggml_tensor * mask_f16, int64_t seq, ggml_tensor * qmask, ggml_tensor * kbuf,
                                    ggml_tensor * vbuf) {
    const int64_t hd = d.head_dim;
    ggml_tensor * qkv = fused_linear(C, w.Wqkv, fused_rms(C, h, w.attn_norm, GGML_TYPE_BF16, rms), w.bqkv, nullptr,
                                     GGML_TYPE_F32, 0);
    ggml_tensor * q16 = fused_rope_kv(C, qkv, positions, kbuf, vbuf, (int) d.n_q, (int) d.n_kv, (int) hd, 0,
                                      d.rope_base);
    auto rows = [&](ggml_tensor * b) {
        return ggml_permute(C, ggml_view_3d(C, b, hd, d.n_kv, seq, b->nb[1], b->nb[2], 0), 0, 2, 1, 3);
    };
    ggml_tensor * fa = ggml_flash_attn_ext(C, ggml_permute(C, q16, 0, 2, 1, 3), rows(kbuf), rows(vbuf), mask_f16,
                                           1.0f/std::sqrt((float) hd), 0.0f, 0.0f);
    ggml_prec_set_acc(fa, GGML_PREC_F32);
    // the buffers are read through views of the leaves: order after the writes
    add_dep(fa, q16);
    ggml_tensor * att;
    if (qmask) {
        att = ggml_cast(C, ggml_mul(C, ggml_reshape_2d(C, fa, d.n_q * hd, seq), qmask), GGML_TYPE_BF16);
    } else {
        retype(fa, GGML_TYPE_BF16);
        att = ggml_reshape_2d(C, fa, d.n_q * hd, seq);
    }
    ggml_tensor * h1 = fused_linear(C, w.Wo, att, nullptr, h, GGML_TYPE_F32, 0);
    ggml_tensor * gu = fused_linear(C, w.Wgu, fused_rms(C, h1, w.ffn_norm, GGML_TYPE_BF16, rms), nullptr, nullptr,
                                    GGML_TYPE_BF16, 0);
    const int64_t n = gu->ne[0]/2;
    ggml_tensor * g = ggml_view_2d(C, gu, n, gu->ne[1], gu->nb[1], 0);
    ggml_tensor * u = ggml_view_2d(C, gu, n, gu->ne[1], gu->nb[1], n * ggml_element_size(gu));
    return fused_linear(C, w.Wdown, fused_swiglu(C, g, u), nullptr, h1, GGML_TYPE_F32, 0);
}

// The K/V buffer pair fused_lm_layer writes, as graph leaves the allocator
// keeps for the whole graph.
inline void fused_lm_kv(ggml_context * C, const FusedLmDims & d, int64_t seq, ggml_tensor ** kbuf,
                        ggml_tensor ** vbuf) {
    *kbuf = ggml_new_tensor_3d(C, GGML_TYPE_F16, d.head_dim, d.n_kv, seq);
    *vbuf = ggml_new_tensor_3d(C, GGML_TYPE_F16, d.head_dim, d.n_kv, seq);
    ggml_set_output(*kbuf);
    ggml_set_output(*vbuf);
}

}  // namespace vla
