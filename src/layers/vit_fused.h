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

/**
 * @file vit_fused.h
 * @brief Fused nodes for a ViT block on SYCL.
 *
 * ggml-sycl runs a LayerNorm as norm -> mul -> add and a biased linear as
 * convert -> mul_mat -> add (-> gelu, -> residual add): one kernel each, and on
 * an iGPU a small kernel costs about as much as the work in it. These
 * GGML_OP_CUSTOM nodes do each chain in one pass:
 *
 *   vit_ln(x, w, b)                 LayerNorm with its affine (F32 or BF16 out)
 *   vit_bias_gelu(y, b)             gelu_tanh(y + b)
 *   vit_bias_res(y, b, r)           r + (y + b)
 *   vit_linear(x, W, b, post, r)    x W^T + b, then gelu_tanh or + r: one oneDNN
 *                                   matmul with post-ops (BF16 x and W)
 *   vit_attn(qkv, heads, hd, seq)   softmax(q k^T / sqrt(hd)) v per head and view,
 *                                   from one fused [3E, T] projection, BF16 out
 *
 * src/sycl/vla_sycl_fused.cpp (and vla_sycl_fmha.cpp for vit_attn) claim them
 * by the magic word in their userdata; the CPU functions below are the same
 * math, for any other backend and the tests. Graphs build them only on SYCL
 * (vit_fused_enabled).
 */

#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace vla {

constexpr uint32_t VIT_FUSED_MAGIC = 0x31545656u;   // "VVT1"

enum class VitFusedKind : uint32_t { LN = 1, BIAS_GELU = 2, BIAS_RES = 3, LINEAR = 4, ATTN = 5 };
enum class VitPost : int { NONE = 0, GELU = 1, RES = 2 };

struct VitFusedSpec {
    uint32_t     magic;
    VitFusedKind kind;
    float        eps;     // LN
    VitPost      post;    // LINEAR
    int          heads;   // ATTN
    int          hd;      // ATTN
    int64_t      seq;     // ATTN: tokens per view
};

/// True on SYCL builds unless VLA_SYCL_FUSED=0.
inline bool vit_fused_enabled() {
#ifdef GGML_USE_SYCL
    static const bool on = [] {
        const char * e = std::getenv("VLA_SYCL_FUSED");
        return !(e && std::strcmp(e, "0") == 0);
    }();
    return on;
#else
    return false;
#endif
}

/// ggml's GELU (tanh form).
inline float vit_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

namespace detail {

// One spec per distinct parameter set, alive for the process (graphs keep
// pointers to it).
inline const VitFusedSpec * vit_spec(VitFusedKind k, float eps, VitPost post = VitPost::NONE, int heads = 0,
                                     int hd = 0, int64_t seq = 0) {
    static std::mutex mu;
    static std::map<std::tuple<uint32_t, uint32_t, int, int, int, int64_t>, std::unique_ptr<VitFusedSpec>> specs;
    uint32_t e;
    std::memcpy(&e, &eps, sizeof(e));
    std::lock_guard<std::mutex> lk(mu);
    auto & p = specs[std::make_tuple((uint32_t) k, e, (int) post, heads, hd, seq)];
    if (!p) p.reset(new VitFusedSpec{VIT_FUSED_MAGIC, k, eps, post, heads, hd, seq});
    return p.get();
}

inline float vit_at(const ggml_tensor * t, int64_t i) {
    return t->type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(static_cast<const ggml_bf16_t *>(t->data)[i])
         : t->type == GGML_TYPE_F16  ? ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(t->data)[i])
                                     : static_cast<const float *>(t->data)[i];
}

inline void vit_put(ggml_tensor * t, int64_t i, float v) {
    if (t->type == GGML_TYPE_BF16) static_cast<ggml_bf16_t *>(t->data)[i] = ggml_fp32_to_bf16(v);
    else                           static_cast<float *>(t->data)[i] = v;
}

inline void vit_fused_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const VitFusedSpec & s = *static_cast<const VitFusedSpec *>(userdata);
    const ggml_tensor * x = dst->src[0];
    const int64_t K = x->ne[0], N = dst->ne[0], T = ggml_nrows(dst);
    if (s.kind == VitFusedKind::ATTN) {
        const int64_t H = s.heads, D = s.hd, E = H * D, S = s.seq, B = T / S;
        const float scale = 1.0f / std::sqrt((float) D);
        std::vector<float> sc((size_t) S);
        for (int64_t job = ith; job < B * H * S; job += nth) {
            const int64_t b = job / (H * S), h = (job / S) % H, i = job % S;
            const int64_t qi = (b * S + i) * 3 * E + h * D;
            float mx = -INFINITY;
            for (int64_t j = 0; j < S; ++j) {
                const int64_t kj = (b * S + j) * 3 * E + E + h * D;
                float a = 0.f;
                for (int64_t d = 0; d < D; ++d) a += vit_at(x, qi + d) * vit_at(x, kj + d);
                sc[j] = a * scale;
                mx = std::fmax(mx, sc[j]);
            }
            float z = 0.f;
            for (int64_t j = 0; j < S; ++j) { sc[j] = std::exp(sc[j] - mx); z += sc[j]; }
            for (int64_t d = 0; d < D; ++d) {
                float o = 0.f;
                for (int64_t j = 0; j < S; ++j) o += sc[j] * vit_at(x, (b * S + j) * 3 * E + 2 * E + h * D + d);
                vit_put(dst, (b * S + i) * E + h * D + d, o / z);
            }
        }
        return;
    }
    const ggml_tensor * a = dst->src[1];   // w (LN), bias, or W (LINEAR)
    for (int64_t t = ith; t < T; t += nth) {
        if (s.kind == VitFusedKind::LN) {
            const ggml_tensor * b = dst->src[2];
            float mean = 0.f;
            for (int64_t k = 0; k < K; ++k) mean += vit_at(x, t * K + k);
            mean /= (float) K;
            float var = 0.f;
            for (int64_t k = 0; k < K; ++k) { const float d = vit_at(x, t * K + k) - mean; var += d * d; }
            const float rstd = 1.0f / std::sqrt(var / (float) K + s.eps);
            for (int64_t k = 0; k < K; ++k)
                vit_put(dst, t * K + k, ((vit_at(x, t * K + k) - mean) * rstd) * vit_at(a, k) + vit_at(dst->src[2], k));
            (void) b;
        } else if (s.kind == VitFusedKind::BIAS_GELU) {
            for (int64_t k = 0; k < K; ++k) vit_put(dst, t * K + k, vit_gelu_tanh(vit_at(x, t * K + k) + vit_at(a, k)));
        } else if (s.kind == VitFusedKind::BIAS_RES) {
            for (int64_t k = 0; k < K; ++k)
                vit_put(dst, t * K + k, vit_at(dst->src[2], t * K + k) + (vit_at(x, t * K + k) + vit_at(a, k)));
        } else {   // LINEAR: a = W [K, N] (row n contiguous), src[2] = bias, src[3] = residual
            for (int64_t n = 0; n < N; ++n) {
                float acc = 0.f;
                for (int64_t k = 0; k < K; ++k) acc += vit_at(x, t * K + k) * vit_at(a, n * K + k);
                float v = acc + vit_at(dst->src[2], n);
                if (s.post == VitPost::GELU) v = vit_gelu_tanh(v);
                if (s.post == VitPost::RES)  v = v + vit_at(dst->src[3], t * N + n);
                vit_put(dst, t * N + n, v);
            }
        }
    }
}

inline ggml_tensor * vit_node(ggml_context * C, const VitFusedSpec * spec, ggml_type type, int64_t ne0,
                              ggml_tensor * x, ggml_tensor * a, ggml_tensor * b, ggml_tensor * c = nullptr) {
    if (!ggml_is_contiguous(x)) x = ggml_cont(C, x);
    ggml_tensor * args[4] = { x, a, b, c };
    const int n = c ? 4 : b ? 3 : a ? 2 : 1;
    return ggml_custom_4d(C, type, ne0, ggml_nrows(x), 1, 1, args, n, vit_fused_cpu, GGML_N_TASKS_MAX,
                          (void *) spec);
}

}  // namespace detail

/// LayerNorm over rows of x (F32), then * w + b; F32 or BF16 out.
inline ggml_tensor * vit_ln(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps,
                            ggml_type out = GGML_TYPE_F32) {
    return detail::vit_node(C, detail::vit_spec(VitFusedKind::LN, eps), out, x->ne[0], x, w, b);
}

/// gelu_tanh(y + b), b broadcast over rows.
inline ggml_tensor * vit_bias_gelu(ggml_context * C, ggml_tensor * y, ggml_tensor * b) {
    return detail::vit_node(C, detail::vit_spec(VitFusedKind::BIAS_GELU, 0.0f), GGML_TYPE_F32, y->ne[0], y, b,
                            nullptr);
}

/// res + (y + b), res shaped like y.
inline ggml_tensor * vit_bias_res(ggml_context * C, ggml_tensor * y, ggml_tensor * b, ggml_tensor * res) {
    if (!ggml_is_contiguous(res)) res = ggml_cont(C, res);
    return detail::vit_node(C, detail::vit_spec(VitFusedKind::BIAS_RES, 0.0f), GGML_TYPE_F32, y->ne[0], y, b, res);
}

/// x [K, T] (BF16) times W [K, N] (BF16, ggml's weight layout) plus b [N], then
/// gelu_tanh (post GELU) or + res [N, T] F32 (post RES); `out` F32 or BF16.
inline ggml_tensor * vit_linear(ggml_context * C, ggml_tensor * x, ggml_tensor * W, ggml_tensor * b, VitPost post,
                                ggml_tensor * res, ggml_type out) {
    if (res && !ggml_is_contiguous(res)) res = ggml_cont(C, res);
    return detail::vit_node(C, detail::vit_spec(VitFusedKind::LINEAR, 0.0f, post), out, W->ne[1], x, W, b,
                            post == VitPost::RES ? res : nullptr);
}

/// Attention over a fused q|k|v projection [3*heads*hd, seq*views] (F32),
/// views independent, no mask; BF16 [heads*hd, seq*views] out.
inline ggml_tensor * vit_attn(ggml_context * C, ggml_tensor * qkv, int heads, int hd, int64_t seq) {
    return detail::vit_node(C, detail::vit_spec(VitFusedKind::ATTN, 0.0f, VitPost::NONE, heads, hd, seq),
                            GGML_TYPE_BF16, (int64_t) heads * hd, qkv, nullptr, nullptr);
}

}  // namespace vla
