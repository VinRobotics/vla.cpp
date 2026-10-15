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
 * @file fq_rope_attn.h
 * @brief pi0.5's action-expert attention in one node: FoldQuant q/k/v GEMM,
 *        RoPE, and the cached-KV flash attention of fmha_kv.h.
 *
 * Per layer and denoising step the expert otherwise runs the q/k/v GEMM, its
 * epilogue, RoPE on q and on k, three F16 conversions and the attention: eight
 * kernels around ~0.25 ms of work. On SYCL this node is three: the oneDNN
 * matmul (int32 sums), one pass that dequantizes, rotates and packs Q (heads
 * folded into rows), K and V as F16, and the sycl-tla kernel
 * (src/sycl/vla_sycl_fmha.cpp). The CPU function below is the same math.
 *
 * Shapes: one KV head (pi0.5's Gemma expert), head_dim 256, NeoX RoPE over the
 * whole head. Output F32 [hd, n_q, seq], as fmha_kv.
 */

#pragma once

#include "foldquant.h"
#include "foldquant_ref.h"
#include "layers/fmha_kv.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace vla {

constexpr uint32_t FQ_ROPE_ATTN_MAGIC = 0x52415146u;   // "FQAR"
constexpr uint32_t FQ_ROPE_PACK_MAGIC = 0x50525146u;   // "FQRP": fq_rope_pack (the prefix)

struct FqRopeAttnSpec {
    uint32_t           magic     = 0;         // FQ_ROPE_ATTN_MAGIC when in use
    const FqGemmSpec * gemm      = nullptr;   // the fused q|k|v site
    int                hd        = 0;
    int                nq        = 0;
    float              freq_base = 10000.0f;
    float              scale     = 0.0f;
};

/// NeoX RoPE angle of pair i (of hd/2) at position p, as ggml-sycl computes it.
inline float fq_rope_theta(int32_t p, int i, int hd, float base) {
    const float theta_scale = std::pow(base, -2.0f / (float) hd);
    return (float) p * std::pow(theta_scale, (float) i);
}

/// CPU reference: src[0..3] = the q|k|v site's W, act blob, wscale, bias|null;
/// src[4] = positions I32 [seq]; src[5], src[6] = cached K, V (F16 [hd, Sc, 1]).
inline void fq_rope_attn_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    const FqRopeAttnSpec & s = *static_cast<const FqRopeAttnSpec *>(userdata);
    if (ith != 0) return;
    const int64_t D = s.hd, H = s.nq, N = s.gemm->N, T = dst->ne[2], half = D / 2;
    const ggml_tensor * pos = dst->src[4], * Kc = dst->src[5], * Vc = dst->src[6];
    const int64_t Sc = Kc->ne[1];
    // q|k|v rows [T][N], as the FoldQuant GEMM node writes them.
    std::vector<float> qkv((size_t) N * T);
    ggml_tensor tmp = *dst;
    tmp.type = GGML_TYPE_F32;
    tmp.ne[0] = N; tmp.ne[1] = T; tmp.ne[2] = tmp.ne[3] = 1;
    tmp.nb[0] = sizeof(float); tmp.nb[1] = (size_t) N * sizeof(float); tmp.nb[2] = tmp.nb[3] = tmp.nb[1] * (size_t) T;
    tmp.data = qkv.data();
    for (int i = 4; i < GGML_MAX_SRC; ++i) tmp.src[i] = nullptr;
    fq_gemm_cpu(&tmp, 0, 1, (void *) s.gemm);
    auto rope = [&](float * v, int32_t p) {
        for (int64_t i = 0; i < half; ++i) {
            const float th = fq_rope_theta(p, (int) i, (int) D, s.freq_base);
            const float c = std::cos(th), sn = std::sin(th), x0 = v[i], x1 = v[i + half];
            v[i]        = x0 * c - x1 * sn;
            v[i + half] = x0 * sn + x1 * c;
        }
    };
    const int32_t * pp = static_cast<const int32_t *>(pos->data);
    for (int64_t t = 0; t < T; ++t) {
        float * row = qkv.data() + (size_t) t * N;
        for (int64_t h = 0; h <= H; ++h) rope(row + h * D, pp[t]);   // q heads, then k
    }
    auto kc = [&](const ggml_tensor * c, int64_t j, int64_t d) {
        return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(c->data)[j * D + d]);
    };
    std::vector<float> sc((size_t) (Sc + T));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h) {
            const float * q = qkv.data() + (size_t) t * N + h * D;
            float mx = -INFINITY;
            for (int64_t j = 0; j < Sc + T; ++j) {
                float a = 0.f;
                for (int64_t d = 0; d < D; ++d)
                    a += q[d] * (j < Sc ? kc(Kc, j, d) : qkv[(size_t) (j - Sc) * N + H * D + d]);
                sc[j] = a * s.scale;
                mx = std::fmax(mx, sc[j]);
            }
            float den = 0.f;
            for (int64_t j = 0; j < Sc + T; ++j) { sc[j] = std::exp(sc[j] - mx); den += sc[j]; }
            float * out = static_cast<float *>(dst->data) + (t * H + h) * D;
            for (int64_t d = 0; d < D; ++d) {
                float o = 0.f;
                for (int64_t j = 0; j < Sc + T; ++j)
                    o += sc[j] * (j < Sc ? kc(Vc, j, d) : qkv[(size_t) (j - Sc) * N + (H + 1) * D + d]);
                out[d] = o / den;
            }
        }
}

/// xq: the q|k|v act blob; positions I32 [seq]; Kc/Vc the cached prefix K/V as
/// packed F16 [hd, Sc, 1]. Returns F32 [hd, n_q, seq].
inline ggml_tensor * fq_rope_attn(ggml_context * C, const FqRopeAttnSpec * spec, const FqLinear & qkv, ggml_tensor * xq,
                                  ggml_tensor * positions, ggml_tensor * Kc, ggml_tensor * Vc) {
    GGML_ASSERT(spec->magic == FQ_ROPE_ATTN_MAGIC && xq->type == GGML_TYPE_I8 && Kc->type == GGML_TYPE_F16);
    ggml_tensor * args[7] = { qkv.w, xq, qkv.wscale, qkv.bias, positions, Kc, Vc };
    ggml_tensor * t = ggml_custom_4d(C, GGML_TYPE_F32, spec->hd, spec->nq, xq->ne[1], 1, args, 7, fq_rope_attn_cpu,
                                     GGML_N_TASKS_MAX, (void *) spec);
    ggml_format_name(t, "%s.rope_attn", ggml_get_name(qkv.w));
    return t;
}

/// fq_rope_pack's CPU reference: src[0..4] as fq_rope_attn; dst F16 [hd, H*seq + 2*seq]:
/// Q rows (head-major, h * seq + t), then K rows, then V rows, q and k rotated.
inline void fq_rope_pack_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    const FqRopeAttnSpec & s = *static_cast<const FqRopeAttnSpec *>(userdata);
    if (ith != 0) return;
    const int64_t D = s.hd, H = s.nq, N = s.gemm->N, T = dst->src[1]->ne[1], half = D / 2;
    std::vector<float> qkv((size_t) N * T);
    ggml_tensor tmp = *dst;
    tmp.type = GGML_TYPE_F32;
    tmp.ne[0] = N; tmp.ne[1] = T; tmp.ne[2] = tmp.ne[3] = 1;
    tmp.nb[0] = sizeof(float); tmp.nb[1] = (size_t) N * sizeof(float); tmp.nb[2] = tmp.nb[3] = tmp.nb[1] * (size_t) T;
    tmp.data = qkv.data();
    for (int i = 4; i < GGML_MAX_SRC; ++i) tmp.src[i] = nullptr;
    fq_gemm_cpu(&tmp, 0, 1, (void *) s.gemm);
    const int32_t * pp = static_cast<const int32_t *>(dst->src[4]->data);
    ggml_fp16_t * out = static_cast<ggml_fp16_t *>(dst->data);
    for (int64_t t = 0; t < T; ++t) {
        const float * row = qkv.data() + (size_t) t * N;
        for (int64_t h = 0; h < H + 2; ++h) {
            ggml_fp16_t * o = out + (h < H ? h * T + t : H * T + (h - H) * T + t) * D;
            for (int64_t i = 0; i < half; ++i) {
                float x0 = row[h * D + i], x1 = row[h * D + i + half];
                if (h <= H) {   // q heads and k
                    const float th = fq_rope_theta(pp[t], (int) i, (int) D, s.freq_base);
                    const float c = std::cos(th), sn = std::sin(th), y0 = x0 * c - x1 * sn, y1 = x0 * sn + x1 * c;
                    x0 = y0; x1 = y1;
                }
                o[i]        = ggml_fp32_to_fp16(x0);
                o[i + half] = ggml_fp32_to_fp16(x1);
            }
        }
    }
}

/// The prefix's q/k/v GEMM, RoPE and F16 packing in one node (one KV head):
/// F16 [hd, H*seq + 2*seq], Q head-major rows then K then V (fq_rope_pack_cpu).
inline ggml_tensor * fq_rope_pack(ggml_context * C, const FqRopeAttnSpec * spec, const FqLinear & qkv, ggml_tensor * xq,
                                  ggml_tensor * positions) {
    GGML_ASSERT(spec->magic == FQ_ROPE_PACK_MAGIC && xq->type == GGML_TYPE_I8);
    const int64_t T = xq->ne[1];
    ggml_tensor * args[5] = { qkv.w, xq, qkv.wscale, qkv.bias, positions };
    ggml_tensor * t = ggml_custom_4d(C, GGML_TYPE_F16, spec->hd, (int64_t) spec->nq * T + 2 * T, 1, 1, args, 5,
                                     fq_rope_pack_cpu, GGML_N_TASKS_MAX, (void *) spec);
    ggml_format_name(t, "%s.rope_pack", ggml_get_name(qkv.w));
    return t;
}

}  // namespace vla
