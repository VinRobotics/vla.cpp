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
 * @file fmha_kv.h
 * @brief Attention over a cached KV prefix plus new keys, without concatenating them.
 *
 * pi0.5's action expert attends to the prefix KV (fixed for the whole denoise
 * loop) plus its own 50 tokens, every layer, every step. The plain graph
 * concatenates the two KV segments and copies them each time. This node takes
 * them apart instead: the prefix K/V once as F16 (cast once per layer, outside
 * the loop) and the new K/V as F32 views, all heads non-causal, no mask.
 *
 * It is a GGML_OP_CUSTOM claimed on SYCL by src/sycl/vla_sycl_fmha.cpp (sycl-tla's
 * cached-KV flash attention, reading the F16 prefix in place). Elsewhere the
 * CPU function below runs it, slowly but exactly; graphs only build it when that
 * SYCL kernel is compiled in (fmha_kv_enabled).
 */

#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace vla {

constexpr uint32_t FMHA_KV_MAGIC = 0x564b4d46u;   // "FMKV"

struct FmhaKvSpec {
    uint32_t magic;
    float    scale;
};

/// True when the SYCL kernel for this node is built in and not turned off
/// (VLA_SYCL_FMHA=0).
inline bool fmha_kv_enabled() {
#ifdef VLA_SYCL_FMHA
    static const bool on = [] {
        const char * e = std::getenv("VLA_SYCL_FMHA");
        return !(e && std::strcmp(e, "0") == 0);
    }();
    return on;
#else
    return false;
#endif
}

/// One spec per distinct scale, alive for the process (graphs keep pointers to it).
inline const FmhaKvSpec * fmha_kv_spec(float scale) {
    static std::mutex mu;
    static std::map<uint32_t, std::unique_ptr<FmhaKvSpec>> specs;
    uint32_t key;
    std::memcpy(&key, &scale, sizeof(key));
    std::lock_guard<std::mutex> lk(mu);
    auto & s = specs[key];
    if (!s) s.reset(new FmhaKvSpec{FMHA_KV_MAGIC, scale});
    return s.get();
}

namespace detail {

inline float fmha_kv_at(const ggml_tensor * t, int64_t d, int64_t s, int64_t h) {
    const char * p = static_cast<const char *>(t->data) + d * t->nb[0] + s * t->nb[1] + h * t->nb[2];
    return t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p))
                                    : *reinterpret_cast<const float *>(p);
}

}  // namespace detail

/// CPU reference: dst [D, Hq, Sq] from Q [D, Sq, Hq] against the keys of
/// Kc/Vc [D, Sc, Hkv] followed by Kn/Vn [D, Sn, Hkv].
inline void fmha_kv_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const float scale = static_cast<const FmhaKvSpec *>(userdata)->scale;
    const ggml_tensor * Q = dst->src[0], * Kc = dst->src[1], * Vc = dst->src[2], * Kn = dst->src[3], * Vn = dst->src[4];
    const int64_t D = Q->ne[0], Sq = Q->ne[1], Hq = Q->ne[2], Sc = Kc->ne[1], Sn = Kn->ne[1], Hkv = Kc->ne[2];
    const int64_t grp = Hq / Hkv, rows = Hq * Sq;
    std::vector<float> sc((size_t) (Sc + Sn)), o((size_t) D);
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t h = row / Sq, s = row % Sq, hk = h / grp;
        float mx = -INFINITY;
        for (int64_t t = 0; t < Sc + Sn; ++t) {
            const ggml_tensor * K = t < Sc ? Kc : Kn;
            const int64_t tt = t < Sc ? t : t - Sc;
            float a = 0.f;
            for (int64_t d = 0; d < D; ++d) a += detail::fmha_kv_at(Q, d, s, h) * detail::fmha_kv_at(K, d, tt, hk);
            sc[t] = a * scale;
            mx = std::fmax(mx, sc[t]);
        }
        float den = 0.f;
        std::fill(o.begin(), o.end(), 0.f);
        for (int64_t t = 0; t < Sc + Sn; ++t) {
            const ggml_tensor * V = t < Sc ? Vc : Vn;
            const int64_t tt = t < Sc ? t : t - Sc;
            const float w = std::exp(sc[t] - mx);
            den += w;
            for (int64_t d = 0; d < D; ++d) o[d] += w * detail::fmha_kv_at(V, d, tt, hk);
        }
        float * out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + s * dst->nb[2] + h * dst->nb[1]);
        for (int64_t d = 0; d < D; ++d) out[d] = o[d] / den;
    }
}

/**
 * @brief softmax(scale * Q K^T) V over [Kc; Kn], as ggml_flash_attn_ext lays it out.
 * @param Q  F32 [D, Sq, Hq], any strides with contiguous rows.
 * @param Kc,Vc Cached prefix, [D, Sc, Hkv]; F16 contiguous for the SYCL kernel.
 * @param Kn,Vn New keys/values, F32 [D, Sn, Hkv] views.
 * @return F32 [D, Hq, Sq].
 */
inline ggml_tensor * fmha_kv(ggml_context * C, ggml_tensor * Q, ggml_tensor * Kc, ggml_tensor * Vc,
                             ggml_tensor * Kn, ggml_tensor * Vn, float scale) {
    ggml_tensor * args[5] = { Q, Kc, Vc, Kn, Vn };
    return ggml_custom_4d(C, GGML_TYPE_F32, Q->ne[0], Q->ne[2], Q->ne[1], 1, args, 5, fmha_kv_cpu,
                          GGML_N_TASKS_MAX, (void *) fmha_kv_spec(scale));
}

}  // namespace vla
