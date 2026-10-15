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
 * @brief Fused elementwise nodes for a ViT block on SYCL.
 *
 * ggml-sycl runs a LayerNorm as norm -> mul -> add and a biased linear as
 * mul_mat -> add (-> gelu, -> residual add): one kernel each, and on an iGPU a
 * small kernel costs about as much as the work in it. These GGML_OP_CUSTOM
 * nodes do each chain in one pass:
 *
 *   vit_ln(x, w, b)          LayerNorm with its affine
 *   vit_bias_gelu(y, b)      gelu_tanh(y + b)
 *   vit_bias_res(y, b, r)    r + (y + b)
 *
 * src/sycl/vla_sycl_fused.cpp claims them by the magic word in their userdata;
 * the CPU functions below are the same math, for any other backend and the
 * tests. Graphs build them only on SYCL (vit_fused_enabled).
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
#include <utility>

namespace vla {

constexpr uint32_t VIT_FUSED_MAGIC = 0x31545656u;   // "VVT1"

enum class VitFusedKind : uint32_t { LN = 1, BIAS_GELU = 2, BIAS_RES = 3 };

struct VitFusedSpec {
    uint32_t     magic;
    VitFusedKind kind;
    float        eps;
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

// One spec per (kind, eps), alive for the process (graphs keep pointers to it).
inline const VitFusedSpec * vit_spec(VitFusedKind k, float eps) {
    static std::mutex mu;
    static std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<VitFusedSpec>> specs;
    uint32_t e;
    std::memcpy(&e, &eps, sizeof(e));
    std::lock_guard<std::mutex> lk(mu);
    auto & p = specs[{ (uint32_t) k, e }];
    if (!p) p.reset(new VitFusedSpec{VIT_FUSED_MAGIC, k, eps});
    return p.get();
}

inline void vit_fused_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const VitFusedSpec & s = *static_cast<const VitFusedSpec *>(userdata);
    const ggml_tensor * x = dst->src[0];
    const int64_t K = dst->ne[0], T = ggml_nrows(dst);
    const float * a = static_cast<const float *>(dst->src[1]->data);   // w (LN) or bias
    for (int64_t t = ith; t < T; t += nth) {
        const float * xr = static_cast<const float *>(x->data) + t * K;
        float *       yr = static_cast<float *>(dst->data) + t * K;
        if (s.kind == VitFusedKind::LN) {
            const float * b = static_cast<const float *>(dst->src[2]->data);
            float mean = 0.f;
            for (int64_t k = 0; k < K; ++k) mean += xr[k];
            mean /= (float) K;
            float var = 0.f;
            for (int64_t k = 0; k < K; ++k) { const float d = xr[k] - mean; var += d * d; }
            const float rstd = 1.0f / std::sqrt(var / (float) K + s.eps);
            for (int64_t k = 0; k < K; ++k) yr[k] = ((xr[k] - mean) * rstd) * a[k] + b[k];
        } else if (s.kind == VitFusedKind::BIAS_GELU) {
            for (int64_t k = 0; k < K; ++k) yr[k] = vit_gelu_tanh(xr[k] + a[k]);
        } else {
            const float * r = static_cast<const float *>(dst->src[2]->data) + t * K;
            for (int64_t k = 0; k < K; ++k) yr[k] = r[k] + (xr[k] + a[k]);
        }
    }
}

inline ggml_tensor * vit_node(ggml_context * C, VitFusedKind k, float eps, ggml_tensor * x, ggml_tensor * a,
                              ggml_tensor * b) {
    if (!ggml_is_contiguous(x)) x = ggml_cont(C, x);
    ggml_tensor * args[3] = { x, a, b };
    return ggml_custom_4d(C, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, b ? 3 : 2,
                          vit_fused_cpu, GGML_N_TASKS_MAX, (void *) vit_spec(k, eps));
}

}  // namespace detail

/// LayerNorm over rows of x (F32), then * w + b.
inline ggml_tensor * vit_ln(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    return detail::vit_node(C, VitFusedKind::LN, eps, x, w, b);
}

/// gelu_tanh(y + b), b broadcast over rows.
inline ggml_tensor * vit_bias_gelu(ggml_context * C, ggml_tensor * y, ggml_tensor * b) {
    return detail::vit_node(C, VitFusedKind::BIAS_GELU, 0.0f, y, b, nullptr);
}

/// res + (y + b), res shaped like y.
inline ggml_tensor * vit_bias_res(ggml_context * C, ggml_tensor * y, ggml_tensor * b, ggml_tensor * res) {
    if (!ggml_is_contiguous(res)) res = ggml_cont(C, res);
    return detail::vit_node(C, VitFusedKind::BIAS_RES, 0.0f, y, b, res);
}

}  // namespace vla
