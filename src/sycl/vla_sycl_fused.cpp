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

// SYCL kernels for the fused ViT nodes of src/layers/vit_fused.h.

#include "sycl/vla_sycl_fused.h"

#include "layers/vit_fused.h"

#include <sycl/sycl.hpp>

#include <cstring>

namespace vla {

namespace {

constexpr int SG   = 16;   // lanes per row
constexpr int ROWS = 8;    // rows (sub-groups) per work-group

const VitFusedSpec * spec_of(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) return nullptr;
    struct custom_params { void * fun; int n_tasks; void * userdata; };
    custom_params cp;
    std::memcpy(&cp, op->op_params, sizeof(cp));
    const auto * s = static_cast<const VitFusedSpec *>(cp.userdata);
    return (s && s->magic == VIT_FUSED_MAGIC) ? s : nullptr;
}

bool f32_rows(const ggml_tensor * t) { return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); }

// One sub-group per row: two passes over the row for the mean and the
// variance (re-read from cache), then the affine.
void run_ln(sycl::queue & q, const float * x, const float * w, const float * b, float * y, int64_t K, int64_t T,
            float eps) {
    const size_t groups = (size_t) ((T + ROWS - 1) / ROWS);
    q.parallel_for(sycl::nd_range<1>(groups * ROWS * SG, ROWS * SG),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int  lane = (int) sg.get_local_linear_id();
        const int64_t t = (int64_t) it.get_group(0) * ROWS + (int64_t) sg.get_group_linear_id();
        if (t >= T) return;
        const float * xr = x + t * K;
        float s = 0.f;
        for (int64_t k = lane; k < K; k += SG) s += xr[k];
        const float mean = sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) K;
        float v = 0.f;
        for (int64_t k = lane; k < K; k += SG) { const float d = xr[k] - mean; v += d * d; }
        const float rstd = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, v, sycl::plus<float>()) / (float) K + eps);
        float * yr = y + t * K;
        for (int64_t k = lane; k < K; k += SG) yr[k] = ((xr[k] - mean) * rstd) * w[k] + b[k];
    });
}

}  // namespace

bool sycl_vit_fused_forward(ggml_tensor * dst, void * queue) {
    const VitFusedSpec * s = spec_of(dst);
    if (!s) return false;
    const ggml_tensor * x = dst->src[0], * a = dst->src[1], * c = dst->src[2];
    if (!f32_rows(dst) || !f32_rows(x) || !f32_rows(a) || ggml_nelements(x) != ggml_nelements(dst)) return false;
    const int64_t K = dst->ne[0], T = ggml_nrows(dst), n = K * T;
    if (a->ne[0] != K) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    const float * xp = static_cast<const float *>(x->data);
    const float * ap = static_cast<const float *>(a->data);
    float *       yp = static_cast<float *>(dst->data);
    switch (s->kind) {
    case VitFusedKind::LN:
        if (!f32_rows(c) || c->ne[0] != K) return false;
        run_ln(q, xp, ap, static_cast<const float *>(c->data), yp, K, T, s->eps);
        return true;
    case VitFusedKind::BIAS_GELU:
        q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
            const float v = xp[i] + ap[i % K];
            yp[i] = 0.5f * v * (1.0f + sycl::tanh(0.7978845608028654f * v * (1.0f + 0.044715f * v * v)));
        });
        return true;
    case VitFusedKind::BIAS_RES: {
        if (!f32_rows(c) || ggml_nelements(c) != n) return false;
        const float * rp = static_cast<const float *>(c->data);
        q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { yp[i] = rp[i] + (xp[i] + ap[i % K]); });
        return true;
    }
    }
    return false;
}

bool sycl_vit_fused_supports(const ggml_tensor * op) { return spec_of(op) != nullptr; }

}  // namespace vla
