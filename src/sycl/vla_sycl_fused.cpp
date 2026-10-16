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
#ifdef VLA_FQ_SYCL_DNNL
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#endif

#include <cstring>
#include <map>
#include <tuple>
#include <unordered_map>

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
// variance (re-read from cache), then the affine. OUT float or bfloat16.
template <class OUT>
void run_ln(sycl::queue & q, const float * x, const float * w, const float * b, OUT * y, int64_t K, int64_t T,
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
        OUT * yr = y + t * K;
        for (int64_t k = lane; k < K; k += SG) yr[k] = OUT(((xr[k] - mean) * rstd) * w[k] + b[k]);
    });
}

#ifdef VLA_FQ_SYCL_DNNL
// vit_linear on oneDNN: one matmul, BF16 x and W in their ggml layouts (x rows
// [T][K], W rows [N][K]), bias and the post-op (gelu_tanh, or + residual) fused,
// F32 or BF16 out. On the Arc B390 this runs SigLIP's projections 1.4-2x faster
// than ggml-sycl's mul_mat (which also converts x and adds the bias apart).
struct VitDnnl {
    dnnl::engine eng;
    dnnl::stream strm;
    std::map<std::tuple<int64_t, int64_t, int64_t, int, int>, dnnl::matmul> prims;   // T, K, N, post, dst bf16
};

VitDnnl & vit_dnnl(sycl::queue & q) {
    static std::unordered_map<sycl::queue *, VitDnnl> states;
    auto it = states.find(&q);
    if (it == states.end()) {
        VitDnnl st;
        st.eng  = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        st.strm = dnnl::sycl_interop::make_stream(st.eng, q);
        it = states.emplace(&q, std::move(st)).first;
    }
    return it->second;
}

bool run_linear(sycl::queue & q, ggml_tensor * dst, const VitFusedSpec & s) {
    const ggml_tensor * x = dst->src[0], * W = dst->src[1], * b = dst->src[2], * r = dst->src[3];
    if (!x || !W || !b || x->type != GGML_TYPE_BF16 || W->type != GGML_TYPE_BF16 || !ggml_is_contiguous(x) ||
        !ggml_is_contiguous(W) || !f32_rows(b) || !ggml_is_contiguous(dst) ||
        (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_BF16))
        return false;
    const int64_t K = x->ne[0], T = ggml_nrows(x), N = W->ne[1];
    if (W->ne[0] != K || dst->ne[0] != N || ggml_nrows(dst) != T || b->ne[0] != N) return false;
    if (s.post == VitPost::RES && (!f32_rows(r) || ggml_nelements(r) != N * T)) return false;
    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    using md  = dnnl::memory::desc;
    VitDnnl & st = vit_dnnl(q);
    const bool obf = dst->type == GGML_TYPE_BF16;
    const md a_md({ T, K }, dt::bf16, tag::ab), w_md({ K, N }, dt::bf16, tag::ba), b_md({ 1, N }, dt::f32, tag::ab);
    const md c_md({ T, N }, obf ? dt::bf16 : dt::f32, tag::ab), r_md({ T, N }, dt::f32, tag::ab);
    const auto key = std::make_tuple(T, K, N, (int) s.post, (int) obf);
    auto it = st.prims.find(key);
    if (it == st.prims.end()) {
        dnnl::primitive_attr attr;
        dnnl::post_ops po;
        if (s.post == VitPost::GELU) po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
        if (s.post == VitPost::RES)  po.append_binary(dnnl::algorithm::binary_add, r_md);
        attr.set_post_ops(po);
        try {
            it = st.prims.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(st.eng, a_md, w_md, b_md, c_md, attr))).first;
        } catch (const dnnl::error &) {
            return false;
        }
    }
    using dnnl::sycl_interop::make_memory;
    using dnnl::sycl_interop::memory_kind;
    std::unordered_map<int, dnnl::memory> args = {
        { DNNL_ARG_SRC,     make_memory(a_md, st.eng, memory_kind::usm, x->data) },
        { DNNL_ARG_WEIGHTS, make_memory(w_md, st.eng, memory_kind::usm, W->data) },
        { DNNL_ARG_BIAS,    make_memory(b_md, st.eng, memory_kind::usm, b->data) },
        { DNNL_ARG_DST,     make_memory(c_md, st.eng, memory_kind::usm, dst->data) },
    };
    if (s.post == VitPost::RES)
        args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                     make_memory(r_md, st.eng, memory_kind::usm, r->data));
    it->second.execute(st.strm, args);
    return true;
}
#endif

}  // namespace

bool sycl_vit_fused_forward(ggml_tensor * dst, void * queue) {
    const VitFusedSpec * s = spec_of(dst);
    if (!s || s->kind == VitFusedKind::ATTN) return false;   // ATTN: src/sycl/vla_sycl_fmha.cpp
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    if (s->kind == VitFusedKind::LINEAR) {
#ifdef VLA_FQ_SYCL_DNNL
        return run_linear(q, dst, *s);
#else
        return false;
#endif
    }
    const ggml_tensor * x = dst->src[0], * a = dst->src[1], * c = dst->src[2];
    const int64_t K = dst->ne[0], T = ggml_nrows(dst), n = K * T;
    if (!f32_rows(x) || !f32_rows(a) || ggml_nelements(x) != ggml_nelements(dst) || a->ne[0] != K ||
        !ggml_is_contiguous(dst))
        return false;
    const float * xp = static_cast<const float *>(x->data);
    const float * ap = static_cast<const float *>(a->data);
    if (s->kind == VitFusedKind::LN) {
        if (!f32_rows(c) || c->ne[0] != K) return false;
        const float * cp = static_cast<const float *>(c->data);
        if (dst->type == GGML_TYPE_BF16)
            run_ln(q, xp, ap, cp, static_cast<sycl::ext::oneapi::bfloat16 *>(dst->data), K, T, s->eps);
        else if (dst->type == GGML_TYPE_F32)
            run_ln(q, xp, ap, cp, static_cast<float *>(dst->data), K, T, s->eps);
        else
            return false;
        return true;
    }
    if (dst->type != GGML_TYPE_F32) return false;
    float * yp = static_cast<float *>(dst->data);
    switch (s->kind) {
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
    default:
        return false;
    }
}

bool sycl_vit_fused_supports(const ggml_tensor * op) { return spec_of(op) != nullptr; }

}  // namespace vla
