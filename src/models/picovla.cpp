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

// PicoVLA (fast_smolvla @ 70d79c8): DINOv3 ConvNeXt-T over each camera, merged
// 2x2 by a softmax gate conditioned on the pooled task embedding; a Llama
// backbone whose layer 0 reads the instruction alone and whose four joint
// layers read [state | images | language | 16 record tokens | 16 past slots];
// and a four-layer action expert that reads, layer for layer, the backbone's
// prefix K/V through small cross projections. MeanFlow in three steps.
//
// The record tokens are a one-frame memory: each call's record outputs are the
// next call's past slots, so the model keeps them between calls and
// Inputs::reset_memory clears them at an episode start. Everything else is one
// cached graph, keyed by the instruction length.

#include "arch.h"
#include "backend.h"
#include "gguf.h"
#include "gguf_reader.h"
#include "loader.h"
#include "model.h"
#include "options.h"
#include "scratch_ctx.h"
#include "layers/attn.h"
#include "layers/embed.h"
#include "layers/ffn.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "modules/preprocess.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vla {
namespace {

constexpr float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImagenetStd[3]  = {0.229f, 0.224f, 0.225f};

// nn.LayerNorm's default, used by the backbone's non-affine skip_norm and
// out_norm, which picovla builds without passing an eps.
constexpr float kLnEps = 1e-5f;

struct LlamaLayerW {
    ggml_tensor *attn_norm, *qkv, *o, *ffn_norm, *gate, *up, *down;
};

struct ExpertLayerW {
    LlamaLayerW   l;
    ggml_tensor * cross_k, * cross_v;
    ggml_tensor * ada_attn_w, * ada_attn_b, * ada_ffn_w, * ada_ffn_b;
};

struct BlockW {
    ggml_tensor *dw_w, *dw_b, *ln_w, *ln_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

struct StageW {
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr, *down_w = nullptr, *down_b = nullptr;
    std::vector<BlockW> blocks;
};

}  // namespace

struct PicoVlaModelArch : public ModelArchBase {
    PicoVlaModelArch() : ModelArchBase(Arch::PICOVLA) {}
    ~PicoVlaModelArch() override {
        graph.release();
        if (weight_buf)  ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend)     ggml_backend_free(backend);
    }

    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             mt          = GGML_TYPE_F32;

    int64_t hidden = 384, inter = 768, n_layers = 4, n_heads = 8, n_kv = 4, head_dim = 48;
    int64_t ex_hidden = 256, ex_inter = 512, n_record = 16, chunk = 16;
    int64_t max_state = 32, max_action = 32, real_state = 8, real_action = 7;
    int64_t n_views = 2, image_size = 448, stem = 4, max_lang = 48, vocab = 49280;
    int     num_steps = 3;
    float   rms_eps = 1e-5f, rope_base = 10000.0f, vis_eps = 1e-6f;
    float   t_lo = 0.01f, t_hi = 0.99f, min_period = 0.1f, max_period = 4.0f, norm_eps = 1e-8f;
    std::vector<int64_t> vis_depths, vis_dims;
    std::vector<int32_t> token_map;  // original id -> row of the compacted table

    ggml_tensor *tok_embd = nullptr, *lm_norm = nullptr, *record_tokens = nullptr;
    std::vector<LlamaLayerW>  vlm;      // [0] is the language-only layer
    std::vector<ExpertLayerW> expert;
    ggml_tensor *state_w = nullptr, *state_b = nullptr, *act_in_w = nullptr, *act_in_b = nullptr;
    ggml_tensor *act_out0_w = nullptr, *act_out0_b = nullptr, *act_out2_w = nullptr, *act_out2_b = nullptr;
    ggml_tensor *time_in_w = nullptr, *time_in_b = nullptr, *time_out_w = nullptr, *time_out_b = nullptr;
    ggml_tensor *cond_w = nullptr, *cond_b = nullptr, *out_ada_w = nullptr, *out_ada_b = nullptr;
    ggml_tensor *stem_w = nullptr, *stem_b = nullptr, *stem_norm_w = nullptr, *stem_norm_b = nullptr;
    std::vector<StageW> stages;
    ggml_tensor *vis_norm_w = nullptr, *vis_norm_b = nullptr;
    ggml_tensor *task_proj = nullptr, *vision_proj = nullptr, *conn_proj = nullptr;
    std::vector<float> state_mean, state_std, action_mean, action_std;

    // The record memory: the previous call's record outputs (out_norm applied),
    // fed back as the past slots. Zero after load and after reset_memory, which
    // is what the reference feeds when it has no past (past_mask false).
    std::vector<float> past;
    std::mt19937       rng{std::random_device{}()};

    struct Key {
        int64_t n_lang = -1;
        bool operator==(const Key & o) const { return n_lang == o.n_lang; }
    };
    struct IO {
        ggml_tensor *pixels = nullptr, *ids = nullptr, *pos = nullptr, *mask = nullptr, *state = nullptr;
        ggml_tensor *past = nullptr, *noise = nullptr, *time = nullptr, *dt = nullptr;
        ggml_tensor *actions = nullptr, *record = nullptr;
    };
    graph_cache<Key, IO> graph;

    int64_t grid() const      { return (image_size / stem) >> (int) (vis_dims.size() - 1); }  // 448 -> 14
    int64_t img_tokens() const { return (grid() / 2) * (grid() / 2); }
    int64_t prefix_len(int64_t L) const { return 1 + n_views*img_tokens() + L; }

    ggml_cgraph * build(ggml_context * C, IO & io, int64_t L) const;
    std::vector<float> predict(const Inputs& in) override;
};

namespace {

// [C, W, H, N] channels-last to [k*k*C, W/k, H/k, N]: each column gathers a
// k x k pixel block, channel fastest, then kx, then ky. The converter orders
// the stride-k conv weights the same way, so the conv is one matmul.
ggml_tensor * space_to_depth(ggml_context * C, ggml_tensor * x, int64_t k) {
    const int64_t c = x->ne[0], w = x->ne[1], h = x->ne[2], n = x->ne[3];
    ggml_tensor * t = ggml_reshape_4d(C, x, c*k, w/k, k, (h/k)*n);
    t = ggml_cont(C, ggml_permute(C, t, 0, 2, 1, 3));
    return ggml_reshape_4d(C, t, c*k*k, w/k, h/k, n);
}

ggml_tensor * stride_conv(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int64_t k) {
    ggml_tensor * s = space_to_depth(C, x, k);
    const int64_t ow = s->ne[1], oh = s->ne[2], n = s->ne[3];
    ggml_tensor * y = linear(C, w, b, ggml_reshape_2d(C, s, s->ne[0], ow*oh*n));
    return ggml_reshape_4d(C, y, w->ne[1], ow, oh, n);
}

// ConvNeXt block on a channels-last [C, W, H, N] map. The depthwise 7x7 runs
// on a permuted view: ggml's channels-last depthwise kernel reads the taps
// channel-fastest, which is how the converter stores them.
ggml_tensor * convnext_block(ggml_context * C, ggml_tensor * x, const BlockW & b, float eps) {
    const int64_t ch = x->ne[0];
    ggml_tensor * k = ggml_permute(C, ggml_reshape_4d(C, b.dw_w, ch, 1, 7, 7), 3, 2, 0, 1);
    ggml_tensor * y = ggml_conv_2d_dw_direct(C, k, ggml_permute(C, x, 2, 0, 1, 3), 1, 1, 3, 3, 1, 1);
    y = ggml_add(C, ggml_permute(C, y, 1, 2, 0, 3), b.dw_b);
    y = layer_norm(C, y, b.ln_w, b.ln_b, eps);
    y = ffn_gelu_erf(C, b.fc1_w, b.fc1_b, b.fc2_w, b.fc2_b, y);
    return ggml_add(C, x, y);
}

// Columns [first, first+n) of a [D, T] activation.
ggml_tensor * cols(ggml_context * C, ggml_tensor * x, int64_t first, int64_t n) {
    return ggml_view_2d(C, x, x->ne[0], n, x->nb[1], (size_t) first*x->nb[1]);
}

// Head-split part `part` (0 q, 1 k, 2 v) of a fused [q | k | v] projection.
ggml_tensor * qkv_part(ggml_context * C, ggml_tensor * qkv, int64_t hd, int64_t nq, int64_t nkv, int part) {
    const size_t  es    = ggml_element_size(qkv);
    const int64_t heads = part == 0 ? nq : nkv;
    const size_t  off   = part == 0 ? 0 : (size_t) (nq + (part - 1)*nkv)*hd*es;
    return ggml_view_3d(C, qkv, hd, heads, qkv->ne[1], (size_t) hd*es, qkv->nb[1], off);
}

// q/k/v arrive as [hd, heads, T]. GQA by ggml_mul_mat broadcast, which pairs
// query head h with K/V head h / (nq/nkv), as the reference's expand+reshape
// does (VLTrBase.attention_forward).
ggml_tensor * attend(ggml_context * C, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                     int64_t hd, int64_t nq) {
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, q, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, k, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, v, 1, 2, 0, 3));
    return attention(C, Q, K, V, mask, 1.0f / std::sqrt((float) hd), hd*nq, q->ne[2]);
}

// policy_utils.apply_rope: the half-split (NeoX) rotation, base 10000.
ggml_tensor * rope_neox(ggml_context * C, ggml_tensor * x, ggml_tensor * pos, int64_t hd, float base) {
    return ggml_rope_ext(C, x, pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

ggml_tensor * swiglu(ggml_context * C, const LlamaLayerW & l, ggml_tensor * x) {
    return ffn_swiglu(C, l.gate, nullptr, l.up, nullptr, l.down, nullptr, x);
}

// RMSNorm without affine, scaled by (1 + Linear(silu(temb))): AdaRMSNorm.
ggml_tensor * ada_rms(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, ggml_tensor * st,
                      float eps) {
    return ggml_mul(C, ggml_rms_norm(C, x, eps), ggml_scale_bias(C, linear(C, w, b, st), 1.0f, 1.0f));
}

}  // namespace

ggml_cgraph * PicoVlaModelArch::build(ggml_context * C, IO & io, int64_t L) const {
    const int64_t P = prefix_len(L), R = n_record, T = P + 2*R, hd = head_dim;
    const int64_t NI = img_tokens();

    // Positions run on past the backbone into the expert's action rows.
    io.pos = ggml_new_tensor_1d(C, GGML_TYPE_I32, P + R + std::max(R, chunk));
    ggml_set_input(io.pos);
    auto pos = [&](int64_t first, int64_t n) {
        return ggml_view_1d(C, io.pos, n, (size_t) first*ggml_element_size(io.pos));
    };

    // Language: Llama layer 0 over the instruction alone, bidirectional. Its
    // output (no final norm) is the language part of the prefix; the masked
    // mean of its normed output is the task embedding that gates the merge.
    io.ids = ggml_new_tensor_1d(C, GGML_TYPE_I32, L);
    ggml_set_input(io.ids);
    ggml_tensor * lang = ggml_get_rows(C, tok_embd, io.ids);
    {
        const LlamaLayerW & l = vlm[0];
        ggml_tensor * qkv = linear(C, l.qkv, nullptr, rms_norm(C, lang, l.attn_norm, rms_eps));
        ggml_tensor * q = rope_neox(C, qkv_part(C, qkv, hd, n_heads, n_kv, 0), pos(0, L), hd, rope_base);
        ggml_tensor * k = rope_neox(C, qkv_part(C, qkv, hd, n_heads, n_kv, 1), pos(0, L), hd, rope_base);
        ggml_tensor * att = attend(C, q, k, qkv_part(C, qkv, hd, n_heads, n_kv, 2), nullptr, hd, n_heads);
        lang = ggml_add(C, lang, linear(C, l.o, nullptr, ggml_reshape_2d(C, att, hidden, L)));
        lang = ggml_add(C, lang, swiglu(C, l, rms_norm(C, lang, l.ffn_norm, rms_eps)));
    }
    ggml_tensor * task = rms_norm(C, lang, lm_norm, rms_eps);
    task = ggml_reshape_1d(C, ggml_mean(C, ggml_cont(C, ggml_transpose(C, task))), hidden);

    // DINOv3 ConvNeXt-T, channels-last [C, W, H, views]. Pixels arrive
    // normalized and interleaved, which already is that layout.
    io.pixels = ggml_new_tensor_4d(C, GGML_TYPE_F32, 3, image_size, image_size, n_views);
    ggml_set_input(io.pixels);
    ggml_tensor * x = stride_conv(C, io.pixels, stem_w, stem_b, stem);
    x = layer_norm(C, x, stem_norm_w, stem_norm_b, vis_eps);
    for (size_t s = 0; s < stages.size(); ++s) {
        const StageW & st = stages[s];
        if (s > 0)
            x = stride_conv(C, layer_norm(C, x, st.norm_w, st.norm_b, vis_eps), st.down_w, st.down_b, 2);
        for (const BlockW & b : st.blocks)
            x = convnext_block(C, x, b, vis_eps);
    }
    x = layer_norm(C, x, vis_norm_w, vis_norm_b, vis_eps);

    // TaskAwareDINOv3Connector: the four tokens of each 2x2 block are weighted
    // by a softmax against the task query (times 4, to keep the scale), then
    // concatenated and projected. space_to_depth orders them dx-fastest, as
    // the reference's reshape+permute does.
    const int64_t vd = x->ne[0];
    ggml_tensor * merged = ggml_reshape_3d(C, space_to_depth(C, x, 2), vd, 4, NI*n_views);
    ggml_tensor * keys   = linear(C, vision_proj, nullptr, ggml_reshape_2d(C, merged, vd, 4*NI*n_views));
    ggml_tensor * query  = linear(C, task_proj, nullptr, ggml_reshape_2d(C, task, hidden, 1));
    ggml_tensor * scores = ggml_reshape_2d(C, ggml_mul_mat(C, keys, query), 4, NI*n_views);
    ggml_tensor * gates  = ggml_scale(C, ggml_soft_max_ext(C, scores, nullptr, 1.0f / std::sqrt((float) query->ne[0]),
                                                           0.0f), 4.0f);
    merged = ggml_mul(C, merged, ggml_reshape_3d(C, gates, 1, 4, NI*n_views));
    ggml_tensor * img = linear(C, conn_proj, nullptr, ggml_reshape_2d(C, merged, 4*vd, NI*n_views));

    // Backbone: [state | images | language | records | past], positions 0..T-1.
    io.state = ggml_new_tensor_1d(C, GGML_TYPE_F32, max_state);
    io.past  = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, R);
    io.mask  = ggml_new_tensor_2d(C, GGML_TYPE_F32, T, T);
    ggml_set_input(io.state);
    ggml_set_input(io.past);
    ggml_set_input(io.mask);
    ggml_tensor * h = linear(C, state_w, state_b, ggml_reshape_2d(C, io.state, max_state, 1));
    h = ggml_concat(C, ggml_concat(C, h, img, 1), lang, 1);
    h = ggml_concat(C, ggml_concat(C, h, record_tokens, 1), io.past, 1);

    // Per joint layer, the prefix part of K (before RoPE) and V through a
    // non-affine LayerNorm across all KV heads: the expert's cache.
    std::vector<ggml_tensor *> kc, vc;
    for (int64_t i = 1; i <= n_layers; ++i) {
        const LlamaLayerW & l = vlm[(size_t) i];
        ggml_tensor * qkv = linear(C, l.qkv, nullptr, rms_norm(C, h, l.attn_norm, rms_eps));
        const size_t  es  = ggml_element_size(qkv);
        kc.push_back(ggml_norm(C, ggml_cont(C, ggml_view_2d(C, qkv, n_kv*hd, P, qkv->nb[1], (size_t) n_heads*hd*es)),
                               kLnEps));
        vc.push_back(ggml_norm(C, ggml_cont(C, ggml_view_2d(C, qkv, n_kv*hd, P, qkv->nb[1],
                                                            (size_t) (n_heads + n_kv)*hd*es)), kLnEps));
        ggml_tensor * q = rope_neox(C, qkv_part(C, qkv, hd, n_heads, n_kv, 0), pos(0, T), hd, rope_base);
        ggml_tensor * k = rope_neox(C, qkv_part(C, qkv, hd, n_heads, n_kv, 1), pos(0, T), hd, rope_base);
        ggml_tensor * att = attend(C, q, k, qkv_part(C, qkv, hd, n_heads, n_kv, 2), io.mask, hd, n_heads);
        h = ggml_add(C, h, linear(C, l.o, nullptr, ggml_reshape_2d(C, att, hidden, T)));
        h = ggml_add(C, h, swiglu(C, l, rms_norm(C, h, l.ffn_norm, rms_eps)));
    }
    // No final RMSNorm: records and past slots leave through out_norm.
    io.record = ggml_norm(C, ggml_cont(C, cols(C, h, P, R)), kLnEps);
    ggml_set_output(io.record);
    ggml_tensor * cond = ggml_norm(C, ggml_cont(C, cols(C, h, P + R, R)), kLnEps);

    // Expert, cond rows. They see the prefix and each other but never the
    // action rows, and carry no time, so they run once per call: what each
    // layer leaves is the RoPE'd [prefix | cond] K/V the action rows attend.
    std::vector<ggml_tensor *> ctx_k, ctx_v;
    ggml_tensor * c = linear(C, cond_w, cond_b, cond);
    for (int64_t i = 0; i < n_layers; ++i) {
        const ExpertLayerW & e = expert[(size_t) i];
        ggml_tensor * pk = ggml_reshape_3d(C, ggml_mul_mat(C, e.cross_k, kc[(size_t) i]), hd, n_kv, P);
        ggml_tensor * pv = ggml_reshape_3d(C, ggml_mul_mat(C, e.cross_v, vc[(size_t) i]), hd, n_kv, P);
        ggml_tensor * qkv = linear(C, e.l.qkv, nullptr, rms_norm(C, c, e.l.attn_norm, rms_eps));
        ggml_tensor * q = rope_neox(C, qkv_part(C, qkv, hd, n_heads, n_kv, 0), pos(P, R), hd, rope_base);
        ggml_tensor * k = rope_neox(C, qkv_part(C, qkv, hd, n_heads, n_kv, 1), pos(P, R), hd, rope_base);
        ggml_tensor * v = ggml_cont(C, qkv_part(C, qkv, hd, n_heads, n_kv, 2));
        ctx_k.push_back(ggml_concat(C, rope_neox(C, pk, pos(0, P), hd, rope_base), k, 2));
        ctx_v.push_back(ggml_concat(C, pv, v, 2));
        ggml_tensor * att = attend(C, q, ctx_k.back(), ctx_v.back(), nullptr, hd, n_heads);
        c = ggml_add(C, c, linear(C, e.l.o, nullptr, ggml_reshape_2d(C, att, n_heads*hd, R)));
        c = ggml_add(C, c, swiglu(C, e.l, rms_norm(C, c, e.l.ffn_norm, rms_eps)));
    }

    // MeanFlow: x <- x - (t - r) * u(x, r, t), every step unrolled. io.time
    // holds each step's [sincos(r) | sincos(t)], io.dt its t - r.
    const int64_t A = chunk, tdim = time_in_w->ne[0];
    io.noise = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_action, A);
    io.time  = ggml_new_tensor_2d(C, GGML_TYPE_F32, tdim, num_steps);
    io.dt    = ggml_new_tensor_1d(C, GGML_TYPE_F32, num_steps);
    ggml_set_input(io.noise);
    ggml_set_input(io.time);
    ggml_set_input(io.dt);
    ggml_tensor * xt = io.noise;
    for (int s = 0; s < num_steps; ++s) {
        ggml_tensor * tin = ggml_view_2d(C, io.time, tdim, 1, io.time->nb[1], (size_t) s*io.time->nb[1]);
        ggml_tensor * dt  = ggml_view_1d(C, io.dt, 1, (size_t) s*sizeof(float));
        ggml_tensor * temb = linear(C, time_out_w, time_out_b, ggml_silu(C, linear(C, time_in_w, time_in_b, tin)));
        ggml_tensor * st   = ggml_silu(C, temb);

        ggml_tensor * a = linear(C, act_in_w, act_in_b, xt);
        for (int64_t i = 0; i < n_layers; ++i) {
            const ExpertLayerW & e = expert[(size_t) i];
            ggml_tensor * qkv = linear(C, e.l.qkv, nullptr, ada_rms(C, a, e.ada_attn_w, e.ada_attn_b, st, rms_eps));
            ggml_tensor * q = rope_neox(C, qkv_part(C, qkv, hd, n_heads, n_kv, 0), pos(P + R, A), hd, rope_base);
            ggml_tensor * k = rope_neox(C, qkv_part(C, qkv, hd, n_heads, n_kv, 1), pos(P + R, A), hd, rope_base);
            ggml_tensor * v = ggml_cont(C, qkv_part(C, qkv, hd, n_heads, n_kv, 2));
            ggml_tensor * att = attend(C, q, ggml_concat(C, ctx_k[(size_t) i], k, 2),
                                       ggml_concat(C, ctx_v[(size_t) i], v, 2), nullptr, hd, n_heads);
            a = ggml_add(C, a, linear(C, e.l.o, nullptr, ggml_reshape_2d(C, att, n_heads*hd, A)));
            a = ggml_add(C, a, swiglu(C, e.l, ada_rms(C, a, e.ada_ffn_w, e.ada_ffn_b, st, rms_eps)));
        }
        ggml_tensor * u = ada_rms(C, a, out_ada_w, out_ada_b, st, rms_eps);
        u = linear(C, act_out2_w, act_out2_b, ggml_silu(C, linear(C, act_out0_w, act_out0_b, u)));
        xt = ggml_sub(C, xt, ggml_mul(C, u, dt));
    }
    io.actions = xt;
    ggml_set_output(io.actions);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_build_forward_expand(gf, io.record);
    ggml_build_forward_expand(gf, io.actions);
    return gf;
}

namespace {

bool read_i64_array(const gguf_reader & g, const char * key, std::vector<int64_t> & out) {
    const int64_t id = gguf_find_key(g.gctx, key);
    if (id < 0 || gguf_get_kv_type(g.gctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g.gctx, id) != GGUF_TYPE_INT32)
        return false;
    const int32_t * d = (const int32_t *) gguf_get_arr_data(g.gctx, id);
    out.assign(d, d + gguf_get_arr_n(g.gctx, id));
    return true;
}

bool load_config(const gguf_reader & g, PicoVlaModelArch & m) {
    auto U = [&](const char * k, int64_t & dst) {
        char key[96];
        std::snprintf(key, sizeof(key), "picovla.%s", k);
        if (g.has(key))
            dst = (int64_t) g.u32(key);
    };
    auto F = [&](const char * k, float & dst) {
        char key[96];
        std::snprintf(key, sizeof(key), "picovla.%s", k);
        if (g.has(key))
            dst = g.f32(key);
    };
    int64_t steps = m.num_steps;
    U("hidden", m.hidden);                 U("intermediate", m.inter);       U("n_layers", m.n_layers);
    U("n_heads", m.n_heads);               U("n_kv_heads", m.n_kv);          U("head_dim", m.head_dim);
    U("expert_hidden", m.ex_hidden);       U("expert_intermediate", m.ex_inter);
    U("n_record", m.n_record);             U("chunk_size", m.chunk);         U("num_steps", steps);
    U("max_state_dim", m.max_state);       U("max_action_dim", m.max_action);
    U("real_state_dim", m.real_state);     U("real_action_dim", m.real_action);
    U("num_views", m.n_views);             U("image_size", m.image_size);    U("stem_patch", m.stem);
    U("tokenizer_max_length", m.max_lang); U("vocab_size", m.vocab);
    F("rms_eps", m.rms_eps);               F("rope_base", m.rope_base);      F("vis_eps", m.vis_eps);
    F("transport_low", m.t_lo);            F("transport_high", m.t_hi);
    F("min_period", m.min_period);         F("max_period", m.max_period);    F("norm_eps", m.norm_eps);
    m.num_steps = (int) steps;

    std::vector<int64_t> map;
    if (!read_i64_array(g, "picovla.vis_depths", m.vis_depths) || !read_i64_array(g, "picovla.vis_dims", m.vis_dims) ||
        !read_i64_array(g, "picovla.token_map", map)) {
        std::fprintf(stderr, "vla(picovla): GGUF lacks vis_depths / vis_dims / token_map\n");
        return false;
    }
    m.token_map.assign(map.begin(), map.end());

    bool ok = m.vis_depths.size() == m.vis_dims.size() && !m.vis_dims.empty() && m.stem == 4 &&
              (int64_t) m.token_map.size() == m.vocab && m.image_size % (m.stem << (m.vis_dims.size() - 1)) == 0 &&
              m.grid() % 2 == 0 && m.n_heads % m.n_kv == 0 && m.num_steps >= 1 && m.t_lo < m.t_hi &&
              m.real_state <= m.max_state && m.real_action <= m.max_action;
    for (int64_t v : { m.hidden, m.inter, m.n_layers, m.n_heads, m.n_kv, m.head_dim, m.ex_hidden, m.ex_inter,
                       m.n_record, m.chunk, m.real_state, m.real_action, m.n_views, m.max_lang })
        ok = ok && v >= 1;
    if (!ok) {
        std::fprintf(stderr, "vla(picovla): inconsistent dimensions in GGUF metadata\n");
        return false;
    }
    return true;
}

LlamaLayerW load_llama(WeightLoader & L, const char * pfx, int64_t i) {
    char n[4][96];
    auto N = [&](int slot, const char * s) {
        std::snprintf(n[slot], sizeof(n[slot]), "%s.blk.%lld.%s", pfx, (long long) i, s);
        return std::string(n[slot]);
    };
    LlamaLayerW w;
    w.attn_norm = L.f32("%s", N(0, "attn_norm.weight").c_str());
    w.qkv = L.fuse_gemm(N(0, "attn_qkv.weight").c_str(),
                        { N(1, "attn_q.weight"), N(2, "attn_k.weight"), N(3, "attn_v.weight") });
    w.o        = L.gemm("%s", N(0, "attn_o.weight").c_str());
    w.ffn_norm = L.f32("%s", N(0, "ffn_norm.weight").c_str());
    w.gate     = L.gemm("%s", N(0, "ffn_gate.weight").c_str());
    w.up       = L.gemm("%s", N(0, "ffn_up.weight").c_str());
    w.down     = L.gemm("%s", N(0, "ffn_down.weight").c_str());
    return w;
}

bool load_weights(PicoVlaModelArch & m, gguf_reader & g) {
    ggml_init_params wp = { (size_t) 16*1024*1024, nullptr, true };
    m.ctx_weights = ggml_init(wp);
    if (!m.ctx_weights)
        return false;
    WeightLoader L("picovla", g, m.ctx_weights, m.mt);

    m.tok_embd      = L.f32("token_embd.weight");
    m.lm_norm       = L.f32("vlm.output_norm.weight");
    m.record_tokens = L.f32("record_tokens");
    for (int64_t i = 0; i <= m.n_layers; ++i)
        m.vlm.push_back(load_llama(L, "vlm", i));

    m.state_w    = L.gemm("state_proj.weight");
    m.state_b    = L.f32("state_proj.bias");
    m.act_in_w   = L.gemm("action_in_proj.weight");
    m.act_in_b   = L.f32("action_in_proj.bias");
    m.act_out0_w = L.gemm("action_out_proj.0.weight");
    m.act_out0_b = L.f32("action_out_proj.0.bias");
    m.act_out2_w = L.gemm("action_out_proj.2.weight");
    m.act_out2_b = L.f32("action_out_proj.2.bias");
    m.time_in_w  = L.gemm("time_mlp_in.weight");
    m.time_in_b  = L.f32("time_mlp_in.bias");
    m.time_out_w = L.gemm("time_mlp_out.weight");
    m.time_out_b = L.f32("time_mlp_out.bias");

    m.cond_w    = L.gemm("aex.cond_proj.weight");
    m.cond_b    = L.f32("aex.cond_proj.bias");
    m.out_ada_w = L.gemm("aex.output_ada.weight");
    m.out_ada_b = L.f32("aex.output_ada.bias");
    for (int64_t i = 0; i < m.n_layers; ++i) {
        ExpertLayerW e;
        e.l          = load_llama(L, "aex", i);
        e.cross_k    = L.gemm("aex.blk.%lld.cross_k.weight", (long long) i);
        e.cross_v    = L.gemm("aex.blk.%lld.cross_v.weight", (long long) i);
        e.ada_attn_w = L.gemm("aex.blk.%lld.ada_attn.weight", (long long) i);
        e.ada_attn_b = L.f32("aex.blk.%lld.ada_attn.bias", (long long) i);
        e.ada_ffn_w  = L.gemm("aex.blk.%lld.ada_ffn.weight", (long long) i);
        e.ada_ffn_b  = L.f32("aex.blk.%lld.ada_ffn.bias", (long long) i);
        m.expert.push_back(e);
    }

    // The depthwise taps and the norms stay F32: ggml's depthwise conv reads
    // F32 or F16 taps only, and they are a rounding error of the tower's size.
    m.stem_w      = L.gemm("vis.stem.weight");
    m.stem_b      = L.f32("vis.stem.bias");
    m.stem_norm_w = L.f32("vis.stem_norm.weight");
    m.stem_norm_b = L.f32("vis.stem_norm.bias");
    m.stages.resize(m.vis_depths.size());
    for (size_t s = 0; s < m.stages.size(); ++s) {
        StageW & st = m.stages[s];
        const long long S = (long long) s;
        if (s > 0) {
            st.norm_w = L.f32("vis.down.%lld.norm.weight", S);
            st.norm_b = L.f32("vis.down.%lld.norm.bias", S);
            st.down_w = L.gemm("vis.down.%lld.weight", S);
            st.down_b = L.f32("vis.down.%lld.bias", S);
        }
        for (int64_t i = 0; i < m.vis_depths[s]; ++i) {
            const long long I = (long long) i;
            BlockW b;
            b.dw_w  = L.f32("vis.blk.%lld.%lld.dw.weight", S, I);
            b.dw_b  = L.f32("vis.blk.%lld.%lld.dw.bias", S, I);
            b.ln_w  = L.f32("vis.blk.%lld.%lld.ln.weight", S, I);
            b.ln_b  = L.f32("vis.blk.%lld.%lld.ln.bias", S, I);
            b.fc1_w = L.gemm("vis.blk.%lld.%lld.fc1.weight", S, I);
            b.fc1_b = L.f32("vis.blk.%lld.%lld.fc1.bias", S, I);
            b.fc2_w = L.gemm("vis.blk.%lld.%lld.fc2.weight", S, I);
            b.fc2_b = L.f32("vis.blk.%lld.%lld.fc2.bias", S, I);
            st.blocks.push_back(b);
        }
    }
    m.vis_norm_w  = L.f32("vis.norm.weight");
    m.vis_norm_b  = L.f32("vis.norm.bias");
    m.task_proj   = L.gemm("conn.task_proj.weight");
    m.vision_proj = L.gemm("conn.vision_proj.weight");
    m.conn_proj   = L.gemm("conn.proj.weight");

    if (!L.upload(m.backend, &m.weight_buf))
        return false;

    const int64_t vd = m.vis_dims.back();
    bool ok = m.tok_embd->ne[0] == m.hidden && m.record_tokens->ne[0] == m.hidden &&
              m.record_tokens->ne[1] == m.n_record && m.state_w->ne[0] == m.max_state &&
              m.act_in_w->ne[0] == m.max_action && m.act_out2_w->ne[1] == m.max_action &&
              m.vlm[0].qkv->ne[1] == (m.n_heads + 2*m.n_kv)*m.head_dim && m.time_in_w->ne[0] % 4 == 0 &&
              m.stem_w->ne[0] == 3*m.stem*m.stem && m.stem_w->ne[1] == m.vis_dims[0] &&
              m.conn_proj->ne[0] == 4*vd && m.conn_proj->ne[1] == m.hidden && m.vision_proj->ne[0] == vd &&
              m.task_proj->ne[0] == m.hidden && m.task_proj->ne[1] == m.vision_proj->ne[1] &&
              m.expert[0].cross_k->ne[0] == m.n_kv*m.head_dim && m.cond_w->ne[1] == m.ex_hidden;
    for (size_t s = 0; s < m.stages.size(); ++s)
        for (const BlockW & b : m.stages[s].blocks)
            ok = ok && b.dw_w->ne[0] == m.vis_dims[s] && b.dw_w->ne[1] == 7 && b.dw_w->ne[2] == 7;
    const int64_t n_rows = m.tok_embd->ne[1];
    for (int32_t r : m.token_map)
        ok = ok && r >= 0 && r < n_rows;
    if (!ok) {
        std::fprintf(stderr, "vla(picovla): tensor shapes disagree with GGUF metadata\n");
        return false;
    }

    auto stats = [&](const char * name, int64_t n, std::vector<float> & dst) {
        dst = g.read_f32(name);
        return (int64_t) dst.size() == n;
    };
    if (!stats("state_mean", m.real_state, m.state_mean) || !stats("state_std", m.real_state, m.state_std) ||
        !stats("action_mean", m.real_action, m.action_mean) || !stats("action_std", m.real_action, m.action_std)) {
        std::fprintf(stderr, "vla(picovla): GGUF normalizer stats are missing or the wrong size\n");
        return false;
    }
    return true;
}

}  // namespace

std::unique_ptr<ModelArchBase> picovla_create(const std::string& mmproj_path,
                                              const std::string& ckpt_path,
                                              const std::string&,
                                              const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(picovla): note - mmproj '%s' is ignored (vision is baked into the GGUF)\n",
                    mmproj_path.c_str());

    auto m = std::make_unique<PicoVlaModelArch>();
    m->mt = opts.weight_dtype.value_or(GGML_TYPE_F32);

    gguf_reader g("picovla");
    if (!g.open(ckpt_path))
        return nullptr;
    if (!g.has("picovla.architecture")) {
        std::fprintf(stderr, "vla(picovla): %s is not a picovla GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, *m) || !resolve_num_steps("picovla", opts, m->num_steps))
        return nullptr;

    const Backend b = backend_init("vla(picovla)", m->n_threads);
    if (!b.handle)
        return nullptr;
    m->backend = b.handle;

    if (!load_weights(*m, g))
        return nullptr;
    m->past.assign((size_t) (m->hidden*m->n_record), 0.0f);

    m->cfg.n_img           = m->n_views * m->img_tokens();
    m->cfg.n_lang          = m->max_lang;
    m->cfg.n_state         = 1;
    m->cfg.n_suffix        = m->chunk;
    m->cfg.hidden          = m->hidden;
    m->cfg.expert_h        = m->ex_hidden;
    m->cfg.intermediate    = m->inter;
    m->cfg.expert_inter    = m->ex_inter;
    m->cfg.n_q_heads       = m->n_heads;
    m->cfg.n_kv_heads      = m->n_kv;
    m->cfg.head_dim        = m->head_dim;
    m->cfg.n_layers        = m->n_layers;
    m->cfg.max_state_dim   = m->max_state;
    m->cfg.max_action_dim  = m->max_action;
    m->cfg.real_state_dim  = m->real_state;
    m->cfg.real_action_dim = m->real_action;
    m->cfg.norm_eps        = m->norm_eps;
    m->cfg.rms_eps         = m->rms_eps;
    m->cfg.min_period      = m->min_period;
    m->cfg.max_period      = m->max_period;
    m->cfg.num_steps       = m->num_steps;
    m->cfg.rope_n_dims     = (int) m->head_dim;
    m->cfg.rope_mode       = GGML_ROPE_TYPE_NEOX;
    m->cfg.rope_freq_base  = m->rope_base;

    std::printf("vla(picovla): weights resident %.2f GiB (%s) - ConvNeXt x%lld views + Llama 1+%lld layers + "
                "expert %lld, chunk %lld, %d MeanFlow steps, %zu-token vocabulary\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0), dtype_name(m->mt),
                (long long) m->n_views, (long long) m->n_layers, (long long) m->ex_hidden, (long long) m->chunk,
                m->num_steps, (size_t) m->tok_embd->ne[1]);
    return m;
}

std::vector<float> PicoVlaModelArch::predict(const Inputs& in) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    stats = Stats{};

    if (in.precomputed_img_emb) {
        std::fprintf(stderr, "vla(picovla): precomputed_img_emb is not supported; pass raw images\n");
        return {};
    }
    if (!in.images || in.n_images != n_views) {
        std::fprintf(stderr, "vla(picovla): expected %lld views (agentview, wrist), got %d\n",
                     (long long) n_views, in.n_images);
        return {};
    }
    for (int64_t i = 0; i < n_views; ++i)
        if (!view_ok("picovla", in.images[i], image_size))
            return {};
    if (!in.lang_tokens || in.n_lang < 1 || in.n_lang > max_lang) {
        std::fprintf(stderr, "vla(picovla): %d language tokens; need 1..%lld\n", in.n_lang, (long long) max_lang);
        return {};
    }
    // Ids the training data never used share the padding row, as after
    // CompactTaskEmbedding.compact().
    const int64_t L = in.n_lang;
    std::vector<int32_t> ids((size_t) L);
    for (int64_t i = 0; i < L; ++i) {
        const int32_t t = in.lang_tokens[i];
        if (t < 0 || t >= vocab) {
            std::fprintf(stderr, "vla(picovla): token %d out of vocab\n", t);
            return {};
        }
        ids[(size_t) i] = token_map[(size_t) t];
    }

    const size_t arena = ggml_tensor_overhead()*8192 + ggml_graph_overhead_custom(8192, false);
    if (!graph.ensure(backend, Key{L}, arena, [&](ggml_context * C, IO & io) { return build(C, io, L); })) {
        std::fprintf(stderr, "vla(picovla): graph build/alloc failed\n");
        return {};
    }
    IO & io = graph.io();

    // Pixels: ImageNet-normalized, kept interleaved (channels-last).
    const int64_t side = image_size, npx = side*side;
    std::vector<float> pixels((size_t) (3*npx*n_views));
    for (int64_t v = 0; v < n_views; ++v) {
        const ImageView & iv = in.images[v];
        float * dst = pixels.data() + (size_t) (v*3*npx);
        for (int64_t p = 0; p < 3*npx; ++p) {
            const int   ch = (int) (p % 3);
            const float px = iv.format == PixelFormat::U8 ? ((const uint8_t *) iv.data)[p] / 255.0f
                                                          : ((const float *) iv.data)[p];
            dst[p] = (px - kImagenetMean[ch]) / kImagenetStd[ch];
        }
    }

    // LeRobot MEAN_STD, then zero padding to max_state_dim.
    std::vector<float> state((size_t) max_state, 0.0f);
    if (in.state)
        for (int64_t i = 0; i < real_state; ++i)
            state[(size_t) i] = (in.state[i] - state_mean[(size_t) i]) / (state_std[(size_t) i] + norm_eps);

    const int64_t P = prefix_len(L), R = n_record, T = P + 2*R;
    std::vector<int32_t> pos((size_t) io.pos->ne[0]);
    for (size_t i = 0; i < pos.size(); ++i)
        pos[i] = (int32_t) i;
    // Prefix rows see the prefix, record rows the prefix and the records, past
    // rows everything: record tokens never read the past, so memory does not
    // compound across calls.
    std::vector<float> mask((size_t) (T*T), 0.0f);
    for (int64_t q = 0; q < T; ++q) {
        const int64_t visible = q < P ? P : q < P + R ? P + R : T;
        for (int64_t k = visible; k < T; ++k)
            mask[(size_t) (q*T + k)] = -INFINITY;
    }

    std::vector<float> noise((size_t) (chunk*max_action));
    if (in.noise) {
        std::copy(in.noise, in.noise + noise.size(), noise.begin());
    } else {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float & v : noise)
            v = dist(rng);
    }

    // Step s integrates from t = 1 - periods[s] down to r = 1 - periods[s+1],
    // periods clamped to the transport range (VLAMeanFlow.sample_actions). The
    // reference builds t and r as float32 tensors, so they round to float here.
    const int64_t tdim = time_in_w->ne[0], half = tdim / 2;
    std::vector<float> time((size_t) (tdim*num_steps)), dt((size_t) num_steps);
    auto period = [&](int s) { return std::min(std::max((float) s / (float) num_steps, t_lo), t_hi); };
    for (int s = 0; s < num_steps; ++s) {
        const float t = 1.0f - period(s), r = 1.0f - period(s + 1);
        const std::vector<float> er = sinusoidal_time_emb(r, half, min_period, max_period);
        const std::vector<float> et = sinusoidal_time_emb(t, half, min_period, max_period);
        float * row = time.data() + (size_t) (s*tdim);
        std::copy(er.begin(), er.end(), row);
        std::copy(et.begin(), et.end(), row + half);
        dt[(size_t) s] = t - r;
    }

    if (in.reset_memory)
        std::fill(past.begin(), past.end(), 0.0f);

    ggml_backend_tensor_set(io.pixels, pixels.data(), 0, ggml_nbytes(io.pixels));
    ggml_backend_tensor_set(io.ids,    ids.data(),    0, ggml_nbytes(io.ids));
    ggml_backend_tensor_set(io.pos,    pos.data(),    0, ggml_nbytes(io.pos));
    ggml_backend_tensor_set(io.mask,   mask.data(),   0, ggml_nbytes(io.mask));
    ggml_backend_tensor_set(io.state,  state.data(),  0, ggml_nbytes(io.state));
    ggml_backend_tensor_set(io.past,   past.data(),   0, ggml_nbytes(io.past));
    ggml_backend_tensor_set(io.noise,  noise.data(),  0, ggml_nbytes(io.noise));
    ggml_backend_tensor_set(io.time,   time.data(),   0, ggml_nbytes(io.time));
    ggml_backend_tensor_set(io.dt,     dt.data(),     0, ggml_nbytes(io.dt));

    const auto tc = clock::now();
    graph_unique_names(graph.graph());
    if (ggml_backend_graph_compute(backend, graph.graph()) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(picovla): compute failed\n");
        return {};
    }
    std::vector<float> out((size_t) (chunk*max_action));
    ggml_backend_tensor_get(io.actions, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_get(io.record,  past.data(), 0, past.size()*sizeof(float));

    for (int64_t a = 0; a < chunk; ++a) {
        float * row = out.data() + (size_t) (a*max_action);
        for (int64_t j = 0; j < max_action; ++j)
            row[j] = j < real_action ? row[j]*action_std[(size_t) j] + action_mean[(size_t) j] : 0.0f;
    }

    stats.ms_inference = std::chrono::duration<float, std::milli>(clock::now() - tc).count();
    stats.ms_total     = std::chrono::duration<float, std::milli>(clock::now() - t0).count();
    return out;
}

}  // namespace vla
