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

// TurboVLA (H-EmbodVis/TurboVLA@b29ab142): DINOv3 ViT-B/16 over each camera,
// BERT-base over the instruction, six Grounding-DINO bi-attention fusion layers
// each followed by a text enhancer, and a three-layer ACT decoder whose twelve
// learned queries read the fused tokens plus two state tokens. One forward pass,
// no denoising loop, so the whole model is one cached graph.

#include "arch.h"
#include "backend.h"
#include "gguf.h"
#include "gguf_reader.h"
#include "loader.h"
#include "model.h"
#include "options.h"
#include "scratch_ctx.h"
#include "layers/attn.h"
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
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace vla {
namespace {

constexpr float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImagenetStd[3]  = {0.229f, 0.224f, 0.225f};

// Epsilons are the module defaults upstream: nn.LayerNorm everywhere except
// BERT, whose config sets 1e-12.
constexpr float kLnEps   = 1e-5f;
constexpr float kBertEps = 1e-12f;

struct VitLayerW {
    ggml_tensor *ln1_w, *ln1_b, *qkv_w, *qkv_b, *o_w, *o_b;
    ggml_tensor *ln2_w, *ln2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

struct BertLayerW {
    ggml_tensor *qkv_w, *qkv_b, *o_w, *o_b, *ln1_w, *ln1_b;
    ggml_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b, *ln2_w, *ln2_b;
};

// v_proj/values_v and l_proj/values_l are fused per side: v_proj(v) is the
// vision query and also the key the language side attends, l_proj(l) the
// other way round, so each side needs one GEMM for both of its projections.
struct FusionLayerW {
    ggml_tensor *norm_v_w, *norm_v_b, *norm_l_w, *norm_l_b;
    ggml_tensor *v_w, *v_b, *l_w, *l_b;
    ggml_tensor *out_v_w, *out_v_b, *out_l_w, *out_l_b;
    ggml_tensor *gamma_v, *gamma_l;
};

struct EnhancerLayerW {
    ggml_tensor *qkv_w, *qkv_b, *o_w, *o_b, *ln1_w, *ln1_b;
    ggml_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b, *ln2_w, *ln2_b;
};

struct DecoderLayerW {
    ggml_tensor *ln1_w, *ln1_b, *self_qkv_w, *self_qkv_b, *self_o_w, *self_o_b;
    ggml_tensor *ln2_w, *ln2_b, *cross_qkv_w, *cross_qkv_b, *cross_o_w, *cross_o_b;
    ggml_tensor *ln3_w, *ln3_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

// A training instruction's BERT token ids and the length its batch was padded
// to. See text_len() for why the length is part of the model.
struct TextGroup {
    std::vector<int32_t> ids;
    int64_t              pad_to = 0;
};

}  // namespace

struct TurboVlaModelArch : public ModelArchBase {
    TurboVlaModelArch() : ModelArchBase(Arch::TURBOVLA) {}
    ~TurboVlaModelArch() override {
        graph.release();
        if (const_buf)   ggml_backend_buffer_free(const_buf);
        if (ctx_const)   ggml_free(ctx_const);
        if (weight_buf)  ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend)     ggml_backend_free(backend);
    }

    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_const   = nullptr;
    ggml_backend_buffer_t const_buf   = nullptr;
    ggml_type             mt          = GGML_TYPE_F32;

    int64_t hidden = 256, n_views = 2, image_size = 256, patch = 16, n_reg = 4;
    int64_t vit_dim = 768, vit_layers = 12, vit_heads = 12;
    int64_t bert_dim = 768, bert_layers = 12, bert_heads = 12, vocab = 30522, bert_max_pos = 512;
    int64_t fusion_layers = 6, fusion_heads = 4, fusion_dim = 1024, enh_heads = 4;
    int64_t dec_layers = 3, dec_heads = 8, horizon = 12, action_dim = 7, state_dim = 8, n_state_tok = 2;
    int64_t text_len_max = 21;
    float   rope_theta = 100.0f;
    int32_t pad_id = 0, cls_id = 101, sep_id = 102, period_id = 1012, question_id = 1029;
    std::vector<TextGroup> text_groups;

    ggml_tensor *patch_w = nullptr, *patch_b = nullptr, *cls_tok = nullptr, *reg_tok = nullptr;
    std::vector<VitLayerW> vit;
    ggml_tensor *vp_in_w = nullptr, *vp_in_b = nullptr, *vp_fc1_w = nullptr, *vp_fc1_b = nullptr;
    ggml_tensor *vp_fc2_w = nullptr, *vp_fc2_b = nullptr, *vp_skip_w = nullptr;
    ggml_tensor *vp_out_w = nullptr, *vp_out_b = nullptr, *view_emb = nullptr;
    ggml_tensor *word_emb = nullptr, *pos_emb = nullptr, *type_emb = nullptr;
    ggml_tensor *emb_ln_w = nullptr, *emb_ln_b = nullptr;
    std::vector<BertLayerW> bert;
    ggml_tensor *text_proj_w = nullptr, *text_proj_b = nullptr;
    std::vector<FusionLayerW>   fusion;
    std::vector<EnhancerLayerW> enhancer;
    ggml_tensor *st_ln_w = nullptr, *st_ln_b = nullptr, *st_fc1_w = nullptr, *st_fc1_b = nullptr;
    ggml_tensor *st_fc2_w = nullptr, *st_fc2_b = nullptr, *st_pos = nullptr, *st_out_w = nullptr, *st_out_b = nullptr;
    ggml_tensor *act_q = nullptr;
    std::vector<DecoderLayerW> dec;
    ggml_tensor *ap0_w = nullptr, *ap0_b = nullptr, *ap1_w = nullptr, *ap1_b = nullptr;
    ggml_tensor *ap2_w = nullptr, *ap2_b = nullptr;

    // DINOv3's RoPE depends only on the patch grid, so its tables are computed
    // once and live next to the weights, shaped [head_dim, 1, T] to broadcast
    // over heads. Rows for the CLS and register tokens are the identity (cos 1,
    // sin 0), which lets the rotation cover the whole sequence instead of
    // splitting off the prefix in every layer. rope_sin carries rotate_half's
    // sign: negative on the first half of the head.
    ggml_tensor *rope_cos = nullptr, *rope_sin = nullptr;

    struct Key {
        int64_t bert_len = -1;
        bool operator==(const Key & o) const { return bert_len == o.bert_len; }
    };
    struct IO {
        ggml_tensor *patches = nullptr, *ids = nullptr, *pos = nullptr, *bert_mask = nullptr;
        ggml_tensor *enh_mask = nullptr, *fus_mask = nullptr, *state = nullptr, *actions = nullptr;
    };
    graph_cache<Key, IO> graph;

    int64_t grid() const      { return image_size / patch; }
    int64_t n_patches() const { return grid() * grid(); }
    int64_t vit_seq() const   { return 1 + n_reg + n_patches(); }

    int64_t text_len(const int32_t * ids, int64_t n) const;
    ggml_cgraph * build(ggml_context * C, IO & io, int64_t bert_len) const;
    bool upload_rope_tables();

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

ggml_tensor * split_heads(ggml_context * C, ggml_tensor * qkv, int64_t hd, int64_t heads,
                          int64_t T, int64_t nv, int part) {
    const size_t es = ggml_element_size(qkv);
    return ggml_view_4d(C, qkv, hd, heads, T, nv, (size_t) hd*es, qkv->nb[1], qkv->nb[2],
                        (size_t) part*hd*heads*es);
}

// Fused-QKV self-attention over [3*dim, T, nv]; returns [dim, T, nv].
ggml_tensor * self_attention(ggml_context * C, ggml_tensor * qkv, ggml_tensor * mask,
                             int64_t hd, int64_t heads, int64_t T, int64_t nv) {
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, split_heads(C, qkv, hd, heads, T, nv, 0), 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, split_heads(C, qkv, hd, heads, T, nv, 1), 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, split_heads(C, qkv, hd, heads, T, nv, 2), 1, 2, 0, 3));
    return attention(C, Q, K, V, mask, 1.0f / std::sqrt((float) hd), hd*heads, T, nv);
}

// Half-split RoPE applied in the projection's own [hd, heads, T, nv] layout, so
// no head permute is needed first: rotate_half(x) is [-x1, x0], and the sign
// lives in the sin table, leaving one concat of two views.
ggml_tensor * rope_heads(ggml_context * C, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_signed) {
    const int64_t half = x->ne[0] / 2;
    const size_t  es   = ggml_element_size(x);
    ggml_tensor * x0 = ggml_view_4d(C, x, half, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor * x1 = ggml_view_4d(C, x, half, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], half*es);
    return ggml_add(C, ggml_mul(C, x, cos_t), ggml_mul(C, ggml_concat(C, x1, x0, 0), sin_signed));
}

// DINOv3 attention. Flash attention takes Q and K as permuted views and casts
// K and V itself, so the rotated projections are never copied again.
ggml_tensor * vit_attention(ggml_context * C, ggml_tensor * qkv, ggml_tensor * cos_t, ggml_tensor * sin_signed,
                            int64_t hd, int64_t heads, int64_t T, int64_t nv, bool flash) {
    const float   scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * q = rope_heads(C, split_heads(C, qkv, hd, heads, T, nv, 0), cos_t, sin_signed);
    ggml_tensor * k = rope_heads(C, split_heads(C, qkv, hd, heads, T, nv, 1), cos_t, sin_signed);
    ggml_tensor * v = split_heads(C, qkv, hd, heads, T, nv, 2);
    if (flash) {
        ggml_tensor * o = flash_attention(C, ggml_permute(C, q, 0, 2, 1, 3), ggml_permute(C, k, 0, 2, 1, 3),
                                          ggml_permute(C, v, 0, 2, 1, 3), nullptr, scale);
        return ggml_reshape_3d(C, o, hd*heads, T, nv);
    }
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, q, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, k, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, v, 1, 2, 0, 3));
    return attention(C, Q, K, V, nullptr, scale, hd*heads, T, nv);
}

// The released BERT wrapper (text/bert.py, from Grounding DINO) splits the
// instruction at [CLS], [SEP], '.' and '?': a token attends only inside its
// sub-sentence and its position restarts there. Tokens outside any span (the
// padding, and a special token at either end) see only themselves at position 0.
void special_token_blocks(const std::vector<int32_t> & ids, const TurboVlaModelArch & m,
                          std::vector<uint8_t> & allowed, std::vector<int32_t> & pos) {
    const int64_t n = (int64_t) ids.size();
    allowed.assign((size_t) n*n, 0);
    pos.assign((size_t) n, 0);
    for (int64_t i = 0; i < n; ++i)
        allowed[(size_t) i*n + i] = 1;
    int64_t prev = 0;
    for (int64_t col = 0; col < n; ++col) {
        const int32_t id = ids[(size_t) col];
        if (id != m.cls_id && id != m.sep_id && id != m.period_id && id != m.question_id)
            continue;
        if (col != 0 && col != n - 1) {
            for (int64_t q = prev + 1; q <= col; ++q) {
                pos[(size_t) q] = (int32_t) (q - prev - 1);
                for (int64_t k = prev + 1; k <= col; ++k)
                    allowed[(size_t) q*n + k] = 1;
            }
        }
        prev = col;
    }
}

}  // namespace

// The ACT decoder attends every text row, padding included (encode_condition
// concatenates them unmasked), so the padded length is part of the output. The
// checkpoint pins it per training instruction (11, 14 or 21 for LIBERO): BERT
// runs over that many tokens and the rows after it, up to text_len_max, are
// zeros that the text projection turns into its bias. Instructions the
// checkpoint does not list are padded to text_len_max. A caller that sends
// trailing [PAD] tokens has chosen the length itself.
int64_t TurboVlaModelArch::text_len(const int32_t * ids, int64_t n) const {
    int64_t core = n;
    while (core > 0 && ids[core - 1] == pad_id)
        --core;
    if (core < n)
        return n;
    for (const TextGroup & g : text_groups)
        if ((int64_t) g.ids.size() == n && std::equal(g.ids.begin(), g.ids.end(), ids))
            return g.pad_to;
    return text_len_max;
}

bool TurboVlaModelArch::upload_rope_tables() {
    // DINOv3 axial RoPE (HF modeling_dinov3_vit.py): patch centres normalized to
    // [-1, 1] per axis, head_dim/4 periods base^(4i/head_dim), angles ordered
    // [h, w, h, w] across the head, rotated half-split. See rope_heads.
    const int64_t hd = vit_dim / vit_heads, q = hd / 4, T = vit_seq(), g = grid(), pre = 1 + n_reg;
    std::vector<float> c((size_t) hd*T, 1.0f), s((size_t) hd*T, 0.0f);
    for (int64_t p = 0; p < n_patches(); ++p) {
        const float h = 2.0f * ((float) (p / g) + 0.5f) / (float) g - 1.0f;
        const float w = 2.0f * ((float) (p % g) + 0.5f) / (float) g - 1.0f;
        float * cr = c.data() + (size_t) (pre + p)*hd;
        float * sr = s.data() + (size_t) (pre + p)*hd;
        for (int64_t i = 0; i < q; ++i) {
            const float period = std::pow(rope_theta, 4.0f * (float) i / (float) hd);
            const float ah = 6.28318530717958647692f * h / period;
            const float aw = 6.28318530717958647692f * w / period;
            for (int64_t blk = 0; blk < 4; ++blk) {
                const float a = (blk % 2 == 0) ? ah : aw;
                cr[blk*q + i] = std::cos(a);
                sr[blk*q + i] = blk < 2 ? -std::sin(a) : std::sin(a);
            }
        }
    }

    ggml_init_params p = { 2*ggml_tensor_overhead(), nullptr, true };
    ctx_const = ggml_init(p);
    if (!ctx_const)
        return false;
    rope_cos = ggml_new_tensor_3d(ctx_const, GGML_TYPE_F32, hd, 1, T);
    rope_sin = ggml_new_tensor_3d(ctx_const, GGML_TYPE_F32, hd, 1, T);
    const_buf = alloc_weights(ctx_const, backend);
    if (!const_buf)
        return false;
    ggml_backend_tensor_set(rope_cos, c.data(), 0, ggml_nbytes(rope_cos));
    ggml_backend_tensor_set(rope_sin, s.data(), 0, ggml_nbytes(rope_sin));
    return true;
}

ggml_cgraph * TurboVlaModelArch::build(ggml_context * C, IO & io, int64_t bert_len) const {
    const int64_t NP = n_patches(), T = vit_seq(), nv = n_views, VS = nv*NP, LT = text_len_max;
    const int64_t pdim = 3*patch*patch;

    // DINOv3. Patches arrive flattened on the host; views are the batch dim.
    io.patches = ggml_new_tensor_3d(C, GGML_TYPE_F32, pdim, NP, nv);
    ggml_set_input(io.patches);
    ggml_tensor * prefix = ggml_reshape_2d(C, cls_tok, vit_dim, 1);
    if (reg_tok)
        prefix = ggml_concat(C, prefix, reg_tok, 1);
    prefix = ggml_repeat_4d(C, prefix, vit_dim, 1 + n_reg, nv, 1);
    ggml_tensor * x = ggml_concat(C, prefix, linear(C, patch_w, patch_b, io.patches), 1);
    for (const VitLayerW & l : vit) {
        ggml_tensor * qkv = linear(C, l.qkv_w, l.qkv_b, layer_norm(C, x, l.ln1_w, l.ln1_b, kLnEps));
        ggml_tensor * att = vit_attention(C, qkv, rope_cos, rope_sin, vit_dim / vit_heads, vit_heads, T, nv,
                                          flash_attn_enabled());
        x = ggml_add(C, x, linear(C, l.o_w, l.o_b, att));
        x = ggml_add(C, x, ffn_gelu_erf(C, l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b,
                                        layer_norm(C, x, l.ln2_w, l.ln2_b, kLnEps)));
    }
    // TurboVLA reads hidden_states[-1], which is before DINOv3's final norm
    // (models/vision_encoder.py:109), then drops the CLS and register tokens.
    x = ggml_view_3d(C, x, vit_dim, NP, nv, x->nb[1], x->nb[2], (size_t) (1 + n_reg)*x->nb[1]);
    x = ggml_reshape_2d(C, ggml_cont(C, x), vit_dim, VS);

    // VisionProjection, then one learned embedding per camera.
    ggml_tensor * mlp = linear(C, vp_fc1_w, vp_fc1_b, layer_norm(C, x, vp_in_w, vp_in_b, kLnEps));
    mlp = linear(C, vp_fc2_w, vp_fc2_b, ggml_gelu_erf(C, mlp));
    ggml_tensor * v = layer_norm(C, ggml_add(C, ggml_mul_mat(C, vp_skip_w, x), mlp), vp_out_w, vp_out_b, kLnEps);
    v = ggml_reshape_2d(C, ggml_add(C, ggml_reshape_3d(C, v, hidden, NP, nv),
                                    ggml_reshape_3d(C, view_emb, hidden, 1, nv)), hidden, VS);

    // BERT over the checkpoint's padded length, token type 0 throughout.
    io.ids = ggml_new_tensor_1d(C, GGML_TYPE_I32, bert_len);
    io.pos = ggml_new_tensor_1d(C, GGML_TYPE_I32, bert_len);
    io.bert_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, bert_len, bert_len);
    ggml_set_input(io.ids);
    ggml_set_input(io.pos);
    ggml_set_input(io.bert_mask);
    ggml_tensor * t = ggml_add(C, ggml_get_rows(C, word_emb, io.ids), ggml_get_rows(C, pos_emb, io.pos));
    t = ggml_add(C, t, ggml_view_1d(C, type_emb, bert_dim, 0));
    t = layer_norm(C, t, emb_ln_w, emb_ln_b, kBertEps);
    for (const BertLayerW & l : bert) {
        ggml_tensor * att = self_attention(C, linear(C, l.qkv_w, l.qkv_b, t), io.bert_mask,
                                           bert_dim / bert_heads, bert_heads, bert_len, 1);
        t = layer_norm(C, ggml_add(C, t, linear(C, l.o_w, l.o_b, att)), l.ln1_w, l.ln1_b, kBertEps);
        t = layer_norm(C, ggml_add(C, t, ffn_gelu_erf(C, l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b, t)),
                       l.ln2_w, l.ln2_b, kBertEps);
    }
    ggml_tensor * lang = linear(C, text_proj_w, text_proj_b, t);
    if (bert_len < LT)
        lang = ggml_concat(C, lang, ggml_repeat_4d(C, text_proj_b, hidden, LT - bert_len, 1, 1), 1);

    // Fusion. One score matrix serves both directions upstream (the language
    // side softmaxes its transpose); two small products are cheaper than a
    // transpose here. Only vision queries mask padded language keys.
    io.enh_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, LT, LT);
    io.fus_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, LT, VS);
    ggml_set_input(io.enh_mask);
    ggml_set_input(io.fus_mask);
    const int64_t fhd = fusion_dim / fusion_heads, ehd = hidden / enh_heads;
    const float   fscale = 1.0f / std::sqrt((float) fhd);
    for (int64_t i = 0; i < fusion_layers; ++i) {
        const FusionLayerW & f = fusion[(size_t) i];
        ggml_tensor * vn = layer_norm(C, v, f.norm_v_w, f.norm_v_b, kLnEps);
        ggml_tensor * ln = layer_norm(C, lang, f.norm_l_w, f.norm_l_b, kLnEps);
        ggml_tensor * vp = linear(C, f.v_w, f.v_b, vn);    // [q_v | values_v]
        ggml_tensor * lp = linear(C, f.l_w, f.l_b, ln);    // [k_l | values_l]
        auto heads = [&](ggml_tensor * p, int64_t L, int part) {
            return ggml_view_3d(C, p, fhd, fusion_heads, L, (size_t) fhd*ggml_element_size(p), p->nb[1],
                                (size_t) part*fusion_dim*ggml_element_size(p));
        };
        ggml_tensor * qv = ggml_cont(C, ggml_permute(C, heads(vp, VS, 0), 0, 2, 1, 3));
        ggml_tensor * kl = ggml_cont(C, ggml_permute(C, heads(lp, LT, 0), 0, 2, 1, 3));
        ggml_tensor * vv = ggml_cont(C, ggml_permute(C, heads(vp, VS, 1), 1, 2, 0, 3));
        ggml_tensor * vl = ggml_cont(C, ggml_permute(C, heads(lp, LT, 1), 1, 2, 0, 3));
        ggml_tensor * dv = attention(C, qv, kl, vl, io.fus_mask, fscale, fusion_dim, VS);
        ggml_tensor * dl = attention(C, kl, qv, vv, nullptr,     fscale, fusion_dim, LT);
        // residual_style "normalized": the gated deltas land on the normed tokens.
        v    = ggml_add(C, vn, ggml_mul(C, linear(C, f.out_v_w, f.out_v_b, dv), f.gamma_v));
        lang = ggml_add(C, ln, ggml_mul(C, linear(C, f.out_l_w, f.out_l_b, dl), f.gamma_l));

        // Text enhancer: post-norm, ReLU, the sub-sentence mask and no
        // key-padding mask (the key_padding_mask argument is commented out
        // upstream, models/components/transformer.py).
        const EnhancerLayerW & e = enhancer[(size_t) i];
        ggml_tensor * att = self_attention(C, linear(C, e.qkv_w, e.qkv_b, lang), io.enh_mask, ehd, enh_heads, LT, 1);
        lang = layer_norm(C, ggml_add(C, lang, linear(C, e.o_w, e.o_b, att)), e.ln1_w, e.ln1_b, kLnEps);
        lang = layer_norm(C, ggml_add(C, lang, ffn_relu(C, e.fc1_w, e.fc1_b, e.fc2_w, e.fc2_b, lang)),
                          e.ln2_w, e.ln2_b, kLnEps);
    }

    // State tokens: LayerNorm, MLP to num_state_tokens*hidden, + position, norm.
    io.state = ggml_new_tensor_1d(C, GGML_TYPE_F32, state_dim);
    ggml_set_input(io.state);
    ggml_tensor * s = linear(C, st_fc1_w, st_fc1_b, layer_norm(C, io.state, st_ln_w, st_ln_b, kLnEps));
    s = ggml_reshape_2d(C, linear(C, st_fc2_w, st_fc2_b, ggml_gelu_erf(C, s)), hidden, n_state_tok);
    s = layer_norm(C, ggml_add(C, s, ggml_reshape_2d(C, st_pos, hidden, n_state_tok)), st_out_w, st_out_b, kLnEps);

    ggml_tensor * mem = ggml_concat(C, ggml_concat(C, v, lang, 1), s, 1);
    const int64_t M = VS + LT + n_state_tok;

    // ACT decoder: nn.TransformerDecoderLayer, norm_first, ReLU, no final norm.
    const int64_t dhd = hidden / dec_heads;
    const float   dscale = 1.0f / std::sqrt((float) dhd);
    ggml_tensor * a = act_q;
    for (const DecoderLayerW & l : dec) {
        ggml_tensor * qkv = linear(C, l.self_qkv_w, l.self_qkv_b, layer_norm(C, a, l.ln1_w, l.ln1_b, kLnEps));
        ggml_tensor * att = self_attention(C, qkv, nullptr, dhd, dec_heads, horizon, 1);
        a = ggml_add(C, a, linear(C, l.self_o_w, l.self_o_b, att));

        // in_proj is [W_q; W_k; W_v]: the queries keep the Q block of their
        // projection, the memory its K and V blocks. Slicing the weight instead
        // saves little and breaks ggml-openvino once the weights are BF16.
        ggml_tensor * q  = linear(C, l.cross_qkv_w, l.cross_qkv_b, layer_norm(C, a, l.ln2_w, l.ln2_b, kLnEps));
        ggml_tensor * kv = linear(C, l.cross_qkv_w, l.cross_qkv_b, mem);
        ggml_tensor * Q = ggml_cont(C, ggml_permute(C, split_heads(C, q, dhd, dec_heads, horizon, 1, 0), 0, 2, 1, 3));
        ggml_tensor * K = ggml_cont(C, ggml_permute(C, split_heads(C, kv, dhd, dec_heads, M, 1, 1), 0, 2, 1, 3));
        ggml_tensor * V = ggml_cont(C, ggml_permute(C, split_heads(C, kv, dhd, dec_heads, M, 1, 2), 1, 2, 0, 3));
        att = attention(C, Q, K, V, nullptr, dscale, hidden, horizon);
        a = ggml_add(C, a, linear(C, l.cross_o_w, l.cross_o_b, att));
        a = ggml_add(C, a, ffn_relu(C, l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b,
                                    layer_norm(C, a, l.ln3_w, l.ln3_b, kLnEps)));
    }
    a = ggml_relu(C, linear(C, ap0_w, ap0_b, a));
    a = ggml_relu(C, linear(C, ap1_w, ap1_b, a));
    io.actions = ggml_tanh(C, linear(C, ap2_w, ap2_b, a));
    ggml_set_output(io.actions);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_build_forward_expand(gf, io.actions);
    return gf;
}

namespace {

bool load_config(const gguf_reader & g, TurboVlaModelArch & m) {
    auto U = [&](const char * k, int64_t & dst) {
        char key[96];
        std::snprintf(key, sizeof(key), "turbovla.%s", k);
        if (!g.has(key))
            return;
        dst = (int64_t) g.u32(key);
    };
    auto I = [&](const char * k, int32_t & dst) {
        int64_t v = dst;
        U(k, v);
        dst = (int32_t) v;
    };
    U("hidden", m.hidden);                   U("num_views", m.n_views);
    U("image_size", m.image_size);           U("patch_size", m.patch);
    U("num_register_tokens", m.n_reg);
    U("vit_dim", m.vit_dim);                 U("vit_layers", m.vit_layers);         U("vit_heads", m.vit_heads);
    U("text_dim", m.bert_dim);               U("text_layers", m.bert_layers);       U("text_heads", m.bert_heads);
    U("vocab_size", m.vocab);
    U("num_fusion_layers", m.fusion_layers); U("fusion_heads", m.fusion_heads);
    U("text_enhancer_heads", m.enh_heads);
    U("num_action_decoder_layers", m.dec_layers); U("action_heads", m.dec_heads);
    U("action_horizon", m.horizon);          U("action_dim", m.action_dim);
    U("state_dim", m.state_dim);             U("num_state_tokens", m.n_state_tok);
    U("max_text_length", m.text_len_max);
    I("pad_token_id", m.pad_id);             I("cls_token_id", m.cls_id);           I("sep_token_id", m.sep_id);
    I("period_token_id", m.period_id);       I("question_token_id", m.question_id);
    if (g.has("turbovla.rope_theta"))
        m.rope_theta = g.f32("turbovla.rope_theta");
    int64_t fhd = m.fusion_dim / m.fusion_heads;
    U("fusion_head_dim", fhd);
    m.fusion_dim = fhd * m.fusion_heads;

    if (m.image_size % m.patch || m.vit_dim % m.vit_heads || (m.vit_dim / m.vit_heads) % 4 ||
        m.bert_dim % m.bert_heads || m.hidden % m.enh_heads || m.hidden % m.dec_heads) {
        std::fprintf(stderr, "vla(turbovla): inconsistent dimensions in GGUF metadata\n");
        return false;
    }

    // The padding table (see text_len). A GGUF without it predates the table and
    // would pad every instruction to max_text_length, which is wrong for the
    // shorter LIBERO groups.
    if (!g.has("turbovla.text_groups.count")) {
        std::fprintf(stderr, "vla(turbovla): GGUF has no text padding table; re-convert it with "
                             "scripts/convert_turbovla_to_gguf.py\n");
        return false;
    }
    const size_t n = g.u32("turbovla.text_groups.count");
    if (n == 0)
        return true;
    const int64_t kt = gguf_find_key(g.gctx, "turbovla.text_groups.tokens");
    const int64_t kl = gguf_find_key(g.gctx, "turbovla.text_groups.lengths");
    const int64_t kp = gguf_find_key(g.gctx, "turbovla.text_groups.pad_to");
    if (kt < 0 || kl < 0 || kp < 0) {
        std::fprintf(stderr, "vla(turbovla): malformed turbovla.text_groups metadata\n");
        return false;
    }
    if (gguf_get_arr_type(g.gctx, kt) != GGUF_TYPE_INT32 || gguf_get_arr_type(g.gctx, kl) != GGUF_TYPE_INT32 ||
        gguf_get_arr_type(g.gctx, kp) != GGUF_TYPE_INT32 || gguf_get_arr_n(g.gctx, kl) != n ||
        gguf_get_arr_n(g.gctx, kp) != n) {
        std::fprintf(stderr, "vla(turbovla): malformed turbovla.text_groups metadata\n");
        return false;
    }
    const int32_t * tok = (const int32_t *) gguf_get_arr_data(g.gctx, kt);
    const int32_t * len = (const int32_t *) gguf_get_arr_data(g.gctx, kl);
    const int32_t * pad = (const int32_t *) gguf_get_arr_data(g.gctx, kp);
    const size_t    n_tok = gguf_get_arr_n(g.gctx, kt);
    size_t off = 0;
    for (size_t i = 0; i < n; ++i) {
        if (len[i] <= 0 || off + (size_t) len[i] > n_tok || pad[i] < len[i] || pad[i] > m.text_len_max) {
            std::fprintf(stderr, "vla(turbovla): malformed turbovla.text_groups entry %zu\n", i);
            return false;
        }
        m.text_groups.push_back({ std::vector<int32_t>(tok + off, tok + off + len[i]), pad[i] });
        off += (size_t) len[i];
    }
    return true;
}

bool load_weights(TurboVlaModelArch & m, gguf_reader & g) {
    ggml_init_params wp = { (size_t) 16*1024*1024, nullptr, true };
    m.ctx_weights = ggml_init(wp);
    if (!m.ctx_weights)
        return false;
    WeightLoader L("turbovla", g, m.ctx_weights, m.mt);
    char b[128];
    auto N = [&](const char * fmt, int64_t i, const char * s) {
        std::snprintf(b, sizeof(b), fmt, (long long) i, s);
        return std::string(b);
    };

    m.cls_tok = L.f32("vit.cls_token");
    if (m.n_reg > 0)
        m.reg_tok = L.f32("vit.register_tokens");
    m.patch_w = L.gemm("vit.patch_embed.weight");
    m.patch_b = L.f32("vit.patch_embed.bias");
    m.vit.resize((size_t) m.vit_layers);
    for (int64_t i = 0; i < m.vit_layers; ++i) {
        VitLayerW & w = m.vit[(size_t) i];
        const char * f = "vit.blk.%lld.%s";
        w.ln1_w = L.f32("%s", N(f, i, "ln1.weight").c_str());
        w.ln1_b = L.f32("%s", N(f, i, "ln1.bias").c_str());
        w.qkv_w = L.fuse_gemm(N(f, i, "attn_qkv.weight").c_str(),
                              { N(f, i, "attn_q.weight"), N(f, i, "attn_k.weight"), N(f, i, "attn_v.weight") });
        w.qkv_b = L.fuse_f32(N(f, i, "attn_qkv.bias").c_str(),
                             { N(f, i, "attn_q.bias"), N(f, i, "attn_k.bias"), N(f, i, "attn_v.bias") });
        w.o_w   = L.gemm("%s", N(f, i, "attn_o.weight").c_str());
        w.o_b   = L.f32("%s", N(f, i, "attn_o.bias").c_str());
        w.ln2_w = L.f32("%s", N(f, i, "ln2.weight").c_str());
        w.ln2_b = L.f32("%s", N(f, i, "ln2.bias").c_str());
        w.fc1_w = L.gemm("%s", N(f, i, "fc1.weight").c_str());
        w.fc1_b = L.f32("%s", N(f, i, "fc1.bias").c_str());
        w.fc2_w = L.gemm("%s", N(f, i, "fc2.weight").c_str());
        w.fc2_b = L.f32("%s", N(f, i, "fc2.bias").c_str());
    }

    m.vp_in_w   = L.f32("vit_proj.input_norm.weight");
    m.vp_in_b   = L.f32("vit_proj.input_norm.bias");
    m.vp_fc1_w  = L.gemm("vit_proj.mlp.0.weight");
    m.vp_fc1_b  = L.f32("vit_proj.mlp.0.bias");
    m.vp_fc2_w  = L.gemm("vit_proj.mlp.3.weight");
    m.vp_fc2_b  = L.f32("vit_proj.mlp.3.bias");
    m.vp_skip_w = L.gemm("vit_proj.skip.weight");
    m.vp_out_w  = L.f32("vit_proj.output_norm.weight");
    m.vp_out_b  = L.f32("vit_proj.output_norm.bias");
    m.view_emb  = L.f32("view_emb");

    m.word_emb = L.f32("text.embed.word_embeddings");
    m.pos_emb  = L.f32("text.embed.position_embeddings");
    m.type_emb = L.f32("text.embed.token_type_embeddings");
    m.emb_ln_w = L.f32("text.embed.LayerNorm.weight");
    m.emb_ln_b = L.f32("text.embed.LayerNorm.bias");
    m.bert.resize((size_t) m.bert_layers);
    for (int64_t i = 0; i < m.bert_layers; ++i) {
        BertLayerW & w = m.bert[(size_t) i];
        const char * f = "text.encoder.layer.%lld.%s";
        w.qkv_w = L.fuse_gemm(N(f, i, "attention.self.qkv.weight").c_str(),
                              { N(f, i, "attention.self.query.weight"), N(f, i, "attention.self.key.weight"),
                                N(f, i, "attention.self.value.weight") });
        w.qkv_b = L.fuse_f32(N(f, i, "attention.self.qkv.bias").c_str(),
                             { N(f, i, "attention.self.query.bias"), N(f, i, "attention.self.key.bias"),
                               N(f, i, "attention.self.value.bias") });
        w.o_w   = L.gemm("%s", N(f, i, "attention.output.dense.weight").c_str());
        w.o_b   = L.f32("%s", N(f, i, "attention.output.dense.bias").c_str());
        w.ln1_w = L.f32("%s", N(f, i, "attention.output.LayerNorm.weight").c_str());
        w.ln1_b = L.f32("%s", N(f, i, "attention.output.LayerNorm.bias").c_str());
        w.fc1_w = L.gemm("%s", N(f, i, "intermediate.dense.weight").c_str());
        w.fc1_b = L.f32("%s", N(f, i, "intermediate.dense.bias").c_str());
        w.fc2_w = L.gemm("%s", N(f, i, "output.dense.weight").c_str());
        w.fc2_b = L.f32("%s", N(f, i, "output.dense.bias").c_str());
        w.ln2_w = L.f32("%s", N(f, i, "output.LayerNorm.weight").c_str());
        w.ln2_b = L.f32("%s", N(f, i, "output.LayerNorm.bias").c_str());
    }
    m.text_proj_w = L.gemm("text_proj.weight");
    m.text_proj_b = L.f32("text_proj.bias");

    m.fusion.resize((size_t) m.fusion_layers);
    m.enhancer.resize((size_t) m.fusion_layers);
    for (int64_t i = 0; i < m.fusion_layers; ++i) {
        FusionLayerW & w = m.fusion[(size_t) i];
        const char * f = "vl_fusion.%lld.%s";
        w.norm_v_w = L.f32("%s", N(f, i, "norm_v.weight").c_str());
        w.norm_v_b = L.f32("%s", N(f, i, "norm_v.bias").c_str());
        w.norm_l_w = L.f32("%s", N(f, i, "norm_l.weight").c_str());
        w.norm_l_b = L.f32("%s", N(f, i, "norm_l.bias").c_str());
        w.v_w = L.fuse_gemm(N(f, i, "v.weight").c_str(), { N(f, i, "v_proj.weight"), N(f, i, "values_v.weight") });
        w.v_b = L.fuse_f32(N(f, i, "v.bias").c_str(),    { N(f, i, "v_proj.bias"),   N(f, i, "values_v.bias") });
        w.l_w = L.fuse_gemm(N(f, i, "l.weight").c_str(), { N(f, i, "l_proj.weight"), N(f, i, "values_l.weight") });
        w.l_b = L.fuse_f32(N(f, i, "l.bias").c_str(),    { N(f, i, "l_proj.bias"),   N(f, i, "values_l.bias") });
        w.out_v_w = L.gemm("%s", N(f, i, "out_v.weight").c_str());
        w.out_v_b = L.f32("%s", N(f, i, "out_v.bias").c_str());
        w.out_l_w = L.gemm("%s", N(f, i, "out_l.weight").c_str());
        w.out_l_b = L.f32("%s", N(f, i, "out_l.bias").c_str());
        w.gamma_v = L.f32("%s", N(f, i, "gamma_v").c_str());
        w.gamma_l = L.f32("%s", N(f, i, "gamma_l").c_str());

        EnhancerLayerW & e = m.enhancer[(size_t) i];
        const char * t = "vl_text.%lld.%s";
        e.qkv_w = L.gemm("%s", N(t, i, "attn_qkv.weight").c_str());
        e.qkv_b = L.f32("%s", N(t, i, "attn_qkv.bias").c_str());
        e.o_w   = L.gemm("%s", N(t, i, "attn_o.weight").c_str());
        e.o_b   = L.f32("%s", N(t, i, "attn_o.bias").c_str());
        e.ln1_w = L.f32("%s", N(t, i, "ln1.weight").c_str());
        e.ln1_b = L.f32("%s", N(t, i, "ln1.bias").c_str());
        e.fc1_w = L.gemm("%s", N(t, i, "fc1.weight").c_str());
        e.fc1_b = L.f32("%s", N(t, i, "fc1.bias").c_str());
        e.fc2_w = L.gemm("%s", N(t, i, "fc2.weight").c_str());
        e.fc2_b = L.f32("%s", N(t, i, "fc2.bias").c_str());
        e.ln2_w = L.f32("%s", N(t, i, "ln2.weight").c_str());
        e.ln2_b = L.f32("%s", N(t, i, "ln2.bias").c_str());
    }

    m.st_ln_w  = L.f32("state.proj.0.weight");
    m.st_ln_b  = L.f32("state.proj.0.bias");
    m.st_fc1_w = L.gemm("state.proj.1.weight");
    m.st_fc1_b = L.f32("state.proj.1.bias");
    m.st_fc2_w = L.gemm("state.proj.4.weight");
    m.st_fc2_b = L.f32("state.proj.4.bias");
    m.st_pos   = L.f32("state.proj.position");
    m.st_out_w = L.f32("state.proj.output_norm.weight");
    m.st_out_b = L.f32("state.proj.output_norm.bias");

    m.act_q = L.f32("act.q.weight");
    m.dec.resize((size_t) m.dec_layers);
    for (int64_t i = 0; i < m.dec_layers; ++i) {
        DecoderLayerW & w = m.dec[(size_t) i];
        const char * f = "act.dec.%lld.%s";
        w.ln1_w = L.f32("%s", N(f, i, "ln1.weight").c_str());
        w.ln1_b = L.f32("%s", N(f, i, "ln1.bias").c_str());
        w.self_qkv_w = L.gemm("%s", N(f, i, "self_qkv.weight").c_str());
        w.self_qkv_b = L.f32("%s", N(f, i, "self_qkv.bias").c_str());
        w.self_o_w = L.gemm("%s", N(f, i, "self_out.weight").c_str());
        w.self_o_b = L.f32("%s", N(f, i, "self_out.bias").c_str());
        w.ln2_w = L.f32("%s", N(f, i, "ln2.weight").c_str());
        w.ln2_b = L.f32("%s", N(f, i, "ln2.bias").c_str());
        w.cross_qkv_w = L.gemm("%s", N(f, i, "cross_qkv.weight").c_str());
        w.cross_qkv_b = L.f32("%s", N(f, i, "cross_qkv.bias").c_str());
        w.cross_o_w = L.gemm("%s", N(f, i, "cross_out.weight").c_str());
        w.cross_o_b = L.f32("%s", N(f, i, "cross_out.bias").c_str());
        w.ln3_w = L.f32("%s", N(f, i, "ln3.weight").c_str());
        w.ln3_b = L.f32("%s", N(f, i, "ln3.bias").c_str());
        w.fc1_w = L.gemm("%s", N(f, i, "fc1.weight").c_str());
        w.fc1_b = L.f32("%s", N(f, i, "fc1.bias").c_str());
        w.fc2_w = L.gemm("%s", N(f, i, "fc2.weight").c_str());
        w.fc2_b = L.f32("%s", N(f, i, "fc2.bias").c_str());
    }
    m.ap0_w = L.gemm("act.proj.0.weight");
    m.ap0_b = L.f32("act.proj.0.bias");
    m.ap1_w = L.gemm("act.proj.1.weight");
    m.ap1_b = L.f32("act.proj.1.bias");
    m.ap2_w = L.gemm("act.proj.2.weight");
    m.ap2_b = L.f32("act.proj.2.bias");

    if (!L.upload(m.backend, &m.weight_buf))
        return false;

    m.bert_max_pos = m.pos_emb->ne[1];
    if (m.vocab != m.word_emb->ne[1] || m.text_len_max > m.bert_max_pos ||
        m.patch_w->ne[0] != 3*m.patch*m.patch || (m.reg_tok && m.reg_tok->ne[1] != m.n_reg) ||
        ggml_nelements(m.view_emb) != m.hidden*m.n_views || m.act_q->ne[1] != m.horizon ||
        m.dec[0].cross_qkv_w->ne[1] != 3*m.hidden) {
        std::fprintf(stderr, "vla(turbovla): tensor shapes disagree with GGUF metadata\n");
        return false;
    }
    return true;
}

// Flattens one view into DINOv3's patch-embedding input: column p is patch p
// (row-major over the grid), rows follow the conv weight's (c, ky, kx) order.
void patchify(const ImageView & v, int64_t side, int64_t P, float * out) {
    const int64_t g = side / P, pdim = 3*P*P;
    for (int64_t p = 0; p < g*g; ++p) {
        const int64_t y0 = (p / g)*P, x0 = (p % g)*P;
        float * col = out + p*pdim;
        for (int64_t c = 0; c < 3; ++c)
            for (int64_t ky = 0; ky < P; ++ky)
                for (int64_t kx = 0; kx < P; ++kx) {
                    const size_t idx = (size_t) ((y0 + ky)*side + x0 + kx)*3 + c;
                    const float px = v.format == PixelFormat::U8 ? ((const uint8_t *) v.data)[idx] / 255.0f
                                                                  : ((const float *) v.data)[idx];
                    col[(c*P + ky)*P + kx] = (px - kImagenetMean[c]) / kImagenetStd[c];
                }
    }
}

}  // namespace

std::unique_ptr<ModelArchBase> turbovla_create(const std::string& mmproj_path,
                                               const std::string& ckpt_path,
                                               const std::string&,
                                               const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(turbovla): note - mmproj '%s' is ignored (vision is baked into the GGUF)\n",
                    mmproj_path.c_str());

    auto m = std::make_unique<TurboVlaModelArch>();
    m->mt = opts.weight_dtype.value_or(GGML_TYPE_F32);

    gguf_reader g("turbovla");
    if (!g.open(ckpt_path))
        return nullptr;
    if (!g.has("turbovla.architecture")) {
        std::fprintf(stderr, "vla(turbovla): %s is not a TurboVLA GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, *m))
        return nullptr;

    const Backend b = backend_init("vla(turbovla)", m->n_threads);
    if (!b.handle)
        return nullptr;
    m->backend = b.handle;

    if (!load_weights(*m, g) || !m->upload_rope_tables())
        return nullptr;

    m->cfg.n_img           = m->n_views * m->n_patches();
    m->cfg.n_lang          = m->text_len_max;
    m->cfg.n_state         = m->n_state_tok;
    m->cfg.n_suffix        = m->horizon;
    m->cfg.hidden          = m->hidden;
    m->cfg.max_state_dim   = m->state_dim;
    m->cfg.real_state_dim  = m->state_dim;
    m->cfg.max_action_dim  = m->action_dim;
    m->cfg.real_action_dim = m->action_dim;

    std::printf("vla(turbovla): weights resident %.2f GiB (%s) - DINOv3 ViT-B/16 x%lld views + BERT + "
                "%lld fusion layers + ACT decoder, horizon %lld, %zu text groups\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0), dtype_name(m->mt),
                (long long) m->n_views, (long long) m->fusion_layers, (long long) m->horizon,
                m->text_groups.size());
    return m;
}

std::vector<float> TurboVlaModelArch::predict(const Inputs& in) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    stats = Stats{};

    if (in.precomputed_img_emb) {
        std::fprintf(stderr, "vla(turbovla): precomputed_img_emb is not supported; pass raw images\n");
        return {};
    }
    if (!in.images || in.n_images != n_views) {
        std::fprintf(stderr, "vla(turbovla): expected %lld views (agentview, wrist), got %d\n",
                     (long long) n_views, in.n_images);
        return {};
    }
    for (int64_t i = 0; i < n_views; ++i) {
        const ImageView & iv = in.images[i];
        if (!view_is_side(iv.data, iv.w, iv.h, image_size)) {
            std::fprintf(stderr, "vla(turbovla): view %lld is %dx%d, expected %lldx%lld\n",
                         (long long) i, iv.w, iv.h, (long long) image_size, (long long) image_size);
            return {};
        }
    }
    if (!in.lang_tokens || in.n_lang < 1 || in.n_lang > text_len_max) {
        std::fprintf(stderr, "vla(turbovla): %d language tokens; need 1..%lld (truncate when tokenizing)\n",
                     in.n_lang, (long long) text_len_max);
        return {};
    }
    // ggml_get_rows does not bound-check.
    for (int i = 0; i < in.n_lang; ++i)
        if (in.lang_tokens[i] < 0 || in.lang_tokens[i] >= vocab) {
            std::fprintf(stderr, "vla(turbovla): token %d out of vocab\n", in.lang_tokens[i]);
            return {};
        }

    const int64_t LT = text_len_max, VS = n_views*n_patches();
    const int64_t bert_len = text_len(in.lang_tokens, in.n_lang);
    std::vector<int32_t> ids((size_t) bert_len, pad_id);
    std::copy(in.lang_tokens, in.lang_tokens + in.n_lang, ids.begin());

    std::vector<uint8_t> allowed;
    std::vector<int32_t> pos;
    special_token_blocks(ids, *this, allowed, pos);
    const float NEG = -INFINITY;
    std::vector<float> bert_mask((size_t) bert_len*bert_len), enh_mask((size_t) LT*LT, NEG), fus_mask((size_t) LT*VS);
    for (int64_t q = 0; q < bert_len; ++q)
        for (int64_t k = 0; k < bert_len; ++k) {
            bert_mask[(size_t) q*bert_len + k] = allowed[(size_t) q*bert_len + k] ? 0.0f : NEG;
            enh_mask[(size_t) q*LT + k] = bert_mask[(size_t) q*bert_len + k];
        }
    for (int64_t q = bert_len; q < LT; ++q)
        enh_mask[(size_t) q*LT + q] = 0.0f;
    for (int64_t k = 0; k < LT; ++k) {
        const float val = (k < bert_len && ids[(size_t) k] != pad_id) ? 0.0f : NEG;
        for (int64_t q = 0; q < VS; ++q)
            fus_mask[(size_t) q*LT + k] = val;
    }

    std::vector<float> patches((size_t) 3*patch*patch*VS);
    for (int64_t i = 0; i < n_views; ++i)
        patchify(in.images[i], image_size, patch, patches.data() + (size_t) i*3*patch*patch*n_patches());
    std::vector<float> state((size_t) state_dim, 0.0f);
    if (in.state)
        std::copy(in.state, in.state + state_dim, state.begin());

    const size_t arena = ggml_tensor_overhead()*8192 + ggml_graph_overhead_custom(8192, false);
    if (!graph.ensure(backend, Key{bert_len}, arena,
                      [&](ggml_context * C, IO & io) { return build(C, io, bert_len); })) {
        std::fprintf(stderr, "vla(turbovla): graph build/alloc failed\n");
        return {};
    }
    IO & io = graph.io();
    ggml_backend_tensor_set(io.patches,   patches.data(),   0, ggml_nbytes(io.patches));
    ggml_backend_tensor_set(io.ids,       ids.data(),       0, ggml_nbytes(io.ids));
    ggml_backend_tensor_set(io.pos,       pos.data(),       0, ggml_nbytes(io.pos));
    ggml_backend_tensor_set(io.bert_mask, bert_mask.data(), 0, ggml_nbytes(io.bert_mask));
    ggml_backend_tensor_set(io.enh_mask,  enh_mask.data(),  0, ggml_nbytes(io.enh_mask));
    ggml_backend_tensor_set(io.fus_mask,  fus_mask.data(),  0, ggml_nbytes(io.fus_mask));
    ggml_backend_tensor_set(io.state,     state.data(),     0, ggml_nbytes(io.state));

    const auto tc = clock::now();
    graph_unique_names(graph.graph());
    if (ggml_backend_graph_compute(backend, graph.graph()) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(turbovla): compute failed\n");
        return {};
    }
    std::vector<float> out((size_t) (horizon*action_dim));
    ggml_backend_tensor_get(io.actions, out.data(), 0, out.size()*sizeof(float));

    stats.ms_inference = std::chrono::duration<float, std::milli>(clock::now() - tc).count();
    stats.ms_total     = std::chrono::duration<float, std::milli>(clock::now() - t0).count();
    return out;
}

}  // namespace vla
