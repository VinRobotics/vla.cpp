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
    * @file turbovla.cpp
    * @brief TurboVLA implementation: DINOv3 ViT + BERT + VL Fusion + ACT decoder.
    */

    #include "arch.h"
    #include "loader.h"
    #include "model.h"
    #include "options.h"

    #include "layers/linear.h"
    #include "layers/norm.h"
    #include "layers/ffn.h"
    #include "layers/attn.h"
    #include "layers/rope.h"
    #include "modules/preprocess.h"
    #include "gguf_reader.h"
    #include "scratch_ctx.h"
    #include "backend.h"

    #include "ggml.h"
    #include "ggml-cpu.h"
    #include "ggml-backend.h"

    #include <algorithm>
    #include <cmath>
    #include <cfloat>
    #include <cstdio>
    #include <cstring>
    #include <memory>
    #include <stdexcept>
    #include <string>
    #include <vector>

    namespace vla {

    struct TurboVLAConfig {
        int64_t hidden = 256;
        int64_t vit_dim = 768;
        int64_t vit_layers = 12;
        int64_t vit_head_dim = 64;
        int64_t vit_heads = 12;
        int64_t text_dim = 768;
        int64_t text_layers = 12;
        int64_t text_head_dim = 64;
        int64_t text_heads = 12;
        int64_t vocab_size = 30522;
        int64_t max_text_length = 256;
        int64_t num_fusion_layers = 6;
        int64_t fusion_heads = 4;
        int64_t fusion_head_dim = 256;
        int64_t text_enhancer_heads = 4;
        int64_t text_enhancer_head_dim = 64;
        int64_t action_heads = 8;
        int64_t action_head_dim = 32;
        int64_t num_text_layers = 6;
        int64_t num_action_decoder_layers = 3;
        int64_t action_dim = 7;
        int64_t state_dim = 8;
        int64_t num_state_tokens = 2;
        int64_t action_horizon = 12;
        int64_t image_size = 256;
        int64_t patch_size = 16;
        int64_t num_views = 2;
        int64_t num_register_tokens = 0;
        float rope_theta = 10000.0f;
        float dropout = 0.1f;
        int64_t pad_token_id = 0;
        int64_t cls_token_id = 101;
        int64_t sep_token_id = 102;
        int64_t period_token_id = 1012;
        int64_t question_token_id = 1029;
    };

    namespace {

    struct VitBlockW {
        ggml_tensor* ln1_w, *ln1_b;
        ggml_tensor* ln2_w, *ln2_b;
        ggml_tensor* attn_q_w, *attn_q_b;
        ggml_tensor* attn_k_w, *attn_k_b;
        ggml_tensor* attn_v_w, *attn_v_b;
        ggml_tensor* attn_o_w, *attn_o_b;
        ggml_tensor* fc1_w, *fc1_b;
        ggml_tensor* fc2_w, *fc2_b;
    };

    struct VisionEncoderW {
        ggml_tensor* cls_token;
        ggml_tensor* patch_embed_w;
        ggml_tensor* patch_embed_b;
        ggml_tensor* register_tokens;
        std::vector<VitBlockW> layers;
        ggml_tensor* final_ln_w;
        ggml_tensor* final_ln_b;
        // One frequency tensor per patch-grid position.
        std::vector<ggml_tensor*> rope_freqs;
    };

    struct VisionProjectionW {
        ggml_tensor* input_norm_w, *input_norm_b;
        ggml_tensor* mlp_0_w, *mlp_0_b;
        ggml_tensor* mlp_3_w, *mlp_3_b;
        ggml_tensor* skip_w;
        ggml_tensor* output_norm_w, *output_norm_b;
    };

    struct BertLayerW {
        ggml_tensor* attn_q_w, *attn_q_b;
        ggml_tensor* attn_k_w, *attn_k_b;
        ggml_tensor* attn_v_w, *attn_v_b;
        ggml_tensor* attn_o_w, *attn_o_b;
        ggml_tensor* ln1_w, *ln1_b;
        ggml_tensor* fc1_w, *fc1_b;
        ggml_tensor* fc2_w, *fc2_b;
        ggml_tensor* ln2_w, *ln2_b;
    };

    struct TextEncoderW {
        ggml_tensor* word_embed;
        ggml_tensor* pos_embed;
        ggml_tensor* tok_type_embed;
        ggml_tensor* embed_ln_w, *embed_ln_b;
        std::vector<BertLayerW> layers;
        ggml_tensor* pooler_w, *pooler_b;
        ggml_tensor* text_proj_w, *text_proj_b;
    };

    struct VlFusionLayerW {
        ggml_tensor* v_proj_w, *v_proj_b;
        ggml_tensor* l_proj_w, *l_proj_b;
        ggml_tensor* values_v_w, *values_v_b;
        ggml_tensor* values_l_w, *values_l_b;
        ggml_tensor* out_v_w, *out_v_b;
        ggml_tensor* out_l_w, *out_l_b;
        ggml_tensor* norm_v_w, *norm_v_b;
        ggml_tensor* norm_l_w, *norm_l_b;
        ggml_tensor* gamma_v;
        ggml_tensor* gamma_l;
    };

    struct VlTextLayerW {
        ggml_tensor* attn_qkv_w, *attn_qkv_b;
        ggml_tensor* attn_o_w, *attn_o_b;
        ggml_tensor* ln1_w, *ln1_b;
        ggml_tensor* fc1_w, *fc1_b;
        ggml_tensor* fc2_w, *fc2_b;
        ggml_tensor* ln2_w, *ln2_b;
    };

    struct ActDecoderLayerW {
        ggml_tensor* self_qkv_w, *self_qkv_b;
        ggml_tensor* self_out_w, *self_out_b;
        ggml_tensor* cross_qkv_w, *cross_qkv_b;
        ggml_tensor* cross_out_w, *cross_out_b;
        ggml_tensor* ln1_w, *ln1_b;
        ggml_tensor* ln2_w, *ln2_b;
        ggml_tensor* ln3_w, *ln3_b;
        ggml_tensor* fc1_w, *fc1_b;
        ggml_tensor* fc2_w, *fc2_b;
    };

    struct ActionDecoderW {
        ggml_tensor* action_q;
        std::vector<ActDecoderLayerW> layers;
        ggml_tensor* proj_0_w, *proj_0_b;
        ggml_tensor* proj_1_w, *proj_1_b;
        ggml_tensor* proj_2_w, *proj_2_b;
    };

    struct StateProjectionW {
        ggml_tensor* net_0_w, *net_0_b;
        ggml_tensor* net_1_w, *net_1_b;
        ggml_tensor* net_4_w, *net_4_b;
        ggml_tensor* output_norm_w, *output_norm_b;
        ggml_tensor* position;
    };

    static void check_heads(const char* tag, ggml_tensor* t, int64_t hd, int64_t heads, int64_t T, int64_t nv = 1) {
        (void) tag;
        const int64_t got = ggml_nelements(t);
        const int64_t expected = hd * heads * T * nv;
        GGML_ASSERT(got == expected);
    }

    }  // anonymous namespace

    struct TurboVLA : public ModelArchBase {
        TurboVLA() : ModelArchBase(Arch::TURBOVLA) {}
        ~TurboVLA() override;

        ggml_backend_t backend = nullptr;
        int n_threads = default_cpu_threads();
        ggml_context* ctx_weights = nullptr;
        ggml_backend_buffer_t weight_buf = nullptr;
        scratch_ctx scratch;

        TurboVLAConfig cfg_;

        VisionEncoderW vision_encoder;
        VisionProjectionW vision_proj;
        TextEncoderW text_encoder;
        std::vector<VlFusionLayerW> fusion_layers;
        std::vector<VlTextLayerW> text_layers;
        ActionDecoderW action_decoder;
        StateProjectionW state_proj;
        ggml_tensor* view_emb = nullptr;

        std::vector<float> predict(const Inputs& in) override;

    private:
        ggml_tensor* vit_layer(ggml_context* C, const VitBlockW& w, ggml_tensor* x,
                                ggml_tensor* rope_cos, ggml_tensor* rope_sin,
                                int64_t internal_seq, int64_t n_heads, int64_t head_dim) const;
        ggml_tensor* bert_layer(ggml_context* C, const BertLayerW& w, ggml_tensor* x,
                                ggml_tensor* self_mask,
                                int64_t seq, int64_t n_heads, int64_t head_dim) const;
        ggml_tensor* vl_fusion_layer(ggml_context* C, const VlFusionLayerW& w,
                                    ggml_tensor* v, ggml_tensor*& l,
                                    ggml_tensor* language_key_mask,
                                    int64_t v_seq, int64_t l_seq,
                                    int64_t n_heads, int64_t head_dim) const;
        ggml_tensor* vl_text_layer(ggml_context* C, const VlTextLayerW& w, ggml_tensor* x,
                                ggml_tensor* self_mask,
                                int64_t seq, int64_t n_heads, int64_t head_dim) const;
        ggml_tensor* act_decoder_layer(ggml_context* C, const ActDecoderLayerW& w,
                                        ggml_tensor* queries, ggml_tensor* memory,
                                        int64_t q_seq, int64_t m_seq,
                                        int64_t n_heads, int64_t head_dim) const;
    };

    TurboVLA::~TurboVLA() {
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend) ggml_backend_free(backend);
    }

    ggml_tensor* TurboVLA::vit_layer(ggml_context* C, const VitBlockW& w, ggml_tensor* x,
                                    ggml_tensor* rope_cos, ggml_tensor* rope_sin,
                                    int64_t internal_seq, int64_t n_heads, int64_t head_dim) const {
        const float scale = 1.0f / std::sqrt((float)head_dim);
        const int64_t hidden = n_heads * head_dim;
        const int64_t patch_start = 1 + cfg_.num_register_tokens;
        const int64_t n_patches = internal_seq - patch_start;
        GGML_ASSERT(n_patches > 0 && rope_cos->ne[1] == n_patches && rope_sin->ne[1] == n_patches);

        ggml_tensor* normed = layer_norm(C, x, w.ln1_w, w.ln1_b, 1e-5f);
        ggml_tensor* q = linear(C, w.attn_q_w, w.attn_q_b, normed);
        ggml_tensor* k = linear(C, w.attn_k_w, w.attn_k_b, normed);
        ggml_tensor* v = linear(C, w.attn_v_w, w.attn_v_b, normed);

        ggml_tensor* Q = to_heads(C, q, head_dim, n_heads, internal_seq);
        ggml_tensor* K = to_heads(C, k, head_dim, n_heads, internal_seq);
        ggml_tensor* V = to_heads_v(C, v, head_dim, n_heads, internal_seq);
        const size_t q_offset = (size_t) patch_start * Q->nb[1];
        ggml_tensor* Q_prefix = ggml_cont(C, ggml_view_3d(C, Q, head_dim, patch_start, n_heads,
                                                            Q->nb[1], Q->nb[2], 0));
        ggml_tensor* K_prefix = ggml_cont(C, ggml_view_3d(C, K, head_dim, patch_start, n_heads,
                                                            K->nb[1], K->nb[2], 0));
        ggml_tensor* Q_patch = ggml_cont(C, ggml_view_3d(C, Q, head_dim, n_patches, n_heads,
                                                           Q->nb[1], Q->nb[2], q_offset));
        ggml_tensor* K_patch = ggml_cont(C, ggml_view_3d(C, K, head_dim, n_patches, n_heads,
                                                           K->nb[1], K->nb[2], (size_t) patch_start*K->nb[1]));
        Q_patch = rope_dinov3_axial(C, Q_patch, rope_cos, rope_sin);
        K_patch = rope_dinov3_axial(C, K_patch, rope_cos, rope_sin);
        Q = ggml_concat(C, Q_prefix, Q_patch, 1);
        K = ggml_concat(C, K_prefix, K_patch, 1);
        ggml_tensor* att = attention(C, Q, K, V, nullptr, scale, hidden, internal_seq);
        ggml_tensor* o = linear(C, w.attn_o_w, w.attn_o_b, att);
        ggml_tensor* h = ggml_add(C, x, o);
        ggml_tensor* mlp = ffn_gelu_erf(C, w.fc1_w, w.fc1_b, w.fc2_w, w.fc2_b,
                                        layer_norm(C, h, w.ln2_w, w.ln2_b, 1e-5f));
        return ggml_add(C, h, mlp);
    }

    ggml_tensor* TurboVLA::bert_layer(ggml_context* C, const BertLayerW& w, ggml_tensor* x,
                                    ggml_tensor* self_mask,
                                    int64_t seq, int64_t n_heads, int64_t head_dim) const {
        const float scale = 1.0f / std::sqrt((float)head_dim);
        const int64_t hidden = n_heads * head_dim;

        // BERT uses post-norm blocks: attention on x, residual + LayerNorm,
        // then FFN, residual + LayerNorm.
        ggml_tensor* q = linear(C, w.attn_q_w, w.attn_q_b, x);
        ggml_tensor* k = linear(C, w.attn_k_w, w.attn_k_b, x);
        ggml_tensor* v = linear(C, w.attn_v_w, w.attn_v_b, x);

        ggml_tensor* Q = to_heads(C, q, head_dim, n_heads, seq);
        ggml_tensor* K = to_heads(C, k, head_dim, n_heads, seq);
        ggml_tensor* V = to_heads_v(C, v, head_dim, n_heads, seq);

        ggml_tensor* att = attention(C, Q, K, V, self_mask, scale, hidden, seq);
        ggml_tensor* o = linear(C, w.attn_o_w, w.attn_o_b, att);
        ggml_tensor* h = layer_norm(C,
                                    ggml_add(C, x, o),
                                    w.ln1_w,
                                    w.ln1_b,
                                    1e-12f);

        ggml_tensor* ffn_out = ffn_gelu_erf(C,
                                            w.fc1_w,
                                            w.fc1_b,
                                            w.fc2_w,
                                            w.fc2_b,
                                            h);
        return layer_norm(C,
                        ggml_add(C, h, ffn_out),
                        w.ln2_w,
                        w.ln2_b,
                        1e-12f);
    }

    ggml_tensor* TurboVLA::vl_fusion_layer(ggml_context* C, const VlFusionLayerW& w,
                                            ggml_tensor* v, ggml_tensor*& l,
                                            ggml_tensor* language_key_mask,
                                            int64_t v_seq, int64_t l_seq,
                                            int64_t n_heads, int64_t head_dim) const {
        const float scale = 1.0f / std::sqrt((float)head_dim);

        const int64_t att_dim = n_heads * head_dim;

        // TurboVLA's normalized residual style adds the deltas to normalized
        // tokens, not the pre-normalization residuals. See
        // H-EmbodVis/TurboVLA@b29ab142, models/components/fusion.py:336-347.
        ggml_tensor* v_norm = layer_norm(C, v, w.norm_v_w, w.norm_v_b, 1e-5f);
        ggml_tensor* l_norm = layer_norm(C, l, w.norm_l_w, w.norm_l_b, 1e-5f);

        // Only vision-to-language attention masks language keys. See
        // H-EmbodVis/TurboVLA@b29ab142, models/components/fusion.py:246-267.
        ggml_tensor* q_v = linear(C, w.v_proj_w, w.v_proj_b, v_norm);
        ggml_tensor* k_l = linear(C, w.l_proj_w, w.l_proj_b, l_norm);
        ggml_tensor* v_l = linear(C, w.values_l_w, w.values_l_b, l_norm);

        ggml_tensor* Q = to_heads(C, q_v, head_dim, n_heads, v_seq);
        ggml_tensor* K = to_heads(C, k_l, head_dim, n_heads, l_seq);
        ggml_tensor* V = to_heads_v(C, v_l, head_dim, n_heads, l_seq);
        ggml_tensor* o = attention(C, Q, K, V, language_key_mask, scale, att_dim, v_seq);
        ggml_tensor* o_proj = linear(C, w.out_v_w, w.out_v_b, o);

        ggml_tensor* gamma_v_scaled = ggml_repeat(C, w.gamma_v, o_proj);
        ggml_tensor* delta_v = ggml_mul(C, gamma_v_scaled, o_proj);

        ggml_tensor* v_out = ggml_add(C, v_norm, delta_v);

        // L attends to V. There is no visual padding, hence no mask on this branch.
        ggml_tensor* q_l = linear(C, w.l_proj_w, w.l_proj_b, l_norm);
        ggml_tensor* k_v = linear(C, w.v_proj_w, w.v_proj_b, v_norm);
        ggml_tensor* v_v = linear(C, w.values_v_w, w.values_v_b, v_norm);

        ggml_tensor* Q_l = to_heads(C, q_l, head_dim, n_heads, l_seq);
        ggml_tensor* K_v = to_heads(C, k_v, head_dim, n_heads, v_seq);
        ggml_tensor* V_v = to_heads_v(C, v_v, head_dim, n_heads, v_seq);
        ggml_tensor* o_l = attention(C, Q_l, K_v, V_v, nullptr, scale, att_dim, l_seq);
        ggml_tensor* o_proj_l = linear(C, w.out_l_w, w.out_l_b, o_l);

        ggml_tensor* gamma_l_scaled = ggml_repeat(C, w.gamma_l, o_proj_l);
        ggml_tensor* delta_l = ggml_mul(C, gamma_l_scaled, o_proj_l);

        l = ggml_add(C, l_norm, delta_l);

        return v_out;
    }

    ggml_tensor* TurboVLA::vl_text_layer(ggml_context* C, const VlTextLayerW& w, ggml_tensor* x,
                                        ggml_tensor* self_mask,
                                        int64_t seq, int64_t n_heads, int64_t head_dim) const {
        // TurboVLA's text enhancer uses post-norm residuals and ReLU.
        const float scale = 1.0f / std::sqrt((float)head_dim);
        const int64_t att_dim = n_heads * head_dim;
        const int64_t n = n_heads;

        ggml_tensor* qkv = linear(C, w.attn_qkv_w, w.attn_qkv_b, x);

        // Most checkpoints store fused [Q; K; V]. Keep compatibility with a
        // checkpoint that stores a single shared projection as well.
        ggml_tensor* q = nullptr;
        ggml_tensor* k = nullptr;
        ggml_tensor* v = nullptr;
        if (qkv->ne[0] == 3 * att_dim) {
            const size_t es = ggml_element_size(qkv);
            q = ggml_cont(C, ggml_view_2d(C, qkv, att_dim, seq, qkv->nb[1], 0));
            k = ggml_cont(C, ggml_view_2d(C, qkv, att_dim, seq, qkv->nb[1], (size_t)att_dim * es));
            v = ggml_cont(C, ggml_view_2d(C, qkv, att_dim, seq, qkv->nb[1], (size_t)2 * att_dim * es));
        } else {
            GGML_ASSERT(qkv->ne[0] == att_dim);
            q = qkv;
            k = qkv;
            v = qkv;
        }

        check_heads("VLText.Q", q, head_dim, n, seq);
        check_heads("VLText.K", k, head_dim, n, seq);
        check_heads("VLText.V", v, head_dim, n, seq);
        ggml_tensor* Q = to_heads(C, q, head_dim, n, seq);
        ggml_tensor* K = to_heads(C, k, head_dim, n, seq);
        ggml_tensor* V = to_heads_v(C, v, head_dim, n, seq);

        ggml_tensor* att = attention(C, Q, K, V, self_mask, scale, att_dim, seq);
        ggml_tensor* o = linear(C, w.attn_o_w, w.attn_o_b, att);
        ggml_tensor* h = ggml_add(C, x, o);

        h = layer_norm(C, h, w.ln1_w, w.ln1_b, 1e-5f);

        ggml_tensor* ff = linear(C, w.fc1_w, w.fc1_b, h);
        ff = ggml_relu(C, ff);
        ggml_tensor* ff_out = linear(C, w.fc2_w, w.fc2_b, ff);

        ggml_tensor* out = ggml_add(C, h, ff_out);
        return layer_norm(C, out, w.ln2_w, w.ln2_b, 1e-5f);
    }

    ggml_tensor* TurboVLA::act_decoder_layer(ggml_context* C, const ActDecoderLayerW& w,
                                            ggml_tensor* queries, ggml_tensor* memory,
                                            int64_t q_seq, int64_t m_seq,
                                            int64_t n_heads, int64_t head_dim) const {
        const float scale = 1.0f / std::sqrt((float)head_dim);
        const int64_t hidden = n_heads * head_dim;

        ggml_tensor* normed = layer_norm(C, queries, w.ln1_w, w.ln1_b, 1e-5f);
        ggml_tensor* self_qkv = linear(C, w.self_qkv_w, w.self_qkv_b, normed);

        ggml_tensor* self_q = nullptr;
        ggml_tensor* self_k = nullptr;
        ggml_tensor* self_v = nullptr;
        if (self_qkv->ne[0] == 3 * hidden) {
            const size_t self_es = ggml_element_size(self_qkv);
            self_q = ggml_cont(C, ggml_view_2d(C, self_qkv, hidden, q_seq, self_qkv->nb[1], 0));
            self_k = ggml_cont(C, ggml_view_2d(C, self_qkv, hidden, q_seq, self_qkv->nb[1], (size_t)hidden * self_es));
            self_v = ggml_cont(C, ggml_view_2d(C, self_qkv, hidden, q_seq, self_qkv->nb[1], (size_t)2 * hidden * self_es));
        } else {
            GGML_ASSERT(self_qkv->ne[0] == hidden);
            self_q = self_qkv;
            self_k = self_qkv;
            self_v = self_qkv;
        }

        check_heads("ACT.self.Q", self_q, head_dim, n_heads, q_seq);
        check_heads("ACT.self.K", self_k, head_dim, n_heads, q_seq);
        check_heads("ACT.self.V", self_v, head_dim, n_heads, q_seq);
        ggml_tensor* Q = to_heads(C, self_q, head_dim, n_heads, q_seq);
        ggml_tensor* K = to_heads(C, self_k, head_dim, n_heads, q_seq);
        ggml_tensor* V = to_heads_v(C, self_v, head_dim, n_heads, q_seq);

        ggml_tensor* att = attention(C, Q, K, V, nullptr, scale, hidden, q_seq);
        ggml_tensor* o = linear(C, w.self_out_w, w.self_out_b, att);
        ggml_tensor* h = ggml_add(C, queries, o);

        // Cross-attention. If cross_qkv is fused [Q;K;V], project both query and
        // memory and select their respective blocks. If the checkpoint stores only
        // a single query projection, preserve the original direct-memory K/V path.
        normed = layer_norm(C, h, w.ln2_w, w.ln2_b, 1e-5f);
        ggml_tensor* cross_qkv_q = linear(C, w.cross_qkv_w, w.cross_qkv_b, normed);

        ggml_tensor* cross_q = nullptr;
        ggml_tensor* cross_k = nullptr;
        ggml_tensor* cross_v = nullptr;
        if (cross_qkv_q->ne[0] == 3 * hidden) {
            ggml_tensor* cross_qkv_m = linear(C, w.cross_qkv_w, w.cross_qkv_b, memory);
            GGML_ASSERT(cross_qkv_m->ne[0] == 3 * hidden);
            const size_t cross_m_es = ggml_element_size(cross_qkv_m);
            cross_q = ggml_cont(C, ggml_view_2d(C, cross_qkv_q, hidden, q_seq, cross_qkv_q->nb[1], 0));
            cross_k = ggml_cont(C, ggml_view_2d(C, cross_qkv_m, hidden, m_seq, cross_qkv_m->nb[1], (size_t)hidden * cross_m_es));
            cross_v = ggml_cont(C, ggml_view_2d(C, cross_qkv_m, hidden, m_seq, cross_qkv_m->nb[1], (size_t)2 * hidden * cross_m_es));
        } else {
            GGML_ASSERT(cross_qkv_q->ne[0] == hidden);
            cross_q = cross_qkv_q;
            cross_k = ggml_is_contiguous(memory) ? memory : ggml_cont(C, memory);
            cross_v = cross_k;
        }
        check_heads("ACT.cross.Q", cross_q, head_dim, n_heads, q_seq);
        check_heads("ACT.cross.K", cross_k, head_dim, n_heads, m_seq);
        check_heads("ACT.cross.V", cross_v, head_dim, n_heads, m_seq);
        ggml_tensor* Q_c = to_heads(C, cross_q, head_dim, n_heads, q_seq);
        ggml_tensor* K_c = to_heads(C, cross_k, head_dim, n_heads, m_seq);
        ggml_tensor* V_c = to_heads_v(C, cross_v, head_dim, n_heads, m_seq);

        ggml_tensor* att_c = attention(C, Q_c, K_c, V_c, nullptr, scale, hidden, q_seq);
        ggml_tensor* o_c = linear(C, w.cross_out_w, w.cross_out_b, att_c);
        h = ggml_add(C, h, o_c);

        // The upstream TransformerDecoderLayer keeps PyTorch's default ReLU
        // activation. See H-EmbodVis/TurboVLA@b29ab142,
        // models/action_head.py:39-47.
        normed = layer_norm(C, h, w.ln3_w, w.ln3_b, 1e-5f);
        ggml_tensor* ffn_out = ffn_relu(C, w.fc1_w, w.fc1_b, w.fc2_w, w.fc2_b, normed);

        return ggml_add(C, h, ffn_out);
    }

    namespace {

    static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
    static const float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

    // Build the flattened patch matrix expected by a patch-embedding weight
    // flattened from [out, C, patch_h, patch_w] to [out, C*patch_h*patch_w].
    // Output is GGML matrix layout [patch_dim, n_patches], i.e. each column is one
    // normalized Cxpatchxpatch image patch.
    void normalize_imagenet_patches(const ImageView& v,
                                    int64_t side,
                                    int64_t patch_size,
                                    std::vector<float>& out) {
        const int64_t grid = side / patch_size;
        const int64_t n_patches = grid * grid;
        const int64_t patch_dim = 3 * patch_size * patch_size;
        out.assign((size_t)patch_dim * n_patches, 0.0f);

        for (int64_t py = 0; py < grid; ++py) {
            for (int64_t px_i = 0; px_i < grid; ++px_i) {
                const int64_t patch_idx = py * grid + px_i;
                for (int64_t c = 0; c < 3; ++c) {
                    for (int64_t ky = 0; ky < patch_size; ++ky) {
                        for (int64_t kx = 0; kx < patch_size; ++kx) {
                            const int64_t h = py * patch_size + ky;
                            const int64_t w = px_i * patch_size + kx;
                            float pixel;
                            if (v.format == PixelFormat::U8) {
                                pixel = ((const uint8_t*)v.data)[(h * side + w) * 3 + c] / 255.0f;
                            } else {
                                pixel = ((const float*)v.data)[(h * side + w) * 3 + c];
                            }
                            const float norm = (pixel - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                            const int64_t feature_idx =
                                (c * patch_size + ky) * patch_size + kx;
                            out[(size_t)feature_idx + (size_t)patch_idx * patch_dim] = norm;
                        }
                    }
                }
            }
        }
    }

    void build_dinov3_rope(int64_t grid, int64_t head_dim, float base,
                            std::vector<float>& cos, std::vector<float>& sin) {
        const int64_t n_patches = grid * grid;
        const int64_t axis_dim = head_dim / 4;
        cos.resize((size_t)head_dim * n_patches);
        sin.resize((size_t)head_dim * n_patches);
        for (int64_t row = 0; row < grid; ++row) {
            const float h = 2.0f * ((float(row) + 0.5f) / float(grid)) - 1.0f;
            for (int64_t col = 0; col < grid; ++col) {
                const float w = 2.0f * ((float(col) + 0.5f) / float(grid)) - 1.0f;
                const size_t offset = (size_t)(row * grid + col) * head_dim;
                for (int64_t i = 0; i < axis_dim; ++i) {
                    const float period = std::pow(base, 4.0f * float(i) / float(head_dim));
                    constexpr float two_pi = 6.28318530717958647692f;
                    const float ah = two_pi * h / period;
                    const float aw = two_pi * w / period;
                    const int64_t dims[4] = {i, axis_dim + i, 2 * axis_dim + i, 3 * axis_dim + i};
                    const float angles[4] = {ah, aw, ah, aw};
                    for (int j = 0; j < 4; ++j) {
                        cos[offset + dims[j]] = std::cos(angles[j]);
                        sin[offset + dims[j]] = std::sin(angles[j]);
                    }
                }
            }
        }
    }

    struct TurboVLATextInputs {
        std::vector<int32_t> token_ids;
        std::vector<int32_t> position_ids;
        std::vector<float> bert_self_mask;
        std::vector<float> enhancer_self_mask;
        std::vector<float> fusion_language_key_mask;
    };

    TurboVLATextInputs build_turbovla_text_inputs(const Inputs& in,
                                                   const TurboVLAConfig& cfg,
                                                   int64_t vision_seq) {
        const int64_t seq = cfg.max_text_length;
        TurboVLATextInputs result;
        result.token_ids.assign((size_t)seq, (int32_t)cfg.pad_token_id);
        result.position_ids.assign((size_t)seq, 0);

        if (in.n_lang > seq) {
            throw std::runtime_error("TurboVLA language input exceeds the checkpoint padding length");
        }
        if (in.lang_tokens && in.n_lang > 0) {
            std::memcpy(result.token_ids.data(), in.lang_tokens,
                        (size_t)in.n_lang * sizeof(int32_t));
        }

        std::vector<uint8_t> valid((size_t)seq, 0);
        for (int64_t i = 0; i < seq; ++i) {
            const bool from_mask = in.attention_mask &&
                ((in.attention_mask_n == seq) || (i < in.attention_mask_n));
            valid[(size_t)i] = from_mask ? in.attention_mask[i] != 0
                                         : result.token_ids[(size_t)i] != cfg.pad_token_id;
        }

        // Port of generate_masks_with_special_tokens from TurboVLA's BERT wrapper.
        // Its block masks separate sub-sentences at [CLS], [SEP], '.' and '?'.
        std::vector<uint8_t> allowed((size_t)seq * seq, 0);
        for (int64_t i = 0; i < seq; ++i) {
            allowed[(size_t)i * seq + i] = 1;
        }
        int64_t previous_special = 0;
        for (int64_t col = 0; col < seq; ++col) {
            const int32_t id = result.token_ids[(size_t)col];
            const bool special = id == cfg.cls_token_id || id == cfg.sep_token_id ||
                                 id == cfg.period_token_id || id == cfg.question_token_id;
            if (!special) {
                continue;
            }
            if (col == 0 || col == seq - 1) {
                result.position_ids[(size_t)col] = 0;
            } else {
                for (int64_t q = previous_special + 1; q <= col; ++q) {
                    result.position_ids[(size_t)q] = (int32_t)(q - previous_special - 1);
                    for (int64_t k = previous_special + 1; k <= col; ++k) {
                        allowed[(size_t)q * seq + k] = 1;
                    }
                }
            }
            previous_special = col;
        }

        result.bert_self_mask.resize((size_t)seq * seq);
        result.enhancer_self_mask.resize((size_t)seq * seq);
        for (int64_t q = 0; q < seq; ++q) {
            for (int64_t k = 0; k < seq; ++k) {
                const bool is_allowed = allowed[(size_t)q * seq + k] != 0;
                result.bert_self_mask[(size_t)q * seq + k] = is_allowed ? 0.0f : -FLT_MAX;
                // The BERT wrapper creates block masks from special tokens; the
                // enhancer additionally masks padded keys. See
                // H-EmbodVis/TurboVLA@b29ab142, text/bert.py:177-212 and
                // models/text_encoder.py:80-94.
                result.enhancer_self_mask[(size_t)q * seq + k] =
                    is_allowed && valid[(size_t)k] ? 0.0f : -FLT_MAX;
            }
        }

        result.fusion_language_key_mask.resize((size_t)vision_seq * seq);
        for (int64_t q = 0; q < vision_seq; ++q) {
            for (int64_t k = 0; k < seq; ++k) {
                result.fusion_language_key_mask[(size_t)q * seq + k] =
                    valid[(size_t)k] ? 0.0f : -FLT_MAX;
            }
        }
        return result;
    }

    }  // anonymous namespace

    std::vector<float> TurboVLA::predict(const Inputs& in) {
        const auto& cfg = cfg_;

        const int64_t patch_size = cfg.patch_size;
        const int64_t img_size = cfg.image_size;
        const int64_t n_patches_per_side = img_size / patch_size;
        const int64_t n_patches = n_patches_per_side * n_patches_per_side;
        const int64_t n_reg = cfg.num_register_tokens;
        const int64_t n_cls = 1;
        const int64_t internal_vit_seq = n_cls + n_reg + n_patches;
        const int64_t n_views = cfg.num_views;
        const int64_t vision_seq = n_views * n_patches;
        const int64_t hidden = cfg.hidden;
        const int64_t vit_dim = cfg.vit_dim;

        // Basic shape sanity checks.  These fail early with a useful message instead
        // of failing later inside ggml_reshape_*.
        if (patch_size <= 0 || img_size <= 0 || img_size % patch_size != 0) {
            fprintf(stderr, "ERROR: invalid image/patch configuration\n");
            return {};
        }
        if (cfg.vit_heads * cfg.vit_head_dim != vit_dim) {
            fprintf(stderr,
                    "ERROR: vit_heads * vit_head_dim = %lld, expected vit_dim = %lld\n",
                    (long long)(cfg.vit_heads * cfg.vit_head_dim),
                    (long long)vit_dim);
            return {};
        }
        if (n_views <= 0 || !in.images || in.n_images < n_views) {
            fprintf(stderr,
                    "ERROR: model expects %lld image view(s), but input has %lld\n",
                    (long long)n_views,
                    (long long)in.n_images);
            return {};
        }

        // Project only patch tokens; DINOv3's CLS and register tokens stay internal.
        const int64_t vision_out_size = n_patches * vit_dim;
        std::vector<float> vision_outputs((size_t)n_views * vision_out_size, 0.0f);

        for (int64_t view_idx = 0; view_idx < n_views; ++view_idx) {
            ggml_context* VC = scratch.reset(256 * 1024 * 1024);  // metadata arena
            if (!VC) {
                fprintf(stderr, "ERROR: vision scratch reset failed\n");
                return {};
            }

            // Patch-embedding input [patch_dim, n_patches]. Patch extraction is
            // performed on the host so that each column is exactly one spatial
            // Cxpatchxpatch block; a plain reshape of CHW image memory does not
            // create spatial patches correctly.
            const int64_t patch_dim = 3 * patch_size * patch_size;
            ggml_tensor* patch_input =
                ggml_new_tensor_2d(VC, GGML_TYPE_F32, patch_dim, n_patches);
            ggml_set_input(patch_input);

            // patch_embed_w is a flattened Conv2D patch projection. GGML linear
            // layout is [in=patch_dim, out=vit_dim], so mul_mat produces
            // [vit_dim, n_patches].
            if (vision_encoder.patch_embed_w->ne[0] != patch_dim) {
                fprintf(stderr,
                        "ERROR: patch_embed_w input dim=%lld, expected patch_dim=%lld\n",
                        (long long)vision_encoder.patch_embed_w->ne[0],
                        (long long)patch_dim);
                return {};
            }

            ggml_tensor* patches_emb =
                ggml_mul_mat(VC, vision_encoder.patch_embed_w, patch_input);
            ggml_tensor* patches_2d = ggml_cont(VC, patches_emb);

            if (patches_2d->ne[0] != vit_dim || patches_2d->ne[1] != n_patches) {
                fprintf(stderr,
                        "ERROR: patch embedding shape is [%lld,%lld], expected [%lld,%lld]\n",
                        (long long)patches_2d->ne[0],
                        (long long)patches_2d->ne[1],
                        (long long)vit_dim,
                        (long long)n_patches);
                return {};
            }

            ggml_tensor* patches_with_bias =
                ggml_add(VC, patches_2d, vision_encoder.patch_embed_b);

            ggml_tensor* rope_cos = ggml_new_tensor_2d(VC, GGML_TYPE_F32, cfg.vit_head_dim, n_patches);
            ggml_tensor* rope_sin = ggml_new_tensor_2d(VC, GGML_TYPE_F32, cfg.vit_head_dim, n_patches);
            ggml_set_input(rope_cos);
            ggml_set_input(rope_sin);
            std::vector<float> rope_cos_host, rope_sin_host;
            build_dinov3_rope(n_patches_per_side, cfg.vit_head_dim, cfg.rope_theta,
                               rope_cos_host, rope_sin_host);

            // Assemble DINOv3's internal sequence: CLS, register tokens, then patches.
            ggml_tensor* cls_2d = ggml_reshape_2d(VC, vision_encoder.cls_token, vit_dim, 1);

            ggml_tensor* vit_out = cls_2d;
            if (n_reg > 0 && vision_encoder.register_tokens) {
                ggml_tensor* reg_2d = ggml_reshape_2d(VC, vision_encoder.register_tokens, vit_dim, n_reg);
                vit_out = ggml_concat(VC, vit_out, reg_2d, 1);
            }
            vit_out = ggml_concat(VC, vit_out, patches_with_bias, 1);

            for (int64_t i = 0; i < cfg.vit_layers; ++i) {
                vit_out = vit_layer(VC,
                                    vision_encoder.layers[i],
                                    vit_out,
                                    rope_cos,
                                    rope_sin,
                                    internal_vit_seq,
                                    cfg.vit_heads,
                                    cfg.vit_head_dim);
            }
            // TurboVLA selects DINOv3's final hidden-state entry, then removes
            // its prefix tokens. See H-EmbodVis/TurboVLA@b29ab142,
            // models/vision_encoder.py:109-114.
            const int64_t patch_token_start = n_cls + n_reg;

            ggml_tensor* vit_patches_only = ggml_view_2d(VC,
                                                         vit_out,
                                                         vit_dim,
                                                         n_patches,
                                                         vit_out->nb[1],
                                                         patch_token_start * vit_out->nb[1]);
            vit_patches_only = ggml_cont(VC, vit_patches_only);

            ggml_set_output(vit_patches_only);

            ggml_cgraph* vg = ggml_new_graph_custom(VC, 8192, false);
            ggml_build_forward_expand(vg, vit_patches_only);

            if (!scratch.alloc(backend, vg)) {
                fprintf(stderr, "ERROR: vision gallocr alloc failed\n");
                return {};
            }

            if (!patch_input->buffer) {
                fprintf(stderr,
                        "ERROR: patch_input has no backend buffer after vision graph allocation; "
                        "vision graph is disconnected\n");
                return {};
            }

            std::vector<float> patch_buf;
            normalize_imagenet_patches(in.images[view_idx],
                                    img_size,
                                    patch_size,
                                    patch_buf);
            ggml_backend_tensor_set(patch_input,
                                    patch_buf.data(),
                                    0,
                                    patch_buf.size() * sizeof(float));
            ggml_backend_tensor_set(rope_cos, rope_cos_host.data(), 0,
                                    rope_cos_host.size() * sizeof(float));
            ggml_backend_tensor_set(rope_sin, rope_sin_host.data(), 0,
                                    rope_sin_host.size() * sizeof(float));

            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "ERROR: vision compute failed\n");
                return {};
            }

            float* out_ptr = vision_outputs.data() + view_idx * vision_out_size;
            ggml_backend_tensor_get(vit_patches_only,
                                    out_ptr,
                                    0,
                                    vision_out_size * sizeof(float));
        }

        ggml_context* C = scratch.reset(512 * 1024 * 1024);  // metadata arena
        if (!C) {
            fprintf(stderr, "ERROR: main scratch reset failed\n");
            return {};
        }

        // Buffers are allocated after graph construction because the context uses no_alloc.
        std::vector<ggml_tensor*> view_tokens((size_t)n_views, nullptr);
        for (int64_t view_idx = 0; view_idx < n_views; ++view_idx) {
            view_tokens[view_idx] =
                ggml_new_tensor_2d(C, GGML_TYPE_F32, vit_dim, n_patches);
            ggml_set_input(view_tokens[view_idx]);
        }

        ggml_tensor* vit_tokens = view_tokens[0];
        for (int64_t view_idx = 1; view_idx < n_views; ++view_idx) {
            vit_tokens = ggml_concat(C, vit_tokens, view_tokens[view_idx], 1);
        }

        // Vision projection: [vit_dim, vision_seq] -> [hidden, vision_seq].
        ggml_tensor* proj =
            layer_norm(C,
                    vit_tokens,
                    vision_proj.input_norm_w,
                    vision_proj.input_norm_b,
                    1e-5f);
        ggml_tensor* mlp_out =
            linear(C, vision_proj.mlp_0_w, vision_proj.mlp_0_b, proj);
        mlp_out = ggml_gelu_erf(C, mlp_out);
        mlp_out =
            linear(C, vision_proj.mlp_3_w, vision_proj.mlp_3_b, mlp_out);

        ggml_tensor* skip_out =
            ggml_mul_mat(C, vision_proj.skip_w, vit_tokens);
        proj = ggml_add(C, mlp_out, skip_out);
        proj = layer_norm(C,
                        proj,
                        vision_proj.output_norm_w,
                        vision_proj.output_norm_b,
                        1e-5f);

        // The checkpoint stores one embedding per camera. Each embedding must form
        // a contiguous patch block: [view0 x 256 patches, view1 x 256 patches].
        const int64_t view_emb_elems = ggml_nelements(view_emb);
        ggml_tensor* view_emb_used = nullptr;
        if (view_emb_elems == (int64_t)hidden * n_views) {
            ggml_tensor* by_view = ggml_reshape_2d(C, view_emb, hidden, n_views);
            for (int64_t view_idx = 0; view_idx < n_views; ++view_idx) {
                ggml_tensor* one_view = ggml_cont(C, ggml_view_2d(
                    C, by_view, hidden, 1, by_view->nb[1], view_idx * by_view->nb[1]));
                ggml_tensor* view_block = ggml_repeat(
                    C, one_view, ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, n_patches));
                view_emb_used = view_emb_used ? ggml_concat(C, view_emb_used, view_block, 1) : view_block;
            }
        } else if (view_emb_elems == (int64_t)hidden) {
            // Single shared [hidden] embedding: repeat to [hidden, vision_seq]
            view_emb_used = ggml_repeat(C, view_emb,
                                         ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, vision_seq));
        } else if (view_emb_elems == (int64_t)hidden * vision_seq) {
            view_emb_used = ggml_reshape_2d(C, view_emb, (int64_t)hidden, vision_seq);
        } else {
            fprintf(stderr,
                    "ERROR: view_emb has %lld values; expected %lld (n_views,hidden), "
                    "%lld (shared), or %lld (per-pos)\n",
                    (long long)view_emb_elems,
                    (long long)((int64_t)hidden * n_views),
                    (long long)(int64_t)hidden,
                    (long long)((int64_t)hidden * vision_seq));
            return {};
        }
        proj = ggml_add(C, proj, view_emb_used);

        const int64_t text_seq = cfg.max_text_length;
        if (text_seq <= 0) {
            fprintf(stderr, "ERROR: text_seq must be positive\n");
            return {};
        }
        TurboVLATextInputs text_inputs;
        try {
            text_inputs = build_turbovla_text_inputs(in, cfg, vision_seq);
        } catch (const std::exception& e) {
            fprintf(stderr, "ERROR: %s\n", e.what());
            return {};
        }
        std::vector<int32_t> tok_type_ids_host((size_t)text_seq, 0);

        ggml_tensor* token_ids =
            ggml_new_tensor_1d(C, GGML_TYPE_I32, text_seq);
        ggml_set_input(token_ids);

        ggml_tensor* pos_ids =
            ggml_new_tensor_1d(C, GGML_TYPE_I32, text_seq);
        ggml_set_input(pos_ids);

        ggml_tensor* tok_type_ids =
            ggml_new_tensor_1d(C, GGML_TYPE_I32, text_seq);
        ggml_set_input(tok_type_ids);

        ggml_tensor* bert_self_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, text_seq, text_seq);
        ggml_tensor* enhancer_self_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, text_seq, text_seq);
        ggml_tensor* fusion_language_key_mask =
            ggml_new_tensor_2d(C, GGML_TYPE_F32, text_seq, vision_seq);
        ggml_set_input(bert_self_mask);
        ggml_set_input(enhancer_self_mask);
        ggml_set_input(fusion_language_key_mask);

        ggml_tensor* pos_emb =
            ggml_get_rows(C, text_encoder.pos_embed, pos_ids);
        ggml_tensor* text_emb =
            ggml_get_rows(C, text_encoder.word_embed, token_ids);
        ggml_tensor* tok_type_emb =
            ggml_get_rows(C, text_encoder.tok_type_embed, tok_type_ids);

        ggml_tensor* text_tokens =
            ggml_add(C, ggml_add(C, text_emb, pos_emb), tok_type_emb);
        text_tokens = layer_norm(C,
                                text_tokens,
                                text_encoder.embed_ln_w,
                                text_encoder.embed_ln_b,
                                1e-12f);

        for (int64_t i = 0; i < cfg.text_layers; ++i) {
            text_tokens = bert_layer(C,
                                    text_encoder.layers[i],
                                    text_tokens,
                                    bert_self_mask,
                                    text_seq,
                                    cfg.text_heads,
                                    cfg.text_head_dim);
        }
        text_tokens =
            linear(C,
                text_encoder.text_proj_w,
                text_encoder.text_proj_b,
                text_tokens);
        ggml_tensor* vl_features = proj;
        ggml_tensor* text_features = text_tokens;

        const int64_t vl_heads = cfg.fusion_heads;
        const int64_t vl_head_dim = cfg.fusion_head_dim;

        const int64_t text_enh_heads = cfg.text_enhancer_heads;
        const int64_t text_enh_head_dim = cfg.text_enhancer_head_dim;

        // Each fusion layer is immediately followed by its paired text enhancer.
        GGML_ASSERT(cfg.num_fusion_layers == cfg.num_text_layers &&
                   "VL fusion and text layers must have same count for interleaved execution");

        for (int64_t i = 0; i < cfg.num_fusion_layers; ++i) {
            vl_features = vl_fusion_layer(C,
                                        fusion_layers[i],
                                        vl_features,
                                        text_features,
                                        fusion_language_key_mask,
                                        vision_seq,
                                        text_seq,
                                        vl_heads,
                                        vl_head_dim);
            text_features = vl_text_layer(C,
                                        text_layers[i],
                                        text_features,
                                        enhancer_self_mask,
                                        text_seq,
                                        text_enh_heads,
                                        text_enh_head_dim);
        }

        ggml_tensor* vl_condition =
            ggml_concat(C, vl_features, text_features, 1);
        const int64_t vl_cond_seq = vision_seq + text_seq;

        const int64_t state_dim = cfg.state_dim;

        std::vector<float> state_host((size_t)state_dim, 0.0f);
        if (in.state) {
            std::copy(in.state, in.state + state_dim, state_host.begin());
        }

        ggml_tensor* state_tensor =
            ggml_new_tensor_2d(C, GGML_TYPE_F32, state_dim, 1);
        ggml_set_input(state_tensor);

        ggml_tensor* s =
            layer_norm(C,
                    state_tensor,
                    state_proj.net_0_w,
                    state_proj.net_0_b,
                    1e-5f);
        s = linear(C, state_proj.net_1_w, state_proj.net_1_b, s);
        s = ggml_gelu_erf(C, s);
        s = linear(C, state_proj.net_4_w, state_proj.net_4_b, s);

        // state projection is expected to produce hidden * num_state_tokens values.
        if (ggml_nelements(s) != hidden * cfg.num_state_tokens) {
            fprintf(stderr,
                    "ERROR: state projection produced %lld values; expected hidden*num_state_tokens=%lld\n",
                    (long long)ggml_nelements(s),
                    (long long)(hidden * cfg.num_state_tokens));
            return {};
        }

        ggml_tensor* state_tokens =
            ggml_reshape_2d(C, s, hidden, cfg.num_state_tokens);
        ggml_tensor* pos_emb_state =
            ggml_reshape_2d(C,
                            state_proj.position,
                            hidden,
                            cfg.num_state_tokens);
        state_tokens = ggml_add(C, state_tokens, pos_emb_state);

        // The projection normalizes only after adding learned positions. See
        // H-EmbodVis/TurboVLA@b29ab142, models/action_head.py:25-31.
        state_tokens = layer_norm(C,
                                state_tokens,
                                state_proj.output_norm_w,
                                state_proj.output_norm_b,
                                1e-5f);

        // State tokens must be part of the decoder memory.
        ggml_tensor* condition =
            ggml_concat(C, vl_condition, state_tokens, 1);
        const int64_t cond_seq = vl_cond_seq + cfg.num_state_tokens;

        ggml_tensor* queries =
            ggml_reshape_2d(C,
                            action_decoder.action_q,
                            hidden,
                            cfg.action_horizon);

        const int64_t act_heads = cfg.action_heads;

        if (hidden % act_heads != 0) {
            fprintf(stderr,
                    "ERROR: hidden=%lld is not divisible by act_heads=%lld\n",
                    (long long)hidden,
                    (long long)act_heads);
            return {};
        }
        const int64_t act_head_dim = cfg.action_head_dim;

        for (int64_t i = 0; i < cfg.num_action_decoder_layers; ++i) {
            queries = act_decoder_layer(C,
                                        action_decoder.layers[i],
                                        queries,
                                        condition,
                                        cfg.action_horizon,
                                        cond_seq,
                                        act_heads,
                                        act_head_dim);
        }

        ggml_tensor* actions =
            linear(C, action_decoder.proj_0_w, action_decoder.proj_0_b, queries);
        actions = ggml_relu(C, actions);
        actions =
            linear(C, action_decoder.proj_1_w, action_decoder.proj_1_b, actions);
        actions = ggml_relu(C, actions);
        actions =
            linear(C, action_decoder.proj_2_w, action_decoder.proj_2_b, actions);
        actions = ggml_tanh(C, actions);
        ggml_set_output(actions);

        ggml_cgraph* gf = ggml_new_graph_custom(C, 16384, false);
        ggml_build_forward_expand(gf, actions);

        if (!scratch.alloc(backend, gf)) {
            fprintf(stderr, "ERROR: scratch alloc failed\n");
            return {};
        }

        // All graph inputs now have backend buffers. Copy them only at this point.
        for (int64_t view_idx = 0; view_idx < n_views; ++view_idx) {
            if (!view_tokens[view_idx]->buffer) {
                fprintf(stderr,
                        "ERROR: view_tokens[%lld] has no backend buffer after main graph allocation\n",
                        (long long)view_idx);
                return {};
            }
            const float* out_ptr =
                vision_outputs.data() + view_idx * vision_out_size;
            ggml_backend_tensor_set(view_tokens[view_idx],
                                    out_ptr,
                                    0,
                                    vision_out_size * sizeof(float));
        }

        if (!token_ids->buffer || !pos_ids->buffer || !tok_type_ids->buffer ||
            !bert_self_mask->buffer || !enhancer_self_mask->buffer ||
            !fusion_language_key_mask->buffer || !state_tensor->buffer) {
            fprintf(stderr,
                    "ERROR: one or more main graph inputs have no backend buffer "
                    "(token=%p pos=%p type=%p bert=%p enhancer=%p fusion=%p state=%p)\n",
                    (void*)token_ids->buffer,
                    (void*)pos_ids->buffer,
                    (void*)tok_type_ids->buffer,
                    (void*)bert_self_mask->buffer,
                    (void*)enhancer_self_mask->buffer,
                    (void*)fusion_language_key_mask->buffer,
                    (void*)state_tensor->buffer);
            return {};
        }

        ggml_backend_tensor_set(token_ids,
                                text_inputs.token_ids.data(),
                                0,
                                text_inputs.token_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pos_ids,
                                text_inputs.position_ids.data(),
                                0,
                                text_inputs.position_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(tok_type_ids,
                                tok_type_ids_host.data(),
                                0,
                                tok_type_ids_host.size() * sizeof(int32_t));
        ggml_backend_tensor_set(bert_self_mask,
                                text_inputs.bert_self_mask.data(),
                                0,
                                text_inputs.bert_self_mask.size() * sizeof(float));
        ggml_backend_tensor_set(enhancer_self_mask,
                                text_inputs.enhancer_self_mask.data(),
                                0,
                                text_inputs.enhancer_self_mask.size() * sizeof(float));
        ggml_backend_tensor_set(fusion_language_key_mask,
                                text_inputs.fusion_language_key_mask.data(),
                                0,
                                text_inputs.fusion_language_key_mask.size() * sizeof(float));
        ggml_backend_tensor_set(state_tensor,
                                state_host.data(),
                                0,
                                state_host.size() * sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: compute failed\n");
            return {};
        }

        const int64_t expected_action_values =
            cfg.action_horizon * cfg.action_dim;
        if (ggml_nelements(actions) != expected_action_values) {
            fprintf(stderr,
                    "ERROR: actions has %lld values, expected %lld (= horizon %lld * dim %lld)\n",
                    (long long)ggml_nelements(actions),
                    (long long)expected_action_values,
                    (long long)cfg.action_horizon,
                    (long long)cfg.action_dim);
            return {};
        }

        std::vector<float> output((size_t)expected_action_values, 0.0f);
        ggml_backend_tensor_get(actions,
                                output.data(),
                                0,
                                output.size() * sizeof(float));

        return output;
    }

    std::unique_ptr<ModelArchBase> turbovla_create(const std::string& /*mmproj_path*/,
                                                const std::string& ckpt_path,
                                                const std::string& /*config_path*/,
                                                const Options& /*opts*/) {
        gguf_reader g("turbovla");
        if (!g.open(ckpt_path)) {
            std::fprintf(stderr, "vla(turbovla): failed to open %s\n", ckpt_path.c_str());
            return nullptr;
        }

        auto* m = new vla::TurboVLA;

        m->cfg_.hidden = (int64_t)g.u32("turbovla.hidden");
        m->cfg_.vit_dim = (int64_t)g.u32("turbovla.vit_dim");
        m->cfg_.vit_layers = (int64_t)g.u32("turbovla.vit_layers");
        m->cfg_.vit_head_dim = (int64_t)g.u32("turbovla.vit_head_dim");
        m->cfg_.vit_heads = (int64_t)g.u32("turbovla.vit_heads");
        m->cfg_.text_dim = (int64_t)g.u32("turbovla.text_dim");
        m->cfg_.text_layers = (int64_t)g.u32("turbovla.text_layers");
        m->cfg_.text_head_dim = (int64_t)g.u32("turbovla.text_head_dim");
        m->cfg_.text_heads = (int64_t)g.u32("turbovla.text_heads");
        m->cfg_.num_fusion_layers = (int64_t)g.u32("turbovla.num_fusion_layers");
        m->cfg_.fusion_heads = (int64_t)g.u32("turbovla.fusion_heads");
        m->cfg_.fusion_head_dim = (int64_t)g.u32("turbovla.fusion_head_dim");
        m->cfg_.text_enhancer_heads = (int64_t)g.u32("turbovla.text_enhancer_heads");
        m->cfg_.text_enhancer_head_dim = (int64_t)g.u32("turbovla.text_enhancer_head_dim");
        m->cfg_.action_heads = (int64_t)g.u32("turbovla.action_heads");
        m->cfg_.action_head_dim = (int64_t)g.u32("turbovla.action_head_dim");
        m->cfg_.num_text_layers = (int64_t)g.u32("turbovla.num_text_layers");
        m->cfg_.num_action_decoder_layers = (int64_t)g.u32("turbovla.num_action_decoder_layers");
        m->cfg_.action_dim = (int64_t)g.u32("turbovla.action_dim");
        m->cfg_.state_dim = (int64_t)g.u32("turbovla.state_dim");
        m->cfg_.num_state_tokens = (int64_t)g.u32("turbovla.num_state_tokens");
        m->cfg_.action_horizon = (int64_t)g.u32("turbovla.action_horizon");
        m->cfg_.image_size = (int64_t)g.u32("turbovla.image_size");
        m->cfg_.patch_size = (int64_t)g.u32("turbovla.patch_size");
        m->cfg_.num_views = (int64_t)g.u32("turbovla.num_views");
        m->cfg_.num_register_tokens = (int64_t)g.u32("turbovla.num_register_tokens");
        if (!g.has("turbovla.rope_theta")) {
            std::fprintf(stderr, "vla(turbovla): GGUF is missing required DINOv3 rope_theta metadata\n");
            return nullptr;
        }
        m->cfg_.rope_theta = g.f32("turbovla.rope_theta");
        m->cfg_.max_text_length = (int64_t)g.u32("turbovla.max_text_length");
        if (!g.has("turbovla.pad_token_id") || !g.has("turbovla.cls_token_id") ||
            !g.has("turbovla.sep_token_id") || !g.has("turbovla.period_token_id") ||
            !g.has("turbovla.question_token_id")) {
            std::fprintf(stderr, "vla(turbovla): GGUF is missing required BERT special-token metadata\n");
            return nullptr;
        }
        m->cfg_.pad_token_id = (int64_t)g.u32("turbovla.pad_token_id");
        m->cfg_.cls_token_id = (int64_t)g.u32("turbovla.cls_token_id");
        m->cfg_.sep_token_id = (int64_t)g.u32("turbovla.sep_token_id");
        m->cfg_.period_token_id = (int64_t)g.u32("turbovla.period_token_id");
        m->cfg_.question_token_id = (int64_t)g.u32("turbovla.question_token_id");
        if (m->cfg_.fusion_heads * m->cfg_.fusion_head_dim != 1024 ||
            m->cfg_.text_enhancer_heads * m->cfg_.text_enhancer_head_dim != m->cfg_.hidden ||
            m->cfg_.action_heads * m->cfg_.action_head_dim != m->cfg_.hidden) {
            std::fprintf(stderr, "vla(turbovla): incompatible attention metadata\n");
            return nullptr;
        }

        // Publish the dimensions used by serving tools. Without this, callers
        // allocate an empty state vector from ModelArchBase::cfg while predict()
        // reads the checkpoint's state_dim values.
        const int64_t patches_per_side = m->cfg_.image_size / m->cfg_.patch_size;
        m->cfg.n_img = m->cfg_.num_views * patches_per_side * patches_per_side;
        m->cfg.n_lang = m->cfg_.max_text_length;
        m->cfg.n_state = m->cfg_.num_state_tokens;
        m->cfg.n_prefix = m->cfg.n_img + m->cfg.n_lang + m->cfg.n_state;
        m->cfg.n_suffix = m->cfg_.action_horizon;
        m->cfg.n_full = m->cfg.n_prefix + m->cfg.n_suffix;
        m->cfg.hidden = m->cfg_.hidden;
        m->cfg.n_q_heads = m->cfg_.action_heads;
        m->cfg.n_kv_heads = m->cfg_.action_heads;
        m->cfg.head_dim = m->cfg_.action_head_dim;
        m->cfg.q_full_dim = m->cfg_.hidden;
        m->cfg.kv_full_dim = m->cfg_.hidden;
        m->cfg.n_layers = m->cfg_.num_action_decoder_layers;
        m->cfg.max_state_dim = m->cfg_.state_dim;
        m->cfg.real_state_dim = m->cfg_.state_dim;
        m->cfg.max_action_dim = m->cfg_.action_dim;
        m->cfg.real_action_dim = m->cfg_.action_dim;
        m->cfg.norm_eps = 1e-5f;
        m->cfg.rms_eps = 1e-5f;
        m->cfg.rope_n_dims = m->cfg_.vit_head_dim;
        m->cfg.rope_freq_base = m->cfg_.rope_theta;
        m->cfg.num_steps = m->cfg_.action_horizon;

        const Backend b = backend_init("vla(turbovla)", m->n_threads);
        if (!b.handle) {
            std::fprintf(stderr, "vla(turbovla): backend_init failed\n");
            return nullptr;
        }
        m->backend = b.handle;

        ggml_init_params wp = { (size_t)16*1024*1024, nullptr, true };
        m->ctx_weights = ggml_init(wp);

        WeightLoader L("turbovla", g, m->ctx_weights, GGML_TYPE_F32);

        m->vision_encoder.cls_token = L.f32("vit.cls_token");
        m->vision_encoder.patch_embed_w = L.gemm("vit.patch_embed.weight");
        m->vision_encoder.patch_embed_b = L.gemm("vit.patch_embed.bias");

        // DINOv3 register tokens: load only when num_register_tokens > 0
        if (m->cfg_.num_register_tokens > 0) {
            m->vision_encoder.register_tokens = L.f32("vit.register_tokens");
            if (!m->vision_encoder.register_tokens) {
                std::fprintf(stderr, "vla(turbovla): WARNING: num_register_tokens=%lld but vit.register_tokens not found\n",
                            (long long)m->cfg_.num_register_tokens);
            }
        }

        m->vision_encoder.layers.resize(m->cfg_.vit_layers);
        for (int64_t i = 0; i < m->cfg_.vit_layers; ++i) {
            auto& w = m->vision_encoder.layers[i];
            w.ln1_w = L.f32("vit.blk.%lld.ln1.weight", (long long) i);
            w.ln1_b = L.f32("vit.blk.%lld.ln1.bias", (long long) i);
            w.attn_q_w = L.f32("vit.blk.%lld.attn_q.weight", (long long) i);
            w.attn_q_b = L.f32("vit.blk.%lld.attn_q.bias", (long long) i);
            w.attn_k_w = L.f32("vit.blk.%lld.attn_k.weight", (long long) i);
            w.attn_k_b = L.f32("vit.blk.%lld.attn_k.bias", (long long) i);
            w.attn_v_w = L.f32("vit.blk.%lld.attn_v.weight", (long long) i);
            w.attn_v_b = L.f32("vit.blk.%lld.attn_v.bias", (long long) i);
            w.attn_o_w = L.f32("vit.blk.%lld.attn_o.weight", (long long) i);
            w.attn_o_b = L.f32("vit.blk.%lld.attn_o.bias", (long long) i);
            w.ln2_w = L.f32("vit.blk.%lld.ln2.weight", (long long) i);
            w.ln2_b = L.f32("vit.blk.%lld.ln2.bias", (long long) i);
            w.fc1_w = L.f32("vit.blk.%lld.fc1.weight", (long long) i);
            w.fc1_b = L.f32("vit.blk.%lld.fc1.bias", (long long) i);
            w.fc2_w = L.f32("vit.blk.%lld.fc2.weight", (long long) i);
            w.fc2_b = L.f32("vit.blk.%lld.fc2.bias", (long long) i);
        }
        m->vision_encoder.final_ln_w = L.f32("vit.final_norm.weight");
        m->vision_encoder.final_ln_b = L.f32("vit.final_norm.bias");


        m->vision_proj.input_norm_w = L.f32("vit_proj.input_norm.weight");
        m->vision_proj.input_norm_b = L.f32("vit_proj.input_norm.bias");
        m->vision_proj.mlp_0_w = L.f32("vit_proj.mlp.0.weight");
        m->vision_proj.mlp_0_b = L.f32("vit_proj.mlp.0.bias");
        m->vision_proj.mlp_3_w = L.f32("vit_proj.mlp.3.weight");
        m->vision_proj.mlp_3_b = L.f32("vit_proj.mlp.3.bias");
        m->vision_proj.skip_w = L.f32("vit_proj.skip.weight");
        m->vision_proj.output_norm_w = L.f32("vit_proj.output_norm.weight");
        m->vision_proj.output_norm_b = L.f32("vit_proj.output_norm.bias");

        m->text_encoder.word_embed = L.f32("text.embed.word_embeddings");
        m->text_encoder.pos_embed = L.f32("text.embed.position_embeddings");
        m->text_encoder.tok_type_embed = L.f32("text.embed.token_type_embeddings");
        m->text_encoder.embed_ln_w = L.f32("text.embed.LayerNorm.weight");
        m->text_encoder.embed_ln_b = L.f32("text.embed.LayerNorm.bias");
        m->text_encoder.pooler_w = L.f32("text.pooler.dense.weight");
        m->text_encoder.pooler_b = L.f32("text.pooler.dense.bias");
        m->text_encoder.text_proj_w = L.f32("text_proj.weight");
        m->text_encoder.text_proj_b = L.f32("text_proj.bias");

        m->text_encoder.layers.resize(m->cfg_.text_layers);
        for (int64_t i = 0; i < m->cfg_.text_layers; ++i) {
            auto& w = m->text_encoder.layers[i];
            w.attn_q_w = L.f32("text.encoder.layer.%lld.attention.self.query.weight", (long long) i);
            w.attn_q_b = L.f32("text.encoder.layer.%lld.attention.self.query.bias", (long long) i);
            w.attn_k_w = L.f32("text.encoder.layer.%lld.attention.self.key.weight", (long long) i);
            w.attn_k_b = L.f32("text.encoder.layer.%lld.attention.self.key.bias", (long long) i);
            w.attn_v_w = L.f32("text.encoder.layer.%lld.attention.self.value.weight", (long long) i);
            w.attn_v_b = L.f32("text.encoder.layer.%lld.attention.self.value.bias", (long long) i);
            w.attn_o_w = L.f32("text.encoder.layer.%lld.attention.output.dense.weight", (long long) i);
            w.attn_o_b = L.f32("text.encoder.layer.%lld.attention.output.dense.bias", (long long) i);
            w.ln1_w = L.f32("text.encoder.layer.%lld.attention.output.LayerNorm.weight", (long long) i);
            w.ln1_b = L.f32("text.encoder.layer.%lld.attention.output.LayerNorm.bias", (long long) i);
            w.fc1_w = L.f32("text.encoder.layer.%lld.intermediate.dense.weight", (long long) i);
            w.fc1_b = L.f32("text.encoder.layer.%lld.intermediate.dense.bias", (long long) i);
            w.fc2_w = L.f32("text.encoder.layer.%lld.output.dense.weight", (long long) i);
            w.fc2_b = L.f32("text.encoder.layer.%lld.output.dense.bias", (long long) i);
            w.ln2_w = L.f32("text.encoder.layer.%lld.output.LayerNorm.weight", (long long) i);
            w.ln2_b = L.f32("text.encoder.layer.%lld.output.LayerNorm.bias", (long long) i);
        }

        m->fusion_layers.resize(m->cfg_.num_fusion_layers);
        for (int64_t i = 0; i < m->cfg_.num_fusion_layers; ++i) {
            auto& w = m->fusion_layers[i];
            w.v_proj_w = L.f32("vl_fusion.%lld.v_proj.weight", (long long) i);
            w.v_proj_b = L.f32("vl_fusion.%lld.v_proj.bias", (long long) i);
            w.l_proj_w = L.f32("vl_fusion.%lld.l_proj.weight", (long long) i);
            w.l_proj_b = L.f32("vl_fusion.%lld.l_proj.bias", (long long) i);
            w.values_v_w = L.f32("vl_fusion.%lld.values_v.weight", (long long) i);
            w.values_v_b = L.f32("vl_fusion.%lld.values_v.bias", (long long) i);
            w.values_l_w = L.f32("vl_fusion.%lld.values_l.weight", (long long) i);
            w.values_l_b = L.f32("vl_fusion.%lld.values_l.bias", (long long) i);
            w.out_v_w = L.f32("vl_fusion.%lld.out_v.weight", (long long) i);
            w.out_v_b = L.f32("vl_fusion.%lld.out_v.bias", (long long) i);
            w.out_l_w = L.f32("vl_fusion.%lld.out_l.weight", (long long) i);
            w.out_l_b = L.f32("vl_fusion.%lld.out_l.bias", (long long) i);
            w.norm_v_w = L.f32("vl_fusion.%lld.norm_v.weight", (long long) i);
            w.norm_v_b = L.f32("vl_fusion.%lld.norm_v.bias", (long long) i);
            w.norm_l_w = L.f32("vl_fusion.%lld.norm_l.weight", (long long) i);
            w.norm_l_b = L.f32("vl_fusion.%lld.norm_l.bias", (long long) i);
            w.gamma_v = L.f32("vl_fusion.%lld.gamma_v", (long long) i);
            w.gamma_l = L.f32("vl_fusion.%lld.gamma_l", (long long) i);
        }

        m->text_layers.resize(m->cfg_.num_text_layers);
        for (int64_t i = 0; i < m->cfg_.num_text_layers; ++i) {
            auto& w = m->text_layers[i];
            w.attn_qkv_w = L.f32("vl_text.%lld.attn_qkv.weight", (long long) i);
            w.attn_qkv_b = L.f32("vl_text.%lld.attn_qkv.bias", (long long) i);
            w.attn_o_w = L.f32("vl_text.%lld.attn_o.weight", (long long) i);
            w.attn_o_b = L.f32("vl_text.%lld.attn_o.bias", (long long) i);
            w.ln1_w = L.f32("vl_text.%lld.ln1.weight", (long long) i);
            w.ln1_b = L.f32("vl_text.%lld.ln1.bias", (long long) i);
            w.fc1_w = L.f32("vl_text.%lld.fc1.weight", (long long) i);
            w.fc1_b = L.f32("vl_text.%lld.fc1.bias", (long long) i);
            w.fc2_w = L.f32("vl_text.%lld.fc2.weight", (long long) i);
            w.fc2_b = L.f32("vl_text.%lld.fc2.bias", (long long) i);
            w.ln2_w = L.f32("vl_text.%lld.ln2.weight", (long long) i);
            w.ln2_b = L.f32("vl_text.%lld.ln2.bias", (long long) i);
        }

        m->action_decoder.action_q = L.f32("act.q.weight");
        m->action_decoder.layers.resize(m->cfg_.num_action_decoder_layers);
        for (int64_t i = 0; i < m->cfg_.num_action_decoder_layers; ++i) {
            auto& w = m->action_decoder.layers[i];
            w.self_qkv_w = L.f32("act.dec.%lld.self_qkv.weight", (long long) i);
            w.self_qkv_b = L.f32("act.dec.%lld.self_qkv.bias", (long long) i);
            w.self_out_w = L.f32("act.dec.%lld.self_out.weight", (long long) i);
            w.self_out_b = L.f32("act.dec.%lld.self_out.bias", (long long) i);
            w.cross_qkv_w = L.f32("act.dec.%lld.cross_qkv.weight", (long long) i);
            w.cross_qkv_b = L.f32("act.dec.%lld.cross_qkv.bias", (long long) i);
            w.cross_out_w = L.f32("act.dec.%lld.cross_out.weight", (long long) i);
            w.cross_out_b = L.f32("act.dec.%lld.cross_out.bias", (long long) i);
            w.ln1_w = L.f32("act.dec.%lld.ln1.weight", (long long) i);
            w.ln1_b = L.f32("act.dec.%lld.ln1.bias", (long long) i);
            w.ln2_w = L.f32("act.dec.%lld.ln2.weight", (long long) i);
            w.ln2_b = L.f32("act.dec.%lld.ln2.bias", (long long) i);
            w.ln3_w = L.f32("act.dec.%lld.ln3.weight", (long long) i);
            w.ln3_b = L.f32("act.dec.%lld.ln3.bias", (long long) i);
            w.fc1_w = L.f32("act.dec.%lld.fc1.weight", (long long) i);
            w.fc1_b = L.f32("act.dec.%lld.fc1.bias", (long long) i);
            w.fc2_w = L.f32("act.dec.%lld.fc2.weight", (long long) i);
            w.fc2_b = L.f32("act.dec.%lld.fc2.bias", (long long) i);
        }
        m->action_decoder.proj_0_w = L.f32("act.proj.0.weight");
        m->action_decoder.proj_0_b = L.f32("act.proj.0.bias");
        m->action_decoder.proj_1_w = L.f32("act.proj.1.weight");
        m->action_decoder.proj_1_b = L.f32("act.proj.1.bias");
        m->action_decoder.proj_2_w = L.f32("act.proj.2.weight");
        m->action_decoder.proj_2_b = L.f32("act.proj.2.bias");

        m->state_proj.net_0_w = L.f32("state.proj.0.weight");
        m->state_proj.net_0_b = L.f32("state.proj.0.bias");
        m->state_proj.net_1_w = L.f32("state.proj.1.weight");
        m->state_proj.net_1_b = L.f32("state.proj.1.bias");
        m->state_proj.net_4_w = L.f32("state.proj.4.weight");
        m->state_proj.net_4_b = L.f32("state.proj.4.bias");
        m->state_proj.output_norm_w = L.f32("state.proj.output_norm.weight");
        m->state_proj.output_norm_b = L.f32("state.proj.output_norm.bias");
        m->state_proj.position = L.f32("state.proj.position");

        m->view_emb = L.f32("view_emb");

        if (!L.upload(m->backend, &m->weight_buf)) {
            std::fprintf(stderr, "vla(turbovla): weight upload failed\n");
            return nullptr;
        }

        std::printf("vla(turbovla): hidden=%lld vit_layers=%lld text_layers=%lld fusion=%lld action_horizon=%lld action_dim=%lld\n",
                    (long long)m->cfg_.hidden,
                    (long long)m->cfg_.vit_layers,
                    (long long)m->cfg_.text_layers,
                    (long long)m->cfg_.num_fusion_layers,
                    (long long)m->cfg_.action_horizon,
                    (long long)m->cfg_.action_dim);
        std::printf("vla(turbovla): DINOv3 config - image_size=%lld patch_size=%lld num_register_tokens=%lld rope_theta=%.1f\n",
                    (long long)m->cfg_.image_size,
                    (long long)m->cfg_.patch_size,
                    (long long)m->cfg_.num_register_tokens,
                    (double)m->cfg_.rope_theta);

        return std::unique_ptr<vla::ModelArchBase>(m);
    }

    }  // namespace vla
