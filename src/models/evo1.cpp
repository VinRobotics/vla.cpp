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

#include "arch.h"
#include "model.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include "gguf.h"
#include "models/gguf_reader.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace vla {
namespace {


struct Qwen2LayerW {
    ggml_tensor *attn_norm, *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *ffn_norm, *Wgate, *Wup, *Wdown;
};
struct DitLayerW {
    ggml_tensor *n1w, *n1b, *n2w, *n2b, *Win, *bin, *Wo, *bo, *f1w, *f1b, *f2w, *f2b;
};
struct ViTLayerW {
    ggml_tensor *n1w, *n1b, *n2w, *n2b, *ls1, *ls2, *Wqkv, *bqkv, *Wproj, *bproj, *Wfc1, *bfc1, *Wfc2, *bfc2;
};

}

struct Evo1ModelArch : public ModelArchBase {
    Evo1ModelArch() : ModelArchBase(Arch::EVO1) {}
    ~Evo1ModelArch() override;

    std::string           gguf_path;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    bool                  is_gpu      = false;
    int                   n_threads   = 4;
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    int64_t lm_hidden=896, lm_layers=14, n_q=14, n_kv=2, lm_head_dim=64, lm_inter=4864;
    int64_t embed_dim=896, dit_layers=8, dit_heads=8, mlp_head_hidden=1024;
    int64_t horizon=50, per_a=24, action_dim=1200, num_steps=32;
    int64_t num_image_token=256, n_images=3, vocab=151674, max_text_length=1024;
    int64_t img_ctx_id=151667, img_start_id=151665, img_end_id=151666, pad_token_id=151643;
    int64_t real_state_dim=8, real_action_dim=7;
    int64_t vit_hidden=1024, vit_layers=24, vit_heads=16, vit_inter=4096, image_size=448, patch_size=14;
    float   lm_rms_eps=1e-6f, proj_ln_eps=1e-5f, norm_eps_denom=1e-8f, vit_ln_eps=1e-6f;
    float   lm_rope_base=1000000.0f;
    bool    have_vision = false;

    ggml_tensor * lm_output_norm = nullptr;
    std::vector<Qwen2LayerW> lm;

    ggml_tensor *vit_patch_w=nullptr,*vit_patch_b=nullptr,*vit_cls=nullptr,*vit_pos=nullptr;
    std::vector<ViTLayerW> vit;
    ggml_tensor *mm_ln_w=nullptr,*mm_ln_b=nullptr,*mm_W1=nullptr,*mm_b1=nullptr,*mm_W2=nullptr,*mm_b2=nullptr;

    ggml_tensor *ae_W1=nullptr,*ae_b1=nullptr,*ae_W2=nullptr,*ae_b2=nullptr,*ae_W3=nullptr,*ae_b3=nullptr,*ae_pos=nullptr;
    std::vector<DitLayerW> dit;
    ggml_tensor *norm_out_w=nullptr,*norm_out_b=nullptr,*seq_pool_w=nullptr,*seq_pool_b=nullptr;
    ggml_tensor *head_W1=nullptr,*head_b1=nullptr,*head_W2=nullptr,*head_b2=nullptr,*time_pos=nullptr;
    ggml_tensor *state_W1=nullptr,*state_b1=nullptr,*state_W2=nullptr,*state_b2=nullptr;

    std::vector<float> state_min, state_max, action_min, action_max;

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

ggml_tensor * build_qwen2_layer(ggml_context * C, const Evo1ModelArch & m, const Qwen2LayerW & w,
                                ggml_tensor * h, ggml_tensor * positions, ggml_tensor * mask, int64_t seq,
                                ggml_tensor * qmask = nullptr) {
    const int64_t hd = m.lm_head_dim, n_q = m.n_q, n_kv = m.n_kv, hq = n_q * hd;
    const float scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * h_n1 = ggml_mul(C, ggml_rms_norm(C, h, m.lm_rms_eps), w.attn_norm);
    ggml_tensor * qp = ggml_add(C, ggml_mul_mat(C, w.Wq, h_n1), w.bq);
    ggml_tensor * kp = ggml_add(C, ggml_mul_mat(C, w.Wk, h_n1), w.bk);
    ggml_tensor * vp = ggml_add(C, ggml_mul_mat(C, w.Wv, h_n1), w.bv);
    ggml_tensor * q_rope = ggml_rope_ext(C, ggml_reshape_3d(C, qp, hd, n_q,  seq), positions, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, m.lm_rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * k_rope = ggml_rope_ext(C, ggml_reshape_3d(C, kp, hd, n_kv, seq), positions, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, m.lm_rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, k_rope, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, vp, hd, n_kv, seq), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, mask, scale, 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(C, V, aw);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, kqv, 0, 2, 1, 3)), hq, seq);
    ggml_tensor * attn_out = ggml_mul_mat(C, w.Wo, att);
    if (qmask) attn_out = ggml_mul(C, attn_out, qmask);
    ggml_tensor * h_attn = ggml_add(C, h, attn_out);
    ggml_tensor * h_n2 = ggml_mul(C, ggml_rms_norm(C, h_attn, m.lm_rms_eps), w.ffn_norm);
    ggml_tensor * gate = ggml_silu(C, ggml_mul_mat(C, w.Wgate, h_n2));
    ggml_tensor * up   = ggml_mul_mat(C, w.Wup, h_n2);
    return ggml_add(C, h_attn, ggml_mul_mat(C, w.Wdown, ggml_mul(C, gate, up)));
}

bool preprocess_image_chw(const ImageView & v, int64_t side, std::vector<float> & out) {
    static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static const float STD [3] = {0.229f, 0.224f, 0.225f};
    if (v.w != (int) side || v.h != (int) side || !v.data) {
        std::fprintf(stderr, "vla(evo1): image view is %dx%d, expected %lldx%lld\n", v.w, v.h, (long long) side, (long long) side);
        return false;
    }
    out.assign((size_t) 3 * side * side, 0.0f);
    for (int64_t h = 0; h < side; ++h)
        for (int64_t w = 0; w < side; ++w)
            for (int64_t c = 0; c < 3; ++c) {
                float px;
                if (v.format == PixelFormat::U8) {
                    px = ((const uint8_t *) v.data)[(h * side + w) * 3 + c] / 255.0f;
                } else {
                    px = ((const float *) v.data)[(h * side + w) * 3 + c];
                }
                out[c * side * side + h * side + w] = (px - MEAN[c]) / STD[c];
            }
    return true;
}

ggml_tensor * build_internvit_layer(ggml_context * C, const Evo1ModelArch & m, const ViTLayerW & w,
                                    ggml_tensor * x, int64_t N) {
    const int64_t H = m.vit_hidden, n_heads = m.vit_heads, hd = H / n_heads;
    const float scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * x_n1 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, m.vit_ln_eps), w.n1w), w.n1b);
    ggml_tensor * qkv = ggml_add(C, ggml_mul_mat(C, w.Wqkv, x_n1), w.bqkv);
    ggml_tensor * q = ggml_cont(C, ggml_view_2d(C, qkv, H, N, qkv->nb[1], 0 * H * ggml_element_size(qkv)));
    ggml_tensor * k = ggml_cont(C, ggml_view_2d(C, qkv, H, N, qkv->nb[1], 1 * H * ggml_element_size(qkv)));
    ggml_tensor * v = ggml_cont(C, ggml_view_2d(C, qkv, H, N, qkv->nb[1], 2 * H * ggml_element_size(qkv)));
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, hd, n_heads, N), 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, hd, n_heads, N), 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, hd, n_heads, N), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(C, V, aw);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, kqv, 0, 2, 1, 3)), H, N);
    ggml_tensor * attn_out = ggml_add(C, ggml_mul_mat(C, w.Wproj, att), w.bproj);
    ggml_tensor * x1 = ggml_add(C, x, ggml_mul(C, attn_out, w.ls1));
    ggml_tensor * x_n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, x1, m.vit_ln_eps), w.n2w), w.n2b);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wfc1, x_n2), w.bfc1);
    ff = ggml_gelu_erf(C, ff);
    ff = ggml_add(C, ggml_mul_mat(C, w.Wfc2, ff), w.bfc2);
    return ggml_add(C, x1, ggml_mul(C, ff, w.ls2));
}

ggml_tensor * build_internvit_view(ggml_context * C, const Evo1ModelArch & m, ggml_tensor * pixels) {
    const int64_t H = m.vit_hidden, grid = m.image_size / m.patch_size, n_patches = grid * grid, n_tok = n_patches + 1;
    const int64_t shuf_c = H * 4, sgrid = grid / 2;

    ggml_tensor * conv = ggml_conv_2d(C, m.vit_patch_w, pixels, (int) m.patch_size, (int) m.patch_size, 0, 0, 1, 1);
    ggml_tensor * patches = ggml_add(C, ggml_cont(C, ggml_transpose(C, ggml_reshape_2d(C, conv, n_patches, H))), m.vit_patch_b);
    ggml_tensor * cls2d = ggml_reshape_2d(C, m.vit_cls, H, 1);
    ggml_tensor * x = ggml_add(C, ggml_concat(C, cls2d, patches, 1), m.vit_pos);

    for (int64_t i = 0; i < m.vit_layers; ++i) x = build_internvit_layer(C, m, m.vit[i], x, n_tok);

    ggml_tensor * pnc = ggml_cont(C, ggml_view_2d(C, x, H, n_patches, x->nb[1], x->nb[1]));
    ggml_tensor * s1 = ggml_reshape_3d(C, pnc, 2 * H, sgrid, grid);
    ggml_tensor * s2 = ggml_cont(C, ggml_permute(C, s1, 0, 2, 1, 3));
    ggml_tensor * s3 = ggml_reshape_3d(C, s2, shuf_c, sgrid, sgrid);
    ggml_tensor * s4 = ggml_cont(C, ggml_permute(C, s3, 0, 2, 1, 3));
    ggml_tensor * shuf = ggml_reshape_2d(C, s4, shuf_c, sgrid * sgrid);
    ggml_tensor * x_ln = ggml_add(C, ggml_mul(C, ggml_norm(C, shuf, m.proj_ln_eps), m.mm_ln_w), m.mm_ln_b);
    ggml_tensor * hh = ggml_add(C, ggml_mul_mat(C, m.mm_W1, x_ln), m.mm_b1);
    hh = ggml_gelu_erf(C, hh);
    return ggml_add(C, ggml_mul_mat(C, m.mm_W2, hh), m.mm_b2);
}

ggml_tensor * inproj_split_w(ggml_context * C, ggml_tensor * Win, int64_t E, int64_t k) {
    return ggml_cont(C, ggml_view_2d(C, Win, E, E, Win->nb[1], (size_t) k * E * E * ggml_element_size(Win)));
}
ggml_tensor * inproj_split_b(ggml_context * C, ggml_tensor * bin, int64_t E, int64_t k) {
    return ggml_cont(C, ggml_view_1d(C, bin, E, (size_t) k * E * ggml_element_size(bin)));
}

bool load_config(const gguf_reader & g, Evo1ModelArch & m, Config & cfg) {
    auto u = [&](const char * k, int64_t & dst) { if (g.has((std::string("evo1.") + k).c_str())) dst = g.u32((std::string("evo1.") + k).c_str()); };
    u("lm_hidden", m.lm_hidden); u("lm_layers_used", m.lm_layers); u("lm_q_heads", m.n_q); u("lm_kv_heads", m.n_kv);
    u("lm_head_dim", m.lm_head_dim); u("lm_inter", m.lm_inter); u("embed_dim", m.embed_dim); u("dit_layers", m.dit_layers);
    u("dit_heads", m.dit_heads); u("mlp_head_hidden", m.mlp_head_hidden); u("horizon", m.horizon); u("per_action_dim", m.per_a);
    u("action_dim", m.action_dim); u("num_inference_timesteps", m.num_steps); u("num_image_token", m.num_image_token);
    u("n_images", m.n_images); u("vocab_size", m.vocab); u("max_text_length", m.max_text_length);
    u("img_context_token_id", m.img_ctx_id); u("img_start_token_id", m.img_start_id); u("img_end_token_id", m.img_end_id);
    u("pad_token_id", m.pad_token_id);
    u("real_state_dim", m.real_state_dim); u("real_action_dim", m.real_action_dim);
    u("vit_hidden", m.vit_hidden); u("vit_layers", m.vit_layers); u("vit_heads", m.vit_heads);
    u("vit_inter", m.vit_inter); u("image_size", m.image_size); u("patch_size", m.patch_size);
    if (g.has("evo1.lm_rms_eps"))  m.lm_rms_eps     = g.f32("evo1.lm_rms_eps");
    if (g.has("evo1.proj_ln_eps")) m.proj_ln_eps    = g.f32("evo1.proj_ln_eps");
    if (g.has("evo1.norm_eps"))    m.norm_eps_denom = g.f32("evo1.norm_eps");
    if (g.has("evo1.vit_ln_eps"))  m.vit_ln_eps     = g.f32("evo1.vit_ln_eps");
    if (g.has("evo1.lm_rope_theta")) m.lm_rope_base = (float) g.f64("evo1.lm_rope_theta");
    if (m.embed_dim != m.lm_hidden) {
        std::fprintf(stderr, "vla(evo1): embed_dim (%lld) != lm_hidden (%lld) - not handled\n",
                     (long long) m.embed_dim, (long long) m.lm_hidden); return false;
    }

    cfg = Config{};
    cfg.n_suffix       = m.horizon;
    cfg.n_full         = m.horizon;
    cfg.hidden         = m.lm_hidden;
    cfg.expert_h       = m.embed_dim;
    cfg.intermediate   = m.lm_inter;
    cfg.n_q_heads      = m.n_q;
    cfg.n_kv_heads     = m.n_kv;
    cfg.head_dim       = m.lm_head_dim;
    cfg.n_layers       = m.lm_layers;
    cfg.max_state_dim  = m.per_a;
    cfg.max_action_dim = m.per_a;
    cfg.real_state_dim = m.real_state_dim;
    cfg.real_action_dim= m.real_action_dim;
    cfg.norm_eps       = m.norm_eps_denom;
    cfg.num_steps      = (int) m.num_steps;
    cfg.rms_eps        = m.lm_rms_eps;
    cfg.rope_n_dims    = (int) m.lm_head_dim;
    cfg.rope_mode      = GGML_ROPE_TYPE_NEOX;
    cfg.rope_freq_base = m.lm_rope_base;
    cfg.n_img          = m.num_image_token;
    cfg.n_lang         = m.max_text_length;
    cfg.n_state        = 1;
    return true;
}

}

Evo1ModelArch::~Evo1ModelArch() {
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> evo1_create(const std::string& mmproj_path,
                                           const std::string& ckpt_path,
                                           const std::string& ) {
    if (!mmproj_path.empty())
        std::printf("vla(evo1): note - mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n",
                    mmproj_path.c_str());

    auto m = std::make_unique<Evo1ModelArch>();
    m->gguf_path = ckpt_path;
    m->matmul_type = std::getenv("VLA_EVO1_F32_WEIGHTS") ? GGML_TYPE_F32 : GGML_TYPE_BF16;

    gguf_reader g("evo1");
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has("evo1.architecture")) {
        std::fprintf(stderr, "vla(evo1): %s is not an evo1 GGUF (no evo1.architecture KV)\n", ckpt_path.c_str()); return nullptr;
    }
    if (!load_config(g, *m, m->cfg)) return nullptr;
    std::printf("vla(evo1): lm=%lldd×%lldL (%lldq/%lldkv×%lld) inter=%lld  embed=%lld dit=%lldL×%lldh  "
                "horizon=%lld per_a=%lld N_steps=%lld  resident matmul=%s\n",
                (long long) m->lm_hidden, (long long) m->lm_layers, (long long) m->n_q, (long long) m->n_kv,
                (long long) m->lm_head_dim, (long long) m->lm_inter, (long long) m->embed_dim,
                (long long) m->dit_layers, (long long) m->dit_heads, (long long) m->horizon, (long long) m->per_a,
                (long long) m->num_steps, m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) { m->is_cuda = true; m->is_gpu = true; std::printf("vla(evo1): backend = CUDA (device 0)\n"); }
    else            std::fprintf(stderr, "vla(evo1): ggml_backend_cuda_init failed; falling back to CPU\n");
#elif defined(GGML_USE_METAL)
    m->backend = ggml_backend_metal_init();
    if (m->backend) { m->is_gpu = true; std::printf("vla(evo1): backend = Metal\n"); }
    else            std::fprintf(stderr, "vla(evo1): ggml_backend_metal_init failed; falling back to CPU\n");
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(evo1): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(evo1): backend = CPU (%d threads)\n", m->n_threads);
    }

    ggml_init_params wp = {  (size_t) 32 * 1024 * 1024,
                             nullptr,  true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) { std::fprintf(stderr, "vla(evo1): ggml_init(ctx_weights) failed\n"); return nullptr; }
    ggml_context * W = m->ctx_weights;

    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(evo1): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, type, ggml_n_dims(gt), gt->ne);
        ggml_set_name(t, name);
        return t;
    };
    auto mk_mm = [&](const char * name) { return mk(name, m->matmul_type); };
    auto mk_f32 = [&](const char * name) { return mk(name, GGML_TYPE_F32); };

    bool ok = true;
    m->lm_output_norm = mk_f32("vlm.output_norm.weight"); ok &= (m->lm_output_norm != nullptr);
    m->lm.resize(m->lm_layers);
    for (int64_t i = 0; i < m->lm_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * suf) { std::snprintf(p, sizeof(p), "vlm.blk.%lld.%s", (long long) i, suf); return p; };
        auto & w = m->lm[i];
        w.attn_norm = mk_f32(N("attn_norm.weight"));
        w.Wq = mk_mm(N("attn_q.weight")); w.bq = mk_f32(N("attn_q.bias"));
        w.Wk = mk_mm(N("attn_k.weight")); w.bk = mk_f32(N("attn_k.bias"));
        w.Wv = mk_mm(N("attn_v.weight")); w.bv = mk_f32(N("attn_v.bias"));
        w.Wo = mk_mm(N("attn_o.weight"));
        w.ffn_norm = mk_f32(N("ffn_norm.weight"));
        w.Wgate = mk_mm(N("ffn_gate.weight")); w.Wup = mk_mm(N("ffn_up.weight")); w.Wdown = mk_mm(N("ffn_down.weight"));
        ok &= w.attn_norm && w.Wq && w.bq && w.Wk && w.bk && w.Wv && w.bv && w.Wo && w.ffn_norm && w.Wgate && w.Wup && w.Wdown;
    }

    m->ae_W1 = mk_mm("aex.ae.W1.weight"); m->ae_b1 = mk_f32("aex.ae.W1.bias");
    m->ae_W2 = mk_mm("aex.ae.W2.weight"); m->ae_b2 = mk_f32("aex.ae.W2.bias");
    m->ae_W3 = mk_mm("aex.ae.W3.weight"); m->ae_b3 = mk_f32("aex.ae.W3.bias");
    m->ae_pos = mk_f32("aex.ae.pos_enc");
    m->dit.resize(m->dit_layers);
    for (int64_t i = 0; i < m->dit_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * suf) { std::snprintf(p, sizeof(p), "aex.blk.%lld.%s", (long long) i, suf); return p; };
        auto & w = m->dit[i];
        w.n1w = mk_f32(N("norm1.weight")); w.n1b = mk_f32(N("norm1.bias"));
        w.n2w = mk_f32(N("norm2.weight")); w.n2b = mk_f32(N("norm2.bias"));
        w.Win = mk_mm(N("attn_in.weight")); w.bin = mk_f32(N("attn_in.bias"));
        w.Wo  = mk_mm(N("attn_out.weight")); w.bo = mk_f32(N("attn_out.bias"));
        w.f1w = mk_mm(N("ff1.weight")); w.f1b = mk_f32(N("ff1.bias"));
        w.f2w = mk_mm(N("ff2.weight")); w.f2b = mk_f32(N("ff2.bias"));
        ok &= w.n1w && w.n1b && w.n2w && w.n2b && w.Win && w.bin && w.Wo && w.bo && w.f1w && w.f1b && w.f2w && w.f2b;
    }
    m->norm_out_w = mk_f32("aex.norm_out.weight"); m->norm_out_b = mk_f32("aex.norm_out.bias");
    m->seq_pool_w = mk_mm("aex.seq_pool.weight");  m->seq_pool_b = mk_f32("aex.seq_pool.bias");
    m->head_W1 = mk_mm("aex.head.fc1.weight"); m->head_b1 = mk_f32("aex.head.fc1.bias");
    m->head_W2 = mk_mm("aex.head.fc2.weight"); m->head_b2 = mk_f32("aex.head.fc2.bias");
    m->time_pos = mk_f32("aex.time_pos_enc");
    m->state_W1 = mk_mm("aex.state_enc.fc1.weight"); m->state_b1 = mk_f32("aex.state_enc.fc1.bias");
    m->state_W2 = mk_mm("aex.state_enc.fc2.weight"); m->state_b2 = mk_f32("aex.state_enc.fc2.bias");
    ok &= m->ae_W1 && m->ae_b1 && m->ae_W2 && m->ae_b2 && m->ae_W3 && m->ae_b3 && m->ae_pos &&
          m->norm_out_w && m->norm_out_b && m->seq_pool_w && m->seq_pool_b && m->head_W1 && m->head_b1 &&
          m->head_W2 && m->head_b2 && m->time_pos && m->state_W1 && m->state_b1 && m->state_W2 && m->state_b2;

    if (g.meta("vit.patch_embd.weight") && ok) {
        m->vit_patch_w = mk("vit.patch_embd.weight", GGML_TYPE_F32);
        m->vit_patch_b = mk_f32("vit.patch_embd.bias");
        m->vit_cls     = mk_f32("vit.class_embd");
        m->vit_pos     = mk_f32("vit.pos_embd");
        m->vit.resize(m->vit_layers);
        for (int64_t i = 0; i < m->vit_layers && ok; ++i) {
            char p[64]; auto N = [&](const char * suf) { std::snprintf(p, sizeof(p), "vit.blk.%lld.%s", (long long) i, suf); return p; };
            auto & w = m->vit[i];
            w.n1w = mk_f32(N("norm1.weight")); w.n1b = mk_f32(N("norm1.bias"));
            w.n2w = mk_f32(N("norm2.weight")); w.n2b = mk_f32(N("norm2.bias"));
            w.ls1 = mk_f32(N("ls1")); w.ls2 = mk_f32(N("ls2"));
            w.Wqkv = mk_mm(N("attn_qkv.weight")); w.bqkv = mk_f32(N("attn_qkv.bias"));
            w.Wproj = mk_mm(N("attn_proj.weight")); w.bproj = mk_f32(N("attn_proj.bias"));
            w.Wfc1 = mk_mm(N("fc1.weight")); w.bfc1 = mk_f32(N("fc1.bias"));
            w.Wfc2 = mk_mm(N("fc2.weight")); w.bfc2 = mk_f32(N("fc2.bias"));
            ok &= w.n1w && w.n1b && w.n2w && w.n2b && w.ls1 && w.ls2 && w.Wqkv && w.bqkv && w.Wproj && w.bproj && w.Wfc1 && w.bfc1 && w.Wfc2 && w.bfc2;
        }
        m->mm_ln_w = mk_f32("mm.ln.weight"); m->mm_ln_b = mk_f32("mm.ln.bias");
        m->mm_W1 = mk_mm("mm.fc1.weight"); m->mm_b1 = mk_f32("mm.fc1.bias");
        m->mm_W2 = mk_mm("mm.fc2.weight"); m->mm_b2 = mk_f32("mm.fc2.bias");
        ok &= m->vit_patch_w && m->vit_patch_b && m->vit_cls && m->vit_pos && m->mm_ln_w && m->mm_ln_b && m->mm_W1 && m->mm_b1 && m->mm_W2 && m->mm_b2;
        m->have_vision = ok;
    } else {
        std::printf("vla(evo1): note - no vit.*/mm.* weights in the GGUF; predict() will require Inputs::precomputed_img_emb\n");
    }
    if (!ok) { std::fprintf(stderr, "vla(evo1): weight tensor setup failed\n"); return nullptr; }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(evo1): ggml_backend_alloc_ctx_tensors failed (OOM?)\n"); return nullptr; }
    for (ggml_tensor * t = ggml_get_first_tensor(W); t; t = ggml_get_next_tensor(W, t)) {
        std::vector<uint8_t> bytes = g.read_convert(ggml_get_name(t), t->type);
        if (bytes.empty() || bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(evo1): failed to load %s (%zu vs %zu bytes)\n",
                         ggml_get_name(t), bytes.size(), ggml_nbytes(t)); return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(evo1): weights resident in %.2f GiB (%s)%s\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0),
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16",
                m->have_vision ? " - incl. InternViT vision tower" : " - vision tower NOT loaded (precomputed_img_emb required)");

    m->state_min  = g.read_f32("state_min");
    m->state_max  = g.read_f32("state_max");
    m->action_min = g.read_f32("action_min");
    m->action_max = g.read_f32("action_max");
    if ((int64_t) m->state_min.size() != m->per_a || (int64_t) m->action_min.size() != m->per_a) {
        std::fprintf(stderr, "vla(evo1): norm-stats length mismatch\n"); return nullptr;
    }

    return m;
}

std::vector<float> Evo1ModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    int64_t n_views = 0;
    const float * img_emb_ptr = nullptr;
    std::vector<float> img_emb_host;
    if (in.precomputed_img_emb) {
        if (in.n_img_views < 1) {
            std::fprintf(stderr, "vla(evo1): precomputed_img_emb set but n_img_views=%d; cannot infer view count\n", in.n_img_views);
            return {};
        }
        n_views = in.n_img_views;
        img_emb_ptr = in.precomputed_img_emb;
    } else if (in.images && in.n_images > 0) {
        if (!have_vision) {
            std::fprintf(stderr, "vla(evo1): GGUF has no vision-tower weights - pass Inputs::precomputed_img_emb instead\n");
            return {};
        }
        n_views = in.n_images;

        ggml_init_params vp = {  (size_t) 32 * 1024 * 1024,  nullptr,  true };
        ggml_context * VC = ggml_init(vp);
        if (!VC) { std::fprintf(stderr, "vla(evo1): ggml_init(vision ctx) failed\n"); return {}; }
        ggml_tensor * t_px = ggml_new_tensor_3d(VC, GGML_TYPE_F32, image_size, image_size, 3); ggml_set_input(t_px);
        ggml_tensor * t_ie = build_internvit_view(VC, *this, t_px);
        ggml_set_output(t_ie);
        ggml_cgraph * vg = ggml_new_graph_custom(VC,  8192,  false);
        ggml_build_forward_expand(vg, t_ie);
        ggml_gallocr_t vga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!vga || !ggml_gallocr_alloc_graph(vga, vg)) {
            std::fprintf(stderr, "vla(evo1): vision ggml_gallocr_alloc_graph failed\n");
            if (vga) ggml_gallocr_free(vga);
            ggml_free(VC);
            return {};
        }
        img_emb_host.assign((size_t) n_views * num_image_token * lm_hidden, 0.0f);
        std::vector<float> chw;
        const auto tv0 = std::chrono::steady_clock::now();
        for (int64_t v = 0; v < n_views; ++v) {
            if (!preprocess_image_chw(in.images[v], image_size, chw)) { ggml_gallocr_free(vga); ggml_free(VC); return {}; }
            ggml_backend_tensor_set(t_px, chw.data(), 0, ggml_nbytes(t_px));
            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(evo1): vision graph compute failed (view %lld)\n", (long long) v);
                ggml_gallocr_free(vga); ggml_free(VC); return {};
            }
            ggml_backend_tensor_get(t_ie, img_emb_host.data() + v * num_image_token * lm_hidden, 0, ggml_nbytes(t_ie));
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tv0).count();
        ggml_gallocr_free(vga);
        ggml_free(VC);
        img_emb_ptr = img_emb_host.data();
    } else {
        std::fprintf(stderr, "vla(evo1): no images and no precomputed_img_emb in the request\n");
        return {};
    }
    const int64_t n_img_tokens = n_views * num_image_token;
    if (n_views != n_images)
        std::fprintf(stderr, "vla(evo1): note - %lld image views (model n_images=%lld); prompt adapts\n", (long long) n_views, (long long) n_images);

    bool pre_built = false;
    for (int j = 0; j < in.n_lang; ++j)
        if (in.lang_tokens[j] == (int32_t) img_ctx_id) { pre_built = true; break; }
    std::vector<int32_t> input_ids;
    input_ids.reserve(max_text_length);
    if (pre_built) {
        for (int j = 0; j < in.n_lang; ++j) input_ids.push_back(in.lang_tokens[j]);
    } else {
        for (int64_t v = 0; v < n_views; ++v) {
            input_ids.push_back((int32_t) img_start_id);
            for (int64_t k = 0; k < num_image_token; ++k) input_ids.push_back((int32_t) img_ctx_id);
            input_ids.push_back((int32_t) img_end_id);
        }
        for (int j = 0; j < in.n_lang; ++j) input_ids.push_back(in.lang_tokens[j]);
    }
    const int64_t n_real = (int64_t) input_ids.size();
    if (n_real > max_text_length) {
        std::fprintf(stderr, "vla(evo1): prompt too long (%lld > %lld)\n", (long long) n_real, (long long) max_text_length);
        return {};
    }
    const int32_t pad_id = (int32_t) pad_token_id;
    input_ids.resize(max_text_length, pad_id);
    const int64_t SEQ = max_text_length;

    gguf_reader g("evo1");
    if (!g.open(gguf_path)) return {};
    std::vector<float> inputs_embeds((size_t) SEQ * lm_hidden);
    if (!g.fetch_rows_f32("token_embd.weight", input_ids, inputs_embeds.data(), lm_hidden)) return {};
    {
        int64_t img_idx = 0;
        for (int64_t p = 0; p < SEQ; ++p) {
            if (input_ids[p] == (int32_t) img_ctx_id) {
                if (img_idx >= n_img_tokens) { std::fprintf(stderr, "vla(evo1): more IMG_CTX tokens than ViT embeds\n"); return {}; }
                std::memcpy(inputs_embeds.data() + p * lm_hidden,
                            img_emb_ptr + img_idx * lm_hidden, lm_hidden * sizeof(float));
                ++img_idx;
            }
        }
        if (img_idx != n_img_tokens) {
            std::fprintf(stderr, "vla(evo1): spliced %lld of %lld ViT embeds - IMG_CTX slot count does not match the images\n", (long long) img_idx, (long long) n_img_tokens);
            return {};
        }
    }

    std::vector<int32_t> attn_ok(SEQ, 0);
    if (in.attention_mask && in.attention_mask_n > 0) {
        if (in.attention_mask_n != (int) SEQ) {
            std::fprintf(stderr, "vla(evo1): attention_mask_n=%d does not match max_text_length=%lld\n",
                         in.attention_mask_n, (long long) SEQ);
            return {};
        }
        for (int64_t p = 0; p < SEQ; ++p) attn_ok[p] = in.attention_mask[p] ? 1 : 0;
    } else {
        for (int64_t p = 0; p < n_real; ++p) attn_ok[p] = 1;
    }

    std::vector<float> state_norm(per_a, 0.0f);
    for (int64_t i = 0; i < per_a; ++i) {
        const float lo = state_min[i], hi = state_max[i];
        float xn = 2.0f * (in.state[i] - lo) / (hi - lo + norm_eps_denom) - 1.0f;
        if (xn < -1.0f) xn = -1.0f;
        if (xn >  1.0f) xn =  1.0f;
        state_norm[i] = xn;
    }

    std::vector<float> x_init((size_t) action_dim);
    if (in.noise) {
        std::memcpy(x_init.data(), in.noise, x_init.size() * sizeof(float));
    } else {
        // Flow matching samples the base action from N(0,1). Box-Muller over a
        // simple LCG keeps this dependency-free; it runs only when the caller
        // omits noise, which the eval client does (noise is sampled here).
        uint64_t s = 0xE701ACE5ULL ^ (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
        auto next_u01 = [&s]() -> double {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            return ((double) (uint32_t) (s >> 32) + 1.0) / 4294967297.0; // in (0,1)
        };
        for (size_t i = 0; i < x_init.size(); i += 2) {
            const double u1 = next_u01(), u2 = next_u01();
            const double r  = std::sqrt(-2.0 * std::log(u1));
            const double th = 6.283185307179586 * u2;
            x_init[i] = (float) (r * std::cos(th));
            if (i + 1 < x_init.size()) x_init[i + 1] = (float) (r * std::sin(th));
        }
    }

    ggml_init_params cp = {  (size_t) 96 * 1024 * 1024,  nullptr,  true };
    ggml_context * C = ggml_init(cp);
    if (!C) { std::fprintf(stderr, "vla(evo1): ggml_init(ctx_compute) failed\n"); return {}; }

    const int64_t E = embed_dim, hd_dit = E / dit_heads;
    const float   scale_dit = 1.0f / std::sqrt((float) hd_dit);
    const int64_t Nctx = SEQ + 1;

    ggml_tensor * t_embeds   = ggml_new_tensor_2d(C, GGML_TYPE_F32, lm_hidden, SEQ);     ggml_set_input(t_embeds);
    ggml_tensor * t_pos      = ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ);                ggml_set_input(t_pos);
    ggml_tensor * t_lmmask   = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ);           ggml_set_input(t_lmmask);

    ggml_tensor * t_qmask    = ggml_new_tensor_2d(C, GGML_TYPE_F32, 1, SEQ);              ggml_set_input(t_qmask);
    ggml_tensor * t_state    = ggml_new_tensor_1d(C, GGML_TYPE_F32, per_a);              ggml_set_input(t_state);
    ggml_tensor * t_x        = ggml_new_tensor_1d(C, GGML_TYPE_F32, action_dim);         ggml_set_input(t_x);
    ggml_tensor * t_amask    = ggml_new_tensor_1d(C, GGML_TYPE_F32, per_a);              ggml_set_input(t_amask);

    ggml_tensor * h = t_embeds;
    for (int64_t i = 0; i < lm_layers; ++i) h = build_qwen2_layer(C, *this, lm[i], h, t_pos, t_lmmask, SEQ, t_qmask);
    ggml_tensor * context = ggml_mul(C, ggml_rms_norm(C, h, lm_rms_eps), lm_output_norm);

    ggml_tensor * se = ggml_relu(C, ggml_add(C, ggml_mul_mat(C, state_W1, t_state), state_b1));
    ggml_tensor * state_tok = ggml_add(C, ggml_mul_mat(C, state_W2, se), state_b2);
    ggml_tensor * context_tokens = ggml_concat(C, context, ggml_reshape_2d(C, state_tok, E, 1), 1);

    struct DC { ggml_tensor *Wq, *bq, *K, *V; };
    std::vector<DC> dc(dit_layers);
    for (int64_t i = 0; i < dit_layers; ++i) {
        const auto & w = dit[i];
        dc[i].Wq = inproj_split_w(C, w.Win, E, 0);
        ggml_tensor * Wk = inproj_split_w(C, w.Win, E, 1);
        ggml_tensor * Wv = inproj_split_w(C, w.Win, E, 2);
        dc[i].bq = inproj_split_b(C, w.bin, E, 0);
        ggml_tensor * bk = inproj_split_b(C, w.bin, E, 1);
        ggml_tensor * bv = inproj_split_b(C, w.bin, E, 2);
        ggml_tensor * kp = ggml_add(C, ggml_mul_mat(C, Wk, context_tokens), bk);
        ggml_tensor * vp = ggml_add(C, ggml_mul_mat(C, Wv, context_tokens), bv);
        dc[i].K = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, kp, hd_dit, dit_heads, Nctx), 0, 2, 1, 3));
        dc[i].V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, vp, hd_dit, dit_heads, Nctx), 1, 2, 0, 3));
    }

    auto denoise = [&](ggml_tensor * x_seq_masked, int64_t time_index) -> ggml_tensor * {
        ggml_tensor * time_emb = ggml_cont(C, ggml_view_1d(C, time_pos, E, (size_t) time_index * E * sizeof(float)));
        ggml_tensor * ae = ggml_relu(C, ggml_add(C, ggml_mul_mat(C, ae_W1, x_seq_masked), ae_b1));
        ae = ggml_add(C, ae, ae_pos);
        ae = ggml_relu(C, ggml_add(C, ggml_mul_mat(C, ae_W2, ae), ae_b2));
        ggml_tensor * x = ggml_add(C, ggml_mul_mat(C, ae_W3, ae), ae_b3);
        for (int64_t i = 0; i < dit_layers; ++i) {
            const auto & w = dit[i]; const auto & c = dc[i];
            ggml_tensor * x_q = ggml_add(C, ggml_mul(C, ggml_norm(C, x, proj_ln_eps), w.n1w), w.n1b);
            ggml_tensor * qp = ggml_add(C, ggml_mul_mat(C, c.Wq, x_q), c.bq);
            ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, qp, hd_dit, dit_heads, horizon), 0, 2, 1, 3));
            ggml_tensor * kq = ggml_mul_mat(C, c.K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            // The Evo-1 reference cross-attends over the full padded context
            // (no key mask), so the action queries see every LM position.
            ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale_dit, 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(C, c.V, aw);
            ggml_tensor * att_pre = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, kqv, 0, 2, 1, 3)), E, horizon);
            ggml_tensor * attn_out = ggml_add(C, ggml_mul_mat(C, w.Wo, att_pre), w.bo);
            ggml_tensor * x1 = ggml_add(C, x, attn_out);
            ggml_tensor * x2 = ggml_add(C, ggml_mul(C, ggml_norm(C, x1, proj_ln_eps), w.n2w), w.n2b);
            x2 = ggml_add(C, x2, time_emb);
            ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.f1w, x2), w.f1b);
            ff = ggml_gelu_erf(C, ff);
            ff = ggml_add(C, ggml_mul_mat(C, w.f2w, ff), w.f2b);
            x = ggml_add(C, x1, ff);
        }
        ggml_tensor * x_no = ggml_add(C, ggml_mul(C, ggml_norm(C, x, proj_ln_eps), norm_out_w), norm_out_b);
        ggml_tensor * x_flat = ggml_reshape_1d(C, ggml_cont(C, x_no), horizon * E);
        ggml_tensor * x_pooled = ggml_add(C, ggml_mul_mat(C, seq_pool_w, x_flat), seq_pool_b);
        ggml_tensor * mh = ggml_relu(C, ggml_add(C, ggml_mul_mat(C, head_W1, x_pooled), head_b1));
        return ggml_add(C, ggml_mul_mat(C, head_W2, mh), head_b2);
    };

    const float dt = 1.0f / (float) num_steps;
    ggml_tensor * x_action = t_x;
    for (int64_t step = 0; step < num_steps; ++step) {
        const int64_t time_index = (int64_t) ((double) step / (double) num_steps * 1000.0);
        ggml_tensor * x_seq = ggml_reshape_2d(C, x_action, per_a, horizon);
        ggml_tensor * x_seq_masked = ggml_mul(C, x_seq, t_amask);
        ggml_tensor * v_t = denoise(x_seq_masked, time_index);
        x_action = ggml_add(C, x_action, ggml_scale(C, v_t, dt));
    }
    ggml_set_name(x_action, "x_final");
    ggml_set_output(x_action);

    ggml_cgraph * gf = ggml_new_graph_custom(C,  32768,  false);
    ggml_build_forward_expand(gf, x_action);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "vla(evo1): ggml_gallocr_alloc_graph failed\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }
    ggml_backend_tensor_set(t_embeds, inputs_embeds.data(), 0, ggml_nbytes(t_embeds));
    { std::vector<int32_t> pp(SEQ); for (int64_t i = 0; i < SEQ; ++i) pp[i] = (int32_t) i; ggml_backend_tensor_set(t_pos, pp.data(), 0, ggml_nbytes(t_pos)); }
    { std::vector<float> mk((size_t) SEQ * SEQ); const float NEG = -std::numeric_limits<float>::infinity();
      for (int64_t q = 0; q < SEQ; ++q) for (int64_t kv = 0; kv < SEQ; ++kv) mk[q * SEQ + kv] = (kv <= q && attn_ok[kv]) ? 0.0f : NEG;
      ggml_backend_tensor_set(t_lmmask, mk.data(), 0, ggml_nbytes(t_lmmask)); }
    ggml_backend_tensor_set(t_state, state_norm.data(), 0, ggml_nbytes(t_state));
    ggml_backend_tensor_set(t_x, x_init.data(), 0, ggml_nbytes(t_x));
    { std::vector<float> am(per_a, 0.0f); for (int64_t i = 0; i < real_action_dim && i < per_a; ++i) am[i] = 1.0f;
      ggml_backend_tensor_set(t_amask, am.data(), 0, ggml_nbytes(t_amask)); }
    { std::vector<float> qm(SEQ, 0.0f); for (int64_t p = 0; p < SEQ; ++p) qm[p] = attn_ok[p] ? 1.0f : 0.0f;
      ggml_backend_tensor_set(t_qmask, qm.data(), 0, ggml_nbytes(t_qmask)); }

    const auto tc0 = std::chrono::steady_clock::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    const auto tc1 = std::chrono::steady_clock::now();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(evo1): ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(galloc); ggml_free(C); return {};
    }
    stats.ms_inference = std::chrono::duration<float, std::milli>(tc1 - tc0).count();

    std::vector<float> x_final((size_t) action_dim);
    ggml_backend_tensor_get(x_action, x_final.data(), 0, x_final.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);

    std::vector<float> out((size_t) horizon * per_a);
    for (int64_t hstep = 0; hstep < horizon; ++hstep)
        for (int64_t c = 0; c < per_a; ++c) {
            const double a = (double) x_final[hstep * per_a + c];
            out[hstep * per_a + c] = (float) ((a + 1.0) / 2.0 * ((double) action_max[c] - (double) action_min[c] + (double) norm_eps_denom) + (double) action_min[c]);
        }
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}
