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
#include "modules/gemma_expert.h"
#include "modules/siglip_vit.h"
#include "options.h"
#include "model.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "backend.h"
#include "gguf.h"
#include "gguf_reader.h"
#include "scratch_ctx.h"
#include "layers/embed.h"
#include "modules/preprocess.h"
#include "env_flag.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace vla {

namespace {


struct ExpertLayerW {
    ggml_tensor * ada_in_w   = nullptr;
    ggml_tensor * ada_in_b   = nullptr;
    ggml_tensor * Wq         = nullptr;
    ggml_tensor * Wk         = nullptr;
    ggml_tensor * Wv         = nullptr;
    ggml_tensor * Wo         = nullptr;
    ggml_tensor * ada_post_w = nullptr;
    ggml_tensor * ada_post_b = nullptr;
    ggml_tensor * Wgate      = nullptr;
    ggml_tensor * Wup        = nullptr;
    ggml_tensor * Wdown      = nullptr;
};

bool ends_with(const std::string & s, const char * sfx) {
    const size_t n = std::strlen(sfx);
    return s.size() >= n && s.compare(s.size()-n, n, sfx) == 0;
}
}

struct Pi05ModelArch : public ModelArchBase {
    Pi05ModelArch() : ModelArchBase(Arch::PI05) {}
    ~Pi05ModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    scratch_ctx           vision_scratch;

    struct MainKey {
        int64_t n_img=-1, n_lang=-1, nsteps=-1;
        bool operator==(const MainKey & o) const {
            return n_img==o.n_img && n_lang==o.n_lang && nsteps==o.nsteps;
        }
    };
    struct MainIO {
        ggml_tensor *t_image_emb=nullptr,*t_lang_emb=nullptr,*t_prefix_pos=nullptr;
        ggml_tensor *t_x0=nullptr,*t_suffix_pos=nullptr,*x_final=nullptr;
        std::vector<ggml_tensor*> t_time;
    };
    graph_cache<MainKey, MainIO> main_graph;
    std::string           ckpt_path_;
    // Opened once at load: reopening per predict re-parses the whole GGUF header.
    gguf_reader           io{"pi05"};
    ggml_type             matmul_type = GGML_TYPE_BF16;
    int64_t               adarms_cond_dim = 0;

    // In-tree SigLIP-So400m/14 vision tower (was llama.cpp clip.cpp mmproj).
    int64_t vit_hidden = 1152, vit_layers = 27, vit_heads = 16;
    int64_t vit_image_size = 224, vit_patch_size = 14, vit_n_tokens = 256;
    float   vit_ln_eps = 1e-6f;
    SigLipTower vit;
    ggml_tensor * mm_proj_w = nullptr, * mm_proj_b = nullptr;

    GemmaStack                  pl;
    std::vector<ExpertLayerW> ex_layers;

    ggml_tensor * ex_final_w = nullptr;
    ggml_tensor * ex_final_b = nullptr;

    ggml_tensor * W_ain  = nullptr, * b_ain  = nullptr;
    ggml_tensor * W_tin  = nullptr, * b_tin  = nullptr;
    ggml_tensor * W_tout = nullptr, * b_tout = nullptr;
    ggml_tensor * W_aout = nullptr, * b_aout = nullptr;

    std::vector<float> state_mean, state_std, action_mean, action_std;
    std::vector<float> action_q01, action_q99;
    bool quantile_norm = false;

    std::mt19937 rng{std::random_device{}()};
    int n_threads = default_cpu_threads();
};

namespace {

// One pre-norm SigLIP encoder block, identical to gr00tn1d5's in-tree tower
// (the PaliGemma vision tower is the same SigLIP-So400m/14). Bidirectional
// attention (nullptr mask), F32 score accumulation, tanh GELU FFN.
ggml_tensor * build_siglip_layer(ggml_context * C, const EncBlockW & w, ggml_tensor * x,
                                 int64_t seq, int64_t heads, int64_t head_dim, int64_t hidden, float ln_eps) {
    const float scale = 1.0f/std::sqrt((float) head_dim);
    ggml_tensor * n1 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.ln1w), w.ln1b);
    ggml_tensor * q = ggml_add(C, ggml_mul_mat(C, w.Wq, n1), w.bq);
    ggml_tensor * k = ggml_add(C, ggml_mul_mat(C, w.Wk, n1), w.bk);
    ggml_tensor * v = ggml_add(C, ggml_mul_mat(C, w.Wv, n1), w.bv);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, head_dim, heads, seq), 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, head_dim, heads, seq), 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, head_dim, heads, seq), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), hidden, seq);
    ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, h1, ln_eps), w.ln2w), w.ln2b);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wfc2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wfc1, n2), w.bfc1))), w.bfc2);
    return ggml_add(C, h1, ff);
}

// CHW-planar float image in [-1,1] for ggml_conv_2d (SigLIP mean/std 0.5).

ggml_tensor * build_vlm_layer(
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions,
        const Config & cfg, int64_t seq, float rope_base,
        ggml_tensor ** k_out, ggml_tensor ** v_out) {
    const int64_t hd  = cfg.head_dim;
    const int64_t nq  = cfg.n_q_heads;
    const int64_t nkv = cfg.n_kv_heads;
    const int64_t qf  = nq * hd;

    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.ln_in);

    ggml_tensor * q = ggml_mul_mat(ctx, w.Wq, x_norm);
    ggml_tensor * k = ggml_mul_mat(ctx, w.Wk, x_norm);
    ggml_tensor * v = ggml_mul_mat(ctx, w.Wv, x_norm);

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q, hd, nq,  seq);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k, hd, nkv, seq);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v, hd, nkv, seq);

    auto rope_call = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, positions, nullptr,
                             (int) hd, GGML_ROPE_TYPE_NEOX, 0,
                             rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);
    };
    ggml_tensor * q_rope = rope_call(q_h);
    ggml_tensor * k_rope = rope_call(k_h);

    if (k_out)
        *k_out = k_rope;
    if (v_out)
        *v_out = v_h;

    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v_h,    1, 2, 0, 3));

    ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    const float scale = 1.f/std::sqrt((float) hd);
    ggml_tensor * attn = ggml_soft_max_ext(ctx, kq,  nullptr, scale, 0.f);
    ggml_tensor * kqv  = ggml_mul_mat(ctx, V, attn);

    ggml_tensor * att_pre = ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq);
    ggml_tensor * o_out = ggml_mul_mat(ctx, w.Wo, att_pre);
    ggml_tensor * h1    = ggml_add(ctx, x_in, o_out);

    ggml_tensor * x_norm_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.ln_post);
    ggml_tensor * gate    = ggml_mul_mat(ctx, w.Wgate, x_norm_mlp);
    ggml_tensor * up      = ggml_mul_mat(ctx, w.Wup,   x_norm_mlp);
    ggml_tensor * inter_t = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
    ggml_tensor * mlp_out = ggml_mul_mat(ctx, w.Wdown, inter_t);
    return ggml_add(ctx, h1, mlp_out);
}

ggml_tensor * build_adarms(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * dense_w,
        ggml_tensor * dense_b, ggml_tensor * cond, int64_t h, float eps,
        ggml_tensor ** gate_out) {
    ggml_tensor * mod   = ggml_add(ctx, ggml_mul_mat(ctx, dense_w, cond), dense_b);
    ggml_tensor * scale = ggml_view_1d(ctx, mod, h, 0);
    ggml_tensor * shift = ggml_view_1d(ctx, mod, h, (size_t) h * sizeof(float));
    ggml_tensor * gate  = ggml_view_1d(ctx, mod, h, (size_t) 2*h * sizeof(float));
    ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);

    ggml_tensor * out = ggml_add(ctx,
        ggml_add(ctx, normed, ggml_mul(ctx, normed, scale)), shift);
    if (gate_out)
        *gate_out = gate;
    return out;
}

ggml_tensor * build_expert_layer(
        ggml_context * ctx, const ExpertLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions, ggml_tensor * cond,
        const Config & cfg, int64_t seq, float rope_base,
        ggml_tensor * cached_K, ggml_tensor * cached_V) {
    const int64_t hd  = cfg.head_dim;
    const int64_t nq  = cfg.n_q_heads;
    const int64_t nkv = cfg.n_kv_heads;
    const int64_t qf  = nq * hd;
    const int64_t h   = cfg.expert_h;

    ggml_tensor * gate_attn = nullptr;
    ggml_tensor * x_norm = build_adarms(ctx, x_in, w.ada_in_w, w.ada_in_b, cond, h, cfg.rms_eps, &gate_attn);

    ggml_tensor * q = ggml_mul_mat(ctx, w.Wq, x_norm);
    ggml_tensor * k = ggml_mul_mat(ctx, w.Wk, x_norm);
    ggml_tensor * v = ggml_mul_mat(ctx, w.Wv, x_norm);

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q, hd, nq,  seq);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k, hd, nkv, seq);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v, hd, nkv, seq);

    auto rope_call = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, positions, nullptr,
                             (int) hd, GGML_ROPE_TYPE_NEOX, 0,
                             rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);
    };
    ggml_tensor * q_rope = rope_call(q_h);
    ggml_tensor * k_rope = rope_call(k_h);

    ggml_tensor * K_full = ggml_concat(ctx, cached_K, k_rope,  2);
    ggml_tensor * V_full = ggml_concat(ctx, cached_V, v_h,     2);

    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));

    ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    const float scale = 1.f/std::sqrt((float) hd);
    ggml_tensor * attn = ggml_soft_max_ext(ctx, kq,  nullptr, scale, 0.f);
    ggml_tensor * kqv  = ggml_mul_mat(ctx, V, attn);

    ggml_tensor * att_pre = ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq);
    ggml_tensor * o_out = ggml_mul_mat(ctx, w.Wo, att_pre);

    ggml_tensor * h1 = ggml_add(ctx, x_in, ggml_mul(ctx, o_out, gate_attn));

    ggml_tensor * gate_ffn = nullptr;
    ggml_tensor * x_norm_mlp = build_adarms(ctx, h1, w.ada_post_w, w.ada_post_b, cond, h, cfg.rms_eps, &gate_ffn);
    ggml_tensor * gate    = ggml_mul_mat(ctx, w.Wgate, x_norm_mlp);
    ggml_tensor * up      = ggml_mul_mat(ctx, w.Wup,   x_norm_mlp);
    ggml_tensor * inter_t = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
    ggml_tensor * mlp_out = ggml_mul_mat(ctx, w.Wdown, inter_t);

    return ggml_add(ctx, h1, ggml_mul(ctx, mlp_out, gate_ffn));
}

bool load_config(const gguf_reader & g, Config & cfg) {
    auto need = [&](const char * k) {
        if (!g.has(k)) {
            std::fprintf(stderr, "vla(pi05): gguf missing key %s\n", k);
            return false;
        }
        return true;
    };
    for (const char * k : {"pi05.hidden", "pi05.intermediate", "pi05.n_q_heads", "pi05.n_kv_heads",
                           "pi05.head_dim", "pi05.n_layers", "pi05.expert_h", "pi05.expert_inter",
                           "pi05.chunk_size", "pi05.num_steps", "pi05.max_state_dim", "pi05.max_action_dim",
                           "pi05.real_state_dim", "pi05.real_action_dim", "pi05.tokenizer_max_length",
                           "pi05.min_period", "pi05.max_period"}) {
        if (!need(k))
            return false;
    }
    cfg = Config{};
    cfg.hidden          = g.u32("pi05.hidden");
    cfg.intermediate    = g.u32("pi05.intermediate");
    cfg.n_q_heads       = g.u32("pi05.n_q_heads");
    cfg.n_kv_heads      = g.u32("pi05.n_kv_heads");
    cfg.head_dim        = g.u32("pi05.head_dim");
    cfg.n_layers        = g.u32("pi05.n_layers");
    cfg.expert_h        = g.u32("pi05.expert_h");
    cfg.expert_inter    = g.u32("pi05.expert_inter");
    cfg.n_suffix        = g.u32("pi05.chunk_size");
    cfg.num_steps       = g.u32("pi05.num_steps");
    cfg.max_state_dim   = g.u32("pi05.max_state_dim");
    cfg.max_action_dim  = g.u32("pi05.max_action_dim");
    cfg.real_state_dim  = g.u32("pi05.real_state_dim");
    cfg.real_action_dim = g.u32("pi05.real_action_dim");
    cfg.n_lang          = g.u32("pi05.tokenizer_max_length");
    cfg.min_period      = g.f64("pi05.min_period");
    cfg.max_period      = g.f64("pi05.max_period");

    cfg.n_state         = 0;
    cfg.n_img           = 256;
    cfg.q_full_dim      = cfg.n_q_heads  * cfg.head_dim;
    cfg.kv_full_dim     = cfg.n_kv_heads*cfg.head_dim;
    cfg.self_attn_every_n = 0;
    cfg.rms_eps         = g.has("pi05.rms_norm_eps") ? g.f32("pi05.rms_norm_eps") : 1e-6f;
    cfg.norm_eps        = g.has("pi05.norm_eps")     ? g.f32("pi05.norm_eps")     : 1e-8f;
    cfg.rope_mode       = GGML_ROPE_TYPE_NEOX;
    cfg.rope_n_dims     = (int) cfg.head_dim;
    cfg.rope_freq_base  = g.has("pi05.rope_theta") ? (float) g.f64("pi05.rope_theta") : 10000.f;
    cfg.n_prefix        = 0;
    cfg.n_full          = 0;
    return true;
}

bool load_stats(gguf_reader & g, Pi05ModelArch & m) {
    const auto & cfg = m.cfg;
    m.state_mean .assign(cfg.real_state_dim,  0.f);
    m.state_std  .assign(cfg.real_state_dim,  1.f);
    m.action_mean.assign(cfg.real_action_dim, 0.f);
    m.action_std .assign(cfg.real_action_dim, 1.f);
    auto read1d = [&](const char * name, std::vector<float> & dst) {
        const ggml_tensor * t = g.meta(name);
        if (!t) {
            std::printf("vla(pi05): %s missing - identity\n", name);
            return;
        }
        if (t->ne[0] != (int64_t) dst.size()) {
            std::printf("vla(pi05): %s dim mismatch - identity\n", name);
            return;
        }
        const std::vector<float> identity = dst;
        if (!g.read_raw(name, dst.data(), dst.size()*sizeof(float))) {
            // A short read leaves dst half-overwritten.
            dst = identity;
            std::printf("vla(pi05): %s read failed - identity\n", name);
        }
    };
    read1d("state_mean",  m.state_mean);
    read1d("state_std",   m.state_std);
    read1d("action_mean", m.action_mean);
    read1d("action_std",  m.action_std);

    if (m.quantile_norm) {
        m.action_q01.assign(cfg.real_action_dim, -1.f);
        m.action_q99.assign(cfg.real_action_dim,  1.f);
        read1d("action_q01", m.action_q01);
        read1d("action_q99", m.action_q99);
    }
    return true;
}

}

Pi05ModelArch::~Pi05ModelArch() {
    if (weight_buf)
        ggml_backend_buffer_free(weight_buf);
    if (ctx_weights)
        ggml_free(ctx_weights);
    if (backend)
        ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> pi05_create(const std::string& mmproj_path,
                                           const std::string& ckpt_path,
                                           const std::string& config_path,
                                           const Options& opts) {
    (void) config_path;

    if (!ends_with(ckpt_path, ".gguf")) {
        std::fprintf(stderr,
            "vla(pi05): ckpt must be a GGUF produced by scripts/convert_pi05_to_gguf.py "
            "(got '%s')\n", ckpt_path.c_str());
        return nullptr;
    }

    auto m = std::make_unique<Pi05ModelArch>();
    m->ckpt_path_  = ckpt_path;
    m->matmul_type = opts.weight_dtype.value_or(GGML_TYPE_BF16);

    if (!m->io.open(ckpt_path))
        return nullptr;
    gguf_reader & g = m->io;
    if (!g.has("pi05.architecture") || g.str("pi05.architecture") != "pi05") {
        std::fprintf(stderr, "vla(pi05): '%s' is not a π0.5 GGUF (pi05.architecture missing/wrong)\n",
                     ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, m->cfg))
        return nullptr;
    const Config & cfg = m->cfg;
    m->adarms_cond_dim = g.has("pi05.adarms_cond_dim") ? g.u32("pi05.adarms_cond_dim") : cfg.expert_h;
    m->quantile_norm   = g.has("pi05.norm_mode") && g.str("pi05.norm_mode") == "quantiles";
    std::printf("vla(pi05): hidden=%lld inter=%lld heads=%lldq/%lldkv x%lld n_layers=%lld "
                "expert_h=%lld expert_inter=%lld chunk=%lld steps=%d real_state=%lld real_action=%lld "
                "max_len=%lld adarms_cond=%lld matmul_weights=%s\n",
                (long long) cfg.hidden, (long long) cfg.intermediate, (long long) cfg.n_q_heads,
                (long long) cfg.n_kv_heads, (long long) cfg.head_dim, (long long) cfg.n_layers,
                (long long) cfg.expert_h, (long long) cfg.expert_inter, (long long) cfg.n_suffix,
                cfg.num_steps, (long long) cfg.real_state_dim, (long long) cfg.real_action_dim,
                (long long) cfg.n_lang, (long long) m->adarms_cond_dim,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    m->n_threads = default_cpu_threads();
    {
        const Backend b = backend_init("vla(pi05)", m->n_threads);
        if (!b.handle) {
            return nullptr;
        }
        m->backend = b.handle;
    }

    // The SigLIP tower is now bundled in the ckpt GGUF; mmproj_path is ignored.
    (void) mmproj_path;
    {
        auto vu = [&](const char * k, int64_t & d) { if (g.has(k)) d = (int64_t) g.u32(k); };
        vu("pi05.vit_hidden", m->vit_hidden); vu("pi05.vit_layers", m->vit_layers);
        vu("pi05.vit_heads",  m->vit_heads);  vu("pi05.image_size", m->vit_image_size);
        vu("pi05.patch_size", m->vit_patch_size); vu("pi05.n_img_tokens", m->vit_n_tokens);
        if (g.has("pi05.vit_ln_eps"))
            m->vit_ln_eps = g.f32("pi05.vit_ln_eps");
        const int64_t grid = m->vit_image_size/m->vit_patch_size;
        if (grid * grid != m->vit_n_tokens || m->vit_n_tokens != cfg.n_img) {
            std::fprintf(stderr, "vla(pi05): vit geometry mismatch (grid^2=%lld n_img_tokens=%lld cfg.n_img=%lld)\n",
                         (long long) (grid * grid), (long long) m->vit_n_tokens, (long long) cfg.n_img);
            return nullptr;
        }
    }

    {
        ggml_init_params wp = { (size_t) 16*1024*1024, nullptr,  true };
        m->ctx_weights = ggml_init(wp);
        if (!m->ctx_weights) {
            std::fprintf(stderr, "vla(pi05): ggml_init(ctx_weights) failed\n");
            return nullptr;
        }
    }
    WeightLoader L("pi05", g, m->ctx_weights, m->matmul_type);

    m->vit.declare(L, "vit", m->vit_layers);
    m->mm_proj_w = L.gemm   ("mm.proj.weight");
    m->mm_proj_b = L.opt_f32("mm.proj.bias");

    m->pl.declare(L, "vlm", cfg.n_layers, false);

    m->ex_layers.resize(cfg.n_layers);
    for (int64_t i=0; i<cfg.n_layers; ++i) {
        ExpertLayerW & w = m->ex_layers[i];
        w.ada_in_w   = L.f32 ("aex.blk.%lld.attn_norm.weight", (long long)i);
        w.ada_in_b   = L.f32 ("aex.blk.%lld.attn_norm.bias",   (long long)i);
        w.Wq         = L.gemm("aex.blk.%lld.attn_q.weight",    (long long)i);
        w.Wk         = L.gemm("aex.blk.%lld.attn_k.weight",    (long long)i);
        w.Wv         = L.gemm("aex.blk.%lld.attn_v.weight",    (long long)i);
        w.Wo         = L.gemm("aex.blk.%lld.attn_o.weight",    (long long)i);
        w.ada_post_w = L.f32 ("aex.blk.%lld.ffn_norm.weight",  (long long)i);
        w.ada_post_b = L.f32 ("aex.blk.%lld.ffn_norm.bias",    (long long)i);
        w.Wgate      = L.gemm("aex.blk.%lld.ffn_gate.weight",  (long long)i);
        w.Wup        = L.gemm("aex.blk.%lld.ffn_up.weight",    (long long)i);
        w.Wdown      = L.gemm("aex.blk.%lld.ffn_down.weight",  (long long)i);
    }

    m->ex_final_w = L.f32("aex.output_norm.weight");
    m->ex_final_b = L.f32("aex.output_norm.bias");
    m->W_ain  = L.f32("action_in_proj.weight");   m->b_ain  = L.f32("action_in_proj.bias");
    m->W_tin  = L.f32("time_mlp_in.weight");      m->b_tin  = L.f32("time_mlp_in.bias");
    m->W_tout = L.f32("time_mlp_out.weight");     m->b_tout = L.f32("time_mlp_out.bias");
    m->W_aout = L.f32("action_out_proj.weight");  m->b_aout = L.f32("action_out_proj.bias");

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;

    std::printf("vla(pi05): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0));

    if (!load_stats(g, *m))
        return nullptr;
    std::printf("vla(pi05): model loaded (n_threads=%d)\n", m->n_threads);
    return m;
}

std::vector<float> Pi05ModelArch::predict(const Inputs& in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    stats = Stats{};

    const Config & cfg = this->cfg;
    const int64_t hidden_pl = cfg.hidden;
    const int64_t hidden_ex = cfg.expert_h;
    const int64_t chunk     = cfg.n_suffix;
    const int64_t n_suf     = chunk;
    const int64_t n_layers  = cfg.n_layers;
    const int64_t max_ad    = cfg.max_action_dim;
    const int     num_steps = cfg.num_steps;
    const float   dt        = -1.0f/(float) num_steps;
    const float   rope_base = cfg.rope_freq_base;

    std::vector<float> img_emb_host;
    int64_t n_img_tokens = 0;
    if (in.precomputed_img_emb) {
        n_img_tokens = (int64_t) in.n_img_views*cfg.n_img;
        img_emb_host.assign(in.precomputed_img_emb,
                            in.precomputed_img_emb+(size_t) n_img_tokens * hidden_pl);
    } else {
        if (in.n_images < 1 || !in.images) {
            std::fprintf(stderr, "vla(pi05): predict: no images and no precomputed_img_emb\n");
            return {};
        }
        const int64_t K = vit_n_tokens, H = hidden_pl, grid = vit_image_size/vit_patch_size;
        n_img_tokens = (int64_t) in.n_images*K;
        img_emb_host.assign((size_t) in.n_images*K * H, 0.0f);

        ggml_context * VC = vision_scratch.reset((size_t) 128*1024*1024);
        if (!VC) { std::fprintf(stderr, "vla(pi05): ggml_init(vision ctx) failed\n"); return {}; }
        ggml_tensor * t_px = ggml_new_tensor_3d(VC, GGML_TYPE_F32, vit_image_size, vit_image_size, 3); ggml_set_input(t_px);
        ggml_tensor * conv = ggml_conv_2d(VC, vit.patch_w, t_px, (int) vit_patch_size, (int) vit_patch_size, 0, 0, 1, 1);
        ggml_tensor * patches = ggml_cont(VC, ggml_transpose(VC, ggml_reshape_2d(VC, conv, grid * grid, vit_hidden)));
        ggml_tensor * h = ggml_add(VC, ggml_add(VC, patches, vit.patch_b), vit.pos);
        for (int64_t i=0; i<vit_layers; ++i)
            h = build_siglip_layer(VC, vit.enc.blk[i], h, K, vit_heads, vit_hidden/vit_heads, vit_hidden, vit_ln_eps);
        h = ggml_add(VC, ggml_mul(VC, ggml_norm(VC, h, vit_ln_eps), vit.post_ln_w), vit.post_ln_b);
        // PaliGemma projector: linear (+ optional bias), then 1/sqrt(hidden) scale (matches clip.cpp siglip.cpp).
        ggml_tensor * proj = ggml_mul_mat(VC, mm_proj_w, h);
        if (mm_proj_b)
            proj = ggml_add(VC, proj, mm_proj_b);
        ggml_tensor * vit_emb = ggml_scale(VC, proj, 1.0f/std::sqrt((float) proj->ne[0]));
        ggml_set_output(vit_emb);

        ggml_cgraph * vg = ggml_new_graph_custom(VC, 8192, false);
        ggml_build_forward_expand(vg, vit_emb);

        if (!vision_scratch.alloc(backend, vg)) {
            std::fprintf(stderr, "vla(pi05): vision gallocr alloc failed\n");
            return {};
        }
        const auto tv0 = clk::now();
        std::vector<float> chw;
        for (int v=0; v<in.n_images; ++v) {
            if (!preprocess_image_chw("pi05", in.images[v], vit_image_size, chw)) { return {}; }
            ggml_backend_tensor_set(t_px, chw.data(), 0, ggml_nbytes(t_px));
            graph_unique_names(vg);
            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(pi05): vision compute failed (view %d)\n", v);
                return {};
            }
            ggml_backend_tensor_get(vit_emb, img_emb_host.data()+(size_t) v * K * H, 0, ggml_nbytes(vit_emb));
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now()-tv0).count();

        // Undo the 1/sqrt(hidden) the shared vision graph applies; pi05 wants raw
        // projector features. Inside this branch on purpose: precomputed_img_emb
        // replaces the tower and is already LM-ready.
        const float img_scale = (float) std::sqrt((double) hidden_pl);
        for (float & x : img_emb_host)
            x *= img_scale;
    }

    if (in.n_lang < 1 || !in.lang_tokens) {
        std::fprintf(stderr, "vla(pi05): predict: empty lang_tokens\n");
        return {};
    }
    const int64_t n_lang   = in.n_lang;
    const int64_t n_prefix = n_img_tokens+n_lang;

    std::vector<int32_t> lang_ids(in.lang_tokens, in.lang_tokens+n_lang);
    std::vector<float> lang_rows((size_t) n_lang * hidden_pl);
    {
        if (!io.fetch_rows_f32("token_embd.weight", lang_ids, lang_rows.data(), hidden_pl)) return {};
    }

    // Prefix + expert graph depends only on the token counts and step count.
    const MainKey mkey{ n_img_tokens, n_lang, num_steps };
    const bool built = main_graph.ensure(backend, mkey, (size_t) 64*1024*1024,
                                         [&](ggml_context * C, MainIO & gio) -> ggml_cgraph * {
    ggml_tensor * t_image_emb = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_img_tokens); ggml_set_input(t_image_emb);
    ggml_tensor * t_lang_emb  = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_lang);       ggml_set_input(t_lang_emb);
    ggml_tensor * t_prefix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_prefix);                ggml_set_input(t_prefix_pos);
    ggml_tensor * t_x0        = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);           ggml_set_input(t_x0);
    ggml_tensor * t_suffix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_suf);                   ggml_set_input(t_suffix_pos);

    std::vector<ggml_tensor *> t_time(num_steps);
    for (int s=0; s<num_steps; ++s) {
        t_time[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, hidden_ex);
        ggml_set_input(t_time[s]);
    }

    const float lang_scale = (float) std::sqrt((double) hidden_pl);
    ggml_tensor * prefix_embs = ggml_concat(C, t_image_emb, ggml_scale(C, t_lang_emb, lang_scale),  1);

    std::vector<ggml_tensor *> cK(n_layers), cV(n_layers);
    {
        ggml_tensor * h = prefix_embs;
        for (int64_t i=0; i<n_layers; ++i) {
            h = build_vlm_layer(C, pl.blk[i], h, t_prefix_pos, cfg, n_prefix, rope_base,
                                &cK[i], &cV[i]);
        }
        (void) h;
    }

    ggml_tensor * x_t = t_x0;
    for (int step=0; step<num_steps; ++step) {

        ggml_tensor * c1   = ggml_silu(C, ggml_add(C, ggml_mul_mat(C, W_tin,  t_time[step]), b_tin));
        ggml_tensor * cond = ggml_silu(C, ggml_add(C, ggml_mul_mat(C, W_tout, c1),           b_tout));

        ggml_tensor * h = ggml_add(C, ggml_mul_mat(C, W_ain, x_t), b_ain);
        for (int64_t i=0; i<n_layers; ++i) {
            h = build_expert_layer(C, ex_layers[i], h, t_suffix_pos, cond, cfg, n_suf, rope_base,
                                   cK[i], cV[i]);
        }

        ggml_tensor * h_final = build_adarms(C, h, ex_final_w, ex_final_b, cond, hidden_ex, cfg.rms_eps, nullptr);
        ggml_tensor * v_t = ggml_add(C, ggml_mul_mat(C, W_aout, h_final), b_aout);
        x_t = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
    }
    ggml_tensor * x_final = x_t;
    ggml_set_output(x_final);

    gio.t_image_emb=t_image_emb; gio.t_lang_emb=t_lang_emb; gio.t_prefix_pos=t_prefix_pos;
    gio.t_x0=t_x0; gio.t_suffix_pos=t_suffix_pos; gio.t_time=t_time; gio.x_final=x_final;

    ggml_cgraph * gf = ggml_new_graph_custom(C,  16384,  false);
    ggml_build_forward_expand(gf, x_final);
    return gf;
    });
    if (!built) { std::fprintf(stderr, "vla(pi05): main graph build failed\n"); return {}; }

    MainIO & gio = main_graph.io();
    ggml_cgraph * gf = main_graph.graph();
    ggml_tensor * t_image_emb = gio.t_image_emb, * t_lang_emb = gio.t_lang_emb;
    ggml_tensor * t_prefix_pos = gio.t_prefix_pos, * t_x0 = gio.t_x0;
    ggml_tensor * t_suffix_pos = gio.t_suffix_pos, * x_final = gio.x_final;
    std::vector<ggml_tensor*> & t_time = gio.t_time;

    ggml_backend_tensor_set(t_image_emb, img_emb_host.data(), 0, ggml_nbytes(t_image_emb));
    ggml_backend_tensor_set(t_lang_emb,  lang_rows.data(),    0, ggml_nbytes(t_lang_emb));
    {
        std::vector<int32_t> pp(n_prefix); for (int64_t i=0; i<n_prefix; ++i) pp[i] = (int32_t) i;
        ggml_backend_tensor_set(t_prefix_pos, pp.data(), 0, ggml_nbytes(t_prefix_pos));
        std::vector<int32_t> sp(n_suf);    for (int64_t i=0; i<n_suf; ++i)    sp[i] = (int32_t) (n_prefix+i);
        ggml_backend_tensor_set(t_suffix_pos, sp.data(), 0, ggml_nbytes(t_suffix_pos));
    }
    {
        std::vector<float> x0h((size_t) max_ad * chunk);
        if (in.noise)
            std::memcpy(x0h.data(), in.noise, x0h.size()*sizeof(float));
        else {
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto & v : x0h)
                v = nd(rng);
        }
        ggml_backend_tensor_set(t_x0, x0h.data(), 0, ggml_nbytes(t_x0));
    }
    for (int s=0; s<num_steps; ++s) {
        const float timestep = 1.0f+(float) s * dt;
        const std::vector<float> tv = sinusoidal_time_emb(timestep, hidden_ex, cfg.min_period, cfg.max_period);
        ggml_backend_tensor_set(t_time[s], tv.data(), 0, ggml_nbytes(t_time[s]));
    }

    graph_unique_names(gf);
    const auto ti0 = clk::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now()-ti0).count();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(pi05): ggml_backend_graph_compute failed (%d)\n", (int) st);
        return {};
    }

    std::vector<float> out((size_t) chunk * max_ad);
    ggml_backend_tensor_get(x_final, out.data(), 0, out.size()*sizeof(float));

    if (!vla::env_flag("VLA_PI05_SKIP_UNNORM")) {
        for (int64_t t=0; t<chunk; ++t) {
            float * row = out.data()+(size_t) t * max_ad;
            for (int64_t j=0; j<max_ad; ++j) {
                if (j >= cfg.real_action_dim)
                    row[j] = 0.0f;
                else if (quantile_norm)
                    row[j] = (row[j]+1.0f)*(action_q99[j]-action_q01[j])*0.5f+action_q01[j];
                else
                    row[j] = row[j]*(action_std[j]+cfg.norm_eps)+action_mean[j];
            }
        }
    }

    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now()-t0).count();
    return out;
}

}
