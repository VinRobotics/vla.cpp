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
#include "options.h"
#include "model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "backend.h"
#include "gguf_reader.h"
#include "scratch_ctx.h"
#include "layers/attn.h"
#include "layers/embed.h"
#include "layers/norm.h"
#include "modules/preprocess.h"
#include "modules/prompt.h"
#include "act_dtype.h"
#include "cuda/vla_cuda_ops.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace vla {

struct Pi0ModelArch : public ModelArchBase {
    Pi0ModelArch() : ModelArchBase(Arch::PI0) {}
    ~Pi0ModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    ggml_context *        ctx_const   = nullptr;
    ggml_backend_buffer_t const_buf   = nullptr;
    scratch_ctx           vision_scratch;

    struct MainKey {
        int64_t n_img=-1, n_lang=-1, nsteps=-1;
        bool operator==(const MainKey & o) const {
            return n_img==o.n_img && n_lang==o.n_lang && nsteps==o.nsteps;
        }
    };
    struct MainIO {
        ggml_tensor *t_image_emb=nullptr,*t_lang_emb=nullptr,*t_prefix_pos=nullptr,*t_state=nullptr;
        ggml_tensor *t_x0=nullptr,*t_suffix_pos=nullptr,*t_full_mask=nullptr,*x_final=nullptr;
    };
    graph_cache<MainKey, MainIO> main_graph;
    // Opened once at load: reopening per predict re-parses the whole GGUF header.
    gguf_reader           io{"pi0"};
    ggml_type             matmul_type = GGML_TYPE_BF16;
    // Activation dtype carried between ops. F32 by default; BF16 under
    // --act-dtype bf16, which removes the per-GEMM F32<->BF16 round trip ggml
    // pays when BF16 weights meet F32 activations. See mm_act/as_type below.
    ggml_type             act_type    = GGML_TYPE_F32;

    PaliVision    vis;
    GemmaStack    pl;
    GemmaStack    ex;

    ggml_tensor * W_sp = nullptr,  * b_sp = nullptr;
    ggml_tensor * W_ain = nullptr, * b_ain = nullptr;
    ggml_tensor * W_at1 = nullptr, * b_at1 = nullptr;
    ggml_tensor * W_at2 = nullptr, * b_at2 = nullptr;
    ggml_tensor * W_aout = nullptr,* b_aout = nullptr;

    std::vector<ggml_tensor *> t_time;

    std::vector<float> state_mean, state_std, action_mean, action_std;

    int n_threads = default_cpu_threads();
};

namespace {

// One pre-norm SigLIP encoder block, identical to gr00tn1d5's in-tree tower
// (the PaliGemma vision tower is the same SigLIP-So400m/14). Bidirectional
// attention (nullptr mask), F32 score accumulation, tanh GELU FFN.
// Fused attention for the SigLIP tower and the PaliGemma/expert stack.
// OPT-IN (--flash-attn): pi0's score matrices are small (~560 keys, 8 heads), so
// fusing them only moved 111.4 ms -> 107.5 ms (3.5%), and ggml's FA computes K/V
// at F16 regardless of the input type. pi0's flash-attention SR was never
// measured, so it stays opt-in on an unquantified risk rather than a measured
// cost. (The evo1 SR drop this used to cite did not reproduce.)
// --act-dtype bf16 is the better lever here: 9.1%, and its SR was measured.

ggml_tensor * build_siglip_layer(ggml_context * C, const EncBlockW & w, ggml_tensor * x,
                                 int64_t seq, const EncCfg & c, ggml_type at) {
    const float scale = 1.0f/std::sqrt((float) c.head_dim);
    ggml_tensor * n1 = layer_norm(C, x, w.ln1w, w.ln1b, c.ln_eps);
    ggml_tensor * q = as_type(C, ggml_add(C, mm_act(C, w.Wq, n1, at), w.bq), GGML_TYPE_F32);
    ggml_tensor * k = as_type(C, ggml_add(C, mm_act(C, w.Wk, n1, at), w.bk), GGML_TYPE_F32);
    ggml_tensor * v = as_type(C, ggml_add(C, mm_act(C, w.Wv, n1, at), w.bv), GGML_TYPE_F32);
    ggml_tensor * Q = to_heads(C, q, c.head_dim, c.heads, seq);
    ggml_tensor * K = to_heads(C, k, c.head_dim, c.heads, seq);
    ggml_tensor * att;
    if (vla::flash_attn_enabled()) {
        // Avoids materialising the per-head score matrix; K/V stay F32 so the
        // numerics track the explicit path below (except on Hexagon, whose
        // kernel takes F16 K/V only; see fa_kv).
        ggml_tensor * V = to_heads(C, v, c.head_dim, c.heads, seq);
        ggml_tensor * fa = ggml_flash_attn_ext(C, Q, vla::fa_kv(C, K), vla::fa_kv(C, V), nullptr, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        att = ggml_reshape_2d(C, fa, c.hidden, seq);
    } else {
        att = attention(C, Q, K, to_heads_v(C, v, c.head_dim, c.heads, seq), nullptr, scale, c.hidden, seq);
    }
    ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, mm_act(C, w.Wo, as_type(C, att, at), at), w.bo));
    ggml_tensor * n2 = layer_norm(C, h1, w.ln2w, w.ln2b, c.ln_eps);
    ggml_tensor * ff = ggml_add(C, mm_act(C, w.Wfc2, vla::gelu(C, ggml_add(C, mm_act(C, w.Wfc1, n2, at), w.bfc1)), at), w.bfc2);
    return ggml_add(C, h1, ff);
}

ggml_tensor * build_embed_suffix(ggml_context * ctx, const Pi0ModelArch & m,
                                 ggml_tensor * state, ggml_tensor * x, ggml_tensor * time_bcast) {
    const ggml_type at = m.act_type;
    ggml_tensor * state_emb       = ggml_add(ctx, mm_act(ctx, m.W_sp,  as_type(ctx, state, at), at), m.b_sp);
    // x is the F32 flow-matching state; time_bcast is an F32 tensor. Both
    // enter the expert in the activation dtype, and ggml_concat needs them to agree.
    ggml_tensor * action_emb      = ggml_add(ctx, mm_act(ctx, m.W_ain, as_type(ctx, x, at), at), m.b_ain);
    ggml_tensor * action_time_in  = ggml_concat(ctx, action_emb, as_type(ctx, time_bcast, at), 0);
    ggml_tensor * mlp1            = ggml_add(ctx, mm_act(ctx, m.W_at1, action_time_in, at), m.b_at1);
    ggml_tensor * mlp1_silu       = ggml_silu(ctx, mlp1);
    ggml_tensor * action_time_emb = ggml_add(ctx, mm_act(ctx, m.W_at2, mlp1_silu, at), m.b_at2);
    ggml_tensor * state_emb_2d    = ggml_reshape_2d(ctx, state_emb, state_emb->ne[0], 1);
    return ggml_concat(ctx, state_emb_2d, action_time_emb, 1);
}

bool load_stats(gguf_reader & g, Pi0ModelArch & m) {
    const auto & cfg = m.cfg;
    m.state_mean .assign(cfg.real_state_dim,  0.f);
    m.state_std  .assign(cfg.real_state_dim,  1.f);
    m.action_mean.assign(cfg.real_action_dim, 0.f);
    m.action_std .assign(cfg.real_action_dim, 1.f);
    bool ok = true;
    ok &= read_pi_stat(g, "state_mean",  m.state_mean);
    ok &= read_pi_stat(g, "state_std",   m.state_std);
    ok &= read_pi_stat(g, "action_mean", m.action_mean);
    ok &= read_pi_stat(g, "action_std",  m.action_std);
    return ok;
}

}

Pi0ModelArch::~Pi0ModelArch() {
    if (weight_buf)
        ggml_backend_buffer_free(weight_buf);
    if (ctx_weights)
        ggml_free(ctx_weights);
    if (const_buf)
        ggml_backend_buffer_free(const_buf);
    if (ctx_const)
        ggml_free(ctx_const);
    if (backend)
        ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> pi0_create(const std::string& mmproj_path,
                                          const std::string& ckpt_path,
                                          const std::string& config_path,
                                          const Options& opts) {
    (void) config_path;

    auto m = std::make_unique<Pi0ModelArch>();
    m->matmul_type = opts.weight_dtype.value_or(vla::default_weight_dtype(GGML_TYPE_BF16));

    gguf_reader & g = m->io;
    if (!load_pi_config(g, ckpt_path, 1, m->cfg) || !resolve_num_steps("pi0", opts, m->cfg.num_steps))
        return nullptr;
    const Config & cfg = m->cfg;
    std::printf("vla(pi0): hidden=%lld inter=%lld heads=%lldq/%lldkv x%lld n_layers=%lld "
                "expert_h=%lld expert_inter=%lld chunk=%lld steps=%d real_state=%lld real_action=%lld "
                "matmul_weights=%s\n",
                (long long) cfg.hidden, (long long) cfg.intermediate, (long long) cfg.n_q_heads,
                (long long) cfg.n_kv_heads, (long long) cfg.head_dim, (long long) cfg.n_layers,
                (long long) cfg.expert_h, (long long) cfg.expert_inter, (long long) cfg.n_suffix,
                cfg.num_steps, (long long) cfg.real_state_dim, (long long) cfg.real_action_dim,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    {
        const Backend b = backend_init("vla(pi0)", m->n_threads);
        if (!b.handle) {
            return nullptr;
        }
        m->backend = b.handle;

        // BF16 activations need BF16-resident weights and the CUDA BF16 GEMM path.
        if (opts.act_dtype.value_or(GGML_TYPE_F32) == GGML_TYPE_BF16) {
            if (b.is_cuda && m->matmul_type == GGML_TYPE_BF16) {
                m->act_type = GGML_TYPE_BF16;
                cuda_register_bf16_ops();   // installs the in-tree BF16 CUDA kernels
                std::printf("vla(pi0): activations = BF16\n");
            } else {
                std::fprintf(stderr, "vla(pi0): --act-dtype bf16 ignored - needs CUDA and BF16 weights\n");
            }
        }
    }

    // The SigLIP tower is now bundled in the ckpt GGUF; mmproj_path is ignored.
    (void) mmproj_path;
    if (!m->vis.load(g, cfg.n_img))
        return nullptr;

    {
        ggml_init_params wp = {  (size_t) 16*1024*1024,  nullptr,  true };
        m->ctx_weights = ggml_init(wp);
        if (!m->ctx_weights) {
            std::fprintf(stderr, "vla(pi0): ggml_init(ctx_weights) failed\n");
            return nullptr;
        }
    }
    WeightLoader L("pi0", g, m->ctx_weights, m->matmul_type);

    m->vis.declare(L);

    m->pl.declare(L, "vlm", cfg.n_layers, false);
    m->ex.declare(L, "aex", cfg.n_layers, true);

    m->W_sp   = L.f32("state_proj.weight");          m->b_sp   = L.f32("state_proj.bias");
    m->W_ain  = L.f32("action_in_proj.weight");      m->b_ain  = L.f32("action_in_proj.bias");
    m->W_at1  = L.f32("action_time_mlp_in.weight");  m->b_at1  = L.f32("action_time_mlp_in.bias");
    m->W_at2  = L.f32("action_time_mlp_out.weight"); m->b_at2  = L.f32("action_time_mlp_out.bias");
    m->W_aout = L.f32("action_out_proj.weight");     m->b_aout = L.f32("action_out_proj.bias");

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;

    std::printf("vla(pi0): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0));

    if (!load_stats(g, *m))
        return nullptr;

    {
        ggml_init_params p = { (size_t) cfg.num_steps*ggml_tensor_overhead(), nullptr, true };
        m->ctx_const = ggml_init(p);
        if (!m->ctx_const) {
            std::fprintf(stderr, "vla(pi0): ggml_init(ctx_const) failed\n");
            return nullptr;
        }
        m->t_time.resize(cfg.num_steps);
        for (ggml_tensor * & t : m->t_time)
            t = ggml_new_tensor_2d(m->ctx_const, GGML_TYPE_F32, cfg.expert_h, cfg.n_suffix);
        m->const_buf = alloc_weights(m->ctx_const, m->backend);
        if (!m->const_buf) {
            std::fprintf(stderr, "vla(pi0): alloc_weights(time embeddings) failed\n");
            return nullptr;
        }
        const float dt = -1.0f/(float) cfg.num_steps;
        std::vector<float> tile((size_t) cfg.expert_h * cfg.n_suffix);
        for (int s=0; s<cfg.num_steps; ++s) {
            const std::vector<float> tv = sinusoidal_time_emb(1.0f+(float) s * dt, cfg.expert_h, cfg.min_period, cfg.max_period);
            for (int64_t c=0; c<cfg.n_suffix; ++c)
                std::memcpy(tile.data()+c * cfg.expert_h, tv.data(), cfg.expert_h * sizeof(float));
            ggml_backend_tensor_set(m->t_time[s], tile.data(), 0, ggml_nbytes(m->t_time[s]));
        }
    }
    std::printf("vla(pi0): model loaded (n_threads=%d)\n", m->n_threads);
    return m;
}

std::vector<float> Pi0ModelArch::predict(const Inputs& in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    stats = Stats{};

    const Config & cfg = this->cfg;
    const int64_t hidden_pl = cfg.hidden;
    const int64_t hidden_ex = cfg.expert_h;
    const int64_t chunk     = cfg.n_suffix;
    const int64_t n_suf     = 1+chunk;
    const int64_t n_layers  = cfg.n_layers;
    const int64_t max_sd    = cfg.max_state_dim;
    const int64_t max_ad    = cfg.max_action_dim;
    const int     num_steps = cfg.num_steps;
    const float   dt        = -1.0f/(float) num_steps;
    const bool    fa        = flash_attn_enabled();

    int64_t n_img_tokens = 0;
    if (in.precomputed_img_emb) {
        if (in.n_img_views < 1) {
            std::fprintf(stderr, "vla(pi0): precomputed_img_emb set but n_img_views=%d\n", in.n_img_views);
            return {};
        }
        n_img_tokens = (int64_t) in.n_img_views*cfg.n_img;
    } else {
        if (in.n_images < 1 || !in.images) {
            std::fprintf(stderr, "vla(pi0): predict: no images and no precomputed_img_emb\n");
            return {};
        }
        n_img_tokens = (int64_t) in.n_images*vis.n_tokens;
    }

    if (in.n_lang < 1 || !in.lang_tokens) {
        std::fprintf(stderr, "vla(pi0): predict: empty lang_tokens\n");
        return {};
    }
    const int64_t n_lang   = in.n_lang;
    const int64_t n_prefix = n_img_tokens+n_lang;
    const int64_t n_total  = n_prefix+n_suf;

    std::vector<int32_t> lang_ids(in.lang_tokens, in.lang_tokens+n_lang);
    std::vector<float> lang_rows((size_t) n_lang * hidden_pl);
    {
        if (!io.fetch_rows_f32("token_embd.weight", lang_ids, lang_rows.data(), hidden_pl)) return {};
    }

    // Prefix + expert graph depends only on the token counts and step count.
    const MainKey mkey{ n_img_tokens, n_lang, num_steps };
    const size_t max_nodes = (size_t) 64*n_layers*(num_steps+1) + 1024;
    const bool built = main_graph.ensure(backend, mkey,
                                         ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false),
                                         [&](ggml_context * C, MainIO & gio) -> ggml_cgraph * {
    ggml_tensor * t_image_emb = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_img_tokens); ggml_set_input(t_image_emb);
    ggml_tensor * t_lang_emb  = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_lang);       ggml_set_input(t_lang_emb);
    ggml_tensor * t_prefix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_prefix);                ggml_set_input(t_prefix_pos);
    ggml_tensor * t_state     = ggml_new_tensor_1d(C, GGML_TYPE_F32, max_sd);                  ggml_set_input(t_state);
    ggml_tensor * t_x0        = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);           ggml_set_input(t_x0);
    ggml_tensor * t_suffix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_suf);                   ggml_set_input(t_suffix_pos);
    ggml_tensor * t_full_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_total, n_suf);          ggml_set_input(t_full_mask);

    const float lang_scale = (float) std::sqrt((double) hidden_pl);
    ggml_tensor * prefix_embs = as_type(C,
        ggml_concat(C, t_image_emb, ggml_scale(C, t_lang_emb, lang_scale),  1), act_type);

    std::vector<ggml_tensor *> cK(n_layers), cV(n_layers);
    {
        ggml_tensor * h = prefix_embs;
        for (int64_t i=0; i<n_layers; ++i) {
            h = gemma_layer(C, pl.blk[i], h, t_prefix_pos, cfg, n_prefix,
                            nullptr, nullptr, nullptr, &cK[i], &cV[i], act_type, fa);
        }
        (void) h;
    }

    // x_t is the flow-matching state; it stays F32 so num_steps Euler updates
    // do not accumulate in 8 mantissa bits.
    ggml_tensor * x_t = t_x0;
    std::vector<ggml_tensor *> v_steps(num_steps);
    for (int step=0; step<num_steps; ++step) {
        ggml_tensor * h = build_embed_suffix(C, *this, t_state, x_t, t_time[step]);
        for (int64_t i=0; i<n_layers; ++i) {
            h = gemma_layer(C, ex.blk[i], h, t_suffix_pos, cfg, n_suf,
                            cK[i], cV[i], t_full_mask, nullptr, nullptr, act_type, fa);
        }
        ggml_tensor * h_final = ggml_mul(C, ggml_rms_norm(C, h, cfg.rms_eps), ex.output_norm);
        // row stride follows h_final's dtype, which is BF16 on the BF16 path
        const size_t rb = (size_t) hidden_ex * ggml_element_size(h_final);
        ggml_tensor * h_actions = ggml_view_2d(C, h_final, hidden_ex, chunk,  rb,  rb);
        // h_actions is an offset slice of whole rows, so it is already contiguous
        ggml_tensor * v_t = as_type(C,
            ggml_add(C, mm_act(C, W_aout, h_actions, act_type), b_aout), GGML_TYPE_F32);
        v_steps[step] = v_t;
        x_t = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
    }
    ggml_tensor * x_final = x_t;
    ggml_set_output(x_final);

    gio.t_image_emb=t_image_emb; gio.t_lang_emb=t_lang_emb; gio.t_prefix_pos=t_prefix_pos;
    gio.t_state=t_state; gio.t_x0=t_x0; gio.t_suffix_pos=t_suffix_pos;
    gio.t_full_mask=t_full_mask; gio.x_final=x_final;

    ggml_cgraph * gf = ggml_new_graph_custom(C, max_nodes, false);
    ggml_build_forward_expand(gf, x_final);
    return gf;
    });
    if (!built) { std::fprintf(stderr, "vla(pi0): main graph build failed\n"); return {}; }

    MainIO & gio = main_graph.io();
    ggml_cgraph * gf = main_graph.graph();
    ggml_tensor * t_image_emb = gio.t_image_emb, * t_lang_emb = gio.t_lang_emb;
    ggml_tensor * t_prefix_pos = gio.t_prefix_pos, * t_state = gio.t_state, * t_x0 = gio.t_x0;
    ggml_tensor * t_suffix_pos = gio.t_suffix_pos, * t_full_mask = gio.t_full_mask;
    ggml_tensor * x_final = gio.x_final;

    if (in.precomputed_img_emb) {
        ggml_backend_tensor_set(t_image_emb, in.precomputed_img_emb, 0, ggml_nbytes(t_image_emb));
    } else {
        const int64_t K = vis.n_tokens, H = hidden_pl, grid = vis.image_size/vis.patch_size;
        ggml_context * VC = vision_scratch.reset((size_t) 128*1024*1024);
        if (!VC) { std::fprintf(stderr, "vla(pi0): ggml_init(vision ctx) failed\n"); return {}; }
        ggml_tensor * t_px = ggml_new_tensor_3d(VC, GGML_TYPE_F32, vis.image_size, vis.image_size, 3); ggml_set_input(t_px);
        // patch embed (conv_2d) stays F32; the tower runs in the activation dtype
        ggml_tensor * h = as_type(VC, vis.vit.embed_conv(VC, t_px, vis.patch_size, grid), act_type);
        for (const EncBlockW & w : vis.vit.enc.blk)
            h = build_siglip_layer(VC, w, h, K, vis.vit.enc.cfg, act_type);
        h = layer_norm(VC, h, vis.vit.post_ln_w, vis.vit.post_ln_b, vis.vit.enc.cfg.ln_eps);
        // PaliGemma projector: linear (+ optional bias).
        ggml_tensor * proj = mm_act(VC, vis.proj_w, h, act_type);
        if (vis.proj_b)
            proj = ggml_add(VC, proj, vis.proj_b);
        ggml_tensor * vit_emb = as_type(VC, proj, GGML_TYPE_F32);
        ggml_set_output(vit_emb);

        ggml_cgraph * vg = ggml_new_graph_custom(VC, 8192, false);
        ggml_build_forward_expand(vg, vit_emb);

        if (!vision_scratch.alloc(backend, vg)) {
            std::fprintf(stderr, "vla(pi0): vision gallocr alloc failed\n");
            return {};
        }
        const auto tv0 = clk::now();
        std::vector<float> chw;
        for (int v=0; v<in.n_images; ++v) {
            if (!preprocess_image_chw("pi0", in.images[v], vis.image_size, chw)) { return {}; }
            ggml_backend_tensor_set(t_px, chw.data(), 0, ggml_nbytes(t_px));
            graph_unique_names(vg);
            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(pi0): vision compute failed (view %d)\n", v);
                return {};
            }
            ggml_tensor * dst = ggml_view_2d(VC, t_image_emb, H, K, t_image_emb->nb[1], (size_t) v * K * t_image_emb->nb[1]);
            if (ggml_backend_view_init(dst) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(pi0): image embedding view failed (view %d)\n", v);
                return {};
            }
            ggml_backend_tensor_copy(vit_emb, dst);
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now()-tv0).count();
    }
    ggml_backend_tensor_set(t_lang_emb,  lang_rows.data(),    0, ggml_nbytes(t_lang_emb));
    {
        std::vector<int32_t> pp(n_prefix); for (int64_t i=0; i<n_prefix; ++i) pp[i] = (int32_t) i;
        ggml_backend_tensor_set(t_prefix_pos, pp.data(), 0, ggml_nbytes(t_prefix_pos));
        std::vector<int32_t> sp(n_suf);     for (int64_t i=0; i<n_suf; ++i)    sp[i] = (int32_t) (n_prefix+i);
        ggml_backend_tensor_set(t_suffix_pos, sp.data(), 0, ggml_nbytes(t_suffix_pos));
    }
    {

        std::vector<float> sh(max_sd, 0.f);
        for (int64_t i=0; i<max_sd; ++i)
            sh[i] = in.state ? in.state[i] : 0.f;
        for (int64_t i=0; i<cfg.real_state_dim && i<max_sd; ++i)
            sh[i] = (sh[i]-state_mean[i])/(state_std[i]+cfg.norm_eps);
        ggml_backend_tensor_set(t_state, sh.data(), 0, ggml_nbytes(t_state));
    }
    {
        std::vector<float> x0h;
        init_noise(in, (size_t) max_ad * chunk, x0h);
        ggml_backend_tensor_set(t_x0, x0h.data(), 0, ggml_nbytes(t_x0));
    }
    {

        std::vector<float> mk((size_t) n_total * n_suf);
        for (int64_t i=0; i<n_suf; ++i)
            for (int64_t j=0; j<n_total; ++j) {
                bool allowed;
                if (j < n_prefix)
                    allowed = true;
                else {
                    const int64_t s = j-n_prefix;
                    allowed = (i == 0) ? (s == 0) : true;
                }
                mk[i * n_total+j] = allowed ? 0.f : -INFINITY;
            }
        ggml_backend_tensor_set(t_full_mask, mk.data(), 0, ggml_nbytes(t_full_mask));
    }
    graph_unique_names(gf);
    const auto ti0 = clk::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now()-ti0).count();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(pi0): ggml_backend_graph_compute failed (%d)\n", (int) st);
        return {};
    }

    std::vector<float> out((size_t) chunk * max_ad);
    ggml_backend_tensor_get(x_final, out.data(), 0, out.size()*sizeof(float));
    for (int64_t t=0; t<chunk; ++t) {
        float * row = out.data()+(size_t) t * max_ad;
        for (int64_t j=0; j<max_ad; ++j)
            row[j] = j < cfg.real_action_dim ? row[j]*(action_std[j]+cfg.norm_eps)+action_mean[j] : 0.0f;
    }

    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now()-t0).count();
    return out;
}

}
