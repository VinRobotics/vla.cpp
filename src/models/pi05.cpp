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
#include "layers/embed.h"
#include "layers/linear.h"
#include "modules/preprocess.h"
#include "modules/prompt.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace vla {

struct Pi05ModelArch : public ModelArchBase {
    Pi05ModelArch() : ModelArchBase(Arch::PI05) {}
    ~Pi05ModelArch() override;

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
        ggml_tensor *t_image_emb=nullptr,*t_lang_emb=nullptr,*t_prefix_pos=nullptr;
        ggml_tensor *t_x0=nullptr,*t_suffix_pos=nullptr,*x_final=nullptr;
    };
    graph_cache<MainKey, MainIO> main_graph;
    // Opened once at load: reopening per predict re-parses the whole GGUF header.
    gguf_reader           io{"pi05"};
    ggml_type             matmul_type = GGML_TYPE_BF16;

    PaliVision                  vis;
    GemmaStack                  pl;
    std::vector<GemmaLayerW>    ex_layers;

    ggml_tensor * W_ain  = nullptr, * b_ain  = nullptr;
    ggml_tensor * W_aout = nullptr, * b_aout = nullptr;

    ggml_tensor * ada_mod = nullptr;

    std::vector<float> action_mean, action_std;
    std::vector<float> action_q01, action_q99;
    bool quantile_norm = false;

    int n_threads = default_cpu_threads();
};

namespace {

ggml_tensor * build_adarms(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * mod, int64_t h, float eps,
        ggml_tensor ** gate_out) {
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
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions, ggml_tensor * mod_attn, ggml_tensor * mod_ffn,
        const Config & cfg, int64_t seq,
        ggml_tensor * cached_K, ggml_tensor * cached_V) {
    const int64_t h = cfg.expert_h;

    ggml_tensor * gate_attn = nullptr;
    ggml_tensor * x_norm = build_adarms(ctx, x_in, mod_attn, h, cfg.rms_eps, &gate_attn);
    ggml_tensor * o_out  = gemma_attn(ctx, w, x_norm, positions, cfg, seq, cached_K, cached_V, nullptr,
                                      nullptr, nullptr, GGML_TYPE_F32, false);

    ggml_tensor * h1 = ggml_add(ctx, x_in, ggml_mul(ctx, o_out, gate_attn));

    ggml_tensor * gate_ffn = nullptr;
    ggml_tensor * x_norm_mlp = build_adarms(ctx, h1, mod_ffn, h, cfg.rms_eps, &gate_ffn);
    ggml_tensor * mlp_out    = gemma_mlp(ctx, w, x_norm_mlp, GGML_TYPE_F32);

    return ggml_add(ctx, h1, ggml_mul(ctx, mlp_out, gate_ffn));
}

bool bake_adarms(gguf_reader & g, Pi05ModelArch & m) {
    const Config & cfg = m.cfg;
    const int64_t h  = cfg.expert_h, nj = 2*cfg.n_layers+1, ns = cfg.num_steps;
    const float   dt = -1.0f/(float) ns;

    ggml_init_params wp = { (size_t) (2*nj+4)*ggml_tensor_overhead(), nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> wctx(ggml_init(wp), ggml_free);
    if (!wctx) {
        std::fprintf(stderr, "vla(pi05): ggml_init(adaRMS weights) failed\n");
        return false;
    }
    WeightLoader L("pi05", g, wctx.get(), m.matmul_type);
    std::vector<ggml_tensor *> W(nj), B(nj);
    for (int64_t i=0; i<cfg.n_layers; ++i) {
        W[2*i]   = L.f32("aex.blk.%lld.attn_norm.weight", (long long)i);
        B[2*i]   = L.f32("aex.blk.%lld.attn_norm.bias",   (long long)i);
        W[2*i+1] = L.f32("aex.blk.%lld.ffn_norm.weight",  (long long)i);
        B[2*i+1] = L.f32("aex.blk.%lld.ffn_norm.bias",    (long long)i);
    }
    W[nj-1] = L.f32("aex.output_norm.weight");
    B[nj-1] = L.f32("aex.output_norm.bias");
    ggml_tensor * W_tin  = L.f32("time_mlp_in.weight"),  * b_tin  = L.f32("time_mlp_in.bias");
    ggml_tensor * W_tout = L.f32("time_mlp_out.weight"), * b_tout = L.f32("time_mlp_out.bias");
    ggml_backend_buffer_t wbuf_raw = nullptr;
    const bool uploaded = L.upload(m.backend, &wbuf_raw);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> wbuf(wbuf_raw, ggml_backend_buffer_free);
    if (!uploaded)
        return false;

    const size_t max_nodes = (size_t) ns*(2*nj+8) + 64;
    scratch_ctx sc;
    ggml_context * C = sc.reset(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));
    if (!C) {
        std::fprintf(stderr, "vla(pi05): ggml_init(adaRMS graph) failed\n");
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(C, max_nodes, false);
    std::vector<ggml_tensor *> t_time(ns), mods((size_t) ns*nj);
    for (int64_t s=0; s<ns; ++s) {
        t_time[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, h);
        ggml_set_input(t_time[s]);
        ggml_tensor * c1   = ggml_silu(C, ggml_add(C, ggml_mul_mat(C, W_tin,  t_time[s]), b_tin));
        ggml_tensor * cond = ggml_silu(C, ggml_add(C, ggml_mul_mat(C, W_tout, c1),        b_tout));
        for (int64_t j=0; j<nj; ++j) {
            ggml_tensor * mod = ggml_add(C, ggml_mul_mat(C, W[j], cond), B[j]);
            ggml_set_output(mod);
            ggml_build_forward_expand(gf, mod);
            mods[s*nj+j] = mod;
        }
    }
    if (!sc.alloc(m.backend, gf)) {
        std::fprintf(stderr, "vla(pi05): adaRMS gallocr alloc failed\n");
        return false;
    }
    for (int64_t s=0; s<ns; ++s) {
        const std::vector<float> tv = sinusoidal_time_emb(1.0f+(float) s * dt, h, cfg.min_period, cfg.max_period);
        ggml_backend_tensor_set(t_time[s], tv.data(), 0, ggml_nbytes(t_time[s]));
    }
    graph_unique_names(gf);
    if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(pi05): adaRMS compute failed\n");
        return false;
    }
    std::vector<float> table((size_t) ns*nj*3*h);
    for (size_t k=0; k<mods.size(); ++k)
        ggml_backend_tensor_get(mods[k], table.data()+k*3*h, 0, ggml_nbytes(mods[k]));

    ggml_init_params p = { ggml_tensor_overhead(), nullptr, true };
    m.ctx_const = ggml_init(p);
    if (!m.ctx_const) {
        std::fprintf(stderr, "vla(pi05): ggml_init(ctx_const) failed\n");
        return false;
    }
    m.ada_mod   = ggml_new_tensor_3d(m.ctx_const, GGML_TYPE_F32, 3*h, nj, ns);
    m.const_buf = alloc_weights(m.ctx_const, m.backend);
    if (!m.const_buf) {
        std::fprintf(stderr, "vla(pi05): alloc_weights(adaRMS table) failed\n");
        return false;
    }
    ggml_backend_tensor_set(m.ada_mod, table.data(), 0, ggml_nbytes(m.ada_mod));
    return true;
}

bool load_stats(gguf_reader & g, Pi05ModelArch & m) {
    const auto & cfg = m.cfg;
    m.action_mean.assign(cfg.real_action_dim, 0.f);
    m.action_std .assign(cfg.real_action_dim, 1.f);
    bool ok = true;
    ok &= read_pi_stat(g, "action_mean", m.action_mean);
    ok &= read_pi_stat(g, "action_std",  m.action_std);

    if (m.quantile_norm) {
        m.action_q01.assign(cfg.real_action_dim, -1.f);
        m.action_q99.assign(cfg.real_action_dim,  1.f);
        ok &= read_pi_stat(g, "action_q01", m.action_q01);
        ok &= read_pi_stat(g, "action_q99", m.action_q99);
    }
    return ok;
}

}

Pi05ModelArch::~Pi05ModelArch() {
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

std::unique_ptr<ModelArchBase> pi05_create(const std::string& mmproj_path,
                                           const std::string& ckpt_path,
                                           const std::string& config_path,
                                           const Options& opts) {
    (void) config_path;

    auto m = std::make_unique<Pi05ModelArch>();
    m->matmul_type = opts.weight_dtype.value_or(vla::default_weight_dtype(GGML_TYPE_BF16));

    gguf_reader & g = m->io;
    if (!load_pi_config(g, ckpt_path, 0, m->cfg) || !resolve_num_steps("pi05", opts, m->cfg.num_steps))
        return nullptr;
    const Config & cfg = m->cfg;
    m->quantile_norm = g.has("pi05.norm_mode") && g.str("pi05.norm_mode") == "quantiles";
    std::printf("vla(pi05): hidden=%lld inter=%lld heads=%lldq/%lldkv x%lld n_layers=%lld "
                "expert_h=%lld expert_inter=%lld chunk=%lld steps=%d real_state=%lld real_action=%lld "
                "max_len=%lld matmul_weights=%s\n",
                (long long) cfg.hidden, (long long) cfg.intermediate, (long long) cfg.n_q_heads,
                (long long) cfg.n_kv_heads, (long long) cfg.head_dim, (long long) cfg.n_layers,
                (long long) cfg.expert_h, (long long) cfg.expert_inter, (long long) cfg.n_suffix,
                cfg.num_steps, (long long) cfg.real_state_dim, (long long) cfg.real_action_dim,
                (long long) cfg.n_lang, m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    {
        const Backend b = backend_init("vla(pi05)", m->n_threads);
        if (!b.handle) {
            return nullptr;
        }
        m->backend = b.handle;
    }

    // The SigLIP tower is now bundled in the ckpt GGUF; mmproj_path is ignored.
    (void) mmproj_path;
    if (!m->vis.load(g, cfg.n_img))
        return nullptr;

    {
        ggml_init_params wp = { (size_t) 16*1024*1024, nullptr,  true };
        m->ctx_weights = ggml_init(wp);
        if (!m->ctx_weights) {
            std::fprintf(stderr, "vla(pi05): ggml_init(ctx_weights) failed\n");
            return nullptr;
        }
    }
    WeightLoader L("pi05", g, m->ctx_weights, m->matmul_type);

    m->vis.declare(L);

    m->pl.declare(L, "vlm", cfg.n_layers, false);

    m->ex_layers.resize(cfg.n_layers);
    for (int64_t i=0; i<cfg.n_layers; ++i) {
        GemmaLayerW & w = m->ex_layers[i];
        w.Wq    = L.gemm("aex.blk.%lld.attn_q.weight",   (long long)i);
        w.Wk    = L.gemm("aex.blk.%lld.attn_k.weight",   (long long)i);
        w.Wv    = L.gemm("aex.blk.%lld.attn_v.weight",   (long long)i);
        w.Wo    = L.gemm("aex.blk.%lld.attn_o.weight",   (long long)i);
        w.Wgate = L.gemm("aex.blk.%lld.ffn_gate.weight", (long long)i);
        w.Wup   = L.gemm("aex.blk.%lld.ffn_up.weight",   (long long)i);
        w.Wdown = L.gemm("aex.blk.%lld.ffn_down.weight", (long long)i);
    }

    m->W_ain  = L.f32("action_in_proj.weight");   m->b_ain  = L.f32("action_in_proj.bias");
    m->W_aout = L.f32("action_out_proj.weight");  m->b_aout = L.f32("action_out_proj.bias");

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;

    std::printf("vla(pi05): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0));

    if (!bake_adarms(g, *m) || !load_stats(g, *m))
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

    int64_t n_img_tokens = 0;
    if (in.precomputed_img_emb) {
        if (in.n_img_views < 1) {
            std::fprintf(stderr, "vla(pi05): precomputed_img_emb set but n_img_views=%d\n", in.n_img_views);
            return {};
        }
        n_img_tokens = (int64_t) in.n_img_views*cfg.n_img;
    } else {
        if (in.n_images < 1 || !in.images) {
            std::fprintf(stderr, "vla(pi05): predict: no images and no precomputed_img_emb\n");
            return {};
        }
        n_img_tokens = (int64_t) in.n_images*vis.n_tokens;
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
    const size_t max_nodes = (size_t) 64*n_layers*(num_steps+1) + 1024;
    const bool built = main_graph.ensure(backend, mkey,
                                         ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false),
                                         [&](ggml_context * C, MainIO & gio) -> ggml_cgraph * {
    ggml_tensor * t_image_emb = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_img_tokens); ggml_set_input(t_image_emb);
    ggml_tensor * t_lang_emb  = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_lang);       ggml_set_input(t_lang_emb);
    ggml_tensor * t_prefix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_prefix);                ggml_set_input(t_prefix_pos);
    ggml_tensor * t_x0        = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);           ggml_set_input(t_x0);
    ggml_tensor * t_suffix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_suf);                   ggml_set_input(t_suffix_pos);

    const float lang_scale = (float) std::sqrt((double) hidden_pl);
    ggml_tensor * prefix_embs = ggml_concat(C, t_image_emb, ggml_scale(C, t_lang_emb, lang_scale),  1);

    std::vector<ggml_tensor *> cK(n_layers), cV(n_layers);
    {
        ggml_tensor * h = prefix_embs;
        for (int64_t i=0; i<n_layers; ++i) {
            h = gemma_layer(C, pl.blk[i], h, t_prefix_pos, cfg, n_prefix,
                            nullptr, nullptr, nullptr, &cK[i], &cV[i]);
        }
        (void) h;
    }

    ggml_tensor * x_t = t_x0;
    for (int step=0; step<num_steps; ++step) {
        auto mod = [&](int64_t j) {
            return ggml_view_1d(C, ada_mod, 3*hidden_ex, step*ada_mod->nb[2]+j*ada_mod->nb[1]);
        };

        ggml_tensor * h = ggml_add(C, ggml_mul_mat(C, W_ain, x_t), b_ain);
        for (int64_t i=0; i<n_layers; ++i) {
            h = build_expert_layer(C, ex_layers[i], h, t_suffix_pos, mod(2*i), mod(2*i+1), cfg, n_suf,
                                   cK[i], cV[i]);
        }

        ggml_tensor * h_final = build_adarms(C, h, mod(2*n_layers), hidden_ex, cfg.rms_eps, nullptr);
        ggml_tensor * v_t = ggml_add(C, ggml_mul_mat(C, W_aout, h_final), b_aout);
        x_t = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
    }
    ggml_tensor * x_final = x_t;
    ggml_set_output(x_final);

    gio.t_image_emb=t_image_emb; gio.t_lang_emb=t_lang_emb; gio.t_prefix_pos=t_prefix_pos;
    gio.t_x0=t_x0; gio.t_suffix_pos=t_suffix_pos; gio.x_final=x_final;

    ggml_cgraph * gf = ggml_new_graph_custom(C, max_nodes, false);
    ggml_build_forward_expand(gf, x_final);
    return gf;
    });
    if (!built) { std::fprintf(stderr, "vla(pi05): main graph build failed\n"); return {}; }

    MainIO & gio = main_graph.io();
    ggml_cgraph * gf = main_graph.graph();
    ggml_tensor * t_image_emb = gio.t_image_emb, * t_lang_emb = gio.t_lang_emb;
    ggml_tensor * t_prefix_pos = gio.t_prefix_pos, * t_x0 = gio.t_x0;
    ggml_tensor * t_suffix_pos = gio.t_suffix_pos, * x_final = gio.x_final;

    if (in.precomputed_img_emb) {
        ggml_backend_tensor_set(t_image_emb, in.precomputed_img_emb, 0, ggml_nbytes(t_image_emb));
    } else {
        const int64_t K = vis.n_tokens, H = hidden_pl, grid = vis.image_size/vis.patch_size;
        ggml_context * VC = vision_scratch.reset((size_t) 128*1024*1024);
        if (!VC) { std::fprintf(stderr, "vla(pi05): ggml_init(vision ctx) failed\n"); return {}; }
        ggml_tensor * t_px = ggml_new_tensor_3d(VC, GGML_TYPE_F32, vis.image_size, vis.image_size, 3); ggml_set_input(t_px);
        ggml_tensor * h = vis.vit.build(VC, vis.vit.embed_conv(VC, t_px, vis.patch_size, grid), K);
        // PaliGemma projector: linear (+ optional bias).
        ggml_tensor * vit_emb = linear(VC, vis.proj_w, vis.proj_b, h);
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
            if (!preprocess_image_chw("pi05", in.images[v], vis.image_size, chw)) { return {}; }
            ggml_backend_tensor_set(t_px, chw.data(), 0, ggml_nbytes(t_px));
            graph_unique_names(vg);
            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(pi05): vision compute failed (view %d)\n", v);
                return {};
            }
            ggml_tensor * dst = ggml_view_2d(VC, t_image_emb, H, K, t_image_emb->nb[1], (size_t) v * K * t_image_emb->nb[1]);
            if (ggml_backend_view_init(dst) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(pi05): image embedding view failed (view %d)\n", v);
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
        std::vector<int32_t> sp(n_suf);    for (int64_t i=0; i<n_suf; ++i)    sp[i] = (int32_t) (n_prefix+i);
        ggml_backend_tensor_set(t_suffix_pos, sp.data(), 0, ggml_nbytes(t_suffix_pos));
    }
    {
        std::vector<float> x0h;
        init_noise(in, (size_t) max_ad * chunk, x0h);
        ggml_backend_tensor_set(t_x0, x0h.data(), 0, ggml_nbytes(t_x0));
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

    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now()-t0).count();
    return out;
}

}
