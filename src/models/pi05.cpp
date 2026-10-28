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
#include "cuda/vla_cuda_ops.h"
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
#include "foldquant.h"
#include "layers/fq_linear.h"
#include "act_dtype.h"
#include "layers/fused_ew.h"
#include "env_flag.h"
#include "cuda/vla_cuda_ops.h"
#include "sycl/vla_sycl_ops.h"
#if defined(GGML_USE_SYCL) && defined(VLA_SYCL_FMHA)
#include "sycl/vla_sycl_fmha.h"
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace vla {

struct Pi05ModelArch : public ModelArchBase {
    Pi05ModelArch() : ModelArchBase(Arch::PI05) {}
    ~Pi05ModelArch() override;

    std::vector<float> predict(const Inputs& in) override;
    bool encode_images(const ImageView* images, int n_images, std::vector<float>& out) override;

    /// Runs the vision tower + projector on every view in one graph; returns its
    /// F32 [hidden, n_images*n_tokens] output, valid until the next call, or null.
    const std::vector<ggml_tensor *> * vision_forward(const ImageView* images, int n_images);

    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    ggml_context *        ctx_const   = nullptr;
    ggml_backend_buffer_t const_buf   = nullptr;
    // Vision tower for all views at once, kept while the view count holds.
    struct VisIO {
        // Pixels, views in order: one [S, S, 3, nv] input when the tower runs
        // batched, else one [S, S, 3, 1] input per view.
        std::vector<ggml_tensor *> t_px;
        // F32 [hidden, nv*K] embeddings, views in order: one output when the
        // tower runs batched, else one per view. Real tensors, not views or rows
        // of a shared output: OpenVINO writes back neither.
        std::vector<ggml_tensor *> emb;
    };
    graph_cache<int64_t, VisIO> vis_graph;
    // Language segment padded up to a multiple of this (0 = exact length), with
    // the padding masked out as attention keys. Set in create (16 on OpenVINO).
    int64_t               lang_bucket = 0;
    // Activation dtype of the GEMMs (--act-dtype bf16 on CUDA/SYCL). The expert's
    // residual stream, its adaRMS gating and the flow-matching state stay F32.
    ggml_type             act_type    = GGML_TYPE_F32;
    // The expert's adaRMS and gated residuals as single fused nodes
    // (layers/fused_ew.h). Only where a backend claims them: SYCL.
    bool                  fused_ew    = false;
    FusedAdarmsSpec       adarms_spec;
    FusedGateResSpec      gateres_spec;
    // K/V written in place into per-layer F16 buffers, q|k|v and gate|up GEMMs
    // fused (modules/gemma_expert.h: KvSlot, gemma_declare_float). SYCL's fused
    // attention only; VLA_PI05_NO_KV_INPLACE=1 keeps the concat graph.
    bool                  kv_inplace  = false;
    // FoldQuant GGUF on SYCL with the sycl-tla kernels: fused q/k/v + RoPE nodes
    // (layers/fq_rope_attn.h), the expert's adaRMS in its act nodes and the
    // prefix K/V read in place as F16 (layers/fmha_kv.h).
    bool                  fq_fast     = false;

    struct MainKey {
        int64_t n_img=-1, n_lang=-1, nsteps=-1;
        bool operator==(const MainKey & o) const {
            return n_img==o.n_img && n_lang==o.n_lang && nsteps==o.nsteps;
        }
    };
    struct MainIO {
        // One prefix input, image rows then language rows: no CONCAT node.
        // t_img_rows views its image rows (filled from the vision graph),
        // t_view_rows each view's share of them.
        ggml_tensor *t_prefix=nullptr,*t_img_rows=nullptr,*t_prefix_pos=nullptr;
        std::vector<ggml_tensor *> t_view_rows;
        ggml_tensor *t_x0=nullptr,*t_suffix_pos=nullptr,*x_final=nullptr;
        ggml_tensor *t_prefix_mask=nullptr,*t_suffix_mask=nullptr;  // lang_bucket only
    };
    graph_cache<MainKey, MainIO> main_graph;
    // Opened once at load: reopening per predict re-parses the whole GGUF header.
    gguf_reader           io{"pi05"};
    ggml_type             matmul_type = GGML_TYPE_BF16;

    PaliVision                  vis;
    bool                        fused_ops = false;
    GemmaStack                  pl;
    std::vector<GemmaLayerW>    ex_layers;

    ggml_tensor * W_ain  = nullptr, * b_ain  = nullptr;
    ggml_tensor * W_aout = nullptr, * b_aout = nullptr;

    ggml_tensor * ada_mod = nullptr;
    // fq_fast: per (step, slot) the 1+scale and shift of ada_mod as tensors of
    // their own, for the act nodes the adaRMS rides in (fq_act_mod). Slot j:
    // 2*layer (attn), 2*layer+1 (ffn), 2*n_layers (out).
    std::vector<ggml_tensor *> ada_scale1, ada_shift;

    std::vector<float> action_mean, action_std;
    std::vector<float> action_q01, action_q99;
    bool quantile_norm = false;

    int n_threads = default_cpu_threads();
};

namespace {

// On a view-taking flash attention (vla::fa_takes_views()), pi05's three
// attention sites (SigLIP, the VLM prefix and the action expert) run fused
// straight off the [hd, H, S] views. Otherwise the unfused graph is used.
bool use_fused_attention() { return vla::flash_attn_enabled() && vla::fa_takes_views(); }

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

// pi0.5's FoldQuant expert on SYCL (fq_fast): this step's adaRMS slots as
// 1 + scale and shift tensors, and the layer's prefix K/V as F16 (null: concat).
struct FqExpertIn {
    ggml_tensor * scale1_attn = nullptr, * shift_attn = nullptr;
    ggml_tensor * scale1_ffn  = nullptr, * shift_ffn  = nullptr;
    ggml_tensor * K16 = nullptr, * V16 = nullptr;
};

ggml_tensor * build_expert_layer(
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions, ggml_tensor * mod_attn, ggml_tensor * mod_ffn,
        const Config & cfg, int64_t seq,
        ggml_tensor * cached_K, ggml_tensor * cached_V, ggml_tensor * mask = nullptr,
        ggml_type at = GGML_TYPE_F32, const FusedAdarmsSpec * fa = nullptr, const FusedGateResSpec * fg = nullptr,
        KvSlot * kv = nullptr, const FqExpertIn * fqx = nullptr) {
    const int64_t h = cfg.expert_h;
    if (fqx && w.act_mod_qkv.has_shift && w.act_mod_gu.has_shift && w.fq_o && w.fq_down) {
        // FoldQuant on SYCL: each adaRMS rides in its act node (rms * (1 + scale)
        // + shift, then quantized), each gated residual in its GEMM epilogue
        // (x + y * gate).
        ggml_tensor * gate_attn = ggml_view_1d(ctx, mod_attn, h, (size_t) 2*h * sizeof(float));
        ggml_tensor * gate_ffn  = ggml_view_1d(ctx, mod_ffn,  h, (size_t) 2*h * sizeof(float));
        ggml_tensor * xq_attn = fq_act_mod(ctx, &w.act_mod_qkv, w.fq_qkv, x_in, fqx->scale1_attn, fqx->shift_attn);
        ggml_tensor * h1 = gemma_attn(ctx, w, nullptr, positions, cfg, seq, cached_K, cached_V, mask, nullptr, nullptr,
                                      GGML_TYPE_F32, use_fused_attention(), nullptr, x_in, nullptr, fqx->K16, fqx->V16,
                                      xq_attn, gate_attn);
        ggml_tensor * xq_ffn = fq_act_mod(ctx, &w.act_mod_gu, w.fq_gu, h1, fqx->scale1_ffn, fqx->shift_ffn);
        return gemma_mlp(ctx, w, nullptr, GGML_TYPE_F32, nullptr, h1, xq_ffn, gate_ffn);
    }
    static const bool q8_fuse = !env_flag("VLA_PI05_NO_Q8_FUSE");
    if (q8_fuse && fa && kv && w.q_qkv && w.q_o && w.q_gu && w.q_down) {
        // W8A8: adaRMS writes each GEMM's INT8 input, and the gated residuals
        // x + out * gate are the o and down GEMMs' epilogues (F32 out).
        auto gate = [&](ggml_tensor * mod) {
            return ggml_view_1d(ctx, mod, h, 2 * h * ggml_element_size(mod));
        };
        ggml_tensor * h1 = gemma_attn_kv(ctx, w, fused_adarms(ctx, x_in, mod_attn, GGML_TYPE_I8, *fa), positions,
                                         cfg, seq, *kv, mask, at, x_in, gate(mod_attn), GGML_TYPE_F32);
        return gemma_mlp_q8(ctx, w, fused_adarms(ctx, h1, mod_ffn, GGML_TYPE_I8, *fa), h1, at, gate(mod_ffn),
                            GGML_TYPE_F32);
    }
    if (fa && fg && !w.fq_q && !w.fq_qkv && !w.fq_gate && !w.fq_gu) {
        // One node per chain: adaRMS straight into the GEMMs' input type, and
        // out * gate + x reading the GEMM output in whatever type it has.
        ggml_tensor * xn = fused_adarms(ctx, x_in, mod_attn, at, *fa);
        ggml_tensor * o  = gemma_attn(ctx, w, xn, positions, cfg, seq, cached_K, cached_V, mask,
                                      nullptr, nullptr, at, use_fused_attention(), nullptr, nullptr, kv);
        ggml_tensor * h1 = fused_gateres(ctx, x_in, o, mod_attn, *fg);
        ggml_tensor * xm = fused_adarms(ctx, h1, mod_ffn, at, *fa);
        return fused_gateres(ctx, h1, gemma_mlp(ctx, w, xm, at), mod_ffn, *fg);
    }

    if (!mask && !kv && gemma_fused(w, at, use_fused_attention(), seq, seq + (cached_K ? cached_K->ne[2] : 0))) {
        auto part = [&](ggml_tensor * mod, int j) {
            return ggml_view_1d(ctx, mod, h, (size_t) j*h*sizeof(float));
        };
        // FoldQuant act nodes run the adaRMS themselves, or read its output in F32.
        const bool bf16 = !w.fq_q;
        auto adarms = [&](ggml_tensor * x, ggml_tensor * mod, const FqLinear & s) {
            ggml_tensor * xq = fq_act_norm(ctx, s, x, part(mod, 0), part(mod, 1), FusedKind::RmsModulate, cfg.rms_eps);
            return xq ? xq : fused_rms_modulate(ctx, x, part(mod, 0), part(mod, 1), cfg.rms_eps, bf16);
        };
        ggml_tensor * o_out = gemma_attn_fused(ctx, w, adarms(x_in, mod_attn, w.fq_q),
            positions, cfg, seq, cached_K, cached_V, nullptr, nullptr, nullptr);
        ggml_tensor * h1 = fused_gate_res(ctx, o_out, part(mod_attn, 2), x_in);
        ggml_tensor * mlp_out = gemma_mlp_fused(ctx, w, adarms(h1, mod_ffn, w.fq_gate));
        return fused_gate_res(ctx, mlp_out, part(mod_ffn, 2), h1);
    }

    ggml_tensor * gate_attn = nullptr;
    ggml_tensor * x_norm = build_adarms(ctx, x_in, mod_attn, h, cfg.rms_eps, &gate_attn);
    // FoldQuant: the act nodes quantize the adaRMS outputs; the residuals are
    // gated (out * gate + x), so they stay out of the GEMM epilogues.
    // mm_act does not convert its input: a BF16 GEMM must be handed BF16 rows,
    // but adaRMS produces F32, whose bits would be misread as BF16.
    ggml_tensor * o_out  = gemma_attn(ctx, w, as_type(ctx, x_norm, at), positions, cfg, seq, cached_K, cached_V, mask,
                                      nullptr, nullptr, at, use_fused_attention(), x_norm, nullptr, kv,
                                      fqx ? fqx->K16 : nullptr, fqx ? fqx->V16 : nullptr);
    o_out = as_type(ctx, o_out, GGML_TYPE_F32);

    ggml_tensor * h1 = ggml_add(ctx, x_in, ggml_mul(ctx, o_out, gate_attn));

    ggml_tensor * gate_ffn = nullptr;
    ggml_tensor * x_norm_mlp = build_adarms(ctx, h1, mod_ffn, h, cfg.rms_eps, &gate_ffn);
    ggml_tensor * mlp_out    = as_type(ctx, gemma_mlp(ctx, w, as_type(ctx, x_norm_mlp, at), at, x_norm_mlp),
                                       GGML_TYPE_F32);

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

    const size_t n_slot = m.fq_fast ? (size_t) ns*nj : 0;
    ggml_init_params p = { (1 + 2*n_slot)*ggml_tensor_overhead(), nullptr, true };
    m.ctx_const = ggml_init(p);
    if (!m.ctx_const) {
        std::fprintf(stderr, "vla(pi05): ggml_init(ctx_const) failed\n");
        return false;
    }
    m.ada_mod   = ggml_new_tensor_3d(m.ctx_const, GGML_TYPE_F32, 3*h, nj, ns);
    m.ada_scale1.resize(n_slot);
    m.ada_shift.resize(n_slot);
    for (size_t k=0; k<n_slot; ++k) {
        m.ada_scale1[k] = ggml_new_tensor_1d(m.ctx_const, GGML_TYPE_F32, h);
        m.ada_shift[k]  = ggml_new_tensor_1d(m.ctx_const, GGML_TYPE_F32, h);
    }
    m.const_buf = alloc_weights(m.ctx_const, m.backend);
    if (!m.const_buf) {
        std::fprintf(stderr, "vla(pi05): alloc_weights(adaRMS table) failed\n");
        return false;
    }
    ggml_backend_tensor_set(m.ada_mod, table.data(), 0, ggml_nbytes(m.ada_mod));
    std::vector<float> scale1((size_t) h);
    for (size_t k=0; k<n_slot; ++k) {
        const float * row = table.data() + k*3*h;
        for (int64_t c=0; c<h; ++c) scale1[c] = 1.0f + row[c];
        ggml_backend_tensor_set(m.ada_scale1[k], scale1.data(), 0, ggml_nbytes(m.ada_scale1[k]));
        ggml_backend_tensor_set(m.ada_shift[k], row + h, 0, ggml_nbytes(m.ada_shift[k]));
    }
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
    FoldQuantSpec fq = foldquant_parse(g, "pi05");
#ifdef GGML_USE_SYCL
    // W4A4 sites on the INT4 DPAS GEMMs (nibble-packed activations), the prefix's
    // and the expert's: at 50 rows they beat oneDNN's s8 x s4 by 2-3x. Without
    // them (no sycl-tla in the build, or VLA_FQ_NO_S4GEMM=1) INT4 activations
    // are stored one code per byte, which oneDNN reads without an unpack.
    {
#ifdef VLA_SYCL_S4GEMM
        const bool s4 = !env_flag("VLA_FQ_NO_S4GEMM");
#else
        const bool s4 = false;
#endif
        fq.llm.s4gemm = fq.action.s4gemm = s4;
        fq.llm.wide4  = fq.action.wide4  = !s4;
    }
#endif
    m->quantile_norm = g.has("pi05.norm_mode") && g.str("pi05.norm_mode") == "quantiles";
    std::printf("vla(pi05): hidden=%lld inter=%lld heads=%lldq/%lldkv x%lld n_layers=%lld "
                "expert_h=%lld expert_inter=%lld chunk=%lld steps=%d real_state=%lld real_action=%lld "
                "max_len=%lld matmul_weights=%s\n",
                (long long) cfg.hidden, (long long) cfg.intermediate, (long long) cfg.n_q_heads,
                (long long) cfg.n_kv_heads, (long long) cfg.head_dim, (long long) cfg.n_layers,
                (long long) cfg.expert_h, (long long) cfg.expert_inter, (long long) cfg.n_suffix,
                cfg.num_steps, (long long) cfg.real_state_dim, (long long) cfg.real_action_dim,
                (long long) cfg.n_lang, m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    bool is_sycl = false;
    {
        const Backend b = backend_init("vla(pi05)", m->n_threads);
        if (!b.handle) {
            return nullptr;
        }
        m->backend = b.handle;
        if (!foldquant_check_backend("vla(pi05)", b, fq, opts.weight_dtype.has_value()))
            return nullptr;
        is_sycl = b.is_sycl;
#if defined(GGML_USE_SYCL) && defined(VLA_SYCL_FMHA) && defined(VLA_FQ_SYCL_DNNL)
        // FoldQuant's fused q/k/v + RoPE nodes and the sycl-tla attention.
        if (b.is_sycl && fq.present && !fq.llm.dequant && !fq.action.dequant && fmha_kv_enabled()) {
            m->fq_fast = true;
            sycl_register_fmha_ops();
        }
#endif

        // pi0.5 writes the robot state into its prompt, so the language length
        // changes every step. A backend that compiles one model per graph shape
        // can pad the segment to a multiple of VLA_PI05_LANG_BUCKET tokens, with
        // the pad masked out as keys; suffix positions still count from the real
        // prefix length, so the result is the same. 0 keeps it exact. OpenVINO
        // compiles and keeps one model per shape, so it defaults to 16.
        m->lang_bucket = b.is_openvino ? 16 : 0;
        // ggml-openvino compiles graphs that do not start with a MUL_MAT in
        // ACCURACY mode, which pins them to f32; pi0.5 holds at f16, so use the
        // precision the GPU is configured with. An explicit setting wins.
        if (b.is_openvino)
            setenv_default("GGML_OPENVINO_EXEC_MODE", "performance");
        // Fused elementwise nodes on SYCL, the backend that claims them.
        // VLA_PI05_NO_FUSE=1 keeps the per-op graph.
        // Off when the backend declines them (VLA_SYCL_NO_FUSED=1).
        if (b.is_sycl && !fq.present && !env_flag("VLA_PI05_NO_FUSE")) {
            sycl_register_fused_ops();
            m->fused_ew        = fused_ew_registered();
            m->adarms_spec.eps = cfg.rms_eps;
            if (m->fused_ew)
                std::printf("vla(pi05): fused expert elementwise ops on (VLA_PI05_NO_FUSE=1 to disable)\n");
        }
        if (const char * e = std::getenv("VLA_PI05_LANG_BUCKET"); e && *e)
            m->lang_bucket = std::max(0L, std::strtol(e, nullptr, 10));
        if (m->lang_bucket > 0)
            std::printf("vla(pi05): language segment padded to a multiple of %lld tokens (masked)\n",
                        (long long) m->lang_bucket);

        // On SYCL, flash attention runs on oneDNN's fused SDPA, so it is the
        // default there unless --flash-attn was given either way.
        if (b.is_sycl && !opts.flash_attn.has_value() && vla::fa_takes_views()) {
            vla::set_flash_attn(true);
            std::printf("vla(pi05): flash attention = on (SYCL default; --flash-attn 0 to disable)\n");
        }

        // BF16 activations, as pi0 has them: BF16-resident weights and
        // a GPU BF16 GEMM path (src/cuda/vla_cuda_bf16.cu, src/sycl/vla_sycl_bf16.cpp).
        // Not with FoldQuant, whose integer sites take F32 activations.
        if (opts.act_dtype.value_or(GGML_TYPE_F32) == GGML_TYPE_BF16) {
            if ((b.is_cuda || b.is_sycl) && m->matmul_type == GGML_TYPE_BF16 && !fq.present) {
                m->act_type = GGML_TYPE_BF16;
                cuda_register_bf16_ops();
                sycl_register_bf16_ops();
                std::printf("vla(pi05): activations = BF16 (expert residual stream F32)\n");
            } else {
                std::fprintf(stderr, "vla(pi05): --act-dtype bf16 ignored - needs CUDA or SYCL, BF16 weights "
                                     "and no FoldQuant\n");
            }
        }
        // CUDA, F32 activations: the bit-exact fused nodes (layers/fused_ops.h).
        if (m->act_type == GGML_TYPE_F32)
            m->fused_ops = enable_fused_ops("pi05", b.is_cuda);
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

    m->kv_inplace = use_fused_attention() && !fq.present && !env_flag("VLA_PI05_NO_KV_INPLACE");
    if (m->kv_inplace)
        std::printf("vla(pi05): K/V in place in F16, fused q|k|v and gate|up GEMMs (VLA_PI05_NO_KV_INPLACE=1 to disable)\n");
    // W8A8 GEMMs (layers/qlinear.h) on the in-place SYCL graph: VLA_PI05_W8A8 =
    // any of vit, vlm, expert (comma-separated), or all (1).
    bool q8_vlm = false, q8_aex = false, q8_vit = false;
    if (const char * e = std::getenv("VLA_PI05_W8A8"); e && *e && std::string(e) != "0") {
        const std::string v = e;
        auto has = [&](const char * k) { return v == "all" || v == "1" || v.find(k) != std::string::npos; };
        q8_vlm = has("vlm");
        q8_aex = has("expert");
        q8_vit = has("vit");
        if (fq.present && is_sycl) {
            // A FoldQuant checkpoint brings its own LM and expert sites; only
            // SigLIP, BF16 in the file, can take the W8A8 GEMMs (fused blocks).
            if (q8_vlm || q8_aex)
                std::fprintf(stderr, "vla(pi05): VLA_PI05_W8A8 vlm/expert ignored - the FoldQuant sites are used\n");
            q8_vlm = q8_aex = false;
            if (q8_vit)
                std::printf("vla(pi05): W8A8 GEMMs (per-token INT8 activations): SigLIP\n");
        } else if (!m->kv_inplace || !m->fused_ew) {
            std::fprintf(stderr, "vla(pi05): VLA_PI05_W8A8 ignored - needs SYCL with fused attention, no FoldQuant\n");
            q8_vlm = q8_aex = q8_vit = false;
        } else {
            std::printf("vla(pi05): W8A8 GEMMs (per-token INT8 activations): %s%s%s\n", q8_vit ? "SigLIP " : "",
                        q8_vlm ? "VLM " : "", q8_aex ? "expert" : "");
        }
    }
    // INT4 expert weights on top of W8A8 (W4A8): VLA_PI05_W4=expert, with
    // VLA_PI05_W4_GROUP K-elements per scale (default 128; 0 = per channel).
    int64_t aex_w4_group = 0;
    if (const char * e = std::getenv("VLA_PI05_W4"); q8_aex && e && std::string(e) == "expert") {
        const char * g = std::getenv("VLA_PI05_W4_GROUP");
        aex_w4_group = g && *g ? std::strtol(g, nullptr, 10) : 128;
        if (aex_w4_group == 0) aex_w4_group = -1;  // per channel: quant_i4 takes K
        std::printf("vla(pi05): expert weights INT4 (W4A8), %s\n",
                    aex_w4_group > 0 ? ("group " + std::to_string(aex_w4_group)).c_str() : "per channel");
    }
    m->vis.vit.enc.cfg.flash_attn = use_fused_attention();
    // Fused SigLIP blocks (modules/encoder.h, EncCfg::fused): SYCL, BF16, fused
    // attention. A FoldQuant GGUF takes them too: its integer sites are the LM's
    // and the expert's, and its vision tower is BF16 like any other checkpoint's.
    // VLA_PI05_NO_VIT_FUSE=1 keeps the per-op graph.
    const bool vit_bf16 = m->act_type == GGML_TYPE_BF16 ||
                          (fq.present && is_sycl && m->matmul_type == GGML_TYPE_BF16 && !env_flag("VLA_PI05_NO_FUSE"));
    m->vis.vit.enc.cfg.fused = is_sycl && vit_bf16 && use_fused_attention() && !env_flag("VLA_PI05_NO_VIT_FUSE");
    if (m->vis.vit.enc.cfg.fused && !m->fused_ew) {
        sycl_register_fused_ops();
        m->vis.vit.enc.cfg.fused = fused_ew_registered();
    }
    // The fused blocks run in BF16; the per-op tower in F32, which takes every
    // view in one graph (its BF16 form is single-view).
    m->vis.vit.enc.cfg.act  = m->vis.vit.enc.cfg.fused ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    m->vis.vit.enc.cfg.int8 = m->vis.vit.enc.cfg.fused && q8_vit;
    if (m->vis.vit.enc.cfg.fused)
        std::printf("vla(pi05): fused SigLIP blocks (VLA_PI05_NO_VIT_FUSE=1 to disable)\n");
    m->vis.declare(L);
    m->vis.vit.enc.cfg.fused_ops = m->fused_ops;   // SigLIP blocks are EncStack blocks

    m->pl.declare(L, "vlm", cfg.n_layers, false, fq.present ? &fq.llm : nullptr, cfg.rms_eps, m->kv_inplace, q8_vlm);
    // fq_fast: the prefix q/k/v GEMM, RoPE and F16 packing as one node
    // (gemma_attn). VLA_FQ_NO_ROPE_PACK=1 keeps them apart (A/B runs).
    if (m->fq_fast && cfg.n_kv_heads == 1 && cfg.head_dim == 256 && !env_flag("VLA_FQ_SYCL_GEMM") &&
        !env_flag("VLA_FQ_NO_ROPE_PACK"))
        for (GemmaLayerW & w : m->pl.blk)
            if (w.fq_qkv) {
                w.rope_pack.magic     = FQ_ROPE_PACK_MAGIC;
                w.rope_pack.gemm      = &w.fq_qkv.gemm;
                w.rope_pack.hd        = (int) cfg.head_dim;
                w.rope_pack.nq        = (int) cfg.n_q_heads;
                w.rope_pack.freq_base = cfg.rope_freq_base;
                w.rope_pack.scale     = 1.0f / std::sqrt((float) cfg.head_dim);
            }

    m->ex_layers.resize(cfg.n_layers);
    for (int64_t i=0; i<cfg.n_layers; ++i) {
        GemmaLayerW & w = m->ex_layers[i];
        const long long ii = (long long) i;
        if (fq.present) {
            // FoldQuant sites (action recipe): the adaRMS output is quantized per
            // site group with the shipped SmoothQuant vector (ascale), no gamma.
            const FqModuleSpec & a = fq.action;
            const std::string b = "aex.blk." + std::to_string(ii);
            gemma_declare_fq_qkv(L, w, a, b, nullptr, 0.0f);
            w.fq_o    = fq_declare_linear(L, a, "o",      false, nullptr, 0.0f, "aex.blk.%lld.attn_o",   ii);
            gemma_declare_fq_gateup(L, w, a, b, nullptr, 0.0f);
            w.fq_down = fq_declare_linear(L, a, "down",   false, nullptr, 0.0f, "aex.blk.%lld.ffn_down", ii);
            gemma_link_glu_act(w);
            if (!w.fq_q != !w.fq_k || !w.fq_q != !w.fq_v || !w.fq_gate != !w.fq_up)
                L.fail("FoldQuant: an expert layer's q/k/v (and gate/up) must all be INT or all float");
        }
        gemma_declare_float(L, w, "aex.blk." + std::to_string(ii), m->kv_inplace, q8_aex, aex_w4_group);
        if (m->fq_fast && w.fq_qkv) {
            // q/k/v GEMM, RoPE and the cached-KV attention as one node (gemma_attn).
            // VLA_FQ_NO_ROPE_ATTN=1 keeps them apart (A/B runs).
            if (cfg.n_kv_heads == 1 && cfg.head_dim == 256 && !env_flag("VLA_FQ_NO_ROPE_ATTN") &&
                !env_flag("VLA_FQ_SYCL_GEMM")) {
                w.rope_attn.magic     = FQ_ROPE_ATTN_MAGIC;
                w.rope_attn.gemm      = &w.fq_qkv.gemm;
                w.rope_attn.hd        = (int) cfg.head_dim;
                w.rope_attn.nq        = (int) cfg.n_q_heads;
                w.rope_attn.freq_base = cfg.rope_freq_base;
                w.rope_attn.scale     = 1.0f / std::sqrt((float) cfg.head_dim);
            }
            // The adaRMS fused into the q/k/v and gate/up act nodes
            // (build_expert_layer). VLA_FQ_NO_ADA=1 keeps them apart (A/B runs).
            if (w.fq_gu && !env_flag("VLA_FQ_NO_ADA")) {
                for (auto * a : { &w.act_mod_qkv, &w.act_mod_gu }) {
                    *a = (a == &w.act_mod_qkv) ? w.fq_qkv.act : w.fq_gu.act;
                    a->has_gamma = true;
                    a->has_shift = true;
                    a->eps       = cfg.rms_eps;
                }
            }
        }
    }
    // Expert attention over contiguous keys: the prefix pack leaves chunk free K/V
    // rows per layer, the expert writes its own there. Above 192 query rows,
    // fq_rope_attn (two oneDNN GEMMs); below, where those lose to the split
    // fmha_kv, fused_rope_kv and the oneDNN SDPA (fq_kv_sdpa; VLA_FQ_NO_KV_SDPA=1
    // keeps fmha_kv). Only when every layer takes the fused nodes.
    // VLA_FQ_NO_KV_TAIL=1: off.
    {
        const bool big = cfg.n_suffix * cfg.n_q_heads > 192;
        bool all = m->fq_fast && !env_flag("VLA_FQ_NO_KV_TAIL") && (big || !env_flag("VLA_FQ_NO_KV_SDPA"));
        for (int64_t i=0; i<cfg.n_layers && all; ++i)
            all = m->pl.blk[i].rope_pack.magic == FQ_ROPE_PACK_MAGIC && m->ex_layers[i].fq_qkv &&
                  (!big || m->ex_layers[i].rope_attn.magic == FQ_ROPE_ATTN_MAGIC);
        if (all && !big) {
            sycl_register_fused_ops();
            all = fused_ew_registered();   // fused_rope_kv declined (VLA_SYCL_NO_FUSED=1): fmha_kv
        }
        if (all) {
            for (int64_t i=0; i<cfg.n_layers; ++i) {
                m->pl.blk[i].rope_pack.extra    = (int) cfg.n_suffix;
                m->ex_layers[i].rope_attn.extra = (int) cfg.n_suffix;
                m->ex_layers[i].fq_kv_sdpa      = !big;
            }
            if (!big)
                std::printf("vla(pi05): expert K/V in the prefix pack, fused attention (VLA_FQ_NO_KV_SDPA=1 to disable)\n");
        }
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

const std::vector<ggml_tensor *> * Pi05ModelArch::vision_forward(const ImageView* images, int n_images) {
    const int64_t K = vis.n_tokens, grid = vis.image_size/vis.patch_size, S = vis.image_size;
    const ggml_type vt = vis.vit.enc.cfg.act;
    const bool built = vis_graph.ensure(backend, (int64_t) n_images, (size_t) 128*1024*1024,
                                        [&](ggml_context * VC, VisIO & vio) -> ggml_cgraph * {
        // Tower + PaliGemma projector (linear, optional bias) over nv views of
        // pixels: F32 [hidden, nv*K]. The patch embed (conv_2d) runs in F32, the
        // tower in vt.
        auto tower = [&](ggml_tensor * px, int64_t nv) {
            ggml_tensor * x = vis.vit.embed_conv(VC, px, vis.patch_size, grid);   // [hidden, K, nv]
            if (vis.vit.enc.cfg.fused)   // the fused blocks take the views as nv*K rows
                x = ggml_reshape_2d(VC, x, x->ne[0], K * nv);
            ggml_tensor * proj = mm_act(VC, vis.proj_w, vis.vit.build(VC, as_type(VC, x, vt), K, nv), vt);
            if (vis.proj_b)
                proj = ggml_add(VC, proj, vis.proj_b);
            proj = as_type(VC, proj, GGML_TYPE_F32);
            if (proj->op == GGML_OP_VIEW || proj->op == GGML_OP_RESHAPE)
                proj = ggml_cont(VC, proj);
            ggml_set_output(proj);
            vio.emb.push_back(proj);
            return proj;
        };
        ggml_cgraph * vg = ggml_new_graph_custom(VC, 8192, false);
        auto new_px = [&](int64_t nv) {
            ggml_tensor * t = ggml_new_tensor_4d(VC, GGML_TYPE_F32, S, S, 3, nv);
            ggml_set_input(t);
            vio.t_px.push_back(t);
            return t;
        };
        vio.t_px.clear();
        vio.emb.clear();
        if (vis.vit.enc.cfg.fused || n_images == 1) {
            // The fused blocks take every view in one pass (nv * K rows per GEMM).
            ggml_build_forward_expand(vg, tower(new_px(n_images), n_images));
        } else {
            // The per-op tower one view at a time (oneDNN's SDPA batches only
            // BF16), an input and an output per view: no CONCAT. Not views of
            // one input: OpenVINO translates a view of a graph input as the
            // whole input.
            for (int v=0; v<n_images; ++v)
                ggml_build_forward_expand(vg, tower(new_px(1), 1));
        }
        return vg;
    });
    if (!built) { std::fprintf(stderr, "vla(pi05): vision graph build failed\n"); return nullptr; }
    const VisIO & vio = vis_graph.io();
    ggml_cgraph * vg = vis_graph.graph();

    const size_t per_view = (size_t) 3*S*S;
    std::vector<float> chw, px(per_view * n_images);
    for (int v=0; v<n_images; ++v) {
        if (!preprocess_image_chw("pi05", images[v], vis.image_size, chw)) return nullptr;
        std::copy(chw.begin(), chw.begin() + per_view, px.begin() + per_view * v);
    }
    size_t off = 0;
    for (ggml_tensor * t : vio.t_px) {
        ggml_backend_tensor_set(t, px.data() + off, 0, ggml_nbytes(t));
        off += (size_t) ggml_nelements(t);
    }
    graph_unique_names(vg);
    if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(pi05): vision compute failed\n");
        return nullptr;
    }
    return &vio.emb;
}

bool Pi05ModelArch::encode_images(const ImageView* images, int n_images, std::vector<float>& out) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    stats = Stats{};
    const std::vector<ggml_tensor *> * embs = vision_forward(images, n_images);
    const bool ok = embs != nullptr;
    if (ok) {
        size_t n = 0;
        for (ggml_tensor * e : *embs) n += (size_t) ggml_nelements(e);
        out.resize(n);
        n = 0;
        for (ggml_tensor * e : *embs) {
            ggml_backend_tensor_get(e, out.data() + n, 0, ggml_nbytes(e));
            n += (size_t) ggml_nelements(e);
        }
    }
    stats.ms_vision = stats.ms_total = std::chrono::duration<float, std::milli>(clk::now()-t0).count();
    if (!ok) out.clear();
    return ok;
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
    const int64_t n_prefix = n_img_tokens+n_lang;      // real tokens
    // Graph sizes: the language segment padded to the bucket (== real without one).
    const int64_t n_lang_g   = lang_bucket > 0 ? (n_lang + lang_bucket - 1) / lang_bucket * lang_bucket : n_lang;
    const int64_t n_prefix_g = n_img_tokens + n_lang_g;
    const bool    padded     = lang_bucket > 0;

    std::vector<int32_t> lang_ids(in.lang_tokens, in.lang_tokens+n_lang);
    std::vector<float> lang_rows((size_t) n_lang_g * hidden_pl, 0.0f);
    {
        if (!io.fetch_rows_f32("token_embd.weight", lang_ids, lang_rows.data(), hidden_pl)) return {};
    }

    // Prefix + expert graph depends only on the token counts and step count.
    const MainKey mkey{ n_img_tokens, n_lang_g, num_steps };
    const size_t max_nodes = (size_t) 64*n_layers*(num_steps+1) + 1024;
    const bool built = main_graph.ensure(backend, mkey,
                                         ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false),
                                         [&](ggml_context * C, MainIO & gio) -> ggml_cgraph * {
    // One prefix input, image rows then language rows (the latter scaled on the
    // host): no CONCAT node, which also keeps the graph SYCL-graph compatible.
    ggml_tensor * t_prefix    = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_prefix_g);   ggml_set_input(t_prefix);
    ggml_tensor * t_prefix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_prefix_g);              ggml_set_input(t_prefix_pos);
    ggml_tensor * t_prefix_mask = nullptr, * t_suffix_mask = nullptr;
    if (padded) {
        t_prefix_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_prefix_g, n_prefix_g);        ggml_set_input(t_prefix_mask);
        t_suffix_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_prefix_g + n_suf, n_suf);     ggml_set_input(t_suffix_mask);
    }
    ggml_tensor * t_x0        = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);           ggml_set_input(t_x0);
    ggml_tensor * t_suffix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_suf);                   ggml_set_input(t_suffix_pos);

    ggml_tensor * prefix_embs = as_type(C, t_prefix, act_type);

    std::vector<ggml_tensor *> cK(n_layers), cV(n_layers);
    std::vector<KvSlot>        kv(kv_inplace ? n_layers : 0);
    for (KvSlot & sl : kv) {
        // Leaves the allocator places up front; output-flagged so it never
        // frees or reuses them while the steps still read them.
        sl.k = ggml_new_tensor_3d(C, GGML_TYPE_F16, cfg.head_dim, cfg.n_kv_heads, n_prefix_g + n_suf);
        sl.v = ggml_new_tensor_3d(C, GGML_TYPE_F16, cfg.head_dim, cfg.n_kv_heads, n_prefix_g + n_suf);
        ggml_set_output(sl.k);
        ggml_set_output(sl.v);
    }
    {
        ggml_tensor * h = prefix_embs;
        for (int64_t i=0; i<n_layers; ++i) {
            if (kv_inplace && pl.blk[i].q_qkv && !env_flag("VLA_PI05_NO_Q8_FUSE"))
                h = gemma_layer_q8(C, pl.blk[i], h, t_prefix_pos, cfg, n_prefix_g, kv[i], t_prefix_mask, act_type,
                                   pl.rms_spec);
            else
                h = gemma_layer(C, pl.blk[i], h, t_prefix_pos, cfg, n_prefix_g,
                                nullptr, nullptr, t_prefix_mask, &cK[i], &cV[i], act_type, use_fused_attention(),
                                kv_inplace ? &kv[i] : nullptr);
        }
        (void) h;
    }
    // fq_fast: the expert reads the prefix K/V as F16, cast here once per layer
    // (or packed so already by fq_rope_pack), not concatenated every step.
    const bool kv16 = fq_fast && use_fused_attention() && !padded;
    std::vector<ggml_tensor *> cK16(n_layers, nullptr), cV16(n_layers, nullptr);
    if (kv16)
        for (int64_t i=0; i<n_layers; ++i) {
            if (cK[i]->type == GGML_TYPE_F16) {   // already F16 [hd, seq, 1] (fq_rope_pack)
                cK16[i] = cK[i];
                cV16[i] = cV[i];
                continue;
            }
            cK16[i] = ggml_cast(C, ggml_permute(C, cK[i], 0, 2, 1, 3), GGML_TYPE_F16);
            cV16[i] = ggml_cast(C, ggml_permute(C, cV[i], 0, 2, 1, 3), GGML_TYPE_F16);
        }

    ggml_tensor * x_t = t_x0;
    for (int step=0; step<num_steps; ++step) {
        auto mod = [&](int64_t j) {
            return ggml_view_1d(C, ada_mod, 3*hidden_ex, step*ada_mod->nb[2]+j*ada_mod->nb[1]);
        };

        ggml_tensor * h = ggml_add(C, ggml_mul_mat(C, W_ain, x_t), b_ain);
        for (int64_t i=0; i<n_layers; ++i) {
            FqExpertIn fqx;
            if (kv16) {
                const size_t k = (size_t) step*(2*n_layers+1) + (size_t) (2*i);
                fqx = { ada_scale1[k], ada_shift[k], ada_scale1[k+1], ada_shift[k+1], cK16[i], cV16[i] };
            }
            h = build_expert_layer(C, ex_layers[i], h, t_suffix_pos, mod(2*i), mod(2*i+1), cfg, n_suf,
                                   cK[i], cV[i], t_suffix_mask, act_type,
                                   fused_ew ? &adarms_spec : nullptr, fused_ew ? &gateres_spec : nullptr,
                                   kv_inplace ? &kv[i] : nullptr, kv16 ? &fqx : nullptr);
        }

        ggml_tensor * h_final = build_adarms(C, h, mod(2*n_layers), hidden_ex, cfg.rms_eps, nullptr);
        ggml_tensor * v_t = ggml_add(C, ggml_mul_mat(C, W_aout, h_final), b_aout);
        x_t = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
    }
    ggml_tensor * x_final = x_t;
    ggml_set_output(x_final);

    gio.t_prefix=t_prefix; gio.t_prefix_pos=t_prefix_pos;
    // Not a graph node: initialized as a view once the graph is allocated.
    gio.t_img_rows=ggml_view_2d(C, t_prefix, hidden_pl, n_img_tokens, t_prefix->nb[1], 0);
    gio.t_view_rows.clear();
    for (int64_t r = 0; r + vis.n_tokens <= n_img_tokens; r += vis.n_tokens)
        gio.t_view_rows.push_back(ggml_view_2d(C, t_prefix, hidden_pl, vis.n_tokens, t_prefix->nb[1],
                                               (size_t) r * t_prefix->nb[1]));
    gio.t_x0=t_x0; gio.t_suffix_pos=t_suffix_pos; gio.x_final=x_final;
    gio.t_prefix_mask=t_prefix_mask; gio.t_suffix_mask=t_suffix_mask;

    ggml_cgraph * gf = ggml_new_graph_custom(C, max_nodes, false);
    ggml_build_forward_expand(gf, x_final);
    return gf;
    });
    if (!built) { std::fprintf(stderr, "vla(pi05): main graph build failed\n"); return {}; }

    MainIO & gio = main_graph.io();
    ggml_cgraph * gf = main_graph.graph();
    ggml_tensor * t_prefix = gio.t_prefix, * t_image_emb = gio.t_img_rows;
    if (!t_image_emb->buffer && ggml_backend_view_init(t_image_emb) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(pi05): image rows view failed\n");
        return {};
    }
    ggml_tensor * t_prefix_pos = gio.t_prefix_pos, * t_x0 = gio.t_x0;
    ggml_tensor * t_suffix_pos = gio.t_suffix_pos, * x_final = gio.x_final;

    if (in.precomputed_img_emb) {
        ggml_backend_tensor_set(t_image_emb, in.precomputed_img_emb, 0, ggml_nbytes(t_image_emb));
    } else {
        const auto tv0 = clk::now();
        const std::vector<ggml_tensor *> * embs = vision_forward(in.images, in.n_images);
        if (!embs) return {};
        if (embs->size() == 1) {
            ggml_backend_tensor_copy(embs->front(), t_image_emb);
        } else {
            for (size_t v = 0; v < embs->size(); ++v) {
                ggml_tensor * rows = gio.t_view_rows[v];
                if (!rows->buffer && ggml_backend_view_init(rows) != GGML_STATUS_SUCCESS) {
                    std::fprintf(stderr, "vla(pi05): image rows view failed\n");
                    return {};
                }
                ggml_backend_tensor_copy((*embs)[v], rows);
            }
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now()-tv0).count();
    }
    {   // Gemma scales the token embeddings by sqrt(hidden): x * s in F32, as ggml_scale did on device.
        const float lang_scale = (float) std::sqrt((double) hidden_pl);
        for (float & v : lang_rows) v *= lang_scale;
    }
    ggml_backend_tensor_set(t_prefix, lang_rows.data(), (size_t) n_img_tokens * t_prefix->nb[1],
                            lang_rows.size() * sizeof(float));
    {
        // Pad tokens continue the count; they are masked as keys, so their
        // positions only shape their own (unused) query rows.
        std::vector<int32_t> pp(n_prefix_g); for (int64_t i=0; i<n_prefix_g; ++i) pp[i] = (int32_t) i;
        ggml_backend_tensor_set(t_prefix_pos, pp.data(), 0, ggml_nbytes(t_prefix_pos));
        std::vector<int32_t> sp(n_suf);    for (int64_t i=0; i<n_suf; ++i)    sp[i] = (int32_t) (n_prefix+i);
        ggml_backend_tensor_set(t_suffix_pos, sp.data(), 0, ggml_nbytes(t_suffix_pos));
    }
    if (padded) {
        const float NEG = -INFINITY;
        std::vector<float> pm((size_t) n_prefix_g * n_prefix_g, 0.0f);
        for (int64_t q = 0; q < n_prefix_g; ++q)
            for (int64_t k = n_prefix; k < n_prefix_g; ++k) pm[(size_t) q * n_prefix_g + k] = NEG;
        ggml_backend_tensor_set(gio.t_prefix_mask, pm.data(), 0, ggml_nbytes(gio.t_prefix_mask));
        const int64_t nk = n_prefix_g + n_suf;
        std::vector<float> sm((size_t) nk * n_suf, 0.0f);
        for (int64_t q = 0; q < n_suf; ++q)
            for (int64_t k = n_prefix; k < n_prefix_g; ++k) sm[(size_t) q * nk + k] = NEG;
        ggml_backend_tensor_set(gio.t_suffix_mask, sm.data(), 0, ggml_nbytes(gio.t_suffix_mask));
    }
    {
        std::vector<float> x0h;
        init_noise(in, (size_t) max_ad * chunk, x0h);
        ggml_backend_tensor_set(t_x0, x0h.data(), 0, ggml_nbytes(t_x0));
    }
    graph_unique_names(gf);
    const auto ti0 = clk::now();
    static const bool split_host = env_flag("VLA_PI05_HOST_SPLIT");
    ggml_status st;
    if (split_host) {
        st = ggml_backend_graph_compute_async(backend, gf);
        const float ms_submit = std::chrono::duration<float, std::milli>(clk::now()-ti0).count();
        ggml_backend_synchronize(backend);
        std::fprintf(stderr, "vla(pi05): main graph %d nodes: host submit %.1f ms, done %.1f ms\n", ggml_graph_n_nodes(gf),
                     ms_submit, std::chrono::duration<float, std::milli>(clk::now()-ti0).count());
    } else {
        st = ggml_backend_graph_compute(backend, gf);
    }
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
