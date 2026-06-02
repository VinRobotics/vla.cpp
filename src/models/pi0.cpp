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

#include "clip.h"
#include "mtmd.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "gguf.h"

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

struct gguf_reader {
    gguf_context * gctx     = nullptr;
    ggml_context * meta_ctx = nullptr;
    FILE *         fp       = nullptr;
    size_t         data_off = 0;

    bool open(const std::string & path) {
        gguf_init_params p{};
        p.no_alloc = true;
        p.ctx      = &meta_ctx;
        gctx = gguf_init_from_file(path.c_str(), p);
        if (!gctx) {
            std::fprintf(stderr, "vla(pi0): gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "vla(pi0): fopen failed for %s\n", path.c_str());
            return false;
        }
        data_off = gguf_get_data_offset(gctx);
        return true;
    }
    ~gguf_reader() {
        if (fp)       std::fclose(fp);
        if (gctx)     gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
    }
    gguf_reader() = default;
    gguf_reader(const gguf_reader &) = delete;
    gguf_reader & operator=(const gguf_reader &) = delete;

    bool has_key(const char * k) const { return gguf_find_key(gctx, k) >= 0; }
    uint32_t    u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }
    float       f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }
    double      f64(const char * k) const { return gguf_get_val_f64(gctx, gguf_find_key(gctx, k)); }
    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }

    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    bool read_raw(const char * name, void * buf) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) { std::fprintf(stderr, "vla(pi0): missing tensor %s\n", name); return false; }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }

    std::vector<uint8_t> read_convert(const char * name, ggml_type target, bool gemma_norm) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(pi0): missing tensor %s\n", name); return {}; }
        const int64_t n = ggml_nelements(t);

        std::vector<float> f32(n);
        if (t->type == GGML_TYPE_F32) {
            if (!read_raw(name, f32.data())) return {};
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp(n);
            if (!read_raw(name, tmp.data())) return {};
            ggml_bf16_to_fp32_row(tmp.data(), f32.data(), n);
        } else {
            std::fprintf(stderr, "vla(pi0): tensor %s has unsupported type %d\n", name, (int) t->type);
            return {};
        }
        if (gemma_norm) for (int64_t i = 0; i < n; ++i) f32[i] += 1.0f;

        if (target == GGML_TYPE_F32) {
            std::vector<uint8_t> out(n * sizeof(float));
            std::memcpy(out.data(), f32.data(), out.size());
            return out;
        }
        if (target == GGML_TYPE_BF16) {
            std::vector<uint8_t> out(n * sizeof(ggml_bf16_t));
            ggml_fp32_to_bf16_row(f32.data(), reinterpret_cast<ggml_bf16_t *>(out.data()), n);
            return out;
        }
        std::fprintf(stderr, "vla(pi0): unsupported resident type %d for %s\n", (int) target, name);
        return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids,
                        float * dst, int64_t cols) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(pi0): missing tensor %s\n", name); return false; }
        if (t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) {
            std::fprintf(stderr, "vla(pi0): %s shape unfit for row-fetch\n", name); return false;
        }
        const int64_t rows = t->ne[1];
        const int64_t id   = gguf_find_tensor(gctx, name);
        const size_t  base = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t  elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
        const size_t  rb   = (size_t) cols * elsz;
        std::vector<uint8_t> row(rb);
        for (size_t k = 0; k < row_ids.size(); ++k) {
            const int32_t r = row_ids[k];
            if (r < 0 || r >= rows) {
                std::fprintf(stderr, "vla(pi0): row %d out of range for %s\n", r, name); return false;
            }
            if (std::fseek(fp, (long) (base + (size_t) r * rb), SEEK_SET) != 0) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

struct GemmaLayerW {
    ggml_tensor * ln_in   = nullptr;
    ggml_tensor * Wq      = nullptr;
    ggml_tensor * Wk      = nullptr;
    ggml_tensor * Wv      = nullptr;
    ggml_tensor * Wo      = nullptr;
    ggml_tensor * ln_post = nullptr;
    ggml_tensor * Wgate   = nullptr;
    ggml_tensor * Wup     = nullptr;
    ggml_tensor * Wdown   = nullptr;
};

bool is_gemma_norm(const std::string & name) { return name.find("norm.weight") != std::string::npos; }

std::vector<float> sinusoidal_time_emb(double t, int64_t dim, double min_p, double max_p) {
    const int64_t half = dim / 2;
    std::vector<float> out(dim);
    for (int64_t i = 0; i < half; ++i) {
        const double frac   = (half == 1) ? 0.0 : double(i) / double(half - 1);
        const double period = min_p * std::pow(max_p / min_p, frac);
        const double s      = (2.0 * M_PI / period) * t;
        out[i]        = (float) std::sin(s);
        out[half + i] = (float) std::cos(s);
    }
    return out;
}

bool ends_with(const std::string & s, const char * sfx) {
    const size_t n = std::strlen(sfx);
    return s.size() >= n && s.compare(s.size() - n, n, sfx) == 0;
}

}

struct Pi0ModelArch : public ModelArchBase {
    Pi0ModelArch() : ModelArchBase(Arch::PI0) {}
    ~Pi0ModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    clip_ctx *            cctx        = nullptr;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    std::string           ckpt_path_;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    std::vector<GemmaLayerW> pl_layers;

    std::vector<GemmaLayerW> ex_layers;
    ggml_tensor * ex_final_norm = nullptr;

    ggml_tensor * W_sp = nullptr,  * b_sp = nullptr;
    ggml_tensor * W_ain = nullptr, * b_ain = nullptr;
    ggml_tensor * W_at1 = nullptr, * b_at1 = nullptr;
    ggml_tensor * W_at2 = nullptr, * b_at2 = nullptr;
    ggml_tensor * W_aout = nullptr,* b_aout = nullptr;

    std::vector<float> state_mean, state_std, action_mean, action_std;

    std::mt19937 rng{std::random_device{}()};
    int n_threads = 4;
};

namespace {

ggml_tensor * build_gemma_layer(
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions,
        const Config & cfg, int64_t seq, float rope_base,
        ggml_tensor * cached_K, ggml_tensor * cached_V, ggml_tensor * mask,
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
        return ggml_rope_ext(ctx, t, positions,  nullptr,
                              (int) hd,  GGML_ROPE_TYPE_NEOX,  0,
                             rope_base,  1.f,  0.f,  1.f,
                              32.f,  1.f);
    };
    ggml_tensor * q_rope = rope_call(q_h);
    ggml_tensor * k_rope = rope_call(k_h);

    if (k_out) *k_out = k_rope;
    if (v_out) *v_out = v_h;

    ggml_tensor * K_full = k_rope;
    ggml_tensor * V_full = v_h;
    if (cached_K && cached_V) {
        K_full = ggml_concat(ctx, cached_K, k_rope,  2);
        V_full = ggml_concat(ctx, cached_V, v_h,     2);
    }

    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));

    ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    const float scale = 1.f / std::sqrt((float) hd);
    ggml_tensor * attn = ggml_soft_max_ext(ctx, kq, mask, scale,  0.f);
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

ggml_tensor * build_embed_suffix(ggml_context * ctx, const Pi0ModelArch & m,
                                 ggml_tensor * state, ggml_tensor * x, ggml_tensor * time_bcast) {
    ggml_tensor * state_emb       = ggml_add(ctx, ggml_mul_mat(ctx, m.W_sp,  state), m.b_sp);
    ggml_tensor * action_emb      = ggml_add(ctx, ggml_mul_mat(ctx, m.W_ain, x),     m.b_ain);
    ggml_tensor * action_time_in  = ggml_concat(ctx, action_emb, time_bcast, 0);
    ggml_tensor * mlp1            = ggml_add(ctx, ggml_mul_mat(ctx, m.W_at1, action_time_in), m.b_at1);
    ggml_tensor * mlp1_silu       = ggml_silu(ctx, mlp1);
    ggml_tensor * action_time_emb = ggml_add(ctx, ggml_mul_mat(ctx, m.W_at2, mlp1_silu), m.b_at2);
    ggml_tensor * state_emb_2d    = ggml_reshape_2d(ctx, state_emb, state_emb->ne[0], 1);
    return ggml_concat(ctx, state_emb_2d, action_time_emb, 1);
}

bool load_config(const gguf_reader & g, Config & cfg) {
    auto need = [&](const char * k) {
        if (!g.has_key(k)) { std::fprintf(stderr, "vla(pi0): gguf missing key %s\n", k); return false; }
        return true;
    };
    for (const char * k : {"pi0.hidden", "pi0.intermediate", "pi0.n_q_heads", "pi0.n_kv_heads",
                           "pi0.head_dim", "pi0.n_layers", "pi0.expert_h", "pi0.expert_inter",
                           "pi0.chunk_size", "pi0.num_steps", "pi0.max_state_dim", "pi0.max_action_dim",
                           "pi0.real_state_dim", "pi0.real_action_dim", "pi0.tokenizer_max_length",
                           "pi0.min_period", "pi0.max_period"}) {
        if (!need(k)) return false;
    }
    cfg = Config{};
    cfg.hidden          = g.u32("pi0.hidden");
    cfg.intermediate    = g.u32("pi0.intermediate");
    cfg.n_q_heads       = g.u32("pi0.n_q_heads");
    cfg.n_kv_heads      = g.u32("pi0.n_kv_heads");
    cfg.head_dim        = g.u32("pi0.head_dim");
    cfg.n_layers        = g.u32("pi0.n_layers");
    cfg.expert_h        = g.u32("pi0.expert_h");
    cfg.expert_inter    = g.u32("pi0.expert_inter");
    cfg.n_suffix        = g.u32("pi0.chunk_size");
    cfg.num_steps       = g.u32("pi0.num_steps");
    cfg.max_state_dim   = g.u32("pi0.max_state_dim");
    cfg.max_action_dim  = g.u32("pi0.max_action_dim");
    cfg.real_state_dim  = g.u32("pi0.real_state_dim");
    cfg.real_action_dim = g.u32("pi0.real_action_dim");
    cfg.n_lang          = g.u32("pi0.tokenizer_max_length");
    cfg.min_period      = g.f64("pi0.min_period");
    cfg.max_period      = g.f64("pi0.max_period");

    cfg.n_state         = 1;
    cfg.n_img           = 256;
    cfg.q_full_dim      = cfg.n_q_heads  * cfg.head_dim;
    cfg.kv_full_dim     = cfg.n_kv_heads * cfg.head_dim;
    cfg.self_attn_every_n = 0;
    cfg.rms_eps         = g.has_key("pi0.rms_norm_eps") ? g.f32("pi0.rms_norm_eps") : 1e-6f;
    cfg.norm_eps        = g.has_key("pi0.norm_eps")     ? g.f32("pi0.norm_eps")     : 1e-8f;
    cfg.rope_mode       = GGML_ROPE_TYPE_NEOX;
    cfg.rope_n_dims     = (int) cfg.head_dim;
    cfg.rope_freq_base  = g.has_key("pi0.rope_theta") ? (float) g.f64("pi0.rope_theta") : 10000.f;
    cfg.n_prefix        = 0;
    cfg.n_full          = 0;
    return true;
}

bool load_stats(gguf_reader & g, Pi0ModelArch & m) {
    const auto & cfg = m.cfg;
    m.state_mean .assign(cfg.real_state_dim,  0.f);
    m.state_std  .assign(cfg.real_state_dim,  1.f);
    m.action_mean.assign(cfg.real_action_dim, 0.f);
    m.action_std .assign(cfg.real_action_dim, 1.f);
    auto read1d = [&](const char * name, std::vector<float> & dst) {
        const ggml_tensor * t = g.meta(name);
        if (!t) { std::printf("vla(pi0): %s missing - identity\n", name); return; }
        if (t->ne[0] != (int64_t) dst.size()) { std::printf("vla(pi0): %s dim mismatch - identity\n", name); return; }
        if (!g.read_raw(name, dst.data())) std::printf("vla(pi0): %s read failed - identity\n", name);
    };
    read1d("state_mean",  m.state_mean);
    read1d("state_std",   m.state_std);
    read1d("action_mean", m.action_mean);
    read1d("action_std",  m.action_std);
    return true;
}

}

Pi0ModelArch::~Pi0ModelArch() {
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
    if (cctx)        clip_free(cctx);
}

std::unique_ptr<ModelArchBase> pi0_create(const std::string& mmproj_path,
                                          const std::string& ckpt_path,
                                          const std::string& config_path) {
    (void) config_path;

    if (!ends_with(ckpt_path, ".gguf")) {
        std::fprintf(stderr,
            "vla(pi0): ckpt must be a GGUF produced by scripts/convert_pi0_to_gguf.py "
            "(got '%s'); direct .safetensors loading for π₀ is not yet supported\n",
            ckpt_path.c_str());
        return nullptr;
    }

    auto m = std::make_unique<Pi0ModelArch>();
    m->ckpt_path_ = ckpt_path;
    m->matmul_type = std::getenv("VLA_PI0_F32_WEIGHTS") ? GGML_TYPE_F32 : GGML_TYPE_BF16;

    gguf_reader g;
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has_key("pi0.architecture") || g.str("pi0.architecture") != "pi0") {
        std::fprintf(stderr, "vla(pi0): '%s' is not a π₀ GGUF (pi0.architecture missing/wrong)\n",
                     ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, m->cfg)) return nullptr;
    const Config & cfg = m->cfg;
    std::printf("vla(pi0): hidden=%lld inter=%lld heads=%lldq/%lldkv x%lld n_layers=%lld "
                "expert_h=%lld expert_inter=%lld chunk=%lld steps=%d real_state=%lld real_action=%lld "
                "matmul_weights=%s\n",
                (long long) cfg.hidden, (long long) cfg.intermediate, (long long) cfg.n_q_heads,
                (long long) cfg.n_kv_heads, (long long) cfg.head_dim, (long long) cfg.n_layers,
                (long long) cfg.expert_h, (long long) cfg.expert_inter, (long long) cfg.n_suffix,
                cfg.num_steps, (long long) cfg.real_state_dim, (long long) cfg.real_action_dim,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init( 0);
    if (m->backend) { m->is_cuda = true; std::printf("vla(pi0): backend = CUDA (device 0)\n"); }
    else            { std::fprintf(stderr, "vla(pi0): ggml_backend_cuda_init failed; falling back to CPU\n"); }
#endif
    {
        const unsigned hw = std::thread::hardware_concurrency();
        m->n_threads = (hw == 0) ? 4 : (int) std::min(hw, 8u);
    }
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(pi0): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(pi0): backend = CPU (%d threads)\n", m->n_threads);
    }

    {
        clip_context_params cp = {};
        cp.use_gpu           = m->is_cuda;
        cp.flash_attn_type   = m->is_cuda ? CLIP_FLASH_ATTN_TYPE_AUTO : CLIP_FLASH_ATTN_TYPE_DISABLED;
        cp.image_min_tokens  = -1;
        cp.image_max_tokens  = -1;
        cp.warmup            = m->is_cuda;
        cp.cb_eval           = nullptr;
        cp.cb_eval_user_data = nullptr;
        clip_init_result r = clip_init(mmproj_path.c_str(), cp);
        if (!r.ctx_v) {
            std::fprintf(stderr, "vla(pi0): clip_init failed for %s\n", mmproj_path.c_str());
            return nullptr;
        }
        m->cctx = r.ctx_v;
        const int img_sz  = clip_get_image_size(m->cctx);
        const int mm_embd = clip_n_mmproj_embd(m->cctx);
        if (img_sz != 224 || mm_embd != (int) cfg.hidden) {
            std::fprintf(stderr, "vla(pi0): mmproj mismatch (image_size=%d mmproj_embd=%d; want 224 / %lld)\n",
                         img_sz, mm_embd, (long long) cfg.hidden);
            return nullptr;
        }
    }

    {
        ggml_init_params wp = {  (size_t) 16 * 1024 * 1024,  nullptr,  true };
        m->ctx_weights = ggml_init(wp);
        if (!m->ctx_weights) { std::fprintf(stderr, "vla(pi0): ggml_init(ctx_weights) failed\n"); return nullptr; }
    }
    ggml_context * W = m->ctx_weights;
    std::vector<ggml_tensor *> weights;

    auto mk = [&](const char * name, ggml_type type, int n_dims, const int64_t * ne) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(pi0): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, type, n_dims, ne);
        ggml_set_name(t, name);
        weights.push_back(t);
        return t;
    };

    auto mk_mm = [&](const char * name) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(pi0): missing tensor %s\n", name); return nullptr; }
        return mk(name, m->matmul_type, GGML_MAX_DIMS, gt->ne);
    };
    auto mk_f32 = [&](const char * name) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(pi0): missing tensor %s\n", name); return nullptr; }
        return mk(name, GGML_TYPE_F32, GGML_MAX_DIMS, gt->ne);
    };

    auto load_layer = [&](const char * tower, int i, GemmaLayerW & lw) -> bool {
        char b[256];
        auto suf = [&](const char * s) { std::snprintf(b, sizeof(b), "%s.blk.%d.%s", tower, i, s); return b; };
        lw.ln_in   = mk_f32(suf("attn_norm.weight"));
        lw.Wq      = mk_mm (suf("attn_q.weight"));
        lw.Wk      = mk_mm (suf("attn_k.weight"));
        lw.Wv      = mk_mm (suf("attn_v.weight"));
        lw.Wo      = mk_mm (suf("attn_o.weight"));
        lw.ln_post = mk_f32(suf("ffn_norm.weight"));
        lw.Wgate   = mk_mm (suf("ffn_gate.weight"));
        lw.Wup     = mk_mm (suf("ffn_up.weight"));
        lw.Wdown   = mk_mm (suf("ffn_down.weight"));
        return lw.ln_in && lw.Wq && lw.Wk && lw.Wv && lw.Wo && lw.ln_post && lw.Wgate && lw.Wup && lw.Wdown;
    };

    m->pl_layers.resize(cfg.n_layers);
    m->ex_layers.resize(cfg.n_layers);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        if (!load_layer("vlm", (int) i, m->pl_layers[i])) return nullptr;
        if (!load_layer("aex", (int) i, m->ex_layers[i])) return nullptr;
    }
    m->ex_final_norm = mk_f32("aex.output_norm.weight");
    m->W_sp   = mk_f32("state_proj.weight");          m->b_sp   = mk_f32("state_proj.bias");
    m->W_ain  = mk_f32("action_in_proj.weight");      m->b_ain  = mk_f32("action_in_proj.bias");
    m->W_at1  = mk_f32("action_time_mlp_in.weight");  m->b_at1  = mk_f32("action_time_mlp_in.bias");
    m->W_at2  = mk_f32("action_time_mlp_out.weight"); m->b_at2  = mk_f32("action_time_mlp_out.bias");
    m->W_aout = mk_f32("action_out_proj.weight");     m->b_aout = mk_f32("action_out_proj.bias");
    for (ggml_tensor * t : weights) if (!t) { std::fprintf(stderr, "vla(pi0): weight tensor creation failed\n"); return nullptr; }
    if (!m->ex_final_norm || !m->W_sp || !m->b_sp || !m->W_ain || !m->b_ain ||
        !m->W_at1 || !m->b_at1 || !m->W_at2 || !m->b_at2 || !m->W_aout || !m->b_aout) {
        std::fprintf(stderr, "vla(pi0): failed to wire projection / norm tensors\n"); return nullptr;
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(pi0): ggml_backend_alloc_ctx_tensors failed (out of memory?)\n"); return nullptr; }
    for (ggml_tensor * t : weights) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type, is_gemma_norm(t->name));
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(pi0): upload size mismatch for %s (%zu vs %zu)\n",
                         t->name, bytes.size(), ggml_nbytes(t));
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(pi0): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0));

    if (!load_stats(g, *m)) return nullptr;
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
    const int64_t n_suf     = 1 + chunk;
    const int64_t n_layers  = cfg.n_layers;
    const int64_t max_sd    = cfg.max_state_dim;
    const int64_t max_ad    = cfg.max_action_dim;
    const int     num_steps = cfg.num_steps;
    const float   dt        = -1.0f / (float) num_steps;
    const float   rope_base = cfg.rope_freq_base;

    std::vector<float> img_emb_host;
    int64_t n_img_tokens = 0;
    if (in.precomputed_img_emb) {
        n_img_tokens = (int64_t) in.n_img_views * cfg.n_img;
        img_emb_host.assign(in.precomputed_img_emb,
                            in.precomputed_img_emb + (size_t) n_img_tokens * hidden_pl);
    } else {
        if (in.n_images < 1 || !in.images) {
            std::fprintf(stderr, "vla(pi0): predict: no images and no precomputed_img_emb\n");
            return {};
        }
        const int    img_sz  = clip_get_image_size(cctx);
        const size_t per_pix = (size_t) 3 * img_sz * img_sz;
        const size_t per_out = clip_embd_nbytes_by_img(cctx, img_sz, img_sz) / sizeof(float);
        img_emb_host.resize(per_out * (size_t) in.n_images);
        std::vector<float> hwc(per_pix);
        const auto tv0 = clk::now();
        for (int v = 0; v < in.n_images; ++v) {
            const ImageView & view = in.images[v];
            if (view.w != img_sz || view.h != img_sz) {
                std::fprintf(stderr, "vla(pi0): image[%d] is %dx%d; π₀ requires %dx%d\n",
                             v, view.w, view.h, img_sz, img_sz);
                return {};
            }
            if (view.format == PixelFormat::U8) {
                const uint8_t * src = static_cast<const uint8_t *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) hwc[i] = (float) src[i] / 127.5f - 1.0f;
            } else {
                const float * src = static_cast<const float *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) hwc[i] = src[i] * 2.0f - 1.0f;
            }

            if (!clip_encode_float_image(cctx, n_threads, hwc.data(), img_sz, img_sz,
                                         img_emb_host.data() + (size_t) v * per_out)) {
                std::fprintf(stderr, "vla(pi0): clip_encode_float_image failed (view %d)\n", v);
                return {};
            }
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now() - tv0).count();
        n_img_tokens = (int64_t) ((per_out / (size_t) hidden_pl) * (size_t) in.n_images);
    }

    if (in.n_lang < 1 || !in.lang_tokens) {
        std::fprintf(stderr, "vla(pi0): predict: empty lang_tokens\n");
        return {};
    }
    const int64_t n_lang   = in.n_lang;
    const int64_t n_prefix = n_img_tokens + n_lang;
    const int64_t n_total  = n_prefix + n_suf;

    std::vector<int32_t> lang_ids(in.lang_tokens, in.lang_tokens + n_lang);
    std::vector<float> lang_rows((size_t) n_lang * hidden_pl);
    {
        gguf_reader g;
        if (!g.open(ckpt_path_)) return {};
        if (!g.fetch_rows_f32("token_embd.weight", lang_ids, lang_rows.data(), hidden_pl)) return {};
    }

    ggml_init_params cp = {  (size_t) 64 * 1024 * 1024,  nullptr,  true };
    ggml_context * C = ggml_init(cp);
    if (!C) { std::fprintf(stderr, "vla(pi0): ggml_init(ctx_compute) failed\n"); return {}; }

    ggml_tensor * t_image_emb = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_img_tokens); ggml_set_input(t_image_emb);
    ggml_tensor * t_lang_emb  = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_lang);       ggml_set_input(t_lang_emb);
    ggml_tensor * t_prefix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_prefix);                ggml_set_input(t_prefix_pos);
    ggml_tensor * t_state     = ggml_new_tensor_1d(C, GGML_TYPE_F32, max_sd);                  ggml_set_input(t_state);
    ggml_tensor * t_x0        = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);           ggml_set_input(t_x0);
    ggml_tensor * t_suffix_pos= ggml_new_tensor_1d(C, GGML_TYPE_I32, n_suf);                   ggml_set_input(t_suffix_pos);
    ggml_tensor * t_full_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_total, n_suf);          ggml_set_input(t_full_mask);
    std::vector<ggml_tensor *> t_time(num_steps);
    for (int s = 0; s < num_steps; ++s) {
        t_time[s] = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_ex, chunk);
        ggml_set_input(t_time[s]);
    }

    const float lang_scale = (float) std::sqrt((double) hidden_pl);
    ggml_tensor * prefix_embs = ggml_concat(C, t_image_emb, ggml_scale(C, t_lang_emb, lang_scale),  1);

    std::vector<ggml_tensor *> cK(n_layers), cV(n_layers);
    {
        ggml_tensor * h = prefix_embs;
        for (int64_t i = 0; i < n_layers; ++i) {
            h = build_gemma_layer(C, pl_layers[i], h, t_prefix_pos, cfg, n_prefix, rope_base,
                                   nullptr,  nullptr,  nullptr,
                                  &cK[i], &cV[i]);
        }
        (void) h;
    }

    ggml_tensor * x_t = t_x0;
    std::vector<ggml_tensor *> v_steps(num_steps);
    for (int step = 0; step < num_steps; ++step) {
        ggml_tensor * h = build_embed_suffix(C, *this, t_state, x_t, t_time[step]);
        for (int64_t i = 0; i < n_layers; ++i) {
            h = build_gemma_layer(C, ex_layers[i], h, t_suffix_pos, cfg, n_suf, rope_base,
                                   cK[i],  cV[i],  t_full_mask,
                                   nullptr,  nullptr);
        }
        ggml_tensor * h_final = ggml_mul(C, ggml_rms_norm(C, h, cfg.rms_eps), ex_final_norm);
        const size_t rb = (size_t) hidden_ex * sizeof(float);
        ggml_tensor * h_actions = ggml_view_2d(C, h_final, hidden_ex, chunk,  rb,  rb);
        ggml_tensor * v_t = ggml_add(C, ggml_mul_mat(C, W_aout, h_actions), b_aout);
        v_steps[step] = v_t;
        x_t = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
    }
    ggml_tensor * x_final = x_t;
    ggml_set_output(x_final);

    ggml_cgraph * gf = ggml_new_graph_custom(C,  16384,  false);
    ggml_build_forward_expand(gf, x_final);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "vla(pi0): ggml_gallocr_alloc_graph failed (out of memory?)\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    ggml_backend_tensor_set(t_image_emb, img_emb_host.data(), 0, ggml_nbytes(t_image_emb));
    ggml_backend_tensor_set(t_lang_emb,  lang_rows.data(),    0, ggml_nbytes(t_lang_emb));
    {
        std::vector<int32_t> pp(n_prefix); for (int64_t i = 0; i < n_prefix; ++i) pp[i] = (int32_t) i;
        ggml_backend_tensor_set(t_prefix_pos, pp.data(), 0, ggml_nbytes(t_prefix_pos));
        std::vector<int32_t> sp(n_suf);     for (int64_t i = 0; i < n_suf; ++i)    sp[i] = (int32_t) (n_prefix + i);
        ggml_backend_tensor_set(t_suffix_pos, sp.data(), 0, ggml_nbytes(t_suffix_pos));
    }
    {

        std::vector<float> sh(max_sd, 0.f);
        for (int64_t i = 0; i < max_sd; ++i) sh[i] = in.state ? in.state[i] : 0.f;
        for (int64_t i = 0; i < cfg.real_state_dim && i < max_sd; ++i)
            sh[i] = (sh[i] - state_mean[i]) / (state_std[i] + cfg.norm_eps);
        ggml_backend_tensor_set(t_state, sh.data(), 0, ggml_nbytes(t_state));
    }
    {
        std::vector<float> x0h((size_t) max_ad * chunk);
        if (in.noise) std::memcpy(x0h.data(), in.noise, x0h.size() * sizeof(float));
        else { std::normal_distribution<float> nd(0.f, 1.f); for (auto & v : x0h) v = nd(rng); }
        ggml_backend_tensor_set(t_x0, x0h.data(), 0, ggml_nbytes(t_x0));
    }
    {

        std::vector<float> mk((size_t) n_total * n_suf);
        for (int64_t i = 0; i < n_suf; ++i)
            for (int64_t j = 0; j < n_total; ++j) {
                bool allowed;
                if (j < n_prefix) allowed = true;
                else { const int64_t s = j - n_prefix; allowed = (i == 0) ? (s == 0) : true; }
                mk[i * n_total + j] = allowed ? 0.f : -INFINITY;
            }
        ggml_backend_tensor_set(t_full_mask, mk.data(), 0, ggml_nbytes(t_full_mask));
    }
    for (int s = 0; s < num_steps; ++s) {
        const float timestep = 1.0f + (float) s * dt;
        const std::vector<float> tv = sinusoidal_time_emb(timestep, hidden_ex, cfg.min_period, cfg.max_period);
        std::vector<float> tile((size_t) hidden_ex * chunk);
        for (int64_t c = 0; c < chunk; ++c) std::memcpy(tile.data() + c * hidden_ex, tv.data(), hidden_ex * sizeof(float));
        ggml_backend_tensor_set(t_time[s], tile.data(), 0, ggml_nbytes(t_time[s]));
    }

    const auto ti0 = clk::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(pi0): ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    std::vector<float> out((size_t) chunk * max_ad);
    ggml_backend_tensor_get(x_final, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);
    for (int64_t t = 0; t < chunk; ++t) {
        float * row = out.data() + (size_t) t * max_ad;
        for (int64_t j = 0; j < cfg.real_action_dim && j < max_ad; ++j)
            row[j] = row[j] * (action_std[j] + cfg.norm_eps) + action_mean[j];
    }

    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
    return out;
}

}
