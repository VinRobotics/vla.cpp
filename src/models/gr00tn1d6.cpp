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

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
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
        if (!gctx) { std::fprintf(stderr, "vla(gr00tn1d6): gguf_init_from_file failed for %s\n", path.c_str()); return false; }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) { std::fprintf(stderr, "vla(gr00tn1d6): fopen failed for %s\n", path.c_str()); return false; }
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

    bool        has(const char * k) const { return gguf_find_key(gctx, k) >= 0; }
    uint32_t    u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }
    float       f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }
    double      f64(const char * k) const { return gguf_get_val_f64(gctx, gguf_find_key(gctx, k)); }
    std::string str(const char * k) const { const int64_t id = gguf_find_key(gctx, k); return id < 0 ? std::string() : std::string(gguf_get_val_str(gctx, id)); }
    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    bool read_raw(const char * name, void * buf) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) { std::fprintf(stderr, "vla(gr00tn1d6): missing tensor %s\n", name); return false; }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }
    std::vector<float> read_f32(const char * name) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(gr00tn1d6): missing tensor %s\n", name); return {}; }
        const int64_t n = ggml_nelements(t);
        std::vector<float> out(n);
        if (t->type == GGML_TYPE_F32) { if (!read_raw(name, out.data())) return {}; }
        else if (t->type == GGML_TYPE_BF16) { std::vector<ggml_bf16_t> tmp(n); if (!read_raw(name, tmp.data())) return {}; ggml_bf16_to_fp32_row(tmp.data(), out.data(), n); }
        else { std::fprintf(stderr, "vla(gr00tn1d6): tensor %s unsupported type %d\n", name, (int) t->type); return {}; }
        return out;
    }
    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        std::vector<float> f = read_f32(name);
        if (f.empty()) return {};
        const int64_t n = (int64_t) f.size();
        if (target == GGML_TYPE_F32)  { std::vector<uint8_t> o(n * 4);  std::memcpy(o.data(), f.data(), o.size()); return o; }
        if (target == GGML_TYPE_BF16) { std::vector<uint8_t> o(n * 2);  ggml_fp32_to_bf16_row(f.data(), reinterpret_cast<ggml_bf16_t *>(o.data()), n); return o; }
        std::fprintf(stderr, "vla(gr00tn1d6): unsupported resident type %d for %s\n", (int) target, name); return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids, float * dst, int64_t cols) {
        const ggml_tensor * t = meta(name);
        if (!t || t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) { std::fprintf(stderr, "vla(gr00tn1d6): %s shape unfit for row-fetch\n", name); return false; }
        const int64_t rows = t->ne[1];
        const int64_t id   = gguf_find_tensor(gctx, name);
        const size_t  base = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t  elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
        const size_t  rb   = (size_t) cols * elsz;
        std::vector<uint8_t> row(rb);
        for (size_t k = 0; k < row_ids.size(); ++k) {
            const int32_t r = row_ids[k];
            if (r < 0 || r >= rows) { std::fprintf(stderr, "vla(gr00tn1d6): row %d out of range for %s\n", r, name); return false; }
            if (std::fseek(fp, (long) (base + (size_t) r * rb), SEEK_SET) != 0) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

struct SigLipLayerW { ggml_tensor *ln1w,*ln1b,*ln2w,*ln2b,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*Wfc1,*bfc1,*Wfc2,*bfc2; };
struct Qwen3LayerW  { ggml_tensor *attn_norm,*Wq,*Wk,*Wv,*Wo,*q_norm,*k_norm,*ffn_norm,*Wgate,*Wup,*Wdown; };
struct DitLayerW    { ggml_tensor *adaln_w,*adaln_b,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*Wff0,*bff0,*Wff2,*bff2; };

}

struct Gr00tN1d6ModelArch : public ModelArchBase {
    Gr00tN1d6ModelArch() : ModelArchBase(Arch::GR00T_N1_6) {}
    ~Gr00tN1d6ModelArch() override;

    std::string           gguf_path;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    bool                  is_gpu      = false;
    int                   n_threads   = 4;
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;

    int64_t vit_hidden=1152, vit_layers=27, vit_heads=16, vit_inter=4304, image_size=224, patch_size=14;
    int64_t vit_num_patches=256, n_img_tokens=64, vit_pixel_shuffle=2, mlp_inner=4608;
    int64_t lm_hidden=2048, lm_layers=16, n_q=16, n_kv=8, lm_head_dim=128, lm_inter=6144, vocab=151680, image_token_index=151669;
    int64_t bb_embed_dim=2048, in_embed_dim=1536, dit_hidden=1536, dit_heads=32, dit_head_dim=48, dit_layers=32, dit_interleave=1, attend_text_every_n=2;
    int64_t action_horizon=50, action_dim=128, max_state_dim=128;
    int64_t num_steps=4, num_buckets=1000, max_embodiments=32, max_seq_len=1024;
    float   vit_ln_eps=1e-6f, lm_rms_eps=1e-6f, ln_eps=1e-5f, norm_out_eps=1e-6f, vlln_eps=1e-5f, connector_ln_eps=1e-5f, lm_rope_base=1000000.0f;
    int64_t embodiment_id = 20;

    ggml_tensor *vit_patch_w=nullptr,*vit_patch_b=nullptr,*vit_pos=nullptr,*vit_post_ln_w=nullptr,*vit_post_ln_b=nullptr;
    std::vector<SigLipLayerW> vit;
    ggml_tensor *mm_ln_w=nullptr,*mm_ln_b=nullptr,*mm_fc1_w=nullptr,*mm_fc1_b=nullptr,*mm_fc2_w=nullptr,*mm_fc2_b=nullptr;

    ggml_tensor *lm_output_norm=nullptr;
    std::vector<Qwen3LayerW> lm;

    ggml_tensor *vlln_w=nullptr,*vlln_b=nullptr;
    ggml_tensor *se_l1W=nullptr,*se_l1b=nullptr,*se_l2W=nullptr,*se_l2b=nullptr;
    ggml_tensor *ae_W1W=nullptr,*ae_W1b=nullptr,*ae_W2W=nullptr,*ae_W2b=nullptr,*ae_W3W=nullptr,*ae_W3b=nullptr;
    ggml_tensor *ad_l1W=nullptr,*ad_l1b=nullptr,*ad_l2W=nullptr,*ad_l2b=nullptr;
    ggml_tensor *pos_embd=nullptr;
    ggml_tensor *te_l1W=nullptr,*te_l1b=nullptr,*te_l2W=nullptr,*te_l2b=nullptr;
    std::vector<DitLayerW> dit;
    ggml_tensor *po1W=nullptr,*po1b=nullptr,*po2W=nullptr,*po2b=nullptr;

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

ggml_tensor * cat_linear(ggml_context * C, ggml_tensor * W3d, ggml_tensor * b2d, int64_t id, ggml_tensor * x) {
    const int64_t out = W3d->ne[0], in = W3d->ne[1];
    ggml_tensor * W_id = ggml_view_2d(C, W3d, out, in, W3d->nb[1], (size_t) id * W3d->nb[2]);
    ggml_tensor * y = ggml_mul_mat(C, ggml_cont(C, ggml_transpose(C, W_id)), x);
    return ggml_add(C, y, ggml_view_1d(C, b2d, out, (size_t) id * b2d->nb[1]));
}

ggml_tensor * adaln(ggml_context * C, ggml_tensor * x, ggml_tensor * temb, ggml_tensor * lw, ggml_tensor * lb, int64_t dim, float eps) {
    ggml_tensor * cond = ggml_add(C, ggml_mul_mat(C, lw, ggml_silu(C, temb)), lb);
    ggml_tensor * sc = ggml_view_1d(C, cond, dim, 0), * sh = ggml_view_1d(C, cond, dim, (size_t) dim * sizeof(float));
    ggml_tensor * xn = ggml_norm(C, x, eps);
    return ggml_add(C, ggml_add(C, xn, ggml_mul(C, xn, sc)), sh);
}

ggml_tensor * build_siglip_layer(ggml_context * C, const SigLipLayerW & w, ggml_tensor * x,
                                 int64_t seq, int64_t heads, int64_t head_dim, int64_t hidden, float ln_eps) {
    const int64_t nv = x->ne[2];
    const float scale = 1.0f / std::sqrt((float) head_dim);
    ggml_tensor * n1 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.ln1w), w.ln1b);
    ggml_tensor * q = ggml_add(C, ggml_mul_mat(C, w.Wq, n1), w.bq);
    ggml_tensor * k = ggml_add(C, ggml_mul_mat(C, w.Wk, n1), w.bk);
    ggml_tensor * v = ggml_add(C, ggml_mul_mat(C, w.Wv, n1), w.bv);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, q, head_dim, heads, seq, nv), 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, k, head_dim, heads, seq, nv), 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, v, head_dim, heads, seq, nv), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_3d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), hidden, seq, nv);
    ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, h1, ln_eps), w.ln2w), w.ln2b);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wfc2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wfc1, n2), w.bfc1))), w.bfc2);
    return ggml_add(C, h1, ff);
}

ggml_tensor * build_qwen3_layer(ggml_context * C, const Gr00tN1d6ModelArch & m, const Qwen3LayerW & w,
                                ggml_tensor * h, ggml_tensor * positions, ggml_tensor * mask, int64_t seq) {
    const int64_t hd = m.lm_head_dim, n_q = m.n_q, n_kv = m.n_kv, hq = n_q * hd;
    const float scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * hn = ggml_mul(C, ggml_rms_norm(C, h, m.lm_rms_eps), w.attn_norm);
    ggml_tensor * qp = ggml_mul_mat(C, w.Wq, hn);
    ggml_tensor * kp = ggml_mul_mat(C, w.Wk, hn);
    ggml_tensor * vp = ggml_mul_mat(C, w.Wv, hn);
    ggml_tensor * qh = ggml_reshape_3d(C, qp, hd, n_q,  seq);
    ggml_tensor * kh = ggml_reshape_3d(C, kp, hd, n_kv, seq);
    ggml_tensor * vh = ggml_reshape_3d(C, vp, hd, n_kv, seq);
    ggml_tensor * qn = ggml_mul(C, ggml_rms_norm(C, qh, m.lm_rms_eps), w.q_norm);
    ggml_tensor * kn = ggml_mul(C, ggml_rms_norm(C, kh, m.lm_rms_eps), w.k_norm);
    ggml_tensor * qr = ggml_rope_ext(C, qn, positions, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, m.lm_rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * kr = ggml_rope_ext(C, kn, positions, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, m.lm_rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, qr, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, kr, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, vh, 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, mask, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), hq, seq);
    ggml_tensor * h_attn = ggml_add(C, h, ggml_mul_mat(C, w.Wo, att));
    ggml_tensor * hn2 = ggml_mul(C, ggml_rms_norm(C, h_attn, m.lm_rms_eps), w.ffn_norm);
    ggml_tensor * gate = ggml_silu(C, ggml_mul_mat(C, w.Wgate, hn2));
    ggml_tensor * up   = ggml_mul_mat(C, w.Wup, hn2);
    return ggml_add(C, h_attn, ggml_mul_mat(C, w.Wdown, ggml_mul(C, gate, up)));
}

void dit_kv(ggml_context * C, const Gr00tN1d6ModelArch & m, const DitLayerW & w, ggml_tensor * kv,
            ggml_tensor ** K_out, ggml_tensor ** V_out) {
    const int64_t hd = m.dit_head_dim, heads = m.dit_heads, Tkv = kv->ne[1];
    ggml_tensor * k = ggml_add(C, ggml_mul_mat(C, w.Wk, kv), w.bk);
    ggml_tensor * v = ggml_add(C, ggml_mul_mat(C, w.Wv, kv), w.bv);
    *K_out = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, hd, heads, Tkv), 0, 2, 1, 3));
    *V_out = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, hd, heads, Tkv), 1, 2, 0, 3));
}

ggml_tensor * build_dit_block(ggml_context * C, const Gr00tN1d6ModelArch & m, const DitLayerW & w,
                              ggml_tensor * h, ggml_tensor * temb, ggml_tensor * enc ,
                              ggml_tensor * K_pre = nullptr, ggml_tensor * V_pre = nullptr) {
    const int64_t hd = m.dit_head_dim, heads = m.dit_heads, dim = m.dit_hidden, Tk = h->ne[1];
    const float scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * n = adaln(C, h, temb, w.adaln_w, w.adaln_b, dim, m.ln_eps);
    ggml_tensor * q = ggml_add(C, ggml_mul_mat(C, w.Wq, n),  w.bq);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, hd, heads, Tk),  0, 2, 1, 3));
    ggml_tensor * K, * V;
    if (K_pre) { K = K_pre; V = V_pre; }
    else       { dit_kv(C, m, w, enc ? enc : n, &K, &V); }
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), dim, Tk);
    ggml_tensor * h1 = ggml_add(C, h, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n3 = ggml_norm(C, h1, m.ln_eps);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wff2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wff0, n3), w.bff0))), w.bff2);
    return ggml_add(C, h1, ff);
}

bool preprocess_image_patches(const ImageView & v, int64_t side, int64_t ps, std::vector<float> & out) {
    if (v.w != (int) side || v.h != (int) side || !v.data) {
        std::fprintf(stderr, "vla(gr00tn1d6): image view is %dx%d, expected %lldx%lld\n", v.w, v.h, (long long) side, (long long) side); return false;
    }
    const int64_t grid = side / ps, pd = 3 * ps * ps, np = grid * grid;
    out.assign((size_t) pd * np, 0.0f);
    auto px = [&](int64_t r, int64_t c, int64_t ch) -> float {
        if (v.format == PixelFormat::U8) return ((const uint8_t *) v.data)[(r * side + c) * 3 + ch] / 255.0f;
        return ((const float *) v.data)[(r * side + c) * 3 + ch];
    };
    for (int64_t row = 0; row < grid; ++row)
        for (int64_t col = 0; col < grid; ++col) {
            const int64_t t = row * grid + col;
            for (int64_t ph = 0; ph < ps; ++ph)
                for (int64_t pw = 0; pw < ps; ++pw)
                    for (int64_t ch = 0; ch < 3; ++ch)
                        out[t * pd + ph * ps * 3 + pw * 3 + ch] = px(row * ps + ph, col * ps + pw, ch) * 2.0f - 1.0f;
        }
    return true;
}

void pixel_shuffle_back(const float * src, int64_t grid, int64_t hidden, int64_t r, float * dst) {
    const int64_t g2 = grid / r, c4 = hidden * r * r;
    for (int64_t y = 0; y < g2; ++y)
        for (int64_t x = 0; x < g2; ++x) {
            const int64_t t = y * g2 + x;
            for (int64_t c = 0; c < hidden; ++c)
                for (int64_t i = 0; i < r; ++i)
                    for (int64_t j = 0; j < r; ++j) {
                        const int64_t pp = (r * y + i) * grid + (r * x + j);
                        const int64_t cp = c * r * r + i * r + j;
                        dst[t * c4 + cp] = src[pp * hidden + c];
                    }
        }
}

void timesteps_proj(int64_t bucket, std::vector<float> & out) {
    const int64_t half = 128; const float lm = std::log(10000.0f); const float t = (float) bucket;
    out.assign(256, 0.0f);
    for (int64_t i = 0; i < half; ++i) { const float emb = t * std::exp(-lm * (float) i / (float) (half - 1)); out[i] = std::cos(emb); out[half + i] = std::sin(emb); }
}

void action_sinusoid(int64_t bucket, int64_t dim, int64_t T, std::vector<float> & out) {
    const int64_t half = dim / 2; const float step = std::log(10000.0f) / (float) half; const float t = (float) bucket;
    out.assign((size_t) T * dim, 0.0f);
    for (int64_t tk = 0; tk < T; ++tk) for (int64_t i = 0; i < half; ++i) { const float emb = t * std::exp(-(float) i * step); out[tk * dim + i] = std::sin(emb); out[tk * dim + half + i] = std::cos(emb); }
}

bool load_config(const gguf_reader & g, Gr00tN1d6ModelArch & m, Config & cfg) {
    auto U = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto fk = [&](const char * s) { static char b[64]; std::snprintf(b, sizeof(b), "gr00t_n1_6.%s", s); return b; };
    U(fk("vit_hidden"), m.vit_hidden); U(fk("vit_layers"), m.vit_layers); U(fk("vit_heads"), m.vit_heads); U(fk("vit_inter"), m.vit_inter);
    U(fk("image_size"), m.image_size); U(fk("patch_size"), m.patch_size);
    U(fk("vit_num_patches"), m.vit_num_patches); U(fk("n_img_tokens"), m.n_img_tokens); U(fk("vit_pixel_shuffle"), m.vit_pixel_shuffle); U(fk("mlp_connector_inner"), m.mlp_inner);
    U(fk("lm_hidden"), m.lm_hidden); U(fk("lm_layers_used"), m.lm_layers); U(fk("lm_q_heads"), m.n_q); U(fk("lm_kv_heads"), m.n_kv);
    U(fk("lm_head_dim"), m.lm_head_dim); U(fk("lm_inter"), m.lm_inter); U(fk("vocab_size"), m.vocab); U(fk("image_token_index"), m.image_token_index);
    U(fk("backbone_embedding_dim"), m.bb_embed_dim); U(fk("input_embedding_dim"), m.in_embed_dim);
    U(fk("dit_hidden"), m.dit_hidden); U(fk("dit_heads"), m.dit_heads); U(fk("dit_head_dim"), m.dit_head_dim); U(fk("dit_layers"), m.dit_layers); U(fk("dit_interleave"), m.dit_interleave);
    U(fk("attend_text_every_n_blocks"), m.attend_text_every_n);
    U(fk("action_horizon"), m.action_horizon); U(fk("action_dim"), m.action_dim); U(fk("max_state_dim"), m.max_state_dim);
    U(fk("num_inference_timesteps"), m.num_steps); U(fk("num_timestep_buckets"), m.num_buckets); U(fk("max_num_embodiments"), m.max_embodiments); U(fk("max_seq_len"), m.max_seq_len);
    F(fk("vit_ln_eps"), m.vit_ln_eps); F(fk("lm_rms_eps"), m.lm_rms_eps); F(fk("ln_eps"), m.ln_eps); F(fk("norm_out_eps"), m.norm_out_eps); F(fk("vlln_eps"), m.vlln_eps); F(fk("connector_ln_eps"), m.connector_ln_eps);
    if (g.has(fk("lm_rope_theta"))) m.lm_rope_base = (float) g.f64(fk("lm_rope_theta"));

    m.embodiment_id = 20;
    {
        const std::string js = g.str(fk("embodiment_id_mapping"));
        auto lookup = [&](const char * key) -> long {
            const std::string k = std::string("\"") + key + "\"";
            size_t p = js.find(k); if (p == std::string::npos) return -1;
            p = js.find(':', p + k.size()); if (p == std::string::npos) return -1;
            return std::strtol(js.c_str() + p + 1, nullptr, 10);
        };
        long gr1 = lookup("gr1"); if (gr1 >= 0) m.embodiment_id = gr1;
        if (const char * e = std::getenv("VLA_GR00T_EMBODIMENT")) {
            char * end = nullptr; long v = std::strtol(e, &end, 10);
            if (end && *end == '\0') m.embodiment_id = v;
            else { long id = lookup(e); if (id >= 0) m.embodiment_id = id; else std::fprintf(stderr, "vla(gr00tn1d6): embodiment tag '%s' not in embodiment_id_mapping; using id %lld\n", e, (long long) m.embodiment_id); }
        }
    }
    if (m.embodiment_id < 0 || m.embodiment_id >= m.max_embodiments) { std::fprintf(stderr, "vla(gr00tn1d6): embodiment id %lld out of range [0,%lld)\n", (long long) m.embodiment_id, (long long) m.max_embodiments); return false; }

    cfg = Config{};
    cfg.n_img = m.n_img_tokens; cfg.n_lang = m.max_seq_len; cfg.n_state = 1;
    cfg.n_suffix = m.action_horizon; cfg.max_state_dim = m.max_state_dim; cfg.max_action_dim = m.action_dim;
    cfg.real_state_dim = m.max_state_dim; cfg.real_action_dim = m.action_dim;
    cfg.hidden = m.lm_hidden; cfg.n_q_heads = m.n_q; cfg.n_kv_heads = m.n_kv; cfg.head_dim = m.lm_head_dim; cfg.n_layers = m.lm_layers;
    cfg.num_steps = (int) m.num_steps; cfg.rms_eps = m.lm_rms_eps;
    cfg.rope_n_dims = (int) m.lm_head_dim; cfg.rope_mode = GGML_ROPE_TYPE_NEOX; cfg.rope_freq_base = m.lm_rope_base;
    cfg.norm_eps = 1e-8f;
    return true;
}

}

Gr00tN1d6ModelArch::~Gr00tN1d6ModelArch() {
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> gr00t_n1_6_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string& ) {
    if (!mmproj_path.empty())
        std::printf("vla(gr00tn1d6): note — mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<Gr00tN1d6ModelArch>();
    m->gguf_path   = ckpt_path;
    m->matmul_type = std::getenv("VLA_GR00T_BF16_WEIGHTS") ? GGML_TYPE_BF16 : GGML_TYPE_F32;

    gguf_reader g;
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has("gr00t_n1_6.architecture")) { std::fprintf(stderr, "vla(gr00tn1d6): %s is not a gr00t_n1_6 GGUF\n", ckpt_path.c_str()); return nullptr; }
    if (!load_config(g, *m, m->cfg)) return nullptr;
    std::printf("vla(gr00tn1d6): vit=%lldd×%lldL×%lldh (Linear patch embed)  pixel_shuffle÷%lld ⇒ n_img_tok=%lld  mlp1=LN(%lld)→Linear→GELU→Linear  "
                "lm=Qwen3 %lldd×%lldL (%lldq/%lldkv×%lld)  dit=AlternateVLDiT %lldL×%lldh×%lld(inner %lld) attend_text_every_n=%lld  in_emb=%lld  "
                "horizon=%lld action_dim=%lld max_state=%lld N_steps=%lld  embodiment=%lld  resident=%s\n",
                (long long) m->vit_hidden, (long long) m->vit_layers, (long long) m->vit_heads, (long long) m->vit_pixel_shuffle, (long long) m->n_img_tokens, (long long) m->mlp_inner,
                (long long) m->lm_hidden, (long long) m->lm_layers, (long long) m->n_q, (long long) m->n_kv, (long long) m->lm_head_dim,
                (long long) m->dit_layers, (long long) m->dit_heads, (long long) m->dit_head_dim, (long long) m->dit_hidden, (long long) m->attend_text_every_n, (long long) m->in_embed_dim,
                (long long) m->action_horizon, (long long) m->action_dim, (long long) m->max_state_dim, (long long) m->num_steps, (long long) m->embodiment_id,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) { m->is_cuda = true; m->is_gpu = true; std::printf("vla(gr00tn1d6): backend = CUDA (device 0)\n"); }
    else            std::fprintf(stderr, "vla(gr00tn1d6): ggml_backend_cuda_init failed; falling back to CPU\n");
#elif defined(GGML_USE_METAL)
    m->backend = ggml_backend_metal_init();
    if (m->backend) { m->is_gpu = true; std::printf("vla(gr00tn1d6): backend = Metal\n"); }
    else            std::fprintf(stderr, "vla(gr00tn1d6): ggml_backend_metal_init failed; falling back to CPU\n");
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(gr00tn1d6): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(gr00tn1d6): backend = CPU (%d threads)\n", m->n_threads);
    }

    ggml_init_params wp = {  (size_t) 32 * 1024 * 1024,  nullptr,  true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) { std::fprintf(stderr, "vla(gr00tn1d6): ggml_init(ctx_weights) failed\n"); return nullptr; }
    ggml_context * W = m->ctx_weights;
    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(gr00tn1d6): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, type, ggml_n_dims(gt), gt->ne);
        ggml_set_name(t, name); return t;
    };
    auto mk_mm  = [&](const char * name) { return mk(name, m->matmul_type); };
    auto mk_f32 = [&](const char * name) { return mk(name, GGML_TYPE_F32); };

    bool ok = true;

    m->vit_patch_w   = mk_mm("vit.patch_embd.weight");
    m->vit_patch_b   = mk_f32("vit.patch_embd.bias");
    m->vit_pos       = mk_f32("vit.pos_embd");
    m->vit_post_ln_w = mk_f32("vit.post_ln.weight"); m->vit_post_ln_b = mk_f32("vit.post_ln.bias");
    m->vit.resize(m->vit_layers);
    for (int64_t i = 0; i < m->vit_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "vit.blk.%lld.%s", (long long) i, s); return p; };
        auto & w = m->vit[i];
        w.ln1w=mk_f32(N("ln1.weight")); w.ln1b=mk_f32(N("ln1.bias")); w.ln2w=mk_f32(N("ln2.weight")); w.ln2b=mk_f32(N("ln2.bias"));
        w.Wq=mk_mm(N("attn_q.weight")); w.bq=mk_f32(N("attn_q.bias")); w.Wk=mk_mm(N("attn_k.weight")); w.bk=mk_f32(N("attn_k.bias"));
        w.Wv=mk_mm(N("attn_v.weight")); w.bv=mk_f32(N("attn_v.bias")); w.Wo=mk_mm(N("attn_o.weight")); w.bo=mk_f32(N("attn_o.bias"));
        w.Wfc1=mk_mm(N("fc1.weight")); w.bfc1=mk_f32(N("fc1.bias")); w.Wfc2=mk_mm(N("fc2.weight")); w.bfc2=mk_f32(N("fc2.bias"));
        ok &= w.ln1w&&w.ln1b&&w.ln2w&&w.ln2b&&w.Wq&&w.bq&&w.Wk&&w.bk&&w.Wv&&w.bv&&w.Wo&&w.bo&&w.Wfc1&&w.bfc1&&w.Wfc2&&w.bfc2;
    }

    m->mm_ln_w  = mk_f32("mm.ln.weight");  m->mm_ln_b  = mk_f32("mm.ln.bias");
    m->mm_fc1_w = mk_mm("mm.fc1.weight");  m->mm_fc1_b = mk_f32("mm.fc1.bias");
    m->mm_fc2_w = mk_mm("mm.fc2.weight");  m->mm_fc2_b = mk_f32("mm.fc2.bias");

    m->lm_output_norm = mk_f32("vlm.output_norm.weight");
    m->lm.resize(m->lm_layers);
    for (int64_t i = 0; i < m->lm_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "vlm.blk.%lld.%s", (long long) i, s); return p; };
        auto & w = m->lm[i];
        w.attn_norm=mk_f32(N("attn_norm.weight"));
        w.Wq=mk_mm(N("attn_q.weight")); w.Wk=mk_mm(N("attn_k.weight")); w.Wv=mk_mm(N("attn_v.weight")); w.Wo=mk_mm(N("attn_o.weight"));
        w.q_norm=mk_f32(N("attn_q_norm.weight")); w.k_norm=mk_f32(N("attn_k_norm.weight")); w.ffn_norm=mk_f32(N("ffn_norm.weight"));
        w.Wgate=mk_mm(N("ffn_gate.weight")); w.Wup=mk_mm(N("ffn_up.weight")); w.Wdown=mk_mm(N("ffn_down.weight"));
        ok &= w.attn_norm&&w.Wq&&w.Wk&&w.Wv&&w.Wo&&w.q_norm&&w.k_norm&&w.ffn_norm&&w.Wgate&&w.Wup&&w.Wdown;
    }

    m->vlln_w=mk_f32("aex.vlln.weight"); m->vlln_b=mk_f32("aex.vlln.bias");

    m->se_l1W=mk_f32("aex.state_enc.l1.W"); m->se_l1b=mk_f32("aex.state_enc.l1.b"); m->se_l2W=mk_f32("aex.state_enc.l2.W"); m->se_l2b=mk_f32("aex.state_enc.l2.b");
    m->ae_W1W=mk_f32("aex.act_enc.W1.W"); m->ae_W1b=mk_f32("aex.act_enc.W1.b"); m->ae_W2W=mk_f32("aex.act_enc.W2.W"); m->ae_W2b=mk_f32("aex.act_enc.W2.b"); m->ae_W3W=mk_f32("aex.act_enc.W3.W"); m->ae_W3b=mk_f32("aex.act_enc.W3.b");
    m->ad_l1W=mk_f32("aex.act_dec.l1.W"); m->ad_l1b=mk_f32("aex.act_dec.l1.b"); m->ad_l2W=mk_f32("aex.act_dec.l2.W"); m->ad_l2b=mk_f32("aex.act_dec.l2.b");
    m->pos_embd=mk_f32("aex.pos_embd");
    m->te_l1W=mk_mm("aex.dit.time_emb.l1.weight"); m->te_l1b=mk_f32("aex.dit.time_emb.l1.bias"); m->te_l2W=mk_mm("aex.dit.time_emb.l2.weight"); m->te_l2b=mk_f32("aex.dit.time_emb.l2.bias");
    m->dit.resize(m->dit_layers);
    for (int64_t i = 0; i < m->dit_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "aex.dit.%lld.%s", (long long) i, s); return p; };
        auto & w = m->dit[i];
        w.adaln_w=mk_mm(N("adaln.weight")); w.adaln_b=mk_f32(N("adaln.bias"));
        w.Wq=mk_mm(N("attn_q.weight")); w.bq=mk_f32(N("attn_q.bias")); w.Wk=mk_mm(N("attn_k.weight")); w.bk=mk_f32(N("attn_k.bias"));
        w.Wv=mk_mm(N("attn_v.weight")); w.bv=mk_f32(N("attn_v.bias")); w.Wo=mk_mm(N("attn_o.weight")); w.bo=mk_f32(N("attn_o.bias"));
        w.Wff0=mk_mm(N("ff0.weight")); w.bff0=mk_f32(N("ff0.bias")); w.Wff2=mk_mm(N("ff2.weight")); w.bff2=mk_f32(N("ff2.bias"));
        ok &= w.adaln_w&&w.adaln_b&&w.Wq&&w.bq&&w.Wk&&w.bk&&w.Wv&&w.bv&&w.Wo&&w.bo&&w.Wff0&&w.bff0&&w.Wff2&&w.bff2;
    }
    m->po1W=mk_mm("aex.dit.proj_out1.weight"); m->po1b=mk_f32("aex.dit.proj_out1.bias"); m->po2W=mk_mm("aex.dit.proj_out2.weight"); m->po2b=mk_f32("aex.dit.proj_out2.bias");
    ok &= m->vit_patch_w&&m->vit_patch_b&&m->vit_pos&&m->vit_post_ln_w&&m->vit_post_ln_b&&m->mm_ln_w&&m->mm_ln_b&&m->mm_fc1_w&&m->mm_fc1_b&&m->mm_fc2_w&&m->mm_fc2_b&&m->lm_output_norm&&
          m->vlln_w&&m->vlln_b&&m->se_l1W&&m->se_l1b&&m->se_l2W&&m->se_l2b&&m->ae_W1W&&m->ae_W1b&&m->ae_W2W&&m->ae_W2b&&m->ae_W3W&&m->ae_W3b&&
          m->ad_l1W&&m->ad_l1b&&m->ad_l2W&&m->ad_l2b&&m->pos_embd&&m->te_l1W&&m->te_l1b&&m->te_l2W&&m->te_l2b&&m->po1W&&m->po1b&&m->po2W&&m->po2b;
    if (!ok) { std::fprintf(stderr, "vla(gr00tn1d6): weight tensor setup failed\n"); return nullptr; }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(gr00tn1d6): ggml_backend_alloc_ctx_tensors failed (OOM?)\n"); return nullptr; }
    for (ggml_tensor * t = ggml_get_first_tensor(W); t; t = ggml_get_next_tensor(W, t)) {
        std::vector<uint8_t> bytes = g.read_convert(ggml_get_name(t), t->type);
        if (bytes.empty() || bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(gr00tn1d6): failed to load %s (%zu vs %zu bytes)\n", ggml_get_name(t), bytes.size(), ggml_nbytes(t)); return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(gr00tn1d6): weights resident in %.2f GiB (%s) — incl. SigLIP2 vision tower; embodiment id %lld\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0), m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16", (long long) m->embodiment_id);
    return m;
}

std::vector<float> Gr00tN1d6ModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    const int64_t H = lm_hidden, K = n_img_tokens, E = in_embed_dim;
    const int64_t grid = image_size / patch_size;
    const int64_t r = vit_pixel_shuffle;
    const int64_t n_patches = grid * grid;
    const int64_t patch_dim = 3 * patch_size * patch_size;
    const int64_t c4 = vit_hidden * r * r;
    const int64_t Nsa = 1 + action_horizon;

    int64_t n_views = 0;
    std::vector<float> img_emb_host;
    const float * img_emb_ptr = nullptr;
    if (in.precomputed_img_emb && in.n_img_views > 0) {
        n_views = in.n_img_views;
        img_emb_ptr = in.precomputed_img_emb;
    } else if (in.images && in.n_images > 0) {
        n_views = in.n_images;
        img_emb_host.assign((size_t) n_views * K * H, 0.0f);

        ggml_init_params vp = { (size_t) 64 * 1024 * 1024, nullptr, true };
        ggml_context * VC = ggml_init(vp);
        if (!VC) { std::fprintf(stderr, "vla(gr00tn1d6): ggml_init(vision ctx A) failed\n"); return {}; }
        ggml_tensor * t_patches = ggml_new_tensor_3d(VC, GGML_TYPE_F32, patch_dim, n_patches, n_views); ggml_set_input(t_patches);
        ggml_tensor * h = ggml_add(VC, ggml_add(VC, ggml_mul_mat(VC, vit_patch_w, t_patches), vit_patch_b), vit_pos);
        for (int64_t i = 0; i < vit_layers; ++i) h = build_siglip_layer(VC, vit[i], h, n_patches, vit_heads, vit_hidden / vit_heads, vit_hidden, vit_ln_eps);
        ggml_tensor * post_ln = ggml_add(VC, ggml_mul(VC, ggml_norm(VC, h, vit_ln_eps), vit_post_ln_w), vit_post_ln_b);
        ggml_set_output(post_ln);
        ggml_cgraph * vgA = ggml_new_graph_custom(VC, 8192, false);
        ggml_build_forward_expand(vgA, post_ln);
        ggml_gallocr_t vgaA = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!vgaA || !ggml_gallocr_alloc_graph(vgaA, vgA)) { std::fprintf(stderr, "vla(gr00tn1d6): vision gallocr A alloc failed\n"); if (vgaA) ggml_gallocr_free(vgaA); ggml_free(VC); return {}; }

        ggml_init_params mp = { (size_t) 16 * 1024 * 1024, nullptr, true };
        ggml_context * MC = ggml_init(mp);
        if (!MC) { std::fprintf(stderr, "vla(gr00tn1d6): ggml_init(vision ctx B) failed\n"); ggml_gallocr_free(vgaA); ggml_free(VC); return {}; }
        ggml_tensor * t_shuf = ggml_new_tensor_3d(MC, GGML_TYPE_F32, c4, K, n_views); ggml_set_input(t_shuf);
        ggml_tensor * mln  = ggml_add(MC, ggml_mul(MC, ggml_norm(MC, t_shuf, connector_ln_eps), mm_ln_w), mm_ln_b);
        ggml_tensor * mz1  = ggml_add(MC, ggml_mul_mat(MC, mm_fc1_w, mln), mm_fc1_b);
        ggml_tensor * vit_embeds = ggml_add(MC, ggml_mul_mat(MC, mm_fc2_w, ggml_gelu_erf(MC, mz1)), mm_fc2_b);
        ggml_set_output(vit_embeds);
        ggml_cgraph * vgB = ggml_new_graph(MC);
        ggml_build_forward_expand(vgB, vit_embeds);
        ggml_gallocr_t vgaB = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!vgaB || !ggml_gallocr_alloc_graph(vgaB, vgB)) { std::fprintf(stderr, "vla(gr00tn1d6): vision gallocr B alloc failed\n"); if (vgaB) ggml_gallocr_free(vgaB); ggml_gallocr_free(vgaA); ggml_free(MC); ggml_free(VC); return {}; }

        const auto tv0 = std::chrono::steady_clock::now();
        std::vector<float> patches,
            patches_all((size_t) patch_dim * n_patches * n_views),
            post_ln_host((size_t) vit_hidden * n_patches * n_views),
            shuf_host((size_t) c4 * K * n_views);
        bool vok = true;
        for (int64_t v = 0; v < n_views && vok; ++v) {
            if (!preprocess_image_patches(in.images[v], image_size, patch_size, patches)) { vok = false; break; }
            std::memcpy(patches_all.data() + (size_t) v * patch_dim * n_patches, patches.data(), patches.size() * sizeof(float));
        }
        if (vok) {
            ggml_backend_tensor_set(t_patches, patches_all.data(), 0, ggml_nbytes(t_patches));
            if (ggml_backend_graph_compute(backend, vgA) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(gr00tn1d6): vision compute A failed\n"); vok = false; }
        }
        if (vok) {
            ggml_backend_tensor_get(post_ln, post_ln_host.data(), 0, ggml_nbytes(post_ln));
            for (int64_t v = 0; v < n_views; ++v)
                pixel_shuffle_back(post_ln_host.data() + (size_t) v * vit_hidden * n_patches, grid, vit_hidden, r, shuf_host.data() + (size_t) v * c4 * K);
            ggml_backend_tensor_set(t_shuf, shuf_host.data(), 0, ggml_nbytes(t_shuf));
            if (ggml_backend_graph_compute(backend, vgB) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(gr00tn1d6): vision compute B failed\n"); vok = false; }
        }
        if (vok) ggml_backend_tensor_get(vit_embeds, img_emb_host.data(), 0, ggml_nbytes(vit_embeds));
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tv0).count();
        ggml_gallocr_free(vgaB); ggml_gallocr_free(vgaA); ggml_free(MC); ggml_free(VC);
        if (!vok) return {};
        img_emb_ptr = img_emb_host.data();
    } else {
        std::fprintf(stderr, "vla(gr00tn1d6): no images and no precomputed_img_emb in the request\n"); return {};
    }
    const int64_t n_img = n_views * K;

    std::vector<int32_t> input_ids;
    int64_t n_img_slots = 0;
    for (int j = 0; j < in.n_lang; ++j) if (in.lang_tokens[j] == (int32_t) image_token_index) ++n_img_slots;
    if (n_img_slots == n_img) {
        input_ids.assign(in.lang_tokens, in.lang_tokens + in.n_lang);
    } else if (n_img_slots == 0) {
        input_ids.reserve(n_img + in.n_lang);
        for (int64_t i = 0; i < n_img; ++i) input_ids.push_back((int32_t) image_token_index);
        for (int j = 0; j < in.n_lang; ++j) input_ids.push_back(in.lang_tokens[j]);
    } else {
        std::fprintf(stderr, "vla(gr00tn1d6): lang_tokens has %lld image-token slots but n_img=%lld; expected 0 (v1 fallback) or %lld (chat-template path)\n",
                     (long long) n_img_slots, (long long) n_img, (long long) n_img);
        return {};
    }
    const int64_t SEQ = (int64_t) input_ids.size();
    if (SEQ > max_seq_len) { std::fprintf(stderr, "vla(gr00tn1d6): prompt too long (%lld > %lld)\n", (long long) SEQ, (long long) max_seq_len); return {}; }

    gguf_reader g;
    if (!g.open(gguf_path)) return {};
    std::vector<float> inputs_embeds((size_t) SEQ * H);
    if (!g.fetch_rows_f32("token_embd.weight", input_ids, inputs_embeds.data(), H)) return {};
    {   int64_t k = 0;
        for (int64_t p = 0; p < SEQ; ++p) if (input_ids[p] == (int32_t) image_token_index) {
            if (k >= n_img) { std::fprintf(stderr, "vla(gr00tn1d6): more <image> tokens than ViT embeds\n"); return {}; }
            std::memcpy(inputs_embeds.data() + p * H, img_emb_ptr + k * H, H * sizeof(float)); ++k;
        }
    }

    std::vector<int32_t> image_pos_idx, text_pos_idx;
    image_pos_idx.reserve((size_t) n_img); text_pos_idx.reserve((size_t) (SEQ - n_img));
    for (int64_t p = 0; p < SEQ; ++p) {
        if (input_ids[p] == (int32_t) image_token_index) image_pos_idx.push_back((int32_t) p);
        else                                              text_pos_idx.push_back((int32_t) p);
    }
    const int64_t SEQ_TXT = (int64_t) text_pos_idx.size();
    if ((int64_t) image_pos_idx.size() != n_img) {
        std::fprintf(stderr, "vla(gr00tn1d6): internal: built %zu image positions, expected %lld\n", image_pos_idx.size(), (long long) n_img); return {};
    }

    const int64_t AD = action_dim, AH = action_horizon;
    std::vector<float> x_init((size_t) AH * AD);
    if (in.noise) std::memcpy(x_init.data(), in.noise, x_init.size() * sizeof(float));
    else { std::mt19937 rng((uint32_t) std::chrono::steady_clock::now().time_since_epoch().count()); std::normal_distribution<float> nd(0.f, 1.f); for (auto & v : x_init) v = nd(rng); }

    ggml_init_params cp = { (size_t) 256 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) { std::fprintf(stderr, "vla(gr00tn1d6): ggml_init(ctx_compute) failed\n"); return {}; }

    ggml_tensor * t_embeds = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);          ggml_set_input(t_embeds);
    ggml_tensor * t_pos    = ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ);             ggml_set_input(t_pos);
    ggml_tensor * t_lmmask = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ);        ggml_set_input(t_lmmask);
    ggml_tensor * t_state  = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_state_dim, 1);ggml_set_input(t_state);
    ggml_tensor * t_x0     = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH);          ggml_set_input(t_x0);

    ggml_tensor * t_img_idx = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_img);           ggml_set_input(t_img_idx);
    ggml_tensor * t_txt_idx = (SEQ_TXT > 0) ? ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ_TXT) : nullptr;
    if (t_txt_idx) ggml_set_input(t_txt_idx);
    std::vector<ggml_tensor *> t_tau(num_steps), t_tproj(num_steps);
    for (int64_t s = 0; s < num_steps; ++s) {
        t_tau[s]   = ggml_new_tensor_2d(C, GGML_TYPE_F32, E, AH); ggml_set_input(t_tau[s]);
        t_tproj[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, 256);   ggml_set_input(t_tproj[s]);
    }

    ggml_tensor * h = t_embeds;
    for (int64_t i = 0; i < lm_layers; ++i) h = build_qwen3_layer(C, *this, lm[i], h, t_pos, t_lmmask, SEQ);
    ggml_tensor * eagle = ggml_mul(C, ggml_rms_norm(C, h, lm_rms_eps), lm_output_norm);

    ggml_tensor * vl_embs = ggml_add(C, ggml_mul(C, ggml_norm(C, eagle, vlln_eps), vlln_w), vlln_b);

    ggml_tensor * vl_img = ggml_get_rows(C, vl_embs, t_img_idx);
    ggml_tensor * vl_txt = (t_txt_idx ? ggml_get_rows(C, vl_embs, t_txt_idx) : vl_img);

    ggml_tensor * state_features = cat_linear(C, se_l2W, se_l2b, embodiment_id, ggml_relu(C, cat_linear(C, se_l1W, se_l1b, embodiment_id, t_state)));

    const float dt = 1.0f / (float) num_steps;
    const int64_t every2 = 2 * attend_text_every_n;

    std::vector<ggml_tensor *> Kc(dit_layers, nullptr), Vc(dit_layers, nullptr);
    for (int64_t i = 0; i < dit_layers; ++i) {
        if (dit_interleave && (i % 2 == 1)) continue;
        ggml_tensor * enc = (i % every2 == 0) ? vl_txt : vl_img;
        dit_kv(C, *this, dit[i], enc, &Kc[i], &Vc[i]);
    }
    ggml_tensor * actions = t_x0;
    for (int64_t s = 0; s < num_steps; ++s) {

        ggml_tensor * temb = ggml_add(C, ggml_mul_mat(C, te_l2W, ggml_silu(C, ggml_add(C, ggml_mul_mat(C, te_l1W, t_tproj[s]), te_l1b))), te_l2b);

        ggml_tensor * a_emb = cat_linear(C, ae_W1W, ae_W1b, embodiment_id, actions);
        ggml_tensor * x_w2  = ggml_silu(C, cat_linear(C, ae_W2W, ae_W2b, embodiment_id, ggml_concat(C, a_emb, t_tau[s], 0)));
        ggml_tensor * af    = ggml_add(C, cat_linear(C, ae_W3W, ae_W3b, embodiment_id, x_w2), ggml_view_2d(C, pos_embd, E, AH, pos_embd->nb[1], 0));

        ggml_tensor * sa = ggml_concat(C, state_features, af, 1);

        ggml_tensor * hh = sa;
        for (int64_t i = 0; i < dit_layers; ++i) {
            ggml_tensor * enc;
            if (dit_interleave && (i % 2 == 1)) enc = nullptr;
            else if (i % every2 == 0)           enc = vl_txt;
            else                                enc = vl_img;
            hh = build_dit_block(C, *this, dit[i], hh, temb, enc, Kc[i], Vc[i]);
        }

        ggml_tensor * po = ggml_add(C, ggml_mul_mat(C, po1W, ggml_silu(C, temb)), po1b);
        ggml_tensor * sh = ggml_view_1d(C, po, dit_hidden, 0), * sc = ggml_view_1d(C, po, dit_hidden, (size_t) dit_hidden * sizeof(float));
        ggml_tensor * hn = ggml_norm(C, hh, norm_out_eps);
        ggml_tensor * h_mod = ggml_add(C, ggml_add(C, hn, ggml_mul(C, hn, sc)), sh);
        ggml_tensor * model_output = ggml_add(C, ggml_mul_mat(C, po2W, h_mod), po2b);

        ggml_tensor * pred = cat_linear(C, ad_l2W, ad_l2b, embodiment_id, ggml_relu(C, cat_linear(C, ad_l1W, ad_l1b, embodiment_id, model_output)));
        ggml_tensor * vel  = ggml_cont(C, ggml_view_2d(C, pred, AD, AH, pred->nb[1], (size_t) (Nsa - AH) * pred->nb[1]));
        actions = ggml_add(C, actions, ggml_scale(C, vel, dt));
    }
    ggml_set_name(actions, "action_pred"); ggml_set_output(actions);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
    ggml_build_forward_expand(gf, actions);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "vla(gr00tn1d6): gallocr alloc failed\n"); if (galloc) ggml_gallocr_free(galloc); ggml_free(C); return {}; }

    ggml_backend_tensor_set(t_embeds, inputs_embeds.data(), 0, ggml_nbytes(t_embeds));
    { std::vector<int32_t> pp(SEQ); for (int64_t i = 0; i < SEQ; ++i) pp[i] = (int32_t) i; ggml_backend_tensor_set(t_pos, pp.data(), 0, ggml_nbytes(t_pos)); }
    { std::vector<float> mk((size_t) SEQ * SEQ); const float NEG = -std::numeric_limits<float>::infinity();
      for (int64_t q = 0; q < SEQ; ++q) for (int64_t kv = 0; kv < SEQ; ++kv) mk[q * SEQ + kv] = (kv <= q) ? 0.0f : NEG;
      ggml_backend_tensor_set(t_lmmask, mk.data(), 0, ggml_nbytes(t_lmmask)); }
    { std::vector<float> st(max_state_dim, 0.0f); for (int64_t i = 0; i < max_state_dim; ++i) st[i] = in.state ? in.state[i] : 0.0f; ggml_backend_tensor_set(t_state, st.data(), 0, ggml_nbytes(t_state)); }
    ggml_backend_tensor_set(t_x0, x_init.data(), 0, ggml_nbytes(t_x0));
    ggml_backend_tensor_set(t_img_idx, image_pos_idx.data(), 0, ggml_nbytes(t_img_idx));
    if (t_txt_idx) ggml_backend_tensor_set(t_txt_idx, text_pos_idx.data(), 0, ggml_nbytes(t_txt_idx));
    for (int64_t s = 0; s < num_steps; ++s) {
        const int64_t bucket = (int64_t) ((double) s / (double) num_steps * (double) num_buckets);
        std::vector<float> tau, tpr; action_sinusoid(bucket, E, AH, tau); timesteps_proj(bucket, tpr);
        ggml_backend_tensor_set(t_tau[s],   tau.data(), 0, ggml_nbytes(t_tau[s]));
        ggml_backend_tensor_set(t_tproj[s], tpr.data(), 0, ggml_nbytes(t_tproj[s]));
    }

    const auto tc0 = std::chrono::steady_clock::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    const auto tc1 = std::chrono::steady_clock::now();
    if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(gr00tn1d6): graph compute failed (%d)\n", (int) st); ggml_gallocr_free(galloc); ggml_free(C); return {}; }
    stats.ms_inference = std::chrono::duration<float, std::milli>(tc1 - tc0).count();

    std::vector<float> out((size_t) AH * AD);
    ggml_backend_tensor_get(actions, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(galloc); ggml_free(C);
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}
