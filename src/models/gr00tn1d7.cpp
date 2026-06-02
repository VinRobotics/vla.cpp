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
        if (!gctx) { std::fprintf(stderr, "vla(gr00tn1d7): gguf_init_from_file failed for %s\n", path.c_str()); return false; }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) { std::fprintf(stderr, "vla(gr00tn1d7): fopen failed for %s\n", path.c_str()); return false; }
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
        if (id < 0) { std::fprintf(stderr, "vla(gr00tn1d7): missing tensor %s\n", name); return false; }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }
    std::vector<float> read_f32(const char * name) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(gr00tn1d7): missing tensor %s\n", name); return {}; }
        const int64_t n = ggml_nelements(t);
        std::vector<float> out(n);
        if (t->type == GGML_TYPE_F32) { if (!read_raw(name, out.data())) return {}; }
        else if (t->type == GGML_TYPE_BF16) { std::vector<ggml_bf16_t> tmp(n); if (!read_raw(name, tmp.data())) return {}; ggml_bf16_to_fp32_row(tmp.data(), out.data(), n); }
        else { std::fprintf(stderr, "vla(gr00tn1d7): tensor %s unsupported type %d\n", name, (int) t->type); return {}; }
        return out;
    }
    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        std::vector<float> f = read_f32(name);
        if (f.empty()) return {};
        const int64_t n = (int64_t) f.size();
        if (target == GGML_TYPE_F32)  { std::vector<uint8_t> o(n * 4);  std::memcpy(o.data(), f.data(), o.size()); return o; }
        if (target == GGML_TYPE_BF16) { std::vector<uint8_t> o(n * 2);  ggml_fp32_to_bf16_row(f.data(), reinterpret_cast<ggml_bf16_t *>(o.data()), n); return o; }
        std::fprintf(stderr, "vla(gr00tn1d7): unsupported resident type %d for %s\n", (int) target, name); return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids, float * dst, int64_t cols) {
        const ggml_tensor * t = meta(name);
        if (!t || t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) { std::fprintf(stderr, "vla(gr00tn1d7): %s shape unfit for row-fetch\n", name); return false; }
        const int64_t rows = t->ne[1];
        const int64_t id   = gguf_find_tensor(gctx, name);
        const size_t  base = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t  elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
        const size_t  rb   = (size_t) cols * elsz;
        std::vector<uint8_t> row(rb);
        for (size_t k = 0; k < row_ids.size(); ++k) {
            const int32_t r = row_ids[k];
            if (r < 0 || r >= rows) { std::fprintf(stderr, "vla(gr00tn1d7): row %d out of range for %s\n", r, name); return false; }
            if (std::fseek(fp, (long) (base + (size_t) r * rb), SEEK_SET) != 0) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

constexpr float CLIP_MEAN[3] = {0.5f, 0.5f, 0.5f};
constexpr float CLIP_STD [3] = {0.5f, 0.5f, 0.5f};

struct VitLayerW   { ggml_tensor *ln1w,*ln1b,*ln2w,*ln2b,*Wqkv,*bqkv,*Wo,*bo,*Wfc1,*bfc1,*Wfc2,*bfc2; };
struct MergerW     { ggml_tensor *nw,*nb,*fc1w,*fc1b,*fc2w,*fc2b; };
struct VlsaLayerW  { ggml_tensor *n1w,*n1b,*n3w,*n3b,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*Wff0,*bff0,*Wff2,*bff2; };
struct Qwen3LayerW { ggml_tensor *attn_norm,*Wq,*Wk,*Wv,*Wo,*q_norm,*k_norm,*ffn_norm,*Wgate,*Wup,*Wdown; };
struct DitLayerW   { ggml_tensor *adaln_w,*adaln_b,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*Wff0,*bff0,*Wff2,*bff2; };

}

struct Gr00tN1d7ModelArch : public ModelArchBase {
    Gr00tN1d7ModelArch() : ModelArchBase(Arch::GR00T_N1_7) {}
    ~Gr00tN1d7ModelArch() override;

    std::string           gguf_path;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    int                   n_threads   = 4;
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;

    int64_t vit_hidden=1024, vit_layers=24, vit_heads=16, vit_inter=4096;
    int64_t patch_size=16, temporal_patch=2, spatial_merge=2, vit_num_pos=2304, vit_patch_flat=1536, vit_merged_dim=4096;
    int64_t deepstack_idx[3] = {5, 11, 17};
    int64_t lm_hidden=2048, lm_layers=16, n_q=16, n_kv=8, lm_head_dim=128, lm_inter=6144, vocab=151936, image_token_index=151655;
    int64_t vlsa_layers=4, vlsa_heads=32, vlsa_head_dim=64, vlsa_ff_inner=8192;
    int64_t bb_embed_dim=2048, in_embed_dim=1536, dit_hidden=1536, dit_heads=32, dit_head_dim=48, dit_layers=32, dit_interleave=1, attend_text_every_n=2;
    int64_t action_horizon=40, action_dim=132, max_state_dim=132;
    int64_t num_steps=4, num_buckets=1000, max_embodiments=32, max_seq_len=1024;
    int64_t image_target_size=256;
    float   vit_ln_eps=1e-6f, vit_rope_base=10000.0f, lm_rms_eps=1e-6f, lm_rope_base=5000000.0f;
    float   vlln_eps=1e-5f, vlsa_ln_eps=1e-5f, ln_eps=1e-5f, norm_out_eps=1e-6f, connector_ln_eps=1e-6f;
    int64_t embodiment_id = 2;

    ggml_tensor *vit_patch_w=nullptr,*vit_patch_b=nullptr,*vit_pos=nullptr;
    std::vector<VitLayerW> vit;
    MergerW deepstack[3];
    MergerW merger;
    ggml_tensor *lm_output_norm=nullptr;
    std::vector<Qwen3LayerW> lm;
    ggml_tensor *vlln_w=nullptr,*vlln_b=nullptr;
    std::vector<VlsaLayerW> vlsa;
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

ggml_tensor * rope2d(ggml_context * C, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_t) {
    const int64_t hd = x->ne[0], S = x->ne[1], Hh = x->ne[2]; const int64_t half = hd / 2;
    ggml_tensor * x1 = ggml_cont(C, ggml_view_3d(C, x, half, S, Hh, x->nb[1], x->nb[2], 0));
    ggml_tensor * x2 = ggml_cont(C, ggml_view_3d(C, x, half, S, Hh, x->nb[1], x->nb[2], (size_t) half * x->nb[0]));
    ggml_tensor * rot = ggml_concat(C, ggml_neg(C, x2), x1, 0);
    return ggml_add(C, ggml_mul(C, x, cos_t), ggml_mul(C, rot, sin_t));
}

ggml_tensor * build_vit_layer(ggml_context * C, const VitLayerW & w, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_t,
                              int64_t seq, int64_t heads, int64_t hd, int64_t hidden, float ln_eps) {
    const float scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * n1 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.ln1w), w.ln1b);
    ggml_tensor * qkv = ggml_add(C, ggml_mul_mat(C, w.Wqkv, n1), w.bqkv);
    ggml_tensor * q = ggml_cont(C, ggml_view_2d(C, qkv, hidden, seq, qkv->nb[1], 0));
    ggml_tensor * k = ggml_cont(C, ggml_view_2d(C, qkv, hidden, seq, qkv->nb[1], (size_t) hidden * qkv->nb[0]));
    ggml_tensor * v = ggml_cont(C, ggml_view_2d(C, qkv, hidden, seq, qkv->nb[1], (size_t) 2 * hidden * qkv->nb[0]));
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, hd, heads, seq), 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, hd, heads, seq), 0, 2, 1, 3));
    Q = rope2d(C, Q, cos_t, sin_t); K = rope2d(C, K, cos_t, sin_t);
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, hd, heads, seq), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), hidden, seq);
    ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, h1, ln_eps), w.ln2w), w.ln2b);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wfc2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wfc1, n2), w.bfc1))), w.bfc2);
    return ggml_add(C, h1, ff);
}

ggml_tensor * build_merger(ggml_context * C, const MergerW & w, ggml_tensor * x, int64_t hidden, int64_t merge2, float ln_eps, bool pre_merge) {
    ggml_tensor * m;
    if (pre_merge) {
        ggml_tensor * xn = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.nw), w.nb);
        const int64_t n_patches = x->ne[1], c_merged = hidden * merge2 * merge2, n_merged = n_patches / (merge2 * merge2);
        m = ggml_reshape_2d(C, ggml_cont(C, xn), c_merged, n_merged);
    } else {
        const int64_t n_patches = x->ne[1], c_merged = hidden * merge2 * merge2, n_merged = n_patches / (merge2 * merge2);
        ggml_tensor * mr = ggml_reshape_2d(C, ggml_cont(C, x), c_merged, n_merged);
        m = ggml_add(C, ggml_mul(C, ggml_norm(C, mr, ln_eps), w.nw), w.nb);
    }
    ggml_tensor * z1 = ggml_add(C, ggml_mul_mat(C, w.fc1w, m), w.fc1b);
    return ggml_add(C, ggml_mul_mat(C, w.fc2w, ggml_gelu(C, z1)), w.fc2b);
}

ggml_tensor * build_vlsa_layer(ggml_context * C, const VlsaLayerW & w, ggml_tensor * x,
                               int64_t seq, int64_t heads, int64_t hd, int64_t hidden, float ln_eps) {
    const float scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * n1 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.n1w), w.n1b);
    ggml_tensor * q = ggml_add(C, ggml_mul_mat(C, w.Wq, n1), w.bq);
    ggml_tensor * k = ggml_add(C, ggml_mul_mat(C, w.Wk, n1), w.bk);
    ggml_tensor * v = ggml_add(C, ggml_mul_mat(C, w.Wv, n1), w.bv);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, hd, heads, seq), 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, hd, heads, seq), 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, hd, heads, seq), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), hidden, seq);
    ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n3 = ggml_add(C, ggml_mul(C, ggml_norm(C, h1, ln_eps), w.n3w), w.n3b);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wff2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wff0, n3), w.bff0))), w.bff2);
    return ggml_add(C, h1, ff);
}

ggml_tensor * build_qwen3_layer(ggml_context * C, const Gr00tN1d7ModelArch & m, const Qwen3LayerW & w,
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
    int sections[4] = { 24, 20, 20, 0 };
    ggml_tensor * qr = ggml_rope_multi(C, qn, positions, nullptr, (int) hd, sections, GGML_ROPE_TYPE_IMROPE, 0, m.lm_rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * kr = ggml_rope_multi(C, kn, positions, nullptr, (int) hd, sections, GGML_ROPE_TYPE_IMROPE, 0, m.lm_rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
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

ggml_tensor * build_dit_block(ggml_context * C, const Gr00tN1d7ModelArch & m, const DitLayerW & w,
                              ggml_tensor * h, ggml_tensor * temb, ggml_tensor * enc ) {
    const int64_t hd = m.dit_head_dim, heads = m.dit_heads, dim = m.dit_hidden, Tk = h->ne[1];
    const float scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * n = adaln(C, h, temb, w.adaln_w, w.adaln_b, dim, m.ln_eps);
    ggml_tensor * kv = enc ? enc : n;
    const int64_t Tkv = kv->ne[1];
    ggml_tensor * q = ggml_add(C, ggml_mul_mat(C, w.Wq, n),  w.bq);
    ggml_tensor * k = ggml_add(C, ggml_mul_mat(C, w.Wk, kv), w.bk);
    ggml_tensor * v = ggml_add(C, ggml_mul_mat(C, w.Wv, kv), w.bv);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, hd, heads, Tk),  0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, hd, heads, Tkv), 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, hd, heads, Tkv), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), dim, Tk);
    ggml_tensor * h1 = ggml_add(C, h, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n3 = ggml_norm(C, h1, m.ln_eps);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wff2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wff0, n3), w.bff0))), w.bff2);
    return ggml_add(C, h1, ff);
}

void merge_block_coords(int64_t gh, int64_t gw, int64_t m, std::vector<int64_t> & row, std::vector<int64_t> & col) {
    const int64_t S = gh * gw; row.assign(S, 0); col.assign(S, 0);
    for (int64_t s = 0; s < S; ++s) {
        int64_t t = s; const int64_t wj = t % m; t /= m; const int64_t wi = t % m; t /= m;
        const int64_t bc = t % (gw / m); t /= (gw / m); const int64_t br = t;
        row[s] = br * m + wi; col[s] = bc * m + wj;
    }
}

void vit_rope_tables(const std::vector<int64_t> & row, const std::vector<int64_t> & col, int64_t hd, double theta,
                     std::vector<float> & cos_t, std::vector<float> & sin_t) {
    const int64_t S = (int64_t) row.size(), nf = hd / 4;
    std::vector<double> invf(nf);
    for (int64_t i = 0; i < nf; ++i) invf[i] = 1.0 / std::pow(theta, (double)(2 * i) / (double)(hd / 2));
    cos_t.assign((size_t) S * hd, 0.0f); sin_t.assign((size_t) S * hd, 0.0f);
    for (int64_t s = 0; s < S; ++s) {
        std::vector<double> emb(hd);
        for (int64_t i = 0; i < nf; ++i) { emb[i] = (double) row[s] * invf[i]; emb[nf + i] = (double) col[s] * invf[i]; }
        for (int64_t i = 0; i < hd / 2; ++i) emb[hd / 2 + i] = emb[i];
        for (int64_t i = 0; i < hd; ++i) { cos_t[s * hd + i] = (float) std::cos(emb[i]); sin_t[s * hd + i] = (float) std::sin(emb[i]); }
    }
}

void interp_pos_embed(const std::vector<float> & table, int64_t num_side, int64_t hidden,
                      const std::vector<int64_t> & row, const std::vector<int64_t> & col, int64_t gh, int64_t gw,
                      std::vector<float> & out) {
    const int64_t S = (int64_t) row.size();
    out.assign((size_t) S * hidden, 0.0f);
    auto src_coord = [&](int64_t k, int64_t g) -> double { return (g <= 1) ? 0.0 : (double) k * (double)(num_side - 1) / (double)(g - 1); };
    for (int64_t s = 0; s < S; ++s) {
        const double hy = src_coord(row[s], gh), wx = src_coord(col[s], gw);
        const int64_t h0 = (int64_t) std::floor(hy), w0 = (int64_t) std::floor(wx);
        const int64_t h1 = std::min(h0 + 1, num_side - 1), w1 = std::min(w0 + 1, num_side - 1);
        const double dh = hy - h0, dw = wx - w0;
        const double c00 = (1 - dh) * (1 - dw), c01 = (1 - dh) * dw, c10 = dh * (1 - dw), c11 = dh * dw;
        const float * T00 = &table[(h0 * num_side + w0) * hidden]; const float * T01 = &table[(h0 * num_side + w1) * hidden];
        const float * T10 = &table[(h1 * num_side + w0) * hidden]; const float * T11 = &table[(h1 * num_side + w1) * hidden];
        for (int64_t c = 0; c < hidden; ++c) out[s * hidden + c] = (float)(c00 * T00[c] + c01 * T01[c] + c10 * T10[c] + c11 * T11[c]);
    }
}

bool preprocess_image_patches(const ImageView & v, int64_t side, int64_t ps, int64_t tps,
                              const std::vector<int64_t> & row, const std::vector<int64_t> & col, std::vector<float> & out) {
    if (v.w != (int) side || v.h != (int) side || !v.data) {
        std::fprintf(stderr, "vla(gr00tn1d7): image view is %dx%d, expected %lldx%lld\n", v.w, v.h, (long long) side, (long long) side); return false;
    }
    const int64_t S = (int64_t) row.size(), pf = 3 * tps * ps * ps;
    out.assign((size_t) pf * S, 0.0f);
    auto px = [&](int64_t r, int64_t c, int64_t ch) -> float {
        if (v.format == PixelFormat::U8) return ((const uint8_t *) v.data)[(r * side + c) * 3 + ch] / 255.0f;
        return ((const float *) v.data)[(r * side + c) * 3 + ch];
    };
    for (int64_t s = 0; s < S; ++s)
        for (int64_t ch = 0; ch < 3; ++ch)
            for (int64_t ph = 0; ph < ps; ++ph)
                for (int64_t pw = 0; pw < ps; ++pw) {
                    const float val = (px(row[s] * ps + ph, col[s] * ps + pw, ch) - CLIP_MEAN[ch]) / CLIP_STD[ch];
                    for (int64_t t = 0; t < tps; ++t) out[s * pf + ch * tps * ps * ps + t * ps * ps + ph * ps + pw] = val;
                }
    return true;
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

bool load_config(const gguf_reader & g, Gr00tN1d7ModelArch & m, Config & cfg) {
    auto U = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto fk = [&](const char * s) { static char b[64]; std::snprintf(b, sizeof(b), "gr00t_n1_7.%s", s); return b; };
    U(fk("vit_hidden"), m.vit_hidden); U(fk("vit_layers"), m.vit_layers); U(fk("vit_heads"), m.vit_heads); U(fk("vit_inter"), m.vit_inter);
    U(fk("patch_size"), m.patch_size); U(fk("temporal_patch_size"), m.temporal_patch); U(fk("spatial_merge_size"), m.spatial_merge);
    U(fk("vit_num_position_embeddings"), m.vit_num_pos); U(fk("vit_patch_flat"), m.vit_patch_flat); U(fk("vit_merged_dim"), m.vit_merged_dim);
    U(fk("deepstack_idx_0"), m.deepstack_idx[0]); U(fk("deepstack_idx_1"), m.deepstack_idx[1]); U(fk("deepstack_idx_2"), m.deepstack_idx[2]);
    U(fk("lm_hidden"), m.lm_hidden); U(fk("lm_layers_used"), m.lm_layers); U(fk("lm_q_heads"), m.n_q); U(fk("lm_kv_heads"), m.n_kv);
    U(fk("lm_head_dim"), m.lm_head_dim); U(fk("lm_inter"), m.lm_inter); U(fk("vocab_size"), m.vocab); U(fk("image_token_index"), m.image_token_index);
    U(fk("vlsa_layers"), m.vlsa_layers); U(fk("vlsa_heads"), m.vlsa_heads); U(fk("vlsa_head_dim"), m.vlsa_head_dim); U(fk("vlsa_ff_inner"), m.vlsa_ff_inner);
    U(fk("backbone_embedding_dim"), m.bb_embed_dim); U(fk("input_embedding_dim"), m.in_embed_dim);
    U(fk("dit_hidden"), m.dit_hidden); U(fk("dit_heads"), m.dit_heads); U(fk("dit_head_dim"), m.dit_head_dim); U(fk("dit_layers"), m.dit_layers); U(fk("dit_interleave"), m.dit_interleave);
    U(fk("attend_text_every_n_blocks"), m.attend_text_every_n);
    U(fk("action_horizon"), m.action_horizon); U(fk("action_dim"), m.action_dim); U(fk("max_state_dim"), m.max_state_dim);
    U(fk("num_inference_timesteps"), m.num_steps); U(fk("num_timestep_buckets"), m.num_buckets); U(fk("max_num_embodiments"), m.max_embodiments); U(fk("max_seq_len"), m.max_seq_len);
    U(fk("image_target_size"), m.image_target_size);
    F(fk("vit_ln_eps"), m.vit_ln_eps); F(fk("lm_rms_eps"), m.lm_rms_eps); F(fk("ln_eps"), m.ln_eps); F(fk("norm_out_eps"), m.norm_out_eps);
    F(fk("vlln_eps"), m.vlln_eps); F(fk("vlsa_ln_eps"), m.vlsa_ln_eps); F(fk("connector_ln_eps"), m.connector_ln_eps); F(fk("vit_rope_theta"), m.vit_rope_base);
    if (g.has(fk("lm_rope_theta"))) m.lm_rope_base = (float) g.f64(fk("lm_rope_theta"));

    m.embodiment_id = 2;
    {
        const std::string js = g.str(fk("embodiment_id_mapping"));
        auto lookup = [&](const char * key) -> long {
            const std::string k = std::string("\"") + key + "\"";
            size_t p = js.find(k); if (p == std::string::npos) return -1;
            p = js.find(':', p + k.size()); if (p == std::string::npos) return -1;
            return std::strtol(js.c_str() + p + 1, nullptr, 10);
        };
        long ls = lookup("libero_sim"); if (ls >= 0) m.embodiment_id = ls;
        if (const char * e = std::getenv("VLA_GR00T_EMBODIMENT")) {
            char * end = nullptr; long v = std::strtol(e, &end, 10);
            if (end && *end == '\0') m.embodiment_id = v;
            else { long id = lookup(e); if (id >= 0) m.embodiment_id = id; else std::fprintf(stderr, "vla(gr00tn1d7): embodiment tag '%s' not in embodiment_id_mapping; using id %lld\n", e, (long long) m.embodiment_id); }
        }
    }
    if (m.embodiment_id < 0 || m.embodiment_id >= m.max_embodiments) { std::fprintf(stderr, "vla(gr00tn1d7): embodiment id %lld out of range [0,%lld)\n", (long long) m.embodiment_id, (long long) m.max_embodiments); return false; }

    cfg = Config{};
    cfg.n_img = 64; cfg.n_lang = m.max_seq_len; cfg.n_state = 1;
    cfg.n_suffix = m.action_horizon; cfg.max_state_dim = m.max_state_dim; cfg.max_action_dim = m.action_dim;
    cfg.real_state_dim = m.max_state_dim; cfg.real_action_dim = m.action_dim;
    cfg.hidden = m.lm_hidden; cfg.n_q_heads = m.n_q; cfg.n_kv_heads = m.n_kv; cfg.head_dim = m.lm_head_dim; cfg.n_layers = m.lm_layers;
    cfg.num_steps = (int) m.num_steps; cfg.rms_eps = m.lm_rms_eps;
    cfg.rope_n_dims = (int) m.lm_head_dim; cfg.rope_mode = GGML_ROPE_TYPE_NEOX; cfg.rope_freq_base = m.lm_rope_base;
    cfg.norm_eps = 1e-8f;
    return true;
}

}

Gr00tN1d7ModelArch::~Gr00tN1d7ModelArch() {
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> gr00t_n1_7_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string& ) {
    if (!mmproj_path.empty())
        std::printf("vla(gr00tn1d7): note - mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<Gr00tN1d7ModelArch>();
    m->gguf_path   = ckpt_path;
    m->matmul_type = std::getenv("VLA_GR00T_BF16_WEIGHTS") ? GGML_TYPE_BF16 : GGML_TYPE_F32;

    gguf_reader g;
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has("gr00t_n1_7.architecture")) { std::fprintf(stderr, "vla(gr00tn1d7): %s is not a gr00t_n1_7 GGUF\n", ckpt_path.c_str()); return nullptr; }
    if (!load_config(g, *m, m->cfg)) return nullptr;
    std::printf("vla(gr00tn1d7): vit=Qwen3-VL %lldd×%lldL×%lldh (Conv3d patch %lld², temporal %lld; learned pos %lld + 2D rope; deepstack@{%lld,%lld,%lld}; merge÷%lld)  "
                "lm=Qwen3-VL %lldd×%lldL (%lldq/%lldkv×%lld, θ=%g)  vlsa=%lldL×%lldh×%lld  dit=AlternateVLDiT %lldL×%lldh×%lld(inner %lld) attend_text_every_n=%lld  "
                "in_emb=%lld  horizon=%lld action_dim=%lld max_state=%lld N_steps=%lld  embodiment=%lld  resident=%s\n",
                (long long) m->vit_hidden, (long long) m->vit_layers, (long long) m->vit_heads, (long long) m->patch_size, (long long) m->temporal_patch,
                (long long) m->vit_num_pos, (long long) m->deepstack_idx[0], (long long) m->deepstack_idx[1], (long long) m->deepstack_idx[2], (long long) m->spatial_merge,
                (long long) m->lm_hidden, (long long) m->lm_layers, (long long) m->n_q, (long long) m->n_kv, (long long) m->lm_head_dim, (double) m->lm_rope_base,
                (long long) m->vlsa_layers, (long long) m->vlsa_heads, (long long) m->vlsa_head_dim,
                (long long) m->dit_layers, (long long) m->dit_heads, (long long) m->dit_head_dim, (long long) m->dit_hidden, (long long) m->attend_text_every_n, (long long) m->in_embed_dim,
                (long long) m->action_horizon, (long long) m->action_dim, (long long) m->max_state_dim, (long long) m->num_steps, (long long) m->embodiment_id,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) { m->is_cuda = true; std::printf("vla(gr00tn1d7): backend = CUDA (device 0)\n"); }
    else            std::fprintf(stderr, "vla(gr00tn1d7): ggml_backend_cuda_init failed; falling back to CPU\n");
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(gr00tn1d7): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(gr00tn1d7): backend = CPU (%d threads)\n", m->n_threads);
    }

    ggml_init_params wp = {  (size_t) 32 * 1024 * 1024,  nullptr,  true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) { std::fprintf(stderr, "vla(gr00tn1d7): ggml_init(ctx_weights) failed\n"); return nullptr; }
    ggml_context * W = m->ctx_weights;
    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(gr00tn1d7): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, type, ggml_n_dims(gt), gt->ne);
        ggml_set_name(t, name); return t;
    };
    auto mk_mm  = [&](const char * name) { return mk(name, m->matmul_type); };
    auto mk_f32 = [&](const char * name) { return mk(name, GGML_TYPE_F32); };

    bool ok = true;

    m->vit_patch_w = mk_mm("vit.patch_embd.weight"); m->vit_patch_b = mk_f32("vit.patch_embd.bias"); m->vit_pos = mk_f32("vit.pos_embd");
    m->vit.resize(m->vit_layers);
    for (int64_t i = 0; i < m->vit_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "vit.blk.%lld.%s", (long long) i, s); return p; };
        auto & w = m->vit[i];
        w.ln1w=mk_f32(N("ln1.weight")); w.ln1b=mk_f32(N("ln1.bias")); w.ln2w=mk_f32(N("ln2.weight")); w.ln2b=mk_f32(N("ln2.bias"));
        w.Wqkv=mk_mm(N("attn_qkv.weight")); w.bqkv=mk_f32(N("attn_qkv.bias")); w.Wo=mk_mm(N("attn_o.weight")); w.bo=mk_f32(N("attn_o.bias"));
        w.Wfc1=mk_mm(N("fc1.weight")); w.bfc1=mk_f32(N("fc1.bias")); w.Wfc2=mk_mm(N("fc2.weight")); w.bfc2=mk_f32(N("fc2.bias"));
        ok &= w.ln1w&&w.ln1b&&w.ln2w&&w.ln2b&&w.Wqkv&&w.bqkv&&w.Wo&&w.bo&&w.Wfc1&&w.bfc1&&w.Wfc2&&w.bfc2;
    }
    for (int j = 0; j < 3; ++j) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "vit.deepstack.%d.%s", j, s); return p; };
        auto & w = m->deepstack[j];
        w.nw=mk_f32(N("norm.weight")); w.nb=mk_f32(N("norm.bias")); w.fc1w=mk_mm(N("fc1.weight")); w.fc1b=mk_f32(N("fc1.bias")); w.fc2w=mk_mm(N("fc2.weight")); w.fc2b=mk_f32(N("fc2.bias"));
        ok &= w.nw&&w.nb&&w.fc1w&&w.fc1b&&w.fc2w&&w.fc2b;
    }
    { auto & w = m->merger;
      w.nw=mk_f32("vit.merger.norm.weight"); w.nb=mk_f32("vit.merger.norm.bias"); w.fc1w=mk_mm("vit.merger.fc1.weight"); w.fc1b=mk_f32("vit.merger.fc1.bias"); w.fc2w=mk_mm("vit.merger.fc2.weight"); w.fc2b=mk_f32("vit.merger.fc2.bias");
      ok &= w.nw&&w.nb&&w.fc1w&&w.fc1b&&w.fc2w&&w.fc2b; }

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
    m->vlsa.resize(m->vlsa_layers);
    for (int64_t i = 0; i < m->vlsa_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "aex.vlsa.%lld.%s", (long long) i, s); return p; };
        auto & w = m->vlsa[i];
        w.n1w=mk_f32(N("norm1.weight")); w.n1b=mk_f32(N("norm1.bias")); w.n3w=mk_f32(N("norm3.weight")); w.n3b=mk_f32(N("norm3.bias"));
        w.Wq=mk_mm(N("attn_q.weight")); w.bq=mk_f32(N("attn_q.bias")); w.Wk=mk_mm(N("attn_k.weight")); w.bk=mk_f32(N("attn_k.bias"));
        w.Wv=mk_mm(N("attn_v.weight")); w.bv=mk_f32(N("attn_v.bias")); w.Wo=mk_mm(N("attn_o.weight")); w.bo=mk_f32(N("attn_o.bias"));
        w.Wff0=mk_mm(N("ff0.weight")); w.bff0=mk_f32(N("ff0.bias")); w.Wff2=mk_mm(N("ff2.weight")); w.bff2=mk_f32(N("ff2.bias"));
        ok &= w.n1w&&w.n1b&&w.n3w&&w.n3b&&w.Wq&&w.bq&&w.Wk&&w.bk&&w.Wv&&w.bv&&w.Wo&&w.bo&&w.Wff0&&w.bff0&&w.Wff2&&w.bff2;
    }
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
    ok &= m->vit_patch_w&&m->vit_patch_b&&m->vit_pos&&m->lm_output_norm&&m->vlln_w&&m->vlln_b&&m->se_l1W&&m->se_l1b&&m->se_l2W&&m->se_l2b&&
          m->ae_W1W&&m->ae_W1b&&m->ae_W2W&&m->ae_W2b&&m->ae_W3W&&m->ae_W3b&&m->ad_l1W&&m->ad_l1b&&m->ad_l2W&&m->ad_l2b&&m->pos_embd&&m->te_l1W&&m->te_l1b&&m->te_l2W&&m->te_l2b&&m->po1W&&m->po1b&&m->po2W&&m->po2b;
    if (!ok) { std::fprintf(stderr, "vla(gr00tn1d7): weight tensor setup failed\n"); return nullptr; }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(gr00tn1d7): ggml_backend_alloc_ctx_tensors failed (OOM?)\n"); return nullptr; }
    for (ggml_tensor * t = ggml_get_first_tensor(W); t; t = ggml_get_next_tensor(W, t)) {
        std::vector<uint8_t> bytes = g.read_convert(ggml_get_name(t), t->type);
        if (bytes.empty() || bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(gr00tn1d7): failed to load %s (%zu vs %zu bytes)\n", ggml_get_name(t), bytes.size(), ggml_nbytes(t)); return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(gr00tn1d7): weights resident in %.2f GiB (%s) - incl. Qwen3-VL vision tower + deepstack + vl_self_attention; embodiment id %lld\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0), m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16", (long long) m->embodiment_id);
    return m;
}

std::vector<float> Gr00tN1d7ModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    const int64_t H = lm_hidden, E = in_embed_dim;
    const int64_t side = image_target_size;
    const int64_t ps = patch_size, m2 = spatial_merge;
    const int64_t grid = side / ps;
    const int64_t n_patches = grid * grid;
    const int64_t K = (grid / m2) * (grid / m2);
    const int64_t hd_vit = vit_hidden / vit_heads;
    const int64_t num_side = (int64_t) std::lround(std::sqrt((double) vit_num_pos));
    const int64_t AD = action_dim, AH = action_horizon, Nsa = 1 + AH;

    std::vector<int64_t> grow, gcol; merge_block_coords(grid, grid, m2, grow, gcol);
    std::vector<float> rope_cos, rope_sin; vit_rope_tables(grow, gcol, hd_vit, (double) vit_rope_base, rope_cos, rope_sin);

    gguf_reader g;
    if (!g.open(gguf_path)) return {};
    std::vector<float> pos_interp;
    {
        std::vector<float> pos_table = g.read_f32("vit.pos_embd");
        if (pos_table.empty() || (int64_t) pos_table.size() != vit_num_pos * vit_hidden) { std::fprintf(stderr, "vla(gr00tn1d7): vit.pos_embd unreadable\n"); return {}; }
        interp_pos_embed(pos_table, num_side, vit_hidden, grow, gcol, grid, grid, pos_interp);
    }

    int64_t n_views = 0;
    std::vector<float> img_emb_host, ds_host[3];
    const float * img_emb_ptr = nullptr;
    if (in.precomputed_img_emb && in.n_img_views > 0) {
        n_views = in.n_img_views; img_emb_ptr = in.precomputed_img_emb;

        for (int j = 0; j < 3; ++j) ds_host[j].assign((size_t) n_views * K * H, 0.0f);
    } else if (in.images && in.n_images > 0) {
        n_views = in.n_images;
        img_emb_host.assign((size_t) n_views * K * H, 0.0f);
        for (int j = 0; j < 3; ++j) ds_host[j].assign((size_t) n_views * K * H, 0.0f);

        ggml_init_params vp = { (size_t) 512 * 1024 * 1024, nullptr, true };
        ggml_context * VC = ggml_init(vp);
        if (!VC) { std::fprintf(stderr, "vla(gr00tn1d7): ggml_init(vision ctx) failed\n"); return {}; }
        ggml_tensor * t_patches = ggml_new_tensor_2d(VC, GGML_TYPE_F32, vit_patch_flat, n_patches); ggml_set_input(t_patches);
        ggml_tensor * t_pos     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, vit_hidden, n_patches);     ggml_set_input(t_pos);
        ggml_tensor * t_cos     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, hd_vit, n_patches);          ggml_set_input(t_cos);
        ggml_tensor * t_sin     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, hd_vit, n_patches);          ggml_set_input(t_sin);
        ggml_tensor * h = ggml_add(VC, ggml_add(VC, ggml_mul_mat(VC, vit_patch_w, t_patches), vit_patch_b), t_pos);

        ggml_set_output(h);
        ggml_tensor * stash[3] = {nullptr, nullptr, nullptr};
        for (int64_t i = 0; i < vit_layers; ++i) {
            h = build_vit_layer(VC, vit[i], h, t_cos, t_sin, n_patches, vit_heads, hd_vit, vit_hidden, vit_ln_eps);
            ggml_set_output(h);
            for (int j = 0; j < 3; ++j) if (i == deepstack_idx[j]) stash[j] = h;
        }
        ggml_tensor * ds_out[3];
        for (int j = 0; j < 3; ++j) { ds_out[j] = build_merger(VC, deepstack[j], stash[j] ? stash[j] : h, vit_hidden, m2, connector_ln_eps, false); ggml_set_output(ds_out[j]); }
        ggml_tensor * vit_embeds = build_merger(VC, merger, h, vit_hidden, m2, connector_ln_eps, true);
        ggml_set_output(vit_embeds);
        ggml_cgraph * vg = ggml_new_graph_custom(VC, 16384, false);
        ggml_build_forward_expand(vg, vit_embeds);
        for (int j = 0; j < 3; ++j) ggml_build_forward_expand(vg, ds_out[j]);
        ggml_gallocr_t vga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!vga || !ggml_gallocr_alloc_graph(vga, vg)) { std::fprintf(stderr, "vla(gr00tn1d7): vision gallocr alloc failed\n"); if (vga) ggml_gallocr_free(vga); ggml_free(VC); return {}; }
        const auto tv0 = std::chrono::steady_clock::now();
        std::vector<float> patches;
        bool vok = true;
        for (int64_t v = 0; v < n_views && vok; ++v) {
            if (!preprocess_image_patches(in.images[v], side, ps, temporal_patch, grow, gcol, patches)) { vok = false; break; }

            ggml_backend_tensor_set(t_pos, pos_interp.data(), 0, ggml_nbytes(t_pos));
            ggml_backend_tensor_set(t_cos, rope_cos.data(), 0, ggml_nbytes(t_cos));
            ggml_backend_tensor_set(t_sin, rope_sin.data(), 0, ggml_nbytes(t_sin));
            ggml_backend_tensor_set(t_patches, patches.data(), 0, ggml_nbytes(t_patches));
            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(gr00tn1d7): vision compute failed\n"); vok = false; break; }
            ggml_backend_tensor_get(vit_embeds, img_emb_host.data() + v * K * H, 0, ggml_nbytes(vit_embeds));
            for (int j = 0; j < 3; ++j) ggml_backend_tensor_get(ds_out[j], ds_host[j].data() + v * K * H, 0, ggml_nbytes(ds_out[j]));
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tv0).count();
        ggml_gallocr_free(vga); ggml_free(VC);
        if (!vok) return {};
        img_emb_ptr = img_emb_host.data();
    } else {
        std::fprintf(stderr, "vla(gr00tn1d7): no images and no precomputed_img_emb in the request\n"); return {};
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
        std::fprintf(stderr, "vla(gr00tn1d7): lang_tokens has %lld image-token slots but n_img=%lld; expected 0 (v1 fallback) or %lld (chat-template path)\n",
                     (long long) n_img_slots, (long long) n_img, (long long) n_img);
        return {};
    }
    const int64_t SEQ = (int64_t) input_ids.size();
    if (SEQ > max_seq_len) { std::fprintf(stderr, "vla(gr00tn1d7): prompt too long (%lld > %lld)\n", (long long) SEQ, (long long) max_seq_len); return {}; }

    std::vector<float> inputs_embeds((size_t) SEQ * H);
    if (!g.fetch_rows_f32("token_embd.weight", input_ids, inputs_embeds.data(), H)) return {};
    {   int64_t k = 0;
        for (int64_t p = 0; p < SEQ; ++p) if (input_ids[p] == (int32_t) image_token_index) {
            if (k >= n_img) { std::fprintf(stderr, "vla(gr00tn1d7): more <image> tokens than ViT embeds\n"); return {}; }
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
        std::fprintf(stderr, "vla(gr00tn1d7): internal: built %zu image positions, expected %lld\n", image_pos_idx.size(), (long long) n_img); return {};
    }

    std::vector<std::vector<float>> ds_pad(3);
    const bool inject_deepstack = (in.images && in.n_images > 0);
    if (inject_deepstack) for (int j = 0; j < 3; ++j) {
        ds_pad[j].assign((size_t) SEQ * H, 0.0f);
        for (int64_t k = 0; k < n_img; ++k) {
            std::memcpy(ds_pad[j].data() + (size_t) image_pos_idx[k] * H,
                        ds_host[j].data() + (size_t) k * H, H * sizeof(float));
        }
    }

    std::vector<float> x_init((size_t) AH * AD);
    if (in.noise) std::memcpy(x_init.data(), in.noise, x_init.size() * sizeof(float));
    else { std::mt19937 rng((uint32_t) std::chrono::steady_clock::now().time_since_epoch().count()); std::normal_distribution<float> nd(0.f, 1.f); for (auto & v : x_init) v = nd(rng); }

    ggml_init_params cp = { (size_t) 256 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) { std::fprintf(stderr, "vla(gr00tn1d7): ggml_init(ctx_compute) failed\n"); return {}; }

    ggml_tensor * t_embeds = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);          ggml_set_input(t_embeds);
    ggml_tensor * t_pos    = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4 * SEQ);         ggml_set_input(t_pos);
    ggml_tensor * t_lmmask = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ);        ggml_set_input(t_lmmask);
    ggml_tensor * t_state  = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_state_dim, 1);ggml_set_input(t_state);
    ggml_tensor * t_x0     = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH);          ggml_set_input(t_x0);
    ggml_tensor * t_ds[3] = {nullptr,nullptr,nullptr};
    if (inject_deepstack) for (int j = 0; j < 3; ++j) { t_ds[j] = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ); ggml_set_input(t_ds[j]); }

    ggml_tensor * t_img_idx = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_img);    ggml_set_input(t_img_idx);
    ggml_tensor * t_txt_idx = (SEQ_TXT > 0)
        ? ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ_TXT) : nullptr;
    if (t_txt_idx) ggml_set_input(t_txt_idx);
    std::vector<ggml_tensor *> t_tau(num_steps), t_tproj(num_steps);
    for (int64_t s = 0; s < num_steps; ++s) {
        t_tau[s]   = ggml_new_tensor_2d(C, GGML_TYPE_F32, E, AH); ggml_set_input(t_tau[s]);
        t_tproj[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, 256);   ggml_set_input(t_tproj[s]);
    }

    ggml_tensor * h = t_embeds;
    std::vector<ggml_tensor*> lm_h_dump;
    for (int64_t i = 0; i < lm_layers; ++i) {
        h = build_qwen3_layer(C, *this, lm[i], h, t_pos, t_lmmask, SEQ);
        if (inject_deepstack && i < 3) h = ggml_add(C, h, t_ds[i]);
        if (std::getenv("VLA_GR00T_N17_DUMP")) { ggml_set_output(h); lm_h_dump.push_back(h); }
    }

    ggml_tensor * eagle = h;
    ggml_set_name(eagle, "eagle"); ggml_set_output(eagle);

    ggml_tensor * vl_embs = ggml_add(C, ggml_mul(C, ggml_norm(C, eagle, vlln_eps), vlln_w), vlln_b);
    std::vector<ggml_tensor*> vlsa_dump;
    if (std::getenv("VLA_GR00T_N17_DUMP")) { ggml_set_output(vl_embs); vlsa_dump.push_back(vl_embs); }
    for (int64_t i = 0; i < vlsa_layers; ++i) {
        vl_embs = build_vlsa_layer(C, vlsa[i], vl_embs, SEQ, vlsa_heads, vlsa_head_dim, bb_embed_dim, vlsa_ln_eps);
        if (std::getenv("VLA_GR00T_N17_DUMP")) { ggml_set_output(vl_embs); vlsa_dump.push_back(vl_embs); }
    }
    ggml_set_name(vl_embs, "vl_embs"); ggml_set_output(vl_embs);

    ggml_tensor * vl_img = ggml_get_rows(C, vl_embs, t_img_idx);
    ggml_tensor * vl_txt = (t_txt_idx ? ggml_get_rows(C, vl_embs, t_txt_idx) : vl_img);

    ggml_tensor * state_features = cat_linear(C, se_l2W, se_l2b, embodiment_id, ggml_relu(C, cat_linear(C, se_l1W, se_l1b, embodiment_id, t_state)));

    const float dt = 1.0f / (float) num_steps;
    const int64_t every2 = 2 * attend_text_every_n;
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
            hh = build_dit_block(C, *this, dit[i], hh, temb, enc);
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
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "vla(gr00tn1d7): gallocr alloc failed\n"); if (galloc) ggml_gallocr_free(galloc); ggml_free(C); return {}; }

    ggml_backend_tensor_set(t_embeds, inputs_embeds.data(), 0, ggml_nbytes(t_embeds));
    {

        const int64_t llm_grid_h = image_target_size / patch_size / spatial_merge;
        const int64_t llm_grid_w = llm_grid_h;

        std::vector<int32_t> pp((size_t) 4 * SEQ, 0);
        int64_t st = 0, st_idx = 0;
        while (st < SEQ) {
            int64_t img_start = -1;
            for (int64_t i = st; i < SEQ; ++i) if (input_ids[i] == (int32_t) image_token_index) { img_start = i; break; }
            const int64_t text_end = (img_start < 0) ? SEQ : img_start;
            const int64_t text_len = text_end - st;
            for (int64_t i = 0; i < text_len; ++i) {
                const int32_t p = (int32_t) (i + st_idx);
                pp[0 * SEQ + (st + i)] = p;
                pp[1 * SEQ + (st + i)] = p;
                pp[2 * SEQ + (st + i)] = p;
            }
            if (img_start < 0) { st_idx += text_len; st = SEQ; break; }
            int64_t img_end = img_start;
            while (img_end < SEQ && input_ids[img_end] == (int32_t) image_token_index) ++img_end;
            const int64_t n_img_tokens = img_end - img_start;
            if (n_img_tokens % (llm_grid_h * llm_grid_w) != 0) {
                std::fprintf(stderr, "vla(gr00tn1d7): image run length %lld not a multiple of %lld (post-merge grid)\n",
                             (long long) n_img_tokens, (long long) (llm_grid_h * llm_grid_w));
                ggml_gallocr_free(galloc); ggml_free(C); return {};
            }
            const int64_t this_t = n_img_tokens / (llm_grid_h * llm_grid_w);
            const int64_t image_offset = text_len + st_idx;
            for (int64_t tt = 0; tt < this_t; ++tt) {
                for (int64_t hh = 0; hh < llm_grid_h; ++hh) {
                    for (int64_t ww = 0; ww < llm_grid_w; ++ww) {
                        const int64_t k = (tt * llm_grid_h + hh) * llm_grid_w + ww;
                        const int64_t tok = img_start + k;
                        pp[0 * SEQ + tok] = (int32_t) (image_offset + tt);
                        pp[1 * SEQ + tok] = (int32_t) (image_offset + hh);
                        pp[2 * SEQ + tok] = (int32_t) (image_offset + ww);
                    }
                }
            }
            int64_t max_image_pos = this_t - 1;
            if (llm_grid_h - 1 > max_image_pos) max_image_pos = llm_grid_h - 1;
            if (llm_grid_w - 1 > max_image_pos) max_image_pos = llm_grid_w - 1;
            st_idx = image_offset + max_image_pos + 1;
            st = img_end;
        }

        std::memcpy(pp.data() + (size_t) 3 * SEQ, pp.data() + (size_t) 0 * SEQ, (size_t) SEQ * sizeof(int32_t));
        ggml_backend_tensor_set(t_pos, pp.data(), 0, ggml_nbytes(t_pos));
    }
    { std::vector<float> mk((size_t) SEQ * SEQ); const float NEG = -std::numeric_limits<float>::infinity();
      for (int64_t q = 0; q < SEQ; ++q) for (int64_t kv = 0; kv < SEQ; ++kv) mk[q * SEQ + kv] = (kv <= q) ? 0.0f : NEG;
      ggml_backend_tensor_set(t_lmmask, mk.data(), 0, ggml_nbytes(t_lmmask)); }
    { std::vector<float> st(max_state_dim, 0.0f); for (int64_t i = 0; i < max_state_dim; ++i) st[i] = in.state ? in.state[i] : 0.0f; ggml_backend_tensor_set(t_state, st.data(), 0, ggml_nbytes(t_state)); }
    ggml_backend_tensor_set(t_x0, x_init.data(), 0, ggml_nbytes(t_x0));
    if (inject_deepstack) for (int j = 0; j < 3; ++j) ggml_backend_tensor_set(t_ds[j], ds_pad[j].data(), 0, ggml_nbytes(t_ds[j]));
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
    if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(gr00tn1d7): graph compute failed (%d)\n", (int) st); ggml_gallocr_free(galloc); ggml_free(C); return {}; }
    stats.ms_inference = std::chrono::duration<float, std::milli>(tc1 - tc0).count();

    std::vector<float> out((size_t) AH * AD);
    ggml_backend_tensor_get(actions, out.data(), 0, out.size() * sizeof(float));

    if (const char * dump = std::getenv("VLA_GR00T_N17_DUMP")) {
        auto dump_t = [&](const char * name, ggml_tensor * t) {
            const int64_t n0 = t->ne[0], n1 = t->ne[1];
            std::vector<float> buf((size_t) n0 * n1);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump, name, (long long) n0, (long long) n1);
            FILE * fp = std::fopen(path, "wb");
            if (fp) { std::fwrite(buf.data(), sizeof(float), buf.size(), fp); std::fclose(fp); std::fprintf(stderr, "vla(gr00tn1d7): dumped %s shape=(%lld,%lld) to %s\n", name, (long long) n1, (long long) n0, path); }
        };
        dump_t("eagle",  eagle);
        dump_t("vl_embs", vl_embs);

        for (size_t li = 0; li < lm_h_dump.size(); ++li) { char nm[32]; std::snprintf(nm, sizeof(nm), "lm_h_%02zu", li); dump_t(nm, lm_h_dump[li]); }

        for (size_t vi = 0; vi < vlsa_dump.size(); ++vi) { char nm[32]; std::snprintf(nm, sizeof(nm), "vlsa_%02zu", vi); dump_t(nm, vlsa_dump[vi]); }

        if (inject_deepstack) {
            auto dump_host = [&](const char * name, const float * data, int64_t n0, int64_t n1) {
                char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump, name, (long long) n0, (long long) n1);
                FILE * fp = std::fopen(path, "wb");
                if (fp) { std::fwrite(data, sizeof(float), (size_t) n0 * n1, fp); std::fclose(fp); std::fprintf(stderr, "vla(gr00tn1d7): dumped %s shape=(%lld,%lld) to %s\n", name, (long long) n1, (long long) n0, path); }
            };

            for (int64_t v = 0; v < n_views; ++v) {
                char nm[32]; std::snprintf(nm, sizeof(nm), "ds0_view%lld", (long long) v);
                dump_host(nm, ds_host[0].data() + v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "ds1_view%lld", (long long) v);
                dump_host(nm, ds_host[1].data() + v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "ds2_view%lld", (long long) v);
                dump_host(nm, ds_host[2].data() + v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "vit_view%lld", (long long) v);
                dump_host(nm, img_emb_host.data() + v * K * H, H, K);
            }
        }
    }
    ggml_gallocr_free(galloc); ggml_free(C);
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}
