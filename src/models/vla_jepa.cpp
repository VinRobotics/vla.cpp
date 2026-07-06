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
#include <random>
#include <set>
#include <string>
#include <vector>

namespace vla {
namespace {


constexpr float CLIP_MEAN[3] = {0.5f, 0.5f, 0.5f};
constexpr float CLIP_STD [3] = {0.5f, 0.5f, 0.5f};

struct VitLayerW   { ggml_tensor *ln1w,*ln1b,*ln2w,*ln2b,*Wqkv,*bqkv,*Wo,*bo,*Wfc1,*bfc1,*Wfc2,*bfc2; };
struct MergerW     { ggml_tensor *nw,*nb,*fc1w,*fc1b,*fc2w,*fc2b; };
struct Qwen3LayerW { ggml_tensor *attn_norm,*Wq,*Wk,*Wv,*Wo,*q_norm,*k_norm,*ffn_norm,*Wgate,*Wup,*Wdown; };
struct DitLayerW   { ggml_tensor *adaln_w,*adaln_b,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*Wff0,*bff0,*Wff2,*bff2; };

}

struct VlaJepaModelArch : public ModelArchBase {
    VlaJepaModelArch() : ModelArchBase(Arch::VLA_JEPA) {}
    ~VlaJepaModelArch() override;

    std::string           gguf_path;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    bool                  is_gpu      = false;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;

    int64_t vit_hidden=1024, vit_layers=24, vit_heads=16, vit_inter=4096;
    int64_t patch_size=16, temporal_patch=2, spatial_merge=2, vit_num_pos=2304, vit_patch_flat=1536, vit_merged_dim=4096;
    int64_t deepstack_idx[3] = {5, 11, 17};
    int64_t lm_hidden=2048, lm_layers=28, n_q=16, n_kv=8, lm_head_dim=128, lm_inter=6144, vocab=151936;
    int64_t image_token_index=151655, embodied_token_id=151697;
    int64_t image_target_size=256;

    int64_t dit_hidden=768, dit_heads=12, dit_head_dim=64, dit_layers=16, cross_dim=2048, output_dim=1024, time_proj_dim=256;
    int64_t action_dim=7, state_dim=8, action_horizon=7, num_future=32, num_steps=4, num_buckets=1000;
    float   vit_ln_eps=1e-6f, vit_rope_base=10000.0f, lm_rms_eps=1e-6f, lm_rope_base=5000000.0f, connector_ln_eps=1e-6f;
    float   dit_ln_eps=1e-5f, dit_norm_out_eps=1e-6f;

    ggml_tensor *vit_patch_w=nullptr,*vit_patch_b=nullptr,*vit_pos=nullptr;
    std::vector<VitLayerW> vit;
    MergerW deepstack[3];
    MergerW merger;
    ggml_tensor *lm_output_norm=nullptr;
    std::vector<Qwen3LayerW> lm;

    ggml_tensor *ae_l1W=nullptr,*ae_l1b=nullptr,*ae_l2W=nullptr,*ae_l2b=nullptr,*ae_l3W=nullptr,*ae_l3b=nullptr;
    ggml_tensor *se_l1W=nullptr,*se_l1b=nullptr,*se_l2W=nullptr,*se_l2b=nullptr;
    ggml_tensor *ad_l1W=nullptr,*ad_l1b=nullptr,*ad_l2W=nullptr,*ad_l2b=nullptr;
    ggml_tensor *future_tokens=nullptr,*pos_embd=nullptr;
    ggml_tensor *te_l1W=nullptr,*te_l1b=nullptr,*te_l2W=nullptr,*te_l2b=nullptr;
    std::vector<DitLayerW> dit;
    ggml_tensor *po1W=nullptr,*po1b=nullptr,*po2W=nullptr,*po2b=nullptr;

    bool                            caches_ready = false;
    std::vector<int64_t>            c_grow, c_gcol;
    std::vector<float>              c_rope_cos, c_rope_sin, c_pos_interp;
    std::vector<std::vector<float>> c_tau, c_tproj;
    std::vector<float>              c_mask; int64_t c_mask_seq = -1;
    gguf_reader                     io;
    bool build_caches();

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

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

ggml_tensor * build_qwen3_layer(ggml_context * C, const VlaJepaModelArch & m, const Qwen3LayerW & w,
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

ggml_tensor * build_dit_block(ggml_context * C, const VlaJepaModelArch & m, const DitLayerW & w,
                              ggml_tensor * h, ggml_tensor * temb, ggml_tensor * enc) {
    const int64_t hd = m.dit_head_dim, heads = m.dit_heads, dim = m.dit_hidden, Tk = h->ne[1];
    const float scale = 1.0f / std::sqrt((float) hd);
    ggml_tensor * n = adaln(C, h, temb, w.adaln_w, w.adaln_b, dim, m.dit_ln_eps);
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
    ggml_tensor * n3 = ggml_norm(C, h1, m.dit_ln_eps);
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
        std::fprintf(stderr, "vla(vla_jepa): image view is %dx%d, expected %lldx%lld\n", v.w, v.h, (long long) side, (long long) side); return false;
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

bool load_config(const gguf_reader & g, VlaJepaModelArch & m, Config & cfg) {
    auto U = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto fk = [&](const char * s) { static char b[64]; std::snprintf(b, sizeof(b), "vla_jepa.%s", s); return b; };
    U(fk("vit_hidden"), m.vit_hidden); U(fk("vit_layers"), m.vit_layers); U(fk("vit_heads"), m.vit_heads); U(fk("vit_inter"), m.vit_inter);
    U(fk("patch_size"), m.patch_size); U(fk("temporal_patch_size"), m.temporal_patch); U(fk("spatial_merge_size"), m.spatial_merge);
    U(fk("vit_num_position_embeddings"), m.vit_num_pos); U(fk("vit_patch_flat"), m.vit_patch_flat); U(fk("vit_merged_dim"), m.vit_merged_dim);
    U(fk("deepstack_idx_0"), m.deepstack_idx[0]); U(fk("deepstack_idx_1"), m.deepstack_idx[1]); U(fk("deepstack_idx_2"), m.deepstack_idx[2]);
    U(fk("lm_hidden"), m.lm_hidden); U(fk("lm_layers"), m.lm_layers); U(fk("lm_q_heads"), m.n_q); U(fk("lm_kv_heads"), m.n_kv);
    U(fk("lm_head_dim"), m.lm_head_dim); U(fk("lm_inter"), m.lm_inter); U(fk("vocab_size"), m.vocab);
    U(fk("image_token_index"), m.image_token_index); U(fk("embodied_action_token_id"), m.embodied_token_id);
    U(fk("image_target_size"), m.image_target_size);
    U(fk("dit_hidden"), m.dit_hidden); U(fk("dit_heads"), m.dit_heads); U(fk("dit_head_dim"), m.dit_head_dim); U(fk("dit_layers"), m.dit_layers);
    U(fk("cross_dim"), m.cross_dim); U(fk("output_dim"), m.output_dim); U(fk("time_proj_dim"), m.time_proj_dim);
    U(fk("action_dim"), m.action_dim); U(fk("state_dim"), m.state_dim); U(fk("action_horizon"), m.action_horizon);
    U(fk("num_future_tokens"), m.num_future); U(fk("num_inference_timesteps"), m.num_steps); U(fk("num_timestep_buckets"), m.num_buckets);
    if (const char * ns = std::getenv("VLA_NUM_STEPS")) {
        char * end = nullptr; long v = std::strtol(ns, &end, 10);
        if (end && *end == '\0' && v >= 1) { m.num_steps = (int64_t) v; std::fprintf(stderr, "vla(vla_jepa): VLA_NUM_STEPS override → num_steps=%lld\n", (long long) v); }
    }
    F(fk("vit_ln_eps"), m.vit_ln_eps); F(fk("lm_rms_eps"), m.lm_rms_eps); F(fk("connector_ln_eps"), m.connector_ln_eps);
    F(fk("vit_rope_theta"), m.vit_rope_base); F(fk("dit_ln_eps"), m.dit_ln_eps); F(fk("dit_norm_out_eps"), m.dit_norm_out_eps);
    if (g.has(fk("lm_rope_theta"))) m.lm_rope_base = (float) g.f64(fk("lm_rope_theta"));

    cfg = Config{};
    cfg.n_img = (m.image_target_size / m.patch_size / m.spatial_merge) * (m.image_target_size / m.patch_size / m.spatial_merge);
    cfg.n_lang = 1024; cfg.n_state = 1;
    cfg.n_suffix = m.action_horizon; cfg.max_state_dim = m.state_dim; cfg.max_action_dim = m.action_dim;
    cfg.real_state_dim = m.state_dim; cfg.real_action_dim = m.action_dim;
    cfg.hidden = m.lm_hidden; cfg.n_q_heads = m.n_q; cfg.n_kv_heads = m.n_kv; cfg.head_dim = m.lm_head_dim; cfg.n_layers = m.lm_layers;
    cfg.num_steps = (int) m.num_steps; cfg.rms_eps = m.lm_rms_eps;
    cfg.rope_n_dims = (int) m.lm_head_dim; cfg.rope_mode = GGML_ROPE_TYPE_IMROPE; cfg.rope_freq_base = m.lm_rope_base;
    cfg.norm_eps = 1e-8f;
    return true;
}

}

VlaJepaModelArch::~VlaJepaModelArch() {
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> vla_jepa_create(const std::string& mmproj_path,
                                               const std::string& ckpt_path,
                                               const std::string& ) {
    if (!mmproj_path.empty())
        std::printf("vla(vla_jepa): note - mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<VlaJepaModelArch>();
    m->gguf_path   = ckpt_path;
    m->matmul_type = std::getenv("VLA_JEPA_BF16_WEIGHTS") ? GGML_TYPE_BF16 : GGML_TYPE_F32;

    gguf_reader g("vla_jepa");
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has("vla_jepa.architecture")) { std::fprintf(stderr, "vla(vla_jepa): %s is not a vla_jepa GGUF\n", ckpt_path.c_str()); return nullptr; }
    if (!load_config(g, *m, m->cfg)) return nullptr;
    std::printf("vla(vla_jepa): vit=Qwen3-VL %lldd×%lldL (deepstack@{%lld,%lld,%lld}, merge÷%lld)  lm=Qwen3-VL %lldd×%lldL (%lldq/%lldkv×%lld, θ=%g)  "
                "dit-B %lldL×%lldh×%lld(inner %lld, cross %lld, out %lld)  horizon=%lld action_dim=%lld state_dim=%lld future=%lld N_steps=%lld  resident=%s\n",
                (long long) m->vit_hidden, (long long) m->vit_layers, (long long) m->deepstack_idx[0], (long long) m->deepstack_idx[1], (long long) m->deepstack_idx[2], (long long) m->spatial_merge,
                (long long) m->lm_hidden, (long long) m->lm_layers, (long long) m->n_q, (long long) m->n_kv, (long long) m->lm_head_dim, (double) m->lm_rope_base,
                (long long) m->dit_layers, (long long) m->dit_heads, (long long) m->dit_head_dim, (long long) m->dit_hidden, (long long) m->cross_dim, (long long) m->output_dim,
                (long long) m->action_horizon, (long long) m->action_dim, (long long) m->state_dim, (long long) m->num_future, (long long) m->num_steps,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) { m->is_cuda = true; m->is_gpu = true; std::printf("vla(vla_jepa): backend = CUDA (device 0)\n"); }
    else            std::fprintf(stderr, "vla(vla_jepa): ggml_backend_cuda_init failed; falling back to CPU\n");
#elif defined(GGML_USE_METAL)
    m->backend = ggml_backend_metal_init();
    if (m->backend) { m->is_gpu = true; std::printf("vla(vla_jepa): backend = Metal\n"); }
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(vla_jepa): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(vla_jepa): backend = CPU (%d threads)\n", m->n_threads);
    }

    ggml_init_params wp = {  (size_t) 32 * 1024 * 1024,  nullptr,  true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) { std::fprintf(stderr, "vla(vla_jepa): ggml_init(ctx_weights) failed\n"); return nullptr; }
    ggml_context * W = m->ctx_weights;
    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(vla_jepa): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, g.resident_type(gt, type), ggml_n_dims(gt), gt->ne);
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

    m->ae_l1W=mk_f32("ah.act_enc.l1.weight"); m->ae_l1b=mk_f32("ah.act_enc.l1.bias");
    m->ae_l2W=mk_f32("ah.act_enc.l2.weight"); m->ae_l2b=mk_f32("ah.act_enc.l2.bias");
    m->ae_l3W=mk_f32("ah.act_enc.l3.weight"); m->ae_l3b=mk_f32("ah.act_enc.l3.bias");
    m->se_l1W=mk_f32("ah.state_enc.l1.weight"); m->se_l1b=mk_f32("ah.state_enc.l1.bias");
    m->se_l2W=mk_f32("ah.state_enc.l2.weight"); m->se_l2b=mk_f32("ah.state_enc.l2.bias");
    m->ad_l1W=mk_f32("ah.act_dec.l1.weight"); m->ad_l1b=mk_f32("ah.act_dec.l1.bias");
    m->ad_l2W=mk_f32("ah.act_dec.l2.weight"); m->ad_l2b=mk_f32("ah.act_dec.l2.bias");
    m->future_tokens=mk_f32("ah.future_tokens"); m->pos_embd=mk_f32("ah.pos_embd");
    m->te_l1W=mk_mm("ah.time_emb.l1.weight"); m->te_l1b=mk_f32("ah.time_emb.l1.bias");
    m->te_l2W=mk_mm("ah.time_emb.l2.weight"); m->te_l2b=mk_f32("ah.time_emb.l2.bias");
    m->dit.resize(m->dit_layers);
    for (int64_t i = 0; i < m->dit_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "ah.dit.%lld.%s", (long long) i, s); return p; };
        auto & w = m->dit[i];
        w.adaln_w=mk_mm(N("adaln.weight")); w.adaln_b=mk_f32(N("adaln.bias"));
        w.Wq=mk_mm(N("attn_q.weight")); w.bq=mk_f32(N("attn_q.bias")); w.Wk=mk_mm(N("attn_k.weight")); w.bk=mk_f32(N("attn_k.bias"));
        w.Wv=mk_mm(N("attn_v.weight")); w.bv=mk_f32(N("attn_v.bias")); w.Wo=mk_mm(N("attn_o.weight")); w.bo=mk_f32(N("attn_o.bias"));
        w.Wff0=mk_mm(N("ff0.weight")); w.bff0=mk_f32(N("ff0.bias")); w.Wff2=mk_mm(N("ff2.weight")); w.bff2=mk_f32(N("ff2.bias"));
        ok &= w.adaln_w&&w.adaln_b&&w.Wq&&w.bq&&w.Wk&&w.bk&&w.Wv&&w.bv&&w.Wo&&w.bo&&w.Wff0&&w.bff0&&w.Wff2&&w.bff2;
    }
    m->po1W=mk_mm("ah.proj_out1.weight"); m->po1b=mk_f32("ah.proj_out1.bias"); m->po2W=mk_mm("ah.proj_out2.weight"); m->po2b=mk_f32("ah.proj_out2.bias");
    ok &= m->vit_patch_w&&m->vit_patch_b&&m->vit_pos&&m->lm_output_norm&&m->ae_l1W&&m->ae_l2W&&m->ae_l3W&&m->se_l1W&&m->se_l2W&&m->ad_l1W&&m->ad_l2W&&
          m->future_tokens&&m->pos_embd&&m->te_l1W&&m->te_l1b&&m->te_l2W&&m->te_l2b&&m->po1W&&m->po1b&&m->po2W&&m->po2b;
    if (!ok) { std::fprintf(stderr, "vla(vla_jepa): weight tensor setup failed\n"); return nullptr; }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(vla_jepa): ggml_backend_alloc_ctx_tensors failed (OOM?)\n"); return nullptr; }
    for (ggml_tensor * t = ggml_get_first_tensor(W); t; t = ggml_get_next_tensor(W, t)) {
        std::vector<uint8_t> bytes = g.read_convert(ggml_get_name(t), t->type);
        if (bytes.empty() || bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(vla_jepa): failed to load %s (%zu vs %zu bytes)\n", ggml_get_name(t), bytes.size(), ggml_nbytes(t)); return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(vla_jepa): weights resident in %.2f GiB (%s) - Qwen3-VL backbone + deepstack + DiT-B head\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0), m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");
    if (!m->build_caches()) { std::fprintf(stderr, "vla(vla_jepa): build_caches failed\n"); return nullptr; }
    return m;
}

bool VlaJepaModelArch::build_caches() {
    if (caches_ready) return true;
    const int64_t side = image_target_size, ps = patch_size, m2 = spatial_merge;
    const int64_t grid = side / ps;
    const int64_t hd_vit = vit_hidden / vit_heads;
    const int64_t num_side = (int64_t) std::lround(std::sqrt((double) vit_num_pos));

    merge_block_coords(grid, grid, m2, c_grow, c_gcol);
    vit_rope_tables(c_grow, c_gcol, hd_vit, (double) vit_rope_base, c_rope_cos, c_rope_sin);

    if (!io.open(gguf_path)) { std::fprintf(stderr, "vla(vla_jepa): build_caches: io.open(%s) failed\n", gguf_path.c_str()); return false; }
    std::vector<float> pos_table = io.read_f32("vit.pos_embd");
    if (pos_table.empty() || (int64_t) pos_table.size() != vit_num_pos * vit_hidden) {
        std::fprintf(stderr, "vla(vla_jepa): build_caches: vit.pos_embd unreadable\n"); return false;
    }
    interp_pos_embed(pos_table, num_side, vit_hidden, c_grow, c_gcol, grid, grid, c_pos_interp);

    c_tau.assign((size_t) num_steps, {}); c_tproj.assign((size_t) num_steps, {});
    for (int64_t s = 0; s < num_steps; ++s) {
        const int64_t bucket = (int64_t) ((double) s / (double) num_steps * (double) num_buckets);
        action_sinusoid(bucket, dit_hidden, action_horizon, c_tau[(size_t) s]);
        timesteps_proj(bucket, c_tproj[(size_t) s]);
    }
    caches_ready = true;
    return true;
}

std::vector<float> VlaJepaModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    const int64_t H = lm_hidden, E = dit_hidden, AD = action_dim, AH = action_horizon, OUTD = output_dim;
    const int64_t side = image_target_size, ps = patch_size, m2 = spatial_merge;
    const int64_t grid = side / ps, n_patches = grid * grid, K = (grid / m2) * (grid / m2);
    const int64_t hd_vit = vit_hidden / vit_heads;
    const int64_t Nseq = 1 + num_future + AH;
    const char * dump_prefix = std::getenv("VLA_JEPA_DUMP");
    if (!caches_ready) { std::fprintf(stderr, "vla(vla_jepa): caches not ready\n"); return {}; }

    auto dump_t = [&](const char * name, ggml_tensor * t) {
        if (!dump_prefix) return;
        const int64_t n0 = t->ne[0], n1 = t->ne[1];
        std::vector<float> buf((size_t) n0 * std::max<int64_t>(1, n1));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump_prefix, name, (long long) n0, (long long) n1);
        FILE * fp = std::fopen(path, "wb"); if (fp) { std::fwrite(buf.data(), sizeof(float), buf.size(), fp); std::fclose(fp); }
    };

    std::vector<float> x_init((size_t) AH * AD);
    if (in.noise) std::memcpy(x_init.data(), in.noise, x_init.size() * sizeof(float));
    else { std::mt19937 rng((uint32_t) std::chrono::steady_clock::now().time_since_epoch().count()); std::normal_distribution<float> nd(0.f, 1.f); for (auto & v : x_init) v = nd(rng); }

    std::vector<float> cond_host((size_t) H * num_future, 0.0f);
    const char * cond_file = std::getenv("VLA_JEPA_COND");
    if (cond_file) {
        FILE * fp = std::fopen(cond_file, "rb");
        if (!fp) { std::fprintf(stderr, "vla(vla_jepa): VLA_JEPA_COND open failed: %s\n", cond_file); return {}; }
        const size_t want = cond_host.size();
        if (std::fread(cond_host.data(), sizeof(float), want, fp) != want) { std::fprintf(stderr, "vla(vla_jepa): VLA_JEPA_COND short read\n"); std::fclose(fp); return {}; }
        std::fclose(fp);
        std::printf("vla(vla_jepa): conditioning injected from %s (action-head isolation)\n", cond_file);
    } else {
        if (in.precomputed_img_emb) {
            std::fprintf(stderr, "vla(vla_jepa): precomputed_img_emb is not supported. The V-JEPA tower also "
                                 "emits deepstack features that a single embedding buffer cannot carry; pass raw images.\n");
            return {};
        }
        int64_t n_views = in.n_images;
        if (n_views <= 0) { std::fprintf(stderr, "vla(vla_jepa): no images in the request\n"); return {}; }
        std::vector<float> img_emb_host((size_t) n_views * K * H), ds_host[3];
        for (int j = 0; j < 3; ++j) ds_host[j].assign((size_t) n_views * K * H, 0.0f);

        std::vector<float> inj_patches; const char * patches_file = std::getenv("VLA_JEPA_PATCHES");
        if (patches_file) {
            FILE * fp = std::fopen(patches_file, "rb");
            if (!fp) { std::fprintf(stderr, "vla(vla_jepa): VLA_JEPA_PATCHES open failed\n"); return {}; }
            inj_patches.resize((size_t) n_views * n_patches * vit_patch_flat);
            if (std::fread(inj_patches.data(), sizeof(float), inj_patches.size(), fp) != inj_patches.size()) { std::fprintf(stderr, "vla(vla_jepa): VLA_JEPA_PATCHES short read\n"); std::fclose(fp); return {}; }
            std::fclose(fp);
            std::printf("vla(vla_jepa): pixel_values injected from %s\n", patches_file);
        }
        if (inj_patches.empty() && !in.images) { std::fprintf(stderr, "vla(vla_jepa): n_images=%d but the images pointer is null\n", in.n_images); return {}; }

        ggml_init_params vp = { (size_t) 512 * 1024 * 1024, nullptr, true };
        ggml_context * VC = ggml_init(vp);
        if (!VC) { std::fprintf(stderr, "vla(vla_jepa): ggml_init(vision ctx) failed\n"); return {}; }
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
        if (!vga || !ggml_gallocr_alloc_graph(vga, vg)) { std::fprintf(stderr, "vla(vla_jepa): vision gallocr alloc failed\n"); if (vga) ggml_gallocr_free(vga); ggml_free(VC); return {}; }
        const auto tv0 = std::chrono::steady_clock::now();
        std::vector<float> patches;
        bool vok = true;
        for (int64_t v = 0; v < n_views && vok; ++v) {
            if (!inj_patches.empty()) {
                ggml_backend_tensor_set(t_patches, inj_patches.data() + v * n_patches * vit_patch_flat, 0, ggml_nbytes(t_patches));
            } else {
                if (!preprocess_image_patches(in.images[v], side, ps, temporal_patch, c_grow, c_gcol, patches)) { vok = false; break; }
                ggml_backend_tensor_set(t_patches, patches.data(), 0, ggml_nbytes(t_patches));
            }
            ggml_backend_tensor_set(t_pos, c_pos_interp.data(), 0, ggml_nbytes(t_pos));
            ggml_backend_tensor_set(t_cos, c_rope_cos.data(), 0, ggml_nbytes(t_cos));
            ggml_backend_tensor_set(t_sin, c_rope_sin.data(), 0, ggml_nbytes(t_sin));
            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(vla_jepa): vision compute failed\n"); vok = false; break; }
            ggml_backend_tensor_get(vit_embeds, img_emb_host.data() + v * K * H, 0, ggml_nbytes(vit_embeds));
            for (int j = 0; j < 3; ++j) ggml_backend_tensor_get(ds_out[j], ds_host[j].data() + v * K * H, 0, ggml_nbytes(ds_out[j]));
            if (dump_prefix) { char nm[32]; std::snprintf(nm, sizeof(nm), "vit_view%lld", (long long) v); char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump_prefix, nm, (long long) H, (long long) K); FILE * fp = std::fopen(path, "wb"); if (fp) { std::fwrite(img_emb_host.data() + v * K * H, sizeof(float), (size_t) K * H, fp); std::fclose(fp); } }
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tv0).count();
        ggml_gallocr_free(vga); ggml_free(VC);
        if (!vok) return {};
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
            std::fprintf(stderr, "vla(vla_jepa): lang_tokens has %lld image slots but n_img=%lld\n", (long long) n_img_slots, (long long) n_img); return {};
        }
        const int64_t SEQ = (int64_t) input_ids.size();

        std::vector<float> inputs_embeds((size_t) SEQ * H);
        if (!io.fetch_rows_f32("token_embd.weight", input_ids, inputs_embeds.data(), H)) return {};
        { int64_t k = 0; for (int64_t p = 0; p < SEQ; ++p) if (input_ids[p] == (int32_t) image_token_index) { std::memcpy(inputs_embeds.data() + p * H, img_emb_host.data() + k * H, H * sizeof(float)); ++k; } }

        std::vector<int32_t> image_pos_idx, emb_pos_idx;
        for (int64_t p = 0; p < SEQ; ++p) {
            if (input_ids[p] == (int32_t) image_token_index) image_pos_idx.push_back((int32_t) p);
            if (input_ids[p] == (int32_t) embodied_token_id)  emb_pos_idx.push_back((int32_t) p);
        }
        if ((int64_t) emb_pos_idx.size() != num_future) { std::fprintf(stderr, "vla(vla_jepa): found %zu embodied tokens, expected %lld\n", emb_pos_idx.size(), (long long) num_future); return {}; }

        std::vector<std::vector<float>> ds_pad(3);
        for (int j = 0; j < 3; ++j) { ds_pad[j].assign((size_t) SEQ * H, 0.0f); for (int64_t k = 0; k < n_img; ++k) std::memcpy(ds_pad[j].data() + (size_t) image_pos_idx[k] * H, ds_host[j].data() + (size_t) k * H, H * sizeof(float)); }

        ggml_init_params cp = { (size_t) 512 * 1024 * 1024, nullptr, true };
        ggml_context * C = ggml_init(cp);
        if (!C) { std::fprintf(stderr, "vla(vla_jepa): ggml_init(LM ctx) failed\n"); return {}; }
        ggml_tensor * t_embeds = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);   ggml_set_input(t_embeds);
        ggml_tensor * t_pos2   = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4 * SEQ);  ggml_set_input(t_pos2);
        ggml_tensor * t_lmmask = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ); ggml_set_input(t_lmmask);
        ggml_tensor * t_emb_idx= ggml_new_tensor_1d(C, GGML_TYPE_I32, num_future); ggml_set_input(t_emb_idx);
        ggml_tensor * t_ds[3];
        for (int j = 0; j < 3; ++j) { t_ds[j] = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ); ggml_set_input(t_ds[j]); }
        ggml_tensor * hh = t_embeds;
        for (int64_t i = 0; i < lm_layers; ++i) {
            hh = build_qwen3_layer(C, *this, lm[i], hh, t_pos2, t_lmmask, SEQ);
            if (i < 3) hh = ggml_add(C, hh, t_ds[i]);
        }
        ggml_tensor * eagle = hh;
        ggml_set_output(eagle);
        ggml_tensor * conditioning = ggml_get_rows(C, eagle, t_emb_idx);
        ggml_set_output(conditioning);
        ggml_cgraph * lg = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(lg, conditioning);
        ggml_gallocr_t lga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!lga || !ggml_gallocr_alloc_graph(lga, lg)) { std::fprintf(stderr, "vla(vla_jepa): LM gallocr alloc failed\n"); if (lga) ggml_gallocr_free(lga); ggml_free(C); return {}; }

        ggml_backend_tensor_set(t_embeds, inputs_embeds.data(), 0, ggml_nbytes(t_embeds));

        {
            const int64_t llm_grid = side / ps / m2;
            std::vector<int32_t> pp((size_t) 4 * SEQ, 0);
            int64_t st = 0, st_idx = 0;
            while (st < SEQ) {
                int64_t img_start = -1;
                for (int64_t i = st; i < SEQ; ++i) if (input_ids[i] == (int32_t) image_token_index) { img_start = i; break; }
                const int64_t text_end = (img_start < 0) ? SEQ : img_start;
                const int64_t text_len = text_end - st;
                for (int64_t i = 0; i < text_len; ++i) { const int32_t p = (int32_t) (i + st_idx); pp[0*SEQ+(st+i)]=p; pp[1*SEQ+(st+i)]=p; pp[2*SEQ+(st+i)]=p; }
                if (img_start < 0) { st_idx += text_len; break; }
                int64_t img_end = img_start; while (img_end < SEQ && input_ids[img_end] == (int32_t) image_token_index) ++img_end;
                const int64_t n_img_tokens = img_end - img_start;
                const int64_t this_t = n_img_tokens / (llm_grid * llm_grid);
                const int64_t image_offset = text_len + st_idx;
                for (int64_t tt = 0; tt < this_t; ++tt) for (int64_t hy = 0; hy < llm_grid; ++hy) for (int64_t wx = 0; wx < llm_grid; ++wx) {
                    const int64_t tok = img_start + (tt * llm_grid + hy) * llm_grid + wx;
                    pp[0*SEQ+tok] = (int32_t)(image_offset + tt); pp[1*SEQ+tok] = (int32_t)(image_offset + hy); pp[2*SEQ+tok] = (int32_t)(image_offset + wx);
                }
                int64_t max_image_pos = this_t - 1; if (llm_grid - 1 > max_image_pos) max_image_pos = llm_grid - 1;
                st_idx = image_offset + max_image_pos + 1; st = img_end;
            }
            std::memcpy(pp.data() + (size_t) 3 * SEQ, pp.data(), (size_t) SEQ * sizeof(int32_t));
            ggml_backend_tensor_set(t_pos2, pp.data(), 0, ggml_nbytes(t_pos2));
        }
        if (c_mask_seq != SEQ) { c_mask.assign((size_t) SEQ * SEQ, 0.0f); const float NEG = -std::numeric_limits<float>::infinity(); for (int64_t q = 0; q < SEQ; ++q) for (int64_t kv = 0; kv < SEQ; ++kv) c_mask[q * SEQ + kv] = (kv <= q) ? 0.0f : NEG; c_mask_seq = SEQ; }
        ggml_backend_tensor_set(t_lmmask, c_mask.data(), 0, ggml_nbytes(t_lmmask));
        ggml_backend_tensor_set(t_emb_idx, emb_pos_idx.data(), 0, ggml_nbytes(t_emb_idx));
        for (int j = 0; j < 3; ++j) ggml_backend_tensor_set(t_ds[j], ds_pad[j].data(), 0, ggml_nbytes(t_ds[j]));

        const auto tp0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, lg) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(vla_jepa): LM compute failed\n"); ggml_gallocr_free(lga); ggml_free(C); return {}; }
        stats.ms_prefill = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tp0).count();
        if (dump_prefix) { dump_t("eagle", eagle); dump_t("conditioning", conditioning); }
        ggml_backend_tensor_get(conditioning, cond_host.data(), 0, cond_host.size() * sizeof(float));
        ggml_gallocr_free(lga); ggml_free(C);
    }

    ggml_init_params hp = { (size_t) 256 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(hp);
    if (!C) { std::fprintf(stderr, "vla(vla_jepa): ggml_init(head ctx) failed\n"); return {}; }
    ggml_tensor * t_cond  = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, num_future); ggml_set_input(t_cond);
    ggml_tensor * t_state = ggml_new_tensor_2d(C, GGML_TYPE_F32, state_dim, 1);  ggml_set_input(t_state);
    ggml_tensor * t_x0    = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH);        ggml_set_input(t_x0);
    std::vector<ggml_tensor *> t_tau(num_steps), t_tproj(num_steps);
    for (int64_t s = 0; s < num_steps; ++s) { t_tau[s] = ggml_new_tensor_2d(C, GGML_TYPE_F32, E, AH); ggml_set_input(t_tau[s]); t_tproj[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, time_proj_dim); ggml_set_input(t_tproj[s]); }

    ggml_tensor * state_features = ggml_add(C, ggml_mul_mat(C, se_l2W, ggml_relu(C, ggml_add(C, ggml_mul_mat(C, se_l1W, t_state), se_l1b))), se_l2b);
    ggml_tensor * future = future_tokens;
    const float dt = 1.0f / (float) num_steps;
    std::vector<ggml_tensor *> step_seq(num_steps), step_pred(num_steps), step_vel(num_steps), step_act(num_steps);

    ggml_tensor * actions = t_x0;
    for (int64_t s = 0; s < num_steps; ++s) {

        ggml_tensor * temb = ggml_add(C, ggml_mul_mat(C, te_l2W, ggml_silu(C, ggml_add(C, ggml_mul_mat(C, te_l1W, t_tproj[s]), te_l1b))), te_l2b);

        ggml_tensor * a_emb = ggml_add(C, ggml_mul_mat(C, ae_l1W, actions), ae_l1b);
        ggml_tensor * cat   = ggml_concat(C, a_emb, t_tau[s], 0);
        ggml_tensor * x2    = ggml_silu(C, ggml_add(C, ggml_mul_mat(C, ae_l2W, cat), ae_l2b));
        ggml_tensor * af    = ggml_add(C, ggml_mul_mat(C, ae_l3W, x2), ae_l3b);
        af = ggml_add(C, af, ggml_view_2d(C, pos_embd, E, AH, pos_embd->nb[1], 0));
        ggml_tensor * seq = ggml_concat(C, ggml_concat(C, state_features, future, 1), af, 1);
        step_seq[s] = seq;
        ggml_tensor * x = seq;
        for (int64_t i = 0; i < dit_layers; ++i) {
            ggml_tensor * enc = (i % 2 == 0) ? t_cond : nullptr;
            x = build_dit_block(C, *this, dit[i], x, temb, enc);
        }

        ggml_tensor * po = ggml_add(C, ggml_mul_mat(C, po1W, ggml_silu(C, temb)), po1b);
        ggml_tensor * sh = ggml_view_1d(C, po, dit_hidden, 0), * sc = ggml_view_1d(C, po, dit_hidden, (size_t) dit_hidden * sizeof(float));
        ggml_tensor * xn = ggml_norm(C, x, dit_norm_out_eps);
        ggml_tensor * h_mod = ggml_add(C, ggml_add(C, xn, ggml_mul(C, xn, sc)), sh);
        ggml_tensor * model_output = ggml_add(C, ggml_mul_mat(C, po2W, h_mod), po2b);
        step_pred[s] = model_output;

        ggml_tensor * last = ggml_cont(C, ggml_view_2d(C, model_output, OUTD, AH, model_output->nb[1], (size_t) (Nseq - AH) * model_output->nb[1]));
        ggml_tensor * vel = ggml_add(C, ggml_mul_mat(C, ad_l2W, ggml_relu(C, ggml_add(C, ggml_mul_mat(C, ad_l1W, last), ad_l1b))), ad_l2b);
        step_vel[s] = vel;
        actions = ggml_add(C, actions, ggml_scale(C, vel, dt));
        step_act[s] = actions;
        if (dump_prefix) { ggml_set_output(step_seq[s]); ggml_set_output(step_pred[s]); ggml_set_output(step_vel[s]); ggml_set_output(step_act[s]); }
    }
    ggml_set_output(actions);
    ggml_cgraph * hg = ggml_new_graph_custom(C, 65536, false);
    ggml_build_forward_expand(hg, actions);
    if (dump_prefix) for (int64_t s = 0; s < num_steps; ++s) { ggml_build_forward_expand(hg, step_seq[s]); ggml_build_forward_expand(hg, step_pred[s]); ggml_build_forward_expand(hg, step_vel[s]); ggml_build_forward_expand(hg, step_act[s]); }
    ggml_gallocr_t hga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!hga || !ggml_gallocr_alloc_graph(hga, hg)) { std::fprintf(stderr, "vla(vla_jepa): head gallocr alloc failed\n"); if (hga) ggml_gallocr_free(hga); ggml_free(C); return {}; }

    ggml_backend_tensor_set(t_cond, cond_host.data(), 0, ggml_nbytes(t_cond));
    { std::vector<float> st(state_dim, 0.0f); for (int64_t i = 0; i < state_dim; ++i) st[i] = in.state ? in.state[i] : 0.0f; ggml_backend_tensor_set(t_state, st.data(), 0, ggml_nbytes(t_state)); }
    ggml_backend_tensor_set(t_x0, x_init.data(), 0, ggml_nbytes(t_x0));
    for (int64_t s = 0; s < num_steps; ++s) { ggml_backend_tensor_set(t_tau[s], c_tau[(size_t) s].data(), 0, ggml_nbytes(t_tau[s])); ggml_backend_tensor_set(t_tproj[s], c_tproj[(size_t) s].data(), 0, ggml_nbytes(t_tproj[s])); }

    const auto td0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(backend, hg) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(vla_jepa): head compute failed\n"); ggml_gallocr_free(hga); ggml_free(C); return {}; }
    stats.ms_denoise = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - td0).count();
    stats.ms_inference = stats.ms_prefill + stats.ms_denoise;

    if (dump_prefix) for (int64_t s = 0; s < num_steps; ++s) {
        char nm[48];
        std::snprintf(nm, sizeof(nm), "step%lld_seq", (long long) s); dump_t(nm, step_seq[s]);
        std::snprintf(nm, sizeof(nm), "step%lld_dit_pred", (long long) s); dump_t(nm, step_pred[s]);
        std::snprintf(nm, sizeof(nm), "step%lld_velocity", (long long) s); dump_t(nm, step_vel[s]);
        std::snprintf(nm, sizeof(nm), "step%lld_actions", (long long) s); dump_t(nm, step_act[s]);
    }

    std::vector<float> out((size_t) AH * AD);
    ggml_backend_tensor_get(actions, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(hga); ggml_free(C);
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}
