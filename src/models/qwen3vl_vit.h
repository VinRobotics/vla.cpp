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

// Qwen3-VL vision tower, shared by GR00T N1.7 and VLA-JEPA.

#pragma once

#include "model.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace vla {

constexpr float QWEN3VL_MEAN[3] = {0.5f, 0.5f, 0.5f};
constexpr float QWEN3VL_STD [3] = {0.5f, 0.5f, 0.5f};

struct VitLayerW { ggml_tensor *ln1w,*ln1b,*ln2w,*ln2b,*Wqkv,*bqkv,*Wo,*bo,*Wfc1,*bfc1,*Wfc2,*bfc2; };
struct MergerW   { ggml_tensor *nw,*nb,*fc1w,*fc1b,*fc2w,*fc2b; };

inline ggml_tensor * rope2d(ggml_context * C, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_t) {
    const int64_t hd = x->ne[0], S = x->ne[1], Hh = x->ne[2]; const int64_t half = hd / 2;
    ggml_tensor * x1 = ggml_cont(C, ggml_view_3d(C, x, half, S, Hh, x->nb[1], x->nb[2], 0));
    ggml_tensor * x2 = ggml_cont(C, ggml_view_3d(C, x, half, S, Hh, x->nb[1], x->nb[2], (size_t) half * x->nb[0]));
    ggml_tensor * rot = ggml_concat(C, ggml_neg(C, x2), x1, 0);
    return ggml_add(C, ggml_mul(C, x, cos_t), ggml_mul(C, rot, sin_t));
}

inline bool fa_enabled() { static const bool e = (std::getenv("VLA_FLASH_ATTN") != nullptr); return e; }

inline ggml_tensor * flash_attn(ggml_context * C, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                ggml_tensor * mask, float scale) {
    ggml_tensor * kf = (k->type == GGML_TYPE_F16) ? k : ggml_cast(C, k, GGML_TYPE_F16);
    ggml_tensor * vf = (v->type == GGML_TYPE_F16) ? v : ggml_cast(C, v, GGML_TYPE_F16);
    ggml_tensor * o  = ggml_flash_attn_ext(C, q, kf, vf, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    return ggml_reshape_2d(C, o, o->ne[0] * o->ne[1], o->ne[2] * o->ne[3]);
}

inline ggml_tensor * build_vit_layer(ggml_context * C, const VitLayerW & w, ggml_tensor * x,
                                     ggml_tensor * cos_t, ggml_tensor * sin_t,
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
    ggml_tensor * att;
    if (fa_enabled()) {
        ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, hd, heads, seq), 0, 2, 1, 3));
        att = flash_attn(C, Q, K, V, nullptr, scale);
    } else {
        ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, hd, heads, seq), 1, 2, 0, 3));
        ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
        att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), hidden, seq);
    }
    ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, h1, ln_eps), w.ln2w), w.ln2b);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wfc2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wfc1, n2), w.bfc1))), w.bfc2);
    return ggml_add(C, h1, ff);
}

// pre_merge normalizes before the reshape, the deepstack taps after.
inline ggml_tensor * build_merger(ggml_context * C, const MergerW & w, ggml_tensor * x,
                                  int64_t hidden, int64_t merge2, float ln_eps, bool pre_merge) {
    const int64_t n_patches = x->ne[1], c_merged = hidden * merge2 * merge2, n_merged = n_patches / (merge2 * merge2);
    ggml_tensor * m;
    if (pre_merge) {
        ggml_tensor * xn = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.nw), w.nb);
        m = ggml_reshape_2d(C, ggml_cont(C, xn), c_merged, n_merged);
    } else {
        ggml_tensor * mr = ggml_reshape_2d(C, ggml_cont(C, x), c_merged, n_merged);
        m = ggml_add(C, ggml_mul(C, ggml_norm(C, mr, ln_eps), w.nw), w.nb);
    }
    ggml_tensor * z1 = ggml_add(C, ggml_mul_mat(C, w.fc1w, m), w.fc1b);
    return ggml_add(C, ggml_mul_mat(C, w.fc2w, ggml_gelu(C, z1)), w.fc2b);
}

// Patch row/col after the spatial merge.
inline void merge_block_coords(int64_t gh, int64_t gw, int64_t m, std::vector<int64_t> & row, std::vector<int64_t> & col) {
    const int64_t S = gh * gw; row.assign(S, 0); col.assign(S, 0);
    for (int64_t s = 0; s < S; ++s) {
        int64_t t = s; const int64_t wj = t % m; t /= m; const int64_t wi = t % m; t /= m;
        const int64_t bc = t % (gw / m); t /= (gw / m); const int64_t br = t;
        row[s] = br * m + wi; col[s] = bc * m + wj;
    }
}

inline void vit_rope_tables(const std::vector<int64_t> & row, const std::vector<int64_t> & col, int64_t hd, double theta,
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

// Bilinear resample of the pretrained position table onto gh x gw.
inline void interp_pos_embed(const std::vector<float> & table, int64_t num_side, int64_t hidden,
                             const std::vector<int64_t> & row, const std::vector<int64_t> & col, int64_t gh, int64_t gw,
                             std::vector<float> & out) {
    const int64_t S = (int64_t) row.size();
    out.assign((size_t) S * hidden, 0.0f);
    auto src_coord = [&](int64_t k, int64_t g) -> double { return (g <= 1) ? 0.0 : (double) k * (double)(num_side - 1) / (double)(g - 1); };
    for (int64_t s = 0; s < S; ++s) {
        // Clamped, not just h1/w1: a grid that the spatial merge does not divide
        // pushes row/col past gh-1 and would index off the end of the table.
        const double lim = (double) (num_side - 1);
        const double hy = std::min(src_coord(row[s], gh), lim), wx = std::min(src_coord(col[s], gw), lim);
        const int64_t h0 = (int64_t) std::floor(hy), w0 = (int64_t) std::floor(wx);
        const int64_t h1 = std::min(h0 + 1, num_side - 1), w1 = std::min(w0 + 1, num_side - 1);
        const double dh = hy - h0, dw = wx - w0;
        const double c00 = (1 - dh) * (1 - dw), c01 = (1 - dh) * dw, c10 = dh * (1 - dw), c11 = dh * dw;
        const float * T00 = &table[(h0 * num_side + w0) * hidden]; const float * T01 = &table[(h0 * num_side + w1) * hidden];
        const float * T10 = &table[(h1 * num_side + w0) * hidden]; const float * T11 = &table[(h1 * num_side + w1) * hidden];
        for (int64_t c = 0; c < hidden; ++c) out[s * hidden + c] = (float)(c00 * T00[c] + c01 * T01[c] + c10 * T10[c] + c11 * T11[c]);
    }
}

// HWC to flat patches. No resize: the view must already be side x side.
inline bool preprocess_image_patches(const char * arch, const ImageView & v, int64_t side, int64_t ps, int64_t tps,
                                     const std::vector<int64_t> & row, const std::vector<int64_t> & col,
                                     std::vector<float> & out) {
    if (v.w != (int) side || v.h != (int) side || !v.data) {
        std::fprintf(stderr, "vla(%s): image view is %dx%d, expected %lldx%lld\n",
                     arch, v.w, v.h, (long long) side, (long long) side);
        return false;
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
                    const float val = (px(row[s] * ps + ph, col[s] * ps + pw, ch) - QWEN3VL_MEAN[ch]) / QWEN3VL_STD[ch];
                    for (int64_t t = 0; t < tps; ++t) out[s * pf + ch * tps * ps * ps + t * ps * ps + ph * ps + pw] = val;
                }
    return true;
}

}  // namespace vla
