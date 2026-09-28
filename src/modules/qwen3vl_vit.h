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

#include "backend.h"
#include "gguf_reader.h"
#include "layers/attn.h"
#include "loader.h"
#include "layers/rope.h"
#include "model.h"
#include "modules/preprocess.h"
#include "scratch_ctx.h"

#include "ggml.h"
#include "options.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vla {

constexpr float QWEN3VL_MEAN[3] = {0.5f, 0.5f, 0.5f};
constexpr float QWEN3VL_STD [3] = {0.5f, 0.5f, 0.5f};

struct VitLayerW { ggml_tensor *ln1w,*ln1b,*ln2w,*ln2b,*Wqkv,*bqkv,*Wo,*bo,*Wfc1,*bfc1,*Wfc2,*bfc2; };
struct MergerW   { ggml_tensor *nw,*nb,*fc1w,*fc1b,*fc2w,*fc2b; };
struct VitIO     { ggml_tensor *t_patches=nullptr,*t_pos=nullptr,*t_cos=nullptr,*t_sin=nullptr,*embeds=nullptr,*ds[3]={}; };

struct Qwen3VLTower {
    int64_t hidden = 1024, layers = 24, heads = 16, patch = 16, temporal = 2, merge = 2;
    int64_t num_pos = 2304, patch_flat = 1536, side = 256;
    int64_t deepstack_idx[3] = {5, 11, 17};
    float   ln_eps = 1e-6f, rope_theta = 10000.0f, conn_eps = 1e-6f;

    std::vector<VitLayerW> blk;
    MergerW                deepstack[3];
    MergerW                merger;
    ggml_tensor *          patch_w = nullptr;
    ggml_tensor *          patch_b = nullptr;
    ggml_tensor *          pos     = nullptr;

    std::vector<int64_t> row, col;
    std::vector<float>   rope_cos, rope_sin, pos_interp;

    int64_t grid() const {
        return side/patch;
    }
    int64_t n_tokens() const {
        return (grid()/merge)*(grid()/merge);
    }

    bool load_config(const char * arch, const gguf_reader & g, const char * ns);

    bool build_caches(const char * arch, gguf_reader & io);

    bool encode(const char * arch, ggml_backend_t backend, graph_cache<int, VitIO> & cache, const ImageView * images,
                int64_t n_views, const float * patches_in, std::vector<float> & emb, std::vector<float> (&ds)[3]) const;

    void declare(WeightLoader & L, const char * prefix) {
        patch_w = L.gemm("%s.patch_embd.weight", prefix);
        patch_b = L.f32 ("%s.patch_embd.bias",   prefix);
        pos     = L.f32 ("%s.pos_embd",          prefix);

        blk.resize(layers);
        for (int64_t i=0; i<layers; ++i) {
            VitLayerW & w = blk[i];
            w.ln1w = L.f32 ("%s.blk.%lld.ln1.weight",      prefix, (long long)i);
            w.ln1b = L.f32 ("%s.blk.%lld.ln1.bias",        prefix, (long long)i);
            w.ln2w = L.f32 ("%s.blk.%lld.ln2.weight",      prefix, (long long)i);
            w.ln2b = L.f32 ("%s.blk.%lld.ln2.bias",        prefix, (long long)i);
            w.Wqkv = L.gemm("%s.blk.%lld.attn_qkv.weight", prefix, (long long)i);
            w.bqkv = L.f32 ("%s.blk.%lld.attn_qkv.bias",   prefix, (long long)i);
            w.Wo   = L.gemm("%s.blk.%lld.attn_o.weight",   prefix, (long long)i);
            w.bo   = L.f32 ("%s.blk.%lld.attn_o.bias",     prefix, (long long)i);
            w.Wfc1 = L.gemm("%s.blk.%lld.fc1.weight",      prefix, (long long)i);
            w.bfc1 = L.f32 ("%s.blk.%lld.fc1.bias",        prefix, (long long)i);
            w.Wfc2 = L.gemm("%s.blk.%lld.fc2.weight",      prefix, (long long)i);
            w.bfc2 = L.f32 ("%s.blk.%lld.fc2.bias",        prefix, (long long)i);
        }

        auto merger_w = [&](MergerW & w, const char * name) {
            w.nw   = L.f32 ("%s.norm.weight", name);
            w.nb   = L.f32 ("%s.norm.bias",   name);
            w.fc1w = L.gemm("%s.fc1.weight",  name);
            w.fc1b = L.f32 ("%s.fc1.bias",    name);
            w.fc2w = L.gemm("%s.fc2.weight",  name);
            w.fc2b = L.f32 ("%s.fc2.bias",    name);
        };

        char name[160];
        for (int j=0; j<3; ++j) {
            std::snprintf(name, sizeof(name), "%s.deepstack.%d", prefix, j);
            merger_w(deepstack[j], name);
        }
        std::snprintf(name, sizeof(name), "%s.merger", prefix);
        merger_w(merger, name);
    }
};

inline ggml_tensor * build_vit_layer(ggml_context * C, const VitLayerW & w, ggml_tensor * x,
                                     ggml_tensor * cos_t, ggml_tensor * sin_t,
                                     int64_t seq, int64_t heads, int64_t hd, int64_t hidden, float ln_eps) {
    const float scale = 1.0f/std::sqrt((float) hd);
    ggml_tensor * n1 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.ln1w), w.ln1b);
    ggml_tensor * qkv = ggml_add(C, ggml_mul_mat(C, w.Wqkv, n1), w.bqkv);
    ggml_tensor * q = ggml_cont(C, ggml_view_2d(C, qkv, hidden, seq, qkv->nb[1], 0));
    ggml_tensor * k = ggml_cont(C, ggml_view_2d(C, qkv, hidden, seq, qkv->nb[1], (size_t) hidden * qkv->nb[0]));
    ggml_tensor * v = ggml_cont(C, ggml_view_2d(C, qkv, hidden, seq, qkv->nb[1], (size_t) 2*hidden * qkv->nb[0]));
    ggml_tensor * Q = rope_2d(C, to_heads(C, q, hd, heads, seq), cos_t, sin_t);
    ggml_tensor * K = rope_2d(C, to_heads(C, k, hd, heads, seq), cos_t, sin_t);
    ggml_tensor * att;
    if (vla::flash_attn_enabled())
        att = flash_attention(C, Q, K, to_heads  (C, v, hd, heads, seq), nullptr, scale);
    else
        att = attention      (C, Q, K, to_heads_v(C, v, hd, heads, seq), nullptr, scale, hidden, seq);
    ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, h1, ln_eps), w.ln2w), w.ln2b);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wfc2, vla::gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wfc1, n2), w.bfc1))), w.bfc2);
    return ggml_add(C, h1, ff);
}

// pre_merge normalizes before the reshape, the deepstack taps after.
inline ggml_tensor * build_merger(ggml_context * C, const MergerW & w, ggml_tensor * x,
                                  int64_t hidden, int64_t merge2, float ln_eps, bool pre_merge) {
    const int64_t n_patches = x->ne[1], c_merged = hidden * merge2*merge2, n_merged = n_patches/(merge2*merge2);
    ggml_tensor * m;
    if (pre_merge) {
        ggml_tensor * xn = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.nw), w.nb);
        m = ggml_reshape_2d(C, ggml_cont(C, xn), c_merged, n_merged);
    } else {
        ggml_tensor * mr = ggml_reshape_2d(C, ggml_cont(C, x), c_merged, n_merged);
        m = ggml_add(C, ggml_mul(C, ggml_norm(C, mr, ln_eps), w.nw), w.nb);
    }
    ggml_tensor * z1 = ggml_add(C, ggml_mul_mat(C, w.fc1w, m), w.fc1b);
    return ggml_add(C, ggml_mul_mat(C, w.fc2w, ggml_gelu_erf(C, z1)), w.fc2b);
}

// Patch row/col after the spatial merge.
inline void merge_block_coords(int64_t gh, int64_t gw, int64_t m, std::vector<int64_t> & row, std::vector<int64_t> & col) {
    const int64_t S = gh * gw; row.assign(S, 0); col.assign(S, 0);
    for (int64_t s=0; s<S; ++s) {
        int64_t t = s; const int64_t wj = t%m; t /= m; const int64_t wi = t%m; t /= m;
        const int64_t bc = t%(gw/m); t /= (gw/m); const int64_t br = t;
        row[s] = br * m+wi; col[s] = bc * m+wj;
    }
}

inline void vit_rope_tables(const std::vector<int64_t> & row, const std::vector<int64_t> & col, int64_t hd, double theta,
                            std::vector<float> & cos_t, std::vector<float> & sin_t) {
    const int64_t S = (int64_t) row.size(), nf = hd/4;
    std::vector<double> invf(nf);
    for (int64_t i=0; i<nf; ++i)
        invf[i] = 1.0/std::pow(theta, (double)(2*i)/(double)(hd/2));
    cos_t.assign((size_t) S * hd, 0.0f); sin_t.assign((size_t) S * hd, 0.0f);
    for (int64_t s=0; s<S; ++s) {
        std::vector<double> emb(hd);
        for (int64_t i=0; i<nf; ++i) {
            emb[i] = (double) row[s]*invf[i];
            emb[nf+i] = (double) col[s]*invf[i];
        }
        for (int64_t i=0; i<hd/2; ++i)
            emb[hd/2+i] = emb[i];
        for (int64_t i=0; i<hd; ++i) {
            cos_t[s * hd+i] = (float) std::cos(emb[i]);
            sin_t[s * hd+i] = (float) std::sin(emb[i]);
        }
    }
}

// Bilinear resample of the pretrained position table onto gh x gw.
inline bool interp_pos_embed(const std::vector<float> & table, int64_t num_side, int64_t hidden,
                             const std::vector<int64_t> & row, const std::vector<int64_t> & col, int64_t gh, int64_t gw,
                             std::vector<float> & out) {
    if (num_side <= 0 || (int64_t) table.size() != num_side*num_side*hidden)
        return false;
    const int64_t S = (int64_t) row.size();
    out.assign((size_t) S * hidden, 0.0f);
    auto src_coord = [&](int64_t k, int64_t g) -> double { return (g <= 1) ? 0.0 : (double) k * (double)(num_side-1)/(double)(g-1); };
    for (int64_t s=0; s<S; ++s) {
        // Clamped, not just h1/w1: a grid that the spatial merge does not divide
        // pushes row/col past gh-1 and would index off the end of the table.
        const double lim = (double) (num_side-1);
        const double hy = std::min(src_coord(row[s], gh), lim), wx = std::min(src_coord(col[s], gw), lim);
        const int64_t h0 = (int64_t) std::floor(hy), w0 = (int64_t) std::floor(wx);
        const int64_t h1 = std::min(h0+1, num_side-1), w1 = std::min(w0+1, num_side-1);
        const double dh = hy-h0, dw = wx-w0;
        const double c00 = (1-dh)*(1-dw), c01 = (1-dh)*dw, c10 = dh * (1-dw), c11 = dh * dw;
        const float * T00 = &table[(h0*num_side+w0)*hidden]; const float * T01 = &table[(h0*num_side+w1)*hidden];
        const float * T10 = &table[(h1*num_side+w0)*hidden]; const float * T11 = &table[(h1*num_side+w1)*hidden];
        for (int64_t c=0; c<hidden; ++c)
            out[s * hidden+c] = (float)(c00*T00[c]+c01*T01[c]+c10*T10[c]+c11*T11[c]);
    }
    return true;
}

// HWC to flat patches. No resize: the view must already be side x side.
inline bool preprocess_image_patches(const char * arch, const ImageView & v, int64_t side, int64_t ps, int64_t tps,
                                     const std::vector<int64_t> & row, const std::vector<int64_t> & col,
                                     std::vector<float> & out) {
    if (!view_ok(arch, v, side))
        return false;
    const int64_t S = (int64_t) row.size(), pf = 3*tps * ps * ps;
    out.assign((size_t) pf * S, 0.0f);
    auto px = [&](int64_t r, int64_t c, int64_t ch) -> float {
        if (v.format == PixelFormat::U8)
            return ((const uint8_t *) v.data)[(r * side+c)*3+ch]/255.0f;
        return ((const float *) v.data)[(r * side+c)*3+ch];
    };
    for (int64_t s=0; s<S; ++s)
        for (int64_t ch=0; ch<3; ++ch)
            for (int64_t ph=0; ph<ps; ++ph)
                for (int64_t pw=0; pw<ps; ++pw) {
                    const float val = (px(row[s]*ps+ph, col[s]*ps+pw, ch)-QWEN3VL_MEAN[ch])/QWEN3VL_STD[ch];
                    for (int64_t t=0; t<tps; ++t)
                        out[s * pf+ch * tps * ps * ps+t * ps * ps+ph * ps+pw] = val;
                }
    return true;
}

inline bool mrope_positions(const char * arch, const std::vector<int32_t> & ids, int32_t image_token, int64_t grid,
                            std::vector<int32_t> & pp) {
    const int64_t seq = (int64_t) ids.size(), g2 = grid*grid;
    pp.assign((size_t) 4*seq, 0);
    int64_t st = 0, st_idx = 0;
    while (st < seq) {
        int64_t img = st;
        while (img < seq && ids[img] != image_token)
            ++img;
        for (int64_t i=st; i<img; ++i)
            pp[i] = pp[seq+i] = pp[2*seq+i] = (int32_t) (st_idx+i-st);
        if (img == seq)
            break;
        int64_t end = img;
        while (end < seq && ids[end] == image_token)
            ++end;
        if ((end-img)%g2 != 0) {
            std::fprintf(stderr, "vla(%s): image run length %lld not a multiple of %lld (post-merge grid)\n",
                         arch, (long long) (end-img), (long long) g2);
            return false;
        }
        const int64_t off = st_idx+img-st;
        for (int64_t k=0; k<end-img; ++k) {
            pp[img+k]       = (int32_t) (off+k/g2);
            pp[seq+img+k]   = (int32_t) (off+k%g2/grid);
            pp[2*seq+img+k] = (int32_t) (off+k%grid);
        }
        st_idx = off+std::max((end-img)/g2, grid);
        st     = end;
    }
    std::copy(pp.begin(), pp.begin()+seq, pp.begin()+3*seq);
    return true;
}

inline bool Qwen3VLTower::load_config(const char * arch, const gguf_reader & g, const char * ns) {
    auto U = [&](const char * k, int64_t & dst) {
        const std::string key = std::string(ns)+"."+k;
        if (g.has(key.c_str()))
            dst = (int64_t) g.u32(key.c_str());
    };
    auto F = [&](const char * k, float & dst) {
        const std::string key = std::string(ns)+"."+k;
        if (g.has(key.c_str()))
            dst = g.f32(key.c_str());
    };
    U("vit_hidden", hidden); U("vit_layers", layers); U("vit_heads", heads);
    U("patch_size", patch); U("temporal_patch_size", temporal); U("spatial_merge_size", merge);
    U("vit_num_position_embeddings", num_pos); U("vit_patch_flat", patch_flat); U("image_target_size", side);
    U("deepstack_idx_0", deepstack_idx[0]); U("deepstack_idx_1", deepstack_idx[1]); U("deepstack_idx_2", deepstack_idx[2]);
    F("vit_ln_eps", ln_eps); F("vit_rope_theta", rope_theta); F("connector_ln_eps", conn_eps);

    // merge_block_coords only enumerates the patch grid exactly when the spatial
    // merge divides it; otherwise it emits rows past the position table.
    if (patch <= 0 || merge <= 0 || side%patch != 0 || (side/patch)%merge != 0) {
        std::fprintf(stderr, "vla(%s): image %lld / patch %lld / merge %lld do not divide evenly\n",
                     arch, (long long) side, (long long) patch, (long long) merge);
        return false;
    }
    if (heads <= 0 || patch_flat != 3*temporal*patch*patch) {
        std::fprintf(stderr, "vla(%s): vit_heads %lld or vit_patch_flat %lld is inconsistent\n",
                     arch, (long long) heads, (long long) patch_flat);
        return false;
    }
    return true;
}

inline bool Qwen3VLTower::build_caches(const char * arch, gguf_reader & io) {
    merge_block_coords(grid(), grid(), merge, row, col);
    vit_rope_tables(row, col, hidden/heads, (double) rope_theta, rope_cos, rope_sin);

    const std::vector<float> table = io.read_f32("vit.pos_embd");
    if (table.empty() || (int64_t) table.size() != num_pos*hidden) {
        std::fprintf(stderr, "vla(%s): build_caches: vit.pos_embd unreadable\n", arch);
        return false;
    }
    const int64_t num_side = (int64_t) std::lround(std::sqrt((double) num_pos));
    if (!interp_pos_embed(table, num_side, hidden, row, col, grid(), grid(), pos_interp)) {
        std::fprintf(stderr, "vla(%s): build_caches: vit_num_position_embeddings %lld is not a square\n",
                     arch, (long long) num_pos);
        return false;
    }
    return true;
}

inline bool Qwen3VLTower::encode(const char * arch, ggml_backend_t backend, graph_cache<int, VitIO> & cache, const ImageView * images,
                                 int64_t n_views, const float * patches_in, std::vector<float> & emb,
                                 std::vector<float> (&ds)[3]) const {
    const int64_t n_patches = grid()*grid(), hd = hidden/heads;

    const bool built = cache.ensure(backend, 0, (size_t) 512*1024*1024, [&](ggml_context * VC, VitIO & io) -> ggml_cgraph * {
        ggml_tensor * t_patches = ggml_new_tensor_2d(VC, GGML_TYPE_F32, patch_flat, n_patches); ggml_set_input(t_patches);
        ggml_tensor * t_pos     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, hidden, n_patches);     ggml_set_input(t_pos);
        ggml_tensor * t_cos     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, hd, n_patches);         ggml_set_input(t_cos);
        ggml_tensor * t_sin     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, hd, n_patches);         ggml_set_input(t_sin);
        ggml_tensor * h = ggml_add(VC, ggml_add(VC, ggml_mul_mat(VC, patch_w, t_patches), patch_b), t_pos);

        ggml_tensor * stash[3] = {nullptr, nullptr, nullptr};
        for (int64_t i=0; i<layers; ++i) {
            h = build_vit_layer(VC, blk[i], h, t_cos, t_sin, n_patches, heads, hd, hidden, ln_eps);
            for (int j=0; j<3; ++j)
                if (i == deepstack_idx[j])
                    stash[j] = h;
        }
        for (int j=0; j<3; ++j) {
            io.ds[j] = build_merger(VC, deepstack[j], stash[j] ? stash[j] : h, hidden, merge, conn_eps, false);
            ggml_set_output(io.ds[j]);
        }
        io.embeds = build_merger(VC, merger, h, hidden, merge, conn_eps, true);
        ggml_set_output(io.embeds);
        io.t_patches = t_patches; io.t_pos = t_pos; io.t_cos = t_cos; io.t_sin = t_sin;

        ggml_cgraph * vg = ggml_new_graph_custom(VC, 16384, false);
        ggml_build_forward_expand(vg, io.embeds);
        for (int j=0; j<3; ++j)
            ggml_build_forward_expand(vg, io.ds[j]);
        return vg;
    });
    if (!built) {
        std::fprintf(stderr, "vla(%s): vision graph build failed\n", arch);
        return false;
    }
    const VitIO & io = cache.io();
    ggml_cgraph * vg = cache.graph();
    ggml_tensor * t_patches = io.t_patches, * t_pos = io.t_pos, * t_cos = io.t_cos, * t_sin = io.t_sin;
    ggml_tensor * embeds = io.embeds;
    ggml_tensor * const * ds_out = io.ds;

    const size_t per_view = (size_t) ggml_nelements(embeds);
    emb.assign((size_t) n_views*per_view, 0.0f);
    for (int j=0; j<3; ++j)
        ds[j].assign((size_t) n_views*per_view, 0.0f);

    std::vector<float> patches;
    for (int64_t v=0; v<n_views; ++v) {
        const float * px = patches_in ? patches_in+(size_t) v*n_patches*patch_flat : nullptr;
        if (!px) {
            if (!preprocess_image_patches(arch, images[v], side, patch, temporal, row, col, patches))
                return false;
            px = patches.data();
        }
        ggml_backend_tensor_set(t_patches, px, 0, ggml_nbytes(t_patches));
        ggml_backend_tensor_set(t_pos, pos_interp.data(), 0, ggml_nbytes(t_pos));
        ggml_backend_tensor_set(t_cos, rope_cos.data(), 0, ggml_nbytes(t_cos));
        ggml_backend_tensor_set(t_sin, rope_sin.data(), 0, ggml_nbytes(t_sin));
        graph_unique_names(vg);
        if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(%s): vision compute failed\n", arch);
            return false;
        }
        ggml_backend_tensor_get(embeds, emb.data()+v*per_view, 0, ggml_nbytes(embeds));
        for (int j=0; j<3; ++j)
            ggml_backend_tensor_get(ds_out[j], ds[j].data()+v*per_view, 0, ggml_nbytes(ds_out[j]));
    }
    return true;
}

}  // namespace vla
