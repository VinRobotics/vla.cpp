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

#include "modules/dit_head.h"

#include "layers/attn.h"
#include "layers/ffn.h"
#include "layers/linear.h"
#include "layers/norm.h"

#include <cmath>
#include <cstdio>

namespace vla {

void DitHead::declare(WeightLoader & L, const char * prefix, bool fuse_qkv, bool interleave) {
    te_l1W = L.gemm("%s.time_emb.l1.weight", prefix);
    te_l1b = L.f32 ("%s.time_emb.l1.bias",   prefix);
    te_l2W = L.gemm("%s.time_emb.l2.weight", prefix);
    te_l2b = L.f32 ("%s.time_emb.l2.bias",   prefix);

    blk.resize(cfg.layers);
    for (int64_t i=0; i<cfg.layers; ++i) {
        DitLayerW & w = blk[i];
        w.adaln_w = L.gemm("%s.%lld.adaln.weight",  prefix, (long long)i);
        w.adaln_b = L.f32 ("%s.%lld.adaln.bias",    prefix, (long long)i);
        w.Wo      = L.gemm("%s.%lld.attn_o.weight", prefix, (long long)i);
        w.bo      = L.f32 ("%s.%lld.attn_o.bias",   prefix, (long long)i);
        w.Wff0    = L.gemm("%s.%lld.ff0.weight",    prefix, (long long)i);
        w.bff0    = L.f32 ("%s.%lld.ff0.bias",      prefix, (long long)i);
        w.Wff2    = L.gemm("%s.%lld.ff2.weight",    prefix, (long long)i);
        w.bff2    = L.f32 ("%s.%lld.ff2.bias",      prefix, (long long)i);

        if (!fuse_qkv) {
            w.Wq = L.gemm("%s.%lld.attn_q.weight", prefix, (long long)i);
            w.bq = L.f32 ("%s.%lld.attn_q.bias",   prefix, (long long)i);
            w.Wk = L.gemm("%s.%lld.attn_k.weight", prefix, (long long)i);
            w.bk = L.f32 ("%s.%lld.attn_k.bias",   prefix, (long long)i);
            w.Wv = L.gemm("%s.%lld.attn_v.weight", prefix, (long long)i);
            w.bv = L.f32 ("%s.%lld.attn_v.bias",   prefix, (long long)i);
            continue;
        }

        char q[192], k[192], v[192], qb[192], kb[192], vb[192], out[192];
        std::snprintf(q,  sizeof(q),  "%s.%lld.attn_q.weight", prefix, (long long)i);
        std::snprintf(k,  sizeof(k),  "%s.%lld.attn_k.weight", prefix, (long long)i);
        std::snprintf(v,  sizeof(v),  "%s.%lld.attn_v.weight", prefix, (long long)i);
        std::snprintf(qb, sizeof(qb), "%s.%lld.attn_q.bias",   prefix, (long long)i);
        std::snprintf(kb, sizeof(kb), "%s.%lld.attn_k.bias",   prefix, (long long)i);
        std::snprintf(vb, sizeof(vb), "%s.%lld.attn_v.bias",   prefix, (long long)i);

        if (interleave && (i%2 == 1)) {
            std::snprintf(out, sizeof(out), "%s.%lld.attn_qkv.fused.w", prefix, (long long)i);
            w.Wqkv = L.fuse_gemm(out, {q, k, v});
            std::snprintf(out, sizeof(out), "%s.%lld.attn_qkv.fused.b", prefix, (long long)i);
            w.bqkv = L.fuse_f32(out, {qb, kb, vb});
        } else {
            w.Wq = L.gemm("%s.%lld.attn_q.weight", prefix, (long long)i);
            w.bq = L.f32 ("%s.%lld.attn_q.bias",   prefix, (long long)i);
            std::snprintf(out, sizeof(out), "%s.%lld.attn_kv.fused.w", prefix, (long long)i);
            w.Wkv = L.fuse_gemm(out, {k, v});
            std::snprintf(out, sizeof(out), "%s.%lld.attn_kv.fused.b", prefix, (long long)i);
            w.bkv = L.fuse_f32(out, {kb, vb});
        }
    }

    po1W = L.gemm("%s.proj_out1.weight", prefix);
    po1b = L.f32 ("%s.proj_out1.bias",   prefix);
    po2W = L.gemm("%s.proj_out2.weight", prefix);
    po2b = L.f32 ("%s.proj_out2.bias",   prefix);
}

void DitHead::kv(ggml_context * C, const DitLayerW & w, ggml_tensor * src,
                 ggml_tensor ** K_out, ggml_tensor ** V_out) const {
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const int64_t Tkv   = src->ne[1];

    if (w.Wkv) {
        ggml_tensor * kvp = linear(C, w.Wkv, w.bkv, src);
        *K_out = ggml_cont(C, ggml_permute(C, head_view(C, kvp, hd, heads, Tkv, cfg.hidden, 2, 0), 0, 2, 1, 3));
        *V_out = ggml_cont(C, ggml_permute(C, head_view(C, kvp, hd, heads, Tkv, cfg.hidden, 2, 1), 1, 2, 0, 3));
        return;
    }
    *K_out = to_heads  (C, linear(C, w.Wk, w.bk, src), hd, heads, Tkv);
    *V_out = to_heads_v(C, linear(C, w.Wv, w.bv, src), hd, heads, Tkv);
}

ggml_tensor * DitHead::block(ggml_context * C, const DitLayerW & w, ggml_tensor * h, ggml_tensor * temb,
                             ggml_tensor * enc, ggml_tensor * K_pre, ggml_tensor * V_pre) const {
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const int64_t dim   = cfg.hidden;
    const int64_t Tk    = h->ne[1];
    const float   scale = 1.0f/std::sqrt((float)hd);

    ggml_tensor * n = adaln(C, h, temb, w.adaln_w, w.adaln_b, dim, cfg.ln_eps);
    ggml_tensor *Q, *K, *V;
    if (!enc && w.Wqkv) {
        ggml_tensor * qkv = linear(C, w.Wqkv, w.bqkv, n);
        Q = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 0), 0, 2, 1, 3));
        K = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 1), 0, 2, 1, 3));
        V = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 2), 1, 2, 0, 3));
    } else {
        Q = to_heads(C, linear(C, w.Wq, w.bq, n), hd, heads, Tk);
        if (K_pre) { K = K_pre; V = V_pre; }
        else       { kv(C, w, enc ? enc : n, &K, &V); }
    }

    ggml_tensor * att = attention(C, Q, K, V, nullptr, scale, dim, Tk);
    ggml_tensor * h1  = ggml_add(C, h, linear(C, w.Wo, w.bo, att));
    ggml_tensor * n3  = ggml_norm(C, h1, cfg.ln_eps);
    return ggml_add(C, h1, ffn_gelu(C, w.Wff0, w.bff0, w.Wff2, w.bff2, n3));
}

ggml_tensor * DitHead::time_emb(ggml_context * C, ggml_tensor * tproj) const {
    return linear(C, te_l2W, te_l2b, ggml_silu(C, linear(C, te_l1W, te_l1b, tproj)));
}

ggml_tensor * DitHead::proj_out(ggml_context * C, ggml_tensor * h, ggml_tensor * temb) const {
    ggml_tensor * po = linear(C, po1W, po1b, ggml_silu(C, temb));
    ggml_tensor * sh = ggml_view_1d(C, po, cfg.hidden, 0);
    ggml_tensor * sc = ggml_view_1d(C, po, cfg.hidden, (size_t)cfg.hidden*sizeof(float));

    ggml_tensor * hn    = ggml_norm(C, h, cfg.norm_out_eps);
    ggml_tensor * h_mod = ggml_add(C, ggml_add(C, hn, ggml_mul(C, hn, sc)), sh);
    return linear(C, po2W, po2b, h_mod);
}

}
