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
#include "layers/fq_linear.h"
#include "layers/linear.h"
#include "layers/norm.h"

#include <cmath>
#include <cstdio>

namespace vla {

void DitHead::declare(WeightLoader & L, const char * prefix, bool fuse_qkv, bool interleave, const char * outer,
                      const FqModuleSpec * fq) {
    if (!outer)
        outer = prefix;

    te_l1W = L.gemm("%s.time_emb.l1.weight", outer);
    te_l1b = L.f32 ("%s.time_emb.l1.bias",   outer);
    te_l2W = L.gemm("%s.time_emb.l2.weight", outer);
    te_l2b = L.f32 ("%s.time_emb.l2.bias",   outer);

    blk.resize(cfg.layers);
    for (int64_t i=0; i<cfg.layers; ++i) {
        DitLayerW & w = blk[i];
        const long long ii = (long long) i;
        // adaLN stays a float GEMM in every FoldQuant scheme (M = 1, weight-only
        // territory); the projections and the FFN are the quantized sites.
        w.adaln_w = L.gemm("%s.%lld.adaln.weight",  prefix, ii);
        w.adaln_b = L.f32 ("%s.%lld.adaln.bias",    prefix, ii);
        if (fq) {
            w.fq_o   = fq_declare_linear(L, *fq, "o",    true, nullptr, 0.0f, "%s.%lld.attn_o", prefix, ii);
            w.fq_ff0 = fq_declare_linear(L, *fq, "ffn0", true, nullptr, 0.0f, "%s.%lld.ff0",    prefix, ii);
            w.fq_ff2 = fq_declare_linear(L, *fq, "ffn2", true, nullptr, 0.0f, "%s.%lld.ff2",    prefix, ii);
        }
        if (!w.fq_o) {
            w.Wo   = L.gemm("%s.%lld.attn_o.weight", prefix, ii);
            w.bo   = L.f32 ("%s.%lld.attn_o.bias",   prefix, ii);
        }
        if (!w.fq_ff0) {
            w.Wff0 = L.gemm("%s.%lld.ff0.weight",    prefix, ii);
            w.bff0 = L.f32 ("%s.%lld.ff0.bias",      prefix, ii);
        }
        if (!w.fq_ff2) {
            w.Wff2 = L.gemm("%s.%lld.ff2.weight",    prefix, ii);
            w.bff2 = L.f32 ("%s.%lld.ff2.bias",      prefix, ii);
        }

        if (!fuse_qkv) {
            if (fq) {
                w.fq_q = fq_declare_linear(L, *fq, "qkv", true, nullptr, 0.0f, "%s.%lld.attn_q", prefix, ii);
                w.fq_k = fq_declare_linear(L, *fq, "qkv", true, nullptr, 0.0f, "%s.%lld.attn_k", prefix, ii);
                w.fq_v = fq_declare_linear(L, *fq, "qkv", true, nullptr, 0.0f, "%s.%lld.attn_v", prefix, ii);
                if ((bool) w.fq_k != (bool) w.fq_v)
                    L.fail("FoldQuant: DiT k/v must both be INT8 or both float in a layer");
            }
            if (!w.fq_q) {
                w.Wq = L.gemm("%s.%lld.attn_q.weight", prefix, ii);
                w.bq = L.f32 ("%s.%lld.attn_q.bias",   prefix, ii);
            }
            if (!w.fq_k) {
                w.Wk = L.gemm("%s.%lld.attn_k.weight", prefix, ii);
                w.bk = L.f32 ("%s.%lld.attn_k.bias",   prefix, ii);
            }
            if (!w.fq_v) {
                w.Wv = L.gemm("%s.%lld.attn_v.weight", prefix, ii);
                w.bv = L.f32 ("%s.%lld.attn_v.bias",   prefix, ii);
            }
            continue;
        }

        char q[192], k[192], v[192], qb[192], kb[192], vb[192], out[192];
        std::snprintf(q,  sizeof(q),  "%s.%lld.attn_q.weight", prefix, (long long)i);
        std::snprintf(k,  sizeof(k),  "%s.%lld.attn_k.weight", prefix, (long long)i);
        std::snprintf(v,  sizeof(v),  "%s.%lld.attn_v.weight", prefix, (long long)i);
        std::snprintf(qb, sizeof(qb), "%s.%lld.attn_q.bias",   prefix, (long long)i);
        std::snprintf(kb, sizeof(kb), "%s.%lld.attn_k.bias",   prefix, (long long)i);
        std::snprintf(vb, sizeof(vb), "%s.%lld.attn_v.bias",   prefix, (long long)i);

        char sq[192], sk[192], sv[192];
        std::snprintf(sq, sizeof(sq), "%s.%lld.attn_q", prefix, ii);
        std::snprintf(sk, sizeof(sk), "%s.%lld.attn_k", prefix, ii);
        std::snprintf(sv, sizeof(sv), "%s.%lld.attn_v", prefix, ii);

        if (interleave && (i%2 == 1)) {
            std::snprintf(out, sizeof(out), "%s.%lld.attn_qkv.fused", prefix, ii);
            if (fq)
                w.fq_qkv = fq_declare_fused(L, *fq, "qkv", true, out, {sq, sk, sv});
            if (!w.fq_qkv) {
                std::snprintf(out, sizeof(out), "%s.%lld.attn_qkv.fused.w", prefix, ii);
                w.Wqkv = L.fuse_gemm(out, {q, k, v});
                std::snprintf(out, sizeof(out), "%s.%lld.attn_qkv.fused.b", prefix, ii);
                w.bqkv = L.fuse_f32(out, {qb, kb, vb});
            }
        } else {
            if (fq)
                w.fq_q = fq_declare_linear(L, *fq, "qkv", true, nullptr, 0.0f, "%s.%lld.attn_q", prefix, ii);
            if (!w.fq_q) {
                w.Wq = L.gemm("%s.%lld.attn_q.weight", prefix, ii);
                w.bq = L.f32 ("%s.%lld.attn_q.bias",   prefix, ii);
            }
            std::snprintf(out, sizeof(out), "%s.%lld.attn_kv.fused", prefix, ii);
            if (fq)
                w.fq_kv = fq_declare_fused(L, *fq, "enc", true, out, {sk, sv});
            if (!w.fq_kv) {
                std::snprintf(out, sizeof(out), "%s.%lld.attn_kv.fused.w", prefix, ii);
                w.Wkv = L.fuse_gemm(out, {k, v});
                std::snprintf(out, sizeof(out), "%s.%lld.attn_kv.fused.b", prefix, ii);
                w.bkv = L.fuse_f32(out, {kb, vb});
            }
        }
    }

    po1W = L.gemm("%s.proj_out1.weight", outer);
    po1b = L.f32 ("%s.proj_out1.bias",   outer);
    po2W = L.gemm("%s.proj_out2.weight", outer);
    po2b = L.f32 ("%s.proj_out2.bias",   outer);
}

void DitHead::kv(ggml_context * C, const DitLayerW & w, ggml_tensor * src,
                 ggml_tensor ** K_out, ggml_tensor ** V_out, ggml_tensor * xq_pre) const {
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const int64_t Tkv   = src->ne[1];

    if (w.Wkv || w.fq_kv) {
        ggml_tensor * kvp = w.fq_kv ? fq_linear(C, w.fq_kv, src) : linear(C, w.Wkv, w.bkv, src);
        *K_out = ggml_cont(C, ggml_permute(C, head_view(C, kvp, hd, heads, Tkv, cfg.hidden, 2, 0), 0, 2, 1, 3));
        *V_out = ggml_cont(C, ggml_permute(C, head_view(C, kvp, hd, heads, Tkv, cfg.hidden, 2, 1), 1, 2, 0, 3));
        return;
    }
    if (w.fq_k) {
        ggml_tensor * xq = xq_pre ? xq_pre : fq_act(C, w.fq_k, src);
        *K_out = to_heads  (C, fq_gemm(C, w.fq_k, xq), hd, heads, Tkv);
        *V_out = to_heads_v(C, fq_gemm(C, w.fq_v, xq), hd, heads, Tkv);
        return;
    }
    *K_out = to_heads  (C, linear(C, w.Wk, w.bk, src), hd, heads, Tkv);
    *V_out = to_heads_v(C, linear(C, w.Wv, w.bv, src), hd, heads, Tkv);
}

// adaLN from a precomputed condition: the tail of layers/norm.h adaln, same ops.
static ggml_tensor * adaln_from_cond(ggml_context * C, ggml_tensor * x, ggml_tensor * cond, int64_t dim, float eps) {
    ggml_tensor * sc = ggml_view_1d(C, cond, dim, 0);
    ggml_tensor * sh = ggml_view_1d(C, cond, dim, (size_t) dim * sizeof(float));
    ggml_tensor * xn = ggml_norm(C, x, eps);
    return ggml_add(C, ggml_add(C, xn, ggml_mul(C, xn, sc)), sh);
}

ggml_tensor * DitHead::adaln_cond(ggml_context * C, const DitLayerW & w, ggml_tensor * temb) const {
    return linear(C, w.adaln_w, w.adaln_b, ggml_silu(C, temb));
}

ggml_tensor * DitHead::proj_out_cond(ggml_context * C, ggml_tensor * temb) const {
    return linear(C, po1W, po1b, ggml_silu(C, temb));
}

ggml_tensor * DitHead::block(ggml_context * C, const DitLayerW & w, ggml_tensor * h, ggml_tensor * temb,
                             ggml_tensor * enc, ggml_tensor * K_pre, ggml_tensor * V_pre, ggml_tensor * cond) const {
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const int64_t dim   = cfg.hidden;
    const int64_t Tk    = h->ne[1];
    const float   scale = 1.0f/std::sqrt((float)hd);

    ggml_tensor * n = cond ? adaln_from_cond(C, h, cond, dim, cfg.ln_eps)
                           : adaln(C, h, temb, w.adaln_w, w.adaln_b, dim, cfg.ln_eps);
    ggml_tensor *Q, *K, *V;
    if (!enc && (w.Wqkv || w.fq_qkv)) {
        ggml_tensor * qkv = w.fq_qkv ? fq_linear(C, w.fq_qkv, n) : linear(C, w.Wqkv, w.bqkv, n);
        Q = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 0), 0, 2, 1, 3));
        K = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 1), 0, 2, 1, 3));
        V = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 2), 1, 2, 0, 3));
    } else {
        // Self-attention with separate FoldQuant q/k/v: one activation blob of n
        // serves all three projections.
        ggml_tensor * xq = (w.fq_q && !enc && !K_pre && w.fq_k) ? fq_act(C, w.fq_q, n) : nullptr;
        Q = to_heads(C, w.fq_q ? (xq ? fq_gemm(C, w.fq_q, xq) : fq_linear(C, w.fq_q, n))
                                : linear(C, w.Wq, w.bq, n), hd, heads, Tk);
        if (K_pre) {
            K = K_pre;
            V = V_pre;
        }
        else       {
            kv(C, w, enc ? enc : n, &K, &V, xq);
        }
    }

    ggml_tensor * att = attention(C, Q, K, V, nullptr, scale, dim, Tk);
    ggml_tensor * o   = w.fq_o ? fq_linear(C, w.fq_o, att) : linear(C, w.Wo, w.bo, att);
    ggml_tensor * h1  = ggml_add(C, h, o);
    ggml_tensor * n3  = ggml_norm(C, h1, cfg.ln_eps);
    ggml_tensor * ff  = w.fq_ff0 ? fq_linear(C, w.fq_ff2, ggml_gelu(C, fq_linear(C, w.fq_ff0, n3)))
                                 : ffn_gelu(C, w.Wff0, w.bff0, w.Wff2, w.bff2, n3);
    return ggml_add(C, h1, ff);
}

ggml_tensor * DitHead::time_emb(ggml_context * C, ggml_tensor * tproj) const {
    return linear(C, te_l2W, te_l2b, ggml_silu(C, linear(C, te_l1W, te_l1b, tproj)));
}

ggml_tensor * DitHead::proj_out(ggml_context * C, ggml_tensor * h, ggml_tensor * temb, ggml_tensor * po) const {
    if (!po) po = proj_out_cond(C, temb);
    ggml_tensor * sh = ggml_view_1d(C, po, cfg.hidden, 0);
    ggml_tensor * sc = ggml_view_1d(C, po, cfg.hidden, (size_t)cfg.hidden*sizeof(float));

    ggml_tensor * hn    = ggml_norm(C, h, cfg.norm_out_eps);
    ggml_tensor * h_mod = ggml_add(C, ggml_add(C, hn, ggml_mul(C, hn, sc)), sh);
    return linear(C, po2W, po2b, h_mod);
}

}
