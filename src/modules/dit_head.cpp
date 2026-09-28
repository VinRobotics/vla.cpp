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

#include "backend.h"
#include "layers/attn.h"
#include "layers/embed.h"
#include "layers/ffn.h"
#include "layers/linear.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>

namespace vla {

void DitHead::declare(WeightLoader & L, const char * prefix, bool fuse_qkv, bool interleave, const char * outer) {
    if (!outer)
        outer = prefix;

    te_l1W = L.gemm("%s.time_emb.l1.weight", outer);
    te_l1b = L.f32 ("%s.time_emb.l1.bias",   outer);
    te_l2W = L.gemm("%s.time_emb.l2.weight", outer);
    te_l2b = L.f32 ("%s.time_emb.l2.bias",   outer);

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

    po1W = L.gemm("%s.proj_out1.weight", outer);
    po1b = L.f32 ("%s.proj_out1.bias",   outer);
    po2W = L.gemm("%s.proj_out2.weight", outer);
    po2b = L.f32 ("%s.proj_out2.bias",   outer);
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

ggml_tensor * DitHead::block(ggml_context * C, const DitLayerW & w, ggml_tensor * h, ggml_tensor * mod,
                             ggml_tensor * enc, ggml_tensor * K_pre, ggml_tensor * V_pre) const {
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const int64_t dim   = cfg.hidden;
    const int64_t Tk    = h->ne[1];
    const float   scale = 1.0f/std::sqrt((float)hd);

    ggml_tensor * sc = ggml_view_1d(C, mod, dim, 0);
    ggml_tensor * sh = ggml_view_1d(C, mod, dim, (size_t)dim*sizeof(float));
    ggml_tensor * xn = ggml_norm(C, h, cfg.ln_eps);
    ggml_tensor * n  = ggml_add(C, ggml_add(C, xn, ggml_mul(C, xn, sc)), sh);
    ggml_tensor *Q, *K, *V;
    if (!enc && w.Wqkv) {
        ggml_tensor * qkv = linear(C, w.Wqkv, w.bqkv, n);
        Q = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 0), 0, 2, 1, 3));
        K = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 1), 0, 2, 1, 3));
        V = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 2), 1, 2, 0, 3));
    } else {
        Q = to_heads(C, linear(C, w.Wq, w.bq, n), hd, heads, Tk);
        if (K_pre) {
            K = K_pre;
            V = V_pre;
        }
        else       {
            kv(C, w, enc ? enc : n, &K, &V);
        }
    }

    ggml_tensor * att = attention(C, Q, K, V, nullptr, scale, dim, Tk);
    ggml_tensor * h1  = ggml_add(C, h, linear(C, w.Wo, w.bo, att));
    ggml_tensor * n3  = ggml_norm(C, h1, cfg.ln_eps);
    return ggml_add(C, h1, ffn_gelu(C, w.Wff0, w.bff0, w.Wff2, w.bff2, n3));
}

ggml_tensor * DitHead::time_emb(ggml_context * C, ggml_tensor * tproj) const {
    return linear(C, te_l2W, te_l2b, ggml_silu(C, linear(C, te_l1W, te_l1b, tproj)));
}

ggml_tensor * DitHead::proj_out(ggml_context * C, ggml_tensor * h, ggml_tensor * mod) const {
    ggml_tensor * sh = ggml_view_1d(C, mod, cfg.hidden, 0);
    ggml_tensor * sc = ggml_view_1d(C, mod, cfg.hidden, (size_t)cfg.hidden*sizeof(float));

    ggml_tensor * hn    = ggml_norm(C, h, cfg.norm_out_eps);
    ggml_tensor * h_mod = ggml_add(C, ggml_add(C, hn, ggml_mul(C, hn, sc)), sh);
    return linear(C, po2W, po2b, h_mod);
}

FlowTimes::~FlowTimes() {
    if (buf)
        ggml_backend_buffer_free(buf);
    if (ctx)
        ggml_free(ctx);
}

bool FlowTimes::build(const char * arch, ggml_backend_t backend, const DitHead & dit,
                      int64_t steps, int64_t buckets, int64_t embed_dim, int64_t horizon) {
    const int64_t dim = dit.cfg.hidden, layers = dit.cfg.layers;
    for (const DitLayerW & w : dit.blk)
        if (w.adaln_w->ne[1] != 2*dim) {
            std::fprintf(stderr, "vla(%s): adaln weight has %lld rows, expected %lld\n",
                         arch, (long long) w.adaln_w->ne[1], (long long) (2*dim));
            return false;
        }
    if (dit.po1W->ne[1] != 2*dim) {
        std::fprintf(stderr, "vla(%s): proj_out1 weight has %lld rows, expected %lld\n",
                     arch, (long long) dit.po1W->ne[1], (long long) (2*dim));
        return false;
    }

    per_step = layers+1;
    ggml_init_params rp = { (size_t) (steps+1)*ggml_tensor_overhead(), nullptr, true };
    ctx = ggml_init(rp);
    if (!ctx)
        return false;
    tau.assign((size_t) steps, nullptr);
    for (int64_t s=0; s<steps; ++s) {
        tau[(size_t) s] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, horizon);
        ggml_format_name(tau[(size_t) s], "flow_tau_%lld", (long long) s);
    }
    mods = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*dim, per_step*steps);
    ggml_set_name(mods, "flow_mods");
    buf  = alloc_weights(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "vla(%s): flow time buffer alloc failed\n", arch);
        return false;
    }

    const size_t nodes = (size_t) steps*(per_step*3+8);
    ggml_init_params gp = { nodes*ggml_tensor_overhead()+ggml_graph_overhead_custom(nodes, false), nullptr, true };
    ggml_context * C = ggml_init(gp);
    if (!C)
        return false;
    ggml_cgraph * gf = ggml_new_graph_custom(C, nodes, false);
    std::vector<ggml_tensor *> t_tproj((size_t) steps), outs;
    for (int64_t s=0; s<steps; ++s) {
        t_tproj[(size_t) s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, 256);
        ggml_set_input(t_tproj[(size_t) s]);
        ggml_tensor * temb = dit.time_emb(C, t_tproj[(size_t) s]);
        for (int64_t i=0; i<=layers; ++i) {
            ggml_tensor * W = i < layers ? dit.blk[(size_t) i].adaln_w : dit.po1W;
            ggml_tensor * b = i < layers ? dit.blk[(size_t) i].adaln_b : dit.po1b;
            ggml_tensor * o = linear(C, W, b, ggml_silu(C, temb));
            ggml_set_output(o);
            ggml_build_forward_expand(gf, o);
            outs.push_back(o);
        }
    }

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ga && ggml_gallocr_alloc_graph(ga, gf);
    if (ok) {
        std::vector<float> tau_h, tproj_h;
        for (int64_t s=0; s<steps; ++s) {
            const int64_t bucket = (int64_t) ((double) s/(double) steps*(double) buckets);
            action_sinusoid(bucket, embed_dim, horizon, tau_h);
            timesteps_proj(bucket, tproj_h);
            ggml_backend_tensor_set(tau[(size_t) s], tau_h.data(), 0, ggml_nbytes(tau[(size_t) s]));
            ggml_backend_tensor_set(t_tproj[(size_t) s], tproj_h.data(), 0, ggml_nbytes(t_tproj[(size_t) s]));
        }
        graph_unique_names(gf);
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        std::vector<float> host((size_t) ggml_nelements(mods));
        for (size_t k=0; k<outs.size(); ++k)
            ggml_backend_tensor_get(outs[k], host.data()+k*(size_t) (2*dim), 0, ggml_nbytes(outs[k]));
        ggml_backend_tensor_set(mods, host.data(), 0, ggml_nbytes(mods));
    }
    if (ga)
        ggml_gallocr_free(ga);
    ggml_free(C);
    if (!ok)
        std::fprintf(stderr, "vla(%s): flow time conditioning compute failed\n", arch);
    return ok;
}

ggml_tensor * FlowTimes::mod(ggml_context * C, int64_t s, int64_t i) const {
    return ggml_view_1d(C, mods, mods->ne[0], (size_t) (s*per_step+i)*mods->nb[1]);
}

}
