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
#include "layers/fq_linear.h"
#include "layers/fused_ops.h"
#include "layers/linear.h"
#include "act_dtype.h"
#include "env_flag.h"
#include "options.h"
#include "sycl/vla_sycl_ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
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
                fq_set_heads(w.fq_q, (int) cfg.head_dim, (int) cfg.heads, 0u);
                fq_set_heads(w.fq_k, (int) cfg.head_dim, (int) cfg.heads, 0u);
                fq_set_heads(w.fq_v, (int) cfg.head_dim, (int) cfg.heads, 1u);
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
            if (fq) {
                w.fq_qkv = fq_declare_fused(L, *fq, "qkv", true, out, {sq, sk, sv});
                fq_set_heads(w.fq_qkv, (int) cfg.head_dim, (int) cfg.heads, 4u);   // parts Q, K, V
            }
            if (!w.fq_qkv) {
                std::snprintf(out, sizeof(out), "%s.%lld.attn_qkv.fused.w", prefix, ii);
                w.Wqkv = L.fuse_gemm(out, {q, k, v});
                std::snprintf(out, sizeof(out), "%s.%lld.attn_qkv.fused.b", prefix, ii);
                w.bqkv = L.fuse_f32(out, {qb, kb, vb});
            }
        } else {
            if (fq) {
                w.fq_q = fq_declare_linear(L, *fq, "qkv", true, nullptr, 0.0f, "%s.%lld.attn_q", prefix, ii);
                fq_set_heads(w.fq_q, (int) cfg.head_dim, (int) cfg.heads, 0u);
            }
            if (!w.fq_q) {
                w.Wq = L.gemm("%s.%lld.attn_q.weight", prefix, ii);
                w.bq = L.f32 ("%s.%lld.attn_q.bias",   prefix, ii);
            }
            std::snprintf(out, sizeof(out), "%s.%lld.attn_kv.fused", prefix, ii);
            if (fq) {
                w.fq_kv = fq_declare_fused(L, *fq, "enc", true, out, {sk, sv});
                fq_set_heads(w.fq_kv, (int) cfg.head_dim, (int) cfg.heads, 2u);    // parts K, V
            }
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

namespace {

// K [hd, T, heads] and V^T [T, hd, heads] in F16, the head dim and the keys
// zero-padded to gemm_width(), as the HMX takes them (Hexagon). Elsewhere both
// come back unchanged.
void hmx_kv(ggml_context * C, ggml_tensor ** K, ggml_tensor ** V) {
    const int64_t hd = (*K)->ne[0], T = (*K)->ne[1], HP = gemm_width(hd), NK = gemm_width(T);
    if (HP > hd || NK > T) {
        *K = ggml_pad(C, *K, (int) (HP-hd), (int) (NK-T), 0, 0);
        *V = ggml_pad(C, *V, (int) (NK-T), (int) (HP-hd), 0, 0);
    }
    *K = fa_kv(C, *K);
    *V = fa_kv(C, *V);
}

}

void DitHead::kv(ggml_context * C, const DitLayerW & w, ggml_tensor * src,
                 ggml_tensor ** K_out, ggml_tensor ** V_out, ggml_tensor * xq_pre, bool cross) const {
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const int64_t Tkv   = src->ne[1];

    if (fused) {
        // [hd, Tkv, heads] views straight into the BF16 projections; the
        // attention reads them in place.
        ggml_tensor * s16 = as_type(C, src, GGML_TYPE_BF16);
        auto heads_of = [&](ggml_tensor * p, int64_t which) {
            const size_t es = ggml_element_size(p);
            return ggml_permute(C, ggml_view_3d(C, p, hd, heads, Tkv, hd * es, p->nb[1], (size_t) which * cfg.hidden * es),
                                0, 2, 1, 3);
        };
        if (w.Wkv) {
            ggml_tensor * kvp = fused_linear(C, w.Wkv, s16, w.bkv, nullptr, GGML_TYPE_BF16, 0);
            *K_out = heads_of(kvp, 0);
            *V_out = heads_of(kvp, 1);
        } else {
            *K_out = heads_of(fused_linear(C, w.Wk, s16, w.bk, nullptr, GGML_TYPE_BF16, 0), 0);
            *V_out = heads_of(fused_linear(C, w.Wv, s16, w.bv, nullptr, GGML_TYPE_BF16, 0), 0);
        }
        return;
    }

    if (w.Wkv || w.fq_kv) {
        ggml_tensor * kvp = w.fq_kv ? fq_linear(C, w.fq_kv, src) : linear(C, w.Wkv, w.bkv, src);
        if (w.fq_kv && w.fq_kv.gemm.heads) {   // written in head layout by the epilogue
            *K_out = fq_head_view(C, kvp, w.fq_kv, 0, Tkv);
            *V_out = fq_head_view(C, kvp, w.fq_kv, 1, Tkv);
            return;
        }
        *K_out = ggml_cont(C, ggml_permute(C, head_view(C, kvp, hd, heads, Tkv, cfg.hidden, 2, 0), 0, 2, 1, 3));
        *V_out = ggml_cont(C, ggml_permute(C, head_view(C, kvp, hd, heads, Tkv, cfg.hidden, 2, 1), 1, 2, 0, 3));
        if (!cfg.fused_ops && (!cross || cfg.cross_kv_f16))
            hmx_kv(C, K_out, V_out);
        return;
    }
    if (w.fq_k) {
        ggml_tensor * xq = xq_pre ? xq_pre : fq_act(C, w.fq_k, src);
        ggml_tensor * kp = fq_gemm(C, w.fq_k, xq), * vp = fq_gemm(C, w.fq_v, xq);
        *K_out = w.fq_k.gemm.heads ? fq_head_view(C, kp, w.fq_k, 0, Tkv) : to_heads  (C, kp, hd, heads, Tkv);
        *V_out = w.fq_v.gemm.heads ? fq_head_view(C, vp, w.fq_v, 0, Tkv) : to_heads_v(C, vp, hd, heads, Tkv);
        return;
    }
    *K_out = to_heads  (C, linear(C, w.Wk, w.bk, src), hd, heads, Tkv);
    *V_out = to_heads_v(C, linear(C, w.Wv, w.bv, src), hd, heads, Tkv);
    if (!cfg.fused_ops && (!cross || cfg.cross_kv_f16))
        hmx_kv(C, K_out, V_out);
}

ggml_tensor * DitHead::key_mask(ggml_context * C, int64_t NK, int64_t Nk, int64_t Tq) const {
    char name[GGML_MAX_NAME];
    std::snprintf(name, sizeof(name), "dit.key_mask.%lld.%lld.%lld", (long long) NK, (long long) Nk, (long long) Tq);
    if (ggml_tensor * m = ggml_get_tensor(C, name))
        return m;
    // 0 for the Nk real keys, -1e30 for the padding: step(i - Nk + 0.5) * -1e30.
    ggml_tensor * m = ggml_step(C, ggml_scale_bias(C, ggml_arange(C, 0.0f, (float) NK, 1.0f), 1.0f, 0.5f - (float) Nk));
    m = ggml_repeat_4d(C, ggml_scale(C, m, -1e30f), NK, Tq, 1, 1);
    ggml_set_name(m, name);
    return m;
}

ggml_tensor * DitHead::block(ggml_context * C, const DitLayerW & w, ggml_tensor * h, ggml_tensor * mod,
                             ggml_tensor * enc, ggml_tensor * K_pre, ggml_tensor * V_pre) const {
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const int64_t dim   = cfg.hidden;
    const int64_t Tk    = h->ne[1];
    const float   scale = 1.0f/std::sqrt((float)hd);

    if (fused) {
        // mod holds 1 + scale, then shift (FlowTimes::build with dit.fused).
        ggml_tensor * n = fused_lnorm(C, h, ggml_view_1d(C, mod, dim, 0),
                                      ggml_view_1d(C, mod, dim, (size_t) dim * sizeof(float)), GGML_TYPE_BF16, ln_spec);
        auto heads_of = [&](ggml_tensor * p, int64_t which) {
            const size_t es = ggml_element_size(p);
            return ggml_permute(C, ggml_view_3d(C, p, hd, heads, Tk, hd * es, p->nb[1], (size_t) which * dim * es),
                                0, 2, 1, 3);
        };
        ggml_tensor *Qf, *Kf, *Vf;
        if (!enc && w.Wqkv) {
            ggml_tensor * qkv = fused_linear(C, w.Wqkv, n, w.bqkv, nullptr, GGML_TYPE_BF16, 0);
            Qf = heads_of(qkv, 0);
            Kf = heads_of(qkv, 1);
            Vf = heads_of(qkv, 2);
        } else {
            Qf = heads_of(fused_linear(C, w.Wq, n, w.bq, nullptr, GGML_TYPE_BF16, 0), 0);
            if (K_pre) {
                Kf = K_pre;
                Vf = V_pre;
            } else {
                kv(C, w, enc ? enc : n, &Kf, &Vf);
            }
        }
        ggml_tensor * fa = ggml_flash_attn_ext(C, Qf, Kf, Vf, nullptr, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        retype(fa, GGML_TYPE_BF16);
        ggml_tensor * h1 = fused_linear(C, w.Wo, ggml_reshape_2d(C, fa, dim, Tk), w.bo, h, GGML_TYPE_F32, 0);
        ggml_tensor * f  = fused_linear(C, w.Wff0, fused_lnorm(C, h1, nullptr, nullptr, GGML_TYPE_BF16, ln_spec),
                                        w.bff0, nullptr, GGML_TYPE_BF16, FUSED_GELU_TANH);
        return fused_linear(C, w.Wff2, f, w.bff2, h1, GGML_TYPE_F32, 0);
    }

    ggml_tensor * sc = ggml_view_1d(C, mod, dim, 0);
    ggml_tensor * sh = ggml_view_1d(C, mod, dim, (size_t)dim*sizeof(float));
    const bool fz = cfg.fused_ops && !w.fq_o && !w.fq_ff0 && !w.fq_qkv && !w.fq_q;
    ggml_tensor * n;
    if (fz) {
        n = fused_ln(C, h, sc, sh, cfg.ln_eps, true, bf16_operand(w.Wqkv ? w.Wqkv : w.Wq, Tk));
    } else if (cfg.fused_ops) {
        n = fused_ln(C, h, sc, sh, cfg.ln_eps, true);
    } else {
        ggml_tensor * xn = ggml_norm(C, h, cfg.ln_eps);
        n = ggml_add(C, ggml_add(C, xn, ggml_mul(C, xn, sc)), sh);
    }
    // FoldQuant: the q (or qkv) act node runs the adaLN itself when its blob is
    // all that reads n (q/k/v sharing it, or a cross-attention layer).
    const FqLinear * nq = (w.fq_qkv && !enc) ? &w.fq_qkv : (w.fq_q && (enc || K_pre || w.fq_k)) ? &w.fq_q : nullptr;
    ggml_tensor * xq_ln = (cfg.fused_ops && nq) ? fq_act_norm(C, *nq, h, sc, sh, FusedKind::LnModulate, cfg.ln_eps) : nullptr;
    ggml_tensor *Q, *K, *V;
    if (fz && !enc && w.Wqkv) {
        // V transposed (bit 2): the non-flash attention takes V^T.
        ggml_tensor * qkv = fused_qkv_heads(C, mm_exact(C, w.Wqkv, n), w.bqkv, nullptr, nullptr, 3, hd, heads, Tk, 1, 4, 0);
        Q = fused_part(C, qkv, 0, false);
        K = fused_part(C, qkv, 1, false);
        V = fused_part(C, qkv, 2, true);
    } else if (fz && K_pre && w.Wq) {
        Q = fused_part(C, fused_qkv_heads(C, mm_exact(C, w.Wq, n), w.bq, nullptr, nullptr, 1, hd, heads, Tk, 1, 0, 0), 0, false);
        K = K_pre;
        V = V_pre;
    } else if (!enc && (w.Wqkv || w.fq_qkv)) {
        ggml_tensor * qkv = w.fq_qkv ? fq_gemm(C, w.fq_qkv, xq_ln ? xq_ln : fq_act(C, w.fq_qkv, n)) : linear(C, w.Wqkv, w.bqkv, n);
        if (w.fq_qkv && w.fq_qkv.gemm.heads) {
            Q = fq_head_view(C, qkv, w.fq_qkv, 0, Tk);
            K = fq_head_view(C, qkv, w.fq_qkv, 1, Tk);
            V = fq_head_view(C, qkv, w.fq_qkv, 2, Tk);
        } else {
        Q = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 0), 0, 2, 1, 3));
        K = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 1), 0, 2, 1, 3));
        V = ggml_cont(C, ggml_permute(C, head_view(C, qkv, hd, heads, Tk, dim, 3, 2), 1, 2, 0, 3));
        if (!cfg.fused_ops)
            hmx_kv(C, &K, &V);
        }
    } else {
        // Self-attention with separate FoldQuant q/k/v: one activation blob of n
        // serves all three projections.
        ggml_tensor * xq = xq_ln ? xq_ln : (w.fq_q && !enc && !K_pre && w.fq_k) ? fq_act(C, w.fq_q, n) : nullptr;
        if (w.fq_q) {
            ggml_tensor * qp = xq ? fq_gemm(C, w.fq_q, xq) : fq_linear(C, w.fq_q, n);
            Q = w.fq_q.gemm.heads ? fq_head_view(C, qp, w.fq_q, 0, Tk) : to_heads(C, qp, hd, heads, Tk);
        } else {
            Q = to_heads(C, linear(C, w.Wq, w.bq, n), hd, heads, Tk);
        }
        if (K_pre) {
            K = K_pre;
            V = V_pre;
        }
        else       {
            kv(C, w, enc ? enc : n, &K, &V, enc ? nullptr : xq, enc != nullptr);
        }
    }

    const bool fusable = cfg.fused_ops && Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 &&
                         V->type == GGML_TYPE_F32 && ggml_is_contiguous(Q) && ggml_is_contiguous(K) && ggml_is_contiguous(V) &&
                         K->ne[1] <= 1024 && V->ne[0] == K->ne[1] && V->ne[1] == K->ne[0];
    const bool exact = fusable && attn_exact_enabled();
    if (fz) {
        ggml_tensor * att = exact ? fused_attn_exact(C, Q, K, V, scale, bf16_operand(w.Wo, Tk))
                                  : attention_rows(C, Q, K, V, scale, dim, Tk, 1, bf16_operand(w.Wo, Tk));
        ggml_tensor * h1  = fused_bias_res(C, mm_exact(C, w.Wo, att), w.bo, h);
        ggml_tensor * n3  = fused_ln(C, h1, nullptr, nullptr, cfg.ln_eps, false, bf16_operand(w.Wff0, Tk));
        ggml_tensor * g   = fused_bias_gelu(C, mm_exact(C, w.Wff0, n3), w.bff0, bf16_operand(w.Wff2, Tk));
        return fused_bias_res(C, mm_exact(C, w.Wff2, g), w.bff2, h1);
    }
    ggml_tensor * att;
    const int64_t Nk = enc ? enc->ne[1] : Tk;
    if (!exact && (K->ne[0] > hd || K->ne[1] > Nk)) {
        // hmx_kv padded K/V: the queries take the head-dim padding too, the
        // padded keys are masked out and the output drops the padded columns.
        if (K->ne[0] > hd)
            Q = ggml_pad(C, Q, (int) (K->ne[0]-hd), 0, 0, 0);
        ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_prec_set_acc(kq, GGML_PREC_F32);
        ggml_tensor * mask = K->ne[1] > Nk ? key_mask(C, K->ne[1], Nk, Tk) : nullptr;
        ggml_tensor * kqv = ggml_mul_mat(C, V, ggml_soft_max_ext(C, kq, mask, scale, 0.0f));
        kqv = ggml_view_3d(C, kqv, hd, Tk, heads, kqv->nb[1], kqv->nb[2], 0);
        att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, kqv, 0, 2, 1, 3)), dim, Tk);
    } else {
        att = exact ? fused_attn_exact(C, Q, K, V, scale, false)
                    : attention(C, Q, K, V, nullptr, scale, dim, Tk);
    }
    // FoldQuant: both residual adds ride in the GEMM epilogues.
    ggml_tensor * h1  = w.fq_o ? fq_linear(C, w.fq_o, att, h) : ggml_add(C, h, linear(C, w.Wo, w.bo, att));
    ggml_tensor * n3  = cfg.fused_ops ? fused_ln(C, h1, nullptr, nullptr, cfg.ln_eps, false) : ggml_norm(C, h1, cfg.ln_eps);
    if (w.fq_ff0) {
        ggml_tensor * xq_ff = cfg.fused_ops ? fq_act_norm(C, w.fq_ff0, h1, nullptr, nullptr, FusedKind::LnPlain, cfg.ln_eps) : nullptr;
        ggml_tensor * f0 = fq_gemm(C, w.fq_ff0, xq_ff ? xq_ff : fq_act(C, w.fq_ff0, n3));
        if (ggml_tensor * dq = cfg.fused_ops ? fq_act_glu(C, w.fq_ff2, f0, nullptr, FusedGlu::Gelu) : nullptr)
            return fq_gemm(C, w.fq_ff2, dq, h1);
        return fq_linear(C, w.fq_ff2, ggml_gelu(C, f0), h1);
    }
    return ggml_add(C, h1, ffn_gelu(C, w.Wff0, w.bff0, w.Wff2, w.bff2, n3));
}

ggml_tensor * DitHead::time_emb(ggml_context * C, ggml_tensor * tproj) const {
    return linear(C, te_l2W, te_l2b, ggml_silu(C, linear(C, te_l1W, te_l1b, tproj)));
}

ggml_tensor * DitHead::proj_out(ggml_context * C, ggml_tensor * h, ggml_tensor * mod) const {
    ggml_tensor * sh = ggml_view_1d(C, mod, cfg.hidden, 0);
    ggml_tensor * sc = ggml_view_1d(C, mod, cfg.hidden, (size_t)cfg.hidden*sizeof(float));
    if (fused)  // sc holds 1 + scale here
        return fused_linear(C, po2W, fused_lnorm(C, h, sc, sh, GGML_TYPE_BF16, out_ln_spec), po2b, nullptr,
                            GGML_TYPE_F32, 0);

    ggml_tensor * hn    = ggml_norm(C, h, cfg.norm_out_eps);
    ggml_tensor * h_mod = ggml_add(C, ggml_add(C, hn, ggml_mul(C, hn, sc)), sh);
    return linear(C, po2W, po2b, h_mod);
}

bool dit_enable_fused(DitHead & dit, const Backend & b, ggml_type weight_type, bool fq, const Options & opts,
                      const char * tag) {
    if (!b.is_sycl || weight_type != GGML_TYPE_BF16 || !fa_takes_views() || fq || !opts.flash_attn.value_or(true) ||
        env_flag("VLA_GR00T_NO_FUSE"))
        return false;
    dit.fused = true;
    dit.ln_spec.eps     = dit.cfg.ln_eps;
    dit.out_ln_spec.eps = dit.cfg.norm_out_eps;
    sycl_register_fused_ops();
    sycl_register_bf16_ops();
    std::printf("vla(%s): fused DiT blocks (VLA_GR00T_NO_FUSE=1 to disable)\n", tag);
    return true;
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
        // The fused blocks take the adaLN as a LayerNorm with weight 1 + scale:
        // scale is the first half of a block's row and the second of proj_out's.
        if (dit.fused)
            for (size_t k=0; k<outs.size(); ++k) {
                const bool proj = (int64_t) (k % (size_t) per_step) == layers;
                float * sc = host.data() + k*(size_t) (2*dim) + (proj ? dim : 0);
                for (int64_t j=0; j<dim; ++j) sc[j] += 1.0f;
            }
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
