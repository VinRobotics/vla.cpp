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

#include "modules/encoder.h"

#include "layers/attn.h"
#include "layers/ffn.h"
#include "layers/fused_ops.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "options.h"
#include "act_dtype.h"
#include "layers/fused_ew.h"
#include "layers/qlinear.h"

#include <cmath>

namespace vla {

void EncStack::declare(WeightLoader & L, const char * prefix, int64_t layers, const EncNames & n) {
    blk.resize(layers);
    ln_spec.eps = cfg.ln_eps;

    for (int64_t i=0; i<layers; ++i) {
        EncBlockW & w = blk[i];
        w.ln1w = L.f32 ("%s.%lld.%s.weight",      prefix, (long long)i, n.ln1);
        w.ln1b = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.ln1);
        w.ln2w = L.f32 ("%s.%lld.%s.weight",      prefix, (long long)i, n.ln2);
        w.ln2b = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.ln2);
        if (cfg.fused) {
            const std::string b = std::string(prefix) + "." + std::to_string(i) + ".attn_";
            w.Wq = w.bq = w.Wk = w.bk = w.Wv = w.bv = nullptr;
            if (cfg.int8)
                w.q_qkv = L.quant_i8(b + "qkv.q8", {b + "q.weight", b + "k.weight", b + "v.weight"});
            else
                w.Wqkv = L.fuse_gemm((b + "qkv.weight").c_str(), {b + "q.weight", b + "k.weight", b + "v.weight"});
            w.bqkv = L.fuse_f32 ((b + "qkv.bias").c_str(),   {b + "q.bias",   b + "k.bias",   b + "v.bias"});
        } else {
            w.Wq   = L.gemm("%s.%lld.attn_q.weight",  prefix, (long long)i);
            w.bq   = L.f32 ("%s.%lld.attn_q.bias",    prefix, (long long)i);
            w.Wk   = L.gemm("%s.%lld.attn_k.weight",  prefix, (long long)i);
            w.bk   = L.f32 ("%s.%lld.attn_k.bias",    prefix, (long long)i);
            w.Wv   = L.gemm("%s.%lld.attn_v.weight",  prefix, (long long)i);
            w.bv   = L.f32 ("%s.%lld.attn_v.bias",    prefix, (long long)i);
        }
        if (cfg.fused && cfg.int8) {
            const std::string b = std::string(prefix) + "." + std::to_string(i) + ".";
            w.Wo = w.Wfc1 = w.Wfc2 = nullptr;
            w.q_o   = L.quant_i8(b + "attn_o.q8", {b + "attn_o.weight"});
            w.q_fc1 = L.quant_i8(b + n.fc1 + ".q8", {b + n.fc1 + ".weight"});
            w.q_fc2 = L.quant_i8(b + n.fc2 + ".q8", {b + n.fc2 + ".weight"});
        } else {
            w.Wo   = L.gemm("%s.%lld.attn_o.weight",  prefix, (long long)i);
            w.Wfc1 = L.gemm("%s.%lld.%s.weight",      prefix, (long long)i, n.fc1);
            w.Wfc2 = L.gemm("%s.%lld.%s.weight",      prefix, (long long)i, n.fc2);
        }
        w.bo   = L.f32 ("%s.%lld.attn_o.bias",    prefix, (long long)i);
        w.bfc1 = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.fc1);
        w.bfc2 = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.fc2);
    }
}

ggml_tensor * EncStack::block(ggml_context * C, const EncBlockW & w, ggml_tensor * x,
                              int64_t seq, int64_t nv) const {
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const float   scale = 1.0f/std::sqrt((float)hd);

    if (cfg.fused) {
        // One node per step (layers/fused_ew.h). Attention reads the q|k|v
        // GEMM's output in place and writes BF16. nv views are nv * seq rows to
        // every GEMM and norm, and a batch of nv to the attention.
        GGML_ASSERT(cfg.act == GGML_TYPE_BF16 && (w.Wqkv || w.q_qkv));
        const ggml_type at = cfg.act;
        const FusedLNormSpec & ln = ln_spec;
        auto lin = [&](ggml_tensor * W, const WeightLoader::QWeight & q, ggml_tensor * in, ggml_tensor * b,
                       ggml_tensor * res, bool gelu) {
            return q ? qlinear(C, q, in, b, res, at, gelu) : fused_linear(C, W, in, b, res, at, gelu);
        };
        ggml_tensor * qkv = lin(w.Wqkv, w.q_qkv, fused_lnorm(C, x, w.ln1w, w.ln1b, at, ln), w.bqkv, nullptr, false);
        const size_t es = ggml_element_size(qkv);
        auto heads_view = [&](int64_t i) {
            return ggml_permute(C, ggml_view_4d(C, qkv, hd, heads, seq, nv, hd * es, qkv->nb[1], seq * qkv->nb[1],
                                                i * cfg.hidden * es),
                                0, 2, 1, 3);
        };
        ggml_tensor * o = ggml_flash_attn_ext(C, heads_view(0), heads_view(1), heads_view(2), nullptr, scale,
                                              0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        retype(o, at);
        ggml_tensor * att = ggml_reshape_2d(C, o, cfg.hidden, seq * nv);
        ggml_tensor * h1  = lin(w.Wo, w.q_o, att, w.bo, x, false);
        ggml_tensor * f1  = lin(w.Wfc1, w.q_fc1, fused_lnorm(C, h1, w.ln2w, w.ln2b, at, ln), w.bfc1, nullptr, true);
        return lin(w.Wfc2, w.q_fc2, f1, w.bfc2, h1, false);
    }

    if (cfg.act != GGML_TYPE_F32) {
        // Reduced-precision activations: GEMMs in cfg.act, scores and softmax
        // in F32. Single view only; no caller of this path passes nv > 1.
        const ggml_type at = cfg.act;
        ggml_tensor * n1 = layer_norm(C, x, w.ln1w, w.ln1b, cfg.ln_eps);
        ggml_tensor * q  = ggml_add(C, mm_act(C, w.Wq, n1, at), w.bq);
        ggml_tensor * k  = ggml_add(C, mm_act(C, w.Wk, n1, at), w.bk);
        ggml_tensor * v  = ggml_add(C, mm_act(C, w.Wv, n1, at), w.bv);
        ggml_tensor * att;
        if (cfg.flash_attn && fa_takes_views()) {
            auto heads_view = [&](ggml_tensor * p) {
                return ggml_permute(C, ggml_reshape_3d(C, p, hd, heads, seq), 0, 2, 1, 3);
            };
            ggml_tensor * o = ggml_flash_attn_ext(C, heads_view(q), heads_view(k), heads_view(v), nullptr, scale,
                                                  0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            att = ggml_reshape_2d(C, o, cfg.hidden, seq);
        } else {
            q = as_type(C, q, GGML_TYPE_F32);
            k = as_type(C, k, GGML_TYPE_F32);
            v = as_type(C, v, GGML_TYPE_F32);
            att = attention(C, to_heads(C, q, hd, heads, seq), to_heads(C, k, hd, heads, seq),
                            to_heads_v(C, v, hd, heads, seq), nullptr, scale, cfg.hidden, seq);
        }
        ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, mm_act(C, w.Wo, as_type(C, att, at), at), w.bo));
        ggml_tensor * n2 = layer_norm(C, h1, w.ln2w, w.ln2b, cfg.ln_eps);
        ggml_tensor * ff = ggml_add(C, mm_act(C, w.Wfc2, gelu(C, ggml_add(C, mm_act(C, w.Wfc1, n2, at), w.bfc1)), at),
                                    w.bfc2);
        return ggml_add(C, h1, ff);
    }

    // Fused path for one view only: with several, ggml multiplies the 3D
    // activations with one batched cuBLAS call, which the fused ops' single
    // GEMM over all views does not reproduce bit for bit.
    if (cfg.fused_ops && nv == 1) {
        const bool fa = cfg.flash_attn;
        ggml_tensor * n1 = fused_ln(C, x, w.ln1w, w.ln1b, cfg.ln_eps, false,
                                    bf16_operand(w.Wq, seq) && bf16_operand(w.Wk, seq) && bf16_operand(w.Wv, seq));
        auto heads_of = [&](ggml_tensor * W, ggml_tensor * b, bool vt, bool f16) {
            return fused_part(C, fused_qkv_heads(C, mm_exact(C, W, n1), b, nullptr, nullptr, 1, hd, heads, seq, nv,
                                                 vt ? 1 : 0, 0, 0, f16), 0, vt);
        };
        // With flash attention K/V are written F16, the type it casts them to.
        ggml_tensor * Q = heads_of(w.Wq, w.bq, false, false), * K = heads_of(w.Wk, w.bk, false, fa);
        ggml_tensor * att = fa ? ggml_reshape_2d(C, flash_attention(C, Q, K, heads_of(w.Wv, w.bv, false, true), nullptr, scale), cfg.hidden, seq*nv)
                               : attention_rows_exact(C, Q, K, heads_of(w.Wv, w.bv, true, false), scale, cfg.hidden, seq, nv, bf16_operand(w.Wo, seq));
        ggml_tensor * h1 = fused_bias_res(C, mm_exact(C, w.Wo, att), w.bo, x);
        ggml_tensor * n2 = fused_ln(C, h1, w.ln2w, w.ln2b, cfg.ln_eps, false, bf16_operand(w.Wfc1, seq));
        ggml_tensor * g  = fused_bias_gelu(C, mm_exact(C, w.Wfc1, n2), w.bfc1, bf16_operand(w.Wfc2, seq));
        return fused_bias_res(C, mm_exact(C, w.Wfc2, g), w.bfc2, h1);
    }

    ggml_tensor * n1 = layer_norm(C, x, w.ln1w, w.ln1b, cfg.ln_eps);
    ggml_tensor * q  = linear(C, w.Wq, w.bq, n1);
    ggml_tensor * k  = linear(C, w.Wk, w.bk, n1);
    ggml_tensor * v  = linear(C, w.Wv, w.bv, n1);

    ggml_tensor * att;
    if (cfg.flash_attn && fa_takes_views()) {
        // A view-taking FA (fa_takes_views) reads the projections in place and
        // converts them itself, so no contiguous copy or F16 cast per operand.
        auto heads_view = [&](ggml_tensor * p) {
            return ggml_permute(C, ggml_reshape_4d(C, p, hd, heads, seq, nv), 0, 2, 1, 3);
        };
        ggml_tensor * o = ggml_flash_attn_ext(C, heads_view(q), heads_view(k), heads_view(v), nullptr, scale,
                                              0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        att = ggml_reshape_3d(C, o, cfg.hidden, seq, nv);
    } else {
        ggml_tensor * Q = to_heads(C, q, hd, heads, seq, nv);
        ggml_tensor * K = to_heads(C, k, hd, heads, seq, nv);
        if (cfg.flash_attn)
            att = flash_attention(C, Q, K, to_heads(C, v, hd, heads, seq, nv), nullptr, scale);
        else
            att = attention(C, Q, K, to_heads_v(C, v, hd, heads, seq, nv), nullptr, scale, cfg.hidden, seq, nv);
    }

    ggml_tensor * h1 = ggml_add(C, x, linear(C, w.Wo, w.bo, att));
    ggml_tensor * n2 = layer_norm(C, h1, w.ln2w, w.ln2b, cfg.ln_eps);
    return ggml_add(C, h1, ffn_gelu(C, w.Wfc1, w.bfc1, w.Wfc2, w.bfc2, n2));
}

ggml_tensor * EncStack::build(ggml_context * C, ggml_tensor * x, int64_t seq, int64_t nv) const {
    for (size_t i=0; i<blk.size(); ++i)
        x = block(C, blk[i], x, seq, nv);

    return x;
}

}
