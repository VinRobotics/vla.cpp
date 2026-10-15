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
#include "layers/fmha_kv.h"
#include "layers/vit_fused.h"
#include "layers/linear.h"
#include "layers/norm.h"

#include <cmath>

namespace vla {

void EncStack::declare(WeightLoader & L, const char * prefix, int64_t layers, const EncNames & n) {
    blk.resize(layers);

    for (int64_t i=0; i<layers; ++i) {
        EncBlockW & w = blk[i];
        w.ln1w = L.f32 ("%s.%lld.%s.weight",      prefix, (long long)i, n.ln1);
        w.ln1b = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.ln1);
        w.ln2w = L.f32 ("%s.%lld.%s.weight",      prefix, (long long)i, n.ln2);
        w.ln2b = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.ln2);
#ifdef GGML_USE_SYCL
        {
            const std::string p = std::string(prefix) + "." + std::to_string(i) + ".attn_";
            w.Wqkv = L.fuse_gemm((p + "qkv.fused.weight").c_str(), { p + "q.weight", p + "k.weight", p + "v.weight" });
            w.bqkv = L.fuse_f32 ((p + "qkv.fused.bias").c_str(),   { p + "q.bias",   p + "k.bias",   p + "v.bias" });
        }
#else
        w.Wq   = L.gemm("%s.%lld.attn_q.weight",  prefix, (long long)i);
        w.bq   = L.f32 ("%s.%lld.attn_q.bias",    prefix, (long long)i);
        w.Wk   = L.gemm("%s.%lld.attn_k.weight",  prefix, (long long)i);
        w.bk   = L.f32 ("%s.%lld.attn_k.bias",    prefix, (long long)i);
        w.Wv   = L.gemm("%s.%lld.attn_v.weight",  prefix, (long long)i);
        w.bv   = L.f32 ("%s.%lld.attn_v.bias",    prefix, (long long)i);
#endif
        w.Wo   = L.gemm("%s.%lld.attn_o.weight",  prefix, (long long)i);
        w.bo   = L.f32 ("%s.%lld.attn_o.bias",    prefix, (long long)i);
        w.Wfc1 = L.gemm("%s.%lld.%s.weight",      prefix, (long long)i, n.fc1);
        w.bfc1 = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.fc1);
        w.Wfc2 = L.gemm("%s.%lld.%s.weight",      prefix, (long long)i, n.fc2);
        w.bfc2 = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.fc2);
    }
}

// The fused-q/k/v block (SYCL). Every view's tokens go through each GEMM at
// once ([hidden, seq*nv], not nv batched [hidden, seq] products), and the heads
// are strided views of the one projection. With the sycl-tla kernel (head_dim
// 72) those views feed the attention node directly: no per-head copies, no F16
// casts; it packs them itself.
static ggml_tensor * block_fused(ggml_context * C, const EncCfg & cfg, const EncBlockW & w, ggml_tensor * x,
                                 int64_t seq, int64_t nv) {
    const int64_t hd = cfg.head_dim, heads = cfg.heads, E = cfg.hidden, T = seq * nv;
    const float   scale = 1.0f/std::sqrt((float)hd);
    ggml_tensor * x2 = ggml_reshape_2d(C, ggml_is_contiguous(x) ? x : ggml_cont(C, x), E, T);

    const bool fz = vit_fused_enabled();   // one-kernel LayerNorm / bias+gelu / bias+residual
    auto ln = [&](ggml_tensor * t, ggml_tensor * g, ggml_tensor * b) {
        return fz ? vit_ln(C, t, g, b, cfg.ln_eps) : layer_norm(C, t, g, b, cfg.ln_eps);
    };
    // BF16 weights on SYCL: every projection one oneDNN matmul with its bias and
    // post-op, LayerNorms and attention handing it BF16 (vit_fused.h).
    if (fz && w.Wqkv->type == GGML_TYPE_BF16 && w.Wo->type == GGML_TYPE_BF16 && w.Wfc1->type == GGML_TYPE_BF16 &&
        w.Wfc2->type == GGML_TYPE_BF16 && cfg.flash_attn && hd == 72 && fmha_kv_enabled()) {
        ggml_tensor * qkv = vit_linear(C, vit_ln(C, x2, w.ln1w, w.ln1b, cfg.ln_eps, GGML_TYPE_BF16), w.Wqkv, w.bqkv,
                                       VitPost::NONE, nullptr, GGML_TYPE_F32);
        ggml_tensor * att = vit_attn(C, qkv, (int) heads, (int) hd, seq);
        ggml_tensor * h1  = vit_linear(C, att, w.Wo, w.bo, VitPost::RES, x2, GGML_TYPE_F32);
        ggml_tensor * f1  = vit_linear(C, vit_ln(C, h1, w.ln2w, w.ln2b, cfg.ln_eps, GGML_TYPE_BF16), w.Wfc1, w.bfc1,
                                       VitPost::GELU, nullptr, GGML_TYPE_BF16);
        return ggml_reshape(C, vit_linear(C, f1, w.Wfc2, w.bfc2, VitPost::RES, h1, GGML_TYPE_F32), x);
    }
    ggml_tensor * qkv = linear(C, w.Wqkv, w.bqkv, ln(x2, w.ln1w, w.ln1b));   // [3E, T]
    const size_t es = ggml_element_size(qkv);
    auto split = [&](int b) {   // [hd, heads, seq, nv] view of block b
        return ggml_view_4d(C, qkv, hd, heads, seq, nv, (size_t) hd*es, (size_t) 3*E*es, (size_t) 3*E*seq*es,
                            (size_t) b*E*es);
    };
    auto as_heads = [&](int b) { return ggml_permute(C, split(b), 0, 2, 1, 3); };   // [hd, seq, heads, nv]

    ggml_tensor * att;
    if (cfg.flash_attn && hd == 72 && fmha_kv_enabled()) {
        ggml_tensor * o = ggml_flash_attn_ext(C, as_heads(0), as_heads(1), as_heads(2), nullptr, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        att = ggml_reshape_2d(C, o, E, T);
    } else if (cfg.flash_attn) {
        att = ggml_reshape_2d(C, flash_attention(C, ggml_cont(C, as_heads(0)), ggml_cont(C, as_heads(1)),
                                                 ggml_cont(C, as_heads(2)), nullptr, scale), E, T);
    } else {
        ggml_tensor * vt = ggml_cont(C, ggml_permute(C, split(2), 1, 2, 0, 3));
        att = ggml_reshape_2d(C, attention(C, ggml_cont(C, as_heads(0)), ggml_cont(C, as_heads(1)), vt, nullptr,
                                           scale, E, seq, nv), E, T);
    }

    if (fz) {
        ggml_tensor * h1 = vit_bias_res(C, ggml_mul_mat(C, w.Wo, att), w.bo, x2);
        ggml_tensor * f1 = vit_bias_gelu(C, ggml_mul_mat(C, w.Wfc1, ln(h1, w.ln2w, w.ln2b)), w.bfc1);
        return ggml_reshape(C, vit_bias_res(C, ggml_mul_mat(C, w.Wfc2, f1), w.bfc2, h1), x);
    }
    ggml_tensor * h1 = ggml_add(C, x2, linear(C, w.Wo, w.bo, att));
    ggml_tensor * n2 = layer_norm(C, h1, w.ln2w, w.ln2b, cfg.ln_eps);
    ggml_tensor * y  = ggml_add(C, h1, ffn_gelu(C, w.Wfc1, w.bfc1, w.Wfc2, w.bfc2, n2));
    return ggml_reshape(C, y, x);
}

ggml_tensor * EncStack::block(ggml_context * C, const EncBlockW & w, ggml_tensor * x,
                              int64_t seq, int64_t nv) const {
    if (w.Wqkv)
        return block_fused(C, cfg, w, x, seq, nv);
    const int64_t hd    = cfg.head_dim;
    const int64_t heads = cfg.heads;
    const float   scale = 1.0f/std::sqrt((float)hd);

    ggml_tensor * n1 = layer_norm(C, x, w.ln1w, w.ln1b, cfg.ln_eps);
    ggml_tensor * q  = linear(C, w.Wq, w.bq, n1);
    ggml_tensor * k  = linear(C, w.Wk, w.bk, n1);
    ggml_tensor * v  = linear(C, w.Wv, w.bv, n1);
    ggml_tensor * Q  = to_heads(C, q, hd, heads, seq, nv);
    ggml_tensor * K  = to_heads(C, k, hd, heads, seq, nv);

    ggml_tensor * att;
    if (cfg.flash_attn)
        att = flash_attention(C, Q, K, to_heads(C, v, hd, heads, seq, nv), nullptr, scale);
    else
        att = attention(C, Q, K, to_heads_v(C, v, hd, heads, seq, nv), nullptr, scale, cfg.hidden, seq, nv);

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
