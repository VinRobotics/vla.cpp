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
        w.Wq   = L.gemm("%s.%lld.attn_q.weight",  prefix, (long long)i);
        w.bq   = L.f32 ("%s.%lld.attn_q.bias",    prefix, (long long)i);
        w.Wk   = L.gemm("%s.%lld.attn_k.weight",  prefix, (long long)i);
        w.bk   = L.f32 ("%s.%lld.attn_k.bias",    prefix, (long long)i);
        w.Wv   = L.gemm("%s.%lld.attn_v.weight",  prefix, (long long)i);
        w.bv   = L.f32 ("%s.%lld.attn_v.bias",    prefix, (long long)i);
        w.Wo   = L.gemm("%s.%lld.attn_o.weight",  prefix, (long long)i);
        w.bo   = L.f32 ("%s.%lld.attn_o.bias",    prefix, (long long)i);
        w.Wfc1 = L.gemm("%s.%lld.%s.weight",      prefix, (long long)i, n.fc1);
        w.bfc1 = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.fc1);
        w.Wfc2 = L.gemm("%s.%lld.%s.weight",      prefix, (long long)i, n.fc2);
        w.bfc2 = L.f32 ("%s.%lld.%s.bias",        prefix, (long long)i, n.fc2);
    }
}

ggml_tensor * EncStack::block(ggml_context * C, const EncBlockW & w, ggml_tensor * x,
                              int64_t seq, int64_t nv) const {
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
