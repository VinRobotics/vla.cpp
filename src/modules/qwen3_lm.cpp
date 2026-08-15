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

#include "modules/qwen3_lm.h"

#include "layers/attn.h"
#include "layers/ffn.h"
#include "layers/norm.h"

#include <cmath>

namespace vla {

void Qwen3LM::declare(WeightLoader & L, const char * prefix) {
    output_norm = L.f32("%s.output_norm.weight", prefix);
    blk.resize(cfg.layers);

    for (int64_t i=0; i<cfg.layers; ++i) {
        Qwen3LayerW & w = blk[i];
        w.attn_norm = L.f32 ("%s.blk.%lld.attn_norm.weight",   prefix, (long long)i);
        w.Wq        = L.gemm("%s.blk.%lld.attn_q.weight",      prefix, (long long)i);
        w.Wk        = L.gemm("%s.blk.%lld.attn_k.weight",      prefix, (long long)i);
        w.Wv        = L.gemm("%s.blk.%lld.attn_v.weight",      prefix, (long long)i);
        w.Wo        = L.gemm("%s.blk.%lld.attn_o.weight",      prefix, (long long)i);
        w.q_norm    = L.f32 ("%s.blk.%lld.attn_q_norm.weight", prefix, (long long)i);
        w.k_norm    = L.f32 ("%s.blk.%lld.attn_k_norm.weight", prefix, (long long)i);
        w.ffn_norm  = L.f32 ("%s.blk.%lld.ffn_norm.weight",    prefix, (long long)i);
        w.Wgate     = L.gemm("%s.blk.%lld.ffn_gate.weight",    prefix, (long long)i);
        w.Wup       = L.gemm("%s.blk.%lld.ffn_up.weight",      prefix, (long long)i);
        w.Wdown     = L.gemm("%s.blk.%lld.ffn_down.weight",    prefix, (long long)i);
    }
}

ggml_tensor * Qwen3LM::block(ggml_context * C, const Qwen3LayerW & w, ggml_tensor * h,
                             ggml_tensor * pos, ggml_tensor * mask, int64_t seq) const {
    const int64_t hd   = cfg.head_dim;
    const int64_t n_q  = cfg.n_q;
    const int64_t n_kv = cfg.n_kv;
    const int64_t hq   = n_q*hd;
    const float   scale= 1.0f/std::sqrt((float)hd);

    ggml_tensor * hn = rms_norm(C, h, w.attn_norm, cfg.rms_eps);
    ggml_tensor * qh = ggml_reshape_3d(C, ggml_mul_mat(C, w.Wq, hn), hd, n_q,  seq);
    ggml_tensor * kh = ggml_reshape_3d(C, ggml_mul_mat(C, w.Wk, hn), hd, n_kv, seq);
    ggml_tensor * vh = ggml_reshape_3d(C, ggml_mul_mat(C, w.Wv, hn), hd, n_kv, seq);

    ggml_tensor * qr = rope(C, cfg.rope, rms_norm(C, qh, w.q_norm, cfg.rms_eps), pos);
    ggml_tensor * kr = rope(C, cfg.rope, rms_norm(C, kh, w.k_norm, cfg.rms_eps), pos);
    ggml_tensor * Q  = ggml_cont(C, ggml_permute(C, qr, 0, 2, 1, 3));
    ggml_tensor * K  = ggml_cont(C, ggml_permute(C, kr, 0, 2, 1, 3));

    ggml_tensor * att;
    if (cfg.flash_attn) {
        ggml_tensor * V = ggml_cont(C, ggml_permute(C, vh, 0, 2, 1, 3));
        att = flash_attention(C, Q, K, V, ggml_cast(C, mask, GGML_TYPE_F16), scale);
    } else {
        ggml_tensor * V = ggml_cont(C, ggml_permute(C, vh, 1, 2, 0, 3));
        att = attention(C, Q, K, V, mask, scale, hq, seq);
    }

    ggml_tensor * h_attn = ggml_add(C, h, ggml_mul_mat(C, w.Wo, att));
    ggml_tensor * hn2    = rms_norm(C, h_attn, w.ffn_norm, cfg.rms_eps);
    return ggml_add(C, h_attn, ffn_swiglu(C, w.Wgate, nullptr, w.Wup, nullptr, w.Wdown, nullptr, hn2));
}

ggml_tensor * Qwen3LM::build(ggml_context * C, ggml_tensor * h,
                             ggml_tensor * pos, ggml_tensor * mask, int64_t seq) const {
    for (int64_t i=0; i<cfg.layers; ++i)
        h = block(C, blk[i], h, pos, mask, seq);

    return rms_norm(C, h, output_norm, cfg.rms_eps);
}

}
