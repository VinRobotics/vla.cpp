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
#include "layers/fq_linear.h"
#include "layers/norm.h"

#include <cmath>

namespace vla {

void Qwen3LM::declare(WeightLoader & L, const char * prefix, const FqModuleSpec * fq) {
    output_norm = L.f32("%s.output_norm.weight", prefix);
    blk.resize(cfg.layers);

    for (int64_t i=0; i<cfg.layers; ++i) {
        Qwen3LayerW & w = blk[i];
        const long long ii = (long long) i;
        w.attn_norm = L.f32 ("%s.blk.%lld.attn_norm.weight",   prefix, ii);
        w.q_norm    = L.f32 ("%s.blk.%lld.attn_q_norm.weight", prefix, ii);
        w.k_norm    = L.f32 ("%s.blk.%lld.attn_k_norm.weight", prefix, ii);
        w.ffn_norm  = L.f32 ("%s.blk.%lld.ffn_norm.weight",    prefix, ii);

        if (fq) {
            // Site keys follow VLA-OPT's site_bits vocabulary: qkv | o | gateup | down.
            // The norm before q/k/v and gate/up is fused into the activation node.
            w.fq_q    = fq_declare_linear(L, *fq, "qkv",    false, w.attn_norm, cfg.rms_eps, "%s.blk.%lld.attn_q",   prefix, ii);
            w.fq_k    = fq_declare_linear(L, *fq, "qkv",    false, w.attn_norm, cfg.rms_eps, "%s.blk.%lld.attn_k",   prefix, ii);
            w.fq_v    = fq_declare_linear(L, *fq, "qkv",    false, w.attn_norm, cfg.rms_eps, "%s.blk.%lld.attn_v",   prefix, ii);
            w.fq_o    = fq_declare_linear(L, *fq, "o",      false, nullptr,     0.0f,        "%s.blk.%lld.attn_o",   prefix, ii);
            w.fq_gate = fq_declare_linear(L, *fq, "gateup", false, w.ffn_norm,  cfg.rms_eps, "%s.blk.%lld.ffn_gate", prefix, ii);
            w.fq_up   = fq_declare_linear(L, *fq, "gateup", false, w.ffn_norm,  cfg.rms_eps, "%s.blk.%lld.ffn_up",   prefix, ii);
            w.fq_down = fq_declare_linear(L, *fq, "down",   false, nullptr,     0.0f,        "%s.blk.%lld.ffn_down", prefix, ii);
            // Sites that share an activation node must be quantized together.
            if ((bool) w.fq_q != (bool) w.fq_k || (bool) w.fq_q != (bool) w.fq_v || (bool) w.fq_gate != (bool) w.fq_up)
                L.fail("FoldQuant: q/k/v (and gate/up) must all be INT8 or all float in a layer");
        }
        if (!w.fq_q)    w.Wq    = L.gemm("%s.blk.%lld.attn_q.weight",   prefix, ii);
        if (!w.fq_k)    w.Wk    = L.gemm("%s.blk.%lld.attn_k.weight",   prefix, ii);
        if (!w.fq_v)    w.Wv    = L.gemm("%s.blk.%lld.attn_v.weight",   prefix, ii);
        if (!w.fq_o)    w.Wo    = L.gemm("%s.blk.%lld.attn_o.weight",   prefix, ii);
        if (!w.fq_gate) w.Wgate = L.gemm("%s.blk.%lld.ffn_gate.weight", prefix, ii);
        if (!w.fq_up)   w.Wup   = L.gemm("%s.blk.%lld.ffn_up.weight",   prefix, ii);
        if (!w.fq_down) w.Wdown = L.gemm("%s.blk.%lld.ffn_down.weight", prefix, ii);
    }
}

ggml_tensor * Qwen3LM::block(ggml_context * C, const Qwen3LayerW & w, ggml_tensor * h,
                             ggml_tensor * pos, ggml_tensor * mask, int64_t seq) const {
    const int64_t hd   = cfg.head_dim;
    const int64_t n_q  = cfg.n_q;
    const int64_t n_kv = cfg.n_kv;
    const int64_t hq   = n_q*hd;
    const float   scale= 1.0f/std::sqrt((float)hd);

    // FoldQuant q/k/v: one activation node (RMSNorm fused) feeds three INT8 GEMMs.
    ggml_tensor * hn = nullptr, * xq = nullptr;
    if (w.fq_q) xq = fq_act(C, w.fq_q, h);
    else        hn = rms_norm(C, h, w.attn_norm, cfg.rms_eps);
    auto proj = [&](const FqLinear & fq, ggml_tensor * W) {
        return fq ? fq_gemm(C, fq, xq) : ggml_mul_mat(C, W, hn);
    };
    ggml_tensor * qh = ggml_reshape_3d(C, proj(w.fq_q, w.Wq), hd, n_q,  seq);
    ggml_tensor * kh = ggml_reshape_3d(C, proj(w.fq_k, w.Wk), hd, n_kv, seq);
    ggml_tensor * vh = ggml_reshape_3d(C, proj(w.fq_v, w.Wv), hd, n_kv, seq);

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

    // o_proj takes the raw attention output (no norm) and is still rotated.
    ggml_tensor * o      = w.fq_o ? fq_linear(C, w.fq_o, att) : ggml_mul_mat(C, w.Wo, att);
    ggml_tensor * h_attn = ggml_add(C, h, o);
    // gate/up (one activation node, ffn_norm fused) and down are independent
    // sites; the float form is ffn_swiglu's op sequence exactly.
    ggml_tensor * gate, * up;
    if (w.fq_gate) {
        ggml_tensor * xq2 = fq_act(C, w.fq_gate, h_attn);
        gate = ggml_silu(C, fq_gemm(C, w.fq_gate, xq2));
        up   = fq_gemm(C, w.fq_up, xq2);
    } else {
        ggml_tensor * hn2 = rms_norm(C, h_attn, w.ffn_norm, cfg.rms_eps);
        gate = ggml_silu(C, ggml_mul_mat(C, w.Wgate, hn2));
        up   = ggml_mul_mat(C, w.Wup, hn2);
    }
    ggml_tensor * mid = ggml_mul(C, gate, up);
    ggml_tensor * ffn = w.fq_down ? fq_linear(C, w.fq_down, mid) : ggml_mul_mat(C, w.Wdown, mid);
    return ggml_add(C, h_attn, ffn);
}

ggml_tensor * Qwen3LM::build(ggml_context * C, ggml_tensor * h,
                             ggml_tensor * pos, ggml_tensor * mask, int64_t seq) const {
    for (int64_t i=0; i<cfg.layers; ++i)
        h = block(C, blk[i], h, pos, mask, seq);

    return rms_norm(C, h, output_norm, cfg.rms_eps);
}

}
