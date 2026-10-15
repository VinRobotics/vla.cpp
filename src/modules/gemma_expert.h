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

// The pieces pi0 and pi0.5 share: a Gemma prefix tower over the image and
// language tokens, an action expert that attends to its K/V, the PaliGemma
// vision tower, and the GGUF config and stats readers.

#pragma once

#include "act_dtype.h"
#include "backend.h"
#include "gguf_reader.h"
#include "layers/norm.h"
#include "foldquant.h"
#include "layers/fq_linear.h"
#include "layers/fmha_kv.h"
#include "layers/fq_rope_attn.h"
#include "loader.h"
#include "model.h"
#include "modules/siglip_vit.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vla {

struct GemmaLayerW {
    ggml_tensor * ln_in   = nullptr;
    ggml_tensor * Wq      = nullptr;
    ggml_tensor * Wk      = nullptr;
    ggml_tensor * Wv      = nullptr;
    ggml_tensor * Wo      = nullptr;
    ggml_tensor * ln_post = nullptr;
    ggml_tensor * Wgate   = nullptr;
    ggml_tensor * Wup     = nullptr;
    ggml_tensor * Wdown   = nullptr;
    // q/k/v and gate/up concatenated into one GEMM each (SYCL; see gemma_declare_float).
    ggml_tensor * Wqkv    = nullptr;
    ggml_tensor * Wgu     = nullptr;
    // FoldQuant sites (pi0.5 prefix tower). The RMSNorm and its folded gamma
    // (ln_in / ln_post, loaded as 1 + w) ride in the q/k/v and gate/up act nodes.
    FqLinear fq_q, fq_k, fq_v, fq_o, fq_gate, fq_up, fq_down;
    // gate and up fused into one FoldQuant GEMM with the GEGLU in its epilogue
    // (SYCL; see gemma_declare_fq_gateup); fq_gate/fq_up stay empty then.
    FqLinear fq_gu;
    // q/k/v as one FoldQuant GEMM (SYCL; gemma_declare_fq_qkv); fq_q/k/v empty then.
    FqLinear fq_qkv;
    // pi0.5's expert on SYCL: the q/k/v and gate/up act specs with the adaRMS
    // fused (has_shift set when in use; see fq_act_mod).
    FqActSpec act_mod_qkv{}, act_mod_gu{};
    // gate/up GEMM + GEGLU + the down act in one node (gemma_link_glu_act).
    FqGluActSpec glu_act{ 0 };
    // q/k/v GEMM + RoPE + cached-KV attention in one node (pi0.5's expert, SYCL).
    FqRopeAttnSpec rope_attn{};
    // q/k/v GEMM + RoPE + F16 packing in one node (pi0.5's prefix, SYCL); the
    // attention reads it in place and k_out / v_out are F16 [hd, seq, 1] views.
    FqRopeAttnSpec rope_pack{};
};

/// Declare a float layer's q/k/v and gate/up GEMMs, `p` being the tensor-name
/// prefix ("vlm.blk.3"). On SYCL each group is one concatenated weight: one GEMM
/// and one F32->BF16 input conversion instead of three (two), which is most of a
/// small-M layer's cost there (pi0.5's action expert runs 50 tokens, 180 times).
inline void gemma_declare_float(WeightLoader & L, GemmaLayerW & w, const std::string & p, bool qkv, bool gateup) {
#ifdef GGML_USE_SYCL
    if (qkv)
        w.Wqkv = L.fuse_gemm((p + ".attn_qkv.fused.weight").c_str(),
                             { p + ".attn_q.weight", p + ".attn_k.weight", p + ".attn_v.weight" });
    if (gateup)
        w.Wgu = L.fuse_gemm((p + ".ffn_gate_up.fused.weight").c_str(), { p + ".ffn_gate.weight", p + ".ffn_up.weight" });
#else
    if (qkv) {
        w.Wq = L.gemm("%s.attn_q.weight", p.c_str());
        w.Wk = L.gemm("%s.attn_k.weight", p.c_str());
        w.Wv = L.gemm("%s.attn_v.weight", p.c_str());
    }
    if (gateup) {
        w.Wgate = L.gemm("%s.ffn_gate.weight", p.c_str());
        w.Wup   = L.gemm("%s.ffn_up.weight",   p.c_str());
    }
#endif
}

/// FoldQuant q/k/v of a layer, `p` the tensor-name prefix. On SYCL, one fused
/// site (one GEMM and epilogue instead of three); elsewhere three sites.
inline void gemma_declare_fq_qkv(WeightLoader & L, GemmaLayerW & w, const FqModuleSpec & fq, const std::string & p,
                                 ggml_tensor * gamma, float eps) {
#ifdef GGML_USE_SYCL
    if (!fq.dequant) {
        w.fq_qkv = fq_declare_fused(L, fq, "qkv", false, p + ".attn_qkv.fused",
                                    { p + ".attn_q", p + ".attn_k", p + ".attn_v" }, gamma, eps);
        if (w.fq_qkv)
            return;
    }
#endif
    w.fq_q = fq_declare_linear(L, fq, "qkv", false, gamma, eps, "%s.attn_q", p.c_str());
    w.fq_k = fq_declare_linear(L, fq, "qkv", false, gamma, eps, "%s.attn_k", p.c_str());
    w.fq_v = fq_declare_linear(L, fq, "qkv", false, gamma, eps, "%s.attn_v", p.c_str());
}

/// SYCL with oneDNN: the fused gate/up site's GEGLU feeds the down act directly
/// (fq_glu_act), when both sites are FoldQuant. VLA_FQ_NO_GLU_ACT=1: apart.
inline void gemma_link_glu_act(GemmaLayerW & w) {
#if defined(GGML_USE_SYCL) && defined(VLA_FQ_SYCL_DNNL)
    // Needs oneDNN's int32 sums: off when VLA_FQ_SYCL_GEMM picks the native GEMMs.
    const char * e = std::getenv("VLA_FQ_NO_GLU_ACT");
    if (w.fq_gu && w.fq_down && !w.fq_down.act.has_gamma && w.fq_down.act.K == w.fq_gu.gemm.N / 2 &&
        w.fq_down.act.K <= 16384 && !(e && *e && *e != '0') && !std::getenv("VLA_FQ_SYCL_GEMM")) {
        w.glu_act.magic = FQ_GLU_ACT_MAGIC;
        w.glu_act.gemm  = &w.fq_gu.gemm;
        w.glu_act.act   = &w.fq_down.act;
    }
#else
    (void) w;
#endif
}

/// FoldQuant gate/up of a layer, `p` the tensor-name prefix. On SYCL, one fused
/// site whose epilogue applies the GEGLU, so neither the [2*inter, T] GEMM
/// output nor a separate GLU pass exists; elsewhere two sites as shipped.
inline void gemma_declare_fq_gateup(WeightLoader & L, GemmaLayerW & w, const FqModuleSpec & fq, const std::string & p,
                                    ggml_tensor * gamma, float eps) {
#ifdef GGML_USE_SYCL
    if (!fq.dequant && fq_glu_enabled()) {
        w.fq_gu = fq_declare_fused(L, fq, "gateup", false, p + ".ffn_gate_up.fused", { p + ".ffn_gate", p + ".ffn_up" },
                                   gamma, eps);
        if (w.fq_gu) {
            w.fq_gu.gemm.glu = 1;
            return;
        }
    }
#endif
    w.fq_gate = fq_declare_linear(L, fq, "gateup", false, gamma, eps, "%s.ffn_gate", p.c_str());
    w.fq_up   = fq_declare_linear(L, fq, "gateup", false, gamma, eps, "%s.ffn_up",   p.c_str());
}

struct GemmaStack {
    std::vector<GemmaLayerW> blk;
    ggml_tensor *            output_norm = nullptr;

    // fq: the LLM FoldQuant spec when the GGUF carries one (pi0.5), else null.
    void declare(WeightLoader & L, const char * prefix, int64_t layers, bool with_output_norm,
                 const FqModuleSpec * fq = nullptr, float rms_eps = 1e-6f) {
        blk.resize(layers);
        for (int64_t i=0; i<layers; ++i) {
            GemmaLayerW & w = blk[i];
            const long long ii = (long long) i;
            w.ln_in   = L.f32_gemma_norm("%s.blk.%lld.attn_norm.weight", prefix, ii);
            w.ln_post = L.f32_gemma_norm("%s.blk.%lld.ffn_norm.weight",  prefix, ii);
            if (fq) {
                gemma_declare_fq_qkv(L, w, *fq, std::string(prefix) + ".blk." + std::to_string(ii), w.ln_in, rms_eps);
                w.fq_o    = fq_declare_linear(L, *fq, "o",      false, nullptr,   0.0f,    "%s.blk.%lld.attn_o",   prefix, ii);
                gemma_declare_fq_gateup(L, w, *fq, std::string(prefix) + ".blk." + std::to_string(ii), w.ln_post, rms_eps);
                w.fq_down = fq_declare_linear(L, *fq, "down",   false, nullptr,   0.0f,    "%s.blk.%lld.ffn_down", prefix, ii);
                gemma_link_glu_act(w);
                if (!w.fq_q != !w.fq_k || !w.fq_q != !w.fq_v || !w.fq_gate != !w.fq_up)
                    L.fail("FoldQuant: a Gemma layer's q/k/v (and gate/up) must all be INT or all float");
            }
            const std::string p = std::string(prefix) + ".blk." + std::to_string(ii);
            gemma_declare_float(L, w, p, !w.fq_q && !w.fq_qkv, !w.fq_gate && !w.fq_gu);
            if (!w.fq_o)
                w.Wo  = L.gemm("%s.blk.%lld.attn_o.weight", prefix, ii);
            if (!w.fq_down)
                w.Wdown = L.gemm("%s.blk.%lld.ffn_down.weight", prefix, ii);
        }
        if (with_output_norm)
            output_norm = L.f32_gemma_norm("%s.output_norm.weight", prefix);
    }
};

inline ggml_tensor * gemma_attn(
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_norm, ggml_tensor * positions,
        const Config & cfg, int64_t seq,
        ggml_tensor * cached_K, ggml_tensor * cached_V, ggml_tensor * mask,
        ggml_tensor ** k_out, ggml_tensor ** v_out, ggml_type at, bool flash,
        ggml_tensor * fq_x = nullptr, ggml_tensor * fq_res = nullptr,
        ggml_tensor * cached_K16 = nullptr, ggml_tensor * cached_V16 = nullptr,
        ggml_tensor * fq_xq = nullptr, ggml_tensor * fq_gate = nullptr) {
    // fq_xq: the q/k/v act blob already made (fq_act_mod); fq_gate: the gate of a
    // gated residual fq_res + o * gate, fused into the o epilogue on SYCL.
    // FoldQuant: q/k/v share one act node over fq_x (the raw residual stream when
    // the RMSNorm and its folded gamma ride in the act node, the adaRMS output
    // otherwise), and fq_res, when given, rides in the o GEMM epilogue. A float
    // layer ignores both.
    const int64_t hd  = cfg.head_dim;
    const int64_t nq  = cfg.n_q_heads;
    const int64_t nkv = cfg.n_kv_heads;
    const int64_t qf  = nq * hd;

    // Q/K/V land in F32: RoPE, the KV cache the suffix passes re-read, and the
    // score/softmax core all stay full precision.
    if (w.rope_attn.magic == FQ_ROPE_ATTN_MAGIC && w.fq_qkv && w.fq_o && flash && cached_K16 && cached_V16 && !mask &&
        !k_out && !v_out && (fq_xq || fq_x) && seq * nq > 192) {
        // One node for q/k/v, RoPE and the attention (src/layers/fq_rope_attn.h).
        ggml_tensor * fa = fq_rope_attn(ctx, &w.rope_attn, w.fq_qkv, fq_xq ? fq_xq : fq_act(ctx, w.fq_qkv, fq_x),
                                        positions, cached_K16, cached_V16);
        return fq_linear(ctx, w.fq_o, ggml_reshape_2d(ctx, fa, qf, seq), fq_res, fq_gate);
    }
    if (w.rope_pack.magic == FQ_ROPE_PACK_MAGIC && w.fq_qkv && flash && !mask && !cached_K && !cached_V && nkv == 1 &&
        (fq_xq || fq_x) && fmha_kv_enabled()) {
        ggml_tensor * pk = fq_rope_pack(ctx, &w.rope_pack, w.fq_qkv, fq_xq ? fq_xq : fq_act(ctx, w.fq_qkv, fq_x), positions);
        const size_t  rs = pk->nb[1];   // one hd row
        ggml_tensor * Q  = ggml_view_3d(ctx, pk, hd, seq, nq, rs, rs * seq, 0);
        ggml_tensor * K  = ggml_view_3d(ctx, pk, hd, seq, 1,  rs, rs * seq, rs * seq * nq);
        ggml_tensor * V  = ggml_view_3d(ctx, pk, hd, seq, 1,  rs, rs * seq, rs * seq * (nq + 1));
        if (k_out) *k_out = K;
        if (v_out) *v_out = V;
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, 1.f/std::sqrt((float) hd), 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        return fq_linear(ctx, w.fq_o, ggml_reshape_2d(ctx, fa, qf, seq), fq_res, fq_gate);
    }
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *qkv_fq = nullptr;
    if (w.fq_qkv) {
        qkv_fq = fq_gemm(ctx, w.fq_qkv, fq_xq ? fq_xq : fq_act(ctx, w.fq_qkv, fq_x));
    } else if (w.fq_q) {
        ggml_tensor * xq = fq_act(ctx, w.fq_q, fq_x);
        q = fq_gemm(ctx, w.fq_q, xq);
        k = fq_gemm(ctx, w.fq_k, xq);
        v = fq_gemm(ctx, w.fq_v, xq);
    } else if (!w.Wqkv) {
        q = as_type(ctx, mm_act(ctx, w.Wq, x_norm, at), GGML_TYPE_F32);
        k = as_type(ctx, mm_act(ctx, w.Wk, x_norm, at), GGML_TYPE_F32);
        v = as_type(ctx, mm_act(ctx, w.Wv, x_norm, at), GGML_TYPE_F32);
    }

    ggml_tensor *q_h, *k_h, *v_h;
    if (w.Wqkv || qkv_fq) {
        // One GEMM; q, k and v are row ranges of each token's output.
        ggml_tensor * qkv = qkv_fq ? qkv_fq : as_type(ctx, mm_act(ctx, w.Wqkv, x_norm, at), GGML_TYPE_F32);
        const size_t es = ggml_element_size(qkv);
        q_h = ggml_view_3d(ctx, qkv, hd, nq,  seq, hd*es, qkv->nb[1], 0);
        k_h = ggml_view_3d(ctx, qkv, hd, nkv, seq, hd*es, qkv->nb[1], (size_t) qf*es);
        v_h = ggml_view_3d(ctx, qkv, hd, nkv, seq, hd*es, qkv->nb[1], (size_t) (qf + nkv*hd)*es);
    } else {
        q_h = ggml_reshape_3d(ctx, q, hd, nq,  seq);
        k_h = ggml_reshape_3d(ctx, k, hd, nkv, seq);
        v_h = ggml_reshape_3d(ctx, v, hd, nkv, seq);
    }

    auto rope_call = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, positions, nullptr,
                             (int) hd, GGML_ROPE_TYPE_NEOX, 0,
                             cfg.rope_freq_base, 1.f, 0.f, 1.f, 32.f, 1.f);
    };
    ggml_tensor * q_rope = rope_call(q_h);
    ggml_tensor * k_rope = rope_call(k_h);

    if (k_out)
        *k_out = k_rope;
    if (v_out)
        *v_out = v_h;

    const float scale = 1.f/std::sqrt((float) hd);
    ggml_tensor * att_pre;
    if (flash && cached_K16 && cached_V16 && !mask) {
        // Cached prefix (F16, cast once per layer) + this step's keys, without
        // concatenating them: src/layers/fmha_kv.h.
        ggml_tensor * fa = fmha_kv(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3), cached_K16, cached_V16,
                                   ggml_permute(ctx, k_rope, 0, 2, 1, 3), ggml_permute(ctx, v_h, 0, 2, 1, 3), scale);
        att_pre = ggml_reshape_2d(ctx, fa, qf, seq);
        if (w.fq_o)
            return fq_linear(ctx, w.fq_o, att_pre, fq_res, fq_gate);
        return mm_act(ctx, w.Wo, as_type(ctx, att_pre, at), at);
    }

    ggml_tensor * K_full = k_rope;
    ggml_tensor * V_full = v_h;
    if (cached_K && cached_V) {
        K_full = ggml_concat(ctx, cached_K, k_rope, 2);
        V_full = ggml_concat(ctx, cached_V, v_h,    2);
    }

    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3));
    if (flash) {
        ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, V_full, 0, 2, 1, 3));
        // ggml_flash_attn_ext asserts an F16 mask. The mask holds only 0 and
        // -inf, both exactly representable in F16, so the cast is lossless.
        ggml_tensor * mask_f16 = mask ? ggml_cast(ctx, mask, GGML_TYPE_F16) : nullptr;
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, fa_kv(ctx, K), fa_kv(ctx, V), mask_f16, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        att_pre = ggml_reshape_2d(ctx, fa, qf, seq);
    } else {
        ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));
        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        ggml_tensor * attn = ggml_soft_max_ext(ctx, kq, mask, scale, 0.f);
        ggml_tensor * kqv  = ggml_mul_mat(ctx, V, attn);
        att_pre = ggml_reshape_2d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq);
    }
    if (w.fq_o)
        return fq_linear(ctx, w.fq_o, att_pre, fq_res);
    return mm_act(ctx, w.Wo, as_type(ctx, att_pre, at), at);
}

// fq_x / fq_res: as in gemma_attn, for the gate/up act node and the down epilogue.
inline ggml_tensor * gemma_mlp(ggml_context * ctx, const GemmaLayerW & w, ggml_tensor * x_norm, ggml_type at,
                               ggml_tensor * fq_x = nullptr, ggml_tensor * fq_res = nullptr,
                               ggml_tensor * fq_xq = nullptr, ggml_tensor * fq_gate = nullptr) {
    if (w.fq_gu && w.glu_act.magic == FQ_GLU_ACT_MAGIC) {   // GEGLU straight into the down act
        ggml_tensor * xq_down = fq_glu_act(ctx, &w.glu_act, w.fq_gu, w.fq_down, fq_xq ? fq_xq : fq_act(ctx, w.fq_gu, fq_x));
        return fq_gemm(ctx, w.fq_down, xq_down, fq_res, fq_gate);
    }
    if (w.fq_gu) {   // one FoldQuant GEMM, GEGLU in its epilogue
        ggml_tensor * inter = fq_gemm(ctx, w.fq_gu, fq_xq ? fq_xq : fq_act(ctx, w.fq_gu, fq_x));
        if (w.fq_down)
            return fq_linear(ctx, w.fq_down, inter, fq_res, fq_gate);
        return mm_act(ctx, w.Wdown, inter, at);
    }
    if (w.Wgu) {   // one GEMM; ggml_geglu splits its rows into gate | up
        ggml_tensor * inter = ggml_geglu(ctx, mm_act(ctx, w.Wgu, x_norm, at));
        if (w.fq_down)
            return fq_linear(ctx, w.fq_down, as_type(ctx, inter, GGML_TYPE_F32), fq_res);
        return mm_act(ctx, w.Wdown, inter, at);
    }
    ggml_tensor *gate, *up;
    if (w.fq_gate) {
        ggml_tensor * xq = fq_act(ctx, w.fq_gate, fq_x);
        gate = fq_gemm(ctx, w.fq_gate, xq);
        up   = fq_gemm(ctx, w.fq_up,   xq);
    } else {
        gate = mm_act(ctx, w.Wgate, x_norm, at);
        up   = mm_act(ctx, w.Wup,   x_norm, at);
    }
    ggml_tensor * inter = geglu(ctx, gate, up);
    if (w.fq_down)
        return fq_linear(ctx, w.fq_down, as_type(ctx, inter, GGML_TYPE_F32), fq_res);
    return mm_act(ctx, w.Wdown, inter, at);
}

inline ggml_tensor * gemma_layer(
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions,
        const Config & cfg, int64_t seq,
        ggml_tensor * cached_K, ggml_tensor * cached_V, ggml_tensor * mask,
        ggml_tensor ** k_out, ggml_tensor ** v_out,
        ggml_type at = GGML_TYPE_F32, bool flash = false) {
    // FoldQuant (pi0.5 prefix): the RMSNorm and its folded gamma ride in the
    // q/k/v and gate/up act nodes, the residual adds in the o / down epilogues.
    ggml_tensor * h1 = w.fq_o
        ? gemma_attn(ctx, w, (w.fq_q || w.fq_qkv) ? nullptr : rms_norm(ctx, x_in, w.ln_in, cfg.rms_eps), positions, cfg, seq,
                     cached_K, cached_V, mask, k_out, v_out, at, flash, x_in, x_in)
        : ggml_add(ctx, x_in,
            gemma_attn(ctx, w, (w.fq_q || w.fq_qkv) ? nullptr : rms_norm(ctx, x_in, w.ln_in, cfg.rms_eps), positions, cfg, seq,
                       cached_K, cached_V, mask, k_out, v_out, at, flash, x_in));
    ggml_tensor * x_norm_mlp = (w.fq_gate || w.fq_gu) ? nullptr : rms_norm(ctx, h1, w.ln_post, cfg.rms_eps);
    return w.fq_down ? gemma_mlp(ctx, w, x_norm_mlp, at, h1, h1)
                     : ggml_add(ctx, h1, gemma_mlp(ctx, w, x_norm_mlp, at, h1));
}

inline std::string pi_key(const gguf_reader & g, const char * s) {
    return std::string(g.arch) + "." + s;
}

// PaliGemma's SigLIP-So400m/14 tower and projector, bundled in the ckpt GGUF.
struct PaliVision {
    int64_t       layers = 27, image_size = 224, patch_size = 14, n_tokens = 256;
    SigLipTower   vit;
    ggml_tensor * proj_w = nullptr, * proj_b = nullptr;

    bool load(const gguf_reader & g, int64_t n_img) {
        EncCfg & c = vit.enc.cfg;
        c.hidden = 1152;
        c.heads  = 16;
        auto u = [&](const char * s, int64_t & d) { if (g.has(pi_key(g, s).c_str())) d = (int64_t) g.u32(pi_key(g, s).c_str()); };
        u("vit_hidden", c.hidden);   u("vit_layers", layers);
        u("vit_heads",  c.heads);    u("image_size", image_size);
        u("patch_size", patch_size); u("n_img_tokens", n_tokens);
        if (g.has(pi_key(g, "vit_ln_eps").c_str()))
            c.ln_eps = g.f32(pi_key(g, "vit_ln_eps").c_str());
        if (patch_size <= 0 || c.heads <= 0 || c.hidden % c.heads || image_size % patch_size) {
            std::fprintf(stderr, "vla(%s): bad vit geometry (image %lld patch %lld hidden %lld heads %lld)\n",
                         g.arch, (long long) image_size, (long long) patch_size,
                         (long long) c.hidden, (long long) c.heads);
            return false;
        }
        const int64_t grid = image_size/patch_size;
        if (grid * grid != n_tokens || n_tokens != n_img) {
            std::fprintf(stderr, "vla(%s): vit geometry mismatch (grid^2=%lld n_img_tokens=%lld cfg.n_img=%lld)\n",
                         g.arch, (long long) (grid * grid), (long long) n_tokens, (long long) n_img);
            return false;
        }
        c.head_dim = c.hidden/c.heads;
        return true;
    }

    void declare(WeightLoader & L) {
        vit.declare(L, "vit", layers);
        proj_w = L.gemm   ("mm.proj.weight");
        proj_b = L.opt_f32("mm.proj.bias");
    }
};

inline bool load_pi_config(gguf_reader & g, const std::string & path, int64_t n_state, Config & cfg) {
    if (path.size() < 5 || path.compare(path.size()-5, 5, ".gguf") != 0) {
        std::fprintf(stderr, "vla(%s): ckpt must be a GGUF produced by scripts/convert_%s_to_gguf.py (got '%s')\n",
                     g.arch, g.arch, path.c_str());
        return false;
    }
    if (!g.open(path))
        return false;

    const std::string ak = pi_key(g, "architecture");
    if (g.str(ak.c_str()) != g.arch) {
        std::fprintf(stderr, "vla(%s): '%s' is not a %s GGUF (%s missing/wrong)\n", g.arch, path.c_str(), g.arch, ak.c_str());
        return false;
    }
    for (const char * s : {"hidden", "intermediate", "n_q_heads", "n_kv_heads", "head_dim", "n_layers",
                           "expert_h", "expert_inter", "chunk_size", "num_steps", "max_state_dim",
                           "max_action_dim", "real_state_dim", "real_action_dim", "tokenizer_max_length",
                           "min_period", "max_period"}) {
        const std::string key = pi_key(g, s);
        if (!g.has(key.c_str())) {
            std::fprintf(stderr, "vla(%s): gguf missing key %s\n", g.arch, key.c_str());
            return false;
        }
    }
    cfg = Config{};
    cfg.hidden          = g.u32(pi_key(g, "hidden").c_str());
    cfg.intermediate    = g.u32(pi_key(g, "intermediate").c_str());
    cfg.n_q_heads       = g.u32(pi_key(g, "n_q_heads").c_str());
    cfg.n_kv_heads      = g.u32(pi_key(g, "n_kv_heads").c_str());
    cfg.head_dim        = g.u32(pi_key(g, "head_dim").c_str());
    cfg.n_layers        = g.u32(pi_key(g, "n_layers").c_str());
    cfg.expert_h        = g.u32(pi_key(g, "expert_h").c_str());
    cfg.expert_inter    = g.u32(pi_key(g, "expert_inter").c_str());
    cfg.n_suffix        = g.u32(pi_key(g, "chunk_size").c_str());
    cfg.num_steps       = g.u32(pi_key(g, "num_steps").c_str());
    cfg.max_state_dim   = g.u32(pi_key(g, "max_state_dim").c_str());
    cfg.max_action_dim  = g.u32(pi_key(g, "max_action_dim").c_str());
    cfg.real_state_dim  = g.u32(pi_key(g, "real_state_dim").c_str());
    cfg.real_action_dim = g.u32(pi_key(g, "real_action_dim").c_str());
    cfg.n_lang          = g.u32(pi_key(g, "tokenizer_max_length").c_str());
    cfg.min_period      = g.f64(pi_key(g, "min_period").c_str());
    cfg.max_period      = g.f64(pi_key(g, "max_period").c_str());
    if (cfg.num_steps < 1 || cfg.num_steps > 1000) {
        std::fprintf(stderr, "vla(%s): num_steps %d out of range [1, 1000]\n", g.arch, cfg.num_steps);
        return false;
    }

    cfg.n_state         = n_state;
    cfg.n_img           = 256;
    cfg.q_full_dim      = cfg.n_q_heads  * cfg.head_dim;
    cfg.kv_full_dim     = cfg.n_kv_heads*cfg.head_dim;
    cfg.self_attn_every_n = 0;
    cfg.rms_eps         = g.has(pi_key(g, "rms_norm_eps").c_str()) ? g.f32(pi_key(g, "rms_norm_eps").c_str()) : 1e-6f;
    cfg.norm_eps        = g.has(pi_key(g, "norm_eps").c_str())     ? g.f32(pi_key(g, "norm_eps").c_str())     : 1e-8f;
    cfg.rope_mode       = GGML_ROPE_TYPE_NEOX;
    cfg.rope_n_dims     = (int) cfg.head_dim;
    cfg.rope_freq_base  = g.has(pi_key(g, "rope_theta").c_str()) ? (float) g.f64(pi_key(g, "rope_theta").c_str()) : 10000.f;
    cfg.n_prefix        = 0;
    cfg.n_full          = 0;
    return true;
}

// Absent stats are a valid checkpoint: identity, carry on. Stats that are
// present but unreadable are not - falling back to identity there hands back
// un-denormalised actions with nothing in the log. Note stderr, not stdout:
// stdout is the action stream tests/predict_check.cpp diffs.
inline bool read_pi_stat(gguf_reader & g, const char * name, std::vector<float> & dst) {
    const ggml_tensor * t = g.meta(name);
    if (!t) {
        std::fprintf(stderr, "vla(%s): %s missing - identity\n", g.arch, name);
        return true;
    }
    if (t->ne[0] != (int64_t) dst.size()) {
        std::fprintf(stderr, "vla(%s): %s is %lld wide, expected %zu\n",
                     g.arch, name, (long long) t->ne[0], dst.size());
        return false;
    }
    if (!g.read_raw(name, dst.data(), dst.size()*sizeof(float))) {
        std::fprintf(stderr, "vla(%s): %s read failed\n", g.arch, name);
        return false;
    }
    return true;
}

}
