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
};

struct GemmaStack {
    std::vector<GemmaLayerW> blk;
    ggml_tensor *            output_norm = nullptr;

    void declare(WeightLoader & L, const char * prefix, int64_t layers, bool with_output_norm) {
        blk.resize(layers);
        for (int64_t i=0; i<layers; ++i) {
            GemmaLayerW & w = blk[i];
            w.ln_in   = L.f32_gemma_norm("%s.blk.%lld.attn_norm.weight", prefix, (long long)i);
            w.Wq      = L.gemm          ("%s.blk.%lld.attn_q.weight",    prefix, (long long)i);
            w.Wk      = L.gemm          ("%s.blk.%lld.attn_k.weight",    prefix, (long long)i);
            w.Wv      = L.gemm          ("%s.blk.%lld.attn_v.weight",    prefix, (long long)i);
            w.Wo      = L.gemm          ("%s.blk.%lld.attn_o.weight",    prefix, (long long)i);
            w.ln_post = L.f32_gemma_norm("%s.blk.%lld.ffn_norm.weight",  prefix, (long long)i);
            w.Wgate   = L.gemm          ("%s.blk.%lld.ffn_gate.weight",  prefix, (long long)i);
            w.Wup     = L.gemm          ("%s.blk.%lld.ffn_up.weight",    prefix, (long long)i);
            w.Wdown   = L.gemm          ("%s.blk.%lld.ffn_down.weight",  prefix, (long long)i);
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
        ggml_tensor ** k_out, ggml_tensor ** v_out, ggml_type at, bool flash) {
    const int64_t hd  = cfg.head_dim;
    const int64_t nq  = cfg.n_q_heads;
    const int64_t nkv = cfg.n_kv_heads;
    const int64_t qf  = nq * hd;

    // Q/K/V land in F32: RoPE, the KV cache the suffix passes re-read, and the
    // score/softmax core all stay full precision.
    ggml_tensor * q = as_type(ctx, mm_act(ctx, w.Wq, x_norm, at), GGML_TYPE_F32);
    ggml_tensor * k = as_type(ctx, mm_act(ctx, w.Wk, x_norm, at), GGML_TYPE_F32);
    ggml_tensor * v = as_type(ctx, mm_act(ctx, w.Wv, x_norm, at), GGML_TYPE_F32);

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q, hd, nq,  seq);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k, hd, nkv, seq);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v, hd, nkv, seq);

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

    ggml_tensor * K_full = k_rope;
    ggml_tensor * V_full = v_h;
    if (cached_K && cached_V) {
        K_full = ggml_concat(ctx, cached_K, k_rope, 2);
        V_full = ggml_concat(ctx, cached_V, v_h,    2);
    }

    const float scale = 1.f/std::sqrt((float) hd);
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3));
    ggml_tensor * att_pre;
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
    return mm_act(ctx, w.Wo, as_type(ctx, att_pre, at), at);
}

inline ggml_tensor * gemma_mlp(ggml_context * ctx, const GemmaLayerW & w, ggml_tensor * x_norm, ggml_type at) {
    ggml_tensor * gate = mm_act(ctx, w.Wgate, x_norm, at);
    ggml_tensor * up   = mm_act(ctx, w.Wup,   x_norm, at);
    return mm_act(ctx, w.Wdown, ggml_mul(ctx, gelu(ctx, gate), up), at);
}

inline ggml_tensor * gemma_layer(
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions,
        const Config & cfg, int64_t seq,
        ggml_tensor * cached_K, ggml_tensor * cached_V, ggml_tensor * mask,
        ggml_tensor ** k_out, ggml_tensor ** v_out,
        ggml_type at = GGML_TYPE_F32, bool flash = false) {
    ggml_tensor * h1 = ggml_add(ctx, x_in,
        gemma_attn(ctx, w, rms_norm(ctx, x_in, w.ln_in, cfg.rms_eps), positions, cfg, seq,
                   cached_K, cached_V, mask, k_out, v_out, at, flash));
    return ggml_add(ctx, h1, gemma_mlp(ctx, w, rms_norm(ctx, h1, w.ln_post, cfg.rms_eps), at));
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
