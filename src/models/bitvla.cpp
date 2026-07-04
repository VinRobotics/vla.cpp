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

#include "arch.h"
#include "model.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "gguf.h"
#include "models/gguf_reader.h"

#ifdef VLA_BITVLA_CUDA_KERNELS
#include "kernels/bitvla/bitvla_lm_cuda.h"
#include "kernels/bitvla/bitvla_vit_cuda.h"
#include "kernels/bitvla/bitvla_fp32head_cuda.h"
#ifdef __GLIBC__
#  include <malloc.h>
#endif
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vla {
namespace {


struct VitLayerW {
    ggml_tensor *ln1w, *ln1b, *ln2w, *ln2b;
    ggml_tensor *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;
    ggml_tensor *Wfc1, *bfc1, *Wfc2, *bfc2;
};
struct LmLayerW {
    ggml_tensor *attn_norm, *attn_sub_norm, *ffn_norm, *ffn_sub_norm;
    ggml_tensor *Wq, *Wk, *Wv, *Wo;
    ggml_tensor *Wgate, *Wup, *Wdown;
    ggml_tensor *Wgate_up = nullptr;
};

void bitvla_act_quant_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ) {
    const int64_t cols = a->ne[0];
    const int64_t rows = ggml_nrows(a);
    const int64_t per  = (rows + nth - 1) / nth;
    const int64_t r0   = ith * per;
    const int64_t r1   = std::min(rows, r0 + per);
    const float * src = (const float *) a->data;
    float * out = (float *) dst->data;
    for (int64_t r = r0; r < r1; ++r) {
        const float * row_in = src + r * cols;
        float * row_out = out + r * cols;
        float amax = 0.0f;
        for (int64_t c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(row_in[c]));
        if (amax < 1e-5f) amax = 1e-5f;
        const float s     = 127.0f / amax;
        const float inv_s = 1.0f / s;
        for (int64_t c = 0; c < cols; ++c) {
            float q = std::nearbyintf(row_in[c] * s);
            if (q >  127.0f) q =  127.0f;
            if (q < -128.0f) q = -128.0f;
            row_out[c] = q * inv_s;
        }
    }
}

}

struct BitvlaModelArch : public ModelArchBase {
    BitvlaModelArch() : ModelArchBase(Arch::BITVLA) {}
    ~BitvlaModelArch() override;

    std::string           gguf_path;
    gguf_reader           emb_reader{"bitvla"};   // stays open for per-step token-embedding row fetches
    std::vector<float>    stop_embed;   // cached constant stop-token embedding row
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    int                   n_threads   = 4;
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;
    bool                  packed_int2 = false;

    int64_t vit_hidden = 1152, vit_layers = 26, vit_heads = 16, vit_head_dim = 72, vit_inter = 4304;
    int64_t image_size = 224, patch_size = 14, n_patches = 256;
    float   vit_ln_eps = 1e-6f;
    int64_t lm_hidden  = 2560, lm_layers = 30, lm_q = 20, lm_kv = 5, lm_head_dim = 128, lm_inter = 6912;
    float   lm_rms_eps = 1e-5f, lm_rope_base = 500000.0f;
    int64_t vocab_size = 128264, lm_max_pos = 4096;
    int64_t num_actions_chunk = 8, action_dim = 7, proprio_dim = 8;
    float   ah_ln_eps = 1e-5f;
    int32_t image_token_id = 128010, proprio_pad_id = 128011, action_begin_id = 128012, stop_id = 128001;

    std::vector<float> q01, q99;
    std::vector<uint8_t> unnorm_mask;

    ggml_tensor *vit_patch_w = nullptr, *vit_patch_b = nullptr, *vit_pos = nullptr;
    std::vector<VitLayerW> vit;
    ggml_tensor *mm_l1_w = nullptr, *mm_l1_b = nullptr, *mm_l2_w = nullptr, *mm_l2_b = nullptr;
    ggml_tensor *pp_fc1_w = nullptr, *pp_fc1_b = nullptr, *pp_fc2_w = nullptr, *pp_fc2_b = nullptr;
    ggml_tensor *embed_tokens = nullptr, *lm_output_norm = nullptr;
    std::vector<LmLayerW> lm;
    ggml_tensor *ah_ln1_w = nullptr, *ah_ln1_b = nullptr, *ah_fc1_w = nullptr, *ah_fc1_b = nullptr;
    ggml_tensor *ah_b0_ln_w = nullptr, *ah_b0_ln_b = nullptr, *ah_b0_w = nullptr, *ah_b0_b = nullptr;
    ggml_tensor *ah_b1_ln_w = nullptr, *ah_b1_ln_b = nullptr, *ah_b1_w = nullptr, *ah_b1_b = nullptr;
    ggml_tensor *ah_ln2_w = nullptr, *ah_ln2_b = nullptr, *ah_fc2_w = nullptr, *ah_fc2_b = nullptr;

#ifdef VLA_BITVLA_CUDA_KERNELS

    bitvla_lm_cuda_ctx*  lm_cuda_ctx  = nullptr;
    bitvla_vit_cuda_ctx* vit_cuda_ctx = nullptr;
    bitvla_fp32head_cuda_ctx* fp32head_cuda_ctx = nullptr;
    bool                 cuda_fp32head_ready = false;
    std::vector<void*>   cuda_devptrs;
    bool                 cuda_lm_ready  = false;
    bool                 cuda_vit_ready = false;

    std::vector<void*>   cpu_kept_ptrs;

    __nv_bfloat16* d_inputs_embeds = nullptr;
    __nv_bfloat16* d_last_hidden   = nullptr;
    __nv_bfloat16* d_action_hidden = nullptr;
    int32_t*       d_action_ids    = nullptr;
    int            cuda_max_seq    = 0;

    __nv_bfloat16* d_vit_patches    = nullptr;
    __nv_bfloat16* d_vit_img_embeds = nullptr;
#endif

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

ggml_tensor * act_quant(ggml_context * C, ggml_tensor * x) {
    return ggml_map_custom1(C, x, bitvla_act_quant_op, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor * bit_linear(ggml_context * C, ggml_tensor * W, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(C, W, act_quant(C, x));
    return b ? ggml_add(C, y, b) : y;
}
ggml_tensor * layernorm(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    return ggml_add(C, ggml_mul(C, ggml_norm(C, x, eps), w), b);
}
ggml_tensor * rmsnorm(ggml_context * C, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(C, ggml_rms_norm(C, x, eps), w);
}

ggml_tensor * build_vit_layer(ggml_context * C, const VitLayerW & w, ggml_tensor * x,
                               int64_t seq, int64_t heads, int64_t head_dim, int64_t hidden, float ln_eps) {
    const float scale = 1.0f / std::sqrt((float) head_dim);
    ggml_tensor * x1 = layernorm(C, x, w.ln1w, w.ln1b, ln_eps);
    ggml_tensor * q  = bit_linear(C, w.Wq, w.bq, x1);
    ggml_tensor * k  = bit_linear(C, w.Wk, w.bk, x1);
    ggml_tensor * v  = bit_linear(C, w.Wv, w.bv, x1);
    ggml_tensor * Q  = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, head_dim, heads, seq), 0, 2, 1, 3));
    ggml_tensor * K  = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, head_dim, heads, seq), 0, 2, 1, 3));
    ggml_tensor * V  = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, head_dim, heads, seq), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * att= ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * y  = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, att), 0, 2, 1, 3)), hidden, seq);
    ggml_tensor * o  = bit_linear(C, w.Wo, w.bo, y);
    ggml_tensor * h1 = ggml_add(C, x, o);
    ggml_tensor * x2 = layernorm(C, h1, w.ln2w, w.ln2b, ln_eps);
    ggml_tensor * f1 = ggml_gelu(C, bit_linear(C, w.Wfc1, w.bfc1, x2));
    ggml_tensor * f2 = bit_linear(C, w.Wfc2, w.bfc2, f1);
    return ggml_add(C, h1, f2);
}

ggml_tensor * build_lm_layer(ggml_context * C, const BitvlaModelArch & m, const LmLayerW & w,
                              ggml_tensor * h, ggml_tensor * positions, int64_t seq) {
    const int64_t hd = m.lm_head_dim, n_q = m.lm_q, n_kv = m.lm_kv, hq = n_q * hd;
    const float scale = 1.0f / std::sqrt((float) hd);

    ggml_tensor * hn = rmsnorm(C, h, w.attn_norm, m.lm_rms_eps);
    ggml_tensor * qp = bit_linear(C, w.Wq, nullptr, hn);
    ggml_tensor * kp = bit_linear(C, w.Wk, nullptr, hn);
    ggml_tensor * vp = bit_linear(C, w.Wv, nullptr, hn);
    ggml_tensor * q3 = ggml_reshape_3d(C, qp, hd, n_q, seq);
    ggml_tensor * k3 = ggml_reshape_3d(C, kp, hd, n_kv, seq);
    ggml_tensor * v3 = ggml_reshape_3d(C, vp, hd, n_kv, seq);
    ggml_tensor * qR = ggml_rope_ext(C, q3, positions, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, m.lm_rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * kR = ggml_rope_ext(C, k3, positions, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, m.lm_rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * Q  = ggml_cont(C, ggml_permute(C, qR, 0, 2, 1, 3));
    ggml_tensor * K  = ggml_cont(C, ggml_permute(C, kR, 0, 2, 1, 3));
    ggml_tensor * V  = ggml_cont(C, ggml_permute(C, v3, 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * att= ggml_soft_max_ext(C, kq,  nullptr, scale, 0.0f);
    ggml_tensor * kqv= ggml_mul_mat(C, V, att);
    ggml_tensor * mer= ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, kqv, 0, 2, 1, 3)), hq, seq);
    ggml_tensor * sub= rmsnorm(C, mer, w.attn_sub_norm, m.lm_rms_eps);
    ggml_tensor * o  = bit_linear(C, w.Wo, nullptr, sub);
    ggml_tensor * h1 = ggml_add(C, h, o);

    ggml_tensor * h2 = rmsnorm(C, h1, w.ffn_norm, m.lm_rms_eps);
    ggml_tensor * g  = bit_linear(C, w.Wgate, nullptr, h2);
    ggml_tensor * u  = bit_linear(C, w.Wup,   nullptr, h2);
    ggml_tensor * gsq= ggml_sqr(C, ggml_relu(C, g));
    ggml_tensor * gu = ggml_mul(C, gsq, u);
    ggml_tensor * fsub= rmsnorm(C, gu, w.ffn_sub_norm, m.lm_rms_eps);
    ggml_tensor * dn = bit_linear(C, w.Wdown, nullptr, fsub);
    return ggml_add(C, h1, dn);
}

bool parse_stats_json(const std::string & js, const char * env_key,
                       int64_t want_dim,
                       std::vector<float> & q01, std::vector<float> & q99,
                       std::vector<uint8_t> & mask, std::string & resolved_key) {
    auto find_obj = [&](const std::string & where, const std::string & key) -> std::string {
        const std::string q = std::string("\"") + key + "\"";
        size_t p = where.find(q); if (p == std::string::npos) return "";
        p = where.find(':', p + q.size()); if (p == std::string::npos) return "";
        size_t s = where.find('{', p); if (s == std::string::npos) return "";
        int depth = 1; size_t i = s + 1;
        while (i < where.size() && depth > 0) {
            if (where[i] == '{') depth++;
            else if (where[i] == '}') depth--;
            if (depth == 0) break;
            i++;
        }
        return where.substr(s, i - s + 1);
    };
    auto parse_array_floats = [](const std::string & body, const char * key, std::vector<float> & out) {
        const std::string q = std::string("\"") + key + "\"";
        size_t p = body.find(q); if (p == std::string::npos) return false;
        p = body.find('[', p); if (p == std::string::npos) return false;
        size_t e = body.find(']', p); if (e == std::string::npos) return false;
        std::string inner = body.substr(p + 1, e - p - 1);
        out.clear();
        const char * s = inner.c_str();
        while (*s) {
            while (*s == ' ' || *s == '\t' || *s == '\n' || *s == ',') s++;
            if (!*s) break;
            char * end = nullptr;
            float v = std::strtof(s, &end);
            if (end == s) break;
            out.push_back(v);
            s = end;
        }
        return true;
    };
    auto parse_array_bools = [](const std::string & body, const char * key, std::vector<uint8_t> & out) {
        const std::string q = std::string("\"") + key + "\"";
        size_t p = body.find(q); if (p == std::string::npos) return false;
        p = body.find('[', p); if (p == std::string::npos) return false;
        size_t e = body.find(']', p); if (e == std::string::npos) return false;
        std::string inner = body.substr(p + 1, e - p - 1);
        out.clear();
        size_t i = 0;
        while (i < inner.size()) {
            while (i < inner.size() && (inner[i] == ' ' || inner[i] == ',' || inner[i] == '\n' || inner[i] == '\t')) i++;
            if (i >= inner.size()) break;
            if (inner.compare(i, 4, "true")  == 0) { out.push_back(1); i += 4; }
            else if (inner.compare(i, 5, "false") == 0) { out.push_back(0); i += 5; }
            else i++;
        }
        return true;
    };

    std::string suite;
    if (env_key && *env_key) {
        suite = env_key;
    } else {

        size_t p = js.find('"');
        if (p == std::string::npos) return false;
        size_t q = js.find('"', p + 1);
        if (q == std::string::npos) return false;
        suite = js.substr(p + 1, q - p - 1);
    }
    resolved_key = suite;

    const std::string suite_obj = find_obj(js, suite);
    if (suite_obj.empty()) { std::fprintf(stderr, "vla(bitvla): suite key '%s' not found in statistics_json\n", suite.c_str()); return false; }
    const std::string action_obj = find_obj(suite_obj, "action");
    if (action_obj.empty()) { std::fprintf(stderr, "vla(bitvla): no action stats under '%s'\n", suite.c_str()); return false; }

    if (!parse_array_floats(action_obj, "q01", q01)) return false;
    if (!parse_array_floats(action_obj, "q99", q99)) return false;
    if (!parse_array_bools (action_obj, "mask", mask)) {

        mask.assign(q01.size(), 1);
    }
    if ((int64_t) q01.size() != want_dim || (int64_t) q99.size() != want_dim || (int64_t) mask.size() != want_dim) {
        std::fprintf(stderr, "vla(bitvla): action stats dim mismatch (q01=%zu q99=%zu mask=%zu, want %lld)\n",
                     q01.size(), q99.size(), mask.size(), (long long) want_dim);
        return false;
    }
    return true;
}

bool load_config(const gguf_reader & g, BitvlaModelArch & m, Config & cfg) {
    auto U = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto I = [&](const char * k, int32_t & dst) { if (g.has(k)) dst = (int32_t) g.u32(k); };
    U("bitvla.vit.hidden",      m.vit_hidden);     U("bitvla.vit.layers",  m.vit_layers);
    U("bitvla.vit.heads",       m.vit_heads);      U("bitvla.vit.head_dim",m.vit_head_dim);
    U("bitvla.vit.inter",       m.vit_inter);
    U("bitvla.vit.image_size",  m.image_size);     U("bitvla.vit.patch_size", m.patch_size);
    U("bitvla.vit.n_patches",   m.n_patches);
    F("bitvla.vit.ln_eps",      m.vit_ln_eps);
    U("bitvla.lm.hidden",       m.lm_hidden);      U("bitvla.lm.layers",   m.lm_layers);
    U("bitvla.lm.q_heads",      m.lm_q);           U("bitvla.lm.kv_heads", m.lm_kv);
    U("bitvla.lm.head_dim",     m.lm_head_dim);    U("bitvla.lm.inter",    m.lm_inter);
    F("bitvla.lm.rope_theta",   m.lm_rope_base);   F("bitvla.lm.rms_eps",  m.lm_rms_eps);
    U("bitvla.lm.max_pos",      m.lm_max_pos);     U("bitvla.lm.vocab_size", m.vocab_size);
    U("bitvla.action.num_actions_chunk", m.num_actions_chunk);
    U("bitvla.action.action_dim",        m.action_dim);
    U("bitvla.action.proprio_dim",       m.proprio_dim);
    F("bitvla.action.ln_eps",            m.ah_ln_eps);
    I("bitvla.tokens.image_id",          m.image_token_id);
    I("bitvla.tokens.proprio_id",        m.proprio_pad_id);
    I("bitvla.tokens.action_begin_id",   m.action_begin_id);
    I("bitvla.tokens.stop_id",           m.stop_id);
    m.packed_int2 = g.has("bitvla.quant.int2_packed") && g.u32("bitvla.quant.int2_packed") != 0;

    const std::string js = g.str("bitvla.statistics_json");
    if (js.empty()) {
        std::fprintf(stderr, "vla(bitvla): bitvla.statistics_json KV missing - un-normalization will pass-through\n");
        m.q01.assign(m.action_dim, 0.0f);
        m.q99.assign(m.action_dim, 0.0f);
        m.unnorm_mask.assign(m.action_dim, 0);
    } else {
        const char * env_suite = std::getenv("VLA_BITVLA_UNNORM_KEY");
        std::string resolved;
        if (!parse_stats_json(js, env_suite, m.action_dim, m.q01, m.q99, m.unnorm_mask, resolved)) {
            std::fprintf(stderr, "vla(bitvla): failed to parse statistics_json - un-normalization will pass-through\n");
            m.q01.assign(m.action_dim, 0.0f);
            m.q99.assign(m.action_dim, 0.0f);
            m.unnorm_mask.assign(m.action_dim, 0);
        } else {
            std::printf("vla(bitvla): un-normalisation key = '%s' (BOUNDS_Q99)\n", resolved.c_str());
        }
    }

    cfg = Config{};
    cfg.n_img = m.n_patches; cfg.n_lang = m.lm_max_pos; cfg.n_state = 1;
    cfg.n_suffix = m.num_actions_chunk; cfg.max_state_dim = m.proprio_dim; cfg.max_action_dim = m.action_dim;
    cfg.real_state_dim = m.proprio_dim; cfg.real_action_dim = m.action_dim;
    cfg.hidden = m.lm_hidden; cfg.n_q_heads = m.lm_q; cfg.n_kv_heads = m.lm_kv;
    cfg.head_dim = m.lm_head_dim; cfg.n_layers = m.lm_layers;
    cfg.num_steps = 0;
    cfg.rms_eps = m.lm_rms_eps;
    cfg.rope_n_dims = (int) m.lm_head_dim;
    cfg.rope_mode = GGML_ROPE_TYPE_NEOX;
    cfg.rope_freq_base = m.lm_rope_base;
    cfg.norm_eps = 1e-8f;
    return true;
}

}

#ifdef VLA_BITVLA_CUDA_KERNELS
namespace {

static void recover_ternary_and_scale(const float* W, int64_t n,
                                       std::vector<int8_t>& ternary, float& absmean) {
    // Per-tensor absmean scale (1/mean|W|), matching scripts/bitvla_int2_pack.py;
    // the int2-packed path bakes the same scale.
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += std::fabs((double) W[i]);
    float mean = n > 0 ? (float) (s / (double) n) : 0.0f;
    if (mean < 1e-5f) mean = 1e-5f;
    absmean = mean;
    const float inv = 1.0f / mean;
    ternary.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        float q = std::nearbyintf(W[i] * inv);
        if (q >  1.0f) q =  1.0f;
        if (q < -1.0f) q = -1.0f;
        ternary[i] = (int8_t) q;
    }
}

static std::vector<uint8_t> pack_ladder_int2(const int8_t* W, int64_t N, int64_t K) {
    constexpr int N_BLOCK = 16, K_BLOCK = 8, K_PER_LOOP = 16;
    constexpr int WMMA_K = 32, K_PER_ITER = K_PER_LOOP * K_BLOCK;
    const int64_t n_slots = N * K / 16;
    std::vector<uint8_t> out(N * K / 4, 0);
    for (int64_t s = 0; s < n_slots; ++s) {
        const int64_t slots_per_block = (N_BLOCK * K) / 16;
        const int64_t n_block  = s / slots_per_block;
        const int64_t in_block = s % slots_per_block;
        const int64_t k_0      = in_block / 128;
        const int64_t in_k0    = in_block % 128;
        const int64_t major_k  = in_k0 / 32;
        const int64_t in_major = in_k0 % 32;
        const int64_t y_half   = in_major / 16;
        const int64_t in_yhalf = in_major % 16;
        const int64_t sub_k    = in_yhalf / 8;
        const int64_t y_in_h   = in_yhalf % 8;
        const int64_t n_global = n_block * N_BLOCK + y_half * 8 + y_in_h;
        const int64_t k_sub    = k_0 * K_PER_ITER + major_k * WMMA_K + sub_k * K_PER_LOOP;
        for (int byte_i = 0; byte_i < 4; ++byte_i) {
            uint8_t b = 0;
            for (int j = 0; j < 4; ++j) {
                const int t = (int) W[n_global * K + (k_sub + byte_i + 4 * j)];
                const uint8_t enc = (uint8_t)(t + 2) & 0x3;
                b |= (enc << (2 * j));
            }
            out[s * 4 + byte_i] = b;
        }
    }
    return out;
}

static inline uint16_t f32_to_bf16_u16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static __nv_bfloat16* upload_bf16_from_f32(const float* h, size_t n, std::vector<void*>& out_ptrs) {
    std::vector<uint16_t> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = f32_to_bf16_u16(h[i]);
    __nv_bfloat16* d = nullptr;
    cudaMalloc(&d, n * sizeof(__nv_bfloat16));
    cudaMemcpy(d, tmp.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    out_ptrs.push_back(d);
    return d;
}
static int8_t* upload_int8(const uint8_t* h, size_t n, std::vector<void*>& out_ptrs) {
    int8_t* d = nullptr;
    cudaMalloc(&d, n);
    cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
    out_ptrs.push_back(d);
    return (int8_t*) d;
}
static float* upload_f32_scales(const float* sc, int n, std::vector<void*>& out_ptrs) {
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, sc, n * sizeof(float), cudaMemcpyHostToDevice);
    out_ptrs.push_back(d);
    return d;
}

static int8_t* pack_and_upload_one(const float* wptr, int64_t N, int64_t K,
                                    float& out_absmean, std::vector<void*>& out_ptrs) {
    std::vector<int8_t> ternary;
    recover_ternary_and_scale(wptr, N * K, ternary, out_absmean);
    auto packed = pack_ladder_int2(ternary.data(), N, K);
    return upload_int8(packed.data(), packed.size(), out_ptrs);
}

static int8_t* pack_and_upload_fused(const std::vector<const float*>& wptrs,
                                      const std::vector<int64_t>& Ns, int64_t K,
                                      std::vector<float>& out_scales,
                                      std::vector<void*>& out_ptrs) {
    int64_t N_total = 0;
    for (int64_t n : Ns) N_total += n;
    std::vector<int8_t> stacked(N_total * K);
    out_scales.clear();
    int64_t row_off = 0;
    for (size_t i = 0; i < wptrs.size(); ++i) {
        std::vector<int8_t> tern;
        float sc;
        recover_ternary_and_scale(wptrs[i], Ns[i] * K, tern, sc);
        std::memcpy(stacked.data() + row_off * K, tern.data(), (size_t) Ns[i] * K);
        out_scales.push_back(sc);
        row_off += Ns[i];
    }
    auto packed = pack_ladder_int2(stacked.data(), N_total, K);
    return upload_int8(packed.data(), packed.size(), out_ptrs);
}

}
#endif

BitvlaModelArch::~BitvlaModelArch() {
#ifdef VLA_BITVLA_CUDA_KERNELS
    if (lm_cuda_ctx)       bitvla_lm_cuda_free(lm_cuda_ctx);
    if (vit_cuda_ctx)      bitvla_vit_cuda_free(vit_cuda_ctx);
    if (fp32head_cuda_ctx) bitvla_fp32head_cuda_free(fp32head_cuda_ctx);
    for (void* p : cuda_devptrs) if (p) cudaFree(p);
    if (d_inputs_embeds)  cudaFree(d_inputs_embeds);
    if (d_last_hidden)    cudaFree(d_last_hidden);
    if (d_action_hidden)  cudaFree(d_action_hidden);
    if (d_action_ids)     cudaFree(d_action_ids);
    if (d_vit_patches)    cudaFree(d_vit_patches);
    if (d_vit_img_embeds) cudaFree(d_vit_img_embeds);
    for (void* p : cpu_kept_ptrs) if (p) std::free(p);
#endif
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> bitvla_create(const std::string& mmproj_path,
                                              const std::string& ckpt_path,
                                              const std::string& ) {
    if (!mmproj_path.empty())
        std::printf("vla(bitvla): note - mmproj '%s' is ignored (the BitSigLIP-L vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<BitvlaModelArch>();
    m->gguf_path   = ckpt_path;
    m->matmul_type = std::getenv("VLA_BITVLA_BF16_WEIGHTS") ? GGML_TYPE_BF16 : GGML_TYPE_F32;

    gguf_reader g("bitvla");
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has("bitvla.architecture") && !g.has("general.architecture")) {
        std::fprintf(stderr, "vla(bitvla): %s is not a bitvla GGUF\n", ckpt_path.c_str()); return nullptr;
    }
    if (!load_config(g, *m, m->cfg)) return nullptr;

    // Keep one reader open for the per-step token-embedding fetches (token_embd
    // stays on disk under int2 packing) and cache the constant stop-token row,
    // so predict() no longer re-opens and re-parses the GGUF twice per call.
    if (!m->emb_reader.open(ckpt_path)) return nullptr;
    m->stop_embed.resize((size_t) m->lm_hidden);
    {
        std::vector<int32_t> sid{ m->stop_id };
        if (!m->emb_reader.fetch_rows_f32("token_embd.weight", sid,
                                          m->stop_embed.data(), m->lm_hidden)) return nullptr;
    }
    std::printf("vla(bitvla): vit=BitSigLIP-L %lldd×%lldL×%lldh@%lld  ⇒ %lld patches  mm=%lld→%lld  "
                "lm=BitNet %lldd×%lldL (%lldq/%lldkv×%lld) rope=%g  chunk×dim=%lld×%lld  vocab=%lld  resident=%s\n",
                (long long) m->vit_hidden, (long long) m->vit_layers, (long long) m->vit_heads, (long long) m->image_size,
                (long long) m->n_patches, (long long) m->vit_hidden, (long long) m->lm_hidden,
                (long long) m->lm_hidden, (long long) m->lm_layers, (long long) m->lm_q, (long long) m->lm_kv, (long long) m->lm_head_dim,
                (double) m->lm_rope_base, (long long) m->num_actions_chunk, (long long) m->action_dim,
                (long long) m->vocab_size, m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    m->backend = ggml_backend_cpu_init();
    if (!m->backend) { std::fprintf(stderr, "vla(bitvla): ggml_backend_cpu_init failed\n"); return nullptr; }
    ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
    std::printf("vla(bitvla): ggml backend = CPU (%d threads) - CUDA LM module activates below if available\n", m->n_threads);

    ggml_init_params wp = {  (size_t) 32 * 1024 * 1024,  nullptr,  true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) { std::fprintf(stderr, "vla(bitvla): ggml_init(ctx_weights) failed\n"); return nullptr; }
    ggml_context * W = m->ctx_weights;
    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(bitvla): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, type, ggml_n_dims(gt), gt->ne);
        ggml_set_name(t, name);
        return t;
    };
    auto mk_mm  = [&](const char * name) { return mk(name, m->matmul_type); };
    auto mk_f32 = [&](const char * name) { return mk(name, GGML_TYPE_F32); };

    auto mk_bit = [&](const char * name) { return mk(name, m->packed_int2 ? GGML_TYPE_I8 : m->matmul_type); };

    bool ok = true;

    m->vit_patch_w = mk_mm("vit.patch_embd.weight");
    m->vit_patch_b = mk_f32("vit.patch_embd.bias");
    m->vit_pos     = mk_f32("vit.pos_embd.weight");
    m->vit.resize(m->vit_layers);
    for (int64_t i = 0; i < m->vit_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "vit.blk.%lld.%s", (long long) i, s); return p; };
        auto & w = m->vit[i];
        w.ln1w=mk_f32(N("ln1.weight")); w.ln1b=mk_f32(N("ln1.bias"));
        w.ln2w=mk_f32(N("ln2.weight")); w.ln2b=mk_f32(N("ln2.bias"));
        w.Wq=mk_bit(N("attn_q.weight")); w.bq=mk_f32(N("attn_q.bias"));
        w.Wk=mk_bit(N("attn_k.weight")); w.bk=mk_f32(N("attn_k.bias"));
        w.Wv=mk_bit(N("attn_v.weight")); w.bv=mk_f32(N("attn_v.bias"));
        w.Wo=mk_bit(N("attn_o.weight")); w.bo=mk_f32(N("attn_o.bias"));
        w.Wfc1=mk_bit(N("fc1.weight")); w.bfc1=mk_f32(N("fc1.bias"));
        w.Wfc2=mk_bit(N("fc2.weight")); w.bfc2=mk_f32(N("fc2.bias"));
        ok &= w.ln1w&&w.ln1b&&w.ln2w&&w.ln2b&&w.Wq&&w.bq&&w.Wk&&w.bk&&w.Wv&&w.bv&&w.Wo&&w.bo&&w.Wfc1&&w.bfc1&&w.Wfc2&&w.bfc2;
    }

    m->mm_l1_w=mk_mm("mm.linear_1.weight"); m->mm_l1_b=mk_f32("mm.linear_1.bias");
    m->mm_l2_w=mk_mm("mm.linear_2.weight"); m->mm_l2_b=mk_f32("mm.linear_2.bias");

    m->pp_fc1_w=mk_f32("aex.proprio.fc1.weight"); m->pp_fc1_b=mk_f32("aex.proprio.fc1.bias");
    m->pp_fc2_w=mk_f32("aex.proprio.fc2.weight"); m->pp_fc2_b=mk_f32("aex.proprio.fc2.bias");

    m->embed_tokens   = m->packed_int2 ? nullptr : mk_mm("token_embd.weight");
    m->lm_output_norm = mk_f32("lm.output_norm.weight");
    m->lm.resize(m->lm_layers);
    for (int64_t i = 0; i < m->lm_layers && ok; ++i) {
        char p[64]; auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "lm.blk.%lld.%s", (long long) i, s); return p; };
        auto & w = m->lm[i];
        w.attn_norm     = mk_f32(N("attn_norm.weight"));
        w.attn_sub_norm = mk_f32(N("attn_sub_norm.weight"));
        w.ffn_norm      = mk_f32(N("ffn_norm.weight"));
        w.ffn_sub_norm  = mk_f32(N("ffn_sub_norm.weight"));
        w.Wq=mk_bit(N("attn_q.weight")); w.Wk=mk_bit(N("attn_k.weight"));
        w.Wv=mk_bit(N("attn_v.weight")); w.Wo=mk_bit(N("attn_o.weight"));
        w.Wdown=mk_bit(N("ffn_down.weight"));
        if (m->packed_int2) {
            w.Wgate_up = mk_bit(N("ffn_gate_up.weight"));
        } else {
            w.Wgate=mk_mm(N("ffn_gate.weight")); w.Wup=mk_mm(N("ffn_up.weight"));
        }
        ok &= w.attn_norm&&w.attn_sub_norm&&w.ffn_norm&&w.ffn_sub_norm&&w.Wq&&w.Wk&&w.Wv&&w.Wo&&w.Wdown&&
              (m->packed_int2 ? (w.Wgate_up != nullptr) : (w.Wgate && w.Wup));
    }

    m->ah_ln1_w =mk_f32("aex.head.ln1.weight"); m->ah_ln1_b =mk_f32("aex.head.ln1.bias");
    m->ah_fc1_w =mk_mm ("aex.head.fc1.weight"); m->ah_fc1_b =mk_f32("aex.head.fc1.bias");
    m->ah_b0_ln_w=mk_f32("aex.head.blk.0.ln.weight"); m->ah_b0_ln_b=mk_f32("aex.head.blk.0.ln.bias");
    m->ah_b0_w  =mk_mm ("aex.head.blk.0.fc.weight"); m->ah_b0_b  =mk_f32("aex.head.blk.0.fc.bias");
    m->ah_b1_ln_w=mk_f32("aex.head.blk.1.ln.weight"); m->ah_b1_ln_b=mk_f32("aex.head.blk.1.ln.bias");
    m->ah_b1_w  =mk_mm ("aex.head.blk.1.fc.weight"); m->ah_b1_b  =mk_f32("aex.head.blk.1.fc.bias");
    m->ah_ln2_w =mk_f32("aex.head.ln2.weight"); m->ah_ln2_b =mk_f32("aex.head.ln2.bias");
    m->ah_fc2_w =mk_mm ("aex.head.fc2.weight"); m->ah_fc2_b =mk_f32("aex.head.fc2.bias");

    ok &= m->vit_patch_w&&m->vit_patch_b&&m->vit_pos&&m->mm_l1_w&&m->mm_l1_b&&m->mm_l2_w&&m->mm_l2_b&&
          m->pp_fc1_w&&m->pp_fc1_b&&m->pp_fc2_w&&m->pp_fc2_b&&(m->embed_tokens||m->packed_int2)&&m->lm_output_norm&&
          m->ah_ln1_w&&m->ah_ln1_b&&m->ah_fc1_w&&m->ah_fc1_b&&
          m->ah_b0_ln_w&&m->ah_b0_ln_b&&m->ah_b0_w&&m->ah_b0_b&&
          m->ah_b1_ln_w&&m->ah_b1_ln_b&&m->ah_b1_w&&m->ah_b1_b&&
          m->ah_ln2_w&&m->ah_ln2_b&&m->ah_fc2_w&&m->ah_fc2_b;
    if (!ok) { std::fprintf(stderr, "vla(bitvla): weight tensor setup failed\n"); return nullptr; }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(bitvla): ggml_backend_alloc_ctx_tensors failed (OOM?)\n"); return nullptr; }
    for (ggml_tensor * t = ggml_get_first_tensor(W); t; t = ggml_get_next_tensor(W, t)) {
        std::vector<uint8_t> bytes = g.read_convert(ggml_get_name(t), t->type);
        if (bytes.empty() || bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(bitvla): failed to load %s (%zu vs %zu bytes)\n", ggml_get_name(t), bytes.size(), ggml_nbytes(t)); return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(bitvla): weights resident in %.2f GiB (%s); image_id=%d proprio_id=%d action_begin_id=%d stop_id=%d\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0),
                m->packed_int2 ? "int2-packed + F32 sidecars"
                               : (m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16"),
                m->image_token_id, m->proprio_pad_id, m->action_begin_id, m->stop_id);

#ifndef VLA_BITVLA_CUDA_KERNELS
    if (m->packed_int2) {
        std::fprintf(stderr, "vla(bitvla): int2-packed GGUF requires a CUDA build "
                             "(VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.\n");
        return nullptr;
    }
#endif

#ifdef VLA_BITVLA_CUDA_KERNELS

    if ((m->packed_int2 || m->matmul_type == GGML_TYPE_F32) && !std::getenv("VLA_BITVLA_NO_CUDA_LM")) {
        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0) {
            cudaSetDevice(0);

            auto load_bit = [&](ggml_tensor * t, int64_t N, int64_t K) -> std::pair<int8_t*, float*> {
                if (m->packed_int2) {
                    int8_t * dp = upload_int8((const uint8_t*) t->data, ggml_nbytes(t), m->cuda_devptrs);
                    std::string nm = ggml_get_name(t);
                    std::string sn = nm.substr(0, nm.size() - 7) + ".scale";
                    std::vector<float> sc = g.read_f32(sn.c_str());
                    float * dws = sc.empty() ? nullptr : upload_f32_scales(sc.data(), (int) sc.size(), m->cuda_devptrs);
                    return { dp, dws };
                }
                float ws;
                int8_t * dp = pack_and_upload_one((const float*) t->data, N, K, ws, m->cuda_devptrs);
                return { dp, upload_f32_scales(&ws, 1, m->cuda_devptrs) };
            };
            const int max_seq = 1024;
            m->cuda_max_seq = max_seq;
            m->lm_cuda_ctx = bitvla_lm_cuda_init(
                (int) m->lm_hidden, (int) m->lm_q, (int) m->lm_kv, (int) m->lm_head_dim,
                (int) m->lm_inter, (int) m->lm_layers, m->lm_rope_base, m->lm_rms_eps, max_seq);
            if (m->lm_cuda_ctx) {
                bool pack_ok = true;
                for (int64_t L = 0; L < m->lm_layers && pack_ok; ++L) {
                    bitvla_lm_layer_cuda lyr{};

                    lyr.attn_norm_w     = upload_bf16_from_f32((const float*) m->lm[L].attn_norm->data,     m->lm_hidden, m->cuda_devptrs);
                    lyr.attn_sub_norm_w = upload_bf16_from_f32((const float*) m->lm[L].attn_sub_norm->data, m->lm_hidden, m->cuda_devptrs);
                    lyr.ffn_norm_w      = upload_bf16_from_f32((const float*) m->lm[L].ffn_norm->data,      m->lm_hidden, m->cuda_devptrs);
                    lyr.ffn_sub_norm_w  = upload_bf16_from_f32((const float*) m->lm[L].ffn_sub_norm->data,  m->lm_inter,  m->cuda_devptrs);

                    const int64_t hq_dim  = m->lm_q  * m->lm_head_dim;
                    const int64_t hkv_dim = m->lm_kv * m->lm_head_dim;
                    { auto r = load_bit(m->lm[L].Wq, hq_dim,       m->lm_hidden); lyr.q_packed = r.first; lyr.q_ws = r.second; }
                    { auto r = load_bit(m->lm[L].Wk, hkv_dim,      m->lm_hidden); lyr.k_packed = r.first; lyr.k_ws = r.second; }
                    { auto r = load_bit(m->lm[L].Wv, hkv_dim,      m->lm_hidden); lyr.v_packed = r.first; lyr.v_ws = r.second; }

                    { auto r = load_bit(m->lm[L].Wo, m->lm_hidden, m->lm_hidden); lyr.o_packed = r.first; lyr.o_ws = r.second; }

                    if (m->packed_int2) {
                        lyr.gate_up_packed = upload_int8((const uint8_t*) m->lm[L].Wgate_up->data, ggml_nbytes(m->lm[L].Wgate_up), m->cuda_devptrs);
                        std::vector<float> sc = g.read_f32(("lm.blk." + std::to_string(L) + ".ffn_gate_up.scale").c_str());
                        lyr.gate_up_ws = upload_f32_scales(sc.data(), (int) sc.size(), m->cuda_devptrs);
                    } else {
                        std::vector<float> ws2;
                        lyr.gate_up_packed = pack_and_upload_fused(
                            { (const float*) m->lm[L].Wgate->data, (const float*) m->lm[L].Wup->data },
                            { m->lm_inter, m->lm_inter },
                            m->lm_hidden, ws2, m->cuda_devptrs);
                        lyr.gate_up_ws = upload_f32_scales(ws2.data(), 2, m->cuda_devptrs);
                    }

                    { auto r = load_bit(m->lm[L].Wdown, m->lm_hidden, m->lm_inter); lyr.down_packed = r.first; lyr.down_ws = r.second; }
                    bitvla_lm_cuda_set_layer(m->lm_cuda_ctx, (int) L, &lyr);
                }
                if (pack_ok) {
                    __nv_bfloat16* onorm = upload_bf16_from_f32((const float*) m->lm_output_norm->data, m->lm_hidden, m->cuda_devptrs);
                    bitvla_lm_cuda_set_output_norm(m->lm_cuda_ctx, onorm);

                    cudaMalloc(&m->d_inputs_embeds, (size_t) max_seq * m->lm_hidden * sizeof(__nv_bfloat16));
                    cudaMalloc(&m->d_last_hidden,   (size_t) max_seq * m->lm_hidden * sizeof(__nv_bfloat16));
                    cudaMalloc(&m->d_action_hidden, (size_t) (m->num_actions_chunk * m->action_dim) * m->lm_hidden * sizeof(__nv_bfloat16));
                    cudaMalloc(&m->d_action_ids,    (size_t) (m->num_actions_chunk * m->action_dim) * sizeof(int32_t));
                    m->cuda_lm_ready = true;
                    size_t packed_bytes = 0;
                    for (void* p : m->cuda_devptrs) {
                        (void) p;

                    }

                    packed_bytes = (size_t) m->lm_layers * (
                        (size_t)(m->lm_q + 2*m->lm_kv) * m->lm_head_dim * m->lm_hidden / 4 +
                        (size_t) m->lm_hidden * m->lm_hidden / 4 +
                        (size_t) 2 * m->lm_inter * m->lm_hidden / 4 +
                        (size_t) m->lm_hidden * m->lm_inter / 4);
                    std::printf("vla(bitvla): CUDA LM forward ENABLED - packed int2 weights = %.2f MiB, max_seq=%d\n",
                                packed_bytes / (1024.0 * 1024.0), max_seq);

                    const int patch_flat = 3 * m->patch_size * m->patch_size;
                    const int mm_out     = (int) m->lm_hidden;
                    const int ffn_pad    = ((m->vit_inter + 127) / 128) * 128;
                    m->vit_cuda_ctx = bitvla_vit_cuda_init(
                        (int) m->vit_layers, (int) m->vit_hidden, (int) m->vit_heads,
                        (int) m->vit_inter,  (int) m->n_patches, patch_flat,
                        m->vit_ln_eps, mm_out);
                    if (m->vit_cuda_ctx) {
                        bool vit_ok = true;
                        for (int64_t L = 0; L < m->vit_layers && vit_ok; ++L) {
                            bitvla_vit_layer_cuda vl{};

                            vl.ln1_w = upload_bf16_from_f32((const float*) m->vit[L].ln1w->data, m->vit_hidden, m->cuda_devptrs);
                            vl.ln1_b = upload_bf16_from_f32((const float*) m->vit[L].ln1b->data, m->vit_hidden, m->cuda_devptrs);
                            vl.ln2_w = upload_bf16_from_f32((const float*) m->vit[L].ln2w->data, m->vit_hidden, m->cuda_devptrs);
                            vl.ln2_b = upload_bf16_from_f32((const float*) m->vit[L].ln2b->data, m->vit_hidden, m->cuda_devptrs);

                            { auto r = load_bit(m->vit[L].Wq, m->vit_hidden, m->vit_hidden); vl.q_packed = r.first; vl.q_ws = r.second; }
                            vl.q_b = upload_bf16_from_f32((const float*) m->vit[L].bq->data, m->vit_hidden, m->cuda_devptrs);
                            { auto r = load_bit(m->vit[L].Wk, m->vit_hidden, m->vit_hidden); vl.k_packed = r.first; vl.k_ws = r.second; }
                            vl.k_b = upload_bf16_from_f32((const float*) m->vit[L].bk->data, m->vit_hidden, m->cuda_devptrs);
                            { auto r = load_bit(m->vit[L].Wv, m->vit_hidden, m->vit_hidden); vl.v_packed = r.first; vl.v_ws = r.second; }
                            vl.v_b = upload_bf16_from_f32((const float*) m->vit[L].bv->data, m->vit_hidden, m->cuda_devptrs);
                            { auto r = load_bit(m->vit[L].Wo, m->vit_hidden, m->vit_hidden); vl.o_packed = r.first; vl.o_ws = r.second; }
                            vl.o_b = upload_bf16_from_f32((const float*) m->vit[L].bo->data, m->vit_hidden, m->cuda_devptrs);

                            { auto r = load_bit(m->vit[L].Wfc1, m->vit_inter, m->vit_hidden); vl.fc1_packed = r.first; vl.fc1_ws = r.second; }
                            vl.fc1_b = upload_bf16_from_f32((const float*) m->vit[L].bfc1->data, m->vit_inter, m->cuda_devptrs);

                            if (m->packed_int2) {
                                { auto r = load_bit(m->vit[L].Wfc2, m->vit_hidden, ffn_pad); vl.fc2_packed = r.first; vl.fc2_ws = r.second; }
                                vl.fc2_b = upload_bf16_from_f32((const float*) m->vit[L].bfc2->data, m->vit_hidden, m->cuda_devptrs);
                            } else {
                                const float* W = (const float*) m->vit[L].Wfc2->data;
                                std::vector<int8_t> tern;
                                float scale;
                                recover_ternary_and_scale(W, m->vit_hidden * m->vit_inter, tern, scale);

                                std::vector<int8_t> padded((size_t) m->vit_hidden * ffn_pad, 0);
                                for (int64_t n = 0; n < m->vit_hidden; ++n) {
                                    std::memcpy(padded.data() + n * ffn_pad,
                                                tern.data() + n * m->vit_inter,
                                                (size_t) m->vit_inter);

                                }
                                auto packed = pack_ladder_int2(padded.data(), m->vit_hidden, ffn_pad);
                                vl.fc2_packed = upload_int8(packed.data(), packed.size(), m->cuda_devptrs);
                                vl.fc2_ws     = upload_f32_scales(&scale, 1, m->cuda_devptrs);
                                vl.fc2_b      = upload_bf16_from_f32((const float*) m->vit[L].bfc2->data, m->vit_hidden, m->cuda_devptrs);
                            }
                            bitvla_vit_cuda_set_layer(m->vit_cuda_ctx, (int) L, &vl);
                        }
                        if (vit_ok) {

                            __nv_bfloat16* pe_w   = upload_bf16_from_f32((const float*) m->vit_patch_w->data, m->vit_hidden * patch_flat, m->cuda_devptrs);
                            __nv_bfloat16* pe_b   = upload_bf16_from_f32((const float*) m->vit_patch_b->data, m->vit_hidden, m->cuda_devptrs);
                            __nv_bfloat16* pos_e  = upload_bf16_from_f32((const float*) m->vit_pos->data,     m->n_patches * m->vit_hidden, m->cuda_devptrs);
                            bitvla_vit_cuda_set_embed(m->vit_cuda_ctx, pe_w, pe_b, pos_e);

                            __nv_bfloat16* mm_W1 = upload_bf16_from_f32((const float*) m->mm_l1_w->data, mm_out * m->vit_hidden, m->cuda_devptrs);
                            __nv_bfloat16* mm_b1 = upload_bf16_from_f32((const float*) m->mm_l1_b->data, mm_out,                 m->cuda_devptrs);
                            __nv_bfloat16* mm_W2 = upload_bf16_from_f32((const float*) m->mm_l2_w->data, mm_out * mm_out,        m->cuda_devptrs);
                            __nv_bfloat16* mm_b2 = upload_bf16_from_f32((const float*) m->mm_l2_b->data, mm_out,                 m->cuda_devptrs);
                            bitvla_vit_cuda_set_mmproj(m->vit_cuda_ctx, mm_W1, mm_b1, mm_W2, mm_b2);

                            cudaMalloc(&m->d_vit_patches,    (size_t) m->n_patches * patch_flat * sizeof(__nv_bfloat16));
                            cudaMalloc(&m->d_vit_img_embeds, (size_t) m->n_patches * mm_out     * sizeof(__nv_bfloat16));
                            m->cuda_vit_ready = true;
                            const size_t vit_packed_bytes = (size_t) m->vit_layers * (
                                4 * (size_t) m->vit_hidden * m->vit_hidden / 4 +
                                (size_t) m->vit_inter * m->vit_hidden / 4 +
                                (size_t) m->vit_hidden * ffn_pad / 4);
                            std::printf("vla(bitvla): CUDA ViT forward ENABLED - packed int2 weights = %.2f MiB, ffn_pad=%d\n",
                                        vit_packed_bytes / (1024.0 * 1024.0), ffn_pad);
                        } else {
                            bitvla_vit_cuda_free(m->vit_cuda_ctx);
                            m->vit_cuda_ctx = nullptr;
                            std::printf("vla(bitvla): CUDA ViT packing failed; falling back to CPU vision\n");
                        }
                    } else {
                        std::fprintf(stderr, "vla(bitvla): bitvla_vit_cuda_init failed; falling back to CPU vision\n");
                    }
                } else {
                    bitvla_lm_cuda_free(m->lm_cuda_ctx);
                    m->lm_cuda_ctx = nullptr;
                    std::printf("vla(bitvla): CUDA LM packing failed; falling back to CPU\n");
                }
            } else {
                std::fprintf(stderr, "vla(bitvla): bitvla_lm_cuda_init failed; falling back to CPU LM\n");
            }
        } else {
            std::printf("vla(bitvla): no CUDA device - using CPU LM forward\n");
        }
    }

    if (m->packed_int2 && !(m->cuda_lm_ready && m->cuda_vit_ready)) {
        std::fprintf(stderr, "vla(bitvla): int2-packed GGUF requires the CUDA LM+ViT path, which "
                             "failed to initialize - use the bf16 GGUF, or fix the CUDA setup.\n");
        return nullptr;
    }

    if (m->cuda_lm_ready && m->cuda_vit_ready &&
        std::getenv("VLA_BITVLA_CPU_HEAD") == nullptr) {
        m->fp32head_cuda_ctx = bitvla_fp32head_cuda_init(
            (int) m->proprio_dim, (int) m->lm_hidden,
            (int) m->num_actions_chunk, (int) m->action_dim,
            m->ah_ln_eps,
            (const float*) m->pp_fc1_w->data,  (const float*) m->pp_fc1_b->data,
            (const float*) m->pp_fc2_w->data,  (const float*) m->pp_fc2_b->data,
            (const float*) m->ah_ln1_w->data,  (const float*) m->ah_ln1_b->data,
            (const float*) m->ah_fc1_w->data,  (const float*) m->ah_fc1_b->data,
            (const float*) m->ah_b0_ln_w->data,(const float*) m->ah_b0_ln_b->data,
            (const float*) m->ah_b0_w->data,   (const float*) m->ah_b0_b->data,
            (const float*) m->ah_b1_ln_w->data,(const float*) m->ah_b1_ln_b->data,
            (const float*) m->ah_b1_w->data,   (const float*) m->ah_b1_b->data,
            (const float*) m->ah_ln2_w->data,  (const float*) m->ah_ln2_b->data,
            (const float*) m->ah_fc2_w->data,  (const float*) m->ah_fc2_b->data);
        if (m->fp32head_cuda_ctx) {
            m->cuda_fp32head_ready = true;

            const int64_t h = m->lm_hidden, p = m->proprio_dim, a = m->action_dim;
            const int64_t adh = a * h;
            const size_t bytes = (size_t)(
                h*p + h + h*h + h +
                2*adh + h*adh + h +
                2 * (2*h + h*h + h) +
                2*h + a*h + a) * sizeof(float);
            std::printf("vla(bitvla): CUDA fp32head (ProprioProj+ActionHead) ENABLED - weights = %.2f MiB on GPU\n",
                        bytes / (1024.0 * 1024.0));
        } else {
            std::fprintf(stderr, "vla(bitvla): bitvla_fp32head_cuda_init failed; falling back to CPU ggml head\n");
        }
    }

    if (m->cuda_lm_ready && m->cuda_vit_ready && m->weight_buf &&
        std::getenv("VLA_BITVLA_KEEP_CPU_WEIGHTS") == nullptr) {
        size_t bytes_kept = 0;
        if (!m->cuda_fp32head_ready) {

            ggml_tensor* keep_tensors[] = {
                m->pp_fc1_w,    m->pp_fc1_b,    m->pp_fc2_w,    m->pp_fc2_b,
                m->ah_ln1_w,    m->ah_ln1_b,    m->ah_fc1_w,    m->ah_fc1_b,
                m->ah_b0_ln_w,  m->ah_b0_ln_b,  m->ah_b0_w,     m->ah_b0_b,
                m->ah_b1_ln_w,  m->ah_b1_ln_b,  m->ah_b1_w,     m->ah_b1_b,
                m->ah_ln2_w,    m->ah_ln2_b,    m->ah_fc2_w,    m->ah_fc2_b,
            };
            for (ggml_tensor* t : keep_tensors) {
                if (!t) continue;
                const size_t nb = ggml_nbytes(t);
                void* copy = std::malloc(nb);
                std::memcpy(copy, t->data, nb);
                t->data = copy;
                m->cpu_kept_ptrs.push_back(copy);
                bytes_kept += nb;
            }
        }
        const double before_gb = ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0);
        ggml_backend_buffer_free(m->weight_buf);
        m->weight_buf = nullptr;

#ifdef __GLIBC__
        malloc_trim(0);
#endif
        if (m->cuda_fp32head_ready) {
            std::printf("vla(bitvla): freed weight_buf (%.2f GiB) - full GPU path (no CPU weights kept)\n",
                        before_gb);
        } else {
            std::printf("vla(bitvla): freed weight_buf (%.2f GiB); kept %.2f MiB of CPU-resident "
                        "ProprioProj+ActionHead weights as standalone copies\n",
                        before_gb, bytes_kept / (1024.0 * 1024.0));
        }
    }
#endif

    return m;
}

std::vector<float> BitvlaModelArch::predict(const Inputs& in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t_start = clk::now();
    stats = Stats{};
    const bool timing_phase = (in.timing_detail == TimingDetail::PHASE);

    const char* _dump_dir = std::getenv("VLA_BITVLA_DUMP_DIR");
    auto _dump_bin = [&](const char* name, const float* data, size_t nelem) {
        if (!_dump_dir) return;
        std::string path = std::string(_dump_dir) + "/" + name + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return;
        std::fwrite(data, sizeof(float), nelem, f);
        std::fclose(f);
    };
    auto _dump_manifest = [&](const std::string& line) {
        if (!_dump_dir) return;
        std::string path = std::string(_dump_dir) + "/manifest.txt";
        FILE* f = std::fopen(path.c_str(), "a");
        if (!f) return;
        std::fwrite(line.data(), 1, line.size(), f);
        std::fputc('\n', f);
        std::fclose(f);
    };
    if (_dump_dir) {

        std::string p = std::string(_dump_dir) + "/manifest.txt";
        FILE* f = std::fopen(p.c_str(), "w"); if (f) std::fclose(f);
    }

    const int64_t H = image_size, P = patch_size, N = n_patches;
    const int64_t hidden_v = vit_hidden, hidden_l = lm_hidden;
    const int64_t patch_flat = 3 * P * P;

    const bool use_precomp_img = (in.precomputed_img_emb != nullptr) && in.n_img_views > 0;
    const int64_t n_views = use_precomp_img ? (int64_t) in.n_img_views : (int64_t) in.n_images;
    if (n_views <= 0) {
        std::fprintf(stderr, "vla(bitvla): no images provided\n");
        return {};
    }

    std::vector<float> img_embeds_host;

    if (!use_precomp_img) {

        img_embeds_host.assign((size_t) n_views * N * hidden_l, 0.0f);
        std::vector<float> patches((size_t) N * patch_flat);
        const auto t_v0 = clk::now();
        for (int64_t v = 0; v < n_views; ++v) {
            const ImageView & iv = in.images[v];
            if (iv.w != (int) H || iv.h != (int) H) {
                std::fprintf(stderr, "vla(bitvla): image view %lld size %dx%d ≠ expected %lldx%lld\n",
                             (long long) v, iv.w, iv.h, (long long) H, (long long) H);
                return {};
            }

            for (int64_t pi = 0; pi < H / P; ++pi)
            for (int64_t pj = 0; pj < H / P; ++pj) {
                const int64_t p_idx = pi * (H / P) + pj;
                float * p_dst = patches.data() + p_idx * patch_flat;
                int64_t k = 0;
                for (int64_t c = 0; c < 3; ++c)
                for (int64_t kh = 0; kh < P; ++kh)
                for (int64_t kw = 0; kw < P; ++kw) {
                    const int64_t h = pi * P + kh;
                    const int64_t w = pj * P + kw;

                    float px;
                    if (iv.format == PixelFormat::U8) {
                        const uint8_t * p = (const uint8_t *) iv.data;
                        px = (float) p[h * H * 3 + w * 3 + c] / 127.5f - 1.0f;
                    } else {
                        const float * p = (const float *) iv.data;
                        px = p[h * H * 3 + w * 3 + c] * 2.0f - 1.0f;
                    }
                    p_dst[k++] = px;
                }
            }

#ifdef VLA_BITVLA_CUDA_KERNELS
            if (cuda_vit_ready) {

                std::vector<uint16_t> patches_bf16((size_t) N * patch_flat);
                for (size_t i = 0; i < patches_bf16.size(); ++i) patches_bf16[i] = f32_to_bf16_u16(patches[i]);
                cudaMemcpy(d_vit_patches, patches_bf16.data(), patches_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
                int rc = bitvla_vit_cuda_forward(vit_cuda_ctx, d_vit_patches, d_vit_img_embeds,  0);
                if (rc != 0) { std::fprintf(stderr, "vla(bitvla): CUDA ViT forward failed (view %lld)\n", (long long) v); return {}; }
                std::vector<uint16_t> img_bf16((size_t) N * hidden_l);
                cudaMemcpy(img_bf16.data(), d_vit_img_embeds, img_bf16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
                float* dst = img_embeds_host.data() + (size_t) v * N * hidden_l;
                for (size_t i = 0; i < img_bf16.size(); ++i) {
                    uint32_t u = ((uint32_t) img_bf16[i]) << 16;
                    float f; std::memcpy(&f, &u, 4);
                    dst[i] = f;
                }
            } else
#endif
            {

            std::vector<uint8_t> meta_buf((size_t) 24 * 1024 * 1024);
            ggml_init_params gp = { meta_buf.size(), meta_buf.data(), true };
            ggml_context * ctx = ggml_init(gp);

            ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, patch_flat, N);
            ggml_set_name(x_in, "patches");

            ggml_tensor * pe = ggml_add(ctx, ggml_mul_mat(ctx, vit_patch_w, x_in), vit_patch_b);
            ggml_tensor * h  = ggml_add(ctx, pe, vit_pos);
            for (int64_t L = 0; L < vit_layers; ++L) {
                h = build_vit_layer(ctx, vit[L], h, N, vit_heads, vit_head_dim, hidden_v, vit_ln_eps);
            }

            ggml_tensor * mm1 = ggml_add(ctx, ggml_mul_mat(ctx, mm_l1_w, h),   mm_l1_b);
            ggml_tensor * mmg = ggml_gelu_erf(ctx, mm1);
            ggml_tensor * mm2 = ggml_add(ctx, ggml_mul_mat(ctx, mm_l2_w, mmg), mm_l2_b);
            ggml_set_name(mm2, "img_embeds");

            ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ggml_cgraph * gf = ggml_new_graph_custom(ctx,  4096,  false);
            ggml_build_forward_expand(gf, mm2);
            if (!ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "vla(bitvla): gallocr_alloc_graph failed (view %lld)\n", (long long) v); ggml_gallocr_free(galloc); ggml_free(ctx); return {}; }
            ggml_backend_tensor_set(x_in, patches.data(), 0, ggml_nbytes(x_in));
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(bitvla): vision graph compute failed (view %lld)\n", (long long) v);
                ggml_gallocr_free(galloc); ggml_free(ctx); return {};
            }
            ggml_backend_tensor_get(mm2, img_embeds_host.data() + (size_t) v * N * hidden_l, 0, (size_t) N * hidden_l * sizeof(float));
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            }
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now() - t_v0).count();
    } else {
        img_embeds_host.assign(in.precomputed_img_emb, in.precomputed_img_emb + (size_t) n_views * N * hidden_l);
    }

    _dump_bin("mm_proj_out", img_embeds_host.data(), img_embeds_host.size());
    _dump_manifest(std::string("mm_proj_out fp32 ") + std::to_string(n_views) + " " + std::to_string(N) + " " + std::to_string(hidden_l));

    std::vector<float> proprio_embed_host((size_t) hidden_l);
#ifdef VLA_BITVLA_CUDA_KERNELS
    if (cuda_fp32head_ready) {
        if (bitvla_fp32head_proprio_forward(fp32head_cuda_ctx, in.state, proprio_embed_host.data(), 0) != 0) {
            std::fprintf(stderr, "vla(bitvla): CUDA proprio forward failed\n"); return {};
        }
    } else
#endif
    {
        std::vector<uint8_t> meta_buf((size_t) 4 * 1024 * 1024);
        ggml_init_params gp = { meta_buf.size(), meta_buf.data(), true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, proprio_dim, 1);
        ggml_set_name(x_in, "state");
        ggml_tensor * h1     = ggml_add(ctx, ggml_mul_mat(ctx, pp_fc1_w, x_in), pp_fc1_b);
        ggml_tensor * h1_gel = ggml_gelu_erf(ctx, h1);
        ggml_tensor * out    = ggml_add(ctx, ggml_mul_mat(ctx, pp_fc2_w, h1_gel), pp_fc2_b);
        ggml_set_name(out, "proprio_embed");

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "vla(bitvla): gallocr failed (proprio)\n"); ggml_gallocr_free(galloc); ggml_free(ctx); return {}; }
        ggml_backend_tensor_set(x_in, in.state, 0, ggml_nbytes(x_in));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla): proprio compute failed\n"); ggml_gallocr_free(galloc); ggml_free(ctx); return {}; }
        ggml_backend_tensor_get(out, proprio_embed_host.data(), 0, (size_t) hidden_l * sizeof(float));
        ggml_gallocr_free(galloc); ggml_free(ctx);
    }

    _dump_bin("proprio_features", proprio_embed_host.data(), proprio_embed_host.size());
    _dump_manifest(std::string("proprio_features fp32 1 ") + std::to_string(hidden_l));

    const int64_t n_lang_in = (int64_t) in.n_lang;
    const int64_t n_img_tok = n_views * N;
    const int64_t n_action  = num_actions_chunk * action_dim;

    int64_t n_image_markers = 0, n_proprio_markers = 0;
    for (int64_t i = 0; i < n_lang_in; ++i) {
        if (in.lang_tokens[i] == image_token_id)     n_image_markers++;
        else if (in.lang_tokens[i] == proprio_pad_id) n_proprio_markers++;
    }
    const bool full_prefix = (n_image_markers == n_img_tok && n_proprio_markers == 1);
    static bool s_logged_path = false;
    if (!s_logged_path) {
        std::printf("vla(bitvla): prefix mode = %s (n_image_markers=%lld n_proprio_markers=%lld n_img_tok=%lld)\n",
                    full_prefix ? "FULL-A (upstream-style: markers in lang_tokens)" : "SEGMENT-B (legacy: append markers)",
                    (long long) n_image_markers, (long long) n_proprio_markers, (long long) n_img_tok);
        s_logged_path = true;
    }
    if (!full_prefix && (n_image_markers != 0 || n_proprio_markers != 0)) {
        std::fprintf(stderr,
                     "vla(bitvla): lang_tokens has %lld image_pad + %lld proprio_pad markers "
                     "(expected either 0+0 for the legacy segment-concat path, or %lld+1 "
                     "for the full-prefix path matching upstream's chat template)\n",
                     (long long) n_image_markers, (long long) n_proprio_markers, (long long) n_img_tok);
        return {};
    }

    const int64_t seq = full_prefix
        ? (n_lang_in + n_action + 1)
        : (n_lang_in + n_img_tok + 1 + n_action + 1);
    if (seq > lm_max_pos) {
        std::fprintf(stderr, "vla(bitvla): seq=%lld > lm_max_pos=%lld\n", (long long) seq, (long long) lm_max_pos);
        return {};
    }

    std::vector<float> inputs_embeds((size_t) seq * hidden_l, 0.0f);

    if (full_prefix) {

        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + n_lang_in);
        for (int32_t id : ids) {
            if (id < 0 || id >= vocab_size) {
                std::fprintf(stderr, "vla(bitvla): prompt token %d out of vocab\n", id); return {};
            }
        }
        if (!emb_reader.fetch_rows_f32("token_embd.weight", ids, inputs_embeds.data(), hidden_l)) return {};

        int64_t k_img = 0;
        for (int64_t i = 0; i < n_lang_in; ++i) {
            const int32_t tok = in.lang_tokens[i];
            if (tok == image_token_id) {
                std::memcpy(inputs_embeds.data() + (size_t) i * hidden_l,
                            img_embeds_host.data() + (size_t) k_img * hidden_l,
                            (size_t) hidden_l * sizeof(float));
                k_img++;
            } else if (tok == proprio_pad_id) {
                std::memcpy(inputs_embeds.data() + (size_t) i * hidden_l,
                            proprio_embed_host.data(),
                            (size_t) hidden_l * sizeof(float));
            }
        }

        std::memcpy(inputs_embeds.data() + (size_t) (seq - 1) * hidden_l,
                    stop_embed.data(), (size_t) hidden_l * sizeof(float));
    } else {

        const int64_t n_prompt = n_lang_in;
        if (n_prompt > 0) {
            std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + n_prompt);
            for (int32_t id : ids) {
                if (id < 0 || id >= vocab_size) {
                    std::fprintf(stderr, "vla(bitvla): prompt token %d out of vocab\n", id); return {};
                }
            }
            if (!emb_reader.fetch_rows_f32("token_embd.weight", ids, inputs_embeds.data(), hidden_l)) return {};
        }
        std::memcpy(inputs_embeds.data() + (size_t) n_prompt * hidden_l,
                    img_embeds_host.data(), (size_t) n_img_tok * hidden_l * sizeof(float));
        std::memcpy(inputs_embeds.data() + (size_t) (n_prompt + n_img_tok) * hidden_l,
                    proprio_embed_host.data(), (size_t) hidden_l * sizeof(float));
        std::memcpy(inputs_embeds.data() + (size_t) (seq - 1) * hidden_l,
                    stop_embed.data(), (size_t) hidden_l * sizeof(float));
    }

    _dump_bin("inputs_embeds", inputs_embeds.data(), inputs_embeds.size());
    _dump_manifest(std::string("inputs_embeds fp32 1 ") + std::to_string(seq) +
                   " " + std::to_string(hidden_l));

    std::vector<float> last_hidden_at_actions((size_t) n_action * hidden_l);
    const auto t_p0 = clk::now();

#ifdef VLA_BITVLA_CUDA_KERNELS
    if (cuda_lm_ready && seq <= cuda_max_seq) {

        std::vector<uint16_t> in_bf16((size_t) seq * hidden_l);
        for (size_t i = 0; i < in_bf16.size(); ++i) in_bf16[i] = f32_to_bf16_u16(inputs_embeds[i]);
        cudaMemcpy(d_inputs_embeds, in_bf16.data(), in_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        int rc = bitvla_lm_cuda_forward(lm_cuda_ctx, d_inputs_embeds, d_last_hidden, (int) seq,  0);
        if (rc != 0) { std::fprintf(stderr, "vla(bitvla): CUDA LM forward failed\n"); return {}; }

        std::vector<int32_t> aids(n_action);
        for (int64_t i = 0; i < n_action; ++i) aids[i] = (int32_t) (seq - 2 - n_action + i);
        cudaMemcpy(d_action_ids, aids.data(), n_action * sizeof(int32_t), cudaMemcpyHostToDevice);
        bitvla_gather_rows_bf16(d_last_hidden, d_action_hidden, d_action_ids, (int) n_action, (int) hidden_l,  0);

        std::vector<uint16_t> out_bf16((size_t) n_action * hidden_l);
        cudaMemcpy(out_bf16.data(), d_action_hidden, out_bf16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < out_bf16.size(); ++i) {
            uint32_t u = ((uint32_t) out_bf16[i]) << 16;
            float f; std::memcpy(&f, &u, 4);
            last_hidden_at_actions[i] = f;
        }
    } else
#endif
    {
        std::vector<uint8_t> meta_buf((size_t) 64 * 1024 * 1024);
        ggml_init_params gp = { meta_buf.size(), meta_buf.data(), true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_l, seq);
        ggml_set_name(x_in, "inputs_embeds");
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
        ggml_set_name(positions, "positions");

        ggml_tensor * h = x_in;
        for (int64_t L = 0; L < lm_layers; ++L) {
            h = build_lm_layer(ctx, *this, lm[L], h, positions, seq);
        }
        ggml_tensor * h_norm = rmsnorm(ctx, h, lm_output_norm, lm_rms_eps);
        ggml_set_name(h_norm, "last_hidden");

        ggml_tensor * action_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_action);
        ggml_set_name(action_ids, "action_ids");
        ggml_tensor * action_hidden = ggml_get_rows(ctx, h_norm, action_ids);
        ggml_set_name(action_hidden, "action_hidden");

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_cgraph * gf = ggml_new_graph_custom(ctx,  32768,  false);
        ggml_build_forward_expand(gf, action_hidden);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "vla(bitvla): gallocr failed (lm)\n"); ggml_gallocr_free(galloc); ggml_free(ctx); return {}; }

        ggml_backend_tensor_set(x_in, inputs_embeds.data(), 0, ggml_nbytes(x_in));
        std::vector<int32_t> pos_v(seq);
        for (int64_t i = 0; i < seq; ++i) pos_v[i] = (int32_t) i;
        ggml_backend_tensor_set(positions, pos_v.data(), 0, ggml_nbytes(positions));
        std::vector<int32_t> aids(n_action);
        for (int64_t i = 0; i < n_action; ++i) aids[i] = (int32_t) (seq - 2 - n_action + i);
        ggml_backend_tensor_set(action_ids, aids.data(), 0, ggml_nbytes(action_ids));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla): lm prefill compute failed\n"); ggml_gallocr_free(galloc); ggml_free(ctx); return {}; }
        ggml_backend_tensor_get(action_hidden, last_hidden_at_actions.data(), 0, (size_t) n_action * hidden_l * sizeof(float));
        ggml_gallocr_free(galloc); ggml_free(ctx);
    }
    if (timing_phase) stats.ms_prefill = std::chrono::duration<float, std::milli>(clk::now() - t_p0).count();

    _dump_bin("ah_input", last_hidden_at_actions.data(), last_hidden_at_actions.size());
    _dump_manifest(std::string("ah_input fp32 1 ") + std::to_string(num_actions_chunk) +
                   " " + std::to_string(action_dim * hidden_l));

    const int64_t chunk  = num_actions_chunk;
    const int64_t in_dim = action_dim * hidden_l;
    std::vector<float> normalized_actions((size_t) chunk * action_dim);
    const auto t_d0 = clk::now();
#ifdef VLA_BITVLA_CUDA_KERNELS
    if (cuda_fp32head_ready) {
        if (bitvla_fp32head_action_forward(fp32head_cuda_ctx,
                                            last_hidden_at_actions.data(),
                                            normalized_actions.data(), 0) != 0) {
            std::fprintf(stderr, "vla(bitvla): CUDA action_head forward failed\n"); return {};
        }
    } else
#endif
    {
        std::vector<uint8_t> meta_buf((size_t) 8 * 1024 * 1024);
        ggml_init_params gp = { meta_buf.size(), meta_buf.data(), true };
        ggml_context * ctx = ggml_init(gp);

        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, chunk);
        ggml_set_name(x, "x");

        ggml_tensor * ln1 = layernorm(ctx, x, ah_ln1_w, ah_ln1_b, ah_ln_eps);
        ggml_tensor * fc1 = ggml_add(ctx, ggml_mul_mat(ctx, ah_fc1_w, ln1), ah_fc1_b);
        ggml_tensor * h   = ggml_relu(ctx, fc1);

        ggml_tensor * b0_ln = layernorm(ctx, h, ah_b0_ln_w, ah_b0_ln_b, ah_ln_eps);
        ggml_tensor * b0    = ggml_add(ctx, ggml_mul_mat(ctx, ah_b0_w, b0_ln), ah_b0_b);
        h = ggml_add(ctx, h, ggml_relu(ctx, b0));

        ggml_tensor * b1_ln = layernorm(ctx, h, ah_b1_ln_w, ah_b1_ln_b, ah_ln_eps);
        ggml_tensor * b1    = ggml_add(ctx, ggml_mul_mat(ctx, ah_b1_w, b1_ln), ah_b1_b);
        h = ggml_add(ctx, h, ggml_relu(ctx, b1));

        ggml_tensor * ln2 = layernorm(ctx, h, ah_ln2_w, ah_ln2_b, ah_ln_eps);
        ggml_tensor * y   = ggml_add(ctx, ggml_mul_mat(ctx, ah_fc2_w, ln2), ah_fc2_b);
        ggml_set_name(y, "y");

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "vla(bitvla): gallocr failed (action_head)\n"); ggml_gallocr_free(galloc); ggml_free(ctx); return {}; }
        ggml_backend_tensor_set(x, last_hidden_at_actions.data(), 0, ggml_nbytes(x));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla): action_head compute failed\n"); ggml_gallocr_free(galloc); ggml_free(ctx); return {}; }
        ggml_backend_tensor_get(y, normalized_actions.data(), 0, (size_t) chunk * action_dim * sizeof(float));
        ggml_gallocr_free(galloc); ggml_free(ctx);
    }
    if (timing_phase) stats.ms_denoise = std::chrono::duration<float, std::milli>(clk::now() - t_d0).count();

    _dump_bin("ah_norm_actions", normalized_actions.data(), normalized_actions.size());
    _dump_manifest(std::string("ah_norm_actions fp32 1 ") + std::to_string(num_actions_chunk) +
                   " " + std::to_string(action_dim));

    std::vector<float> actions = std::move(normalized_actions);
    for (int64_t t = 0; t < chunk; ++t) {
        for (int64_t d = 0; d < action_dim; ++d) {
            const float a = actions[t * action_dim + d];
            if (unnorm_mask[d]) {
                actions[t * action_dim + d] = 0.5f * (a + 1.0f) * (q99[d] - q01[d] + 1e-8f) + q01[d];
            }

        }
    }

    stats.ms_inference = (timing_phase
        ? stats.ms_prefill + stats.ms_denoise
        : std::chrono::duration<float, std::milli>(clk::now() - t_start).count() - stats.ms_vision);
    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t_start).count();
    return actions;
}

}
