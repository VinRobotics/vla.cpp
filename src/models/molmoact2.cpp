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

// Ai2 MolmoAct2 (allenai/MolmoAct2-LIBERO and siblings), continuous actions. The reference is the
// checkpoint's modeling_molmoact2.py (MolmoAct2ForConditionalGeneration.predict_action):
//
//   views    each resized to 378x378 (torchvision bilinear, no antialias, rounded to uint8), [-1, 1]
//   tower    SigLIP-so400m/14, 25 layers; the outputs of layers 24 and 18 concatenated (2304 wide)
//   adapter  2x2 attention pooling (the query is the mean of the valid patches), SwiGLU into the LM:
//            196 tokens per view, ADDED to the <im_patch> token embeddings
//   LM       Qwen3-style, 36 layers, causal except among image tokens (bidirectional there); only its
//            per-layer K (after q/k-norm and RoPE) and V are used
//   expert   36 adaLN blocks, one per LM layer: self-attention (q/k RMSNorm, RoPE over the chunk),
//            cross-attention to that layer's projected, RMS-normed K/V (the <|im_end|> tokens masked),
//            SwiGLU; 10 Euler steps from noise at t=0 to the action at t=1
//   output   the first action_dim columns un-normalized with the baked norm_tag statistics
//
// The client builds the prompt (task, discretized state, setup and control text; see
// eval/client/vla_cpp_client.py) and sends its token ids with one <|image|> per view; each is
// expanded here into <im_start> <im_patch>x196 <im_end>. The time-only adaLN modulations are
// computed once at load, on the host, for the configured step count.

#include "arch.h"
#include "backend.h"
#include "gguf_reader.h"
#include "layers/attn.h"
#include "layers/ffn.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "loader.h"
#include "model.h"
#include "modules/encoder.h"
#include "modules/preprocess.h"
#include "modules/prompt.h"
#include "modules/qwen3_lm.h"
#include "options.h"
#include "scratch_ctx.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace vla {

namespace {

constexpr float kAeNormEps = 1e-6f;  // the expert's block, context and final RMSNorms (hard-coded upstream)
constexpr float kAeRopeBase = 10000.0f;

struct AeBlockW {
    ggml_tensor *Wqkv = nullptr, *bqkv = nullptr, *Wo = nullptr, *bo = nullptr;
    ggml_tensor *Wcq = nullptr, *bcq = nullptr, *Wco = nullptr, *bco = nullptr;
    ggml_tensor *Wg = nullptr, *bg = nullptr, *Wu = nullptr, *bu = nullptr, *Wd = nullptr, *bd = nullptr;
};

struct MainKey {
    int64_t nv = -1, seq = -1;
    bool    dump = false;
    bool operator==(const MainKey & o) const {
        return nv == o.nv && seq == o.seq && dump == o.dump;
    }
};

struct MainIO {
    ggml_tensor *patches = nullptr, *gather = nullptr, *inv_cnt = nullptr, *pool_mask = nullptr;
    ggml_tensor *vzero = nullptr, *lzero = nullptr, *embeds = nullptr, *scatter = nullptr, *pos = nullptr;
    ggml_tensor *lm_mask = nullptr, *ctx_mask = nullptr, *ae_pos = nullptr, *x0 = nullptr, *dim_mask = nullptr;
    ggml_tensor *actions = nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> dumps;
};

}  // namespace

struct Molmoact2ModelArch : public ModelArchBase {
    Molmoact2ModelArch() : ModelArchBase(Arch::MOLMOACT2) {}
    ~Molmoact2ModelArch() override;

    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_const   = nullptr;
    ggml_backend_buffer_t const_buf   = nullptr;
    ggml_type             matmul_type = GGML_TYPE_BF16;
    gguf_reader           io{"molmoact2"};

    int64_t image_size = 378, patch = 14, grid = 27, pool_h = 2, pool_w = 2, gh = 14, gw = 14;
    int64_t vit_hidden = 1152, vit_heads = 16, vit_head_dim = 72, vit_inter = 4304, vit_layers = 25;
    std::vector<int64_t> feat_layers;
    int64_t ad_hidden = 1152, ad_heads = 16, ad_head_dim = 72, ad_inter = 9728;
    int64_t lm_hidden = 2560, lm_layers = 36, n_q = 32, n_kv = 8, lm_head_dim = 128, lm_inter = 9728, vocab = 0;
    int64_t ae_hidden = 768, ae_heads = 8, ae_layers = 36, ae_inter = 3072, ae_time_dim = 256;
    int64_t max_action_dim = 32, horizon = 10, action_dim = 7, state_dim = 8, num_steps = 10;
    int64_t max_seq = 1024;
    float   vit_ln_eps = 1e-6f, lm_rms_eps = 1e-6f, qk_eps = 1e-6f;
    double  lm_rope_theta = 5e6;
    bool    mask_dims = true, mask_eos = true;

    int32_t tok_image_prompt = -1, tok_image_start = -1, tok_image_end = -1, tok_image_patch = -1;
    int32_t tok_bos = -1, tok_eos = -1, tok_action_start = -1, tok_action_end = -1;
    std::vector<int32_t> image_tokens;   // bidirectional attention among these

    std::string        norm_mode;
    std::vector<float> a_lo, a_hi, a_mask;

    ggml_tensor *patch_w = nullptr, *patch_b = nullptr, *pos_embd = nullptr;
    EncStack     vit;
    ggml_tensor *pq_w = nullptr, *pq_b = nullptr, *pk_w = nullptr, *pk_b = nullptr;
    ggml_tensor *pv_w = nullptr, *pv_b = nullptr, *po_w = nullptr, *po_b = nullptr;
    ggml_tensor *proj_gate = nullptr, *proj_up = nullptr, *proj_down = nullptr;
    Qwen3LM      lm;
    std::vector<AeBlockW> ae;
    ggml_tensor *act_in_w = nullptr, *act_in_b = nullptr, *ctx_k = nullptr, *ctx_v = nullptr;
    ggml_tensor *act_out_w = nullptr, *act_out_b = nullptr;
    // [9*ae_hidden, ae_layers, num_steps] and [2*ae_hidden, num_steps]: shift, 1+scale, gate per step.
    ggml_tensor *mod_tab = nullptr, *fmod_tab = nullptr;

    std::vector<int32_t> pool_idx;       // [gh*gw][pool_h*pool_w], -1 outside the grid
    std::vector<float>   pool_inv_cnt;   // 1 / valid patches per token

    graph_cache<MainKey, MainIO> mg;

    std::vector<float> predict(const Inputs & in) override;
};

namespace {

bool load_config(gguf_reader & g, const Options & opts, Molmoact2ModelArch & m) {
    auto U = [&](const char * k, int64_t & dst) {
        char key[96];
        std::snprintf(key, sizeof(key), "molmoact2.%s", k);
        if (!g.has(key)) {
            std::fprintf(stderr, "vla(molmoact2): GGUF lacks %s\n", key);
            return false;
        }
        dst = (int64_t) g.u32(key);
        return true;
    };
    auto T = [&](const char * k, int32_t & dst) {
        int64_t v = 0;
        if (!U(k, v))
            return false;
        dst = (int32_t) v;
        return true;
    };
    bool ok = true;
    ok &= U("image_size", m.image_size) && U("patch_size", m.patch) && U("pool_h", m.pool_h) && U("pool_w", m.pool_w);
    ok &= U("vit_hidden", m.vit_hidden) && U("vit_heads", m.vit_heads) && U("vit_head_dim", m.vit_head_dim);
    ok &= U("vit_inter", m.vit_inter) && U("vit_layers", m.vit_layers);
    ok &= U("adapter_hidden", m.ad_hidden) && U("adapter_heads", m.ad_heads) && U("adapter_head_dim", m.ad_head_dim);
    ok &= U("adapter_inter", m.ad_inter);
    ok &= U("lm_hidden", m.lm_hidden) && U("lm_layers", m.lm_layers) && U("lm_q_heads", m.n_q);
    ok &= U("lm_kv_heads", m.n_kv) && U("lm_head_dim", m.lm_head_dim) && U("lm_inter", m.lm_inter);
    ok &= U("vocab_size", m.vocab);
    ok &= U("aex_hidden", m.ae_hidden) && U("aex_heads", m.ae_heads) && U("aex_layers", m.ae_layers);
    ok &= U("aex_inter", m.ae_inter) && U("aex_time_dim", m.ae_time_dim);
    ok &= U("max_action_dim", m.max_action_dim) && U("action_horizon", m.horizon) && U("action_dim", m.action_dim);
    ok &= U("state_dim", m.state_dim) && U("num_steps", m.num_steps);
    ok &= T("image_prompt_id", m.tok_image_prompt) && T("image_start_id", m.tok_image_start);
    ok &= T("image_end_id", m.tok_image_end) && T("image_patch_id", m.tok_image_patch);
    ok &= T("bos_id", m.tok_bos) && T("eos_id", m.tok_eos);
    ok &= T("action_start_id", m.tok_action_start) && T("action_end_id", m.tok_action_end);
    int64_t n_feat = 0;
    ok &= U("vit_n_feature_layers", n_feat);
    if (!ok)
        return false;
    m.feat_layers.clear();
    for (int64_t i = 0; i < n_feat; ++i) {
        char k[48];
        std::snprintf(k, sizeof(k), "vit_feature_layer.%lld", (long long) i);
        int64_t l = 0;
        if (!U(k, l))
            return false;
        if (l < 0 || l >= m.vit_layers) {
            std::fprintf(stderr, "vla(molmoact2): feature layer %lld outside the %lld-layer tower\n",
                         (long long) l, (long long) m.vit_layers);
            return false;
        }
        m.feat_layers.push_back(l);
    }
    {
        const int64_t id = gguf_find_key(g.gctx, "molmoact2.image_token_ids");
        if (id < 0 || gguf_get_kv_type(g.gctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g.gctx, id) != GGUF_TYPE_INT32) {
            std::fprintf(stderr, "vla(molmoact2): GGUF lacks the int32 array molmoact2.image_token_ids\n");
            return false;
        }
        const int32_t * p = (const int32_t *) gguf_get_arr_data(g.gctx, id);
        m.image_tokens.assign(p, p + gguf_get_arr_n(g.gctx, id));
    }
    m.vit_ln_eps = g.f32("molmoact2.vit_ln_eps");
    m.lm_rms_eps = g.f32("molmoact2.lm_rms_eps");
    m.qk_eps     = g.f32("molmoact2.aex_qk_norm_eps");
    m.lm_rope_theta = g.f64("molmoact2.lm_rope_theta");
    m.norm_mode  = g.str("molmoact2.norm_mode");
    auto B = [&](const char * k, bool def) {
        const int64_t id = gguf_find_key(g.gctx, k);
        return id >= 0 && gguf_get_kv_type(g.gctx, id) == GGUF_TYPE_BOOL ? gguf_get_val_bool(g.gctx, id) : def;
    };
    m.mask_dims = B("molmoact2.mask_action_dim_padding", true);
    m.mask_eos  = B("molmoact2.encoder_mask_eos", true);

    if (m.norm_mode != "q01_q99" && m.norm_mode != "q10_q90" && m.norm_mode != "min_max" && m.norm_mode != "mean_std") {
        std::fprintf(stderr, "vla(molmoact2): unsupported norm_mode '%s'\n", m.norm_mode.c_str());
        return false;
    }
    if (m.image_size % m.patch != 0 || m.vit_hidden != m.vit_heads * m.vit_head_dim ||
        m.ad_hidden != m.ad_heads * m.ad_head_dim || m.ae_hidden % m.ae_heads != 0 ||
        m.ae_layers != m.lm_layers || m.horizon < 1 || m.action_dim < 1 || m.action_dim > m.max_action_dim ||
        (m.ae_hidden / m.ae_heads) % 2 != 0) {
        std::fprintf(stderr, "vla(molmoact2): inconsistent hyper-parameters in the GGUF\n");
        return false;
    }
    if (!resolve_num_steps("molmoact2", opts, m.num_steps))
        return false;

    m.grid = m.image_size / m.patch;
    m.gh   = (m.grid + m.pool_h - 1) / m.pool_h;
    m.gw   = (m.grid + m.pool_w - 1) / m.pool_w;

    m.vit.cfg.hidden   = m.vit_hidden;
    m.vit.cfg.heads    = m.vit_heads;
    m.vit.cfg.head_dim = m.vit_head_dim;
    m.vit.cfg.ln_eps   = m.vit_ln_eps;
    m.vit.cfg.flash_attn = flash_attn_enabled();

    m.lm.cfg.hidden   = m.lm_hidden;
    m.lm.cfg.layers   = m.lm_layers;
    m.lm.cfg.n_q      = m.n_q;
    m.lm.cfg.n_kv     = m.n_kv;
    m.lm.cfg.head_dim = m.lm_head_dim;
    m.lm.cfg.inter    = m.lm_inter;
    m.lm.cfg.rms_eps  = m.lm_rms_eps;
    m.lm.cfg.flash_attn = opts.flash_attn.value_or(false);
    m.lm.cfg.rope.type      = GGML_ROPE_TYPE_NEOX;
    m.lm.cfg.rope.n_dims    = (int) m.lm_head_dim;
    m.lm.cfg.rope.freq_base = (float) m.lm_rope_theta;

    Config & c = m.cfg;
    c = Config{};
    c.n_img = m.gh * m.gw;
    c.n_lang = m.max_seq;
    c.n_state = 0;
    c.n_suffix = m.horizon;
    c.hidden = m.lm_hidden;
    c.expert_h = m.ae_hidden;
    c.intermediate = m.lm_inter;
    c.expert_inter = m.ae_inter;
    c.n_q_heads = m.n_q;
    c.n_kv_heads = m.n_kv;
    c.head_dim = m.lm_head_dim;
    c.q_full_dim = m.n_q * m.lm_head_dim;
    c.kv_full_dim = m.n_kv * m.lm_head_dim;
    c.n_layers = m.lm_layers;
    // The state reaches the model as tokens in the prompt; the field is accepted and ignored.
    c.max_state_dim = m.state_dim;
    c.real_state_dim = m.state_dim;
    c.max_action_dim = m.max_action_dim;
    c.real_action_dim = m.action_dim;
    c.num_steps = (int) m.num_steps;
    c.rms_eps = m.lm_rms_eps;
    c.norm_eps = m.lm_rms_eps;
    c.rope_n_dims = (int) m.lm_head_dim;
    c.rope_mode = GGML_ROPE_TYPE_NEOX;
    c.rope_freq_base = (float) m.lm_rope_theta;
    c.denormalized = true;
    return true;
}

bool load_stats(gguf_reader & g, Molmoact2ModelArch & m) {
    m.a_lo   = g.read_f32("molmoact2.action_low");
    m.a_hi   = g.read_f32("molmoact2.action_high");
    m.a_mask = g.read_f32("molmoact2.action_mask");
    const size_t n = (size_t) m.action_dim;
    if (m.a_lo.size() != n || m.a_hi.size() != n || m.a_mask.size() != n) {
        std::fprintf(stderr, "vla(molmoact2): action statistics are not %zu wide\n", n);
        return false;
    }
    return true;
}

// The pooling layout of the processor's arange_for_pooling: the grid padded with -1 to a multiple of the
// pool (the extra row/column split floor/ceil before/after), then one token per pool_h x pool_w window.
void build_pool_layout(Molmoact2ModelArch & m) {
    const int64_t J = m.pool_h * m.pool_w;
    const int64_t top = (m.gh * m.pool_h - m.grid) / 2, left = (m.gw * m.pool_w - m.grid) / 2;
    m.pool_idx.assign((size_t) (m.gh * m.gw * J), -1);
    m.pool_inv_cnt.assign((size_t) (m.gh * m.gw), 1.0f);
    for (int64_t h = 0; h < m.gh; ++h)
        for (int64_t w = 0; w < m.gw; ++w) {
            const int64_t t = h * m.gw + w;
            int cnt = 0;
            for (int64_t dh = 0; dh < m.pool_h; ++dh)
                for (int64_t dw = 0; dw < m.pool_w; ++dw) {
                    const int64_t r = h * m.pool_h + dh - top, c = w * m.pool_w + dw - left;
                    if (r >= 0 && r < m.grid && c >= 0 && c < m.grid) {
                        m.pool_idx[(size_t) (t * J + dh * m.pool_w + dw)] = (int32_t) (r * m.grid + c);
                        ++cnt;
                    }
                }
            // sum / count, with count 1, 2 or 4: the reciprocal is exact.
            m.pool_inv_cnt[(size_t) t] = cnt > 0 ? 1.0f / (float) cnt : 1.0f;
        }
}

float silu_f(float x) {
    return x / (1.0f + std::exp(-x));
}

// out[s][r] = b[r] + W[r] . a[s] for every step s, W [rows x cols] row-major; double accumulation.
void gemv_steps(const std::vector<float> & W, const std::vector<float> & b, const std::vector<float> & a,
                int64_t rows, int64_t cols, int64_t steps, std::vector<float> & out, int n_threads) {
    out.assign((size_t) (steps * rows), 0.0f);
    auto work = [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const float * w = W.data() + r * cols;
            for (int64_t s = 0; s < steps; ++s) {
                const float * x = a.data() + s * cols;
                double acc = 0.0;
                for (int64_t k = 0; k < cols; ++k)
                    acc += (double) w[k] * (double) x[k];
                out[(size_t) (s * rows + r)] = (float) acc + b[(size_t) r];
            }
        }
    };
    const int nt = std::max(1, std::min<int>(n_threads, (int) (rows / 64 + 1)));
    std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i)
        th.emplace_back(work, rows * i / nt, rows * (i + 1) / nt);
    for (auto & t : th)
        t.join();
}

// The adaLN modulation of every block at every solver step depends on t alone, so it is computed once:
// the sinusoidal embedding, time MLP and SiLU, then each block's Linear. The scale chunks are stored as
// 1 + scale, the factor _modulate multiplies by.
bool build_mod_tables(Molmoact2ModelArch & m) {
    gguf_reader & g = m.io;
    const int64_t D = m.ae_hidden, TD = m.ae_time_dim, NS = m.num_steps, L = m.ae_layers;
    auto need = [&](const char * name, size_t n, std::vector<float> & v) {
        v = g.read_f32(name);
        if (v.size() != n) {
            std::fprintf(stderr, "vla(molmoact2): %s has %zu values, expected %zu\n", name, v.size(), n);
            return false;
        }
        return true;
    };
    std::vector<float> w1, b1, w2, b2;
    if (!need("aex.time_in.weight", (size_t) (D * TD), w1) || !need("aex.time_in.bias", (size_t) D, b1) ||
        !need("aex.time_out.weight", (size_t) (D * D), w2) || !need("aex.time_out.bias", (size_t) D, b2))
        return false;

    // SinusoidalTimeEmbedding: freq_i = exp(i * -ln(1e4) / (half - 1)), [sin(t f), cos(t f)], all f32.
    const int64_t half = TD / 2;
    const float   c = (float) (-std::log(10000.0) / (double) std::max<int64_t>(half - 1, 1));
    std::vector<float> emb((size_t) (NS * TD), 0.0f);
    for (int64_t s = 0; s < NS; ++s) {
        const float t = (float) ((double) s / (double) NS);
        for (int64_t i = 0; i < half; ++i) {
            const float arg = t * std::exp((float) i * c);
            emb[(size_t) (s * TD + i)]        = std::sin(arg);
            emb[(size_t) (s * TD + half + i)] = std::cos(arg);
        }
    }
    std::vector<float> h1, cond;
    gemv_steps(w1, b1, emb, D, TD, NS, h1, m.n_threads);
    for (float & v : h1)
        v = silu_f(v);
    gemv_steps(w2, b2, h1, D, D, NS, cond, m.n_threads);
    for (float & v : cond)
        v = silu_f(v);   // every modulation Linear reads SiLU(conditioning)

    const int64_t R = 9 * D;
    std::vector<float> tab((size_t) (NS * L * R)), ftab((size_t) (NS * 2 * D)), out, w, b;
    for (int64_t l = 0; l < L; ++l) {
        char wn[64], bn[64];
        std::snprintf(wn, sizeof(wn), "aex.blk.%lld.mod.weight", (long long) l);
        std::snprintf(bn, sizeof(bn), "aex.blk.%lld.mod.bias", (long long) l);
        if (!need(wn, (size_t) (R * D), w) || !need(bn, (size_t) R, b))
            return false;
        gemv_steps(w, b, cond, R, D, NS, out, m.n_threads);
        for (int64_t s = 0; s < NS; ++s)
            for (int64_t r = 0; r < R; ++r) {
                const int64_t chunk = r / D;
                const float   v = out[(size_t) (s * R + r)];
                tab[(size_t) ((s * L + l) * R + r)] = (chunk == 1 || chunk == 4 || chunk == 7) ? 1.0f + v : v;
            }
    }
    if (!need("aex.final_mod.weight", (size_t) (2 * D * D), w) || !need("aex.final_mod.bias", (size_t) (2 * D), b))
        return false;
    gemv_steps(w, b, cond, 2 * D, D, NS, out, m.n_threads);
    for (int64_t s = 0; s < NS; ++s)
        for (int64_t r = 0; r < 2 * D; ++r) {
            const float v = out[(size_t) (s * 2 * D + r)];
            ftab[(size_t) (s * 2 * D + r)] = r >= D ? 1.0f + v : v;
        }

    ggml_init_params p = { 4 * ggml_tensor_overhead(), nullptr, true };
    m.ctx_const = ggml_init(p);
    if (!m.ctx_const)
        return false;
    m.mod_tab  = ggml_new_tensor_3d(m.ctx_const, GGML_TYPE_F32, R, L, NS);
    m.fmod_tab = ggml_new_tensor_2d(m.ctx_const, GGML_TYPE_F32, 2 * D, NS);
    m.const_buf = alloc_weights(m.ctx_const, m.backend);
    if (!m.const_buf) {
        std::fprintf(stderr, "vla(molmoact2): cannot allocate the modulation tables\n");
        return false;
    }
    ggml_backend_tensor_set(m.mod_tab, tab.data(), 0, ggml_nbytes(m.mod_tab));
    ggml_backend_tensor_set(m.fmod_tab, ftab.data(), 0, ggml_nbytes(m.fmod_tab));
    return true;
}

// torchvision Resize(BILINEAR, antialias=False) as the MolmoAct2 processor runs it on a uint8 HWC array:
// F.interpolate (align_corners=False) in float, then round half to even back to uint8. With 3 channels
// PyTorch's CPU kernel is the generic separable one (upsample_generic_Nd_kernel_impl): each row
// interpolated along the width, then the two rows along the height, unfused. A float [0, 1] view is scaled
// to [0, 255] first.
void resize_u8(const ImageView & v, int64_t side, std::vector<uint8_t> & out) {
    struct Tap { int64_t i0, i1; float l0, l1; };
    auto taps = [](int64_t in, int64_t n) {
        std::vector<Tap> t((size_t) n);
        const float scale = (float) in / (float) n;
        for (int64_t o = 0; o < n; ++o) {
            if (in == n) {
                t[(size_t) o] = { o, o, 1.0f, 0.0f };
                continue;
            }
            float real = scale * ((float) o + 0.5f) - 0.5f;
            if (real < 0.0f)
                real = 0.0f;
            const int64_t i0 = std::min<int64_t>((int64_t) std::floor(real), in - 1);
            const float   l1 = std::min(std::max(real - (float) i0, 0.0f), 1.0f);
            t[(size_t) o] = { i0, i0 + (i0 < in - 1 ? 1 : 0), 1.0f - l1, l1 };
        }
        return t;
    };
    const std::vector<Tap> th = taps(v.h, side), tw = taps(v.w, side);
    auto px = [&](int64_t r, int64_t c, int ch) -> float {
        const size_t i = (size_t) ((r * v.w + c) * 3 + ch);
        return v.format == PixelFormat::U8 ? (float) ((const uint8_t *) v.data)[i] : ((const float *) v.data)[i] * 255.0f;
    };
    out.resize((size_t) (side * side * 3));
    for (int64_t oh = 0; oh < side; ++oh) {
        const Tap & a = th[(size_t) oh];
        for (int64_t ow = 0; ow < side; ++ow) {
            const Tap & b = tw[(size_t) ow];
            for (int ch = 0; ch < 3; ++ch) {
                const float r0  = px(a.i0, b.i0, ch) * b.l0 + px(a.i0, b.i1, ch) * b.l1;
                const float r1  = px(a.i1, b.i0, ch) * b.l0 + px(a.i1, b.i1, ch) * b.l1;
                const float val = r0 * a.l0 + r1 * a.l1;
                out[(size_t) ((oh * side + ow) * 3 + ch)] =
                    (uint8_t) std::min(255.0f, std::max(0.0f, std::nearbyint(val)));
            }
        }
    }
}

}  // namespace

Molmoact2ModelArch::~Molmoact2ModelArch() {
    mg.release();
    if (weight_buf)
        ggml_backend_buffer_free(weight_buf);
    if (ctx_weights)
        ggml_free(ctx_weights);
    if (const_buf)
        ggml_backend_buffer_free(const_buf);
    if (ctx_const)
        ggml_free(ctx_const);
    if (backend)
        ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> molmoact2_create(const std::string & mmproj_path,
                                                const std::string & ckpt_path,
                                                const std::string &,
                                                const Options & opts) {
    if (!mmproj_path.empty())
        std::printf("vla(molmoact2): note - mmproj '%s' is ignored (the vision tower is bundled in the GGUF)\n",
                    mmproj_path.c_str());
    auto m = std::make_unique<Molmoact2ModelArch>();
    m->matmul_type = opts.weight_dtype.value_or(default_weight_dtype(GGML_TYPE_BF16));
    if (!m->io.open(ckpt_path))
        return nullptr;
    gguf_reader & g = m->io;
    if (!g.has("molmoact2.architecture")) {
        std::fprintf(stderr, "vla(molmoact2): %s is not a molmoact2 GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, opts, *m) || !load_stats(g, *m))
        return nullptr;
    build_pool_layout(*m);

    std::printf("vla(molmoact2): vit=SigLIP %lldd x %lldL @%lldpx (features of layers",
                (long long) m->vit_hidden, (long long) m->vit_layers, (long long) m->image_size);
    for (int64_t l : m->feat_layers)
        std::printf(" %lld", (long long) l);
    std::printf(")  pool=%lldx%lld -> %lld tok/view  lm=%lldd x %lldL (%lldq/%lldkv x %lld, theta=%g)  "
                "expert=%lldd x %lldL x %lldh  horizon=%lld action_dim=%lld/%lld steps=%lld norm=%s  resident=%s\n",
                (long long) m->pool_h, (long long) m->pool_w, (long long) (m->gh * m->gw), (long long) m->lm_hidden,
                (long long) m->lm_layers, (long long) m->n_q, (long long) m->n_kv, (long long) m->lm_head_dim,
                m->lm_rope_theta, (long long) m->ae_hidden, (long long) m->ae_layers, (long long) m->ae_heads,
                (long long) m->horizon, (long long) m->action_dim, (long long) m->max_action_dim,
                (long long) m->num_steps, m->norm_mode.c_str(), dtype_name(m->matmul_type));

    const Backend b = backend_init("vla(molmoact2)", m->n_threads);
    if (!b.handle)
        return nullptr;
    m->backend = b.handle;

    ggml_init_params wp = { (size_t) 32 * 1024 * 1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) {
        std::fprintf(stderr, "vla(molmoact2): ggml_init(ctx_weights) failed\n");
        return nullptr;
    }
    WeightLoader L("molmoact2", g, m->ctx_weights, m->matmul_type);
    m->patch_w  = L.gemm("vit.patch_embd.weight");
    m->patch_b  = L.f32("vit.patch_embd.bias");
    m->pos_embd = L.f32("vit.pos_embd");
    m->vit.declare(L, "vit.blk", m->vit_layers);
    m->pq_w = L.gemm("mm.pool.attn_q.weight"); m->pq_b = L.f32("mm.pool.attn_q.bias");
    m->pk_w = L.gemm("mm.pool.attn_k.weight"); m->pk_b = L.f32("mm.pool.attn_k.bias");
    m->pv_w = L.gemm("mm.pool.attn_v.weight"); m->pv_b = L.f32("mm.pool.attn_v.bias");
    m->po_w = L.gemm("mm.pool.attn_o.weight"); m->po_b = L.f32("mm.pool.attn_o.bias");
    m->proj_gate = L.gemm("mm.proj.ffn_gate.weight");
    m->proj_up   = L.gemm("mm.proj.ffn_up.weight");
    m->proj_down = L.gemm("mm.proj.ffn_down.weight");
    m->lm.declare(L, "vlm");
    m->act_in_w  = L.gemm("aex.action_in.weight");  m->act_in_b  = L.f32("aex.action_in.bias");
    m->act_out_w = L.gemm("aex.action_out.weight"); m->act_out_b = L.f32("aex.action_out.bias");
    m->ctx_k = L.gemm("aex.ctx_k.weight");
    m->ctx_v = L.gemm("aex.ctx_v.weight");
    m->ae.resize((size_t) m->ae_layers);
    for (int64_t i = 0; i < m->ae_layers; ++i) {
        AeBlockW & w = m->ae[(size_t) i];
        const long long ii = (long long) i;
        w.Wqkv = L.gemm("aex.blk.%lld.attn_qkv.weight", ii); w.bqkv = L.f32("aex.blk.%lld.attn_qkv.bias", ii);
        w.Wo   = L.gemm("aex.blk.%lld.attn_o.weight", ii);   w.bo   = L.f32("aex.blk.%lld.attn_o.bias", ii);
        w.Wcq  = L.gemm("aex.blk.%lld.cross_q.weight", ii);  w.bcq  = L.f32("aex.blk.%lld.cross_q.bias", ii);
        w.Wco  = L.gemm("aex.blk.%lld.cross_o.weight", ii);  w.bco  = L.f32("aex.blk.%lld.cross_o.bias", ii);
        w.Wg   = L.gemm("aex.blk.%lld.ffn_gate.weight", ii); w.bg   = L.f32("aex.blk.%lld.ffn_gate.bias", ii);
        w.Wu   = L.gemm("aex.blk.%lld.ffn_up.weight", ii);   w.bu   = L.f32("aex.blk.%lld.ffn_up.bias", ii);
        w.Wd   = L.gemm("aex.blk.%lld.ffn_down.weight", ii); w.bd   = L.f32("aex.blk.%lld.ffn_down.bias", ii);
    }
    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;
    // Shapes the graph relies on, checked once the tensors are declared.
    if (m->patch_w->ne[0] != 3 * m->patch * m->patch || m->pos_embd->ne[0] != m->vit_hidden ||
        m->pos_embd->ne[1] != m->grid * m->grid ||
        m->pq_w->ne[0] != m->vit_hidden * (int64_t) m->feat_layers.size() || m->pq_w->ne[1] != m->ad_hidden ||
        m->proj_down->ne[1] != m->lm_hidden || m->ctx_k->ne[0] != m->n_kv * m->lm_head_dim ||
        m->ctx_k->ne[1] != m->ae_hidden || m->act_in_w->ne[0] != m->max_action_dim ||
        m->act_out_w->ne[1] != m->max_action_dim || m->ae[0].Wqkv->ne[1] != 3 * m->ae_hidden) {
        std::fprintf(stderr, "vla(molmoact2): tensor shapes do not match the GGUF hyper-parameters\n");
        return nullptr;
    }
    if (!build_mod_tables(*m))
        return nullptr;
    std::printf("vla(molmoact2): weights resident in %.2f GiB (%s); token embeddings read from the GGUF\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0), dtype_name(m->matmul_type));
    return m;
}

std::vector<float> Molmoact2ModelArch::predict(const Inputs & in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};
    const char * dump_dir = std::getenv("VLA_MOLMOACT2_DUMP");

    if (in.precomputed_img_emb && in.n_img_views > 0) {
        std::fprintf(stderr, "vla(molmoact2): precomputed image embeddings are not supported\n");
        return {};
    }
    if (!in.images || in.n_images < 1) {
        std::fprintf(stderr, "vla(molmoact2): no images in the request\n");
        return {};
    }
    if (!in.lang_tokens || in.n_lang < 1) {
        std::fprintf(stderr, "vla(molmoact2): empty lang_tokens\n");
        return {};
    }
    const int64_t nv = in.n_images, P = grid * grid, PD = 3 * patch * patch, NT = gh * gw, T = nv * NT;
    const int64_t J = pool_h * pool_w, AD = max_action_dim, H = horizon;

    // Views: resize, then patchify in the processor's (row, col, channel) order.
    const auto tv0 = std::chrono::steady_clock::now();
    std::vector<float> patches((size_t) (nv * P * PD)), one;
    std::vector<uint8_t> resized;
    for (int64_t v = 0; v < nv; ++v) {
        const ImageView & iv = in.images[v];
        if (!iv.data || iv.w < 1 || iv.h < 1) {
            std::fprintf(stderr, "vla(molmoact2): image view %lld is empty\n", (long long) v);
            return {};
        }
        resize_u8(iv, image_size, resized);
        const ImageView rv{ resized.data(), (int) image_size, (int) image_size, PixelFormat::U8 };
        if (!preprocess_image_patches("molmoact2", rv, image_size, patch, one))
            return {};
        std::memcpy(patches.data() + v * P * PD, one.data(), (size_t) (P * PD) * sizeof(float));
    }
    stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tv0).count();

    // Prompt: BOS if the client left it out (the processor's insert_bos), then each <|image|> expanded.
    std::vector<int32_t> src(in.lang_tokens, in.lang_tokens + in.n_lang);
    if (src.front() != tok_bos)
        src.insert(src.begin(), tok_bos);
    int64_t slots = 0;
    for (int32_t t : src)
        slots += t == tok_image_prompt;
    if (slots != 0 && slots != nv) {
        std::fprintf(stderr, "vla(molmoact2): the prompt has %lld <|image|> slots for %lld views\n",
                     (long long) slots, (long long) nv);
        return {};
    }
    std::vector<int32_t> ids;
    ids.reserve(src.size() + (size_t) (nv * (NT + 2)));
    auto image_block = [&]() {
        ids.push_back(tok_image_start);
        ids.insert(ids.end(), (size_t) NT, tok_image_patch);
        ids.push_back(tok_image_end);
    };
    for (size_t i = 0; i < src.size(); ++i) {
        if (slots == 0 && i == 1)
            for (int64_t v = 0; v < nv; ++v)   // no slots: the views go right after BOS
                image_block();
        if (src[i] == tok_image_prompt)
            image_block();
        else
            ids.push_back(src[i]);
    }
    if (slots == 0 && src.size() == 1)
        for (int64_t v = 0; v < nv; ++v)
            image_block();
    const int64_t S = (int64_t) ids.size();
    if (S > max_seq + nv * (NT + 2)) {
        std::fprintf(stderr, "vla(molmoact2): prompt too long (%lld tokens)\n", (long long) S);
        return {};
    }
    for (int32_t t : ids)
        if (t < 0 || t >= vocab) {
            std::fprintf(stderr, "vla(molmoact2): token %d outside the %lld-entry vocabulary\n", t, (long long) vocab);
            return {};
        }

    std::vector<int32_t> scatter((size_t) S, (int32_t) T), pos((size_t) S);
    std::vector<uint8_t> is_img((size_t) S, 0);
    int64_t k_img = 0;
    for (int64_t p = 0; p < S; ++p) {
        pos[(size_t) p] = (int32_t) p;
        if (ids[(size_t) p] == tok_image_patch)
            scatter[(size_t) p] = (int32_t) k_img++;
        is_img[(size_t) p] = std::find(image_tokens.begin(), image_tokens.end(), ids[(size_t) p]) != image_tokens.end();
    }
    if (k_img != T) {
        std::fprintf(stderr, "vla(molmoact2): %lld <im_patch> tokens for %lld image features\n",
                     (long long) k_img, (long long) T);
        return {};
    }
    // LM: causal, and bidirectional among the image tokens (token_type_ids).
    std::vector<float> lm_mask((size_t) (S * S));
    for (int64_t q = 0; q < S; ++q)
        for (int64_t k = 0; k < S; ++k)
            lm_mask[(size_t) (q * S + k)] = (k <= q || (is_img[(size_t) q] && is_img[(size_t) k])) ? 0.0f : -INFINITY;
    // Expert cross-attention: _get_encoder_attention_mask drops <|im_end|> (the eos, BOS included) and any
    // <action_start>..<action_end> span when the checkpoint was trained on both action modes.
    std::vector<uint8_t> keep((size_t) S, 1);
    if (mask_eos) {
        for (int64_t p = 0; p < S; ++p)
            if (ids[(size_t) p] == tok_eos)
                keep[(size_t) p] = 0;
        for (int64_t p = 0; p < S; ++p) {
            if (ids[(size_t) p] != tok_action_start)
                continue;
            int64_t e = p;
            while (e < S && ids[(size_t) e] != tok_action_end)
                ++e;
            for (int64_t q = p; q <= std::min(e, S - 1); ++q)
                keep[(size_t) q] = 0;
            p = e;
        }
    }
    std::vector<float> ctx_mask((size_t) (S * H));
    for (int64_t q = 0; q < H; ++q)
        for (int64_t k = 0; k < S; ++k)
            ctx_mask[(size_t) (q * S + k)] = keep[(size_t) k] ? 0.0f : -INFINITY;

    std::vector<float> embeds((size_t) (S * lm_hidden));
    if (!io.fetch_rows_f32("token_embd.weight", ids, embeds.data(), lm_hidden))
        return {};

    std::vector<int32_t> gather((size_t) (J * T));
    for (int64_t v = 0; v < nv; ++v)
        for (int64_t i = 0; i < NT * J; ++i) {
            const int32_t p = pool_idx[(size_t) i];
            gather[(size_t) (v * NT * J + i)] = p < 0 ? (int32_t) (nv * P) : (int32_t) (v * P + p);
        }
    std::vector<float> inv_cnt((size_t) T), pmask((size_t) (J * ad_heads * T));
    for (int64_t t = 0; t < T; ++t) {
        inv_cnt[(size_t) t] = pool_inv_cnt[(size_t) (t % NT)];
        for (int64_t h = 0; h < ad_heads; ++h)
            for (int64_t j = 0; j < J; ++j)
                pmask[(size_t) ((t * ad_heads + h) * J + j)] = pool_idx[(size_t) ((t % NT) * J + j)] < 0 ? -INFINITY : 0.0f;
    }
    std::vector<int32_t> ae_pos((size_t) H);
    for (int64_t i = 0; i < H; ++i)
        ae_pos[(size_t) i] = (int32_t) i;
    std::vector<float> dim_mask((size_t) AD, 1.0f);
    if (mask_dims)
        for (int64_t d = action_dim; d < AD; ++d)
            dim_mask[(size_t) d] = 0.0f;
    std::vector<float> x0;
    init_noise(in, (size_t) (H * AD), x0);

    const MainKey key{ nv, S, dump_dir != nullptr };
    const size_t nodes = 16384 + (size_t) num_steps * (size_t) (ae_layers * 96 + 64);
    const bool built = mg.ensure(backend, key, nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false),
                                 [&](ggml_context * C, MainIO & o) -> ggml_cgraph * {
        auto input = [&](ggml_tensor * t) {
            ggml_set_input(t);
            return t;
        };
        auto keep_out = [&](const char * name, ggml_tensor * t) {
            if (dump_dir) {
                ggml_set_output(t);
                o.dumps.emplace_back(name, t);
            }
        };
        const int64_t F = vit_hidden * (int64_t) feat_layers.size();
        o.patches   = input(ggml_new_tensor_3d(C, GGML_TYPE_F32, PD, P, nv));
        o.gather    = input(ggml_new_tensor_1d(C, GGML_TYPE_I32, J * T));
        o.inv_cnt   = input(ggml_new_tensor_2d(C, GGML_TYPE_F32, 1, T));
        o.pool_mask = input(ggml_new_tensor_4d(C, GGML_TYPE_F32, J, 1, ad_heads, T));
        o.vzero     = input(ggml_new_tensor_2d(C, GGML_TYPE_F32, F, 1));
        o.lzero     = input(ggml_new_tensor_2d(C, GGML_TYPE_F32, lm_hidden, 1));
        o.embeds    = input(ggml_new_tensor_2d(C, GGML_TYPE_F32, lm_hidden, S));
        o.scatter   = input(ggml_new_tensor_1d(C, GGML_TYPE_I32, S));
        o.pos       = input(ggml_new_tensor_1d(C, GGML_TYPE_I32, S));
        o.lm_mask   = input(ggml_new_tensor_2d(C, GGML_TYPE_F32, S, S));
        o.ctx_mask  = input(ggml_new_tensor_2d(C, GGML_TYPE_F32, S, H));
        o.ae_pos    = input(ggml_new_tensor_1d(C, GGML_TYPE_I32, H));
        o.x0        = input(ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, H));
        o.dim_mask  = input(ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, 1));

        // Tower: every view in one batch; keep the outputs of the adapter's layers.
        ggml_tensor * x = ggml_add(C, linear(C, patch_w, patch_b, o.patches), pos_embd);
        std::vector<ggml_tensor *> feats(feat_layers.size(), nullptr);
        const int64_t last = *std::max_element(feat_layers.begin(), feat_layers.end());
        for (int64_t i = 0; i <= last; ++i) {
            x = vit.block(C, vit.blk[(size_t) i], x, P, nv);
            for (size_t f = 0; f < feat_layers.size(); ++f)
                if (feat_layers[f] == i)
                    feats[f] = x;
        }
        ggml_tensor * fcat = feats[0];
        for (size_t f = 1; f < feats.size(); ++f)
            fcat = ggml_concat(C, fcat, feats[f], 0);
        fcat = ggml_concat(C, ggml_reshape_2d(C, fcat, F, P * nv), o.vzero, 1);   // last column: zeros

        // Attention pooling: K/V are the window's patches (zeros outside the grid, masked), the query their mean.
        const int64_t ahd = ad_head_dim, ah = ad_heads;
        ggml_tensor * G  = ggml_get_rows(C, fcat, o.gather);                                  // [F, J*T]
        ggml_tensor * Gp = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, G, F, J, T), 1, 0, 2, 3));  // [J, F, T]
        ggml_tensor * query = ggml_mul(C, ggml_reshape_2d(C, ggml_sum_rows(C, Gp), F, T), o.inv_cnt);
        ggml_tensor * Q = ggml_reshape_4d(C, linear(C, pq_w, pq_b, query), ahd, 1, ah, T);
        ggml_tensor * K = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, linear(C, pk_w, pk_b, G), ahd, ah, J, T), 0, 2, 1, 3));
        ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, linear(C, pv_w, pv_b, G), ahd, ah, J, T), 1, 2, 0, 3));
        ggml_tensor * kq = ggml_mul_mat(C, K, Q);                                              // [J, 1, ah, T]
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        ggml_tensor * pw = ggml_soft_max_ext(C, kq, o.pool_mask, 1.0f / std::sqrt((float) ahd), 0.0f);
        ggml_tensor * pooled = linear(C, po_w, po_b, ggml_reshape_2d(C, ggml_mul_mat(C, V, pw), ad_hidden, T));
        ggml_tensor * img = ggml_mul_mat(C, proj_down, ggml_mul(C, ggml_silu(C, ggml_mul_mat(C, proj_gate, pooled)),
                                                                   ggml_mul_mat(C, proj_up, pooled)));   // [lm_hidden, T]
        keep_out("img_emb", img);

        // LM: image features added onto the <im_patch> embeddings; only the per-layer K/V are used.
        ggml_tensor * h = ggml_add(C, o.embeds, ggml_get_rows(C, ggml_concat(C, img, o.lzero, 1), o.scatter));
        std::vector<Qwen3KV> kv((size_t) lm_layers);
        for (int64_t i = 0; i < lm_layers; ++i)
            h = lm.block(C, lm.blk[(size_t) i], h, o.pos, o.lm_mask, S, &kv[(size_t) i]);
        keep_out("lm_k0", kv[0].k);
        keep_out("lm_v_last", kv[(size_t) lm_layers - 1].v);

        // Expert context, per layer: K and V through context_{k,v}_proj and the context RMSNorm, K also
        // through the cross-attention's per-head k_norm.
        const int64_t ehd = ae_hidden / ae_heads, eh = ae_heads, D = ae_hidden;
        std::vector<ggml_tensor *> Kc((size_t) ae_layers), Vc((size_t) ae_layers);
        for (int64_t l = 0; l < ae_layers; ++l) {
            ggml_tensor * kf = ggml_reshape_2d(C, kv[(size_t) l].k, n_kv * lm_head_dim, S);
            ggml_tensor * vf = ggml_reshape_2d(C, kv[(size_t) l].v, n_kv * lm_head_dim, S);
            ggml_tensor * ck = ggml_rms_norm(C, ggml_reshape_3d(C, ggml_rms_norm(C, ggml_mul_mat(C, ctx_k, kf), kAeNormEps),
                                                                ehd, eh, S), qk_eps);
            ggml_tensor * cv = ggml_reshape_3d(C, ggml_rms_norm(C, ggml_mul_mat(C, ctx_v, vf), kAeNormEps), ehd, eh, S);
            Kc[(size_t) l] = ggml_cont(C, ggml_permute(C, ck, 0, 2, 1, 3));   // [ehd, S, eh]
            Vc[(size_t) l] = ggml_cont(C, ggml_permute(C, cv, 1, 2, 0, 3));   // [S, ehd, eh]
        }

        const float scale = 1.0f / std::sqrt((float) ehd), dt = 1.0f / (float) num_steps;
        auto modulate = [&](ggml_tensor * t, ggml_tensor * shift, ggml_tensor * scale1) {
            return ggml_add(C, ggml_mul(C, ggml_rms_norm(C, t, kAeNormEps), scale1), shift);
        };
        auto heads_q = [&](ggml_tensor * t) {   // [ehd, eh, H] -> [ehd, H, eh]
            return ggml_cont(C, ggml_permute(C, t, 0, 2, 1, 3));
        };
        auto rope_ae = [&](ggml_tensor * t) {
            return ggml_rope_ext(C, t, o.ae_pos, nullptr, (int) ehd, GGML_ROPE_TYPE_NEOX, 0, kAeRopeBase, 1.0f,
                                 0.0f, 1.0f, 0.0f, 0.0f);
        };
        ggml_tensor * traj = mask_dims ? ggml_mul(C, o.x0, o.dim_mask) : o.x0;
        const size_t es = sizeof(float);
        for (int64_t s = 0; s < num_steps; ++s) {
            ggml_tensor * xa = linear(C, act_in_w, act_in_b, traj);   // [D, H]
            for (int64_t l = 0; l < ae_layers; ++l) {
                const AeBlockW & w = ae[(size_t) l];
                const size_t base = (size_t) ((s * ae_layers + l) * 9 * D) * es;
                auto mod = [&](int c) {
                    return ggml_view_1d(C, mod_tab, D, base + (size_t) (c * D) * es);
                };
                ggml_tensor * qkv = linear(C, w.Wqkv, w.bqkv, modulate(xa, mod(0), mod(1)));   // [3D, H]
                auto part = [&](int i) {
                    return ggml_cont(C, ggml_view_3d(C, qkv, ehd, eh, H, ehd * es, qkv->nb[1], (size_t) (i * D) * es));
                };
                ggml_tensor * q = heads_q(rope_ae(ggml_rms_norm(C, part(0), qk_eps)));
                ggml_tensor * k = heads_q(rope_ae(ggml_rms_norm(C, part(1), qk_eps)));
                ggml_tensor * v = ggml_cont(C, ggml_permute(C, part(2), 1, 2, 0, 3));
                ggml_tensor * att = attention(C, q, k, v, nullptr, scale, D, H);
                xa = ggml_add(C, xa, ggml_mul(C, linear(C, w.Wo, w.bo, att), mod(2)));

                ggml_tensor * cq = linear(C, w.Wcq, w.bcq, modulate(xa, mod(3), mod(4)));
                cq = heads_q(ggml_rms_norm(C, ggml_reshape_3d(C, cq, ehd, eh, H), qk_eps));
                att = attention(C, cq, Kc[(size_t) l], Vc[(size_t) l], o.ctx_mask, scale, D, H);
                xa = ggml_add(C, xa, ggml_mul(C, linear(C, w.Wco, w.bco, att), mod(5)));

                ggml_tensor * ff = ffn_swiglu(C, w.Wg, w.bg, w.Wu, w.bu, w.Wd, w.bd, modulate(xa, mod(6), mod(7)));
                xa = ggml_add(C, xa, ggml_mul(C, ff, mod(8)));
            }
            const size_t fb = (size_t) (s * 2 * D) * es;
            ggml_tensor * vel = linear(C, act_out_w, act_out_b,
                                       modulate(xa, ggml_view_1d(C, fmod_tab, D, fb),
                                                ggml_view_1d(C, fmod_tab, D, fb + (size_t) D * es)));   // [AD, H]
            if (mask_dims)
                vel = ggml_mul(C, vel, o.dim_mask);
            if (s == 0)
                keep_out("velocity0", vel);
            traj = ggml_add(C, traj, ggml_scale(C, vel, dt));
        }
        o.actions = traj;
        ggml_set_output(traj);
        ggml_cgraph * gf = ggml_new_graph_custom(C, nodes, false);
        ggml_build_forward_expand(gf, traj);
        for (auto & d : o.dumps)
            ggml_build_forward_expand(gf, d.second);
        return gf;
    });
    if (!built) {
        std::fprintf(stderr, "vla(molmoact2): graph build failed\n");
        return {};
    }
    MainIO & o = mg.io();
    const std::vector<float> vzero((size_t) o.vzero->ne[0], 0.0f), lzero((size_t) lm_hidden, 0.0f);
    ggml_backend_tensor_set(o.patches, patches.data(), 0, ggml_nbytes(o.patches));
    ggml_backend_tensor_set(o.gather, gather.data(), 0, ggml_nbytes(o.gather));
    ggml_backend_tensor_set(o.inv_cnt, inv_cnt.data(), 0, ggml_nbytes(o.inv_cnt));
    ggml_backend_tensor_set(o.pool_mask, pmask.data(), 0, ggml_nbytes(o.pool_mask));
    ggml_backend_tensor_set(o.vzero, vzero.data(), 0, ggml_nbytes(o.vzero));
    ggml_backend_tensor_set(o.lzero, lzero.data(), 0, ggml_nbytes(o.lzero));
    ggml_backend_tensor_set(o.embeds, embeds.data(), 0, ggml_nbytes(o.embeds));
    ggml_backend_tensor_set(o.scatter, scatter.data(), 0, ggml_nbytes(o.scatter));
    ggml_backend_tensor_set(o.pos, pos.data(), 0, ggml_nbytes(o.pos));
    ggml_backend_tensor_set(o.lm_mask, lm_mask.data(), 0, ggml_nbytes(o.lm_mask));
    ggml_backend_tensor_set(o.ctx_mask, ctx_mask.data(), 0, ggml_nbytes(o.ctx_mask));
    ggml_backend_tensor_set(o.ae_pos, ae_pos.data(), 0, ggml_nbytes(o.ae_pos));
    ggml_backend_tensor_set(o.x0, x0.data(), 0, ggml_nbytes(o.x0));
    ggml_backend_tensor_set(o.dim_mask, dim_mask.data(), 0, ggml_nbytes(o.dim_mask));

    ggml_cgraph * gf = mg.graph();
    graph_unique_names(gf);
    const auto tc0 = std::chrono::steady_clock::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    stats.ms_inference = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tc0).count();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(molmoact2): graph compute failed (%d)\n", (int) st);
        mg.release();
        return {};
    }

    std::vector<float> out((size_t) (H * AD));
    ggml_backend_tensor_get(o.actions, out.data(), 0, out.size() * sizeof(float));
    if (dump_dir) {
        for (auto & d : o.dumps) {
            std::vector<float> buf((size_t) ggml_nelements(d.second));
            ggml_backend_tensor_get(d.second, buf.data(), 0, buf.size() * sizeof(float));
            const std::string path = std::string(dump_dir) + "/" + d.first + ".f32";
            if (FILE * fp = std::fopen(path.c_str(), "wb")) {
                std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
                std::fclose(fp);
            }
        }
        auto host = [&](const char * name, const void * data, size_t bytes) {
            const std::string path = std::string(dump_dir) + "/" + name;
            if (FILE * fp = std::fopen(path.c_str(), "wb")) {
                std::fwrite(data, 1, bytes, fp);
                std::fclose(fp);
            }
        };
        host("actions_raw.f32", out.data(), out.size() * sizeof(float));
        host("patches.f32", patches.data(), patches.size() * sizeof(float));
        host("input_ids.i32", ids.data(), ids.size() * sizeof(int32_t));
    }

    // _FeatureNormalizer.unnormalize: clip to [-1, 1] for the range modes, then the affine map on the
    // masked dims; unmasked dims (the gripper) keep the clipped value.
    const bool range = norm_mode != "mean_std";
    for (int64_t t = 0; t < H; ++t) {
        float * row = out.data() + t * AD;
        for (int64_t d = 0; d < AD; ++d) {
            if (d >= action_dim) {
                row[d] = 0.0f;
                continue;
            }
            const float a = range ? std::min(1.0f, std::max(-1.0f, row[d])) : row[d];
            const float lo = a_lo[(size_t) d], hi = a_hi[(size_t) d];
            const float u = range ? (a + 1.0f) * (hi - lo) / 2.0f + lo : a * hi + lo;
            row[d] = a_mask[(size_t) d] != 0.0f ? u : a;
        }
    }
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace vla
