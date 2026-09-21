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
#include "backend.h"
#include "gguf_reader.h"
#include "loader.h"
#include "model.h"
#include "models/octo.h"
#include "modules/preprocess.h"
#include "scratch_ctx.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "nlohmann/json.hpp"
#include "sentencepiece_processor.h"

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace vla {
namespace {

constexpr int kHidden     = 384;
constexpr int kPatchEmbed = 512;
constexpr int kTaskTokens = 16;

// The obs/task position-embedding tables are converted as one max-horizon slab,
// the same size for every checkpoint; window_size indexes into it.
constexpr int kMaxHorizon = 10;

// LowdimObsTokenizer emits one token per state dimension, and octo-small-1.5's
// proprio is 7-dim in every checkpoint that has one.
constexpr int kProprioTokens = 7;

// A caller that supplies Inputs::noise is replaying a chunk, so the reverse
// process has to draw the same per-step noise too, not just the same sample.
constexpr uint32_t kReplaySeed = 20260921u;

struct OctoRuntime {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w   = nullptr;
    std::unordered_map<std::string, ggml_tensor *> by_name;

    // One cache per camera: the views differ in side and token count, so they
    // cannot share a graph.
    struct ObsKey {
        int side  = -1;
        int n_tok = -1;
        int steps = -1;
        bool operator==(const ObsKey& o) const {
            return side == o.side && n_tok == o.n_tok && steps == o.steps;
        }
    };
    struct ObsIO {
        ggml_tensor * input = nullptr;
        ggml_tensor * rows  = nullptr;
        ggml_tensor * pos   = nullptr;
    };
    graph_cache<ObsKey, ObsIO> obs_primary, obs_wrist;

    struct ProprioKey {
        int in_dim = -1;
        int steps  = -1;
        bool operator==(const ProprioKey& o) const {
            return in_dim == o.in_dim && steps == o.steps;
        }
    };
    struct ProprioIO {
        ggml_tensor * tokens = nullptr;
        ggml_tensor * rows   = nullptr;
        ggml_tensor * pos    = nullptr;
    };
    graph_cache<ProprioKey, ProprioIO> proprio;

    struct T5Key {
        int seq = -1;
        bool operator==(const T5Key& o) const {
            return seq == o.seq;
        }
    };
    struct T5IO {
        ggml_tensor * ids     = nullptr;
        ggml_tensor * bucket  = nullptr;
        ggml_tensor * padmask = nullptr;
        ggml_tensor * out     = nullptr;
    };
    graph_cache<T5Key, T5IO> t5;

    struct LangKey {
        int steps = -1;
        bool operator==(const LangKey& o) const {
            return steps == o.steps;
        }
    };
    struct LangIO {
        ggml_tensor * in       = nullptr;
        ggml_tensor * pos      = nullptr;
        ggml_tensor * repeated = nullptr;
    };
    graph_cache<LangKey, LangIO> language;

    struct BtKey {
        int seq       = -1;
        int n_readout = -1;
        bool operator==(const BtKey& o) const {
            return seq == o.seq && n_readout == o.n_readout;
        }
    };
    struct BtIO {
        ggml_tensor * input       = nullptr;
        ggml_tensor * mask        = nullptr;
        ggml_tensor * readout_idx = nullptr;
        ggml_tensor * out         = nullptr;
    };
    graph_cache<BtKey, BtIO> transformer;

    // The whole reverse process lives in one graph. The steps are sequentially
    // dependent so they cannot be batched, but with the graph cached a frame
    // costs one submission instead of twenty.
    struct DiffKey {
        int width  = -1;
        int action = -1;
        int steps  = -1;
        bool operator==(const DiffKey& o) const {
            return width == o.width && action == o.action && steps == o.steps;
        }
    };
    struct DiffIO {
        ggml_tensor * obs   = nullptr;
        ggml_tensor * x0    = nullptr;
        ggml_tensor * z     = nullptr;
        ggml_tensor * times = nullptr;
        ggml_tensor * out   = nullptr;
    };
    graph_cache<DiffKey, DiffIO> diffusion;

    struct L1Key {
        int width  = -1;
        int action = -1;
        bool operator==(const L1Key& o) const {
            return width == o.width && action == o.action;
        }
    };
    struct L1IO {
        ggml_tensor * readout = nullptr;
        ggml_tensor * out     = nullptr;
    };
    graph_cache<L1Key, L1IO> l1_head;

    // The T5 encoder and the projection after it depend only on the
    // instruction, which holds for as long as the robot pursues one task.
    std::vector<int32_t> lang_key;
    std::vector<float>   lang_pos;
    std::vector<float>   lang_repeated;
    int                  lang_steps = -1;

    struct ActionStats {
        std::vector<float>   mean, stdv;
        std::vector<uint8_t> mask;
    };
    struct ProprioStats {
        std::vector<float> mean, stdv;
    };
    std::string  stats_key;
    bool         stats_loaded = false;
    ActionStats  action_stats;
    ProprioStats proprio_stats;
    bool         has_proprio_stats = false;

    void init(ggml_backend_t b, ggml_context * w) {
        backend = b;
        ctx_w   = w;
        // ggml_get_tensor is a linear strcmp scan, and building the stage graphs
        // looks up a few hundred weights by name.
        by_name.clear();
        for (ggml_tensor * t = ggml_get_first_tensor(w); t; t = ggml_get_next_tensor(w, t))
            by_name.emplace(t->name, t);
    }

    ggml_tensor * weight(const char * name) const {
        auto it = by_name.find(name);
        if (it == by_name.end()) {
            std::fprintf(stderr, "vla(octo): missing resident weight %s\n", name);
            return nullptr;
        }
        return it->second;
    }

    /// Must run before the backend the caches allocated from is freed.
    void reset() {
        obs_primary.release();
        obs_wrist.release();
        proprio.release();
        t5.release();
        language.release();
        transformer.release();
        diffusion.release();
        l1_head.release();
    }

    ~OctoRuntime() {
        reset();
    }
};

// Relabels duplicate node names for ggml-openvino (a no-op elsewhere).
bool octo_compute(OctoRuntime& rt, ggml_cgraph * gf, const char * what) {
    graph_unique_names(gf);
    const ggml_status st = ggml_backend_graph_compute(rt.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): %s graph compute failed (%d)\n", what, (int) st);
        return false;
    }
    return true;
}

struct OctoModelArch : public ModelArchBase {
    OctoModelArch() : ModelArchBase(Arch::OCTO) {}
    ~OctoModelArch() override {
        rt.reset();
        if (weight_buf)  ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend)     ggml_backend_free(backend);
    }

    std::string           gguf_path;
    ggml_backend_t        backend     = nullptr;
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    int                   n_threads   = default_cpu_threads();

    int64_t hidden          = 384;
    int64_t blocks          = 12;
    int64_t heads           = 6;
    int64_t ffn             = 1536;
    int64_t window_size     = 2;
    int64_t action_horizon  = 4;
    int64_t action_dim      = 7;
    int64_t primary_tokens  = 256;
    int64_t wrist_tokens    = 64;
    int64_t primary_size    = 256;
    int64_t wrist_size      = 128;
    int64_t language_tokens = kTaskTokens;
    int64_t diffusion_steps = 20;
    float   diffusion_s     = 0.008f;
    float   max_action      = 5.0f;
    int32_t pad_id          = 0;

    // "diffusion" for the published checkpoints, "l1" for the fine-tuned ones.
    std::string head_type = "diffusion";
    // Proprio has no metadata key of its own; presence of the projection weight
    // is what says the checkpoint has a LowdimObsTokenizer. Its in-dim is 1
    // (continuous) or 256 (bin one-hot).
    bool    has_proprio    = false;
    int64_t proprio_in_dim = 0;

    gguf_reader  io{"octo"};
    OctoRuntime  rt;
    std::mt19937 rng{std::random_device{}()};

    std::vector<float> predict(const Inputs& in) override;
};

void detect_proprio(const gguf_reader& g, bool& has_proprio, int64_t& proprio_in_dim) {
    const ggml_tensor * proj = g.meta("octo.obs.proprio.proj.weight");
    has_proprio    = proj != nullptr;
    proprio_in_dim = has_proprio ? proj->ne[0] : 0;
}

// The converter writes a whole-numbered scalar as an integer, so the diffusion
// constants are not all one GGUF type.
float scalar_key(const gguf_reader& g, const char * key) {
    const int64_t id = gguf_find_key(g.gctx, key);
    if (id >= 0) {
        switch (gguf_get_kv_type(g.gctx, id)) {
            case GGUF_TYPE_FLOAT32: return         gguf_get_val_f32(g.gctx, id);
            case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(g.gctx, id);
            case GGUF_TYPE_UINT32:  return (float) gguf_get_val_u32(g.gctx, id);
            case GGUF_TYPE_INT32:   return (float) gguf_get_val_i32(g.gctx, id);
            default: break;
        }
    }
    std::fprintf(stderr, "vla(octo): %s is not a scalar number\n", key);
    return 0.f;
}

bool require_key(const gguf_reader& g, const char * key) {
    if (!g.has(key)) {
        std::fprintf(stderr, "vla(octo): missing metadata %s\n", key);
        return false;
    }
    return true;
}

bool load_config(const gguf_reader& g, OctoModelArch& m) {
    const char * keys[] = {
        "octo.architecture",
        "octo.embedding_length",
        "octo.block_count",
        "octo.attention.head_count",
        "octo.feed_forward_length",
        "octo.attention.layer_norm_eps",
        "octo.window_size",
        "octo.action.horizon",
        "octo.action.dim",
        "octo.readout.count",
        "octo.tokens.primary",
        "octo.tokens.wrist",
        "octo.tokens.language",
        "octo.image.primary_size",
        "octo.image.wrist_size",
        "octo.diffusion.steps",
        "octo.diffusion.beta_schedule",
        "octo.diffusion.s",
        "octo.diffusion.max_action",
        "octo.dataset_statistics",
    };
    for (const char * key : keys) {
        if (!require_key(g, key))
            return false;
    }
    if (g.str("octo.architecture") != "octo-small-1.5") {
        std::fprintf(stderr, "vla(octo): octo.architecture=%s, expected octo-small-1.5\n",
                     g.str("octo.architecture").c_str());
        return false;
    }
    if (g.str("octo.diffusion.beta_schedule") != "cosine") {
        std::fprintf(stderr, "vla(octo): octo.diffusion.beta_schedule=%s, only cosine is implemented\n",
                     g.str("octo.diffusion.beta_schedule").c_str());
        return false;
    }

    m.hidden          = g.u32("octo.embedding_length");
    m.blocks          = g.u32("octo.block_count");
    m.heads           = g.u32("octo.attention.head_count");
    m.ffn             = g.u32("octo.feed_forward_length");
    m.window_size     = g.u32("octo.window_size");
    m.action_horizon  = g.u32("octo.action.horizon");
    m.action_dim      = g.u32("octo.action.dim");
    m.primary_tokens  = g.u32("octo.tokens.primary");
    m.wrist_tokens    = g.u32("octo.tokens.wrist");
    m.primary_size    = g.u32("octo.image.primary_size");
    m.wrist_size      = g.u32("octo.image.wrist_size");
    m.language_tokens = g.u32("octo.tokens.language");
    m.diffusion_steps = g.u32("octo.diffusion.steps");
    m.diffusion_s     = scalar_key(g, "octo.diffusion.s");
    m.max_action      = scalar_key(g, "octo.diffusion.max_action");
    m.pad_id          = g.has("octo.tokenizer.pad_id") ? (int32_t) g.u32("octo.tokenizer.pad_id") : 0;
    m.head_type       = g.has("octo.action.head_type") ? g.str("octo.action.head_type") : "diffusion";
    detect_proprio(g, m.has_proprio, m.proprio_in_dim);

    if (m.has_proprio && m.proprio_in_dim != 1 && m.proprio_in_dim != 256) {
        std::fprintf(stderr, "vla(octo): octo.obs.proprio.proj.weight in-dim=%lld, expected 1 or 256\n",
                     (long long) m.proprio_in_dim);
        return false;
    }
    if (m.hidden != kHidden || m.blocks != 12 || m.heads != 6 || m.ffn != 1536) {
        std::fprintf(stderr, "vla(octo): metadata does not match the octo-small-1.5 backbone\n");
        return false;
    }
    if (m.diffusion_steps < 1 || m.diffusion_s <= 0.f || m.max_action <= 0.f) {
        std::fprintf(stderr, "vla(octo): diffusion steps=%lld s=%g max_action=%g must all be positive\n",
                     (long long) m.diffusion_steps, (double) m.diffusion_s, (double) m.max_action);
        return false;
    }
    if (m.language_tokens != kTaskTokens) {
        std::fprintf(stderr, "vla(octo): octo.tokens.language=%lld, the T5 encoder is built for %d\n",
                     (long long) m.language_tokens, kTaskTokens);
        return false;
    }
    if (g.u32("octo.readout.count") != 1) {
        std::fprintf(stderr, "vla(octo): octo.readout.count=%u, only the single action readout is implemented\n",
                     g.u32("octo.readout.count"));
        return false;
    }
    // window_size is per-checkpoint (bridge pretrain 2, the LIBERO finetunes 1);
    // any value the shared position-embedding slab covers is a legal slice.
    if (m.window_size < 1 || m.window_size > kMaxHorizon) {
        std::fprintf(stderr, "vla(octo): octo.window_size=%lld outside [1, %d]\n",
                     (long long) m.window_size, kMaxHorizon);
        return false;
    }

    const int64_t proprio_dim = m.has_proprio ? kProprioTokens : 0;

    m.cfg.n_img             = m.primary_tokens+m.wrist_tokens;
    m.cfg.n_lang            = m.language_tokens;
    m.cfg.n_state           = proprio_dim;
    m.cfg.n_prefix          = m.language_tokens+m.window_size*(m.primary_tokens+m.wrist_tokens+m.language_tokens);
    m.cfg.n_suffix          = m.action_horizon;
    m.cfg.n_full            = m.cfg.n_prefix+m.window_size;
    m.cfg.hidden            = m.hidden;
    m.cfg.expert_h          = 256;
    m.cfg.intermediate      = m.ffn;
    m.cfg.expert_inter      = 256;
    m.cfg.n_q_heads         = m.heads;
    m.cfg.n_kv_heads        = m.heads;
    m.cfg.head_dim          = m.hidden/m.heads;
    m.cfg.q_full_dim        = m.hidden;
    m.cfg.kv_full_dim       = m.hidden;
    m.cfg.n_layers          = m.blocks;
    m.cfg.self_attn_every_n = 1;
    m.cfg.max_state_dim     = proprio_dim;
    m.cfg.max_action_dim    = m.action_dim;
    m.cfg.real_state_dim    = proprio_dim;
    m.cfg.real_action_dim   = m.action_dim;
    m.cfg.norm_eps          = g.f32("octo.attention.layer_norm_eps");
    m.cfg.num_steps         = (int) m.diffusion_steps;
    return true;
}

// SmallStem's StdConv standardizes its kernel per output channel on every
// forward pass. The kernel is frozen at inference, so the result is too: bake it
// in once instead of paying a round trip per camera per frame.
bool is_stem_conv_weight(const char * name) {
    return std::strstr(name, "octo.obs.") == name &&
           std::strstr(name, ".stem.")      != nullptr &&
           std::strstr(name, ".conv.weight") != nullptr;
}

void standardize_conv_weight(float * w, int64_t oc, int64_t n) {
    for (int64_t o=0; o<oc; ++o) {
        float * row = w+o*n;

        double mean = 0.0;
        for (int64_t i=0; i<n; ++i)
            mean += row[i];
        mean /= (double) n;

        double var = 0.0;
        for (int64_t i=0; i<n; ++i) {
            const double d = (double) row[i]-mean;
            var += d*d;
        }

        const float inv = 1.0f/std::sqrt((float) (var/(double) n)+1e-10f);
        for (int64_t i=0; i<n; ++i)
            row[i] = ((float) row[i]-(float) mean)*inv;
    }
}

bool load_weights(OctoModelArch& m, gguf_reader& g) {
    ggml_init_params wp = { (size_t) 16*1024*1024, nullptr, true };
    m.ctx_weights = ggml_init(wp);
    if (!m.ctx_weights) {
        std::fprintf(stderr, "vla(octo): ggml_init(ctx_weights) failed\n");
        return false;
    }

    // Every tensor the converter emits is consumed, and the optional groups
    // (wrist, proprio, either action head) differ per checkpoint, so the file's
    // own tensor list is the declaration.
    WeightLoader L("octo", g, m.ctx_weights, GGML_TYPE_F32);
    const int64_t n = gguf_get_n_tensors(g.gctx);
    for (int64_t i=0; i<n; ++i)
        L.f32("%s", gguf_get_tensor_name(g.gctx, i));
    if (!L.upload(m.backend, &m.weight_buf))
        return false;

    std::vector<float> row;
    for (ggml_tensor * t = ggml_get_first_tensor(m.ctx_weights); t; t = ggml_get_next_tensor(m.ctx_weights, t)) {
        if (!is_stem_conv_weight(ggml_get_name(t)))
            continue;
        row.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, row.data(), 0, ggml_nbytes(t));
        // ggml ne = [kw, kh, in, out]: one contiguous block per output channel.
        standardize_conv_weight(row.data(), t->ne[3], t->ne[0]*t->ne[1]*t->ne[2]);
        ggml_backend_tensor_set(t, row.data(), 0, ggml_nbytes(t));
    }
    return true;
}

// Host copy of a resident weight, for the few that a call transforms before they
// become graph operands. Goes through the backend: the buffer may be on device.
std::vector<float> tensor_to_vec(const ggml_tensor * t) {
    std::vector<float> out((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

// SmallStem16 for one camera view: four standardized-conv + GroupNorm + ReLU
// stages at stride 2, a 1x1 patch embedding, a projection to the model width,
// and the per-timestep position embedding.
//
// `obs` holds the normalized CHW frames for `steps`, `task` the single goal
// frame, which is concatenated onto every one of them as channels 3..5.
bool run_obs_tokenizer_graph(OctoRuntime& rt,
                             graph_cache<OctoRuntime::ObsKey, OctoRuntime::ObsIO>& cache,
                             const char * view,
                             const std::vector<float>& obs,
                             const std::vector<float>& task,
                             int side,
                             int n_tok,
                             int steps,
                             const std::vector<int32_t>& pos_rows,
                             std::vector<float>& pos) {
    const size_t frame = (size_t) 3*side*side;
    if (obs.size() != frame*(size_t) steps || task.size() != frame) {
        std::fprintf(stderr, "vla(octo): unexpected input image shape for side=%d\n", side);
        return false;
    }
    if ((int) pos_rows.size() != steps)
        return false;
    pos.resize((size_t) steps*n_tok*kHidden);

    const OctoRuntime::ObsKey key{side, n_tok, steps};
    const bool built = cache.ensure(rt.backend, key, (size_t) 32*1024*1024,
                                    [&](ggml_context * C, OctoRuntime::ObsIO& io) -> ggml_cgraph * {
        ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, side, side, 6, steps);
        ggml_set_name(x, "octo.obs.input_norm");
        ggml_set_input(x);
        io.input = x;

        char rname[160];
        auto weight = [&](const char * suffix, int li) -> ggml_tensor * {
            if (li < 0) std::snprintf(rname, sizeof(rname), "octo.obs.%s.%s", view, suffix);
            else        std::snprintf(rname, sizeof(rname), "octo.obs.%s.stem.%d.%s", view, li, suffix);
            return rt.weight(rname);
        };
        // Biases and scales arrive as [ch] against a [w,h,ch,steps] feature map;
        // the reshape is a view, and ggml_add/ggml_mul broadcast from there.
        auto per_channel = [&](ggml_tensor * t) {
            return t ? ggml_reshape_4d(C, t, 1, 1, t->ne[0], 1) : nullptr;
        };

        for (int li=0; li<4; ++li) {
            ggml_tensor * cw = weight("conv.weight", li);
            ggml_tensor * cb = per_channel(weight("conv.bias", li));
            ggml_tensor * gw = per_channel(weight("gn.weight", li));
            ggml_tensor * gb = per_channel(weight("gn.bias", li));
            if (!cw || !cb || !gw || !gb)
                return nullptr;

            x = ggml_conv_2d(C, cw, x, 2, 2, 1, 1, 1, 1);
            x = ggml_add(C, x, cb);
            x = ggml_group_norm(C, x, 32, 1e-5f);
            x = ggml_add(C, ggml_mul(C, x, gw), gb);
            x = ggml_relu(C, x);
        }

        ggml_tensor * pw    = weight("patch_embd.weight", -1);
        ggml_tensor * pb    = per_channel(weight("patch_embd.bias", -1));
        ggml_tensor * jw    = weight("proj.weight", -1);
        ggml_tensor * jb    = weight("proj.bias", -1);
        ggml_tensor * pos_r = weight("pos_embd", -1);
        if (!pw || !pb || !jw || !jb || !pos_r)
            return nullptr;

        ggml_tensor * patch = ggml_add(C, ggml_conv_2d(C, pw, x, 1, 1, 0, 0, 1, 1), pb);
        ggml_tensor * tok   = ggml_cont(C, ggml_reshape_3d(C,
            ggml_cont(C, ggml_permute(C, patch, 1, 2, 0, 3)), kPatchEmbed, n_tok, steps));

        // The timesteps in the sequence need not be the leading ones, so the
        // rows are gathered rather than sliced. See OctoSeqLayout.
        ggml_tensor * rows = ggml_new_tensor_1d(C, GGML_TYPE_I32, steps);
        ggml_set_name(rows, "octo.obs.pos_embd.rows");
        ggml_set_input(rows);
        io.rows = rows;
        ggml_tensor * pe = ggml_reshape_3d(C,
            ggml_get_rows(C, ggml_reshape_2d(C, pos_r, kHidden*n_tok, pos_r->ne[2]), rows),
            kHidden, n_tok, steps);

        ggml_tensor * out = ggml_add(C, ggml_add(C, ggml_mul_mat(C, jw, tok), jb), pe);
        ggml_set_name(out, "obs.tokenizer.pos");
        ggml_set_output(out);
        io.pos = out;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
        ggml_build_forward_expand(gf, out);
        return gf;
    });
    if (!built) {
        std::fprintf(stderr, "vla(octo): obs tokenizer graph build failed\n");
        return false;
    }

    OctoRuntime::ObsIO& io = cache.io();
    std::vector<float> input((size_t) steps*2*frame);
    for (int t=0; t<steps; ++t) {
        const size_t dst = (size_t) t*2*frame;
        std::copy_n(obs.begin()+(ptrdiff_t) ((size_t) t*frame), frame, input.begin()+(ptrdiff_t) dst);
        std::copy_n(task.begin(), frame, input.begin()+(ptrdiff_t) (dst+frame));
    }
    ggml_backend_tensor_set(io.input, input.data(), 0, ggml_nbytes(io.input));
    ggml_backend_tensor_set(io.rows, pos_rows.data(), 0, ggml_nbytes(io.rows));
    if (!octo_compute(rt, cache.graph(), "obs tokenizer"))
        return false;
    ggml_backend_tensor_get(io.pos, pos.data(), 0, ggml_nbytes(io.pos));
    return true;
}

// One token per state dimension. The z-scored value either feeds the shared
// Linear(1,384) directly or is bucketed into a one-hot(256) first; both converge
// on the same graph once tokens_in is built.
bool run_proprio_tokenizer_graph(OctoRuntime& rt,
                                 const std::vector<float>& proprio_norm,
                                 int in_dim,
                                 int steps,
                                 const std::vector<int32_t>& pos_rows,
                                 std::vector<float>& pos_out) {
    constexpr int n_dims = kProprioTokens;
    if (proprio_norm.size() != (size_t) steps*n_dims)
        return false;
    if ((int) pos_rows.size() != steps)
        return false;
    if (in_dim != 1 && in_dim != 256) {
        std::fprintf(stderr, "vla(octo): proprio tokenizer in_dim=%d, expected 1 or 256\n", in_dim);
        return false;
    }
    pos_out.resize((size_t) steps*n_dims*kHidden);

    std::vector<float> tokens_in((size_t) in_dim*n_dims*steps, 0.0f);
    if (in_dim == 1) {
        for (size_t i=0; i<proprio_norm.size(); ++i)
            tokens_in[i] = proprio_norm[i];
    } else {
        ggml_tensor * thresholds_r = rt.weight("octo.obs.proprio.bin_thresholds");
        if (!thresholds_r)
            return false;
        // torch.bucketize: the index is the count of boundaries <= x.
        const std::vector<float> thresholds = tensor_to_vec(thresholds_r);
        const int n_thresh = (int) thresholds.size();
        for (int t=0; t<steps; ++t) {
            for (int d=0; d<n_dims; ++d) {
                const float v = proprio_norm[(size_t) t*n_dims+d];
                int bucket = 0;
                while (bucket < n_thresh && thresholds[(size_t) bucket] <= v)
                    ++bucket;
                bucket = std::min(bucket, in_dim-1);
                tokens_in[((size_t) t*n_dims+d)*in_dim+bucket] = 1.0f;
            }
        }
    }

    const OctoRuntime::ProprioKey key{in_dim, steps};
    const bool built = rt.proprio.ensure(rt.backend, key, (size_t) 4*1024*1024,
                                         [&](ggml_context * C, OctoRuntime::ProprioIO& io) -> ggml_cgraph * {
        ggml_tensor * proj_w = rt.weight("octo.obs.proprio.proj.weight");
        ggml_tensor * proj_b = rt.weight("octo.obs.proprio.proj.bias");
        ggml_tensor * pos_r  = rt.weight("octo.obs.proprio.pos_embd");
        if (!proj_w || !proj_b || !pos_r)
            return nullptr;
        if (pos_r->ne[2] < steps) {
            std::fprintf(stderr, "vla(octo): octo.obs.proprio.pos_embd too small for %d timesteps\n", steps);
            return nullptr;
        }

        ggml_tensor * x = ggml_new_tensor_3d(C, GGML_TYPE_F32, in_dim, n_dims, steps);
        ggml_set_name(x, "octo.obs.proprio.tokens_in");
        ggml_set_input(x);
        io.tokens = x;

        ggml_tensor * rows = ggml_new_tensor_1d(C, GGML_TYPE_I32, steps);
        ggml_set_name(rows, "octo.obs.proprio.pos_embd.rows");
        ggml_set_input(rows);
        io.rows = rows;

        ggml_tensor * pe = ggml_reshape_3d(C,
            ggml_get_rows(C, ggml_reshape_2d(C, pos_r, kHidden*n_dims, pos_r->ne[2]), rows),
            kHidden, n_dims, steps);
        ggml_tensor * out = ggml_add(C, ggml_add(C, ggml_mul_mat(C, proj_w, x), proj_b), pe);
        ggml_set_name(out, "obs.proprio.pos");
        ggml_set_output(out);
        io.pos = out;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 256, false);
        ggml_build_forward_expand(gf, out);
        return gf;
    });
    if (!built) {
        std::fprintf(stderr, "vla(octo): proprio tokenizer graph build failed\n");
        return false;
    }

    OctoRuntime::ProprioIO& io = rt.proprio.io();
    ggml_backend_tensor_set(io.tokens, tokens_in.data(), 0, ggml_nbytes(io.tokens));
    ggml_backend_tensor_set(io.rows, pos_rows.data(), 0, ggml_nbytes(io.rows));
    if (!octo_compute(rt, rt.proprio.graph(), "proprio tokenizer"))
        return false;
    ggml_backend_tensor_get(io.pos, pos_out.data(), 0, ggml_nbytes(io.pos));
    return true;
}

// Language projection and position embedding, then repeat_task_tokens: the same
// 16 projected tokens are duplicated into every observation timestep.
bool run_language_graph(OctoRuntime& rt,
                        const std::vector<float>& t5,
                        int steps,
                        std::vector<float>& pos,
                        std::vector<float>& repeated) {
    if (t5.size() != (size_t) kTaskTokens*768) {
        std::fprintf(stderr, "vla(octo): expected a T5 output of %dx768\n", kTaskTokens);
        return false;
    }
    pos.resize((size_t) kTaskTokens*kHidden);
    repeated.resize((size_t) steps*kTaskTokens*kHidden);

    const bool built = rt.language.ensure(rt.backend, OctoRuntime::LangKey{steps}, (size_t) 4*1024*1024,
                                          [&](ggml_context * C, OctoRuntime::LangIO& io) -> ggml_cgraph * {
        ggml_tensor * jw = rt.weight("octo.task.language.proj.weight");
        ggml_tensor * jb = rt.weight("octo.task.language.proj.bias");
        ggml_tensor * pe = rt.weight("octo.task.language.pos_embd");
        if (!jw || !jb || !pe)
            return nullptr;

        ggml_tensor * in = ggml_new_tensor_3d(C, GGML_TYPE_F32, 768, kTaskTokens, 1);
        ggml_set_name(in, "octo.task.language.t5_inject");
        ggml_set_input(in);
        io.in = in;

        ggml_tensor * pos_t = ggml_add(C, ggml_add(C, ggml_mul_mat(C, jw, in), jb), pe);
        ggml_set_name(pos_t, "task_language.pos");
        ggml_set_output(pos_t);
        io.pos = pos_t;

        ggml_tensor * rep = ggml_repeat_4d(C, pos_t, kHidden, kTaskTokens, steps, 1);
        ggml_set_name(rep, "obs_task_language.repeated");
        ggml_set_output(rep);
        io.repeated = rep;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 1024, false);
        ggml_build_forward_expand(gf, rep);
        return gf;
    });
    if (!built) {
        std::fprintf(stderr, "vla(octo): language graph build failed\n");
        return false;
    }

    OctoRuntime::LangIO& io = rt.language.io();
    ggml_backend_tensor_set(io.in, t5.data(), 0, ggml_nbytes(io.in));
    if (!octo_compute(rt, rt.language.graph(), "language"))
        return false;
    ggml_backend_tensor_get(io.pos, pos.data(), 0, ggml_nbytes(io.pos));
    ggml_backend_tensor_get(io.repeated, repeated.data(), 0, ggml_nbytes(io.repeated));
    return true;
}

// relative_position = key_pos - query_pos, bidirectional, 32 buckets, max
// distance 128 -- HF T5Attention._relative_position_bucket.
int32_t t5_relative_position_bucket(int32_t query_pos, int32_t key_pos, int32_t n_buckets, int32_t max_distance) {
    const int32_t nb                = n_buckets/2;
    const int32_t relative_position = key_pos-query_pos;
    const int32_t rp                = std::abs(relative_position);
    const int32_t max_exact         = nb/2;

    int32_t bucket = relative_position > 0 ? nb : 0;
    if (rp < max_exact) {
        bucket += rp;
    } else {
        const float v = (float) max_exact+std::log((float) rp/(float) max_exact)/
                                          std::log((float) max_distance/(float) max_exact)*(float) (nb-max_exact);
        bucket += std::min((int32_t) std::floor(v), nb-1);
    }
    return bucket;
}

// T5-base encoder over the instruction: 12 pre-norm blocks, bidirectional
// attention with the relative-position bias shared from block 0. The bucket
// table and pad mask are host-computed, so the graph depends only on the length.
bool run_t5_encoder_graph(OctoRuntime& rt,
                          const std::vector<int32_t>& input_ids,
                          const std::vector<int32_t>& attention_mask,
                          std::vector<float>& t5_out) {
    constexpr int   hidden       = 768;
    constexpr int   heads        = 12;
    constexpr int   head_dim     = 64;
    constexpr int   seq          = kTaskTokens;
    constexpr int   n_buckets    = 32;
    constexpr int   max_distance = 128;
    constexpr float ln_eps       = 1e-6f;
    if (input_ids.size() != seq || attention_mask.size() != seq) {
        std::fprintf(stderr, "vla(octo): T5 encoder expected %d input_ids/attention_mask\n", seq);
        return false;
    }

    std::vector<int32_t> bucket_idx((size_t) seq*seq);
    std::vector<float>   padmask((size_t) seq*seq);
    for (int q=0; q<seq; ++q) {
        for (int k=0; k<seq; ++k) {
            bucket_idx[(size_t) q*seq+k] = t5_relative_position_bucket(q, k, n_buckets, max_distance);
            padmask   [(size_t) q*seq+k] = attention_mask[(size_t) k] != 0 ? 0.0f : -FLT_MAX;
        }
    }

    const bool built = rt.t5.ensure(rt.backend, OctoRuntime::T5Key{seq}, (size_t) 16*1024*1024,
                                    [&](ggml_context * C, OctoRuntime::T5IO& io) -> ggml_cgraph * {
        ggml_tensor * tok_embd = rt.weight("octo.t5.tok_embd.weight");
        ggml_tensor * rel_b    = rt.weight("octo.t5.blk.0.attn_rel_b.weight");
        ggml_tensor * outw     = rt.weight("octo.t5.output_norm.weight");
        if (!tok_embd || !rel_b || !outw)
            return nullptr;

        char rname[160];
        ggml_tensor * blk_w[12][8];
        const char * leaves[8] = {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                                  "attn_o.weight", "ffn_norm.weight", "ffn_up.weight", "ffn_down.weight"};
        for (int i=0; i<12; ++i) {
            for (int j=0; j<8; ++j) {
                std::snprintf(rname, sizeof(rname), "octo.t5.blk.%d.%s", i, leaves[j]);
                blk_w[i][j] = rt.weight(rname);
                if (!blk_w[i][j])
                    return nullptr;
            }
        }

        ggml_tensor * ids = ggml_new_tensor_1d(C, GGML_TYPE_I32, seq);
        ggml_set_name(ids, "octo.t5.input_ids");
        ggml_set_input(ids);
        io.ids = ids;

        ggml_tensor * bucket = ggml_new_tensor_2d(C, GGML_TYPE_I32, seq, seq);
        ggml_set_name(bucket, "octo.t5.pos_bucket");
        ggml_set_input(bucket);
        io.bucket = bucket;

        ggml_tensor * padmask_t = ggml_new_tensor_2d(C, GGML_TYPE_F32, seq, seq);
        ggml_set_name(padmask_t, "octo.t5.padmask");
        ggml_set_input(padmask_t);
        io.padmask = padmask_t;

        ggml_tensor * x = ggml_get_rows(C, tok_embd, ids);
        ggml_set_name(x, "octo.t5.input_embed");

        ggml_tensor * pos_bias = ggml_get_rows(C, rel_b, ggml_reshape_1d(C, bucket, (int64_t) seq*seq));
        pos_bias = ggml_reshape_3d(C, pos_bias, heads, seq, seq);
        pos_bias = ggml_cont(C, ggml_permute(C, pos_bias, 2, 0, 1, 3));
        ggml_tensor * mask = ggml_add(C, pos_bias, padmask_t);

        for (int i=0; i<12; ++i) {
            ggml_tensor * n1 = ggml_mul(C, ggml_rms_norm(C, x, ln_eps), blk_w[i][0]);
            ggml_tensor * Q  = ggml_mul_mat(C, blk_w[i][1], n1);
            ggml_tensor * K  = ggml_mul_mat(C, blk_w[i][2], n1);
            ggml_tensor * V  = ggml_mul_mat(C, blk_w[i][3], n1);
            ggml_tensor * Qh = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, Q, head_dim, heads, seq), 0, 2, 1, 3));
            ggml_tensor * Kh = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, K, head_dim, heads, seq), 0, 2, 1, 3));
            ggml_tensor * Vh = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, V, head_dim, heads, seq), 1, 2, 0, 3));

            ggml_tensor * scores = ggml_mul_mat(C, Kh, Qh);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            // T5 folds 1/sqrt(d_k) into the weights, so the scale here is 1.
            ggml_tensor * probs    = ggml_soft_max_ext(C, scores, mask, 1.0f, 0.0f);
            ggml_tensor * attended = ggml_mul_mat(C, Vh, probs);
            ggml_tensor * merged   = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, attended, 0, 2, 1, 3)), hidden, seq);
            x = ggml_add(C, x, ggml_mul_mat(C, blk_w[i][4], merged));

            ggml_tensor * n2 = ggml_mul(C, ggml_rms_norm(C, x, ln_eps), blk_w[i][5]);
            ggml_tensor * h  = ggml_relu(C, ggml_mul_mat(C, blk_w[i][6], n2));
            x = ggml_add(C, x, ggml_mul_mat(C, blk_w[i][7], h));
        }

        ggml_tensor * out = ggml_mul(C, ggml_rms_norm(C, x, ln_eps), outw);
        ggml_set_name(out, "t5.out");
        ggml_set_output(out);
        io.out = out;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 4096, false);
        ggml_build_forward_expand(gf, out);
        return gf;
    });
    if (!built) {
        std::fprintf(stderr, "vla(octo): T5 encoder graph build failed\n");
        return false;
    }

    OctoRuntime::T5IO& io = rt.t5.io();
    ggml_backend_tensor_set(io.ids,     input_ids.data(),      0, ggml_nbytes(io.ids));
    ggml_backend_tensor_set(io.bucket,  bucket_idx.data(),     0, ggml_nbytes(io.bucket));
    ggml_backend_tensor_set(io.padmask, padmask.data(),        0, ggml_nbytes(io.padmask));
    if (!octo_compute(rt, rt.t5.graph(), "T5 encoder"))
        return false;
    t5_out.resize((size_t) hidden*seq);
    ggml_backend_tensor_get(io.out, t5_out.data(), 0, ggml_nbytes(io.out));
    return true;
}

// Which tokenizer a run of sequence tokens came from. TASK is the once-only
// language prefix; the repeated task tokens count as observation, matching
// repeat_task_tokens.
enum class OctoGroup { TASK, PRIMARY, WRIST, PROPRIO, LANGUAGE, READOUT };

struct OctoSeqRun {
    OctoGroup group;
    int       timestep;   ///< -1 for the task prefix.
    int       n_tokens;
    int       src_step;   ///< Index of this timestep inside that group's own buffer.
    bool      key_valid;  ///< False => masked out as a key for every query.
};

// The sequence the block transformer actually sees.
//
// Observation tokens belonging to a padded timestep are left out rather than
// emitted and then masked. They cannot affect the result: the pad mask makes
// them invalid keys for every query, and only the readout rows are read back.
// Dropping them shortens a cold-start window_size=2 sequence from 690 tokens to
// 370 and lets the conv stem run over the frames that are actually live.
struct OctoSeqLayout {
    std::vector<OctoSeqRun> runs;
    int                     seq = 0;
    std::vector<int32_t>    primary_steps;   ///< Original timestep of each emitted batch entry.
    std::vector<int32_t>    wrist_steps;
    std::vector<int32_t>    proprio_steps;
    std::vector<int32_t>    readout_seq_idx; ///< Sequence position of each timestep's readout token.
};

// Per-timestep order is primary -> wrist -> proprio -> repeated language ->
// readout, with the task prefix once at the front.
OctoSeqLayout build_seq_layout(bool task_valid,
                               const std::vector<uint8_t>& primary_valid,
                               const std::vector<uint8_t>& wrist_valid,
                               const std::vector<uint8_t>& timestep_valid,
                               int window_size,
                               int n_primary,
                               int n_wrist,
                               int n_proprio) {
    OctoSeqLayout L;
    auto emit = [&](OctoGroup g, int t, int n, int src, bool valid) {
        if (n <= 0)
            return;
        L.runs.push_back({g, t, n, src, valid});
        L.seq += n;
    };

    emit(OctoGroup::TASK, -1, kTaskTokens, 0, task_valid);
    for (int t=0; t<window_size; ++t) {
        const bool live = timestep_valid[(size_t) t] != 0;
        if (live && primary_valid[(size_t) t]) {
            emit(OctoGroup::PRIMARY, t, n_primary, (int) L.primary_steps.size(), true);
            L.primary_steps.push_back(t);
        }
        if (live && wrist_valid[(size_t) t]) {
            emit(OctoGroup::WRIST, t, n_wrist, (int) L.wrist_steps.size(), true);
            L.wrist_steps.push_back(t);
        }
        if (live && n_proprio > 0) {
            emit(OctoGroup::PROPRIO, t, n_proprio, (int) L.proprio_steps.size(), true);
            L.proprio_steps.push_back(t);
        }
        // The repeated task tokens and the readout query stay in for every
        // timestep: both are valid keys for later ones regardless of the mask.
        emit(OctoGroup::LANGUAGE, t, kTaskTokens, t, task_valid);
        L.readout_seq_idx.push_back(L.seq);
        emit(OctoGroup::READOUT, t, 1, t, true);
    }
    return L;
}

// Each group's buffer is indexed by the run's src_step: the compacted batch
// index for the tokenized groups, the original timestep for the rest.
bool assemble_transformer_input(const OctoSeqLayout& layout,
                                const std::vector<float>& task_language,
                                const std::vector<float>& obs_primary,
                                const std::vector<float>& obs_wrist,
                                const std::vector<float>& obs_proprio,
                                const std::vector<float>& repeated_language,
                                const std::vector<float>& readout_pos,
                                std::vector<float>& input) {
    input.assign((size_t) layout.seq*kHidden, 0.0f);

    size_t dst = 0;
    for (const OctoSeqRun& r : layout.runs) {
        const std::vector<float> * src = nullptr;
        switch (r.group) {
            case OctoGroup::TASK:     src = &task_language;     break;
            case OctoGroup::PRIMARY:  src = &obs_primary;       break;
            case OctoGroup::WRIST:    src = &obs_wrist;         break;
            case OctoGroup::PROPRIO:  src = &obs_proprio;       break;
            case OctoGroup::LANGUAGE: src = &repeated_language; break;
            case OctoGroup::READOUT:  src = &readout_pos;       break;
        }
        const size_t n   = (size_t) r.n_tokens*kHidden;
        const size_t off = (size_t) r.src_step*n;
        if (src->size() < off+n) {
            std::fprintf(stderr, "vla(octo): invalid tensor size while assembling block transformer input\n");
            return false;
        }
        std::copy_n(src->begin()+(ptrdiff_t) off, n, input.begin()+(ptrdiff_t) dst);
        dst += n;
    }
    return true;
}

// Additive mask, 0 where attention is allowed and -FLT_MAX where it is blocked.
// A task token sees only task tokens; an observation token sees the task prefix
// plus every observation token at its own timestep or earlier; a readout token
// sees those plus the readouts up to its own timestep. A key the pad mask says
// is not real is blocked for everyone. Shape is [seq,seq] with no head axis --
// ggml_soft_max_ext broadcasts a mask whose ne2 is 1 over all heads.
void build_transformer_mask(const OctoSeqLayout& layout, std::vector<float>& mask) {
    const int seq = layout.seq;

    std::vector<OctoGroup> group((size_t) seq);
    std::vector<int>       timestep((size_t) seq);
    std::vector<uint8_t>   key_valid((size_t) seq);
    int i = 0;
    for (const OctoSeqRun& r : layout.runs) {
        for (int k=0; k<r.n_tokens; ++k, ++i) {
            group    [(size_t) i] = r.group;
            timestep [(size_t) i] = r.timestep;
            key_valid[(size_t) i] = r.key_valid ? 1 : 0;
        }
    }

    mask.assign((size_t) seq*seq, 0.0f);
    for (int q=0; q<seq; ++q) {
        const OctoGroup qg = group[(size_t) q];
        const int       qt = timestep[(size_t) q];
        for (int k=0; k<seq; ++k) {
            const OctoGroup kg        = group[(size_t) k];
            const bool      k_task    = kg == OctoGroup::TASK;
            const bool      k_readout = kg == OctoGroup::READOUT;
            const bool      k_obs     = !k_task && !k_readout;

            bool allowed;
            if (qg == OctoGroup::TASK)
                allowed = k_task;
            else if (qg == OctoGroup::READOUT)
                allowed = k_task || ((k_obs || k_readout) && timestep[(size_t) k] <= qt);
            else
                allowed = k_task || (k_obs && timestep[(size_t) k] <= qt);

            if (!allowed || !key_valid[(size_t) k])
                mask[(size_t) q*seq+k] = -FLT_MAX;
        }
    }
}

// 12 pre-norm encoder blocks over the assembled sequence. Only the readout rows
// are gathered back out; everything else the blocks compute is intermediate.
bool run_transformer_graph(OctoRuntime& rt,
                           const OctoSeqLayout& layout,
                           const std::vector<float>& input,
                           const std::vector<float>& blocked_mask,
                           std::vector<float>& readout_action) {
    constexpr int   heads      = 6;
    constexpr int   head_dim   = 64;
    constexpr float ln_eps     = 1e-6f;
    constexpr float attn_scale = 0.125f;
    const int seq       = layout.seq;
    const int n_readout = (int) layout.readout_seq_idx.size();
    if (input.size() != (size_t) kHidden*seq || blocked_mask.size() != (size_t) seq*seq)
        return false;

    const OctoRuntime::BtKey key{seq, n_readout};
    const bool built = rt.transformer.ensure(rt.backend, key, (size_t) 32*1024*1024,
                                             [&](ggml_context * C, OctoRuntime::BtIO& io) -> ggml_cgraph * {
        char rname[160];
        ggml_tensor * blk_w[12][12];
        const char * leaves[12] = {"attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                                   "attn_o.weight", "attn_o.bias", "ffn_norm.weight", "ffn_norm.bias",
                                   "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias"};
        for (int i=0; i<12; ++i) {
            for (int j=0; j<12; ++j) {
                std::snprintf(rname, sizeof(rname), "octo.blk.%d.%s", i, leaves[j]);
                blk_w[i][j] = rt.weight(rname);
                if (!blk_w[i][j])
                    return nullptr;
            }
        }
        ggml_tensor * out_w = rt.weight("octo.output_norm.weight");
        ggml_tensor * out_b = rt.weight("octo.output_norm.bias");
        if (!out_w || !out_b)
            return nullptr;

        ggml_tensor * x = ggml_new_tensor_2d(C, GGML_TYPE_F32, kHidden, seq);
        ggml_set_name(x, "octo.block_transformer.input");
        ggml_set_input(x);
        io.input = x;

        ggml_tensor * mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, seq, seq);
        ggml_set_name(mask, "octo.block_transformer.additive_mask");
        ggml_set_input(mask);
        io.mask = mask;

        ggml_tensor * readout_idx = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_readout);
        ggml_set_name(readout_idx, "octo.block_transformer.readout_idx");
        ggml_set_input(readout_idx);
        io.readout_idx = readout_idx;

        for (int i=0; i<12; ++i) {
            ggml_tensor * n1  = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), blk_w[i][0]), blk_w[i][1]);
            ggml_tensor * qkv = ggml_add(C, ggml_mul_mat(C, blk_w[i][2], n1), blk_w[i][3]);
            ggml_tensor * q   = ggml_cont(C, ggml_view_2d(C, qkv, kHidden, seq, qkv->nb[1], 0));
            ggml_tensor * k   = ggml_cont(C, ggml_view_2d(C, qkv, kHidden, seq, qkv->nb[1], (size_t) kHidden*qkv->nb[0]));
            ggml_tensor * v   = ggml_cont(C, ggml_view_2d(C, qkv, kHidden, seq, qkv->nb[1], (size_t) 2*kHidden*qkv->nb[0]));
            ggml_tensor * Q   = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, head_dim, heads, seq), 0, 2, 1, 3));
            ggml_tensor * K   = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, head_dim, heads, seq), 0, 2, 1, 3));
            ggml_tensor * V   = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, head_dim, heads, seq), 1, 2, 0, 3));

            ggml_tensor * scores = ggml_mul_mat(C, K, Q);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            ggml_tensor * probs    = ggml_soft_max_ext(C, scores, mask, attn_scale, 0.0f);
            ggml_tensor * attended = ggml_mul_mat(C, V, probs);
            ggml_tensor * merged   = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, attended, 0, 2, 1, 3)), kHidden, seq);
            ggml_tensor * attn_out = ggml_add(C, ggml_mul_mat(C, blk_w[i][4], merged), blk_w[i][5]);
            ggml_tensor * residual = ggml_add(C, x, attn_out);

            ggml_tensor * n2  = ggml_add(C, ggml_mul(C, ggml_norm(C, residual, ln_eps), blk_w[i][6]), blk_w[i][7]);
            ggml_tensor * mlp = ggml_add(C, ggml_mul_mat(C, blk_w[i][8], n2), blk_w[i][9]);
            mlp = ggml_gelu_erf(C, mlp);
            mlp = ggml_add(C, ggml_mul_mat(C, blk_w[i][10], mlp), blk_w[i][11]);
            x = ggml_add(C, residual, mlp);
        }

        ggml_tensor * output = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), out_w), out_b);
        // The readouts are not evenly spaced once padded timesteps drop their
        // observation groups, so they are gathered rather than strided.
        ggml_tensor * readout = ggml_get_rows(C, output, readout_idx);
        ggml_set_name(readout, "bt.readout_action");
        ggml_set_output(readout);
        io.out = readout;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
        ggml_build_forward_expand(gf, readout);
        return gf;
    });
    if (!built) {
        std::fprintf(stderr, "vla(octo): block transformer graph build failed\n");
        return false;
    }

    OctoRuntime::BtIO& io = rt.transformer.io();
    ggml_backend_tensor_set(io.input,       input.data(),                   0, ggml_nbytes(io.input));
    ggml_backend_tensor_set(io.mask,        blocked_mask.data(),            0, ggml_nbytes(io.mask));
    ggml_backend_tensor_set(io.readout_idx, layout.readout_seq_idx.data(),  0, ggml_nbytes(io.readout_idx));
    if (!octo_compute(rt, rt.transformer.graph(), "block transformer"))
        return false;
    readout_action.resize((size_t) kHidden*n_readout);
    ggml_backend_tensor_get(io.out, readout_action.data(), 0, ggml_nbytes(io.out));
    return true;
}

struct OctoDiffusion {
    int   steps      = 20;
    float s          = 0.008f;
    float max_action = 5.0f;
};

struct OctoDiffusionSchedule {
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alpha_hats;
};

OctoDiffusionSchedule make_cosine_schedule(int steps, float s) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double     ds = (double) s;

    std::vector<double> alpha_cum((size_t) steps+1);
    for (int i=0; i<=steps; ++i) {
        const double t = (double) i/(double) steps;
        const double v = std::cos((t+ds)/(1.0+ds)*pi*0.5);
        alpha_cum[(size_t) i] = v*v;
    }

    OctoDiffusionSchedule sched;
    sched.betas.resize((size_t) steps);
    sched.alphas.resize((size_t) steps);
    sched.alpha_hats.resize((size_t) steps);

    const double first = alpha_cum[0];
    float        cum   = 1.0f;
    for (int i=0; i<steps; ++i) {
        const double a0   = alpha_cum[(size_t) i]/first;
        const double a1   = alpha_cum[(size_t) i+1]/first;
        const float  beta = (float) std::min(std::max(1.0-a1/a0, 0.0), 0.999);
        sched.betas [(size_t) i] = beta;
        sched.alphas[(size_t) i] = 1.0f-beta;
        cum *= sched.alphas[(size_t) i];
        sched.alpha_hats[(size_t) i] = cum;
    }
    return sched;
}

// Pre-resolved so a 20-step chain does not re-look-up 27 tensors per step.
struct OctoScoreActorWeights {
    ggml_tensor * time_w;
    ggml_tensor * c0w,  * c0b;
    ggml_tensor * c1w,  * c1b;
    ggml_tensor * rinw, * rinb;
    ggml_tensor * routw,* routb;
    ggml_tensor * blk[3][6];
};

bool resolve_score_actor_weights(const OctoRuntime& rt, OctoScoreActorWeights& w) {
    w.time_w = rt.weight("octo.head.diffusion.time_fourier.weight");
    w.c0w    = rt.weight("octo.head.diffusion.cond.0.weight");
    w.c0b    = rt.weight("octo.head.diffusion.cond.0.bias");
    w.c1w    = rt.weight("octo.head.diffusion.cond.1.weight");
    w.c1b    = rt.weight("octo.head.diffusion.cond.1.bias");
    w.rinw   = rt.weight("octo.head.diffusion.reverse.in.weight");
    w.rinb   = rt.weight("octo.head.diffusion.reverse.in.bias");
    w.routw  = rt.weight("octo.head.diffusion.reverse.out.weight");
    w.routb  = rt.weight("octo.head.diffusion.reverse.out.bias");
    if (!w.time_w || !w.c0w || !w.c0b || !w.c1w || !w.c1b || !w.rinw || !w.rinb || !w.routw || !w.routb)
        return false;

    char rname[160];
    const char * leaves[6] = {"ln.weight", "ln.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<6; ++j) {
            std::snprintf(rname, sizeof(rname), "octo.head.diffusion.reverse.blk.%d.%s", i, leaves[j]);
            w.blk[i][j] = rt.weight(rname);
            if (!w.blk[i][j])
                return false;
        }
    }
    return true;
}

// One reverse step as graph nodes: Fourier time embedding, conditioning MLP,
// concat(cond, readout, noisy action), three residual blocks, eps.
ggml_tensor * build_score_actor(ggml_context * ctx,
                                const OctoScoreActorWeights& w,
                                ggml_tensor * time,
                                ggml_tensor * obs,
                                ggml_tensor * actions) {
    constexpr float ln_eps = 1e-6f;
    constexpr float two_pi = 6.2831853071795864769f;

    ggml_tensor * f       = ggml_scale(ctx, ggml_mul_mat(ctx, w.time_w, time), two_pi);
    ggml_tensor * time_ff = ggml_concat(ctx, ggml_cos(ctx, f), ggml_sin(ctx, f), 0);
    ggml_tensor * cond    = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w.c0w, time_ff), w.c0b));
    cond = ggml_add(ctx, ggml_mul_mat(ctx, w.c1w, cond), w.c1b);

    ggml_tensor * reverse_input = ggml_concat(ctx, ggml_concat(ctx, cond, obs, 0), actions, 0);
    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, w.rinw, reverse_input), w.rinb);
    for (int i=0; i<3; ++i) {
        ggml_tensor * residual = x;
        ggml_tensor * h = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), w.blk[i][0]), w.blk[i][1]);
        h = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w.blk[i][2], h), w.blk[i][3]));
        h = ggml_add(ctx, ggml_mul_mat(ctx, w.blk[i][4], h), w.blk[i][5]);
        x = ggml_add(ctx, residual, h);
    }
    return ggml_add(ctx, ggml_mul_mat(ctx, w.routw, ggml_silu(ctx, x)), w.routb);
}

// The DDPM reverse process as ONE graph. The steps are sequentially dependent so
// they cannot be batched, but chaining them inside a cached graph costs a frame
// one submission and one readback instead of twenty.
//
// Every window row denoises independently -- width is a batch axis all the way
// through build_score_actor -- so `noise`, which covers the one row that is read
// back, is replicated across the rest rather than mixed into them.
bool run_diffusion(OctoRuntime& rt,
                   std::mt19937& rng,
                   const float * noise,
                   const std::vector<float>& readout_action,
                   int window_size,
                   int action_total,
                   const OctoDiffusion& diff,
                   std::vector<float>& final_actions) {
    const int width  = window_size;
    const int action = action_total;
    const int steps  = diff.steps;
    if (readout_action.size() != (size_t) kHidden*width || width < 1 || action < 1 || steps < 1)
        return false;

    const OctoDiffusionSchedule sched = make_cosine_schedule(steps, diff.s);
    const size_t                chunk = (size_t) width*action;

    // Drawn up front in the order and count a step-at-a-time loop would use, so
    // one RNG state yields one trajectory. The last step adds no noise.
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> initial_noise(chunk);
    if (noise) {
        for (int i=0; i<width; ++i)
            std::copy_n(noise, action, initial_noise.begin()+(ptrdiff_t) ((size_t) i*action));
    } else {
        for (float& v : initial_noise)
            v = normal(rng);
    }

    std::vector<float> z(chunk*(size_t) steps, 0.0f);
    std::vector<float> times((size_t) width*steps);
    for (int step=0; step<steps; ++step) {
        const int time_value = steps-1-step;
        if (time_value > 0) {
            for (size_t i=0; i<chunk; ++i)
                z[(size_t) step*chunk+i] = normal(rng);
        }
        for (int i=0; i<width; ++i)
            times[(size_t) step*width+i] = (float) time_value;
    }

    const OctoRuntime::DiffKey key{width, action, steps};
    const bool built = rt.diffusion.ensure(rt.backend, key, (size_t) 32*1024*1024,
                                           [&](ggml_context * C, OctoRuntime::DiffIO& io) -> ggml_cgraph * {
        OctoScoreActorWeights w{};
        if (!resolve_score_actor_weights(rt, w))
            return nullptr;
        if (w.routw->ne[1] != action) {
            std::fprintf(stderr, "vla(octo): diffusion head emits %lld values, not horizon*dim=%d\n",
                         (long long) w.routw->ne[1], action);
            return nullptr;
        }

        ggml_tensor * obs = ggml_new_tensor_2d(C, GGML_TYPE_F32, kHidden, width);
        ggml_set_name(obs, "action_head.readout_embedding");
        ggml_set_input(obs);
        io.obs = obs;

        ggml_tensor * x0 = ggml_new_tensor_2d(C, GGML_TYPE_F32, action, width);
        ggml_set_name(x0, "action_head.initial_noise");
        ggml_set_input(x0);
        io.x0 = x0;

        ggml_tensor * z_all = ggml_new_tensor_3d(C, GGML_TYPE_F32, action, width, steps);
        ggml_set_name(z_all, "action_head.step_noise");
        ggml_set_input(z_all);
        io.z = z_all;

        ggml_tensor * times_all = ggml_new_tensor_3d(C, GGML_TYPE_F32, 1, width, steps);
        ggml_set_name(times_all, "action_head.time");
        ggml_set_input(times_all);
        io.times = times_all;

        ggml_tensor * x = x0;
        for (int step=0; step<steps; ++step) {
            const int   time_value = steps-1-step;
            const float alpha      = sched.alphas[(size_t) time_value];
            const float beta       = sched.betas[(size_t) time_value];
            const float alpha_hat  = sched.alpha_hats[(size_t) time_value];
            const float alpha_1    = 1.0f/std::sqrt(alpha);
            const float alpha_2    = (1.0f-alpha)/std::sqrt(1.0f-alpha_hat);

            ggml_tensor * time = ggml_view_2d(C, times_all, 1, width, times_all->nb[1],
                                              (size_t) step*times_all->nb[2]);
            ggml_tensor * eps  = build_score_actor(C, w, time, obs, x);
            ggml_tensor * y    = ggml_scale(C, ggml_add(C, x, ggml_scale(C, eps, -alpha_2)), alpha_1);
            if (time_value > 0) {
                ggml_tensor * zs = ggml_view_2d(C, z_all, action, width, z_all->nb[1],
                                                (size_t) step*z_all->nb[2]);
                y = ggml_add(C, y, ggml_scale(C, zs, std::sqrt(beta)));
            }
            x = ggml_clamp(C, y, -diff.max_action, diff.max_action);
        }

        // sample_actions returns the last window timestep's chunk.
        ggml_tensor * out = ggml_cont(C, ggml_view_1d(C, x, action, (size_t) (width-1)*x->nb[1]));
        ggml_set_name(out, "action_head.final_actions");
        ggml_set_output(out);
        io.out = out;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 4096, false);
        ggml_build_forward_expand(gf, out);
        return gf;
    });
    if (!built) {
        std::fprintf(stderr, "vla(octo): diffusion graph build failed\n");
        return false;
    }

    OctoRuntime::DiffIO& io = rt.diffusion.io();
    ggml_backend_tensor_set(io.obs,   readout_action.data(), 0, ggml_nbytes(io.obs));
    ggml_backend_tensor_set(io.x0,    initial_noise.data(),  0, ggml_nbytes(io.x0));
    ggml_backend_tensor_set(io.z,     z.data(),              0, ggml_nbytes(io.z));
    ggml_backend_tensor_set(io.times, times.data(),          0, ggml_nbytes(io.times));
    if (!octo_compute(rt, rt.diffusion.graph(), "diffusion"))
        return false;
    final_actions.resize((size_t) action);
    ggml_backend_tensor_get(io.out, final_actions.data(), 0, ggml_nbytes(io.out));
    return true;
}

// ContinuousActionHead: a MAP head over the readout rows, then the mean
// projection. Replaces the diffusion head entirely when head_type is "l1".
bool run_l1_action_head_graph(OctoRuntime& rt,
                              const std::vector<float>& readout_action,
                              int window_size,
                              int action_total,
                              float max_action,
                              std::vector<float>& final_actions) {
    constexpr int   map_heads    = 8;               // 8, not the block transformer's 6.
    constexpr int   map_head_dim = kHidden/map_heads;
    constexpr float ln_eps       = 1e-6f;
    const int width = window_size;
    if (readout_action.size() != (size_t) kHidden*width || action_total <= 0)
        return false;

    const OctoRuntime::L1Key key{width, action_total};
    const bool built = rt.l1_head.ensure(rt.backend, key, (size_t) 8*1024*1024,
                                         [&](ggml_context * C, OctoRuntime::L1IO& io) -> ggml_cgraph * {
        ggml_tensor * probe      = rt.weight("octo.head.l1.map.probe");
        ggml_tensor * qkv_w      = rt.weight("octo.head.l1.map.attn_qkv.weight");
        ggml_tensor * qkv_b      = rt.weight("octo.head.l1.map.attn_qkv.bias");
        ggml_tensor * o_w        = rt.weight("octo.head.l1.map.attn_o.weight");
        ggml_tensor * o_b        = rt.weight("octo.head.l1.map.attn_o.bias");
        ggml_tensor * norm_w     = rt.weight("octo.head.l1.map.norm.weight");
        ggml_tensor * norm_b     = rt.weight("octo.head.l1.map.norm.bias");
        ggml_tensor * ffn_up_w   = rt.weight("octo.head.l1.map.ffn_up.weight");
        ggml_tensor * ffn_up_b   = rt.weight("octo.head.l1.map.ffn_up.bias");
        ggml_tensor * ffn_down_w = rt.weight("octo.head.l1.map.ffn_down.weight");
        ggml_tensor * ffn_down_b = rt.weight("octo.head.l1.map.ffn_down.bias");
        ggml_tensor * mean_w     = rt.weight("octo.head.l1.mean_proj.weight");
        ggml_tensor * mean_b     = rt.weight("octo.head.l1.mean_proj.bias");
        if (!probe || !qkv_w || !qkv_b || !o_w || !o_b || !norm_w || !norm_b ||
            !ffn_up_w || !ffn_up_b || !ffn_down_w || !ffn_down_b || !mean_w || !mean_b)
            return nullptr;

        ggml_tensor * x = ggml_new_tensor_2d(C, GGML_TYPE_F32, kHidden, width);
        ggml_set_name(x, "l1_head.readout_action");
        ggml_set_input(x);
        io.readout = x;

        // Q comes from the probe and K/V from x through the SAME combined
        // in_proj_weight, so each needs its own mul_mat and drops the slices it
        // does not use -- nn.MultiheadAttention keeps the [Wq;Wk;Wv] row blocks
        // whatever is fed through it.
        ggml_tensor * qkv_probe = ggml_add(C, ggml_mul_mat(C, qkv_w, probe), qkv_b);
        ggml_tensor * q         = ggml_cont(C, ggml_view_2d(C, qkv_probe, kHidden, 1, qkv_probe->nb[1], 0));
        ggml_tensor * qkv_x     = ggml_add(C, ggml_mul_mat(C, qkv_w, x), qkv_b);
        ggml_tensor * k         = ggml_cont(C, ggml_view_2d(C, qkv_x, kHidden, width, qkv_x->nb[1], (size_t) kHidden*qkv_x->nb[0]));
        ggml_tensor * v         = ggml_cont(C, ggml_view_2d(C, qkv_x, kHidden, width, qkv_x->nb[1], (size_t) 2*kHidden*qkv_x->nb[0]));

        // Heads on ne2 and window on ne3 are both batch axes mul_mat loops over,
        // never cross-multiplied, which keeps each timestep independent.
        ggml_tensor * Qh = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, q, map_head_dim, map_heads, 1, 1), 0, 2, 1, 3));
        ggml_tensor * Kh = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, k, map_head_dim, map_heads, 1, width), 0, 2, 1, 3));
        ggml_tensor * Vh = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, v, map_head_dim, map_heads, 1, width), 1, 2, 0, 3));

        ggml_tensor * scores = ggml_mul_mat(C, Qh, Kh);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        // One readout token per timestep, so this softmax is over a single logit
        // and always yields 1.0. Kept as the real op in case that changes.
        ggml_tensor * probs    = ggml_soft_max_ext(C, scores, nullptr, 1.0f/std::sqrt((float) map_head_dim), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(C, Vh, probs);
        ggml_tensor * merged   = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, attended, 0, 2, 1, 3)), kHidden, width);

        ggml_tensor * attn_out = ggml_add(C, ggml_mul_mat(C, o_w, merged), o_b);
        ggml_tensor * y        = ggml_add(C, ggml_mul(C, ggml_norm(C, attn_out, ln_eps), norm_w), norm_b);
        ggml_tensor * h        = ggml_gelu_erf(C, ggml_add(C, ggml_mul_mat(C, ffn_up_w, y), ffn_up_b));
        h = ggml_add(C, ggml_mul_mat(C, ffn_down_w, h), ffn_down_b);
        // The residual is onto attn_out, before the norm, as in MAPHead.
        ggml_tensor * emb = ggml_add(C, attn_out, h);

        ggml_tensor * mean_raw = ggml_add(C, ggml_mul_mat(C, mean_w, emb), mean_b);
        ggml_tensor * out = ggml_scale(C, ggml_tanh(C, ggml_scale(C, mean_raw, 1.0f/max_action)), max_action);
        ggml_set_name(out, "l1_head.mean_normalized");
        ggml_set_output(out);
        if (ggml_nelements(out) != (int64_t) action_total*width) {
            std::fprintf(stderr, "vla(octo): L1 mean_proj out-dim=%lld does not match horizon*dim=%d\n",
                         (long long) out->ne[0], action_total);
            return nullptr;
        }
        io.out = out;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 512, false);
        ggml_build_forward_expand(gf, out);
        return gf;
    });
    if (!built) {
        std::fprintf(stderr, "vla(octo): L1 head graph build failed\n");
        return false;
    }

    OctoRuntime::L1IO& io = rt.l1_head.io();
    ggml_backend_tensor_set(io.readout, readout_action.data(), 0, ggml_nbytes(io.readout));
    if (!octo_compute(rt, rt.l1_head.graph(), "L1 head"))
        return false;

    std::vector<float> mean_normalized((size_t) action_total*width);
    ggml_backend_tensor_get(io.out, mean_normalized.data(), 0, ggml_nbytes(io.out));
    // predict_action takes the last window timestep, as the diffusion head does.
    final_actions.resize((size_t) action_total);
    std::copy_n(mean_normalized.begin()+(ptrdiff_t) ((size_t) action_total*(width-1)),
                (size_t) action_total, final_actions.begin());
    return true;
}

bool read_kv_u8_array(const gguf_reader& g, const char * key, std::vector<uint8_t>& out) {
    const int64_t id = gguf_find_key(g.gctx, key);
    if (id < 0) {
        std::fprintf(stderr, "vla(octo): missing metadata %s\n", key);
        return false;
    }
    if (gguf_get_kv_type(g.gctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g.gctx, id) != GGUF_TYPE_UINT8) {
        std::fprintf(stderr, "vla(octo): %s is not a UINT8 array\n", key);
        return false;
    }
    const size_t    n    = gguf_get_arr_n(g.gctx, id);
    const uint8_t * data = (const uint8_t *) gguf_get_arr_data(g.gctx, id);
    out.assign(data, data+n);
    return true;
}

// Which top-level key of octo.dataset_statistics to un-normalize against, when
// the caller did not pin one down. In order: VLA_OCTO_UNNORM_DATASET, the sole
// key if there is only one, then bridge_dataset, which is what the pretrain
// checkpoint's ~25-key OXE mix was always read against. Several real candidates
// and no hint is a failure, not a guess.
bool resolve_unnorm_dataset_key(const nlohmann::json& stats, std::string& key) {
    if (const char * env = std::getenv("VLA_OCTO_UNNORM_DATASET"); env && env[0] != '\0') {
        key = env;
        return true;
    }
    if (stats.is_object() && stats.size() == 1) {
        key = stats.begin().key();
        return true;
    }
    if (stats.is_object() && stats.contains("bridge_dataset")) {
        key = "bridge_dataset";
        return true;
    }
    std::fprintf(stderr,
                 "vla(octo): cannot resolve the unnorm dataset key (%zu candidates in "
                 "octo.dataset_statistics); set VLA_OCTO_UNNORM_DATASET\n",
                 stats.is_object() ? stats.size() : (size_t) 0);
    return false;
}

// Single-dataset checkpoints write the stats block flat, with no dataset-name
// wrapper. A member literally named "action" holding a "mean" is what tells the
// two shapes apart; no real dataset name collides with that.
bool resolve_stats_block(const nlohmann::json& j, const std::string& dataset_key_in,
                         const nlohmann::json ** out) {
    if (j.is_object() && j.contains("action") && j["action"].is_object() && j["action"].contains("mean")) {
        *out = &j;
        return true;
    }
    std::string dataset_key = dataset_key_in;
    if (dataset_key.empty() && !resolve_unnorm_dataset_key(j, dataset_key))
        return false;
    if (!j.contains(dataset_key)) {
        std::fprintf(stderr, "vla(octo): dataset_statistics has no key %s\n", dataset_key.c_str());
        return false;
    }
    *out = &j[dataset_key];
    return true;
}

// Parses octo.dataset_statistics once and caches the blocks on `rt`: it is a
// JSON blob in the metadata, ~25 datasets wide for the pretrain checkpoint, and
// re-reading 21 floats out of it per request is pure overhead.
bool ensure_stats(OctoRuntime& rt, gguf_reader& g, const std::string& dataset_key_in, int64_t action_dim) {
    if (rt.stats_loaded && rt.stats_key == dataset_key_in)
        return true;

    const std::string stats_json = g.str("octo.dataset_statistics");
    if (stats_json.empty()) {
        std::fprintf(stderr, "vla(octo): missing octo.dataset_statistics\n");
        return false;
    }
    nlohmann::json j = nlohmann::json::parse(stats_json, nullptr, false);
    if (j.is_discarded()) {
        std::fprintf(stderr, "vla(octo): octo.dataset_statistics is not valid JSON\n");
        return false;
    }
    const nlohmann::json * block = nullptr;
    if (!resolve_stats_block(j, dataset_key_in, &block))
        return false;
    if (!block->contains("action")) {
        std::fprintf(stderr, "vla(octo): dataset_statistics block has no .action\n");
        return false;
    }

    const auto& act = (*block)["action"];
    OctoRuntime::ActionStats a;
    a.mean = act.at("mean").get<std::vector<float>>();
    a.stdv = act.at("std").get<std::vector<float>>();
    for (bool b : act.at("mask").get<std::vector<bool>>())
        a.mask.push_back(b ? 1 : 0);
    if (a.mask.size() != (size_t) action_dim || a.mean.size() != (size_t) action_dim ||
        a.stdv.size() != (size_t) action_dim) {
        std::fprintf(stderr, "vla(octo): dataset_statistics/action is not %lld-dim\n", (long long) action_dim);
        return false;
    }

    OctoRuntime::ProprioStats pr;
    bool has_pr = false;
    if (block->contains("proprio")) {
        const auto& p = (*block)["proprio"];
        pr.mean = p.at("mean").get<std::vector<float>>();
        pr.stdv = p.at("std").get<std::vector<float>>();
        if (pr.mean.size() != (size_t) kProprioTokens || pr.stdv.size() != (size_t) kProprioTokens) {
            std::fprintf(stderr, "vla(octo): dataset_statistics/proprio is not %d-dim\n", kProprioTokens);
            return false;
        }
        has_pr = true;
    }

    rt.action_stats      = std::move(a);
    rt.proprio_stats     = std::move(pr);
    rt.has_proprio_stats = has_pr;
    rt.stats_key         = dataset_key_in;
    rt.stats_loaded      = true;
    return true;
}

// unnorm[d] = mask[d] ? norm[d]*std[d]+mean[d] : norm[d]. The masked-out dims
// (the gripper, in every LIBERO block) pass through untouched.
bool unnormalize_action(const OctoRuntime& rt,
                        const std::vector<float>& normalized_flat,
                        std::vector<float>& unnorm_flat) {
    const OctoRuntime::ActionStats& a = rt.action_stats;
    const size_t dim = a.mask.size();
    if (dim == 0 || normalized_flat.empty() || normalized_flat.size()%dim != 0) {
        std::fprintf(stderr, "vla(octo): unexpected action shape for un-normalization\n");
        return false;
    }

    const size_t horizon = normalized_flat.size()/dim;
    unnorm_flat.resize(normalized_flat.size());
    for (size_t t=0; t<horizon; ++t) {
        for (size_t d=0; d<dim; ++d) {
            const float norm = normalized_flat[t*dim+d];
            unnorm_flat[t*dim+d] = a.mask[d] ? norm*a.stdv[d]+a.mean[d] : norm;
        }
    }
    return true;
}

// The server path already receives side x side frames, which is what
// modules/preprocess.h expects for every arch; vla-cli can hand us any file.
void resize_hwc(const uint8_t * src, int sw, int sh, int side, std::vector<uint8_t>& dst) {
    dst.resize((size_t) 3*side*side);
    for (int y=0; y<side; ++y) {
        const int sy = std::min(sh-1, (int) ((int64_t) y*sh/side));
        for (int x=0; x<side; ++x) {
            const int       sx  = std::min(sw-1, (int) ((int64_t) x*sw/side));
            const uint8_t * px  = src+((size_t) sy*sw+sx)*3;
            uint8_t *       out = dst.data()+((size_t) y*side+x)*3;
            out[0] = px[0];
            out[1] = px[1];
            out[2] = px[2];
        }
    }
}

bool octo_prepare_view(const ImageView& v, int side, std::vector<float>& out) {
    if (v.format != PixelFormat::U8 || !v.data) {
        std::fprintf(stderr, "vla(octo): predict only supports PixelFormat::U8 images\n");
        return false;
    }
    if (v.w == side && v.h == side)
        return preprocess_image_chw("octo", v, side, out);

    std::vector<uint8_t> resized;
    resize_hwc((const uint8_t *) v.data, v.w, v.h, side, resized);
    const ImageView rv{resized.data(), side, side, PixelFormat::U8};
    return preprocess_image_chw("octo", rv, side, out);
}

struct OctoFrame {
    std::vector<float>   primary;      ///< [3,primary_size,primary_size] normalized.
    std::vector<float>   wrist;        ///< Empty when the client sent no wrist view.
    bool                 wrist_real = false;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    std::vector<float>   proprio_raw;  ///< Original units; z-scored here, not by the caller.
    const float *        noise = nullptr;
};

bool run_pipeline(OctoModelArch& m, const OctoFrame& f,
                  std::vector<float>& unnormalized_out,
                  float& ms_vision_out, float& ms_inference_out) {
    using clock = std::chrono::steady_clock;

    OctoRuntime& rt = m.rt;
    if (m.head_type != "diffusion" && m.head_type != "l1") {
        std::fprintf(stderr, "vla(octo): head_type=%s is not implemented\n", m.head_type.c_str());
        return false;
    }

    const int window_size  = (int) m.window_size;
    const int action_total = (int) (m.action_horizon*m.action_dim);
    const int n_proprio    = m.has_proprio ? kProprioTokens : 0;
    if (!ensure_stats(rt, m.io, "", m.action_dim))
        return false;

    // Cold start: history is filled with copies of the one live frame and every
    // slot but the last is marked padding, as HistoryWrapper.reset() does.
    std::vector<uint8_t> primary_valid ((size_t) window_size, 1);
    std::vector<uint8_t> wrist_valid   ((size_t) window_size, f.wrist_real ? 1 : 0);
    std::vector<uint8_t> timestep_valid((size_t) window_size, 0);
    timestep_valid[(size_t) window_size-1] = 1;

    const OctoSeqLayout layout = build_seq_layout(true, primary_valid, wrist_valid, timestep_valid,
                                                  window_size, (int) m.primary_tokens,
                                                  (int) m.wrist_tokens, n_proprio);

    const auto t_vision0 = clock::now();
    std::vector<float> primary_pos, wrist_pos, proprio_pos;
    // Every live window slot holds the same frame, so it is replicated rather
    // than re-decoded. Language-only conditioning means no goal image: the task
    // frame is a zero uint8 image, which normalizes to -1.
    auto tokenize_view = [&](graph_cache<OctoRuntime::ObsKey, OctoRuntime::ObsIO>& cache,
                             const char * view, const std::vector<float>& frame, int side, int n_tok,
                             const std::vector<int32_t>& steps, std::vector<float>& out) {
        const size_t n = (size_t) 3*side*side;
        if (frame.size() != n) {
            std::fprintf(stderr, "vla(octo): %s frame is %zu floats, expected %zu\n", view, frame.size(), n);
            return false;
        }
        std::vector<float> obs(n*steps.size());
        for (size_t i=0; i<steps.size(); ++i)
            std::copy(frame.begin(), frame.end(), obs.begin()+(ptrdiff_t) (i*n));

        const std::vector<float> task(n, -1.0f);
        return run_obs_tokenizer_graph(rt, cache, view, obs, task, side, n_tok, (int) steps.size(), steps, out);
    };

    if (!layout.primary_steps.empty() &&
        !tokenize_view(rt.obs_primary, "primary", f.primary, (int) m.primary_size,
                       (int) m.primary_tokens, layout.primary_steps, primary_pos))
        return false;
    if (!layout.wrist_steps.empty() &&
        !tokenize_view(rt.obs_wrist, "wrist", f.wrist, (int) m.wrist_size,
                       (int) m.wrist_tokens, layout.wrist_steps, wrist_pos))
        return false;

    if (!layout.proprio_steps.empty()) {
        if (!rt.has_proprio_stats) {
            std::fprintf(stderr, "vla(octo): proprio checkpoint but dataset_statistics has no .proprio block\n");
            return false;
        }
        const std::vector<float>& mean    = rt.proprio_stats.mean;
        const std::vector<float>& stdv    = rt.proprio_stats.stdv;
        const int                 n_steps = (int) layout.proprio_steps.size();

        std::vector<float> proprio_norm((size_t) n_steps*kProprioTokens);
        for (int t=0; t<n_steps; ++t) {
            for (int d=0; d<kProprioTokens; ++d) {
                const float raw = d < (int) f.proprio_raw.size() ? f.proprio_raw[(size_t) d] : 0.0f;
                proprio_norm[(size_t) t*kProprioTokens+d] = (raw-mean[(size_t) d])/stdv[(size_t) d];
            }
        }
        if (!run_proprio_tokenizer_graph(rt, proprio_norm, (int) m.proprio_in_dim, n_steps,
                                         layout.proprio_steps, proprio_pos))
            return false;
    }
    ms_vision_out = std::chrono::duration<float, std::milli>(clock::now()-t_vision0).count();

    const auto t_inference0 = clock::now();

    std::vector<int32_t> lang_key = f.input_ids;
    lang_key.insert(lang_key.end(), f.attention_mask.begin(), f.attention_mask.end());
    if (rt.lang_key != lang_key || rt.lang_steps != window_size) {
        std::vector<float> t5_out;
        if (!run_t5_encoder_graph(rt, f.input_ids, f.attention_mask, t5_out))
            return false;
        if (!run_language_graph(rt, t5_out, window_size, rt.lang_pos, rt.lang_repeated))
            return false;
        rt.lang_key   = std::move(lang_key);
        rt.lang_steps = window_size;
    }

    ggml_tensor * readout_pos_r = rt.weight("octo.readout.action.pos_embd");
    if (!readout_pos_r)
        return false;
    const std::vector<float> readout_pos = tensor_to_vec(readout_pos_r);

    std::vector<float> input, mask;
    if (!assemble_transformer_input(layout, rt.lang_pos, primary_pos, wrist_pos, proprio_pos,
                                    rt.lang_repeated, readout_pos, input))
        return false;
    build_transformer_mask(layout, mask);

    std::vector<float> readout_action;
    if (!run_transformer_graph(rt, layout, input, mask, readout_action))
        return false;

    std::vector<float> normalized;
    if (m.head_type == "l1") {
        if (!run_l1_action_head_graph(rt, readout_action, window_size, action_total, m.max_action, normalized))
            return false;
    } else {
        const OctoDiffusion diff{(int) m.diffusion_steps, m.diffusion_s, m.max_action};
        if (!run_diffusion(rt, m.rng, f.noise, readout_action, window_size, action_total, diff, normalized))
            return false;
    }
    ms_inference_out = std::chrono::duration<float, std::milli>(clock::now()-t_inference0).count();

    return unnormalize_action(rt, normalized, unnormalized_out);
}

}  // namespace

std::unique_ptr<ModelArchBase> octo_create(const std::string& mmproj_path,
                                           const std::string& ckpt_path,
                                           const std::string&) {
    if (!mmproj_path.empty())
        std::printf("vla(octo): note - mmproj '%s' is ignored (Octo ships one GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<OctoModelArch>();
    m->gguf_path = ckpt_path;

    if (!m->io.open(ckpt_path))
        return nullptr;
    if (!m->io.has("octo.architecture")) {
        std::fprintf(stderr, "vla(octo): %s is not an Octo GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(m->io, *m))
        return nullptr;

    const Backend b = backend_init("vla(octo)", m->n_threads);
    if (!b.handle)
        return nullptr;
    m->backend = b.handle;

    if (!load_weights(*m, m->io))
        return nullptr;
    m->rt.init(m->backend, m->ctx_weights);

    std::printf("vla(octo): weights resident in %.2f GiB (F32) - head=%s window=%lld horizon=%lld "
                "action_dim=%lld proprio=%s\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0),
                m->head_type.c_str(), (long long) m->window_size, (long long) m->action_horizon,
                (long long) m->action_dim, m->has_proprio ? "yes" : "no");
    return m;
}

bool octo_tokenize_text(const std::string& ckpt_path,
                        const std::string& text,
                        std::vector<int32_t>& input_ids,
                        std::vector<int32_t>& attention_mask) {
    gguf_reader g{"octo"};
    if (!g.open(ckpt_path))
        return false;

    std::vector<uint8_t> spm_bytes;
    if (!read_kv_u8_array(g, "octo.tokenizer.spm_model", spm_bytes))
        return false;
    const uint32_t eos_id     = g.has("octo.tokenizer.eos_id") ? g.u32("octo.tokenizer.eos_id") : 1;
    const uint32_t pad_id     = g.has("octo.tokenizer.pad_id") ? g.u32("octo.tokenizer.pad_id") : 0;
    const int64_t  max_length = g.has("octo.tokens.language") ? g.u32("octo.tokens.language") : kTaskTokens;

    sentencepiece::SentencePieceProcessor sp;
    const auto status = sp.LoadFromSerializedProto(
        absl::string_view(reinterpret_cast<const char *>(spm_bytes.data()), spm_bytes.size()));
    if (!status.ok()) {
        std::fprintf(stderr, "vla(octo): sentencepiece LoadFromSerializedProto failed: %s\n",
                     status.ToString().c_str());
        return false;
    }

    std::vector<int> ids = sp.EncodeAsIds(text);
    if ((int64_t) ids.size() > max_length-1)
        ids.resize((size_t) (max_length-1));
    input_ids.assign(ids.begin(), ids.end());
    input_ids.push_back((int32_t) eos_id);
    attention_mask.assign(input_ids.size(), 1);
    input_ids.resize((size_t) max_length, (int32_t) pad_id);
    attention_mask.resize((size_t) max_length, 0);
    return true;
}

// Unlike the other archs, this returns the action in world units rather than the
// normalized one: Octo's dataset_statistics lives inside the multi-hundred-MB
// checkpoint, not in a small sibling stats.json a client could hold, so
// un-normalizing server-side is the only way a client avoids shipping the GGUF
// to read its metadata. The client still owns the gripper convention.
std::vector<float> OctoModelArch::predict(const Inputs& in) {
    const auto t_total0 = std::chrono::steady_clock::now();
    stats = Stats{};

    if (in.n_images < 1 || !in.images) {
        std::fprintf(stderr, "vla(octo): predict needs at least the primary image\n");
        return {};
    }
    if (in.n_lang != (int) language_tokens) {
        std::fprintf(stderr, "vla(octo): predict expects exactly %lld language tokens "
                             "(t5-base, padding=\"max_length\"), got %d\n",
                     (long long) language_tokens, in.n_lang);
        return {};
    }
    if (in.attention_mask && in.attention_mask_n != (int) language_tokens) {
        std::fprintf(stderr, "vla(octo): attention_mask_n=%d does not match the %lld language tokens\n",
                     in.attention_mask_n, (long long) language_tokens);
        return {};
    }

    OctoFrame f;
    f.wrist_real = in.n_images >= 2;
    if (!octo_prepare_view(in.images[0], (int) primary_size, f.primary))
        return {};
    // A wrist view we were not given is left out of the sequence entirely rather
    // than zero-filled and masked. See build_seq_layout.
    if (f.wrist_real && !octo_prepare_view(in.images[1], (int) wrist_size, f.wrist))
        return {};

    f.input_ids.assign(in.lang_tokens, in.lang_tokens+in.n_lang);
    if (in.attention_mask) {
        f.attention_mask.assign(in.attention_mask, in.attention_mask+in.attention_mask_n);
    } else {
        // T5's encoder needs real padding information. With right-padded ids the
        // pad token carries it, which reproduces the mask the client's own
        // tokenizer would have sent.
        f.attention_mask.resize(f.input_ids.size());
        for (size_t i=0; i<f.input_ids.size(); ++i)
            f.attention_mask[i] = f.input_ids[i] != pad_id ? 1 : 0;
    }

    if (has_proprio) {
        // Raw proprio in original units: run_pipeline z-scores it, so a caller
        // must not. A missing state zero-fills, as the other archs do, but says
        // so: a silently zeroed reading skews every action it produces.
        f.proprio_raw.assign((size_t) kProprioTokens, 0.0f);
        if (in.state) {
            for (int64_t d=0; d<kProprioTokens; ++d)
                f.proprio_raw[(size_t) d] = in.state[d];
        } else {
            std::fprintf(stderr, "vla(octo): proprio checkpoint but Inputs::state is null - "
                                 "predicting from an all-zero state\n");
        }
    }

    // A caller replaying a chunk pins the sample; the rest of the chain has to
    // follow the same trajectory, so the step noise is reseeded with it.
    f.noise = in.noise;
    if (in.noise)
        rng.seed(kReplaySeed);

    std::vector<float> unnormalized;
    float ms_vision    = 0.f;
    float ms_inference = 0.f;
    if (!run_pipeline(*this, f, unnormalized, ms_vision, ms_inference))
        return {};

    stats.ms_vision    = ms_vision;
    stats.ms_inference = ms_inference;
    stats.ms_total     = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now()-t_total0).count();
    return unnormalized;
}

}  // namespace vla
