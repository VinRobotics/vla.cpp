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

// TODO: SnapFlow (arXiv 2604.05656) distills cfg.num_steps to 1; load the
// distilled action-expert weights and force num_steps = 1 at the denoise loops.

#include "arch.h"
#include "gguf_reader.h"
#include "layers/attn.h"
#include "layers/embed.h"
#include "layers/ffn.h"
#include "layers/norm.h"
#include "layers/rope.h"
#include "modules/siglip_vit.h"
#include "options.h"
#include "model.h"
#include "modules/preprocess.h"
#include "scratch_ctx.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "backend.h"

#include "nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vla {

namespace {

using json = nlohmann::json;

struct st_tensor_info {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t off_begin;
    uint64_t off_end;
};

struct safetensors {
    std::ifstream file;
    uint64_t data_blob_start = 0;
    std::map<std::string, st_tensor_info> tensors;

    bool open(const std::string & path) {
        file.open(path, std::ios::binary);
        if (!file)
            return false;
        uint64_t header_size = 0;
        file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
        std::string header_str(header_size, '\0');
        file.read(header_str.data(), header_size);
        data_blob_start = sizeof(uint64_t)+header_size;
        json j = json::parse(header_str);
        for (auto it=j.begin(); it!=j.end(); ++it) {
            if (it.key() == "__metadata__")
                continue;
            const auto & v = it.value();
            st_tensor_info info;
            info.dtype = v.at("dtype").get<std::string>();
            info.shape = v.at("shape").get<std::vector<int64_t>>();
            info.off_begin = v.at("data_offsets")[0].get<uint64_t>();
            info.off_end   = v.at("data_offsets")[1].get<uint64_t>();
            tensors.emplace(it.key(), std::move(info));
        }
        return true;
    }

    bool read_to_f32(const std::string & name, float * dst,
                     const std::vector<int64_t> & expected_shape) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            std::fprintf(stderr, "vla: tensor not found: %s\n", name.c_str());
            return false;
        }
        const auto & info = it->second;
        if (info.shape != expected_shape) {
            std::fprintf(stderr, "vla: shape mismatch for %s\n", name.c_str());
            return false;
        }
        if (info.dtype != "BF16" && info.dtype != "F32") {
            std::fprintf(stderr, "vla: unsupported dtype for %s: %s\n",
                         name.c_str(), info.dtype.c_str());
            return false;
        }
        // dst is sized from expected_shape, so the declared span has to match it.
        // Without this a file can name the right shape and a longer span.
        const size_t elsz = (info.dtype == "BF16") ? sizeof(ggml_bf16_t) : sizeof(float);
        size_t want = elsz;
        for (const int64_t d : info.shape) {
            if (d < 0) {
                std::fprintf(stderr, "vla: negative dim for %s\n", name.c_str());
                return false;
            }
            want *= (size_t) d;
        }
        if (info.off_end < info.off_begin || info.off_end-info.off_begin != want) {
            std::fprintf(stderr, "vla: bad data_offsets for %s\n", name.c_str());
            return false;
        }
        const size_t bytes = info.off_end-info.off_begin;
        file.seekg(data_blob_start+info.off_begin, std::ios::beg);
        if (info.dtype == "BF16") {
            std::vector<ggml_bf16_t> tmp(bytes/sizeof(ggml_bf16_t));
            file.read(reinterpret_cast<char *>(tmp.data()), bytes);
            ggml_bf16_to_fp32_row(tmp.data(), dst, tmp.size());
        } else {
            file.read(reinterpret_cast<char *>(dst), bytes);
        }
        return !file.fail();
    }

    bool read_raw(const std::string & name, void * dst, size_t expected_bytes,
                  const char * expected_dtype) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            std::fprintf(stderr, "vla: tensor not found: %s\n", name.c_str());
            return false;
        }
        const auto & info = it->second;
        if ((info.off_end-info.off_begin) != expected_bytes ||
            info.dtype != expected_dtype) {
            std::fprintf(stderr, "vla: bad raw read for %s\n", name.c_str());
            return false;
        }
        file.seekg(data_blob_start+info.off_begin, std::ios::beg);
        file.read(static_cast<char *>(dst), expected_bytes);
        return true;
    }
};

struct gguf_source : gguf_reader {
    gguf_source() : gguf_reader("smolvla") {}

    static bool shape_matches(const ggml_tensor * t, const std::vector<int64_t> & pt_shape) {
        const int nd_used = std::max(1, (int) pt_shape.size());
        if (nd_used > GGML_MAX_DIMS)
            return false;
        for (int d=0; d<(int) pt_shape.size(); ++d) {
            const int64_t expected = pt_shape[pt_shape.size()-1-d];
            if (t->ne[d] != expected)
                return false;
        }
        for (int d=(int) pt_shape.size(); d<GGML_MAX_DIMS; ++d) {
            if (t->ne[d] != 1)
                return false;
        }
        return true;
    }

    bool read_to_f32(const std::string & name, float * dst,
                     const std::vector<int64_t> & expected_shape) {
        const ggml_tensor * t = meta(name.c_str());
        if (!t) {
            std::fprintf(stderr, "vla(smolvla): gguf tensor not found: %s\n", name.c_str());
            return false;
        }
        if (!shape_matches(t, expected_shape)) {
            std::fprintf(stderr, "vla(smolvla): gguf shape mismatch for %s\n", name.c_str());
            return false;
        }
        const size_t bytes = ggml_nbytes(t);
        if (t->type == GGML_TYPE_F32)
            return read_raw(name.c_str(), dst, bytes);
        // A requantized file (scripts/quantize_gguf.py) may pack a tensor
        // this model keeps float; unpack it rather than refuse the file.
        const ggml_type_traits * tt = ggml_get_type_traits(t->type);
        if (!tt->to_float) {
            std::fprintf(stderr, "vla(smolvla): gguf cannot convert %s for %s\n",
                         ggml_type_name(t->type), name.c_str());
            return false;
        }
        std::vector<uint8_t> tmp(bytes);
        if (!read_raw(name.c_str(), tmp.data(), bytes))
            return false;
        tt->to_float(tmp.data(), dst, ggml_nelements(t));
        return true;
    }

    /// Type of a tensor in the file, or GGML_TYPE_COUNT if it is absent.
    ggml_type file_type(const std::string & name) const {
        const ggml_tensor * t = meta(name.c_str());
        return t ? t->type : GGML_TYPE_COUNT;
    }

    /// Bytes stored as they are: the caller made the tensor that type.
    bool read_packed(const std::string & name, void * dst, ggml_type want, size_t expected_bytes) {
        if (file_type(name) != want) {
            std::fprintf(stderr, "vla(smolvla): gguf bad packed read for %s\n", name.c_str());
            return false;
        }
        return read_raw(name.c_str(), dst, expected_bytes);
    }
};

struct LayerW {
    ggml_tensor * Wln_in;
    ggml_tensor * Wq;
    ggml_tensor * Wk;
    ggml_tensor * Wv;
    ggml_tensor * Wo;
    ggml_tensor * Wln_post;
    ggml_tensor * Wgate;
    ggml_tensor * Wup;
    ggml_tensor * Wdown;
};

bool expert_self_attn(const Config & cfg, int64_t i) {
    return cfg.self_attn_every_n > 0 && i%cfg.self_attn_every_n == 0;
}

}

struct SmolVLAModelArch : public ModelArchBase {
    SmolVLAModelArch() : ModelArchBase(Arch::SMOLVLA) {}
    ~SmolVLAModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    // In-tree SigLIP-B/16 vision tower (was llama.cpp clip.cpp mmproj).
    int64_t vit_hidden = 768, vit_layers = 12, vit_heads = 12, vit_inter = 3072;
    int64_t vit_patch = 16, vit_image = 512, vit_scale = 4, vit_n_tokens = 64;
    float   vit_ln_eps = 1e-6f;
    SigLipTower   vit;
    ggml_tensor * mm_fc = nullptr;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;

    ggml_type             weight_dtype = GGML_TYPE_BF16;

    ggml_context * ctx_weights = nullptr;
    scratch_ctx vision_scratch;
    scratch_ctx connector_scratch;

    ggml_tensor *  E_lang   = nullptr;
    ggml_tensor *  Wstate   = nullptr;
    ggml_tensor *  bstate   = nullptr;

    std::vector<LayerW> vlm_layers;

    std::vector<LayerW> expert_layers;
    ggml_tensor *  Wnorm_expert = nullptr;

    ggml_tensor *  W_ain   = nullptr;
    ggml_tensor *  b_ain   = nullptr;
    ggml_tensor *  W_at1   = nullptr;
    ggml_tensor *  b_at1   = nullptr;
    ggml_tensor *  W_at2   = nullptr;
    ggml_tensor *  b_at2   = nullptr;

    ggml_tensor *  W_aout  = nullptr;
    ggml_tensor *  b_aout  = nullptr;

    std::vector<float> state_mean, state_std;
    std::vector<float> action_mean, action_std;

    std::mt19937   rng{std::random_device{}()};

    struct MainIO {
        ggml_tensor *img_emb = nullptr, *lang_ids = nullptr, *state = nullptr, *x0 = nullptr;
        ggml_tensor *mask_prefill = nullptr, *pos_prefill = nullptr, *mask_full = nullptr;
        ggml_tensor *mask_pfx_only = nullptr, *pos_full = nullptr, *pos_rebased = nullptr, *x_t = nullptr;
        std::vector<ggml_tensor *> k_cache, v_cache, k_leaf, v_leaf;
    };
    graph_cache<int, MainIO> main_graph;

    std::vector<ggml_tensor *> time_bcasts;
};

namespace {

// Fused attention in the SigLIP tower. OPT-IN (VLA_SMOLVLA_FA=1), not default.
// It cuts the vision stage from 33.2 ms to 22.1 ms (total 68.5 -> 55.8 ms), which
// is enough to beat compiled PyTorch — but ggml's CUDA flash attention computes
// K/V at F16 regardless of input type (fattn.cu accepts F32 K/V only by
// reinterpreting it as F16), and that measured 92/100 on libero_object against
// 96/100 for explicit attention. evo1 showed the same ~4-5 pp drop, so the
// default stays on the accuracy-preserving path.

// One pre-norm SigLIP encoder block (SmolVLM2 tower), same graph as the other
// in-tree models. Bidirectional attention, F32 score accumulation, tanh GELU.
ggml_tensor * build_siglip_layer(ggml_context * C, const EncCfg & c, const EncBlockW & w,
                                 ggml_tensor * x, int64_t seq) {
    const float scale = 1.0f/std::sqrt((float) c.head_dim);
    ggml_tensor * n1 = layer_norm(C, x, w.ln1w, w.ln1b, c.ln_eps);
    ggml_tensor * Q = to_heads(C, linear(C, w.Wq, w.bq, n1), c.head_dim, c.heads, seq);
    ggml_tensor * K = to_heads(C, linear(C, w.Wk, w.bk, n1), c.head_dim, c.heads, seq);
    ggml_tensor * v = linear(C, w.Wv, w.bv, n1);
    ggml_tensor * att;
    if (vla::flash_attn_enabled()) {
        // The tower runs 1024 tokens (512/16 grid) over 12 layers, so the
        // explicit path below materialises a 1024x1024 score matrix per head —
        // written by the matmul, read and rewritten by the softmax, then read
        // again by the AV matmul. That traffic, not the FLOPs, is why the vision
        // stage is roughly half of smolvla's latency. K/V stay F32 so the
        // numerics match the explicit path; the expert layers below already call
        // this op the same way.
        ggml_tensor * V = to_heads(C, v, c.head_dim, c.heads, seq);
        ggml_tensor * fa = ggml_flash_attn_ext(C, Q, vla::fa_kv(C, K), vla::fa_kv(C, V), nullptr, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        att = ggml_reshape_2d(C, fa, c.hidden, seq);
    } else {
        att = attention(C, Q, K, to_heads_v(C, v, c.head_dim, c.heads, seq), nullptr, scale, c.hidden, seq);
    }
    ggml_tensor * h1 = ggml_add(C, x, linear(C, w.Wo, w.bo, att));
    return ggml_add(C, h1, ffn_gelu(C, w.Wfc1, w.bfc1, w.Wfc2, w.bfc2, layer_norm(C, h1, w.ln2w, w.ln2b, c.ln_eps)));
}

void set_derived_config(Config & cfg) {
    cfg.rms_eps        = 1e-5f;
    cfg.rope_mode      = GGML_ROPE_TYPE_NEOX;
    cfg.rope_freq_base = 10000.f;

    cfg.n_state     = 1;
    cfg.q_full_dim  = cfg.n_q_heads  * cfg.head_dim;
    cfg.kv_full_dim = cfg.n_kv_heads*cfg.head_dim;
    cfg.rope_n_dims = static_cast<int>(cfg.head_dim);
}

bool load_config_from_json(const std::string & path, Config & cfg) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "vla: cannot open config %s\n", path.c_str());
        return false;
    }
    json j;
    try {
        f >> j;
    } catch (const json::exception & e) {
        std::fprintf(stderr, "vla: failed to parse %s: %s\n", path.c_str(), e.what());
        return false;
    }
    for (const char * k : {"adapt_to_pi_aloha", "add_image_special_tokens"}) {
        if (j.contains(k) && j[k].is_boolean() && j[k].get<bool>()) {
            std::fprintf(stderr, "vla(smolvla): %s=true in %s is not supported\n", k, path.c_str());
            return false;
        }
    }

    cfg.hidden        = 960;
    cfg.n_q_heads     = 15;
    cfg.n_kv_heads    = 5;
    cfg.head_dim      = 64;
    cfg.intermediate  = 2560;

    try {
        cfg.n_suffix          = j.at("chunk_size").get<int64_t>();
        cfg.num_steps         = j.at("num_steps").get<int>();
        cfg.max_state_dim     = j.at("max_state_dim").get<int64_t>();
        cfg.max_action_dim    = j.at("max_action_dim").get<int64_t>();
        cfg.min_period        = j.at("min_period").get<double>();
        cfg.max_period        = j.at("max_period").get<double>();
        cfg.self_attn_every_n = j.at("self_attn_every_n_layers").get<int>();
        cfg.n_lang            = j.at("tokenizer_max_length").get<int64_t>();

        cfg.n_layers          = j.at("num_vlm_layers").get<int64_t>();

        const double mul      = j.at("expert_width_multiplier").get<double>();
        cfg.expert_h          = static_cast<int64_t>(std::round(double(cfg.hidden)*mul));

        cfg.real_state_dim  = j.at("input_features").at("observation.state").at("shape").at(0).get<int64_t>();
        cfg.real_action_dim = j.at("output_features").at("action").at("shape").at(0).get<int64_t>();
    } catch (const json::exception & e) {
        std::fprintf(stderr, "vla: missing/bad field in %s: %s\n", path.c_str(), e.what());
        return false;
    }

    set_derived_config(cfg);
    cfg.norm_eps    = 1e-8f;
    return true;
}

void load_normalizer_stats(const std::string & model_dir, SmolVLAModelArch & m) {
    const auto & cfg = m.cfg;

    m.state_mean .assign(cfg.real_state_dim,  0.f);
    m.state_std  .assign(cfg.real_state_dim,  1.f);
    m.action_mean.assign(cfg.real_action_dim, 0.f);
    m.action_std .assign(cfg.real_action_dim, 1.f);

    auto load_one = [&](const std::string & meta_path, const std::string & registry_name,
                        const std::string & mean_key, const std::string & std_key,
                        std::vector<float> & mean_out, std::vector<float> & std_out,
                        int64_t expected_dim, const char * label) {
        std::ifstream f(meta_path);
        if (!f) {
            std::printf("vla: %s: %s not found - using identity stats\n",
                        label, meta_path.c_str());
            return;
        }
        json meta;
        try { f >> meta; } catch (const json::exception & e) {
            std::fprintf(stderr, "vla: %s: failed to parse %s: %s\n",
                         label, meta_path.c_str(), e.what());
            return;
        }

        std::string state_file;
        for (const auto & step : meta.at("steps")) {
            if (step.value("registry_name", std::string{}) == registry_name) {
                state_file = step.value("state_file", std::string{});
                if (step.contains("config") && step["config"].contains("eps")) {
                    m.cfg.norm_eps = step["config"].at("eps").get<float>();
                }
                break;
            }
        }
        if (state_file.empty()) {
            std::printf("vla: %s: no %s step in %s - using identity stats\n",
                        label, registry_name.c_str(), meta_path.c_str());
            return;
        }
        const std::string sf_path = model_dir + "/" + state_file;
        safetensors st;
        if (!st.open(sf_path)) {
            std::fprintf(stderr, "vla: %s: cannot open %s\n", label, sf_path.c_str());
            return;
        }
        if (st.tensors.find(mean_key) == st.tensors.end() ||
            st.tensors.find(std_key)  == st.tensors.end()) {
            std::printf("vla: %s: %s lacks '%s'/'%s' - using identity stats\n",
                        label, sf_path.c_str(), mean_key.c_str(), std_key.c_str());
            return;
        }
        if (!st.read_to_f32(mean_key, mean_out.data(), {expected_dim}) ||
            !st.read_to_f32(std_key,  std_out.data(),  {expected_dim})) {
            std::fprintf(stderr, "vla: %s: failed to load %s/%s from %s\n",
                         label, mean_key.c_str(), std_key.c_str(), sf_path.c_str());
            mean_out.assign(expected_dim, 0.f);
            std_out .assign(expected_dim, 1.f);
            return;
        }
        std::printf("vla: %s stats loaded from %s\n", label, sf_path.c_str());
    };

    load_one(model_dir + "/policy_preprocessor.json", "normalizer_processor",
             "observation.state.mean", "observation.state.std",
             m.state_mean, m.state_std, cfg.real_state_dim, "state");
    load_one(model_dir + "/policy_postprocessor.json", "unnormalizer_processor",
             "action.mean", "action.std",
             m.action_mean, m.action_std, cfg.real_action_dim, "action");
}

std::string dir_of(const std::string & path) {
    const auto pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? std::string(".") : path.substr(0, pos);
}

bool ends_with_gguf(const std::string & path) {
    static const std::string sfx = ".gguf";
    return path.size() >= sfx.size()
        && path.compare(path.size()-sfx.size(), sfx.size(), sfx) == 0;
}

bool load_config_from_gguf(const gguf_source & st, Config & cfg) {
    const std::string arch = st.str("smolvla.architecture");
    if (arch != "smolvla") {
        std::fprintf(stderr, "vla(smolvla): gguf architecture = '%s' (expected 'smolvla')\n",
                     arch.c_str());
        return false;
    }

    auto need = [&](const char * key, gguf_type type) -> bool {
        int64_t id;
        if (!st.typed_key(key, type, &id)) {
            std::fprintf(stderr, "vla(smolvla): gguf missing key '%s'\n", key);
            return false;
        }
        return true;
    };

    for (const char * k : {
            "smolvla.hidden", "smolvla.intermediate",
            "smolvla.n_q_heads", "smolvla.n_kv_heads", "smolvla.head_dim",
            "smolvla.n_layers", "smolvla.vocab_size",
            "smolvla.expert_h", "smolvla.expert_inter",
            "smolvla.chunk_size", "smolvla.num_steps",
            "smolvla.max_state_dim", "smolvla.max_action_dim",
            "smolvla.real_state_dim", "smolvla.real_action_dim",
            "smolvla.self_attn_every_n_layers", "smolvla.tokenizer_max_length"}) {
        if (!need(k, GGUF_TYPE_UINT32))
            return false;
    }
    if (!need("smolvla.min_period", GGUF_TYPE_FLOAT64) || !need("smolvla.max_period", GGUF_TYPE_FLOAT64))
        return false;

    cfg.hidden        = st.u32("smolvla.hidden");
    cfg.intermediate  = st.u32("smolvla.intermediate");
    cfg.n_q_heads     = st.u32("smolvla.n_q_heads");
    cfg.n_kv_heads    = st.u32("smolvla.n_kv_heads");
    cfg.head_dim      = st.u32("smolvla.head_dim");
    cfg.n_layers      = st.u32("smolvla.n_layers");
    cfg.expert_h      = st.u32("smolvla.expert_h");
    cfg.expert_inter  = st.u32("smolvla.expert_inter");
    cfg.n_suffix      = st.u32("smolvla.chunk_size");
    cfg.num_steps     = st.u32("smolvla.num_steps");
    cfg.max_state_dim = st.u32("smolvla.max_state_dim");
    cfg.max_action_dim= st.u32("smolvla.max_action_dim");
    cfg.real_state_dim  = st.u32("smolvla.real_state_dim");
    cfg.real_action_dim = st.u32("smolvla.real_action_dim");
    cfg.self_attn_every_n = st.u32("smolvla.self_attn_every_n_layers");
    cfg.n_lang        = st.u32("smolvla.tokenizer_max_length");
    cfg.min_period    = st.f64("smolvla.min_period");
    cfg.max_period    = st.f64("smolvla.max_period");

    set_derived_config(cfg);
    cfg.norm_eps = st.has("smolvla.norm_eps") ? st.f32("smolvla.norm_eps") : 1e-8f;
    return true;
}

bool load_normalizer_stats_from_gguf(gguf_source & st, SmolVLAModelArch & m) {
    const auto & cfg = m.cfg;
    m.state_mean .resize(cfg.real_state_dim);
    m.state_std  .resize(cfg.real_state_dim);
    m.action_mean.resize(cfg.real_action_dim);
    m.action_std .resize(cfg.real_action_dim);
    return st.read_to_f32("state_mean",  m.state_mean.data(),  {cfg.real_state_dim})
        && st.read_to_f32("state_std",   m.state_std.data(),   {cfg.real_state_dim})
        && st.read_to_f32("action_mean", m.action_mean.data(), {cfg.real_action_dim})
        && st.read_to_f32("action_std",  m.action_std.data(),  {cfg.real_action_dim});
}

std::string hf_to_gguf(const std::string & n) {
    static const char * VLM_LAYER_PFX = "model.vlm_with_expert.vlm.model.text_model.";
    static const char * AEX_LAYER_PFX = "model.vlm_with_expert.lm_expert.";
    static const char * MODEL_PFX     = "model.";

    auto map_suffix = [](const std::string & s) -> std::string {
        if (s == "input_layernorm.weight")
            return "attn_norm.weight";
        if (s == "self_attn.q_proj.weight")
            return "attn_q.weight";
        if (s == "self_attn.k_proj.weight")
            return "attn_k.weight";
        if (s == "self_attn.v_proj.weight")
            return "attn_v.weight";
        if (s == "self_attn.o_proj.weight")
            return "attn_o.weight";
        if (s == "post_attention_layernorm.weight")
            return "ffn_norm.weight";
        if (s == "mlp.gate_proj.weight")
            return "ffn_gate.weight";
        if (s == "mlp.up_proj.weight")
            return "ffn_up.weight";
        if (s == "mlp.down_proj.weight")
            return "ffn_down.weight";
        return s;
    };
    auto starts_with = [](const std::string & s, const char * pfx) -> bool {
        const size_t plen = std::strlen(pfx);
        return s.size() >= plen && s.compare(0, plen, pfx) == 0;
    };

    if (n == "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight")
        return "token_embd.weight";
    if (n == "model.vlm_with_expert.lm_expert.norm.weight")
        return "aex.output_norm.weight";

    auto layer_translate = [&](const std::string & rest, const char * dst_blk) -> std::string {

        if (!starts_with(rest, "layers."))
            return n;
        const size_t end_i = rest.find('.', 7);
        if (end_i == std::string::npos)
            return n;
        const std::string idx = rest.substr(7, end_i-7);
        const std::string suf = rest.substr(end_i+1);
        return std::string(dst_blk) + ".blk." + idx + "." + map_suffix(suf);
    };

    if (starts_with(n, VLM_LAYER_PFX)) {
        return layer_translate(n.substr(std::strlen(VLM_LAYER_PFX)), "vlm");
    }
    if (starts_with(n, AEX_LAYER_PFX)) {
        return layer_translate(n.substr(std::strlen(AEX_LAYER_PFX)), "aex");
    }

    static const char * VIS_PFX = "model.vlm_with_expert.vlm.model.vision_model.";
    if (n == "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight")
        return "mm.fc.weight";
    if (starts_with(n, VIS_PFX)) {
        const std::string rest = n.substr(std::strlen(VIS_PFX));
        if (rest == "embeddings.patch_embedding.weight")
            return "vit.patch_embd.weight";
        if (rest == "embeddings.patch_embedding.bias")
            return "vit.patch_embd.bias";
        if (rest == "embeddings.position_embedding.weight")
            return "vit.pos_embd";
        if (rest == "post_layernorm.weight")
            return "vit.post_ln.weight";
        if (rest == "post_layernorm.bias")
            return "vit.post_ln.bias";
        if (starts_with(rest, "encoder.layers.")) {
            const size_t e = rest.find('.', 15);
            if (e == std::string::npos)
                return n;
            const std::string idx = rest.substr(15, e-15);
            const std::string suf = rest.substr(e+1);
            std::string ds;
            if      (suf == "layer_norm1.weight")
                ds = "ln1.weight";
            else if (suf == "layer_norm1.bias")       ds = "ln1.bias";
            else if (suf == "layer_norm2.weight")     ds = "ln2.weight";
            else if (suf == "layer_norm2.bias")       ds = "ln2.bias";
            else if (suf == "self_attn.q_proj.weight")   ds = "attn_q.weight";
            else if (suf == "self_attn.q_proj.bias")     ds = "attn_q.bias";
            else if (suf == "self_attn.k_proj.weight")   ds = "attn_k.weight";
            else if (suf == "self_attn.k_proj.bias")     ds = "attn_k.bias";
            else if (suf == "self_attn.v_proj.weight")   ds = "attn_v.weight";
            else if (suf == "self_attn.v_proj.bias")     ds = "attn_v.bias";
            else if (suf == "self_attn.out_proj.weight") ds = "attn_o.weight";
            else if (suf == "self_attn.out_proj.bias")   ds = "attn_o.bias";
            else if (suf == "mlp.fc1.weight")            ds = "fc1.weight";
            else if (suf == "mlp.fc1.bias")              ds = "fc1.bias";
            else if (suf == "mlp.fc2.weight")            ds = "fc2.weight";
            else if (suf == "mlp.fc2.bias")              ds = "fc2.bias";
            else
                return n;
            return "vit.blk." + idx + "." + ds;
        }
        return n;
    }

    if (starts_with(n, MODEL_PFX)) {

        return n.substr(std::strlen(MODEL_PFX));
    }
    return n;
}


RopeSpec rope_spec(const Config & cfg) {
    return RopeSpec{cfg.rope_mode, cfg.rope_n_dims, {}, cfg.rope_freq_base};
}

ggml_tensor * mm_w(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * r = ggml_mul_mat(ctx, w, x);
    if (vla::mm_prec_f32_enabled())
        ggml_prec_set_acc(r, GGML_PREC_F32);
    return r;
}

ggml_tensor * attn_mlp(ggml_context * ctx, const LayerW & w, ggml_tensor * x_in,
                       ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                       const Config & cfg, int64_t seq) {
    ggml_tensor * Q = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(ctx, vla::fa_kv(ctx, k), 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(ctx, vla::fa_kv(ctx, v), 0, 2, 1, 3);
    const float scale = 1.f/std::sqrt(static_cast<float>(cfg.head_dim));
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.f, 0.f);
    ggml_prec_set_acc(fa, GGML_PREC_F32);
    ggml_tensor * h1 = ggml_add(ctx, x_in, mm_w(ctx, w.Wo, ggml_reshape_2d(ctx, fa, cfg.q_full_dim, seq)));

    ggml_tensor * x_norm_mlp = rms_norm(ctx, h1, w.Wln_post, cfg.rms_eps);
    ggml_tensor * inter      = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, w.Wgate, x_norm_mlp)),
                                        mm_w(ctx, w.Wup, x_norm_mlp));
    return ggml_add(ctx, h1, mm_w(ctx, w.Wdown, inter));
}

ggml_tensor * build_vlm_layer(ggml_context * ctx, const LayerW & w,
                              ggml_tensor * x_in, ggml_tensor * mask,
                              ggml_tensor * positions, const Config & cfg,
                              ggml_tensor ** k_out, ggml_tensor ** v_out) {
    ggml_tensor * x_norm = rms_norm(ctx, x_in, w.Wln_in, cfg.rms_eps);
    ggml_tensor * q_h = ggml_reshape_3d(ctx, mm_w(ctx, w.Wq, x_norm), cfg.head_dim, cfg.n_q_heads,  cfg.n_prefix);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, mm_w(ctx, w.Wk, x_norm), cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);
    *v_out = ggml_reshape_3d(ctx, mm_w(ctx, w.Wv, x_norm), cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);
    *k_out = rope(ctx, rope_spec(cfg), k_h, positions);
    return attn_mlp(ctx, w, x_in, rope(ctx, rope_spec(cfg), q_h, positions), *k_out, *v_out, mask, cfg, cfg.n_prefix);
}

ggml_tensor * build_expert_self_attn_layer(
    ggml_context * ctx, const LayerW & w,
    ggml_tensor * x_in, ggml_tensor * cached_K, ggml_tensor * cached_V,
    ggml_tensor * positions_full, ggml_tensor * mask_full, const Config & cfg)
{
    ggml_tensor * x_norm = rms_norm(ctx, x_in, w.Wln_in, cfg.rms_eps);
    ggml_tensor * q_h = ggml_reshape_3d(ctx, mm_w(ctx, w.Wq, x_norm), cfg.head_dim, cfg.n_q_heads,  cfg.n_suffix);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, mm_w(ctx, w.Wk, x_norm), cfg.head_dim, cfg.n_kv_heads, cfg.n_suffix);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, mm_w(ctx, w.Wv, x_norm), cfg.head_dim, cfg.n_kv_heads, cfg.n_suffix);

    ggml_tensor * K_full = ggml_concat(ctx, cached_K, rope(ctx, rope_spec(cfg), k_h, positions_full), 2);
    ggml_tensor * V_full = ggml_concat(ctx, cached_V, v_h, 2);
    return attn_mlp(ctx, w, x_in, rope(ctx, rope_spec(cfg), q_h, positions_full), K_full, V_full, mask_full, cfg, cfg.n_suffix);
}

// Reproject the constant prefix cache into a cross-attn layer's K/V. Depends only
// on the prefix, so it is built once and shared across all denoise steps.
void expert_cross_kv(ggml_context * ctx, const LayerW & w,
                     ggml_tensor * cached_K, ggml_tensor * cached_V, const Config & cfg,
                     ggml_tensor ** K_repro, ggml_tensor ** V_repro) {
    ggml_tensor * cK_flat = ggml_reshape_2d(ctx, cached_K, cfg.kv_full_dim, cfg.n_prefix);
    ggml_tensor * cV_flat = ggml_reshape_2d(ctx, cached_V, cfg.kv_full_dim, cfg.n_prefix);
    *K_repro = ggml_reshape_3d(ctx, mm_w(ctx, w.Wk, cK_flat), cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);
    *V_repro = ggml_reshape_3d(ctx, mm_w(ctx, w.Wv, cV_flat), cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);
}

ggml_tensor * build_expert_cross_attn_layer(
    ggml_context * ctx, const LayerW & w,
    ggml_tensor * x_in, ggml_tensor * K_repro, ggml_tensor * V_repro,
    ggml_tensor * positions_rebased, ggml_tensor * mask_prefix_only,
    const Config & cfg)
{
    ggml_tensor * x_norm = rms_norm(ctx, x_in, w.Wln_in, cfg.rms_eps);
    ggml_tensor * q_h    = ggml_reshape_3d(ctx, mm_w(ctx, w.Wq, x_norm), cfg.head_dim, cfg.n_q_heads, cfg.n_suffix);
    return attn_mlp(ctx, w, x_in, rope(ctx, rope_spec(cfg), q_h, positions_rebased), K_repro, V_repro,
                    mask_prefix_only, cfg, cfg.n_suffix);
}

}

namespace {

static void vram_probe(ggml_backend_t backend, const char * label) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev)
        return;
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    static size_t prev_free = 0;
    static bool   have_prev = false;
    const double MiB = 1024.0*1024.0;
    const long long used = (long long)(total_b-free_b);
    if (have_prev) {
        const long long delta = (long long)prev_free-(long long)free_b;
        std::printf("vla: [vram] %-22s used=%.1f MiB  free=%.1f MiB  (+%.1f MiB)\n",
                    label, used/MiB, free_b/MiB, delta/MiB);
    } else {
        std::printf("vla: [vram] %-22s used=%.1f MiB  free=%.1f MiB\n",
                    label, used/MiB, free_b/MiB);
    }
    prev_free = free_b;
    have_prev = true;
}


static void backend_set_from_f32(ggml_tensor * t, const float * src, int64_t n) {
    switch (t->type) {
        case GGML_TYPE_F32:
            ggml_backend_tensor_set(t, src, 0, n * sizeof(float));
            break;
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_fp32_to_fp16_row(src, tmp.data(), n);
            ggml_backend_tensor_set(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
            break;
        }
        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> tmp(n);
            ggml_fp32_to_bf16_row(src, tmp.data(), n);
            ggml_backend_tensor_set(t, tmp.data(), 0, n * sizeof(ggml_bf16_t));
            break;
        }
        default:
            std::fprintf(stderr, "vla: backend_set_from_f32: unsupported dtype %d\n",
                         (int) t->type);
            break;
    }
}

std::unique_ptr<SmolVLAModelArch> smolvla_load_impl(ggml_type weight_dtype,
                                                    const std::string& ckpt_path,
                                                    const std::string& config_path) {
    auto m = std::make_unique<SmolVLAModelArch>();

    const bool use_gguf = ends_with_gguf(ckpt_path);
    const std::string cfg_path = config_path.empty() ? dir_of(ckpt_path) + "/config.json" : config_path;
    safetensors  st;
    gguf_source  gst;

    if (use_gguf) {
        if (!gst.open(ckpt_path) || !load_config_from_gguf(gst, m->cfg))
            return nullptr;
        std::printf("vla: config = %s (gguf KV)\n", ckpt_path.c_str());
    } else {
        if (!load_config_from_json(cfg_path, m->cfg))
            return nullptr;
        std::printf("vla: config = %s\n", cfg_path.c_str());
    }
    if (m->cfg.num_steps < 1 || m->cfg.num_steps > 1000) {
        std::fprintf(stderr, "vla(smolvla): num_steps %d out of range [1, 1000]\n", m->cfg.num_steps);
        return nullptr;
    }

    {
        const Backend b = backend_init("vla", default_cpu_threads());
        if (!b.handle)
            return nullptr;
        m->backend = b.handle;
    }
    vram_probe(m->backend, "after backend init");

    m->weight_dtype = weight_dtype;
    std::printf("vla: tower weights resident as %s\n", ggml_type_name(m->weight_dtype));

    // Vision tower geometry: from gguf KV (self-contained ckpt), else SmolVLM2-500M defaults.
    if (use_gguf) {
        auto vu = [&](const char * k, int64_t & d) { if (gst.has(k)) d = (int64_t) gst.u32(k); };
        vu("smolvla.vit_hidden", m->vit_hidden); vu("smolvla.vit_layers", m->vit_layers);
        vu("smolvla.vit_heads",  m->vit_heads);  vu("smolvla.patch_size", m->vit_patch);
        vu("smolvla.image_size", m->vit_image);  vu("smolvla.vit_pixel_shuffle", m->vit_scale);
        vu("smolvla.n_img_tokens", m->vit_n_tokens); vu("smolvla.vit_inter", m->vit_inter);
        if (gst.has("smolvla.vit_ln_eps"))
            m->vit_ln_eps = gst.f32("smolvla.vit_ln_eps");
    }
    {
        if (m->vit_patch <= 0 || m->vit_scale <= 0 || m->vit_heads <= 0 ||
            m->vit_image % m->vit_patch || (m->vit_image/m->vit_patch) % m->vit_scale ||
            m->vit_hidden % m->vit_heads) {
            std::fprintf(stderr, "vla(smolvla): bad vit geometry (image %lld patch %lld shuffle %lld hidden %lld heads %lld)\n",
                         (long long) m->vit_image, (long long) m->vit_patch, (long long) m->vit_scale,
                         (long long) m->vit_hidden, (long long) m->vit_heads);
            return nullptr;
        }
        const int64_t grid = m->vit_image/m->vit_patch;
        const int64_t k = grid/m->vit_scale;
        if (k * k != m->vit_n_tokens) {
            std::fprintf(stderr, "vla: smolvla vit geometry mismatch (grid=%lld scale=%lld -> %lld tokens, KV says %lld)\n",
                         (long long) grid, (long long) m->vit_scale, (long long) (k * k), (long long) m->vit_n_tokens);
            return nullptr;
        }
        m->cfg.n_img = m->vit_n_tokens;
        m->vit.enc.cfg = {m->vit_hidden, m->vit_heads, m->vit_hidden/m->vit_heads, m->vit_ln_eps, false};
    }
    m->cfg.n_prefix = m->cfg.n_img+m->cfg.n_lang+m->cfg.n_state;
    m->cfg.n_full   = m->cfg.n_prefix+m->cfg.n_suffix;

    if (!use_gguf) {
        if (!st.open(ckpt_path)) {
            std::fprintf(stderr, "vla: failed to open %s\n", ckpt_path.c_str());
            return nullptr;
        }
        if (m->cfg.n_layers <= 0) {
            const std::string prefix = "model.vlm_with_expert.vlm.model.text_model.layers.";
            int max_layer = -1;
            for (const auto & kv : st.tensors) {
                if (kv.first.compare(0, prefix.size(), prefix) == 0) {
                    const int idx = std::atoi(kv.first.c_str()+prefix.size());
                    if (idx > max_layer)
                        max_layer = idx;
                }
            }
            if (max_layer < 0) {
                std::fprintf(stderr, "vla: cannot infer n_layers from %s\n", ckpt_path.c_str());
                return nullptr;
            }
            m->cfg.n_layers = max_layer+1;
        }
        {
            const auto it = st.tensors.find("model.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight");
            if (it == st.tensors.end() || it->second.shape.size() != 2) {
                std::fprintf(stderr, "vla: missing/malformed expert gate_proj for shape derivation\n");
                return nullptr;
            }

            m->cfg.expert_inter = it->second.shape[0];
            if (it->second.shape[1] != m->cfg.expert_h) {
                std::fprintf(stderr, "vla: expert_h mismatch - config implies %lld, "
                                     "checkpoint gate_proj has %lld\n",
                             (long long) m->cfg.expert_h, (long long) it->second.shape[1]);
                return nullptr;
            }
        }
    }

    std::printf("vla: cfg  hidden=%lld expert_h=%lld expert_inter=%lld n_layers=%lld "
                "n_img=%lld n_lang=%lld n_suffix=%lld max_state=%lld max_action=%lld\n",
                (long long) m->cfg.hidden,        (long long) m->cfg.expert_h,
                (long long) m->cfg.expert_inter,  (long long) m->cfg.n_layers,
                (long long) m->cfg.n_img,         (long long) m->cfg.n_lang,
                (long long) m->cfg.n_suffix,
                (long long) m->cfg.max_state_dim, (long long) m->cfg.max_action_dim);
    std::printf("vla: cfg  real_state_dim=%lld real_action_dim=%lld\n",
                (long long) m->cfg.real_state_dim, (long long) m->cfg.real_action_dim);

    if (use_gguf) {
        if (!load_normalizer_stats_from_gguf(gst, *m)) {
            std::fprintf(stderr, "vla: failed to load normalizer stats from gguf\n");
            return nullptr;
        }
    } else {
        load_normalizer_stats(dir_of(cfg_path), *m);
    }

    ggml_init_params gparams = {
         size_t(32)*1024*1024,
         nullptr,
         true,
    };
    m->ctx_weights = ggml_init(gparams);
    if (!m->ctx_weights) {
        std::fprintf(stderr, "vla: ggml_init (weights) failed\n");
        return nullptr;
    }

    auto * ctx = m->ctx_weights;
    const auto & cfg = m->cfg;
    const ggml_type wdt = m->weight_dtype;

    struct PendingF32  { std::string name; ggml_tensor * t; std::vector<int64_t> shape; };
    std::vector<PendingF32>  pending_f32;

    m->E_lang = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, cfg.hidden,  49280);

    m->Wstate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.max_state_dim, cfg.hidden);
    m->bstate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.hidden);
    pending_f32.push_back({"model.state_proj.weight", m->Wstate, {cfg.hidden, cfg.max_state_dim}});
    pending_f32.push_back({"model.state_proj.bias",   m->bstate, {cfg.hidden}});

    // Vision tower weights (SigLIP-B/16 encoder + single-linear pixel-shuffle connector).
    {
        const int64_t H = m->vit_hidden, FF = m->vit_inter, P = m->vit_patch;
        const int64_t grid = m->vit_image/P, n_patches = grid * grid;
        const int64_t c4 = H * m->vit_scale*m->vit_scale;
        const char * VP = "model.vlm_with_expert.vlm.model.vision_model.";
        SigLipTower & vt = m->vit;
        vt.patch_w   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, P, P, 3, H);
        vt.patch_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        vt.pos       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_patches);
        vt.post_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        vt.post_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        pending_f32.push_back({std::string(VP) + "embeddings.patch_embedding.weight", vt.patch_w, {H, 3, P, P}});
        pending_f32.push_back({std::string(VP) + "embeddings.patch_embedding.bias",   vt.patch_b, {H}});
        pending_f32.push_back({std::string(VP) + "embeddings.position_embedding.weight", vt.pos, {n_patches, H}});
        pending_f32.push_back({std::string(VP) + "post_layernorm.weight", vt.post_ln_w, {H}});
        pending_f32.push_back({std::string(VP) + "post_layernorm.bias",   vt.post_ln_b, {H}});
        vt.enc.blk.resize(m->vit_layers);
        for (int64_t i=0; i<m->vit_layers; ++i) {
            EncBlockW & w = vt.enc.blk[i];
            char pb[256]; std::snprintf(pb, sizeof(pb), "%sencoder.layers.%lld.", VP, (long long) i);
            const std::string pf = pb;
            w.ln1w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H); w.ln1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            w.ln2w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H); w.ln2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            w.Wq = ggml_new_tensor_2d(ctx, wdt, H, H); w.bq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            w.Wk = ggml_new_tensor_2d(ctx, wdt, H, H); w.bk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            w.Wv = ggml_new_tensor_2d(ctx, wdt, H, H); w.bv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            w.Wo = ggml_new_tensor_2d(ctx, wdt, H, H); w.bo = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            w.Wfc1 = ggml_new_tensor_2d(ctx, wdt, H, FF);  w.bfc1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FF);
            w.Wfc2 = ggml_new_tensor_2d(ctx, wdt, FF, H);  w.bfc2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            pending_f32.push_back({pf + "layer_norm1.weight", w.ln1w, {H}}); pending_f32.push_back({pf + "layer_norm1.bias", w.ln1b, {H}});
            pending_f32.push_back({pf + "layer_norm2.weight", w.ln2w, {H}}); pending_f32.push_back({pf + "layer_norm2.bias", w.ln2b, {H}});
            pending_f32.push_back({pf + "self_attn.q_proj.weight", w.Wq, {H, H}}); pending_f32.push_back({pf + "self_attn.q_proj.bias", w.bq, {H}});
            pending_f32.push_back({pf + "self_attn.k_proj.weight", w.Wk, {H, H}}); pending_f32.push_back({pf + "self_attn.k_proj.bias", w.bk, {H}});
            pending_f32.push_back({pf + "self_attn.v_proj.weight", w.Wv, {H, H}}); pending_f32.push_back({pf + "self_attn.v_proj.bias", w.bv, {H}});
            pending_f32.push_back({pf + "self_attn.out_proj.weight", w.Wo, {H, H}}); pending_f32.push_back({pf + "self_attn.out_proj.bias", w.bo, {H}});
            pending_f32.push_back({pf + "mlp.fc1.weight", w.Wfc1, {FF, H}}); pending_f32.push_back({pf + "mlp.fc1.bias", w.bfc1, {FF}});
            pending_f32.push_back({pf + "mlp.fc2.weight", w.Wfc2, {H, FF}}); pending_f32.push_back({pf + "mlp.fc2.bias", w.bfc2, {H}});
        }
        m->mm_fc = ggml_new_tensor_2d(ctx, wdt, c4, cfg.hidden);
        pending_f32.push_back({"model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
                               m->mm_fc, {cfg.hidden, c4}});
    }

    m->vlm_layers.resize(cfg.n_layers);
    for (int i=0; i<cfg.n_layers; ++i) {
        LayerW & w = m->vlm_layers[i];
        w.Wln_in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.hidden);
        w.Wq       = ggml_new_tensor_2d(ctx, wdt, cfg.hidden, cfg.q_full_dim);
        w.Wk       = ggml_new_tensor_2d(ctx, wdt, cfg.hidden, cfg.kv_full_dim);
        w.Wv       = ggml_new_tensor_2d(ctx, wdt, cfg.hidden, cfg.kv_full_dim);
        w.Wo       = ggml_new_tensor_2d(ctx, wdt, cfg.q_full_dim, cfg.hidden);
        w.Wln_post = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.hidden);
        w.Wgate    = ggml_new_tensor_2d(ctx, wdt, cfg.hidden,       cfg.intermediate);
        w.Wup      = ggml_new_tensor_2d(ctx, wdt, cfg.hidden,       cfg.intermediate);
        w.Wdown    = ggml_new_tensor_2d(ctx, wdt, cfg.intermediate, cfg.hidden);

        char p[256]; std::snprintf(p, sizeof(p),
                                   "model.vlm_with_expert.vlm.model.text_model.layers.%d.", i);
        const std::string pf = p;
        pending_f32.push_back({pf + "input_layernorm.weight",          w.Wln_in,   {cfg.hidden}});
        pending_f32.push_back({pf + "self_attn.q_proj.weight",         w.Wq,       {cfg.q_full_dim,  cfg.hidden}});
        pending_f32.push_back({pf + "self_attn.k_proj.weight",         w.Wk,       {cfg.kv_full_dim, cfg.hidden}});
        pending_f32.push_back({pf + "self_attn.v_proj.weight",         w.Wv,       {cfg.kv_full_dim, cfg.hidden}});
        pending_f32.push_back({pf + "self_attn.o_proj.weight",         w.Wo,       {cfg.hidden,      cfg.q_full_dim}});
        pending_f32.push_back({pf + "post_attention_layernorm.weight", w.Wln_post, {cfg.hidden}});
        pending_f32.push_back({pf + "mlp.gate_proj.weight",            w.Wgate,    {cfg.intermediate, cfg.hidden}});
        pending_f32.push_back({pf + "mlp.up_proj.weight",              w.Wup,      {cfg.intermediate, cfg.hidden}});
        pending_f32.push_back({pf + "mlp.down_proj.weight",            w.Wdown,    {cfg.hidden,       cfg.intermediate}});
    }

    m->expert_layers.resize(cfg.n_layers);
    for (int i=0; i<cfg.n_layers; ++i) {
        LayerW & w = m->expert_layers[i];
        const bool self_attn = expert_self_attn(cfg, i);
        w.Wln_in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
        w.Wq       = ggml_new_tensor_2d(ctx, wdt, cfg.expert_h, cfg.q_full_dim);
        if (self_attn) {
            w.Wk = ggml_new_tensor_2d(ctx, wdt, cfg.expert_h,    cfg.kv_full_dim);
            w.Wv = ggml_new_tensor_2d(ctx, wdt, cfg.expert_h,    cfg.kv_full_dim);
        } else {
            w.Wk = ggml_new_tensor_2d(ctx, wdt, cfg.kv_full_dim, cfg.kv_full_dim);
            w.Wv = ggml_new_tensor_2d(ctx, wdt, cfg.kv_full_dim, cfg.kv_full_dim);
        }
        w.Wo       = ggml_new_tensor_2d(ctx, wdt, cfg.q_full_dim, cfg.expert_h);
        w.Wln_post = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
        w.Wgate    = ggml_new_tensor_2d(ctx, wdt, cfg.expert_h,     cfg.expert_inter);
        w.Wup      = ggml_new_tensor_2d(ctx, wdt, cfg.expert_h,     cfg.expert_inter);
        w.Wdown    = ggml_new_tensor_2d(ctx, wdt, cfg.expert_inter, cfg.expert_h);

        char p[256]; std::snprintf(p, sizeof(p),
                                   "model.vlm_with_expert.lm_expert.layers.%d.", i);
        const std::string pf = p;
        pending_f32.push_back({pf + "input_layernorm.weight",  w.Wln_in, {cfg.expert_h}});
        pending_f32.push_back({pf + "self_attn.q_proj.weight", w.Wq,     {cfg.q_full_dim, cfg.expert_h}});
        if (self_attn) {
            pending_f32.push_back({pf + "self_attn.k_proj.weight", w.Wk, {cfg.kv_full_dim, cfg.expert_h}});
            pending_f32.push_back({pf + "self_attn.v_proj.weight", w.Wv, {cfg.kv_full_dim, cfg.expert_h}});
        } else {
            pending_f32.push_back({pf + "self_attn.k_proj.weight", w.Wk, {cfg.kv_full_dim, cfg.kv_full_dim}});
            pending_f32.push_back({pf + "self_attn.v_proj.weight", w.Wv, {cfg.kv_full_dim, cfg.kv_full_dim}});
        }
        pending_f32.push_back({pf + "self_attn.o_proj.weight",         w.Wo,       {cfg.expert_h, cfg.q_full_dim}});
        pending_f32.push_back({pf + "post_attention_layernorm.weight", w.Wln_post, {cfg.expert_h}});
        pending_f32.push_back({pf + "mlp.gate_proj.weight",            w.Wgate,    {cfg.expert_inter, cfg.expert_h}});
        pending_f32.push_back({pf + "mlp.up_proj.weight",              w.Wup,      {cfg.expert_inter, cfg.expert_h}});
        pending_f32.push_back({pf + "mlp.down_proj.weight",            w.Wdown,    {cfg.expert_h, cfg.expert_inter}});
    }
    m->Wnorm_expert = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
    pending_f32.push_back({"model.vlm_with_expert.lm_expert.norm.weight",
                           m->Wnorm_expert, {cfg.expert_h}});

    m->W_ain = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.max_action_dim, cfg.expert_h);
    m->b_ain = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
    m->W_at1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*cfg.expert_h, cfg.expert_h);
    m->b_at1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
    m->W_at2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.expert_h, cfg.expert_h);
    m->b_at2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
    pending_f32.push_back({"model.action_in_proj.weight",      m->W_ain, {cfg.expert_h, cfg.max_action_dim}});
    pending_f32.push_back({"model.action_in_proj.bias",        m->b_ain, {cfg.expert_h}});
    pending_f32.push_back({"model.action_time_mlp_in.weight",  m->W_at1, {cfg.expert_h, 2*cfg.expert_h}});
    pending_f32.push_back({"model.action_time_mlp_in.bias",    m->b_at1, {cfg.expert_h}});
    pending_f32.push_back({"model.action_time_mlp_out.weight", m->W_at2, {cfg.expert_h, cfg.expert_h}});
    pending_f32.push_back({"model.action_time_mlp_out.bias",   m->b_at2, {cfg.expert_h}});

    m->W_aout = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.expert_h, cfg.max_action_dim);
    m->b_aout = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.max_action_dim);
    pending_f32.push_back({"model.action_out_proj.weight", m->W_aout, {cfg.max_action_dim, cfg.expert_h}});
    pending_f32.push_back({"model.action_out_proj.bias",   m->b_aout, {cfg.max_action_dim}});

    m->time_bcasts.assign(cfg.num_steps, nullptr);
    for (int step=0; step<cfg.num_steps; ++step) {
        m->time_bcasts[step] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   cfg.expert_h, cfg.n_suffix);
    }

    // A GEMM weight the file stores packed (Q8_0, Q4_0, ...) stays packed and
    // ggml_mul_mat dequantizes at compute, as in the shared WeightLoader. Every
    // tensor above that uses wdt is a mul_mat operand, so those are the ones
    // eligible. Retyping is safe here: nothing is allocated yet.
    struct PendingPacked { std::string name; ggml_tensor * t; };
    std::vector<PendingPacked> pending_packed;
    if (use_gguf) {
        std::vector<PendingF32> keep;
        keep.reserve(pending_f32.size());
        for (auto & p : pending_f32) {
            const ggml_type ft = gst.file_type(hf_to_gguf(p.name));
            if (p.t->type == wdt && ft != GGML_TYPE_COUNT && ggml_is_quantized(ft) &&
                p.t->ne[0] % ggml_blck_size(ft) == 0) {
                p.t->type  = ft;
                p.t->nb[0] = ggml_type_size(ft);
                p.t->nb[1] = p.t->nb[0] * (p.t->ne[0] / ggml_blck_size(ft));
                for (int d = 2; d < GGML_MAX_DIMS; ++d)
                    p.t->nb[d] = p.t->nb[d-1] * p.t->ne[d-1];
                pending_packed.push_back({p.name, p.t});
            } else {
                keep.push_back(std::move(p));
            }
        }
        pending_f32.swap(keep);
        if (!pending_packed.empty())
            std::printf("vla: %zu GEMM weights kept packed as in the file (%s)\n",
                        pending_packed.size(), ggml_type_name(pending_packed[0].t->type));
    }

    m->weight_buf = alloc_weights(m->ctx_weights, m->backend);
    if (!m->weight_buf) {
        std::fprintf(stderr, "vla: alloc_weights failed\n");
        return nullptr;
    }
    std::printf("vla: [vram] weight_buf = %.1f MiB\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0));
    vram_probe(m->backend, "after weights alloc");

    auto stream_f32 = [&](const std::string & hf_name, ggml_tensor * t,
                          const std::vector<int64_t> & shape) -> bool {
        std::vector<float> hbuf(ggml_nelements(t));
        const bool ok = use_gguf
            ? gst.read_to_f32(hf_to_gguf(hf_name), hbuf.data(), shape)
            : st .read_to_f32(hf_name,             hbuf.data(), shape);
        if (!ok) {
            std::fprintf(stderr, "vla: read_to_f32 failed for %s\n", hf_name.c_str());
            return false;
        }
        backend_set_from_f32(t, hbuf.data(), ggml_nelements(t));
        return true;
    };

    for (auto & p : pending_f32) {
        if (!stream_f32(p.name, p.t, p.shape))
            return nullptr;
    }
    {
        const std::string emb = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight";
        std::vector<ggml_bf16_t> hbuf(ggml_nelements(m->E_lang));
        const bool ok = use_gguf
            ? gst.read_packed(hf_to_gguf(emb), hbuf.data(), GGML_TYPE_BF16, ggml_nbytes(m->E_lang))
            : st .read_raw(emb,                hbuf.data(), ggml_nbytes(m->E_lang), "BF16");
        if (!ok) {
            std::fprintf(stderr, "vla: read_raw failed for %s\n", emb.c_str());
            return nullptr;
        }
        ggml_backend_tensor_set(m->E_lang, hbuf.data(), 0, ggml_nbytes(m->E_lang));
    }
    for (auto & p : pending_packed) {
        std::vector<uint8_t> hbuf(ggml_nbytes(p.t));
        if (!gst.read_packed(hf_to_gguf(p.name), hbuf.data(), p.t->type, hbuf.size()))
            return nullptr;
        ggml_backend_tensor_set(p.t, hbuf.data(), 0, hbuf.size());
    }

    {
        const float dt = -1.f/static_cast<float>(cfg.num_steps);
        for (int step=0; step<cfg.num_steps; ++step) {
            const double time = 1.0+double(step)*double(dt);
            const auto te = sinusoidal_time_emb(time, cfg.expert_h,
                                                cfg.min_period, cfg.max_period);
            std::vector<float> tile(cfg.expert_h*cfg.n_suffix);
            for (int64_t t=0; t<cfg.n_suffix; ++t) {
                std::memcpy(tile.data()+t * cfg.expert_h,
                            te.data(), cfg.expert_h*sizeof(float));
            }
            ggml_backend_tensor_set(m->time_bcasts[step], tile.data(),
                                    0, tile.size()*sizeof(float));
        }
    }

    return m;
}
}

namespace {
void build_graph(SmolVLAModelArch * m, ggml_context * ctx, SmolVLAModelArch::MainIO & io,
                 int n_views, int64_t n_lang, bool kv_leaves) {
    Config cfg = m->cfg;
    cfg.n_img    = m->cfg.n_img*int64_t(n_views);
    cfg.n_lang   = n_lang;
    cfg.n_prefix = cfg.n_img+cfg.n_lang+cfg.n_state;
    cfg.n_full   = cfg.n_prefix+cfg.n_suffix;

    io.img_emb       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.hidden,         cfg.n_img);
    io.lang_ids      = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_lang);
    io.state         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.max_state_dim);
    io.x0            = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.max_action_dim, cfg.n_suffix);
    io.mask_prefill  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_prefix, cfg.n_prefix);
    io.pos_prefill   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_prefix);
    io.mask_full     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_full,   cfg.n_suffix);
    io.mask_pfx_only = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_prefix, cfg.n_suffix);
    io.pos_full      = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_suffix);
    io.pos_rebased   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_suffix);
    for (ggml_tensor * t : {io.img_emb, io.lang_ids, io.state, io.x0,
                            io.mask_prefill, io.pos_prefill, io.mask_full,
                            io.mask_pfx_only, io.pos_full, io.pos_rebased}) {
        ggml_set_input(t);
    }

    const float lang_scale = std::sqrt(static_cast<float>(cfg.hidden));
    ggml_tensor * img_emb_scaled  = ggml_scale(ctx, io.img_emb, lang_scale);
    ggml_tensor * lang_emb_scaled = ggml_scale(ctx, ggml_get_rows(ctx, m->E_lang, io.lang_ids), lang_scale);
    ggml_tensor * state_emb       = ggml_reshape_2d(ctx, linear(ctx, m->Wstate, m->bstate, io.state), cfg.hidden, 1);
    ggml_tensor * prefix_embs     = ggml_concat(ctx, ggml_concat(ctx, img_emb_scaled, lang_emb_scaled, 1), state_emb, 1);

    ggml_tensor * mask_prefill_f16 = ggml_cast(ctx, io.mask_prefill,  GGML_TYPE_F16);
    ggml_tensor * mask_full_f16    = ggml_cast(ctx, io.mask_full,     GGML_TYPE_F16);
    ggml_tensor * mask_pfx_only_f16= ggml_cast(ctx, io.mask_pfx_only, GGML_TYPE_F16);
    io.k_cache.resize(cfg.n_layers);
    io.v_cache.resize(cfg.n_layers);
    {
        ggml_tensor * h = prefix_embs;
        for (int i=0; i<cfg.n_layers; ++i) {
            h = build_vlm_layer(ctx, m->vlm_layers[i], h, mask_prefill_f16, io.pos_prefill,
                                cfg, &io.k_cache[i], &io.v_cache[i]);
        }
    }

    if (kv_leaves) {
        for (int i=0; i<cfg.n_layers; ++i) {
            io.k_leaf.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix));
            io.v_leaf.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix));
        }
    }
    const std::vector<ggml_tensor *> & K = kv_leaves ? io.k_leaf : io.k_cache;
    const std::vector<ggml_tensor *> & V = kv_leaves ? io.v_leaf : io.v_cache;

    // Reproject each cross-attn layer's prefix K/V once; reused every denoise step.
    std::vector<ggml_tensor *> xk_cache(cfg.n_layers, nullptr);
    std::vector<ggml_tensor *> xv_cache(cfg.n_layers, nullptr);
    for (int li=0; li<cfg.n_layers; ++li) {
        if (!expert_self_attn(cfg, li))
            expert_cross_kv(ctx, m->expert_layers[li], K[li], V[li], cfg, &xk_cache[li], &xv_cache[li]);
    }

    const float dt = -1.f/static_cast<float>(cfg.num_steps);
    ggml_tensor * x_t = io.x0;
    for (int step=0; step<cfg.num_steps; ++step) {
        ggml_tensor * action_emb     = linear(ctx, m->W_ain, m->b_ain, x_t);
        ggml_tensor * action_time_in = ggml_concat(ctx, action_emb, m->time_bcasts[step], 0);
        ggml_tensor * mlp1           = linear(ctx, m->W_at1, m->b_at1, action_time_in);
        ggml_tensor * h              = linear(ctx, m->W_at2, m->b_at2, ggml_silu(ctx, mlp1));
        for (int li=0; li<cfg.n_layers; ++li) {
            if (expert_self_attn(cfg, li)) {
                h = build_expert_self_attn_layer(ctx, m->expert_layers[li], h, K[li], V[li],
                                                 io.pos_full, mask_full_f16, cfg);
            } else {
                h = build_expert_cross_attn_layer(ctx, m->expert_layers[li], h,
                                                  xk_cache[li], xv_cache[li],
                                                  io.pos_rebased, mask_pfx_only_f16, cfg);
            }
        }
        ggml_tensor * v_t = linear(ctx, m->W_aout, m->b_aout, rms_norm(ctx, h, m->Wnorm_expert, cfg.rms_eps));
        x_t = ggml_add(ctx, x_t, ggml_scale(ctx, v_t, dt));
    }

    ggml_set_output(x_t);
    io.x_t = x_t;
}
}

SmolVLAModelArch::~SmolVLAModelArch() {
    if (weight_buf)
        ggml_backend_buffer_free(weight_buf);
    if (ctx_weights)
        ggml_free(ctx_weights);
    if (backend)
        ggml_backend_free(backend);
}

namespace {
std::vector<float> predict_impl(SmolVLAModelArch* m, const Inputs& in) {
    using clock = std::chrono::steady_clock;
    const auto t_total_begin = clock::now();

    m->stats = Stats{};

    const Config & cfg = m->cfg;

    if (in.n_lang < 1 || in.n_lang > int(cfg.n_lang)) {
        std::fprintf(stderr, "vla: lang_tokens length %d out of range [1, %lld]\n",
                     in.n_lang, (long long) cfg.n_lang);
        return {};
    }

    // The language tokens index E_lang via ggml_get_rows, which does not bound
    // its indices. Reject any token id outside the embedding table before the
    // gather so an out-of-range id cannot read past the weights.
    const int64_t vocab_rows = m->E_lang ? m->E_lang->ne[1] : 0;
    for (int i=0; i<in.n_lang; ++i) {
        if (in.lang_tokens[i] < 0 || in.lang_tokens[i] >= vocab_rows) {
            std::fprintf(stderr, "vla: lang_tokens[%d]=%d out of vocab range [0, %lld)\n",
                         i, in.lang_tokens[i], (long long) vocab_rows);
            return {};
        }
    }

    const size_t per_view_n = size_t(cfg.n_img*cfg.hidden);

    int    n_views   = 0;
    size_t img_emb_n = 0;
    std::vector<float> img_emb_pre;

    if (in.precomputed_img_emb != nullptr) {
        if (in.n_img_views < 1) {
            std::fprintf(stderr, "vla: precomputed_img_emb set but n_img_views=%d\n",
                         in.n_img_views);
            return {};
        }
        n_views   = in.n_img_views;
        img_emb_n = per_view_n * size_t(n_views);
        img_emb_pre.assign(in.precomputed_img_emb, in.precomputed_img_emb+img_emb_n);

    } else {
        if (in.n_images < 1 || in.images == nullptr) {
            std::fprintf(stderr, "vla: at least one image is required\n");
            return {};
        }
        n_views   = in.n_images;
        img_emb_n = per_view_n * size_t(n_views);
        img_emb_pre.resize(img_emb_n);

        const int64_t H = m->vit_hidden, grid = m->vit_image/m->vit_patch, n_patches = grid * grid;
        const int64_t s = m->vit_scale, c4 = H * s * s, K = m->vit_n_tokens;
        const auto t_vision_begin = clock::now();

        // Graph A: SigLIP ViT (conv patch-embed -> +pos -> layers -> post_ln), plain sequential positions.
        ggml_context * VC = m->vision_scratch.reset(size_t(256)*1024*1024);
        if (!VC) { std::fprintf(stderr, "vla(smolvla): ggml_init(vision ctx) failed\n"); return {}; }
        ggml_tensor * t_px = ggml_new_tensor_3d(VC, GGML_TYPE_F32, m->vit_image, m->vit_image, 3); ggml_set_input(t_px);
        ggml_tensor * hv = m->vit.embed_conv(VC, t_px, m->vit_patch, grid);
        for (const EncBlockW & w : m->vit.enc.blk)
            hv = build_siglip_layer(VC, m->vit.enc.cfg, w, hv, n_patches);
        ggml_tensor * post_ln = layer_norm(VC, hv, m->vit.post_ln_w, m->vit.post_ln_b, m->vit.enc.cfg.ln_eps);
        ggml_set_output(post_ln);
        ggml_cgraph * gA = ggml_new_graph_custom(VC, 8192, false);
        ggml_build_forward_expand(gA, post_ln);
        if (!m->vision_scratch.alloc(m->backend, gA)) {
            std::fprintf(stderr, "vla(smolvla): vision gallocr A alloc failed\n");
            return {};
        }

        // Graph B: pixel-shuffle connector, a single bias-free matmul (c4 -> hidden).
        ggml_context * MC = m->connector_scratch.reset(size_t(64)*1024*1024);
        if (!MC) { std::fprintf(stderr, "vla(smolvla): ggml_init(connector ctx) failed\n"); return {}; }
        ggml_tensor * t_shuf = ggml_new_tensor_2d(MC, GGML_TYPE_F32, c4, K); ggml_set_input(t_shuf);
        ggml_tensor * img_embeds = ggml_mul_mat(MC, m->mm_fc, t_shuf);
        ggml_set_output(img_embeds);
        ggml_cgraph * gB = ggml_new_graph(MC);
        ggml_build_forward_expand(gB, img_embeds);
        if (!m->connector_scratch.alloc(m->backend, gB)) {
            std::fprintf(stderr, "vla(smolvla): vision gallocr B alloc failed\n");
            return {};
        }

        std::vector<float> chw, post_host((size_t) H * n_patches), shuf_host((size_t) c4*K);
        bool vok = true;
        for (int v=0; v<n_views && vok; ++v) {
            if (!preprocess_image_chw("smolvla", in.images[v], m->vit_image, chw)) {
                vok = false;
                break;
            }
            ggml_backend_tensor_set(t_px, chw.data(), 0, ggml_nbytes(t_px));
            graph_unique_names(gA);
            if (ggml_backend_graph_compute(m->backend, gA) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(smolvla): vision compute A failed (view %d)\n", v); vok = false; break;
            }
            ggml_backend_tensor_get(post_ln, post_host.data(), 0, ggml_nbytes(post_ln));
            pixel_shuffle_hf(post_host.data(), shuf_host.data(), H, grid, s);
            ggml_backend_tensor_set(t_shuf, shuf_host.data(), 0, ggml_nbytes(t_shuf));
            graph_unique_names(gB);
            if (ggml_backend_graph_compute(m->backend, gB) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(smolvla): connector compute failed (view %d)\n", v); vok = false; break;
            }
            ggml_backend_tensor_get(img_embeds, img_emb_pre.data()+size_t(v)*per_view_n, 0, ggml_nbytes(img_embeds));
        }
        if (!vok) return {};
        m->stats.ms_vision = std::chrono::duration<float, std::milli>(clock::now()-t_vision_begin).count();
    }

    const bool    phase     = in.timing_detail == TimingDetail::PHASE;
    const int64_t n_img     = cfg.n_img*int64_t(n_views);
    const int64_t n_lang    = phase ? in.n_lang : cfg.n_lang;
    const int64_t n_prefix  = n_img+n_lang+cfg.n_state;
    const int64_t n_full    = n_prefix+cfg.n_suffix;
    const int64_t pad_start = n_img+in.n_lang;
    const int64_t pad_end   = n_img+n_lang;

    const size_t max_nodes = size_t(64)*cfg.n_layers*(cfg.num_steps+1) + 1024;
    const size_t arena     = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false);

    SmolVLAModelArch::MainIO * io = nullptr;
    ggml_cgraph * gf = nullptr;
    SmolVLAModelArch::MainIO phase_io;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> phase_ctx(nullptr, ggml_free);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> phase_buf(nullptr, ggml_backend_buffer_free);
    if (!phase) {
        const bool built = m->main_graph.ensure(m->backend, n_views, arena,
            [&](ggml_context * C, SmolVLAModelArch::MainIO & gio) -> ggml_cgraph * {
                build_graph(m, C, gio, n_views, n_lang, false);
                ggml_cgraph * g = ggml_new_graph_custom(C, max_nodes, false);
                ggml_build_forward_expand(g, gio.x_t);
                return g;
            });
        if (!built) {
            std::fprintf(stderr, "vla: cached graph build failed\n");
            return {};
        }
        io = &m->main_graph.io();
        gf = m->main_graph.graph();
    } else {
        ggml_init_params gparams = { arena + ggml_graph_overhead_custom(4096, false), nullptr, true };
        phase_ctx.reset(ggml_init(gparams));
        if (!phase_ctx) {
            std::fprintf(stderr, "vla: ggml_init (compute) failed\n");
            return {};
        }
        build_graph(m, phase_ctx.get(), phase_io, n_views, n_lang, true);
        // Graph inputs, not weights: tag them so a backend that reads buffer usage
        // (ggml-openvino) does not mistake the default ANY for a KV cache. gallocr
        // tags its own arena the same way.
        phase_buf.reset(ggml_backend_alloc_ctx_tensors(phase_ctx.get(), m->backend));
        if (!phase_buf) {
            std::fprintf(stderr, "vla: ggml_backend_alloc_ctx_tensors (compute) failed\n");
            return {};
        }
        ggml_backend_buffer_set_usage(phase_buf.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        io = &phase_io;
    }

    std::vector<float> state_host(cfg.max_state_dim, 0.0f);
    if (in.state)
        std::memcpy(state_host.data(), in.state, cfg.max_state_dim*sizeof(float));
    for (int64_t i=0; i<cfg.real_state_dim && i<cfg.max_state_dim; ++i) {
        state_host[i] = (state_host[i]-m->state_mean[i])/(m->state_std[i]+cfg.norm_eps);
    }

    std::vector<float> noise_host(cfg.n_suffix*cfg.max_action_dim);
    if (in.noise) {
        std::memcpy(noise_host.data(), in.noise, noise_host.size()*sizeof(float));
    } else {
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto & v : noise_host)
            v = dist(m->rng);
    }

    std::vector<int32_t> lang_host(n_lang, 0);
    std::memcpy(lang_host.data(), in.lang_tokens, in.n_lang*sizeof(int32_t));

    const int64_t state_pos       = n_img+in.n_lang;
    const int64_t suffix_pos_base = state_pos+1;
    std::vector<float>   mask_prefill_host(n_prefix * n_prefix);
    std::vector<int32_t> pos_prefill_host (n_prefix);
    for (int64_t i=0; i<n_prefix; ++i) {
        for (int64_t j=0; j<n_prefix; ++j) {
            bool blocked = false;
            if ((i < n_prefix-1) && (j == n_prefix-1))
                blocked = true;
            if (j >= pad_start && j < pad_end)
                blocked = true;
            mask_prefill_host[i * n_prefix+j] = blocked ? -INFINITY : 0.f;
        }
        pos_prefill_host[i] = (i == n_prefix-1)
            ? static_cast<int32_t>(state_pos)
            : static_cast<int32_t>(i);
    }

    std::vector<float>   mask_full_host       (n_full   * cfg.n_suffix);
    std::vector<float>   mask_prefix_only_host(n_prefix * cfg.n_suffix, 0.f);
    std::vector<int32_t> pos_full_host        (cfg.n_suffix);
    std::vector<int32_t> pos_rebased_host     (cfg.n_suffix);
    for (int64_t i=0; i<cfg.n_suffix; ++i) {
        for (int64_t j=0; j<n_full; ++j) {
            bool blocked;
            if (j < n_prefix) {
                blocked = (j >= pad_start && j < pad_end);
            } else {
                blocked = ((j-n_prefix) > i);
            }
            mask_full_host[i * n_full+j] = blocked ? -INFINITY : 0.f;
        }
        for (int64_t j=0; j<n_prefix; ++j) {
            if (j >= pad_start && j < pad_end) {
                mask_prefix_only_host[i * n_prefix+j] = -INFINITY;
            }
        }
        pos_full_host   [i] = static_cast<int32_t>(suffix_pos_base+i);
        pos_rebased_host[i] = static_cast<int32_t>(i);
    }

    ggml_backend_tensor_set(io->img_emb,       img_emb_pre.data(),    0, img_emb_n * sizeof(float));
    ggml_backend_tensor_set(io->lang_ids,      lang_host.data(),      0, lang_host.size()*sizeof(int32_t));
    ggml_backend_tensor_set(io->state,         state_host.data(),     0, cfg.max_state_dim*sizeof(float));
    ggml_backend_tensor_set(io->x0,            noise_host.data(),     0, noise_host.size()*sizeof(float));
    ggml_backend_tensor_set(io->mask_prefill,  mask_prefill_host.data(),     0, mask_prefill_host.size()     * sizeof(float));
    ggml_backend_tensor_set(io->pos_prefill,   pos_prefill_host.data(),      0, pos_prefill_host.size()      * sizeof(int32_t));
    ggml_backend_tensor_set(io->mask_full,     mask_full_host.data(),        0, mask_full_host.size()        * sizeof(float));
    ggml_backend_tensor_set(io->mask_pfx_only, mask_prefix_only_host.data(), 0, mask_prefix_only_host.size()*sizeof(float));
    ggml_backend_tensor_set(io->pos_full,      pos_full_host.data(),         0, pos_full_host.size()         * sizeof(int32_t));
    ggml_backend_tensor_set(io->pos_rebased,   pos_rebased_host.data(),      0, pos_rebased_host.size()      * sizeof(int32_t));

    if (phase) {
        ggml_cgraph * gf_pre = ggml_new_graph_custom(phase_ctx.get(), 4096, false);
        for (int i=0; i<cfg.n_layers; ++i) {
            ggml_build_forward_expand(gf_pre, io->k_cache[i]);
            ggml_build_forward_expand(gf_pre, io->v_cache[i]);
        }
        graph_unique_names(gf_pre);
        const auto t0 = clock::now();
        if (ggml_backend_graph_compute(m->backend, gf_pre) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla: ggml prefill compute failed\n");
            return {};
        }
        m->stats.ms_prefill = std::chrono::duration<float, std::milli>(clock::now()-t0).count();

        for (int i=0; i<cfg.n_layers; ++i) {
            ggml_backend_tensor_copy(io->k_cache[i], io->k_leaf[i]);
            ggml_backend_tensor_copy(io->v_cache[i], io->v_leaf[i]);
        }
        gf = ggml_new_graph_custom(phase_ctx.get(), max_nodes, false);
        ggml_build_forward_expand(gf, io->x_t);
    }

    graph_unique_names(gf);
    const auto t0 = clock::now();
    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla: ggml compute failed\n");
        return {};
    }
    const float ms = std::chrono::duration<float, std::milli>(clock::now()-t0).count();
    if (phase)
        m->stats.ms_denoise = ms;
    m->stats.ms_inference = m->stats.ms_prefill+ms;

    std::vector<float> out(cfg.n_suffix*cfg.max_action_dim);
    ggml_backend_tensor_get(io->x_t, out.data(), 0, out.size()*sizeof(float));
    for (int64_t r=0; r<cfg.n_suffix; ++r) {
        float * row = out.data()+r * cfg.max_action_dim;
        for (int64_t j=0; j<cfg.max_action_dim; ++j) {
            row[j] = j < cfg.real_action_dim ? row[j]*(m->action_std[j]+cfg.norm_eps)+m->action_mean[j] : 0.0f;
        }
    }

    m->stats.ms_total = std::chrono::duration<float, std::milli>(
        clock::now()-t_total_begin).count();
    return out;
}
}

std::vector<float> SmolVLAModelArch::predict(const Inputs& in) {
    return predict_impl(this, in);
}

std::unique_ptr<ModelArchBase> smolvla_create(const std::string&,
                                              const std::string& ckpt_path,
                                              const std::string& config_path,
                                              const Options& opts) {
    return smolvla_load_impl(opts.weight_dtype.value_or(vla::default_weight_dtype(GGML_TYPE_BF16)),
                             ckpt_path, config_path);
}

}
