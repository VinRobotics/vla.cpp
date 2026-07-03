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
#include "model.h"
#include "vision_common.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
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
        if (!file) return false;
        uint64_t header_size = 0;
        file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
        std::string header_str(header_size, '\0');
        file.read(header_str.data(), header_size);
        data_blob_start = sizeof(uint64_t) + header_size;
        json j = json::parse(header_str);
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() == "__metadata__") continue;
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
        const size_t bytes = info.off_end - info.off_begin;
        file.seekg(data_blob_start + info.off_begin, std::ios::beg);
        if (info.dtype == "BF16") {
            std::vector<ggml_bf16_t> tmp(bytes / sizeof(ggml_bf16_t));
            file.read(reinterpret_cast<char *>(tmp.data()), bytes);
            ggml_bf16_to_fp32_row(tmp.data(), dst, tmp.size());
        } else if (info.dtype == "F32") {
            file.read(reinterpret_cast<char *>(dst), bytes);
        } else {
            std::fprintf(stderr, "vla: unsupported dtype for %s: %s\n",
                         name.c_str(), info.dtype.c_str());
            return false;
        }
        return true;
    }

    bool read_raw(const std::string & name, void * dst, size_t expected_bytes,
                  const char * expected_dtype) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            std::fprintf(stderr, "vla: tensor not found: %s\n", name.c_str());
            return false;
        }
        const auto & info = it->second;
        if ((info.off_end - info.off_begin) != expected_bytes ||
            info.dtype != expected_dtype) {
            std::fprintf(stderr, "vla: bad raw read for %s\n", name.c_str());
            return false;
        }
        file.seekg(data_blob_start + info.off_begin, std::ios::beg);
        file.read(static_cast<char *>(dst), expected_bytes);
        return true;
    }
};

struct gguf_source {
    struct gguf_context * gctx     = nullptr;
    struct ggml_context * meta_ctx = nullptr;
    FILE *                fp       = nullptr;
    size_t                data_off = 0;

    bool open(const std::string & path) {
        gguf_init_params p{};
        p.no_alloc = true;
        p.ctx      = &meta_ctx;
        gctx = gguf_init_from_file(path.c_str(), p);
        if (!gctx) {
            std::fprintf(stderr, "vla: gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "vla: fopen failed for %s\n", path.c_str());
            gguf_free(gctx); gctx = nullptr;
            ggml_free(meta_ctx); meta_ctx = nullptr;
            return false;
        }
        data_off = gguf_get_data_offset(gctx);
        return true;
    }

    ~gguf_source() {
        if (fp)       std::fclose(fp);
        if (gctx)     gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
    }

    static bool shape_matches(const ggml_tensor * t, const std::vector<int64_t> & pt_shape) {
        const int nd_used = std::max(1, (int) pt_shape.size());
        if (nd_used > GGML_MAX_DIMS) return false;
        for (int d = 0; d < (int) pt_shape.size(); ++d) {
            const int64_t expected = pt_shape[pt_shape.size() - 1 - d];
            if (t->ne[d] != expected) return false;
        }
        for (int d = (int) pt_shape.size(); d < GGML_MAX_DIMS; ++d) {
            if (t->ne[d] != 1) return false;
        }
        return true;
    }

    bool read_to_f32(const std::string & name, float * dst,
                     const std::vector<int64_t> & expected_shape) {
        ggml_tensor * t = ggml_get_tensor(meta_ctx, name.c_str());
        if (!t) {
            std::fprintf(stderr, "vla: gguf tensor not found: %s\n", name.c_str());
            return false;
        }
        if (!shape_matches(t, expected_shape)) {
            std::fprintf(stderr, "vla: gguf shape mismatch for %s\n", name.c_str());
            return false;
        }
        const int64_t id     = gguf_find_tensor(gctx, name.c_str());
        const size_t  offset = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t  bytes  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) offset, SEEK_SET) != 0) {
            std::fprintf(stderr, "vla: fseek failed for %s\n", name.c_str());
            return false;
        }
        if (t->type == GGML_TYPE_F32) {
            if (std::fread(dst, 1, bytes, fp) != bytes) return false;
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp(bytes / sizeof(ggml_bf16_t));
            if (std::fread(tmp.data(), 1, bytes, fp) != bytes) return false;
            ggml_bf16_to_fp32_row(tmp.data(), dst, tmp.size());
        } else {
            std::fprintf(stderr, "vla: gguf unsupported dtype %d for %s\n",
                         (int) t->type, name.c_str());
            return false;
        }
        return true;
    }

    bool read_raw(const std::string & name, void * dst, size_t expected_bytes,
                  const char * expected_dtype) {
        ggml_tensor * t = ggml_get_tensor(meta_ctx, name.c_str());
        if (!t) {
            std::fprintf(stderr, "vla: gguf tensor not found: %s\n", name.c_str());
            return false;
        }
        const int64_t id    = gguf_find_tensor(gctx, name.c_str());
        const size_t  bytes = gguf_get_tensor_size(gctx, id);
        const bool ok_dtype = (std::strcmp(expected_dtype, "BF16") == 0 && t->type == GGML_TYPE_BF16)
                            || (std::strcmp(expected_dtype, "F32") == 0 && t->type == GGML_TYPE_F32);
        if (bytes != expected_bytes || !ok_dtype) {
            std::fprintf(stderr, "vla: gguf bad raw read for %s\n", name.c_str());
            return false;
        }
        const size_t offset = data_off + gguf_get_tensor_offset(gctx, id);
        if (std::fseek(fp, (long) offset, SEEK_SET) != 0) return false;
        return std::fread(dst, 1, bytes, fp) == bytes;
    }

    int64_t find_key(const char * key) const {
        return gguf_find_key(gctx, key);
    }
    bool has_key(const char * key) const {
        return find_key(key) >= 0;
    }
    uint32_t get_u32(const char * key) const { return gguf_get_val_u32(gctx, find_key(key)); }
    int32_t  get_i32(const char * key) const { return gguf_get_val_i32(gctx, find_key(key)); }
    float    get_f32(const char * key) const { return gguf_get_val_f32(gctx, find_key(key)); }
    double   get_f64(const char * key) const { return gguf_get_val_f64(gctx, find_key(key)); }
    std::string get_str(const char * key) const { return gguf_get_val_str(gctx, find_key(key)); }

    bool has_tensor(const char * name) const {
        return ggml_get_tensor(meta_ctx, name) != nullptr;
    }
};

struct VlmLayerW {
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

struct ExpertLayerW {
    bool is_self_attn;
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

// SigLIP-B/16 vision block weights (SmolVLM2 tower, built in-tree).
struct SigLipLayerW { ggml_tensor *ln1w,*ln1b,*ln2w,*ln2b,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*Wfc1,*bfc1,*Wfc2,*bfc2; };

}

struct SmolVLAModelArch : public ModelArchBase {
    SmolVLAModelArch() : ModelArchBase(Arch::SMOLVLA) {}
    ~SmolVLAModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    // In-tree SigLIP-B/16 vision tower (was llama.cpp clip.cpp mmproj).
    int64_t vit_hidden = 768, vit_layers = 12, vit_heads = 12, vit_inter = 3072;
    int64_t vit_patch = 16, vit_image = 512, vit_scale = 4, vit_n_tokens = 64;
    float   vit_ln_eps = 1e-6f;
    ggml_tensor * vit_patch_w = nullptr, * vit_patch_b = nullptr, * vit_pos = nullptr;
    ggml_tensor * vit_post_ln_w = nullptr, * vit_post_ln_b = nullptr, * mm_fc = nullptr;
    std::vector<SigLipLayerW> vit;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    bool                  is_cuda     = false;
    bool                  is_gpu      = false;

    ggml_type             weight_dtype = GGML_TYPE_BF16;

    ggml_context * ctx_weights = nullptr;

    ggml_tensor *  E_lang   = nullptr;
    ggml_tensor *  Wstate   = nullptr;
    ggml_tensor *  bstate   = nullptr;

    std::vector<VlmLayerW> vlm_layers;
    ggml_tensor *  Wnorm_vlm = nullptr;

    std::vector<ExpertLayerW> expert_layers;
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

    ggml_context *        ctx_compute     = nullptr;
    ggml_gallocr_t        galloc          = nullptr;
    ggml_cgraph *         gf_cached       = nullptr;
    int                   cached_n_views  = 0;

    ggml_tensor * in_img_emb       = nullptr;
    ggml_tensor * in_lang_ids      = nullptr;
    ggml_tensor * in_state         = nullptr;
    ggml_tensor * in_x0            = nullptr;
    ggml_tensor * in_mask_prefill  = nullptr;
    ggml_tensor * in_pos_prefill   = nullptr;
    ggml_tensor * in_mask_full     = nullptr;
    ggml_tensor * in_mask_pfx_only = nullptr;
    ggml_tensor * in_pos_full      = nullptr;
    ggml_tensor * in_pos_rebased   = nullptr;

    ggml_tensor * out_x_t          = nullptr;

    std::vector<ggml_tensor *> time_bcasts;
};

namespace {

// One pre-norm SigLIP encoder block (SmolVLM2 tower), same graph as the other
// in-tree models. Bidirectional attention, F32 score accumulation, tanh GELU.
ggml_tensor * build_siglip_layer(ggml_context * C, const SigLipLayerW & w, ggml_tensor * x,
                                 int64_t seq, int64_t heads, int64_t head_dim, int64_t hidden, float ln_eps) {
    const float scale = 1.0f / std::sqrt((float) head_dim);
    ggml_tensor * n1 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, ln_eps), w.ln1w), w.ln1b);
    ggml_tensor * q = ggml_add(C, ggml_mul_mat(C, w.Wq, n1), w.bq);
    ggml_tensor * k = ggml_add(C, ggml_mul_mat(C, w.Wk, n1), w.bk);
    ggml_tensor * v = ggml_add(C, ggml_mul_mat(C, w.Wv, n1), w.bv);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, head_dim, heads, seq), 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, head_dim, heads, seq), 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, head_dim, heads, seq), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(C, K, Q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), hidden, seq);
    ggml_tensor * h1 = ggml_add(C, x, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, h1, ln_eps), w.ln2w), w.ln2b);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wfc2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wfc1, n2), w.bfc1))), w.bfc2);
    return ggml_add(C, h1, ff);
}

// CHW-planar float image in [-1,1] for ggml_conv_2d (SigLIP mean/std 0.5).
bool preprocess_image_chw(const ImageView & v, int64_t side, std::vector<float> & out) {
    if (v.w != (int) side || v.h != (int) side || !v.data) {
        std::fprintf(stderr, "vla(smolvla): image view is %dx%d, expected %lldx%lld\n",
                     v.w, v.h, (long long) side, (long long) side);
        return false;
    }
    out.assign((size_t) 3 * side * side, 0.0f);
    for (int64_t h = 0; h < side; ++h)
        for (int64_t w = 0; w < side; ++w)
            for (int64_t c = 0; c < 3; ++c) {
                float px;
                if (v.format == PixelFormat::U8) px = ((const uint8_t *) v.data)[(h * side + w) * 3 + c] / 255.0f;
                else                              px = ((const float  *) v.data)[(h * side + w) * 3 + c];
                out[c * side * side + h * side + w] = px * 2.0f - 1.0f;
            }
    return true;
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

    cfg.hidden        = 960;
    cfg.n_q_heads     = 15;
    cfg.n_kv_heads    = 5;
    cfg.head_dim      = 64;
    cfg.intermediate  = 2560;

    cfg.rms_eps        = 1e-5f;
    cfg.rope_mode      = GGML_ROPE_TYPE_NEOX;
    cfg.rope_freq_base = 10000.f;

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
        cfg.expert_h          = static_cast<int64_t>(std::round(double(cfg.hidden) * mul));

        cfg.real_state_dim  = j.at("input_features").at("observation.state").at("shape").at(0).get<int64_t>();
        cfg.real_action_dim = j.at("output_features").at("action").at("shape").at(0).get<int64_t>();
    } catch (const json::exception & e) {
        std::fprintf(stderr, "vla: missing/bad field in %s: %s\n", path.c_str(), e.what());
        return false;
    }

    cfg.n_state     = 1;
    cfg.q_full_dim  = cfg.n_q_heads  * cfg.head_dim;
    cfg.kv_full_dim = cfg.n_kv_heads * cfg.head_dim;
    cfg.rope_n_dims = static_cast<int>(cfg.head_dim);

    cfg.norm_eps    = 1e-8f;
    return true;
}

bool load_normalizer_stats(const std::string & model_dir, SmolVLAModelArch & m) {
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
    return true;
}

std::string default_config_path(const std::string & ckpt_path) {
    const auto pos = ckpt_path.find_last_of("/\\");
    const std::string dir = (pos == std::string::npos) ? std::string(".")
                                                       : ckpt_path.substr(0, pos);
    return dir + "/config.json";
}

bool ends_with_gguf(const std::string & path) {
    static const std::string sfx = ".gguf";
    return path.size() >= sfx.size()
        && path.compare(path.size() - sfx.size(), sfx.size(), sfx) == 0;
}

bool load_config_from_gguf(const gguf_source & st, Config & cfg) {
    if (!st.has_key("smolvla.architecture")) {
        std::fprintf(stderr, "vla: gguf missing key 'smolvla.architecture'\n");
        return false;
    }
    const std::string arch = st.get_str("smolvla.architecture");
    if (arch != "smolvla") {
        std::fprintf(stderr, "vla: gguf architecture = '%s' (expected 'smolvla')\n",
                     arch.c_str());
        return false;
    }

    auto need = [&](const char * key) -> bool {
        if (!st.has_key(key)) {
            std::fprintf(stderr, "vla: gguf missing key '%s'\n", key);
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
            "smolvla.self_attn_every_n_layers", "smolvla.tokenizer_max_length",
            "smolvla.min_period", "smolvla.max_period"}) {
        if (!need(k)) return false;
    }

    cfg.hidden        = st.get_u32("smolvla.hidden");
    cfg.intermediate  = st.get_u32("smolvla.intermediate");
    cfg.n_q_heads     = st.get_u32("smolvla.n_q_heads");
    cfg.n_kv_heads    = st.get_u32("smolvla.n_kv_heads");
    cfg.head_dim      = st.get_u32("smolvla.head_dim");
    cfg.n_layers      = st.get_u32("smolvla.n_layers");
    cfg.expert_h      = st.get_u32("smolvla.expert_h");
    cfg.expert_inter  = st.get_u32("smolvla.expert_inter");
    cfg.n_suffix      = st.get_u32("smolvla.chunk_size");
    cfg.num_steps     = st.get_u32("smolvla.num_steps");
    cfg.max_state_dim = st.get_u32("smolvla.max_state_dim");
    cfg.max_action_dim= st.get_u32("smolvla.max_action_dim");
    cfg.real_state_dim  = st.get_u32("smolvla.real_state_dim");
    cfg.real_action_dim = st.get_u32("smolvla.real_action_dim");
    cfg.self_attn_every_n = st.get_u32("smolvla.self_attn_every_n_layers");
    cfg.n_lang        = st.get_u32("smolvla.tokenizer_max_length");
    cfg.min_period    = st.get_f64("smolvla.min_period");
    cfg.max_period    = st.get_f64("smolvla.max_period");

    cfg.rms_eps        = 1e-5f;
    cfg.rope_mode      = GGML_ROPE_TYPE_NEOX;
    cfg.rope_freq_base = 10000.f;

    cfg.n_state     = 1;
    cfg.q_full_dim  = cfg.n_q_heads  * cfg.head_dim;
    cfg.kv_full_dim = cfg.n_kv_heads * cfg.head_dim;
    cfg.rope_n_dims = static_cast<int>(cfg.head_dim);

    cfg.norm_eps = st.has_key("smolvla.norm_eps") ? st.get_f32("smolvla.norm_eps") : 1e-8f;
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
        if (s == "input_layernorm.weight")          return "attn_norm.weight";
        if (s == "self_attn.q_proj.weight")         return "attn_q.weight";
        if (s == "self_attn.k_proj.weight")         return "attn_k.weight";
        if (s == "self_attn.v_proj.weight")         return "attn_v.weight";
        if (s == "self_attn.o_proj.weight")         return "attn_o.weight";
        if (s == "post_attention_layernorm.weight") return "ffn_norm.weight";
        if (s == "mlp.gate_proj.weight")            return "ffn_gate.weight";
        if (s == "mlp.up_proj.weight")              return "ffn_up.weight";
        if (s == "mlp.down_proj.weight")            return "ffn_down.weight";
        return s;
    };
    auto starts_with = [](const std::string & s, const char * pfx) -> bool {
        const size_t plen = std::strlen(pfx);
        return s.size() >= plen && s.compare(0, plen, pfx) == 0;
    };

    if (n == "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight")
        return "token_embd.weight";
    if (n == "model.vlm_with_expert.vlm.model.text_model.norm.weight")
        return "vlm.output_norm.weight";
    if (n == "model.vlm_with_expert.lm_expert.norm.weight")
        return "aex.output_norm.weight";

    auto layer_translate = [&](const std::string & rest, const char * dst_blk) -> std::string {

        if (!starts_with(rest, "layers.")) return n;
        const size_t end_i = rest.find('.', 7);
        if (end_i == std::string::npos) return n;
        const std::string idx = rest.substr(7, end_i - 7);
        const std::string suf = rest.substr(end_i + 1);
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
        if (rest == "embeddings.patch_embedding.weight")   return "vit.patch_embd.weight";
        if (rest == "embeddings.patch_embedding.bias")     return "vit.patch_embd.bias";
        if (rest == "embeddings.position_embedding.weight") return "vit.pos_embd";
        if (rest == "post_layernorm.weight")               return "vit.post_ln.weight";
        if (rest == "post_layernorm.bias")                 return "vit.post_ln.bias";
        if (starts_with(rest, "encoder.layers.")) {
            const size_t e = rest.find('.', 15);
            if (e == std::string::npos) return n;
            const std::string idx = rest.substr(15, e - 15);
            const std::string suf = rest.substr(e + 1);
            std::string ds;
            if      (suf == "layer_norm1.weight")     ds = "ln1.weight";
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
            else return n;
            return "vit.blk." + idx + "." + ds;
        }
        return n;
    }

    if (starts_with(n, MODEL_PFX)) {

        return n.substr(std::strlen(MODEL_PFX));
    }
    return n;
}

std::vector<float> sinusoidal_time_emb(double timestep, int64_t dim,
                                       double min_period, double max_period) {
    const int64_t half = dim / 2;
    std::vector<float> out(dim);
    for (int64_t i = 0; i < half; ++i) {
        const double frac   = (half == 1) ? 0.0 : double(i) / double(half - 1);
        const double period = min_period * std::pow(max_period / min_period, frac);
        const double scale  = 2.0 * M_PI / period;
        const double s      = scale * timestep;
        out[i]        = static_cast<float>(std::sin(s));
        out[half + i] = static_cast<float>(std::cos(s));
    }
    return out;
}

ggml_tensor * rope_q_or_k(ggml_context * ctx, ggml_tensor * x,
                          ggml_tensor * positions, const Config & cfg) {
    return ggml_rope_ext(ctx, x, positions,  nullptr,
                         cfg.rope_n_dims, cfg.rope_mode,  0,
                         cfg.rope_freq_base,  1.f,
                          0.f,  1.f,
                          32.f,  1.f);
}

static inline bool tower_mm_f32_prec() {
    const char * e = std::getenv("VLA_MM_PREC");
    return !(e && std::strcmp(e, "default") == 0);
}
static inline ggml_tensor * mm_w(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * r = ggml_mul_mat(ctx, w, x);
    if (tower_mm_f32_prec()) ggml_mul_mat_set_prec(r, GGML_PREC_F32);
    return r;
}

ggml_tensor * build_vlm_layer(ggml_context * ctx, const VlmLayerW & w,
                              ggml_tensor * x_in, ggml_tensor * mask,
                              ggml_tensor * positions, const Config & cfg,
                              ggml_tensor ** k_out, ggml_tensor ** v_out) {
    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.Wln_in);
    ggml_tensor * q_proj = mm_w(ctx, w.Wq, x_norm);
    ggml_tensor * k_proj = mm_w(ctx, w.Wk, x_norm);
    ggml_tensor * v_proj = mm_w(ctx, w.Wv, x_norm);

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q_proj, cfg.head_dim, cfg.n_q_heads,  cfg.n_prefix);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k_proj, cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v_proj, cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);

    ggml_tensor * q_rope = rope_q_or_k(ctx, q_h, positions, cfg);
    ggml_tensor * k_rope = rope_q_or_k(ctx, k_h, positions, cfg);
    *k_out = k_rope;
    *v_out = v_h;

    ggml_tensor * Q = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(ctx, v_h,    0, 2, 1, 3);
    const float scale = 1.f / std::sqrt(static_cast<float>(cfg.head_dim));
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale,
                                            0.f,  0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att_pre_o = ggml_reshape_2d(ctx, fa, cfg.q_full_dim, cfg.n_prefix);
    ggml_tensor * o_out = mm_w(ctx, w.Wo, att_pre_o);
    ggml_tensor * h1    = ggml_add(ctx, x_in, o_out);

    ggml_tensor * x_norm_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.Wln_post);
    ggml_tensor * gate    = mm_w(ctx, w.Wgate, x_norm_mlp);
    ggml_tensor * up      = mm_w(ctx, w.Wup,   x_norm_mlp);
    ggml_tensor * inter   = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_tensor * mlp_out = mm_w(ctx, w.Wdown, inter);
    return ggml_add(ctx, h1, mlp_out);
}

ggml_tensor * build_expert_self_attn_layer(
    ggml_context * ctx, const ExpertLayerW & w,
    ggml_tensor * x_in, ggml_tensor * cached_K, ggml_tensor * cached_V,
    ggml_tensor * positions_full, ggml_tensor * mask_full, const Config & cfg)
{
    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.Wln_in);
    ggml_tensor * q_proj = mm_w(ctx, w.Wq, x_norm);
    ggml_tensor * k_proj = mm_w(ctx, w.Wk, x_norm);
    ggml_tensor * v_proj = mm_w(ctx, w.Wv, x_norm);

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q_proj, cfg.head_dim, cfg.n_q_heads,  cfg.n_suffix);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k_proj, cfg.head_dim, cfg.n_kv_heads, cfg.n_suffix);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v_proj, cfg.head_dim, cfg.n_kv_heads, cfg.n_suffix);

    ggml_tensor * q_rope = rope_q_or_k(ctx, q_h, positions_full, cfg);
    ggml_tensor * k_rope = rope_q_or_k(ctx, k_h, positions_full, cfg);

    ggml_tensor * K_full = ggml_concat(ctx, cached_K, k_rope, 2);
    ggml_tensor * V_full = ggml_concat(ctx, cached_V, v_h,    2);

    ggml_tensor * Q  = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
    ggml_tensor * Kp = ggml_permute(ctx, K_full, 0, 2, 1, 3);
    ggml_tensor * Vp = ggml_permute(ctx, V_full, 0, 2, 1, 3);
    const float scale = 1.f / std::sqrt(static_cast<float>(cfg.head_dim));
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, Kp, Vp, mask_full, scale,
                                            0.f,  0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att_pre_o = ggml_reshape_2d(ctx, fa, cfg.q_full_dim, cfg.n_suffix);
    ggml_tensor * h1 = ggml_add(ctx, x_in, mm_w(ctx, w.Wo, att_pre_o));

    ggml_tensor * x_norm_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.Wln_post);
    ggml_tensor * inter      = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, w.Wgate, x_norm_mlp)),
                                        mm_w(ctx, w.Wup, x_norm_mlp));
    return ggml_add(ctx, h1, mm_w(ctx, w.Wdown, inter));
}

ggml_tensor * build_expert_cross_attn_layer(
    ggml_context * ctx, const ExpertLayerW & w,
    ggml_tensor * x_in, ggml_tensor * cached_K, ggml_tensor * cached_V,
    ggml_tensor * positions_rebased, ggml_tensor * mask_prefix_only,
    const Config & cfg)
{
    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.Wln_in);

    ggml_tensor * q_proj = mm_w(ctx, w.Wq, x_norm);
    ggml_tensor * q_h    = ggml_reshape_3d(ctx, q_proj, cfg.head_dim, cfg.n_q_heads, cfg.n_suffix);
    ggml_tensor * q_rope = rope_q_or_k(ctx, q_h, positions_rebased, cfg);

    ggml_tensor * cK_flat = ggml_reshape_2d(ctx, cached_K, cfg.kv_full_dim, cfg.n_prefix);
    ggml_tensor * cV_flat = ggml_reshape_2d(ctx, cached_V, cfg.kv_full_dim, cfg.n_prefix);
    ggml_tensor * K_repro = ggml_reshape_3d(ctx, mm_w(ctx, w.Wk, cK_flat),
                                            cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);
    ggml_tensor * V_repro = ggml_reshape_3d(ctx, mm_w(ctx, w.Wv, cV_flat),
                                            cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);

    ggml_tensor * Q  = ggml_permute(ctx, q_rope,  0, 2, 1, 3);
    ggml_tensor * Kp = ggml_permute(ctx, K_repro, 0, 2, 1, 3);
    ggml_tensor * Vp = ggml_permute(ctx, V_repro, 0, 2, 1, 3);
    const float scale = 1.f / std::sqrt(static_cast<float>(cfg.head_dim));
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, Kp, Vp, mask_prefix_only, scale,
                                            0.f,  0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att_pre_o = ggml_reshape_2d(ctx, fa, cfg.q_full_dim, cfg.n_suffix);
    ggml_tensor * h1 = ggml_add(ctx, x_in, mm_w(ctx, w.Wo, att_pre_o));

    ggml_tensor * x_norm_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.Wln_post);
    ggml_tensor * inter      = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, w.Wgate, x_norm_mlp)),
                                        mm_w(ctx, w.Wup, x_norm_mlp));
    return ggml_add(ctx, h1, mm_w(ctx, w.Wdown, inter));
}

}

namespace {

static void vram_probe(ggml_backend_t backend, const char * label) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) return;
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    static size_t prev_free = 0;
    static bool   have_prev = false;
    const double MiB = 1024.0 * 1024.0;
    const long long used = (long long)(total_b - free_b);
    if (have_prev) {
        const long long delta = (long long)prev_free - (long long)free_b;
        std::printf("vla: [vram] %-22s used=%.1f MiB  free=%.1f MiB  (+%.1f MiB)\n",
                    label, used / MiB, free_b / MiB, delta / MiB);
    } else {
        std::printf("vla: [vram] %-22s used=%.1f MiB  free=%.1f MiB\n",
                    label, used / MiB, free_b / MiB);
    }
    prev_free = free_b;
    have_prev = true;
}

static ggml_type resolve_weight_dtype() {
    const char * e = std::getenv("VLA_WEIGHT_DTYPE");
    if (!e) return GGML_TYPE_BF16;
    if (std::strcmp(e, "f32")  == 0) return GGML_TYPE_F32;
    if (std::strcmp(e, "bf16") == 0) return GGML_TYPE_BF16;
    if (std::strcmp(e, "f16")  == 0) return GGML_TYPE_F16;
    std::fprintf(stderr, "vla: unknown VLA_WEIGHT_DTYPE='%s', using bf16\n", e);
    return GGML_TYPE_BF16;
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

SmolVLAModelArch* smolvla_load_impl(const std::string& mmproj_path,
                                    const std::string& ckpt_path,
                                    const std::string& config_path) {
    auto* m = new SmolVLAModelArch();

    const bool use_gguf = ends_with_gguf(ckpt_path);
    safetensors  st;
    gguf_source  gst;

    if (use_gguf) {
        if (!gst.open(ckpt_path)) {
            delete m;
            return nullptr;
        }
        if (!load_config_from_gguf(gst, m->cfg)) {
            delete m;
            return nullptr;
        }
        std::printf("vla: config = %s (gguf KV)\n", ckpt_path.c_str());
    } else {
        const std::string cfg_path = config_path.empty() ? default_config_path(ckpt_path)
                                                         : config_path;
        if (!load_config_from_json(cfg_path, m->cfg)) {
            delete m;
            return nullptr;
        }
        std::printf("vla: config = %s\n", cfg_path.c_str());
    }

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init( 0);
    if (m->backend) {
        m->is_cuda = true;
        m->is_gpu  = true;
        std::printf("vla: backend = CUDA (device 0)\n");
    } else {
        std::fprintf(stderr, "vla: ggml_backend_cuda_init failed; falling back to CPU\n");
    }
#elif defined(GGML_USE_METAL)
    m->backend = ggml_backend_metal_init();
    if (m->backend) {
        m->is_gpu = true;
        std::printf("vla: backend = Metal\n");
    } else {
        std::fprintf(stderr, "vla: ggml_backend_metal_init failed; falling back to CPU\n");
    }
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) {
            std::fprintf(stderr, "vla: ggml_backend_cpu_init failed\n");
            delete m;
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(m->backend, 4);
        std::printf("vla: backend = CPU (4 threads)\n");
    }
    vram_probe(m->backend, "after backend init");

    m->weight_dtype = resolve_weight_dtype();
    std::printf("vla: tower weights resident as %s\n", ggml_type_name(m->weight_dtype));

    // Vision tower geometry: from gguf KV (self-contained ckpt), else SmolVLM2-500M defaults.
    (void) mmproj_path;
    if (use_gguf) {
        auto vu = [&](const char * k, int64_t & d) { if (gst.has_key(k)) d = (int64_t) gst.get_u32(k); };
        vu("smolvla.vit_hidden", m->vit_hidden); vu("smolvla.vit_layers", m->vit_layers);
        vu("smolvla.vit_heads",  m->vit_heads);  vu("smolvla.patch_size", m->vit_patch);
        vu("smolvla.image_size", m->vit_image);  vu("smolvla.vit_pixel_shuffle", m->vit_scale);
        vu("smolvla.n_img_tokens", m->vit_n_tokens); vu("smolvla.vit_inter", m->vit_inter);
        if (gst.has_key("smolvla.vit_ln_eps")) m->vit_ln_eps = gst.get_f32("smolvla.vit_ln_eps");
    }
    {
        const int64_t grid = m->vit_image / m->vit_patch;
        const int64_t k = grid / m->vit_scale;
        if (k * k != m->vit_n_tokens) {
            std::fprintf(stderr, "vla: smolvla vit geometry mismatch (grid=%lld scale=%lld -> %lld tokens, KV says %lld)\n",
                         (long long) grid, (long long) m->vit_scale, (long long) (k * k), (long long) m->vit_n_tokens);
            ggml_backend_free(m->backend);
            delete m;
            return nullptr;
        }
        m->cfg.n_img = m->vit_n_tokens;
    }
    m->cfg.n_prefix = m->cfg.n_img + m->cfg.n_lang + m->cfg.n_state;
    m->cfg.n_full   = m->cfg.n_prefix + m->cfg.n_suffix;

    if (!use_gguf) {
        if (!st.open(ckpt_path)) {
            std::fprintf(stderr, "vla: failed to open %s\n", ckpt_path.c_str());
            ggml_backend_free(m->backend);
            delete m;
            return nullptr;
        }
        if (m->cfg.n_layers <= 0) {
            const std::string prefix = "model.vlm_with_expert.vlm.model.text_model.layers.";
            int max_layer = -1;
            for (const auto & kv : st.tensors) {
                if (kv.first.compare(0, prefix.size(), prefix) == 0) {
                    const int idx = std::atoi(kv.first.c_str() + prefix.size());
                    if (idx > max_layer) max_layer = idx;
                }
            }
            if (max_layer < 0) {
                std::fprintf(stderr, "vla: cannot infer n_layers from %s\n", ckpt_path.c_str());
                ggml_backend_free(m->backend);
                delete m;
                return nullptr;
            }
            m->cfg.n_layers = max_layer + 1;
        }
        {
            const auto it = st.tensors.find("model.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight");
            if (it == st.tensors.end() || it->second.shape.size() != 2) {
                std::fprintf(stderr, "vla: missing/malformed expert gate_proj for shape derivation\n");
                ggml_backend_free(m->backend);
                delete m;
                return nullptr;
            }

            m->cfg.expert_inter = it->second.shape[0];
            if (it->second.shape[1] != m->cfg.expert_h) {
                std::fprintf(stderr, "vla: expert_h mismatch - config implies %lld, "
                                     "checkpoint gate_proj has %lld\n",
                             (long long) m->cfg.expert_h, (long long) it->second.shape[1]);
                ggml_backend_free(m->backend);
                delete m;
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
            ggml_backend_free(m->backend);
            delete m;
            return nullptr;
        }
    } else {
        const std::string cfg_path_local = config_path.empty()
            ? default_config_path(ckpt_path) : config_path;
        const auto pos = cfg_path_local.find_last_of("/\\");
        const std::string model_dir = (pos == std::string::npos) ? std::string(".")
                                                                 : cfg_path_local.substr(0, pos);
        load_normalizer_stats(model_dir, *m);
    }

    ggml_init_params gparams = {
         size_t(32) * 1024 * 1024,
         nullptr,
         true,
    };
    m->ctx_weights = ggml_init(gparams);
    if (!m->ctx_weights) {
        std::fprintf(stderr, "vla: ggml_init (weights) failed\n");
        ggml_backend_free(m->backend);
        delete m;
        return nullptr;
    }

    auto * ctx = m->ctx_weights;
    const auto & cfg = m->cfg;
    const ggml_type wdt = m->weight_dtype;

    struct PendingF32  { std::string name; ggml_tensor * t; std::vector<int64_t> shape; };
    struct PendingBF16 { std::string name; ggml_tensor * t; };
    std::vector<PendingF32>  pending_f32;
    std::vector<PendingBF16> pending_bf16;

    m->E_lang = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, cfg.hidden,  49280);
    pending_bf16.push_back({
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight", m->E_lang});

    m->Wstate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.max_state_dim, cfg.hidden);
    m->bstate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.hidden);
    pending_f32.push_back({"model.state_proj.weight", m->Wstate, {cfg.hidden, cfg.max_state_dim}});
    pending_f32.push_back({"model.state_proj.bias",   m->bstate, {cfg.hidden}});

    // Vision tower weights (SigLIP-B/16 encoder + single-linear pixel-shuffle connector).
    {
        const int64_t H = m->vit_hidden, FF = m->vit_inter, P = m->vit_patch;
        const int64_t grid = m->vit_image / P, n_patches = grid * grid;
        const int64_t c4 = H * m->vit_scale * m->vit_scale;
        const char * VP = "model.vlm_with_expert.vlm.model.vision_model.";
        m->vit_patch_w   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, P, P, 3, H);
        m->vit_patch_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        m->vit_pos       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_patches);
        m->vit_post_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        m->vit_post_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        pending_f32.push_back({std::string(VP) + "embeddings.patch_embedding.weight", m->vit_patch_w, {H, 3, P, P}});
        pending_f32.push_back({std::string(VP) + "embeddings.patch_embedding.bias",   m->vit_patch_b, {H}});
        pending_f32.push_back({std::string(VP) + "embeddings.position_embedding.weight", m->vit_pos, {n_patches, H}});
        pending_f32.push_back({std::string(VP) + "post_layernorm.weight", m->vit_post_ln_w, {H}});
        pending_f32.push_back({std::string(VP) + "post_layernorm.bias",   m->vit_post_ln_b, {H}});
        m->vit.resize(m->vit_layers);
        for (int64_t i = 0; i < m->vit_layers; ++i) {
            SigLipLayerW & w = m->vit[i];
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
    for (int i = 0; i < cfg.n_layers; ++i) {
        VlmLayerW & w = m->vlm_layers[i];
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
    m->Wnorm_vlm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.hidden);
    pending_f32.push_back({"model.vlm_with_expert.vlm.model.text_model.norm.weight",
                           m->Wnorm_vlm, {cfg.hidden}});

    m->expert_layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        ExpertLayerW & w = m->expert_layers[i];
        w.is_self_attn = (i % cfg.self_attn_every_n == 0);
        w.Wln_in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
        w.Wq       = ggml_new_tensor_2d(ctx, wdt, cfg.expert_h, cfg.q_full_dim);
        if (w.is_self_attn) {
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
        if (w.is_self_attn) {
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
    m->W_at1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * cfg.expert_h, cfg.expert_h);
    m->b_at1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
    m->W_at2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.expert_h, cfg.expert_h);
    m->b_at2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.expert_h);
    pending_f32.push_back({"model.action_in_proj.weight",      m->W_ain, {cfg.expert_h, cfg.max_action_dim}});
    pending_f32.push_back({"model.action_in_proj.bias",        m->b_ain, {cfg.expert_h}});
    pending_f32.push_back({"model.action_time_mlp_in.weight",  m->W_at1, {cfg.expert_h, 2 * cfg.expert_h}});
    pending_f32.push_back({"model.action_time_mlp_in.bias",    m->b_at1, {cfg.expert_h}});
    pending_f32.push_back({"model.action_time_mlp_out.weight", m->W_at2, {cfg.expert_h, cfg.expert_h}});
    pending_f32.push_back({"model.action_time_mlp_out.bias",   m->b_at2, {cfg.expert_h}});

    m->W_aout = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.expert_h, cfg.max_action_dim);
    m->b_aout = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.max_action_dim);
    pending_f32.push_back({"model.action_out_proj.weight", m->W_aout, {cfg.max_action_dim, cfg.expert_h}});
    pending_f32.push_back({"model.action_out_proj.bias",   m->b_aout, {cfg.max_action_dim}});

    m->time_bcasts.assign(cfg.num_steps, nullptr);
    for (int step = 0; step < cfg.num_steps; ++step) {
        m->time_bcasts[step] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   cfg.expert_h, cfg.n_suffix);
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) {
        std::fprintf(stderr, "vla: ggml_backend_alloc_ctx_tensors (weights) failed\n");
        delete m;
        return nullptr;
    }
    std::printf("vla: [vram] weight_buf = %.1f MiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0));
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
    auto stream_bf16 = [&](const std::string & hf_name, ggml_tensor * t) -> bool {
        std::vector<ggml_bf16_t> hbuf(ggml_nelements(t));
        const bool ok = use_gguf
            ? gst.read_raw(hf_to_gguf(hf_name), hbuf.data(), ggml_nbytes(t), "BF16")
            : st .read_raw(hf_name,             hbuf.data(), ggml_nbytes(t), "BF16");
        if (!ok) {
            std::fprintf(stderr, "vla: read_raw failed for %s\n", hf_name.c_str());
            return false;
        }
        ggml_backend_tensor_set(t, hbuf.data(), 0, ggml_nbytes(t));
        return true;
    };

    for (auto & p : pending_f32) {
        if (!stream_f32(p.name, p.t, p.shape)) { delete m; return nullptr; }
    }
    for (auto & p : pending_bf16) {
        if (!stream_bf16(p.name, p.t))         { delete m; return nullptr; }
    }

    {
        const float dt = -1.f / static_cast<float>(cfg.num_steps);
        for (int step = 0; step < cfg.num_steps; ++step) {
            const double time = 1.0 + double(step) * double(dt);
            const auto te = sinusoidal_time_emb(time, cfg.expert_h,
                                                cfg.min_period, cfg.max_period);
            std::vector<float> tile(cfg.expert_h * cfg.n_suffix);
            for (int64_t t = 0; t < cfg.n_suffix; ++t) {
                std::memcpy(tile.data() + t * cfg.expert_h,
                            te.data(), cfg.expert_h * sizeof(float));
            }
            ggml_backend_tensor_set(m->time_bcasts[step], tile.data(),
                                    0, tile.size() * sizeof(float));
        }
    }

    return m;
}
}

namespace {
bool build_compute_graph(SmolVLAModelArch* m, int n_views) {
    if (n_views < 1) {
        std::fprintf(stderr, "vla: build_compute_graph: n_views=%d invalid\n", n_views);
        return false;
    }
    const Config & cfg_model = m->cfg;
    Config cfg = cfg_model;

    cfg.n_img = cfg_model.n_img * int64_t(n_views);

    ggml_init_params gparams = {
         size_t(64) * 1024 * 1024,
         nullptr,
         true,
    };
    ggml_context * ctx = ggml_init(gparams);
    if (!ctx) {
        std::fprintf(stderr, "vla: build_compute_graph: ggml_init failed\n");
        return false;
    }

    const int64_t n_lang_max   = cfg.n_lang;
    const int64_t n_prefix_max = cfg.n_img + n_lang_max + cfg.n_state;
    const int64_t n_full_max   = n_prefix_max + cfg.n_suffix;

    ggml_tensor * img_emb_in     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.hidden,         cfg.n_img);
    ggml_tensor * lang_ids       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_lang_max);
    ggml_tensor * state_t        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.max_state_dim);
    ggml_tensor * x0             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.max_action_dim, cfg.n_suffix);

    ggml_tensor * mask_prefill   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_prefix_max, n_prefix_max);
    ggml_tensor * pos_prefill    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_prefix_max);
    ggml_tensor * mask_full      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_full_max,   cfg.n_suffix);
    ggml_tensor * mask_pfx_only  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_prefix_max, cfg.n_suffix);
    ggml_tensor * pos_full       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_suffix);
    ggml_tensor * pos_rebased    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_suffix);

    for (ggml_tensor * t : {img_emb_in, lang_ids, state_t, x0,
                            mask_prefill, pos_prefill, mask_full,
                            mask_pfx_only, pos_full, pos_rebased}) {
        ggml_set_input(t);
    }

    Config cfg_built = cfg;
    cfg_built.n_lang   = n_lang_max;
    cfg_built.n_prefix = n_prefix_max;
    cfg_built.n_full   = n_full_max;

    const float lang_scale = std::sqrt(static_cast<float>(cfg.hidden));
    ggml_tensor * img_emb_scaled  = ggml_scale(ctx, img_emb_in, lang_scale);
    ggml_tensor * lang_raw        = ggml_get_rows(ctx, m->E_lang, lang_ids);
    ggml_tensor * lang_emb_scaled = ggml_scale(ctx, lang_raw, lang_scale);
    ggml_tensor * state_pre       = ggml_add(ctx, ggml_mul_mat(ctx, m->Wstate, state_t), m->bstate);
    ggml_tensor * state_emb       = ggml_reshape_2d(ctx, state_pre, cfg.hidden, 1);
    ggml_tensor * embs_il         = ggml_concat(ctx, img_emb_scaled, lang_emb_scaled, 1);
    ggml_tensor * prefix_embs     = ggml_concat(ctx, embs_il, state_emb, 1);

    ggml_tensor * mask_prefill_f16 = ggml_cast(ctx, mask_prefill,  GGML_TYPE_F16);
    ggml_tensor * mask_full_f16    = ggml_cast(ctx, mask_full,     GGML_TYPE_F16);
    ggml_tensor * mask_pfx_only_f16= ggml_cast(ctx, mask_pfx_only, GGML_TYPE_F16);
    std::vector<ggml_tensor *> k_cache(cfg.n_layers);
    std::vector<ggml_tensor *> v_cache(cfg.n_layers);
    {
        ggml_tensor * h = prefix_embs;
        for (int i = 0; i < cfg.n_layers; ++i) {
            h = build_vlm_layer(ctx, m->vlm_layers[i], h, mask_prefill_f16, pos_prefill,
                                cfg_built, &k_cache[i], &v_cache[i]);
        }
    }

    const float dt = -1.f / static_cast<float>(cfg.num_steps);
    ggml_tensor * x_t = x0;

    for (int step = 0; step < cfg.num_steps; ++step) {
        ggml_tensor * time_bcast     = m->time_bcasts[step];
        ggml_tensor * action_emb     = ggml_add(ctx, ggml_mul_mat(ctx, m->W_ain, x_t), m->b_ain);
        ggml_tensor * action_time_in = ggml_concat(ctx, action_emb, time_bcast, 0);
        ggml_tensor * mlp1           = ggml_add(ctx, ggml_mul_mat(ctx, m->W_at1, action_time_in), m->b_at1);
        ggml_tensor * suffix_embs    = ggml_add(ctx,
                                                ggml_mul_mat(ctx, m->W_at2, ggml_silu(ctx, mlp1)),
                                                m->b_at2);
        ggml_tensor * h = suffix_embs;
        for (int li = 0; li < cfg.n_layers; ++li) {
            if (m->expert_layers[li].is_self_attn) {
                h = build_expert_self_attn_layer(ctx, m->expert_layers[li], h,
                                                 k_cache[li], v_cache[li],
                                                 pos_full, mask_full_f16, cfg_built);
            } else {
                h = build_expert_cross_attn_layer(ctx, m->expert_layers[li], h,
                                                  k_cache[li], v_cache[li],
                                                  pos_rebased, mask_pfx_only_f16, cfg_built);
            }
        }
        ggml_tensor * h_final = ggml_mul(ctx, ggml_rms_norm(ctx, h, cfg.rms_eps), m->Wnorm_expert);
        ggml_tensor * v_t     = ggml_add(ctx, ggml_mul_mat(ctx, m->W_aout, h_final), m->b_aout);
        x_t = ggml_add(ctx, x_t, ggml_scale(ctx, v_t, dt));
    }

    ggml_set_output(x_t);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx,  16384,  false);
    ggml_build_forward_expand(gf, x_t);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(m->backend);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!galloc) {
        std::fprintf(stderr, "vla: build_compute_graph: ggml_gallocr_new failed\n");
        ggml_free(ctx);
        return false;
    }
    if (!ggml_gallocr_reserve(galloc, gf)) {
        std::fprintf(stderr, "vla: build_compute_graph: ggml_gallocr_reserve failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    std::printf("vla: [vram] gallocr compute buf = %.1f MiB\n",
                ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0));
    vram_probe(m->backend, "after gallocr reserve");

    m->ctx_compute       = ctx;
    m->galloc            = galloc;
    m->gf_cached         = gf;
    m->in_img_emb        = img_emb_in;
    m->in_lang_ids       = lang_ids;
    m->in_state          = state_t;
    m->in_x0             = x0;
    m->in_mask_prefill   = mask_prefill;
    m->in_pos_prefill    = pos_prefill;
    m->in_mask_full      = mask_full;
    m->in_mask_pfx_only  = mask_pfx_only;
    m->in_pos_full       = pos_full;
    m->in_pos_rebased    = pos_rebased;
    m->out_x_t           = x_t;

    return true;
}
}

SmolVLAModelArch::~SmolVLAModelArch() {

    if (galloc)      ggml_gallocr_free(galloc);
    if (ctx_compute) ggml_free(ctx_compute);
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

namespace {
std::vector<float> predict_impl(SmolVLAModelArch* m, const Inputs& in) {
    using clock = std::chrono::steady_clock;
    const auto t_total_begin = clock::now();

    m->stats = Stats{};

    Config cfg = m->cfg;

    const size_t per_view_n = size_t(m->cfg.n_img * cfg.hidden);

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
        img_emb_pre.assign(in.precomputed_img_emb, in.precomputed_img_emb + img_emb_n);

    } else {
        if (in.n_images < 1 || in.images == nullptr) {
            std::fprintf(stderr, "vla: at least one image is required\n");
            return {};
        }
        n_views   = in.n_images;
        img_emb_n = per_view_n * size_t(n_views);
        img_emb_pre.resize(img_emb_n);

        const int64_t H = m->vit_hidden, grid = m->vit_image / m->vit_patch, n_patches = grid * grid;
        const int64_t s = m->vit_scale, c4 = H * s * s, K = m->vit_n_tokens;
        const auto t_vision_begin = clock::now();

        // Graph A: SigLIP ViT (conv patch-embed -> +pos -> layers -> post_ln), plain sequential positions.
        ggml_init_params vpA = { size_t(256) * 1024 * 1024, nullptr, true };
        ggml_context * VC = ggml_init(vpA);
        if (!VC) { std::fprintf(stderr, "vla(smolvla): ggml_init(vision ctx) failed\n"); return {}; }
        ggml_tensor * t_px = ggml_new_tensor_3d(VC, GGML_TYPE_F32, m->vit_image, m->vit_image, 3); ggml_set_input(t_px);
        ggml_tensor * conv = ggml_conv_2d(VC, m->vit_patch_w, t_px, (int) m->vit_patch, (int) m->vit_patch, 0, 0, 1, 1);
        ggml_tensor * patches = ggml_cont(VC, ggml_transpose(VC, ggml_reshape_2d(VC, conv, n_patches, H)));
        ggml_tensor * hv = ggml_add(VC, ggml_add(VC, patches, m->vit_patch_b), m->vit_pos);
        for (int64_t i = 0; i < m->vit_layers; ++i)
            hv = build_siglip_layer(VC, m->vit[i], hv, n_patches, m->vit_heads, H / m->vit_heads, H, m->vit_ln_eps);
        ggml_tensor * post_ln = ggml_add(VC, ggml_mul(VC, ggml_norm(VC, hv, m->vit_ln_eps), m->vit_post_ln_w), m->vit_post_ln_b);
        ggml_set_output(post_ln);
        ggml_cgraph * gA = ggml_new_graph_custom(VC, 8192, false);
        ggml_build_forward_expand(gA, post_ln);
        ggml_gallocr_t vgA = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
        if (!vgA || !ggml_gallocr_alloc_graph(vgA, gA)) {
            std::fprintf(stderr, "vla(smolvla): vision gallocr A alloc failed\n");
            if (vgA) ggml_gallocr_free(vgA);
            ggml_free(VC);
            return {};
        }

        // Graph B: pixel-shuffle connector, a single bias-free matmul (c4 -> hidden).
        ggml_init_params vpB = { size_t(64) * 1024 * 1024, nullptr, true };
        ggml_context * MC = ggml_init(vpB);
        if (!MC) { std::fprintf(stderr, "vla(smolvla): ggml_init(connector ctx) failed\n"); ggml_gallocr_free(vgA); ggml_free(VC); return {}; }
        ggml_tensor * t_shuf = ggml_new_tensor_2d(MC, GGML_TYPE_F32, c4, K); ggml_set_input(t_shuf);
        ggml_tensor * img_embeds = ggml_mul_mat(MC, m->mm_fc, t_shuf);
        ggml_set_output(img_embeds);
        ggml_cgraph * gB = ggml_new_graph(MC);
        ggml_build_forward_expand(gB, img_embeds);
        ggml_gallocr_t vgB = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
        if (!vgB || !ggml_gallocr_alloc_graph(vgB, gB)) {
            std::fprintf(stderr, "vla(smolvla): vision gallocr B alloc failed\n");
            if (vgB) ggml_gallocr_free(vgB);
            ggml_gallocr_free(vgA); ggml_free(MC); ggml_free(VC);
            return {};
        }

        std::vector<float> chw, post_host((size_t) H * n_patches), shuf_host((size_t) c4 * K);
        bool vok = true;
        for (int v = 0; v < n_views && vok; ++v) {
            if (!preprocess_image_chw(in.images[v], m->vit_image, chw)) { vok = false; break; }
            ggml_backend_tensor_set(t_px, chw.data(), 0, ggml_nbytes(t_px));
            if (ggml_backend_graph_compute(m->backend, gA) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(smolvla): vision compute A failed (view %d)\n", v); vok = false; break;
            }
            ggml_backend_tensor_get(post_ln, post_host.data(), 0, ggml_nbytes(post_ln));
            pixel_shuffle_hf(post_host.data(), shuf_host.data(), H, grid, s);
            ggml_backend_tensor_set(t_shuf, shuf_host.data(), 0, ggml_nbytes(t_shuf));
            if (ggml_backend_graph_compute(m->backend, gB) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(smolvla): connector compute failed (view %d)\n", v); vok = false; break;
            }
            ggml_backend_tensor_get(img_embeds, img_emb_pre.data() + size_t(v) * per_view_n, 0, ggml_nbytes(img_embeds));
        }
        ggml_gallocr_free(vgB); ggml_free(MC); ggml_gallocr_free(vgA); ggml_free(VC);
        if (!vok) return {};
        m->stats.ms_vision = std::chrono::duration<float, std::milli>(clock::now() - t_vision_begin).count();
    }

    cfg.n_img    = m->cfg.n_img * int64_t(n_views);
    cfg.n_prefix = cfg.n_img + cfg.n_lang + cfg.n_state;
    cfg.n_full   = cfg.n_prefix + cfg.n_suffix;

    if (in.n_lang < 1 || in.n_lang > int(cfg.n_lang)) {
        std::fprintf(stderr, "vla: lang_tokens length %d out of range [1, %lld]\n",
                     in.n_lang, (long long) cfg.n_lang);
        return {};
    }

    // The language tokens index E_lang via ggml_get_rows, which does not bound
    // its indices. Reject any token id outside the embedding table before the
    // gather so an out-of-range id cannot read past the weights.
    const int64_t vocab_rows = m->E_lang ? m->E_lang->ne[1] : 0;
    for (int i = 0; i < in.n_lang; ++i) {
        if (in.lang_tokens[i] < 0 || in.lang_tokens[i] >= vocab_rows) {
            std::fprintf(stderr, "vla: lang_tokens[%d]=%d out of vocab range [0, %lld)\n",
                         i, in.lang_tokens[i], (long long) vocab_rows);
            return {};
        }
    }

    if (in.timing_detail == TimingDetail::NONE) {

        if (m->gf_cached == nullptr || m->cached_n_views != n_views) {
            if (m->galloc)      ggml_gallocr_free(m->galloc);
            if (m->ctx_compute) ggml_free(m->ctx_compute);
            m->galloc      = nullptr;
            m->ctx_compute = nullptr;
            m->gf_cached   = nullptr;
            if (!build_compute_graph(m, n_views)) {
                std::fprintf(stderr, "vla: cached graph build failed\n");
                return {};
            }
            m->cached_n_views = n_views;
        }

        const int64_t n_lang_max   = m->cfg.n_lang;
        const int64_t n_prefix_max = cfg.n_img + n_lang_max + cfg.n_state;
        const int64_t n_full_max   = n_prefix_max + cfg.n_suffix;
        const int64_t pad_start    = cfg.n_img + in.n_lang;
        const int64_t pad_end      = cfg.n_img + n_lang_max;

        std::vector<float> state_host(cfg.max_state_dim);
        std::memcpy(state_host.data(), in.state, cfg.max_state_dim * sizeof(float));
        for (int64_t i = 0; i < cfg.real_state_dim; ++i) {
            state_host[i] = (state_host[i] - m->state_mean[i]) / (m->state_std[i] + cfg.norm_eps);
        }

        std::vector<float> noise_host(cfg.n_suffix * cfg.max_action_dim);
        if (in.noise) {
            std::memcpy(noise_host.data(), in.noise, noise_host.size() * sizeof(float));
        } else {
            std::normal_distribution<float> dist(0.f, 1.f);
            for (auto & v : noise_host) v = dist(m->rng);
        }

        std::vector<int32_t> lang_host(n_lang_max, 0);
        std::memcpy(lang_host.data(), in.lang_tokens, in.n_lang * sizeof(int32_t));

        const int64_t state_pos       = cfg.n_img + in.n_lang;
        const int64_t suffix_pos_base = state_pos + 1;
        std::vector<float>   mask_prefill_host(n_prefix_max * n_prefix_max);
        std::vector<int32_t> pos_prefill_host (n_prefix_max);
        for (int64_t i = 0; i < n_prefix_max; ++i) {
            for (int64_t j = 0; j < n_prefix_max; ++j) {
                bool blocked = false;
                if ((i < n_prefix_max - 1) && (j == n_prefix_max - 1)) blocked = true;
                if (j >= pad_start && j < pad_end)                     blocked = true;
                mask_prefill_host[i * n_prefix_max + j] = blocked ? -INFINITY : 0.f;
            }
            pos_prefill_host[i] = (i == n_prefix_max - 1)
                ? static_cast<int32_t>(state_pos)
                : static_cast<int32_t>(i);
        }

        std::vector<float>   mask_full_host       (n_full_max   * cfg.n_suffix);
        std::vector<float>   mask_prefix_only_host(n_prefix_max * cfg.n_suffix, 0.f);
        std::vector<int32_t> pos_full_host        (cfg.n_suffix);
        std::vector<int32_t> pos_rebased_host     (cfg.n_suffix);
        for (int64_t i = 0; i < cfg.n_suffix; ++i) {
            for (int64_t j = 0; j < n_full_max; ++j) {
                bool blocked;
                if (j < n_prefix_max) {
                    blocked = (j >= pad_start && j < pad_end);
                } else {
                    blocked = ((j - n_prefix_max) > i);
                }
                mask_full_host[i * n_full_max + j] = blocked ? -INFINITY : 0.f;
            }
            for (int64_t j = 0; j < n_prefix_max; ++j) {
                if (j >= pad_start && j < pad_end) {
                    mask_prefix_only_host[i * n_prefix_max + j] = -INFINITY;
                }
            }
            pos_full_host   [i] = static_cast<int32_t>(suffix_pos_base + i);
            pos_rebased_host[i] = static_cast<int32_t>(i);
        }

        if (!ggml_gallocr_alloc_graph(m->galloc, m->gf_cached)) {
            std::fprintf(stderr, "vla: ggml_gallocr_alloc_graph failed\n");
            return {};
        }

        ggml_backend_tensor_set(m->in_img_emb,       img_emb_pre.data(),    0, img_emb_n * sizeof(float));
        ggml_backend_tensor_set(m->in_lang_ids,      lang_host.data(),      0, n_lang_max        * sizeof(int32_t));
        ggml_backend_tensor_set(m->in_state,         state_host.data(),     0, cfg.max_state_dim * sizeof(float));
        ggml_backend_tensor_set(m->in_x0,            noise_host.data(),     0, noise_host.size() * sizeof(float));
        ggml_backend_tensor_set(m->in_mask_prefill,  mask_prefill_host.data(),     0, mask_prefill_host.size()     * sizeof(float));
        ggml_backend_tensor_set(m->in_pos_prefill,   pos_prefill_host.data(),      0, pos_prefill_host.size()      * sizeof(int32_t));
        ggml_backend_tensor_set(m->in_mask_full,     mask_full_host.data(),        0, mask_full_host.size()        * sizeof(float));
        ggml_backend_tensor_set(m->in_mask_pfx_only, mask_prefix_only_host.data(), 0, mask_prefix_only_host.size() * sizeof(float));
        ggml_backend_tensor_set(m->in_pos_full,      pos_full_host.data(),         0, pos_full_host.size()         * sizeof(int32_t));
        ggml_backend_tensor_set(m->in_pos_rebased,   pos_rebased_host.data(),      0, pos_rebased_host.size()      * sizeof(int32_t));

        const auto t0 = clock::now();
        if (ggml_backend_graph_compute(m->backend, m->gf_cached) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla: ggml compute (cached) failed\n");
            return {};
        }
        m->stats.ms_inference = std::chrono::duration<float, std::milli>(
            clock::now() - t0).count();

        std::vector<float> out(cfg.n_suffix * cfg.max_action_dim);
        ggml_backend_tensor_get(m->out_x_t, out.data(), 0, out.size() * sizeof(float));
        for (int64_t r = 0; r < cfg.n_suffix; ++r) {
            float * row = out.data() + r * cfg.max_action_dim;
            for (int64_t j = 0; j < cfg.real_action_dim; ++j) {
                row[j] = row[j] * (m->action_std[j] + cfg.norm_eps) + m->action_mean[j];
            }
        }

        m->stats.ms_total = std::chrono::duration<float, std::milli>(
            clock::now() - t_total_begin).count();
        return out;
    }

    ggml_init_params gparams = {
         size_t(64) * 1024 * 1024,
         nullptr,
         true,
    };
    ggml_context * ctx = ggml_init(gparams);
    if (!ctx) {
        std::fprintf(stderr, "vla: ggml_init (compute) failed\n");
        return {};
    }

    if (in.n_lang < 1 || in.n_lang > int(cfg.n_lang)) {
        std::fprintf(stderr, "vla: lang_tokens length %d out of range [1, %lld]\n",
                     in.n_lang, (long long) cfg.n_lang);
        ggml_free(ctx);
        return {};
    }

    cfg.n_lang   = in.n_lang;
    cfg.n_prefix = cfg.n_img + cfg.n_lang + cfg.n_state;
    cfg.n_full   = cfg.n_prefix + cfg.n_suffix;
    ggml_tensor * img_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.hidden, cfg.n_img);
    ggml_tensor * lang_ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_lang);
    ggml_tensor * state_t    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.max_state_dim);
    ggml_tensor * x0         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.max_action_dim, cfg.n_suffix);

    ggml_tensor * mask_prefill     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_prefix, cfg.n_prefix);
    ggml_tensor * pos_prefill      = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_prefix);
    ggml_tensor * mask_full        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_full,   cfg.n_suffix);
    ggml_tensor * mask_prefix_only = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_prefix, cfg.n_suffix);
    ggml_tensor * pos_full         = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_suffix);
    ggml_tensor * pos_rebased      = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cfg.n_suffix);

    std::vector<float> state_host(cfg.max_state_dim);
    std::memcpy(state_host.data(), in.state, cfg.max_state_dim * sizeof(float));
    for (int64_t i = 0; i < cfg.real_state_dim; ++i) {
        state_host[i] = (state_host[i] - m->state_mean[i]) / (m->state_std[i] + cfg.norm_eps);
    }

    std::vector<float> noise_host(cfg.n_suffix * cfg.max_action_dim);
    if (in.noise) {
        std::memcpy(noise_host.data(), in.noise, noise_host.size() * sizeof(float));
    } else {
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto & v : noise_host) v = dist(m->rng);
    }

    std::vector<float>   mask_prefill_host(cfg.n_prefix * cfg.n_prefix);
    std::vector<int32_t> pos_prefill_host (cfg.n_prefix);
    for (int64_t i = 0; i < cfg.n_prefix; ++i) {
        for (int64_t j = 0; j < cfg.n_prefix; ++j) {
            const bool blocked = (i < cfg.n_prefix - 1) && (j == cfg.n_prefix - 1);
            mask_prefill_host[i * cfg.n_prefix + j] = blocked ? -INFINITY : 0.f;
        }
        pos_prefill_host[i] = static_cast<int32_t>(i);
    }

    std::vector<float>   mask_full_host       (cfg.n_full   * cfg.n_suffix);
    std::vector<float>   mask_prefix_only_host(cfg.n_prefix * cfg.n_suffix, 0.f);
    std::vector<int32_t> pos_full_host        (cfg.n_suffix);
    std::vector<int32_t> pos_rebased_host     (cfg.n_suffix);
    for (int64_t i = 0; i < cfg.n_suffix; ++i) {
        for (int64_t j = 0; j < cfg.n_full; ++j) {
            bool blocked;
            if (j < cfg.n_prefix) blocked = false;
            else                  blocked = ((j - cfg.n_prefix) > i);
            mask_full_host[i * cfg.n_full + j] = blocked ? -INFINITY : 0.f;
        }
        pos_full_host   [i] = static_cast<int32_t>(cfg.n_prefix + i);
        pos_rebased_host[i] = static_cast<int32_t>(i);
    }

    const float lang_scale = std::sqrt(static_cast<float>(cfg.hidden));
    ggml_tensor * img_emb_scaled  = ggml_scale(ctx, img_emb_in, lang_scale);
    ggml_tensor * lang_raw        = ggml_get_rows(ctx, m->E_lang, lang_ids);
    ggml_tensor * lang_emb_scaled = ggml_scale(ctx, lang_raw, lang_scale);
    ggml_tensor * state_pre = ggml_add(ctx, ggml_mul_mat(ctx, m->Wstate, state_t), m->bstate);
    ggml_tensor * state_emb = ggml_reshape_2d(ctx, state_pre, cfg.hidden, 1);
    ggml_tensor * embs_il   = ggml_concat(ctx, img_emb_scaled, lang_emb_scaled, 1);
    ggml_tensor * prefix_embs = ggml_concat(ctx, embs_il, state_emb, 1);

    ggml_tensor * mask_prefill_f16     = ggml_cast(ctx, mask_prefill,     GGML_TYPE_F16);
    ggml_tensor * mask_full_f16        = ggml_cast(ctx, mask_full,        GGML_TYPE_F16);
    ggml_tensor * mask_prefix_only_f16 = ggml_cast(ctx, mask_prefix_only, GGML_TYPE_F16);
    std::vector<ggml_tensor *> k_cache(cfg.n_layers);
    std::vector<ggml_tensor *> v_cache(cfg.n_layers);
    {
        ggml_tensor * h = prefix_embs;
        for (int i = 0; i < cfg.n_layers; ++i) {
            h = build_vlm_layer(ctx, m->vlm_layers[i], h, mask_prefill_f16, pos_prefill,
                                cfg, &k_cache[i], &v_cache[i]);
        }
    }

    std::vector<ggml_tensor *> K_storage(cfg.n_layers);
    std::vector<ggml_tensor *> V_storage(cfg.n_layers);
    if (in.timing_detail == TimingDetail::PHASE) {
        for (int i = 0; i < cfg.n_layers; ++i) {
            K_storage[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                              cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);
            V_storage[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                              cfg.head_dim, cfg.n_kv_heads, cfg.n_prefix);
        }
    }

    auto ms_since = [](auto t0) {
        return std::chrono::duration<float, std::milli>(clock::now() - t0).count();
    };

    auto & K_ref = (in.timing_detail == TimingDetail::PHASE) ? K_storage : k_cache;
    auto & V_ref = (in.timing_detail == TimingDetail::PHASE) ? V_storage : v_cache;

    const float dt = -1.f / static_cast<float>(cfg.num_steps);
    ggml_tensor * x_t = x0;
    std::vector<ggml_tensor *>     time_bcasts(cfg.num_steps, nullptr);
    std::vector<std::vector<float>> time_host  (cfg.num_steps);

    for (int step = 0; step < cfg.num_steps; ++step) {
        const double time = 1.0 + static_cast<double>(step) * static_cast<double>(dt);
        time_host[step] = sinusoidal_time_emb(time, cfg.expert_h, cfg.min_period, cfg.max_period);

        ggml_tensor * time_bcast = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.expert_h, cfg.n_suffix);
        time_bcasts[step] = time_bcast;

        ggml_tensor * action_emb     = ggml_add(ctx, ggml_mul_mat(ctx, m->W_ain, x_t), m->b_ain);
        ggml_tensor * action_time_in = ggml_concat(ctx, action_emb, time_bcast, 0);
        ggml_tensor * mlp1           = ggml_add(ctx, ggml_mul_mat(ctx, m->W_at1, action_time_in), m->b_at1);
        ggml_tensor * suffix_embs    = ggml_add(ctx,
                                                ggml_mul_mat(ctx, m->W_at2, ggml_silu(ctx, mlp1)),
                                                m->b_at2);

        ggml_tensor * h = suffix_embs;
        for (int li = 0; li < cfg.n_layers; ++li) {
            if (m->expert_layers[li].is_self_attn) {
                h = build_expert_self_attn_layer(ctx, m->expert_layers[li], h,
                                                 K_ref[li], V_ref[li],
                                                 pos_full, mask_full_f16, cfg);
            } else {
                h = build_expert_cross_attn_layer(ctx, m->expert_layers[li], h,
                                                  K_ref[li], V_ref[li],
                                                  pos_rebased, mask_prefix_only_f16, cfg);
            }
        }
        ggml_tensor * h_final = ggml_mul(ctx, ggml_rms_norm(ctx, h, cfg.rms_eps), m->Wnorm_expert);
        ggml_tensor * v_t     = ggml_add(ctx, ggml_mul_mat(ctx, m->W_aout, h_final), m->b_aout);
        x_t = ggml_add(ctx, x_t, ggml_scale(ctx, v_t, dt));
    }

    ggml_backend_buffer_t compute_buf = ggml_backend_alloc_ctx_tensors(ctx, m->backend);
    if (!compute_buf) {
        std::fprintf(stderr, "vla: ggml_backend_alloc_ctx_tensors (compute) failed\n");
        ggml_free(ctx);
        return {};
    }

    ggml_backend_tensor_set(img_emb_in,       img_emb_pre.data(),    0, img_emb_n * sizeof(float));
    ggml_backend_tensor_set(lang_ids,         in.lang_tokens,        0, cfg.n_lang        * sizeof(int32_t));
    ggml_backend_tensor_set(state_t,          state_host.data(),     0, cfg.max_state_dim * sizeof(float));
    ggml_backend_tensor_set(x0,               noise_host.data(),     0, noise_host.size()        * sizeof(float));
    ggml_backend_tensor_set(mask_prefill,     mask_prefill_host.data(),     0, mask_prefill_host.size()     * sizeof(float));
    ggml_backend_tensor_set(pos_prefill,      pos_prefill_host.data(),      0, pos_prefill_host.size()      * sizeof(int32_t));
    ggml_backend_tensor_set(mask_full,        mask_full_host.data(),        0, mask_full_host.size()        * sizeof(float));
    ggml_backend_tensor_set(mask_prefix_only, mask_prefix_only_host.data(), 0, mask_prefix_only_host.size() * sizeof(float));
    ggml_backend_tensor_set(pos_full,         pos_full_host.data(),         0, pos_full_host.size()         * sizeof(int32_t));
    ggml_backend_tensor_set(pos_rebased,      pos_rebased_host.data(),      0, pos_rebased_host.size()      * sizeof(int32_t));
    for (int step = 0; step < cfg.num_steps; ++step) {

        std::vector<float> tile(cfg.expert_h * cfg.n_suffix);
        for (int64_t t = 0; t < cfg.n_suffix; ++t) {
            std::memcpy(tile.data() + t * cfg.expert_h,
                        time_host[step].data(), cfg.expert_h * sizeof(float));
        }
        ggml_backend_tensor_set(time_bcasts[step], tile.data(), 0, tile.size() * sizeof(float));
    }

    if (in.timing_detail == TimingDetail::PHASE) {

        ggml_cgraph * gf_pre = ggml_new_graph_custom(ctx,  4096,  false);
        for (int i = 0; i < cfg.n_layers; ++i) {
            ggml_build_forward_expand(gf_pre, k_cache[i]);
            ggml_build_forward_expand(gf_pre, v_cache[i]);
        }
        const auto t0 = clock::now();
        if (ggml_backend_graph_compute(m->backend, gf_pre) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla: ggml prefill compute failed\n");
            ggml_backend_buffer_free(compute_buf);
            ggml_free(ctx);
            return {};
        }
        m->stats.ms_prefill = ms_since(t0);

        for (int i = 0; i < cfg.n_layers; ++i) {
            ggml_backend_tensor_copy(k_cache[i], K_storage[i]);
            ggml_backend_tensor_copy(v_cache[i], V_storage[i]);
        }
    }

    {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx,  16384,  false);
        ggml_build_forward_expand(gf, x_t);
        const auto t0 = clock::now();
        if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla: ggml compute failed\n");
            ggml_backend_buffer_free(compute_buf);
            ggml_free(ctx);
            return {};
        }
        const float ms = ms_since(t0);
        if (in.timing_detail == TimingDetail::PHASE) {
            m->stats.ms_denoise = ms;
        }

        m->stats.ms_inference = m->stats.ms_prefill + ms;
    }

    std::vector<float> out(cfg.n_suffix * cfg.max_action_dim);
    ggml_backend_tensor_get(x_t, out.data(), 0, out.size() * sizeof(float));

    for (int64_t r = 0; r < cfg.n_suffix; ++r) {
        float * row = out.data() + r * cfg.max_action_dim;
        for (int64_t j = 0; j < cfg.real_action_dim; ++j) {
            row[j] = row[j] * (m->action_std[j] + cfg.norm_eps) + m->action_mean[j];
        }
    }

    ggml_backend_buffer_free(compute_buf);
    ggml_free(ctx);

    m->stats.ms_total = std::chrono::duration<float, std::milli>(
        clock::now() - t_total_begin).count();
    return out;
}
}

std::vector<float> SmolVLAModelArch::predict(const Inputs& in) {
    return predict_impl(this, in);
}

std::unique_ptr<ModelArchBase> smolvla_create(const std::string& mmproj_path,
                                              const std::string& ckpt_path,
                                              const std::string& config_path) {
    SmolVLAModelArch* raw = smolvla_load_impl(mmproj_path, ckpt_path, config_path);
    if (!raw) return nullptr;
    return std::unique_ptr<ModelArchBase>(raw);
}

}
