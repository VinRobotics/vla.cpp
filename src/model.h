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

/**
 * @file model.h
 * @brief Public C++ API of the vla.cpp inference engine.
 *
 * A @ref vla::Model is the opaque handle returned by @ref vla::model_load.
 * Callers construct an @ref vla::Inputs struct (images, language tokens,
 * proprioception, noise), pass it to @ref vla::predict, and receive a flat
 * vector of normalised actions. @ref vla::model_config exposes the resolved
 * hyper-parameters needed to interpret that vector, and @ref vla::last_stats
 * reports per-phase timings for the most recent call.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vla {

/**
 * @brief Resolved hyper-parameters of a loaded model.
 *
 * Populated from the GGUF metadata at load time. Sequence counts are in
 * tokens; @c hidden / @c expert_h / @c intermediate refer to model widths.
 * The action chunk returned by @ref predict has shape
 * @c [num_steps, max_action_dim] in row-major order; only the first
 * @c real_action_dim columns are valid, the rest are zero padding.
 */
struct Config {
    int64_t n_img;            ///< Image-token count fed to the LM.
    int64_t n_lang;           ///< Language-token capacity.

    int64_t n_state;          ///< Proprioception-token count.
    int64_t n_prefix;         ///< Prefix length (vision + lang + state).
    int64_t n_suffix;         ///< Suffix length (action queries).
    int64_t n_full;           ///< Combined sequence length seen by attention.

    int64_t hidden;           ///< LM hidden width.
    int64_t expert_h;         ///< Action-expert hidden width.
    int64_t intermediate;     ///< LM FFN inner width.
    int64_t expert_inter;     ///< Action-expert FFN inner width.
    int64_t n_q_heads;        ///< Number of attention query heads.
    int64_t n_kv_heads;       ///< Number of attention KV heads (GQA).
    int64_t head_dim;         ///< Per-head dimension.
    int64_t q_full_dim;       ///< Flattened query projection width.
    int64_t kv_full_dim;      ///< Flattened key/value projection width.
    int64_t n_layers;         ///< Transformer layer count.
    int     self_attn_every_n;///< Cadence of self-attn vs. cross-attn layers.

    int64_t max_state_dim;    ///< Max proprio dim the checkpoint supports.
    int64_t max_action_dim;   ///< Max action dim the checkpoint supports.
    int64_t real_state_dim;   ///< Active proprio dim for this embodiment.
    int64_t real_action_dim;  ///< Active action dim for this embodiment.
    float   norm_eps;         ///< Epsilon for the LM normalisation layers.
    double  min_period;       ///< Flow-matching schedule lower bound.
    double  max_period;       ///< Flow-matching schedule upper bound.
    int     num_steps;        ///< Denoising / flow-matching step count.

    float   rms_eps;          ///< Epsilon for RMSNorm.

    int     rope_n_dims;      ///< RoPE rotation width (per head).
    int     rope_mode;        ///< RoPE variant (NeoX / GPT-J / etc).
    float   rope_freq_base;   ///< RoPE base frequency.
};

/// Opaque engine handle; created by @ref model_load and released by
/// @ref model_free.
struct Model;

/**
 * @brief Granularity of timing data accumulated during @ref predict.
 */
enum class TimingDetail {
    NONE,   ///< Only @c ms_total is populated.
    PHASE,  ///< Per-phase timings (vision, prefill, denoise, ...).
};

/**
 * @brief In-memory pixel format consumed by @ref predict.
 */
enum class PixelFormat {
    U8,         ///< 8-bit interleaved RGB.
    F32_RGB_01, ///< Float32 interleaved RGB in [0, 1].
};

/**
 * @brief Lightweight view over a single image in caller-owned memory.
 */
struct ImageView {
    const void*  data;                       ///< Pointer to the first pixel.
    int          w;                          ///< Image width in pixels.
    int          h;                          ///< Image height in pixels.
    PixelFormat  format = PixelFormat::U8;   ///< Pixel format of @ref data.
};

/**
 * @brief Inputs for one @ref predict call.
 *
 * All pointers are borrowed from the caller and must remain valid for the
 * duration of @ref predict. Pass either @ref images (and the engine will
 * run the vision tower) or pre-computed embeddings via
 * @ref precomputed_img_emb -- never both.
 */
struct Inputs {
    const ImageView* images;          ///< Camera views (host memory).
    int              n_images;        ///< Number of @ref images.

    /// Pre-computed image embeddings; bypasses the vision tower.
    const float*     precomputed_img_emb = nullptr;
    int              n_img_views         = 0; ///< Number of views in
                                              ///  @ref precomputed_img_emb.

    const int32_t*   lang_tokens;     ///< Tokenised language instruction.
    int              n_lang;          ///< Length of @ref lang_tokens.

    const float*     state;           ///< Proprioception, length @c max_state_dim
                                      ///  (pad @c real_state_dim..max with zeros).
    const float*     noise;           ///< Initial noise for the action expert.

    /// Optional per-token language attention mask. Only Evo-1 honors this; the
    /// other architectures derive their own mask and ignore it.
    const int32_t*   attention_mask   = nullptr;
    int              attention_mask_n = 0;     ///< Length of @ref attention_mask.

    TimingDetail     timing_detail    = TimingDetail::NONE;
};

/**
 * @brief Load a model from one (vision-baked) or two (mmproj + ckpt) GGUFs.
 *
 * The architecture is detected from the checkpoint via
 * @ref detect_arch_from_ckpt. Fails loud: a missing file, unknown
 * architecture, or shape mismatch aborts rather than returning a
 * half-initialised handle.
 *
 * @param mmproj_path Path to the vision-tower GGUF, or empty if the
 *        architecture bakes vision into @p ckpt_path.
 * @param ckpt_path   Path to the main checkpoint GGUF.
 * @param config_path Optional JSON override; empty to use bundled config.
 * @return Owning handle. Free with @ref model_free.
 */
Model* model_load(const std::string& mmproj_path, const std::string& ckpt_path,
                  const std::string& config_path = "");

/**
 * @brief Release a model handle returned by @ref model_load.
 * @param m Handle to free; may be @c nullptr.
 */
void   model_free(Model* m);

/**
 * @brief Resolved config of a loaded model.
 * @param m A handle from @ref model_load.
 * @return Reference valid for the lifetime of @p m.
 */
const Config& model_config(const Model* m);

/**
 * @brief Run one forward pass.
 *
 * Returns the predicted action chunk, normalised to the model's training
 * statistics. The caller is responsible for un-normalising into world
 * units. NaN/Inf inputs cause the call to abort.
 *
 * @param m  A handle from @ref model_load.
 * @param in Filled-in @ref Inputs struct.
 * @return Action chunk of length @c num_steps * max_action_dim, in
 *         row-major order (first @c real_action_dim columns valid).
 */
std::vector<float> predict(Model* m, const Inputs& in);

/**
 * @brief Wall-clock timings of the most recent @ref predict call.
 *
 * Times are in milliseconds. Per-phase fields are zero unless
 * @ref Inputs::timing_detail was @ref TimingDetail::PHASE.
 */
struct Stats {
    float ms_total      = 0.f; ///< Total time spent in @ref predict.
    float ms_vision     = 0.f; ///< Vision-tower forward pass.
    float ms_inference  = 0.f; ///< LM + action-expert combined.
    float ms_prefill    = 0.f; ///< Prefix prefill (vision + lang + state).
    float ms_denoise    = 0.f; ///< Flow-matching denoising loop.
};

/**
 * @brief Stats from the most recent call to @ref predict on @p m.
 * @param m A handle from @ref model_load.
 * @return Reference valid until the next @ref predict call.
 */
const Stats& last_stats(const Model* m);

}
