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

// Stable C ABI over src/model.h, so bindings do not have to match a C++ compiler
// or standard library.
//
// Ownership: vla_model_load pairs with vla_model_free, vla_predict pairs with
// vla_free_actions, and every pointer in vla_inputs is borrowed for the call.

#ifndef VLA_H
#define VLA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define VLA_API __declspec(dllexport)
#else
#  define VLA_API __attribute__((visibility("default")))
#endif

// Bumped on any incompatible change to the structs or functions below.
#define VLA_ABI_VERSION 1

typedef struct vla_model vla_model;

typedef enum {
    VLA_OK             =  0,
    VLA_ERR_ARG        = -1,  ///< Null handle or malformed vla_inputs.
    VLA_ERR_PREDICT    = -2,  ///< The engine returned no actions.
    VLA_ERR_EXCEPTION  = -3,  ///< A C++ exception was caught at the boundary.
} vla_status;

typedef enum {
    VLA_PIXEL_U8         = 0,  ///< 8-bit interleaved RGB.
    VLA_PIXEL_F32_RGB_01 = 1,  ///< Float32 interleaved RGB in [0, 1].
} vla_pixel_format;

typedef enum {
    VLA_TIMING_NONE  = 0,
    VLA_TIMING_PHASE = 1,
} vla_timing_detail;

/// Mirrors vla::Config. See src/model.h for the per-field meaning.
typedef struct {
    int64_t n_img;
    int64_t n_lang;
    int64_t n_state;
    int64_t n_prefix;
    int64_t n_suffix;
    int64_t n_full;

    int64_t hidden;
    int64_t expert_h;
    int64_t intermediate;
    int64_t expert_inter;
    int64_t n_q_heads;
    int64_t n_kv_heads;
    int64_t head_dim;
    int64_t q_full_dim;
    int64_t kv_full_dim;
    int64_t n_layers;
    int32_t self_attn_every_n;

    int64_t max_state_dim;
    int64_t max_action_dim;
    int64_t real_state_dim;
    int64_t real_action_dim;
    float   norm_eps;
    double  min_period;
    double  max_period;
    int32_t num_steps;

    float   rms_eps;
    int32_t rope_n_dims;
    int32_t rope_mode;
    float   rope_freq_base;

    /// Non-zero if vla_predict already applied the dataset statistics. Zero for
    /// the GR00T family and VLA-JEPA, whose callers un-normalise themselves.
    int32_t denormalized;
} vla_config;

typedef struct {
    const void * data;    ///< First pixel, caller-owned.
    int32_t      w;
    int32_t      h;
    int32_t      format;  ///< A vla_pixel_format value.
} vla_image;

typedef struct {
    const vla_image * images;
    int32_t           n_images;

    /// Optional, replaces the vision tower. Layout [n_img_views * n_img, hidden],
    /// already in the scale that arch's LM expects. Pass NULL to use images.
    const float *     precomputed_img_emb;
    int32_t           n_img_views;

    const int32_t *   lang_tokens;
    int32_t           n_lang;

    /// Length max_state_dim; pad real_state_dim..max with zeros.
    const float *     state;
    /// Length n_suffix * max_action_dim, or NULL to sample internally.
    const float *     noise;

    /// Only Evo-1 reads this; other archs derive their own mask.
    const int32_t *   attention_mask;
    int32_t           attention_mask_n;

    int32_t           timing_detail;  ///< A vla_timing_detail value.
} vla_inputs;

/// Milliseconds. Phase fields are zero unless timing_detail was VLA_TIMING_PHASE.
typedef struct {
    float ms_total;
    float ms_vision;
    float ms_inference;
    float ms_prefill;
    float ms_denoise;
} vla_stats;

/// VLA_ABI_VERSION of the loaded library, for a runtime compatibility check.
VLA_API int32_t vla_abi_version(void);

/// mmproj_path may be NULL or "" for archs that bake vision into the checkpoint.
/// config_path may be NULL. Returns NULL on failure.
VLA_API vla_model * vla_model_load(const char * mmproj_path,
                                   const char * ckpt_path,
                                   const char * config_path);

VLA_API void vla_model_free(vla_model * m);

/// Fills out with the resolved config. Returns a vla_status.
VLA_API int32_t vla_model_config(const vla_model * m, vla_config * out);

/// Runs one forward pass. On VLA_OK, *out_actions points to *out_n floats that
/// the caller releases with vla_free_actions. Row-major [num_steps or n_suffix,
/// max_action_dim]; only the first real_action_dim columns carry values.
VLA_API int32_t vla_predict(vla_model * m, const vla_inputs * in,
                            float ** out_actions, int64_t * out_n);

VLA_API void vla_free_actions(float * actions);

/// Timings of the most recent vla_predict on this handle.
VLA_API int32_t vla_last_stats(const vla_model * m, vla_stats * out);

#ifdef __cplusplus
}
#endif

#endif  // VLA_H
