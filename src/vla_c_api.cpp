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

// Type conversion at the C boundary, and a barrier for C++ exceptions.

#include "vla.h"

#include "model.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {

// vla_model is opaque to C, so it can just wrap the C++ handle.
struct vla_model_impl {
    vla::Model * m = nullptr;
};

vla::PixelFormat to_pixel_format(int32_t f) {
    return f == VLA_PIXEL_F32_RGB_01 ? vla::PixelFormat::F32_RGB_01
                                     : vla::PixelFormat::U8;
}

}  // namespace

struct vla_model : vla_model_impl {};

extern "C" {

int32_t vla_abi_version(void) {
    return VLA_ABI_VERSION;
}

vla_model * vla_model_load(const char * mmproj_path, const char * ckpt_path,
                           const char * config_path) {
    if (!ckpt_path)
        return nullptr;
    try {
        vla::Model * m = vla::model_load(mmproj_path ? mmproj_path : "",
                                         ckpt_path,
                                         config_path ? config_path : "");
        if (!m)
            return nullptr;
        // Owns the engine until the handle exists, so a throwing new does not
        // strand the whole model.
        std::unique_ptr<vla::Model, void (*)(vla::Model *)> guard(m, vla::model_free);
        auto * h = new vla_model();
        h->m = guard.release();
        return h;
    } catch (...) {
        return nullptr;
    }
}

void vla_model_free(vla_model * h) {
    if (!h)
        return;
    vla::model_free(h->m);
    delete h;
}

int32_t vla_model_config(const vla_model * h, vla_config * out) {
    if (!h || !h->m || !out)
        return VLA_ERR_ARG;
    try {
        const vla::Config & c = vla::model_config(h->m);
        *out = vla_config{};
        out->n_img             = c.n_img;
        out->n_lang            = c.n_lang;
        out->n_state           = c.n_state;
        out->n_prefix          = c.n_prefix;
        out->n_suffix          = c.n_suffix;
        out->n_full            = c.n_full;
        out->hidden            = c.hidden;
        out->expert_h          = c.expert_h;
        out->intermediate      = c.intermediate;
        out->expert_inter      = c.expert_inter;
        out->n_q_heads         = c.n_q_heads;
        out->n_kv_heads        = c.n_kv_heads;
        out->head_dim          = c.head_dim;
        out->q_full_dim        = c.q_full_dim;
        out->kv_full_dim       = c.kv_full_dim;
        out->n_layers          = c.n_layers;
        out->self_attn_every_n = c.self_attn_every_n;
        out->max_state_dim     = c.max_state_dim;
        out->max_action_dim    = c.max_action_dim;
        out->real_state_dim    = c.real_state_dim;
        out->real_action_dim   = c.real_action_dim;
        out->norm_eps          = c.norm_eps;
        out->min_period        = c.min_period;
        out->max_period        = c.max_period;
        out->num_steps         = c.num_steps;
        out->rms_eps           = c.rms_eps;
        out->rope_n_dims       = c.rope_n_dims;
        out->rope_mode         = c.rope_mode;
        out->rope_freq_base    = c.rope_freq_base;
        out->denormalized      = c.denormalized ? 1 : 0;
        return VLA_OK;
    } catch (...) {
        return VLA_ERR_EXCEPTION;
    }
}

int32_t vla_predict(vla_model * h, const vla_inputs * in,
                    float ** out_actions, int64_t * out_n) {
    if (!h || !h->m || !in || !out_actions || !out_n)
        return VLA_ERR_ARG;
    *out_actions = nullptr;
    *out_n = 0;
    if (in->n_images < 0 || in->n_lang < 0 || in->n_img_views < 0)
        return VLA_ERR_ARG;
    if (in->n_images > 0 && !in->images)
        return VLA_ERR_ARG;
    if (in->n_lang  > 0 && !in->lang_tokens)
        return VLA_ERR_ARG;

    try {
        std::vector<vla::ImageView> views((size_t) (in->n_images > 0 ? in->n_images : 0));
        for (size_t i = 0; i < views.size(); ++i) {
            views[i] = vla::ImageView{ in->images[i].data,
                                       in->images[i].w,
                                       in->images[i].h,
                                       to_pixel_format(in->images[i].format) };
        }

        vla::Inputs ci{};
        ci.images              = views.empty() ? nullptr : views.data();
        ci.n_images            = (int) views.size();
        ci.precomputed_img_emb = in->precomputed_img_emb;
        ci.n_img_views         = in->n_img_views;
        ci.lang_tokens         = in->lang_tokens;
        ci.n_lang              = in->n_lang;
        ci.state               = in->state;
        ci.noise               = in->noise;
        ci.attention_mask      = in->attention_mask;
        ci.attention_mask_n    = in->attention_mask_n;
        ci.timing_detail       = in->timing_detail == VLA_TIMING_PHASE
                                     ? vla::TimingDetail::PHASE
                                     : vla::TimingDetail::NONE;

        const std::vector<float> act = vla::predict(h->m, ci);
        if (act.empty())
            return VLA_ERR_PREDICT;

        // malloc pairs with vla_free_actions, which callers may replace.
        float * buf = (float *) std::malloc(act.size()*sizeof(float));
        if (!buf)
            return VLA_ERR_EXCEPTION;
        std::memcpy(buf, act.data(), act.size()*sizeof(float));
        *out_actions = buf;
        *out_n = (int64_t) act.size();
        return VLA_OK;
    } catch (...) {
        return VLA_ERR_EXCEPTION;
    }
}

void vla_free_actions(float * actions) {
    std::free(actions);
}

int32_t vla_last_stats(const vla_model * h, vla_stats * out) {
    if (!h || !h->m || !out)
        return VLA_ERR_ARG;
    try {
        const vla::Stats & s = vla::last_stats(h->m);
        out->ms_total     = s.ms_total;
        out->ms_vision    = s.ms_vision;
        out->ms_inference = s.ms_inference;
        out->ms_prefill   = s.ms_prefill;
        out->ms_denoise   = s.ms_denoise;
        return VLA_OK;
    } catch (...) {
        return VLA_ERR_EXCEPTION;
    }
}

}  // extern "C"
