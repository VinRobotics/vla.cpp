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

#include "vlm/engine.h"

#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <climits>
#include <cstdio>

namespace vlm {

namespace {

void ensure_global_init() {
    static bool done = false;
    if (!done) {
        ggml_time_init();
        common_init();
        done = true;
    }
}
}

struct Engine::Impl {
    common_init_result_ptr    llama_init;
    llama_model *             model  = nullptr;
    llama_context *           lctx   = nullptr;
    const llama_vocab *       vocab  = nullptr;
    mtmd::context_ptr         vision;
    common_chat_templates_ptr tmpls;
    llama_batch               batch{};
    int32_t                   n_batch = 2048;
    bool                      batch_inited = false;
    bool                      use_jinja    = true;

    ~Impl() {
        if (batch_inited) {
            llama_batch_free(batch);
        }
    }
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

bool Engine::loaded() const { return impl_ && impl_->lctx != nullptr; }

bool Engine::load(const LoadParams & lp) {
    ensure_global_init();

    if (lp.model_path.empty() || lp.mmproj_path.empty()) {
        std::fprintf(stderr, "vlm: load requires both model_path and mmproj_path\n");
        return false;
    }

    common_params params;
    params.model.path     = lp.model_path;
    params.mmproj.path     = lp.mmproj_path;
    params.mmproj_use_gpu  = lp.mmproj_use_gpu;
    params.n_ctx           = lp.n_ctx;
    params.n_gpu_layers    = lp.n_gpu_layers;
    if (lp.n_threads > 0) {
        params.cpuparams.n_threads = lp.n_threads;
    }

    impl_->llama_init = common_init_from_params(params);
    impl_->model = impl_->llama_init->model();
    impl_->lctx  = impl_->llama_init->context();
    if (!impl_->model || !impl_->lctx) {
        std::fprintf(stderr, "vlm: common_init_from_params failed (model=%s)\n",
                     lp.model_path.c_str());
        return false;
    }
    impl_->vocab   = llama_model_get_vocab(impl_->model);
    impl_->n_batch = params.n_batch;

    if (!llama_model_chat_template(impl_->model, nullptr) && params.chat_template.empty()) {
        std::fprintf(stderr, "vlm: model has no chat template; pass one via the GGUF\n");
        return false;
    }
    impl_->tmpls     = common_chat_templates_init(impl_->model, params.chat_template);
    impl_->use_jinja = params.use_jinja;

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu         = lp.mmproj_use_gpu;
    mparams.print_timings   = false;
    mparams.n_threads       = params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn_type;
    mparams.warmup          = params.warmup;
    impl_->vision.reset(mtmd_init_from_file(lp.mmproj_path.c_str(), impl_->model, mparams));
    if (!impl_->vision.get()) {
        std::fprintf(stderr, "vlm: failed to load vision tower from %s\n", lp.mmproj_path.c_str());
        return false;
    }

    impl_->batch        = llama_batch_init(1, 0, 1);
    impl_->batch_inited = true;
    return true;
}

namespace {

bool bitmap_to_image(mtmd::bitmap & bmp, Image & out) {
    if (!bmp.ptr || mtmd_bitmap_is_audio(bmp.ptr.get())) {
        return false;
    }
    out.width  = bmp.nx();
    out.height = bmp.ny();
    out.rgb.assign(bmp.data(), bmp.data() + bmp.n_bytes());
    return true;
}
}

bool Engine::decode_image_file(const std::string & path, Image & out) const {
    if (!loaded()) {
        return false;
    }
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(impl_->vision.get(), path.c_str()));
    return bitmap_to_image(bmp, out);
}

bool Engine::decode_image_buf(const uint8_t * data, size_t len, Image & out) const {
    if (!loaded() || !data || len == 0) {
        return false;
    }
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(impl_->vision.get(), data, len));
    return bitmap_to_image(bmp, out);
}

ChatResult Engine::chat(const std::vector<Message> & messages,
                        const std::vector<Image> &   images,
                        const SamplingParams &       sampling,
                        const TokenCallback &        on_token) {
    ChatResult res;
    if (!loaded()) {
        res.finish_reason = "error";
        res.error         = "engine not loaded";
        return res;
    }
    if (messages.empty()) {
        res.finish_reason = "error";
        res.error         = "no messages";
        return res;
    }

    llama_memory_clear(llama_get_memory(impl_->lctx), true);
    llama_pos n_past = 0;
    std::vector<common_chat_msg> chat_history;

    common_params_sampling sp;
    sp.seed  = sampling.seed;
    sp.top_k = sampling.top_k;
    sp.top_p = sampling.top_p;
    sp.temp  = sampling.temperature;
    common_sampler * smpl = common_sampler_init(impl_->model, sp);
    if (!smpl) {
        res.finish_reason = "error";
        res.error         = "common_sampler_init failed";
        return res;
    }

    int img_msg_idx = -1;
    for (int i = (int) messages.size() - 1; i >= 0; --i) {
        if (messages[i].role == "user") { img_msg_idx = i; break; }
    }
    if (img_msg_idx < 0) {
        img_msg_idx = (int) messages.size() - 1;
    }

    const char * marker = mtmd_default_marker();
    const int64_t t_prefill_start = ggml_time_us();

    for (size_t i = 0; i < messages.size(); ++i) {
        common_chat_msg msg;
        msg.role    = messages[i].role;
        msg.content = messages[i].content;

        std::vector<mtmd::bitmap> bmps;
        if ((int) i == img_msg_idx && !images.empty()) {

            if (msg.content.find(marker) == std::string::npos) {
                std::string prefix;
                for (size_t k = 0; k < images.size(); ++k) prefix += marker;
                msg.content = prefix + msg.content;
            }
            for (const auto & im : images) {
                if (im.rgb.size() != (size_t) im.width * im.height * 3) {
                    res.finish_reason = "error";
                    res.error         = "image rgb size != w*h*3";
                    common_sampler_free(smpl);
                    return res;
                }
                bmps.emplace_back(mtmd_bitmap_init(im.width, im.height, im.rgb.data()));
            }
        }

        const bool add_special = chat_history.empty();
        const std::string formatted =
            common_chat_format_single(impl_->tmpls.get(), chat_history, msg,
                                      msg.role == "user", impl_->use_jinja);
        chat_history.push_back(msg);

        mtmd_input_text text;
        text.text          = formatted.c_str();
        text.add_special   = add_special;
        text.parse_special = true;

        std::vector<const mtmd_bitmap *> bmp_ptrs(bmps.size());
        for (size_t k = 0; k < bmps.size(); ++k) bmp_ptrs[k] = bmps[k].ptr.get();

        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        if (mtmd_tokenize(impl_->vision.get(), chunks.ptr.get(), &text,
                          bmp_ptrs.data(), bmp_ptrs.size()) != 0) {
            res.finish_reason = "error";
            res.error         = "mtmd_tokenize failed";
            common_sampler_free(smpl);
            return res;
        }

        llama_pos new_n_past = n_past;
        if (mtmd_helper_eval_chunks(impl_->vision.get(), impl_->lctx, chunks.ptr.get(),
                                    n_past, 0, impl_->n_batch,
                                    true, &new_n_past) != 0) {
            res.finish_reason = "error";
            res.error         = "mtmd_helper_eval_chunks failed";
            common_sampler_free(smpl);
            return res;
        }
        n_past = new_n_past;
    }
    res.prompt_tokens = (int32_t) n_past;
    res.ms_prefill    = (ggml_time_us() - t_prefill_start) / 1000.0f;

    const int n_predict = sampling.max_tokens <= 0 ? INT_MAX : sampling.max_tokens;
    std::vector<llama_token> generated;
    const int64_t t_decode_start = ggml_time_us();
    res.finish_reason = "length";
    for (int i = 0; i < n_predict; ++i) {
        const llama_token tok = common_sampler_sample(smpl, impl_->lctx, -1);
        common_sampler_accept(smpl, tok, true);
        generated.push_back(tok);

        if (llama_vocab_is_eog(impl_->vocab, tok)) {
            res.finish_reason = "stop";
            break;
        }

        const std::string piece = common_token_to_piece(impl_->lctx, tok);
        res.text += piece;
        if (on_token && !on_token(piece)) {
            res.finish_reason = "cancelled";
            break;
        }

        common_batch_clear(impl_->batch);
        common_batch_add(impl_->batch, tok, n_past++, {0}, true);
        if (llama_decode(impl_->lctx, impl_->batch) != 0) {
            res.finish_reason = "error";
            res.error         = "llama_decode failed";
            break;
        }
    }
    res.completion_tokens = (int32_t) generated.size();
    res.ms_decode         = (ggml_time_us() - t_decode_start) / 1000.0f;

    if (res.finish_reason != "error") {
        res.text = common_detokenize(impl_->lctx, generated, false);
    }

    common_sampler_free(smpl);
    return res;
}

}
