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
 * @file engine.h
 * @brief Chat / image-grounded text generation built on llama.cpp.
 *
 * The @ref vlm namespace is independent of the @c vla action-prediction
 * pipeline: it exposes a small "load model, send messages + images, get
 * text" surface used by @c vlm-server and the example chat client. The
 * implementation owns the underlying llama.cpp model, KV cache, and
 * (when provided) the mmproj vision tower.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vlm {

/**
 * @brief One message in a chat transcript.
 *
 * @c role follows the OpenAI convention: @c "system", @c "user",
 * @c "assistant". @c content is plain UTF-8 text; image attachments are
 * passed separately via the @c images argument to @ref Engine::chat.
 */
struct Message {
    std::string role;     ///< Speaker role.
    std::string content;  ///< Message body (plain text).
};

/**
 * @brief Decoded image in host memory, interleaved 8-bit RGB.
 */
struct Image {
    uint32_t             width  = 0; ///< Image width in pixels.
    uint32_t             height = 0; ///< Image height in pixels.
    std::vector<uint8_t> rgb;        ///< @c width*height*3 bytes, row-major.
};

/**
 * @brief Sampling knobs for text generation.
 */
struct SamplingParams {
    float    temperature = 0.8f;        ///< Softmax temperature.
    float    top_p       = 0.95f;       ///< Nucleus sampling cumulative mass.
    int32_t  top_k       = 40;          ///< Top-k truncation.
    uint32_t seed        = 0xFFFFFFFFu; ///< RNG seed; 0xFFFFFFFF = random.
    int32_t  max_tokens  = 256;         ///< Hard cap on decoded tokens.
};

/**
 * @brief Load-time configuration for @ref Engine::load.
 */
struct LoadParams {
    std::string model_path;             ///< Path to the LM GGUF.
    std::string mmproj_path;            ///< Path to the vision-tower GGUF (optional).
    int32_t     n_ctx          = 4096;  ///< KV-cache context length.
    int32_t     n_gpu_layers   = 999;   ///< Layers offloaded to GPU (999 = all).
    bool        mmproj_use_gpu = true;  ///< Offload the mmproj to GPU.
    int32_t     n_threads      = 0;     ///< CPU thread count (0 = auto).
};

/**
 * @brief Outcome of one @ref Engine::chat call.
 *
 * On failure @ref error is set and the other fields are unspecified.
 */
struct ChatResult {
    std::string text;                   ///< Generated assistant message.
    std::string finish_reason;          ///< "stop", "length", or backend code.
    int32_t     prompt_tokens     = 0;  ///< Tokens fed to the model.
    int32_t     completion_tokens = 0;  ///< Tokens generated.
    float       ms_prefill        = 0.f;///< Prompt-prefill wall time (ms).
    float       ms_decode         = 0.f;///< Token-decode wall time (ms).
    std::string error;                  ///< Empty on success.
};

/**
 * @brief Streaming callback. Return @c false to abort generation early.
 * @param piece The newly emitted text fragment (already detokenised).
 */
using TokenCallback = std::function<bool(const std::string& piece)>;

/**
 * @brief Thread-unsafe VLM engine wrapping a llama.cpp context.
 *
 * One @ref Engine owns one model and is intended to be driven by a single
 * caller at a time. @ref load must be called exactly once before
 * @ref chat. Copy and assignment are disabled.
 */
class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    /**
     * @brief Load the LM (and optional vision tower) into memory.
     * @param params Paths and runtime knobs.
     * @return @c true on success; @c false if any required file is missing
     *         or the backend reports a load error.
     */
    bool load(const LoadParams& params);

    /// @return @c true once @ref load has succeeded.
    bool loaded() const;

    /**
     * @brief Decode an image file (jpg/png/...) into @p out.
     * @param path Path to the encoded image.
     * @param[out] out Decoded RGB image.
     * @return @c true on success; @c false on decode error.
     */
    bool decode_image_file(const std::string& path, Image& out) const;

    /**
     * @brief Decode an in-memory encoded image into @p out.
     * @param data Pointer to the encoded byte stream.
     * @param len  Length of @p data in bytes.
     * @param[out] out Decoded RGB image.
     * @return @c true on success; @c false on decode error.
     */
    bool decode_image_buf(const uint8_t* data, size_t len, Image& out) const;

    /**
     * @brief Run one chat completion.
     *
     * The conversation is materialised from @p messages; @p images are
     * attached as multimodal inputs to the latest @c user message in the
     * order they appear. When @p on_token is supplied it is invoked for
     * every detokenised fragment as it is produced.
     *
     * @param messages Chat transcript.
     * @param images   Image attachments for the latest user turn.
     * @param sampling Sampling knobs.
     * @param on_token Optional streaming callback.
     * @return Final result, including the full generated text.
     */
    ChatResult chat(const std::vector<Message>& messages,
                    const std::vector<Image>&   images,
                    const SamplingParams&       sampling,
                    const TokenCallback&        on_token = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
