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

// Cosmos3 Policy outside the graphs (src/models/cosmos3_policy.cpp): the prompt (lerobot's Cosmos3PromptProcessorStep and
// Cosmos3TokenizerProcessorStep) and the LIBERO action decode (Cosmos3LiberoActionProcessorStep).
//
//  * The prompt is the JSON caption of the task in the chat template of the understanding stream, then the
//    end-of-turn token and <|vision_start|>. The tokenizer is the model's byte-level BPE (131,072 tokens),
//    read from the GGUF (`tokenizer.ggml.*`, the keys llama.cpp uses). It handles ASCII text only: its
//    pre-tokenizer splits on Unicode letter classes, which are not implemented here, so other text is refused.
//  * rot6d -> axis-angle follows the lerobot step: the two columns and their cross product form a matrix that
//    is projected onto SO(3), then converted to a rotation vector, in double precision.

#pragma once

#include "gguf_reader.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vla {

struct Cosmos3Prompt {
    // Loads the tokenizer and the caption settings from a Cosmos3 Policy GGUF written with
    // `--policy-dir`. False, with a reason on stderr, when the file has none.
    bool load(const gguf_reader & g);

    // The ids of the prompt of `task` (what the policy feeds the understanding stream, unpadded). False,
    // with a reason in `err`, for text this tokenizer does not handle.
    bool tokens(const std::string & task, std::vector<int32_t> & ids, std::string & err) const;

    // The JSON caption (Cosmos3PromptProcessorStep.caption with 0 idle frames) and the full text the
    // tokenizer sees (chat template, end-of-turn and <|vision_start|> included); for tests and debugging.
    std::string caption(const std::string & task) const;
    std::string chat_text(const std::string & task) const;

    // Token ids of a text with special tokens recognised in it, as Hugging Face's tokenizer does.
    bool encode(const std::string & text, std::vector<int32_t> & ids, std::string & err) const;

private:
    bool bpe_segment(const std::string & text, std::vector<int32_t> & ids, std::string & err) const;

    std::vector<std::string>                     tokens_;      // id -> byte-level spelling
    std::unordered_map<std::string, int32_t>     vocab_;       // byte-level spelling -> id
    std::unordered_map<std::string, int32_t>     rank_;        // "left right" -> merge priority
    std::vector<std::pair<std::string, int32_t>> specials_;    // raw text -> id, longest first
    bool                                         special_first_[256] = {};  // bytes a special can start with
    std::string                                  byte_str_[256];  // byte -> its byte-level character (UTF-8)

    // Caption settings.
    std::string view_description_;
    std::string aspect_ratio_;
    double      fps_       = 20.0;
    int         chunk_     = 16;
    int         canvas_h_  = 192;
    int         canvas_w_  = 320;
    bool        use_state_ = false;
};

// The LIBERO command of one 10-wide model action [dpos(3), rot6d(6), gripper]: [dpos(3), axis-angle(3), gripper]
// with the gripper mapped from 1 open / 0 closed to -1 open / +1 closed.
void cosmos3_libero_action(const float * action10, float * out7);

}  // namespace vla
