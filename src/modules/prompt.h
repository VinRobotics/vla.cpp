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

// Token-sequence assembly for the archs whose backbone takes one interleaved
// image/text stream: build the id list, note where the image slots landed, then
// swap the tower's embeddings into those rows.

#pragma once

#include "gguf_reader.h"
#include "model.h"

#include <cstdint>
#include <vector>

namespace vla {

struct Prompt {
    std::vector<int32_t> ids;
    std::vector<int32_t> image_pos;
    std::vector<int32_t> text_pos;

    int64_t len()     const { return (int64_t) ids.size(); }
    int64_t n_text()  const { return (int64_t) text_pos.size(); }
};

// Accepts a caller-supplied stream that already carries exactly n_img image
// placeholders, or one with none, in which case the placeholders are prepended.
// Any other count is a mismatch between the tokenizer and the tower.
bool build_prompt(const char * arch, const Inputs & in, int64_t n_img,
                  int32_t image_token, int64_t max_seq, Prompt & out);

// Embedding-table rows for the prompt, with the image rows overwritten by the
// tower output.
bool fetch_embeds(const char * arch, gguf_reader & io, const Prompt & p,
                  const float * img_emb, int64_t hidden, std::vector<float> & out);

// The request's noise if it carried any, else a fresh N(0,1) draw.
void init_noise(const Inputs & in, size_t n, std::vector<float> & out);

}
