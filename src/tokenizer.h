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

#pragma once

#include "gguf_reader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vla {

bool spm_encode(const gguf_reader& g, const char * arch, const std::string& text,
                std::vector<int32_t>& ids, bool add_bos = false);

bool has_spm_tokenizer(const std::string& ckpt_path, const char * arch);

bool prompt_text(const char * arch, const std::string& text, const std::vector<float>& state,
                 const std::vector<float>& q01, const std::vector<float>& q99, std::string& out);

bool tokenize_prompt(const std::string& ckpt_path, const char * arch, const std::string& text,
                     const std::vector<float>& state, std::vector<int32_t>& ids,
                     std::vector<int32_t>& attention_mask);

}  // namespace vla
