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

// A test-only C shim over Cosmos3-Edge's prompt and action decode, so a Python harness (the closed-loop
// LIBERO eval) can run the C++ code instead of lerobot's. Not part of libvla and not installed.

#include "models/cosmos3_edge_prompt.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>

extern "C" {

// The ids of the prompt of `task`, from the tokenizer in `gguf`. Returns their number, or -1 (the reason in
// `err`) when the GGUF has no tokenizer, the text is not supported or `cap` is too small.
int64_t cosmos3_shim_tokenize(const char * gguf, const char * task, int32_t * ids, int64_t cap, char * err, int64_t errcap) {
    static std::map<std::string, std::unique_ptr<vla::Cosmos3Prompt>> cache;
    auto fail = [&](const std::string & m) {
        std::snprintf(err, (size_t) errcap, "%s", m.c_str());
        return (int64_t) -1;
    };
    auto & slot = cache[gguf];
    if (!slot) {
        vla::gguf_reader g("cosmos3_edge");
        if (!g.open(gguf))
            return fail("cannot open the GGUF");
        auto p = std::make_unique<vla::Cosmos3Prompt>();
        if (!p->load(g))
            return fail("the GGUF has no tokenizer");
        slot = std::move(p);
    }
    std::vector<int32_t> v;
    std::string          e;
    if (!slot->tokens(task, v, e))
        return fail(e);
    if ((int64_t) v.size() > cap)
        return fail("buffer too small");
    std::memcpy(ids, v.data(), v.size() * sizeof(int32_t));
    return (int64_t) v.size();
}

// The LIBERO command of one 10-wide model action.
void cosmos3_shim_libero_action(const float * in10, float * out7) {
    vla::cosmos3_libero_action(in10, out7);
}

}  // extern "C"
