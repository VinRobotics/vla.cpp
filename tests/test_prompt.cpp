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

#include "tokenizer.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace vla;

namespace {

const std::vector<float> kQ01   = {-0.5f, -0.3f, 0.0f, 1.0f, -1.0f, 0.2f, -0.04f, 0.0f};
const std::vector<float> kQ99   = {0.5f, 0.3f, 1.2f, 3.0f, 1.0f, 0.2f, 0.04f, 0.01f};
const std::vector<float> kState = {0.1f, -0.9f, 0.6f, 3.5f, -1.0f, 0.2f, 0.0390625f, 0.00499f};
const char * kPi05Text = "  Pick up the black_bowl and place it on the plate  ";

std::string prompt(const char * arch, const std::string & text, const std::vector<float> & state = {}) {
    std::string out;
    assert(prompt_text(arch, text, state, kQ01, kQ99, out));
    return out;
}

std::vector<int32_t> tokenize(const char * spm_path, const char * arch, uint32_t max_length,
                              const std::string & text, const std::vector<float> & state) {
    std::ifstream f(spm_path, std::ios::binary);
    const std::vector<char> spm((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    assert(!spm.empty());

    const std::string a = arch;
    gguf_context * ctx = gguf_init_empty();
    gguf_set_arr_data(ctx, (a + ".tokenizer.spm_model").c_str(), GGUF_TYPE_UINT8, spm.data(), spm.size());
    if (max_length)
        gguf_set_val_u32(ctx, (a + ".tokenizer_max_length").c_str(), max_length);
    gguf_set_arr_data(ctx, "pi05.state.q01", GGUF_TYPE_FLOAT32, kQ01.data(), kQ01.size());
    gguf_set_arr_data(ctx, "pi05.state.q99", GGUF_TYPE_FLOAT32, kQ99.data(), kQ99.size());
    const std::string path = (std::filesystem::temp_directory_path() / ("vla_test_prompt_" + a + ".gguf")).string();
    assert(gguf_write_to_file(ctx, path.c_str(), true));
    gguf_free(ctx);

    std::vector<int32_t> ids, attn;
    assert(tokenize_prompt(path, arch, text, state, ids, attn));
    std::filesystem::remove(path);
    return ids;
}

}  // namespace

int main(int argc, char ** argv) {
    assert(prompt("pi0", "pick up the bowl") == "pick up the bowl\n");
    assert(prompt("pi0", "pick up the bowl\n") == "pick up the bowl\n");
    assert(prompt("openvla_oft", "Put Both the SOUP in the basket") ==
           "In: What action should the robot take to put both the soup in the basket?\nOut:");
    assert(prompt("pi05", kPi05Text, kState) ==
           "Task: Pick up the black bowl and place it on the plate, "
           "State: 153 -1 128 255 0 255 253 127;\nAction: ");

    std::string out;
    assert(!prompt_text("pi05", "x", {0.0f}, kQ01, kQ99, out));
    assert(!prompt_text("pi05", "x", kState, {}, {}, out));
    assert(!prompt_text("smolvla", "x", {}, {}, {}, out));

    if (argc < 3) {
        std::printf("test_prompt: templates OK; pass paligemma and llama-2 tokenizer.model to check ids\n");
        return 0;
    }
    const std::vector<int32_t> pi0 = {2, 18075, 908, 573, 14581, 108};
    assert(tokenize(argv[1], "pi0", 48, "pick up the bowl", {}) == pi0);

    std::string long_text = "pick up the bowl";
    for (int i=0; i<19; ++i)
        long_text += " pick up the bowl";
    const std::vector<int32_t> pi0_long = tokenize(argv[1], "pi0", 48, long_text, {});
    assert(pi0_long.size() == 48 && pi0_long[47] == 573);

    const std::vector<int32_t> pi05 = {
        2, 7071, 235292, 17350, 908, 573, 2656, 14581, 578, 2040, 665, 611, 573, 8811, 235269,
        3040, 235292, 235248, 235274, 235308, 235304, 728, 235274, 235248, 235274, 235284, 235321,
        235248, 235284, 235308, 235308, 235248, 235276, 235248, 235284, 235308, 235308, 235248,
        235284, 235308, 235304, 235248, 235274, 235284, 235324, 235289, 108, 4022, 235292, 235248};
    assert(tokenize(argv[1], "pi05", 200, kPi05Text, kState) == pi05);

    const std::vector<int32_t> oft = {
        1, 512, 29901, 1724, 3158, 881, 278, 19964, 2125, 304, 1925, 1716, 278, 22968, 22300, 322,
        278, 6454, 1219, 12507, 346, 297, 278, 25972, 29973, 13, 3744, 29901, 29871};
    assert(tokenize(argv[2], "openvla_oft", 0,
                    "Put Both the alphabet soup and the tomato sauce in the basket", {}) == oft);

    std::printf("test_prompt: templates and tokenizer ids OK\n");
    return 0;
}
