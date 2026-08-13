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

#include "modules/prompt.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

namespace vla {

bool build_prompt(const char * arch, const Inputs & in, int64_t n_img,
                  int32_t image_token, int64_t max_seq, Prompt & out) {
    out = Prompt{};

    int64_t slots = 0;
    for (int j = 0; j < in.n_lang; ++j)
        if (in.lang_tokens[j] == image_token)
            ++slots;

    if (slots == n_img) {
        out.ids.assign(in.lang_tokens, in.lang_tokens+in.n_lang);
    } else if (slots == 0) {
        out.ids.reserve((size_t)(n_img+in.n_lang));
        for (int64_t i = 0; i < n_img; ++i)
            out.ids.push_back(image_token);
        for (int j = 0; j < in.n_lang; ++j)
            out.ids.push_back(in.lang_tokens[j]);
    } else {
        std::fprintf(stderr, "vla(%s): lang_tokens has %lld image-token slots but n_img=%lld; expected 0 or %lld\n",
                     arch, (long long) slots, (long long) n_img, (long long) n_img);
        return false;
    }

    const int64_t seq = out.len();
    if (seq > max_seq) {
        std::fprintf(stderr, "vla(%s): prompt too long (%lld > %lld)\n", arch, (long long) seq, (long long) max_seq);
        return false;
    }

    out.image_pos.reserve((size_t) n_img);
    out.text_pos.reserve((size_t)(seq-n_img));
    for (int64_t p = 0; p < seq; ++p) {
        if (out.ids[p] == image_token)
            out.image_pos.push_back((int32_t) p);
        else
            out.text_pos.push_back((int32_t) p);
    }
    return true;
}

bool fetch_embeds(const char * arch, gguf_reader & io, const Prompt & p,
                  const float * img_emb, int64_t hidden, std::vector<float> & out) {
    const int64_t seq = p.len();
    out.assign((size_t) seq*hidden, 0.0f);
    if (!io.fetch_rows_f32("token_embd.weight", p.ids, out.data(), hidden))
        return false;

    for (size_t k = 0; k < p.image_pos.size(); ++k)
        std::memcpy(out.data()+(size_t) p.image_pos[k]*hidden, img_emb+k*hidden, hidden*sizeof(float));

    (void) arch;
    return true;
}

void init_noise(const Inputs & in, size_t n, std::vector<float> & out) {
    out.assign(n, 0.0f);
    if (in.noise) {
        std::memcpy(out.data(), in.noise, n*sizeof(float));
        return;
    }

    std::mt19937 rng((uint32_t) std::chrono::steady_clock::now().time_since_epoch().count());
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : out)
        v = nd(rng);
}

}
