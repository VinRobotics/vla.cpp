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

// Cosmos3-Edge's prompt and action decode against reference data made with lerobot and Hugging Face's
// tokenizer (tests/py/make_cosmos3_edge_fixtures.py):
//   * rot6d -> LIBERO command, always;
//   * caption, chat template and BPE ids, when a Cosmos3-Edge GGUF made with --policy-dir is given
//     (argv[2] or VLA_COSMOS3_EDGE_GGUF); otherwise that part is skipped.
//
//   test_cosmos3_edge_prompt <tests/data> [model.gguf]

#include "models/cosmos3_edge_prompt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace vla;

namespace {

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::string              cur;
    for (char c : s) {
        if (c == sep) { out.push_back(cur); cur.clear(); } else { cur += c; }
    }
    out.push_back(cur);
    return out;
}

std::string unhex(const std::string & h) {
    std::string o;
    for (size_t i = 0; i + 1 < h.size(); i += 2)
        o += (char) std::strtol(h.substr(i, 2).c_str(), nullptr, 16);
    return o;
}

std::vector<int32_t> ints(const std::string & s) {
    std::vector<int32_t> v;
    std::istringstream   is(s);
    int                  x;
    while (is >> x)
        v.push_back(x);
    return v;
}

std::vector<float> floats(const std::string & s) {
    std::vector<float> v;
    std::istringstream is(s);
    float              x;
    while (is >> x)
        v.push_back(x);
    return v;
}

std::vector<std::string> lines(const std::string & path) {
    std::ifstream f(path);
    std::vector<std::string> out;
    for (std::string l; std::getline(f, l);)
        if (!l.empty())
            out.push_back(l);
    return out;
}

int fails = 0;

void check(bool ok, const char * what, const std::string & detail = "") {
    if (!ok) {
        fails++;
        std::fprintf(stderr, "FAIL %s %s\n", what, detail.c_str());
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <tests/data> [model.gguf]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];

    // ---- rot6d -> LIBERO command --------------------------------------------------------------------
    {
        const auto rows = lines(dir + "/cosmos3_edge_rot6d.tsv");
        double     worst = 0;
        for (const auto & l : rows) {
            const auto cols = split(l, '\t');
            const auto in = floats(cols.at(0)), want = floats(cols.at(1));
            float      got[7];
            cosmos3_libero_action(in.data(), got);
            for (int i = 0; i < 7; i++)
                worst = std::fmax(worst, std::fabs((double) got[i] - (double) want[(size_t) i]));
        }
        std::printf("rot6d: %zu rows, max |d| %.2e\n", rows.size(), worst);
        check(!rows.empty() && worst < 3e-6, "rot6d decode", "max |d| " + std::to_string(worst));
    }

    // ---- caption, chat template, BPE -----------------------------------------------------------------
    const char * gguf_path = argc > 2 ? argv[2] : std::getenv("VLA_COSMOS3_EDGE_GGUF");
    if (!gguf_path || !*gguf_path) {
        std::printf("prompt: skipped (no GGUF: pass one, or set VLA_COSMOS3_EDGE_GGUF)\n");
        return fails ? 1 : 0;
    }
    gguf_reader g("cosmos3_edge");
    if (!g.open(gguf_path)) {
        std::fprintf(stderr, "cannot open %s\n", gguf_path);
        return 2;
    }
    Cosmos3Prompt p;
    if (!p.load(g))
        return 2;

    {
        const auto rows = lines(dir + "/cosmos3_edge_prompts.tsv");
        int        bad  = 0;
        for (const auto & l : rows) {
            const auto cols = split(l, '\t');
            const std::string task = unhex(cols.at(0)), want_caption = unhex(cols.at(1));
            const auto        want_ids = ints(cols.at(2));
            std::vector<int32_t> ids;
            std::string          err;
            const bool ok_tok = p.tokens(task, ids, err);
            const std::string cap = p.caption(task);
            const bool same_cap = cap == want_caption || (want_caption.empty() && cap.empty());
            if (!ok_tok || !same_cap || ids != want_ids) {
                bad++;
                std::fprintf(stderr, "FAIL prompt for task \"%s\": tokenized %d, caption %s, ids %s %s\n", task.c_str(),
                             (int) ok_tok, same_cap ? "same" : "DIFFERENT", ids == want_ids ? "same" : "DIFFERENT", err.c_str());
            }
        }
        std::printf("prompts: %zu tasks, %d differ (caption text and ids)\n", rows.size(), bad);
        check(!rows.empty() && bad == 0, "prompts");
    }
    {
        const auto rows = lines(dir + "/cosmos3_edge_bpe.tsv");
        int        bad  = 0;
        for (const auto & l : rows) {
            const auto cols = split(l, '\t');
            const std::string text = unhex(cols.at(0));
            std::vector<int32_t> ids;
            std::string          err;
            if (!p.encode(text, ids, err) || ids != ints(cols.at(1))) {
                bad++;
                if (bad <= 3)
                    std::fprintf(stderr, "FAIL bpe text of %zu bytes %s\n", text.size(), err.c_str());
            }
        }
        std::printf("bpe: %zu texts, %d differ\n", rows.size(), bad);
        check(!rows.empty() && bad == 0, "bpe");
    }
    // Text this tokenizer must refuse rather than get wrong.
    {
        std::vector<int32_t> ids;
        std::string          err;
        check(!p.tokens("caf\xc3\xa9 au lait", ids, err), "non-ASCII task is refused");
    }
    std::printf(fails ? "FAILED\n" : "OK\n");
    return fails ? 1 : 0;
}
