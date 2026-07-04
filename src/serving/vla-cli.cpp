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

// One-shot action prediction from the command line. Loads a model, decodes an
// image plus an already-tokenized instruction, runs one predict(), and prints
// the action chunk. No server, no simulator. Tokenization stays in the Python
// client, so language is passed as token ids here.
//
//   vla-cli [--mmproj m.gguf] --ckpt c.gguf --image img.jpg [--image img2.jpg]
//           --tokens id,id,... [--state f,f,...] [--pretty]

#include "model.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
// stb pulls in its full implementation here; keep its unused-function noise out
// of our -Wall -Wextra output.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "stb_image.h"
#pragma GCC diagnostic pop

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vla;

namespace {

std::vector<int32_t> parse_ints(const std::string & s) {
    std::vector<int32_t> v;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ')) ++i;
        if (i >= s.size()) break;
        char * e = nullptr;
        long x = std::strtol(s.c_str() + i, &e, 10);
        if (e == s.c_str() + i) { ++i; continue; }
        v.push_back((int32_t) x);
        i = (size_t) (e - s.c_str());
    }
    return v;
}

std::vector<float> parse_floats(const std::string & s) {
    std::vector<float> v;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ')) ++i;
        if (i >= s.size()) break;
        char * e = nullptr;
        float x = std::strtof(s.c_str() + i, &e);
        if (e == s.c_str() + i) { ++i; continue; }
        v.push_back(x);
        i = (size_t) (e - s.c_str());
    }
    return v;
}

// Decode an image file to interleaved RGB8. buf must outlive the ImageView.
bool load_image(const char * path, std::vector<uint8_t> & buf, int & w, int & h) {
    int ch = 0;
    unsigned char * px = stbi_load(path, &w, &h, &ch, 3);
    if (!px) {
        std::fprintf(stderr, "vla-cli: cannot load image %s: %s\n", path, stbi_failure_reason());
        return false;
    }
    buf.assign(px, px + size_t(3) * w * h);
    stbi_image_free(px);
    return true;
}

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s [--mmproj m.gguf] --ckpt c.gguf --image img.jpg [--image ...]\n"
        "          --tokens id,id,... [--state f,f,...] [--pretty]\n"
        "  --mmproj   vision-tower GGUF (SmolVLA/pi0/pi0.5); omit for baked-vision archs\n"
        "  --ckpt     model checkpoint GGUF\n"
        "  --image    image file, repeat for multi-view (decoded via stb_image)\n"
        "  --tokens   language token ids, comma-separated (tokenize in the client)\n"
        "  --state    proprioception floats, comma-separated (default zeros)\n"
        "  --pretty   print one action row (max_action_dim values) per line\n",
        prog);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string mmproj, ckpt, tokens_s, state_s;
    std::vector<std::string> image_paths;
    bool pretty = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "vla-cli: %s needs a value\n", name); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--mmproj")  mmproj = need("--mmproj");
        else if (a == "--ckpt")    ckpt = need("--ckpt");
        else if (a == "--image")   image_paths.push_back(need("--image"));
        else if (a == "--tokens")  tokens_s = need("--tokens");
        else if (a == "--state")   state_s = need("--state");
        else if (a == "--pretty")  pretty = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "vla-cli: unknown argument %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (ckpt.empty() || image_paths.empty() || tokens_s.empty()) { usage(argv[0]); return 1; }

    Model * m = model_load(mmproj, ckpt, "");
    if (!m) { std::fprintf(stderr, "vla-cli: model_load failed\n"); return 1; }
    const Config & cfg = model_config(m);

    std::vector<std::vector<uint8_t>> imgbuf(image_paths.size());
    std::vector<ImageView>            views(image_paths.size());
    for (size_t v = 0; v < image_paths.size(); ++v) {
        int w = 0, h = 0;
        if (!load_image(image_paths[v].c_str(), imgbuf[v], w, h)) { model_free(m); return 1; }
        views[v] = ImageView{ imgbuf[v].data(), w, h, PixelFormat::U8 };
    }

    std::vector<int32_t> lang  = parse_ints(tokens_s);
    std::vector<float>   state = parse_floats(state_s);
    state.resize((size_t) cfg.max_state_dim, 0.0f);
    if (lang.empty()) { std::fprintf(stderr, "vla-cli: --tokens parsed to nothing\n"); model_free(m); return 1; }

    Inputs in{};
    in.images      = views.data();
    in.n_images    = (int) views.size();
    in.lang_tokens = lang.data();
    in.n_lang      = (int) lang.size();
    in.state       = state.data();
    in.noise       = nullptr;  // predict() samples N(0,1) when omitted

    std::vector<float> act = predict(m, in);
    if (act.empty()) { std::fprintf(stderr, "vla-cli: predict failed\n"); model_free(m); return 2; }

    const int64_t cols = cfg.max_action_dim > 0 ? cfg.max_action_dim : 1;
    if (pretty) {
        for (size_t i = 0; i < act.size(); ++i)
            std::printf("%.6g%c", act[i], ((int64_t) (i + 1) % cols == 0) ? '\n' : ' ');
    } else {
        std::printf("action_len=%zu\n", act.size());
        for (float x : act) std::printf("%.9g\n", x);
    }
    std::fflush(stdout);

    model_free(m);
    return 0;
}
