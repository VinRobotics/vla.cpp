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
// image plus an instruction, runs one predict(), and prints the action chunk.
// No server, no simulator. There is no tokenizer in the C++ core, so --text
// shells out to scripts/tokenize_prompt.py; --tokens takes ids directly.
//
//   vla-cli [--mmproj m.gguf] --ckpt c.gguf --image img.jpg [--image img2.jpg]
//           (--text "pick up the bowl" | --tokens id,id,...) [--state f,f,...] [--pretty]

#include "arch.h"
#include "model.h"
#include "serving/hf_fetch.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
// stb pulls in its full implementation here; keep its unused-function noise out
// of our -Wall -Wextra output.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "stb_image.h"
#pragma GCC diagnostic pop

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vla;

namespace {

// Comma/space separated ints. Returns false on junk or out-of-int32 values.
bool parse_ints(const std::string & s, std::vector<int32_t> & out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ')) ++i;
        if (i >= s.size()) break;
        errno = 0;
        char * e = nullptr;
        long long x = std::strtoll(s.c_str() + i, &e, 10);
        if (e == s.c_str() + i) { std::fprintf(stderr, "vla-cli: bad token near '%s'\n", s.c_str() + i); return false; }
        if (errno == ERANGE || x < INT32_MIN || x > INT32_MAX) { std::fprintf(stderr, "vla-cli: token %lld out of int32 range\n", x); return false; }
        out.push_back((int32_t) x);
        i = (size_t) (e - s.c_str());
    }
    return true;
}

// Comma/space separated floats. Returns false on junk or non-finite values.
bool parse_floats(const std::string & s, std::vector<float> & out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ')) ++i;
        if (i >= s.size()) break;
        char * e = nullptr;
        float x = std::strtof(s.c_str() + i, &e);
        if (e == s.c_str() + i) { std::fprintf(stderr, "vla-cli: bad number near '%s'\n", s.c_str() + i); return false; }
        if (!std::isfinite(x)) { std::fprintf(stderr, "vla-cli: non-finite value in --state\n"); return false; }
        out.push_back(x);
        i = (size_t) (e - s.c_str());
    }
    return true;
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

const char * arch_slug(Arch a) {
    switch (a) {
        case Arch::SMOLVLA:     return "smolvla";
        case Arch::PI0:         return "pi0";
        case Arch::PI05:        return "pi05";
        case Arch::EVO1:        return "evo1";
        case Arch::GR00T_N1_5:  return "gr00t_n1_5";
        case Arch::GR00T_N1_6:  return "gr00t_n1_6";
        case Arch::GR00T_N1_7:  return "gr00t_n1_7";
        case Arch::BITVLA:      return "bitvla";
        case Arch::VLA_ADAPTER: return "vla_adapter";
        case Arch::OPENVLA_OFT: return "openvla_oft";
        case Arch::VLA_JEPA:    return "vla_jepa";
    }
    return "";
}

// The instruction reaches a shell command, so keep it to plain prose.
bool text_ok(const std::string & s) {
    if (s.empty() || s.size() > 512) return false;
    for (const char c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == ' ' || c == '.' || c == ',' ||
                        c == '-' || c == '_' || c == '\'';
        if (!ok) return false;
    }
    return true;
}

// Ask scripts/tokenize_prompt.py for the ids, using the tokenizer the arch was
// trained with. Returns "" and explains on stderr.
std::string tokenize_text(const std::string & ckpt, const std::string & text) {
    Arch arch;
    if (!detect_arch_from_ckpt(ckpt, &arch)) {
        std::fprintf(stderr, "vla-cli: cannot detect the arch of %s for --text\n", ckpt.c_str());
        return "";
    }
    if (!text_ok(text)) {
        std::fprintf(stderr, "vla-cli: --text takes plain prose (letters, digits, space . , - _ ')\n");
        return "";
    }
    std::string esc;
    for (const char c : text) { if (c == '\'') esc += "'\\''"; else esc += c; }
    // Env first so a packaged binary can point at its own copy of the script.
    const char * env = std::getenv("VLA_TOKENIZE_SCRIPT");
    const std::string script = (env && *env) ? std::string(env)
                                             : std::string(VLA_SOURCE_DIR) + "/scripts/tokenize_prompt.py";
    const char * py = std::getenv("VLA_PYTHON");
    const std::string interp = (py && *py) ? std::string(py) : std::string("python3");
    const std::string cmd = "'" + interp + "' '" + script + "' --arch " + arch_slug(arch) +
                            " --text '" + esc + "'";

    FILE * fp = popen(cmd.c_str(), "r");
    if (!fp) { std::fprintf(stderr, "vla-cli: cannot run %s\n", cmd.c_str()); return ""; }
    std::string out;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), fp)) out += buf;
    if (pclose(fp) != 0) {
        std::fprintf(stderr,
                     "vla-cli: tokenizing failed. Install the client extras with\n"
                     "         pip install -e \".[client]\"\n"
                     "         (VLA_PYTHON selects a different interpreter)\n");
        return "";
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s [--mmproj m.gguf] (--ckpt c.gguf | -hf user/repo) --image img.jpg [--image ...]\n"
        "          (--text \"...\" | --tokens id,id,...) [--state f,f,...] [--pretty]\n"
        "  --mmproj   vision-tower GGUF (SmolVLA/pi0/pi0.5); omit for baked-vision archs\n"
        "  --ckpt     model checkpoint GGUF\n"
        "  -hf        HuggingFace repo, user/repo[:file.gguf], cached under $VLA_CACHE\n"
        "  --image    image file, repeat for multi-view (decoded via stb_image)\n"
        "  --text     instruction, tokenized by scripts/tokenize_prompt.py (needs transformers)\n"
        "  --tokens   language token ids, comma-separated, if you tokenized already\n"
        "  --state    proprioception floats, comma-separated (default zeros)\n"
        "  --pretty   print one action row (max_action_dim values) per line\n",
        prog);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string mmproj, ckpt, hf, tokens_s, state_s, text_s;
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
        else if (a == "-hf")       hf   = need("-hf");
        else if (a == "--image")   image_paths.push_back(need("--image"));
        else if (a == "--tokens")  tokens_s = need("--tokens");
        else if (a == "--text")    text_s = need("--text");
        else if (a == "--state")   state_s = need("--state");
        else if (a == "--pretty")  pretty = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "vla-cli: unknown argument %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (!hf.empty()) {
        if (!ckpt.empty()) { std::fprintf(stderr, "vla-cli: pass --ckpt or -hf, not both\n"); return 1; }
        ckpt = vla::hf_resolve(hf);
        if (ckpt.empty()) return 1;
    }
    if (ckpt.empty() || image_paths.empty() || (tokens_s.empty() && text_s.empty())) { usage(argv[0]); return 1; }
    if (!tokens_s.empty() && !text_s.empty()) { std::fprintf(stderr, "vla-cli: pass --text or --tokens, not both\n"); return 1; }
    if (!text_s.empty()) {
        tokens_s = tokenize_text(ckpt, text_s);
        if (tokens_s.empty()) return 1;
        std::fprintf(stderr, "vla-cli: --text tokenized to %s\n", tokens_s.c_str());
    }

    // Validate the cheap args before loading the model.
    std::vector<int32_t> lang;
    std::vector<float>   state;
    if (!parse_ints(tokens_s, lang) || !parse_floats(state_s, state)) return 1;
    if (lang.empty()) { std::fprintf(stderr, "vla-cli: --tokens parsed to nothing\n"); return 1; }

    Model * m = model_load(mmproj, ckpt, "");
    if (!m) { std::fprintf(stderr, "vla-cli: model_load failed\n"); return 1; }
    const Config & cfg = model_config(m);

    if (!state.empty() && (int64_t) state.size() != cfg.max_state_dim)
        std::fprintf(stderr, "vla-cli: --state has %zu values, model expects %lld; padding or truncating\n",
                     state.size(), (long long) cfg.max_state_dim);
    state.resize((size_t) cfg.max_state_dim, 0.0f);

    std::vector<std::vector<uint8_t>> imgbuf(image_paths.size());
    std::vector<ImageView>            views(image_paths.size());
    for (size_t v = 0; v < image_paths.size(); ++v) {
        int w = 0, h = 0;
        if (!load_image(image_paths[v].c_str(), imgbuf[v], w, h)) { model_free(m); return 1; }
        views[v] = ImageView{ imgbuf[v].data(), w, h, PixelFormat::U8 };
    }

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
