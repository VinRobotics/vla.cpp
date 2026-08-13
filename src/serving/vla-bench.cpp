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

// Latency for one checkpoint. Synthetic inputs: engine only, no task success.

#include "model.h"
#include "serving/hf_fetch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s (--ckpt c.gguf | -hf user/repo) [--mmproj m.gguf]\n"
        "          [--label name] [--images N] [--size N] [--tokens N]\n"
        "          [--extra-token ID] [--extra-count N] [--warmup N] [--reps N] [--markdown]\n"
        "  --label    row label (default: the checkpoint filename)\n"
        "  --images   camera views (default 1)\n"
        "  --size     square input side in pixels (default 224)\n"
        "  --tokens   language token count (default 16)\n"
        "  --extra-token  token id appended --extra-count times (VLA-JEPA needs its\n"
        "                 <embodied> tokens)\n"
        "  --warmup   untimed calls before measuring (default 3)\n"
        "  --reps     timed calls (default 20)\n"
        "  --markdown print a markdown table row instead of a plain summary\n",
        prog);
}

// v must be sorted.
double percentile(const std::vector<double> & v, double p) {
    if (v.empty())
        return 0.0;
    const double idx = p * (double) (v.size() - 1);
    const size_t lo = (size_t) std::floor(idx), hi = (size_t) std::ceil(idx);
    return v[lo] + (v[hi] - v[lo]) * (idx - (double) lo);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string ckpt, mmproj, hf, label;
    int n_images = 1, side = 224, n_tokens = 16, warmup = 3, reps = 20;
    int extra_token = -1, extra_count = 0;
    bool markdown = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "vla-bench: %s needs a value\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--ckpt")
            ckpt     = need("--ckpt");
        else if (a == "-hf")        hf       = need("-hf");
        else if (a == "--mmproj")   mmproj   = need("--mmproj");
        else if (a == "--label")    label    = need("--label");
        else if (a == "--images")   n_images = std::atoi(need("--images"));
        else if (a == "--size")     side     = std::atoi(need("--size"));
        else if (a == "--tokens")   n_tokens = std::atoi(need("--tokens"));
        else if (a == "--extra-token") extra_token = std::atoi(need("--extra-token"));
        else if (a == "--extra-count") extra_count = std::atoi(need("--extra-count"));
        else if (a == "--warmup")   warmup   = std::atoi(need("--warmup"));
        else if (a == "--reps")     reps     = std::atoi(need("--reps"));
        else if (a == "--markdown") markdown = true;
        else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        }
        else {
            std::fprintf(stderr, "vla-bench: unknown argument %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (!hf.empty()) {
        if (!ckpt.empty()) {
            std::fprintf(stderr, "vla-bench: pass --ckpt or -hf, not both\n");
            return 1;
        }
        ckpt = vla::hf_resolve(hf);
        if (ckpt.empty())
            return 1;
    }
    if (ckpt.empty()) {
        usage(argv[0]);
        return 1;
    }
    if (n_images < 1 || side < 16 || n_tokens < 1 || warmup < 0 || reps < 1) {
        std::fprintf(stderr, "vla-bench: --images/--size/--tokens/--reps must be positive\n");
        return 1;
    }
    if (label.empty()) {
        const size_t slash = ckpt.find_last_of('/');
        label = (slash == std::string::npos) ? ckpt : ckpt.substr(slash + 1);
    }

    vla::Model * m = vla::model_load(mmproj, ckpt, "");
    if (!m) {
        std::fprintf(stderr, "vla-bench: model_load failed\n");
        return 1;
    }
    const vla::Config & cfg = vla::model_config(m);

    std::vector<std::vector<uint8_t>> pixels(n_images, std::vector<uint8_t>((size_t) 3 * side * side));
    std::vector<vla::ImageView> views(n_images);
    for (int v = 0; v < n_images; ++v) {
        for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x)
        for (int c = 0; c < 3; ++c)
            pixels[v][((size_t) y * side + x) * 3 + c] = (uint8_t) ((x + 2 * y + 40 * c + 17 * v) & 0xFF);
        views[v] = vla::ImageView{ pixels[v].data(), side, side, vla::PixelFormat::U8 };
    }

    std::vector<int32_t> lang((size_t) n_tokens);
    for (int i = 0; i < n_tokens; ++i)
        lang[i] = 1 + (i % 100);
    if (extra_token >= 0 && extra_count > 0)
        lang.insert(lang.end(), (size_t) extra_count, extra_token);

    std::vector<float> state((size_t) cfg.max_state_dim, 0.0f);
    for (int64_t i = 0; i < cfg.real_state_dim && i < cfg.max_state_dim; ++i)
        state[i] = 0.01f * (float) (i + 1);

    std::vector<float> noise((size_t) cfg.max_action_dim * (size_t) cfg.n_suffix);
    for (size_t i = 0; i < noise.size(); ++i)
        noise[i] = 0.001f * (float) ((i * 2654435761u) % 1000) - 0.5f;

    vla::Inputs in{};
    in.images      = views.data();
    in.n_images    = n_images;
    in.lang_tokens = lang.data();
    in.n_lang      = (int) lang.size();
    in.state       = state.data();
    in.noise       = noise.data();

    for (int i = 0; i < warmup; ++i) {
        if (vla::predict(m, in).empty()) {
            std::fprintf(stderr, "vla-bench: predict failed\n");
            vla::model_free(m);
            return 1;
        }
    }

    std::vector<double> ms;
    ms.reserve((size_t) reps);
    double vision_sum = 0.0;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> out = vla::predict(m, in);
        const auto t1 = std::chrono::steady_clock::now();
        if (out.empty()) {
            std::fprintf(stderr, "vla-bench: predict failed at rep %d\n", i);
            vla::model_free(m);
            return 1;
        }
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        vision_sum += vla::last_stats(m).ms_vision;
    }

    std::sort(ms.begin(), ms.end());
    const double lo     = ms.front();
    const double p50    = percentile(ms, 0.50);
    const double p90    = percentile(ms, 0.90);
    const double vision = vision_sum / (double) reps;

    if (markdown) {
        std::printf("| %s | %d | %d | %d | %.1f | %.1f | %.1f | %.1f |\n",
                    label.c_str(), n_images, side, n_tokens, lo, p50, p90, vision);
    } else {
        std::printf("%s: min %.1f ms  p50 %.1f ms  p90 %.1f ms  vision %.1f ms  (%d views, %dx%d, %d tokens, %d reps)\n",
                    label.c_str(), lo, p50, p90, vision, n_images, side, side, n_tokens, reps);
    }

    vla::model_free(m);
    return 0;
}
