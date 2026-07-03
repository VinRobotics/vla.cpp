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

// Deterministic output-equality + timing harness for vla::predict. Feeds fixed
// images/language/state/noise and prints the action chunk; diff stdout before and
// after a change to confirm numerics are unchanged (bitvla ignores noise, the
// others read it, so fixed noise is reproducible for all archs).
//
//   predict_check <ckpt.gguf> [mmproj.gguf] [n_images]
//   env: VLA_IMG_SIZE (square input, default 224), VLA_BENCH_ITERS (>0 = time it)

#include "model.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace vla;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <ckpt.gguf> [mmproj.gguf] [n_images]\n", argv[0]);
        return 1;
    }
    const char* ckpt     = argv[1];
    const char* mmproj   = (argc > 2 && argv[2][0] && argv[2][0] != '-') ? argv[2] : "";
    const int   n_images = argc > 3 ? std::atoi(argv[3]) : 2;

    Model* m = model_load(mmproj, ckpt, "");
    if (!m) {
        std::fprintf(stderr, "model_load failed\n");
        return 1;
    }
    const Config& cfg = model_config(m);
    std::fprintf(stderr,
        "loaded: real_state=%lld max_state=%lld real_action=%lld max_action=%lld "
        "n_suffix=%lld num_steps=%d\n",
        (long long)cfg.real_state_dim, (long long)cfg.max_state_dim,
        (long long)cfg.real_action_dim, (long long)cfg.max_action_dim,
        (long long)cfg.n_suffix, cfg.num_steps);

    const char* isz = std::getenv("VLA_IMG_SIZE");
    const int W = isz ? std::atoi(isz) : 224, H = W;
    std::vector<std::vector<uint8_t>> imgbuf(n_images, std::vector<uint8_t>((size_t)3 * W * H));
    std::vector<ImageView> views(n_images);
    for (int v = 0; v < n_images; ++v) {
        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        for (int c = 0; c < 3; ++c)
            imgbuf[v][((size_t)y * W + x) * 3 + c] = (uint8_t)((x + 2 * y + 40 * c + 17 * v) & 0xFF);
        views[v] = ImageView{ imgbuf[v].data(), W, H, PixelFormat::U8 };
    }

    std::vector<int32_t> lang = {1, 100, 200, 300, 400, 2};
    std::vector<float>   state((size_t)cfg.max_state_dim, 0.0f);
    for (int i = 0; i < (int)cfg.real_state_dim; ++i) state[i] = 0.01f * (float)(i + 1);

    const size_t noise_n = (size_t)cfg.max_action_dim * (size_t)cfg.n_suffix;
    std::vector<float> noise(noise_n);
    for (size_t i = 0; i < noise_n; ++i)
        noise[i] = 0.001f * (float)((i * 2654435761u) % 1000) - 0.5f;

    Inputs in{};
    in.images        = views.data();
    in.n_images      = n_images;
    in.lang_tokens   = lang.data();
    in.n_lang        = (int)lang.size();
    in.state         = state.data();
    in.noise         = noise.data();
    in.timing_detail = TimingDetail::NONE;

    std::vector<float> act = predict(m, in);
    std::printf("action_len=%zu\n", act.size());
    for (size_t i = 0; i < act.size(); ++i) std::printf("%.9g\n", (double)act[i]);
    std::fflush(stdout);

    const char* it_env = std::getenv("VLA_BENCH_ITERS");
    const int iters = it_env ? std::atoi(it_env) : 0;
    if (iters > 0) {
        for (int w = 0; w < 3; ++w) (void)predict(m, in);
        double best = 1e30, sum = 0.0;
        for (int i = 0; i < iters; ++i) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            std::vector<float> a = predict(m, in);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            best = ms < best ? ms : best;
            sum += ms;
        }
        const Stats& st = last_stats(m);
        std::fprintf(stderr, "predict() over %d iters: min=%.3f ms  avg=%.3f ms\n",
                     iters, best, sum / iters);
        std::fprintf(stderr, "last split: vision=%.2f  inference=%.2f  total=%.2f ms\n",
                     st.ms_vision, st.ms_inference, st.ms_total);
    }

    model_free(m);
    return act.empty() ? 2 : 0;
}
