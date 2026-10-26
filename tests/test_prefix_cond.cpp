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

// Pins layers/prefix_cond.h without a checkpoint: an Euler loop from t = 1 to
// 0 with a random stand-in velocity per step, corrected by prefix_cond_velocity.
// With prev set, the first `delay` output rows land on the normalised previous
// chunk; with no prev, every step's velocity, and so the output, is bit-identical
// to the uncorrected loop.

#include "layers/prefix_cond.h"

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const int64_t D = 8, ROWS = 10, REAL = 6, STEPS = 10;

// Runs the Euler loop; returns x_0 [ROWS, D]. W, Y as built by prefix_cond_inputs.
static std::vector<float> integrate(const std::vector<float> & x_init, const std::vector<std::vector<float>> & vs,
                                    const std::vector<float> & w, const std::vector<float> & y, bool correct) {
    ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, false };
    ggml_context * C = ggml_init(ip);
    auto make = [&](const std::vector<float> & v) {
        ggml_tensor * t = ggml_new_tensor_2d(C, GGML_TYPE_F32, D, ROWS);
        std::memcpy(t->data, v.data(), v.size() * sizeof(float));
        return t;
    };
    ggml_tensor * x = make(x_init);
    ggml_tensor * W = make(w);
    ggml_tensor * Y = make(y);
    const float dt = -1.0f / float(STEPS);
    for (int64_t step = 0; step < STEPS; ++step) {
        ggml_tensor * v = make(vs[step]);
        if (correct) v = vla::prefix_cond_velocity(C, x, v, Y, W, 1.f + float(step) * dt);
        x = ggml_add(C, x, ggml_scale(C, v, dt));
    }
    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, x);
    ggml_graph_compute_with_ctx(C, gf, 1);
    std::vector<float> out(size_t(ROWS * D));
    std::memcpy(out.data(), x->data, out.size() * sizeof(float));
    ggml_free(C);
    return out;
}

int main() {
    std::mt19937 rng(0);
    std::normal_distribution<float> n01(0.f, 1.f);
    auto randv = [&](size_t n) { std::vector<float> v(n); for (float & e : v) e = n01(rng); return v; };
    const std::vector<float> x_init = randv(size_t(ROWS * D));
    std::vector<std::vector<float>> vs;
    for (int64_t s = 0; s < STEPS; ++s) vs.push_back(randv(size_t(ROWS * D)));
    const std::vector<float> mean = randv(size_t(REAL)), stdv = { 0.5f, 1.f, 2.f, 3.f, 0.25f, 1.5f };
    const float eps = 1e-8f;

    // --- Weights: 1 on the first `delay` rows, decaying, 0 from n_prev on and on padded columns. ---
    const int64_t n_prev = 7, delay = 3;
    const std::vector<float> prev = randv(size_t(ROWS * D));
    std::vector<float> w, y;
    vla::prefix_cond_inputs(prev.data(), n_prev, delay, ROWS, D, REAL, mean.data(), stdv.data(), eps, w, y);
    for (int64_t r = 0; r < ROWS; ++r)
        for (int64_t j = 0; j < D; ++j) {
            const float wr = w[r * D + j];
            if (j >= REAL || r >= n_prev) assert(wr == 0.f);
            else if (r < delay) assert(wr == 1.f);
            else assert(wr > 0.f && wr < 1.f);
        }
    for (int64_t r = delay + 1; r < n_prev; ++r) assert(w[r * D] < w[(r - 1) * D]);  // decays

    // --- With prev: the first `delay` rows of the output equal Y (normalised prev). ---
    {
        const std::vector<float> out = integrate(x_init, vs, w, y, true);
        float worst = 0.f;
        for (int64_t r = 0; r < delay; ++r)
            for (int64_t j = 0; j < REAL; ++j) {
                const float want = (prev[r * D + j] - mean[j]) / (stdv[j] + eps);
                worst = std::fmax(worst, std::fabs(out[r * D + j] - want));
            }
        std::printf("prefix rows vs normalised prev: max |d| = %.3g\n", (double) worst);
        assert(worst <= 1e-5f);
        // Rows past n_prev are untouched by the correction (W = 0 there).
        const std::vector<float> plain = integrate(x_init, vs, w, y, false);
        for (int64_t r = n_prev; r < ROWS; ++r)
            for (int64_t j = 0; j < D; ++j) assert(out[r * D + j] == plain[r * D + j]);
    }

    // --- Without prev: W = 0 everywhere, output bit-identical to the uncorrected loop. ---
    {
        std::vector<float> w0, y0;
        vla::prefix_cond_inputs(nullptr, 0, delay, ROWS, D, REAL, mean.data(), stdv.data(), eps, w0, y0);
        for (float e : w0) assert(e == 0.f);
        const std::vector<float> a = integrate(x_init, vs, w0, y0, true);
        const std::vector<float> b = integrate(x_init, vs, w0, y0, false);
        assert(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    }

    std::printf("prefix_cond OK\n");
    return 0;
}
