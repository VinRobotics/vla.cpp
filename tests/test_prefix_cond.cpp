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

// Pins layers/prefix_cond.h without a checkpoint: (1) the prefix weights of every schedule and the
// clamped guidance weight against values produced by LeRobot's RTCProcessor (tests/data/rtc_reference.inc,
// regenerated with tests/gen_rtc_reference.py); (2) an Euler loop from t = 1 to 0 with a random
// stand-in velocity per step, corrected by prefix_cond_velocity: with no prev the output is
// bit-identical to the uncorrected loop, rows past the execution horizon are untouched, and the
// prefix rows end closer to the normalised previous chunk than without the correction; (3) archs
// that do not override supports_prefix_cond() refuse the input.

#include "layers/prefix_cond.h"
#include "arch.h"

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "data/rtc_reference.inc"

static const int64_t D = 8, ROWS = 10, REAL = 6, STEPS = 10;

static vla::RtcSchedule schedule_of(const std::string & s) {
    if (s == "ZEROS") return vla::RtcSchedule::ZEROS;
    if (s == "ONES") return vla::RtcSchedule::ONES;
    if (s == "LINEAR") return vla::RtcSchedule::LINEAR;
    return vla::RtcSchedule::EXP;
}

// Runs the Euler loop; returns x_0 [ROWS, D]. W, Y as built by prefix_cond_inputs.
static std::vector<float> integrate(const std::vector<float> & x_init, const std::vector<std::vector<float>> & vs,
                                    const std::vector<float> & w, const std::vector<float> & y, bool correct,
                                    float max_gw) {
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
        const float t = 1.f + float(step) * dt;
        if (correct) v = vla::prefix_cond_velocity(C, x, v, Y, W, t, vla::guidance_weight(t, max_gw));
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

namespace {
struct NoRtcArch : vla::ModelArchBase {
    NoRtcArch() : vla::ModelArchBase(vla::Arch::PI0) {}
    std::vector<float> predict(const vla::Inputs &) override { return {}; }
};
}  // namespace

int main() {
    // --- (1) LeRobot parity: prefix weights of every schedule, guidance weight with its clamp. ---
    for (const RtcWeightCase & c : kRtcWeights) {
        const std::vector<float> got = vla::prefix_weights(schedule_of(c.schedule), c.start, c.end, c.total);
        assert(got.size() == c.w.size());
        for (size_t i = 0; i < got.size(); ++i) assert(std::fabs(got[i] - c.w[i]) <= 1e-6f);
    }
    for (const RtcGwCase & c : kRtcGw) assert(std::fabs(vla::guidance_weight(c.t, c.max_gw) - c.gw) <= 1e-5f * std::fmax(1.f, c.gw));
    std::printf("LeRobot parity: %zu weight cases, %zu guidance cases\n", kRtcWeights.size(), kRtcGw.size());

    std::mt19937 rng(0);
    std::normal_distribution<float> n01(0.f, 1.f);
    auto randv = [&](size_t n) { std::vector<float> v(n); for (float & e : v) e = n01(rng); return v; };
    const std::vector<float> x_init = randv(size_t(ROWS * D));
    std::vector<std::vector<float>> vs;
    for (int64_t s = 0; s < STEPS; ++s) vs.push_back(randv(size_t(ROWS * D)));
    const std::vector<float> mean = randv(size_t(REAL)), stdv = { 0.5f, 1.f, 2.f, 3.f, 0.25f, 1.5f };
    const float eps = 1e-8f;

    // --- (2a) Inputs: W = LeRobot weights on the first `real` columns of the first rows, 0 elsewhere; Y = normalised prev. ---
    vla::RtcConfig rtc;  // LeRobot defaults: execution_horizon 10, max_guidance_weight 10, LINEAR
    assert(rtc.execution_horizon == 10 && rtc.max_guidance_weight == 10.0f && rtc.schedule == vla::RtcSchedule::LINEAR);
    const int64_t n_prev = 7, delay = 3;  // horizon = min(10, 7) = 7
    const std::vector<float> prev = randv(size_t(ROWS * D));
    std::vector<float> w, y;
    vla::prefix_cond_inputs(prev.data(), n_prev, delay, ROWS, D, REAL, mean.data(), stdv.data(), eps, rtc, w, y);
    const std::vector<float> want_w = vla::prefix_weights(vla::RtcSchedule::LINEAR, delay, n_prev, ROWS);
    for (int64_t r = 0; r < ROWS; ++r)
        for (int64_t j = 0; j < D; ++j) {
            const float wr = w[r * D + j];
            if (j >= REAL || r >= n_prev) assert(wr == 0.f && y[r * D + j] == 0.f);
            else {
                assert(wr == want_w[r]);
                assert(std::fabs(y[r * D + j] - (prev[r * D + j] - mean[j]) / (stdv[j] + eps)) <= 1e-6f);
            }
        }
    // An execution horizon below n_prev ends the weights there; one above is cut to n_prev.
    {
        vla::RtcConfig r2 = rtc; r2.execution_horizon = 5;
        std::vector<float> w2, y2;
        vla::prefix_cond_inputs(prev.data(), n_prev, delay, ROWS, D, REAL, mean.data(), stdv.data(), eps, r2, w2, y2);
        for (int64_t r = 5; r < ROWS; ++r) assert(w2[r * D] == 0.f);
        assert(w2[4 * D] > 0.f);
    }

    // --- (2b) With prev: rows past the horizon are untouched; the prefix rows end nearer the normalised prev. ---
    {
        const std::vector<float> out   = integrate(x_init, vs, w, y, true, rtc.max_guidance_weight);
        const std::vector<float> plain = integrate(x_init, vs, w, y, false, rtc.max_guidance_weight);
        for (int64_t r = n_prev; r < ROWS; ++r)
            for (int64_t j = 0; j < D; ++j) assert(out[r * D + j] == plain[r * D + j]);
        double e_out = 0, e_plain = 0;
        for (int64_t r = 0; r < delay; ++r)
            for (int64_t j = 0; j < REAL; ++j) {
                const float want = (prev[r * D + j] - mean[j]) / (stdv[j] + eps);
                e_out += std::fabs(out[r * D + j] - want);
                e_plain += std::fabs(plain[r * D + j] - want);
            }
        std::printf("prefix rows |x0 - prev|: corrected %.4g, uncorrected %.4g\n", e_out, e_plain);
        assert(e_out < 0.5 * e_plain);
    }

    // --- (2c) Without prev: W = 0 everywhere, output bit-identical to the uncorrected loop. ---
    {
        std::vector<float> w0, y0;
        vla::prefix_cond_inputs(nullptr, 0, delay, ROWS, D, REAL, mean.data(), stdv.data(), eps, rtc, w0, y0);
        for (float e : w0) assert(e == 0.f);
        const std::vector<float> a = integrate(x_init, vs, w0, y0, true, rtc.max_guidance_weight);
        const std::vector<float> b = integrate(x_init, vs, w0, y0, false, rtc.max_guidance_weight);
        assert(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    }

    // --- (3) An arch that does not override the flag refuses prev_actions. ---
    {
        NoRtcArch a;
        assert(!a.supports_prefix_cond());
    }

    std::printf("prefix_cond OK\n");
    return 0;
}
