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

// Real-time chunking across chunks, without gradients: at every flow-matching step
// the predicted clean chunk is pulled onto the not-yet-executed rest of the previous
// chunk. The weights and the guidance weight follow LeRobot's RTCProcessor
// (policies/rtc/modeling_rtc.py) with the RTCConfig options in model.h (RtcConfig);
// the exact guidance multiplies the error by the Jacobian of x1 = x_t - t v with
// respect to x_t, here the identity is used (the weighted error directly, as
// huggingface/lerobot#4945 keeps as its default). Pinned by tests/test_prefix_cond.cpp,
// which needs no checkpoint and compares against values produced by LeRobot's code.

#pragma once

#include "ggml.h"
#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vla {

// LeRobot's RTCProcessor.get_prefix_weights(start, end, total).
inline std::vector<float> prefix_weights(RtcSchedule schedule, int64_t start, int64_t end, int64_t total) {
    start = std::min(start, end);
    std::vector<float> w(size_t(std::max<int64_t>(total, 0)), 0.0f);
    if (schedule == RtcSchedule::ZEROS) {
        for (int64_t i = 0; i < std::min(start, total); ++i) w[size_t(i)] = 1.0f;
        return w;
    }
    if (schedule == RtcSchedule::ONES) {
        for (int64_t i = 0; i < std::min(end, total); ++i) w[size_t(i)] = 1.0f;
        return w;
    }
    // LINEAR / EXP: leading ones, then linspace(1, 0, steps + 2)[1:-1] (squared by expm1 for EXP), then zeros.
    const int64_t skip  = std::max<int64_t>(total - end, 0);
    const int64_t steps = total - skip - start;
    const int64_t ones  = std::min(start, total);
    for (int64_t i = 0; i < ones; ++i) w[size_t(i)] = 1.0f;
    if (end <= start || steps <= 0) return w;
    for (int64_t i = 0; i < steps && ones + i < total; ++i) {
        const double lin = 1.0 - double(i + 1) / double(steps + 1);
        w[size_t(ones + i)] = schedule == RtcSchedule::EXP ? float(lin * std::expm1(lin) / std::expm1(1.0)) : float(lin);
    }
    return w;
}

// LeRobot's guidance weight at flow time t (1 -> 0; LeRobot's tau = 1 - t), clamped by max_guidance_weight.
inline float guidance_weight(float t, float max_guidance_weight) {
    const double tau = 1.0 - double(t);
    if (!(tau > 0.0) || !(t > 0.0)) return max_guidance_weight;  // 1/0 -> +inf -> clamped (nan_to_num posinf)
    const double c      = (1.0 - tau) / tau;
    const double inv_r2 = ((1.0 - tau) * (1.0 - tau) + tau * tau) / ((1.0 - tau) * (1.0 - tau));
    return float(std::min(c * inv_r2, double(max_guidance_weight)));
}

// Host side: prefix weights W and targets Y, both [n_rows, max_dim] row-major. `prev` is the rest of
// the previous chunk in robot units (n_prev rows of max_dim); Y is prev normalised with
// mean / (std + eps) (zero rows past n_prev, as LeRobot pads). W is get_prefix_weights(delay,
// min(execution_horizon, n_prev), n_rows). Only the first real_dim columns are set, everything
// else stays 0, so W = 0 (no prev) leaves the model unchanged.
inline void prefix_cond_inputs(const float * prev, int64_t n_prev, int64_t delay, int64_t n_rows, int64_t max_dim,
                               int64_t real_dim, const float * mean, const float * std, float eps, const RtcConfig & rtc,
                               std::vector<float> & w_out, std::vector<float> & y_out) {
    w_out.assign(size_t(n_rows * max_dim), 0.0f);
    y_out.assign(size_t(n_rows * max_dim), 0.0f);
    if (!prev || n_prev <= 0) return;
    const int64_t rows = std::min<int64_t>(n_prev, n_rows);
    const int64_t end  = std::min<int64_t>(rtc.execution_horizon, rows);
    const std::vector<float> w = prefix_weights(rtc.schedule, std::max<int64_t>(delay, 0), end, n_rows);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < real_dim && j < max_dim; ++j) {
            w_out[size_t(r * max_dim + j)] = w[size_t(r)];
            y_out[size_t(r * max_dim + j)] = (prev[r * max_dim + j] - mean[j]) / (std[j] + eps);
        }
}

// Graph side, one Euler step at flow time t: x1 = x_t - t v_t is the predicted clean chunk and
// v_t' = v_t - gw * W (Y - x1). W = 0 returns v_t's value unchanged.
inline ggml_tensor * prefix_cond_velocity(ggml_context * ctx, ggml_tensor * x_t, ggml_tensor * v_t, ggml_tensor * y,
                                          ggml_tensor * w, float t, float gw) {
    ggml_tensor * x1  = ggml_sub(ctx, x_t, ggml_scale(ctx, v_t, t));
    ggml_tensor * err = ggml_mul(ctx, ggml_sub(ctx, y, x1), w);
    return ggml_sub(ctx, v_t, ggml_scale(ctx, err, gw));
}

}  // namespace vla
