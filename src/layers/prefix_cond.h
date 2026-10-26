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

// Prefix conditioning across chunks (gradient-free real-time chunking): the
// new chunk's first rows are pulled onto the not-yet-executed rest of the
// previous chunk at every flow-matching step. Pinned by
// tests/test_prefix_cond.cpp, which needs no checkpoint.

#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vla {

// Host side: prefix weights W and targets Y, both [n_rows, max_dim] row-major.
// W is RTC's prefix weight (Eq. 5, LeRobot's EXP schedule): 1 on the first
// `delay` rows, c * expm1(c) / (e - 1) after, 0 from row n_prev on. Y is prev
// normalised with mean / (std + eps). Only the first real_dim columns are set;
// everything else stays 0, so W = 0 (no prev) leaves the model unchanged.
inline void prefix_cond_inputs(const float * prev, int64_t n_prev, int64_t delay, int64_t n_rows, int64_t max_dim,
                               int64_t real_dim, const float * mean, const float * std, float eps,
                               std::vector<float> & w_out, std::vector<float> & y_out) {
    w_out.assign(size_t(n_rows * max_dim), 0.0f);
    y_out.assign(size_t(n_rows * max_dim), 0.0f);
    if (!prev || n_prev <= 0) return;
    const int64_t end = std::min<int64_t>(n_prev, n_rows);
    const int64_t d   = std::clamp<int64_t>(delay, 0, end);
    for (int64_t r = 0; r < end; ++r) {
        float w = 1.f;
        if (r >= d) {
            const double c = 1.0 - double(r - d + 1) / double(end - d + 1);
            w = float(c * std::expm1(c) / std::expm1(1.0));
        }
        for (int64_t j = 0; j < real_dim && j < max_dim; ++j) {
            w_out[r * max_dim + j] = w;
            y_out[r * max_dim + j] = (prev[r * max_dim + j] - mean[j]) / (std[j] + eps);
        }
    }
}

// Graph side, one Euler step at flow time t: x1 = x_t - t v_t is the predicted
// clean chunk; it is moved onto Y by W, and the velocity is recomputed from it:
// v_t' = v_t - W (Y - x1) / t. W = 0 returns v_t's value unchanged.
inline ggml_tensor * prefix_cond_velocity(ggml_context * ctx, ggml_tensor * x_t, ggml_tensor * v_t, ggml_tensor * y,
                                          ggml_tensor * w, float t) {
    ggml_tensor * x1  = ggml_sub(ctx, x_t, ggml_scale(ctx, v_t, t));
    ggml_tensor * err = ggml_mul(ctx, ggml_sub(ctx, y, x1), w);
    return ggml_sub(ctx, v_t, ggml_scale(ctx, err, 1.f / t));
}

}  // namespace vla
