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

// DiT head pieces shared verbatim by GR00T N1.5/N1.6/N1.7 and VLA-JEPA. dit_kv
// and build_dit_block take a per-arch struct and stay in their own files.

#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace vla {

// Row id of a stacked [out, in, n_embodiment] weight, applied to x.
inline ggml_tensor * cat_linear(ggml_context * C, ggml_tensor * W3d, ggml_tensor * b2d, int64_t id, ggml_tensor * x) {
    const int64_t out = W3d->ne[0], in = W3d->ne[1];
    ggml_tensor * W_id = ggml_view_2d(C, W3d, out, in, W3d->nb[1], (size_t) id * W3d->nb[2]);
    ggml_tensor * y = ggml_mul_mat(C, ggml_cont(C, ggml_transpose(C, W_id)), x);
    return ggml_add(C, y, ggml_view_1d(C, b2d, out, (size_t) id * b2d->nb[1]));
}

// Per-block AdaLN. The conditioning vector is (scale, shift) in that order; the
// final projection layer in each arch uses (shift, scale) instead.
inline ggml_tensor * adaln(ggml_context * C, ggml_tensor * x, ggml_tensor * temb, ggml_tensor * lw, ggml_tensor * lb, int64_t dim, float eps) {
    ggml_tensor * cond = ggml_add(C, ggml_mul_mat(C, lw, ggml_silu(C, temb)), lb);
    ggml_tensor * sc = ggml_view_1d(C, cond, dim, 0), * sh = ggml_view_1d(C, cond, dim, (size_t) dim * sizeof(float));
    ggml_tensor * xn = ggml_norm(C, x, eps);
    return ggml_add(C, ggml_add(C, xn, ggml_mul(C, xn, sc)), sh);
}

// cos first, then sin. Opposite order to action_sinusoid; both match the
// reference and are pinned by tests/test_dit_common.cpp.
inline void timesteps_proj(int64_t bucket, std::vector<float> & out) {
    const int64_t half = 128; const float lm = std::log(10000.0f); const float t = (float) bucket;
    out.assign(256, 0.0f);
    for (int64_t i=0; i<half; ++i) {
        const float emb = t * std::exp(-lm * (float) i/(float) (half-1));
        out[i] = std::cos(emb);
        out[half+i] = std::sin(emb);
    }
}

// Broadcast across the horizon. sin first, then cos.
inline void action_sinusoid(int64_t bucket, int64_t dim, int64_t T, std::vector<float> & out) {
    const int64_t half = dim/2; const float step = std::log(10000.0f)/(float) half; const float t = (float) bucket;
    out.assign((size_t) T * dim, 0.0f);
    for (int64_t tk=0; tk<T; ++tk) for (int64_t i = 0; i < half; ++i) {
        const float emb = t * std::exp(-(float) i * step);
        out[tk * dim+i] = std::sin(emb);
        out[tk * dim+half+i] = std::cos(emb);
    }
}

// Log-spaced periods rather than frequencies, the openpi convention shared by
// pi0, pi0.5 and SmolVLA.
inline std::vector<float> sinusoidal_time_emb(double t, int64_t dim, double min_p, double max_p) {
    const int64_t half = dim/2;
    std::vector<float> out(dim);
    for (int64_t i=0; i<half; ++i) {
        const double frac   = (half == 1) ? 0.0 : double(i)/double(half-1);
        const double period = min_p * std::pow(max_p/min_p, frac);
        const double s      = (2.0*M_PI/period)*t;
        out[i]        = (float) std::sin(s);
        out[half+i] = (float) std::cos(s);
    }
    return out;
}

// Additive causal mask, -inf above the diagonal.
inline void build_causal_mask(int64_t seq, std::vector<float> & out) {
    out.assign((size_t) seq * seq, 0.0f);
    const float NEG = -std::numeric_limits<float>::infinity();
    for (int64_t q=0; q<seq; ++q)
        for (int64_t kv=q+1; kv<seq; ++kv)
            out[q * seq+kv] = NEG;
}

}  // namespace vla
