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

// Pins the two DiT time embeddings shared by GR00T N1.5/N1.6/N1.7 and VLA-JEPA.
// Pure host functions, so they can be tested without a checkpoint, which the
// end-to-end sha oracle cannot do for three of the four archs. The trig orders
// are opposite and both match the reference.

#include "layers/embed.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static bool close(float a, float b) { return std::fabs(a - b) <= 1e-6f * (1.0f + std::fabs(b)); }

int main() {
    // --- timesteps_proj: 256 wide, cos in [0,128), sin in [128,256). ---
    {
        std::vector<float> out;
        vla::timesteps_proj(0, out);
        assert(out.size() == 256);
        // bucket 0 -> emb == 0 for every i -> cos(0)=1, sin(0)=0.
        for (int i = 0; i < 128; ++i) { assert(close(out[i], 1.0f)); assert(close(out[128 + i], 0.0f)); }

        vla::timesteps_proj(250, out);
        const float lm = std::log(10000.0f);
        for (int i : { 0, 1, 63, 127 }) {
            const float emb = 250.0f * std::exp(-lm * (float) i / 127.0f);
            assert(close(out[i],       std::cos(emb)));   // cos first
            assert(close(out[128 + i], std::sin(emb)));   // sin second
        }
        // The i=0 term has exp(0)=1, so it is the raw timestep.
        assert(close(out[0], std::cos(250.0f)));
    }

    // --- action_sinusoid: [T, dim], sin in the low half, cos in the high half. ---
    {
        const int64_t dim = 8, T = 3;
        std::vector<float> out;
        vla::action_sinusoid(0, dim, T, out);
        assert(out.size() == (size_t) (T * dim));
        for (int64_t tk = 0; tk < T; ++tk)
            for (int64_t i = 0; i < dim / 2; ++i) {
                assert(close(out[tk * dim + i], 0.0f));              // sin(0)
                assert(close(out[tk * dim + dim / 2 + i], 1.0f));    // cos(0)
            }

        vla::action_sinusoid(500, dim, T, out);
        const float step = std::log(10000.0f) / (float) (dim / 2);
        for (int64_t i = 0; i < dim / 2; ++i) {
            const float emb = 500.0f * std::exp(-(float) i * step);
            assert(close(out[i],           std::sin(emb)));  // sin first
            assert(close(out[dim / 2 + i], std::cos(emb)));  // cos second
        }
        // Every horizon step carries the same embedding.
        for (int64_t tk = 1; tk < T; ++tk)
            for (int64_t i = 0; i < dim; ++i)
                assert(close(out[tk * dim + i], out[i]));
    }

    // --- The orders are opposite; a consistency cleanup must fail here. ---
    {
        std::vector<float> tp, as;
        vla::timesteps_proj(7, tp);
        vla::action_sinusoid(7, 256, 1, as);
        // Same width and same bucket, but tp[0] is a cosine and as[0] is a sine.
        assert(as.size() == tp.size());
        assert(!close(tp[0], as[0]));
        assert(close(tp[0], std::cos(7.0f)));
        assert(close(as[0], std::sin(7.0f)));
    }

    std::printf("dit common: ok\n");
    return 0;
}
