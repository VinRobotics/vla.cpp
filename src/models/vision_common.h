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

// Small pure vision helpers shared by the in-tree towers, split out so they can
// be unit-tested without a model or a GPU.

#pragma once

#include "model.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace vla {

// A tower that reads side*side*3 from a view needs the view to be exactly that
// size with real data, else it runs past the buffer.
inline bool view_is_side(const void * data, int w, int h, int64_t side) {
    return data != nullptr && (int64_t) w == side && (int64_t) h == side;
}

// IDEFICS3/SmolVLM pixel-shuffle (space-to-depth), c-innermost channel order.
// src [embed, n_patches] row-major (patch p, channel e) -> dst [embed*s^2, (grid/s)^2].
inline void pixel_shuffle_hf(const float * src, float * dst,
                             int64_t embed, int64_t grid, int64_t s) {
    const int64_t g2 = grid / s, c4 = embed * s * s;
    for (int64_t h2 = 0; h2 < g2; ++h2)
        for (int64_t w2 = 0; w2 < g2; ++w2) {
            const int64_t t = h2 * g2 + w2;
            for (int64_t hs = 0; hs < s; ++hs)
                for (int64_t ws = 0; ws < s; ++ws) {
                    const int64_t p    = (h2 * s + hs) * grid + (w2 * s + ws);
                    const int64_t base = (hs * s + ws) * embed;
                    std::memcpy(dst + t * c4 + base, src + p * embed,
                                (size_t) embed * sizeof(float));
                }
        }
}

// HWC to CHW planar in [-1, 1], the SigLIP convention used by SmolVLA, pi0, pi0.5
// and GR00T N1.5. No resize: the view must already be side x side. arch only
// labels the error.
inline bool preprocess_image_chw(const char * arch, const ImageView & v, int64_t side,
                                 std::vector<float> & out) {
    if (v.w != (int) side || v.h != (int) side || !v.data) {
        std::fprintf(stderr, "vla(%s): image view is %dx%d, expected %lldx%lld\n",
                     arch, v.w, v.h, (long long) side, (long long) side);
        return false;
    }
    out.assign((size_t) 3 * side * side, 0.0f);
    for (int64_t h = 0; h < side; ++h)
        for (int64_t w = 0; w < side; ++w)
            for (int64_t c = 0; c < 3; ++c) {
                float px;
                if (v.format == PixelFormat::U8) px = ((const uint8_t *) v.data)[(h * side + w) * 3 + c] / 255.0f;
                else                             px = ((const float  *) v.data)[(h * side + w) * 3 + c];
                out[c * side * side + h * side + w] = px * 2.0f - 1.0f;
            }
    return true;
}

}  // namespace vla
