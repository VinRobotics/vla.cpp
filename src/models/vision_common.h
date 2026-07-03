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

#include <cstdint>
#include <cstring>

namespace vla {

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

}  // namespace vla
