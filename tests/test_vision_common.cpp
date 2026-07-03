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

// Unit test for the IDEFICS3/SmolVLM pixel-shuffle channel order: a c-outermost
// (wrong) arrangement fails the hard-coded expectations below.

#include "models/vision_common.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    // embed=2, grid=4, s=2 -> g2=2, n_patches=16, c4=8, out tokens=4.
    // src[p*embed + e] = 1000*p + e, unique per (patch, channel).
    const int64_t embed = 2, grid = 4, s = 2;
    const int64_t n_patches = grid * grid, c4 = embed * s * s, tokens = (grid / s) * (grid / s);
    std::vector<float> src((size_t) embed * n_patches);
    for (int64_t p = 0; p < n_patches; ++p)
        for (int64_t e = 0; e < embed; ++e)
            src[p * embed + e] = 1000.0f * (float) p + (float) e;

    std::vector<float> dst((size_t) c4 * tokens, -1.0f);
    vla::pixel_shuffle_hf(src.data(), dst.data(), embed, grid, s);

    // Hand-computed token 0 (h2=0,w2=0): inner patches {p0,p1,p4,p5}, channel
    // innermost. A c-outermost layout would order these differently.
    const float expect0[8] = {0, 1, 1000, 1001, 4000, 4001, 5000, 5001};
    for (int i = 0; i < 8; ++i) {
        if (dst[i] != expect0[i]) {
            std::fprintf(stderr, "pixel_shuffle_hf token0[%d]=%g, expected %g (c-order wrong?)\n",
                         i, (double) dst[i], (double) expect0[i]);
            return 1;
        }
    }
    if (dst[8] != 2000.0f) {  // token 1 (w2=1), inner patch p2, channel 0
        std::fprintf(stderr, "pixel_shuffle_hf token1[0]=%g, expected 2000\n", (double) dst[8]);
        return 1;
    }

    // Full mapping check against the reference index formula.
    const int64_t g2 = grid / s;
    for (int64_t h2 = 0; h2 < g2; ++h2)
    for (int64_t w2 = 0; w2 < g2; ++w2)
    for (int64_t hs = 0; hs < s; ++hs)
    for (int64_t ws = 0; ws < s; ++ws)
    for (int64_t e = 0; e < embed; ++e) {
        const int64_t t = h2 * g2 + w2;
        const int64_t p = (h2 * s + hs) * grid + (w2 * s + ws);
        const float got = dst[t * c4 + (hs * s + ws) * embed + e];
        const float want = 1000.0f * (float) p + (float) e;
        assert(got == want);
        (void) got; (void) want;
    }

    std::printf("test_vision_common: pixel_shuffle_hf c-innermost order OK\n");
    return 0;
}
