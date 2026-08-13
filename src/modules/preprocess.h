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

// HWC to a [3*ps*ps, grid*grid] patch table in [-1, 1], the GEMM patch-embed
// input GR00T N1.6 uses in place of a conv2d.
inline bool preprocess_image_patches(const char * arch, const ImageView & v, int64_t side, int64_t ps,
                                     std::vector<float> & out) {
    if (v.w != (int) side || v.h != (int) side || !v.data) {
        std::fprintf(stderr, "vla(%s): image view is %dx%d, expected %lldx%lld\n",
                     arch, v.w, v.h, (long long) side, (long long) side);
        return false;
    }
    const int64_t grid = side/ps, pd = 3*ps*ps, np = grid*grid;
    out.assign((size_t) pd*np, 0.0f);

    auto px = [&](int64_t r, int64_t c, int64_t ch) -> float {
        if (v.format == PixelFormat::U8) return ((const uint8_t *) v.data)[(r*side+c)*3+ch]/255.0f;
        return ((const float *) v.data)[(r*side+c)*3+ch];
    };

    for (int64_t row = 0; row < grid; ++row)
        for (int64_t col = 0; col < grid; ++col) {
            const int64_t t = row*grid+col;
            for (int64_t ph = 0; ph < ps; ++ph)
                for (int64_t pw = 0; pw < ps; ++pw)
                    for (int64_t ch = 0; ch < 3; ++ch)
                        out[t*pd+ph*ps*3+pw*3+ch] = px(row*ps+ph, col*ps+pw, ch)*2.0f-1.0f;
        }
    return true;
}

// Pixel shuffle with c-outermost channel order, the inverse layout to
// pixel_shuffle_hf above. GR00T N1.6's connector expects this one.
inline void pixel_shuffle_back(const float * src, int64_t grid, int64_t hidden, int64_t r, float * dst) {
    const int64_t g2 = grid/r, c4 = hidden*r*r;
    for (int64_t y = 0; y < g2; ++y)
        for (int64_t x = 0; x < g2; ++x) {
            const int64_t t = y*g2+x;
            for (int64_t c = 0; c < hidden; ++c)
                for (int64_t i = 0; i < r; ++i)
                    for (int64_t j = 0; j < r; ++j) {
                        const int64_t pp = (r*y+i)*grid+(r*x+j);
                        const int64_t cp = c*r*r+i*r+j;
                        dst[t*c4+cp] = src[pp*hidden+c];
                    }
        }
}

}  // namespace vla
