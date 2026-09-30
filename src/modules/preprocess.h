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
#include <vector>

namespace vla {

// A tower that reads side*side*3 from a view needs the view to be exactly that
// size with real data, else it runs past the buffer.
inline bool view_is_side(const void * data, int w, int h, int64_t side) {
    return data != nullptr && (int64_t) w == side && (int64_t) h == side;
}

inline bool view_ok(const char * arch, const ImageView & v, int64_t side) {
    if (view_is_side(v.data, v.w, v.h, side))
        return true;
    std::fprintf(stderr, "vla(%s): image view is %dx%d, expected %lldx%lld\n",
                 arch, v.w, v.h, (long long) side, (long long) side);
    return false;
}

// HWC to CHW planar with per-channel mean/std. No resize: the view must
// already be side x side. arch only labels the error.
inline bool preprocess_image_chw(const char * arch, const ImageView & v, int64_t side,
                                 const float mean[3], const float std_[3], std::vector<float> & out) {
    if (!view_ok(arch, v, side))
        return false;
    out.assign((size_t) 3*side * side, 0.0f);
    for (int64_t h=0; h<side; ++h)
        for (int64_t w=0; w<side; ++w)
            for (int64_t c=0; c<3; ++c) {
                float px;
                if (v.format == PixelFormat::U8)
                    px = ((const uint8_t *) v.data)[(h * side+w)*3+c]/255.0f;
                else
                    px = ((const float  *) v.data)[(h * side+w)*3+c];
                out[c * side * side+h * side+w] = (px-mean[c])/std_[c];
            }
    return true;
}

// [-1, 1], the SigLIP convention used by SmolVLA, pi0, pi0.5 and GR00T N1.5.
inline bool preprocess_image_chw(const char * arch, const ImageView & v, int64_t side,
                                 std::vector<float> & out) {
    static const float half[3] = {0.5f, 0.5f, 0.5f};
    return preprocess_image_chw(arch, v, side, half, half, out);
}

inline bool preprocess_image_patches(const char * arch, const ImageView & v, int64_t side, int64_t ps,
                                     std::vector<float> & out) {
    if (!view_ok(arch, v, side))
        return false;
    const int64_t grid = side/ps, pd = 3*ps*ps, np = grid*grid;
    out.assign((size_t) pd*np, 0.0f);

    auto px = [&](int64_t r, int64_t c, int64_t ch) -> float {
        if (v.format == PixelFormat::U8)
            return ((const uint8_t *) v.data)[(r*side+c)*3+ch]/255.0f;
        return ((const float *) v.data)[(r*side+c)*3+ch];
    };

    for (int64_t row=0; row<grid; ++row)
        for (int64_t col=0; col<grid; ++col) {
            const int64_t t = row*grid+col;
            for (int64_t ph=0; ph<ps; ++ph)
                for (int64_t pw=0; pw<ps; ++pw)
                    for (int64_t ch=0; ch<3; ++ch)
                        out[t*pd+ph*ps*3+pw*3+ch] = px(row*ps+ph, col*ps+pw, ch)*2.0f-1.0f;
        }
    return true;
}

}  // namespace vla
