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

// Unit test for the shared image view check and CHW preprocessing.

#include "modules/preprocess.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    // view_is_side: only an exact side x side view with real data passes.
    int dummy = 0;
    assert(vla::view_is_side(&dummy, 224, 224, 224));
    assert(!vla::view_is_side(nullptr, 224, 224, 224));
    assert(!vla::view_is_side(&dummy, 32, 32, 224));
    assert(!vla::view_is_side(&dummy, 224, 32, 224));
    assert(!vla::view_is_side(&dummy, 32, 224, 224));

    const uint8_t rgb[2*2*3] = {0, 51, 102, 153, 204, 255, 255, 0, 128, 64, 32, 16};
    const vla::ImageView iv{rgb, 2, 2};
    const float mean[3] = {0.485f, 0.456f, 0.406f}, std_[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> siglip, imnet;
    assert(vla::preprocess_image_chw("test", iv, 2, siglip));
    assert(vla::preprocess_image_chw("test", iv, 2, mean, std_, imnet));
    assert(siglip.size() == 12 && imnet.size() == 12);
    for (int64_t c = 0; c < 3; ++c)
        for (int64_t p = 0; p < 4; ++p) {
            const float px = rgb[p * 3 + c] / 255.0f;
            assert(siglip[c * 4 + p] == px * 2.0f - 1.0f);
            assert(imnet[c * 4 + p] == (px - mean[c]) / std_[c]);
        }
    assert(!vla::preprocess_image_chw("test", iv, 4, siglip));
    assert(!vla::preprocess_image_chw("test", vla::ImageView{nullptr, 2, 2}, 2, mean, std_, imnet));

    std::printf("test_vision_common: OK\n");
    return 0;
}
