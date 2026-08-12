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

// Pins the two rotary conventions so a cleanup cannot swap one for the other.
//
// VLA-Adapter's action head pairs an interleaved rotation with a half-split
// frequency table, so a rotation pair gets two angles. The reference does the same
// (action_heads.py:163 vs :137-140) and the weights were trained on it, so making
// it self-consistent would break the shipped checkpoints.

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// src/models/vla_adapter.cpp hrot(): out[2k] = -x[2k+1], out[2k+1] = x[2k].
std::vector<float> rotate_interleaved(const std::vector<float> & x) {
    const size_t hd = x.size();
    std::vector<float> out(hd);
    for (size_t k = 0; k < hd / 2; ++k) {
        out[2 * k]     = -x[2 * k + 1];
        out[2 * k + 1] =  x[2 * k];
    }
    return out;
}

// HuggingFace rotate_half, the NeoX convention ggml_rope_ext implements.
std::vector<float> rotate_half(const std::vector<float> & x) {
    const size_t hd = x.size(), half = hd / 2;
    std::vector<float> out(hd);
    for (size_t i = 0; i < half; ++i) {
        out[i]        = -x[half + i];
        out[half + i] =  x[i];
    }
    return out;
}

// src/models/vla_adapter.cpp fill_cs(): j = mi % half, inv = base^(-2j/HD).
double adapter_angle(size_t mi, size_t hd, double base, double t) {
    const size_t j = mi % (hd / 2);
    return t * (1.0 / std::pow(base, (2.0 * (double) j) / (double) hd));
}

}  // namespace

int main() {
    const size_t hd   = 8;
    const double base = 10000.0;
    const double t    = 3.0;

    const std::vector<float> x = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // 1. The two rotations differ, so a swap is observable.
    const std::vector<float> ri = rotate_interleaved(x);
    const std::vector<float> rh = rotate_half(x);
    assert(ri != rh);
    assert(ri[0] == -2.0f && ri[1] == 1.0f);   // interleaved pairs (x0,x1)
    assert(rh[0] == -5.0f && rh[4] == 1.0f);   // half-split pairs  (x0,x4)

    // 2. The adapter frequency table is half-split: index i and i+half share an
    //    angle. That is what makes it mismatch the interleaved rotation above.
    for (size_t i = 0; i < hd / 2; ++i) {
        assert(adapter_angle(i, hd, base, t) == adapter_angle(i + hd / 2, hd, base, t));
    }

    // 3. The mismatch: an interleaved pair (2k, 2k+1) does not share an angle
    //    under this table. Pinned on purpose, see the header.
    bool any_pair_differs = false;
    for (size_t k = 0; k < hd / 2; ++k) {
        if (adapter_angle(2 * k, hd, base, t) != adapter_angle(2 * k + 1, hd, base, t)) {
            any_pair_differs = true;
        }
    }
    assert(any_pair_differs);

    // 4. An interleaved table (j = mi / 2) would make every pair agree. That is
    //    the fix that must not be applied.
    for (size_t k = 0; k < hd / 2; ++k) {
        const auto interleaved_angle = [&](size_t mi) {
            return t * (1.0 / std::pow(base, (2.0 * (double) (mi / 2)) / (double) hd));
        };
        assert(interleaved_angle(2 * k) == interleaved_angle(2 * k + 1));
    }

    std::printf("rope conventions: ok\n");
    return 0;
}
