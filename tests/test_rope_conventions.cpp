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

#include "layers/rope.h"

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

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

std::vector<float> run_rot(const std::vector<float> & x) {
    ggml_init_params ip = { (size_t) 16 * 1024 * 1024, nullptr, false };
    ggml_context * C = ggml_init(ip);
    const int64_t hd = (int64_t) x.size();
    ggml_tensor * t = ggml_new_tensor_3d(C, GGML_TYPE_F32, hd, 1, 1);
    for (int64_t i = 0; i < hd; ++i) ggml_set_f32_1d(t, (int) i, x[i]);
    ggml_tensor * r = vla::rope_pairwise_rot(C, t, hd);
    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(C, gf, 1);
    std::vector<float> out(hd);
    for (int64_t i = 0; i < hd; ++i) out[i] = ggml_get_f32_1d(r, (int) i);
    ggml_free(C);
    return out;
}

}  // namespace

int main() {
    const int64_t hd   = 8;
    const float   base = 10000.0f;
    const int64_t T    = 4;

    const std::vector<float> x = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // 1. The two rotations differ, so a swap is observable.
    const std::vector<float> ri = run_rot(x);
    const std::vector<float> rh = rotate_half(x);
    assert(ri != rh);
    for (int64_t k = 0; k < hd / 2; ++k) {
        assert(ri[2 * k] == -x[2 * k + 1] && ri[2 * k + 1] == x[2 * k]);   // interleaved pairs (x0,x1)
    }
    assert(rh[0] == -5.0f && rh[4] == 1.0f);   // half-split pairs  (x0,x4)

    // 2. The adapter frequency table is half-split: index i and i+half share an
    //    angle. That is what makes it mismatch the interleaved rotation above.
    std::vector<float> cs, sn;
    vla::rope_pairwise_table(hd, T, base, cs, sn);
    assert(cs.size() == (size_t) (hd * T) && sn.size() == cs.size());
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < hd / 2; ++i) {
            assert(cs[t * hd + i] == cs[t * hd + i + hd / 2]);
            assert(sn[t * hd + i] == sn[t * hd + i + hd / 2]);
        }
    }
    assert(cs[3 * hd + 1] == (float) std::cos(3.0 * (1.0 / std::pow(10000.0, 2.0 / 8.0))));

    // 3. The mismatch: an interleaved pair (2k, 2k+1) does not share an angle
    //    under this table. Pinned on purpose, see the header.
    bool any_pair_differs = false;
    for (int64_t k = 0; k < hd / 2; ++k) {
        if (sn[1 * hd + 2 * k] != sn[1 * hd + 2 * k + 1]) {
            any_pair_differs = true;
        }
    }
    assert(any_pair_differs);

    std::printf("rope conventions: ok\n");
    return 0;
}
