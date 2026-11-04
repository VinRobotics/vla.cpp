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


#pragma once

#include "ggml.h"

#include <cstddef>

namespace vla {

inline ggml_tensor * linear(ggml_context * C, ggml_tensor * W, ggml_tensor * b, ggml_tensor * x) {
#ifdef GGML_USE_HEXAGON
    // Hexagon computes a 2D weight against a batch of activations wrong, and the
    // fallback wrapper runs that on the CPU; the same rows as one 2D matmul stay
    // on the NPU.
    if (W->ne[2] * W->ne[3] == 1 && x->ne[2] * x->ne[3] > 1 && ggml_is_contiguous(x)) {
        ggml_tensor * y = ggml_mul_mat(C, W, ggml_reshape_2d(C, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]));
        if (b)
            y = ggml_add(C, y, b);
        return ggml_reshape_4d(C, y, W->ne[1], x->ne[1], x->ne[2], x->ne[3]);
    }
#endif
    ggml_tensor * y = ggml_mul_mat(C, W, x);
    return b ? ggml_add(C, y, b) : y;
}

// Block `blk` of a fused [nblk*E, T] projection, laid out as heads.
inline ggml_tensor * head_view(ggml_context * C, ggml_tensor * proj, int64_t hd, int64_t heads,
                               int64_t T, int64_t E, int nblk, int blk) {
    const size_t es = ggml_element_size(proj);
    return ggml_view_3d(C, proj, hd, heads, T, (size_t)hd*es, (size_t)nblk*E*es, (size_t)blk*E*es);
}

}
