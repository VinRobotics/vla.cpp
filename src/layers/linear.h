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

// Projections. A null bias emits no add node.

#pragma once

#include "ggml.h"

#include <cstddef>

namespace vla {

inline ggml_tensor * linear(ggml_context * C, ggml_tensor * W, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(C, W, x);
    return b ? ggml_add(C, y, b) : y;
}

// One row of a stacked [out, in, n_embodiment] weight: the GR00T action expert
// keeps a per-embodiment copy of every projection in one tensor.
inline ggml_tensor * cat_linear(ggml_context * C, ggml_tensor * W3d, ggml_tensor * b2d, int64_t id, ggml_tensor * x) {
    const int64_t out = W3d->ne[0];
    const int64_t in  = W3d->ne[1];

    ggml_tensor * W_id = ggml_view_2d(C, W3d, out, in, W3d->nb[1], (size_t)id*W3d->nb[2]);
    ggml_tensor * y    = ggml_mul_mat(C, ggml_cont(C, ggml_transpose(C, W_id)), x);
    return ggml_add(C, y, ggml_view_1d(C, b2d, out, (size_t)id*b2d->nb[1]));
}

// Slice block `blk` out of a fused [nblk*E, T] projection, laid out as heads.
inline ggml_tensor * head_view(ggml_context * C, ggml_tensor * proj, int64_t hd, int64_t heads,
                               int64_t T, int64_t E, int nblk, int blk) {
    const size_t es = ggml_element_size(proj);
    return ggml_view_3d(C, proj, hd, heads, T, (size_t)hd*es, (size_t)nblk*E*es, (size_t)blk*E*es);
}

}
