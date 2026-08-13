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


// nv == 1 emits the same nodes as the 2d/3d spelling, since ggml pads every
// shape to four dimensions.

#pragma once

#include "ggml.h"

#include <cstdint>

namespace vla {

inline ggml_tensor * to_heads(ggml_context * C, ggml_tensor * p, int64_t hd, int64_t heads,
                              int64_t T, int64_t nv = 1) {
    return ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, p, hd, heads, T, nv), 0, 2, 1, 3));
}

// V is pre-transposed so that mul_mat(V, aw) lands the right way round.
inline ggml_tensor * to_heads_v(ggml_context * C, ggml_tensor * p, int64_t hd, int64_t heads,
                                int64_t T, int64_t nv = 1) {
    return ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, p, hd, heads, T, nv), 1, 2, 0, 3));
}

inline ggml_tensor * attention(ggml_context * C, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
                               ggml_tensor * mask, float scale, int64_t dim, int64_t T, int64_t nv = 1) {
    ggml_tensor * kq = ggml_mul_mat(C, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    ggml_tensor * aw  = ggml_soft_max_ext(C, kq, mask, scale, 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(C, V, aw);
    return ggml_reshape_3d(C, ggml_cont(C, ggml_permute(C, kqv, 0, 2, 1, 3)), dim, T, nv);
}

// Takes V laid out like Q/K, not the transposed to_heads_v form.
inline ggml_tensor * flash_attention(ggml_context * C, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
                                     ggml_tensor * mask, float scale) {
    ggml_tensor * kf = K->type == GGML_TYPE_F16 ? K : ggml_cast(C, K, GGML_TYPE_F16);
    ggml_tensor * vf = V->type == GGML_TYPE_F16 ? V : ggml_cast(C, V, GGML_TYPE_F16);

    ggml_tensor * o = ggml_flash_attn_ext(C, Q, kf, vf, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    return ggml_reshape_2d(C, o, o->ne[0]*o->ne[1], o->ne[2]*o->ne[3]);
}

}
