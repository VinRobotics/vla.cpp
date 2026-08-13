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

// Rotary embeddings. Three conventions live in the tree and they are not
// interchangeable; the pairing is pinned by tests/test_rope_conventions.cpp.
//
//   RopeSpec       ggml rope, NEOX or IMROPE. The Qwen3 backbones.
//   rope_2d        precomputed tables, half-split rotation. The Qwen3-VL tower,
//                  whose 2d grid positions ggml_rope has no spelling for.
//   rope_pairwise  precomputed tables, adjacent-pair rotation. VLA-Adapter's
//                  action head pairs a half-split frequency table with an
//                  interleaved rotation; the reference does the same, so the
//                  mismatch is deliberate.

#pragma once

#include "ggml.h"

#include <cstdint>

namespace vla {

// Which ggml rope call a backbone wants, and with what parameters. `sections`
// is read only when type is GGML_ROPE_TYPE_IMROPE.
struct RopeSpec {
    int   type       = GGML_ROPE_TYPE_NEOX;
    int   n_dims     = 0;
    int   sections[4]= {0, 0, 0, 0};
    float freq_base  = 10000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor= 1.0f;
    float beta_fast  = 32.0f;
    float beta_slow  = 1.0f;
};

inline ggml_tensor * rope(ggml_context * C, const RopeSpec & r, ggml_tensor * x, ggml_tensor * pos) {
    if (r.type == GGML_ROPE_TYPE_IMROPE) {
        int sect[4] = { r.sections[0], r.sections[1], r.sections[2], r.sections[3] };
        return ggml_rope_multi(C, x, pos, nullptr, r.n_dims, sect, r.type, 0,
                               r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow);
    }
    return ggml_rope_ext(C, x, pos, nullptr, r.n_dims, r.type, 0,
                         r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow);
}

// Half-split rotation against precomputed tables: (x1, x2) -> (-x2, x1).
inline ggml_tensor * rope_2d(ggml_context * C, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_t) {
    const int64_t hd = x->ne[0];
    const int64_t S  = x->ne[1];
    const int64_t Hh = x->ne[2];
    const int64_t half = hd/2;

    ggml_tensor * x1  = ggml_cont(C, ggml_view_3d(C, x, half, S, Hh, x->nb[1], x->nb[2], 0));
    ggml_tensor * x2  = ggml_cont(C, ggml_view_3d(C, x, half, S, Hh, x->nb[1], x->nb[2], (size_t)half*x->nb[0]));
    ggml_tensor * rot = ggml_concat(C, ggml_neg(C, x2), x1, 0);
    return ggml_add(C, ggml_mul(C, x, cos_t), ggml_mul(C, rot, sin_t));
}

// Adjacent-pair rotation: (even, odd) -> (-odd, even).
inline ggml_tensor * rope_pairwise_rot(ggml_context * C, ggml_tensor * x, int64_t HD) {
    const int64_t L = x->ne[1];
    const int64_t H = x->ne[2];

    ggml_tensor * xp = ggml_reshape_4d(C, x, 2, HD/2, L, H);
    ggml_tensor * ev = ggml_cont(C, ggml_view_4d(C, xp, 1, HD/2, L, H, xp->nb[1], xp->nb[2], xp->nb[3], 0));
    ggml_tensor * od = ggml_cont(C, ggml_view_4d(C, xp, 1, HD/2, L, H, xp->nb[1], xp->nb[2], xp->nb[3], xp->nb[0]));
    return ggml_reshape_3d(C, ggml_concat(C, ggml_scale(C, od, -1.0f), ev, 0), HD, L, H);
}

inline ggml_tensor * rope_pairwise(ggml_context * C, ggml_tensor * x, ggml_tensor * cs, ggml_tensor * sn, int64_t HD) {
    ggml_tensor * c = ggml_reshape_3d(C, cs, HD, x->ne[1], 1);
    ggml_tensor * s = ggml_reshape_3d(C, sn, HD, x->ne[1], 1);
    return ggml_add(C, ggml_mul(C, x, c), ggml_mul(C, rope_pairwise_rot(C, x, HD), s));
}

}
