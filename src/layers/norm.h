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

#include "layers/linear.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace vla {

inline ggml_tensor * layer_norm(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    return ggml_add(C, ggml_mul(C, ggml_norm(C, x, eps), w), b);
}

inline ggml_tensor * rms_norm(ggml_context * C, ggml_tensor * x, ggml_tensor * w, float eps) {
    ggml_tensor * n = ggml_rms_norm(C, x, eps);
    return w ? ggml_mul(C, n, w) : n;
}

// cond is (scale, shift) here; the DiT final projection uses (shift, scale).
inline ggml_tensor * adaln(ggml_context * C, ggml_tensor * x, ggml_tensor * temb,
                           ggml_tensor * lw, ggml_tensor * lb, int64_t dim, float eps) {
    ggml_tensor * cond = linear(C, lw, lb, ggml_silu(C, temb));
    ggml_tensor * sc   = ggml_view_1d(C, cond, dim, 0);
    ggml_tensor * sh   = ggml_view_1d(C, cond, dim, (size_t)dim*sizeof(float));

    ggml_tensor * xn = ggml_norm(C, x, eps);
    return ggml_add(C, ggml_add(C, xn, ggml_mul(C, xn, sc)), sh);
}

}
