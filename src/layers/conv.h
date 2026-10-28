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

namespace vla {

// x [W, H, C, N] with w [KW, KH, C, OC] to [OW, OH, OC, N], no bias, as im2col
// and one F32 GEMM. The patches are the GEMM's first operand, so the result
// lands as [OW*OH*N, OC]: one image is already in the input layout, a batch is
// one permute away from it.
inline ggml_tensor * conv_2d_f32(ggml_context * C, ggml_tensor * w, ggml_tensor * x, int stride, int pad) {
    ggml_tensor * col = ggml_im2col(C, w, x, stride, stride, pad, pad, 1, 1, true, GGML_TYPE_F32);
    ggml_tensor * y   = ggml_mul_mat(C,
        ggml_reshape_2d(C, col, col->ne[0], col->ne[3]*col->ne[2]*col->ne[1]),
        ggml_reshape_2d(C, w, w->ne[0]*w->ne[1]*w->ne[2], w->ne[3]));
    if (col->ne[3] == 1)
        return ggml_reshape_4d(C, y, col->ne[1], col->ne[2], w->ne[3], 1);
    y = ggml_reshape_4d(C, y, col->ne[1], col->ne[2], col->ne[3], w->ne[3]);
    return ggml_cont(C, ggml_permute(C, y, 0, 1, 3, 2));
}

}
