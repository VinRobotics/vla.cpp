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

// Feed-forward blocks.

#pragma once

#include "layers/linear.h"

#include "ggml.h"

namespace vla {

// tanh-approximate GELU.
inline ggml_tensor * ffn_gelu(ggml_context * C, ggml_tensor * W1, ggml_tensor * b1,
                              ggml_tensor * W2, ggml_tensor * b2, ggml_tensor * x) {
    return linear(C, W2, b2, ggml_gelu(C, linear(C, W1, b1, x)));
}

// Exact erf GELU, what DINOv2 and SigLIP-so400m were trained with.
inline ggml_tensor * ffn_gelu_erf(ggml_context * C, ggml_tensor * W1, ggml_tensor * b1,
                                  ggml_tensor * W2, ggml_tensor * b2, ggml_tensor * x) {
    return linear(C, W2, b2, ggml_gelu_erf(C, linear(C, W1, b1, x)));
}

// down(silu(gate(x)) * up(x)).
inline ggml_tensor * ffn_swiglu(ggml_context * C, ggml_tensor * Wg, ggml_tensor * bg,
                                ggml_tensor * Wu, ggml_tensor * bu,
                                ggml_tensor * Wd, ggml_tensor * bd, ggml_tensor * x) {
    ggml_tensor * gate = ggml_silu(C, linear(C, Wg, bg, x));
    ggml_tensor * up   = linear(C, Wu, bu, x);
    return linear(C, Wd, bd, ggml_mul(C, gate, up));
}

}
