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

// SigLIP vision tower: learned position embeddings, post-LN encoder blocks, a
// final LayerNorm, and no CLS token.
//
// The patch embedding has two spellings. GR00T N1.5, pi0, pi0.5 and SmolVLA
// feed a CHW image through a conv2d; N1.6 pre-patchifies on the host and feeds
// a GEMM. Both land on the same [hidden, patches] activation.

#pragma once

#include "modules/encoder.h"

#include "ggml.h"

#include <cstdint>

namespace vla {

struct SigLipTower {
    EncStack     enc;
    ggml_tensor *patch_w   = nullptr, *patch_b   = nullptr;
    ggml_tensor *pos       = nullptr;
    ggml_tensor *post_ln_w = nullptr, *post_ln_b = nullptr;

    // Reads "<prefix>.patch_embd.*", "<prefix>.pos_embd", "<prefix>.post_ln.*"
    // and "<prefix>.blk.<i>.*".
    void declare(WeightLoader & L, const char * prefix, int64_t layers, bool patch_embd_is_gemm = false);

    ggml_tensor * embed_conv(ggml_context * C, ggml_tensor * pixels, int64_t patch, int64_t grid) const;
    ggml_tensor * embed_patches(ggml_context * C, ggml_tensor * patches) const;

    // Encoder blocks, then the final LayerNorm.
    ggml_tensor * build(ggml_context * C, ggml_tensor * h, int64_t seq, int64_t nv = 1) const;
};

}
