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

// Post-LN transformer encoder: full self-attention with biases, a GELU MLP, and
// a LayerNorm before each residual. The SigLIP vision blocks and the GR00T
// vision-language self-attention blocks are the same computation under
// different tensor names, so both declare through EncNames.

#pragma once

#include "loader.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace vla {

struct EncBlockW {
    ggml_tensor *ln1w, *ln1b, *ln2w, *ln2b;
    ggml_tensor *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;
    ggml_tensor *Wfc1, *bfc1, *Wfc2, *bfc2;
};

struct EncNames {
    const char * ln1 = "ln1";
    const char * ln2 = "ln2";
    const char * fc1 = "fc1";
    const char * fc2 = "fc2";
};

struct EncCfg {
    int64_t hidden     = 0;
    int64_t heads      = 0;
    int64_t head_dim   = 0;
    float   ln_eps     = 1e-6f;
    bool    flash_attn = false;
};

struct EncStack {
    EncCfg                 cfg;
    std::vector<EncBlockW> blk;

    // Reads "<prefix>.<i>.*". SigLIP checkpoints prefix their blocks with
    // "blk", so callers pass e.g. "vit.blk".
    void declare(WeightLoader & L, const char * prefix, int64_t layers, const EncNames & n = EncNames{});

    ggml_tensor * block(ggml_context * C, const EncBlockW & w, ggml_tensor * x,
                        int64_t seq, int64_t nv = 1) const;

    ggml_tensor * build(ggml_context * C, ggml_tensor * x, int64_t seq, int64_t nv = 1) const;
};

}
