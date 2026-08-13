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

// Prefill only: a VLA runs one pass per action chunk, so there is no KV cache.

#pragma once

#include "layers/rope.h"
#include "loader.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace vla {

struct Qwen3LayerW {
    ggml_tensor *attn_norm, *Wq, *Wk, *Wv, *Wo, *q_norm, *k_norm, *ffn_norm, *Wgate, *Wup, *Wdown;
};

struct Qwen3Cfg {
    int64_t  hidden   = 2048;
    int64_t  layers   = 16;
    int64_t  n_q      = 16;
    int64_t  n_kv     = 8;
    int64_t  head_dim = 128;
    int64_t  inter    = 6144;
    float    rms_eps  = 1e-6f;
    RopeSpec rope;
    bool     flash_attn = false;
};

struct Qwen3LM {
    Qwen3Cfg                 cfg;
    std::vector<Qwen3LayerW> blk;
    ggml_tensor *            output_norm = nullptr;

    void declare(WeightLoader & L, const char * prefix);

    ggml_tensor * block(ggml_context * C, const Qwen3LayerW & w, ggml_tensor * h,
                        ggml_tensor * pos, ggml_tensor * mask, int64_t seq) const;

    ggml_tensor * build(ggml_context * C, ggml_tensor * h,
                        ggml_tensor * pos, ggml_tensor * mask, int64_t seq) const;
};

}
