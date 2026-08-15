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

// Gemma decoder stack. pi0, pi0.5 and SmolVLA each run two of these with a
// shared attention: a prefix tower over the image and language tokens, and an
// action expert over the state and noisy-action tokens.

#pragma once

#include "loader.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace vla {

struct GemmaLayerW {
    ggml_tensor * ln_in   = nullptr;
    ggml_tensor * Wq      = nullptr;
    ggml_tensor * Wk      = nullptr;
    ggml_tensor * Wv      = nullptr;
    ggml_tensor * Wo      = nullptr;
    ggml_tensor * ln_post = nullptr;
    ggml_tensor * Wgate   = nullptr;
    ggml_tensor * Wup     = nullptr;
    ggml_tensor * Wdown   = nullptr;
};

struct GemmaStack {
    std::vector<GemmaLayerW> blk;
    ggml_tensor *            output_norm = nullptr;

    void declare(WeightLoader & L, const char * prefix, int64_t layers, bool with_output_norm) {
        blk.resize(layers);
        for (int64_t i=0; i<layers; ++i) {
            GemmaLayerW & w = blk[i];
            w.ln_in   = L.f32_gemma_norm("%s.blk.%lld.attn_norm.weight", prefix, (long long)i);
            w.Wq      = L.gemm          ("%s.blk.%lld.attn_q.weight",    prefix, (long long)i);
            w.Wk      = L.gemm          ("%s.blk.%lld.attn_k.weight",    prefix, (long long)i);
            w.Wv      = L.gemm          ("%s.blk.%lld.attn_v.weight",    prefix, (long long)i);
            w.Wo      = L.gemm          ("%s.blk.%lld.attn_o.weight",    prefix, (long long)i);
            w.ln_post = L.f32_gemma_norm("%s.blk.%lld.ffn_norm.weight",  prefix, (long long)i);
            w.Wgate   = L.gemm          ("%s.blk.%lld.ffn_gate.weight",  prefix, (long long)i);
            w.Wup     = L.gemm          ("%s.blk.%lld.ffn_up.weight",    prefix, (long long)i);
            w.Wdown   = L.gemm          ("%s.blk.%lld.ffn_down.weight",  prefix, (long long)i);
        }
        if (with_output_norm)
            output_norm = L.f32_gemma_norm("%s.output_norm.weight", prefix);
    }
};

}
