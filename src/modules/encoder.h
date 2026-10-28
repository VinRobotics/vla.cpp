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

// Pre-LN encoder. The SigLIP vision blocks and the GR00T vision-language
// self-attention blocks are the same computation under different tensor names,
// which is what EncNames selects.

#pragma once

#include "layers/fused_ew.h"
#include "loader.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace vla {

struct EncBlockW {
    ggml_tensor *ln1w, *ln1b, *ln2w, *ln2b;
    ggml_tensor *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;
    ggml_tensor *Wfc1, *bfc1, *Wfc2, *bfc2;
    // cfg.fused: q|k|v weights and biases concatenated (Wq..bv above are null).
    ggml_tensor *Wqkv = nullptr, *bqkv = nullptr;
    // cfg.int8: the four GEMMs as W8A8 (layers/qlinear.h) in place of Wqkv, Wo,
    // Wfc1, Wfc2.
    WeightLoader::QWeight q_qkv, q_o, q_fc1, q_fc2;
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
    // Activation dtype for the GEMMs (mm_act); F32 is the classic graph. The
    // input must already be in this type (cast the patch embedding).
    ggml_type act = GGML_TYPE_F32;
    // Reduced-precision act on a backend that claims layers/fused_ew.h nodes:
    // GEMMs take bias, GELU and residual as epilogue, q|k|v is one GEMM, and
    // attention reads its output directly. Set before declare().
    bool fused = false;
    // With fused: the GEMMs as W8A8 with per-token INT8 activations.
    bool int8 = false;
    // F32 act on CUDA: layers/fused_ops.h nodes, bit-identical to this graph.
    bool fused_ops = false;
};

struct EncStack {
    EncCfg                 cfg;
    std::vector<EncBlockW> blk;
    FusedLNormSpec         ln_spec;  // cfg.fused; its address rides in the graph

    void declare(WeightLoader & L, const char * prefix, int64_t layers, const EncNames & n = EncNames{});

    ggml_tensor * block(ggml_context * C, const EncBlockW & w, ggml_tensor * x,
                        int64_t seq, int64_t nv = 1) const;

    ggml_tensor * build(ggml_context * C, ggml_tensor * x, int64_t seq, int64_t nv = 1) const;
};

}
