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

// A block cross-attends when `enc` is non-null and self-attends otherwise; the
// per-layer choice is the model's. Cross-attention K/V depend only on `enc`, so
// a model hoists them out of the solver loop and passes them via K_pre/V_pre.

#pragma once

#include "loader.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace vla {

struct DitLayerW {
    ggml_tensor *adaln_w, *adaln_b;
    ggml_tensor *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;
    ggml_tensor *Wff0, *bff0, *Wff2, *bff2;

    ggml_tensor *Wqkv = nullptr, *bqkv = nullptr, *Wkv = nullptr, *bkv = nullptr;
};

struct DitCfg {
    int64_t hidden       = 1536;
    int64_t heads        = 32;
    int64_t head_dim     = 48;
    int64_t layers       = 16;
    float   ln_eps       = 1e-5f;
    float   norm_out_eps = 1e-6f;
};

struct DitHead {
    DitCfg                 cfg;
    std::vector<DitLayerW> blk;
    ggml_tensor *te_l1W = nullptr, *te_l1b = nullptr, *te_l2W = nullptr, *te_l2b = nullptr;
    ggml_tensor *po1W   = nullptr, *po1b   = nullptr, *po2W   = nullptr, *po2b   = nullptr;

    // outer names time_emb and proj_out when they do not sit under the block
    // prefix; null means they do.
    void declare(WeightLoader & L, const char * prefix, bool fuse_qkv = false, bool interleave = false,
                 const char * outer = nullptr);

    void kv(ggml_context * C, const DitLayerW & w, ggml_tensor * src,
            ggml_tensor ** K_out, ggml_tensor ** V_out) const;

    ggml_tensor * block(ggml_context * C, const DitLayerW & w, ggml_tensor * h, ggml_tensor * temb,
                        ggml_tensor * enc, ggml_tensor * K_pre = nullptr, ggml_tensor * V_pre = nullptr) const;

    ggml_tensor * time_emb(ggml_context * C, ggml_tensor * tproj) const;

    // (shift, scale) adaLN, opposite to layers/norm.h adaln.
    ggml_tensor * proj_out(ggml_context * C, ggml_tensor * h, ggml_tensor * temb) const;
};

}
