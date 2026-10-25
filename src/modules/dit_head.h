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

#include "foldquant.h"
#include "layers/fused_ew.h"
#include "loader.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace vla {

struct DitLayerW {
    ggml_tensor *adaln_w, *adaln_b;
    ggml_tensor *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;
    ggml_tensor *Wff0, *bff0, *Wff2, *bff2;

    ggml_tensor *Wqkv = nullptr, *bqkv = nullptr, *Wkv = nullptr, *bkv = nullptr;

    // FoldQuant sites (docs/QUANTIZATION.md); empty where the file ships a float
    // GEMM. fq_qkv / fq_kv are the fused forms of fq_q/fq_k/fq_v.
    FqLinear fq_q, fq_k, fq_v, fq_o, fq_ff0, fq_ff2, fq_qkv, fq_kv;
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
    // The fused graph (SYCL, BF16 weights, view-taking flash attention, no
    // FoldQuant): each block as 7 nodes (layers/fused_ew.h) - the adaLN as a
    // LayerNorm whose weight and bias are 1 + scale and shift (FlowTimes keeps
    // 1 + scale), attention reading the q|k|v and K/V GEMM outputs in place,
    // the residual adds and the GELU as GEMM epilogues. Set before FlowTimes::build.
    bool                   fused = false;
    FusedLNormSpec         ln_spec, out_ln_spec;
    ggml_tensor *te_l1W = nullptr, *te_l1b = nullptr, *te_l2W = nullptr, *te_l2b = nullptr;
    ggml_tensor *po1W   = nullptr, *po1b   = nullptr, *po2W   = nullptr, *po2b   = nullptr;

    // outer names time_emb and proj_out when they do not sit under the block
    // prefix; null means they do.
    // fq: the action module's FoldQuant parameters when the GGUF carries them.
    void declare(WeightLoader & L, const char * prefix, bool fuse_qkv = false, bool interleave = false,
                 const char * outer = nullptr, const FqModuleSpec * fq = nullptr);

    // xq_pre: an activation blob already computed from src (shared with q).
    void kv(ggml_context * C, const DitLayerW & w, ggml_tensor * src,
            ggml_tensor ** K_out, ggml_tensor ** V_out, ggml_tensor * xq_pre = nullptr) const;

    ggml_tensor * block(ggml_context * C, const DitLayerW & w, ggml_tensor * h, ggml_tensor * mod,
                        ggml_tensor * enc, ggml_tensor * K_pre = nullptr, ggml_tensor * V_pre = nullptr) const;

    ggml_tensor * time_emb(ggml_context * C, ggml_tensor * tproj) const;

    // (shift, scale), opposite to the blocks.
    ggml_tensor * proj_out(ggml_context * C, ggml_tensor * h, ggml_tensor * mod) const;
};

struct Backend;
struct Options;

// Sets dit.fused where the fused graph runs: SYCL, BF16-resident weights,
// view-taking flash attention, no FoldQuant (fq), --flash-attn not 0, and
// VLA_GR00T_NO_FUSE unset. Call after cfg is read and before FlowTimes::build.
bool dit_enable_fused(DitHead & dit, const Backend & b, ggml_type weight_type, bool fq, const Options & opts,
                      const char * tag);

struct FlowTimes {
    std::vector<ggml_tensor *> tau;

    FlowTimes() = default;
    FlowTimes(const FlowTimes &) = delete;
    FlowTimes & operator=(const FlowTimes &) = delete;
    ~FlowTimes();

    bool build(const char * arch, ggml_backend_t backend, const DitHead & dit,
               int64_t steps, int64_t buckets, int64_t embed_dim, int64_t horizon);

    ggml_tensor * mod(ggml_context * C, int64_t s, int64_t i) const;

private:
    ggml_tensor *         mods = nullptr;
    int64_t               per_step = 0;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

}
