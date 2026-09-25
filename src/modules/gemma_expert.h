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

#include "foldquant.h"
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
    // FoldQuant sites (pi0.5 prefix tower). The RMSNorm and its folded gamma
    // (ln_in / ln_post, loaded as 1 + w) ride in the q/k/v and gate/up act nodes.
    FqLinear fq_q, fq_k, fq_v, fq_o, fq_gate, fq_up, fq_down;
};

struct GemmaStack {
    std::vector<GemmaLayerW> blk;
    ggml_tensor *            output_norm = nullptr;

    // fq: the LLM FoldQuant spec when the GGUF carries one (pi0.5), else null.
    void declare(WeightLoader & L, const char * prefix, int64_t layers, bool with_output_norm,
                 const FqModuleSpec * fq = nullptr, float rms_eps = 1e-6f) {
        blk.resize(layers);
        for (int64_t i=0; i<layers; ++i) {
            GemmaLayerW & w = blk[i];
            const long long ii = (long long) i;
            w.ln_in   = L.f32_gemma_norm("%s.blk.%lld.attn_norm.weight", prefix, ii);
            w.ln_post = L.f32_gemma_norm("%s.blk.%lld.ffn_norm.weight",  prefix, ii);
            if (fq) {
                w.fq_q    = fq_declare_linear(L, *fq, "qkv",    false, w.ln_in,   rms_eps, "%s.blk.%lld.attn_q",   prefix, ii);
                w.fq_k    = fq_declare_linear(L, *fq, "qkv",    false, w.ln_in,   rms_eps, "%s.blk.%lld.attn_k",   prefix, ii);
                w.fq_v    = fq_declare_linear(L, *fq, "qkv",    false, w.ln_in,   rms_eps, "%s.blk.%lld.attn_v",   prefix, ii);
                w.fq_o    = fq_declare_linear(L, *fq, "o",      false, nullptr,   0.0f,    "%s.blk.%lld.attn_o",   prefix, ii);
                w.fq_gate = fq_declare_linear(L, *fq, "gateup", false, w.ln_post, rms_eps, "%s.blk.%lld.ffn_gate", prefix, ii);
                w.fq_up   = fq_declare_linear(L, *fq, "gateup", false, w.ln_post, rms_eps, "%s.blk.%lld.ffn_up",   prefix, ii);
                w.fq_down = fq_declare_linear(L, *fq, "down",   false, nullptr,   0.0f,    "%s.blk.%lld.ffn_down", prefix, ii);
                if (!w.fq_q != !w.fq_k || !w.fq_q != !w.fq_v || !w.fq_gate != !w.fq_up)
                    L.fail("FoldQuant: a Gemma layer's q/k/v (and gate/up) must all be INT or all float");
            }
            if (!w.fq_q) {
                w.Wq  = L.gemm("%s.blk.%lld.attn_q.weight", prefix, ii);
                w.Wk  = L.gemm("%s.blk.%lld.attn_k.weight", prefix, ii);
                w.Wv  = L.gemm("%s.blk.%lld.attn_v.weight", prefix, ii);
            }
            if (!w.fq_o)
                w.Wo  = L.gemm("%s.blk.%lld.attn_o.weight", prefix, ii);
            if (!w.fq_gate) {
                w.Wgate = L.gemm("%s.blk.%lld.ffn_gate.weight", prefix, ii);
                w.Wup   = L.gemm("%s.blk.%lld.ffn_up.weight",   prefix, ii);
            }
            if (!w.fq_down)
                w.Wdown = L.gemm("%s.blk.%lld.ffn_down.weight", prefix, ii);
        }
        if (with_output_norm)
            output_norm = L.f32_gemma_norm("%s.output_norm.weight", prefix);
    }
};

}
