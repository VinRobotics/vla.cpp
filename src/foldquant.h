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

// FoldQuant: real-quantized linears exported by VLA-OPT (docs/QUANTIZATION.md).
//
// A FoldQuant site ships `<site>.weight` as GGML_TYPE_I8 codes (per-output-row
// symmetric, in a block-Hadamard-rotated frame), `<site>.wscale` F32[N], and
// optionally `<site>.ascale` F32[K] (the static SmoothQuant vector). At runtime
// the activation is [RMS-normed,] [divided,] rotated by the same butterfly,
// quantized per token to INT8, multiplied on integer units and dequantized.
//
// Both backends run the same two GGML_OP_CUSTOM nodes per site (fq_act,
// fq_gemm; see layers/fq_linear.h): the CPU backend executes the reference in
// foldquant_ref.h, the CUDA backend claims them through the ggml extension hook.

#pragma once

#include "ggml.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vla {

struct gguf_reader;
struct Backend;
class  WeightLoader;

constexpr uint32_t FQ_ACT_MAGIC  = 0x31414651u;  // "FQA1"
constexpr uint32_t FQ_GEMM_MAGIC = 0x31474651u;  // "FQG1"

// Bytes after the codes in every row of the activation blob: the per-token
// float scale sits at byte K_pack, the rest is padding that keeps rows 16-byte
// aligned (K is a multiple of 64 wherever a site is accepted).
constexpr int64_t FQ_ACT_TAIL = 16;

// Per-module scheme parameters parsed from `<arch>.quant.*`.
struct FqModuleSpec {
    int   wbits       = 8;      // 8 | 4
    int   abits       = 8;      // 8 | 4
    int   rot_block   = 64;     // nominal; narrowed per site by fq_rot_block_for()
    bool  fold_before = false;  // ascale divides before (true) or after the butterfly
    float clip        = 1.0f;   // act_clip_ratio (INT4 activations only)
    std::map<std::string, int> site_bits;  // per-site override, e.g. {"o":8,"down":8}
    std::string scheme;

    int wbits_for(const char * site_key) const;
    int abits_for(const char * site_key) const;
};

struct FoldQuantSpec {
    bool         present = false;
    FqModuleSpec llm, action;
    std::string  method, applied_at, provenance;
};

// Static per-node parameters. Graph nodes reference them by pointer (custom-op
// userdata), so they must outlive every graph: they live inside the module
// weight structs, which are sized once at declare time and never resized.
struct FqActSpec {
    uint32_t magic       = FQ_ACT_MAGIC;
    int64_t  K           = 0;
    int      abits       = 8;
    int      rot_block   = 1;      // 1 = no rotation
    bool     fold_before = false;
    bool     has_gamma   = false;  // fused RMSNorm with the folded gamma
    bool     has_ascale  = false;  // static SmoothQuant vector shipped with the site
    float    clip        = 1.0f;
    float    eps         = 0.0f;   // RMSNorm epsilon when has_gamma
};

struct FqGemmSpec {
    uint32_t magic = FQ_GEMM_MAGIC;
    int64_t  K     = 0;
    int64_t  N     = 0;
    int      wbits = 8;
};

struct FqLinear {
    ggml_tensor * w      = nullptr;  // I8  [K_pack, N]; null => not a FoldQuant site
    ggml_tensor * wscale = nullptr;  // F32 [N]
    ggml_tensor * ascale = nullptr;  // F32 [K] or null
    ggml_tensor * gamma  = nullptr;  // F32 [K] folded RMSNorm gamma or null
    ggml_tensor * bias   = nullptr;  // F32 [N] or null
    FqActSpec     act;
    FqGemmSpec    gemm;

    explicit operator bool() const { return w != nullptr; }
};

// Activation blob layout shared by the CPU reference, the CUDA kernels and the
// tests: I8 tensor, ne = [row_bytes, T]; row t = codes[0, K_pack) then float scale.
inline int64_t fq_act_kpack(int64_t K, int abits)     { return abits == 4 ? K / 2 : K; }
inline int64_t fq_act_row_bytes(int64_t K, int abits) { return fq_act_kpack(K, abits) + FQ_ACT_TAIL; }
inline int64_t fq_w_kpack(int64_t K, int wbits)       { return wbits == 4 ? K / 2 : K; }

// Largest power of two <= nominal that divides K (VLA-OPT foldq.rotation_block_for);
// 1 means no rotation.
int fq_rot_block_for(int64_t K, int nominal);

// `<prefix>.quant.method` == "foldquant". Never asserts on a wrong KV type.
bool          foldquant_present(const gguf_reader & g, const char * prefix);
FoldQuantSpec foldquant_parse  (const gguf_reader & g, const char * prefix);

// Load-time policy: CUDA (registers the kernels) or CPU. Everything else is
// refused: GGML_OP_CUSTOM has no implementation there and the core drives one
// backend with no per-op fallback.
bool foldquant_check_backend(const char * tag, const Backend & b, const FoldQuantSpec & fq, bool weight_dtype_set);

// Declares `<site>.weight` (I8), `.wscale`, optional `.ascale` and optional
// `.bias`. Returns an empty FqLinear (w == nullptr) when `<site>.weight` is not
// I8, so the caller falls back to its stock declare. `gamma` is the already
// declared folded norm weight fused into the activation node, or null.
FqLinear fq_declare_linear(WeightLoader & L, const FqModuleSpec & mod, const char * site_key,
                           bool has_bias, ggml_tensor * gamma, float eps,
                           const char * site_fmt, ...) __attribute__((format(printf, 7, 8)));

// Several sites sharing one input transform (DiT q/k/v, k/v) fused into one
// weight: codes and wscale concatenate along N; ascale must agree and is kept once.
FqLinear fq_declare_fused(WeightLoader & L, const FqModuleSpec & mod, const char * site_key,
                          bool has_bias, const std::string & out_base, const std::vector<std::string> & sites);

}  // namespace vla
