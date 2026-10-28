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

// FoldQuant on ggml's SYCL backend (src/sycl/vla_sycl_foldquant.cpp).

#pragma once

#include <cstdint>

struct ggml_tensor;

namespace vla {

struct FqGemmSpec;

// The int32 sums [M][N] of a FoldQuant GEMM node's matmul (oneDNN), for a
// fused node that applies its own epilogue: w, blob, wscale as the gemm node's
// src[0..2]. Valid until the next FoldQuant GEMM on the queue; null when oneDNN
// is not in the build or not usable. *xs_off: the per-row scale's byte offset
// in a blob row.
int32_t * sycl_fq_gemm_acc(void * queue, const ggml_tensor * w, const ggml_tensor * blob, const ggml_tensor * ws,
                           const FqGemmSpec & s, int64_t * xs_off);

// Adds the FoldQuant kernels to the SYCL extension hook (sycl/vla_sycl_ext.h,
// scripts/patch_ggml_sycl_ext_hook.py). Idempotent; call once the SYCL backend is up.
void sycl_register_foldquant_ops();

}  // namespace vla
