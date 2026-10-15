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

// sycl-tla flash attention on ggml's SYCL backend (src/sycl/vla_sycl_fmha.cpp).

#pragma once

struct ggml_tensor;

namespace vla {

// Runs a GGML_OP_FLASH_ATTN_EXT node with sycl-tla's fused kernel when its
// shape is one the kernel was checked on; false leaves the node to ggml-sycl.
bool sycl_fmha_forward(ggml_tensor * dst, void * queue);

// The cached-KV attention node of src/layers/fmha_kv.h: claim and run it.
bool sycl_fmha_kv_forward(ggml_tensor * dst, void * queue);
bool sycl_fmha_kv_supports(const ggml_tensor * op);

// vit_attn of src/layers/vit_fused.h (head_dim 72).
bool sycl_vit_attn_forward(ggml_tensor * dst, void * queue);

}  // namespace vla
