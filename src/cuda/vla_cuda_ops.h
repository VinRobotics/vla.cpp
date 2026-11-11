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

#pragma once

// Registration for the in-tree CUDA kernel families that ride the ggml
// extension hook: the BF16 activation ops (src/cuda/vla_cuda_bf16.cu) and the
// FoldQuant INT8/INT4 linears (src/cuda/vla_cuda_foldquant.cu). Both install
// through one dispatcher (src/cuda/vla_cuda_ext.cu) so they compose.
//
// Off every other build: without CUDA there is no hook to install and these
// paths are unreachable anyway, so this compiles to nothing.

#include "backend.h"
#include "layers/fused_ops.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vla {

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)
void cuda_register_bf16_ops();
void cuda_register_foldquant_ops();
bool cuda_register_fused_ops(const char * arch, int device);
#else
inline void cuda_register_bf16_ops() {}
inline void cuda_register_foldquant_ops() {}
inline bool cuda_register_fused_ops(const char *, int) { return false; }
#endif

// The fused CUDA nodes (layers/fused_ops.h) on a live CUDA backend, unless
// VLA_FUSED_OPS=0. Check device/kernel compatibility before emitting CUSTOM
// nodes: declining a CUSTOM node at execution time cannot unfuse the graph.
inline bool enable_fused_ops(const char * arch, bool is_cuda) {
    const char * fo = std::getenv("VLA_FUSED_OPS");
    if (!is_cuda || (fo && std::strcmp(fo, "0") == 0))
        return false;
    if (!cuda_register_fused_ops(arch, backend_device_index()))
        return false;
    fused_ops_active() = true;
    std::printf("vla(%s): fused CUDA ops on (VLA_FUSED_OPS=0 to disable)\n", arch);
    return true;
}

}  // namespace vla
