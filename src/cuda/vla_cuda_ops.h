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

// Registration for the in-tree BF16 CUDA kernels (src/cuda/vla_cuda_bf16.cu).
//
// Off every other build: without CUDA there is no hook to install and the BF16
// activation path is unreachable anyway, so this compiles to nothing.

namespace vla {

#ifdef GGML_USE_CUDA
void cuda_register_bf16_ops();
#else
inline void cuda_register_bf16_ops() {}
#endif

}  // namespace vla
