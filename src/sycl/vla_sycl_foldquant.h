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

namespace vla {

// Points the SYCL extension hook (scripts/patch_ggml_sycl_ext_hook.py) at the
// FoldQuant kernels. Idempotent; call once the SYCL backend is up.
void sycl_register_foldquant_ops();

}  // namespace vla
