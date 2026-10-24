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

// The one owner of ggml-sycl's extension hook (scripts/patch_ggml_sycl_ext_hook.py).
//
// The hook is a single pair of function pointers, but several in-tree
// extensions (FoldQuant, the BF16 ops, the fused ops) may be needed by one
// model. Each registers here instead of writing the pointers, and the
// dispatcher asks them in registration order until one claims the node.

#pragma once

struct ggml_tensor;

namespace vla {

using sycl_ext_forward_fn  = bool (*)(ggml_tensor * dst, void * queue);
using sycl_ext_supports_fn = bool (*)(const ggml_tensor * op);

// Add an extension and install the dispatcher. @p supports answers for the
// GGML_OP_CUSTOM nodes it claims and may be null. Registering the same
// @p forward twice is a no-op. @p front puts it ahead of the ones already
// registered: for an extension that claims a few built-in nodes an earlier one
// would otherwise take (the sycl-tla attention over a FoldQuant prefix pack).
void sycl_ext_register(sycl_ext_forward_fn forward, sycl_ext_supports_fn supports, bool front = false);

}  // namespace vla
