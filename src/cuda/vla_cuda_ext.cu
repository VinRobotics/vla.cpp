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

// The ggml CUDA extension hook is one function pointer. Several in-tree kernel
// families ride it (BF16 activations, FoldQuant linears), so this dispatcher is
// the only thing ever stored there: each family registers a handler that
// returns false, with no side effects, for nodes that are not its own.

#include "ggml.h"

#include <cstddef>

extern "C" {
typedef bool (*ggml_cuda_ext_forward_t)(struct ggml_tensor * dst, void * stream);
extern ggml_cuda_ext_forward_t ggml_cuda_ext_forward;
}

namespace vla {

using cuda_ext_fn = bool (*)(ggml_tensor *, void *);

namespace {

constexpr int MAX_HANDLERS = 4;
cuda_ext_fn g_handlers[MAX_HANDLERS] = {};
int         g_n_handlers = 0;

extern "C" bool vla_cuda_ext_dispatch(ggml_tensor * dst, void * stream) {
    for (int i = 0; i < g_n_handlers; ++i)
        if (g_handlers[i](dst, stream))
            return true;
    return false;
}

}  // namespace

// Idempotent per handler. Order of registration is irrelevant: handlers
// self-select on the node (type, op, magic word) and never overlap.
void cuda_ext_add_handler(cuda_ext_fn fn) {
    for (int i = 0; i < g_n_handlers; ++i)
        if (g_handlers[i] == fn)
            return;
    if (g_n_handlers < MAX_HANDLERS)
        g_handlers[g_n_handlers++] = fn;
    ggml_cuda_ext_forward = vla_cuda_ext_dispatch;
}

}  // namespace vla
