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

// Compute context and graph allocator reused across predict calls; rebuilding
// them costs 2-4 ms on the larger graphs. One scratch per graph role, and
// tensors die at the next reset.

#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstddef>

namespace vla {

class scratch_ctx {
public:
    scratch_ctx() = default;
    scratch_ctx(const scratch_ctx &) = delete;
    scratch_ctx & operator=(const scratch_ctx &) = delete;
    ~scratch_ctx() { release(); }

    ggml_context * reset(size_t arena) {
        if (ctx_) { ggml_reset(ctx_); return ctx_; }
        ggml_init_params p = { arena, nullptr, true };
        ctx_ = ggml_init(p);
        return ctx_;
    }

    bool alloc(ggml_backend_t backend, ggml_cgraph * gf) {
        if (!galloc_) galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return galloc_ && ggml_gallocr_alloc_graph(galloc_, gf);
    }

    void release() {
        if (galloc_) { ggml_gallocr_free(galloc_); galloc_ = nullptr; }
        if (ctx_)    { ggml_free(ctx_);            ctx_    = nullptr; }
    }

private:
    ggml_context * ctx_    = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
};

}  // namespace vla
