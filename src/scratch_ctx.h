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
//
// graph_cache keeps the built graph too, for the archs whose shape depends on a
// small key. ggml-cuda can then capture and replay it, which needs the node list
// to stay put.

#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstddef>
#include <utility>

namespace vla {

class scratch_ctx {
public:
    scratch_ctx() = default;
    scratch_ctx(const scratch_ctx &) = delete;
    scratch_ctx & operator=(const scratch_ctx &) = delete;
    ~scratch_ctx() {
        release();
    }

    ggml_context * reset(size_t arena) {
        // Growing matters: an arena sized from the input shape would otherwise
        // keep the first call's smaller pool and abort in ggml_new_tensor.
        // Every call site passes a constant today.
        if (ctx_ && arena > arena_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        if (ctx_) {
            ggml_reset(ctx_);
            return ctx_;
        }
        ggml_init_params p = { arena, nullptr, true };
        ctx_   = ggml_init(p);
        arena_ = arena;
        return ctx_;
    }

    bool alloc(ggml_backend_t backend, ggml_cgraph * gf) {
        if (!galloc_)
            galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return galloc_ && ggml_gallocr_alloc_graph(galloc_, gf);
    }

    void release() {
        if (galloc_) {
            ggml_gallocr_free(galloc_);
            galloc_ = nullptr;
        }
        if (ctx_)    {
            ggml_free(ctx_);
            ctx_    = nullptr;
        }
    }

private:
    ggml_context * ctx_    = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    size_t         arena_  = 0;
};

// Key is whatever shape the graph depends on (it needs operator==); IO holds the
// input/output tensor handles the arch fills in each call. build(ctx, io) emits
// the graph and returns it, or null to fail the call.
template <class Key, class IO>
class graph_cache {
public:
    graph_cache() = default;
    graph_cache(const graph_cache &) = delete;
    graph_cache & operator=(const graph_cache &) = delete;
    ~graph_cache() {
        release();
    }

    template <class Build>
    bool ensure(ggml_backend_t backend, const Key & key, size_t arena, Build && build) {
        if (valid_ && key_ == key)
            return true;
        release();
        ggml_init_params p = { arena, nullptr, true };
        ctx_ = ggml_init(p);
        if (!ctx_)
            return false;
        io_ = IO{};
        gf_ = build(ctx_, io_);
        if (!gf_) {
            release();
            return false;
        }
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc_ || !ggml_gallocr_alloc_graph(galloc_, gf_)) {
            release();
            return false;
        }
        key_   = key;
        valid_ = true;
        return true;
    }

    IO &          io()    {
        return io_;
    }
    ggml_cgraph * graph() {
        return gf_;
    }

    void release() {
        if (galloc_) {
            ggml_gallocr_free(galloc_);
            galloc_ = nullptr;
        }
        if (ctx_)    {
            ggml_free(ctx_);
            ctx_    = nullptr;
        }
        gf_ = nullptr;
        io_ = IO{};
        valid_ = false;
    }

private:
    ggml_context * ctx_    = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    ggml_cgraph *  gf_     = nullptr;
    Key            key_{};
    IO             io_{};
    bool           valid_  = false;
};

}  // namespace vla
