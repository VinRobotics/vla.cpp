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

#include "sycl/vla_sycl_ext.h"

#include <mutex>

// Must match the typedefs the hook patch inserts into ggml-sycl.cpp.
extern "C" {
typedef bool (*ggml_sycl_ext_forward_t)(struct ggml_tensor * dst, void * queue);
typedef bool (*ggml_sycl_ext_supports_t)(const struct ggml_tensor * op);
extern ggml_sycl_ext_forward_t  ggml_sycl_ext_forward;
extern ggml_sycl_ext_supports_t ggml_sycl_ext_supports;
}

namespace vla {

namespace {

constexpr int MAX_EXT = 8;

struct Ext {
    sycl_ext_forward_fn  forward;
    sycl_ext_supports_fn supports;
};

// Written only under the mutex, before the first graph that needs the entry
// runs; read lock-free by the dispatcher on the compute path.
Ext        g_ext[MAX_EXT];
int        g_n_ext = 0;
std::mutex g_mu;

bool dispatch(ggml_tensor * dst, void * queue) {
    for (int i = 0; i < g_n_ext; ++i)
        if (g_ext[i].forward(dst, queue)) return true;
    return false;
}

bool supports(const ggml_tensor * op) {
    for (int i = 0; i < g_n_ext; ++i)
        if (g_ext[i].supports && g_ext[i].supports(op)) return true;
    return false;
}

}  // namespace

void sycl_ext_register(sycl_ext_forward_fn forward, sycl_ext_supports_fn sup, bool front) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (int i = 0; i < g_n_ext; ++i)
        if (g_ext[i].forward == forward) return;
    if (g_n_ext == MAX_EXT) return;
    const int at = front ? 0 : g_n_ext;
    for (int i = g_n_ext; i > at; --i)
        g_ext[i] = g_ext[i - 1];
    g_ext[at] = {forward, sup};
    ++g_n_ext;
    ggml_sycl_ext_forward  = dispatch;
    ggml_sycl_ext_supports = supports;
}

}  // namespace vla
