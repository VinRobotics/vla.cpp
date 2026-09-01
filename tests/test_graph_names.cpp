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

// Pins vla::graph_unique_names, which ggml-openvino depends on to tell two
// same-named nodes apart. Defines GGML_USE_OPENVINO so the body compiles on any
// build; nothing here calls into the backend, so only ggml is linked.
//
// The bug this exists to catch: renaming through ggml_format_name passes the
// tensor's own name as the "%s" source, and glibc empties it instead of
// appending, so every duplicate collapsed to "#<index>".

#define GGML_USE_OPENVINO
#include "backend.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

int main() {
    ggml_init_params p = { 64u * 1024 * 1024, nullptr, true };
    ggml_context *   C = ggml_init(p);
    assert(C);

    ggml_tensor * x = ggml_new_tensor_2d(C, GGML_TYPE_F32, 8, 4);
    ggml_set_name(x, "x");

    // Unnamed reshapes all land on the same ggml-derived name.
    ggml_cgraph * gf = ggml_new_graph_custom(C, 256, false);
    for (int i = 0; i < 6; ++i)
        ggml_build_forward_expand(gf, ggml_reshape_2d(C, ggml_scale(C, x, 1.0f + i), 4, 8));

    const int n = ggml_graph_n_nodes(gf);
    assert(n >= 12);

    std::vector<std::string> before;
    for (int i = 0; i < n; ++i)
        before.emplace_back(ggml_get_name(ggml_graph_node(gf, i)));

    // Precondition: without the pass the graph really does carry duplicates.
    assert(std::set<std::string>(before.begin(), before.end()).size() < (size_t) n);

    vla::graph_unique_names(gf);

    std::set<std::string> post;
    for (int i = 0; i < n; ++i) {
        const std::string nm = ggml_get_name(ggml_graph_node(gf, i));
        assert(!nm.empty());
        assert(post.insert(nm).second);                     // every node distinct
        assert(nm.find(before[i]) != std::string::npos);    // and still says what it was
    }

    // Idempotent: the graph cache hands the same graph back on every predict.
    std::set<std::string> again;
    vla::graph_unique_names(gf);
    for (int i = 0; i < n; ++i)
        again.insert(ggml_get_name(ggml_graph_node(gf, i)));
    assert(again == post);

    ggml_free(C);
    std::printf("test_graph_names: OK (%d nodes)\n", n);
    return 0;
}
