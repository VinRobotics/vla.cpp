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

#include "backend_fallback.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
// Internal: the backend vtable, and struct ggml_cgraph for slicing a graph.
// ggml_graph_view itself is not exported from ggml-base.dll, so the slice is
// built here the same way.
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace vla {
namespace {

enum class Where : uint8_t { Accel, Cpu, Either };

struct FallbackCtx {
    ggml_backend_t accel = nullptr;
    ggml_backend_t cpu   = nullptr;
    std::string    name;

    // Host copies of weights some CPU-side op reads, made once. Keyed by tensor:
    // the wrapper lives exactly as long as the model that owns the weights.
    std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> weights;

    // Host storage for one CPU run; the inner vectors keep their capacity across
    // runs, so a steady-state predict does not allocate.
    std::vector<std::vector<uint8_t>> pool;
    size_t                            pool_used = 0;

    std::vector<Where>    where;
    std::set<std::string> reported;
    bool                  stats   = false;
    bool                  hexagon = false;  // accel is an HTP session, see accel_runs()

    uint64_t n_graphs = 0, n_split_graphs = 0, n_cpu_nodes = 0, n_cpu_runs = 0;
    uint64_t bytes_in = 0, bytes_out = 0;
};

uint8_t * take(FallbackCtx * fc, size_t n) {
    if (fc->pool_used == fc->pool.size())
        fc->pool.emplace_back();
    std::vector<uint8_t> & v = fc->pool[fc->pool_used++];
    if (v.size() < n)
        v.resize(n);
    return v.data();
}

// Ops that only reinterpret memory. They compute nothing, so they go wherever
// their neighbours go instead of splitting a run.
bool is_view_op(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

ggml_backend_buffer_t buffer_of(const ggml_tensor * t) {
    return t->view_src ? t->view_src->buffer : t->buffer;
}

bool is_weight(const ggml_tensor * t) {
    ggml_backend_buffer_t b = buffer_of(t);
    return b && ggml_backend_buffer_get_usage(b) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

bool is_host(const ggml_tensor * t) {
    ggml_backend_buffer_t b = buffer_of(t);
    return b && ggml_backend_buffer_is_host(b);
}

void report(FallbackCtx * fc, const ggml_tensor * t, const char * why) {
    std::string key = std::string(ggml_op_desc(t)) + "/" + ggml_type_name(t->type);
    if (t->src[0])
        key += std::string("/") + ggml_type_name(t->src[0]->type);
    if (!fc->reported.insert(key).second)
        return;
    std::fprintf(stderr,
                 "vla: %s %s %s (%s <- %s, [%lld,%lld,%lld,%lld], e.g. '%s'); running it on CPU\n",
                 fc->name.c_str(), why, ggml_op_desc(t), ggml_type_name(t->type),
                 t->src[0] ? ggml_type_name(t->src[0]->type) : "-",
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                 t->name);
}

// A host tensor holding what `t` holds on the device, for the CPU to read.
ggml_tensor * host_input(FallbackCtx * fc, ggml_context * meta,
                         std::unordered_map<const ggml_tensor *, ggml_tensor *> & shadow,
                         ggml_tensor * t) {
    auto it = shadow.find(t);
    if (it != shadow.end())
        return it->second;

    ggml_tensor * s = ggml_new_tensor(meta, t->type, GGML_MAX_DIMS, t->ne);
    std::memcpy(s->nb, t->nb, sizeof(s->nb));
    ggml_set_name(s, t->name);

    // ggml_nbytes follows the strides, so a permuted or strided view copies the
    // span it covers, and the copy keeps the same strides.
    const size_t n = ggml_nbytes(t);
    if (is_host(t)) {
        s->data = t->data;
    } else if (is_weight(t)) {
        std::vector<uint8_t> & w = fc->weights[t];
        if (w.empty()) {
            // get_tensor undoes the accelerator's repacking, so this is plain ggml layout.
            w.resize(n);
            ggml_backend_tensor_get(t, w.data(), 0, n);
            fc->bytes_in += n;
        }
        s->data = w.data();
    } else {
        s->data = take(fc, n);
        ggml_backend_tensor_get(t, s->data, 0, n);
        fc->bytes_in += n;
    }
    shadow[t] = s;
    return s;
}

enum ggml_status run_on_cpu(FallbackCtx * fc, ggml_cgraph * g, int i0, int i1) {
    // The accelerator may still be writing what this run reads.
    ggml_backend_synchronize(fc->accel);

    const int    n_run    = i1 - i0;
    const size_t max_objs = (size_t) n_run * (GGML_MAX_SRC + 2);
    const size_t gsize    = std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, max_objs);
    ggml_init_params p = {
        ggml_tensor_overhead() * max_objs + ggml_graph_overhead_custom(gsize, false),
        nullptr,
        true,
    };
    ggml_context * meta = ggml_init(p);
    if (!meta)
        return GGML_STATUS_ALLOC_FAILED;
    ggml_cgraph * sg = ggml_new_graph_custom(meta, gsize, false);

    fc->pool_used = 0;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> shadow;
    shadow.reserve(max_objs);

    for (int k = i0; k < i1; ++k) {
        ggml_tensor * node = g->nodes[k];
        ggml_tensor * s    = ggml_new_tensor(meta, node->type, GGML_MAX_DIMS, node->ne);
        s->op    = node->op;
        s->flags = node->flags;
        std::memcpy(s->op_params, node->op_params, sizeof(s->op_params));
        std::memcpy(s->nb, node->nb, sizeof(s->nb));
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (node->src[j])
                s->src[j] = host_input(fc, meta, shadow, node->src[j]);
        }
        if (node->view_src) {
            // In-place ops and views write through their source, so they must
            // land in the same host copy the source got.
            ggml_tensor * vs = host_input(fc, meta, shadow, node->view_src);
            s->view_src  = vs;
            s->view_offs = node->view_offs;
            s->data      = (char *) vs->data + node->view_offs;
        } else {
            s->data = take(fc, ggml_nbytes(node));
        }
        ggml_set_name(s, node->name);
        shadow[node] = s;
        ggml_build_forward_expand(sg, s);
    }

    const enum ggml_status st = ggml_backend_graph_compute(fc->cpu, sg);

    if (st == GGML_STATUS_SUCCESS) {
        // Pure views wrote nothing; every other node's result goes back, since
        // later accelerator runs (or the arch's tensor_get) read it there.
        for (int k = i0; k < i1; ++k) {
            ggml_tensor * node = g->nodes[k];
            if (is_view_op(node) || !node->data || !buffer_of(node))
                continue;
            const size_t n = ggml_nbytes(node);
            ggml_backend_tensor_set(node, shadow[node]->data, 0, n);
            fc->bytes_out += n;
        }
    }
    fc->n_cpu_nodes += (uint64_t) n_run;
    fc->n_cpu_runs  += 1;
    ggml_free(meta);
    return st;
}

enum ggml_status run_on_accel(FallbackCtx * fc, ggml_cgraph * g, int i0, int i1) {
    // Same slice ggml_graph_view makes: the node range, no leafs, and the parent's
    // hash set so use counts (which fusion reads) still resolve.
    ggml_cgraph v = *g;
    v.size      = 0;
    v.n_nodes   = i1 - i0;
    v.n_leafs   = 0;
    v.nodes     = g->nodes + i0;
    v.grads     = nullptr;
    v.grad_accs = nullptr;
    v.leafs     = nullptr;
    v.uid       = 0;
    return ggml_backend_graph_compute_async(fc->accel, &v);
}

const char * fb_get_name(ggml_backend_t be) {
    return static_cast<FallbackCtx *>(be->context)->name.c_str();
}

void fb_free(ggml_backend_t be) {
    auto * fc = static_cast<FallbackCtx *>(be->context);
    if (fc->n_split_graphs > 0) {
        std::fprintf(stderr,
                     "vla: %s: %llu of %llu graphs split; %llu nodes in %llu runs went to CPU, "
                     "%.1f MiB copied in, %.1f MiB out, %.1f MiB of weights mirrored on host\n",
                     fc->name.c_str(),
                     (unsigned long long) fc->n_split_graphs, (unsigned long long) fc->n_graphs,
                     (unsigned long long) fc->n_cpu_nodes, (unsigned long long) fc->n_cpu_runs,
                     fc->bytes_in / 1048576.0, fc->bytes_out / 1048576.0,
                     [fc] {
                         size_t s = 0;
                         for (auto & kv : fc->weights) s += kv.second.size();
                         return s / 1048576.0;
                     }());
    }
    ggml_backend_free(fc->accel);
    ggml_backend_free(fc->cpu);
    delete fc;
    delete be;
}

void fb_synchronize(ggml_backend_t be) {
    ggml_backend_synchronize(static_cast<FallbackCtx *>(be->context)->accel);
}

// Whether b broadcasts over some dim d (b->ne[d] == 1 < a->ne[d]) while
// spanning a higher one. Row vectors and trailing broadcasts have no gap.
bool broadcast_has_gap(const ggml_tensor * a, const ggml_tensor * b) {
    bool broadcast_below = false;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (b->ne[d] == 1 && a->ne[d] > 1)
            broadcast_below = true;
        else if (b->ne[d] > 1 && broadcast_below)
            return true;
    }
    return false;
}

// Whether the accelerator should run t. Normally its own supports_op, but an
// op it claims and computes differently from the CPU reference is refused here.
bool accel_runs(FallbackCtx * fc, const ggml_tensor * t, const char ** why) {
    *why = "rejects";
    if (fc->hexagon && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_GELU) {
        // ggml-hexagon maps GELU and GELU_QUICK to one kernel, x*sigmoid(1.702x),
        // which is GELU_QUICK. ggml_gelu is the tanh form; on SmolVLA's SigLIP
        // tower the difference flips the gripper channel of the action chunk.
        *why = "computes GELU as GELU_QUICK for";
        return false;
    }
    if (fc->hexagon &&
        (t->op == GGML_OP_ADD || t->op == GGML_OP_SUB || t->op == GGML_OP_MUL || t->op == GGML_OP_DIV) &&
        broadcast_has_gap(t->src[0], t->src[1])) {
        // Claimed, but a src1 that broadcasts over one dim while spanning a
        // higher one comes out wrong. TurboVLA's RoPE multiplies
        // [64,12,261,2] by a cos table [64,1,261,1]: max|delta| 0.72 on the
        // action chunk, 1.2e-3 with these on the CPU.
        *why = "miscomputes a gapped broadcast in";
        return false;
    }
    if (fc->hexagon && t->op == GGML_OP_IM2COL) {
        // Written for patch embeddings (kernel == stride > 1, no padding, no
        // dilation) but claimed for any geometry. Octo's stride-2, pad-1 stem
        // convs and its 1x1 projection come out wrong: max|delta| 1.05, 5.2e-4
        // with them on the CPU. SigLIP's 16x16/16 patch embed is fine.
        const int32_t * p  = (const int32_t *) t->op_params;  // s0 s1 p0 p1 d0 d1 is_2D
        const int64_t   kw = t->src[0]->ne[0];
        const int64_t   kh = p[6] ? t->src[0]->ne[1] : 1;
        const bool patch = p[2] == 0 && p[3] == 0 && p[4] == 1 && p[5] == 1 && kw > 1 &&
                           p[0] == kw && (!p[6] || p[1] == kh);
        if (!patch) {
            *why = "miscomputes an overlapping or padded";
            return false;
        }
    }
    return ggml_backend_supports_op(fc->accel, t);
}

// The node whose memory t reads, looking through pure views.
const ggml_tensor * producer(const ggml_tensor * t) {
    while (t && t->op != GGML_OP_NONE && is_view_op(t))
        t = t->src[0];
    return t;
}

enum ggml_status fb_graph_compute(ggml_backend_t be, ggml_cgraph * g) {
    auto *    fc = static_cast<FallbackCtx *>(be->context);
    const int n  = g->n_nodes;
    fc->n_graphs++;

    fc->where.resize((size_t) n);
    bool any_cpu = false;
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = g->nodes[i];
        if (is_view_op(t) || ggml_is_empty(t)) {
            fc->where[i] = Where::Either;
        } else if (const char * why; accel_runs(fc, t, &why)) {
            fc->where[i] = Where::Accel;
        } else {
            fc->where[i] = Where::Cpu;
            any_cpu      = true;
            report(fc, t, why);
        }
    }
    if (!any_cpu)
        return ggml_backend_graph_compute_async(fc->accel, g);
    fc->n_split_graphs++;

    if (fc->hexagon) {
        // An F32->F16 CPY computed on the NPU reads back wrong on the host, while
        // the NPU's own consumers see it right (SmolVLA: flash attention moved to
        // the CPU gets max|delta| 1.2 on its F16 K/V, 1.7e-3 once the cast moves
        // with it). So a copy feeding the CPU is done on the CPU too. Walking
        // backwards lets a pulled copy pull the copy that feeds it.
        std::unordered_map<const ggml_tensor *, int> idx;
        idx.reserve((size_t) n);
        for (int i = 0; i < n; ++i)
            idx[g->nodes[i]] = i;
        for (int i = n - 1; i >= 0; --i) {
            if (fc->where[i] != Where::Cpu)
                continue;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * p = producer(g->nodes[i]->src[s]);
                if (!p)
                    continue;
                auto it = idx.find(p);
                if (it == idx.end() || fc->where[it->second] != Where::Accel)
                    continue;
                if (p->op == GGML_OP_CPY || p->op == GGML_OP_CONT || p->op == GGML_OP_DUP)
                    fc->where[it->second] = Where::Cpu;
            }
        }
    }

    Where prev = Where::Accel;
    for (int i = 0; i < n; ++i) {
        if (fc->where[i] == Where::Either)
            fc->where[i] = prev;
        prev = fc->where[i];
    }

    int n_runs = 0;
    for (int i = 0; i < n;) {
        int j = i + 1;
        while (j < n && fc->where[j] == fc->where[i])
            ++j;
        const enum ggml_status st = fc->where[i] == Where::Cpu ? run_on_cpu(fc, g, i, j)
                                                               : run_on_accel(fc, g, i, j);
        if (st != GGML_STATUS_SUCCESS)
            return st;
        ++n_runs;
        i = j;
    }
    if (fc->stats) {
        std::fprintf(stderr, "vla: %s: graph of %d nodes ran as %d runs\n", fc->name.c_str(), n, n_runs);
    }
    return GGML_STATUS_SUCCESS;
}

ggml_guid_t fb_guid() {
    static ggml_guid guid = { 0x76, 0x6c, 0x61, 0x2d, 0x66, 0x61, 0x6c, 0x6c,
                              0x62, 0x61, 0x63, 0x6b, 0x2d, 0x63, 0x70, 0x75 };
    return &guid;
}

}  // namespace

ggml_backend_t fallback_backend_new(ggml_backend_t accel, int n_threads) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        ggml_backend_free(accel);
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(cpu, n_threads);

    auto * fc  = new FallbackCtx;
    fc->accel  = accel;
    fc->cpu    = cpu;
    fc->name   = std::string(ggml_backend_name(accel)) + "+CPU";
    const char * st = std::getenv("VLA_FALLBACK_STATS");
    fc->stats  = st && st[0] == '1';
    fc->hexagon = std::strncmp(ggml_backend_name(accel), "HTP", 3) == 0;

    ggml_backend_i iface = {};
    iface.get_name      = fb_get_name;
    iface.free          = fb_free;
    iface.synchronize   = fb_synchronize;
    iface.graph_compute = fb_graph_compute;
    // Everything else stays null: tensor_set/get then go through the buffer,
    // which is the accelerator's, and nobody here plans graphs or uses events.

    return new ggml_backend{ fb_guid(), iface, accel->device, fc };
}

void hexagon_default_skel_path() {
#ifdef _WIN32
    size_t len = 0;
    if (getenv_s(&len, nullptr, 0, "ADSP_LIBRARY_PATH") == 0 && len > 1)
        return;
    char path[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH)
        return;
    char * slash = std::strrchr(path, '\\');
    if (!slash)
        return;
    *slash = '\0';
    // _putenv_s updates the process environment too, which is where the FastRPC
    // loader in libcdsprpc.dll reads it from.
    _putenv_s("ADSP_LIBRARY_PATH", path);
#endif
}

}  // namespace vla
