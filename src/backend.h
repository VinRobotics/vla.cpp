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

/**
 * @file backend.h
 * @brief Compute-backend selection shared by every in-tree arch.
 *
 * Each arch used to open-code the same accelerator-then-CPU ladder, so adding a
 * backend meant editing a dozen files. They all call @ref vla::backend_init
 * instead; the ladder lives here once.
 *
 * Exactly one accelerator is compiled in, picked by the CMake flag that was
 * used (`GGML_CUDA` / `GGML_SYCL` / `GGML_METAL` / `GGML_OPENVINO`). There is no
 * per-op CPU fallback: the core drives a single backend through `gallocr` rather
 * than a scheduler, so an arch that hits an op the backend does not implement
 * asserts at predict time instead of silently limping.
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_OPENVINO
#include "ggml-openvino.h"
#endif

#include <cstdio>
#include <cstring>
#include <cstdlib>
#ifdef GGML_USE_OPENVINO
#include <string>
#include <unordered_set>
#endif
#if defined(GGML_USE_SYCL) || defined(GGML_USE_OPENVINO)
#include <stdlib.h>  // setenv / _putenv_s
#include <mutex>
#endif

namespace vla {

#if defined(GGML_USE_SYCL) || defined(GGML_USE_OPENVINO)
// setenv is POSIX. _putenv_s has no "do not overwrite" mode, so check first.
// Empty counts as unset; an empty KEY= in a compose file is not a choice.
inline void setenv_default(const char * key, const char * val) {
#ifdef _WIN32
    size_t len = 0;
    if (getenv_s(&len, nullptr, 0, key) == 0 && len > 1)  // len counts the NUL
        return;
    _putenv_s(key, val);
#else
    const char * cur = std::getenv(key);
    if (cur && *cur)
        return;
    setenv(key, val, /*overwrite=*/1);
#endif
}
#endif

/// Outcome of @ref backend_init. @c handle is null only if even the CPU backend
/// failed to come up, which callers treat as a fatal load error.
struct Backend {
    ggml_backend_t handle = nullptr;
    /// True only for a live CUDA backend, never for the CPU fallback. The BF16
    /// activation path needs the CUDA BF16 GEMM and elementwise kernels, so
    /// archs gate on this rather than on the compiled-in accelerator.
    bool           is_cuda = false;
};

/**
 * @brief Give every tensor in a built graph a name unique within that graph.
 *
 * ggml derives a name for a result from its source -- `ggml_reshape_2d` of an
 * unnamed tensor is called " (reshaped)" -- so a graph whose intermediates were
 * never named ends up with many tensors sharing one name. That is legal ggml,
 * and llama.cpp never trips over it because it labels every node it builds.
 *
 * ggml-openvino keys its translation map on those names: two nodes with the same
 * name silently become one, and the graph it hands OpenVINO wires the wrong
 * tensor into the next op. Renaming duplicates in place before compute is enough
 * to keep them apart, and only the OpenVINO build pays for it -- elsewhere this
 * compiles to nothing, so backend logs and profiles keep the names ggml chose.
 *
 * @param gf Graph to relabel, already built.
 */
inline void graph_unique_names([[maybe_unused]] ggml_cgraph * gf) {
#ifdef GGML_USE_OPENVINO
    // Leafs cannot collide: ggml names an unnamed one "leaf_<i>" as it walks the
    // graph, and a named one came from the checkpoint. Only results carry a name
    // derived from their source, so only nodes are checked.
    const int n = ggml_graph_n_nodes(gf);

    std::unordered_set<std::string> seen;
    seen.reserve((size_t) n);

    char buf[GGML_MAX_NAME];
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (seen.insert(ggml_get_name(t)).second) continue;
        // Local buffer: ggml_format_name would pass t->name as both destination
        // and "%s" source, and glibc empties it. Index leads so truncation eats
        // the tail, not the distinguisher. k steps by n to stay out of range.
        for (int k = i; ; k += n) {
            std::snprintf(buf, sizeof(buf), "%d#%s", k, ggml_get_name(t));
            if (seen.insert(buf).second) break;
        }
        ggml_set_name(t, buf);
    }
#endif
}

/**
 * @brief Allocate an arch's weights and tag the buffer as holding weights.
 *
 * `ggml_backend_alloc_ctx_tensors` leaves the buffer on the default
 * `GGML_BACKEND_BUFFER_USAGE_ANY`, which backends read as "this is not weights".
 * ggml-openvino takes it literally and classifies every ANY tensor as a KV
 * cache, giving it a dynamic sequence dimension; the first translator that asks
 * a weight for its static shape then throws `to_shape was called on a dynamic
 * shape`. Tagging the buffer is what llama.cpp does with its own weights, and it
 * is what lets the OpenVINO frontend fold them in as constants.
 *
 * @return The weight buffer, or null if the allocation failed (OOM).
 */
inline ggml_backend_buffer_t alloc_weights(ggml_context * ctx, ggml_backend_t backend) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf) {
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    return buf;
}

/// GPU ordinal for CUDA and SYCL; `VLA_DEVICE` overrides. Junk is rejected, not
/// silently read as device 0.
inline int backend_device_index() {
    const char * e = std::getenv("VLA_DEVICE");
    if (!e || !*e)
        return 0;
    char * end = nullptr;
    const long idx = std::strtol(e, &end, 10);
    if (*end != '\0' || idx < 0 || idx > 1024) {
        std::fprintf(stderr, "vla: ignoring VLA_DEVICE='%s' (not a device index); using 0\n", e);
        return 0;
    }
    return (int) idx;
}

/**
 * @brief Bring up the best compute backend available to this build.
 *
 * @param tag        Log prefix identifying the arch, e.g. @c "vla(pi0)".
 * @param n_threads  Thread count handed to the CPU backend if it is used.
 * @return The backend plus the flags the arch records about it.
 */
inline Backend backend_init(const char * tag, int n_threads) {
    Backend b;

#ifdef GGML_USE_CUDA
    {
        const int dev = backend_device_index();
        b.handle = ggml_backend_cuda_init(dev);
        if (b.handle) {
            b.is_cuda = true;
            std::printf("%s: backend = CUDA (device %d)\n", tag, dev);
        } else {
            std::fprintf(stderr, "%s: ggml_backend_cuda_init failed; falling back to CPU\n", tag);
        }
    }
#elif defined(GGML_USE_SYCL)
    {
        // ggml-sycl's VMM pool hands out virtual-memory-backed pointers that
        // oneDNN cannot wrap in a dnnl::memory: the GEMM aborts the process with
        // "could not create a memory object". Any src0 that is not already F32
        // (BF16, F16, and every quantized type) is converted into that pool
        // first, so the crash hits most checkpoints. Turning the pool off is
        // also the faster of the two workarounds -- measurably better than
        // disabling oneDNN outright. Only a default: an explicit setting wins,
        // for Intel GPUs where the pool is worth keeping.
        // ggml reads this on the first SYCL entry point, so it must be set here.
        // call_once: concurrent model_load would race on the environment.
        static std::once_flag vmm_once;
        std::call_once(vmm_once, [] { setenv_default("GGML_SYCL_ENABLE_VMM", "0"); });

        // ggml_backend_sycl_init() guards the device index with assert(), which
        // a Release build compiles out and then indexes past the device array.
        // Range-check here so a SYCL build on a box with no Intel GPU (or a bad
        // VLA_DEVICE) lands on CPU instead of corrupting memory.
        const int dev   = backend_device_index();
        const int n_dev = ggml_backend_sycl_get_device_count();
        if (dev >= n_dev) {
            std::fprintf(stderr, "%s: SYCL device %d out of range (%d visible); falling back to CPU\n",
                         tag, dev, n_dev);
        } else if ((b.handle = ggml_backend_sycl_init(dev)) != nullptr) {
            char desc[256] = { 0 };
            ggml_backend_sycl_get_device_description(dev, desc, sizeof(desc));
            std::printf("%s: backend = SYCL (device %d: %s)\n", tag, dev, desc);
        } else {
            std::fprintf(stderr, "%s: ggml_backend_sycl_init failed; falling back to CPU\n", tag);
        }
    }
#elif defined(GGML_USE_METAL)
    {
        b.handle = ggml_backend_metal_init();
        if (b.handle) {
            std::printf("%s: backend = Metal\n", tag);
        } else {
            std::fprintf(stderr, "%s: ggml_backend_metal_init failed; falling back to CPU\n", tag);
        }
    }
#elif defined(GGML_USE_OPENVINO)
    {
        // ggml-openvino translates a graph under 20 nodes literally and anything
        // larger through a decoder-only-LLM model builder that infers a KV cache
        // and one position input. No vla.cpp graph is that shape and all of them
        // are far larger, so raise the bar the literal path is chosen under;
        // scripts/patch_ggml_openvino.py is what makes the threshold settable.
        // Only a default: an explicit setting wins. The patched reader latches it
        // on the first graph_compute, so here is early enough. call_once because
        // a concurrent model_load would race on the environment.
        static std::once_flag naive_once;
        std::call_once(naive_once, [] { setenv_default("GGML_OPENVINO_NAIVE_GRAPH_SIZE", "1000000"); });

        // pi0 runs its whole 10-step denoise loop inside a single graph, so the
        // GPU plugin's F16 arithmetic compounds across every step with nothing to
        // reset it. On the continuous action dims that shows up as ~4e-2 against
        // an F32 reference, and it is enough to move the bistable gripper channel
        // across its threshold a step early -- which reads as a 1.7 error on a
        // metric, and as the gripper closing late on a robot. Asking the GPU for
        // F32 puts it back at 6.5e-5, and costs about 3x (383 ms -> 1170 ms).
        // Only pi0 needs it: every other arch is inside the bar on the GPU at F16.
        // A default, so GGML_OPENVINO_GPU_PRECISION=f16 still wins.
        // Exact match: `tag` is the log prefix, and "vla(pi05)" contains "vla(pi0)",
        // so anything looser would drag pi0.5 in too -- it does not need this.
        if (tag && std::strcmp(tag, "vla(pi0)") == 0) {
            setenv_default("GGML_OPENVINO_GPU_PRECISION", "f32");
        }

        // ggml exposes OpenVINO as a single device, so VLA_DEVICE does not apply:
        // the target is chosen by name through GGML_OPENVINO_DEVICE (CPU / GPU /
        // NPU) and resolved inside ggml.
        //
        // OpenVINO's blob cache reloads a graph that computes the wrong thing:
        // cold run correct, next run wrong, nothing logged. A warning is no use
        // when stderr goes nowhere, so clear it and make opting back in explicit.
        if (const char * cd = std::getenv("GGML_OPENVINO_CACHE_DIR"); cd && *cd) {
            const char * allow = std::getenv("VLA_ALLOW_OV_CACHE");
            const bool   keep  = allow && allow[0] == '1' && allow[1] == '\0';
            std::fprintf(stderr,
                         "%s: WARNING GGML_OPENVINO_CACHE_DIR is set. Reloading cached blobs has been\n"
                         "%s:         seen to produce silently incorrect actions on the GPU plugin.\n"
                         "%s:         %s See docs/backend/ov.md.\n",
                         tag, tag, tag,
                         keep ? "Kept: VLA_ALLOW_OV_CACHE=1."
                              : "Ignoring it; set VLA_ALLOW_OV_CACHE=1 to keep it.");
            if (!keep) {
#ifdef _WIN32
                _putenv_s("GGML_OPENVINO_CACHE_DIR", "");
#else
                unsetenv("GGML_OPENVINO_CACHE_DIR");
#endif
            }
        }

        // ggml falls back to CPU silently when the requested device is missing,
        // so this line is the request. ggml logs what actually ran.
        const char * want = std::getenv("GGML_OPENVINO_DEVICE");
        b.handle = ggml_backend_openvino_init(0);
        if (b.handle) {
            std::printf("%s: backend = OPENVINO (asked for %s, see ggml's \"using device\" line)\n",
                        tag, (want && *want) ? want : "CPU");
        } else {
            std::fprintf(stderr, "%s: ggml_backend_openvino_init failed; falling back to CPU\n", tag);
        }
    }
#endif

    if (!b.handle) {
        b.handle = ggml_backend_cpu_init();
        if (!b.handle) {
            std::fprintf(stderr, "%s: ggml_backend_cpu_init failed\n", tag);
            return b;
        }
        ggml_backend_cpu_set_n_threads(b.handle, n_threads);
        std::printf("%s: backend = CPU (%d threads)\n", tag, n_threads);
    }
    return b;
}

}  // namespace vla
