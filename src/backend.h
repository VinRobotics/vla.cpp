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
 * used (`GGML_CUDA` / `GGML_SYCL` / `GGML_METAL`). There is no per-op CPU
 * fallback: the core drives a single backend through `gallocr` rather than a
 * scheduler, so an arch that hits an op the backend does not implement asserts
 * at predict time instead of silently limping.
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

#include <cstdio>
#include <cstdlib>
#ifdef GGML_USE_SYCL
#include <stdlib.h>  // setenv / _putenv_s
#include <mutex>
#endif

namespace vla {

#ifdef GGML_USE_SYCL
// setenv is POSIX. _putenv_s has no "do not overwrite" mode, so check first.
inline void setenv_default(const char * key, const char * val) {
#ifdef _WIN32
    size_t len = 0;
    if (getenv_s(&len, nullptr, 0, key) == 0 && len > 0) return;
    _putenv_s(key, val);
#else
    setenv(key, val, /*overwrite=*/0);
#endif
}
#endif

/// Outcome of @ref backend_init. @c handle is null only if even the CPU backend
/// failed to come up, which callers treat as a fatal load error.
struct Backend {
    ggml_backend_t handle  = nullptr;
    bool           is_cuda = false;
    bool           is_gpu  = false;
};

/// GPU ordinal for CUDA and SYCL; `VLA_DEVICE` overrides. Junk is rejected, not
/// silently read as device 0.
inline int backend_device_index() {
    const char * e = std::getenv("VLA_DEVICE");
    if (!e || !*e) return 0;
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
            b.is_gpu  = true;
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
            b.is_gpu = true;
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
            b.is_gpu = true;
            std::printf("%s: backend = Metal\n", tag);
        } else {
            std::fprintf(stderr, "%s: ggml_backend_metal_init failed; falling back to CPU\n", tag);
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
