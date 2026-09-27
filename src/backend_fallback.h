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
 * @file backend_fallback.h
 * @brief An accelerator backend that hands the ops it rejects to the CPU.
 *
 * The core drives one backend through `gallocr`, with no scheduler, and that is
 * fine for CUDA, Metal, SYCL and OpenVINO because each of them runs every op the
 * archs build. Hexagon and OpenCL do not: an NPU rejects an op it has no kernel
 * for (GELU_ERF) or whose tile does not fit its VTCM, and the answer depends on
 * tensor shapes, so it cannot be known before the graph exists.
 *
 * Instead of a `ggml_backend_sched` at every call site, the arch gets one
 * backend that looks like the accelerator -- same device, same buffer type, so
 * weights and activations still live on it -- but whose graph_compute walks the
 * graph and splits it: runs of ops the accelerator accepts go to it unchanged,
 * and runs it rejects are copied to host memory, computed on a CPU backend, and
 * copied back. Weights read on the CPU side are copied once and kept.
 *
 * Only built with GGML_HEXAGON or GGML_OPENCL; nothing else links it.
 */

#pragma once

#include "ggml-backend.h"

namespace vla {

/**
 * @brief Wrap @p accel so that ops it cannot run execute on the CPU.
 *
 * @param accel     Accelerator backend; ownership passes to the wrapper, which
 *                  frees it. Must not be null.
 * @param n_threads Thread count for the CPU backend that runs rejected ops.
 * @return The wrapper, or null if the CPU backend could not be created (in
 *         which case @p accel has been freed).
 */
ggml_backend_t fallback_backend_new(ggml_backend_t accel, int n_threads);

/**
 * @brief Point the Hexagon FastRPC loader at this executable's directory.
 *
 * The HTP skels (`libggml-htp-vNN.so` and their signed catalog) are copied
 * beside the binaries at build time, and FastRPC looks for them only on
 * `ADSP_LIBRARY_PATH`. Sets it to the executable's directory unless the
 * environment already names one. No-op off Windows.
 */
void hexagon_default_skel_path();

}  // namespace vla
