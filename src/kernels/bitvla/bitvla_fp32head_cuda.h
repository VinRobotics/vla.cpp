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
 * @file bitvla_fp32head_cuda.h
 * @brief FP32 BitVLA action head + proprioception projection.
 *
 * BitVLA's LM and ViT are 1.58-bit ternary; its small action head and
 * proprio projector are kept in FP32 (cuBLAS GEMM) to preserve regression
 * accuracy on continuous outputs. This header exposes the two-stage
 * forward path:
 *
 *   1. @ref bitvla_fp32head_proprio_forward turns a host-side state
 *      vector into a hidden conditioning vector for the LM.
 *   2. @ref bitvla_fp32head_action_forward consumes the LM's pooled
 *      output and produces a normalised action chunk on the host.
 *
 * The context is constructed once with @ref bitvla_fp32head_cuda_init,
 * which uploads all weights to device memory, and released with
 * @ref bitvla_fp32head_cuda_free.
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

/// Opaque context for the FP32 action head; allocate with
/// @ref bitvla_fp32head_cuda_init.
typedef struct bitvla_fp32head_cuda_ctx bitvla_fp32head_cuda_ctx;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate and upload all FP32 head weights to device memory.
 *
 * Naming convention: @c pp_* are the two-layer proprioception projector
 * (state -> LM hidden), @c ah_* are the action head proper -- two
 * pre/post LayerNorms (@c ln1/@c ln2), two outer FCs (@c fc1/@c fc2),
 * and two residual MLP blocks (@c b0/@c b1 each with its own LayerNorm
 * and weight matrix).
 *
 * All weight pointers are host arrays; they are copied to the device
 * during this call and may be freed by the caller on return.
 *
 * @param proprio_dim Proprioception input dimension.
 * @param lm_hidden   LM hidden width (head output of the proprio projector).
 * @param chunk       Action chunk length (number of time-steps).
 * @param action_dim  Per-step action dimension.
 * @param ah_ln_eps   Epsilon for the action head's LayerNorms.
 * @return Owning context; release with @ref bitvla_fp32head_cuda_free.
 */
bitvla_fp32head_cuda_ctx* bitvla_fp32head_cuda_init(
    int proprio_dim,
    int lm_hidden,
    int chunk,
    int action_dim,
    float ah_ln_eps,

    const float* pp_fc1_w,
    const float* pp_fc1_b,
    const float* pp_fc2_w,
    const float* pp_fc2_b,

    const float* ah_ln1_w,
    const float* ah_ln1_b,
    const float* ah_fc1_w,
    const float* ah_fc1_b,
    const float* ah_b0_lnw,
    const float* ah_b0_lnb,
    const float* ah_b0_w,
    const float* ah_b0_b,
    const float* ah_b1_lnw,
    const float* ah_b1_lnb,
    const float* ah_b1_w,
    const float* ah_b1_b,
    const float* ah_ln2_w,
    const float* ah_ln2_b,
    const float* ah_fc2_w,
    const float* ah_fc2_b);

/**
 * @brief Project a host-side proprioception vector into LM hidden space.
 * @param ctx        Context returned by @ref bitvla_fp32head_cuda_init.
 * @param host_state Length-@c proprio_dim FP32 input on the host.
 * @param host_out   Length-@c lm_hidden  FP32 output on the host.
 * @param stream     CUDA stream used for the H2D / D2H transfers.
 * @return 0 on success, non-zero on dispatch failure.
 */
int bitvla_fp32head_proprio_forward(
    bitvla_fp32head_cuda_ctx* ctx,
    const float* host_state,
    float* host_out,
    cudaStream_t stream);

/**
 * @brief Run the action head on the LM's pooled hidden state.
 * @param ctx              Context returned by @ref bitvla_fp32head_cuda_init.
 * @param host_ah_input    Length-@c lm_hidden FP32 input on the host.
 * @param host_norm_actions Length-@c chunk*action_dim FP32 output on the
 *        host, normalised to training statistics. The caller un-normalises
 *        to world units.
 * @param stream           CUDA stream used for the H2D / D2H transfers.
 * @return 0 on success, non-zero on dispatch failure.
 */
int bitvla_fp32head_action_forward(
    bitvla_fp32head_cuda_ctx* ctx,
    const float* host_ah_input,
    float* host_norm_actions,
    cudaStream_t stream);

/**
 * @brief Release a context returned by @ref bitvla_fp32head_cuda_init.
 * @param ctx Context to free; may be @c nullptr.
 */
void bitvla_fp32head_cuda_free(bitvla_fp32head_cuda_ctx* ctx);

#ifdef __cplusplus
}
#endif
