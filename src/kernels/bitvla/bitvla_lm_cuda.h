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
 * @file bitvla_lm_cuda.h
 * @brief CUDA helpers and forward path for the BitVLA language model.
 *
 * The BitVLA LM stores attention/FFN projections in a 1.58-bit ternary
 * packed format ("ladder int8xint2") alongside FP32 weight scales. This
 * header exposes:
 *
 *   * Standalone bf16 ops (norm / RoPE / softmax / activations) used by
 *     both the LM and the ViT.
 *   * @ref bitvla_lm_cuda_ctx, an opaque context that owns the device-side
 *     LM state, plus its init / set-layer / forward / free entry points.
 *
 * All functions are @c extern @c "C" so they can be called from C++ or C
 * driver code. Streams are passed in explicitly; the kernels never call
 * @c cudaDeviceSynchronize() themselves.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Row-wise RMSNorm in bf16.
 * @param x  Input matrix (M x K), device pointer.
 * @param w  Per-channel scale (length K), device pointer.
 * @param out Output matrix (M x K), device pointer; may equal @p x.
 * @param eps Epsilon added to the variance.
 * @param M  Number of rows.
 * @param K  Row length.
 * @param stream CUDA stream.
 */
void bitvla_rmsnorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                         __nv_bfloat16* out, float eps, int M, int K,
                         cudaStream_t stream);

/**
 * @brief Apply NeoX-style rotary position embeddings in place.
 * @param inout   Tensor of shape (H, S, D), bf16 device pointer.
 * @param cos_tab Per-position cosine table (S x D/2), device pointer.
 * @param sin_tab Per-position sine table   (S x D/2), device pointer.
 * @param H       Number of heads.
 * @param S       Sequence length.
 * @param D       Per-head dimension; must be even.
 * @param stream  CUDA stream.
 */
void bitvla_rope_neox_bf16(__nv_bfloat16* inout, const float* cos_tab,
                           const float* sin_tab, int H, int S, int D,
                           cudaStream_t stream);

/**
 * @brief Numerically-stable scaled softmax along the inner (S) axis.
 * @param inout  Tensor laid out as (n_rows, S), bf16 device pointer.
 * @param scale  Multiplier applied before the softmax (usually 1/sqrt(d)).
 * @param n_rows Number of rows.
 * @param S      Softmax length.
 * @param stream CUDA stream.
 */
void bitvla_softmax_scaled_bf16(__nv_bfloat16* inout, float scale,
                                int n_rows, int S, cudaStream_t stream);

/**
 * @brief Elementwise @c relu(g)^2 * u (BitVLA squared-ReLU FFN gate).
 * @param g   Gate input (N), bf16 device pointer.
 * @param u   Up input    (N), bf16 device pointer.
 * @param out Output      (N), bf16 device pointer.
 * @param N   Element count.
 * @param stream CUDA stream.
 */
void bitvla_squared_relu_mul_bf16(const __nv_bfloat16* g, const __nv_bfloat16* u,
                                  __nv_bfloat16* out, int N, cudaStream_t stream);

/**
 * @brief Elementwise bf16 add (@p out = @p a + @p b).
 * @param a Length-@p N input, device pointer.
 * @param b Length-@p N input, device pointer.
 * @param out Length-@p N output, device pointer.
 * @param N   Element count.
 * @param stream CUDA stream.
 */
void bitvla_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                     __nv_bfloat16* out, int N, cudaStream_t stream);

/**
 * @brief GQA repeat: tile KV heads up to the number of Q heads.
 * @param in   Source tensor (n_kv, seq, hd).
 * @param out  Destination  (n_q, seq, hd).
 * @param n_q  Target head count.
 * @param n_kv Source head count; @p n_q must be a multiple of @p n_kv.
 * @param seq  Sequence length.
 * @param hd   Per-head dimension.
 * @param stream CUDA stream.
 */
void bitvla_repeat_kv_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                           int n_q, int n_kv, int seq, int hd, cudaStream_t stream);

/**
 * @brief Transpose [S, N*hd] -> [N, S, hd] (split last axis into heads).
 *
 * @param in   Source tensor laid out as (S, N*hd).
 * @param out  Destination tensor laid out as (N, S, hd).
 * @param S    Sequence length.
 * @param N    Head count.
 * @param hd   Per-head dimension.
 * @param stream CUDA stream.
 */
void bitvla_transpose_sNhd_to_NshHd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                                         int S, int N, int hd, cudaStream_t stream);

/**
 * @brief Inverse of @ref bitvla_transpose_sNhd_to_NshHd_bf16.
 *
 * @param in   Source tensor (N, S, hd).
 * @param out  Destination tensor (S, N*hd).
 * @param N    Head count.
 * @param S    Sequence length.
 * @param hd   Per-head dimension.
 * @param stream CUDA stream.
 */
void bitvla_transpose_NshHd_to_sNhd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                                         int N, int S, int hd, cudaStream_t stream);

/**
 * @brief Gather rows by index (bf16, K-wide rows).
 * @param in       Source matrix (rows x K).
 * @param out      Output matrix (n_rows x K).
 * @param row_ids  Length-@p n_rows index list (int32).
 * @param n_rows   Number of rows to gather.
 * @param K        Row length.
 * @param stream   CUDA stream.
 */
void bitvla_gather_rows_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                             const int32_t* row_ids, int n_rows, int K,
                             cudaStream_t stream);

/**
 * @brief Affine LayerNorm in bf16 (mean/variance + scale + bias).
 * @param x  Input matrix (M x K), bf16 device pointer.
 * @param w  Per-channel scale (length K).
 * @param b  Per-channel bias  (length K).
 * @param out Output matrix (M x K); may equal @p x.
 * @param eps Epsilon added to the variance.
 * @param M  Number of rows.
 * @param K  Row length.
 * @param stream CUDA stream.
 */
void bitvla_layernorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                           const __nv_bfloat16* b, __nv_bfloat16* out,
                           float eps, int M, int K, cudaStream_t stream);

/**
 * @brief Elementwise tanh-approximation GELU (BitVLA ViT activation).
 * @param x   Length-@p N input, bf16 device pointer.
 * @param out Length-@p N output, bf16 device pointer.
 * @param N   Element count.
 * @param stream CUDA stream.
 */
void bitvla_gelu_tanh_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                           int N, cudaStream_t stream);

/**
 * @brief Broadcast-add a per-channel bias to a (M x K) matrix in bf16.
 * @param x    Input matrix (M x K).
 * @param bias Per-channel bias (length K).
 * @param out  Output matrix (M x K); may equal @p x.
 * @param M    Number of rows.
 * @param K    Row length.
 * @param stream CUDA stream.
 */
void bitvla_add_bias_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                          __nv_bfloat16* out, int M, int K, cudaStream_t stream);

/**
 * @brief Zero out columns [@p start_col, @p total_cols) of an
 *        (M x @p total_cols) bf16 matrix, leaving the first @p start_col
 *        columns untouched.
 *
 * Used to mask out padding lanes added by the ternary GEMM's column tiling.
 */
void bitvla_zero_tail_bf16(__nv_bfloat16* x, int M, int total_cols,
                           int start_col, cudaStream_t stream);

/**
 * @brief Per-layer weight pointers for one BitVLA LM transformer block.
 *
 * All pointers live in device memory and are owned by the caller. The
 * @c *_packed weights are ternary (i2) values, @c *_ws their per-group
 * FP32 scales. Norm weights are bf16.
 */
typedef struct {
    __nv_bfloat16* attn_norm_w;     ///< Pre-attention RMSNorm scale.
    __nv_bfloat16* attn_sub_norm_w; ///< BitNet attention sub-norm scale.
    __nv_bfloat16* ffn_norm_w;      ///< Pre-FFN RMSNorm scale.
    __nv_bfloat16* ffn_sub_norm_w;  ///< BitNet FFN sub-norm scale.

    int8_t*        q_packed;        ///< Ternary Q projection weights.
    float*         q_ws;            ///< Per-group scales for @ref q_packed.
    int8_t*        k_packed;        ///< Ternary K projection weights.
    float*         k_ws;            ///< Per-group scales for @ref k_packed.
    int8_t*        v_packed;        ///< Ternary V projection weights.
    float*         v_ws;            ///< Per-group scales for @ref v_packed.
    int8_t*        o_packed;        ///< Ternary output projection weights.
    float*         o_ws;            ///< Per-group scales for @ref o_packed.
    int8_t*        gate_up_packed;  ///< Fused gate+up FFN weights.
    float*         gate_up_ws;      ///< Per-group scales for @ref gate_up_packed.
    int8_t*        down_packed;     ///< FFN down-projection weights.
    float*         down_ws;         ///< Per-group scales for @ref down_packed.
} bitvla_lm_layer_cuda;

/// Opaque LM context; allocate with @ref bitvla_lm_cuda_init.
typedef struct bitvla_lm_cuda_ctx bitvla_lm_cuda_ctx;

/**
 * @brief Allocate device buffers and metadata for the BitVLA LM.
 * @param hidden    LM hidden width.
 * @param n_q       Number of query heads.
 * @param n_kv      Number of KV heads (GQA).
 * @param head_dim  Per-head dimension.
 * @param ffn       FFN inner width.
 * @param n_layers  Transformer layer count.
 * @param rope_base RoPE base frequency.
 * @param rms_eps   Epsilon for RMSNorm.
 * @param max_seq   Maximum sequence length supported by the workspace.
 * @return Owning context; release with @ref bitvla_lm_cuda_free.
 */
bitvla_lm_cuda_ctx* bitvla_lm_cuda_init(int hidden, int n_q, int n_kv, int head_dim,
                                        int ffn, int n_layers,
                                        float rope_base, float rms_eps,
                                        int max_seq);

/**
 * @brief Release a context returned by @ref bitvla_lm_cuda_init.
 * @param ctx Context to free; may be @c nullptr.
 */
void bitvla_lm_cuda_free(bitvla_lm_cuda_ctx* ctx);

/**
 * @brief Bind device pointers for transformer layer @p L.
 * @param ctx   LM context.
 * @param L     Zero-based layer index, in [0, n_layers).
 * @param layer Pointers to the layer's weights.
 */
void bitvla_lm_cuda_set_layer(bitvla_lm_cuda_ctx* ctx, int L,
                              const bitvla_lm_layer_cuda* layer);

/**
 * @brief Bind the final (output) RMSNorm scale.
 * @param ctx LM context.
 * @param w   Device pointer to a length-@c hidden bf16 vector.
 */
void bitvla_lm_cuda_set_output_norm(bitvla_lm_cuda_ctx* ctx, const __nv_bfloat16* w);

/**
 * @brief Run the LM forward pass on a single batch.
 * @param ctx    LM context with all layers bound.
 * @param d_in   Input embeddings (@p seq x hidden), bf16 device pointer.
 * @param d_out  Output hidden states (@p seq x hidden), bf16 device pointer.
 * @param seq    Sequence length; must be <= the workspace's @c max_seq.
 * @param stream CUDA stream.
 * @return 0 on success, non-zero on dispatch failure.
 */
int bitvla_lm_cuda_forward(bitvla_lm_cuda_ctx* ctx,
                           const __nv_bfloat16* d_in,
                           __nv_bfloat16* d_out,
                           int seq, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
