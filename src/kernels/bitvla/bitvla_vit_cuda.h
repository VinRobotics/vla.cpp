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
 * @file bitvla_vit_cuda.h
 * @brief CUDA forward path for the BitVLA vision tower (ViT + mmproj).
 *
 * Mirrors the layout of @ref bitvla_lm_cuda.h: an opaque
 * @ref bitvla_vit_cuda_ctx owns device buffers; per-layer weights are
 * registered via @ref bitvla_vit_cuda_set_layer; the patch embed and
 * the 2-layer mmproj projector are set separately, and a single
 * @ref bitvla_vit_cuda_forward call produces patch-token embeddings
 * ready to feed the LM.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Per-layer weight pointers for one BitVLA ViT block.
 *
 * Norms are bf16, biases bf16. The dense projections (Q/K/V/O and the
 * two FFN matmuls) are 1.58-bit ternary (i2) packs paired with FP32
 * per-group scales.
 */
typedef struct {
    __nv_bfloat16* ln1_w; ///< Pre-attention LayerNorm scale.
    __nv_bfloat16* ln1_b; ///< Pre-attention LayerNorm bias.
    __nv_bfloat16* ln2_w; ///< Pre-FFN LayerNorm scale.
    __nv_bfloat16* ln2_b; ///< Pre-FFN LayerNorm bias.

    int8_t* q_packed;     ///< Ternary Q projection weights.
    float*  q_ws;         ///< Per-group scales for @ref q_packed.
    __nv_bfloat16* q_b;   ///< Q projection bias.
    int8_t* k_packed;     ///< Ternary K projection weights.
    float*  k_ws;         ///< Per-group scales for @ref k_packed.
    __nv_bfloat16* k_b;   ///< K projection bias.
    int8_t* v_packed;     ///< Ternary V projection weights.
    float*  v_ws;         ///< Per-group scales for @ref v_packed.
    __nv_bfloat16* v_b;   ///< V projection bias.
    int8_t* o_packed;     ///< Ternary output projection weights.
    float*  o_ws;         ///< Per-group scales for @ref o_packed.
    __nv_bfloat16* o_b;   ///< Output projection bias.
    int8_t* fc1_packed;   ///< FFN up-projection (hidden -> ffn).
    float*  fc1_ws;       ///< Per-group scales for @ref fc1_packed.
    __nv_bfloat16* fc1_b; ///< FFN up bias.
    int8_t* fc2_packed;   ///< FFN down-projection (ffn -> hidden).
    float*  fc2_ws;       ///< Per-group scales for @ref fc2_packed.
    __nv_bfloat16* fc2_b; ///< FFN down bias.
} bitvla_vit_layer_cuda;

/// Opaque ViT context; allocate with @ref bitvla_vit_cuda_init.
typedef struct bitvla_vit_cuda_ctx bitvla_vit_cuda_ctx;

/**
 * @brief Allocate device buffers and metadata for the BitVLA ViT.
 * @param n_layers   Transformer layer count.
 * @param hidden     ViT hidden width.
 * @param n_heads    Attention head count.
 * @param ffn        FFN inner width.
 * @param n_patches  Token count after patch embed (e.g. 256 for 16x16 grid).
 * @param patch_flat Flattened patch input dimension (channels * patch * patch).
 * @param ln_eps     Epsilon for LayerNorm.
 * @param mm_out     Output dimension of the mmproj projector.
 * @return Owning context; release with @ref bitvla_vit_cuda_free.
 */
bitvla_vit_cuda_ctx* bitvla_vit_cuda_init(int n_layers, int hidden, int n_heads, int ffn,
                                          int n_patches, int patch_flat,
                                          float ln_eps, int mm_out);

/**
 * @brief Release a context returned by @ref bitvla_vit_cuda_init.
 * @param ctx Context to free; may be @c nullptr.
 */
void bitvla_vit_cuda_free(bitvla_vit_cuda_ctx* ctx);

/**
 * @brief Bind device pointers for ViT layer @p L.
 * @param ctx   ViT context.
 * @param L     Zero-based layer index, in [0, n_layers).
 * @param layer Pointers to the layer's weights.
 */
void bitvla_vit_cuda_set_layer(bitvla_vit_cuda_ctx* ctx, int L,
                               const bitvla_vit_layer_cuda* layer);

/**
 * @brief Bind the patch-embed projection and positional embedding table.
 * @param ctx      ViT context.
 * @param patch_w  Patch-embed weight (hidden x patch_flat), bf16.
 * @param patch_b  Patch-embed bias (length hidden), bf16.
 * @param pos_emb  Positional embedding table (n_patches x hidden), bf16.
 */
void bitvla_vit_cuda_set_embed(bitvla_vit_cuda_ctx* ctx,
                               const __nv_bfloat16* patch_w,
                               const __nv_bfloat16* patch_b,
                               const __nv_bfloat16* pos_emb);

/**
 * @brief Bind the two FC layers of the mmproj projector.
 * @param ctx  ViT context.
 * @param W1   First FC weight (hidden_proj x hidden), bf16.
 * @param b1   First FC bias (length hidden_proj), bf16.
 * @param W2   Second FC weight (mm_out x hidden_proj), bf16.
 * @param b2   Second FC bias (length mm_out), bf16.
 */
void bitvla_vit_cuda_set_mmproj(bitvla_vit_cuda_ctx* ctx,
                                const __nv_bfloat16* W1, const __nv_bfloat16* b1,
                                const __nv_bfloat16* W2, const __nv_bfloat16* b2);

/**
 * @brief Run the ViT forward pass on a single image.
 *
 * Computes patch embed -> @c n_layers transformer blocks -> mmproj
 * projector, writing one row per patch token to @p d_out.
 *
 * @param ctx       ViT context with all weights bound.
 * @param d_patches Input patches (n_patches x patch_flat), bf16 device pointer.
 * @param d_out     Output tokens (n_patches x mm_out), bf16 device pointer.
 * @param stream    CUDA stream.
 * @return 0 on success, non-zero on dispatch failure.
 */
int bitvla_vit_cuda_forward(bitvla_vit_cuda_ctx* ctx,
                            const __nv_bfloat16* d_patches,
                            __nv_bfloat16* d_out,
                            cudaStream_t stream);

#ifdef __cplusplus
}
#endif
