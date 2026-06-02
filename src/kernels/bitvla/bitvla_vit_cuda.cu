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

#include "bitvla_vit_cuda.h"
#include "bitvla_lm_cuda.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" void bitlinear_int8xint2_m(int8_t* A, int8_t* B, __nv_bfloat16* out,
                                       float* s, float* ws,
                                       int M, int N, int K, cudaStream_t stream);
extern "C" void bitvla_act_quant_cuda(const __nv_bfloat16* in, int8_t* out,
                                       float* scales,
                                       int M, int K, cudaStream_t stream);

__global__ void gelu_erf_bf16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int N) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) return;
    const float x = __bfloat162float(in[i]);
    const float inv_sqrt2 = 0.7071067811865475f;
    out[i] = __float2bfloat16(0.5f * x * (1.0f + erff(x * inv_sqrt2)));
}
static void gelu_erf_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int N, cudaStream_t stream) {
    constexpr int B = 256;
    gelu_erf_bf16_kernel<<<dim3((N + B - 1) / B, 1, 1), dim3(B, 1, 1), 0, stream>>>(in, out, N);
}

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "vla(bitvla_vit_cuda): %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return -1; } } while (0)
#define CUDA_OKV(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "vla(bitvla_vit_cuda): %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return nullptr; } } while (0)

struct bitvla_vit_cuda_ctx {
    int n_layers, hidden, n_heads, head_dim, ffn, n_patches, patch_flat, mm_out;
    float ln_eps;
    int ffn_pad;

    cublasHandle_t cublas;

    __nv_bfloat16* patch_w   = nullptr;
    __nv_bfloat16* patch_b   = nullptr;
    __nv_bfloat16* pos_emb   = nullptr;

    __nv_bfloat16* mm_W1     = nullptr;
    __nv_bfloat16* mm_b1     = nullptr;
    __nv_bfloat16* mm_W2     = nullptr;
    __nv_bfloat16* mm_b2     = nullptr;

    std::vector<bitvla_vit_layer_cuda> layers;

    __nv_bfloat16* d_h            = nullptr;
    __nv_bfloat16* d_h_norm       = nullptr;
    int8_t*        d_act_int8_h   = nullptr;
    int8_t*        d_act_int8_ffn = nullptr;
    float*         d_act_s        = nullptr;
    __nv_bfloat16* d_q_proj       = nullptr;
    __nv_bfloat16* d_k_proj       = nullptr;
    __nv_bfloat16* d_v_proj       = nullptr;
    __nv_bfloat16* d_q_HShd       = nullptr;
    __nv_bfloat16* d_k_HShd       = nullptr;
    __nv_bfloat16* d_v_HShd       = nullptr;
    __nv_bfloat16* d_scores       = nullptr;
    __nv_bfloat16* d_attn_out     = nullptr;
    __nv_bfloat16* d_attn_merged  = nullptr;
    __nv_bfloat16* d_o_out        = nullptr;
    __nv_bfloat16* d_fc1_dense    = nullptr;
    __nv_bfloat16* d_fc1_padded   = nullptr;
    __nv_bfloat16* d_fc2_out      = nullptr;
    __nv_bfloat16* d_mm_h1        = nullptr;
};

bitvla_vit_cuda_ctx* bitvla_vit_cuda_init(int n_layers, int hidden, int n_heads, int ffn,
                                            int n_patches, int patch_flat,
                                            float ln_eps, int mm_out)
{
    auto* ctx = new bitvla_vit_cuda_ctx{};
    ctx->n_layers = n_layers;
    ctx->hidden   = hidden;
    ctx->n_heads  = n_heads;
    ctx->head_dim = hidden / n_heads;
    ctx->ffn      = ffn;
    ctx->n_patches  = n_patches;
    ctx->patch_flat = patch_flat;
    ctx->ln_eps   = ln_eps;
    ctx->mm_out   = mm_out;

    ctx->ffn_pad = ((ffn + 127) / 128) * 128;
    ctx->layers.resize(n_layers);

    if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(bitvla_vit_cuda): cublasCreate failed\n"); delete ctx; return nullptr;
    }

    const size_t bf16 = sizeof(__nv_bfloat16);
    CUDA_OKV(cudaMalloc(&ctx->d_h,            (size_t) n_patches * hidden  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_h_norm,       (size_t) n_patches * hidden  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_act_int8_h,   (size_t) n_patches * hidden));
    CUDA_OKV(cudaMalloc(&ctx->d_act_int8_ffn, (size_t) n_patches * ctx->ffn_pad));
    CUDA_OKV(cudaMalloc(&ctx->d_act_s,        (size_t) n_patches * sizeof(float)));
    CUDA_OKV(cudaMalloc(&ctx->d_q_proj,       (size_t) n_patches * hidden  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_k_proj,       (size_t) n_patches * hidden  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_v_proj,       (size_t) n_patches * hidden  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_q_HShd,       (size_t) n_heads * n_patches * ctx->head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_k_HShd,       (size_t) n_heads * n_patches * ctx->head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_v_HShd,       (size_t) n_heads * n_patches * ctx->head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_scores,       (size_t) n_heads * n_patches * n_patches    * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_attn_out,     (size_t) n_heads * n_patches * ctx->head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_attn_merged,  (size_t) n_patches * hidden  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_o_out,        (size_t) n_patches * hidden  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_fc1_dense,    (size_t) n_patches * ffn         * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_fc1_padded,   (size_t) n_patches * ctx->ffn_pad * bf16));

    CUDA_OKV(cudaMemset(ctx->d_fc1_padded, 0, (size_t) n_patches * ctx->ffn_pad * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_fc2_out,      (size_t) n_patches * hidden  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_mm_h1,        (size_t) n_patches * mm_out  * bf16));
    return ctx;
}

void bitvla_vit_cuda_free(bitvla_vit_cuda_ctx* ctx) {
    if (!ctx) return;
    cublasDestroy(ctx->cublas);
    cudaFree(ctx->d_h);   cudaFree(ctx->d_h_norm);
    cudaFree(ctx->d_act_int8_h);   cudaFree(ctx->d_act_int8_ffn);   cudaFree(ctx->d_act_s);
    cudaFree(ctx->d_q_proj); cudaFree(ctx->d_k_proj); cudaFree(ctx->d_v_proj);
    cudaFree(ctx->d_q_HShd); cudaFree(ctx->d_k_HShd); cudaFree(ctx->d_v_HShd);
    cudaFree(ctx->d_scores); cudaFree(ctx->d_attn_out); cudaFree(ctx->d_attn_merged);
    cudaFree(ctx->d_o_out);
    cudaFree(ctx->d_fc1_dense); cudaFree(ctx->d_fc1_padded); cudaFree(ctx->d_fc2_out);
    cudaFree(ctx->d_mm_h1);
    delete ctx;
}

void bitvla_vit_cuda_set_layer(bitvla_vit_cuda_ctx* ctx, int L,
                                const bitvla_vit_layer_cuda* layer) {
    ctx->layers[L] = *layer;
}
void bitvla_vit_cuda_set_embed(bitvla_vit_cuda_ctx* ctx,
                                const __nv_bfloat16* patch_w,
                                const __nv_bfloat16* patch_b,
                                const __nv_bfloat16* pos_emb) {
    ctx->patch_w = const_cast<__nv_bfloat16*>(patch_w);
    ctx->patch_b = const_cast<__nv_bfloat16*>(patch_b);
    ctx->pos_emb = const_cast<__nv_bfloat16*>(pos_emb);
}
void bitvla_vit_cuda_set_mmproj(bitvla_vit_cuda_ctx* ctx,
                                 const __nv_bfloat16* W1, const __nv_bfloat16* b1,
                                 const __nv_bfloat16* W2, const __nv_bfloat16* b2) {
    ctx->mm_W1 = const_cast<__nv_bfloat16*>(W1);
    ctx->mm_b1 = const_cast<__nv_bfloat16*>(b1);
    ctx->mm_W2 = const_cast<__nv_bfloat16*>(W2);
    ctx->mm_b2 = const_cast<__nv_bfloat16*>(b2);
}

static int run_vit_layer(bitvla_vit_cuda_ctx* ctx, int L, cudaStream_t stream) {
    auto& lr = ctx->layers[L];
    const int seq = ctx->n_patches, H = ctx->hidden, n_heads = ctx->n_heads;
    const int hd  = ctx->head_dim, ffn = ctx->ffn, ffn_pad = ctx->ffn_pad;

    bitvla_layernorm_bf16(ctx->d_h, lr.ln1_w, lr.ln1_b, ctx->d_h_norm, ctx->ln_eps, seq, H, stream);
    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, H, stream);

    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.q_packed, ctx->d_q_proj, ctx->d_act_s, lr.q_ws, seq, H, H, stream);
    bitvla_add_bias_bf16(ctx->d_q_proj, lr.q_b, ctx->d_q_proj, seq, H, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.k_packed, ctx->d_k_proj, ctx->d_act_s, lr.k_ws, seq, H, H, stream);
    bitvla_add_bias_bf16(ctx->d_k_proj, lr.k_b, ctx->d_k_proj, seq, H, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.v_packed, ctx->d_v_proj, ctx->d_act_s, lr.v_ws, seq, H, H, stream);
    bitvla_add_bias_bf16(ctx->d_v_proj, lr.v_b, ctx->d_v_proj, seq, H, stream);

    bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_q_proj, ctx->d_q_HShd, seq, n_heads, hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_k_proj, ctx->d_k_HShd, seq, n_heads, hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(ctx->d_v_proj, ctx->d_v_HShd, seq, n_heads, hd, stream);

    cublasSetStream(ctx->cublas, stream);
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cbs;
    cbs = cublasGemmStridedBatchedEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        seq, seq, hd, &alpha,
        ctx->d_k_HShd, CUDA_R_16BF, hd, (long long) seq * hd,
        ctx->d_q_HShd, CUDA_R_16BF, hd, (long long) seq * hd,
        &beta,
        ctx->d_scores, CUDA_R_16BF, seq, (long long) seq * seq,
        n_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cbs != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla_vit_cuda): QK^T gemm @L%d failed (%d)\n", L, cbs); return -1; }

    const float scl = 1.0f / std::sqrt((float) hd);
    bitvla_softmax_scaled_bf16(ctx->d_scores, scl, n_heads * seq, seq, stream);

    cbs = cublasGemmStridedBatchedEx(
        ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        hd, seq, seq, &alpha,
        ctx->d_v_HShd, CUDA_R_16BF, hd,  (long long) seq * hd,
        ctx->d_scores, CUDA_R_16BF, seq, (long long) seq * seq,
        &beta,
        ctx->d_attn_out, CUDA_R_16BF, hd, (long long) seq * hd,
        n_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cbs != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla_vit_cuda): attn@V gemm @L%d failed (%d)\n", L, cbs); return -1; }

    bitvla_transpose_NshHd_to_sNhd_bf16(ctx->d_attn_out, ctx->d_attn_merged, n_heads, seq, hd, stream);

    bitvla_act_quant_cuda(ctx->d_attn_merged, ctx->d_act_int8_h, ctx->d_act_s, seq, H, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.o_packed, ctx->d_o_out, ctx->d_act_s, lr.o_ws, seq, H, H, stream);
    bitvla_add_bias_bf16(ctx->d_o_out, lr.o_b, ctx->d_o_out, seq, H, stream);

    bitvla_add_bf16(ctx->d_h, ctx->d_o_out, ctx->d_h, seq * H, stream);

    bitvla_layernorm_bf16(ctx->d_h, lr.ln2_w, lr.ln2_b, ctx->d_h_norm, ctx->ln_eps, seq, H, stream);
    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, H, stream);

    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.fc1_packed, ctx->d_fc1_dense,
                          ctx->d_act_s, lr.fc1_ws, seq, ffn, H, stream);
    bitvla_add_bias_bf16(ctx->d_fc1_dense, lr.fc1_b, ctx->d_fc1_dense, seq, ffn, stream);

    bitvla_gelu_tanh_bf16(ctx->d_fc1_dense, ctx->d_fc1_dense, seq * ffn, stream);

    cudaMemcpy2DAsync(
        ctx->d_fc1_padded, (size_t) ffn_pad * sizeof(__nv_bfloat16),
        ctx->d_fc1_dense,  (size_t) ffn     * sizeof(__nv_bfloat16),
        (size_t) ffn * sizeof(__nv_bfloat16),
        seq,
        cudaMemcpyDeviceToDevice, stream);

    bitvla_act_quant_cuda(ctx->d_fc1_padded, ctx->d_act_int8_ffn, ctx->d_act_s, seq, ffn_pad, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_ffn, lr.fc2_packed, ctx->d_fc2_out,
                          ctx->d_act_s, lr.fc2_ws, seq, H, ffn_pad, stream);
    bitvla_add_bias_bf16(ctx->d_fc2_out, lr.fc2_b, ctx->d_fc2_out, seq, H, stream);

    bitvla_add_bf16(ctx->d_h, ctx->d_fc2_out, ctx->d_h, seq * H, stream);
    return 0;
}

int bitvla_vit_cuda_forward(bitvla_vit_cuda_ctx* ctx,
                             const __nv_bfloat16* d_patches,
                             __nv_bfloat16* d_out,
                             cudaStream_t stream)
{
    const int seq = ctx->n_patches, H = ctx->hidden, P = ctx->patch_flat, M = ctx->mm_out;
    cublasSetStream(ctx->cublas, stream);
    const float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t cbs = cublasGemmEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        H, seq, P, &alpha,
        ctx->patch_w, CUDA_R_16BF, P,
        d_patches,    CUDA_R_16BF, P,
        &beta,
        ctx->d_h,     CUDA_R_16BF, H,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cbs != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla_vit_cuda): patch_embed gemm failed (%d)\n", cbs); return -1; }

    bitvla_add_bias_bf16(ctx->d_h, ctx->patch_b, ctx->d_h, seq, H, stream);

    bitvla_add_bf16(ctx->d_h, ctx->pos_emb, ctx->d_h, seq * H, stream);

    for (int L = 0; L < ctx->n_layers; ++L) {
        int rc = run_vit_layer(ctx, L, stream);
        if (rc != 0) return rc;
    }

    cbs = cublasGemmEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        M, seq, H, &alpha,
        ctx->mm_W1, CUDA_R_16BF, H,
        ctx->d_h,   CUDA_R_16BF, H,
        &beta,
        ctx->d_mm_h1, CUDA_R_16BF, M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cbs != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla_vit_cuda): MM linear_1 gemm failed (%d)\n", cbs); return -1; }
    bitvla_add_bias_bf16(ctx->d_mm_h1, ctx->mm_b1, ctx->d_mm_h1, seq, M, stream);
    gelu_erf_bf16(ctx->d_mm_h1, ctx->d_mm_h1, seq * M, stream);

    cbs = cublasGemmEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        M, seq, M, &alpha,
        ctx->mm_W2,   CUDA_R_16BF, M,
        ctx->d_mm_h1, CUDA_R_16BF, M,
        &beta,
        d_out, CUDA_R_16BF, M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cbs != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla_vit_cuda): MM linear_2 gemm failed (%d)\n", cbs); return -1; }
    bitvla_add_bias_bf16(d_out, ctx->mm_b2, d_out, seq, M, stream);
    return 0;
}
