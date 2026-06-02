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

#include "bitvla_fp32head_cuda.h"
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_OK_RET(c) do { cudaError_t _e = (c); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "vla(bitvla_fp32head): %s at %s:%d (%s)\n", cudaGetErrorString(_e), __FILE__, __LINE__, #c); \
    return -1; } } while (0)
#define CUDA_OK_NULL(c) do { cudaError_t _e = (c); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "vla(bitvla_fp32head): %s at %s:%d (%s)\n", cudaGetErrorString(_e), __FILE__, __LINE__, #c); \
    return nullptr; } } while (0)

struct bitvla_fp32head_cuda_ctx {
    cublasHandle_t cublas;
    int proprio_dim;
    int lm_hidden;
    int chunk;
    int action_dim;
    int ah_in_dim;
    float ah_ln_eps;

    float* pp_fc1_w; float* pp_fc1_b;
    float* pp_fc2_w; float* pp_fc2_b;

    float* d_state;
    float* d_pp_h1;
    float* d_pp_out;

    float* ah_ln1_w; float* ah_ln1_b;
    float* ah_fc1_w; float* ah_fc1_b;
    float* ah_b0_lnw; float* ah_b0_lnb; float* ah_b0_w; float* ah_b0_b;
    float* ah_b1_lnw; float* ah_b1_lnb; float* ah_b1_w; float* ah_b1_b;
    float* ah_ln2_w; float* ah_ln2_b;
    float* ah_fc2_w; float* ah_fc2_b;

    float* d_ah_in;
    float* d_ah_norm_big;
    float* d_ah_h;
    float* d_ah_tmp;
    float* d_ah_tmp2;
    float* d_ah_out;
};

static float* upload_f32(const float* h, size_t n) {
    float* d = nullptr;
    cudaError_t e = cudaMalloc(&d, n * sizeof(float));
    if (e != cudaSuccess) { std::fprintf(stderr, "vla(bitvla_fp32head): cudaMalloc failed (%zu)\n", n); return nullptr; }
    e = cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { std::fprintf(stderr, "vla(bitvla_fp32head): cudaMemcpy H2D failed (%zu)\n", n); cudaFree(d); return nullptr; }
    return d;
}

__global__ void gelu_erf_fp32_kernel(const float* __restrict__ in, float* __restrict__ out, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = in[i];
    out[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

__global__ void relu_fp32_kernel(float* __restrict__ inout, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = inout[i];
    if (v < 0.0f) inout[i] = 0.0f;
}

template <int BLOCK>
__global__ void layernorm_fp32_kernel(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       const float* __restrict__ b,
                                       float* __restrict__ out,
                                       float eps, int K) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = x + (size_t)m * K;
    float*       o   = out + (size_t)m * K;

    float sum = 0.0f;
    for (int k = tid; k < K; k += BLOCK) sum += row[k];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    __shared__ float smem[32];
    if ((tid & 31) == 0) smem[tid >> 5] = sum;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK + 31) / 32) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) smem[0] = v;
    }
    __syncthreads();
    const float mean = smem[0] / (float)K;

    float vsum = 0.0f;
    for (int k = tid; k < K; k += BLOCK) {
        float v = row[k] - mean;
        vsum += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) vsum += __shfl_down_sync(0xffffffff, vsum, off);
    if ((tid & 31) == 0) smem[tid >> 5] = vsum;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK + 31) / 32) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) smem[0] = v;
    }
    __syncthreads();
    const float inv_std = rsqrtf(smem[0] / (float)K + eps);

    for (int k = tid; k < K; k += BLOCK) {
        o[k] = (row[k] - mean) * inv_std * w[k] + b[k];
    }
}

__global__ void add_bias_fp32_kernel(const float* x, const float* bias,
                                      float* out, int M, int K) {
    const int m = blockIdx.x;
    const int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const size_t i = (size_t)m * K + k;
    out[i] = x[i] + bias[k];
}

__global__ void add_fp32_kernel(const float* a, const float* b, float* out, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a[i] + b[i];
}

static int linear_bias_fp32(
    cublasHandle_t cublas, cudaStream_t stream,
    const float* W, const float* bias,
    const float* x,
    float* out,
    int M, int N_out, int K_in) {
    cublasSetStream(cublas, stream);
    const float a = 1.0f, b = 0.0f;
    cublasStatus_t cbs = cublasSgemm(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N_out, M, K_in, &a,
        W, K_in,
        x, K_in,
        &b,
        out, N_out);
    if (cbs != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(bitvla_fp32head): sgemm failed (%d) M=%d N=%d K=%d\n", cbs, M, N_out, K_in);
        return -1;
    }
    if (bias) {
        const int B = 256;
        const int n_kb = (N_out + B - 1) / B;
        add_bias_fp32_kernel<<<dim3(M, n_kb, 1), dim3(B, 1, 1), 0, stream>>>(out, bias, out, M, N_out);
    }
    return 0;
}

extern "C" bitvla_fp32head_cuda_ctx* bitvla_fp32head_cuda_init(
    int proprio_dim, int lm_hidden, int chunk, int action_dim,
    float ah_ln_eps,
    const float* pp_fc1_w, const float* pp_fc1_b,
    const float* pp_fc2_w, const float* pp_fc2_b,
    const float* ah_ln1_w, const float* ah_ln1_b,
    const float* ah_fc1_w, const float* ah_fc1_b,
    const float* ah_b0_lnw, const float* ah_b0_lnb,
    const float* ah_b0_w,   const float* ah_b0_b,
    const float* ah_b1_lnw, const float* ah_b1_lnb,
    const float* ah_b1_w,   const float* ah_b1_b,
    const float* ah_ln2_w,  const float* ah_ln2_b,
    const float* ah_fc2_w,  const float* ah_fc2_b)
{
    auto* ctx = new bitvla_fp32head_cuda_ctx;
    std::memset(ctx, 0, sizeof(*ctx));
    if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(bitvla_fp32head): cublasCreate failed\n");
        delete ctx; return nullptr;
    }
    ctx->proprio_dim = proprio_dim;
    ctx->lm_hidden   = lm_hidden;
    ctx->chunk       = chunk;
    ctx->action_dim  = action_dim;
    ctx->ah_in_dim   = action_dim * lm_hidden;
    ctx->ah_ln_eps   = ah_ln_eps;

    auto try_upload = [&](const float* h, size_t n, const char* ) -> float* {
        return upload_f32(h, n);
    };
    ctx->pp_fc1_w = try_upload(pp_fc1_w, (size_t)lm_hidden * proprio_dim, "pp_fc1_w");
    ctx->pp_fc1_b = try_upload(pp_fc1_b, (size_t)lm_hidden, "pp_fc1_b");
    ctx->pp_fc2_w = try_upload(pp_fc2_w, (size_t)lm_hidden * lm_hidden, "pp_fc2_w");
    ctx->pp_fc2_b = try_upload(pp_fc2_b, (size_t)lm_hidden, "pp_fc2_b");

    ctx->ah_ln1_w = try_upload(ah_ln1_w, (size_t)ctx->ah_in_dim, "ah_ln1_w");
    ctx->ah_ln1_b = try_upload(ah_ln1_b, (size_t)ctx->ah_in_dim, "ah_ln1_b");
    ctx->ah_fc1_w = try_upload(ah_fc1_w, (size_t)lm_hidden * ctx->ah_in_dim, "ah_fc1_w");
    ctx->ah_fc1_b = try_upload(ah_fc1_b, (size_t)lm_hidden, "ah_fc1_b");

    ctx->ah_b0_lnw = try_upload(ah_b0_lnw, (size_t)lm_hidden, "ah_b0_lnw");
    ctx->ah_b0_lnb = try_upload(ah_b0_lnb, (size_t)lm_hidden, "ah_b0_lnb");
    ctx->ah_b0_w   = try_upload(ah_b0_w,   (size_t)lm_hidden * lm_hidden, "ah_b0_w");
    ctx->ah_b0_b   = try_upload(ah_b0_b,   (size_t)lm_hidden, "ah_b0_b");

    ctx->ah_b1_lnw = try_upload(ah_b1_lnw, (size_t)lm_hidden, "ah_b1_lnw");
    ctx->ah_b1_lnb = try_upload(ah_b1_lnb, (size_t)lm_hidden, "ah_b1_lnb");
    ctx->ah_b1_w   = try_upload(ah_b1_w,   (size_t)lm_hidden * lm_hidden, "ah_b1_w");
    ctx->ah_b1_b   = try_upload(ah_b1_b,   (size_t)lm_hidden, "ah_b1_b");

    ctx->ah_ln2_w = try_upload(ah_ln2_w, (size_t)lm_hidden, "ah_ln2_w");
    ctx->ah_ln2_b = try_upload(ah_ln2_b, (size_t)lm_hidden, "ah_ln2_b");
    ctx->ah_fc2_w = try_upload(ah_fc2_w, (size_t)action_dim * lm_hidden, "ah_fc2_w");
    ctx->ah_fc2_b = try_upload(ah_fc2_b, (size_t)action_dim, "ah_fc2_b");

    CUDA_OK_NULL(cudaMalloc(&ctx->d_state,   (size_t)proprio_dim * sizeof(float)));
    CUDA_OK_NULL(cudaMalloc(&ctx->d_pp_h1,   (size_t)lm_hidden  * sizeof(float)));
    CUDA_OK_NULL(cudaMalloc(&ctx->d_pp_out,  (size_t)lm_hidden  * sizeof(float)));

    CUDA_OK_NULL(cudaMalloc(&ctx->d_ah_in,       (size_t)chunk * ctx->ah_in_dim * sizeof(float)));
    CUDA_OK_NULL(cudaMalloc(&ctx->d_ah_norm_big, (size_t)chunk * ctx->ah_in_dim * sizeof(float)));
    CUDA_OK_NULL(cudaMalloc(&ctx->d_ah_h,         (size_t)chunk * lm_hidden  * sizeof(float)));
    CUDA_OK_NULL(cudaMalloc(&ctx->d_ah_tmp,       (size_t)chunk * lm_hidden  * sizeof(float)));
    CUDA_OK_NULL(cudaMalloc(&ctx->d_ah_tmp2,      (size_t)chunk * lm_hidden  * sizeof(float)));
    CUDA_OK_NULL(cudaMalloc(&ctx->d_ah_out,       (size_t)chunk * action_dim * sizeof(float)));

    void* required[] = {
        ctx->pp_fc1_w, ctx->pp_fc1_b, ctx->pp_fc2_w, ctx->pp_fc2_b,
        ctx->ah_ln1_w, ctx->ah_ln1_b, ctx->ah_fc1_w, ctx->ah_fc1_b,
        ctx->ah_b0_lnw, ctx->ah_b0_lnb, ctx->ah_b0_w, ctx->ah_b0_b,
        ctx->ah_b1_lnw, ctx->ah_b1_lnb, ctx->ah_b1_w, ctx->ah_b1_b,
        ctx->ah_ln2_w, ctx->ah_ln2_b, ctx->ah_fc2_w, ctx->ah_fc2_b,
    };
    for (void* p : required) if (!p) {
        std::fprintf(stderr, "vla(bitvla_fp32head): a weight upload returned null; init failed\n");
        bitvla_fp32head_cuda_free(ctx);
        return nullptr;
    }
    return ctx;
}

extern "C" int bitvla_fp32head_proprio_forward(
    bitvla_fp32head_cuda_ctx* ctx,
    const float* host_state,
    float* host_out,
    cudaStream_t stream)
{

    CUDA_OK_RET(cudaMemcpyAsync(ctx->d_state, host_state, (size_t)ctx->proprio_dim * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

    if (linear_bias_fp32(ctx->cublas, stream, ctx->pp_fc1_w, ctx->pp_fc1_b,
                         ctx->d_state, ctx->d_pp_h1, 1, ctx->lm_hidden, ctx->proprio_dim) != 0) return -1;

    {
        const int B = 256;
        const int N = ctx->lm_hidden;
        gelu_erf_fp32_kernel<<<dim3((N + B - 1) / B), dim3(B), 0, stream>>>(ctx->d_pp_h1, ctx->d_pp_h1, N);
    }

    if (linear_bias_fp32(ctx->cublas, stream, ctx->pp_fc2_w, ctx->pp_fc2_b,
                         ctx->d_pp_h1, ctx->d_pp_out, 1, ctx->lm_hidden, ctx->lm_hidden) != 0) return -1;

    CUDA_OK_RET(cudaMemcpyAsync(host_out, ctx->d_pp_out, (size_t)ctx->lm_hidden * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    CUDA_OK_RET(cudaStreamSynchronize(stream));
    return 0;
}

extern "C" int bitvla_fp32head_action_forward(
    bitvla_fp32head_cuda_ctx* ctx,
    const float* host_ah_input,
    float* host_norm_actions,
    cudaStream_t stream)
{
    const int M  = ctx->chunk;
    const int Di = ctx->ah_in_dim;
    const int H  = ctx->lm_hidden;
    const int A  = ctx->action_dim;

    CUDA_OK_RET(cudaMemcpyAsync(ctx->d_ah_in, host_ah_input, (size_t)M * Di * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

    {
        constexpr int B = 256;
        layernorm_fp32_kernel<B><<<dim3(M), dim3(B), 0, stream>>>(
            ctx->d_ah_in, ctx->ah_ln1_w, ctx->ah_ln1_b, ctx->d_ah_norm_big, ctx->ah_ln_eps, Di);
    }

    if (linear_bias_fp32(ctx->cublas, stream, ctx->ah_fc1_w, ctx->ah_fc1_b,
                         ctx->d_ah_norm_big, ctx->d_ah_h, M, H, Di) != 0) return -1;

    {
        const int B = 256;
        const int N = M * H;
        relu_fp32_kernel<<<dim3((N + B - 1) / B), dim3(B), 0, stream>>>(ctx->d_ah_h, N);
    }

    {
        constexpr int B = 256;
        layernorm_fp32_kernel<B><<<dim3(M), dim3(B), 0, stream>>>(
            ctx->d_ah_h, ctx->ah_b0_lnw, ctx->ah_b0_lnb, ctx->d_ah_tmp, ctx->ah_ln_eps, H);
    }

    if (linear_bias_fp32(ctx->cublas, stream, ctx->ah_b0_w, ctx->ah_b0_b,
                         ctx->d_ah_tmp, ctx->d_ah_tmp2, M, H, H) != 0) return -1;

    {
        const int B = 256;
        const int N = M * H;
        relu_fp32_kernel<<<dim3((N + B - 1) / B), dim3(B), 0, stream>>>(ctx->d_ah_tmp2, N);
    }

    {
        const int B = 256;
        const int N = M * H;
        add_fp32_kernel<<<dim3((N + B - 1) / B), dim3(B), 0, stream>>>(ctx->d_ah_h, ctx->d_ah_tmp2, ctx->d_ah_h, N);
    }

    {
        constexpr int B = 256;
        layernorm_fp32_kernel<B><<<dim3(M), dim3(B), 0, stream>>>(
            ctx->d_ah_h, ctx->ah_b1_lnw, ctx->ah_b1_lnb, ctx->d_ah_tmp, ctx->ah_ln_eps, H);
    }
    if (linear_bias_fp32(ctx->cublas, stream, ctx->ah_b1_w, ctx->ah_b1_b,
                         ctx->d_ah_tmp, ctx->d_ah_tmp2, M, H, H) != 0) return -1;
    {
        const int B = 256;
        const int N = M * H;
        relu_fp32_kernel<<<dim3((N + B - 1) / B), dim3(B), 0, stream>>>(ctx->d_ah_tmp2, N);
    }
    {
        const int B = 256;
        const int N = M * H;
        add_fp32_kernel<<<dim3((N + B - 1) / B), dim3(B), 0, stream>>>(ctx->d_ah_h, ctx->d_ah_tmp2, ctx->d_ah_h, N);
    }

    {
        constexpr int B = 256;
        layernorm_fp32_kernel<B><<<dim3(M), dim3(B), 0, stream>>>(
            ctx->d_ah_h, ctx->ah_ln2_w, ctx->ah_ln2_b, ctx->d_ah_tmp, ctx->ah_ln_eps, H);
    }
    if (linear_bias_fp32(ctx->cublas, stream, ctx->ah_fc2_w, ctx->ah_fc2_b,
                         ctx->d_ah_tmp, ctx->d_ah_out, M, A, H) != 0) return -1;

    CUDA_OK_RET(cudaMemcpyAsync(host_norm_actions, ctx->d_ah_out, (size_t)M * A * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    CUDA_OK_RET(cudaStreamSynchronize(stream));
    return 0;
}

extern "C" void bitvla_fp32head_cuda_free(bitvla_fp32head_cuda_ctx* ctx) {
    if (!ctx) return;
    float* ws[] = {
        ctx->pp_fc1_w, ctx->pp_fc1_b, ctx->pp_fc2_w, ctx->pp_fc2_b,
        ctx->ah_ln1_w, ctx->ah_ln1_b, ctx->ah_fc1_w, ctx->ah_fc1_b,
        ctx->ah_b0_lnw, ctx->ah_b0_lnb, ctx->ah_b0_w, ctx->ah_b0_b,
        ctx->ah_b1_lnw, ctx->ah_b1_lnb, ctx->ah_b1_w, ctx->ah_b1_b,
        ctx->ah_ln2_w, ctx->ah_ln2_b, ctx->ah_fc2_w, ctx->ah_fc2_b,
        ctx->d_state, ctx->d_pp_h1, ctx->d_pp_out,
        ctx->d_ah_in, ctx->d_ah_norm_big, ctx->d_ah_h, ctx->d_ah_tmp, ctx->d_ah_tmp2, ctx->d_ah_out,
    };
    for (float* p : ws) if (p) cudaFree(p);
    if (ctx->cublas) cublasDestroy(ctx->cublas);
    delete ctx;
}
