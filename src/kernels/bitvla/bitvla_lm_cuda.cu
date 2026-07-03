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

#include "bitvla_lm_cuda.h"
#include "bitnet_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void bitlinear_int8xint2_m(int8_t* A, int8_t* B, __nv_bfloat16* out,
                                       float* s, float* ws,
                                       int M, int N, int K, cudaStream_t stream);
extern "C" void bitvla_act_quant_cuda(const __nv_bfloat16* in, int8_t* out,
                                       float* scales,
                                       int M, int K, cudaStream_t stream);

extern "C" void gate_up_fused_sqrelu_mul_bf16(const __nv_bfloat16* gu, __nv_bfloat16* out,
                                               int seq, int ffn, cudaStream_t stream);

template <int BLOCK>
__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     __nv_bfloat16* __restrict__ out,
                                     float eps, int K)
{
    const int m   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const __nv_bfloat16* row = x + m * K;
    __nv_bfloat16*       o   = out + m * K;

    float ss = 0.0f;
    for (int k = tid; k < K; k += BLOCK) {
        float v = __bfloat162float(row[k]);
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    __shared__ float smem[32];
    if ((tid & 31) == 0) smem[tid >> 5] = ss;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK + 31) / 32) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) smem[0] = v;
    }
    __syncthreads();
    const float mean  = smem[0] / (float)K;
    const float scale = rsqrtf(mean + eps);

    for (int k = tid; k < K; k += BLOCK) {
        float v  = __bfloat162float(row[k]) * scale;
        float wv = __bfloat162float(w[k]);
        o[k] = __float2bfloat16(v * wv);
    }
}

template <int BLOCK>
__global__ void rope_neox_bf16_kernel(__nv_bfloat16* __restrict__ inout,
                                       const float* __restrict__ cos_tab,
                                       const float* __restrict__ sin_tab,
                                       int S, int D)
{
    const int h    = (int)blockIdx.x;
    const int s    = (int)blockIdx.y;
    const int tid  = (int)threadIdx.x;
    const int half = D / 2;
    __nv_bfloat16* row = inout + ((size_t)h * S + s) * D;
    const float* c_row = cos_tab + (size_t)s * half;
    const float* s_row = sin_tab + (size_t)s * half;
    for (int k = tid; k < half; k += BLOCK) {
        float c = c_row[k];
        float si= s_row[k];
        float a = __bfloat162float(row[k]);
        float b = __bfloat162float(row[k + half]);
        row[k]        = __float2bfloat16(a * c - b * si);
        row[k + half] = __float2bfloat16(b * c + a * si);
    }
}

template <int BLOCK>
__global__ void softmax_scaled_bf16_kernel(__nv_bfloat16* __restrict__ inout,
                                            float scale, int S)
{
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    __nv_bfloat16* r = inout + (size_t)row * S;

    float mx = -INFINITY;
    for (int i = tid; i < S; i += BLOCK) {
        float v = __bfloat162float(r[i]) * scale;
        if (v > mx) mx = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xffffffff, mx, off);
        if (other > mx) mx = other;
    }
    __shared__ float smem[32];
    if ((tid & 31) == 0) smem[tid >> 5] = mx;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK + 31) / 32) ? smem[tid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(0xffffffff, v, off);
            if (other > v) v = other;
        }
        if (tid == 0) smem[0] = v;
    }
    __syncthreads();
    const float max_v = smem[0];

    float s_sum = 0.0f;
    for (int i = tid; i < S; i += BLOCK) {
        s_sum += expf(__bfloat162float(r[i]) * scale - max_v);
    }
    for (int off = 16; off > 0; off >>= 1) s_sum += __shfl_down_sync(0xffffffff, s_sum, off);
    if ((tid & 31) == 0) smem[tid >> 5] = s_sum;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < (BLOCK + 31) / 32) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) smem[0] = v;
    }
    __syncthreads();
    const float inv_sum = 1.0f / smem[0];

    for (int i = tid; i < S; i += BLOCK) {
        float v = expf(__bfloat162float(r[i]) * scale - max_v) * inv_sum;
        r[i] = __float2bfloat16(v);
    }
}

__global__ void squared_relu_mul_bf16_kernel(const __nv_bfloat16* g,
                                              const __nv_bfloat16* u,
                                              __nv_bfloat16* out, int N) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) return;
    float gv = __bfloat162float(g[i]);
    if (gv < 0.0f) gv = 0.0f;
    out[i] = __float2bfloat16(gv * gv * __bfloat162float(u[i]));
}

__global__ void add_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                 __nv_bfloat16* out, int N) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) return;
    out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

__global__ void repeat_kv_bf16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out,
                                       int n_q, int n_kv, int seq, int hd) {
    const int q_h = (int)blockIdx.x;
    const int s   = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int kv_h = q_h * n_kv / n_q;
    for (int k = tid; k < hd; k += blockDim.x) {
        out[((size_t)q_h * seq + s) * hd + k] = in[((size_t)kv_h * seq + s) * hd + k];
    }
}

__global__ void transpose_sHhd_to_HShd_bf16_kernel(const __nv_bfloat16* in,
                                                    __nv_bfloat16* out,
                                                    int S, int H, int hd) {
    const int s   = (int)blockIdx.y;
    const int h   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    for (int k = tid; k < hd; k += blockDim.x) {
        out[((size_t)h * S + s) * hd + k] = in[((size_t)s * H + h) * hd + k];
    }
}

__global__ void transpose_HShd_to_sHhd_bf16_kernel(const __nv_bfloat16* in,
                                                    __nv_bfloat16* out,
                                                    int H, int S, int hd) {
    const int h   = (int)blockIdx.x;
    const int s   = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    for (int k = tid; k < hd; k += blockDim.x) {
        out[((size_t)s * H + h) * hd + k] = in[((size_t)h * S + s) * hd + k];
    }
}

__global__ void gather_rows_bf16_kernel(const __nv_bfloat16* in,
                                         __nv_bfloat16* out,
                                         const int32_t* row_ids,
                                         int K) {
    const int m = (int)blockIdx.x;
    const int r = row_ids[m];
    const int tid = (int)threadIdx.x;
    for (int k = tid; k < K; k += blockDim.x) {
        out[(size_t)m * K + k] = in[(size_t)r * K + k];
    }
}

extern "C" void bitvla_rmsnorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                     __nv_bfloat16* out, float eps, int M, int K,
                                     cudaStream_t stream) {
    constexpr int B = 256;
    rmsnorm_bf16_kernel<B><<<dim3(M, 1, 1), dim3(B, 1, 1), 0, stream>>>(x, w, out, eps, K);
}
extern "C" void bitvla_rope_neox_bf16(__nv_bfloat16* inout, const float* cos_tab,
                                       const float* sin_tab, int H, int S, int D,
                                       cudaStream_t stream) {
    constexpr int B = 128;
    rope_neox_bf16_kernel<B><<<dim3(H, S, 1), dim3(B, 1, 1), 0, stream>>>(inout, cos_tab, sin_tab, S, D);
}
extern "C" void bitvla_softmax_scaled_bf16(__nv_bfloat16* inout, float scale,
                                            int n_rows, int S, cudaStream_t stream) {
    constexpr int B = 256;
    softmax_scaled_bf16_kernel<B><<<dim3(n_rows, 1, 1), dim3(B, 1, 1), 0, stream>>>(inout, scale, S);
}
extern "C" void bitvla_squared_relu_mul_bf16(const __nv_bfloat16* g, const __nv_bfloat16* u,
                                              __nv_bfloat16* out, int N, cudaStream_t stream) {
    constexpr int B = 256;
    squared_relu_mul_bf16_kernel<<<dim3((N + B - 1) / B, 1, 1), dim3(B, 1, 1), 0, stream>>>(g, u, out, N);
}
extern "C" void bitvla_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                 __nv_bfloat16* out, int N, cudaStream_t stream) {
    constexpr int B = 256;
    add_bf16_kernel<<<dim3((N + B - 1) / B, 1, 1), dim3(B, 1, 1), 0, stream>>>(a, b, out, N);
}
extern "C" void bitvla_repeat_kv_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                                       int n_q, int n_kv, int seq, int hd, cudaStream_t stream) {
    constexpr int B = 128;
    repeat_kv_bf16_kernel<<<dim3(n_q, seq, 1), dim3(B, 1, 1), 0, stream>>>(in, out, n_q, n_kv, seq, hd);
}
extern "C" void bitvla_transpose_sNhd_to_NshHd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                                                     int S, int N, int hd, cudaStream_t stream) {
    constexpr int B = 64;
    transpose_sHhd_to_HShd_bf16_kernel<<<dim3(N, S, 1), dim3(B, 1, 1), 0, stream>>>(in, out, S, N, hd);
}
extern "C" void bitvla_transpose_NshHd_to_sNhd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                                                     int N, int S, int hd, cudaStream_t stream) {
    constexpr int B = 64;
    transpose_HShd_to_sHhd_bf16_kernel<<<dim3(N, S, 1), dim3(B, 1, 1), 0, stream>>>(in, out, N, S, hd);
}
extern "C" void bitvla_gather_rows_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                                         const int32_t* row_ids, int n_rows, int K,
                                         cudaStream_t stream) {
    constexpr int B = 128;
    gather_rows_bf16_kernel<<<dim3(n_rows, 1, 1), dim3(B, 1, 1), 0, stream>>>(in, out, row_ids, K);
}

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "vla(bitvla_lm_cuda): %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return -1; } } while (0)
#define CUDA_OKV(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "vla(bitvla_lm_cuda): %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return nullptr; } } while (0)

struct bitvla_lm_cuda_ctx {
    int hidden, n_q, n_kv, head_dim, ffn, n_layers, max_seq;
    float rope_base, rms_eps;
    int hidden_q, hidden_kv;

    cublasHandle_t cublas;
    std::vector<bitvla_lm_layer_cuda> layers;
    __nv_bfloat16* output_norm_w = nullptr;

    float* d_cos = nullptr;
    float* d_sin = nullptr;

    __nv_bfloat16* d_h         = nullptr;
    __nv_bfloat16* d_h_norm    = nullptr;
    int8_t*        d_act_int8_h= nullptr;
    float*         d_act_s     = nullptr;
    __nv_bfloat16* d_qkv       = nullptr;
    __nv_bfloat16* d_q_HShd    = nullptr;
    __nv_bfloat16* d_k_HShd    = nullptr;
    __nv_bfloat16* d_v_HShd    = nullptr;
    __nv_bfloat16* d_k_rep     = nullptr;
    __nv_bfloat16* d_v_rep     = nullptr;
    __nv_bfloat16* d_scores    = nullptr;
    __nv_bfloat16* d_attn_out  = nullptr;
    __nv_bfloat16* d_attn_merged = nullptr;
    __nv_bfloat16* d_o_out     = nullptr;
    int8_t*        d_act_int8_ffn = nullptr;
    __nv_bfloat16* d_gate_up   = nullptr;
    __nv_bfloat16* d_gate_sq_up= nullptr;
    __nv_bfloat16* d_down_out  = nullptr;
};

extern "C" bitvla_lm_cuda_ctx* bitvla_lm_cuda_init(int hidden, int n_q, int n_kv, int head_dim,
                                                     int ffn, int n_layers,
                                                     float rope_base, float rms_eps, int max_seq)
{
    auto* ctx = new bitvla_lm_cuda_ctx{};
    ctx->hidden = hidden; ctx->n_q = n_q; ctx->n_kv = n_kv; ctx->head_dim = head_dim;
    ctx->ffn = ffn; ctx->n_layers = n_layers;
    ctx->rope_base = rope_base; ctx->rms_eps = rms_eps; ctx->max_seq = max_seq;
    ctx->hidden_q  = n_q  * head_dim;
    ctx->hidden_kv = n_kv * head_dim;
    ctx->layers.resize(n_layers);

    cublasStatus_t cbs = cublasCreate(&ctx->cublas);
    if (cbs != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla_lm_cuda): cublasCreate failed\n"); delete ctx; return nullptr; }

    const int half = head_dim / 2;
    std::vector<float> h_cos((size_t)max_seq * half), h_sin((size_t)max_seq * half);
    for (int s = 0; s < max_seq; ++s) {
        for (int k = 0; k < half; ++k) {
            float freq = 1.0f / std::pow(rope_base, (float)(2 * k) / (float)head_dim);
            float ang  = (float)s * freq;
            h_cos[(size_t)s * half + k] = std::cos(ang);
            h_sin[(size_t)s * half + k] = std::sin(ang);
        }
    }
    CUDA_OKV(cudaMalloc(&ctx->d_cos, (size_t)max_seq * half * sizeof(float)));
    CUDA_OKV(cudaMalloc(&ctx->d_sin, (size_t)max_seq * half * sizeof(float)));
    CUDA_OKV(cudaMemcpy(ctx->d_cos, h_cos.data(), h_cos.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OKV(cudaMemcpy(ctx->d_sin, h_sin.data(), h_sin.size() * sizeof(float), cudaMemcpyHostToDevice));

    const size_t bf16 = sizeof(__nv_bfloat16);
    CUDA_OKV(cudaMalloc(&ctx->d_h,           (size_t)max_seq * hidden       * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_h_norm,      (size_t)max_seq * hidden       * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_act_int8_h,  (size_t)max_seq * hidden));
    CUDA_OKV(cudaMalloc(&ctx->d_act_s,       (size_t)max_seq                * sizeof(float)));
    CUDA_OKV(cudaMalloc(&ctx->d_qkv,         (size_t)max_seq * (ctx->hidden_q + 2 * ctx->hidden_kv) * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_q_HShd,      (size_t)n_q  * max_seq * head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_k_HShd,      (size_t)n_kv * max_seq * head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_v_HShd,      (size_t)n_kv * max_seq * head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_k_rep,       (size_t)n_q  * max_seq * head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_v_rep,       (size_t)n_q  * max_seq * head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_scores,      (size_t)n_q  * max_seq * max_seq  * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_attn_out,    (size_t)n_q  * max_seq * head_dim * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_attn_merged, (size_t)max_seq * ctx->hidden_q   * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_o_out,       (size_t)max_seq * hidden          * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_act_int8_ffn,(size_t)max_seq * ffn));
    CUDA_OKV(cudaMalloc(&ctx->d_gate_up,     (size_t)max_seq * 2 * ffn * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_gate_sq_up,  (size_t)max_seq * ffn     * bf16));
    CUDA_OKV(cudaMalloc(&ctx->d_down_out,    (size_t)max_seq * hidden  * bf16));
    return ctx;
}

extern "C" void bitvla_lm_cuda_free(bitvla_lm_cuda_ctx* ctx) {
    if (!ctx) return;
    cublasDestroy(ctx->cublas);
    cudaFree(ctx->d_cos);  cudaFree(ctx->d_sin);
    cudaFree(ctx->d_h);    cudaFree(ctx->d_h_norm);
    cudaFree(ctx->d_act_int8_h);    cudaFree(ctx->d_act_s);
    cudaFree(ctx->d_qkv);
    cudaFree(ctx->d_q_HShd); cudaFree(ctx->d_k_HShd); cudaFree(ctx->d_v_HShd);
    cudaFree(ctx->d_k_rep);  cudaFree(ctx->d_v_rep);
    cudaFree(ctx->d_scores); cudaFree(ctx->d_attn_out); cudaFree(ctx->d_attn_merged);
    cudaFree(ctx->d_o_out);
    cudaFree(ctx->d_act_int8_ffn);
    cudaFree(ctx->d_gate_up); cudaFree(ctx->d_gate_sq_up); cudaFree(ctx->d_down_out);
    delete ctx;
}

extern "C" void bitvla_lm_cuda_set_layer(bitvla_lm_cuda_ctx* ctx, int L,
                                          const bitvla_lm_layer_cuda* layer) {
    ctx->layers[L] = *layer;
}
extern "C" void bitvla_lm_cuda_set_output_norm(bitvla_lm_cuda_ctx* ctx, const __nv_bfloat16* w) {
    ctx->output_norm_w = const_cast<__nv_bfloat16*>(w);
}

static int run_layer(bitvla_lm_cuda_ctx* ctx, int L, int seq, cudaStream_t stream) {
    const auto& lr = ctx->layers[L];
    const int hidden = ctx->hidden, n_q = ctx->n_q, n_kv = ctx->n_kv, hd = ctx->head_dim;
    const int ffn = ctx->ffn, hq = ctx->hidden_q, hkv = ctx->hidden_kv;

    const char* l0_dir = (L == 0) ? std::getenv("VLA_BITVLA_DUMP_L0") : nullptr;
    auto l0_dump = [&](const char* name, const __nv_bfloat16* d_ptr, size_t n) {
        if (!l0_dir) return;
        cudaStreamSynchronize(stream);
        std::vector<__nv_bfloat16> tmp(n);
        std::vector<float> f32(n);
        cudaMemcpy(tmp.data(), d_ptr, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n; ++i) {
            uint16_t u = *reinterpret_cast<uint16_t*>(&tmp[i]);
            uint32_t b = ((uint32_t) u) << 16;
            float f; std::memcpy(&f, &b, 4);
            f32[i] = f;
        }
        std::string path = std::string(l0_dir) + "/" + name + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f) { std::fwrite(f32.data(), sizeof(float), n, f); std::fclose(f); }
    };

    bitvla_rmsnorm_bf16(ctx->d_h, lr.attn_norm_w, ctx->d_h_norm, ctx->rms_eps, seq, hidden, stream);
    l0_dump("L0_01_attn_norm", ctx->d_h_norm, (size_t)seq * hidden);

    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, hidden, stream);

    __nv_bfloat16* q_dense = ctx->d_qkv;
    __nv_bfloat16* k_dense = ctx->d_qkv + (size_t)seq * hq;
    __nv_bfloat16* v_dense = ctx->d_qkv + (size_t)seq * (hq + hkv);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.q_packed, q_dense,
                          ctx->d_act_s, lr.q_ws, seq, hq,  hidden, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.k_packed, k_dense,
                          ctx->d_act_s, lr.k_ws, seq, hkv, hidden, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.v_packed, v_dense,
                          ctx->d_act_s, lr.v_ws, seq, hkv, hidden, stream);
    l0_dump("L0_02_qkv_proj", ctx->d_qkv, (size_t)seq * (hq + 2*hkv));

    // Split the interleaved [seq, H*hd] projections into head-major [H, seq, hd]
    // with one kernel per tensor instead of a cudaMemcpy2DAsync per head (30 tiny
    // launches per layer). Same transpose the ViT head-split already uses.
    bitvla_transpose_sNhd_to_NshHd_bf16(q_dense, ctx->d_q_HShd, seq, n_q,  hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(k_dense, ctx->d_k_HShd, seq, n_kv, hd, stream);
    bitvla_transpose_sNhd_to_NshHd_bf16(v_dense, ctx->d_v_HShd, seq, n_kv, hd, stream);

    bitvla_rope_neox_bf16(ctx->d_q_HShd, ctx->d_cos, ctx->d_sin, n_q,  seq, hd, stream);
    bitvla_rope_neox_bf16(ctx->d_k_HShd, ctx->d_cos, ctx->d_sin, n_kv, seq, hd, stream);

    bitvla_repeat_kv_bf16(ctx->d_k_HShd, ctx->d_k_rep, n_q, n_kv, seq, hd, stream);
    bitvla_repeat_kv_bf16(ctx->d_v_HShd, ctx->d_v_rep, n_q, n_kv, seq, hd, stream);

    cublasSetStream(ctx->cublas, stream);
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cbs = cublasGemmStridedBatchedEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        seq, seq, hd,
        &alpha,
        ctx->d_k_rep, CUDA_R_16BF, hd, (long long)seq * hd,
        ctx->d_q_HShd,CUDA_R_16BF, hd, (long long)seq * hd,
        &beta,
        ctx->d_scores,CUDA_R_16BF, seq, (long long)seq * seq,
        n_q,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cbs != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla_lm_cuda): QK^T gemm failed @L%d (%d)\n", L, cbs); return -1; }

    const float scl = 1.0f / std::sqrt((float)hd);
    bitvla_softmax_scaled_bf16(ctx->d_scores, scl, n_q * seq, seq, stream);

    cbs = cublasGemmStridedBatchedEx(
        ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        hd, seq, seq,
        &alpha,
        ctx->d_v_rep, CUDA_R_16BF, hd,  (long long)seq * hd,
        ctx->d_scores,CUDA_R_16BF, seq, (long long)seq * seq,
        &beta,
        ctx->d_attn_out, CUDA_R_16BF, hd, (long long)seq * hd,
        n_q,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cbs != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vla(bitvla_lm_cuda): attn@V gemm failed @L%d (%d)\n", L, cbs); return -1; }

    bitvla_transpose_NshHd_to_sNhd_bf16(ctx->d_attn_out, ctx->d_attn_merged, n_q, seq, hd, stream);

    l0_dump("L0_03_attn_merged_pre_subnorm", ctx->d_attn_merged, (size_t)seq * hq);

    bitvla_rmsnorm_bf16(ctx->d_attn_merged, lr.attn_sub_norm_w, ctx->d_h_norm, ctx->rms_eps, seq, hq, stream);
    l0_dump("L0_04_attn_sub_norm", ctx->d_h_norm, (size_t)seq * hq);

    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, hq, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.o_packed, ctx->d_o_out,
                          ctx->d_act_s, lr.o_ws, seq, hidden, hq, stream);

    l0_dump("L0_05_o_proj_out", ctx->d_o_out, (size_t)seq * hidden);

    bitvla_add_bf16(ctx->d_h, ctx->d_o_out, ctx->d_h, seq * hidden, stream);
    l0_dump("L0_06_after_attn_residual", ctx->d_h, (size_t)seq * hidden);

    bitvla_rmsnorm_bf16(ctx->d_h, lr.ffn_norm_w, ctx->d_h_norm, ctx->rms_eps, seq, hidden, stream);
    l0_dump("L0_07_ffn_norm", ctx->d_h_norm, (size_t)seq * hidden);
    bitvla_act_quant_cuda(ctx->d_h_norm, ctx->d_act_int8_h, ctx->d_act_s, seq, hidden, stream);

    bitlinear_int8xint2_m(ctx->d_act_int8_h, lr.gate_up_packed, ctx->d_gate_up,
                          ctx->d_act_s, lr.gate_up_ws, seq, 2 * ffn, hidden, stream);

    gate_up_fused_sqrelu_mul_bf16(ctx->d_gate_up, ctx->d_gate_sq_up, seq, ffn, stream);

    bitvla_rmsnorm_bf16(ctx->d_gate_sq_up, lr.ffn_sub_norm_w, ctx->d_gate_sq_up,
                        ctx->rms_eps, seq, ffn, stream);

    l0_dump("L0_08_ffn_sub_norm", ctx->d_gate_sq_up, (size_t)seq * ffn);

    bitvla_act_quant_cuda(ctx->d_gate_sq_up, ctx->d_act_int8_ffn, ctx->d_act_s, seq, ffn, stream);
    bitlinear_int8xint2_m(ctx->d_act_int8_ffn, lr.down_packed, ctx->d_down_out,
                          ctx->d_act_s, lr.down_ws, seq, hidden, ffn, stream);
    l0_dump("L0_09_down_out", ctx->d_down_out, (size_t)seq * hidden);

    bitvla_add_bf16(ctx->d_h, ctx->d_down_out, ctx->d_h, seq * hidden, stream);
    l0_dump("L0_10_final", ctx->d_h, (size_t)seq * hidden);

    return 0;
}

__global__ void gate_up_fused_sqrelu_mul_bf16_kernel(const __nv_bfloat16* __restrict__ gu,
                                                      __nv_bfloat16* __restrict__ out,
                                                      int seq, int ffn) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = seq * ffn;
    if (idx >= total) return;
    const int s = idx / ffn;
    const int k = idx % ffn;
    const size_t row_base = (size_t)s * 2 * ffn;
    float g = __bfloat162float(gu[row_base + k]);
    float u = __bfloat162float(gu[row_base + ffn + k]);
    if (g < 0.0f) g = 0.0f;
    out[(size_t)idx] = __float2bfloat16(g * g * u);
}
extern "C" void gate_up_fused_sqrelu_mul_bf16(const __nv_bfloat16* gu, __nv_bfloat16* out,
                                               int seq, int ffn, cudaStream_t stream) {
    const int total = seq * ffn;
    const int B = 256;
    gate_up_fused_sqrelu_mul_bf16_kernel<<<dim3((total + B - 1) / B, 1, 1),
                                            dim3(B, 1, 1), 0, stream>>>(gu, out, seq, ffn);
}

template <int BLOCK>
__global__ void layernorm_bias_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ w,
                                            const __nv_bfloat16* __restrict__ b,
                                            __nv_bfloat16* __restrict__ out,
                                            float eps, int K)
{
    const int m   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const __nv_bfloat16* row = x + (size_t)m * K;
    __nv_bfloat16*       o   = out + (size_t)m * K;

    float sum = 0.0f;
    for (int k = tid; k < K; k += BLOCK) sum += __bfloat162float(row[k]);
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
        float v = __bfloat162float(row[k]) - mean;
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
        float v = (__bfloat162float(row[k]) - mean) * inv_std;
        float wv = __bfloat162float(w[k]);
        float bv = __bfloat162float(b[k]);
        o[k] = __float2bfloat16(v * wv + bv);
    }
}

__global__ void gelu_tanh_bf16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int N) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) return;
    float x = __bfloat162float(in[i]);

    const float kAlpha = 0.7978845608028654f;
    const float kBeta  = 0.044715f;
    float u = kAlpha * (x + kBeta * x * x * x);
    float t = tanhf(u);
    out[i] = __float2bfloat16(0.5f * x * (1.0f + t));
}

__global__ void add_bias_bf16_kernel(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                                      __nv_bfloat16* out, int M, int K) {
    const int m = (int)blockIdx.x;
    const int k = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (k >= K) return;
    const size_t i = (size_t)m * K + k;
    out[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(bias[k]));
}

__global__ void zero_tail_bf16_kernel(__nv_bfloat16* x, int total_cols, int start_col) {
    const int m = (int)blockIdx.x;
    const int k = (int)(start_col + blockIdx.y * blockDim.x + threadIdx.x);
    if (k >= total_cols) return;
    x[(size_t)m * total_cols + k] = __float2bfloat16(0.0f);
}

extern "C" void bitvla_layernorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                       const __nv_bfloat16* b, __nv_bfloat16* out,
                                       float eps, int M, int K, cudaStream_t stream) {
    constexpr int B = 256;
    layernorm_bias_bf16_kernel<B><<<dim3(M, 1, 1), dim3(B, 1, 1), 0, stream>>>(x, w, b, out, eps, K);
}
extern "C" void bitvla_gelu_tanh_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                                       int N, cudaStream_t stream) {
    constexpr int B = 256;
    gelu_tanh_bf16_kernel<<<dim3((N + B - 1) / B, 1, 1), dim3(B, 1, 1), 0, stream>>>(x, out, N);
}
extern "C" void bitvla_add_bias_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                                      __nv_bfloat16* out, int M, int K, cudaStream_t stream) {
    constexpr int B = 256;
    const int n_kb = (K + B - 1) / B;
    add_bias_bf16_kernel<<<dim3(M, n_kb, 1), dim3(B, 1, 1), 0, stream>>>(x, bias, out, M, K);
}
extern "C" void bitvla_zero_tail_bf16(__nv_bfloat16* x, int M, int total_cols,
                                       int start_col, cudaStream_t stream) {
    if (start_col >= total_cols) return;
    constexpr int B = 128;
    const int len = total_cols - start_col;
    const int n_kb = (len + B - 1) / B;
    zero_tail_bf16_kernel<<<dim3(M, n_kb, 1), dim3(B, 1, 1), 0, stream>>>(x, total_cols, start_col);
}

extern "C" int bitvla_lm_cuda_forward(bitvla_lm_cuda_ctx* ctx,
                                       const __nv_bfloat16* d_in,
                                       __nv_bfloat16* d_out,
                                       int seq, cudaStream_t stream)
{
    if (seq > ctx->max_seq) {
        std::fprintf(stderr, "vla(bitvla_lm_cuda): seq=%d > max_seq=%d\n", seq, ctx->max_seq);
        return -1;
    }

    const char* dump_dir = std::getenv("VLA_BITVLA_DUMP_LM_LAYERS");
    std::vector<__nv_bfloat16> h_dump;
    std::vector<float>         h_dump_f32;
    auto dump_to_file = [&](const char* name, const __nv_bfloat16* d_ptr) {
        if (!dump_dir) return;
        const size_t n = (size_t) seq * ctx->hidden;
        h_dump.resize(n); h_dump_f32.resize(n);
        cudaMemcpy(h_dump.data(), d_ptr, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n; ++i) {
            uint16_t u = *reinterpret_cast<uint16_t*>(&h_dump[i]);
            uint32_t b = ((uint32_t) u) << 16;
            float f; std::memcpy(&f, &b, 4);
            h_dump_f32[i] = f;
        }
        std::string path = std::string(dump_dir) + "/" + name + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f) { std::fwrite(h_dump_f32.data(), sizeof(float), n, f); std::fclose(f); }
    };

    CUDA_OK(cudaMemcpyAsync(ctx->d_h, d_in, (size_t)seq * ctx->hidden * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToDevice, stream));
    if (dump_dir) {
        cudaStreamSynchronize(stream);
        dump_to_file("lm_layer_input", ctx->d_h);
    }
    for (int L = 0; L < ctx->n_layers; ++L) {
        int rc = run_layer(ctx, L, seq, stream);
        if (rc != 0) return rc;
        if (dump_dir) {
            cudaStreamSynchronize(stream);
            char name[64]; std::snprintf(name, sizeof(name), "lm_layer_%d", L);
            dump_to_file(name, ctx->d_h);
        }
    }

    bitvla_rmsnorm_bf16(ctx->d_h, ctx->output_norm_w, d_out, ctx->rms_eps, seq, ctx->hidden, stream);
    if (dump_dir) {
        cudaStreamSynchronize(stream);
        dump_to_file("lm_final", d_out);
    }
    return 0;
}
