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

// Micro-benchmark of the FoldQuant CUDA kernels at GR00T N1.6/N1.7 shapes:
// prologue + INT8 GEMM time, effective weight-streaming bandwidth, and a
// cuBLAS BF16 GEMM of the same shape as the number to beat. Built with the
// CUDA tree, not registered with ctest (needs a GPU, read by hand).
//
//   ./build-cuda/tests/foldquant_gemm_check [reps]

#include "kernels/foldquant/fq_kernels.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

struct Shape { const char * name; int M, N, K; bool gamma; };

// --stress: run the FFN shapes many times and hash the outputs; any change
// between iterations is a race in the kernels (the graph is otherwise fixed).
static uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const uint8_t *) p)[i]; h *= 1099511628211ull; }
    return h;
}

static int stress(cudaStream_t st, int iters) {
    const Shape shapes[] = {
        { "ffn gate M=87", 87, 6144, 2048, true },
        { "ffn down M=87", 87, 2048, 6144, false },
        { "qkv M=87",      87, 2048, 2048, true },
    };
    for (const Shape & s : shapes) {
        const int64_t M = s.M, N = s.N, K = s.K, rb = K + 16;
        std::vector<float> hx((size_t) M * K), hg(K), hws(N);
        std::vector<int8_t> hw((size_t) N * K);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 37) % 23) - 11.0f;
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = (int8_t) ((i * 53) % 255 - 127);
        for (int64_t i = 0; i < K; ++i) hg[i] = 1.0f + 0.001f * (float) (i % 7);
        for (int64_t i = 0; i < N; ++i) hws[i] = 0.01f;
        float *x, *g, *ws, *y; int8_t *w, *blob;
        CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&g, K * 4)); CK(cudaMalloc(&ws, N * 4));
        CK(cudaMalloc(&y, (size_t) M * N * 4)); CK(cudaMalloc(&w, hw.size())); CK(cudaMalloc(&blob, (size_t) M * rb));
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g, hg.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ws, hws.data(), N * 4, cudaMemcpyHostToDevice));
        vla::fq::ActArgs a{}; a.x = x; a.gamma = s.gamma ? g : nullptr; a.blob = blob; a.row_bytes = rb;
        a.M = M; a.K = K; a.abits = 8; a.rot_block = 64; a.clip = 1.0f; a.eps = 1e-6f; a.inv_sqrt_bs = 0.125f;
        vla::fq::GemmArgs gm{}; gm.w = w; gm.blob = blob; gm.wscale = ws; gm.y = y; gm.M = M; gm.N = N; gm.K = K;
        gm.row_bytes = rb; gm.wbits = 8; gm.abits = 8;
        std::vector<uint8_t> hb((size_t) M * rb); std::vector<float> hy((size_t) M * N);
        uint64_t hb0 = 0, hy0 = 0; int bad_b = 0, bad_y = 0;
        for (int it = 0; it < iters; ++it) {
            CK(cudaMemsetAsync(blob, 0xAB, (size_t) M * rb, st));
            CK(cudaMemsetAsync(y, 0xCD, (size_t) M * N * 4, st));
            if (vla::fq::launch_act(a, st) != cudaSuccess) { std::printf("act launch failed\n"); return 1; }
            if (vla::fq::launch_gemm(gm, st) != cudaSuccess) { std::printf("gemm launch failed\n"); return 1; }
            CK(cudaMemcpyAsync(hb.data(), blob, hb.size(), cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            // hash codes + scale of each row (padding bytes are never written)
            uint64_t hbb = 14695981039346656037ull;
            for (int64_t m = 0; m < M; ++m) hbb ^= fnv(hb.data() + m * rb, K + 4), hbb *= 1099511628211ull;
            const uint64_t hyy = fnv(hy.data(), hy.size() * 4);
            if (it == 0) { hb0 = hbb; hy0 = hyy; }
            else { bad_b += hbb != hb0; bad_y += hyy != hy0; }
        }
        std::printf("stress %-16s blob changed %d/%d  y changed %d/%d\n", s.name, bad_b, iters - 1, bad_y, iters - 1);
        cudaFree(x); cudaFree(g); cudaFree(ws); cudaFree(y); cudaFree(w); cudaFree(blob);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--stress") {
        cudaStream_t st; CK(cudaStreamCreate(&st));
        return stress(st, argc > 2 ? std::atoi(argv[2]) : 200);
    }
    // --cold: rotate every kernel over enough weight copies that no site stays
    // in the 4 MB L2 between reps - what a model run sees, where each site's
    // weights are streamed from DRAM once per request. Without it the loop
    // re-runs one L2-hot site and overstates the achievable bandwidth.
    bool cold = false;
    int  reps = 50;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cold") cold = true;
        else reps = std::atoi(argv[i]);
    }
    const Shape shapes[] = {
        { "dit qkv  M=41",     41,   4608, 1536, false },
        { "dit o    M=41",     41,   1536, 1536, false },
        { "dit ff0  M=41",     41,   6144, 1536, false },
        { "dit ff2  M=41",     41,   1536, 6144, false },
        { "llm qkv  M=160",    160,  4096, 2048, true  },
        { "llm o    M=160",    160,  2048, 2048, false },
        { "llm gate+up M=160", 160,  12288, 2048, true },
        { "llm down M=160",    160,  2048, 6144, false },
        { "llm qkv  M=1024",   1024, 4096, 2048, true  },
    };
    cublasHandle_t cublas; cublasCreate(&cublas);
    cudaStream_t st; CK(cudaStreamCreate(&st)); cublasSetStream(cublas, st);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    {   // burn-in so the governor raises the clock before anything is timed
        float * junk; CK(cudaMalloc(&junk, 64 << 20));
        for (int i = 0; i < 200; ++i) CK(cudaMemsetAsync(junk, i, 64 << 20, st));
        CK(cudaStreamSynchronize(st)); cudaFree(junk);
    }
    std::printf("%-20s %8s %8s %8s %9s %9s %9s\n", "shape", "prolog us", "gemm us", "total us", "W GB/s", "bf16 us", "speedup");
    for (const Shape & s : shapes) {
        const int64_t M = s.M, N = s.N, K = s.K;
        const int64_t rb = K + 16;
        std::vector<float> hx((size_t) M * K), hg(K), hws(N);
        std::vector<int8_t> hw((size_t) N * K);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 37) % 23) - 11.0f;
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = (int8_t) ((i * 53) % 255 - 127);
        for (int64_t i = 0; i < K; ++i) hg[i] = 1.0f;
        for (int64_t i = 0; i < N; ++i) hws[i] = 0.01f;

        const int copies = cold ? (int) std::max<int64_t>(2, (48ll << 20) / ((int64_t) N * K) + 1) : 1;
        float *x, *g, *ws, *y; int8_t *blob;
        std::vector<int8_t *> w(copies);
        CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&g, K * 4)); CK(cudaMalloc(&ws, N * 4));
        CK(cudaMalloc(&y, (size_t) M * N * 4)); CK(cudaMalloc(&blob, (size_t) M * rb));
        for (int c = 0; c < copies; ++c) {
            CK(cudaMalloc(&w[c], hw.size()));
            CK(cudaMemcpy(w[c], hw.data(), hw.size(), cudaMemcpyHostToDevice));
        }
        int rot = 0;
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g, hg.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ws, hws.data(), N * 4, cudaMemcpyHostToDevice));

        vla::fq::ActArgs a{}; a.x = x; a.ascale = nullptr; a.gamma = s.gamma ? g : nullptr; a.blob = blob; a.row_bytes = rb;
        a.M = M; a.K = K; a.abits = 8; a.rot_block = 64; a.fold_before = false; a.clip = 1.0f; a.eps = 1e-6f; a.inv_sqrt_bs = 0.125f;
        vla::fq::GemmArgs gm{}; gm.w = w[0]; gm.blob = blob; gm.wscale = ws; gm.bias = nullptr; gm.y = y; gm.M = M; gm.N = N; gm.K = K;
        gm.row_bytes = rb; gm.wbits = 8; gm.abits = 8;

        // DVFS: the governor ramps the clock with sustained load, so the three
        // kernels are measured interleaved in several rounds and the best round
        // of each is kept - the ratios are what matters, at whatever clock.
        auto time = [&](auto fn) -> float {
            for (int i = 0; i < 5; ++i) fn();
            CK(cudaEventRecord(e0, st));
            for (int i = 0; i < reps; ++i) fn();
            CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
            float ms; cudaEventElapsedTime(&ms, e0, e1); return ms * 1000.0f / reps;
        };
        std::vector<__nv_bfloat16 *> wb(copies); __nv_bfloat16 * xb; float * yb;
        for (int c = 0; c < copies; ++c) CK(cudaMalloc(&wb[c], (size_t) N * K * 2));
        CK(cudaMalloc(&xb, (size_t) M * K * 2)); CK(cudaMalloc(&yb, (size_t) M * N * 4));
        const float one = 1.0f, zero = 0.0f;
        auto bf16 = [&] {
            // cuBLAS BF16 GEMM with F32 accumulate, the shape vla.cpp's bf16 path runs (weights [N,K], x [M,K]).
            cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) M, (int) K, &one, wb[rot++ % copies], CUDA_R_16BF, (int) K,
                         xb, CUDA_R_16BF, (int) K, &zero, yb, CUDA_R_32F, (int) N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        };
        float t_pro = 1e30f, t_gemm = 1e30f, t_bf16 = 1e30f;
        for (int round = 0; round < 4; ++round) {
            t_bf16 = std::min(t_bf16, time(bf16));
            t_pro  = std::min(t_pro,  time([&] { vla::fq::launch_act(a, st); }));
            t_gemm = std::min(t_gemm, time([&] { gm.w = w[rot++ % copies]; vla::fq::launch_gemm(gm, st); }));
        }
        if (cudaGetLastError() != cudaSuccess) { std::printf("launch error\n"); return 1; }
        const double gbps = (double) N * K / ((t_gemm) * 1e-6) / 1e9;
        std::printf("%-20s %8.1f %8.1f %8.1f %9.1f %9.1f %8.2fx\n", s.name, t_pro, t_gemm, t_pro + t_gemm, gbps, t_bf16,
                    t_bf16 / (t_pro + t_gemm));
        cudaFree(x); cudaFree(g); cudaFree(ws); cudaFree(y); cudaFree(blob); cudaFree(xb); cudaFree(yb);
        for (int c = 0; c < copies; ++c) { cudaFree(w[c]); cudaFree(wb[c]); }
    }
    return 0;
}
