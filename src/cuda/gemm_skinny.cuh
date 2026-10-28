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

// Skinny BF16 GEMM (few tokens, S <= 8*NT) that reproduces cuBLAS's
// ampere_s16816 BF16 kernels bit for bit: Y[s][n] = sum_k W[n][k] X[s][k] as a
// chain of mma.m16n8k16 in K order per output; with SL slices, slice q of
// every KT-wide k-tile accumulates on its own (KT/SL wide) and the slices are
// summed in order at the end (the sliced1xSL kernels). Which structure a
// shape's cublasGemmEx call uses is found by comparison (gemm_exact's plan);
// the gain is in the tiling: more CTAs over N and a deeper cp.async pipeline.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace skinny {

constexpr int PAD = 8;

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool valid) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], const void * p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldm_x2(uint32_t (&r)[2], const void * p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// CTA: RG row groups of 16 output features x SL slice warps; STG pipeline stages.
template <int NT, int RG, int SL, int KT, int STG, int BK>
__global__ void __launch_bounds__(RG*SL*32)
k_gemm_skinny(const __nv_bfloat16 * __restrict__ W, const __nv_bfloat16 * __restrict__ X, float * __restrict__ Y,
              int N, int S, int K) {
    // blockIdx.y: the view (one GEMM per view, as gemm_views)
    X += (size_t) blockIdx.y*S*K;
    Y += (size_t) blockIdx.y*S*N;
    static_assert(KT % SL == 0 && (KT/SL) % 16 == 0 && BK % KT == 0, "slice width: a multiple of 16");
    constexpr int SW = KT/SL, RW = RG*16, RX = NT*8, NTH = RG*SL*32, LDS = BK+PAD;
    extern __shared__ __align__(16) __nv_bfloat16 sm[];
    __nv_bfloat16 * sW = sm, * sX = sm+STG*RW*LDS;   // [STG][RW][LDS], [STG][RX][LDS]
    const int tid = threadIdx.x, warp = tid/32, lane = tid%32;
    const int rg = warp%RG, sl = warp/RG;
    const int n0 = blockIdx.x*RW;
    const int nk = K/BK;

    auto load = [&](int kb, int stg) {
        const int k0 = kb*BK;
        for (int c = tid; c < RW*(BK/8); c += NTH) {
            const int r = c/(BK/8), kc = (c%(BK/8))*8;
            cp_async16(sW+((size_t) stg*RW+r)*LDS+kc, W+(size_t) (n0+r)*K+k0+kc, true);
        }
        for (int c = tid; c < RX*(BK/8); c += NTH) {
            const int r = c/(BK/8), kc = (c%(BK/8))*8;
            const bool v = r < S;
            cp_async16(sX+((size_t) stg*RX+r)*LDS+kc, X+(size_t) (v ? r : 0)*K+k0+kc, v);
        }
    };

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

#pragma unroll
    for (int s = 0; s < STG-1; ++s) {
        if (s < nk) load(s, s);
        cp_commit();
    }
    for (int kb = 0; kb < nk; ++kb) {
        cp_wait<STG-2>();
        __syncthreads();
        {   // refill the stage consumed in the previous iteration
            const int nb = kb+STG-1;
            if (nb < nk) load(nb, nb%STG);
            cp_commit();
        }
        const int stg = kb%STG;
        const __nv_bfloat16 * w = sW+((size_t) stg*RW+rg*16)*LDS;
        const __nv_bfloat16 * x = sX+(size_t) stg*RX*LDS;
#pragma unroll
        for (int t = 0; t < BK/KT; ++t) {
#pragma unroll
            for (int kk = t*KT+sl*SW; kk < t*KT+(sl+1)*SW; kk += 16) {
                uint32_t a[4];
                ldm_x4(a, w+(size_t) (lane%16)*LDS+kk+(lane/16)*8);
#pragma unroll
                for (int j = 0; j < NT; j += 2) {
                    if (j+1 < NT) {
                        uint32_t b[4];
                        // m0/m1: tile j at k, k+8; m2/m3: tile j+1
                        ldm_x4(b, x+(size_t) (j*8+(lane%8)+(lane/16)*8)*LDS+kk+((lane/8)%2)*8);
                        mma16816(acc[j], a, b[0], b[1]);
                        mma16816(acc[j+1], a, b[2], b[3]);
                    } else {
                        uint32_t b[2];
                        ldm_x2(b, x+(size_t) (j*8+(lane%8))*LDS+kk+((lane/8)%2)*8);
                        mma16816(acc[j], a, b[0], b[1]);
                    }
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
    // Slices 1.. hand their partials to slice 0 through shared memory.
    float * red = (float *) sm;   // [SL-1][RG][NT*4][32]
    if (SL > 1) {
        if (sl > 0)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) red[(((sl-1)*RG+rg)*NT*4+j*4+i)*32+lane] = acc[j][i];
        __syncthreads();
        if (sl > 0) return;
#pragma unroll
        for (int q = 1; q < SL; ++q)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[j][i] = acc[j][i] + red[(((q-1)*RG+rg)*NT*4+j*4+i)*32+lane];
    }
    const int g = lane/4, t4 = lane%4, n = n0+rg*16+g;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int s = j*8+t4*2;
        if (s < S)   { Y[(size_t) s*N+n] = acc[j][0];     Y[(size_t) s*N+n+8] = acc[j][2]; }
        if (s+1 < S) { Y[(size_t) (s+1)*N+n] = acc[j][1]; Y[(size_t) (s+1)*N+n+8] = acc[j][3]; }
    }
}

template <int NT, int RG, int SL, int KT, int STG, int BK>
inline size_t smem_bytes() {
    const size_t pipe = (size_t) STG*(RG*16+NT*8)*(BK+PAD)*2, red = (size_t) (SL-1)*RG*NT*4*32*4;
    return pipe > red ? pipe : red;
}

template <int NT, int RG, int SL, int KT, int STG, int BK>
inline bool launch(const __nv_bfloat16 * W, const __nv_bfloat16 * X, float * Y, int N, int S, int K, int nv, cudaStream_t st) {
    if (S > NT*8 || N % (RG*16) || K % BK) return false;
    const size_t shm = smem_bytes<NT, RG, SL, KT, STG, BK>();
    if (shm > 48*1024 &&
        cudaFuncSetAttribute(k_gemm_skinny<NT, RG, SL, KT, STG, BK>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm) != cudaSuccess)
        return false;
    k_gemm_skinny<NT, RG, SL, KT, STG, BK><<<dim3(N/(RG*16), nv), RG*SL*32, shm, st>>>(W, X, Y, N, S, K);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace skinny

namespace skinny {

using LaunchFn = bool (*)(const __nv_bfloat16 *, const __nv_bfloat16 *, float *, int, int, int, int, cudaStream_t);

struct Cand {
    LaunchFn fn;
    const char * name;
};

// Candidates for S tokens: n8 tiles NT (the smallest instantiated >= S/8), K
// slicing 1 or 2 of a 64-wide k-tile, and a few CTA shapes / pipelines.
template <int NT>
inline void cands_nt(std::vector<Cand> & v) {
    v.push_back({launch<NT, 8, 1, 64, 3, 64>,  "sl1 rg8 bk64 st3"});
    v.push_back({launch<NT, 4, 1, 64, 3, 128>, "sl1 rg4 bk128 st3"});
    v.push_back({launch<NT, 4, 1, 64, 4, 128>, "sl1 rg4 bk128 st4"});
    v.push_back({launch<NT, 2, 1, 64, 3, 256>, "sl1 rg2 bk256 st3"});
    v.push_back({launch<NT, 8, 2, 64, 3, 64>,  "sl2 rg8 bk64 st3"});
    v.push_back({launch<NT, 4, 2, 64, 3, 128>, "sl2 rg4 bk128 st3"});
    v.push_back({launch<NT, 4, 2, 64, 4, 128>, "sl2 rg4 bk128 st4"});
    v.push_back({launch<NT, 2, 2, 64, 3, 256>, "sl2 rg2 bk256 st3"});
}

inline std::vector<Cand> cands(int S) {
    std::vector<Cand> v;
    if      (S <= 16) cands_nt<2>(v);
    else if (S <= 32) cands_nt<4>(v);
    else if (S <= 56) cands_nt<7>(v);
    else if (S <= 64) cands_nt<8>(v);
    return v;
}

}  // namespace skinny
