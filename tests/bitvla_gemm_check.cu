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
 * @file bitvla_gemm_check.cu
 * @brief A/B correctness + throughput check for the BitVLA ternary GEMM.
 *
 * The wide kernel (@c ladder_int8xint2_kernel_m_wide) changes only how work is
 * tiled, not the arithmetic: identical int8 x int2 products accumulated in
 * int32 and scaled by the same per-row and per-column-group factors. So its
 * output must be *bit-identical* to the one-tile-per-CTA kernel, not merely
 * close -- an epsilon here would hide a real indexing bug behind bf16 rounding.
 *
 * Runs every shape the production dispatch supports, at the sequence lengths
 * the LM and ViT actually use, plus a ragged M to exercise the row tail.
 */

#include "../src/kernels/bitvla/bitnet_kernels.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_OK(expr)                                                          \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),  \
                   __FILE__, __LINE__);                                        \
      return 2;                                                                \
    }                                                                          \
  } while (0)

namespace {

struct Shape {
  const char* name;
  int N, K, ws_num;
};

// Every shape in bitlinear_int8xint2_m's dispatch.
const Shape kShapes[] = {
    {"lm.q/o     ", 2560, 2560, 1},
    {"lm.k/v     ", 640, 2560, 1},
    {"lm.gate_up ", 13824, 2560, 2},
    {"lm.down    ", 2560, 6912, 1},
    {"vit.qkvo   ", 1152, 1152, 1},
    {"vit.fc1    ", 4304, 1152, 1},
    {"vit.fc2    ", 1152, 4352, 1},
    {"head.qkv   ", 3840, 2560, 3},
};

float bf16_to_f32(__nv_bfloat16 h) {
  uint16_t u;
  std::memcpy(&u, &h, 2);
  uint32_t b = ((uint32_t)u) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

int run_shape(const Shape& sh, int M, int nt, bool& ok) {
  const size_t a_sz = (size_t)M * sh.K;
  const size_t b_sz = (size_t)sh.N * sh.K / 4;  // int2: 4 values per byte
  const size_t o_sz = (size_t)M * sh.N;

  std::mt19937 rng(1234u + (unsigned)sh.N + (unsigned)M);
  std::vector<int8_t> h_a(a_sz);
  std::vector<int8_t> h_b(b_sz);
  std::vector<float> h_s(M), h_ws(sh.ws_num);
  for (auto& v : h_a) v = (int8_t)((int)(rng() % 255) - 127);
  for (auto& v : h_b) v = (int8_t)(rng() % 256);
  for (auto& v : h_s) v = 40.0f + (float)(rng() % 100);
  for (auto& v : h_ws) v = 0.01f + 0.001f * (float)(rng() % 50);

  int8_t *d_a, *d_b;
  __nv_bfloat16 *d_ref, *d_new;
  float *d_s, *d_ws;
  CUDA_OK(cudaMalloc(&d_a, a_sz));
  CUDA_OK(cudaMalloc(&d_b, b_sz));
  CUDA_OK(cudaMalloc(&d_ref, o_sz * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMalloc(&d_new, o_sz * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMalloc(&d_s, M * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_ws, sh.ws_num * sizeof(float)));
  CUDA_OK(cudaMemcpy(d_a, h_a.data(), a_sz, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_b, h_b.data(), b_sz, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_s, h_s.data(), M * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_ws, h_ws.data(), sh.ws_num * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(d_ref, 0, o_sz * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMemset(d_new, 0, o_sz * sizeof(__nv_bfloat16)));

  // Dispatch by shape, mirroring bitlinear_int8xint2_m. Templated on N/K, so
  // this has to be a chain rather than a loop.
#define BOTH(NN, KK, WS)                                                        \
  if (sh.N == (NN) && sh.K == (KK)) {                                           \
    launch_ladder_int8xint2_m<NN, KK, WS, 128>(d_a, d_b, d_ref, d_s, d_ws, M, 0);\
    if (nt == 1)                                                                \
      launch_ladder_int8xint2_m_wide<NN, KK, WS, 128, 1>(d_a, d_b, d_new, d_s, d_ws, M, 0); \
    else if (nt == 2)                                                           \
      launch_ladder_int8xint2_m_wide<NN, KK, WS, 128, 2>(d_a, d_b, d_new, d_s, d_ws, M, 0); \
    else                                                                        \
      launch_ladder_int8xint2_m_wide<NN, KK, WS, 128, 4>(d_a, d_b, d_new, d_s, d_ws, M, 0); \
  }
  BOTH(2560, 2560, 1)
  BOTH(640, 2560, 1)
  BOTH(13824, 2560, 2)
  BOTH(2560, 6912, 1)
  BOTH(1152, 1152, 1)
  BOTH(4304, 1152, 1)
  BOTH(1152, 4352, 1)
  BOTH(3840, 2560, 3)
#undef BOTH
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaGetLastError());

  std::vector<__nv_bfloat16> h_ref(o_sz), h_new(o_sz);
  CUDA_OK(cudaMemcpy(h_ref.data(), d_ref, o_sz * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(h_new.data(), d_new, o_sz * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  size_t mismatches = 0;
  size_t first = 0;
  for (size_t i = 0; i < o_sz; ++i) {
    uint16_t a, b;
    std::memcpy(&a, &h_ref[i], 2);
    std::memcpy(&b, &h_new[i], 2);
    if (a != b) {
      if (mismatches == 0) first = i;
      ++mismatches;
    }
  }

  // Timing: median of 20 after 5 warmup, so a stray clock excursion does not
  // decide the reported speedup.
  auto bench = [&](bool wide) -> float {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<float> ms;
    for (int it = 0; it < 25; ++it) {
      cudaEventRecord(e0);
#define TIME_ONE(NN, KK, WS)                                                    \
  if (sh.N == (NN) && sh.K == (KK)) {                                           \
    if (!wide)                                                                  \
      launch_ladder_int8xint2_m<NN, KK, WS, 128>(d_a, d_b, d_ref, d_s, d_ws, M, 0);\
    else if (nt == 1)                                                           \
      launch_ladder_int8xint2_m_wide<NN, KK, WS, 128, 1>(d_a, d_b, d_new, d_s, d_ws, M, 0); \
    else if (nt == 2)                                                           \
      launch_ladder_int8xint2_m_wide<NN, KK, WS, 128, 2>(d_a, d_b, d_new, d_s, d_ws, M, 0); \
    else                                                                        \
      launch_ladder_int8xint2_m_wide<NN, KK, WS, 128, 4>(d_a, d_b, d_new, d_s, d_ws, M, 0); \
  }
      TIME_ONE(2560, 2560, 1)
      TIME_ONE(640, 2560, 1)
      TIME_ONE(13824, 2560, 2)
      TIME_ONE(2560, 6912, 1)
      TIME_ONE(1152, 1152, 1)
      TIME_ONE(4304, 1152, 1)
      TIME_ONE(1152, 4352, 1)
      TIME_ONE(3840, 2560, 3)
#undef TIME_ONE
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float t = 0;
      cudaEventElapsedTime(&t, e0, e1);
      if (it >= 5) ms.push_back(t);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
  };

  const float t_ref = bench(false);
  const float t_new = bench(true);
  const double macs = (double)M * sh.N * sh.K;

  std::printf("%s M=%-4d nt=%d  ref %7.3f ms (%5.1f TOPS)   wide %7.3f ms (%5.1f TOPS)"
              "   %4.2fx   %s\n",
              sh.name, M, nt, t_ref, macs / (t_ref * 1e-3) / 1e12, t_new,
              macs / (t_new * 1e-3) / 1e12, t_ref / t_new,
              mismatches == 0 ? "bit-identical"
                              : "MISMATCH");
  if (mismatches != 0) {
    std::printf("    %zu/%zu differ, first at %zu: ref %g new %g\n", mismatches,
                o_sz, first, bf16_to_f32(h_ref[first]), bf16_to_f32(h_new[first]));
    ok = false;
  }

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_ref); cudaFree(d_new);
  cudaFree(d_s); cudaFree(d_ws);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // Default run is the production tiling only; pass "sweep" to compare
  // N_TILES = 1/2/4 per shape, which is how the per-shape choice was made.
  const bool sweep = (argc > 1 && std::strcmp(argv[1], "sweep") == 0);
  bool ok = true;
  // 600: the LM sequence length for a LIBERO prompt (512 image markers + 1
  // proprio + prompt + 56 action slots + stop). 256: the ViT, which runs one
  // 256-patch view per call rather than batching both. 517: ragged, to
  // exercise the M tail (517 = 4*128 + 5).
  const int kMs[] = {600, 256, 517};
  const int kSweepTiles[] = {1, 2, 4};
  for (const auto& sh : kShapes) {
    for (int m : kMs) {
      if (sweep) {
        for (int nt : kSweepTiles) {
          int rc = run_shape(sh, m, nt, ok);
          if (rc) return rc;
        }
        std::printf("\n");
      } else {
        int rc = run_shape(sh, m, bitvla_n_tiles_for(sh.N, sh.K), ok);
        if (rc) return rc;
      }
    }
  }
  std::printf("\n%s\n", ok ? "PASS: wide kernel is bit-identical on every shape"
                           : "FAIL: outputs differ");
  return ok ? 0 : 1;
}
