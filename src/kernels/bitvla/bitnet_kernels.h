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
 * @file bitnet_kernels.h
 * @brief BitNet "ladder" int8 x int2 GEMM kernels (header-only).
 *
 * Lowest-level CUDA kernels for the BitVLA 1.58-bit ternary matmul. The
 * weight matrix @c B is packed two ternary values per byte (i2 packing);
 * the activation matrix @c A is FP32-quantised to int8 row-wise via
 * @ref act_quant_kernel. The decode lanes use the @c lop3.b32 PTX
 * intrinsic to expand i2 to i8, then run a single warp-level @c wmma
 * fragment multiply per tile.
 *
 * Two GEMM entry points are provided:
 *   * @ref ladder_int8xint2_kernel - single-row (M=1) decode kernel
 *     used for next-token / single-query inference.
 *   * @ref ladder_int8xint2_kernel_m + @ref launch_ladder_int8xint2_m
 *     - multi-row (M>1) variant for prefill and ViT batches.
 *
 * This header is meant to be included by the per-tier CUDA `.cu` files
 * (@c bitvla_lm_cuda.cu, @c bitvla_vit_cuda.cu); it is not part of the
 * public engine API.
 */

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

/**
 * @brief Decode a packed int2 word into N int8 values via @c lop3.b32.
 *
 * Unpacks 4 ternary {-1, 0, +1} values per 32-bit lane and subtracts the
 * 0x02 bias so the result is centred around zero.
 *
 * @tparam T1 Source word type (read once as a uint).
 * @tparam T2 Destination element type (treated as int8 lanes).
 * @param _i2s Pointer to one packed int2 word.
 * @param _i8s Pointer to the output buffer (length @p N int8).
 * @param N    Number of int8 values to produce; default 16.
 */
template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{

  uint *i8s = reinterpret_cast<uint *>(_i8s);

  uint const i2s = *_i2s;

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x03030303;
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000;

#pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
    i8s[i] = __vsubss4(i8s[i], 0x02020202);
  }
}

/**
 * @brief Single-row ternary GEMM kernel (M = 1).
 *
 * Computes one row of @c dtype_transform[0,:] = (A * B^T) / s[0] * ws,
 * with @c A in int8, @c B packed as int2 (decoded on the fly), accumulated
 * in int32 via @c __dp4a, then scaled back to bf16. The output bias
 * @c ws is applied per @c ws_num column groups.
 *
 * @tparam M             Always 1 in this overload; kept for symmetry with
 *                       the multi-row kernel.
 * @tparam N             Output column count.
 * @tparam K             Reduction dimension.
 * @tparam ws_num        Number of column groups sharing one bias entry.
 * @tparam K_block_size  Threads collaborating along K (warp width).
 * @tparam N_block_size  Threads collaborating along N (warp height).
 */
template <int M, int N, int K, int ws_num, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ dtype_transform, float* __restrict__ s, float* __restrict__ ws) {
  constexpr int K_per_loop = 16;
  constexpr int wmma_K = 32;
  constexpr int wmma_N = 16;
  int in_thread_C_local[1];
  signed char A_local[K_per_loop];
  int B_reshape_local[1];
  signed char B_decode_local[K_per_loop];
  int red_buf0[1];
  in_thread_C_local[0] = 0;
  #pragma unroll
  for (int k_0 = 0; k_0 < K/(K_per_loop * K_block_size); ++k_0) {
    *(int4*)(A_local + 0) = *(int4*)(A + ((k_0 * K_per_loop * K_block_size) + (((int)threadIdx.x) * K_per_loop)));
    B_reshape_local[0] = *(int*)(B +
      (((int)blockIdx.x) * N_block_size * K / 4) +
      (k_0 * K_block_size * K_per_loop * wmma_N / 4) +
      ((((int)threadIdx.x) >> 1) * wmma_K * wmma_N / 4) +
      ((((int)threadIdx.y) >> 3) * (wmma_K * wmma_N / 2) / 4) +
      ((((int)threadIdx.x) & 1) * (wmma_K * wmma_N / 4) / 4) +
      ((((int)threadIdx.y) & 7) * (wmma_K / 2) / 4)
      );
    decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
    #pragma unroll
    for (int k_2_0 = 0; k_2_0 < 4; ++k_2_0) {
      in_thread_C_local[0] = __dp4a(*(int *)&A_local[((k_2_0 * 4))],*(int *)&B_decode_local[((k_2_0 * 4))], in_thread_C_local[0]);
    }
  }
  red_buf0[0] = in_thread_C_local[0];
  #pragma unroll
  for (int offset = K_block_size/2; offset > 0; offset /= 2) {
    red_buf0[0] += __shfl_down_sync(__activemask(), red_buf0[0], offset, K_block_size);
  }
  int out_idx = ((((int)blockIdx.x) * N_block_size) + ((int)threadIdx.y));
  int ws_idx = out_idx / (N / ws_num);
  if (threadIdx.x == 0)
    dtype_transform[out_idx] = __float2bfloat16(((float)red_buf0[0]) / s[0] * ws[ws_idx]);
}

/**
 * @brief Multi-row ternary GEMM kernel (M > 1) using @c wmma fragments.
 *
 * Tiles M rows in chunks of @p M_ROWS per CTA. Each warp processes
 * @c TILES_PER_WARP = @p M_ROWS / 64 row-tiles, accumulating int32 via
 * @c wmma::mma_sync against B fragments decoded from the i2 pack into
 * shared memory.
 *
 * @tparam N      Output column count.
 * @tparam K      Reduction dimension.
 * @tparam ws_num Number of column groups sharing one scale entry.
 * @tparam M_ROWS Rows per CTA along M; must be a multiple of 64.
 * @param A,B    Activation (int8) and weight (int2-packed int8) buffers.
 * @param out    bf16 output (M x N).
 * @param s,ws   Per-row activation scale and per-column-group output scale.
 * @param M      Actual row count (may be < blockDim.y * M_ROWS - tail
 *               threads predicate against this).
 */
template <int N, int K, int ws_num, int M_ROWS>
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel_m(
    int8_t* __restrict__ A, int8_t* __restrict__ B,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ s, float* __restrict__ ws, int M)
{
  using namespace nvcuda;
  constexpr int N_block_size = 16;
  constexpr int K_per_loop = 16, wmma_K = 32, wmma_N = 16;
  constexpr int K_CHUNK = 128;
  constexpr int TILES_PER_WARP = M_ROWS / (4 * 16);

  const int tx  = (int)threadIdx.x;
  const int ty  = (int)threadIdx.y;
  const int tid = ty * 8 + tx;
  const int warp = tid >> 5;
  const int m_base = (int)blockIdx.y * M_ROWS;

  __shared__ signed char A_smem[M_ROWS][K_CHUNK];
  __shared__ signed char W_smem[16][K_CHUNK];
  __shared__ int         C_smem[M_ROWS][16];

  int B_reshape_local[1];
  signed char B_decode_local[K_per_loop];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag[TILES_PER_WARP];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[TILES_PER_WARP];
  #pragma unroll
  for (int t = 0; t < TILES_PER_WARP; ++t) wmma::fill_fragment(acc[t], 0);

  for (int k_0 = 0; k_0 < K / K_CHUNK; ++k_0) {
    #pragma unroll
    for (int r = 0; r < (M_ROWS * K_CHUNK / 16) / 128; ++r) {
      const int idx = tid + r * 128;
      const int m   = idx >> 3;
      const int kk  = (idx & 7) * 16;
      const int mrow = m_base + m;
      const int8_t* aptr = A + ((mrow < M ? mrow : 0) * K) + k_0 * K_CHUNK + kk;
      *(int4*)(&A_smem[m][kk]) = *(const int4*)aptr;
    }
    B_reshape_local[0] = *(int*)(B +
      (((int)blockIdx.x) * N_block_size * K / 4) +
      (k_0 * 8 * K_per_loop * wmma_N / 4) +
      ((tx >> 1) * wmma_K * wmma_N / 4) +
      ((ty >> 3) * (wmma_K * wmma_N / 2) / 4) +
      ((tx & 1) * (wmma_K * wmma_N / 4) / 4) +
      ((ty & 7) * (wmma_K / 2) / 4));
    decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
    *(int4*)(&W_smem[ty][tx * 16]) = *(int4*)(&B_decode_local[0]);
    __syncthreads();

    #pragma unroll
    for (int k16 = 0; k16 < K_CHUNK / 16; ++k16) {
      wmma::load_matrix_sync(b_frag, &W_smem[0][k16 * 16], K_CHUNK);
      #pragma unroll
      for (int t = 0; t < TILES_PER_WARP; ++t) {
        wmma::load_matrix_sync(a_frag[t], &A_smem[warp * 16 + t * 64][k16 * 16], K_CHUNK);
        wmma::mma_sync(acc[t], a_frag[t], b_frag, acc[t]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int t = 0; t < TILES_PER_WARP; ++t)
    wmma::store_matrix_sync(&C_smem[warp * 16 + t * 64][0], acc[t], 16, wmma::mem_row_major);
  __syncthreads();

  const int n_base = ((int)blockIdx.x) * N_block_size;
  const int ws_idx = n_base / (N / ws_num);
  const float wsv = ws[ws_idx];
  #pragma unroll
  for (int e = 0; e < (M_ROWS * 16) / 128; ++e) {
    const int lin = tid + e * 128;
    const int ml  = lin >> 4;
    const int col = lin & 15;
    const int m = m_base + ml;
    if (m < M)
      out[m * N + n_base + col] = __float2bfloat16(((float)C_smem[ml][col]) / s[m] * wsv);
  }
}

/**
 * @brief Launch helper for @ref ladder_int8xint2_kernel_m.
 *
 * Builds the launch grid (one CTA per (N/16, M/M_ROWS) tile) and
 * forwards to the kernel.
 */
template <int N, int K, int ws_num, int M_ROWS>
static inline void launch_ladder_int8xint2_m(
    int8_t* A, int8_t* B, __nv_bfloat16* out,
    float* s, float* ws, int M, cudaStream_t stream) {
  ladder_int8xint2_kernel_m<N, K, ws_num, M_ROWS>
    <<<dim3(N / 16, (M + M_ROWS - 1) / M_ROWS, 1), dim3(8, 16, 1), 0, stream>>>(A, B, out, s, ws, M);
}

/**
 * @brief Row-wise int8 quantisation of a bf16 activation matrix.
 *
 * For each row of @p in, computes @c amax = max(|x|), then scales the
 * row by @c 127/amax and round-to-nearest to int8. The per-row scale is
 * written to @p scales; the int8 row is written to @p out. Used to
 * prepare activations for the ternary GEMM kernels above.
 *
 * @tparam BLOCK_THREADS Threads per CTA; must be a multiple of 32.
 * @param in     bf16 activation matrix (M x K), device pointer. M is
 *               passed implicitly via @c blockIdx.x.
 * @param out    int8 quantised matrix (M x K), device pointer.
 * @param scales Per-row scales (length M), device pointer.
 * @param K      Row length.
 */
template <int BLOCK_THREADS>
__global__ void act_quant_kernel(
    const __nv_bfloat16* __restrict__ in,
    int8_t* __restrict__ out,
    float* __restrict__ scales,
    int K)
{
    const int m   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const __nv_bfloat16* row_in  = in  + m * K;
    int8_t*              row_out = out + m * K;

    float local_max = 0.0f;
    for (int k = tid; k < K; k += BLOCK_THREADS) {
        float v = fabsf(__bfloat162float(row_in[k]));
        if (v > local_max) local_max = v;
    }

    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_max, off);
        if (other > local_max) local_max = other;
    }

    __shared__ float smem[32];
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (BLOCK_THREADS + 31) / 32) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(0xffffffff, v, off);
            if (other > v) v = other;
        }
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    const float amax  = smem[0] < 1e-5f ? 1e-5f : smem[0];
    const float scale = 127.0f / amax;
    if (tid == 0) scales[m] = scale;

    for (int k = tid; k < K; k += BLOCK_THREADS) {
        float v = __bfloat162float(row_in[k]) * scale;
        float q = nearbyintf(v);
        if (q >  127.0f) q =  127.0f;
        if (q < -128.0f) q = -128.0f;
        row_out[k] = (int8_t)q;
    }
}
