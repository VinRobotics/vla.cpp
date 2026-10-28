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

#include "bitnet_kernels.h"
#include "bitnet_mma.cuh"

#include "env_flag.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <set>

[[noreturn]] static void bitlinear_unsupported_shape(const char * which,
                                                    int M, int N, int K) {
    std::fprintf(stderr,
        "vla(bitvla): %s has no specialised kernel for (M=%d, N=%d, K=%d). "
        "Register the shape in bitnet_kernels.cu and rebuild.\n",
        which, M, N, K);
    std::abort();
}

// The wide kernel amortises the shared A block over 4 column tiles instead of
// 1 (see ladder_int8xint2_kernel_m_wide). It is the default; set
// VLA_BITVLA_NARROW_GEMM=1 to fall back to the one-tile-per-CTA kernel, which
// is what the A/B correctness harness and any regression bisect want.
static bool bitlinear_use_wide() {
    static const bool wide = !vla::env_flag("VLA_BITVLA_NARROW_GEMM");
    return wide;
}

// bitnet_mma.cuh kernels: same bits as the ladder kernels. VLA_BITVLA_MMA=0
// restores the ladder kernels and the original LM/ViT layer code.
bool bitlinear_mma_enabled() {
    static const bool on = vla::env_flag("VLA_BITVLA_MMA", true);
    return on;
}

// True once set_layer has repacked the weights into the fragment layout.
static bool g_frag = false;

// Buffers already in the fragment layout. A model's free removes its own, so a
// later model whose weights land at the same addresses is repacked again.
static std::mutex g_repacked_mu;
static std::set<const void *> g_repacked;

void bitlinear_repack_for_mma(int8_t * w, int N, int K) {
    {
        std::lock_guard<std::mutex> lk(g_repacked_mu);
        if (!g_repacked.insert(w).second)   // never repack the same buffer twice
            return;
    }
    const cudaError_t e = bitmma::repack_frag(w, N, K, 0);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "vla(bitvla): weight repack failed (N=%d K=%d): %s\n", N, K, cudaGetErrorString(e));
        std::abort();
    }
    g_frag = true;
}

void bitlinear_repack_forget(const void * w) {
    std::lock_guard<std::mutex> lk(g_repacked_mu);
    g_repacked.erase(w);
}

// Index of the tile height that pads M the least (first wins ties).
static int bm_pick(int M, std::initializer_list<int> bms) {
    int best = 0, i = 0, best_rows = 1 << 30;
    for (int bm : bms) {
        const int rows = (M + bm-1)/bm*bm;
        if (rows < best_rows) { best_rows = rows; best = i; }
        ++i;
    }
    return best;
}

// Tile configurations measured on Jetson AGX Orin at M ~330 (LM prompt) and
// M = 256 (one ViT view).
void bitlinear_mma_segs(const int8_t * A, const float * s, bitmma::Segs segs, int M, int K, cudaStream_t stream) {
    segs.frag = g_frag;
    int N = 0;
    for (int i = 0; i < segs.n; ++i) N += segs.s[i].N;
    cudaError_t e;
    if (segs.s[0].hd > 0) {                // q/k/v stored per head
        if (bm_pick(M, {32, 48}) == 0) e = bitmma::launch<32, 128, 32, 32, 4>(A, s, segs, M, K, stream);
        else                           e = bitmma::launch<48, 128, 48, 32, 4>(A, s, segs, M, K, stream);
    } else if (K >= 2560 && N >= 2560) {   // lm.o, lm.gate_up, lm.down
        switch (bm_pick(M, {112, 64, 48})) {
            case 0:  e = bitmma::launch<112, 128, 112, 32, 3>(A, s, segs, M, K, stream); break;
            case 1:  e = bitmma::launch<64, 256, 64, 64, 3>(A, s, segs, M, K, stream); break;
            default: e = bitmma::launch<48, 128, 48, 32, 4>(A, s, segs, M, K, stream); break;
        }
    } else if (N == 640 || K == 4352) {    // lm.k/v alone, vit.fc2
        e = bitmma::launch<32, 128, 32, 32, 4>(A, s, segs, M, K, stream);
    } else {                               // vit.q/k/v/o, vit.fc1
        e = bitmma::launch<32, 256, 32, 64, 4>(A, s, segs, M, K, stream);
    }
    if (e != cudaSuccess) {
        std::fprintf(stderr, "vla(bitvla): bitnet mma launch failed (M=%d N=%d K=%d): %s\n", M, N, K, cudaGetErrorString(e));
        std::abort();
    }
}

static void bitlinear_mma(const int8_t * A, const int8_t * B, __nv_bfloat16 * out,
                          const float * s, const float * ws, int M, int N, int K, int ws_num,
                          cudaStream_t stream) {
    bitmma::Segs sg{};
    sg.n = 1;
    sg.s[0] = {B, out, ws, N, N/ws_num, 0};
    bitlinear_mma_segs(A, s, sg, M, K, stream);
}

extern "C" void bitlinear_int8xint2_m(
    int8_t* input0, int8_t* input1,
    __nv_bfloat16* output0,
    float* s, float* ws,
    int M, int N, int K, cudaStream_t stream)
{
    if (bitlinear_mma_enabled()) {
        int ws_num = 0;
        if      ((N == 2560 && K == 2560) || (N == 640 && K == 2560) || (N == 2560 && K == 6912) ||
                 (N == 1152 && K == 1152) || (N == 4304 && K == 1152) || (N == 1152 && K == 4352)) ws_num = 1;
        else if (N == 13824 && K == 2560) ws_num = 2;
        else
            bitlinear_unsupported_shape("bitlinear_int8xint2_m", M, N, K);
        bitlinear_mma(input0, input1, output0, s, ws, M, N, K, ws_num, stream);
        return;
    }
    if (bitlinear_use_wide()) {
#define WIDE(NN, KK, WS) \
    launch_ladder_int8xint2_m_wide<NN, KK, WS, 128, bitvla_n_tiles_for(NN, KK)>( \
        input0, input1, output0, s, ws, M, stream)
        if      (N == 2560  && K == 2560)
            WIDE(2560,  2560, 1);
        else if (N == 640   && K == 2560) WIDE(640,   2560, 1);
        else if (N == 13824 && K == 2560) WIDE(13824, 2560, 2);
        else if (N == 2560  && K == 6912) WIDE(2560,  6912, 1);

        else if (N == 1152  && K == 1152) WIDE(1152,  1152, 1);
        else if (N == 4304  && K == 1152) WIDE(4304,  1152, 1);
        else if (N == 1152  && K == 4352) WIDE(1152,  4352, 1);
        else
            bitlinear_unsupported_shape("bitlinear_int8xint2_m", M, N, K);
#undef WIDE
        return;
    }

    if      (N == 2560  && K == 2560)
        launch_ladder_int8xint2_m<2560,  2560, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 640   && K == 2560) launch_ladder_int8xint2_m<640,   2560, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 13824 && K == 2560) launch_ladder_int8xint2_m<13824, 2560, 2, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 2560  && K == 6912) launch_ladder_int8xint2_m<2560,  6912, 1, 128>(input0, input1, output0, s, ws, M, stream);

    else if (N == 1152  && K == 1152) launch_ladder_int8xint2_m<1152,  1152, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 4304  && K == 1152) launch_ladder_int8xint2_m<4304,  1152, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 1152  && K == 4352) launch_ladder_int8xint2_m<1152,  4352, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else
        bitlinear_unsupported_shape("bitlinear_int8xint2_m", M, N, K);
}

extern "C" void bitvla_act_quant_cuda(
    const __nv_bfloat16* in, int8_t* out, float* scales,
    int M, int K, int ld_out, cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    act_quant_kernel<BLOCK_THREADS><<<dim3(M, 1, 1), dim3(BLOCK_THREADS, 1, 1), 0, stream>>>(in, out, scales, K, ld_out);
}
