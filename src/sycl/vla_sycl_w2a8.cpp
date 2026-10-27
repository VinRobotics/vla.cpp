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

// W2A8 linear for the pi0.5 action expert on Intel Xe3 (ESIMD, XMX DPAS):
// 2-bit weights with an integer unit per group x INT8 per-token activations.
//
//   w[n, k] = (2 q - 3) * u[n, k / 64],  q in 0..3, u in 1..42;   out[m, n] = sx[m] sn[n] sum_k a[m, k] w[n, k]
//
// The weight blob ("W2P", WeightLoader::quant_w2p) has one block per 16-column
// tile: [K/32 k-steps x 128 B codes][K/64 groups x 16 B units][pad to 64 B]. The
// four 2-bit codes in a byte of a k-step's 32 dwords P[j] belong to the same
// byte position of the four VNNI dwords D[j + 32 t] (t = 0..3) of that k-step's
// s8 B tile, all in column j % 16, so each B dword is
//   ((P >> 2t) & 0x03030303) * 2u + (128 - 3u) * 0x01010101   (per byte 2qu + 128 - 3u, in 2..254: no carries)
//   ^ 0x80808080                                                (= (2q - 3) u as a signed byte)
// four or five dword ops per B dword, then plain s8 x s8 DPAS with int32
// accumulators and the two float scales at the end. |w| <= 126 fits s8.
//
// One thread owns one 16-column tile (or a gate|up tile pair) and up to 64 (56)
// rows; with K >= 2048 two threads split K and sum their int32 partials through
// SLM. A is read row-major with 2D block loads one k-step ahead of the DPAS; the
// weight stream two groups ahead. The epilogue applies the scales, bias, the
// per-channel gate and the residual (F32/BF16/F16 out), or, for a tile pair
// whose weights were packed gate/up interleaved, gelu_tanh(gate) * up (F16 out,
// N/2 columns): the GeGLU that feeds the down GEMM, without the gate|up output.

#include "ggml.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdint>
#include <cstdio>

namespace {

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace xe  = sycl::ext::intel::experimental::esimd;
using P = xmx::dpas_argument_type;

constexpr int CN = 128, AN = 64;  // int32 accumulator dwords per tile, A dwords per row tile

struct W2Args {
    const uint8_t * W;     // W2P blob, N/16 tiles of TS bytes
    const float *   SN;    // per-row weight scale [N] (tile order)
    const int8_t *  A;     // int8 activations, row-major [M][K]
    const float *   SX;    // per-token activation scale [M]
    void *          Y;     // [M][N] of dst_type (GeGLU: [M][N/2] F16), row stride y_ld elements
    const float *   bias;  // [N] or null
    const void *    res;   // [M][N] of res_type, row stride res_ld elements, or null
    const float *   gate;  // [N] or null
    int M, N, K, KS, TS, dst_type, res_type, res_ld, y_ld;
};

// The MTT A tiles (8 rows x 32 k each) of one k-step from row-major A: 32-row 2D
// block loads; rows past M read as zero.
template <int MTT>
ESIMD_INLINE void load_a(const int8_t * A, int K, int M, int row0, int step, simd<int, AN * MTT> & dst) {
    constexpr int R = MTT * 8, H0 = R < 32 ? R : 32, H1 = R - H0;
    simd<int8_t, H0 * 32> b0 = xe::lsc_load_2d<int8_t, 32, H0>(A, K - 1, M - 1, K - 1, step * 32, row0);
    dst.template select<H0 * 8, 1>(0) = b0.template bit_cast_view<int>();
    if constexpr (H1 > 0) {
        simd<int8_t, H1 * 32> b1 = xe::lsc_load_2d<int8_t, 32, H1>(A, K - 1, M - 1, K - 1, step * 32, row0 + 32);
        dst.template select<H1 * 8, 1>(H0 * 8) = b1.template bit_cast_view<int>();
    }
}

ESIMD_INLINE simd<uint16_t, 16> to_bf16(simd<float, 16> f) {
    simd<uint32_t, 16> u = f.template bit_cast_view<uint32_t>();
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;  // round to nearest even
    return convert<uint16_t>(u);
}

ESIMD_INLINE simd<float, 16> load_res(const void * res, int type, int64_t off) {
    if (type == GGML_TYPE_F32) return xe::lsc_block_load<float, 16>((const float *) res + off);
    simd<uint32_t, 8>  packed = xe::lsc_block_load<uint32_t, 8>((const uint32_t *) ((const uint16_t *) res + off));
    simd<uint16_t, 16> hv     = packed.template bit_cast_view<uint16_t>();
    if (type == GGML_TYPE_BF16) {
        simd<uint32_t, 16> u = convert<uint32_t>(hv) << 16;
        return u.template bit_cast_view<float>();
    }
    simd<sycl::half, 16> hf = hv.template bit_cast_view<sycl::half>();
    return convert<float>(hf);
}

ESIMD_INLINE void store_row(void * Y, int type, int64_t off, simd<float, 16> v) {
    if (type == GGML_TYPE_F32) {
        xe::lsc_block_store<float, 16>((float *) Y + off, v);
        return;
    }
    simd<uint16_t, 16> h;
    if (type == GGML_TYPE_BF16) {
        h = to_bf16(v);
    } else {
        simd<sycl::half, 16> hf = convert<sycl::half>(v);
        h = hf.template bit_cast_view<uint16_t>();
    }
    xe::lsc_block_store<uint32_t, 8>((uint32_t *) ((uint16_t *) Y + off), h.template bit_cast_view<uint32_t>());
}

// gelu_tanh(x) = x * sigmoid(2u), u = sqrt(2/pi) x (1 + 0.044715 x^2): the same
// function as gelu_tanh_fast in vla_sycl_fused.cpp.
ESIMD_INLINE simd<float, 16> gelu_tanh(simd<float, 16> x) {
    simd<float, 16> u2 = x * (1.0f + 0.044715f * x * x) * 1.5957691216057308f;
    return x / (exp(-u2) + 1.0f);
}

// S: split-K threads; NTT: 16-column tiles per thread (2 = a gate|up pair with the GeGLU epilogue); MTT: row tiles.
template <int S, int NTT, int MTT, bool GEGLU>
void launch(sycl::queue & q, const W2Args & a) {
    constexpr int ROWS = MTT * 8;
    const int NT = a.N / 16, NWG = NT / NTT, MC = (a.M + ROWS - 1) / ROWS, KSt = a.KS / S;
    const W2Args ar = a;
    auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        if constexpr (S > 1) slm_init<S * MTT * NTT * CN * 4>();
        const int wg = (int) it.get_group(0), s = (int) it.get_local_id(0);
        const int ntg = wg % NWG, mc = wg / NWG, nt0 = ntg * NTT, row0 = mc * ROWS;
        const int mrows = ar.M - row0 < ROWS ? ar.M - row0 : ROWS;
        const int nact = (mrows + 7) / 8;  // active row tiles (uniform)
        const int s0 = s * KSt, g0 = s0 / 2, NGt = KSt / 2;
        const uint8_t * wt = ar.W + (size_t) nt0 * ar.TS;
        const int * codes = (const int *) wt + (size_t) s0 * 32;
        const uint8_t * units = wt + (size_t) ar.KS * 128 + (size_t) g0 * 16;
        simd<int, CN> acc[MTT][NTT];
#pragma unroll
        for (int i = 0; i < MTT; ++i)
#pragma unroll
            for (int j = 0; j < NTT; ++j) acc[i][j] = 0;
        // weight stream: two groups (2 x 256 B of codes + 2 x 16 B of units) in flight per tile
        simd<int, 64> Pring[2][NTT];
        simd<uint32_t, 4> uring[2][NTT];
#pragma unroll
        for (int d = 0; d < 2; ++d)
#pragma unroll
            for (int j = 0; j < NTT; ++j) {
                Pring[d][j] = block_load<int, 64>(codes + (size_t) j * ar.TS / 4 + d * 64, overaligned_tag<16>{});
                uring[d][j] = xe::lsc_block_load<uint32_t, 4>((const uint32_t *) (units + (size_t) j * ar.TS + d * 16));
            }
        simd<int, AN * MTT> Abuf[2];
        load_a<MTT>(ar.A, ar.K, ar.M, row0, s0, Abuf[0]);
        const simd<uint32_t, 32> H(0x80808080u);
        for (int gi = 0; gi < NGt; gi += 2) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int g = gi + u;
                simd<uint32_t, 32> mul2u[NTT], badd[NTT];
#pragma unroll
                for (int j = 0; j < NTT; ++j) {
                    simd<uint8_t, 16>  ub  = uring[u][j].template bit_cast_view<uint8_t>();
                    simd<uint32_t, 16> u16 = convert<uint32_t>(ub);
                    simd<uint32_t, 32> u32 = u16.template replicate<2>();
                    mul2u[j] = u32 * 2u;
                    badd[j]  = simd<uint32_t, 32>(0x80808080u) - u32 * 0x03030303u;  // 128 - 3u per byte
                }
#pragma unroll
                for (int ss = 0; ss < 2; ++ss) {
                    const int step = s0 + g * 2 + ss;
                    const int par = ss;  // k-step parity within the group (static)
                    if (step + 1 < s0 + KSt) load_a<MTT>(ar.A, ar.K, ar.M, row0, step + 1, Abuf[par ^ 1]);
#pragma unroll
                    for (int j = 0; j < NTT; ++j) {
                        simd<int, 32>      Pi = Pring[u][j].template select<32, 1>(ss * 32);
                        simd<uint32_t, 32> Pc = Pi.template bit_cast_view<uint32_t>();
                        simd<int, CN>      B;
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            simd<uint32_t, 32> qb = (t ? (Pc >> (uint32_t) (2 * t)) : Pc) & 0x03030303u;  // q per byte
                            simd<uint32_t, 32> wb = (qb * mul2u[j] + badd[j]) ^ H;                       // (2q-3) u
                            B.template select<32, 1>(32 * t) = wb.template bit_cast_view<int>();
                        }
#pragma unroll
                        for (int i = 0; i < MTT; ++i) {
                            if (i >= nact) break;
                            simd<int, AN> Ai = Abuf[par].template select<AN, 1>(i * AN);
                            acc[i][j] = xmx::dpas<8, 8, int, int, int, int, P::s8, P::s8>(acc[i][j], B, Ai);
                        }
                    }
                }
                if (g + 2 < NGt) {
#pragma unroll
                    for (int j = 0; j < NTT; ++j) {
                        Pring[u][j] = block_load<int, 64>(codes + (size_t) j * ar.TS / 4 + (size_t) (g + 2) * 64, overaligned_tag<16>{});
                        uring[u][j] = xe::lsc_block_load<uint32_t, 4>((const uint32_t *) (units + (size_t) j * ar.TS + (g + 2) * 16));
                    }
                }
            }
        }
        if constexpr (S > 1) {
#pragma unroll
            for (int i = 0; i < MTT; ++i)
#pragma unroll
                for (int j = 0; j < NTT; ++j)
                    slm_block_store<int, CN>((uint32_t) (((s * MTT + i) * NTT + j) * CN * 4), acc[i][j]);
            barrier();
        }
        simd<float, 16> sn[NTT], b16[NTT], g16[NTT];
#pragma unroll
        for (int j = 0; j < NTT; ++j) {
            sn[j]  = xe::lsc_block_load<float, 16>(ar.SN + (nt0 + j) * 16);
            b16[j] = 0.0f;
            g16[j] = 1.0f;
            if (ar.bias) b16[j] = xe::lsc_block_load<float, 16>(ar.bias + (nt0 + j) * 16);
            if (ar.gate) g16[j] = xe::lsc_block_load<float, 16>(ar.gate + (nt0 + j) * 16);
        }
#pragma unroll
        for (int i = 0; i < MTT; ++i) {
            if (i >= nact || i % S != s) continue;
            simd<float, CN> f[NTT];
#pragma unroll
            for (int j = 0; j < NTT; ++j) {
                simd<int, CN> tot = acc[i][j];
                if constexpr (S > 1) {
                    tot = 0;
#pragma unroll
                    for (int s2 = 0; s2 < S; ++s2)
                        tot += slm_block_load<int, CN>((uint32_t) (((s2 * MTT + i) * NTT + j) * CN * 4));
                }
                f[j] = convert<float>(tot) * sn[j].template replicate<8>();
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int m = row0 + i * 8 + r;
                if (m < ar.M) {
                    if constexpr (GEGLU) {
                        // tile pair (gate, up) -> gelu_tanh(gate) * up, F16, column block ntg
                        simd<float, 16> gv = f[0].template select<16, 1>(r * 16) * ar.SX[m];
                        simd<float, 16> uv = f[NTT - 1].template select<16, 1>(r * 16) * ar.SX[m];
                        store_row(ar.Y, GGML_TYPE_F16, (int64_t) m * ar.y_ld + ntg * 16, gelu_tanh(gv) * uv);
                    } else {
#pragma unroll
                        for (int j = 0; j < NTT; ++j) {
                            simd<float, 16> v = f[j].template select<16, 1>(r * 16) * ar.SX[m] + b16[j];
                            v *= g16[j];
                            if (ar.res) v += load_res(ar.res, ar.res_type, (int64_t) m * ar.res_ld + (nt0 + j) * 16);
                            store_row(ar.Y, ar.dst_type, (int64_t) m * ar.y_ld + (nt0 + j) * 16, v);
                        }
                    }
                }
            }
        }
    };
    q.parallel_for(sycl::nd_range<1>(NWG * MC * S, S),
                   sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> }, kern);
}

}  // namespace

namespace vla {

// Called from the W8A8 linear (vla_sycl_bf16.cpp) once the INT8 activation rows
// and their scales are in place. geglu: the blob's tiles are gate/up pairs and
// dst is gelu_tanh(gate) * up, F16 [N/2, M]. false: shape not handled.
bool sycl_w2a8_linear(sycl::queue & q, const ggml_tensor * dst, const uint8_t * W, const float * sn, const int8_t * qx,
                      const float * sx, long long M, long long N, long long K, long long TS, const ggml_tensor * bias,
                      const ggml_tensor * res, const ggml_tensor * gate, bool geglu) {
    if (N % 16 || K % 128 || K > (1 << 23) || M > (1 << 23) || ((uintptr_t) qx % 64) || dst->nb[1] % 64) return false;
    if (geglu) {
        if (N % 32 || dst->type != GGML_TYPE_F16 || bias || res || gate) return false;
    } else if ((dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_BF16 && dst->type != GGML_TYPE_F16) ||
               (res && res->type != GGML_TYPE_F32 && res->type != GGML_TYPE_BF16 && res->type != GGML_TYPE_F16) ||
               (bias && bias->type != GGML_TYPE_F32) || (gate && gate->type != GGML_TYPE_F32) || (res && res->nb[1] % 64)) {
        return false;
    }
    W2Args a;
    a.W = W; a.SN = sn; a.A = qx; a.SX = sx; a.Y = dst->data;
    a.bias = bias ? (const float *) bias->data : nullptr;
    a.res  = res ? res->data : nullptr;
    a.gate = gate ? (const float *) gate->data : nullptr;
    a.M = (int) M; a.N = (int) N; a.K = (int) K; a.KS = (int) (K / 32); a.TS = (int) TS;
    a.dst_type = dst->type;
    a.res_type = res ? res->type : GGML_TYPE_F32;
    a.res_ld   = res ? (int) (res->nb[1] / ggml_type_size(res->type)) : 0;
    a.y_ld     = (int) (dst->nb[1] / ggml_type_size(dst->type));
    // split K in two when a thread would otherwise walk 64+ k-steps (K >= 2048)
    const bool split = K >= 2048 && (K / 32 / 2) % 2 == 0;
    if (geglu) {
        if (split) launch<2, 2, 7, true>(q, a);
        else       launch<1, 2, 7, true>(q, a);
    } else {
        if (split) launch<2, 1, 8, false>(q, a);
        else       launch<1, 1, 8, false>(q, a);
    }
    return true;
}

}  // namespace vla
