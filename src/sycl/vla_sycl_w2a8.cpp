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
// 2-bit weights with power-of-two group shifts x INT8 per-token activations.
//
//   w[n, k] = (2 q - 3) << e[n, k / 64],  q in 0..3, e in 0..4;  out[m, n] = sx[m] sn[n] sum_k a[m, k] w[n, k]
//
// The weight blob ("W2P", WeightLoader::quant_w2p) has one block per 16-column
// tile: [K/32 k-steps x 128 B codes][K/64 groups x 8 B exponent nibbles][pad to
// 64 B]. The four 2-bit codes in a byte of a k-step's 32 dwords P[j] belong to
// the same byte position of the four VNNI dwords D[j + 32 t] (t = 0..3) of that
// k-step's s8 B tile, all in column j % 16, so each B dword is
//   rol(P[j], e + 1 - 2 t) & (0x03030303 << (e + 1))   = q << (e+1) per byte,
// and (u | 0x80..) - (3 << e) * 0x0101.. ^ 0x80.. is the per-byte signed value
// (2q - 3) << e without borrows between bytes: four dword ops per B dword, then
// plain s8 x s8 DPAS with int32 accumulators and the two float scales at the end.
//
// One thread owns one 16-column tile and up to 64 rows (8 DPAS row tiles); with
// K >= 2048 two threads split K and sum their int32 partials through SLM. A is
// read row-major with 2D block loads one k-step ahead of the DPAS; the weight
// stream two groups ahead. The epilogue applies the scales, bias, the per-channel
// gate and the residual, and stores F32, BF16 or F16.

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

constexpr int CN = 128, AN = 64, MTT = 8;  // int32 accumulator dwords per tile, A dwords per tile, row tiles per thread

struct W2Args {
    const uint8_t * W;     // W2P blob, N/16 tiles of TS bytes
    const float *   SN;    // per-row weight scale [N]
    const int8_t *  A;     // int8 activations, row-major [M][K]
    const float *   SX;    // per-token activation scale [M]
    void *          Y;     // [M][N] of dst_type, row stride y_ld elements
    const float *   bias;  // [N] or null
    const void *    res;   // [M][N] of res_type, row stride res_ld elements, or null
    const float *   gate;  // [N] or null
    int M, N, K, KS, TS, NG, dst_type, res_type, res_ld, y_ld;
};

// The 8 exponent nibbles of one group -> 16 dwords (column n = 0..15).
ESIMD_INLINE simd<uint32_t, 16> decode_e(simd<uint32_t, 2> eb) {
    simd<uint32_t, 16> e;
    simd<uint32_t, 8> by = eb.template bit_cast_view<uint8_t>().template select<8, 1>(0);
    e.template select<8, 2>(0) = by & 0xFu;
    e.template select<8, 2>(1) = by >> 4;
    return e;
}

// The 8 A tiles (64 rows x 32 k) of one k-step from row-major A: two 32-row 2D
// block loads; rows past M read as zero.
ESIMD_INLINE void load_a(const int8_t * A, int K, int M, int row0, int step, simd<int, AN * MTT> & dst) {
    simd<int8_t, 1024> b0 = xe::lsc_load_2d<int8_t, 32, 32>(A, K - 1, M - 1, K - 1, step * 32, row0);
    dst.template select<256, 1>(0) = b0.template bit_cast_view<int>();
    simd<int8_t, 1024> b1 = xe::lsc_load_2d<int8_t, 32, 32>(A, K - 1, M - 1, K - 1, step * 32, row0 + 32);
    dst.template select<256, 1>(256) = b1.template bit_cast_view<int>();
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

template <int S>
void launch(sycl::queue & q, const W2Args & a) {
    const int NT = a.N / 16, MC = (a.M + MTT * 8 - 1) / (MTT * 8), KSt = a.KS / S;
    const W2Args ar = a;
    auto kern = [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        if constexpr (S > 1) slm_init<S * MTT * CN * 4>();
        const int wg = (int) it.get_group(0), s = (int) it.get_local_id(0);
        const int nt = wg % NT, mc = wg / NT, row0 = mc * MTT * 8;
        const int mrows = ar.M - row0 < MTT * 8 ? ar.M - row0 : MTT * 8;
        const int nact = (mrows + 7) / 8;  // active row tiles (uniform)
        const int s0 = s * KSt, g0 = s0 / 2, NGt = KSt / 2;
        const uint8_t * wt = ar.W + (size_t) nt * ar.TS;
        const int * codes = (const int *) wt + (size_t) s0 * 32;
        const uint8_t * exps = wt + (size_t) ar.KS * 128 + (size_t) g0 * 8;
        simd<int, CN> acc[MTT];
#pragma unroll
        for (int i = 0; i < MTT; ++i) acc[i] = 0;
        // weight stream: two groups (2 x 256 B of codes + exponents) in flight
        simd<int, 64> Pring[2];
        simd<uint32_t, 2> ering[2];
#pragma unroll
        for (int d = 0; d < 2; ++d) {
            Pring[d] = block_load<int, 64>(codes + d * 64, overaligned_tag<16>{});
            ering[d] = xe::lsc_block_load<uint32_t, 2>((const uint32_t *) (exps + d * 8));
        }
        simd<int, AN * MTT> Abuf[2];
        load_a(ar.A, ar.K, ar.M, row0, s0, Abuf[0]);
        for (int gi = 0; gi < NGt; gi += 2) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int g = gi + u;
                const simd<uint32_t, 16> e16 = decode_e(ering[u]);
                const simd<uint32_t, 32> e32 = e16.template replicate<2>();
                const simd<uint32_t, 32> rot0 = e32 + 1u;
                const simd<uint32_t, 32> mask = simd<uint32_t, 32>(0x03030303u) << (e32 + 1u);
                const simd<uint32_t, 32> bias = simd<uint32_t, 32>(0x03030303u) << e32;
                const simd<uint32_t, 32> H(0x80808080u);
#pragma unroll
                for (int ss = 0; ss < 2; ++ss) {
                    const int step = s0 + g * 2 + ss;
                    const int par = ss;  // k-step parity within the group (static)
                    if (step + 1 < s0 + KSt) load_a(ar.A, ar.K, ar.M, row0, step + 1, Abuf[par ^ 1]);
                    simd<int, 32> Pi = Pring[u].template select<32, 1>(ss * 32);
                    simd<uint32_t, 32> Pc = Pi.template bit_cast_view<uint32_t>();
                    simd<int, CN> B;
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        simd<uint32_t, 32> rot = (rot0 - (uint32_t) (2 * t)) & 31u;
                        simd<uint32_t, 32> ub  = rol<uint32_t>(Pc, rot) & mask;  // q << (e+1) per byte
                        ub = ((ub | H) - bias) ^ H;                              // (2q-3) << e per byte, no borrows
                        B.template select<32, 1>(32 * t) = ub.template bit_cast_view<int>();
                    }
#pragma unroll
                    for (int i = 0; i < MTT; ++i) {
                        if (i >= nact) break;
                        simd<int, AN> Ai = Abuf[par].template select<AN, 1>(i * AN);
                        acc[i] = xmx::dpas<8, 8, int, int, int, int, P::s8, P::s8>(acc[i], B, Ai);
                    }
                }
                if (g + 2 < NGt) {
                    Pring[u] = block_load<int, 64>(codes + (size_t) (g + 2) * 64, overaligned_tag<16>{});
                    ering[u] = xe::lsc_block_load<uint32_t, 2>((const uint32_t *) (exps + (g + 2) * 8));
                }
            }
        }
        if constexpr (S > 1) {
#pragma unroll
            for (int i = 0; i < MTT; ++i) slm_block_store<int, CN>((uint32_t) ((s * MTT + i) * CN * 4), acc[i]);
            barrier();
        }
        const simd<float, 16> sn16 = xe::lsc_block_load<float, 16>(ar.SN + nt * 16);
        simd<float, 16> b16 = 0.0f, g16 = 1.0f;
        if (ar.bias) b16 = xe::lsc_block_load<float, 16>(ar.bias + nt * 16);
        if (ar.gate) g16 = xe::lsc_block_load<float, 16>(ar.gate + nt * 16);
#pragma unroll
        for (int i = 0; i < MTT; ++i) {
            if (i >= nact || i % S != s) continue;
            simd<int, CN> tot = acc[i];
            if constexpr (S > 1) {
                tot = 0;
#pragma unroll
                for (int s2 = 0; s2 < S; ++s2) tot += slm_block_load<int, CN>((uint32_t) ((s2 * MTT + i) * CN * 4));
            }
            simd<float, CN> f = convert<float>(tot) * sn16.template replicate<8>();
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int m = row0 + i * 8 + r;
                if (m < ar.M) {
                    simd<float, 16> v = f.template select<16, 1>(r * 16) * ar.SX[m] + b16;
                    v *= g16;
                    if (ar.res) v += load_res(ar.res, ar.res_type, (int64_t) m * ar.res_ld + nt * 16);
                    store_row(ar.Y, ar.dst_type, (int64_t) m * ar.y_ld + nt * 16, v);
                }
            }
        }
    };
    q.parallel_for(sycl::nd_range<1>(NT * MC * S, S),
                   sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> }, kern);
}

}  // namespace

namespace vla {

// Called from the W8A8 linear (vla_sycl_bf16.cpp) once the INT8 activation rows
// and their scales are in place. false: shape not handled (the caller reports).
bool sycl_w2a8_linear(sycl::queue & q, const ggml_tensor * dst, const uint8_t * W, const float * sn, const int8_t * qx,
                      const float * sx, long long M, long long N, long long K, long long TS, const ggml_tensor * bias,
                      const ggml_tensor * res, const ggml_tensor * gate) {
    if (N % 16 || K % 128 || (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_BF16 && dst->type != GGML_TYPE_F16) ||
        (res && res->type != GGML_TYPE_F32 && res->type != GGML_TYPE_BF16 && res->type != GGML_TYPE_F16) ||
        (bias && bias->type != GGML_TYPE_F32) || (gate && gate->type != GGML_TYPE_F32) ||
        dst->nb[1] % 64 || (res && res->nb[1] % 64) || ((uintptr_t) qx % 64) || K > (1 << 23) || M > (1 << 23)) {
        return false;
    }
    W2Args a;
    a.W = W; a.SN = sn; a.A = qx; a.SX = sx; a.Y = dst->data;
    a.bias = bias ? (const float *) bias->data : nullptr;
    a.res  = res ? res->data : nullptr;
    a.gate = gate ? (const float *) gate->data : nullptr;
    a.M = (int) M; a.N = (int) N; a.K = (int) K; a.KS = (int) (K / 32); a.TS = (int) TS; a.NG = (int) (K / 64);
    a.dst_type = dst->type;
    a.res_type = res ? res->type : GGML_TYPE_F32;
    a.res_ld   = res ? (int) (res->nb[1] / ggml_type_size(res->type)) : 0;
    a.y_ld     = (int) (dst->nb[1] / ggml_type_size(dst->type));
    // split K in two when a thread would otherwise walk 64+ k-steps (K >= 2048)
    if (K >= 2048 && (K / 32 / 2) % 2 == 0) launch<2>(q, a);
    else                                    launch<1>(q, a);
    return true;
}

}  // namespace vla
