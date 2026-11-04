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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

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
    // RoPE + K/V-write epilogue (EPI_ROPE): tile pair p = (head hh, cols c*16.. and c*16 + hd/2..)
    const int32_t * pos;   // [M]
    const float *   rt;    // cos/sin table [npos][hd]: cos(pos * ts^i) for i < hd/2, then sin
    void *          kb;    // K cache, F16; element strides kb_s1 (head), kb_s2 (row); rows row0 + m
    void *          vb;
    int nq, nkv, hd, row0, kb_s1, kb_s2, vb_s1, vb_s2, npos;
};

enum { EPI_NONE = 0, EPI_GEGLU = 1, EPI_ROPE = 2 };

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
template <int S, int NTT, int MTT, int EPI>
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
                    if constexpr (EPI == EPI_ROPE) {
                        // pair ntg: head hh, columns i = c*16 + lane and i + hd/2
                        const int tph = ar.hd / 32, hh = ntg / tph, c = ntg % tph;
                        simd<float, 16> x0 = f[0].template select<16, 1>(r * 16) * ar.SX[m];
                        simd<float, 16> x1 = f[1].template select<16, 1>(r * 16) * ar.SX[m];
                        uint16_t * o;
                        if (hh < ar.nq)              o = (uint16_t *) ar.Y + ((int64_t) m * ar.nq + hh) * ar.hd;
                        else if (hh < ar.nq + ar.nkv) o = (uint16_t *) ar.kb + (int64_t) (ar.row0 + m) * ar.kb_s2 + (hh - ar.nq) * ar.kb_s1;
                        else                         o = (uint16_t *) ar.vb + (int64_t) (ar.row0 + m) * ar.vb_s2 + (hh - ar.nq - ar.nkv) * ar.vb_s1;
                        if (hh < ar.nq + ar.nkv) {
                            int p = ar.pos[m]; p = p < ar.npos ? p : ar.npos - 1;
                            const float * tb = ar.rt + (int64_t) p * ar.hd + c * 16;
                            simd<float, 16> cs = xe::lsc_block_load<float, 16>(tb);
                            simd<float, 16> sn = xe::lsc_block_load<float, 16>(tb + ar.hd / 2);
                            simd<float, 16> y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs;
                            x0 = y0; x1 = y1;
                        }
                        store_row(o, GGML_TYPE_F16, c * 16, x0);
                        store_row(o, GGML_TYPE_F16, c * 16 + ar.hd / 2, x1);
                    } else if constexpr (EPI == EPI_GEGLU) {
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

// Work split per call. A work-group owns NTT 16-column tiles and MTT*8 rows; S threads split K. A thread walks
// K/32/S k-steps in pairs of 64-k groups, two groups per loop trip, so KS/S must be a multiple of 4.
// At M = 50, one 56/64-row work-group per tile fills a 4-Xe-core part (32 XVEs) but leaves a 12-Xe-core part
// (96 XVEs) mostly idle. Measured on the Arc B390 (SYCL graph, weights from DRAM), per expert layer
// q|k|v 26.7 -> 15.7 us, o 12.0 -> 9.7, gate|up 33.3 -> 28.9, down 17.8 -> 14.8: on parts with >= 64 XVEs use 32-row
// blocks (the second block's weight reads hit the cache), then split K until there are ~2 threads per XVE; K >= 2048
// is always split at least in two. On 32-XVE parts this picks the original split (K >= 2048 -> 2, else 1).
// The int32 partials are summed exactly, so every choice gives bit-identical results.
// VLA_W2_SPLIT=1|2|4|8 and VLA_W2_MTT=2|4|7 (rows/8; 7 = 56/64-row blocks) override, for tuning.
static int env_int(const char * name) {
    const char * e = std::getenv(name);
    return e ? std::atoi(e) : 0;
}

static long long xve_count(sycl::queue & q) {
    static const long long n = q.get_device().get_info<sycl::info::device::max_compute_units>();
    return n;
}

int pick_split(sycl::queue & q, long long wgs, int KS) {
    auto ok = [KS](int s) { return KS % (4 * s) == 0; };
    static const int env = env_int("VLA_W2_SPLIT");
    if (env > 0) {
        int s = env >= 8 ? 8 : env >= 4 ? 4 : env >= 2 ? 2 : 1;
        while (s > 1 && !ok(s)) s /= 2;
        return s;
    }
    int s = KS >= 64 && ok(2) ? 2 : 1;
    while (s < 8 && wgs * s < 2 * xve_count(q) && ok(s * 2)) s *= 2;
    return s;
}

template <int NTT, int MTT, int EPI>
void launch_s(sycl::queue & q, const W2Args & a) {
    const int S = pick_split(q, (long long) (a.N / 16 / NTT) * ((a.M + MTT * 8 - 1) / (MTT * 8)), a.KS);
    switch (S) {
        case 8:  launch<8, NTT, MTT, EPI>(q, a); break;
        case 4:  launch<4, NTT, MTT, EPI>(q, a); break;
        case 2:  launch<2, NTT, MTT, EPI>(q, a); break;
        default: launch<1, NTT, MTT, EPI>(q, a); break;
    }
}

template <int NTT, int MTT, int EPI>
void dispatch(sycl::queue & q, const W2Args & a) {
    static const int env = env_int("VLA_W2_MTT");
    const int mtt = env > 0 ? env : (xve_count(q) >= 64 ? 4 : MTT);
    if (mtt == 2)      launch_s<NTT, 2, EPI>(q, a);
    else if (mtt == 4) launch_s<NTT, 4, EPI>(q, a);
    else               launch_s<NTT, MTT, EPI>(q, a);
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
    if (geglu) dispatch<2, 7, EPI_GEGLU>(q, a);
    else       dispatch<1, 8, EPI_NONE>(q, a);
    return true;
}

// cos/sin table for the RoPE epilogue: [npos][hd], cos(p * ts^i) for i < hd/2 then sin, computed in double on the
// host once per (hd, theta_scale) and kept on the device (the queue is in order; one table serves every call).
static const float * rope_table(sycl::queue & q, int hd, float theta_scale, int npos) {
    static std::mutex mu;
    struct Entry { int hd; float ts; int npos; float * dev; };
    static std::vector<Entry> tables;
    std::lock_guard<std::mutex> lock(mu);
    for (const Entry & e : tables)
        if (e.hd == hd && e.ts == theta_scale && e.npos == npos) return e.dev;
    std::vector<float> h((size_t) npos * hd);
    for (int p = 0; p < npos; ++p)
        for (int i = 0; i < hd / 2; ++i) {
            const double th = (double) p * std::pow((double) theta_scale, (double) i);
            h[(size_t) p * hd + i]          = (float) std::cos(th);
            h[(size_t) p * hd + hd / 2 + i] = (float) std::sin(th);
        }
    float * dev = sycl::malloc_device<float>(h.size(), q);
    // through a queue of its own: a wait on the compute queue is illegal while it records a SYCL graph
    sycl::queue(q.get_context(), q.get_device()).memcpy(dev, h.data(), h.size() * sizeof(float)).wait();
    tables.push_back({ hd, theta_scale, npos, dev });
    return dev;
}

bool sycl_w2a8_ropekv(sycl::queue & q, const ggml_tensor * dst, const uint8_t * W, const float * sn, const int8_t * qx,
                      const float * sx, long long M, long long K, long long TS, const int32_t * pos, ggml_tensor * kb,
                      ggml_tensor * vb, int nq, int nkv, int hd, long long row0, float theta_scale) {
    const long long N = (long long) (nq + 2 * nkv) * hd;
    if (hd % 32 || K % 128 || K > (1 << 23) || M > (1 << 23) || ((uintptr_t) qx % 64) || dst->type != GGML_TYPE_F16 ||
        kb->type != GGML_TYPE_F16 || vb->type != GGML_TYPE_F16 || kb->nb[1] % 32 || kb->nb[2] % 32 || vb->nb[1] % 32 ||
        vb->nb[2] % 32)
        return false;
    constexpr int NPOS = 8192;
    W2Args a{};
    a.W = W; a.SN = sn; a.A = qx; a.SX = sx; a.Y = dst->data;
    a.M = (int) M; a.N = (int) N; a.K = (int) K; a.KS = (int) (K / 32); a.TS = (int) TS;
    a.dst_type = GGML_TYPE_F16; a.y_ld = hd;
    a.pos = pos; a.rt = rope_table(q, hd, theta_scale, NPOS); a.npos = NPOS;
    a.kb = kb->data; a.vb = vb->data; a.nq = nq; a.nkv = nkv; a.hd = hd; a.row0 = (int) row0;
    a.kb_s1 = (int) (kb->nb[1] / 2); a.kb_s2 = (int) (kb->nb[2] / 2);
    a.vb_s1 = (int) (vb->nb[1] / 2); a.vb_s2 = (int) (vb->nb[2] / 2);
    dispatch<2, 7, EPI_ROPE>(q, a);
    return true;
}

}  // namespace vla
