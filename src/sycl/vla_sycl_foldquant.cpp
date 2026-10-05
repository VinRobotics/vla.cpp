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

// FoldQuant's two custom nodes (src/layers/fq_linear.h) on ggml's SYCL backend,
// reached through the hook scripts/patch_ggml_sycl_ext_hook.py adds; the SYCL
// counterpart of src/cuda/vla_cuda_foldquant.cu and src/kernels/foldquant/.
//
// fq_act: the reduction tree foldquant_ref.h mirrors, with lane l of a 32-wide
// sub-group owning the 64-element chunks l, l+32, ... of a row, so the codes and
// scales are bit-identical to the CPU reference. Rows up to K = 8192 (all a VLA
// has but its 16384-wide down projections) spread over a work-group; longer ones
// run one sub-group per row.
//
// fq_gemm: INT8/INT4 x INT8/INT4 -> INT32, then ((float) acc * act_scale[m]) *
// wscale[n] (+ bias[n]) (+ residual), the reference's epilogue order. The sums
// are integer, so any tiling gives the reference's result: oneDNN's int8 matmul
// when ggml-sycl is built with it (GGML_SYCL_DNN, the default), else this file's
// GEMV (M <= 32) and XMX kernels.
//
// Exactness rests on the build flags in CMakeLists.txt: -ffp-contract=off for
// the source, and -cl-fp32-correctly-rounded-divide-sqrt for the device JIT,
// without which the driver's division and sqrt are approximate and move codes
// across rounding ties. test_foldquant_sycl_op checks every path byte for byte.

#include "sycl/vla_sycl_foldquant.h"

#include "foldquant.h"
#include "foldquant_ref.h"
#include "ggml.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <unordered_map>

#ifdef VLA_FQ_SYCL_DNNL
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#endif

extern "C" {
typedef bool (*ggml_sycl_ext_forward_t)(struct ggml_tensor * dst, void * queue);
typedef bool (*ggml_sycl_ext_supports_t)(const struct ggml_tensor * op);
extern ggml_sycl_ext_forward_t  ggml_sycl_ext_forward;
extern ggml_sycl_ext_supports_t ggml_sycl_ext_supports;
}

namespace vla {

namespace {

constexpr int CHUNK = 64;
constexpr int SG    = 32;      // sub-group width: the reference's lane count
constexpr int ROWS  = 4;       // rows (sub-groups) per work-group

struct ActParams {
    const float * x;           // [M][x_stride]
    int64_t       x_stride;
    const float * ascale;      // [K] or null
    const float * gamma;       // [K] or null
    int8_t *      blob;        // [M][row_bytes]
    int64_t       row_bytes;
    int64_t       M, K;
    bool          fold_before;
    float         clip, eps, inv_sqrt_bs;
};

struct GemmParams {
    const int8_t * w;          // [N][K_pack]
    const int8_t * blob;       // [M][row_bytes]
    const float *  wscale;     // [N]
    const float *  bias;       // [N] or null
    const float *  res;        // [M][N] or null
    float *        y;          // [M][N]
    int64_t        M, N, K, row_bytes;
};

const void * fq_userdata(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CUSTOM) return nullptr;
    struct custom_params { void * fun; int n_tasks; void * userdata; };   // ggml_custom_op_params
    custom_params p;
    std::memcpy(&p, op->op_params, sizeof(p));
    return p.userdata;
}

uint32_t fq_magic(const void * ud) {
    uint32_t m = 0;
    if (ud) std::memcpy(&m, ud, sizeof(m));
    return m;
}

// Load one 64-float chunk, apply the pre-quant chain and rotate it. ROT is a
// compile-time constant so every index into v[] is static (registers).
template <int ROT>
inline void load_chunk(const ActParams & a, const float * x, int64_t c0, float rstd, float * v) {
    for (int i = 0; i < CHUNK; ++i) v[i] = x[c0 + i];
    if (a.gamma)
        for (int i = 0; i < CHUNK; ++i) v[i] = (v[i] * rstd) * a.gamma[c0 + i];
    if (a.ascale && a.fold_before)
        for (int i = 0; i < CHUNK; ++i) v[i] = v[i] / a.ascale[c0 + i];
    if (ROT > 1) {
        // Natural-order Sylvester butterfly on every ROT sub-block of the chunk,
        // the pairs and stage order of fwht_row.
        for (int h = 1; h < ROT; h <<= 1)
            for (int p = 0; p < CHUNK / 2; ++p) {
                const int   j = ((p / h) * 2 * h) + (p % h);
                const float u = v[j], w = v[j + h];
                v[j]     = u + w;
                v[j + h] = u - w;
            }
        for (int i = 0; i < CHUNK; ++i) v[i] = v[i] * a.inv_sqrt_bs;
    }
    if (a.ascale && !a.fold_before)
        for (int i = 0; i < CHUNK; ++i) v[i] = v[i] / a.ascale[c0 + i];
}

template <int ABITS, int ROT>
void act_kernel(const ActParams a, const sycl::nd_item<1> & it) {
    const auto    sg   = it.get_sub_group();
    const int     lane = (int) sg.get_local_linear_id();
    const int64_t m    = (int64_t) it.get_group(0) * ROWS + (int64_t) sg.get_group_linear_id();
    if (m >= a.M) return;
    const int64_t K       = a.K;
    const int     nchunks = (int) (K / CHUNK);
    const float * x       = a.x + m * a.x_stride;

    float rstd = 0.0f;
    if (a.gamma) {
        float p = 0.0f;
        for (int c = lane; c < nchunks; c += SG)
            for (int i = 0; i < CHUNK; ++i) {
                const float q = x[(int64_t) c * CHUNK + i];
                p = p + (q * q);
            }
        for (int off = 16; off > 0; off >>= 1) p = p + sycl::permute_group_by_xor(sg, p, off);
        rstd = 1.0f / sycl::sqrt((p / (float) K) + a.eps);
    }

    float v[CHUNK];
    float mx = 0.0f;
    for (int c = lane; c < nchunks; c += SG) {
        load_chunk<ROT>(a, x, (int64_t) c * CHUNK, rstd, v);
        for (int i = 0; i < CHUNK; ++i) mx = sycl::fmax(mx, sycl::fabs(v[i]));
    }
    for (int off = 16; off > 0; off >>= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, off));

    const float qmax  = ABITS == 4 ? 7.0f : 127.0f;
    float       scale = (a.clip * mx) / qmax;
    if (scale < 1e-12f) scale = 1e-12f;
    // Multiply by the reciprocal, as the reference and the TensorRT kernels do.
    const float inv = 1.0f / scale;

    int8_t * row = a.blob + m * a.row_bytes;
    for (int c = lane; c < nchunks; c += SG) {
        if (nchunks > SG) load_chunk<ROT>(a, x, (int64_t) c * CHUNK, rstd, v);
        if (ABITS == 8) {
            int8_t * dst = row + (int64_t) c * CHUNK;
            for (int i = 0; i < CHUNK; ++i) {
                float q = sycl::rint(v[i] * inv);
                q = sycl::fmin(qmax, sycl::fmax(-qmax, q));
                dst[i] = (int8_t) (int) q;
            }
        } else {
            int8_t * dst = row + (int64_t) c * (CHUNK / 2);
            for (int i = 0; i < CHUNK; i += 2) {
                float q0 = sycl::rint(v[i] * inv), q1 = sycl::rint(v[i + 1] * inv);
                q0 = sycl::fmin(qmax, sycl::fmax(-qmax, q0));
                q1 = sycl::fmin(qmax, sycl::fmax(-qmax, q1));
                dst[i / 2] = (int8_t) (((uint32_t) (int) q0 & 0xFu) | (((uint32_t) (int) q1 & 0xFu) << 4));
            }
        }
    }
    if (lane == 0) {
        float * s = (float *) (row + (ABITS == 8 ? K : K / 2));
        *s = scale;
    }
}

// Work-group-per-row variant for the few rows a VLA quantizes (10-41 action
// tokens, a few hundred LLM tokens): one sub-group per row leaves the GPU nearly
// idle. Here RW sub-groups share a row; sub-group w takes chunks w, w + RW, ...
// and lane l owns the element pair (2l, 2l + 1) of each chunk. The 64-wide
// butterfly runs stage h = 1 inside the lane and the other stages across lanes,
// forming exactly the (a + b, a - b) pairs fwht_row forms, and the RMSNorm sum
// of squares keeps the reference's lane-partial order (sub-group 0 runs the
// per-row loop), so the codes are still bit-identical. The CUDA path's
// act_row_kernel, ported.
constexpr int RW = 8, RT = RW * SG, MAXC = 16;   // K <= RW * MAXC * 64

template <int ROT>
inline void rot_pair(const sycl::sub_group & sg, float & v0, float & v1, int lane, float inv_sqrt_bs) {
    if (ROT > 1) {
        { const float u = v0, w = v1; v0 = u + w; v1 = u - w; }   // h = 1: pair (2l, 2l+1)
        for (int h = 2; h < ROT; h <<= 1) {
            const int   half  = h >> 1;
            const float o0    = sycl::permute_group_by_xor(sg, v0, half);
            const float o1    = sycl::permute_group_by_xor(sg, v1, half);
            const bool  upper = (lane & half) != 0;                        // this lane holds j + h
            v0 = upper ? o0 - v0 : v0 + o0;
            v1 = upper ? o1 - v1 : v1 + o1;
        }
        v0 = v0 * inv_sqrt_bs;
        v1 = v1 * inv_sqrt_bs;
    }
}

template <int ABITS, int ROT>
void act_row_kernel(const ActParams a, const sycl::nd_item<1> & it, float * red, float * s_rstd) {
    const auto    sg      = it.get_sub_group();
    const int     w       = (int) sg.get_group_linear_id();
    const int     lane    = (int) sg.get_local_linear_id();
    const int64_t m       = (int64_t) it.get_group(0);
    const int64_t K       = a.K;
    const int     nchunks = (int) (K / CHUNK);
    const float * x       = a.x + m * a.x_stride;

    if (a.gamma) {
        if (w == 0) {
            float p = 0.0f;
            for (int c = lane; c < nchunks; c += SG)
                for (int i = 0; i < CHUNK; ++i) {
                    const float q = x[(int64_t) c * CHUNK + i];
                    p = p + (q * q);
                }
            for (int off = 16; off > 0; off >>= 1) p = p + sycl::permute_group_by_xor(sg, p, off);
            if (lane == 0) *s_rstd = 1.0f / sycl::sqrt((p / (float) K) + a.eps);
        }
        sycl::group_barrier(it.get_group());
    }
    const float rstd = a.gamma ? *s_rstd : 0.0f;

    float v0[MAXC], v1[MAXC];
    float mx = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = w + i * RW;
        if (c < nchunks) {
            const int64_t k0 = (int64_t) c * CHUNK + 2 * lane;
            float e0 = x[k0], e1 = x[k0 + 1];
            if (a.gamma) { e0 = (e0 * rstd) * a.gamma[k0]; e1 = (e1 * rstd) * a.gamma[k0 + 1]; }
            if (a.ascale && a.fold_before) { e0 = e0 / a.ascale[k0]; e1 = e1 / a.ascale[k0 + 1]; }
            rot_pair<ROT>(sg, e0, e1, lane, a.inv_sqrt_bs);
            if (a.ascale && !a.fold_before) { e0 = e0 / a.ascale[k0]; e1 = e1 / a.ascale[k0 + 1]; }
            v0[i] = e0; v1[i] = e1;
            mx = sycl::fmax(mx, sycl::fmax(sycl::fabs(e0), sycl::fabs(e1)));
        }
    }
    for (int off = 16; off > 0; off >>= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, off));
    if (lane == 0) red[w] = mx;
    sycl::group_barrier(it.get_group());
    mx = red[0];
    for (int j = 1; j < RW; ++j) mx = sycl::fmax(mx, red[j]);

    const float qmax  = ABITS == 4 ? 7.0f : 127.0f;
    float       scale = (a.clip * mx) / qmax;
    if (scale < 1e-12f) scale = 1e-12f;
    const float inv = 1.0f / scale;
    int8_t * row = a.blob + m * a.row_bytes;
    #pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        const int c = w + i * RW;
        if (c < nchunks) {
            float q0 = sycl::rint(v0[i] * inv), q1 = sycl::rint(v1[i] * inv);
            q0 = sycl::fmin(qmax, sycl::fmax(-qmax, q0));
            q1 = sycl::fmin(qmax, sycl::fmax(-qmax, q1));
            if (ABITS == 8) {
                row[(int64_t) c * CHUNK + 2 * lane]     = (int8_t) (int) q0;
                row[(int64_t) c * CHUNK + 2 * lane + 1] = (int8_t) (int) q1;
            } else {
                row[(int64_t) c * (CHUNK / 2) + lane] =
                    (int8_t) (((uint32_t) (int) q0 & 0xFu) | (((uint32_t) (int) q1 & 0xFu) << 4));
            }
        }
    }
    if (it.get_local_linear_id() == 0) {
        float * sp = (float *) (row + (ABITS == 8 ? K : K / 2));
        *sp = scale;
    }
}

template <int ABITS, int ROT>
void launch_act_row(sycl::queue & q, const ActParams & a) {
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(RW + 1), h);
        h.parallel_for(sycl::nd_range<1>((size_t) a.M * RT, RT), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            float * r = red.template get_multi_ptr<sycl::access::decorated::no>().get();
            act_row_kernel<ABITS, ROT>(a, it, r, r + RW);
        });
    });
}

template <int ABITS, int ROT>
void launch_act_rot(sycl::queue & q, const ActParams & a) {
    if (a.K / CHUNK <= RW * MAXC) {
        launch_act_row<ABITS, ROT>(q, a);
        return;
    }
    const size_t groups = (size_t) ((a.M + ROWS - 1) / ROWS);
    q.parallel_for(sycl::nd_range<1>(groups * ROWS * SG, ROWS * SG),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] { act_kernel<ABITS, ROT>(a, it); });
}

template <int ABITS>
bool launch_act_bits(sycl::queue & q, const ActParams & a, int rot) {
    switch (rot) {
        case 64: launch_act_rot<ABITS, 64>(q, a); return true;
        case 32: launch_act_rot<ABITS, 32>(q, a); return true;
        case 16: launch_act_rot<ABITS, 16>(q, a); return true;
        case 8:  launch_act_rot<ABITS, 8 >(q, a); return true;
        case 4:  launch_act_rot<ABITS, 4 >(q, a); return true;
        case 2:  launch_act_rot<ABITS, 2 >(q, a); return true;
        case 1:
        case 0:  launch_act_rot<ABITS, 1 >(q, a); return true;
        default: return false;
    }
}

// Signed nibble k of a packed row (low nibble = even column).
inline int nib(const int8_t * row, int64_t k) {
    const int b = (uint8_t) row[k >> 1];
    const int v = (k & 1) ? (b >> 4) : (b & 0xF);
    return v >= 8 ? v - 16 : v;
}

// One work-item per output element, a TM x TN tile per work-group whose
// activation and weight rows are staged through local memory in K steps.
constexpr int TM = 16, TN = 16, TK = 64;

template <int WBITS, int ABITS>
void launch_gemm_bits(sycl::queue & q, const GemmParams & g, bool has_bias, bool has_res) {
    const size_t gm = (size_t) ((g.M + TM - 1) / TM) * TM, gn = (size_t) ((g.N + TN - 1) / TN) * TN;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int8_t, 2> As(sycl::range<2>(TM, TK), h);
        sycl::local_accessor<int8_t, 2> Ws(sycl::range<2>(TN, TK), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(gm, gn), sycl::range<2>(TM, TN)), [=](sycl::nd_item<2> it) {
            const int     lm = (int) it.get_local_id(0), ln = (int) it.get_local_id(1);
            const int64_t m  = (int64_t) it.get_global_id(0), n = (int64_t) it.get_global_id(1);
            const int64_t m0 = m - lm, n0 = n - ln;
            const int     tid = lm * TN + ln;
            int32_t acc = 0;
            for (int64_t k0 = 0; k0 < g.K; k0 += TK) {
                // Stage TM activation rows and TN weight rows of TK codes each, unpacked to int8.
                for (int e = tid; e < TM * TK; e += TM * TN) {
                    const int r = e / TK, kk = e % TK;
                    const int64_t mm = m0 + r;
                    int8_t v = 0;
                    if (mm < g.M) {
                        const int8_t * arow = g.blob + mm * g.row_bytes;
                        v = ABITS == 4 ? (int8_t) nib(arow, k0 + kk) : arow[k0 + kk];
                    }
                    As[r][kk] = v;
                }
                for (int e = tid; e < TN * TK; e += TM * TN) {
                    const int r = e / TK, kk = e % TK;
                    const int64_t nn = n0 + r;
                    int8_t v = 0;
                    if (nn < g.N) {
                        const int8_t * wrow = g.w + nn * (WBITS == 4 ? g.K / 2 : g.K);
                        v = WBITS == 4 ? (int8_t) nib(wrow, k0 + kk) : wrow[k0 + kk];
                    }
                    Ws[r][kk] = v;
                }
                sycl::group_barrier(it.get_group());
                for (int kk = 0; kk < TK; ++kk)
                    acc += (int32_t) As[lm][kk] * (int32_t) Ws[ln][kk];
                sycl::group_barrier(it.get_group());
            }
            if (m < g.M && n < g.N) {
                float xs;
                std::memcpy(&xs, g.blob + m * g.row_bytes + (ABITS == 4 ? g.K / 2 : g.K), sizeof(float));
                float v = ((float) acc * xs) * g.wscale[n];
                if (has_bias) v = v + g.bias[n];
                if (has_res)  v = v + g.res[m * g.N + n];
                g.y[m * g.N + n] = v;
            }
        });
    });
}

// XMX path: int8 joint_matrix (8x32 A, 32x16 B, int32 8x16 accumulator) on
// 16-wide sub-groups. A work-group of XS sub-groups computes a XM x (XS*16)
// tile; per XK-wide K step it stages the activation tile row-major and the
// weight tile in the VNNI layout the B operand wants ([k/4][n*4 + k%4]),
// unpacking INT4 nibbles to int8 on the way, so every width runs on the int8
// engines. The accumulator is int32 and the epilogue the same as above, so the
// output is still bit-identical to the reference.
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int XM = 32, XS = 4, XN = XS * 16, XK = 32, XT = XS * 16;

template <int WBITS, int ABITS>
void launch_gemm_xmx(sycl::queue & q, const GemmParams & g, bool has_bias, bool has_res) {
    const size_t groups_m = (size_t) ((g.M + XM - 1) / XM), groups_n = (size_t) (g.N / XN);
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int8_t, 1>  As(sycl::range<1>(XM * XK), h);
        sycl::local_accessor<int8_t, 1>  Bs(sycl::range<1>(XK * XN), h);
        sycl::local_accessor<int32_t, 1> Cs(sycl::range<1>(XM * XN), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(groups_m, groups_n * XT), sycl::range<2>(1, XT)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto    sg   = it.get_sub_group();
            const int     sgid = (int) sg.get_group_linear_id();
            const int     tid  = (int) it.get_local_id(1);
            const int64_t m0   = (int64_t) it.get_group(0) * XM;
            const int64_t n0   = (int64_t) it.get_group(1) * XN;
            auto a_ptr = As.template get_multi_ptr<sycl::access::decorated::no>();
            auto b_ptr = Bs.template get_multi_ptr<sycl::access::decorated::no>();
            auto c_ptr = Cs.template get_multi_ptr<sycl::access::decorated::no>();

            jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> acc[XM / 8];
            #pragma unroll
            for (int i = 0; i < XM / 8; ++i) jm::joint_matrix_fill(sg, acc[i], 0);

            for (int64_t k0 = 0; k0 < g.K; k0 += XK) {
                for (int e = tid; e < XM * XK; e += XT) {
                    const int r = e / XK, kk = e % XK;
                    const int64_t mm = m0 + r;
                    int8_t v = 0;
                    if (mm < g.M) {
                        const int8_t * arow = g.blob + mm * g.row_bytes;
                        v = ABITS == 4 ? (int8_t) nib(arow, k0 + kk) : arow[k0 + kk];
                    }
                    As[e] = v;
                }
                for (int e = tid; e < XN * XK; e += XT) {
                    const int nl = e / XK, kk = e % XK;   // weight row (output column) nl, element kk
                    const int8_t * wrow = g.w + (n0 + nl) * (WBITS == 4 ? g.K / 2 : g.K);
                    const int8_t v = WBITS == 4 ? (int8_t) nib(wrow, k0 + kk) : wrow[k0 + kk];
                    Bs[(kk / 4) * (XN * 4) + nl * 4 + (kk % 4)] = v;
                }
                sycl::group_barrier(it.get_group());
                jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, XK, 16, jm::layout::ext_intel_packed> mb;
                jm::joint_matrix_load(sg, mb, b_ptr + sgid * 16 * 4, XN * 4);
                #pragma unroll
                for (int i = 0; i < XM / 8; ++i) {
                    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, XK, jm::layout::row_major> ma;
                    jm::joint_matrix_load(sg, ma, a_ptr + i * 8 * XK, XK);
                    jm::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                }
                sycl::group_barrier(it.get_group());
            }
            #pragma unroll
            for (int i = 0; i < XM / 8; ++i)
                jm::joint_matrix_store(sg, acc[i], c_ptr + i * 8 * XN + sgid * 16, XN, jm::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int e = tid; e < XM * XN; e += XT) {
                const int r = e / XN, c = e % XN;
                const int64_t m = m0 + r, n = n0 + c;
                if (m >= g.M) continue;
                float xs;
                std::memcpy(&xs, g.blob + m * g.row_bytes + (ABITS == 4 ? g.K / 2 : g.K), sizeof(float));
                float v = ((float) Cs[e] * xs) * g.wscale[n];
                if (has_bias) v = v + g.bias[n];
                if (has_res)  v = v + g.res[m * g.N + n];
                g.y[m * g.N + n] = v;
            }
        });
    });
}

// Small-M path (the action expert's 10-41 tokens): GEMV-style. A 16-wide
// sub-group owns one output column; its lanes stream the weight row 8 codes at a
// time and accumulate int32 dot products against every token row, which are
// then combined with an exact integer sub-group reduction. Weights are read once
// per column, coalesced across the lanes, which is what a weight-streaming
// shape needs; the epilogue is the reference's.
constexpr int SM_MAX = 32, SM_SG = 16, SM_COLS = 8;   // M <= SM_MAX, SM_COLS columns per work-group

template <int BITS>
inline void load8(const int8_t * p, int64_t k, int8_t * v) {   // codes k .. k+7 (k % 8 == 0)
    if (BITS == 8) {
        for (int j = 0; j < 8; ++j) v[j] = p[k + j];
    } else {
        const uint8_t * b = (const uint8_t *) p + (k >> 1);
        for (int j = 0; j < 4; ++j) {
            const int lo = b[j] & 0xF, hi = b[j] >> 4;
            v[2 * j]     = (int8_t) (lo >= 8 ? lo - 16 : lo);
            v[2 * j + 1] = (int8_t) (hi >= 8 ? hi - 16 : hi);
        }
    }
}

template <int WBITS, int ABITS>
void launch_gemm_small(sycl::queue & q, const GemmParams & g, bool has_bias, bool has_res) {
    const size_t groups = (size_t) ((g.N + SM_COLS - 1) / SM_COLS);
    q.parallel_for(sycl::nd_range<1>(groups * SM_COLS * SM_SG, SM_COLS * SM_SG),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SM_SG)]] {
        const auto    sg   = it.get_sub_group();
        const int     lane = (int) sg.get_local_linear_id();
        const int64_t n    = (int64_t) it.get_group(0) * SM_COLS + (int64_t) sg.get_group_linear_id();
        if (n >= g.N) return;
        const int8_t * wrow = g.w + n * (WBITS == 4 ? g.K / 2 : g.K);
        int32_t acc[SM_MAX];
        for (int m = 0; m < SM_MAX; ++m) acc[m] = 0;
        for (int64_t k = (int64_t) lane * 8; k < g.K; k += SM_SG * 8) {
            int8_t wv[8];
            load8<WBITS>(wrow, k, wv);
            for (int m = 0; m < SM_MAX; ++m) {
                if (m >= g.M) break;
                int8_t av[8];
                load8<ABITS>(g.blob + (int64_t) m * g.row_bytes, k, av);
                int32_t d = 0;
                for (int j = 0; j < 8; ++j) d += (int32_t) wv[j] * (int32_t) av[j];
                acc[m] += d;
            }
        }
        for (int m = 0; m < SM_MAX; ++m) {
            if (m >= g.M) break;
            const int32_t total = sycl::reduce_over_group(sg, acc[m], sycl::plus<int32_t>());
            if (lane == 0) {
                float xs;
                std::memcpy(&xs, g.blob + (int64_t) m * g.row_bytes + (ABITS == 4 ? g.K / 2 : g.K), sizeof(float));
                float v = ((float) total * xs) * g.wscale[n];
                if (has_bias) v = v + g.bias[n];
                if (has_res)  v = v + g.res[(int64_t) m * g.N + n];
                g.y[(int64_t) m * g.N + n] = v;
            }
        }
    });
}

// Whether the device has the int8 8x16x32 XMX combination the kernel above uses.
bool use_xmx(const sycl::queue & q) {
    static const bool ok = [&] {
        namespace syclex = sycl::ext::oneapi::experimental;
        try {
            for (const auto & c : q.get_device().get_info<syclex::info::device::matrix_combinations>())
                if (c.atype == syclex::matrix::matrix_type::sint8 && c.btype == syclex::matrix::matrix_type::sint8 &&
                    c.ctype == syclex::matrix::matrix_type::sint32 && c.nsize == 16 && c.ksize == 32 &&
                    (c.msize == 8 || c.max_msize >= 8))
                    return true;
        } catch (...) {
        }
        return false;
    }();
    return ok;
}

bool run_act(sycl::queue & q, ggml_tensor * dst, const FqActSpec & s) {
    const ggml_tensor * x = dst->src[0];
    int si = 1;
    const ggml_tensor * gamma  = s.has_gamma  ? dst->src[si++] : nullptr;
    const ggml_tensor * ascale = s.has_ascale ? dst->src[si++] : nullptr;
    if (!x || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || s.K % CHUNK != 0 || s.rot_block > CHUNK)
        return false;
    ActParams a{};
    a.x           = (const float *) x->data;
    a.x_stride    = (int64_t) (x->nb[1] / sizeof(float));
    a.ascale      = ascale ? (const float *) ascale->data : nullptr;
    a.gamma       = gamma ? (const float *) gamma->data : nullptr;
    a.blob        = (int8_t *) dst->data;
    a.row_bytes   = dst->ne[0];
    a.M           = dst->ne[1] * dst->ne[2] * dst->ne[3];
    a.K           = s.K;
    a.fold_before = s.fold_before;
    a.clip        = s.clip;
    a.eps         = s.eps;
    a.inv_sqrt_bs = s.rot_block > 1 ? fqref::inv_sqrt_block(s.rot_block) : 1.0f;
    if (a.M == 0) return true;
    return s.abits == 8 ? launch_act_bits<8>(q, a, s.rot_block)
         : s.abits == 4 ? launch_act_bits<4>(q, a, s.rot_block)
         : false;
}

#ifdef VLA_FQ_SYCL_DNNL
// Prefill shapes on oneDNN's int8 matmul (XMX): INT8 (or unpacked INT4) codes
// in, int32 sums out into a scratch buffer, then the reference epilogue. Integer
// sums are exact in any order, so the output is the same as on the other paths;
// test_foldquant_sycl_op checks that at full K.
struct DnnlState {
    dnnl::engine eng;
    dnnl::stream strm;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int>, dnnl::matmul> prims;   // M, N, K, lda, s4 weights
    int8_t *  a8 = nullptr; size_t a8_cap = 0;    // A4 activations unpacked to int8
    int8_t *  w8 = nullptr; size_t w8_cap = 0;    // W4 weights unpacked to int8, when oneDNN takes no s4
    int32_t * acc = nullptr; size_t acc_cap = 0;  // the int32 sums
    bool s4_ok = true;
};

template <typename T>
T * grow(sycl::queue & q, T *& p, size_t & cap, size_t n) {
    if (n > cap) {
        if (p) { q.wait(); sycl::free(p, q); }
        p   = sycl::malloc_device<T>(n, q);
        cap = n;
    }
    return p;
}

DnnlState & dnnl_state(sycl::queue & q) {
    static std::unordered_map<sycl::queue *, DnnlState> states;
    auto it = states.find(&q);
    if (it == states.end()) {
        DnnlState st;
        st.eng  = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        st.strm = dnnl::sycl_interop::make_stream(st.eng, q);
        it = states.emplace(&q, std::move(st)).first;
    }
    return it->second;
}

// Unpacks `rows` nibble-packed rows (stride `src_stride` bytes) of K codes to int8.
void unpack_rows(sycl::queue & q, const int8_t * src, int64_t src_stride, int8_t * dst, int64_t rows, int64_t K) {
    const int64_t half = K / 2;
    q.parallel_for(sycl::range<2>((size_t) rows, (size_t) half), [=](sycl::item<2> it) {
        const int64_t r = (int64_t) it.get_id(0), j = (int64_t) it.get_id(1);
        const int b = (uint8_t) src[r * src_stride + j];
        const int lo = b & 0xF, hi = b >> 4;
        dst[r * K + 2 * j]     = (int8_t) (lo >= 8 ? lo - 16 : lo);
        dst[r * K + 2 * j + 1] = (int8_t) (hi >= 8 ? hi - 16 : hi);
    });
}

template <int ABITS>
void launch_epilogue(sycl::queue & q, const GemmParams & g, const int32_t * acc, bool has_bias, bool has_res) {
    q.parallel_for(sycl::range<2>((size_t) g.M, (size_t) g.N), [=](sycl::item<2> it) {
        const int64_t m = (int64_t) it.get_id(0), n = (int64_t) it.get_id(1);
        float xs;
        std::memcpy(&xs, g.blob + m * g.row_bytes + (ABITS == 4 ? g.K / 2 : g.K), sizeof(float));
        float v = ((float) acc[m * g.N + n] * xs) * g.wscale[n];
        if (has_bias) v = v + g.bias[n];
        if (has_res)  v = v + g.res[m * g.N + n];
        g.y[m * g.N + n] = v;
    });
}

bool gemm_dnnl_impl(sycl::queue & q, const GemmParams & g, int wbits, int abits, bool has_bias, bool has_res);

// oneDNN on a GPU, unless it cannot drive it (no engine, no int8 matmul): then
// this file's kernels, for good on that queue. GPUs only: on a CPU device
// oneDNN's int8 matmul is not exact everywhere (on CPUs without VNNI it sums
// pairs in saturating 16-bit lanes, vpmaddubsw), and vla.cpp only runs ggml-sycl
// on GPUs anyway; the CPU device is test_foldquant_sycl_op's stand-in.
bool gemm_dnnl(sycl::queue & q, const GemmParams & g, int wbits, int abits, bool has_bias, bool has_res) {
    static std::unordered_map<sycl::queue *, bool> unusable;
    if (unusable.find(&q) == unusable.end()) unusable[&q] = !q.get_device().is_gpu();
    if (unusable[&q]) return false;
    try {
        if (gemm_dnnl_impl(q, g, wbits, abits, has_bias, has_res)) return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FoldQuant SYCL: oneDNN unavailable on this device (%s); using the native GEMMs\n", e.what());
    }
    unusable[&q] = true;
    return false;
}

bool gemm_dnnl_impl(sycl::queue & q, const GemmParams & g, int wbits, int abits, bool has_bias, bool has_res) {
    using dt  = dnnl::memory::data_type;
    using md  = dnnl::memory::desc;
    DnnlState & st = dnnl_state(q);
    const int64_t M = g.M, N = g.N, K = g.K;

    const int8_t * a   = g.blob;
    int64_t        lda = g.row_bytes;
    if (abits == 4) {
        unpack_rows(q, g.blob, g.row_bytes, grow(q, st.a8, st.a8_cap, (size_t) (M * K)), M, K);
        a   = st.a8;
        lda = K;
    }
    bool s4 = wbits == 4 && st.s4_ok;
    auto make = [&](bool w4) {
        const md src_md({ M, K }, dt::s8, { lda, 1 });
        const md wei_md({ K, N }, w4 ? dt::s4 : dt::s8, { 1, K });
        const md dst_md({ M, N }, dt::s32, { N, 1 });
        return dnnl::matmul(dnnl::matmul::primitive_desc(st.eng, src_md, wei_md, dst_md));
    };
    auto key = std::make_tuple(M, N, K, lda, s4 ? 1 : 0);
    auto it  = st.prims.find(key);
    if (it == st.prims.end()) {
        try {
            it = st.prims.emplace(key, make(s4)).first;
        } catch (const dnnl::error &) {
            if (!s4) return false;
            st.s4_ok = false;   // no s4 x s8 matmul here: unpack the weights instead
            std::fprintf(stderr, "FoldQuant SYCL: oneDNN has no s8 x s4 matmul here; INT4 weights are unpacked per call\n");
            s4  = false;
            key = std::make_tuple(M, N, K, lda, 0);
            it  = st.prims.find(key);
            if (it == st.prims.end()) it = st.prims.emplace(key, make(false)).first;
        }
    }
    const int8_t * w = g.w;
    if (wbits == 4 && !s4) {
        unpack_rows(q, g.w, K / 2, grow(q, st.w8, st.w8_cap, (size_t) (N * K)), N, K);
        w = st.w8;
    }
    int32_t * acc = grow(q, st.acc, st.acc_cap, (size_t) (M * N));
    using dnnl::sycl_interop::make_memory;
    using dnnl::sycl_interop::memory_kind;
    const md src_md({ M, K }, dt::s8, { lda, 1 });
    const md wei_md({ K, N }, s4 ? dt::s4 : dt::s8, { 1, K });
    const md dst_md({ M, N }, dt::s32, { N, 1 });
    auto src_m = make_memory(src_md, st.eng, memory_kind::usm, const_cast<int8_t *>(a));
    auto wei_m = make_memory(wei_md, st.eng, memory_kind::usm, const_cast<int8_t *>(w));
    auto dst_m = make_memory(dst_md, st.eng, memory_kind::usm, acc);
    it->second.execute(st.strm, { { DNNL_ARG_SRC, src_m }, { DNNL_ARG_WEIGHTS, wei_m }, { DNNL_ARG_DST, dst_m } });
    if (abits == 4) launch_epilogue<4>(q, g, acc, has_bias, has_res);
    else            launch_epilogue<8>(q, g, acc, has_bias, has_res);
    return true;
}
#endif

bool run_gemm(sycl::queue & q, ggml_tensor * dst, const FqGemmSpec & s, int abits) {
    const ggml_tensor * w = dst->src[0], * blob = dst->src[1], * ws = dst->src[2];
    const ggml_tensor * bias = dst->src[3], * res = dst->src[4];
    if (!w || !blob || !ws || s.heads != 0) return false;
    GemmParams g{};
    g.w         = (const int8_t *) w->data;
    g.blob      = (const int8_t *) blob->data;
    g.wscale    = (const float *) ws->data;
    g.bias      = bias ? (const float *) bias->data : nullptr;
    g.res       = res ? (const float *) res->data : nullptr;
    g.y         = (float *) dst->data;
    g.M         = dst->ne[1];
    g.N         = s.N;
    g.K         = s.K;
    g.row_bytes = blob->ne[0];
    if (g.M == 0) return true;
    // oneDNN when built with it; VLA_FQ_SYCL_GEMM=native picks this file's kernels
    // (the GEMV for M <= SM_MAX, else XMX) and =simple the plain tiled one. Read
    // per call so test_foldquant_sycl_op can cover every path.
    const char * mode   = std::getenv("VLA_FQ_SYCL_GEMM");
    const bool   simple = mode && std::strcmp(mode, "simple") == 0;
    const bool   native = simple || (mode && std::strcmp(mode, "native") == 0);
#ifdef VLA_FQ_SYCL_DNNL
    if (!native && gemm_dnnl(q, g, s.wbits, abits, bias, res))
        return true;
#endif
    if (!simple && g.M <= SM_MAX && g.K % (SM_SG * 8) == 0) {
        if (s.wbits == 8 && abits == 8) launch_gemm_small<8, 8>(q, g, bias, res);
        else if (s.wbits == 8 && abits == 4) launch_gemm_small<8, 4>(q, g, bias, res);
        else if (s.wbits == 4 && abits == 8) launch_gemm_small<4, 8>(q, g, bias, res);
        else if (s.wbits == 4 && abits == 4) launch_gemm_small<4, 4>(q, g, bias, res);
        else return false;
        return true;
    }
    if (!simple && use_xmx(q) && g.N % XN == 0 && g.K % XK == 0) {
        if (s.wbits == 8 && abits == 8) launch_gemm_xmx<8, 8>(q, g, bias, res);
        else if (s.wbits == 8 && abits == 4) launch_gemm_xmx<8, 4>(q, g, bias, res);
        else if (s.wbits == 4 && abits == 8) launch_gemm_xmx<4, 8>(q, g, bias, res);
        else if (s.wbits == 4 && abits == 4) launch_gemm_xmx<4, 4>(q, g, bias, res);
        else return false;
        return true;
    }
    if (s.wbits == 8 && abits == 8) launch_gemm_bits<8, 8>(q, g, bias, res);
    else if (s.wbits == 8 && abits == 4) launch_gemm_bits<8, 4>(q, g, bias, res);
    else if (s.wbits == 4 && abits == 8) launch_gemm_bits<4, 8>(q, g, bias, res);
    else if (s.wbits == 4 && abits == 4) launch_gemm_bits<4, 4>(q, g, bias, res);
    else return false;
    return true;
}

// The gemm node's activation width is the one its blob was quantized at: the
// blob row holds K (A8) or K/2 (A4) code bytes then the scale and its padding.
int blob_abits(const ggml_tensor * blob, int64_t K) {
    return blob->ne[0] == fq_act_row_bytes(K, 4) ? 4 : 8;
}

bool dispatch(ggml_tensor * dst, void * queue) {
    const void * ud = fq_userdata(dst);
    const uint32_t m = fq_magic(ud);
    if (m != FQ_ACT_MAGIC && m != FQ_GEMM_MAGIC) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    if (m == FQ_ACT_MAGIC)
        return run_act(q, dst, *static_cast<const FqActSpec *>(ud));
    const FqGemmSpec & s = *static_cast<const FqGemmSpec *>(ud);
    return run_gemm(q, dst, s, blob_abits(dst->src[1], s.K));
}

bool supports(const ggml_tensor * op) {
    const uint32_t m = fq_magic(fq_userdata(op));
    if (m == FQ_ACT_MAGIC) return true;
    if (m == FQ_GEMM_MAGIC) return static_cast<const FqGemmSpec *>(fq_userdata(op))->heads == 0;
    return false;
}

}  // namespace

void sycl_register_foldquant_ops() {
    ggml_sycl_ext_forward  = dispatch;
    ggml_sycl_ext_supports = supports;
}

}  // namespace vla
