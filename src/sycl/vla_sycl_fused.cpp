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

// SYCL kernels for the fused elementwise nodes in layers/fused_ew.h, behind the
// extension hook (sycl/vla_sycl_ext.cpp). One work-group per row: the row is
// read once, reduced in registers for the norm, and written once in the type
// the next GEMM wants.

#include "layers/fused_ew.h"
#include "layers/qlinear.h"
#include "sycl/vla_sycl_ext.h"
#include "sycl/vla_sycl_ops.h"
#include "env_flag.h"

#include "ggml.h"

#include <sycl/sycl.hpp>

#include <cstring>

namespace vla {

bool sycl_fused_linear(ggml_tensor * dst, const void * spec, sycl::queue & q);  // vla_sycl_bf16.cpp
bool sycl_qlinear(ggml_tensor * dst, const void * spec, sycl::queue & q);       // vla_sycl_bf16.cpp

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int WG = 256;

const void * custom_userdata(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CUSTOM) return nullptr;
    struct custom_params { void * fun; int n_tasks; void * userdata; };  // ggml_custom_op_params
    custom_params p;
    std::memcpy(&p, op->op_params, sizeof(p));
    return p.userdata;
}

uint32_t magic_of(const ggml_tensor * op) {
    const void * ud = custom_userdata(op);
    uint32_t m = 0;
    if (ud) std::memcpy(&m, ud, sizeof(m));
    return m;
}

// Rows of h elements, laid out [h, ne1, ne2, ne3] with row stride nb[1] and the
// higher dims packed, as an ADD/MUL output (or a row-contiguous view) is.
bool rows_packed(const ggml_tensor * t) {
    return t->nb[0] == ggml_type_size(t->type) && t->nb[2] == t->nb[1] * (size_t) t->ne[1] &&
           t->nb[3] == t->nb[2] * (size_t) t->ne[2];
}

template <typename T> inline float ld(const T * p, int64_t i) { return (float) p[i]; }
template <typename T> inline void st(T * p, int64_t i, float v) { p[i] = T(v); }

// A row of K values into a q8 blob (layers/fused_ew.h): max |v| over the
// work-group, then the codes. val(i) is evaluated twice, identically.
template <typename F>
inline void q8_row(const sycl::nd_item<1> & it, int64_t K, int8_t * codes, float * scale, F val) {
    const int tid  = (int) it.get_local_id(0);
    float     amax = 0.0f;
    for (int64_t i = tid; i < K; i += WG) amax = sycl::fmax(amax, sycl::fabs(val(i)));
    amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
    const float sc  = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / sc;
    for (int64_t i = tid; i < K; i += WG)
        codes[i] = (int8_t) sycl::rint(sycl::fmin(127.0f, sycl::fmax(-127.0f, val(i) * inv)));
    if (tid == 0) *scale = sc;
}

// q8_row over chunks of 8 consecutive elements (K % 8 == 0): val8(c, v) fills
// v[0..7] with elements 8c..8c+7, so rows are read with 16- or 32-byte loads
// and written 8 codes at a time.
template <typename F>
inline void q8_row8(const sycl::nd_item<1> & it, int64_t K, int8_t * codes, float * scale, F val8) {
    const int     tid  = (int) it.get_local_id(0);
    const int64_t nc   = K / 8;
    float         amax = 0.0f;
    float         v[8];
    for (int64_t c = tid; c < nc; c += WG) {
        val8(c, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) amax = sycl::fmax(amax, sycl::fabs(v[j]));
    }
    amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
    const float sc  = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / sc;
    for (int64_t c = tid; c < nc; c += WG) {
        val8(c, v);
        sycl::vec<int8_t, 8> o;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = (int8_t) sycl::rint(sycl::fmin(127.0f, sycl::fmax(-127.0f, v[j] * inv)));
        reinterpret_cast<sycl::vec<int8_t, 8> *>(codes)[c] = o;
    }
    if (tid == 0) *scale = sc;
}

template <typename T> inline sycl::vec<float, 8> load8(const T * p, int64_t c) {
    const sycl::vec<T, 8> t = reinterpret_cast<const sycl::vec<T, 8> *>(p)[c];
    sycl::vec<float, 8>   f;
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = (float) t[j];
    return f;
}

// Rows of K elements at row stride `stride` bytes, starting at base: 8-element
// chunks are 8 * sizeof(T)-aligned.
inline bool rows_vec8(const void * base, size_t stride, int64_t K, size_t es) {
    return K % 8 == 0 && (uintptr_t) base % (8 * es) == 0 && stride % (8 * es) == 0;
}

struct k_adarms_q8;

void run_adarms_q8(sycl::queue & q, const ggml_tensor * dst, float eps) {
    const ggml_tensor * x = dst->src[0];
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    const char *  xb = (const char *) x->data;
    const float * mod = (const float *) dst->src[1]->data;
    int8_t *      codes = (int8_t *) dst->data;
    float *       scales = (float *) (codes + rows * h);
    const size_t  xs = x->nb[1];
    const bool    v8 = rows_vec8(xb, xs, h, 4) && (uintptr_t) mod % 32 == 0 && h % 8 == 0;
    q.parallel_for<k_adarms_q8>(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
                                [=](sycl::nd_item<1> it) {
        const int64_t r   = it.get_group(0);
        const int     tid = (int) it.get_local_id(0);
        const float * xr  = (const float *) (xb + r * xs);
        float ss = 0.0f;
        for (int64_t i = tid; i < h; i += WG) ss += xr[i] * xr[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float inv = 1.0f / sycl::sqrt(ss / (float) h + eps);
        if (v8) {
            q8_row8(it, h, codes + r * h, scales + r, [&](int64_t c, float * v) {
                const sycl::vec<float, 8> xv = load8(xr, c), sv = load8(mod, c), bv = load8(mod + h, c);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float n = xv[j] * inv;
                    v[j] = (n + n * sv[j]) + bv[j];
                }
            });
        } else {
            q8_row(it, h, codes + r * h, scales + r, [&](int64_t i) {
                const float n = xr[i] * inv;
                return (n + n * mod[i]) + mod[h + i];
            });
        }
    });
}

template <typename TI>
struct k_rms_q8;

template <typename TI>
void run_rms_q8(sycl::queue & q, const ggml_tensor * dst, float eps) {
    const ggml_tensor * x = dst->src[0];
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    const char *  xb = (const char *) x->data;
    const float * w  = (const float *) dst->src[1]->data;
    int8_t *      codes = (int8_t *) dst->data;
    float *       scales = (float *) (codes + rows * h);
    const size_t  xs = x->nb[1];
    const bool    v8 = rows_vec8(xb, xs, h, sizeof(TI)) && (uintptr_t) w % 32 == 0;
    q.parallel_for<k_rms_q8<TI>>(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
                                 [=](sycl::nd_item<1> it) {
        const int64_t r   = it.get_group(0);
        const int     tid = (int) it.get_local_id(0);
        const TI *    xr  = (const TI *) (xb + r * xs);
        float ss = 0.0f;
        if (v8) {
            for (int64_t c = tid; c < h / 8; c += WG) {
                const sycl::vec<float, 8> xv = load8(xr, c);
#pragma unroll
                for (int j = 0; j < 8; ++j) ss += xv[j] * xv[j];
            }
        } else {
            for (int64_t i = tid; i < h; i += WG) {
                const float v = (float) xr[i];
                ss += v * v;
            }
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float inv = sycl::rsqrt(ss / (float) h + eps);
        if (v8) {
            q8_row8(it, h, codes + r * h, scales + r, [&](int64_t c, float * v) {
                const sycl::vec<float, 8> xv = load8(xr, c), wv = load8(w, c);
#pragma unroll
                for (int j = 0; j < 8; ++j) v[j] = xv[j] * inv * wv[j];
            });
        } else {
            q8_row(it, h, codes + r * h, scales + r, [&](int64_t i) { return (float) xr[i] * inv * w[i]; });
        }
    });
}

template <typename TI, typename TO>
struct k_rms;

// fused_rms: RMSNorm with its weight into rows of TO, one work-group per row.
template <typename TI, typename TO>
void run_rms(sycl::queue & q, const ggml_tensor * dst, float eps) {
    const ggml_tensor * x = dst->src[0];
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    const char *  xb = (const char *) x->data;
    const float * w  = (const float *) dst->src[1]->data;
    char *        ob = (char *) dst->data;
    const size_t  xs = x->nb[1], os = dst->nb[1];
    q.parallel_for<k_rms<TI, TO>>(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
                                  [=](sycl::nd_item<1> it) {
        const int64_t r   = it.get_group(0);
        const int     tid = (int) it.get_local_id(0);
        const TI *    xr  = (const TI *) (xb + r * xs);
        float ss = 0.0f;
        for (int64_t i = tid; i < h; i += WG) {
            const float v = (float) xr[i];
            ss += v * v;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) h + eps);
        TO * orow = (TO *) (ob + r * os);
        for (int64_t i = tid; i < h; i += WG) orow[i] = TO((float) xr[i] * inv * w[i]);
    });
}

template <typename TO>
struct k_adarms;

template <typename TO>
void run_adarms(sycl::queue & q, const ggml_tensor * dst, float eps) {
    const ggml_tensor * x = dst->src[0];
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    const char *  xb = (const char *) x->data;
    const float * mod = (const float *) dst->src[1]->data;
    char *        ob = (char *) dst->data;
    const size_t  xs = x->nb[1], os = dst->nb[1];
    q.parallel_for<k_adarms<TO>>(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
                                 [=](sycl::nd_item<1> it) {
        const int64_t r   = it.get_group(0);
        const int     tid = (int) it.get_local_id(0);
        const float * xr  = (const float *) (xb + r * xs);
        float ss = 0.0f;
        for (int64_t i = tid; i < h; i += WG) ss += xr[i] * xr[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float inv = 1.0f / sycl::sqrt(ss / (float) h + eps);
        TO * orow = (TO *) (ob + r * os);
        for (int64_t i = tid; i < h; i += WG) {
            const float n = xr[i] * inv;
            st(orow, i, (n + n * mod[i]) + mod[h + i]);
        }
    });
}

template <typename TI>
struct k_gateres;

template <typename TI>
void run_gateres(sycl::queue & q, const ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * o = dst->src[1];
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    const char *  xb = (const char *) x->data;
    const char *  obsrc = (const char *) o->data;
    const float * gate = (const float *) dst->src[2]->data + 2 * h;
    char *        db = (char *) dst->data;
    const size_t  xs = x->nb[1], os = o->nb[1], ds = dst->nb[1];
    q.parallel_for<k_gateres<TI>>(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
                                  [=](sycl::nd_item<1> it) {
        const int64_t r   = it.get_group(0);
        const int     tid = (int) it.get_local_id(0);
        const float * xr  = (const float *) (xb + r * xs);
        const TI *    orow = (const TI *) (obsrc + r * os);
        float *       dr  = (float *) (db + r * ds);
        for (int64_t i = tid; i < h; i += WG) dr[i] = xr[i] + ld(orow, i) * gate[i];
    });
}

template <typename T>
struct k_geglu;

// gelu_tanh(x) = 0.5x(1 + tanh(u)) = x * sigmoid(2u): one exp instead of a
// tanh, the same function.
inline float gelu_tanh_fast(float x) {
    const float u2 = 1.5957691216057308f * x * (1.0f + 0.044715f * x * x);  // 2u
    return x / (1.0f + sycl::native::exp(-u2));
}

// The gate activation of a FusedGegluSpec node: SiLU or tanh GELU.
inline float glu_act(bool silu, float x) {
    return silu ? x / (1.0f + sycl::native::exp(-x)) : gelu_tanh_fast(x);
}

bool glu_silu(const ggml_tensor * dst) {
    return static_cast<const FusedGegluSpec *>(custom_userdata(dst))->silu != 0;
}

// One work-item per 4 consecutive elements of a row (n % 4 == 0, which every
// FFN width here is): 8-byte BF16 loads and stores instead of 2-byte ones.
template <typename T>
void run_geglu(sycl::queue & q, const ggml_tensor * dst) {
    const ggml_tensor * g = dst->src[0];
    const ggml_tensor * u = dst->src[1];
    const int64_t n = g->ne[0], rows = ggml_nrows(g), n4 = n / 4;
    const char * gb = (const char *) g->data;
    const char * ub = (const char *) u->data;
    char *       db = (char *) dst->data;
    const size_t gs = g->nb[1], us = u->nb[1], ds = dst->nb[1];
    const bool   silu = glu_silu(dst);
    using V = sycl::vec<T, 4>;
    q.parallel_for<k_geglu<T>>(sycl::range<2>((size_t) rows, (size_t) n4), [=](sycl::item<2> it) {
        const int64_t r = (int64_t) it[0], c = (int64_t) it[1];
        const V gv = reinterpret_cast<const V *>(gb + r * gs)[c];
        const V uv = reinterpret_cast<const V *>(ub + r * us)[c];
        V o;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = T(glu_act(silu, (float) gv[j]) * (float) uv[j]);
        reinterpret_cast<V *>(db + r * ds)[c] = o;
    });
}

template <typename TI, typename TO>
struct k_lnorm;

// LayerNorm + affine, one work-group per row: the row is read three times from
// cache (mean, variance, the write) and written once, in the next GEMM's type.
template <typename TI, typename TO>
struct k_lnorm8;

template <typename TI, typename TO>
void run_lnorm(sycl::queue & q, const ggml_tensor * dst, float eps) {
    const ggml_tensor * x = dst->src[0];
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    const char *  xb = (const char *) x->data;
    // no affine (both null): the scalar kernel, with w = 1 and b = 0
    const float * w  = dst->src[1] ? (const float *) dst->src[1]->data : nullptr;
    const float * b  = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;
    char *        ob = (char *) dst->data;
    const size_t  xs = x->nb[1], os = dst->nb[1];
    const sycl::nd_range<1> nr(sycl::range<1>(rows * WG), sycl::range<1>(WG));
    if (w && rows_vec8(xb, xs, h, sizeof(TI)) && rows_vec8(ob, os, h, sizeof(TO)) && (uintptr_t) w % 32 == 0 &&
        (uintptr_t) b % 32 == 0) {
        q.parallel_for<k_lnorm8<TI, TO>>(nr, [=](sycl::nd_item<1> it) {
            const int64_t r   = it.get_group(0);
            const int     tid = (int) it.get_local_id(0);
            const TI *    xr  = (const TI *) (xb + r * xs);
            const int64_t nc  = h / 8;
            float s = 0.0f;
            for (int64_t c = tid; c < nc; c += WG) {
                const sycl::vec<float, 8> v = load8(xr, c);
#pragma unroll
                for (int j = 0; j < 8; ++j) s += v[j];
            }
            const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) h;
            float ss = 0.0f;
            for (int64_t c = tid; c < nc; c += WG) {
                const sycl::vec<float, 8> v = load8(xr, c);
#pragma unroll
                for (int j = 0; j < 8; ++j) ss += (v[j] - mean) * (v[j] - mean);
            }
            const float inv =
                sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) h + eps);
            auto * orow = (sycl::vec<TO, 8> *) (ob + r * os);
            for (int64_t c = tid; c < nc; c += WG) {
                const sycl::vec<float, 8> v = load8(xr, c), wv = load8(w, c), bv = load8(b, c);
                sycl::vec<TO, 8> o;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = TO((v[j] - mean) * inv * wv[j] + bv[j]);
                orow[c] = o;
            }
        });
        return;
    }
    q.parallel_for<k_lnorm<TI, TO>>(nr, [=](sycl::nd_item<1> it) {
        const int64_t r   = it.get_group(0);
        const int     tid = (int) it.get_local_id(0);
        const TI *    xr  = (const TI *) (xb + r * xs);
        // Two passes, mean then the centred sum of squares: the residual
        // streams this normalises carry offsets large next to their spread.
        float s = 0.0f;
        for (int64_t i = tid; i < h; i += WG) s += (float) xr[i];
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) h;
        float ss = 0.0f;
        for (int64_t i = tid; i < h; i += WG) {
            const float d = (float) xr[i] - mean;
            ss += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) h + eps);
        TO * orow = (TO *) (ob + r * os);
        for (int64_t i = tid; i < h; i += WG)
            orow[i] = TO(((float) xr[i] - mean) * inv * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f));
    });
}

template <typename T>
struct k_geglu_q8;

template <typename T>
void run_geglu_q8(sycl::queue & q, const ggml_tensor * dst) {
    const ggml_tensor * g = dst->src[0];
    const ggml_tensor * u = dst->src[1];
    const int64_t n = g->ne[0], rows = ggml_nrows(g);
    const char * gb = (const char *) g->data;
    const char * ub = (const char *) u->data;
    int8_t *     codes = (int8_t *) dst->data;
    float *      scales = (float *) (codes + rows * n);
    const size_t gs = g->nb[1], us = u->nb[1];
    const bool   v8 = rows_vec8(gb, gs, n, sizeof(T)) && rows_vec8(ub, us, n, sizeof(T));
    const bool   silu = glu_silu(dst);
    q.parallel_for<k_geglu_q8<T>>(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
                                  [=](sycl::nd_item<1> it) {
        const int64_t r  = it.get_group(0);
        const T *     gr = (const T *) (gb + r * gs);
        const T *     ur = (const T *) (ub + r * us);
        if (v8) {
            q8_row8(it, n, codes + r * n, scales + r, [&](int64_t c, float * v) {
                const sycl::vec<float, 8> gv = load8(gr, c), uv = load8(ur, c);
#pragma unroll
                for (int j = 0; j < 8; ++j) v[j] = glu_act(silu, gv[j]) * uv[j];
            });
        } else {
            q8_row(it, n, codes + r * n, scales + r,
                   [&](int64_t i) { return glu_act(silu, (float) gr[i]) * (float) ur[i]; });
        }
    });
}

template <typename TO>
struct k_ropekv;

// TO: the q out and K/V buffer type, F16 (TO) or BF16 (the table mode).
template <typename TO>
void run_ropekv(sycl::queue & q, ggml_tensor * dst) {
    const auto &        s   = *static_cast<const FusedRopeKvSpec *>(custom_userdata(dst));
    const ggml_tensor * qkv = dst->src[0];
    const int32_t *     pos = (const int32_t *) dst->src[1]->data;
    const ggml_tensor * kb  = dst->src[2];
    const ggml_tensor * vb  = dst->src[3];
    const int64_t T = qkv->ne[1], h2 = s.hd / 2, heads = s.nq + 2 * s.nkv;
    const char *  xb = (const char *) qkv->data;
    const size_t  xs = qkv->nb[1];
    TO *  qo = (TO *) dst->data;
    TO *  ko = (TO *) ((char *) kb->data + s.row0 * kb->nb[2]);
    TO *  vo = (TO *) ((char *) vb->data + s.row0 * vb->nb[2]);
    const int64_t nq = s.nq, nkv = s.nkv, hd = s.hd;
    const float   ts = s.theta_scale;
    // table: cos / sin rows per token in place of the positions
    const float * ct = s.table ? (const float *) dst->src[1]->data : nullptr;
    const float * st = s.table ? (const float *) dst->src[4]->data : nullptr;
    if (s.table) pos = nullptr;
    q.parallel_for<k_ropekv<TO>>(sycl::range<3>((size_t) T, (size_t) heads, (size_t) h2), [=](sycl::item<3> it) {
        const int64_t t = it[0], hh = it[1], i = it[2];
        const float * x = (const float *) (xb + t * xs) + hh * hd;
        const float   x0 = x[i], x1 = x[i + h2];
        TO *  o;
        if (hh < nq)            o = qo + (t * nq + hh) * hd;
        else if (hh < nq + nkv) o = ko + (t * nkv + hh - nq) * hd;
        else {
            o = vo + (t * nkv + hh - nq - nkv) * hd;
            o[i]      = (TO) x0;
            o[i + h2] = (TO) x1;
            return;
        }
        if (ct) {
            const float * c = ct + t * hd, * sn = st + t * hd;
            o[i]      = (TO) (x0 * c[i] - x1 * sn[i]);
            o[i + h2] = (TO) (x1 * c[i + h2] + x0 * sn[i + h2]);
            return;
        }
        // ggml-sycl's rope_neox with freq_scale 1, ext_factor 0, attn_factor 1.
        const float theta = (float) pos[t] * sycl::pow(ts, (float) i);
        const float c = sycl::cos(theta), sn = sycl::sin(theta);
        o[i]      = (TO) (x0 * c - x1 * sn);
        o[i + h2] = (TO) (x0 * sn + x1 * c);
    });
}

// A q8 blob of rows x h: one I8 row of rows * (h + 4) bytes.
bool is_blob(const ggml_tensor * op, const ggml_tensor * like) {
    return op->type == GGML_TYPE_I8 && ggml_is_contiguous(op) && op->ne[1] == 1 &&
           op->ne[0] == ggml_nrows(like) * (like->ne[0] + 4);
}

bool supported(const ggml_tensor * op) {
    static const bool off = env_flag("VLA_SYCL_NO_FUSED");
    if (off) return false;
    const uint32_t m = magic_of(op);
    if (m == FUSED_ADARMS_MAGIC) {
        return op->src[0]->type == GGML_TYPE_F32 && rows_packed(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
               (is_blob(op, op->src[0]) ||
                (rows_packed(op) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_BF16)));
    }
    if (m == FUSED_ROPEKV_MAGIC) {
        // Contiguous q out and K/V buffers ([hd, nkv, n_ctx], F16, or BF16 all three).
        return op->src[0]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == 4 && ggml_is_contiguous(op) &&
               ggml_is_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) &&
               (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16) && op->src[2]->type == op->type &&
               op->src[3]->type == op->type;
    }
    if (m == FUSED_RMSQ_MAGIC) {
        return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) &&
               rows_packed(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
               (is_blob(op, op->src[0]) ||
                (rows_packed(op) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_BF16)));
    }
    if (m == FUSED_GATERES_MAGIC) {
        return op->src[0]->type == GGML_TYPE_F32 && rows_packed(op->src[0]) && rows_packed(op->src[1]) &&
               rows_packed(op) && (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_BF16) &&
               ggml_is_contiguous(op->src[2]);
    }
    if (m == FUSED_LINEAR_MAGIC) {
        const ggml_tensor * W = op->src[0], * x = op->src[1], * b = op->src[2], * r = op->src[3];
        return W->type == GGML_TYPE_BF16 && ggml_is_contiguous(W) && x->type == GGML_TYPE_BF16 &&
               x->nb[0] == 2 && x->ne[2] == 1 && x->ne[3] == 1 && (!b || ggml_is_contiguous(b)) &&
               ggml_is_contiguous(op) && (op->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_F32) &&
               (!r || (r->nb[0] == ggml_type_size(r->type) && r->ne[0] == op->ne[0] && r->ne[1] == op->ne[1] &&
                       (r->type == GGML_TYPE_BF16 || r->type == GGML_TYPE_F32)));
    }
    if (m == QLINEAR_MAGIC) {
        const ggml_tensor * W = op->src[0], * x = op->src[2], * r = op->src[4];
        const int64_t K = qlinear_k(W, static_cast<const QLinearSpec *>(custom_userdata(op))->bits);
        const bool x_ok = x->type == GGML_TYPE_I8
            ? ggml_is_contiguous(x) && x->ne[1] == 1 && x->ne[0] % (K + 4) == 0 && K % 4 == 0
            : (x->type == GGML_TYPE_BF16 || x->type == GGML_TYPE_F32) && x->nb[0] == ggml_type_size(x->type) &&
              x->ne[2] == 1 && x->ne[3] == 1;
        const ggml_tensor * g = op->src[5];
        return W->type == GGML_TYPE_I8 && ggml_is_contiguous(W) && op->src[1]->type == GGML_TYPE_F32 && x_ok &&
               (!g || (g->type == GGML_TYPE_F32 && ggml_is_contiguous(g))) && ggml_is_contiguous(op) &&
               (op->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_F32) &&
               (!r || (r->nb[0] == ggml_type_size(r->type) && (r->type == GGML_TYPE_BF16 || r->type == GGML_TYPE_F32)));
    }
    if (m == FUSED_LNORM_MAGIC) {
        return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) &&
               rows_packed(op->src[0]) && rows_packed(op) &&
               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_BF16);
    }
    if (m == FUSED_GEGLU_MAGIC && op->type == GGML_TYPE_I8) {
        const ggml_tensor * g = op->src[0], * u = op->src[1];
        return (g->type == GGML_TYPE_BF16 || g->type == GGML_TYPE_F32) && u->type == g->type &&
               rows_packed(g) && rows_packed(u) && is_blob(op, g);
    }
    if (m == FUSED_GEGLU_MAGIC) {
        const size_t es = ggml_type_size(op->type);
        return (op->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_F32) && op->src[0]->type == op->type &&
               op->src[1]->type == op->type && rows_packed(op->src[0]) && rows_packed(op->src[1]) &&
               rows_packed(op) && op->ne[0] % 4 == 0 && op->src[0]->nb[1] % (4 * es) == 0 &&
               op->src[1]->nb[1] % (4 * es) == 0 && op->nb[1] % (4 * es) == 0 &&
               (uintptr_t) op->src[1]->view_offs % (4 * es) == 0;
    }
    return false;
}

bool forward(ggml_tensor * dst, void * queue) {
    if (!supported(dst)) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    if (magic_of(dst) == FUSED_LINEAR_MAGIC) return sycl_fused_linear(dst, custom_userdata(dst), q);
    if (magic_of(dst) == QLINEAR_MAGIC) return sycl_qlinear(dst, custom_userdata(dst), q);
    if (magic_of(dst) == FUSED_LNORM_MAGIC) {
        const float eps = static_cast<const FusedLNormSpec *>(custom_userdata(dst))->eps;
        const bool  bi  = dst->src[0]->type == GGML_TYPE_BF16, bo = dst->type == GGML_TYPE_BF16;
        if (bi && bo)       run_lnorm<bf16, bf16>(q, dst, eps);
        else if (bi)        run_lnorm<bf16, float>(q, dst, eps);
        else if (bo)        run_lnorm<float, bf16>(q, dst, eps);
        else                run_lnorm<float, float>(q, dst, eps);
        return true;
    }
    if (magic_of(dst) == FUSED_ROPEKV_MAGIC) {
        if (dst->type == GGML_TYPE_BF16) run_ropekv<bf16>(q, dst);
        else                             run_ropekv<sycl::half>(q, dst);
        return true;
    }
    if (magic_of(dst) == FUSED_RMSQ_MAGIC) {
        const float eps = static_cast<const FusedRmsSpec *>(custom_userdata(dst))->eps;
        const bool  bi  = dst->src[0]->type == GGML_TYPE_BF16;
        if (dst->type == GGML_TYPE_I8) {
            if (bi) run_rms_q8<bf16>(q, dst, eps);
            else    run_rms_q8<float>(q, dst, eps);
        } else if (dst->type == GGML_TYPE_BF16) {
            if (bi) run_rms<bf16, bf16>(q, dst, eps);
            else    run_rms<float, bf16>(q, dst, eps);
        } else {
            if (bi) run_rms<bf16, float>(q, dst, eps);
            else    run_rms<float, float>(q, dst, eps);
        }
        return true;
    }
    if (magic_of(dst) == FUSED_GEGLU_MAGIC && dst->type == GGML_TYPE_I8) {
        if (dst->src[0]->type == GGML_TYPE_BF16) run_geglu_q8<bf16>(q, dst);
        else                                     run_geglu_q8<float>(q, dst);
        return true;
    }
    if (magic_of(dst) == FUSED_GEGLU_MAGIC) {
        if (dst->type == GGML_TYPE_BF16) run_geglu<bf16>(q, dst);
        else                             run_geglu<float>(q, dst);
        return true;
    }
    if (magic_of(dst) == FUSED_ADARMS_MAGIC) {
        const auto & s = *static_cast<const FusedAdarmsSpec *>(custom_userdata(dst));
        if (dst->type == GGML_TYPE_I8)        run_adarms_q8(q, dst, s.eps);
        else if (dst->type == GGML_TYPE_BF16) run_adarms<bf16>(q, dst, s.eps);
        else                             run_adarms<float>(q, dst, s.eps);
        return true;
    }
    if (dst->src[1]->type == GGML_TYPE_BF16) run_gateres<bf16>(q, dst);
    else                                     run_gateres<float>(q, dst);
    return true;
}

}  // namespace

void sycl_register_fused_ops() {
    sycl_ext_register(forward, supported);
    fused_ew_registered() = true;
}

}  // namespace vla
