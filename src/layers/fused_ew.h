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

// Fused elementwise chains as graph nodes.
//
// ggml runs every ADD, MUL, RMS_NORM and CPY as its own kernel, each reading and
// writing the whole activation tensor; on a bandwidth-bound GPU that dominates
// the time outside the GEMMs. These nodes do each chain (adaRMS, gated residual,
// geglu, ...) in one pass.
//
// Same encoding as FoldQuant (layers/fq_linear.h): GGML_OP_CUSTOM with a magic
// word at the head of its userdata. The CPU backend runs the reference below;
// SYCL claims the node through src/sycl/vla_sycl_fused.cpp. Only build them for
// a backend that claims them, or the scheduler moves the node to the CPU.

#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <tuple>
#include <mutex>
#include <map>

namespace vla {

constexpr uint32_t FUSED_ADARMS_MAGIC = 0x534d5241;  // "ARMS"
constexpr uint32_t FUSED_GATERES_MAGIC = 0x53455247; // "GRES"
constexpr uint32_t FUSED_GEGLU_MAGIC   = 0x554c4547; // "GELU"
constexpr uint32_t FUSED_LINEAR_MAGIC  = 0x524e494c; // "LINR"
constexpr uint32_t FUSED_LNORM_MAGIC   = 0x4d524f4e; // "NORM"
constexpr uint32_t FUSED_RMSQ_MAGIC    = 0x51534d52; // "RMSQ"
constexpr uint32_t FUSED_ROPEKV_MAGIC  = 0x564b5052; // "RPKV"

// NEOX RoPE on the q and k heads of one fused q|k|v projection, in one pass
// that also stores the cache rows: out = rope(q) as F16 [hd, nq, T]; rope(k)
// and v go to rows [row0, row0 + T) of the F16 [hd, nkv, n_ctx] K and V
// buffers (srcs 2 and 3, the KvSlot buffers, written in place; their readers
// are ordered after this node). Same arithmetic as ggml's RoPE with no
// frequency scaling: theta = pos * base^(-2i/hd). With table set
// (fused_rope_kv_table), src 1 and 4 are cos and sin F32 [hd, T] instead of the
// positions: rotate-half against per-row tables, as layers/rope.h rope_2d.
struct FusedRopeKvSpec {
    uint32_t magic = FUSED_ROPEKV_MAGIC;
    int32_t  nq = 0, nkv = 0, hd = 0;
    int64_t  row0 = 0;
    float    theta_scale = 1.0f;  // base^(-2/hd)
    int32_t  table = 0;
};

// A per-token INT8 activation (layers/qlinear.h takes it in place of a float
// input): M rows of K codes, then the M F32 row scales, as a 1-D I8 tensor of
// M * (K + 4) bytes. A node whose output type is I8 writes one of these: the
// producing kernel already holds the row, so it quantizes it on the way out
// rather than leaving that to a pass of its own.
inline int64_t q8_rows(const ggml_tensor * blob, int64_t K) {
    return blob->ne[0] / (K + 4);
}

// out[r, i] = n + n * scale[i] + shift[i],  n = x[r, i] / sqrt(mean_i(x[r]^2) + eps)
// mod is the [3h] F32 modulation vector (scale | shift | gate); out has the
// requested type (F32 or BF16). The arithmetic order matches build_adarms.
struct FusedAdarmsSpec {
    uint32_t magic = FUSED_ADARMS_MAGIC;
    float    eps   = 1e-6f;
};

// out[r, i] = x[r, i] + o[r, i] * gate[i]; x F32, o F32 or BF16, out F32.
// gate is the third third of the [3h] modulation vector.
struct FusedGateResSpec {
    uint32_t magic = FUSED_GATERES_MAGIC;
};

// out = gelu_tanh(gate) * up, all three of one type (BF16 here: ggml-sycl's own
// GEGLU takes F32/F16 only). Same tanh form as ggml_gelu. silu: silu(gate) * up
// instead (SwiGLU, the Llama MLP).
struct FusedGegluSpec {
    uint32_t magic = FUSED_GEGLU_MAGIC;
    int32_t  silu  = 0;
};

// out = act(x W^T + b) (+ res): one GEMM with its bias, activation and residual
// as epilogue (oneDNN post-ops on SYCL), where ggml runs a bias add, an
// activation and a residual add as separate passes over the [N, M] result.
// W [K, N] BF16, x [K, M] BF16, b [N] F32, res [N, M] of the output type.
struct FusedLinearSpec {
    uint32_t magic = FUSED_LINEAR_MAGIC;
    int32_t  gelu  = 0;  // after the bias: 1 gelu_tanh, 2 gelu_erf (FUSED_GELU_*)
};

// The activation a fused_linear / qlinear applies after its bias.
constexpr int FUSED_GELU_TANH = 1;
constexpr int FUSED_GELU_ERF  = 2;

// out = x / sqrt(mean(x^2) + eps) * w per row (w F32, already 1 + w for a Gemma
// norm), written as a per-token INT8 blob for the GEMM that follows, or as F32
// / BF16 rows (fused_rms).
struct FusedRmsSpec {
    uint32_t magic = FUSED_RMSQ_MAGIC;
    float    eps   = 1e-6f;
};

// out = (x - mean) / sqrt(var + eps) * w + b per row, in one pass; x F32 or
// BF16, w/b F32, out in the requested type.
struct FusedLNormSpec {
    uint32_t magic = FUSED_LNORM_MAGIC;
    float    eps   = 1e-6f;
};

inline float fused_gelu_tanh(float x) {
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(c * x * (1.0f + 0.044715f * x * x)));
}

inline float fused_gelu(int mode, float x) {
    return mode == FUSED_GELU_ERF ? 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f)) : fused_gelu_tanh(x);
}

inline float fused_load(const ggml_tensor * t, const char * row, int64_t i) {
    if (t->type == GGML_TYPE_BF16) {
        uint32_t u = (uint32_t) ((const uint16_t *) row)[i] << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    return ((const float *) row)[i];
}

inline void fused_store(ggml_tensor * t, char * row, int64_t i, float v) {
    if (t->type == GGML_TYPE_BF16) {
        ((ggml_bf16_t *) row)[i] = ggml_fp32_to_bf16(v);
    } else {
        ((float *) row)[i] = v;
    }
}

// Row m of K values into a q8 blob (or a float dst row): scale max|v| / 127.
inline void fused_store_row(ggml_tensor * dst, int64_t m, int64_t K, const float * v) {
    if (dst->type != GGML_TYPE_I8) {
        char * dr = (char *) dst->data + m * dst->nb[1];
        for (int64_t i = 0; i < K; ++i) fused_store(dst, dr, i, v[i]);
        return;
    }
    const int64_t M = q8_rows(dst, K);
    float amax = 0.0f;
    for (int64_t i = 0; i < K; ++i) amax = std::fmax(amax, std::fabs(v[i]));
    const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
    int8_t * codes = (int8_t *) dst->data + m * K;
    for (int64_t i = 0; i < K; ++i)
        codes[i] = (int8_t) std::lrint(std::fmin(127.0f, std::fmax(-127.0f, v[i] / sc)));
    ((float *) ((int8_t *) dst->data + M * K))[m] = sc;
}

inline void fused_adarms_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto & s = *static_cast<const FusedAdarmsSpec *>(ud);
    const ggml_tensor * x = dst->src[0];
    const float * mod = (const float *) dst->src[1]->data;
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    std::vector<float> v((size_t) h);
    for (int64_t r = ith; r < rows; r += nth) {
        const float * xr = (const float *) ((const char *) x->data + r * x->nb[1]);
        double ss = 0.0;
        for (int64_t i = 0; i < h; ++i) ss += (double) xr[i] * xr[i];
        const float inv = 1.0f / std::sqrt((float) (ss / (double) h) + s.eps);
        for (int64_t i = 0; i < h; ++i) {
            const float n = xr[i] * inv;
            v[i] = (n + n * mod[i]) + mod[h + i];
        }
        fused_store_row(dst, r, h, v.data());
    }
}

inline void fused_rms_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto & s = *static_cast<const FusedRmsSpec *>(ud);
    const ggml_tensor * x = dst->src[0];
    const float * w = (const float *) dst->src[1]->data;
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    std::vector<float> v((size_t) h);
    for (int64_t r = ith; r < rows; r += nth) {
        const char * xr = (const char *) x->data + r * x->nb[1];
        double ss = 0.0;
        for (int64_t i = 0; i < h; ++i) { const double t = fused_load(x, xr, i); ss += t * t; }
        const float inv = 1.0f / std::sqrt((float) (ss / (double) h) + s.eps);
        for (int64_t i = 0; i < h; ++i) v[i] = fused_load(x, xr, i) * inv * w[i];
        fused_store_row(dst, r, h, v.data());
    }
}

inline void fused_gateres_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * o = dst->src[1];
    const float * mod = (const float *) dst->src[2]->data;
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    for (int64_t r = ith; r < rows; r += nth) {
        const float * xr = (const float *) ((const char *) x->data + r * x->nb[1]);
        const char *  orow = (const char *) o->data + r * o->nb[1];
        float * dr = (float *) ((char *) dst->data + r * dst->nb[1]);
        for (int64_t i = 0; i < h; ++i) dr[i] = xr[i] + fused_load(o, orow, i) * mod[2 * h + i];
    }
}

inline float fused_silu(float x) { return x / (1.0f + std::exp(-x)); }

inline void fused_geglu_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const bool silu = static_cast<const FusedGegluSpec *>(ud)->silu != 0;
    const ggml_tensor * g = dst->src[0];
    const ggml_tensor * u = dst->src[1];
    const int64_t n = g->ne[0], rows = ggml_nrows(g);
    std::vector<float> v((size_t) n);
    for (int64_t r = ith; r < rows; r += nth) {
        const char * gr = (const char *) g->data + r * g->nb[1];
        const char * ur = (const char *) u->data + r * u->nb[1];
        for (int64_t i = 0; i < n; ++i) {
            const float a = fused_load(g, gr, i);
            v[i] = (silu ? fused_silu(a) : fused_gelu_tanh(a)) * fused_load(u, ur, i);
        }
        fused_store_row(dst, r, n, v.data());
    }
}

// The output of a row-wise fused node: [h, rows...] of out_type, or a q8 blob.
inline ggml_tensor * fused_row_out(ggml_context * C, ggml_tensor * like, ggml_type out_type, ggml_tensor ** args,
                                   int n, ggml_custom_op_t fun, const void * spec) {
    if (out_type == GGML_TYPE_I8)
        return ggml_custom_4d(C, GGML_TYPE_I8, ggml_nrows(like) * (like->ne[0] + 4), 1, 1, 1, args, n, fun,
                              GGML_N_TASKS_MAX, (void *) spec);
    return ggml_custom_4d(C, out_type, like->ne[0], like->ne[1], like->ne[2], like->ne[3], args, n, fun,
                          GGML_N_TASKS_MAX, (void *) spec);
}

// gate, up: [n, T] of one type. Returns gelu_tanh(gate) * up in that type, or
// as a q8 blob (out_type I8).
inline ggml_tensor * fused_geglu(ggml_context * C, ggml_tensor * gate, ggml_tensor * up,
                                 ggml_type out_type = GGML_TYPE_COUNT, bool silu = false) {
    static const FusedGegluSpec gelu_spec{FUSED_GEGLU_MAGIC, 0}, silu_spec{FUSED_GEGLU_MAGIC, 1};
    GGML_ASSERT(ggml_are_same_shape(gate, up) && gate->type == up->type);
    ggml_tensor * args[2] = { gate, up };
    ggml_tensor * t = fused_row_out(C, gate, out_type == GGML_TYPE_COUNT ? gate->type : out_type, args, 2,
                                    fused_geglu_cpu, silu ? &silu_spec : &gelu_spec);
    ggml_set_name(t, silu ? "fused_swiglu" : "fused_geglu");
    return t;
}

// silu(gate) * up, as fused_geglu.
inline ggml_tensor * fused_swiglu(ggml_context * C, ggml_tensor * gate, ggml_tensor * up,
                                  ggml_type out_type = GGML_TYPE_COUNT) {
    return fused_geglu(C, gate, up, out_type, true);
}

inline void fused_linear_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto & s = *static_cast<const FusedLinearSpec *>(ud);
    const ggml_tensor * W = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * b = dst->src[2];
    const ggml_tensor * res = dst->src[3];
    const int64_t K = x->ne[0], M = x->ne[1], N = W->ne[1];
    for (int64_t m = ith; m < M; m += nth) {
        const char * xr = (const char *) x->data + m * x->nb[1];
        char *       dr = (char *) dst->data + m * dst->nb[1];
        for (int64_t n = 0; n < N; ++n) {
            const char * wr = (const char *) W->data + n * W->nb[1];
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += fused_load(W, wr, k) * fused_load(x, xr, k);
            if (b) acc += ((const float *) b->data)[n];
            if (s.gelu) acc = fused_gelu(s.gelu, acc);
            if (res) acc += fused_load(res, (const char *) res->data + m * res->nb[1], n);
            fused_store(dst, dr, n, acc);
        }
    }
}

inline void fused_lnorm_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto & s = *static_cast<const FusedLNormSpec *>(ud);
    const ggml_tensor * x = dst->src[0];
    const float * w = dst->src[1] ? (const float *) dst->src[1]->data : nullptr;
    const float * b = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;
    const int64_t h = x->ne[0], rows = ggml_nrows(x);
    for (int64_t r = ith; r < rows; r += nth) {
        const char * xr = (const char *) x->data + r * x->nb[1];
        double sum = 0.0, sq = 0.0;
        for (int64_t i = 0; i < h; ++i) { const double v = fused_load(x, xr, i); sum += v; sq += v * v; }
        const double mean = sum / (double) h;
        const float  inv  = 1.0f / std::sqrt((float) (sq / (double) h - mean * mean) + s.eps);
        char * dr = (char *) dst->data + r * dst->nb[1];
        for (int64_t i = 0; i < h; ++i)
            fused_store(dst, dr, i, (fused_load(x, xr, i) - (float) mean) * inv * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f));
    }
}

inline void fused_ropekv_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto & s = *static_cast<const FusedRopeKvSpec *>(ud);
    const ggml_tensor * qkv = dst->src[0];
    const int32_t *     pos = s.table ? nullptr : (const int32_t *) dst->src[1]->data;
    ggml_tensor *       kb  = dst->src[2];
    ggml_tensor *       vb  = dst->src[3];
    const int64_t T = qkv->ne[1], hd = s.hd, h2 = hd / 2;
    // F16 out and buffers, or BF16 all three (the table mode may ask for it)
    const bool bf = dst->type == GGML_TYPE_BF16;
    auto put = [bf](void * o, int64_t i, float v) {
        if (bf) ((ggml_bf16_t *) o)[i] = ggml_fp32_to_bf16(v);
        else    ((ggml_fp16_t *) o)[i] = ggml_fp32_to_fp16(v);
    };
    for (int64_t t = ith; t < T; t += nth) {
        const float * row = (const float *) ((const char *) qkv->data + t * qkv->nb[1]);
        for (int64_t hh = 0; hh < s.nq + 2 * s.nkv; ++hh) {
            const float * x = row + hh * hd;
            char * o;
            if (hh < s.nq)
                o = (char *) dst->data + t * dst->nb[2] + hh * dst->nb[1];
            else if (hh < s.nq + s.nkv)
                o = (char *) kb->data + (s.row0 + t) * kb->nb[2] + (hh - s.nq) * kb->nb[1];
            else
                o = (char *) vb->data + (s.row0 + t) * vb->nb[2] + (hh - s.nq - s.nkv) * vb->nb[1];
            for (int64_t i = 0; i < h2; ++i) {
                if (hh >= s.nq + s.nkv) {
                    put(o, i, x[i]);
                    put(o, i + h2, x[i + h2]);
                    continue;
                }
                if (s.table) {
                    const float * cs = (const float *) dst->src[1]->data + t * hd;
                    const float * sn = (const float *) dst->src[4]->data + t * hd;
                    put(o, i,      x[i] * cs[i] - x[i + h2] * sn[i]);
                    put(o, i + h2, x[i + h2] * cs[i + h2] + x[i] * sn[i + h2]);
                    continue;
                }
                const float th = (float) pos[t] * std::pow(s.theta_scale, (float) i);
                const float c = std::cos(th), sn = std::sin(th);
                put(o, i,      x[i] * c - x[i + h2] * sn);
                put(o, i + h2, x[i] * sn + x[i + h2] * c);
            }
        }
    }
}

// The spec a rope_kv node points at, one per distinct value, at a stable address.
inline const FusedRopeKvSpec * fused_ropekv_spec(int nq, int nkv, int hd, int64_t row0, float base, bool table = false) {
    static std::map<std::tuple<int, int, int, int64_t, float, bool>, FusedRopeKvSpec> specs;
    static std::mutex                                                                 mu;
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_tuple(nq, nkv, hd, row0, base, table);
    auto it = specs.find(key);
    if (it == specs.end())
        it = specs.emplace(key, FusedRopeKvSpec{FUSED_ROPEKV_MAGIC, nq, nkv, hd, row0,
                                                std::pow(base, -2.0f / (float) hd), table ? 1 : 0}).first;
    return &it->second;
}

// Set by sycl_register_fused_ops(): the process has a backend that claims these
// nodes, so the shared helpers (geglu) may emit them.
inline bool & fused_ew_registered() {
    static bool on = false;
    return on;
}

// x: F32 [h, T] (rows contiguous); mod: F32 [3h]. Returns [h, T] of out_type,
// or a q8 blob (out_type I8).
inline ggml_tensor * fused_adarms(ggml_context * C, ggml_tensor * x, ggml_tensor * mod, ggml_type out_type,
                                  const FusedAdarmsSpec & spec) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && mod->ne[0] == 3 * x->ne[0]);
    ggml_tensor * args[2] = { x, mod };
    ggml_tensor * t = fused_row_out(C, x, out_type, args, 2, fused_adarms_cpu, &spec);
    ggml_set_name(t, "fused_adarms");
    return t;
}

// RMSNorm with its weight, as a q8 blob for the GEMM that follows. x: [h, T]
// F32 or BF16, rows dense; w: F32 [h].
inline ggml_tensor * fused_rms_q8(ggml_context * C, ggml_tensor * x, ggml_tensor * w, const FusedRmsSpec & spec) {
    GGML_ASSERT(w->type == GGML_TYPE_F32 && w->ne[0] == x->ne[0] && x->nb[0] == ggml_type_size(x->type));
    ggml_tensor * args[2] = { x, w };
    ggml_tensor * t = fused_row_out(C, x, GGML_TYPE_I8, args, 2, fused_rms_cpu, &spec);
    ggml_set_name(t, "fused_rms_q8");
    return t;
}

// The same RMSNorm written as [h, T] rows of out_type (F32 or BF16).
inline ggml_tensor * fused_rms(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_type out_type,
                               const FusedRmsSpec & spec) {
    GGML_ASSERT(w->type == GGML_TYPE_F32 && w->ne[0] == x->ne[0] && x->nb[0] == ggml_type_size(x->type));
    GGML_ASSERT(out_type == GGML_TYPE_F32 || out_type == GGML_TYPE_BF16);
    ggml_tensor * args[2] = { x, w };
    ggml_tensor * t = fused_row_out(C, x, out_type, args, 2, fused_rms_cpu, &spec);
    ggml_set_name(t, "fused_rms");
    return t;
}

// x: F32 [h, T]; o: F32 or BF16 [h, T]; mod: F32 [3h]. Returns F32 [h, T].
inline ggml_tensor * fused_gateres(ggml_context * C, ggml_tensor * x, ggml_tensor * o, ggml_tensor * mod,
                                   const FusedGateResSpec & spec) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_are_same_shape(x, o) && mod->ne[0] == 3 * x->ne[0]);
    GGML_ASSERT(o->type == GGML_TYPE_F32 || o->type == GGML_TYPE_BF16);
    ggml_tensor * args[3] = { x, o, mod };
    ggml_tensor * t = ggml_custom_4d(C, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, 3,
                                     fused_gateres_cpu, GGML_N_TASKS_MAX, (void *) &spec);
    ggml_set_name(t, "fused_gateres");
    return t;
}

// W [K, N] BF16, x [K, M] BF16 (rows dense), b [N] F32 or null, res [N, M] or
// null. Returns [N, M] of out_type.
inline ggml_tensor * fused_linear(ggml_context * C, ggml_tensor * W, ggml_tensor * x, ggml_tensor * b,
                                  ggml_tensor * res, ggml_type out_type, int gelu) {
    // gelu: 0, or FUSED_GELU_TANH (true) / FUSED_GELU_ERF.
    static const FusedLinearSpec specs[3] = {{FUSED_LINEAR_MAGIC, 0}, {FUSED_LINEAR_MAGIC, FUSED_GELU_TANH},
                                             {FUSED_LINEAR_MAGIC, FUSED_GELU_ERF}};
    GGML_ASSERT(gelu >= 0 && gelu <= FUSED_GELU_ERF);
    GGML_ASSERT(W->ne[0] == x->ne[0] && x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(!b || (b->type == GGML_TYPE_F32 && b->ne[0] == W->ne[1]));
    ggml_tensor * args[4] = { W, x, b, res };
    const int n = res ? 4 : 3;
    ggml_tensor * t = ggml_custom_4d(C, out_type, W->ne[1], x->ne[1], 1, 1, args, n, fused_linear_cpu,
                                     GGML_N_TASKS_MAX, (void *) &specs[gelu]);
    ggml_set_name(t, "fused_linear");
    return t;
}

// x: [h, T] F32 or BF16; w, b: F32 [h], or both null for no affine. Returns
// [h, T] of out_type.
inline ggml_tensor * fused_lnorm(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                                 ggml_type out_type, const FusedLNormSpec & spec) {
    GGML_ASSERT(!w == !b);
    GGML_ASSERT(!w || (w->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && w->ne[0] == x->ne[0]));
    ggml_tensor * args[3] = { x, w, b };
    ggml_tensor * t = ggml_custom_4d(C, out_type, x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, w ? 3 : 1,
                                     fused_lnorm_cpu, GGML_N_TASKS_MAX, (void *) &spec);
    ggml_set_name(t, "fused_lnorm");
    return t;
}

// qkv: F32 [(nq + 2 nkv) hd, T], rows dense; pos: I32 [T]; kbuf/vbuf: F16
// [hd, nkv, n_ctx]. Returns rope(q) as F16 [hd, nq, T].
inline ggml_tensor * fused_rope_kv(ggml_context * C, ggml_tensor * qkv, ggml_tensor * pos, ggml_tensor * kbuf,
                                   ggml_tensor * vbuf, int nq, int nkv, int hd, int64_t row0, float base) {
    GGML_ASSERT(qkv->type == GGML_TYPE_F32 && qkv->ne[0] == (int64_t) (nq + 2 * nkv) * hd && qkv->nb[0] == 4);
    GGML_ASSERT(kbuf->type == GGML_TYPE_F16 && vbuf->type == GGML_TYPE_F16 && row0 + qkv->ne[1] <= kbuf->ne[2]);
    ggml_tensor * args[4] = { qkv, pos, kbuf, vbuf };
    ggml_tensor * t = ggml_custom_4d(C, GGML_TYPE_F16, hd, nq, qkv->ne[1], 1, args, 4, fused_ropekv_cpu,
                                     GGML_N_TASKS_MAX, (void *) fused_ropekv_spec(nq, nkv, hd, row0, base));
    ggml_set_name(t, "fused_rope_kv");
    return t;
}

// fused_rope_kv against cos / sin tables F32 [hd, T] (row t for token t) in
// place of the positions: out[i] = x[i] cos[i] - x[i+h] sin[i], out[i+h] =
// x[i+h] cos[i+h] + x[i] sin[i+h], h = hd/2 (layers/rope.h rope_2d).
// The buffers' type (F16 or BF16) is the output's.
inline ggml_tensor * fused_rope_kv_table(ggml_context * C, ggml_tensor * qkv, ggml_tensor * cos_t, ggml_tensor * sin_t,
                                         ggml_tensor * kbuf, ggml_tensor * vbuf, int nq, int nkv, int hd,
                                         int64_t row0) {
    GGML_ASSERT(qkv->type == GGML_TYPE_F32 && qkv->ne[0] == (int64_t) (nq + 2 * nkv) * hd && qkv->nb[0] == 4);
    GGML_ASSERT((kbuf->type == GGML_TYPE_F16 || kbuf->type == GGML_TYPE_BF16) && vbuf->type == kbuf->type &&
                row0 + qkv->ne[1] <= kbuf->ne[2]);
    GGML_ASSERT(cos_t->type == GGML_TYPE_F32 && sin_t->type == GGML_TYPE_F32 && ggml_is_contiguous(cos_t) &&
                ggml_is_contiguous(sin_t) && cos_t->ne[0] == hd && cos_t->ne[1] >= qkv->ne[1] &&
                ggml_are_same_shape(cos_t, sin_t));
    ggml_tensor * args[5] = { qkv, cos_t, kbuf, vbuf, sin_t };
    ggml_tensor * t = ggml_custom_4d(C, kbuf->type, hd, nq, qkv->ne[1], 1, args, 5, fused_ropekv_cpu,
                                     GGML_N_TASKS_MAX, (void *) fused_ropekv_spec(nq, nkv, hd, row0, 0.0f, true));
    ggml_set_name(t, "fused_rope_kv_table");
    return t;
}

}  // namespace vla
