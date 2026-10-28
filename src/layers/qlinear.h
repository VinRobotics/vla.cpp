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

// W8A8 linear with dynamic per-token activation scales, as one graph node.
// Weights are symmetric per-output-channel INT8 (WeightLoader::quant_i8); each
// activation row gets its own scale (max |x| / 127), accumulation is exact int32:
//
//   out[m, n] = act(sx[m] * sw[n] * sum_k qx[m, k] * qw[n, k] + b[n]) (* gate[n]) (+ res[m, n])
//
// x is a float activation or a q8 blob from a fused producer (layers/fused_ew.h).
// Encoded like layers/fused_ew.h: GGML_OP_CUSTOM with a magic word, a CPU
// reference here, claimed on SYCL by src/sycl/vla_sycl_bf16.cpp. Build it only
// for a backend that claims it.

#pragma once

#include "fused_ew.h"
#include "loader.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace vla {

constexpr uint32_t QLINEAR_MAGIC = 0x4e494c51;  // "QLIN"

struct QLinearSpec {
    uint32_t magic = QLINEAR_MAGIC;
    int32_t  gelu  = 0;
    int32_t  bits  = 8;  // weight codes: 8, or 4 (two per byte along K, low nibble first)
    int32_t  group = 0;  // INT4: K-elements per weight scale (scale[g * N + n])
};

// The spec a node points at, one per distinct value, at a stable address.
inline const QLinearSpec * qlinear_spec(int gelu, int bits, int64_t group) {
    static std::map<std::tuple<int, int, int64_t>, QLinearSpec> specs;
    static std::mutex                                            mu;
    std::lock_guard<std::mutex> lock(mu);
    auto it = specs.find({gelu, bits, group});
    if (it == specs.end())
        it = specs.emplace(std::make_tuple(gelu, bits, group),
                           QLinearSpec{QLINEAR_MAGIC, gelu, bits, (int32_t) group}).first;
    return &it->second;
}

inline int64_t qlinear_k(const ggml_tensor * w, int bits) {
    return bits == 4 ? 2 * w->ne[0] : w->ne[0];
}

using QWeight = WeightLoader::QWeight;

// The quantization the SYCL kernel applies to an activation row, for the
// reference below: scale = max|x| / 127, q = round(x / scale).
inline float qlinear_quant_row(const ggml_tensor * x, const char * row, int64_t K, std::vector<int8_t> & q) {
    float amax = 0.0f;
    for (int64_t k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(fused_load(x, row, k)));
    const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
    q.resize((size_t) K);
    for (int64_t k = 0; k < K; ++k)
        q[k] = (int8_t) std::lrint(std::fmin(127.0f, std::fmax(-127.0f, fused_load(x, row, k) / sc)));
    return sc;
}

// srcs: W codes I8 [K, N], W scales F32 [N], x [K, M] (F32/BF16) or a q8 blob,
// bias F32 [N] or null, res [N, M] or null, gate F32 [N] or null.
inline void qlinear_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto & s = *static_cast<const QLinearSpec *>(ud);
    const ggml_tensor * W  = dst->src[0];
    const float *       sw = (const float *) dst->src[1]->data;
    const ggml_tensor * x  = dst->src[2];
    const ggml_tensor * b  = dst->src[3];
    const ggml_tensor * r  = dst->src[4];
    const float *       g  = dst->src[5] ? (const float *) dst->src[5]->data : nullptr;
    const int64_t K = qlinear_k(W, s.bits), N = W->ne[1];
    const int64_t G = s.bits == 4 ? s.group : K;
    const bool    blob = x->type == GGML_TYPE_I8;
    const int64_t M = blob ? q8_rows(x, K) : x->ne[1];
    std::vector<int8_t> qx;
    for (int64_t m = ith; m < M; m += nth) {
        float sx;
        const int8_t * xq;
        if (blob) {
            xq = (const int8_t *) x->data + m * K;
            sx = ((const float *) ((const int8_t *) x->data + M * K))[m];
        } else {
            sx = qlinear_quant_row(x, (const char *) x->data + m * x->nb[1], K, qx);
            xq = qx.data();
        }
        char * dr = (char *) dst->data + m * dst->nb[1];
        for (int64_t n = 0; n < N; ++n) {
            float v = 0.0f;
            if (s.bits == 4) {
                const uint8_t * wr = (const uint8_t *) W->data + n * K / 2;
                for (int64_t k0 = 0; k0 < K; k0 += G) {
                    int32_t acc = 0;
                    for (int64_t k = k0; k < k0 + G; ++k) {
                        const int c = (wr[k / 2] >> (4 * (k & 1))) & 0xf;
                        acc += (int32_t) xq[k] * (c >= 8 ? c - 16 : c);
                    }
                    v += (float) acc * sw[(k0 / G) * N + n];
                }
                v *= sx;
            } else {
                const int8_t * wr = (const int8_t *) W->data + n * K;
                int32_t acc = 0;
                for (int64_t k = 0; k < K; ++k) acc += (int32_t) xq[k] * wr[k];
                v = (float) acc * sx * sw[n];
            }
            if (b) v += ((const float *) b->data)[n];
            if (s.gelu) v = fused_gelu_tanh(v);
            if (g) v *= g[n];
            if (r) v += fused_load(r, (const char *) r->data + m * r->nb[1], n);
            fused_store(dst, dr, n, v);
        }
    }
}

// x: [K, M] F32 or BF16 with dense rows, or a q8 blob of K-wide rows. gate, if
// given, scales the result per output channel before res is added (pi0.5's
// gated residual: res + out * gate). Returns [N, M] of out_type.
inline ggml_tensor * qlinear(ggml_context * C, const QWeight & q, ggml_tensor * x, ggml_tensor * bias,
                             ggml_tensor * res, ggml_type out_type, bool gelu = false, ggml_tensor * gate = nullptr) {
    const int64_t K = qlinear_k(q.w, q.bits);
    const bool    blob = x->type == GGML_TYPE_I8;
    GGML_ASSERT(blob ? x->ne[0] % (K + 4) == 0
                     : (K == x->ne[0] && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == ggml_type_size(x->type)));
    GGML_ASSERT(!gate || (gate->type == GGML_TYPE_F32 && gate->ne[0] == q.w->ne[1]));
    const int64_t M = blob ? q8_rows(x, K) : x->ne[1];
    ggml_tensor * args[6] = { q.w, q.s, x, bias, res, gate };
    const int n = gate ? 6 : res ? 5 : 4;
    ggml_tensor * t = ggml_custom_4d(C, out_type, q.w->ne[1], M, 1, 1, args, n, qlinear_cpu, GGML_N_TASKS_MAX,
                                     (void *) qlinear_spec(gelu, q.bits, q.group));
    ggml_format_name(t, "%s.q8", ggml_get_name(q.w));
    return t;
}

}  // namespace vla
