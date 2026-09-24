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

// FoldQuant reference bodies. This translation unit is compiled with
// -ffp-contract=off (CMakeLists.txt); see foldquant_ref.h for why that matters.

#include "foldquant_ref.h"
#include "env_flag.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace vla {

static bool fq_stats_enabled() {
    static const bool on = env_flag("VLA_FQ_STATS");
    return on;
}

namespace fqref {

// The kernel runs one warp per row: lane l owns the 64-element chunks
// c = l, l+32, l+64, ... of the row (K is a multiple of 64), accumulates its
// partial sum of squares over them in element order, and the 32 partials are
// combined by an xor-shuffle butterfly. NT is the number of partials.

// Kernel-order reduction of the 32 lane partials. Clobbers p.
float block_sum(float * p) {
    float t[32];
    for (int off = 16; off > 0; off >>= 1) {
        for (int l = 0; l < 32; ++l) t[l] = p[l] + p[l ^ off];
        std::memcpy(p, t, sizeof(t));
    }
    return p[0];
}

// In-place natural-order Sylvester-Hadamard butterfly on every bs-block of a
// row, normalised by inv_sqrt_bs = 1/sqrt(bs) afterwards (same stage order and
// pairing as VLA-OPT fwht.cuh::block_fwht_smem).
void fwht_row(float * y, int64_t K, int bs, float inv_sqrt_bs) {
    for (int64_t b = 0; b < K; b += bs) {
        float * v = y + b;
        for (int h = 1; h < bs; h <<= 1)
            for (int i = 0; i < bs; i += 2*h)
                for (int j = i; j < i + h; ++j) {
                    const float a = v[j], c = v[j + h];
                    v[j]     = a + c;
                    v[j + h] = a - c;
                }
        for (int j = 0; j < bs; ++j) v[j] = v[j] * inv_sqrt_bs;
    }
}

float inv_sqrt_block(int bs) {
    return (float) (1.0 / std::sqrt((double) bs));
}

float qmax_for(int bits) {
    return bits == 4 ? 7.0f : 127.0f;
}

// One activation row: x[K] -> codes + scale. tmp and partial are scratch
// (K floats and NT floats).
void act_row(const float * x, const float * ascale, const float * gamma, const FqActSpec & s,
                    float * tmp, float * partial, int8_t * codes_out, float * scale_out) {
    const int64_t K = s.K;

    if (s.has_gamma && gamma) {
        for (int t = 0; t < NT; ++t) partial[t] = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            const float v = x[k];
            const int   lane = (int) ((k / CHUNK) % NT);
            partial[lane] = partial[lane] + v * v;
        }
        const float sumsq = block_sum(partial);
        const float rstd  = 1.0f / std::sqrt(sumsq / (float) K + s.eps);
        for (int64_t k = 0; k < K; ++k) tmp[k] = (x[k] * rstd) * gamma[k];
    } else {
        for (int64_t k = 0; k < K; ++k) tmp[k] = x[k];
    }

    if (ascale && s.fold_before)
        for (int64_t k = 0; k < K; ++k) tmp[k] = tmp[k] / ascale[k];

    if (s.rot_block > 1)
        fwht_row(tmp, K, s.rot_block, inv_sqrt_block(s.rot_block));

    if (ascale && !s.fold_before)
        for (int64_t k = 0; k < K; ++k) tmp[k] = tmp[k] / ascale[k];

    float amax = 0.0f;
    for (int64_t k = 0; k < K; ++k) amax = std::max(amax, std::fabs(tmp[k]));

    const float qmax  = qmax_for(s.abits);
    float       scale = (s.clip * amax) / qmax;
    if (scale < 1e-12f) scale = 1e-12f;
    *scale_out = scale;
    // Multiply by the reciprocal rather than divide: what VLA-OPT's TensorRT
    // kernels do, so the deployed engines and this runtime round alike.
    const float inv = 1.0f / scale;

    if (s.abits == 4) {
        for (int64_t k = 0; k < K; k += 2) {
            float q0 = std::nearbyint(tmp[k]     * inv);
            float q1 = std::nearbyint(tmp[k + 1] * inv);
            q0 = std::min(qmax, std::max(-qmax, q0));
            q1 = std::min(qmax, std::max(-qmax, q1));
            const int i0 = (int) q0, i1 = (int) q1;
            codes_out[k / 2] = (int8_t) ((i0 & 0xF) | ((i1 & 0xF) << 4));
        }
    } else {
        for (int64_t k = 0; k < K; ++k) {
            float q = std::nearbyint(tmp[k] * inv);
            q = std::min(qmax, std::max(-qmax, q));
            codes_out[k] = (int8_t) (int) q;
        }
    }
}

// Unpack a nibble row (low nibble = even column, two's complement) to int8.
void unpack_nibbles(const int8_t * packed, int64_t K, int8_t * out) {
    for (int64_t k = 0; k < K; k += 2) {
        const uint8_t b = (uint8_t) packed[k / 2];
        out[k]     = (int8_t) ((int8_t) (b << 4) >> 4);
        out[k + 1] = (int8_t) ((int8_t) b >> 4);
    }
}

// y[n] for one (token, output row): exact int32 accumulation, then the
// kernel's epilogue order.
float gemm_dot(const int8_t * w_row, const int8_t * x_row, int64_t K, float xs, float ws, float bias) {
    int32_t acc = 0;
    for (int64_t k = 0; k < K; ++k) acc += (int32_t) w_row[k] * (int32_t) x_row[k];
    float v = ((float) acc * xs) * ws;
    return v + bias;
}

}  // namespace fqref

// GGML_OP_CUSTOM entry points. Sources are packed without holes:
//   fq_act : src[0]=x F32 [K, T...] contiguous, then gamma F32[K] if has_gamma,
//            then ascale F32[K] if has_ascale; dst I8 [row_bytes, T]
//   fq_gemm: src[0]=w I8 [K_pack, N], src[1]=xq blob, src[2]=wscale F32[N], src[3]=bias F32[N] or null,
//            src[4]=residual F32 [N, T] or absent (the residual add fused into the epilogue); dst F32 [N, T]

void fq_act_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const FqActSpec & s = *(const FqActSpec *) userdata;
    if (s.magic != FQ_ACT_MAGIC) return;

    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * g = nullptr, * as = nullptr;
    fq_act_srcs(dst, s, &g, &as);

    const int64_t K   = s.K;
    const int64_t T   = dst->ne[1];
    const int64_t rb  = fq_act_row_bytes(K, s.abits);
    const int64_t kp  = fq_act_kpack(K, s.abits);
    const int64_t per = (T + nth - 1) / nth;

    // VLA_FQ_STATS=1: one line per node with checksums of every input, so two
    // backends running this same reference can be diffed node by node.
    if (ith == 0 && fq_stats_enabled()) {
        double sum = 0.0, sumsq = 0.0, mx = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            const float * xr = (const float *) ((const char *) x->data + t * x->nb[1]);
            for (int64_t k = 0; k < K; ++k) { const double v = xr[k]; sum += v; sumsq += v*v; mx = std::fmax(mx, std::fabs(v)); }
        }
        double gs = 0.0, as_ = 0.0;
        if (g)  for (int64_t k = 0; k < K; ++k) gs  += ((const float *) g->data)[k];
        if (as) for (int64_t k = 0; k < K; ++k) as_ += ((const float *) as->data)[k];
        std::fprintf(stderr, "FQSTAT act  %-44s T=%lld x sum=%.9g sumsq=%.9g max=%.9g gamma=%.9g ascale=%.9g\n",
                     ggml_get_name(dst), (long long) T, sum, sumsq, mx, gs, as_);
        // VLA_FQ_DUMP=<dir>: also write this node's input rows as raw F32 so two
        // backends can be compared element by element (first occurrence only).
        if (const char * dir = std::getenv("VLA_FQ_DUMP")) {
            static int n_dumped = 0;
            if (n_dumped < 64) {
                char path[1024];
                std::snprintf(path, sizeof path, "%s/%s.%d.f32", dir, ggml_get_name(dst), n_dumped++);
                if (FILE * f = std::fopen(path, "wb")) {
                    for (int64_t t = 0; t < T; ++t)
                        std::fwrite((const char *) x->data + t * x->nb[1], sizeof(float), (size_t) K, f);
                    std::fclose(f);
                }
            }
        }
    }
    const int64_t t0  = (int64_t) ith * per;
    const int64_t t1  = std::min(T, t0 + per);

    const float * xp = (const float *) x->data;
    const float * ap = as ? (const float *) as->data : nullptr;
    const float * gp = g  ? (const float *) g->data  : nullptr;

    std::vector<float> tmp((size_t) K);
    float partial[fqref::NT];
    for (int64_t t = t0; t < t1; ++t) {
        uint8_t * row = (uint8_t *) dst->data + (size_t) t * rb;
        fqref::act_row((const float *) ((const char *) xp + (size_t) t * x->nb[1]), ap, gp, s, tmp.data(), partial,
                       (int8_t *) row, (float *) (row + kp));
    }
}

// Where output element (row t, column n) lands: plain row-major, or the head
// layout described in FqGemmSpec (same rule as the CUDA epilogues).
static inline size_t fq_out_index(const FqGemmSpec & s, int64_t T, int64_t t, int64_t n) {
    if (!s.heads) return (size_t) t * s.N + n;
    const int64_t dim = (int64_t) s.head_dim * s.heads;
    const int64_t p = n / dim, r = n - p * dim, h = r / s.head_dim, d = r - h * s.head_dim;
    const int64_t base = p * dim * T;
    return (size_t) (((s.vmask >> p) & 1) ? base + (h * s.head_dim + d) * T + t
                                          : base + (h * T + t) * s.head_dim + d);
}

void fq_gemm_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const FqGemmSpec & s = *(const FqGemmSpec *) userdata;
    if (s.magic != FQ_GEMM_MAGIC) return;

    const ggml_tensor * w  = dst->src[0];
    const ggml_tensor * xq = dst->src[1];
    const ggml_tensor * ws = dst->src[2];
    const ggml_tensor * b  = dst->src[3];
    const ggml_tensor * r  = dst->src[4];          // optional residual [N, T]

    const int64_t K     = s.K;
    const int64_t N     = s.N;
    const int64_t T     = dst->ne[1];
    const int     abits = (xq->ne[0] == K + FQ_ACT_TAIL) ? 8 : 4;
    const int64_t rb    = fq_act_row_bytes(K, abits);
    const int64_t kp_a  = fq_act_kpack(K, abits);
    const int64_t kp_w  = fq_w_kpack(K, s.wbits);

    if (ith == 0 && fq_stats_enabled()) {
        long long csum = 0; double ssum = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            const int8_t * row = (const int8_t *) xq->data + t * rb;
            for (int64_t k = 0; k < kp_a; ++k) csum += row[k];
            float sc; std::memcpy(&sc, row + kp_a, sizeof sc); ssum += sc;
        }
        long long wsum = 0;
        for (int64_t i = 0; i < (int64_t) N * kp_w; ++i) wsum += ((const int8_t *) w->data)[i];
        double wss = 0.0, bs = 0.0;
        for (int64_t n = 0; n < N; ++n) wss += ((const float *) ws->data)[n];
        if (b) for (int64_t n = 0; n < N; ++n) bs += ((const float *) b->data)[n];
        std::fprintf(stderr, "FQSTAT gemm %-44s T=%lld codes=%lld scales=%.9g w=%lld wscale=%.9g bias=%.9g\n",
                     ggml_get_name(dst), (long long) T, csum, ssum, wsum, wss, bs);
    }

    const int8_t * wp = (const int8_t *) w->data;
    const uint8_t * xp = (const uint8_t *) xq->data;
    const float * wsp = (const float *) ws->data;
    const float * bp  = b ? (const float *) b->data : nullptr;
    const float * rp  = r ? (const float *) r->data : nullptr;
    float * y = (float *) dst->data;

    // Activations unpacked once per thread when they are nibbles.
    std::vector<int8_t> xa;
    if (abits == 4) {
        xa.resize((size_t) T * K);
        for (int64_t t = 0; t < T; ++t)
            fqref::unpack_nibbles((const int8_t *) (xp + (size_t) t * rb), K, xa.data() + (size_t) t * K);
    }
    std::vector<int8_t> wrow_buf(s.wbits == 4 ? (size_t) K : 0);

    const int64_t per = (N + nth - 1) / nth;
    const int64_t n0  = (int64_t) ith * per;
    const int64_t n1  = std::min(N, n0 + per);
    for (int64_t n = n0; n < n1; ++n) {
        const int8_t * wrow = wp + (size_t) n * kp_w;
        if (s.wbits == 4) {
            fqref::unpack_nibbles(wrow, K, wrow_buf.data());
            wrow = wrow_buf.data();
        }
        const float wsn = wsp[n];
        const float bn  = bp ? bp[n] : 0.0f;
        for (int64_t t = 0; t < T; ++t) {
            const uint8_t * row = xp + (size_t) t * rb;
            const int8_t * xrow = (abits == 4) ? xa.data() + (size_t) t * K : (const int8_t *) row;
            float xs;
            std::memcpy(&xs, row + kp_a, sizeof(float));
            float v = fqref::gemm_dot(wrow, xrow, K, xs, wsn, bn);
            if (rp) v = v + rp[(size_t) t * N + n];
            y[fq_out_index(s, T, t, n)] = v;
        }
    }
}

}  // namespace vla
