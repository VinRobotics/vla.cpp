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

// FoldQuant CPU reference ops, run as GGML_OP_CUSTOM nodes on the CPU backend
// with several threads, checked three ways:
//   1. bit-identical to the reference functions called row by row (graph
//      plumbing, src slots, thread partitioning);
//   2. within +-1 code of an independent double-precision dense-Hadamard
//      implementation (the math itself);
//   3. a golden FNV-1a checksum of the codes for the fixed LCG input, shared
//      with tests/py/test_foldquant_ref.py so the numpy reference and the C++
//      one cannot drift apart.

#include "foldquant.h"
#include "foldquant_ref.h"
#include "layers/fq_linear.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    // Same generator as scripts/foldquant_ref.py: value in [-1, 1].
    float next() {
        s = s * 1664525u + 1013904223u;
        return ((float) ((s >> 8) & 0xFFFFu) / 65535.0f) * 2.0f - 1.0f;
    }
};

uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

struct Case {
    const char * name;
    int  wbits, abits, rot;
    bool gamma, ascale, fold_before, bias;
};

constexpr int64_t K = 128, N = 64, T = 5;

// FNV-1a of the INT8 codes of the first case (no norm, rot 64) for the LCG input.
constexpr uint64_t FQ_GOLDEN_FNV = 0xb97c1c173d2d0278ull;

// Independent double-precision reference of the activation transform.
void ref_act_double(const std::vector<float> & x, const std::vector<float> & as, const std::vector<float> & ga,
                    const vla::FqActSpec & s, std::vector<int> & codes, std::vector<double> & scales) {
    codes.assign((size_t) T * K, 0);
    scales.assign((size_t) T, 0.0);
    const int bs = s.rot_block;
    std::vector<double> H((size_t) bs * bs, 0.0);
    for (int i = 0; i < bs; ++i)
        for (int j = 0; j < bs; ++j) {
            int bits = i & j, par = 0;
            while (bits) { par ^= bits & 1; bits >>= 1; }
            H[(size_t) i * bs + j] = (par ? -1.0 : 1.0) / std::sqrt((double) bs);
        }
    const double qmax = s.abits == 4 ? 7.0 : 127.0;
    for (int64_t t = 0; t < T; ++t) {
        std::vector<double> y(K);
        for (int64_t k = 0; k < K; ++k) y[k] = x[(size_t) t * K + k];
        if (s.has_gamma) {
            double ss = 0.0;
            for (int64_t k = 0; k < K; ++k) ss += y[k] * y[k];
            const double rstd = 1.0 / std::sqrt(ss / (double) K + (double) s.eps);
            for (int64_t k = 0; k < K; ++k) y[k] = y[k] * rstd * ga[k];
        }
        if (!as.empty() && s.fold_before) for (int64_t k = 0; k < K; ++k) y[k] /= as[k];
        if (bs > 1) {
            std::vector<double> z(K);
            for (int64_t b = 0; b < K; b += bs)
                for (int i = 0; i < bs; ++i) {
                    double acc = 0.0;
                    for (int j = 0; j < bs; ++j) acc += H[(size_t) i * bs + j] * y[b + j];
                    z[b + i] = acc;
                }
            y = z;
        }
        if (!as.empty() && !s.fold_before) for (int64_t k = 0; k < K; ++k) y[k] /= as[k];
        double amax = 0.0;
        for (int64_t k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(y[k]));
        double scale = (double) s.clip * amax / qmax;
        if (scale < 1e-12) scale = 1e-12;
        scales[t] = scale;
        for (int64_t k = 0; k < K; ++k) {
            double q = std::nearbyint(y[k] / scale);
            q = std::fmin(qmax, std::fmax(-qmax, q));
            codes[(size_t) t * K + k] = (int) q;
        }
    }
}

int run_case(ggml_backend_t backend, const Case & c, bool print_hash) {
    std::printf("case %-22s W%dA%d rot%-3d gamma=%d ascale=%d before=%d bias=%d\n",
                c.name, c.wbits, c.abits, c.rot, c.gamma, c.ascale, c.fold_before, c.bias);

    Lcg rng(0x5eed1234u);
    std::vector<float> hx((size_t) K * T), hw((size_t) K * N), hws(N), hb(N), has(K), hga(K);
    for (auto & v : hx)  v = rng.next() * 4.0f;
    for (auto & v : hw)  v = rng.next();
    for (auto & v : hws) v = 0.01f + 0.02f * std::fabs(rng.next());
    for (auto & v : hb)  v = rng.next() * 0.5f;
    for (auto & v : has) v = 0.5f + std::fabs(rng.next());
    for (auto & v : hga) v = 0.75f + 0.5f * std::fabs(rng.next());

    // Weight codes: per-output-row symmetric of the float w (rounded here; the
    // GEMM only needs *some* valid codes).
    const float wq = c.wbits == 4 ? 7.0f : 127.0f;
    std::vector<int8_t> wcodes((size_t) K * N);
    for (int64_t n = 0; n < N; ++n) {
        float amax = 0.f;
        for (int64_t k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(hw[(size_t) n * K + k]));
        const float sc = amax / wq;
        for (int64_t k = 0; k < K; ++k)
            wcodes[(size_t) n * K + k] = (int8_t) std::nearbyint(hw[(size_t) n * K + k] / sc);
    }
    const int64_t kpw = vla::fq_w_kpack(K, c.wbits);
    std::vector<int8_t> wpacked((size_t) kpw * N);
    if (c.wbits == 4) {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; k += 2)
                wpacked[(size_t) n * kpw + k / 2] = (int8_t) ((wcodes[(size_t) n * K + k] & 0xF) | ((wcodes[(size_t) n * K + k + 1] & 0xF) << 4));
    } else {
        wpacked = wcodes;
    }

    ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, false };
    ggml_context * C = ggml_init(ip);

    ggml_tensor * x  = ggml_new_tensor_2d(C, GGML_TYPE_F32, K, T);
    ggml_tensor * w  = ggml_new_tensor_2d(C, GGML_TYPE_I8, kpw, N);
    ggml_tensor * ws = ggml_new_tensor_1d(C, GGML_TYPE_F32, N);
    ggml_tensor * b  = c.bias   ? ggml_new_tensor_1d(C, GGML_TYPE_F32, N) : nullptr;
    ggml_tensor * as = c.ascale ? ggml_new_tensor_1d(C, GGML_TYPE_F32, K) : nullptr;
    ggml_tensor * ga = c.gamma  ? ggml_new_tensor_1d(C, GGML_TYPE_F32, K) : nullptr;
    ggml_set_name(w, "site");
    std::memcpy(x->data, hx.data(), ggml_nbytes(x));
    std::memcpy(w->data, wpacked.data(), ggml_nbytes(w));
    std::memcpy(ws->data, hws.data(), ggml_nbytes(ws));
    if (b)  std::memcpy(b->data,  hb.data(),  ggml_nbytes(b));
    if (as) std::memcpy(as->data, has.data(), ggml_nbytes(as));
    if (ga) std::memcpy(ga->data, hga.data(), ggml_nbytes(ga));

    vla::FqLinear s;
    s.w = w; s.wscale = ws; s.bias = b; s.ascale = as; s.gamma = ga;
    s.act.K = K; s.act.abits = c.abits; s.act.rot_block = c.rot; s.act.fold_before = c.fold_before;
    s.act.has_gamma = c.gamma; s.act.has_ascale = c.ascale; s.act.clip = c.abits == 4 ? 0.9f : 1.0f; s.act.eps = 1e-6f;
    s.gemm.K = K; s.gemm.N = N; s.gemm.wbits = c.wbits;

    ggml_tensor * xq = vla::fq_act(C, s, x);
    ggml_tensor * y  = vla::fq_gemm(C, s, xq);
    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, y);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: graph compute\n");
        return 1;
    }

    // 1. bit-identical to the reference functions.
    const int64_t rb = vla::fq_act_row_bytes(K, c.abits);
    const int64_t kp = vla::fq_act_kpack(K, c.abits);
    std::vector<uint8_t> blob((size_t) rb * T);
    {
        std::vector<float> tmp(K); float partial[vla::fqref::NT];
        for (int64_t t = 0; t < T; ++t) {
            uint8_t * row = blob.data() + (size_t) t * rb;
            std::memset(row, 0, rb);
            vla::fqref::act_row(hx.data() + (size_t) t * K, as ? has.data() : nullptr, ga ? hga.data() : nullptr,
                                s.act, tmp.data(), partial, (int8_t *) row, (float *) (row + kp));
        }
    }
    for (int64_t t = 0; t < T; ++t) {
        const uint8_t * got = (const uint8_t *) xq->data + (size_t) t * rb;
        if (std::memcmp(got, blob.data() + (size_t) t * rb, kp + 4) != 0) {
            std::printf("FAIL: row %lld of the act blob differs from the reference\n", (long long) t);
            return 1;
        }
    }
    std::vector<int8_t> xcodes((size_t) T * K);
    for (int64_t t = 0; t < T; ++t) {
        const int8_t * row = (const int8_t *) xq->data + (size_t) t * rb;
        if (c.abits == 4) vla::fqref::unpack_nibbles(row, K, xcodes.data() + (size_t) t * K);
        else std::memcpy(xcodes.data() + (size_t) t * K, row, K);
    }
    for (int64_t t = 0; t < T; ++t) {
        float xs; std::memcpy(&xs, (const uint8_t *) xq->data + (size_t) t * rb + kp, 4);
        for (int64_t n = 0; n < N; ++n) {
            const float want = vla::fqref::gemm_dot(wcodes.data() + (size_t) n * K, xcodes.data() + (size_t) t * K,
                                                    K, xs, hws[n], b ? hb[n] : 0.0f);
            const float got  = ((const float *) y->data)[(size_t) t * N + n];
            if (std::memcmp(&want, &got, 4) != 0) {
                std::printf("FAIL: y[%lld][%lld] = %g, reference %g\n", (long long) t, (long long) n, got, want);
                return 1;
            }
        }
    }

    // 2. the math, against double precision.
    std::vector<int> dcodes; std::vector<double> dscales;
    ref_act_double(hx, as ? has : std::vector<float>(), hga, s.act, dcodes, dscales);
    int off_by_one = 0;
    for (int64_t t = 0; t < T; ++t) {
        float xs; std::memcpy(&xs, (const uint8_t *) xq->data + (size_t) t * rb + kp, 4);
        if (std::fabs((double) xs - dscales[t]) > 1e-5 * dscales[t]) {
            std::printf("FAIL: row %lld scale %g vs double %g\n", (long long) t, xs, dscales[t]);
            return 1;
        }
        for (int64_t k = 0; k < K; ++k) {
            const int d = std::abs((int) xcodes[(size_t) t * K + k] - dcodes[(size_t) t * K + k]);
            if (d > 1) { std::printf("FAIL: code [%lld][%lld] off by %d\n", (long long) t, (long long) k, d); return 1; }
            off_by_one += d;
        }
    }
    if (off_by_one > (int) (T * K) / 100) {
        std::printf("FAIL: %d codes differ from the double reference by one\n", off_by_one);
        return 1;
    }

    if (print_hash)
        std::printf("  codes fnv1a = 0x%016llx\n", (unsigned long long) fnv1a((const uint8_t *) xcodes.data(), xcodes.size()));
    std::printf("  ok (%d codes off by one vs double)\n", off_by_one);
    ggml_free(C);
    return 0;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);

    const Case cases[] = {
        { "w8a8_rot64",          8, 8, 64, false, false, false, true  },
        { "w8a8_rot64_gamma",    8, 8, 64, true,  false, false, false },
        { "w8a8_rot32_pre",      8, 8, 32, false, true,  true,  true  },
        { "w8a8_rot64_post",     8, 8, 64, false, true,  false, false },
        { "w8a8_norot",          8, 8, 1,  false, false, false, false },
        { "w4a8_rot64_gamma",    4, 8, 64, true,  false, false, true  },
        { "w4a4_rot64_pre",      4, 4, 64, false, true,  true,  true  },
    };
    int i = 0;
    for (const Case & c : cases) {
        if (run_case(backend, c, i == 0)) return 1;
        ++i;
    }

    // 3. golden checksum of the first case (no gamma: its reduction order is the
    // one thing numpy cannot mirror exactly).
    {
        Lcg rng(0x5eed1234u);
        std::vector<float> hx((size_t) K * T);
        for (auto & v : hx) v = rng.next() * 4.0f;
        vla::FqActSpec a; a.K = K; a.abits = 8; a.rot_block = 64;
        std::vector<int8_t> codes((size_t) T * K);
        std::vector<float> tmp(K); float partial[vla::fqref::NT]; float sc;
        for (int64_t t = 0; t < T; ++t)
            vla::fqref::act_row(hx.data() + (size_t) t * K, nullptr, nullptr, a, tmp.data(), partial,
                                codes.data() + (size_t) t * K, &sc);
        const uint64_t h = fnv1a((const uint8_t *) codes.data(), codes.size());
        const uint64_t golden = FQ_GOLDEN_FNV;   // pinned by tests/py/test_foldquant_ref.py too
        if (h != golden) {
            std::printf("FAIL: golden checksum 0x%016llx, expected 0x%016llx\n",
                        (unsigned long long) h, (unsigned long long) golden);
            return 1;
        }
    }

    std::printf("PASS\n");
    ggml_backend_free(backend);
    return 0;
}
