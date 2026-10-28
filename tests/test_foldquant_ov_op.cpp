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

// FoldQuant's two custom nodes on ggml's OpenVINO backend (translated by
// src/openvino/foldquant_ov.cpp) against the CPU reference (foldquant_ref.h):
// the cases of test_foldquant_cpu_op.cpp, with the site's tensors in a weights
// buffer as a real model has them, on the device GGML_OPENVINO_DEVICE names.
//
// Codes are computed in a different float order than the reference (OpenVINO's
// own reductions), so a value sitting on a rounding boundary can land one code
// over; the output must match the reference wherever every code agrees and may
// differ by one activation step where one does not. Built only with -DGGML_OPENVINO=ON.

#include "foldquant.h"
#include "foldquant_ref.h"
#include "layers/fq_linear.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-openvino.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return ((float) ((s >> 8) & 0xFFFFu) / 65535.0f) * 2.0f - 1.0f;
    }
};

struct Case {
    const char * name;
    int  wbits, abits, rot;
    bool gamma, ascale, fold_before, bias;
};

constexpr int64_t K = 128, N = 64, T = 5;

int run_case(ggml_backend_t backend, const Case & c) {
    Lcg rng(0x5eed1234u);
    std::vector<float> hx((size_t) K * T), hw((size_t) K * N), hws(N), hb(N), has(K), hga(K);
    for (auto & v : hx)  v = rng.next() * 4.0f;
    for (auto & v : hw)  v = rng.next();
    for (auto & v : hws) v = 0.01f + 0.02f * std::fabs(rng.next());
    for (auto & v : hb)  v = rng.next() * 0.5f;
    for (auto & v : has) v = 0.5f + std::fabs(rng.next());
    for (auto & v : hga) v = 0.75f + 0.5f * std::fabs(rng.next());

    const float wq = c.wbits == 4 ? 7.0f : 127.0f;
    std::vector<int8_t> wcodes((size_t) K * N);
    for (int64_t n = 0; n < N; ++n) {
        float amax = 0.f;
        for (int64_t k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(hw[(size_t) n * K + k]));
        for (int64_t k = 0; k < K; ++k)
            wcodes[(size_t) n * K + k] = (int8_t) std::nearbyint(hw[(size_t) n * K + k] / (amax / wq));
    }
    const int64_t kpw = vla::fq_w_kpack(K, c.wbits);
    std::vector<int8_t> wpacked((size_t) kpw * N);
    if (c.wbits == 4) {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; k += 2)
                wpacked[(size_t) n * kpw + k / 2] =
                    (int8_t) ((wcodes[(size_t) n * K + k] & 0xF) | ((wcodes[(size_t) n * K + k + 1] & 0xF) << 4));
    } else {
        wpacked = wcodes;
    }

    // The site's tensors live in a weights buffer, as a model's do: that is what
    // makes the backend turn them into constants.
    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * W = ggml_init(wp);
    ggml_tensor * w  = ggml_new_tensor_2d(W, GGML_TYPE_I8, kpw, N);
    ggml_tensor * ws = ggml_new_tensor_1d(W, GGML_TYPE_F32, N);
    ggml_tensor * b  = c.bias   ? ggml_new_tensor_1d(W, GGML_TYPE_F32, N) : nullptr;
    ggml_tensor * as = c.ascale ? ggml_new_tensor_1d(W, GGML_TYPE_F32, K) : nullptr;
    ggml_tensor * ga = c.gamma  ? ggml_new_tensor_1d(W, GGML_TYPE_F32, K) : nullptr;
    ggml_set_name(w, "site.weight"); ggml_set_name(ws, "site.wscale");
    if (b) ggml_set_name(b, "site.bias");
    if (as) ggml_set_name(as, "site.ascale");
    if (ga) ggml_set_name(ga, "site.gamma");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(W, backend);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w, wpacked.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(ws, hws.data(), 0, ggml_nbytes(ws));
    if (b)  ggml_backend_tensor_set(b,  hb.data(),  0, ggml_nbytes(b));
    if (as) ggml_backend_tensor_set(as, has.data(), 0, ggml_nbytes(as));
    if (ga) ggml_backend_tensor_set(ga, hga.data(), 0, ggml_nbytes(ga));

    vla::FqLinear s;
    s.w = w; s.wscale = ws; s.bias = b; s.ascale = as; s.gamma = ga;
    s.act.K = K; s.act.abits = c.abits; s.act.rot_block = c.rot; s.act.fold_before = c.fold_before;
    s.act.has_gamma = c.gamma; s.act.has_ascale = c.ascale; s.act.clip = c.abits == 4 ? 0.9f : 1.0f; s.act.eps = 1e-6f;
    s.gemm.K = K; s.gemm.N = N; s.gemm.wbits = c.wbits;

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * C = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(C, GGML_TYPE_F32, K, T);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * y = vla::fq_gemm(C, s, vla::fq_act(C, s, x));
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t ga_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga_alloc, gf)) { std::printf("FAIL: alloc\n"); return 1; }
    ggml_backend_tensor_set(x, hx.data(), 0, ggml_nbytes(x));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: %s: graph compute\n", c.name);
        return 1;
    }
    std::vector<float> got((size_t) N * T);
    ggml_backend_tensor_get(y, got.data(), 0, ggml_nbytes(y));

    // CPU reference: the activation codes, then the integer GEMM.
    int rows_exact = 0, rows_off = 0;
    double max_rel = 0.0;
    std::vector<float> tmp(K);
    float partial[vla::fqref::NT];
    const int64_t kp = vla::fq_act_kpack(K, c.abits);
    std::vector<int8_t> blob((size_t) kp), codes((size_t) K);
    for (int64_t t = 0; t < T; ++t) {
        float xs;
        vla::fqref::act_row(hx.data() + (size_t) t * K, as ? has.data() : nullptr, ga ? hga.data() : nullptr,
                            s.act, tmp.data(), partial, blob.data(), &xs);
        if (c.abits == 4) vla::fqref::unpack_nibbles(blob.data(), K, codes.data());
        else codes = blob;
        double row_err = 0.0, row_ref = 0.0;
        for (int64_t n = 0; n < N; ++n) {
            const float want = vla::fqref::gemm_dot(wcodes.data() + (size_t) n * K, codes.data(), K, xs, hws[n],
                                                    b ? hb[n] : 0.0f);
            row_err = std::fmax(row_err, std::fabs((double) got[(size_t) t * N + n] - want));
            row_ref = std::fmax(row_ref, std::fabs((double) want));
        }
        const double rel = row_err / std::fmax(row_ref, 1e-6);
        max_rel = std::fmax(max_rel, rel);
        // Every code equal: float accumulation differences only. One code over on a
        // rounding boundary moves the row by at most |w| * xs * wscale per output.
        if (rel <= 1e-4) ++rows_exact; else ++rows_off;
    }
    const bool ok = rows_off <= 1 && max_rel <= (c.abits == 4 ? 0.05 : 0.01);
    std::printf("%s case %-20s W%dA%d rot%-2d gamma=%d ascale=%d before=%d bias=%d: %d/%lld rows exact, max rel %.2e\n",
                ok ? "ok  " : "FAIL", c.name, c.wbits, c.abits, c.rot, c.gamma, c.ascale, c.fold_before, c.bias,
                rows_exact, (long long) T, max_rel);
    ggml_gallocr_free(ga_alloc);
    ggml_free(C);
    ggml_backend_buffer_free(wbuf);
    ggml_free(W);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_openvino_init(0);
    if (!backend) {
        std::printf("SKIP: no OpenVINO backend\n");
        return 0;
    }
    const Case cases[] = {
        { "w8a8_rot64",          8, 8, 64, false, false, false, true  },
        { "w8a8_rot64_gamma",    8, 8, 64, true,  false, false, false },
        { "w8a8_rot32_pre",      8, 8, 32, false, true,  true,  true  },
        { "w8a8_rot64_post",     8, 8, 64, false, true,  false, false },
        { "w8a8_norot",          8, 8, 1,  false, false, false, false },
        { "w4a8_rot64_gamma",    4, 8, 64, true,  false, false, true  },
        { "w4a4_rot64_pre",      4, 4, 64, false, true,  true,  true  },
    };
    int fails = 0;
    for (const Case & c : cases)
        fails += run_case(backend, c);
    std::printf("%s\n", fails ? "test_foldquant_ov_op: FAILED" : "test_foldquant_ov_op: PASS");
    ggml_backend_free(backend);
    return fails ? 1 : 0;
}
