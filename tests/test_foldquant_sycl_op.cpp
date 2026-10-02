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

// FoldQuant's two custom nodes on ggml's SYCL backend (src/sycl/
// vla_sycl_foldquant.cpp) against the CPU reference (foldquant_ref.h), on the
// cases of test_foldquant_cpu_op.cpp: the activation blob must match byte for
// byte and every output bit for bit, as the CUDA kernels do. Built only with
// -DGGML_SYCL=ON.

#include "foldquant.h"
#include "foldquant_ref.h"
#include "layers/fq_linear.h"
#include "sycl/vla_sycl_foldquant.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    bool gamma, ascale, fold_before, bias, residual;
};

// The shape under test; main() runs the cases over a list of them.
int64_t K = 256, N = 128, T = 37;

int run_case(ggml_backend_t backend, const Case & c) {
    Lcg rng(0x5eed1234u);
    std::vector<float> hx((size_t) K * T), hw((size_t) K * N), hws(N), hb(N), has(K), hga(K), hr((size_t) N * T);
    for (auto & v : hx)  v = rng.next() * 4.0f;
    for (auto & v : hw)  v = rng.next();
    for (auto & v : hws) v = 0.01f + 0.02f * std::fabs(rng.next());
    for (auto & v : hb)  v = rng.next() * 0.5f;
    for (auto & v : has) v = 0.5f + std::fabs(rng.next());
    for (auto & v : hga) v = 0.75f + 0.5f * std::fabs(rng.next());
    for (auto & v : hr)  v = rng.next();

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

    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * W = ggml_init(wp);
    ggml_tensor * w  = ggml_new_tensor_2d(W, GGML_TYPE_I8, kpw, N);
    ggml_tensor * ws = ggml_new_tensor_1d(W, GGML_TYPE_F32, N);
    ggml_tensor * b  = c.bias   ? ggml_new_tensor_1d(W, GGML_TYPE_F32, N) : nullptr;
    ggml_tensor * as = c.ascale ? ggml_new_tensor_1d(W, GGML_TYPE_F32, K) : nullptr;
    ggml_tensor * ga = c.gamma  ? ggml_new_tensor_1d(W, GGML_TYPE_F32, K) : nullptr;
    ggml_set_name(w, "site.weight");
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
    ggml_tensor * r = c.residual ? ggml_new_tensor_2d(C, GGML_TYPE_F32, N, T) : nullptr;
    ggml_set_input(x);
    if (r) ggml_set_input(r);
    ggml_tensor * xq = vla::fq_act(C, s, x);
    ggml_tensor * y  = vla::fq_gemm(C, s, xq, r);
    ggml_set_output(xq);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { std::printf("FAIL: alloc\n"); return 1; }
    ggml_backend_tensor_set(x, hx.data(), 0, ggml_nbytes(x));
    if (r) ggml_backend_tensor_set(r, hr.data(), 0, ggml_nbytes(r));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: %s: graph compute\n", c.name);
        return 1;
    }
    const int64_t rb = vla::fq_act_row_bytes(K, c.abits), kp = vla::fq_act_kpack(K, c.abits);
    std::vector<uint8_t> got_blob((size_t) rb * T);
    std::vector<float>   got_y((size_t) N * T);
    ggml_backend_tensor_get(xq, got_blob.data(), 0, got_blob.size());
    ggml_backend_tensor_get(y, got_y.data(), 0, got_y.size() * sizeof(float));

    // Reference: blob rows, then the integer GEMM and its epilogue.
    int bad_rows = 0, bad_y = 0;
    std::vector<float> tmp(K);
    float partial[vla::fqref::NT];
    std::vector<uint8_t> row((size_t) rb);
    std::vector<int8_t>  codes((size_t) K);
    for (int64_t t = 0; t < T; ++t) {
        std::memset(row.data(), 0, rb);
        vla::fqref::act_row(hx.data() + (size_t) t * K, as ? has.data() : nullptr, ga ? hga.data() : nullptr,
                            s.act, tmp.data(), partial, (int8_t *) row.data(), (float *) (row.data() + kp));
        if (std::memcmp(row.data(), got_blob.data() + (size_t) t * rb, kp + 4) != 0) ++bad_rows;
        if (c.abits == 4) vla::fqref::unpack_nibbles((const int8_t *) row.data(), K, codes.data());
        else std::memcpy(codes.data(), row.data(), K);
        float xs;
        std::memcpy(&xs, row.data() + kp, 4);
        for (int64_t n = 0; n < N; ++n) {
            float want = vla::fqref::gemm_dot(wcodes.data() + (size_t) n * K, codes.data(), K, xs, hws[n],
                                              b ? hb[n] : 0.0f);
            if (r) want = want + hr[(size_t) t * N + n];
            if (std::memcmp(&want, &got_y[(size_t) t * N + n], 4) != 0) {
                if (bad_y < 3 && std::getenv("VLA_FQ_TEST_VERBOSE"))
                    std::printf("    y[%lld][%lld] got %.9g want %.9g (blob row %s)\n", (long long) t, (long long) n,
                                got_y[(size_t) t * N + n], want,
                                std::memcmp(row.data(), got_blob.data() + (size_t) t * rb, kp + 4) ? "differs" : "same");
                ++bad_y;
            }
        }
    }
    const bool ok = bad_rows == 0 && bad_y == 0;
    std::printf("%s case %-22s W%dA%d rot%-2d gamma=%d ascale=%d before=%d bias=%d res=%d: "
                "%d/%lld blob rows differ, %d/%lld outputs differ\n",
                ok ? "ok  " : "FAIL", c.name, c.wbits, c.abits, c.rot, c.gamma, c.ascale, c.fold_before, c.bias,
                c.residual, bad_rows, (long long) T, bad_y, (long long) (N * T));
    ggml_gallocr_free(alloc);
    ggml_free(C);
    ggml_backend_buffer_free(wbuf);
    ggml_free(W);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: no SYCL device\n");
        return 0;
    }
    vla::sycl_register_foldquant_ops();
    const Case cases[] = {
        { "w8a8_rot64",          8, 8, 64, false, false, false, true,  false },
        { "w8a8_rot64_gamma",    8, 8, 64, true,  false, false, false, true  },
        { "w8a8_rot32_pre",      8, 8, 32, false, true,  true,  true,  false },
        { "w8a8_rot64_post",     8, 8, 64, false, true,  false, false, false },
        { "w8a8_norot",          8, 8, 1,  false, false, false, false, false },
        { "w4a8_rot64_gamma",    4, 8, 64, true,  false, false, true,  true  },
        { "w4a4_rot64_pre",      4, 4, 64, false, true,  true,  true,  false },
        { "w8a4_rot64_gamma",    8, 4, 64, true,  false, false, false, false },
    };
    // Small shapes over every path, then production ones: K = 16384 (a down
    // projection: more chunks than the per-row act kernel holds, so the warp
    // kernel), K = 6144 and 2048 rows through the per-row kernel, a prefill M
    // through the XMX GEMM and a decode M through the GEMV, at full K so the int32
    // sums reach the magnitudes a real layer produces.
    struct Shape { int64_t K, N, T; };
    const Shape shapes[] = {
        { 256, 128, 37 }, { 256, 128, 10 }, { 256, 200, 37 },
        { 16384, 256, 300 }, { 16384, 256, 10 }, { 6144, 192, 41 }, { 2048, 2048, 64 },
    };
    // Every GEMM path: the default (oneDNN when built with it), this repo's
    // GEMV + XMX kernels, and the plain tiled one.
    int fails = 0;
    for (const char * path : { "", "native", "simple" }) {
        setenv("VLA_FQ_SYCL_GEMM", path, 1);
        for (const Shape & sh : shapes) {
            K = sh.K; N = sh.N; T = sh.T;
            std::printf("GEMM path '%s', K = %lld, N = %lld, T = %lld\n", path, (long long) K, (long long) N,
                        (long long) T);
            for (const Case & c : cases)
                fails += run_case(backend, c);
        }
    }
    std::printf("%s\n", fails ? "test_foldquant_sycl_op: FAILED" : "test_foldquant_sycl_op: PASS");
    ggml_backend_free(backend);
    return fails ? 1 : 0;
}
