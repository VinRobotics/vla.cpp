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

// Every op the BF16 activation path needs, driven through real ggml dispatch
// and checked against the same graph in F32.
//
// This is the regression test for src/cuda/vla_cuda_bf16.cu and for the one
// hook it rides on (scripts/patch_ggml_cuda_ext_hook.py). It is what catches a
// llama.cpp bump that moves the hook anchor or changes an op's semantics: with
// the hook uninstalled, ggml aborts on the first BF16 op instead of silently
// returning something plausible.
//
// The chain deliberately includes ggml_mul(ggml_rms_norm(x), w), which is the
// pattern the CUDA backend tries to fuse and whose fusion check asserts F32.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include "cuda/vla_cuda_ops.h"
#include "models/act_dtype.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t K = 128;   // reduction / feature dim
constexpr int64_t M = 64;    // rows
constexpr int64_t N = 96;    // output features
constexpr float   EPS = 1e-5f;

// Builds the chain at `at`. Inputs are always F32 tensors; the BF16 run casts in
// at the top and back out at the bottom, exactly as the models do.
ggml_tensor * build_chain(ggml_context * C, ggml_tensor * x, ggml_tensor * w_norm,
                          ggml_tensor * W, ggml_tensor * bias, ggml_type at) {
    ggml_tensor * h = vla::as_type(C, x, at);
    h = ggml_mul(C, ggml_rms_norm(C, h, EPS), w_norm);   // RMS_NORM (+ MUL, F32 weight)
    h = vla::mm_act(C, W, h, at);                        // MUL_MAT
    h = ggml_add(C, h, bias);                            // ADD, F32 bias
    h = ggml_silu(C, h);                                 // UNARY SILU
    h = ggml_scale(C, h, 0.5f);                          // SCALE
    h = ggml_add(C, ggml_norm(C, h, EPS), h);            // NORM (+ ADD, BF16 x BF16)
    h = ggml_gelu_erf(C, h);                             // UNARY GELU_ERF
    return vla::as_type(C, h, GGML_TYPE_F32);
}

std::vector<float> run(ggml_backend_t backend, ggml_type at,
                       const std::vector<float> & hx, const std::vector<float> & hw,
                       const std::vector<float> & hW, const std::vector<float> & hb) {
    ggml_init_params p = { (size_t) 32*1024*1024, nullptr, true };
    ggml_context * C = ggml_init(p);

    ggml_tensor * x      = ggml_new_tensor_2d(C, GGML_TYPE_F32, K, M);
    ggml_tensor * w_norm = ggml_new_tensor_1d(C, GGML_TYPE_F32, K);
    // The weight must be resident in the activation dtype for mm_act to take the
    // typed-GEMM path; that mirrors a BF16 checkpoint.
    ggml_tensor * W      = ggml_new_tensor_2d(C, at, K, N);
    ggml_tensor * bias   = ggml_new_tensor_1d(C, GGML_TYPE_F32, N);
    for (ggml_tensor * t : {x, w_norm, W, bias}) ggml_set_input(t);

    ggml_tensor * out = build_chain(C, x, w_norm, W, bias, at);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        std::fprintf(stderr, "alloc failed\n");
        return {};
    }

    ggml_backend_tensor_set(x,      hx.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(w_norm, hw.data(), 0, ggml_nbytes(w_norm));
    ggml_backend_tensor_set(bias,   hb.data(), 0, ggml_nbytes(bias));
    if (at == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> t(hW.size());
        ggml_fp32_to_bf16_row(hW.data(), t.data(), (int64_t) hW.size());
        ggml_backend_tensor_set(W, t.data(), 0, ggml_nbytes(W));
    } else {
        ggml_backend_tensor_set(W, hW.data(), 0, ggml_nbytes(W));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed (%s)\n", ggml_type_name(at));
        return {};
    }

    std::vector<float> host((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, host.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(ga);
    ggml_free(C);
    return host;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        // No usable GPU: nothing to check, and this must not fail a CPU-only CI run.
        std::printf("bf16_cuda_ops: no CUDA device, skipping\n");
        return 0;
    }
    vla::cuda_register_bf16_ops();

    std::vector<float> hx((size_t) K*M), hw((size_t) K), hW((size_t) K*N), hb((size_t) N);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = ((float) ((i*37) % 23) - 11.0f) / 8.0f;
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.5f + ((float) ((i*11) % 7)) / 16.0f;
    for (size_t i = 0; i < hW.size(); ++i) hW[i] = ((float) ((i*53) % 19) - 9.0f) / 32.0f;
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = ((float) ((i*17) % 5) - 2.0f) / 16.0f;

    const std::vector<float> ref = run(backend, GGML_TYPE_F32,  hx, hw, hW, hb);
    const std::vector<float> got = run(backend, GGML_TYPE_BF16, hx, hw, hW, hb);
    if (ref.empty() || got.empty() || ref.size() != got.size()) {
        std::printf("FAIL: a run produced no output\n");
        return 1;
    }

    // Normalised against the reference's own scale: gelu_erf drives many outputs
    // near zero, where a per-element relative error is meaningless.
    double max_abs = 0.0, max_diff = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_abs  = std::fmax(max_abs, std::fabs((double) ref[i]));
        max_diff = std::fmax(max_diff, std::fabs((double) ref[i] - (double) got[i]));
    }
    const double rel = max_abs > 0.0 ? max_diff / max_abs : max_diff;

    // BF16 carries 8 mantissa bits, and this chain is 8 ops deep.
    const double tol = 0.05;
    std::printf("bf16_cuda_ops: max |ref| %.4f, max diff %.4f, normalised %.4f (tol %.2f)\n",
                max_abs, max_diff, rel, tol);

    if (max_abs == 0.0) { std::printf("FAIL: reference is all zeros\n"); return 1; }
    if (rel > tol)      { std::printf("FAIL\n"); return 1; }

    std::printf("PASS\n");
    ggml_backend_free(backend);
    return 0;
}
