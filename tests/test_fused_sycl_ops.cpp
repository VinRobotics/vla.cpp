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

// The row-wise fused nodes of layers/fused_ew.h that SmolVLA's fused graph
// uses (fused_rms into F32 / BF16 rows, fused_swiglu into BF16 rows or a q8
// blob), fused_geglu next to them, and fused_linear with the erf GELU and a
// residual (Evo-1's InternViT and DiT MLPs), and fused_lnorm without affine, run on SYCL (src/sycl/vla_sycl_fused.cpp)
// and checked against the same graph on the CPU backend, whose reference is
// the header's. Gate and up are views of one gate|up tensor, as in the models.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"

#include "layers/fused_ew.h"
#include "sycl/vla_sycl_ops.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t H = 480;   // RMSNorm width (SmolVLA's expert)
constexpr int64_t N = 1280;  // gate / up width
constexpr int64_t M = 50;    // rows

struct Outs {
    std::vector<float> rms_f32, rms_bf16, swiglu, geglu, swiglu_q8, lin_erf, ln_plain;
};

std::vector<float> get_f32(ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// A q8 blob of M rows of n codes, then M F32 scales, decoded to floats.
std::vector<float> get_q8(ggml_tensor * t, int64_t n) {
    std::vector<int8_t> b((size_t) ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    std::vector<float> v((size_t) (M * n));
    for (int64_t m = 0; m < M; ++m) {
        float sc;
        std::memcpy(&sc, b.data() + M * n + 4 * m, 4);
        for (int64_t i = 0; i < n; ++i) v[(size_t) (m * n + i)] = sc * b[(size_t) (m * n + i)];
    }
    return v;
}

bool run(ggml_backend_t backend, const std::vector<float> & hx, const std::vector<float> & hw,
         const std::vector<float> & hgu, const std::vector<float> & hwl, Outs & o) {
    static const vla::FusedRmsSpec rms{vla::FUSED_RMSQ_MAGIC, 1e-5f};
    ggml_init_params p = { (size_t) 16*1024*1024, nullptr, true };
    ggml_context * C = ggml_init(p);

    ggml_tensor * x  = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, M);
    ggml_tensor * w  = ggml_new_tensor_1d(C, GGML_TYPE_F32, H);
    ggml_tensor * gu = ggml_new_tensor_2d(C, GGML_TYPE_F32, 2 * N, M);
    ggml_tensor * wl = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, N);
    for (ggml_tensor * t : {x, w, gu, wl}) ggml_set_input(t);

    ggml_tensor * r32 = vla::fused_rms(C, x, w, GGML_TYPE_F32, rms);
    ggml_tensor * r16 = ggml_cast(C, vla::fused_rms(C, x, w, GGML_TYPE_BF16, rms), GGML_TYPE_F32);
    ggml_tensor * g16 = ggml_cast(C, gu, GGML_TYPE_BF16);
    ggml_tensor * g   = ggml_view_2d(C, g16, N, M, g16->nb[1], 0);
    ggml_tensor * u   = ggml_view_2d(C, g16, N, M, g16->nb[1], N * ggml_element_size(g16));
    ggml_tensor * sw  = ggml_cast(C, vla::fused_swiglu(C, g, u), GGML_TYPE_F32);
    ggml_tensor * ge  = ggml_cast(C, vla::fused_geglu(C, g, u), GGML_TYPE_F32);
    ggml_tensor * sq  = vla::fused_swiglu(C, g, u, GGML_TYPE_I8);
    // [H, M] x [H, N]^T + b, erf GELU, + res; the bias and residual are
    // slices of the gate|up input.
    ggml_tensor * Wl  = ggml_cast(C, wl, GGML_TYPE_BF16);
    ggml_tensor * bl  = ggml_view_1d(C, gu, N, 0);
    ggml_tensor * rl  = ggml_view_2d(C, gu, N, M, gu->nb[1], 0);
    ggml_tensor * le  = vla::fused_linear(C, Wl, ggml_cast(C, x, GGML_TYPE_BF16), bl, rl, GGML_TYPE_F32,
                                          vla::FUSED_GELU_ERF);
    // LayerNorm without affine (the GR00T DiT's second norm), into BF16
    static const vla::FusedLNormSpec ln{vla::FUSED_LNORM_MAGIC, 1e-5f};
    ggml_tensor * lp  = ggml_cast(C, vla::fused_lnorm(C, x, nullptr, nullptr, GGML_TYPE_BF16, ln), GGML_TYPE_F32);
    ggml_tensor * outs[] = {r32, r16, sw, ge, sq, le, lp};

    ggml_cgraph * gf = ggml_new_graph(C);
    for (ggml_tensor * t : outs) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ga && ggml_gallocr_alloc_graph(ga, gf);
    if (ok) {
        ggml_backend_tensor_set(x,  hx.data(),  0, ggml_nbytes(x));
        ggml_backend_tensor_set(w,  hw.data(),  0, ggml_nbytes(w));
        ggml_backend_tensor_set(gu, hgu.data(), 0, ggml_nbytes(gu));
        ggml_backend_tensor_set(wl, hwl.data(), 0, ggml_nbytes(wl));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        o.rms_f32   = get_f32(r32);
        o.rms_bf16  = get_f32(r16);
        o.swiglu    = get_f32(sw);
        o.geglu     = get_f32(ge);
        o.swiglu_q8 = get_q8(sq, N);
        o.lin_erf   = get_f32(le);
        o.ln_plain  = get_f32(lp);
    }
    ggml_gallocr_free(ga);
    ggml_free(C);
    return ok;
}

// max |ref - got| / max |ref|.
double rel_diff(const std::vector<float> & ref, const std::vector<float> & got) {
    double max_abs = 0.0, max_diff = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_abs  = std::fmax(max_abs, std::fabs((double) ref[i]));
        max_diff = std::fmax(max_diff, std::fabs((double) ref[i] - (double) got[i]));
    }
    return max_abs > 0.0 ? max_diff / max_abs : 1.0;
}

}  // namespace

int main() {
    if (!std::getenv("GGML_SYCL_ENABLE_VMM")) {
        setenv("GGML_SYCL_ENABLE_VMM", "0", /*overwrite=*/0);
    }
    // As tests/test_bf16_sycl_ops.cpp: skip without a SYCL GPU.
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty() ||
        ggml_backend_sycl_get_device_count() < 1) {
        std::printf("fused_sycl_ops: no SYCL GPU, skipping\n");
        return 0;
    }
    ggml_backend_t sy = ggml_backend_sycl_init(0);
    if (!sy) {
        std::printf("fused_sycl_ops: ggml_backend_sycl_init failed, skipping\n");
        return 0;
    }
    vla::sycl_register_fused_ops();
    ggml_backend_t cpu = ggml_backend_cpu_init();

    std::vector<float> hx((size_t) (H * M)), hw((size_t) H), hgu((size_t) (2 * N * M)), hwl((size_t) (H * N));
    for (size_t i = 0; i < hx.size(); ++i)  hx[i]  = ((float) ((i * 37) % 23) - 11.0f) / 4.0f;
    for (size_t i = 0; i < hw.size(); ++i)  hw[i]  = 0.5f + ((float) ((i * 11) % 7)) / 16.0f;
    for (size_t i = 0; i < hgu.size(); ++i) hgu[i] = ((float) ((i * 53) % 41) - 20.0f) / 5.0f;
    for (size_t i = 0; i < hwl.size(); ++i) hwl[i] = ((float) ((i * 29) % 31) - 15.0f) / 256.0f;

    Outs ref, got;
    if (!run(cpu, hx, hw, hgu, hwl, ref) || !run(sy, hx, hw, hgu, hwl, got)) {
        std::printf("FAIL: a run failed\n");
        return 1;
    }

    // F32 rows: rsqrt vs 1/sqrt. BF16 rows: one rounding of 8 mantissa bits.
    // SwiGLU / GeGLU: BF16 out, and the kernel's exp is the native one. The q8
    // blob: codes may differ by one where native exp moves a value across a
    // rounding boundary, so allow one step of the row's scale (amax / 127).
    struct Case { const char * name; const std::vector<float> & r, & g; double tol; };
    const Case cases[] = {
        {"fused_rms f32",   ref.rms_f32,   got.rms_f32,   1e-5},
        {"fused_rms bf16",  ref.rms_bf16,  got.rms_bf16,  1e-2},
        {"fused_swiglu",    ref.swiglu,    got.swiglu,    1e-2},
        {"fused_geglu",     ref.geglu,     got.geglu,     1e-2},
        {"fused_swiglu q8", ref.swiglu_q8, got.swiglu_q8, 1.5 / 127.0},
        // F32 accumulation in another order. Tight enough to tell the erf
        // GELU from the tanh one (~4e-5 apart here).
        {"fused_linear erf", ref.lin_erf,  got.lin_erf,   1e-5},
        {"fused_lnorm none", ref.ln_plain, got.ln_plain,  1e-2},
    };
    int fails = 0;
    for (const Case & c : cases) {
        const double d = c.r.size() == c.g.size() ? rel_diff(c.r, c.g) : 1.0;
        const bool   ok = d <= c.tol;
        std::printf("fused_sycl_ops: %-16s normalised diff %.2e (tol %.1e) %s\n", c.name, d, c.tol, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(sy);
    if (fails) { std::printf("FAIL\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
