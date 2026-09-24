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

// FoldQuant nodes on the CUDA backend versus the CPU reference: the activation
// blob and the F32 GEMM output must be byte-identical (int32 accumulation,
// fixed reduction tree, no FMA contraction on either side). Also covers the
// ggml plumbing the kernels depend on: the custom-op userdata decode, the hook
// dispatcher composing with the BF16 handler, and gallocr sizing of the blob.
// Runs once per ctest registration: plain, and under VLA_FQ_CPU_REF=1 (staging
// shim). Skips itself (exit 0) when no CUDA device is present.

#include "foldquant.h"
#include "foldquant_ref.h"
#include "layers/fq_linear.h"
#include "act_dtype.h"
#include "cuda/vla_cuda_ops.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
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

struct Shape { int64_t K, N, T; };

struct Case {
    const char * name;
    int  wbits, abits, rot;
    bool gamma, ascale, fold_before, bias;
    bool res = false;   // residual fused into the GEMM epilogue
    int  heads = 0;     // >0: head-layout epilogue (hd = 64, 1-3 parts, last part V)
};

struct Inputs {
    std::vector<float>  x, ws, b, as, ga, res;
    std::vector<int8_t> w;   // packed
};

Inputs make_inputs(const Shape & sh, const Case & c, uint32_t seed) {
    Inputs in;
    Lcg rng(seed);
    in.x.resize((size_t) sh.K * sh.T); in.ws.resize(sh.N); in.b.resize(sh.N); in.as.resize(sh.K); in.ga.resize(sh.K);
    for (auto & v : in.x)  v = rng.next() * 4.0f;
    for (auto & v : in.ws) v = 0.01f + 0.02f * std::fabs(rng.next());
    for (auto & v : in.b)  v = rng.next() * 0.5f;
    in.res.resize((size_t) sh.N * sh.T);
    for (auto & v : in.res) v = rng.next() * 3.0f;
    for (auto & v : in.as) v = 0.5f + std::fabs(rng.next());
    for (auto & v : in.ga) v = 0.75f + 0.5f * std::fabs(rng.next());
    const int64_t kpw = vla::fq_w_kpack(sh.K, c.wbits);
    in.w.resize((size_t) kpw * sh.N);
    const int wq = c.wbits == 4 ? 7 : 127;
    for (int64_t n = 0; n < sh.N; ++n) {
        std::vector<int> codes(sh.K);
        for (int64_t k = 0; k < sh.K; ++k) codes[k] = (int) std::nearbyint(rng.next() * (float) wq);
        if (c.wbits == 4)
            for (int64_t k = 0; k < sh.K; k += 2)
                in.w[(size_t) n * kpw + k / 2] = (int8_t) ((codes[k] & 0xF) | ((codes[k + 1] & 0xF) << 4));
        else
            for (int64_t k = 0; k < sh.K; ++k) in.w[(size_t) n * kpw + k] = (int8_t) codes[k];
    }
    return in;
}

struct Result {
    std::vector<uint8_t> blob;
    std::vector<float>   y;
    std::vector<float>   bf16_probe;   // output of a BF16 mul_mat_t node in the same graph
};

Result run(ggml_backend_t backend, const Shape & sh, const Case & c, const Inputs & in, bool with_bf16_probe) {
    Result r;
    ggml_init_params p = { (size_t) 64 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(p);

    const int64_t kpw = vla::fq_w_kpack(sh.K, c.wbits);
    ggml_tensor * x  = ggml_new_tensor_2d(C, GGML_TYPE_F32, sh.K, sh.T);
    ggml_tensor * w  = ggml_new_tensor_2d(C, GGML_TYPE_I8, kpw, sh.N);
    ggml_tensor * ws = ggml_new_tensor_1d(C, GGML_TYPE_F32, sh.N);
    ggml_tensor * b  = c.bias   ? ggml_new_tensor_1d(C, GGML_TYPE_F32, sh.N) : nullptr;
    ggml_tensor * rs = c.res    ? ggml_new_tensor_2d(C, GGML_TYPE_F32, sh.N, sh.T) : nullptr;
    ggml_tensor * as = c.ascale ? ggml_new_tensor_1d(C, GGML_TYPE_F32, sh.K) : nullptr;
    ggml_tensor * ga = c.gamma  ? ggml_new_tensor_1d(C, GGML_TYPE_F32, sh.K) : nullptr;
    ggml_tensor * Wb = with_bf16_probe ? ggml_new_tensor_2d(C, GGML_TYPE_BF16, sh.K, 64) : nullptr;
    for (ggml_tensor * t : {x, w, ws, b, as, ga, Wb, rs}) if (t) ggml_set_input(t);
    ggml_set_name(w, "site");

    vla::FqLinear s;
    s.w = w; s.wscale = ws; s.bias = b; s.ascale = as; s.gamma = ga;
    s.act.K = sh.K; s.act.abits = c.abits; s.act.rot_block = c.rot; s.act.fold_before = c.fold_before;
    s.act.has_gamma = c.gamma; s.act.has_ascale = c.ascale; s.act.clip = c.abits == 4 ? 0.9f : 1.0f; s.act.eps = 1e-6f;
    s.gemm.K = sh.K; s.gemm.N = sh.N; s.gemm.wbits = c.wbits;
    if (c.heads) {   // hd = 64; as many parts as the model uses (q/k/v = 3, k/v = 2, one), last part V
        const int parts = sh.N % 192 == 0 ? 3 : sh.N % 128 == 0 ? 2 : 1;
        if (sh.N % (64 * parts) == 0)
            vla::fq_set_heads(s, 64, (int) (sh.N / (64 * parts)), 1u << (parts - 1));
    }

    ggml_tensor * xq = vla::fq_act(C, s, x);
    ggml_tensor * y  = vla::fq_gemm(C, s, xq, rs);
    ggml_set_output(xq); ggml_set_output(y);
    ggml_tensor * probe = nullptr;
    if (Wb) {
        // The BF16 handler must still see its nodes when both are registered.
        ggml_tensor * xb = vla::as_type(C, x, GGML_TYPE_BF16);
        probe = vla::as_type(C, vla::mm_act(C, Wb, xb, GGML_TYPE_BF16), GGML_TYPE_F32);
        ggml_set_output(probe);
    }

    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, y);
    if (probe) ggml_build_forward_expand(gf, probe);

    ggml_gallocr_t ga_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ga_ || !ggml_gallocr_alloc_graph(ga_, gf)) { std::fprintf(stderr, "alloc failed\n"); return r; }

    ggml_backend_tensor_set(x,  in.x.data(),  0, ggml_nbytes(x));
    ggml_backend_tensor_set(w,  in.w.data(),  0, ggml_nbytes(w));
    ggml_backend_tensor_set(ws, in.ws.data(), 0, ggml_nbytes(ws));
    if (b)  ggml_backend_tensor_set(b,  in.b.data(),  0, ggml_nbytes(b));
    if (rs) ggml_backend_tensor_set(rs, in.res.data(), 0, ggml_nbytes(rs));
    if (as) ggml_backend_tensor_set(as, in.as.data(), 0, ggml_nbytes(as));
    if (ga) ggml_backend_tensor_set(ga, in.ga.data(), 0, ggml_nbytes(ga));
    if (Wb) {
        std::vector<float> hw((size_t) sh.K * 64);
        Lcg rng(7u);
        for (auto & v : hw) v = rng.next();
        std::vector<ggml_bf16_t> t(hw.size());
        ggml_fp32_to_bf16_row(hw.data(), t.data(), (int64_t) hw.size());
        ggml_backend_tensor_set(Wb, t.data(), 0, ggml_nbytes(Wb));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed\n"); return r; }

    r.blob.resize(ggml_nbytes(xq));
    ggml_backend_tensor_get(xq, r.blob.data(), 0, r.blob.size());
    r.y.resize((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, r.y.data(), 0, ggml_nbytes(y));
    if (probe) {
        r.bf16_probe.resize((size_t) ggml_nelements(probe));
        ggml_backend_tensor_get(probe, r.bf16_probe.data(), 0, ggml_nbytes(probe));
    }
    ggml_gallocr_free(ga_);
    ggml_free(C);
    return r;
}

// The CPU op reads its spec through userdata; the CUDA decoder mirrors the
// private ggml_custom_op_params layout. Pin that the pointer round-trips.
bool check_userdata_layout() {
    ggml_init_params p = { (size_t) 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_1d(C, GGML_TYPE_F32, 16);
    int marker = 42;
    ggml_tensor * args[1] = { a };
    ggml_tensor * t = ggml_custom_4d(C, GGML_TYPE_F32, 16, 1, 1, 1, args, 1, nullptr, GGML_N_TASKS_MAX, &marker);
    struct { void * fun; int n_tasks; void * userdata; } mirror;
    std::memcpy(&mirror, t->op_params, sizeof(mirror));
    const bool ok = mirror.userdata == &marker && mirror.n_tasks == GGML_N_TASKS_MAX && mirror.fun == nullptr;
    ggml_free(C);
    return ok;
}

}  // namespace

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) {
        std::printf("foldquant_cuda_op: no CUDA device, skipping\n");
        return 0;
    }
    if (!check_userdata_layout()) {
        std::printf("FAIL: ggml_custom_op_params layout changed; update vla_cuda_foldquant.cu\n");
        return 1;
    }
    vla::cuda_register_bf16_ops();
    vla::cuda_register_foldquant_ops();

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);

    const Case cases[] = {
        { "w8a8_rot64_bias",     8, 8, 64, false, false, false, true  },
        { "w8a8_rot64_bias_res", 8, 8, 64, false, false, false, true, true },
        { "w4a4_rot64_res",      4, 4, 64, false, true,  true,  false, true },
        { "w8a8_rot64_heads",    8, 8, 64, false, false, false, true,  false, 4 },
        { "w4a4_rot64_heads",    4, 4, 64, false, true,  true,  true,  false, 2 },
        { "w8a8_rot64_gamma",    8, 8, 64, true,  false, false, false },
        { "w8a8_rot32_pre",      8, 8, 32, false, true,  true,  true  },
        { "w8a8_rot64_post",     8, 8, 64, false, true,  false, false },
        { "w8a8_norot",          8, 8, 1,  false, false, false, false },
        { "w4a8_rot64_gamma",    4, 8, 64, true,  false, false, true  },   // CPU-reference fallback on CUDA today
        { "w4a4_rot64_pre",      4, 4, 64, false, true,  true,  true  },
    };
    const Shape shapes[] = {
        { 128,  64,  5   },
        { 2048, 4096, 41 },   // GR00T N1.7 LLM q/k/v at a DiT-sized M
        { 2048, 2048, 130 },  // o_proj, M above one tile with a tail
        { 6144, 2048, 130 },  // ffn_down
        { 1536, 4608, 41  },  // DiT fused qkv
        { 2048, 12288, 160 }, // fused gate+up at the N1.7 prefill: the 128x64x128 tile
        { 2048, 2048, 193 },  // 192-row tile with a one-row second M tile
        { 2048, 4096, 300 },  // prefix past 256 tokens: 128-row tile, three M tiles
    };

    int n = 0;
    for (const Shape & sh : shapes) {
        for (const Case & c : cases) {
            if (sh.K > 128 && (c.wbits != 8 || c.abits != 8)) continue;   // big W4 cases are slow on the host path
            const Inputs in = make_inputs(sh, c, 0x1234u + (uint32_t) n);
            const Result ref = run(cpu,  sh, c, in, false);
            const Result got = run(cuda, sh, c, in, true);
            if (ref.blob.empty() || got.blob.empty() || ref.y.empty() || got.y.empty()) {
                std::printf("FAIL: %s K=%lld N=%lld T=%lld produced no output\n", c.name, (long long) sh.K, (long long) sh.N, (long long) sh.T);
                return 1;
            }
            // Compare codes + scale of every row; the padding bytes are uninitialised.
            const int64_t rb = vla::fq_act_row_bytes(sh.K, c.abits), kp = vla::fq_act_kpack(sh.K, c.abits);
            for (int64_t t = 0; t < sh.T; ++t)
                if (std::memcmp(ref.blob.data() + (size_t) t * rb, got.blob.data() + (size_t) t * rb, kp + 4) != 0) {
                    std::printf("FAIL: %s K=%lld N=%lld T=%lld: act row %lld differs\n", c.name,
                                (long long) sh.K, (long long) sh.N, (long long) sh.T, (long long) t);
                    return 1;
                }
            if (std::memcmp(ref.y.data(), got.y.data(), ref.y.size() * sizeof(float)) != 0) {
                size_t first = 0;
                while (first < ref.y.size() && ref.y[first] == got.y[first]) ++first;
                std::printf("FAIL: %s K=%lld N=%lld T=%lld: y[%zu] = %.9g vs CPU %.9g\n", c.name,
                            (long long) sh.K, (long long) sh.N, (long long) sh.T, first, got.y[first], ref.y[first]);
                return 1;
            }
            if (got.bf16_probe.empty()) { std::printf("FAIL: BF16 probe missing\n"); return 1; }
            std::printf("ok   %-18s K=%-5lld N=%-5lld T=%-4lld\n", c.name, (long long) sh.K, (long long) sh.N, (long long) sh.T);
            ++n;
        }
    }
    std::printf("PASS (%d shape/case pairs)\n", n);
    ggml_backend_free(cpu);
    ggml_backend_free(cuda);
    return 0;
}
