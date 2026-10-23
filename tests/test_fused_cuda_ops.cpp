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

// The fused CUDA nodes of layers/fused_ops.h against the ggml chains they
// replace, on the same inputs: the outputs must be bit-identical. Covers the
// epilogues, the head layouts with RoPE, gemm_exact (mm_exact) and attn_exact
// (with K/V broadcast and a mask), at the widths the models use. Skips
// (exit 0) without a CUDA device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include "backend.h"
#include "cuda/vla_cuda_ops.h"
#include "layers/attn.h"
#include "layers/fused_ops.h"
#include "layers/linear.h"
#include "layers/norm.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

enum class Fill { Rand, Pos, Mask };

struct In {
    ggml_type type;
    int64_t   ne[4];
    Fill      fill = Fill::Rand;
};

using Build = std::function<ggml_tensor *(ggml_context *, const std::vector<ggml_tensor *> &, bool fused)>;

struct Case {
    std::string     name;
    std::vector<In> ins;
    Build           build;
};

void fill(ggml_tensor * t, Fill f, uint32_t seed) {
    const int64_t n = ggml_nelements(t);
    if (f == Fill::Pos) {
        std::vector<int32_t> v((size_t) n);
        for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (int32_t) i;
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        return;
    }
    std::vector<float> v((size_t) n);
    if (f == Fill::Mask) {   // causal: key k visible to query q when k <= q
        const int64_t nk = t->ne[0];
        for (int64_t i = 0; i < n; ++i) v[(size_t) i] = i%nk <= i/nk ? 0.0f : -INFINITY;
    } else {
        for (auto & x : v) {
            seed = seed*1664525u+1013904223u;
            x = ((float) (seed >> 8)/16777216.0f)*4.0f-2.0f;
        }
    }
    if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> b((size_t) n);
        ggml_fp32_to_bf16_row(v.data(), b.data(), n);
        ggml_backend_tensor_set(t, b.data(), 0, ggml_nbytes(t));
    } else {
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
}

std::vector<uint8_t> run(ggml_backend_t backend, const Case & c, bool fused) {
    ggml_init_params p = { (size_t) 16*1024*1024, nullptr, true };
    ggml_context * C = ggml_init(p);
    std::vector<ggml_tensor *> ins;
    for (const In & in : c.ins) {
        ggml_tensor * t = ggml_new_tensor_4d(C, in.type, in.ne[0], in.ne[1], in.ne[2], in.ne[3]);
        ggml_set_input(t);
        ins.push_back(t);
    }
    ggml_tensor * out = c.build(C, ins, fused);
    if (out->type != GGML_TYPE_F32)
        out = ggml_cast(C, out, GGML_TYPE_F32);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph_custom(C, 4096, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<uint8_t> host;
    if (ga && ggml_gallocr_alloc_graph(ga, gf)) {
        for (size_t i = 0; i < ins.size(); ++i)
            fill(ins[i], c.ins[i].fill, 1234u+97u*(uint32_t) i);
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            host.resize(ggml_nbytes(out));
            ggml_backend_tensor_get(out, host.data(), 0, host.size());
        }
    }
    ggml_gallocr_free(ga);
    ggml_free(C);
    return host;
}

constexpr float EPS = 1e-6f;

std::vector<Case> cases() {
    using T = std::vector<ggml_tensor *>;
    const ggml_type F = GGML_TYPE_F32, B = GGML_TYPE_BF16, I = GGML_TYPE_I32;
    std::vector<Case> v;

    for (int64_t D : {768, 1152}) {
        v.push_back({"ln_affine/" + std::to_string(D), {{F, {D, 77, 1, 1}}, {F, {D, 1, 1, 1}}, {F, {D, 1, 1, 1}}},
            [](ggml_context * C, const T & t, bool f) {
                return f ? vla::fused_ln(C, t[0], t[1], t[2], EPS, false) : vla::layer_norm(C, t[0], t[1], t[2], EPS);
            }});
    }
    v.push_back({"ln_modulate", {{F, {1536, 41, 1, 1}}, {F, {1536, 1, 1, 1}}, {F, {1536, 1, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            if (f) return vla::fused_ln(C, t[0], t[1], t[2], EPS, true);
            ggml_tensor * n = ggml_norm(C, t[0], EPS);
            return ggml_add(C, ggml_add(C, n, ggml_mul(C, n, t[1])), t[2]);
        }});
    for (int64_t D : {896, 1024, 2048}) {
        v.push_back({"rms_mul/" + std::to_string(D), {{F, {D, 53, 1, 1}}, {F, {D, 1, 1, 1}}},
            [](ggml_context * C, const T & t, bool f) {
                return f ? vla::fused_rms_mul(C, t[0], t[1], EPS, false) : vla::rms_norm(C, t[0], t[1], EPS);
            }});
    }
    v.push_back({"rms_modulate", {{F, {1024, 50, 1, 1}}, {F, {1024, 1, 1, 1}}, {F, {1024, 1, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            if (f) return vla::fused_rms_modulate(C, t[0], t[1], t[2], EPS, false);
            ggml_tensor * n = ggml_rms_norm(C, t[0], EPS);
            return ggml_add(C, ggml_add(C, n, ggml_mul(C, n, t[1])), t[2]);
        }});
    v.push_back({"bias_res", {{F, {768, 61, 1, 1}}, {F, {768, 1, 1, 1}}, {F, {768, 61, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::fused_bias_res(C, t[0], t[1], t[2]) : ggml_add(C, t[2], ggml_add(C, t[0], t[1]));
        }});
    v.push_back({"bias_gelu", {{F, {3072, 33, 1, 1}}, {F, {3072, 1, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::fused_bias_gelu(C, t[0], t[1]) : vla::gelu(C, ggml_add(C, t[0], t[1]));
        }});
    v.push_back({"bias_gelu_erf", {{F, {4096, 29, 1, 1}}, {F, {4096, 1, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::fused_bias_gelu_erf(C, t[0], t[1]) : ggml_gelu_erf(C, ggml_add(C, t[0], t[1]));
        }});
    v.push_back({"bias_gate_res", {{F, {1024, 37, 1, 1}}, {F, {1024, 1, 1, 1}}, {F, {1024, 1, 1, 1}}, {F, {1024, 37, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::fused_bias_gate_res(C, t[0], t[1], t[2], t[3])
                     : ggml_add(C, t[3], ggml_mul(C, ggml_add(C, t[0], t[1]), t[2]));
        }});
    v.push_back({"gate_res", {{F, {1024, 37, 1, 1}}, {F, {1024, 1, 1, 1}}, {F, {1024, 37, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::fused_gate_res(C, t[0], t[1], t[2]) : ggml_add(C, t[2], ggml_mul(C, t[0], t[1]));
        }});
    v.push_back({"silu_mul", {{F, {4864, 31, 1, 1}}, {F, {4864, 31, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::fused_silu_mul(C, t[0], t[1], false) : ggml_mul(C, ggml_silu(C, t[0]), t[1]);
        }});
    v.push_back({"gelu_mul", {{F, {4096, 31, 1, 1}}, {F, {4096, 31, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::fused_gelu_mul(C, t[0], t[1], false) : vla::geglu(C, t[0], t[1]);
        }});

    // BF16-weight GEMMs: gemm_exact against ggml's own cuBLAS call, for an F32
    // operand and for a BF16 operand produced by a fused norm.
    v.push_back({"mm_exact", {{B, {896, 4864, 1, 1}}, {F, {896, 300, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::mm_exact(C, t[0], t[1]) : ggml_mul_mat(C, t[0], t[1]);
        }});
    v.push_back({"ln_bf16_mm_exact", {{B, {1152, 1152, 1, 1}}, {F, {1152, 256, 1, 1}}, {F, {1152, 1, 1, 1}}, {F, {1152, 1, 1, 1}}},
        [](ggml_context * C, const T & t, bool f) {
            return f ? vla::mm_exact(C, t[0], vla::fused_ln(C, t[1], t[2], t[3], EPS, false, true))
                     : ggml_mul_mat(C, t[0], vla::layer_norm(C, t[1], t[2], t[3], EPS));
        }});

    // A ViT block's attention: bias + head split + V^T, then attn_exact, at
    // the head dims the towers use.
    for (int64_t hd : {64, 72}) {
        const int64_t H = 16, S = 257, D = hd*H;
        v.push_back({"qkv_heads_attn/hd" + std::to_string(hd), {{F, {3*D, S, 1, 1}}, {F, {3*D, 1, 1, 1}}},
            [hd, H, S, D](ggml_context * C, const T & t, bool f) {
                const float scale = 1.0f/std::sqrt((float) hd);
                if (f) {
                    ggml_tensor * qkv = vla::fused_qkv_heads(C, t[0], t[1], nullptr, nullptr, 3, hd, H, S, 1, 0b100, 0);
                    return vla::attention_rows_exact(C, vla::fused_part(C, qkv, 0, false), vla::fused_part(C, qkv, 1, false),
                                                     vla::fused_part(C, qkv, 2, true), scale, D, S, 1, false);
                }
                ggml_tensor * y = ggml_add(C, t[0], t[1]);
                auto part = [&](int p) { return ggml_cont(C, ggml_view_2d(C, y, D, S, y->nb[1], (size_t) p*D*sizeof(float))); };
                return vla::attention(C, vla::to_heads(C, part(0), hd, H, S), vla::to_heads(C, part(1), hd, H, S),
                                      vla::to_heads_v(C, part(2), hd, H, S), nullptr, scale, D, S);
            }});
    }

    // An LM layer's attention: bias + NEOX RoPE + head layout, K/V broadcast
    // over the query heads (GQA) and a causal mask.
    {
        constexpr int64_t hd = 64, Hq = 14, Hkv = 2, S = 120;
        v.push_back({"heads_rope_gqa_mask", {{F, {hd*Hq, S, 1, 1}}, {F, {hd*Hkv, S, 1, 1}}, {F, {hd*Hkv, S, 1, 1}},
                                              {F, {hd*Hq, 1, 1, 1}}, {F, {hd*Hkv, 1, 1, 1}}, {F, {hd*Hkv, 1, 1, 1}},
                                              {I, {S, 1, 1, 1}, Fill::Pos}, {F, {S, S, 1, 1}, Fill::Mask}},
            [](ggml_context * C, const T & t, bool f) {
                const float scale = 1.0f/std::sqrt((float) hd), base = 1e6f;
                if (f) {
                    ggml_tensor * Q  = vla::fused_heads_rope(C, t[0], t[6], nullptr, hd, Hq,  base, false, t[3]);
                    ggml_tensor * K  = vla::fused_heads_rope(C, t[1], t[6], nullptr, hd, Hkv, base, false, t[4]);
                    ggml_tensor * Vt = vla::fused_heads_rope(C, t[2], nullptr, nullptr, hd, Hkv, base, true, t[5]);
                    return vla::fused_attn_exact(C, Q, K, Vt, scale, false, 1, t[7]);
                }
                auto rope = [&](ggml_tensor * x, ggml_tensor * b, int64_t h) {
                    return ggml_rope_ext(C, ggml_reshape_3d(C, ggml_add(C, x, b), hd, h, S), t[6], nullptr, (int) hd,
                                         GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
                };
                ggml_tensor * Q = ggml_cont(C, ggml_permute(C, rope(t[0], t[3], Hq), 0, 2, 1, 3));
                ggml_tensor * K = ggml_cont(C, ggml_permute(C, rope(t[1], t[4], Hkv), 0, 2, 1, 3));
                ggml_tensor * V = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, ggml_add(C, t[2], t[5]), hd, Hkv, S), 1, 2, 0, 3));
                return vla::attention(C, Q, K, V, t[7], scale, hd*Hq, S);
            }});
    }
    return v;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::printf("fused_cuda_ops: no CUDA device, skipping\n");
        return 0;
    }
    if (!vla::enable_fused_ops("test", true)) {
        std::printf("fused_cuda_ops: fused ops disabled (VLA_FUSED_OPS=0), skipping\n");
        ggml_backend_free(backend);
        return 0;
    }
    int failed = 0;
    for (const Case & c : cases()) {
        const std::vector<uint8_t> ref = run(backend, c, false), got = run(backend, c, true);
        const bool ok = !ref.empty() && ref.size() == got.size() && std::memcmp(ref.data(), got.data(), ref.size()) == 0;
        size_t n_diff = 0;
        if (!ok && ref.size() == got.size())
            for (size_t i = 0; i < ref.size(); i += 4)
                n_diff += std::memcmp(ref.data()+i, got.data()+i, 4) != 0;
        std::printf("%-22s %s", c.name.c_str(), ok ? "bit-identical\n" : "FAIL");
        if (!ok) std::printf(" (%zu of %zu values differ)\n", n_diff, ref.size()/4);
        failed += !ok;
    }
    ggml_backend_free(backend);
    std::printf(failed ? "FAIL\n" : "PASS\n");
    return failed ? 1 : 0;
}
