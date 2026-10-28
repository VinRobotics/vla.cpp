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

#include "loader.h"

#include "backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace vla {

namespace {

constexpr size_t NAME_CAP = 256;

// Resident type of a float rebuild: the requested GEMM type when it is a float
// type, F32 otherwise.
ggml_type float_type(ggml_type want) {
    return (want == GGML_TYPE_F32 || want == GGML_TYPE_F16 || want == GGML_TYPE_BF16) ? want : GGML_TYPE_F32;
}

}

void WeightLoader::as_float(const std::string & name, int64_t K, int64_t N, FloatMaker make) {
    as_float_[name] = AsFloat{K, N, std::move(make)};
}

bool WeightLoader::read_resident(const std::string & name, ggml_type type, bool gemma_norm, std::vector<uint8_t> & out) {
    const auto it = as_float_.find(name);
    if (it == as_float_.end()) {
        out = g_.read_convert(name.c_str(), type, gemma_norm);
        return !out.empty();
    }
    const AsFloat & a = it->second;
    std::vector<float> f;
    if (!a.make(f) || (int64_t) f.size() != a.K * a.N) {
        std::fprintf(stderr, "vla(%s): float rebuild of %s failed\n", arch_, name.c_str());
        return false;
    }
    out.assign(ggml_row_size(type, a.K) * (size_t) a.N, 0);
    ggml_quantize_chunk(type, f.data(), out.data(), 0, a.N, a.K, nullptr);
    return true;
}

ggml_tensor * WeightLoader::declare(ggml_type want, bool required, bool gemma_norm,
                                    const char * fmt, va_list ap) {
    char name[NAME_CAP];
    const int n = std::vsnprintf(name, sizeof(name), fmt, ap);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        std::fprintf(stderr, "vla(%s): tensor name too long for a %zu-byte buffer\n", arch_, sizeof(name));
        ok_ = false;
        return nullptr;
    }

    const ggml_tensor * src = g_.meta(name);
    if (!src) {
        if (required) {
            std::fprintf(stderr, "vla(%s): missing tensor %s\n", arch_, name);
            ok_ = false;
        }
        return nullptr;
    }

    // INT8 in the file is a FoldQuant site (docs/QUANTIZATION.md), which only
    // typed(GGML_TYPE_I8) may declare; a float GEMM declare would otherwise fail
    // later in read_convert with a size mismatch that says nothing useful.
    if (src->type == GGML_TYPE_I8 && want != GGML_TYPE_I8) {
        // ... unless the site was registered to be read back as a float weight.
        const auto it = as_float_.find(name);
        if (it != as_float_.end()) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx_, float_type(want), it->second.K, it->second.N);
            if (!t) {
                std::fprintf(stderr, "vla(%s): ggml_new_tensor failed for %s\n", arch_, name);
                ok_ = false;
                return nullptr;
            }
            ggml_set_name(t, name);
            return t;
        }
        std::fprintf(stderr, "vla(%s): %s is INT8 (FoldQuant) but this site is not FoldQuant-aware in this arch\n",
                     arch_, name);
        ok_ = false;
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor(ctx_, g_.resident_type(src, want), ggml_n_dims(src), src->ne);
    if (!t) {
        std::fprintf(stderr, "vla(%s): ggml_new_tensor failed for %s (weight context too small?)\n", arch_, name);
        ok_ = false;
        return nullptr;
    }
    ggml_set_name(t, name);

    if (gemma_norm)
        gemma_norms_.push_back(name);
    return t;
}

#define VLA_DECLARE_FN(fn, type, required, gemma)                       \
    ggml_tensor * WeightLoader::fn(const char * fmt, ...) {             \
        va_list ap;                                                     \
        va_start(ap, fmt);                                              \
        ggml_tensor * t = declare(type, required, gemma, fmt, ap);      \
        va_end(ap);                                                     \
        return t;                                                       \
    }

VLA_DECLARE_FN(gemm,           gemm_,          true,  false)
VLA_DECLARE_FN(f32,            GGML_TYPE_F32,  true,  false)
VLA_DECLARE_FN(opt_gemm,       gemm_,          false, false)
VLA_DECLARE_FN(opt_f32,        GGML_TYPE_F32,  false, false)
VLA_DECLARE_FN(f32_gemma_norm, GGML_TYPE_F32,  true,  true)

#undef VLA_DECLARE_FN

ggml_tensor * WeightLoader::typed(ggml_type want, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ggml_tensor * t = declare(want, true, false, fmt, ap);
    va_end(ap);
    return t;
}

ggml_tensor * WeightLoader::opt_typed(ggml_type want, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ggml_tensor * t = declare(want, false, false, fmt, ap);
    va_end(ap);
    return t;
}

void WeightLoader::fail(const char * what) {
    std::fprintf(stderr, "vla(%s): %s\n", arch_, what);
    ok_ = false;
}

ggml_tensor * WeightLoader::fuse_typed(ggml_type want, const char * out_name, const std::vector<std::string> & srcs) {
    return fuse(want, out_name, srcs);
}

ggml_tensor * WeightLoader::fuse_gemm(const char * out_name, const std::vector<std::string> & srcs) {
    return fuse(gemm_, out_name, srcs);
}

ggml_tensor * WeightLoader::fuse_f32(const char * out_name, const std::vector<std::string> & srcs) {
    return fuse(GGML_TYPE_F32, out_name, srcs);
}

ggml_tensor * WeightLoader::fuse(ggml_type want, const char * out_name, const std::vector<std::string> & srcs) {
    if (srcs.empty()) {
        ok_ = false;
        return nullptr;
    }

    const ggml_tensor * first = g_.meta(srcs[0].c_str());
    if (!first) {
        std::fprintf(stderr, "vla(%s): missing tensor %s\n", arch_, srcs[0].c_str());
        ok_ = false;
        return nullptr;
    }

    // A source registered with as_float() reads as an F32 [K, N] tensor.
    struct View { ggml_type file, resident; int64_t ne0, ne1; int nd; };
    auto view = [&](const std::string & s, const ggml_tensor * gs) {
        const auto it = as_float_.find(s);
        if (it != as_float_.end())
            return View{GGML_TYPE_F32, float_type(want), it->second.K, it->second.N, 2};
        return View{gs->type, g_.resident_type(gs, want), gs->ne[0], gs->ne[1], ggml_n_dims(gs)};
    };
    const View      v0   = view(srcs[0], first);
    const ggml_type rt   = v0.resident;
    const bool      is1d = v0.nd == 1;
    int64_t         rows = 0;
    for (const std::string & s : srcs) {
        const ggml_tensor * gs = g_.meta(s.c_str());
        if (!gs) {
            std::fprintf(stderr, "vla(%s): missing tensor %s\n", arch_, s.c_str());
            ok_ = false;
            return nullptr;
        }
        // Same resident type and row shape; a tensor copied raw (a packed type,
        // FoldQuant INT8 codes) also needs every source in that same type.
        const View v = view(s, gs);
        if (v.resident != rt || (rt == v0.file && v.file != v0.file) ||
            (!is1d && v.ne0 != v0.ne0) || v.nd != v0.nd) {
            std::fprintf(stderr, "vla(%s): %s does not match %s for fusing\n", arch_, s.c_str(), srcs[0].c_str());
            ok_ = false;
            return nullptr;
        }
        rows += is1d ? v.ne0 : v.ne1;
    }

    ggml_tensor * t = is1d ? ggml_new_tensor_1d(ctx_, rt, rows)
                           : ggml_new_tensor_2d(ctx_, rt, v0.ne0, rows);
    if (!t) {
        std::fprintf(stderr, "vla(%s): ggml_new_tensor failed for %s\n", arch_, out_name);
        ok_ = false;
        return nullptr;
    }
    ggml_set_name(t, out_name);
    fused_.push_back(Fused{t, srcs});
    return t;
}

WeightLoader::QWeight WeightLoader::quant_i8(const std::string & out_name, const std::vector<std::string> & srcs) {
    return quant(out_name, srcs, 8, 0);
}

WeightLoader::QWeight WeightLoader::quant_i4(const std::string & out_name, const std::vector<std::string> & srcs,
                                             int64_t group) {
    return quant(out_name, srcs, 4, group);
}

WeightLoader::QWeight WeightLoader::quant(const std::string & out_name, const std::vector<std::string> & srcs,
                                          int bits, int64_t group) {
    int64_t K = -1, N = 0;
    for (const std::string & s : srcs) {
        const ggml_tensor * gs = g_.meta(s.c_str());
        if (!gs || ggml_n_dims(gs) != 2 || (K >= 0 && gs->ne[0] != K)) {
            std::fprintf(stderr, "vla(%s): %s missing or not a [%lld, N] weight for INT%d\n", arch_, s.c_str(),
                         (long long) K, bits);
            ok_ = false;
            return {};
        }
        K = gs->ne[0];
        N += gs->ne[1];
    }
    QWeight q;
    q.bits  = bits;
    q.group = bits == 4 ? (group > 0 ? group : K) : 0;
    if (bits == 4 && (K % 2 || K % q.group)) {
        std::fprintf(stderr, "vla(%s): %s: K %lld not a multiple of the INT4 group %lld\n", arch_, out_name.c_str(),
                     (long long) K, (long long) q.group);
        ok_ = false;
        return {};
    }
    q.w = ggml_new_tensor_2d(ctx_, GGML_TYPE_I8, bits == 4 ? K / 2 : K, N);
    q.s = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, bits == 4 ? (K / q.group) * N : N);
    if (!q.w || !q.s) {
        std::fprintf(stderr, "vla(%s): ggml_new_tensor failed for %s\n", arch_, out_name.c_str());
        ok_ = false;
        return {};
    }
    ggml_set_name(q.w, out_name.c_str());
    ggml_set_name(q.s, (out_name + ".scale").c_str());
    quant_.push_back(Quant{q, K, srcs});
    return q;
}

bool WeightLoader::upload(ggml_backend_t backend, ggml_backend_buffer_t * out_buf) {
    if (!ok_) {
        std::fprintf(stderr, "vla(%s): weight tensor setup failed\n", arch_);
        return false;
    }

    ggml_backend_buffer_t buf = alloc_weights(ctx_, backend);
    if (!buf) {
        std::fprintf(stderr, "vla(%s): alloc_weights failed (OOM?)\n", arch_);
        return false;
    }
    *out_buf = buf;

    for (ggml_tensor * t=ggml_get_first_tensor(ctx_); t; t=ggml_get_next_tensor(ctx_, t)) {
        const char * name = ggml_get_name(t);
        const bool fused = std::any_of(fused_.begin(), fused_.end(),
                                       [&](const Fused & f) { return f.dst == t; }) ||
                           std::any_of(quant_.begin(), quant_.end(),
                                       [&](const Quant & f) { return f.q.w == t || f.q.s == t; });
        if (fused)
            continue;

        const bool gn = std::find(gemma_norms_.begin(), gemma_norms_.end(), name) != gemma_norms_.end();

        std::vector<uint8_t> bytes;
        if (!read_resident(name, t->type, gn, bytes) || bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr,
                         "vla(%s): failed to load %s (got %zu bytes, expected %zu, type=%d)\n",
                         arch_, name, bytes.size(), ggml_nbytes(t), (int) t->type);
            return false;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }

    for (const Fused & f : fused_) {
        std::vector<uint8_t> parts;
        for (const std::string & s : f.srcs) {
            std::vector<uint8_t> b;
            if (!read_resident(s, f.dst->type, false, b)) {
                std::fprintf(stderr, "vla(%s): fused fill: read %s failed\n", arch_, s.c_str());
                return false;
            }
            parts.insert(parts.end(), b.begin(), b.end());
        }
        if (parts.size() != ggml_nbytes(f.dst)) {
            std::fprintf(stderr, "vla(%s): fused fill: %s size %zu vs %zu\n",
                         arch_, ggml_get_name(f.dst), parts.size(), ggml_nbytes(f.dst));
            return false;
        }
        ggml_backend_tensor_set(f.dst, parts.data(), 0, parts.size());
    }

    for (const Quant & f : quant_) {
        const int64_t K = f.K;
        const int64_t g = f.q.bits == 4 ? f.q.group : K;
        const float   qmax = f.q.bits == 4 ? 7.0f : 127.0f;
        std::vector<int8_t> codes;   // INT8: one per byte; INT4: packed below
        std::vector<float>  scales;  // [n][K/g] while filling
        for (const std::string & s : f.srcs) {
            std::vector<uint8_t> b;
            if (!read_resident(s, GGML_TYPE_F32, false, b) || b.size() % (K * sizeof(float))) {
                std::fprintf(stderr, "vla(%s): INT fill: read %s failed\n", arch_, s.c_str());
                return false;
            }
            const float * w = (const float *) b.data();
            const int64_t n = (int64_t) (b.size() / (K * sizeof(float)));
            for (int64_t r = 0; r < n; ++r) {
                const float * row = w + r * K;
                for (int64_t k0 = 0; k0 < K; k0 += g) {
                    float amax = 0.0f;
                    for (int64_t k = k0; k < k0 + g; ++k) amax = std::max(amax, std::fabs(row[k]));
                    const float sc = amax > 0.0f ? amax / qmax : 1.0f;
                    scales.push_back(sc);
                    for (int64_t k = k0; k < k0 + g; ++k)
                        codes.push_back((int8_t) std::lrint(std::min(qmax, std::max(-qmax, row[k] / sc))));
                }
            }
        }
        const int64_t N = (int64_t) (codes.size() / K);
        if (N != f.q.w->ne[1]) {
            std::fprintf(stderr, "vla(%s): INT fill: %s rows %lld vs %lld\n", arch_, ggml_get_name(f.q.w),
                         (long long) N, (long long) f.q.w->ne[1]);
            return false;
        }
        if (f.q.bits == 4) {
            std::vector<uint8_t> packed((size_t) (N * K / 2));
            for (size_t i = 0; i < packed.size(); ++i)
                packed[i] = (uint8_t) ((codes[2 * i] & 0xf) | ((codes[2 * i + 1] & 0xf) << 4));
            std::vector<float> st((size_t) ((K / g) * N));  // [K/g][N]
            for (int64_t n = 0; n < N; ++n)
                for (int64_t j = 0; j < K / g; ++j) st[j * N + n] = scales[n * (K / g) + j];
            ggml_backend_tensor_set(f.q.w, packed.data(), 0, packed.size());
            ggml_backend_tensor_set(f.q.s, st.data(), 0, st.size() * sizeof(float));
            continue;
        }
        ggml_backend_tensor_set(f.q.w, codes.data(), 0, codes.size());
        ggml_backend_tensor_set(f.q.s, scales.data(), 0, scales.size() * sizeof(float));
    }
    return true;
}

}
