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

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace vla {

namespace {

constexpr size_t NAME_CAP = 256;

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

    ggml_tensor * t = ggml_new_tensor(ctx_, g_.resident_type(src, want), ggml_n_dims(src), src->ne);
    if (!t) {
        std::fprintf(stderr, "vla(%s): ggml_new_tensor failed for %s (weight context too small?)\n", arch_, name);
        ok_ = false;
        return nullptr;
    }
    ggml_set_name(t, name);

    if (gemma_norm) gemma_norms_.push_back(name);
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

ggml_tensor * WeightLoader::fuse_gemm(const char * out_name, const std::vector<std::string> & srcs) {
    return fuse(gemm_, out_name, srcs);
}

ggml_tensor * WeightLoader::fuse_f32(const char * out_name, const std::vector<std::string> & srcs) {
    return fuse(GGML_TYPE_F32, out_name, srcs);
}

ggml_tensor * WeightLoader::fuse(ggml_type want, const char * out_name, const std::vector<std::string> & srcs) {
    if (srcs.empty()) { ok_ = false; return nullptr; }

    const ggml_tensor * first = g_.meta(srcs[0].c_str());
    if (!first) {
        std::fprintf(stderr, "vla(%s): missing tensor %s\n", arch_, srcs[0].c_str());
        ok_ = false;
        return nullptr;
    }

    const bool is1d = ggml_n_dims(first) == 1;
    int64_t    rows = 0;
    for (const std::string & s : srcs) {
        const ggml_tensor * gs = g_.meta(s.c_str());
        if (!gs) {
            std::fprintf(stderr, "vla(%s): missing tensor %s\n", arch_, s.c_str());
            ok_ = false;
            return nullptr;
        }
        rows += is1d ? gs->ne[0] : gs->ne[1];
    }

    ggml_tensor * t = is1d ? ggml_new_tensor_1d(ctx_, want, rows)
                           : ggml_new_tensor_2d(ctx_, want, first->ne[0], rows);
    if (!t) {
        std::fprintf(stderr, "vla(%s): ggml_new_tensor failed for %s\n", arch_, out_name);
        ok_ = false;
        return nullptr;
    }
    ggml_set_name(t, out_name);
    fused_.push_back(Fused{t, srcs});
    return t;
}

bool WeightLoader::upload(ggml_backend_t backend, ggml_backend_buffer_t * out_buf) {
    if (!ok_) {
        std::fprintf(stderr, "vla(%s): weight tensor setup failed\n", arch_);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buf) {
        std::fprintf(stderr, "vla(%s): ggml_backend_alloc_ctx_tensors failed (OOM?)\n", arch_);
        return false;
    }
    *out_buf = buf;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) {
        const char * name = ggml_get_name(t);
        const bool fused = std::any_of(fused_.begin(), fused_.end(),
                                       [&](const Fused & f) { return f.dst == t; });
        if (fused) continue;

        const bool gn = std::find(gemma_norms_.begin(), gemma_norms_.end(), name) != gemma_norms_.end();

        std::vector<uint8_t> bytes = g_.read_convert(name, t->type, gn);
        if (bytes.empty() || bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(%s): failed to load %s (%zu vs %zu bytes)\n",
                         arch_, name, bytes.size(), ggml_nbytes(t));
            return false;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }

    for (const Fused & f : fused_) {
        std::vector<uint8_t> buf;
        for (const std::string & s : f.srcs) {
            std::vector<uint8_t> b = g_.read_convert(s.c_str(), f.dst->type);
            if (b.empty()) {
                std::fprintf(stderr, "vla(%s): fused fill: read %s failed\n", arch_, s.c_str());
                return false;
            }
            buf.insert(buf.end(), b.begin(), b.end());
        }
        if (buf.size() != ggml_nbytes(f.dst)) {
            std::fprintf(stderr, "vla(%s): fused fill: %s size %zu vs %zu\n",
                         arch_, ggml_get_name(f.dst), buf.size(), ggml_nbytes(f.dst));
            return false;
        }
        ggml_backend_tensor_set(f.dst, buf.data(), 0, buf.size());
    }
    return true;
}

}
