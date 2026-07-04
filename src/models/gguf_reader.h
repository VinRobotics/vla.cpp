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

// Shared GGUF reader for the in-tree model loaders. Reads a tensor as F32/BF16
// (dequantizing BF16), passes packed bytes through when the resident type is not
// float, and fetches token-embedding rows on demand. The arch name only labels
// error messages.

#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vla {

struct gguf_reader {
    const char *   arch     = "vla";
    gguf_context * gctx     = nullptr;
    ggml_context * meta_ctx = nullptr;
    FILE *         fp       = nullptr;
    size_t         data_off = 0;

    explicit gguf_reader(const char * arch_ = "vla") : arch(arch_) {}
    ~gguf_reader() {
        if (fp)       std::fclose(fp);
        if (gctx)     gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
    }
    gguf_reader(const gguf_reader &) = delete;
    gguf_reader & operator=(const gguf_reader &) = delete;

    bool open(const std::string & path) {
        gguf_init_params p{};
        p.no_alloc = true;
        p.ctx      = &meta_ctx;
        gctx = gguf_init_from_file(path.c_str(), p);
        if (!gctx) { std::fprintf(stderr, "vla(%s): gguf_init_from_file failed for %s\n", arch, path.c_str()); return false; }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) { std::fprintf(stderr, "vla(%s): fopen failed for %s\n", arch, path.c_str()); return false; }
        data_off = gguf_get_data_offset(gctx);
        return true;
    }

    bool        has(const char * k) const { return gguf_find_key(gctx, k) >= 0; }
    uint32_t    u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }
    float       f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }
    double      f64(const char * k) const { return gguf_get_val_f64(gctx, gguf_find_key(gctx, k)); }
    std::string str(const char * k) const { const int64_t id = gguf_find_key(gctx, k); return id < 0 ? std::string() : std::string(gguf_get_val_str(gctx, id)); }
    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    bool read_raw(const char * name, void * buf) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) { std::fprintf(stderr, "vla(%s): missing tensor %s\n", arch, name); return false; }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }

    std::vector<float> read_f32(const char * name) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(%s): missing tensor %s\n", arch, name); return {}; }
        const int64_t n = ggml_nelements(t);
        std::vector<float> out(n);
        if (t->type == GGML_TYPE_F32) { if (!read_raw(name, out.data())) return {}; }
        else if (t->type == GGML_TYPE_BF16) { std::vector<ggml_bf16_t> tmp(n); if (!read_raw(name, tmp.data())) return {}; ggml_bf16_to_fp32_row(tmp.data(), out.data(), n); }
        else { std::fprintf(stderr, "vla(%s): tensor %s unsupported type %d\n", arch, name, (int) t->type); return {}; }
        return out;
    }

    // F32/BF16 targets dequantize to that resident type. A non-float target (I8,
    // and the packed quant types once converters emit them) is stored raw so
    // ggml_mul_mat dequantizes at compute. gemma_norm adds 1.0 per weight.
    std::vector<uint8_t> read_convert(const char * name, ggml_type target, bool gemma_norm = false) {
        if (target == GGML_TYPE_I8) {
            const ggml_tensor * t = meta(name);
            if (!t) { std::fprintf(stderr, "vla(%s): missing tensor %s\n", arch, name); return {}; }
            std::vector<uint8_t> o(ggml_nbytes(t));
            if (!read_raw(name, o.data())) return {};
            return o;
        }
        std::vector<float> f = read_f32(name);
        if (f.empty()) return {};
        const int64_t n = (int64_t) f.size();
        if (gemma_norm) for (int64_t i = 0; i < n; ++i) f[i] += 1.0f;
        if (target == GGML_TYPE_F32)  { std::vector<uint8_t> o(n * sizeof(float));      std::memcpy(o.data(), f.data(), o.size()); return o; }
        if (target == GGML_TYPE_BF16) { std::vector<uint8_t> o(n * sizeof(ggml_bf16_t)); ggml_fp32_to_bf16_row(f.data(), reinterpret_cast<ggml_bf16_t *>(o.data()), n); return o; }
        std::fprintf(stderr, "vla(%s): unsupported resident type %d for %s\n", arch, (int) target, name); return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids, float * dst, int64_t cols) {
        const ggml_tensor * t = meta(name);
        if (!t || t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) { std::fprintf(stderr, "vla(%s): %s shape unfit for row-fetch\n", arch, name); return false; }
        const int64_t rows = t->ne[1];
        const int64_t id   = gguf_find_tensor(gctx, name);
        const size_t  base = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t  elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
        const size_t  rb   = (size_t) cols * elsz;
        std::vector<uint8_t> row(rb);
        for (size_t k = 0; k < row_ids.size(); ++k) {
            const int32_t r = row_ids[k];
            if (r < 0 || r >= rows) { std::fprintf(stderr, "vla(%s): row %d out of range for %s\n", arch, r, name); return false; }
            if (std::fseek(fp, (long) (base + (size_t) r * rb), SEEK_SET) != 0) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

}  // namespace vla
