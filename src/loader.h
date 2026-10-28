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

// gemm() lands in the model's matmul type, unless the GGUF holds the tensor
// quantized, in which case it stays packed and ggml dequantizes at compute.
// A miss is recorded and surfaces once, at ok().

#pragma once

#include "gguf_reader.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdarg>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace vla {

class WeightLoader {
public:
    WeightLoader(const char * arch, gguf_reader & g, ggml_context * ctx, ggml_type gemm_type)
        : arch_(arch), g_(g), ctx_(ctx), gemm_(gemm_type) {}

    WeightLoader(const WeightLoader &)             = delete;
    WeightLoader & operator=(const WeightLoader &) = delete;

    ggml_tensor * gemm(const char * fmt, ...) __attribute__((format(printf, 2, 3)));
    ggml_tensor * f32 (const char * fmt, ...) __attribute__((format(printf, 2, 3)));

    // Explicit resident type, for weights that are neither a plain GEMM input
    // nor F32 (BitVLA's int2-packed ternary blocks).
    ggml_tensor * typed(ggml_type want, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
    ggml_tensor * opt_typed(ggml_type want, const char * fmt, ...) __attribute__((format(printf, 3, 4)));

    // A miss is not an error.
    ggml_tensor * opt_gemm(const char * fmt, ...) __attribute__((format(printf, 2, 3)));
    ggml_tensor * opt_f32 (const char * fmt, ...) __attribute__((format(printf, 2, 3)));

    // Gemma norms are centred on zero and add 1 at use.
    ggml_tensor * f32_gemma_norm(const char * fmt, ...) __attribute__((format(printf, 2, 3)));

    // Several GGUF tensors concatenated into one resident tensor; out_name is
    // synthetic and need not exist in the file.
    ggml_tensor * fuse_gemm(const char * out_name, const std::vector<std::string> & srcs);
    ggml_tensor * fuse_f32 (const char * out_name, const std::vector<std::string> & srcs);
    ggml_tensor * fuse_typed(ggml_type want, const char * out_name, const std::vector<std::string> & srcs);

    // Float GGUF weights [K, N_i], concatenated along N, as symmetric per-output-
    // channel INT8: codes I8 [K, N] and scales F32 [N], w[n][k] ~ code * scale[n].
    // Quantized from F32 at upload (layers/qlinear.h runs them).
    struct QWeight {
        ggml_tensor * w = nullptr, * s = nullptr;
        int           bits = 8;
        int64_t       group = 0;  // INT4: K-elements per scale (0: per channel)
        explicit operator bool() const { return w != nullptr; }
    };
    QWeight quant_i8(const std::string & out_name, const std::vector<std::string> & srcs);
    // The same as symmetric INT4 (codes -7..7, two per byte along K, low nibble
    // first: I8 [K/2, N]) with one F32 scale per `group` K-elements and output
    // channel (F32 [N, K/group], scale[g * N + n]); group 0 = one per channel.
    QWeight quant_i4(const std::string & out_name, const std::vector<std::string> & srcs, int64_t group);

    // A tensor of another type in the file (a FoldQuant site's INT codes) that
    // gemm/opt_gemm/fuse_gemm declare as a float [K, N] weight in the resident
    // GEMM type; make() returns its N*K floats (row-major [N][K]) at upload.
    using FloatMaker = std::function<bool(std::vector<float> &)>;
    void as_float(const std::string & name, int64_t K, int64_t N, FloatMaker make);

    // For helpers that inspect the file before declaring (foldquant.cpp).
    gguf_reader & reader() {
        return g_;
    }
    // Record a failure detected outside declare(); upload() then refuses.
    void fail(const char * what);

    ggml_type gemm_type() const {
        return gemm_;
    }
    bool      ok()        const {
        return ok_;
    }

    bool upload(ggml_backend_t backend, ggml_backend_buffer_t * out_buf);

private:
    ggml_tensor * declare(ggml_type want, bool required, bool gemma_norm, const char * fmt, va_list ap);
    ggml_tensor * fuse(ggml_type want, const char * out_name, const std::vector<std::string> & srcs);
    QWeight       quant(const std::string & out_name, const std::vector<std::string> & srcs, int bits, int64_t group);
    // Bytes of a resident tensor filled from GGUF tensor `name` (a registered
    // float rebuild, or the file's tensor converted).
    bool read_resident(const std::string & name, ggml_type type, bool gemma_norm, std::vector<uint8_t> & out);

    struct AsFloat {
        int64_t    K = 0, N = 0;
        FloatMaker make;
    };
    std::map<std::string, AsFloat> as_float_;

    const char *   arch_;
    gguf_reader &  g_;
    ggml_context * ctx_;
    ggml_type      gemm_;
    bool           ok_ = true;

    struct Fused {
        ggml_tensor *            dst;
        std::vector<std::string> srcs;
    };

    std::vector<std::string> gemma_norms_;
    std::vector<Fused>       fused_;
    struct Quant {
        QWeight                  q;
        int64_t                  K = 0;
        std::vector<std::string> srcs;
    };
    std::vector<Quant>       quant_;
};

}
