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

    // A miss is not an error.
    ggml_tensor * opt_gemm(const char * fmt, ...) __attribute__((format(printf, 2, 3)));
    ggml_tensor * opt_f32 (const char * fmt, ...) __attribute__((format(printf, 2, 3)));

    // Gemma norms are centred on zero and add 1 at use.
    ggml_tensor * f32_gemma_norm(const char * fmt, ...) __attribute__((format(printf, 2, 3)));

    // Several GGUF tensors concatenated into one resident tensor; out_name is
    // synthetic and need not exist in the file.
    ggml_tensor * fuse_gemm(const char * out_name, const std::vector<std::string> & srcs);
    ggml_tensor * fuse_f32 (const char * out_name, const std::vector<std::string> & srcs);

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
};

}
