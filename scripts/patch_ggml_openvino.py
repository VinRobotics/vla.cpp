#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nine fixes to the fetched ggml OpenVINO backend.

ggml-openvino is written against llama.cpp's graphs: one decoder-only
transformer, one position input, an F16 KV cache. vla.cpp drives it with vision
towers and action experts instead, which is legal ggml but nothing the backend
has seen. Five of the hunks below are places where an llama.cpp-shaped
assumption is narrower than the ggml contract; the sixth is a missing cache on
the path those graphs take. Together they are what lets SmolVLA and pi0.5 run
end to end on the CPU, GPU and NPU plugins. Number 4 is the one that matters
most and the one worth upstreaming.

See docs/backend/ov.md for the measured results and for what is still blocked.

  1. ggml-openvino-extra.cpp - pick the *Intel* OpenCL platform.
     `GGML_OPENVINO_DEVICE=GPU` builds an OpenVINO remote context on an OpenCL
     queue and takes the first platform the ICD loader reports. With more than
     one runtime installed (an NVIDIA card next to the Intel iGPU, POCL,
     Rusticl) that is whichever `/etc/OpenCL/vendors/*.icd` sorted first, and
     the GPU plugin only accepts an Intel context -- it aborts at startup with
     "Incompatible OpenCL runtime: program is not in expected ELF format".
     Selecting by `CL_PLATFORM_VENDOR` leaves single-runtime boxes unchanged.

  2. ggml-decoder.cpp - narrow RESHAPE op_case 3.
     Case 3 is the KV-cache flatten, `[512,1024,1,1] -> [1,524288,1,1]`, and it
     emits a shape with `-1` in dim 2 and 1 in dim 3. Its guard only tests
     `src->ne[0]*ne[1]*ne[2] == node->ne[1]`, which also matches the kernel
     reshape inside `ggml_conv_2d` (`[16,16,3,768] -> [768,768]`) and mangles
     it. The real case always has `node->ne[0] == 1`; requiring that sends the
     conv kernel to case 6, the plain reshape.

  3. openvino/op/flash_attn_ext.cpp - convert K/V to F16 with Q.
     The translator converts Q, the mask and the scale to F16 because
     llama.cpp's KV cache already is. vla.cpp keeps K/V in F32, and OpenVINO's
     SDPA rejects mixed input types ("Mixed input types are not supported").
     Converting K/V too matches the precision the translator already chose.

  4. ggml-decoder.{h,cpp} - stop distinct position inputs aliasing each other.
     Every tensor feeding a ROPE's second input is renamed to one graph
     parameter called "inp_pos", because an llama.cpp graph has exactly one.
     add_rope_sin_cos() then builds a single shared sin/cos table from it. Every
     vla.cpp arch has several position tensors -- SmolVLA passes a prefill, a
     full and a rebased one -- so they alias each other and every ROPE takes the
     table built from whichever won, which fails shape inference:
     "Multiply (Split[1]:f32[1,113,5,32], Multiply[0]:f32[1,50,1,32])
      Argument shapes are inconsistent."
     When the graph has more than one, keep each tensor's own name. Nothing is
     then called "inp_pos", so add_rope_sin_cos() returns early and the existing
     fallback in translate_rope() builds sin/cos per op from its own position
     input. Single-position graphs are untouched and keep the shared table.
     This one is what carries an arch through to a full prediction.

  5. utils.{h,cpp} - cache what the naive path compiles.
     The dynamic and static paths keep a `graph_key`-indexed cache of the
     decoder and the compiled infer request; the naive path has none, so it
     rebuilt the decoder, re-converted the model and called compile_model() on
     every single ggml_backend_graph_compute. That is the dominant cost once a
     real graph goes through it: SmolVLA on the CPU plugin drops from 22.7 s to
     1.8 s per prediction with the cache in place. A hit rebinds the cached
     decoder to the new graph through the existing update_io(), which is how the
     dynamic path already handles freshly built tensors.

  6. openvino/op_table.cpp - add the four missing op translators.
     RELU, GELU_ERF, NEG and SQR have no entry in the table, and with no per-op
     CPU fallback in the core an arch that uses one cannot run at all. Between
     them they block every arch except SmolVLA, pi0 and pi0.5. All four map
     onto ov ops directly: Relu, Gelu (whose default is the exact erf form
     GELU_ERF asks for), Negative, and -- since no single-input ov op squares --
     Multiply with the input on both sides.

  7. openvino/utils.cpp - give a folded weight its full rank before slicing.
     A ggml tensor that is 2-D folds in as a rank-2 ov constant, which is what a
     GEMM operand wants, but process_view_input_new() indexes a viewed tensor at
     its full ggml rank, so the slice axis lands outside it:
     "Slice (Constant aex.blk.0.attn_in.weight[0]:bf16[2688,896], ...)
      Axis 2 out of the tensor rank range [-2, 1]."
     Evo-1 hits it by viewing Q, K and V out of one fused attn_in weight.
     Left-pad the input with 1s, which is the shape ggml gave it anyway.

  8. openvino/op/concat.cpp - align input ranks.
     The same rank-2 constants reach CONCAT, which unlike the broadcasting
     elementwise ops needs both inputs at the graph's rank or the axis falls
     outside them. Evo-1 concatenates a CLS weight onto 4-D patch embeddings;
     SmolVLA concatenates a precomputed time tile onto a 4-D activation. Padding
     here is what let vla.cpp drop an arch-specific workaround that had moved
     SmolVLA's time tiles into their own buffer.

  9. utils.cpp - make the naive-path graph-size threshold settable.
     Graphs under 20 nodes bypass the LLM decoder and translate literally, with
     static shapes and no KV-cache inference. That literal path is the one that
     suits a vision tower, but a vision tower is ~450 nodes. The constant
     becomes `GGML_OPENVINO_NAIVE_GRAPH_SIZE`; src/backend.h defaults it high
     for vla.cpp and an explicit setting still wins.

Idempotent - re-running on a patched tree is a no-op, so a reconfigure that
re-populates the FetchContent source dir is safe either way.

Usage: scripts/patch_ggml_openvino.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp:"

HELPER = """// vla.cpp: select the Intel OpenCL platform. With several OpenCL runtimes
// installed the first platform is not always Intel's, and the OpenVINO GPU
// plugin only accepts an Intel context. Cached: the ICD list cannot change
// under a running process.
static cl_platform_id ggml_openvino_get_intel_platform() {
    static cl_platform_id platform = nullptr;
    static bool searched = false;
    if (searched) {
        return platform;
    }
    searched = true;

    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        return nullptr;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return nullptr;
    }

    for (cl_platform_id p : platforms) {
        char vendor[256] = "";
        if (clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr) != CL_SUCCESS) {
            continue;
        }
        if (strstr(vendor, "Intel") != nullptr) {
            platform = p;
            break;
        }
    }
    return platform;
}

"""

USM_LOOKUP = """        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (%s_fn) clGetExtensionFunctionAddressForPlatform(platform, "%s");
"""

USM_LOOKUP_NEW = """        cl_platform_id platform = ggml_openvino_get_intel_platform();
        if (platform != nullptr) {
            fn = (%s_fn) clGetExtensionFunctionAddressForPlatform(platform, "%s");
"""

NAIVE_COMPUTE_OLD = """enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    bool naive = true;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
    if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    } else {
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    }
    if (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), config).create_infer_request());
    } else {
        infer_request =
            std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
    }

    auto ov_params = model->get_parameters();"""

NAIVE_COMPUTE_NEW = """enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config,
                               std::shared_ptr<ov_runtime_context> r_ctx) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    // vla.cpp: reuse the decoder, the converted model and the compiled infer
    // request across calls on the same graph, the way the dynamic and static
    // paths already do. Conversion plus compile_model dominates a naive call, so
    // without this every graph_compute pays it again.
    static const bool cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    const graph_key key(cgraph);

    std::shared_ptr<naive_runtime_ctx> entry;
    bool cache_hit = false;
    if (cache_enabled && r_ctx != nullptr) {
        std::lock_guard<std::mutex> lock(r_ctx->ctx_mutex);
        auto it = r_ctx->naive_cache.find(key);
        if (it != r_ctx->naive_cache.end()) {
            entry = it->second;
            cache_hit = true;
        } else {
            entry = std::make_shared<naive_runtime_ctx>();
            r_ctx->naive_cache[key] = entry;
        }
    } else {
        entry = std::make_shared<naive_runtime_ctx>();
    }

    // One graph at a time: an ov::InferRequest is not re-entrant, and a hit
    // rebinds the decoder to this cgraph.
    std::lock_guard<std::mutex> entry_lock(entry->mutex);

    bool naive = true;
    std::shared_ptr<GgmlOvDecoder> decoder;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::InferRequest> infer_request;

    if (cache_hit && entry->infer_request != nullptr) {
        decoder = entry->decoder;
        model = entry->model;
        infer_request = entry->infer_request;
        // Same shapes, new tensors: point the decoder at this call's graph.
        decoder->update_io(cgraph);
    } else {
        auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
        decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
        auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
        model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
        if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
            ov::serialize(model, "IR_naive.xml");
        }

        auto remote_context = ggml_openvino_get_remote_context();
        if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
            // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
            core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
        } else {
            core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
        }
        if (remote_context.has_value()) {
            infer_request = std::make_shared<ov::InferRequest>(
                core.compile_model(model, remote_context.value(), config).create_infer_request());
        } else {
            infer_request =
                std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
        }

        entry->decoder = decoder;
        entry->model = model;
        entry->infer_request = infer_request;
    }

    auto ov_params = model->get_parameters();"""

# file -> [(anchor, replacement), ...]. Every anchor must match exactly once.
EDITS = {
    "ggml/src/ggml-openvino/ggml-openvino-extra.cpp": [
        ("#include <optional>\n", "#include <optional>\n#include <vector>\n"),
        ("void ggml_openvino_device_config::init() {", HELPER + "void ggml_openvino_device_config::init() {"),
        (
            """        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL platform: %d\\n", err);
            return;
        }
""",
            """        cl_int err;
        cl_platform_id platform = ggml_openvino_get_intel_platform();
        if (platform == nullptr) {
            GGML_LOG_ERROR("Failed to find an Intel OpenCL platform\\n");
            return;
        }
""",
        ),
        (USM_LOOKUP % (("clEnqueueMemFillINTEL",) * 2), USM_LOOKUP_NEW % (("clEnqueueMemFillINTEL",) * 2)),
        (USM_LOOKUP % (("clEnqueueMemcpyINTEL",) * 2), USM_LOOKUP_NEW % (("clEnqueueMemcpyINTEL",) * 2)),
    ],
    "ggml/src/ggml-openvino/openvino/op/concat.cpp": [
        (
            """#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>""",
            """#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/unsqueeze.hpp>""",
        ),
        ("#include <memory>", "#include <memory>\n#include <numeric>"),
        (
            """    const auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{input_0, input_1}, axis);""",
            """    // vla.cpp: a weight that is 2-D in ggml is folded in as a rank-2 constant,
    // because that is what a GEMM operand wants. Concat is the one op where that
    // matters: it needs both inputs at the graph's rank, or the axis computed
    // below falls outside them. Left-pad the shorter one with 1s, which is the
    // shape ggml gave it anyway.
    auto align_rank = [rank](ov::Output<ov::Node> in) {
        const auto & ps = in.get_partial_shape();
        if (ps.rank().is_dynamic() || ps.rank().get_length() >= rank) {
            return in;
        }
        std::vector<int64_t> axes(rank - ps.rank().get_length());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_node = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
        return ov::Output<ov::Node>(std::make_shared<ov::op::v0::Unsqueeze>(in, axes_node));
    };
    input_0 = align_rank(input_0);
    input_1 = align_rank(input_1);

    const auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{input_0, input_1}, axis);""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/utils.cpp": [
        ("#include <memory>", "#include <memory>\n#include <numeric>"),
        (
            "#include <openvino/op/transpose.hpp>",
            "#include <openvino/op/transpose.hpp>\n#include <openvino/op/unsqueeze.hpp>",
        ),
        (
            """    size_t view_input_size = context.get_view_input_size(input_index);
    if (view_input_size == 0) {
        // No view inputs, return the input as is
        return input;
    }
""",
            """    size_t view_input_size = context.get_view_input_size(input_index);
    if (view_input_size == 0) {
        // No view inputs, return the input as is
        return input;
    }

    // vla.cpp: a ggml tensor that is 2-D folds in as a rank-2 ov constant, which
    // is what a GEMM operand wants, but every slice below indexes the tensor at
    // its full ggml rank. Left-pad with 1s so the axes line up. Evo-1 hits this
    // by viewing Q/K/V out of one fused attn_in weight.
    {
        const auto src_ggml_shape = context.get_view_input_src_ggml_shape(input_index, 0);
        const auto & in_ps = input.get_partial_shape();
        if (in_ps.rank().is_static() && (size_t) in_ps.rank().get_length() < src_ggml_shape.size()) {
            std::vector<int64_t> axes(src_ggml_shape.size() - (size_t) in_ps.rank().get_length());
            std::iota(axes.begin(), axes.end(), 0);
            input = std::make_shared<ov::op::v0::Unsqueeze>(
                input, ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes));
        }
    }
""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op_table.cpp": [
        (
            """#include <openvino/op/multiply.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/tanh.hpp>

namespace ov {
namespace frontend {
namespace ggml {

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {""",
            """#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/tanh.hpp>

namespace ov {
namespace frontend {
namespace ggml {

namespace op {
// vla.cpp: no ov op takes one input and squares it, so pair the input with
// itself rather than route it through Power and a constant exponent.
OutputVector translate_sqr(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(input, input);
    return rename_outputs_with_suffix({res}, context.get_name());
}
}  // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {""",
        ),
        (
            """        {"GGML_UNARY_OP_GELU",      op::translate_1to1_match_1_input<v7::Gelu>     },""",
            """        {"GGML_UNARY_OP_GELU",      op::translate_1to1_match_1_input<v7::Gelu>     },
        // vla.cpp: ov's Gelu defaults to the exact erf formulation, which is what
        // GELU_ERF asks for. GGML_UNARY_OP_GELU above is ggml's tanh
        // approximation and keeps the mapping it already had.
        {"GGML_UNARY_OP_GELU_ERF",  op::translate_1to1_match_1_input<v7::Gelu>     },
        {"GGML_UNARY_OP_RELU",      op::translate_1to1_match_1_input<v0::Relu>     },
        {"GGML_OP_NEG",             op::translate_1to1_match_1_input<v0::Negative> },
        {"GGML_OP_SQR",             op::translate_sqr                              },""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/flash_attn_ext.cpp": [
        (
            """    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);""",
            """    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    // vla.cpp: Q, the mask and the scale below are all forced to F16 because
    // llama.cpp's KV cache already is. K/V that arrive as F32 have to come along
    // or SDPA rejects the mix.
    if (k.get_element_type() != ov::element::f16) {
        k = std::make_shared<ov::op::v0::Convert>(k, ov::element::f16);
    }
    if (v.get_element_type() != ov::element::f16) {
        v = std::make_shared<ov::op::v0::Convert>(v, ov::element::f16);
    }""",
        ),
    ],
    "ggml/src/ggml-openvino/ggml-decoder.h": [
        (
            """    std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) {
        if (is_inp_pos(tensor, op)) {
            return "inp_pos";
        }""",
            """    // vla.cpp: only collapse ROPE position inputs onto one "inp_pos" parameter
    // when the graph really has one. See scripts/patch_ggml_openvino.py.
    bool has_multiple_inp_pos() const;

    std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) {
        if (is_inp_pos(tensor, op)) {
            return has_multiple_inp_pos() ? std::string(tensor->name) : std::string("inp_pos");
        }""",
        ),
        (
            "    ggml_cgraph * m_cgraph = nullptr;",
            "    ggml_cgraph * m_cgraph = nullptr;\n"
            "    mutable int m_multi_inp_pos = -1;  // vla.cpp: lazily computed, -1 = unknown",
        ),
    ],
    "ggml/src/ggml-openvino/ggml-decoder.cpp": [
        (
            """        } else if (src->ne[0] * src->ne[1] * src->ne[2] == node->ne[1]) {
            op_case = 3;""",
            """            // vla.cpp: case 3 is the KV-cache flatten, whose result is always
            // [1, n, 1, 1]. Without the ne[0] test it also swallows the kernel
            // reshape ggml_conv_2d emits and rewrites it to the wrong shape.
        } else if (src->ne[0] * src->ne[1] * src->ne[2] == node->ne[1] && node->ne[0] == 1) {
            op_case = 3;""",
        ),
        (
            "int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {",
            """bool GgmlOvDecoder::has_multiple_inp_pos() const {
    if (m_multi_inp_pos < 0) {
        std::set<const ggml_tensor *> seen;
        for (int i = 0; i < m_cgraph->n_nodes && seen.size() < 2; i++) {
            const ggml_tensor * node = m_cgraph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC && node->src[j] != nullptr; j++) {
                if (is_inp_pos(node->src[j], node)) {
                    seen.insert(node->src[j]);
                }
            }
        }
        m_multi_inp_pos = seen.size() > 1 ? 1 : 0;
    }
    return m_multi_inp_pos == 1;
}

int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {""",
        ),
    ],
    "ggml/src/ggml-openvino/utils.h": [
        (
            "struct decoder_runtime_ctx {",
            """// vla.cpp: what naive_compute() reuses across calls on the same graph. Without
// it that path rebuilt the decoder, re-converted the model and called
// compile_model() on every ggml_backend_graph_compute, which dominated runtime.
struct naive_runtime_ctx {
    std::mutex mutex;
    std::shared_ptr<GgmlOvDecoder> decoder;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::InferRequest> infer_request;
};

struct decoder_runtime_ctx {""",
        ),
        (
            "    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;",
            "    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;\n"
            "    std::unordered_map<graph_key, std::shared_ptr<naive_runtime_ctx>, graph_key_hash> naive_cache;",
        ),
        (
            """        decoder_cache.clear();
        infer_request_cache.clear();""",
            """        decoder_cache.clear();
        naive_cache.clear();
        infer_request_cache.clear();""",
        ),
        (
            """enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config);""",
            """enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config,
                               std::shared_ptr<ov_runtime_context> r_ctx);""",
        ),
    ],
    "ggml/src/ggml-openvino/utils.cpp": [
        (
            """        if (!is_model_splitted(cgraph)) {
            return naive_compute(cgraph, core, device, config);
        }""",
            """        if (!is_model_splitted(cgraph)) {
            return naive_compute(cgraph, core, device, config, r_ctx);
        }""",
        ),
        (
            """    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }""",
            """    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config, r_ctx);
    }""",
        ),
        (
            NAIVE_COMPUTE_OLD,
            NAIVE_COMPUTE_NEW,
        ),
        (
            """bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;""",
            """bool is_naive(ggml_cgraph * cgraph) {
    // vla.cpp: the literal translation path suits any graph that is not a
    // decoder-only LLM, so let the caller raise the bar it is chosen under.
    static const int naive_graph_size_threshold = [] {
        const char * env = getenv("GGML_OPENVINO_NAIVE_GRAPH_SIZE");
        return (env && *env) ? atoi(env) : 20;
    }();""",
        ),
    ],
}


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    for rel, edits in EDITS.items():
        src = root / rel
        if not src.is_file():
            print(f"patch_ggml_openvino: {src} not found", file=sys.stderr)
            return 1
        text = src.read_text()
        if MARKER in text:
            print(f"patch_ggml_openvino: {rel} already patched")
            continue
        for anchor, replacement in edits:
            n = text.count(anchor)
            if n != 1:
                print(f"patch_ggml_openvino: anchor matched {n} times in {rel}, expected 1:\n{anchor}",
                      file=sys.stderr)
                return 1
            text = text.replace(anchor, replacement)
        src.write_text(text)
        print(f"patch_ggml_openvino: patched {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
