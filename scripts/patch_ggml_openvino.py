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

"""Thirteen fixes to the fetched ggml OpenVINO backend, and the FoldQuant hook.

ggml-openvino is written against llama.cpp's graphs: one decoder-only
transformer, one position input, an F16 KV cache. vla.cpp drives it with vision
towers and action experts instead, which is legal ggml but nothing the backend
has seen. Six of the fixes below narrow an llama.cpp-shaped assumption back to
the ggml contract; the rest are about the path those graphs take and the
precision the plugins run them at. Together they are what lets the archs in
docs/backend/ov.md run end to end on the CPU, GPU and NPU plugins. Numbers 2
and 3 are the ones worth upstreaming.

Upstream took five earlier fixes between b11223 and v1.0.0, so they are gone
from here: the Intel OpenCL platform (the GPU context now comes from
OpenVINO's own), the RESHAPE op_case 3 guard, both GELU modes, the IMROPE
sector bound, and the 2D im2col -> Convolution fusion.

A hunk that applies is not a hunk that runs. Whenever VLA_LLAMA_TAG moves, check
that each patched function still has a live caller, not just that its anchor
still matches: b10729 moved the position-input naming out of the GgmlOvDecoder
member that fix 2 patches and into a free function, and the fix went silently
dead while applying cleanly.

See docs/backend/ov.md for the measured results and for what is still blocked.

  1. openvino/op/flash_attn_ext.cpp - convert K/V to F16 with Q.
     The translator converts Q, the mask and the scale to F16 because
     llama.cpp's KV cache already is. vla.cpp keeps K/V in F32, and OpenVINO's
     SDPA rejects mixed input types ("Mixed input types are not supported").
     Converting K/V too matches the precision the translator already chose.

  2. ggml-decoder.{h,cpp} - stop distinct position inputs aliasing each other.
     Every tensor feeding a ROPE's second input is renamed to one graph
     parameter called "inp_pos", because an llama.cpp graph has exactly one.
     add_rope_sin_cos() then builds a single shared sin/cos table from it. Every
     vla.cpp arch has several position tensors -- SmolVLA passes a prefill, a
     full and a rebased one -- so they alias each other and every ROPE takes the
     table built from whichever won, which fails shape inference:
     "Multiply (Split[1]:f32[1,113,5,32], Multiply[0]:f32[1,50,1,32])
      Argument shapes are inconsistent."
     When the graph has more than one, keep each tensor's own name. Nothing is
     then called "inp_pos", so translate_rope() falls back to building sin/cos
     per op from its own position input. Single-position graphs are untouched.
     This one is what carries an arch through to a full prediction.

     The naming lives in the free function get_tensor_graph_input_ov_name() in
     ggml-decoder.cpp. The GgmlOvDecoder member of the same intent has had no
     callers since b10729, so only its new has_multiple_inp_pos() declaration
     goes into the header.

     openvino/op/rope.cpp - key the per-op sin/cos cache on the position input.
     translate_rope() caches each sin/cos table under the ROPE's op_params
     alone, so two ROPEs with the same parameters share one table whatever
     their positions. pi0, pi0.5 and SmolVLA rotate the prefix and the action
     suffix with the same parameters and different position tensors, and the
     suffix then multiplies by the prefix's table:
     "Multiply (Reshape[0]:f32[1,50,1,256], Concat[0]:f32[1,528,1,256])
      Argument shapes are inconsistent." on every OpenVINO device.

  3. utils.{h,cpp} - cache what the naive path compiles.
     The dynamic and static paths keep a `graph_key`-indexed cache of the
     decoder and the compiled infer request; the naive path has none, so it
     rebuilt the decoder, re-converted the model and called compile_model() on
     every single ggml_backend_graph_compute. That is the dominant cost once a
     real graph goes through it: SmolVLA on the CPU plugin drops from 22.7 s to
     1.4 s per prediction with the cache in place. A hit rebinds the cached
     decoder to the new graph through the existing update_io(), which is how the
     dynamic path already handles freshly built tensors.
     The key is `naive_key`, not `graph_key`: the latter is n_nodes, the first
     and last node name and the input names, which two graphs of the same size
     can share, and a compiled model is bound to the shapes it was built for.
     Reusing one across a shape change returns another graph's answer with no
     error, so the key mixes in every node's op and shape. The map is bounded;
     see the comment on the flush. In front of it sits a cheaper fingerprint of
     the graph object and its tensor addresses: on a hit the cached decoder is
     already bound to these tensors, so both naive_key and update_io(), costly
     on a graph of thousands of nodes, are skipped.

  4. openvino/utils.cpp - give a folded weight its full rank before slicing.
     A ggml tensor that is 2-D folds in as a rank-2 ov constant, which is what a
     GEMM operand wants, but process_view_input_new() indexes a viewed tensor at
     its full ggml rank, so the slice axis lands outside it:
     "Slice (Constant aex.blk.0.attn_in.weight[0]:bf16[2688,896], ...)
      Axis 2 out of the tensor rank range [-2, 1]."
     Evo-1 hits it by viewing Q, K and V out of one fused attn_in weight.
     Left-pad the input with 1s, which is the shape ggml gave it anyway.

  5. openvino/op/concat.cpp - align input ranks.
     The same rank-2 constants reach CONCAT, which unlike the broadcasting
     elementwise ops needs both inputs at the graph's rank or the axis falls
     outside them. Evo-1 concatenates a CLS weight onto 4-D patch embeddings;
     SmolVLA concatenates a precomputed time tile onto a 4-D activation. Padding
     here is what saves each arch from working around it, e.g. by moving
     SmolVLA's time tiles into a buffer of their own.

  6. utils.cpp - make the naive-path graph-size threshold settable.
     Graphs under 20 nodes bypass the LLM decoder and translate literally, with
     static shapes and no KV-cache inference. That literal path is the one that
     suits a vision tower, but a vision tower is ~450 nodes. The constant
     becomes `GGML_OPENVINO_NAIVE_GRAPH_SIZE`; src/backend.h defaults it high
     for vla.cpp and an explicit setting still wins. Parsed with strtol, because
     atoi turns junk into 0 and that would send every graph down the LLM builder
     with nothing said.

  7. ggml-decoder.cpp - require a ROPE before taking PERMUTE op_case 2.
     op_case 2 rewrites the tensor as [n_seq, -1, n_heads, head_size] and only
     then transposes, which is correct for llama.cpp's rope'd query and nothing
     else. The classifier reached it for ANY permute whose source is a view of a
     non-leaf, so GR00T N1.7's DiT cross-attention V - ggml_permute(view, 1,2,0,3)
     over a fused KV projection - was reshaped into a shape unrelated to it and
     came out with its elements rearranged: V was 139% wrong while K, which uses
     permute(0,2,1,3) and happened to survive the same rewrite, was 0.04%. That
     turned into a 27% error in the final actions. Walking the view/reshape/cont
     chain and requiring a ROPE at the end sends every other permute to op_case 1,
     the plain transpose. This is what makes GR00T N1.7 correct (27% -> 0.005%).

  8. ggml-decoder.cpp + openvino/op/add.cpp - do not stack two elementwise adds
     on a GEMM. The GPU plugin folds elementwise ops into the preceding GEMM as
     post-ops. Given ADD(ADD(residual, GEMM), graph_input) it folds both, and the
     second operand is silently lost: the result equals the inner add, as though
     the outer one never ran. Nothing is logged. An llama.cpp graph never builds
     that chain - one residual add per sub-block - but a VLA does, wherever a
     tower's features are added on top of an FFN residual. It cost VLA-JEPA,
     GR00T N1.7 and pi0 their GPU support, while every other arch was unaffected.
     Addition is associative, so re-hang the outer add on the inner one's
     non-GEMM operand: the GEMM keeps exactly one post-op and the arithmetic is
     unchanged. op_case 2/3 records which of the inner add's operands is the
     GEMM, because ggml's operand order is not fixed.
     Found by bisecting VLA-JEPA with GGML_OPENVINO_DEBUG_NODE: its ViT and DiT
     graphs matched the CPU plugin to 0.2%, the VLM prefill was already wrong at
     the end of layer 0, and within that layer the divergence was one node - the
     FFN residual add under the deepstack add. The same fusion path already has a
     known defect with broadcast DIV; see ggml_backend_openvino_supports_op.

  9. ggml-openvino-extra.cpp - expose the GPU plugin's inference precision.
     The GPU plugin computes in f16 unless told otherwise, which is most of why
     it is fast, and for nearly every graph that is the right trade. It is not
     the right trade for a graph carrying a long serial chain - a flow-matching
     denoise loop unrolled inside one graph - because the error compounds across
     every step with nothing to reset it, and a saturating output channel can
     then cross its threshold in the wrong place. pi0 does exactly that: on the
     GPU its continuous action dims land 4e-2 from an F32 reference and its
     bistable gripper flips one step early, which is a 1.7 max|delta| on a metric
     and a late grasp on a robot. GGML_OPENVINO_GPU_PRECISION=f32 puts it at
     6.5e-5. Exposed rather than forced: f32 costs about 3x on this plugin, so
     src/backend.h defaults it for pi0 alone and an explicit setting still wins.

 10. utils.cpp - let a model opt every naive graph into PERFORMANCE mode.
     naive_compute() compiles any graph that does not open with a MUL_MAT in
     ACCURACY mode, which also pins the inference precision to f32, so a vision
     tower (which opens with its patch convolution) ran at f32 whatever
     GGML_OPENVINO_GPU_PRECISION asked for. GGML_OPENVINO_EXEC_MODE=performance
     compiles every graph in PERFORMANCE mode; a model that holds at f16 sets it.

 11. openvino/op/rms_norm.cpp - an RMSNorm that cannot overflow f16, for the
     NPU. The NPU computes in f16 only, and a 2048-wide sum of squares overflows
     once activations reach |x| ~ 6; GR00T's language models carry outliers of
     1e4 and more. Each row is scaled by its own maximum first. On by default
     when GGML_OPENVINO_DEVICE is the NPU; GGML_OPENVINO_RMS_SAFE=0/1 overrides.

 12. ggml-openvino-extra.cpp - GGML_OPENVINO_NPU_PROPS="KEY=VALUE,..." lands
     verbatim in the NPU compile config, after the defaults, e.g. NPU_TURBO=YES
     to keep the clock up between a policy server's requests.

 13. utils.cpp - no NPUW for naive graphs on the NPU. The NPU compile config
     turns NPUW on for llama.cpp's decoder layers, and its partitioning hands the
     compiler dynamic Reshape inputs that static VLA graphs never had. The
     static path already drops it for a graph without a KV cache; this does the
     same for naive graphs.

 14. op_table.{h,cpp}, ggml-openvino.cpp, ggml-quants.cpp, ggml-decoder.cpp -
     the hook for FoldQuant. A FoldQuant GGUF builds two GGML_OP_CUSTOM nodes
     per INT8/INT4 site, which ggml leaves to the CPU backend. The translator
     that maps them onto OpenVINO ops lives in-tree (src/openvino/
     foldquant_ov.cpp, compiled into this target by vla.cpp's CMakeLists); this
     hunk only registers it for GGML_OP_CUSTOM, lets the selected device's
     supports_op accept FoldQuant's nodes (and no other custom op) ahead of the
     type checks that would refuse their INT8 tensors, and keeps an INT8 weight
     as an i8 constant.

The GPU precision, NPU properties and exec mode are read with getenv rather
than the backend's env allowlist: the allowlist is cached at backend init, and
a hunk that adds to it anchors on a list upstream keeps editing.

Idempotent - re-running on a patched tree is a no-op, so a reconfigure that
re-populates the FetchContent source dir is safe either way.

Usage: scripts/patch_ggml_openvino.py [<llama-src-dir>]
"""

import pathlib
import re
import sys

NAIVE_COMPUTE_OLD = """enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
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
    ov::AnyMap compile_config = config;
    if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } else {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    if (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } else {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (const auto & param : model->get_parameters()) {
        input_names.push_back(param->get_friendly_name());
    }
    for (const auto & result : model->get_results()) {
        output_names.push_back(result->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    input_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_lock.unlock();
"""

NAIVE_COMPUTE_NEW = """// vla.cpp: identity of one concrete graph, cheap enough to take per call:
// naive_graph_sig plus the address and data pointer of every node and source.
// vla.cpp reuses one graph object per shape, so a match means the same tensors
// in the same buffers, and the decoder bound to them last time still is. That
// skips naive_key and update_io(), both costly on a graph of thousands of nodes.
inline uint64_t naive_graph_fp(const ggml_cgraph * cgraph) {
    uint64_t h = naive_graph_sig(cgraph);
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix((uint64_t) (uintptr_t) cgraph);
    mix((uint64_t) cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        mix((uint64_t) (uintptr_t) node);
        mix((uint64_t) (uintptr_t) node->data);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = node->src[s];
            if (src) {
                mix((uint64_t) (uintptr_t) src);
                mix((uint64_t) (uintptr_t) src->data);
            }
        }
    }
    return h;
}

enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config,
                               const std::shared_ptr<ov_runtime_context> & r_ctx) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    // vla.cpp: reuse the decoder and the compiled infer request across calls on
    // the same graph, the way the dynamic and static paths already do.
    // Conversion plus compile_model dominates a naive call, so without this every
    // graph_compute pays it again.
    static const bool cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    std::shared_ptr<naive_runtime_ctx> entry;
    // vla.cpp: try the fingerprint first; only a miss builds naive_key.
    const uint64_t graph_fp = cache_enabled ? naive_graph_fp(cgraph) : 0;
    if (cache_enabled) {
        std::lock_guard<std::mutex> lock(r_ctx->ctx_mutex);
        auto fit = r_ctx->naive_fp_cache.find(graph_fp);
        if (fit != r_ctx->naive_fp_cache.end()) {
            entry = fit->second;
        }
    }
    if (cache_enabled && entry == nullptr) {
        const naive_key key(cgraph);
        std::lock_guard<std::mutex> lock(r_ctx->ctx_mutex);
        auto it = r_ctx->naive_cache.find(key);
        if (it != r_ctx->naive_cache.end()) {
            entry = it->second;
            r_ctx->naive_fp_cache[graph_fp] = entry;
        } else {
            // Each entry holds a compiled model, so this cannot grow forever.
            // Flush rather than evict: a caller sees a handful of shapes, and an
            // LRU is only worth its bookkeeping once that stops being true.
            if (r_ctx->naive_cache.size() >= 32) {
                r_ctx->naive_cache.clear();
                r_ctx->naive_fp_cache.clear();
                r_ctx->naive_io_fp.clear();
            }
            entry = std::make_shared<naive_runtime_ctx>();
            r_ctx->naive_cache[key] = entry;
            r_ctx->naive_fp_cache[graph_fp] = entry;
        }
    }

    std::shared_ptr<GgmlOvDecoder> decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    if (entry != nullptr && entry->infer_request != nullptr) {
        decoder = entry->decoder;
        infer_request = entry->infer_request;
        input_names = entry->input_names;
        output_names = entry->output_names;
        // Same shapes, new tensors: point the decoder at this call's graph,
        // unless it is the graph the decoder is already bound to.
        bool bound;
        {
            std::lock_guard<std::mutex> lock(r_ctx->ctx_mutex);
            bound = r_ctx->naive_io_fp[entry.get()] == graph_fp;
        }
        if (!bound) {
            decoder->update_io(cgraph);
            std::lock_guard<std::mutex> lock(r_ctx->ctx_mutex);
            r_ctx->naive_io_fp[entry.get()] = graph_fp;
        }
    } else {
        std::unique_lock<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        bool naive = true;
        auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
        decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
        auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
        auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
        if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
            ov::serialize(model, "IR_naive.xml");
        }

        auto remote_context = ggml_openvino_get_remote_context();
        ov::AnyMap compile_config = config;
        if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
            // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
            compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
        } else {
            compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
        }
        if (remote_context.has_value()) {
            infer_request = std::make_shared<ov::InferRequest>(
                core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
        } else {
            infer_request = std::make_shared<ov::InferRequest>(
                core.compile_model(model, device, compile_config).create_infer_request());
        }
        for (const auto & param : model->get_parameters()) {
            input_names.push_back(param->get_friendly_name());
        }
        for (const auto & result : model->get_results()) {
            output_names.push_back(result->get_friendly_name());
        }
        // Destroy the frontend graph under the compilation lock as well: it can
        // still own edges into the shared weight nodes.
        model.reset();
        input_model.reset();
        decoder->clear_model_weights();
        model_weights.clear();
        compile_lock.unlock();

        if (entry != nullptr) {
            entry->decoder = decoder;
            entry->infer_request = infer_request;
            entry->input_names = input_names;
            entry->output_names = output_names;
            std::lock_guard<std::mutex> lock(r_ctx->ctx_mutex);
            r_ctx->naive_io_fp[entry.get()] = graph_fp;
        }
    }
"""

# file -> [(anchor, replacement), ...]. Every anchor must match exactly once.
EDITS = {
    "ggml/src/ggml-openvino/ggml-openvino-extra.cpp": [
        (
            """    } else if (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }""",
            """    } else if (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }

    // vla.cpp: the GPU plugin computes in f16 unless told otherwise, which is why
    // it is fast. A model whose graph carries a long serial chain -- a denoise
    // loop unrolled inside one graph -- compounds that across every step, and a
    // saturating output channel can then cross its threshold in the wrong place.
    // Exposed rather than forced: f32 costs roughly 3x on this plugin. getenv,
    // not the env allowlist, so this hunk does not anchor on upstream's list.
    if (has_prefix(device_name, "GPU")) {
        const char * gpu_prec = getenv("GGML_OPENVINO_GPU_PRECISION");
        if (gpu_prec != nullptr && strcmp(gpu_prec, "f32") == 0) {
            compile_config.insert(ov::hint::inference_precision(ov::element::f32));
        } else if (gpu_prec != nullptr && *gpu_prec && strcmp(gpu_prec, "f16") != 0) {
            GGML_LOG_WARN("GGML OpenVINO Backend: GGML_OPENVINO_GPU_PRECISION=%s is not f16 or f32, ignoring\\n",
                          gpu_prec);
        }
    }""",
        ),
        (
            """        if (compilation_mode_params && strlen(compilation_mode_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_mode_params;
        }
""",
            """        if (compilation_mode_params && strlen(compilation_mode_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_mode_params;
        }
        // vla.cpp: GGML_OPENVINO_NPU_PROPS="KEY=VALUE,KEY=VALUE" lands verbatim in
        // the compile config. NPU_TURBO=YES keeps the clock up between requests:
        // a policy server's 50-500 ms gaps otherwise cost 50-70 ms per request.
        // Later keys override the defaults above.
        if (const char * props = getenv("GGML_OPENVINO_NPU_PROPS"); props && *props) {
            std::string list(props);
            size_t start = 0;
            while (start < list.size()) {
                size_t end = list.find(',', start);
                if (end == std::string::npos) {
                    end = list.size();
                }
                const std::string kv = list.substr(start, end - start);
                const size_t eq = kv.find('=');
                if (eq != std::string::npos && eq > 0) {
                    compile_config[kv.substr(0, eq)] = kv.substr(eq + 1);
                }
                start = end + 1;
            }
        }
""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/add.cpp": [
        (
            """    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(input_0, input_1);""",
            """    // vla.cpp: re-hang the outer add on the inner one's non-GEMM operand so the
    // GEMM is left with a single post-op. Addition is associative.
    ov::Output<ov::Node> res;
    const int oc = context.get_op_case();
    auto inner = input_0.get_node_shared_ptr();
    if ((oc == 2 || oc == 3) && inner->get_input_size() == 2) {
        const size_t keep = (oc == 2) ? 0 : 1;
        const size_t fold = 1 - keep;
        auto folded = std::make_shared<ov::op::v1::Add>(inner->input_value(fold), input_1);
        res = std::make_shared<ov::op::v1::Add>(inner->input_value(keep), folded);
    } else {
        res = std::make_shared<ov::op::v1::Add>(input_0, input_1);
    }""",
        ),
    ],
    # vla.cpp: an RMSNorm that cannot overflow f16, for the NPU.
    "ggml/src/ggml-openvino/openvino/op/rms_norm.cpp": [
        (
            "#include <memory>\n",
            "#include <cmath>\n#include <cstdlib>\n#include <cstring>\n#include <memory>\n#include <string>\n",
            "#include <cmath>\n#include <cstdlib>\n",
        ),
        (
            "#include <openvino/op/reduce_mean.hpp>\n",
            "#include <openvino/op/abs.hpp>\n#include <openvino/op/maximum.hpp>\n#include <openvino/op/reduce_max.hpp>\n#include <openvino/op/reduce_mean.hpp>\n",
            "#include <openvino/op/reduce_max.hpp>\n",
        ),
        (
            """    auto square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);
""",
            """    // vla.cpp: AN RMSNORM THAT CANNOT OVERFLOW F16, FOR THE NPU.
    //
    // The NPU computes in f16 only. A 2048-wide sum of squares overflows 65504
    // once activations reach |x| ~ 6, and GR00T's language models carry
    // residual outliers of 1e4 and more, so the plain norm below returned NaN.
    // Scale each row by its own maximum, a = max(|x|) floored at 1e-2:
    //     x / sqrt(mean(x^2) + eps) ~= x / max(a * sqrt(mean((x/a)^2)), sqrt(eps))
    // Every (x/a)^2 is <= 1, so nothing in the chain can overflow. Two NPU
    // details shape the rest:
    // - the epsilon is applied as max(., sqrt(eps)), not as eps/a^2 inside the
    //   sqrt: eps/a^2 is ~1e-8, deep in f16's subnormals, and the NPU compiler
    //   miscompiled that branch in some compiles (GR00T N1.6 came out all-NaN
    //   from one compile and finite from another). The two forms differ only on
    //   rows whose rms is near sqrt(eps), by at most sqrt(2);
    // - both divisions are pre-scaled by 1/16: the NPU divides through an f16
    //   reciprocal, and 1/d is flushed to 0 once d > 16384, which zeroed x/a
    //   for GR00T N1.5's 1.8e4 outlier rows (LIBERO 0/100). 1/16 is exact.
    // On by default when GGML_OPENVINO_DEVICE is the NPU; GGML_OPENVINO_RMS_SAFE
    // =0/1 overrides for any device.
    {
        const char * rs_env = std::getenv("GGML_OPENVINO_RMS_SAFE");
        const char * rs_dev = std::getenv("GGML_OPENVINO_DEVICE");
        const bool   rms_safe = (rs_env && *rs_env) ? std::atoi(rs_env) != 0 :
                                                      (rs_dev && std::string(rs_dev).rfind("NPU", 0) == 0);
        if (rms_safe) {
            float eps_s;
            memcpy(&eps_s, context.get_output_op_params(), sizeof(float));
            const auto et   = input_node.get_element_type();
            auto       axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
            auto       s16  = ov::op::v0::Constant::create(et, ov::Shape{1}, {1.0f / 16.0f});
            auto       amax = std::make_shared<ov::op::v1::Maximum>(
                std::make_shared<ov::op::v1::ReduceMax>(std::make_shared<ov::op::v0::Abs>(input_node), axis, true),
                ov::op::v0::Constant::create(et, ov::Shape{1}, {1e-2f}));
            auto x16   = std::make_shared<ov::op::v1::Multiply>(input_node, s16);
            auto xs    = std::make_shared<ov::op::v1::Divide>(x16, std::make_shared<ov::op::v1::Multiply>(amax, s16));
            auto ms    = std::make_shared<ov::op::v1::ReduceMean>(std::make_shared<ov::op::v1::Multiply>(xs, xs), axis, true);
            auto denom = std::make_shared<ov::op::v1::Maximum>(
                std::make_shared<ov::op::v1::Multiply>(amax, std::make_shared<ov::op::v0::Sqrt>(ms)),
                ov::op::v0::Constant::create(et, ov::Shape{1}, {std::sqrt(eps_s)}));
            auto res_s = std::make_shared<ov::op::v1::Divide>(x16, std::make_shared<ov::op::v1::Multiply>(denom, s16));
            return rename_outputs_with_suffix({res_s}, context.get_name());
        }
    }

    auto square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);
""",
            "// vla.cpp: AN RMSNORM THAT CANNOT OVERFLOW F16, FOR THE NPU.",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/rope.cpp": [
        (
            """        if (context.get_input_size() == 3) {
            cache_key += "_ff_" + context.get_input_names()[2];
        }
""",
            """        if (context.get_input_size() == 3) {
            cache_key += "_ff_" + context.get_input_names()[2];
        }
        // vla.cpp: two ROPEs with the same op_params but different position
        // inputs (a prefix and an action suffix) must not share one table.
        cache_key += "_pos_" + context.get_input_names()[1];
""",
        ),
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
            """        {"GGML_OP_ADD",             op::translate_add                              },""",
            """        {"GGML_OP_ADD",             op::translate_add                              },
        // vla.cpp: FoldQuant's two custom nodes (src/openvino/foldquant_ov.cpp).
        {"GGML_OP_CUSTOM",          op::translate_vla_foldquant                    },""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op_table.h": [
        (
            """GGML_OP_CONVERTER(translate_add);""",
            """GGML_OP_CONVERTER(translate_add);
GGML_OP_CONVERTER(translate_vla_foldquant);  // vla.cpp: src/openvino/foldquant_ov.cpp""",
        ),
    ],
    "ggml/src/ggml-openvino/ggml-openvino.cpp": [
        (
            """static ggml_openvino_op_support ggml_backend_openvino_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);
""",
            """// vla.cpp: defined in src/openvino/foldquant_ov.cpp.
bool vla_foldquant_ov_supports(const ggml_tensor * op);

static ggml_openvino_op_support ggml_backend_openvino_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);
""",
        ),
        (
            """        return {false, "device is not the selected OpenVINO device"};
    }
""",
            """        return {false, "device is not the selected OpenVINO device"};
    }

    // vla.cpp: FoldQuant's custom nodes carry INT8 weights and an INT8 blob,
    // which the type checks below would refuse; they have their own test.
    // After the guard above, so only the selected device claims them.
    if (op->op == GGML_OP_CUSTOM) {
        if (vla_foldquant_ov_supports(op)) {
            return {true, ""};
        }
        return {false, "GGML_OP_CUSTOM other than FoldQuant's"};
    }
""",
        ),
    ],
    "ggml/src/ggml-openvino/ggml-quants.cpp": [
        (
            """    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        switch (tensor->type) {""",
            """    // vla.cpp: GGML_TYPE_I8 is a FoldQuant site's codes, kept as an i8 constant.
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16 ||
        tensor->type == GGML_TYPE_I8) {
        ov::element::Type element_type;
        switch (tensor->type) {
        case GGML_TYPE_I8:
            element_type = ov::element::i8;
            break;""",
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
            """    std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) const {""",
            """    // vla.cpp: whether the graph has more than one ROPE position input. The
    // free function get_tensor_graph_input_ov_name() in ggml-decoder.cpp asks.
    bool has_multiple_inp_pos() const;

    std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) const {""",
        ),
        (
            "    ggml_cgraph * m_cgraph = nullptr;",
            "    ggml_cgraph * m_cgraph = nullptr;\n"
            "    mutable int m_multi_inp_pos = -1;  // vla.cpp: lazily computed, -1 = unknown",
        ),
    ],
    "ggml/src/ggml-openvino/ggml-decoder.cpp": [
        (
            """    case GGML_OP_ADD: {
        if (is_moe_expert_sum_add(node)) {""",
            """    case GGML_OP_ADD: {
        // vla.cpp: ADD(ADD(.., GEMM), graph-input). Two elementwise ops chained on a
        // GEMM; the GPU plugin folds both in as post-ops and loses the second operand.
        // op_case 2 = the inner add's input 0 is the GEMM, 3 = its input 1 is.
        {
            const ggml_tensor * inner = node->src[0];
            const ggml_tensor * other = node->src[1];
            if (inner && other && inner->op == GGML_OP_ADD && other->op == GGML_OP_NONE) {
                for (int k = 0; k < 2; k++) {
                    const ggml_tensor * s = inner->src[k];
                    while (s && s->src[0] &&
                           (s->op == GGML_OP_VIEW || s->op == GGML_OP_RESHAPE || s->op == GGML_OP_CONT)) {
                        s = s->src[0];
                    }
                    if (s && s->op == GGML_OP_MUL_MAT) {
                        op_case = (k == 0) ? 2 : 3;
                    }
                }
            }
        }
        if (is_moe_expert_sum_add(node)) {""",
        ),
        (
            """    if (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        return "inp_pos";
    }""",
            """    if (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        // vla.cpp: this free function is the live naming path -- the
        // GgmlOvDecoder member of the same intent has no callers.
        // Only collapse ROPE position inputs onto one "inp_pos" parameter when
        // the graph really has one. See scripts/patch_ggml_openvino.py.
        return decoder->has_multiple_inp_pos() ? get_tensor_ov_name(cgraph, tensor) : std::string("inp_pos");
    }""",
        ),
        (
            """        } else {
            // rope'ed query tensor
            op_case = 2;""",
            """        } else {
            // vla.cpp: op_case 2 rewrites the tensor as [n_seq, -1, n_heads,
            // head_size] before transposing, which is only correct for
            // llama.cpp's rope'd query. It was reached by ANY permute whose
            // source is a view of a non-leaf, so a DiT head split -- GR00T
            // N1.7's cross-attention V, ggml_permute(view, 1,2,0,3) -- was
            // reshaped into a shape that has nothing to do with it and came out
            // with its elements rearranged. Require the rope.
            const ggml_tensor * prod = node->src[0];
            while (prod && prod->src[0] &&
                   (prod->op == GGML_OP_VIEW || prod->op == GGML_OP_RESHAPE || prod->op == GGML_OP_CONT)) {
                prod = prod->src[0];
            }
            op_case = (prod && prod->op == GGML_OP_ROPE) ? 2 : 1;""",
        ),
        (
            "int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {",
            """bool GgmlOvDecoder::has_multiple_inp_pos() const {
    if (m_cgraph == nullptr) {
        return false;
    }
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
        (
            """    static const std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,""",
            """    // vla.cpp: GGML_TYPE_I8 is a FoldQuant site's codes (see ggml-quants.cpp).
    static const std::set<ggml_type> weight_types = {GGML_TYPE_I8,   GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,""",
        ),
    ],
    "ggml/src/ggml-openvino/utils.h": [
        (
            "struct decoder_runtime_ctx {",
            """// vla.cpp: what naive_compute() reuses across calls on the same graph. Without
// it that path rebuilt the decoder, re-converted the model and called
// compile_model() on every ggml_backend_graph_compute, which dominated runtime.
struct naive_runtime_ctx {
    std::shared_ptr<GgmlOvDecoder> decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

// vla.cpp: graph_key is {n_nodes, first name, last name, input names}, which two
// graphs of the same size can share. A compiled model is bound to the shapes it
// was built for, so reusing one across a shape change returns another graph's
// answer with no error. Mix the ops and shapes in as well.
inline uint64_t naive_graph_sig(const ggml_cgraph * cgraph) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        mix((uint64_t) node->op);
        mix((uint64_t) node->type);
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            mix((uint64_t) node->ne[d]);
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = node->src[s];
            mix(src ? (uint64_t) src->type + 1 : 0ull);
            for (int d = 0; src && d < GGML_MAX_DIMS; d++) {
                mix((uint64_t) src->ne[d]);
            }
        }
    }
    return h;
}

struct naive_key {
    graph_key base;
    uint64_t  sig;

    naive_key(const ggml_cgraph * cgraph) : base(cgraph), sig(naive_graph_sig(cgraph)) {}

    bool operator==(const naive_key & other) const { return sig == other.sig && base == other.base; }
};

struct naive_key_hash {
    size_t operator()(const naive_key & key) const {
        size_t h = graph_key_hash{}(key.base);
        return h ^ (std::hash<uint64_t>{}(key.sig) + 0x9e3779b9 + (h << 6) + (h >> 2));
    }
};

struct decoder_runtime_ctx {""",
        ),
        (
            "    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;",
            "    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;\n"
            "    std::unordered_map<naive_key, std::shared_ptr<naive_runtime_ctx>, naive_key_hash> naive_cache;\n"
            "    // vla.cpp: naive_graph_fp -> entry, and the fingerprint each entry's decoder is bound to.\n"
            "    std::unordered_map<uint64_t, std::shared_ptr<naive_runtime_ctx>> naive_fp_cache;\n"
            "    std::unordered_map<const naive_runtime_ctx *, uint64_t> naive_io_fp;",
        ),
        (
            """        decoder_cache.clear();
        infer_request_cache.clear();""",
            """        decoder_cache.clear();
        naive_cache.clear();
        infer_request_cache.clear();
        naive_fp_cache.clear();
        naive_io_fp.clear();""",
        ),
    ],
    "ggml/src/ggml-openvino/utils.cpp": [
        (
            """        if (!model_is_splitted) {
            return naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }""",
            """        if (!model_is_splitted) {
            return naive_compute(cgraph, core, device, config, r_ctx);
        }""",
        ),
        (
            """    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }""",
            """    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config, r_ctx);
    }""",
        ),
        (
            NAIVE_COMPUTE_OLD,
            NAIVE_COMPUTE_NEW,
            # The two hunks below insert into this replacement, so it cannot be
            # its own marker. This line survives them and is unique to it.
            "// vla.cpp: try the fingerprint first; only a miss builds naive_key.",
        ),
        (
            """bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;""",
            """bool is_naive(ggml_cgraph * cgraph) {
    // vla.cpp: the literal translation path suits any graph that is not a
    // decoder-only LLM, so let the caller raise the bar it is chosen under.
    // Junk parses to 0 under atoi, which would silently send every graph down
    // the LLM builder, so reject anything that is not a whole positive number.
    static const int naive_graph_size_threshold = [] {
        const char * env = getenv("GGML_OPENVINO_NAIVE_GRAPH_SIZE");
        if (env == nullptr || *env == '\\0') {
            return 20;
        }
        char *     end = nullptr;
        const long val = strtol(env, &end, 10);
        const int  n   = (int) val;
        if (*end != '\\0' || val <= 0 || (long) n != val) {
            GGML_LOG_WARN("GGML OpenVINO Backend: ignoring GGML_OPENVINO_NAIVE_GRAPH_SIZE='%s'\\n", env);
            return 20;
        }
        return n;
    }();""",
        ),
        # vla.cpp: two vla.cpp-only knobs inside naive_compute, kept out of the
        # replacement above so scripts/upstream_split.py can leave them behind.
        (
            """            compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
        }""",
            """            compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
        }
        // vla.cpp: ACCURACY also pins the precision to f32. A model that holds at
        // f16 opts every graph into PERFORMANCE. getenv, not the env allowlist:
        // that is cached at backend init, and the model sets this afterwards.
        const char * exec_mode = getenv("GGML_OPENVINO_EXEC_MODE");
        if (exec_mode != nullptr && strcmp(exec_mode, "performance") == 0) {
            compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
        }""",
        ),
        (
            """        if (exec_mode != nullptr && strcmp(exec_mode, "performance") == 0) {
            compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
        }
        if (remote_context.has_value()) {""",
            """        if (exec_mode != nullptr && strcmp(exec_mode, "performance") == 0) {
            compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
        }
        // vla.cpp: NO NPUW FOR NAIVE GRAPHS ON THE NPU. The NPU compile config
        // turns on NPUW (FOLD, FUNCALL_FOR_ALL) for llama.cpp's decoder layers,
        // and its partitioning hands the compiler subgraphs with dynamic Reshape
        // inputs: Evo-1's LM + DiT, fully static as exported, failed with
        // "IE.Reshape doesn't support dynamic shapes" in 4 s under that config
        // and compiled in 113 s without it (pi0.5, GR00T and VLA-JEPA the same).
        // A naive graph is never a KV-cached decoder, so NPUW does nothing for it.
        if (ggml_openvino_is_npu()) {
            for (auto it = compile_config.begin(); it != compile_config.end();) {
                if (it->first.rfind("NPUW", 0) == 0 || it->first == "NPU_USE_NPUW") {
                    it = compile_config.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (remote_context.has_value()) {""",
            "// vla.cpp: NO NPUW FOR NAIVE GRAPHS ON THE NPU.",
        ),
    ],
}


# A duplicate key in EDITS silently drops the earlier entry's hunks -- Python keeps
# the last one and nothing complains. That happened once and took the whole Intel
# OpenCL fix out of the patch, so compare the literal keys against the dict.
_keys = re.findall(r'^    "([^"]+)": \[$', pathlib.Path(__file__).read_text(), re.M)
if len(_keys) != len(EDITS):
    _dup = sorted({k for k in _keys if _keys.count(k) > 1})
    sys.exit(f"patch_ggml_openvino: EDITS has {len(_keys)} literal keys but {len(EDITS)} entries; "
             f"duplicate key(s): {_dup}")


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    pending = []
    for rel, edits in EDITS.items():
        src = root / rel
        if not src.is_file():
            print(f"patch_ggml_openvino: {src} not found", file=sys.stderr)
            return 1
        original = src.read_text()
        text = original
        for anchor, replacement, *marker in edits:
            # Per hunk, not per file. The hunk list grows between commits, so a
            # tree patched by an older checkout still needs the newer ones and a
            # file-wide marker would skip them, leaving a build that runs and is
            # quietly wrong. A hunk that later hunks edit into names a marker
            # of its own, since its replacement text no longer appears whole.
            if (marker[0] if marker else replacement) in text:
                continue
            n = text.count(anchor)
            if n != 1:
                print(f"patch_ggml_openvino: anchor matched {n} times in {rel}, expected 1:\n{anchor}\n"
                      f"A tree patched by an older checkout reads like this. Delete the fetched "
                      f"llama.cpp (rm -rf <build>/_deps) and reconfigure. If the tree is clean, the "
                      f"anchor no longer matches VLA_LLAMA_TAG and needs re-targeting.",
                      file=sys.stderr)
                return 1
            text = text.replace(anchor, replacement, 1)
        if text != original:
            pending.append((src, text, rel))

    # Every anchor matched, so nothing was half-written on the way here.
    for src, text, rel in pending:
        src.write_text(text)
        print(f"patch_ggml_openvino: patched {rel}")
    if not pending:
        print("patch_ggml_openvino: already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
