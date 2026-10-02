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

// FoldQuant's two GGML_OP_CUSTOM nodes (src/layers/fq_linear.h) translated into
// OpenVINO ops, so a FoldQuant GGUF runs natively on ggml's OpenVINO backend.
// Compiled into the ggml-openvino target by vla.cpp's CMakeLists; the two hook
// points (the op table entry and supports_op) are added by
// scripts/patch_ggml_openvino.py.
//
// fq_act  (x[, gamma][, ascale]) -> [.., T, K] F32: the integer-valued codes times
//         the per-token scale. ggml declares an I8 blob for this node; only fq_gemm
//         reads it, so the translator is free to carry it as floats.
// fq_gemm (w, act[, bias][, residual]) -> [.., T, N] F32.
//
// The arithmetic is the CPU reference's (src/foldquant_ref.cpp): RMSNorm with the
// folded gamma, the ascale divide before or after the block-normalised
// Sylvester-Hadamard rotation, scale = max(clip * amax / qmax, 1e-12), codes =
// clamp(round_half_even(y * (1 / scale)), -qmax, qmax). The weights stay INT8 or
// INT4 constants (W4 reinterprets the nibble bytes in place: OpenVINO's i4 packs
// the even element in the low nibble, as FoldQuant does) and are dequantized by
// wscale in the decompression pattern the plugins keep compressed.

#include "openvino/node_context.h"
#include "openvino/op_table.h"
#include "openvino/utils.h"

#include "foldquant.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <openvino/op/abs.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/round.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sqrt.hpp>
#include <vector>

namespace {

const void * fq_userdata(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CUSTOM)
        return nullptr;
    const auto * p = reinterpret_cast<const ggml_custom_op_params *>(op->op_params);
    return p->userdata;
}

uint32_t fq_magic(const void * userdata) {
    uint32_t m = 0;
    if (userdata)
        std::memcpy(&m, userdata, sizeof(m));
    return m;
}

}  // namespace

// supports_op hook (declared again in the patched ggml-openvino.cpp): FoldQuant's
// nodes, and no other GGML_OP_CUSTOM.
bool vla_foldquant_ov_supports(const ggml_tensor * op);
bool vla_foldquant_ov_supports(const ggml_tensor * op) {
    const uint32_t m = fq_magic(fq_userdata(op));
    if (m == vla::FQ_ACT_MAGIC) {
        const auto * s = static_cast<const vla::FqActSpec *>(fq_userdata(op));
        return op->src[0] && op->src[0]->type == GGML_TYPE_F32 && s->rot_block <= 64;
    }
    if (m == vla::FQ_GEMM_MAGIC) {
        const auto * s = static_cast<const vla::FqGemmSpec *>(fq_userdata(op));
        // The head-laid-out epilogue writes a layout the graph reads through views
        // of raw memory; the vla.cpp side turns it off on this backend.
        return s->heads == 0 && op->src[0] && op->src[0]->type == GGML_TYPE_I8;
    }
    return false;
}

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

using ov::op::v0::Constant;

Output<Node> f32_scalar(float v) {
    return Constant::create(element::f32, Shape{1}, {v});
}

Output<Node> i64_vec(std::vector<int64_t> v) {
    return Constant::create(element::i64, Shape{v.size()}, v);
}

// Normalised natural-order Sylvester-Hadamard matrix of size n (symmetric).
Output<Node> hadamard(int n) {
    std::vector<float> h((size_t) n * n);
    const float norm = 1.0f / std::sqrt((float) n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            h[(size_t) i * n + j] = (__builtin_popcount((unsigned) (i & j)) & 1) ? -norm : norm;
    return Constant::create(element::f32, Shape{(size_t) n, (size_t) n}, h);
}

// A [K] / [1, K] weight vector shaped to broadcast along the last axis.
Output<Node> row_vector(const Output<Node> & v, int64_t K) {
    Output<Node> r = v;
    if (r.get_element_type() != element::f32)
        r = std::make_shared<ov::op::v0::Convert>(r, element::f32);
    return std::make_shared<ov::op::v1::Reshape>(r, i64_vec({K}), false);
}

OutputVector translate_act(const NodeContext & context, const vla::FqActSpec & s) {
    const int64_t K = s.K;
    Output<Node> x = process_view_input_new(context, 0);
    if (x.get_element_type() != element::f32)
        x = std::make_shared<ov::op::v0::Convert>(x, element::f32);
    size_t next = 1;
    Output<Node> y = x;
    if (s.has_gamma) {
        const Output<Node> gamma = row_vector(context.get_input((int) next++), K);
        auto ms   = std::make_shared<ov::op::v1::ReduceMean>(std::make_shared<ov::op::v1::Multiply>(x, x),
                                                             i64_vec({-1}), true);
        auto rstd = std::make_shared<ov::op::v1::Divide>(
            f32_scalar(1.0f), std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(ms, f32_scalar(s.eps))));
        y = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v1::Multiply>(x, rstd), gamma);
    }
    Output<Node> a;
    if (s.has_ascale)
        a = row_vector(context.get_input((int) next++), K);
    if (s.has_ascale && s.fold_before)
        y = std::make_shared<ov::op::v1::Divide>(y, a);
    if (s.rot_block > 1) {
        // Rotate each rot_block chunk of the last axis: [.., K] -> [.., K/rb, rb] . H -> [.., K].
        auto shape = std::make_shared<ov::op::v3::ShapeOf>(y, element::i64);
        auto blk   = std::make_shared<ov::op::v1::Reshape>(y, i64_vec({-1, K / s.rot_block, s.rot_block}), false);
        auto rot   = std::make_shared<ov::op::v0::MatMul>(blk, hadamard(s.rot_block), false, false);
        y = std::make_shared<ov::op::v1::Reshape>(rot, shape, false);
    }
    if (s.has_ascale && !s.fold_before)
        y = std::make_shared<ov::op::v1::Divide>(y, a);

    const float qmax  = s.abits == 4 ? 7.0f : 127.0f;
    auto amax  = std::make_shared<ov::op::v1::ReduceMax>(std::make_shared<ov::op::v0::Abs>(y), i64_vec({-1}), true);
    auto scale = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Multiply>(amax, f32_scalar(s.clip)),
                                             f32_scalar(qmax)),
        f32_scalar(1e-12f));
    auto inv   = std::make_shared<ov::op::v1::Divide>(f32_scalar(1.0f), scale);
    auto q     = std::make_shared<ov::op::v0::Clamp>(
        std::make_shared<ov::op::v5::Round>(std::make_shared<ov::op::v1::Multiply>(y, inv),
                                            ov::op::v5::Round::RoundMode::HALF_TO_EVEN),
        -qmax, qmax);
    // codes * scale: the per-token scale factors out of the GEMM, so carrying it
    // inside the activation saves the codes/scale split at every consumer.
    auto out = std::make_shared<ov::op::v1::Multiply>(q, scale);
    return rename_outputs_with_suffix({out}, context.get_name());
}

OutputVector translate_gemm(const NodeContext & context, const vla::FqGemmSpec & s) {
    const int64_t K = s.K, N = s.N;

    // INT codes [N, K_pack] as the backend's weight constant; W4 reinterprets the
    // same bytes as an i4 [N, K] constant (the weight buffer outlives the model).
    auto wnode = std::dynamic_pointer_cast<Constant>(context.get_input(0).get_node_shared_ptr());
    OPENVINO_ASSERT(wnode, "FoldQuant: weight of ", context.get_name(), " is not a constant");
    std::shared_ptr<Node> w = wnode;
    if (s.wbits == 4) {
        ov::Tensor t(element::i4, Shape{(size_t) N, (size_t) K}, const_cast<void *>(wnode->get_data_ptr()));
        w = std::make_shared<Constant>(t);
    } else if (wnode->get_shape() != Shape{(size_t) N, (size_t) K}) {
        w = std::make_shared<ov::op::v1::Reshape>(wnode, i64_vec({N, K}), false);
    }
    // Decompression: Convert(INT) * wscale per output row.
    auto wscale = std::make_shared<ov::op::v1::Reshape>(row_vector(context.get_input(2), N), i64_vec({N, 1}), false);
    auto wf     = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v0::Convert>(w, element::f32), wscale);

    // The activation arrives as codes * per-token scale (see translate_act).
    Output<Node> y = std::make_shared<ov::op::v0::MatMul>(context.get_input(1), wf, false, true);

    // Optional sources after (w, act, wscale): the bias [N] and/or the residual
    // shaped like the output; ggml drops null sources, so tell them by size.
    for (size_t i = 3; i < context.get_input_size(); ++i) {
        const Output<Node> extra = context.get_input((int) i);
        const PartialShape ps = context.get_input_shape(i);
        const bool is_bias = ps.is_static() && ov::shape_size(ps.to_shape()) == (size_t) N;
        Output<Node> e = is_bias ? row_vector(extra, N) : extra;
        if (e.get_element_type() != element::f32)
            e = std::make_shared<ov::op::v0::Convert>(e, element::f32);
        y = std::make_shared<ov::op::v1::Add>(y, e);
    }
    return rename_outputs_with_suffix({y}, context.get_name());
}

}  // namespace

OutputVector translate_vla_foldquant(const NodeContext & context) {
    const int32_t * params = context.get_output_op_params();
    const void * ud = reinterpret_cast<const ggml_custom_op_params *>(params)->userdata;
    switch (fq_magic(ud)) {
    case vla::FQ_ACT_MAGIC:
        return translate_act(context, *static_cast<const vla::FqActSpec *>(ud));
    case vla::FQ_GEMM_MAGIC:
        return translate_gemm(context, *static_cast<const vla::FqGemmSpec *>(ud));
    default:
        OPENVINO_THROW("GGML_OP_CUSTOM ", context.get_name(), " is not a FoldQuant node");
    }
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
