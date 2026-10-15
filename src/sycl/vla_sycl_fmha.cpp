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


// Flash attention for Intel GPUs from sycl-tla (CUTLASS for SYCL), behind the
// SYCL extension hook (scripts/patch_ggml_sycl_ext_hook.py).
//
// ggml-sycl lowers GGML_OP_FLASH_ATTN_EXT with head_dim 256 to an F32 oneMKL
// GEMM, a softmax and a second GEMM. sycl-tla's fused FMHA kernel does the same
// in one pass on the XMX units. It is claimed only for the shapes it was
// checked on (head_dim 256, no mask, no sinks, no ALiBi, no softcap, batch 1):
// everything else returns false and ggml-sycl runs it as before.
//
// Q, K and V arrive as F32 (src/backend.h fa_kv keeps them F32 off Hexagon);
// one kernel converts them to F16 in a scratch buffer, the FMHA kernel reads
// that, accumulates in F32 and writes F32 straight into dst.
//
// VLA_SYCL_FMHA=0 turns the claim off.

#include "sycl/vla_sycl_fmha.h"

#include "ggml.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace vla {
namespace {

using namespace cute;

using Elt = cute::half_t;

// q128 x 16 subgroups: the fastest of the four tile shapes measured on the
// Arc B390 for pi0.5's prefix (528x528) and action-expert (50x578) attention.
using ShapeQK  = Shape<_128, _64, _32>;
using ShapePV  = Shape<_128, _32, _64>;
using ShapeOut = Shape<_128, _256>;
using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
constexpr int kHeadDim = 256;
constexpr int kPipelineStages = 2;

constexpr int SGTileQ = get<0>(shape_div(ShapeQK{}, shape(SubgroupLayoutQK{})))();
using MMAOp = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, Elt>;
using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

using StrideQ = Stride<int, _1, int, int>;   // (seq, d, head, batch)
using StrideK = Stride<int, _1, int, int>;
using StrideV = Stride<_1, int, int, int>;   // (d, seq, head, batch)
using StrideO = Stride<int, _1, int, int>;
using StrideS = Stride<_1, int, int, int>;

template <class T, class S>
using DummyTensor = decltype(make_tensor(make_gmem_ptr(static_cast<T *>(nullptr)),
                                         make_layout(repeat<rank_v<S>>(1), S{})));

using TensorQ = DummyTensor<Elt, StrideQ>;
using TensorK = DummyTensor<Elt, StrideK>;
using TensorV = DummyTensor<Elt, StrideV>;
using TensorO = DummyTensor<float, StrideO>;
using TensorS = DummyTensor<float, StrideS>;

using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOp>, Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOp>, Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;
constexpr int VTiles = get<1>(ShapeOut{}) / get<1>(ShapePV{});

using Mainloop = cutlass::fmha::collective::FMHAFwdMainloop<
    cutlass::fmha::XeDefault<kPipelineStages>, /*Causal=*/false, /*BlockScale=*/false,
    /*F8kvF16mma=*/false, /*PerTensorScale=*/false, /*CachedKV=*/false, /*PagedKV=*/false,
    TiledMMAQK, TiledMMAPV, VTiles,
    TensorQ, TensorK, TensorV, TensorS, TensorS, TensorS,
    TensorK, TensorV,
    void, void, void, void, void>;
using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<Mainloop, ShapeOut, TensorO, void>;
using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<false>;
// Batch 1, grouped heads allowed (pi0.5 has 8 query heads per KV head).
using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<true, false>;
using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShape, Mainloop, Epilogue, Scheduler>;

bool env_off() {
    const char * e = std::getenv("VLA_SYCL_FMHA");
    return e && std::strcmp(e, "0") == 0;
}

// F16 scratch for the converted Q, K and V; grows, never shrinks. The queue is
// in-order, so one buffer serves every node in turn.
struct Scratch {
    std::mutex mu;
    sycl::context ctx;
    Elt * ptr = nullptr;
    size_t n = 0;
};

Elt * scratch(sycl::queue & q, size_t n) {
    static Scratch s;
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.ptr && s.ctx == q.get_context() && s.n >= n) return s.ptr;
    if (s.ptr) {
        q.wait();
        sycl::free(s.ptr, s.ctx);
    }
    s.ctx = q.get_context();
    s.ptr = sycl::malloc_device<Elt>(n, q);
    s.n = s.ptr ? n : 0;
    return s.ptr;
}

int eu_count(sycl::queue & q) {
    static const int n = (int) q.get_device().get_info<sycl::info::device::max_compute_units>();
    return n;
}

// [D, S, H] F32 view with byte strides -> packed [H][S][D] F16.
void to_f16(sycl::queue & q, const ggml_tensor * t, Elt * dst) {
    const int64_t D = t->ne[0], S = t->ne[1], H = t->ne[2];
    const size_t nb1 = t->nb[1], nb2 = t->nb[2];
    const char * src = static_cast<const char *>(t->data);
    q.parallel_for(sycl::range<3>(H, S, D), [=](sycl::id<3> i) {
        const int64_t h = i[0], s = i[1], d = i[2];
        const float v = *reinterpret_cast<const float *>(src + h * nb2 + s * nb1 + d * sizeof(float));
        dst[(h * S + s) * D + d] = Elt(v);
    });
}

}  // namespace

bool sycl_fmha_forward(ggml_tensor * dst, void * queue) {
    if (dst->op != GGML_OP_FLASH_ATTN_EXT) return false;
    static const bool off = env_off();
    if (off) return false;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (dst->src[3] || dst->src[4]) return false;                  // mask, sinks
    float scale, max_bias, softcap;
    std::memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) return false;
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F32 || V->type != GGML_TYPE_F32) return false;
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) return false;
    if (Q->ne[0] != kHeadDim || K->ne[0] != kHeadDim || V->ne[0] != kHeadDim) return false;
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) return false;
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(float) || V->nb[0] != sizeof(float)) return false;
    const int64_t Sq = Q->ne[1], Hq = Q->ne[2], Skv = K->ne[1], Hkv = K->ne[2];
    if (V->ne[1] != Skv || V->ne[2] != Hkv || Hkv < 1 || Hq % Hkv != 0) return false;

    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    const size_t nq = (size_t) Hq * Sq * kHeadDim, nkv = (size_t) Hkv * Skv * kHeadDim;
    Elt * buf = scratch(q, nq + 2 * nkv);
    if (!buf) return false;
    Elt * q16 = buf, * k16 = buf + nq, * v16 = buf + nq + nkv;
    to_f16(q, Q, q16);
    to_f16(q, K, k16);
    to_f16(q, V, v16);

    ProblemShape shape{};
    shape.batch = 1;
    shape.num_heads_q = (int) Hq;
    shape.num_heads_kv = (int) Hkv;
    shape.seq_len_qo = (int) Sq;
    shape.seq_len_kv = (int) Skv;
    shape.seq_len_kv_cache = 0;
    shape.head_size_qk = kHeadDim;
    shape.head_size_vo = kHeadDim;

    typename Kernel::KernelArguments ka{};
    ka.shape = shape;
    ka.Q = q16; ka.dQ = StrideQ{kHeadDim, _1{}, (int) (Sq * kHeadDim), (int) nq};
    ka.K = k16; ka.dK = StrideK{kHeadDim, _1{}, (int) (Skv * kHeadDim), (int) nkv};
    ka.V = v16; ka.dV = StrideV{_1{}, kHeadDim, (int) (Skv * kHeadDim), (int) nkv};
    // dst is [D, Hq, Sq]: a row of D floats per (token, head).
    ka.O = static_cast<float *>(dst->data);
    ka.dO = StrideO{(int) (Hq * kHeadDim), _1{}, kHeadDim, (int) (Sq * Hq * kHeadDim)};
    ka.K_cache = nullptr;
    ka.V_cache = nullptr;

    cutlass::KernelHardwareInfo hw{};
    hw.sm_count = eu_count(q);
    typename Kernel::Arguments args{ka, {scale, nullptr, 0, nullptr}, {}, hw};
    if (!Kernel::can_implement(args)) return false;
    const auto params = Kernel::to_underlying_arguments(args, nullptr);

    namespace syclex  = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    const dim3 block = Kernel::get_block_shape();
    const dim3 grid  = Kernel::get_grid_shape(params);
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(Kernel::SharedStorageSize),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
        intelex::grf_size<256>,
    };
    compat::experimental::launch_policy policy{compat::dim3(grid.x, grid.y, grid.z),
                                               compat::dim3(block.x, block.y, block.z),
                                               launch_props, kernel_props};
    compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel, false>(policy, q, params);
    return true;
}

}  // namespace vla
