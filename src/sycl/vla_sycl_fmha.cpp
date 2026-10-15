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
// everything else returns false and ggml-sycl runs it as before. The same
// kernel, in its cached-KV form, runs the fmha_kv node (src/layers/fmha_kv.h):
// pi0.5's action expert over the prefix KV plus its own keys, not concatenated.
//
// Q (and K/V when they are not F16 already) are converted to F16 in a scratch
// buffer; the kernel accumulates in F32 and writes F32 straight into dst.
// sycl-tla needs scripts/patch_sycl_tla_fmha.py for key counts that are not a
// whole 64-key tile (CMake applies it); tests/fmha_sycl_check covers both nodes.
//
// VLA_SYCL_FMHA=0 turns both claims off.

#include "sycl/vla_sycl_fmha.h"
#include "layers/fmha_kv.h"

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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace vla {
namespace {

using namespace cute;

using Elt = cute::half_t;

// Tile shapes. q128 x 16 sub-groups: the fastest of four measured on the Arc
// B390 for pi0.5's prefix (561 queries x 8 heads). q64 x 8: the action expert
// with its 8 query heads folded into the rows (one KV head), 400 rows.
constexpr int kHeadDim = 256;
constexpr int kPipelineStages = 2;

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
using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<false>;
// Batch 1, grouped heads allowed (pi0.5 has 8 query heads per KV head).
using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<true, false>;

// QM query rows x KT keys per tile; SGN sub-groups along the rows, SGK along
// the keys (SGK > 1: the decode shape, partial softmaxes reduced in the epilogue).
template <int QM, int SGN, bool CachedKV, int HD = kHeadDim, class Sched = Scheduler, int KT = 64, int SGK = 1,
          int Stages = kPipelineStages, bool VarLen = false, bool Lse = false>
struct Fmha {
    using ShapeQK  = Shape<Int<QM>, Int<KT>, _32>;
    using ShapePV  = Shape<Int<QM>, _32, Int<KT>>;
    using ShapeOut = Shape<Int<QM>, Int<HD>>;
    using SubgroupLayoutQK = Layout<Shape<Int<SGN>, Int<SGK>, _1>>;
    static constexpr int SGTileQ = get<0>(shape_div(ShapeQK{}, shape(SubgroupLayoutQK{})))();
    using MMAOp = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, Elt>;
    using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOp>, Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOp>, Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;
    static constexpr int VTiles = get<1>(ShapeOut{}) / get<1>(ShapePV{});
    using Mainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        cutlass::fmha::XeDefault<Stages>, /*Causal=*/false, /*BlockScale=*/false,
        /*F8kvF16mma=*/false, /*PerTensorScale=*/false, CachedKV, /*PagedKV=*/false,
        TiledMMAQK, TiledMMAPV, VTiles,
        TensorQ, TensorK, TensorV, TensorS, TensorS, TensorS,
        TensorK, TensorV,
        void, void, void, void, void>;
    using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<Mainloop, ShapeOut, TensorO, void, Lse>;
    using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemShape<VarLen>, Mainloop,
                                                          Epilogue, Sched>;
};

// SigLIP's head_dim 72, padded with zeros to the kernel's 96 (zero columns add
// nothing to Q.K, and V's become output columns that are dropped); views on the
// batch axis, one KV head per query head.
constexpr int kPadDim = 96;
using FmhaVit = Fmha<128, 8, false, kPadDim, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, true>>;

bool env_off() {
    static const bool off = [] {
        const char * e = std::getenv("VLA_SYCL_FMHA");
        return e && std::strcmp(e, "0") == 0;
    }();
    return off;
}

// Device scratch for the F16 inputs; grows, never shrinks. The queue is
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

// [D, S, H] tensor (F32 or F16, rows contiguous, any row/head strides) -> packed
// F16, [H][S][D], or [S][H][D] with fold (query heads folded into the rows).
void to_f16(sycl::queue & q, const ggml_tensor * t, Elt * dst, bool fold = false) {
    const int64_t S = t->ne[1], H = t->ne[2];
    const size_t nb1 = t->nb[1], nb2 = t->nb[2];
    const bool f16 = t->type == GGML_TYPE_F16;
    const char * src = static_cast<const char *>(t->data);
    q.parallel_for(sycl::range<3>(H, S, kHeadDim), [=](sycl::id<3> i) {
        const int64_t h = i[0], s = i[1], d = i[2];
        const char * e = src + h * nb2 + s * nb1;
        const float v = f16 ? (float) reinterpret_cast<const sycl::half *>(e)[d] : reinterpret_cast<const float *>(e)[d];
        dst[(fold ? s * H + h : h * S + s) * kHeadDim + d] = Elt(v);
    });
}

bool rows_ok(const ggml_tensor * t, bool f16_ok) {
    return t->ne[0] == kHeadDim && t->ne[3] == 1 &&
           (t->type == GGML_TYPE_F32 || (f16_ok && t->type == GGML_TYPE_F16)) &&
           t->nb[0] == ggml_type_size(t->type);
}

bool packed_f16(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F16 && ggml_is_contiguous(t);
}

// Packed F16 view of t: t itself when it is one already, else converted into *spare.
const Elt * f16_of(sycl::queue & q, const ggml_tensor * t, Elt *& spare) {
    if (packed_f16(t)) return static_cast<const Elt *>(t->data);
    Elt * d = spare;
    to_f16(q, t, d);
    spare += (size_t) t->ne[1] * t->ne[2] * kHeadDim;
    return d;
}

// One kernel launch. K/V: S_kv keys of each KV head, packed F16 [Hkv][S_kv][D];
// with CachedKV, Kc/Vc hold S_c keys that come first. O: [D, Hq, Sq] F32.
template <class Cfg>
void launch(sycl::queue & q, const Elt * q16, int64_t Sq, int64_t Hq, int64_t Hkv, const Elt * k16, const Elt * v16,
            int64_t Skv, const Elt * kc16, const Elt * vc16, int64_t Sc, float scale, float * O) {
    using K = typename Cfg::Kernel;
    ProblemShape shape{};
    shape.batch = 1;
    shape.num_heads_q = (int) Hq;
    shape.num_heads_kv = (int) Hkv;
    shape.seq_len_qo = (int) Sq;
    shape.seq_len_kv = (int) Skv;
    shape.seq_len_kv_cache = (int) Sc;
    shape.head_size_qk = kHeadDim;
    shape.head_size_vo = kHeadDim;

    typename K::KernelArguments ka{};
    ka.shape = shape;
    ka.Q = q16; ka.dQ = StrideQ{kHeadDim, _1{}, (int) (Sq * kHeadDim), 0};
    ka.K = k16; ka.dK = StrideK{kHeadDim, _1{}, (int) (Skv * kHeadDim), 0};
    ka.V = v16; ka.dV = StrideV{_1{}, kHeadDim, (int) (Skv * kHeadDim), 0};
    ka.O = O;   // a row of D floats per (token, head)
    ka.dO = StrideO{(int) (Hq * kHeadDim), _1{}, kHeadDim, 0};
    ka.K_cache = kc16; ka.dK_cache = StrideK{kHeadDim, _1{}, (int) (Sc * kHeadDim), 0};
    ka.V_cache = vc16; ka.dV_cache = StrideV{_1{}, kHeadDim, (int) (Sc * kHeadDim), 0};

    cutlass::KernelHardwareInfo hw{};
    hw.sm_count = eu_count(q);
    typename K::Arguments args{ka, {scale, nullptr, 0, nullptr}, {}, hw};
    const auto params = K::to_underlying_arguments(args, nullptr);

    namespace syclex  = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    const dim3 block = K::get_block_shape();
    const dim3 grid  = K::get_grid_shape(params);
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(K::SharedStorageSize),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
        intelex::grf_size<256>,
    };
    compat::experimental::launch_policy policy{compat::dim3(grid.x, grid.y, grid.z),
                                               compat::dim3(block.x, block.y, block.z),
                                               launch_props, kernel_props};
    compat::experimental::launch<cutlass::device_kernel<K>, K, false>(policy, q, params);
}

// [D, S, H, B] tensor (F32 or F16, rows contiguous) -> packed F16 [B][H][S][Dp],
// columns D..Dp-1 zero.
void pack_pad(sycl::queue & q, const ggml_tensor * t, Elt * dst, int64_t Dp) {
    const int64_t D = t->ne[0], S = t->ne[1], H = t->ne[2], B = t->ne[3];
    const size_t nb1 = t->nb[1], nb2 = t->nb[2], nb3 = t->nb[3];
    const bool f16 = t->type == GGML_TYPE_F16;
    const char * src = static_cast<const char *>(t->data);
    q.parallel_for(sycl::range<3>(B * H, S, Dp), [=](sycl::id<3> i) {
        const int64_t b = i[0] / H, h = i[0] % H, s = i[1], d = i[2];
        float v = 0.f;
        if (d < D) {
            const char * e = src + b * nb3 + h * nb2 + s * nb1;
            v = f16 ? (float) reinterpret_cast<const sycl::half *>(e)[d] : reinterpret_cast<const float *>(e)[d];
        }
        dst[((b * H + h) * S + s) * Dp + d] = Elt(v);
    });
}

// Packed [B][H][S][Dp] Q/K/V (one KV head per query head) -> O [B][S][H][Dp] F32.
template <class Cfg>
void launch_batched(sycl::queue & q, const Elt * q16, const Elt * k16, const Elt * v16, int64_t B, int64_t S,
                    int64_t Skv, int64_t H, int64_t Dp, float scale, float * O) {
    using K = typename Cfg::Kernel;
    ProblemShape shape{};
    shape.batch = (int) B;
    shape.num_heads_q = (int) H;
    shape.num_heads_kv = (int) H;
    shape.seq_len_qo = (int) S;
    shape.seq_len_kv = (int) Skv;
    shape.seq_len_kv_cache = 0;
    shape.head_size_qk = (int) Dp;
    shape.head_size_vo = (int) Dp;

    typename K::KernelArguments ka{};
    ka.shape = shape;
    ka.Q = q16; ka.dQ = StrideQ{(int) Dp, _1{}, (int) (S * Dp), (int) (H * S * Dp)};
    ka.K = k16; ka.dK = StrideK{(int) Dp, _1{}, (int) (Skv * Dp), (int) (H * Skv * Dp)};
    ka.V = v16; ka.dV = StrideV{_1{}, (int) Dp, (int) (Skv * Dp), (int) (H * Skv * Dp)};
    ka.O = O;
    ka.dO = StrideO{(int) (H * Dp), _1{}, (int) Dp, (int) (S * H * Dp)};
    ka.K_cache = nullptr;
    ka.V_cache = nullptr;

    cutlass::KernelHardwareInfo hw{};
    hw.sm_count = eu_count(q);
    typename K::Arguments args{ka, {scale, nullptr, 0, nullptr}, {}, hw};
    const auto params = K::to_underlying_arguments(args, nullptr);

    namespace syclex  = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    const dim3 block = K::get_block_shape();
    const dim3 grid  = K::get_grid_shape(params);
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(K::SharedStorageSize),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
        intelex::grf_size<256>,
    };
    compat::experimental::launch_policy policy{compat::dim3(grid.x, grid.y, grid.z),
                                               compat::dim3(block.x, block.y, block.z),
                                               launch_props, kernel_props};
    compat::experimental::launch<cutlass::device_kernel<K>, K, false>(policy, q, params);
}

// The head_dim-72 node (SigLIP): no mask, Hq == Hkv, any batch.
bool fmha_vit(sycl::queue & q, ggml_tensor * dst, const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V,
              float scale) {
    const int64_t D = Q->ne[0], S = Q->ne[1], H = Q->ne[2], B = Q->ne[3], Skv = K->ne[1];
    for (const ggml_tensor * t : {Q, K, V})
        if (t->ne[0] != D || t->nb[0] != ggml_type_size(t->type) || t->ne[3] != B ||
            !(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16))
            return false;
    if (K->ne[2] != H || V->ne[2] != H || V->ne[1] != Skv || dst->ne[0] != D || dst->ne[1] != H || dst->ne[2] != S)
        return false;
    const int64_t Dp = kPadDim, nq = B * H * S * Dp, nkv = B * H * Skv * Dp;
    Elt * buf = scratch(q, (size_t) (nq + 2 * nkv + 2 * nq));   // F16 Q, K, V, then the F32 output
    if (!buf) return false;
    Elt * q16 = buf, * k16 = q16 + nq, * v16 = k16 + nkv;
    float * o32 = reinterpret_cast<float *>(v16 + nkv);
    pack_pad(q, Q, q16, Dp);
    pack_pad(q, K, k16, Dp);
    pack_pad(q, V, v16, Dp);
    launch_batched<FmhaVit>(q, q16, k16, v16, B, S, Skv, H, Dp, scale, o32);
    float * out = static_cast<float *>(dst->data);
    q.parallel_for(sycl::range<2>(B * S * H, D), [=](sycl::id<2> i) {
        out[i[0] * D + i[1]] = o32[i[0] * Dp + i[1]];
    });
    return true;
}

uint32_t custom_magic(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CUSTOM) return 0;
    struct custom_params { void * fun; int n_tasks; void * userdata; };   // ggml_custom_op_params
    custom_params p;
    std::memcpy(&p, op->op_params, sizeof(p));
    uint32_t m = 0;
    if (p.userdata) std::memcpy(&m, p.userdata, sizeof(m));
    return m;
}

bool fmha_kv_shape_ok(const ggml_tensor * op) {
    const ggml_tensor * Q = op->src[0], * Kc = op->src[1], * Vc = op->src[2], * Kn = op->src[3], * Vn = op->src[4];
    if (!Q || !Kc || !Vc || !Kn || !Vn) return false;
    if (!rows_ok(Q, false) || !rows_ok(Kc, true) || !rows_ok(Vc, true) || !rows_ok(Kn, true) || !rows_ok(Vn, true))
        return false;
    const int64_t Hq = Q->ne[2], Hkv = Kc->ne[2];
    return Vc->ne[1] == Kc->ne[1] && Vn->ne[1] == Kn->ne[1] && Vc->ne[2] == Hkv && Kn->ne[2] == Hkv &&
           Vn->ne[2] == Hkv && Hkv >= 1 && Hq % Hkv == 0 && Kc->ne[1] > 0 && Kn->ne[1] > 0 &&
           op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
}

}  // namespace

bool sycl_fmha_forward(ggml_tensor * dst, void * queue) {
    if (dst->op != GGML_OP_FLASH_ATTN_EXT || env_off()) return false;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (dst->src[3] || dst->src[4]) return false;                  // mask, sinks
    float scale, max_bias, softcap;
    std::memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) return false;
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) return false;
    if (Q->ne[0] == 72)
        return fmha_vit(*static_cast<sycl::queue *>(queue), dst, Q, K, V, scale);
    if (!rows_ok(Q, false) || !rows_ok(K, true) || !rows_ok(V, true)) return false;
    const int64_t Sq = Q->ne[1], Hq = Q->ne[2], Skv = K->ne[1], Hkv = K->ne[2];
    if (V->ne[1] != Skv || V->ne[2] != Hkv || Hkv < 1 || Hq % Hkv != 0) return false;

    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    Elt * spare = scratch(q, (size_t) (Hq * Sq + 2 * Hkv * Skv) * kHeadDim);
    if (!spare) return false;
    const Elt * q16 = spare;
    to_f16(q, Q, spare);
    spare += (size_t) Hq * Sq * kHeadDim;
    const Elt * k16 = f16_of(q, K, spare), * v16 = f16_of(q, V, spare);
    launch<Fmha<128, 16, false>>(q, q16, Sq, Hq, Hkv, k16, v16, Skv, nullptr, nullptr, 0, scale, static_cast<float *>(dst->data));
    return true;
}

bool sycl_fmha_kv_supports(const ggml_tensor * op) {
    return custom_magic(op) == FMHA_KV_MAGIC && fmha_kv_shape_ok(op);
}

// Split-KV: the keys [Kc; Kn] cut into ns chunks, one batch entry per chunk
// (varlen, so the last chunk may be shorter), every chunk against all R folded
// query rows, each writing its partial O and log2-sum-exp; a combine pass
// merges them. One q64 tile walks ~S/ns keys instead of S, and the launch has
// ns times the work-groups: pi0.5's expert has 400 rows (7 tiles) over ~611 keys.
using FmhaSplit = Fmha<64, 8, false, kHeadDim, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, true>, 64, 1,
                       kPipelineStages, true, true>;

void * split_scratch(sycl::queue & q, size_t bytes) {
    static std::mutex mu;
    static void * p = nullptr;
    static size_t cap = 0;
    static sycl::context ctx = q.get_context();
    std::lock_guard<std::mutex> lk(mu);
    if (p && cap >= bytes && ctx == q.get_context()) return p;
    if (p) { q.wait(); sycl::free(p, ctx); }
    ctx = q.get_context();
    p = sycl::malloc_device(bytes, q);
    cap = p ? bytes : 0;
    return p;
}

// Chunks over S keys for R folded rows: 128 keys (two 64-key tiles) each, and
// only while the rows alone give few work-groups (R <= 192, three q64 tiles).
// On the Arc B390, 80 rows over ~600 keys: 0.12 -> 0.05 ms; 400 rows (pi0.5's
// 50 action tokens x 8 heads) gain less in the kernel than the pack and
// combine passes cost. VLA_FMHA_SPLITS=n forces n (1 = no split).
int64_t split_count(int64_t R, int64_t S) {
    static const int forced = [] { const char * e = std::getenv("VLA_FMHA_SPLITS"); return e ? std::atoi(e) : 0; }();
    if (forced > 0) return std::min<int64_t>(forced, S);
    return (R > 192 || S < 256) ? 1 : (S + 127) / 128;
}

bool fmha_kv_split(sycl::queue & q, const ggml_tensor * Q, const Elt * kc16, const Elt * vc16, int64_t Sc,
                   const ggml_tensor * Kn, const ggml_tensor * Vn, float scale, float * out, int64_t ns) {
    const int64_t Sq = Q->ne[1], Hq = Q->ne[2], R = Sq * Hq, Sn = Kn->ne[1], S = Sc + Sn, D = kHeadDim;
    const int64_t c = (S + ns - 1) / ns;
    ns = (S + c - 1) / c;
    // F16 Q x ns, K, V | F32 partial O, LSE | int cumulative lengths (q, kv, cache)
    const size_t b_q = (size_t) ns * R * D * sizeof(Elt), b_kv = (size_t) S * D * sizeof(Elt);
    const size_t b_o = (size_t) ns * R * D * sizeof(float), b_l = (size_t) ns * R * sizeof(float);
    const size_t b_c = (size_t) 3 * (ns + 1) * sizeof(int);
    char * base = static_cast<char *>(split_scratch(q, b_q + 2 * b_kv + b_o + b_l + b_c));
    if (!base) return false;
    Elt * q16 = reinterpret_cast<Elt *>(base);
    Elt * k16 = reinterpret_cast<Elt *>(base + b_q);
    Elt * v16 = reinterpret_cast<Elt *>(base + b_q + b_kv);
    float * op = reinterpret_cast<float *>(base + b_q + 2 * b_kv);
    float * lse = reinterpret_cast<float *>(base + b_q + 2 * b_kv + b_o);
    int * cu = reinterpret_cast<int *>(base + b_q + 2 * b_kv + b_o + b_l);

    const char * qs = static_cast<const char *>(Q->data);
    const size_t qnb1 = Q->nb[1], qnb2 = Q->nb[2];
    const bool qf16 = Q->type == GGML_TYPE_F16;
    const char * kn = static_cast<const char *>(Kn->data), * vn = static_cast<const char *>(Vn->data);
    const size_t knb1 = Kn->nb[1], vnb1 = Vn->nb[1];
    const bool kf16 = Kn->type == GGML_TYPE_F16, vf16 = Vn->type == GGML_TYPE_F16;
    const int64_t rows = std::max(ns * R, S);
    q.parallel_for(sycl::range<2>((size_t) rows, (size_t) D), [=](sycl::id<2> i) {
        const int64_t r = i[0], d = i[1];
        if (r < ns * R) {   // Q row r % R (token-major: s * Hq + h), copy r / R
            const int64_t rr = r % R, s = rr / Hq, h = rr % Hq;
            const char * e = qs + h * qnb2 + s * qnb1;
            q16[r * D + d] = Elt(qf16 ? (float) reinterpret_cast<const sycl::half *>(e)[d] : reinterpret_cast<const float *>(e)[d]);
        }
        if (r < S) {
            if (r < Sc) {
                k16[r * D + d] = kc16[r * D + d];
                v16[r * D + d] = vc16[r * D + d];
            } else {
                const char * ek = kn + (r - Sc) * knb1, * ev = vn + (r - Sc) * vnb1;
                k16[r * D + d] = Elt(kf16 ? (float) reinterpret_cast<const sycl::half *>(ek)[d] : reinterpret_cast<const float *>(ek)[d]);
                v16[r * D + d] = Elt(vf16 ? (float) reinterpret_cast<const sycl::half *>(ev)[d] : reinterpret_cast<const float *>(ev)[d]);
            }
        }
        if (r <= ns && d == 0) {
            cu[r]              = (int) (r * R);
            cu[ns + 1 + r]     = (int) std::min<int64_t>(r * c, S);
            cu[2 * (ns + 1) + r] = 0;
        }
    });

    using K = FmhaSplit::Kernel;
    using VL = cutlass::fmha::collective::VariableLength;
    cutlass::fmha::kernel::FMHAProblemShape<true> shape{};
    shape.batch = (int) ns;
    shape.num_heads_q = 1;
    shape.num_heads_kv = 1;
    shape.seq_len_qo = VL{ (int) R, cu };
    shape.seq_len_kv = VL{ (int) c, cu + ns + 1 };
    shape.seq_len_kv_cache = VL{ 0, cu + 2 * (ns + 1) };
    shape.head_size_qk = (int) D;
    shape.head_size_vo = (int) D;

    typename K::KernelArguments ka{};
    ka.shape = shape;
    ka.Q = q16; ka.dQ = StrideQ{(int) D, _1{}, (int) (R * D), (int) (R * D)};
    ka.K = k16; ka.dK = StrideK{(int) D, _1{}, (int) (S * D), 0};
    ka.V = v16; ka.dV = StrideV{_1{}, (int) D, (int) (S * D), 0};
    ka.O = op;  ka.dO = StrideO{(int) D, _1{}, (int) (R * D), (int) (R * D)};

    cutlass::KernelHardwareInfo hw{};
    hw.sm_count = eu_count(q);
    typename K::Arguments args{ka, {scale, nullptr, 0, nullptr}, {lse, (int) R, 1}, hw};
    const auto params = K::to_underlying_arguments(args, nullptr);

    namespace syclex  = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    const dim3 block = K::get_block_shape();
    const dim3 grid  = K::get_grid_shape(params);
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(K::SharedStorageSize),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
        intelex::grf_size<256>,
    };
    compat::experimental::launch_policy policy{compat::dim3(grid.x, grid.y, grid.z),
                                               compat::dim3(block.x, block.y, block.z),
                                               launch_props, kernel_props};
    compat::experimental::launch<cutlass::device_kernel<K>, K, false>(policy, q, params);

    // out[r] = sum_b 2^(lse_b - m) O_b[r] / sum_b 2^(lse_b - m)
    q.parallel_for(sycl::range<2>((size_t) R, (size_t) D), [=](sycl::id<2> i) {
        const int64_t r = i[0], d = i[1];
        float m = -INFINITY;
        for (int64_t b = 0; b < ns; ++b) m = sycl::fmax(m, lse[b * R + r]);
        float num = 0.f, den = 0.f;
        for (int64_t b = 0; b < ns; ++b) {
            const float w = sycl::exp2(lse[b * R + r] - m);
            num += w * op[(b * R + r) * D + d];
            den += w;
        }
        out[r * D + d] = num / den;
    });
    return true;
}

bool sycl_fmha_kv_forward(ggml_tensor * dst, void * queue) {
    if (custom_magic(dst) != FMHA_KV_MAGIC || env_off() || !fmha_kv_shape_ok(dst)) return false;
    struct custom_params { void * fun; int n_tasks; void * userdata; };
    custom_params cp;
    std::memcpy(&cp, dst->op_params, sizeof(cp));
    const float scale = static_cast<const FmhaKvSpec *>(cp.userdata)->scale;
    const ggml_tensor * Q = dst->src[0], * Kc = dst->src[1], * Vc = dst->src[2], * Kn = dst->src[3], * Vn = dst->src[4];
    const int64_t Sq = Q->ne[1], Hq = Q->ne[2], Sc = Kc->ne[1], Sn = Kn->ne[1], Hkv = Kc->ne[2];

    // The cache is read in place when it is packed F16 (the graph casts it once
    // per layer); the new keys and Q are converted on the way.
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    // One KV head: the query heads fold into the rows (token-major, so row
    // s*Hq + h is exactly dst's row), and every row of a q64 tile is a real
    // query instead of 50 of 128, with K/V streamed once per tile, not per head.
    const bool fold = Hkv == 1;
    float * O = static_cast<float *>(dst->data);
    if (fold && packed_f16(Kc) && packed_f16(Vc)) {
        const int64_t ns = split_count(Sq * Hq, Sc + Sn);
        if (ns > 1)
            return fmha_kv_split(q, Q, static_cast<const Elt *>(Kc->data), static_cast<const Elt *>(Vc->data), Sc, Kn, Vn,
                                 scale, O, ns);
    }
    Elt * spare = scratch(q, (size_t) (Hq * Sq + 2 * Hkv * (Sc + Sn)) * kHeadDim);
    if (!spare) return false;
    const Elt * q16 = spare;
    to_f16(q, Q, spare, fold);
    spare += (size_t) Hq * Sq * kHeadDim;
    const Elt * kc16 = f16_of(q, Kc, spare), * vc16 = f16_of(q, Vc, spare);
    const Elt * kn16 = f16_of(q, Kn, spare), * vn16 = f16_of(q, Vn, spare);
    if (fold)
        launch<Fmha<64, 8, true>>(q, q16, Sq * Hq, 1, 1, kn16, vn16, Sn, kc16, vc16, Sc, scale, O);
    else
        launch<Fmha<128, 16, true>>(q, q16, Sq, Hq, Hkv, kn16, vn16, Sn, kc16, vc16, Sc, scale, O);
    return true;
}

}  // namespace vla
