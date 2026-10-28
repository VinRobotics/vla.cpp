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
#include "env_flag.h"
#include "layers/fmha_kv.h"
#include "layers/fq_rope_attn.h"
#include "sycl/vla_sycl_ext.h"
#include "sycl/vla_sycl_foldquant.h"
#ifdef VLA_FQ_SYCL_DNNL
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#endif

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
#include <unordered_map>
#include <map>
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
    if (!rows_ok(Q, true) || !rows_ok(K, true) || !rows_ok(V, true)) return false;
    const int64_t Sq = Q->ne[1], Hq = Q->ne[2], Skv = K->ne[1], Hkv = K->ne[2];
    if (V->ne[1] != Skv || V->ne[2] != Hkv || Hkv < 1 || Hq % Hkv != 0) return false;

    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    Elt * spare = scratch(q, (size_t) (Hq * Sq + 2 * Hkv * Skv) * kHeadDim);
    if (!spare) return false;
    const Elt * q16 = f16_of(q, Q, spare);   // in place when packed F16 (fq_rope_pack)
    const Elt * k16 = f16_of(q, K, spare), * v16 = f16_of(q, V, spare);
    launch<Fmha<128, 16, false>>(q, q16, Sq, Hq, Hkv, k16, v16, Skv, nullptr, nullptr, 0, scale, static_cast<float *>(dst->data));
    return true;
}

#ifdef VLA_FQ_SYCL_DNNL
// Attention as two oneDNN GEMMs and a softmax, all keys contiguous: S = Q K^T
// (F16 in, F32 out), P = softmax(scale * S) in F16, O = P V (F32 out). For
// pi0.5's expert (400 folded rows, ~611 keys) 0.06 ms against 0.10-0.13 for the
// tiled flash kernel, which runs only 7 work-groups of 10 serial key tiles.
struct AttnDnnl {
    dnnl::engine eng;
    dnnl::stream strm;
    std::map<std::pair<int64_t, int64_t>, std::pair<dnnl::matmul, dnnl::matmul>> prims;   // (R, S)
    float * s = nullptr; size_t s_cap = 0;
    sycl::half * p = nullptr; size_t p_cap = 0;
};

AttnDnnl & attn_dnnl(sycl::queue & q) {
    static std::unordered_map<sycl::queue *, AttnDnnl> states;
    auto it = states.find(&q);
    if (it == states.end()) {
        AttnDnnl st;
        st.eng  = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        st.strm = dnnl::sycl_interop::make_stream(st.eng, q);
        it = states.emplace(&q, std::move(st)).first;
    }
    return it->second;
}

template <class T>
T * attn_grow(sycl::queue & q, T *& p, size_t & cap, size_t n) {
    if (p && cap >= n) return p;
    if (p) { q.wait(); sycl::free(p, q); }
    p = sycl::malloc_device<T>(n, q);
    cap = p ? n : 0;
    return p;
}

// q16 [R][D], k16 / v16 [S][D] F16 contiguous; O [R][D] F32.
bool attn_gemm(sycl::queue & q, const Elt * q16, const Elt * k16, const Elt * v16, int64_t R, int64_t S, float scale,
               float * O) {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    using md = dnnl::memory::desc;
    const int64_t D = kHeadDim;
    AttnDnnl & st = attn_dnnl(q);
    const md q_md({ R, D }, dt::f16, tag::ab), kT_md({ D, S }, dt::f16, tag::ba), s_md({ R, S }, dt::f32, tag::ab);
    const md p_md({ R, S }, dt::f16, tag::ab), v_md({ S, D }, dt::f16, tag::ab), o_md({ R, D }, dt::f32, tag::ab);
    auto it = st.prims.find({ R, S });
    if (it == st.prims.end()) {
        try {
            it = st.prims.emplace(std::make_pair(R, S),
                                  std::make_pair(dnnl::matmul(dnnl::matmul::primitive_desc(st.eng, q_md, kT_md, s_md)),
                                                 dnnl::matmul(dnnl::matmul::primitive_desc(st.eng, p_md, v_md, o_md)))).first;
        } catch (const dnnl::error &) {
            return false;
        }
    }
    float * sb = attn_grow(q, st.s, st.s_cap, (size_t) (R * S));
    sycl::half * pb = attn_grow(q, st.p, st.p_cap, (size_t) (R * S));
    if (!sb || !pb) return false;
    using dnnl::sycl_interop::make_memory;
    const auto U = dnnl::sycl_interop::memory_kind::usm;
    it->second.first.execute(st.strm, { { DNNL_ARG_SRC, make_memory(q_md, st.eng, U, const_cast<Elt *>(q16)) },
                                        { DNNL_ARG_WEIGHTS, make_memory(kT_md, st.eng, U, const_cast<Elt *>(k16)) },
                                        { DNNL_ARG_DST, make_memory(s_md, st.eng, U, sb) } });
    const int iS = (int) S;
    q.parallel_for(sycl::nd_range<1>((size_t) R * 32, 32), [=](sycl::nd_item<1> it2) [[sycl::reqd_sub_group_size(32)]] {
        const auto sg = it2.get_sub_group();
        const int r = (int) it2.get_group(0), l = (int) sg.get_local_linear_id();
        const float * srow = sb + (size_t) r * iS;
        float m = -INFINITY;
        for (int j = l; j < iS; j += 32) m = sycl::fmax(m, srow[j]);
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        float z = 0.f;
        for (int j = l; j < iS; j += 32) z += sycl::exp((srow[j] - m) * scale);
        z = sycl::reduce_over_group(sg, z, sycl::plus<float>());
        const float iz = 1.0f / z;
        sycl::half * prow = pb + (size_t) r * iS;
        for (int j = l; j < iS; j += 32) prow[j] = sycl::half(sycl::exp((srow[j] - m) * scale) * iz);
    });
    it->second.second.execute(st.strm, { { DNNL_ARG_SRC, make_memory(p_md, st.eng, U, pb) },
                                         { DNNL_ARG_WEIGHTS, make_memory(v_md, st.eng, U, const_cast<Elt *>(v16)) },
                                         { DNNL_ARG_DST, make_memory(o_md, st.eng, U, O) } });
    return true;
}
#endif

// fq_rope_attn (src/layers/fq_rope_attn.h): the q|k|v int32 sums from oneDNN, one
// pass that dequantizes (the GEMM epilogue's math), applies NeoX RoPE to q and k
// and packs Q folded [S*Hq][D], K and V [S][D] as F16, then the cached-KV kernel.
bool sycl_fq_rope_attn_forward(ggml_tensor * dst, void * queue) {
    if (dst->op != GGML_OP_CUSTOM || env_off()) return false;
    struct custom_params { void * fun; int n_tasks; void * userdata; };
    custom_params cp;
    std::memcpy(&cp, dst->op_params, sizeof(cp));
    const auto * sp = static_cast<const FqRopeAttnSpec *>(cp.userdata);
    if (!sp || sp->magic != FQ_ROPE_ATTN_MAGIC || sp->hd != kHeadDim) return false;
    const ggml_tensor * w = dst->src[0], * blob = dst->src[1], * ws = dst->src[2], * bias = dst->src[3];
    const ggml_tensor * pos = dst->src[4], * Kc = dst->src[5], * Vc = dst->src[6];
    if (!w || !blob || !ws || !pos || pos->type != GGML_TYPE_I32 || !packed_f16(Kc) || !packed_f16(Vc) ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst))
        return false;
    const int64_t D = sp->hd, H = sp->nq, N = sp->gemm->N, Sn = blob->ne[1], R = Sn * H;
    const bool    tail = sp->extra > 0;   // Kc/Vc end in Sn free rows for this step's K/V
    const int64_t Sc = Kc->ne[1] - (tail ? Sn : 0);
    if (tail && sp->extra != Sn) return false;
    if (N != (H + 2) * D || Kc->ne[2] != 1 || pos->ne[0] != Sn) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    int64_t xs_off = 0;
    const int32_t * acc = sycl_fq_gemm_acc(queue, w, blob, ws, *sp->gemm, &xs_off);
    if (!acc) return false;
    Elt * spare = scratch(q, (size_t) (R + 2 * Sn) * D);
    if (!spare) return false;
    Elt * q16 = spare, * kn16 = q16 + R * D, * vn16 = kn16 + Sn * D;
    if (tail) {   // write this step's K/V right after the cached ones
        kn16 = static_cast<Elt *>(Kc->data) + Sc * D;
        vn16 = static_cast<Elt *>(Vc->data) + Sc * D;
    }
    const int8_t * bl = static_cast<const int8_t *>(blob->data);
    const int64_t rb = blob->ne[0];
    const float * wsp = static_cast<const float *>(ws->data);
    const float * bp = bias ? static_cast<const float *>(bias->data) : nullptr;
    const int32_t * pp = static_cast<const int32_t *>(pos->data);
    const float theta_scale = std::pow(sp->freq_base, -2.0f / (float) D);
    const int64_t half = D / 2;
    // Row r < R: Q of token r / H, head r % H (dst's own row order); then the
    // Sn K rows, then the Sn V rows. Each item writes one pair (i, i + half).
    q.parallel_for(sycl::range<2>((size_t) (R + 2 * Sn), (size_t) half), [=](sycl::id<2> it) {
        const int64_t r = it[0], i = it[1];
        int64_t t, col;
        Elt * out;
        bool rot = true;
        if (r < R)           { t = r / H; col = (r % H) * D;       out = q16 + r * D; }
        else if (r < R + Sn) { t = r - R; col = H * D;             out = kn16 + t * D; }
        else                 { t = r - R - Sn; col = (H + 1) * D;  out = vn16 + t * D; rot = false; }
        float xs;
        std::memcpy(&xs, bl + t * rb + xs_off, sizeof(float));
        const int32_t * ar = acc + t * N;
        float x0 = ((float) ar[col + i] * xs) * wsp[col + i];
        float x1 = ((float) ar[col + i + half] * xs) * wsp[col + i + half];
        if (bp) { x0 = x0 + bp[col + i]; x1 = x1 + bp[col + i + half]; }
        if (rot) {
            const float th = (float) pp[t] * sycl::pow(theta_scale, (float) i);
            const float c = sycl::cos(th), s = sycl::sin(th);
            const float y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
            x0 = y0; x1 = y1;
        }
        out[i] = Elt(x0);
        out[i + half] = Elt(x1);
    });
#ifdef VLA_FQ_SYCL_DNNL
    if (tail)
        return attn_gemm(q, q16, static_cast<const Elt *>(Kc->data), static_cast<const Elt *>(Vc->data), R, Sc + Sn,
                         sp->scale, static_cast<float *>(dst->data));
#endif
    if (tail) return false;
    launch<Fmha<64, 8, true>>(q, q16, R, 1, 1, kn16, vn16, Sn, static_cast<const Elt *>(Kc->data),
                              static_cast<const Elt *>(Vc->data), Sc, sp->scale, static_cast<float *>(dst->data));
    return true;
}

// fq_rope_pack: the prefix's q|k|v int32 sums, dequantized, RoPE on q and k,
// packed F16 (Q head-major rows, then K, then V) for the flash-attention node
// and the expert's cached K/V.
bool sycl_fq_rope_pack_forward(ggml_tensor * dst, void * queue) {
    if (dst->op != GGML_OP_CUSTOM) return false;
    struct custom_params { void * fun; int n_tasks; void * userdata; };
    custom_params cp;
    std::memcpy(&cp, dst->op_params, sizeof(cp));
    const auto * sp = static_cast<const FqRopeAttnSpec *>(cp.userdata);
    if (!sp || sp->magic != FQ_ROPE_PACK_MAGIC) return false;
    const ggml_tensor * w = dst->src[0], * blob = dst->src[1], * ws = dst->src[2], * bias = dst->src[3], * pos = dst->src[4];
    const int64_t D = sp->hd, H = sp->nq, N = sp->gemm->N, T = blob->ne[1];
    const int64_t E = sp->extra;
    if (!pos || pos->type != GGML_TYPE_I32 || pos->ne[0] != T || N != (H + 2) * D || dst->type != GGML_TYPE_F16 ||
        dst->ne[0] != D || dst->ne[1] != H * T + 2 * (T + E) || (D % 2) != 0)
        return false;
    int64_t xs_off = 0;
    const int32_t * acc = sycl_fq_gemm_acc(queue, w, blob, ws, *sp->gemm, &xs_off);
    if (!acc) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    const int8_t * bl = static_cast<const int8_t *>(blob->data);
    const int rb = (int) blob->ne[0], xo = (int) xs_off, iN = (int) N, iD = (int) D, iH = (int) H, iT = (int) T, iE = (int) E;
    const float * wsp = static_cast<const float *>(ws->data);
    const float * bp = bias ? static_cast<const float *>(bias->data) : nullptr;
    const int32_t * pp = static_cast<const int32_t *>(pos->data);
    const float theta_scale = std::pow(sp->freq_base, -2.0f / (float) D);
    const int half = (int) D / 2;
    auto * out = static_cast<sycl::half *>(dst->data);
    // Item (part, t, i): part < H a q head, H = k, H + 1 = v; pair (i, i + half).
    q.parallel_for(sycl::range<3>((size_t) (H + 2), (size_t) T, (size_t) half), [=](sycl::id<3> it) {
        const int h = (int) it[0], t = (int) it[1], i = (int) it[2];
        const int col = h * iD;
        float xs;
        std::memcpy(&xs, bl + (size_t) t * rb + xo, sizeof(float));
        const int32_t * ar = acc + (size_t) t * iN;
        float x0 = ((float) ar[col + i] * xs) * wsp[col + i];
        float x1 = ((float) ar[col + i + half] * xs) * wsp[col + i + half];
        if (bp) { x0 = x0 + bp[col + i]; x1 = x1 + bp[col + i + half]; }
        if (h <= iH) {
            const float th = (float) pp[t] * sycl::pow(theta_scale, (float) i);
            const float c = sycl::cos(th), s = sycl::sin(th);
            const float y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
            x0 = y0; x1 = y1;
        }
        sycl::half * o = out + (size_t) (h < iH ? h * iT + t : iH * iT + (h - iH) * (iT + iE) + t) * iD;
        o[i] = sycl::half(x0);
        o[i + half] = sycl::half(x1);
    });
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

namespace {

// The hook entry. main's oneDNN SDPA (vla_sycl_attn.cpp) owns FLASH_ATTN_EXT;
// this one sits ahead of it but takes only the attention over a FoldQuant
// prefix pack (fq_rope_pack's packed F16 rows, read in place), plus its own
// custom nodes. The expert's attention over the pack's tail rows (Q from
// fused_rope_kv, GemmaLayerW::fq_kv_sdpa) stays with oneDNN, and so does the
// prefix's unless VLA_FQ_TLA_PREFIX=1: on Arc B390 oneDNN's SDPA runs pi0.5's
// 560-token, 8-head prefix faster.
bool fmha_dispatch(ggml_tensor * dst, void * queue) {
    static const bool tla_prefix = env_flag("VLA_FQ_TLA_PREFIX");
    if (dst->op == GGML_OP_FLASH_ATTN_EXT)
        return tla_prefix && dst->src[0] && dst->src[1] && custom_magic(dst->src[0]->view_src) == FQ_ROPE_PACK_MAGIC &&
               custom_magic(dst->src[1]->view_src) == FQ_ROPE_PACK_MAGIC && sycl_fmha_forward(dst, queue);
    return sycl_fmha_kv_forward(dst, queue) || sycl_fq_rope_attn_forward(dst, queue) ||
           sycl_fq_rope_pack_forward(dst, queue);
}

bool fmha_supports(const ggml_tensor * op) {
    if (env_off()) return false;
    const uint32_t m = custom_magic(op);
    return sycl_fmha_kv_supports(op) || m == FQ_ROPE_ATTN_MAGIC || m == FQ_ROPE_PACK_MAGIC;
}

}  // namespace

void sycl_register_fmha_ops() {
    sycl_ext_register(fmha_dispatch, fmha_supports, /*front=*/true);
}

}  // namespace vla
