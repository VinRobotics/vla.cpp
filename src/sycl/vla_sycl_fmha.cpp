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
// that, accumulates in F32 and writes F32 straight into dst. It takes the keys
// in whole 64-key tiles only; the remainder is merged in by merge_tail.
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

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
constexpr int kTileKV  = 64;   // get<1>(ShapeQK)
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
// LSE out: the KV tail past the last full 64-key tile is merged in afterwards.
using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<Mainloop, ShapeOut, TensorO, void, /*EnableLSE=*/true>;
using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<false>;
// Batch 1, grouped heads allowed (pi0.5 has 8 query heads per KV head).
using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<true, false>;
using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShape, Mainloop, Epilogue, Scheduler>;

bool env_off() {
    const char * e = std::getenv("VLA_SYCL_FMHA");
    return e && std::strcmp(e, "0") == 0;
}

// Device scratch for the F16 Q, K, V and the LSE; grows, never shrinks. The
// queue is in-order, so one buffer serves every node in turn.
struct Scratch {
    std::mutex mu;
    sycl::context ctx;
    char * ptr = nullptr;
    size_t n = 0;
};

char * scratch(sycl::queue & q, size_t bytes) {
    static Scratch s;
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.ptr && s.ctx == q.get_context() && s.n >= bytes) return s.ptr;
    if (s.ptr) {
        q.wait();
        sycl::free(s.ptr, s.ctx);
    }
    s.ctx = q.get_context();
    s.ptr = sycl::malloc_device<char>(bytes, q);
    s.n = s.ptr ? bytes : 0;
    return s.ptr;
}

int eu_count(sycl::queue & q) {
    static const int n = (int) q.get_device().get_info<sycl::info::device::max_compute_units>();
    return n;
}

// First S rows of a [D, *, H] F32 view with byte strides -> packed [H][S][D] F16.
void to_f16(sycl::queue & q, const ggml_tensor * t, Elt * dst, int64_t S) {
    const int64_t D = t->ne[0], H = t->ne[2];
    const size_t nb1 = t->nb[1], nb2 = t->nb[2];
    const char * src = static_cast<const char *>(t->data);
    q.parallel_for(sycl::range<3>(H, S, D), [=](sycl::id<3> i) {
        const int64_t h = i[0], s = i[1], d = i[2];
        const float v = *reinterpret_cast<const float *>(src + h * nb2 + s * nb1 + d * sizeof(float));
        dst[(h * S + s) * D + d] = Elt(v);
    });
}

// The keys past the last full KV tile, merged into the FMHA output. On the Arc
// B390 the kernel's own remainder masking gives wrong rows (any S_kv % 64 != 0;
// tests/fmha_sycl_check), so the kernel only sees whole tiles and this does the
// rest exactly: one work-group per (head, query) row, one work-item per channel.
// lse2 is the kernel's log2(sum exp(scale * s)) over its keys; empty when
// kv0 == 0. The tail is read from the F32 inputs.
void merge_tail(sycl::queue & q, const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V,
                float * O, const float * lse2, int64_t kv0, float scale) {
    const int64_t Sq = Q->ne[1], Hq = Q->ne[2], Skv = K->ne[1], Hkv = K->ne[2];
    const int64_t grp = Hq / Hkv, ntail = Skv - kv0;
    const char * qp = static_cast<const char *>(Q->data);
    const char * kp = static_cast<const char *>(K->data);
    const char * vp = static_cast<const char *>(V->data);
    const size_t qb1 = Q->nb[1], qb2 = Q->nb[2], kb1 = K->nb[1], kb2 = K->nb[2], vb1 = V->nb[1], vb2 = V->nb[2];
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(kTileKV), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(Hq * Sq * kHeadDim), sycl::range<1>(kHeadDim)),
                       [=](sycl::nd_item<1> it) {
            const int64_t row = it.get_group(0), d = it.get_local_id(0);
            const int64_t hq = row / Sq, s = row % Sq, hk = hq / grp;
            const float qd = *reinterpret_cast<const float *>(qp + hq * qb2 + s * qb1 + d * sizeof(float));
            float m = -INFINITY;
            for (int64_t t = 0; t < ntail; ++t) {
                const float kd = *reinterpret_cast<const float *>(kp + hk * kb2 + (kv0 + t) * kb1 + d * sizeof(float));
                const float st = sycl::reduce_over_group(it.get_group(), qd * kd, sycl::plus<float>()) * scale;
                if (d == 0) sc[t] = st;
                m = sycl::fmax(m, st);
            }
            sycl::group_barrier(it.get_group());
            float l = 0.f, acc = 0.f;
            for (int64_t t = 0; t < ntail; ++t) {
                const float w = sycl::exp(sc[t] - m);
                l += w;
                acc += w * *reinterpret_cast<const float *>(vp + hk * vb2 + (kv0 + t) * vb1 + d * sizeof(float));
            }
            float * o = O + (s * Hq + hq) * kHeadDim + d;
            if (kv0 == 0) {
                *o = acc / l;
                return;
            }
            // Natural-log mass of each part, then the usual two-way softmax merge.
            const float a = lse2[hq * Sq + s] * 0.69314718056f, b = m + sycl::log(l);
            const float mx = sycl::fmax(a, b), wa = sycl::exp(a - mx), wb = sycl::exp(b - mx);
            *o = (wa * *o + wb * (acc / l)) / (wa + wb);
        });
    });
}

// VLA_SYCL_FMHA_CHECK=1: after each claimed node, recompute it on the host in
// double from the F32 inputs and print the worst error and the input ranges.
void check_against_host(sycl::queue & q, const ggml_tensor * dst, const ggml_tensor * Q,
                        const ggml_tensor * K, const ggml_tensor * V, float scale) {
    auto fetch = [&](const ggml_tensor * t) {
        std::vector<float> h(ggml_nbytes(t) / sizeof(float));
        q.memcpy(h.data(), t->data, ggml_nbytes(t)).wait();
        return h;
    };
    q.wait();
    const auto hq = fetch(Q), hk = fetch(K), hv = fetch(V), ho = fetch(dst);
    const int64_t D = Q->ne[0], Sq = Q->ne[1], Hq = Q->ne[2], Skv = K->ne[1], Hkv = K->ne[2];
    auto at = [](const std::vector<float> & h, const ggml_tensor * t, int64_t d, int64_t s, int64_t hh) {
        return h[(hh * t->nb[2] + s * t->nb[1]) / sizeof(float) + d];
    };
    float amax_q = 0, amax_k = 0, amax_v = 0;
    for (float x : hq) amax_q = std::max(amax_q, std::fabs(x));
    for (float x : hk) amax_k = std::max(amax_k, std::fabs(x));
    for (float x : hv) amax_v = std::max(amax_v, std::fabs(x));
    double worst = 0, ref_max = 0;
    std::vector<double> sc(Skv), o(D);
    for (int64_t h = 0; h < Hq; ++h) {
        const int64_t hk_i = h / (Hq / Hkv);
        for (int64_t s = 0; s < Sq; s += std::max<int64_t>(1, Sq / 7)) {
            double m = -1e300;
            for (int64_t t = 0; t < Skv; ++t) {
                double a = 0;
                for (int64_t d = 0; d < D; ++d) a += (double) at(hq, Q, d, s, h) * at(hk, K, d, t, hk_i);
                sc[t] = a * scale; m = std::max(m, sc[t]);
            }
            double z = 0;
            for (int64_t t = 0; t < Skv; ++t) { sc[t] = std::exp(sc[t] - m); z += sc[t]; }
            std::fill(o.begin(), o.end(), 0.0);
            for (int64_t t = 0; t < Skv; ++t)
                for (int64_t d = 0; d < D; ++d) o[d] += sc[t] / z * at(hv, V, d, t, hk_i);
            for (int64_t d = 0; d < D; ++d) {
                const double got = ho[(s * Hq + h) * D + d];
                worst = std::max(worst, std::fabs(got - o[d]));
                ref_max = std::max(ref_max, std::fabs(o[d]));
            }
        }
    }
    std::fprintf(stderr, "vla(fmha) CHECK %-28s Sq=%lld Skv=%lld Hq=%lld Hkv=%lld  max|err| %.3g (ref max %.3g)  "
                "absmax q %.3g k %.3g v %.3g\n", ggml_get_name(dst), (long long) Sq, (long long) Skv,
                (long long) Hq, (long long) Hkv, worst, ref_max, amax_q, amax_k, amax_v);
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
    const int64_t kv0 = Skv / kTileKV * kTileKV;   // keys the kernel takes; the rest is the tail
    const size_t nq = (size_t) Hq * Sq * kHeadDim, nkv = (size_t) Hkv * kv0 * kHeadDim;
    char * buf = scratch(q, (nq + 2 * nkv) * sizeof(Elt) + (size_t) Hq * Sq * sizeof(float));
    if (!buf) return false;
    Elt * q16 = reinterpret_cast<Elt *>(buf), * k16 = q16 + nq, * v16 = k16 + nkv;
    float * lse2 = reinterpret_cast<float *>(v16 + nkv);
    if (kv0 == 0) {   // shorter than one tile: the tail pass is the whole attention
        merge_tail(q, Q, K, V, static_cast<float *>(dst->data), nullptr, 0, scale);
        return true;
    }
    to_f16(q, Q, q16, Sq);
    to_f16(q, K, k16, kv0);
    to_f16(q, V, v16, kv0);

    ProblemShape shape{};
    shape.batch = 1;
    shape.num_heads_q = (int) Hq;
    shape.num_heads_kv = (int) Hkv;
    shape.seq_len_qo = (int) Sq;
    shape.seq_len_kv = (int) kv0;
    shape.seq_len_kv_cache = 0;
    shape.head_size_qk = kHeadDim;
    shape.head_size_vo = kHeadDim;

    typename Kernel::KernelArguments ka{};
    ka.shape = shape;
    ka.Q = q16; ka.dQ = StrideQ{kHeadDim, _1{}, (int) (Sq * kHeadDim), (int) nq};
    ka.K = k16; ka.dK = StrideK{kHeadDim, _1{}, (int) (kv0 * kHeadDim), (int) nkv};
    ka.V = v16; ka.dV = StrideV{_1{}, kHeadDim, (int) (kv0 * kHeadDim), (int) nkv};
    // dst is [D, Hq, Sq]: a row of D floats per (token, head).
    ka.O = static_cast<float *>(dst->data);
    ka.dO = StrideO{(int) (Hq * kHeadDim), _1{}, kHeadDim, (int) (Sq * Hq * kHeadDim)};
    ka.Lse = kv0 < Skv ? lse2 : nullptr;
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
    if (kv0 < Skv)
        merge_tail(q, Q, K, V, static_cast<float *>(dst->data), lse2, kv0, scale);
    static const bool check = [] { const char * e = std::getenv("VLA_SYCL_FMHA_CHECK"); return e && *e && *e != '0'; }();
    if (check) check_against_host(q, dst, Q, K, V, scale);
    return true;
}

}  // namespace vla
