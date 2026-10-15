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
// that, accumulates in F32 and writes F32 into dst. It takes the keys in whole
// 64-key tiles only, split along its batch axis when there are too few query
// tiles to fill the GPU; merge() folds the splits and the remaining keys in.
//
// VLA_SYCL_FMHA=0 turns the claim off.

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
// Grouped heads (pi0.5 has 8 query heads per KV head) and a batch axis, which
// carries the KV splits (see sycl_fmha_forward).
using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false>;
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

// Rows of D values in a ggml tensor: element (d, s, h) at p + s*nbs + h*nbh.
struct Rows {
    const char * p;
    size_t nbs, nbh;
    bool f16;
};

Rows rows_of(const ggml_tensor * t, int64_t s0 = 0) {
    return { static_cast<const char *>(t->data) + s0 * t->nb[1], t->nb[1], t->nb[2], t->type == GGML_TYPE_F16 };
}

// n rows of each of H heads -> packed F16 [H][n][D].
void to_f16(sycl::queue & q, Rows r, int64_t n, int64_t H, Elt * dst) {
    q.parallel_for(sycl::range<3>(H, n, kHeadDim), [=](sycl::id<3> i) {
        const int64_t h = i[0], s = i[1], d = i[2];
        const char * e = r.p + h * r.nbh + s * r.nbs;
        const float v = r.f16 ? (float) reinterpret_cast<const sycl::half *>(e)[d] : reinterpret_cast<const float *>(e)[d];
        dst[(h * n + s) * kHeadDim + d] = Elt(v);
    });
}

// n rows of each of H heads -> F32 rows [off, off + n) of a [H][ld][D] buffer.
void to_f32(sycl::queue & q, Rows r, int64_t n, int64_t H, float * dst, int64_t ld, int64_t off) {
    if (n <= 0) return;
    q.parallel_for(sycl::range<3>(H, n, kHeadDim), [=](sycl::id<3> i) {
        const int64_t h = i[0], s = i[1], d = i[2];
        const char * e = r.p + h * r.nbh + s * r.nbs;
        dst[(h * ld + off + s) * kHeadDim + d] =
            r.f16 ? (float) reinterpret_cast<const sycl::half *>(e)[d] : reinterpret_cast<const float *>(e)[d];
    });
}

// Folds the kernel's per-split partial outputs and the keys it did not see into
// O, exactly. On the Arc B390 the kernel's own remainder masking gives wrong rows
// for any S_kv % 64 != 0 (tests/fmha_sycl_check), so it only ever gets whole KV
// tiles; the rest (kt/vt, F32 [Hkv][ntail][D]) is scored here. ob/lse2 hold
// nsplit partial outputs [split][head][query][D] and their log2(sum exp(scale*s)).
// Three light passes: tail scores, a row of mixing weights per (head, query),
// then a weighted sum per output element.
void merge(sycl::queue & q, Rows qr, int64_t Sq, int64_t Hq, int64_t Hkv, const float * kt, const float * vt,
           int64_t ntail, float * O, const float * ob, const float * lse2, float * coef, int nsplit, float scale) {
    const int64_t grp = Hq / Hkv, rows = Hq * Sq, nc = nsplit + ntail;
    // coef[row][0..nsplit) weigh the splits, coef[row][nsplit + t] tail key t.
    if (ntail > 0)
        q.parallel_for(sycl::range<2>(rows, ntail), [=](sycl::id<2> i) {
            const int64_t row = i[0], t = i[1], hq = row / Sq, s = row % Sq, hk = hq / grp;
            const sycl::float4 * a4 = reinterpret_cast<const sycl::float4 *>(qr.p + hq * qr.nbh + s * qr.nbs);
            const sycl::float4 * k4 = reinterpret_cast<const sycl::float4 *>(kt + (hk * ntail + t) * kHeadDim);
            sycl::float4 a{0.f};
            for (int j = 0; j < kHeadDim / 4; ++j) a += a4[j] * k4[j];
            coef[row * nc + nsplit + t] = (a.x() + a.y() + a.z() + a.w()) * scale;
        });
    q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> i) {
        const int64_t row = i[0];
        float * c = coef + row * nc;
        // Each split stands for exp(lse) of softmax mass at once; each tail key for exp(score).
        float mx = -INFINITY;
        for (int b = 0; b < nsplit; ++b) mx = sycl::fmax(mx, lse2[b * rows + row] * 0.69314718056f);
        for (int64_t t = 0; t < ntail; ++t) mx = sycl::fmax(mx, c[nsplit + t]);
        float den = 0.f;
        for (int b = 0; b < nsplit; ++b) { c[b] = sycl::exp(lse2[b * rows + row] * 0.69314718056f - mx); den += c[b]; }
        for (int64_t t = 0; t < ntail; ++t) { c[nsplit + t] = sycl::exp(c[nsplit + t] - mx); den += c[nsplit + t]; }
        const float inv = 1.f / den;
        for (int64_t j = 0; j < nc; ++j) c[j] *= inv;
    });
    q.parallel_for(sycl::range<2>(rows, kHeadDim), [=](sycl::id<2> i) {
        const int64_t row = i[0], d = i[1], hq = row / Sq, s = row % Sq, hk = hq / grp;
        const float * c = coef + row * nc;
        float acc = 0.f;
        for (int b = 0; b < nsplit; ++b) acc += c[b] * ob[(b * rows + row) * kHeadDim + d];
        for (int64_t t = 0; t < ntail; ++t) acc += c[nsplit + t] * vt[(hk * ntail + t) * kHeadDim + d];
        O[(s * Hq + hq) * kHeadDim + d] = acc;
    });
}

// How many ways to split the whole KV tiles, each split a whole number of
// tiles. On the Arc B390 splitting did not pay end to end for pi0.5 (the extra
// merge passes cost what the fuller GPU saves), so the default is 1.
// VLA_SYCL_FMHA_SPLIT=n forces n (rounded down to a divisor).
int pick_split(int64_t nblk) {
    static const int forced = [] { const char * e = std::getenv("VLA_SYCL_FMHA_SPLIT"); return e ? std::atoi(e) : 0; }();
    int64_t want = std::min<int64_t>(std::max(forced, 1), nblk);
    while (nblk % want) --want;
    return (int) want;
}

// The kernel over keys [0, kv0) (kv0 a whole number of tiles) of K/V, packed
// F16 [Hkv][ldk][D]. Writes O ([D, Hq, Sq] F32) itself when direct, else
// nsplit partial outputs to ob and their LSE to lse2.
void run_kernel(sycl::queue & q, const Elt * q16, const Elt * k16, const Elt * v16, int64_t ldk, int64_t kv0,
                int64_t Sq, int64_t Hq, int64_t Hkv, float scale, int nsplit, bool direct, float * O, float * ob,
                float * lse2) {
    const int64_t chunk = kv0 / nsplit, nq = Hq * Sq * kHeadDim;
    ProblemShape shape{};
    shape.batch = nsplit;
    shape.num_heads_q = (int) Hq;
    shape.num_heads_kv = (int) Hkv;
    shape.seq_len_qo = (int) Sq;
    shape.seq_len_kv = (int) chunk;
    shape.seq_len_kv_cache = 0;
    shape.head_size_qk = kHeadDim;
    shape.head_size_vo = kHeadDim;

    // Batch b is split b: the same Q (batch stride 0) against keys
    // [b*chunk, (b+1)*chunk) of each KV head.
    typename Kernel::KernelArguments ka{};
    ka.shape = shape;
    ka.Q = q16; ka.dQ = StrideQ{kHeadDim, _1{}, (int) (Sq * kHeadDim), 0};
    ka.K = k16; ka.dK = StrideK{kHeadDim, _1{}, (int) (ldk * kHeadDim), (int) (chunk * kHeadDim)};
    ka.V = v16; ka.dV = StrideV{_1{}, kHeadDim, (int) (ldk * kHeadDim), (int) (chunk * kHeadDim)};
    if (direct) {
        // O is [D, Hq, Sq]: a row of D floats per (token, head).
        ka.O = O;
        ka.dO = StrideO{(int) (Hq * kHeadDim), _1{}, kHeadDim, 0};
    } else {
        ka.O = ob;
        ka.dO = StrideO{kHeadDim, _1{}, (int) (Sq * kHeadDim), (int) nq};
        ka.Lse = lse2;
    }
    ka.K_cache = nullptr;
    ka.V_cache = nullptr;

    cutlass::KernelHardwareInfo hw{};
    hw.sm_count = eu_count(q);
    typename Kernel::Arguments args{ka, {scale, nullptr, 0, nullptr}, {}, hw};
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
}

// Device scratch carved for one node.
struct Buffers {
    Elt * q16, * k16, * v16;
    float * kt, * vt, * ob, * lse2, * coef;
};

// Attention of Q rows against cache rows [0, kv0) (F16, packed [Hkv][ldk][D],
// or converted from kc/vc when they are not) plus ntail tail rows already in
// b.kt/b.vt, written into O.
bool attend(sycl::queue & q, Rows qr, int64_t Sq, int64_t Hq, int64_t Hkv, const Elt * k16, const Elt * v16,
            int64_t ldk, int64_t kv0, int64_t ntail, float scale, const Buffers & b, float * O) {
    const int nsplit = kv0 ? pick_split(kv0 / kTileKV) : 0;
    const bool direct = nsplit == 1 && ntail == 0;
    if (nsplit > 0) {
        to_f16(q, qr, Sq, Hq, b.q16);
        run_kernel(q, b.q16, k16, v16, ldk, kv0, Sq, Hq, Hkv, scale, nsplit, direct, O, b.ob, b.lse2);
    }
    if (!direct)
        merge(q, qr, Sq, Hq, Hkv, b.kt, b.vt, ntail, O, b.ob, b.lse2, b.coef, nsplit, scale);
    return true;
}

Buffers carve(sycl::queue & q, int64_t Sq, int64_t Hq, int64_t Hkv, int64_t nkv16, int64_t kv0, int64_t ntail,
              bool & ok) {
    const int nsplit = kv0 ? pick_split(kv0 / kTileKV) : 0;
    const size_t nq = (size_t) Hq * Sq * kHeadDim, nkv = (size_t) Hkv * nkv16 * kHeadDim;
    const size_t nt = (size_t) Hkv * ntail * kHeadDim, nob = (size_t) nsplit * nq, nlse = (size_t) nsplit * Hq * Sq;
    const size_t ncoef = (size_t) Hq * Sq * (nsplit + ntail);
    // F16 first (multiples of D elements), then the F32 rows (float4 reads), then the rest.
    char * buf = scratch(q, (nq + 2 * nkv) * sizeof(Elt) + (2 * nt + nob + nlse + ncoef) * sizeof(float));
    Buffers b{};
    ok = buf != nullptr;
    if (!ok) return b;
    b.q16 = reinterpret_cast<Elt *>(buf);
    b.k16 = b.q16 + nq;
    b.v16 = b.k16 + nkv;
    b.kt = reinterpret_cast<float *>(b.v16 + nkv);
    b.vt = b.kt + nt;
    b.ob = b.vt + nt;
    b.lse2 = b.ob + nob;
    b.coef = b.lse2 + nlse;
    return b;
}

bool rows_ok(const ggml_tensor * t, bool f16_ok) {
    return t->ne[0] == kHeadDim && t->ne[3] == 1 &&
           (t->type == GGML_TYPE_F32 || (f16_ok && t->type == GGML_TYPE_F16)) &&
           t->nb[0] == ggml_type_size(t->type) && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0;
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
    if (!rows_ok(Q, false) || !rows_ok(K, false) || !rows_ok(V, false)) return false;
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) return false;
    const int64_t Sq = Q->ne[1], Hq = Q->ne[2], Skv = K->ne[1], Hkv = K->ne[2];
    if (V->ne[1] != Skv || V->ne[2] != Hkv || Hkv < 1 || Hq % Hkv != 0) return false;

    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    const int64_t kv0 = Skv / kTileKV * kTileKV, ntail = Skv - kv0;   // whole tiles to the kernel
    bool ok = false;
    const Buffers b = carve(q, Sq, Hq, Hkv, kv0, kv0, ntail, ok);
    if (!ok) return false;
    if (kv0) {
        to_f16(q, rows_of(K), kv0, Hkv, b.k16);
        to_f16(q, rows_of(V), kv0, Hkv, b.v16);
    }
    to_f32(q, rows_of(K, kv0), ntail, Hkv, b.kt, ntail, 0);
    to_f32(q, rows_of(V, kv0), ntail, Hkv, b.vt, ntail, 0);
    attend(q, rows_of(Q), Sq, Hq, Hkv, b.k16, b.v16, kv0, kv0, ntail, scale, b, static_cast<float *>(dst->data));
    static const bool check = [] { const char * e = std::getenv("VLA_SYCL_FMHA_CHECK"); return e && *e && *e != '0'; }();
    if (check) check_against_host(q, dst, Q, K, V, scale);
    return true;
}

namespace {

bool fmha_kv_shape_ok(const ggml_tensor * op) {
    const ggml_tensor * Q = op->src[0], * Kc = op->src[1], * Vc = op->src[2], * Kn = op->src[3], * Vn = op->src[4];
    if (!Q || !Kc || !Vc || !Kn || !Vn) return false;
    if (!rows_ok(Q, false) || !rows_ok(Kc, true) || !rows_ok(Vc, true) || !rows_ok(Kn, false) || !rows_ok(Vn, false))
        return false;
    const int64_t Hq = Q->ne[2], Hkv = Kc->ne[2];
    return Vc->ne[1] == Kc->ne[1] && Vn->ne[1] == Kn->ne[1] && Vc->ne[2] == Hkv && Kn->ne[2] == Hkv &&
           Vn->ne[2] == Hkv && Hkv >= 1 && Hq % Hkv == 0 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
           Kc->ne[1] + Kn->ne[1] > 0;
}

bool packed_f16(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F16 && ggml_is_contiguous(t);
}

}  // namespace

bool sycl_fmha_kv_supports(const ggml_tensor * op) {
    return custom_magic(op) == FMHA_KV_MAGIC && fmha_kv_shape_ok(op);
}

bool sycl_fmha_kv_forward(ggml_tensor * dst, void * queue) {
    if (custom_magic(dst) != FMHA_KV_MAGIC) return false;
    if (!fmha_kv_shape_ok(dst)) return false;
    struct custom_params { void * fun; int n_tasks; void * userdata; };
    custom_params cp;
    std::memcpy(&cp, dst->op_params, sizeof(cp));
    const float scale = static_cast<const FmhaKvSpec *>(cp.userdata)->scale;
    const ggml_tensor * Q = dst->src[0], * Kc = dst->src[1], * Vc = dst->src[2], * Kn = dst->src[3], * Vn = dst->src[4];
    const int64_t Sq = Q->ne[1], Hq = Q->ne[2], Sc = Kc->ne[1], Sn = Kn->ne[1], Hkv = Kc->ne[2];

    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    // Whole tiles of the cache go to the kernel, read in place when they are
    // already packed F16 (the graph casts them once per layer); the rest of the
    // cache and every new key are the tail.
    const int64_t kv0 = Sc / kTileKV * kTileKV, ntail = Sc - kv0 + Sn;
    const bool in_place = packed_f16(Kc) && packed_f16(Vc);
    bool ok = false;
    const Buffers b = carve(q, Sq, Hq, Hkv, in_place ? 0 : kv0, kv0, ntail, ok);
    if (!ok) return false;
    const Elt * k16 = in_place ? static_cast<const Elt *>(Kc->data) : b.k16;
    const Elt * v16 = in_place ? static_cast<const Elt *>(Vc->data) : b.v16;
    const int64_t ldk = in_place ? Sc : kv0;
    if (!in_place && kv0) {
        to_f16(q, rows_of(Kc), kv0, Hkv, b.k16);
        to_f16(q, rows_of(Vc), kv0, Hkv, b.v16);
    }
    to_f32(q, rows_of(Kc, kv0), Sc - kv0, Hkv, b.kt, ntail, 0);
    to_f32(q, rows_of(Vc, kv0), Sc - kv0, Hkv, b.vt, ntail, 0);
    to_f32(q, rows_of(Kn), Sn, Hkv, b.kt, ntail, Sc - kv0);
    to_f32(q, rows_of(Vn), Sn, Hkv, b.vt, ntail, Sc - kv0);
    return attend(q, rows_of(Q), Sq, Hq, Hkv, k16, v16, ldk, kv0, ntail, scale, b, static_cast<float *>(dst->data));
}

}  // namespace vla
