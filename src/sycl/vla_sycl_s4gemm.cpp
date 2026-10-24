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

// INT4 x INT4 GEMMs on Xe DPAS (s4 x s4 -> s32) from sycl-tla's CuTe, for
// FoldQuant W4A4 sites (pi0.5's prefix). oneDNN has no s4 x s4 matmul, so its
// path runs INT4 sites at the INT8 rate; DPAS's int4 mode doubles that. Built
// on sycl-tla's examples/cute/tutorial/xe_gemm.cpp main loop. The same main
// loop on s8 x s8 serves INT8 sites at the action expert's few rows, where it
// carries the epilogue that oneDNN's matmul would leave to a second kernel.
//
// A: nibble-packed activation blob rows (FoldQuant act layout: K/2 code bytes,
//    then the float row scale), pitch = blob row bytes.
// B: nibble-packed weight rows [N][K/2] (low nibble = even column).

#include "sycl/vla_sycl_s4gemm.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cute/tensor.hpp>

#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace vla {

namespace {

using namespace cute;

using MmaOp   = XE_DPAS_TT<8, int32_t, int4_t, int4_t>;
// s32 kernels, by K (Arc B390, 561 rows): 128 x 128 for the long-K down site
// (0.33 -> 0.31 ms), 64 x 256 otherwise (q/k/v 0.058 -> 0.044, o 0.047 -> 0.038).
using SGL4x4  = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
using SGL2x8s = Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>;
using MmaLongK  = typename TiledMMAHelper<MMA_Atom<MmaOp>, Layout<Shape<_128, _128, _128>>, SGL4x4>::TiledMMA;
using MmaShortK = typename TiledMMAHelper<MMA_Atom<MmaOp>, Layout<Shape<_64, _256, _128>>, SGL2x8s>::TiledMMA;
// Up to 16 rows (pi0.5's 10 action tokens): 64 x 256 tiles left o/down with 4
// work-groups on the Arc B390; 16 x 64 gives 16 (q/k/v 40), W4A4 87.7 -> 83.9 ms.
using SGL1x4  = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using MmaSmallM = typename TiledMMAHelper<MMA_Atom<MmaOp>, Layout<Shape<_16, _64, _128>>, SGL1x4>::TiledMMA;
// INT8 x INT8 (FoldQuant W8A8 sites), the small-M shape only: larger M stays on
// oneDNN's int8 matmul.
using MmaOp8     = XE_DPAS_TT<8, int32_t, int8_t, int8_t>;
using MmaSmallM8 = typename TiledMMAHelper<MMA_Atom<MmaOp8>, Layout<Shape<_16, _64, _128>>, SGL1x4>::TiledMMA;
// The GEGLU kernel: two accumulators (gate, up) per work-group; 64 x 256 measured
// best among 5 tilings on the Arc B390 (561 x 2 x 16384 x 2048: 0.84 ms).
using SGL2x8  = Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>;
using MmaHalf = typename TiledMMAHelper<MMA_Atom<MmaOp>, Layout<Shape<_64, _256, _128>>, SGL2x8>::TiledMMA;
constexpr int GLU_M = 64, GLU_N = 256;
// Small parts (4 Xe cores, e.g. Core Ultra 7 356H, 528 rows): the tiles above leave
// the INT4 DPAS at ~38 TOPS there. These reach 54-55 TOPS: 192 x 128 for the long-K
// down site (0.93 -> 0.66 ms), and for the GEGLU 48 x 64 tiles (48 divides the
// prefix's 528 rows) at GRF 128, whose 8 threads per vector engine hide the load
// latency better than the larger tile at GRF 256 (1.84 -> 1.29 ms). Prefetch
// distance 2 on both.
using SGL2x4      = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
using MmaLongKXe4 = typename TiledMMAHelper<MMA_Atom<MmaOp>, Layout<Shape<_192, _128, _128>>, SGL4x4>::TiledMMA;
using MmaHalfXe4  = typename TiledMMAHelper<MMA_Atom<MmaOp>, Layout<Shape<_48, _64, _128>>, SGL2x4>::TiledMMA;
constexpr int GLU_M_XE4 = 48, GLU_N_XE4 = 64;

// VLA_FQ_S4_NO_SMALL_M=1: the 64-row tiles at every M.
bool no_small_m() {
    static const bool off = [] { const char * e = std::getenv("VLA_FQ_S4_NO_SMALL_M"); return e && *e && *e != '0'; }();
    return off;
}

template <class T>
auto row_major(T * p, int rows, int cols, int pitch) {
    return make_tensor(make_gmem_ptr(p), make_layout(make_shape(rows, cols), make_stride(pitch, _1{})));
}

// GELU (tanh form) with tanh(z) = 1 - 2 / (e^2z + 1) on the native exp: the
// precise tanh made the epilogue cost as much as the main loop. The result is
// rounded to BF16 anyway.
inline float gelu_tanh_dev(float x) {
    const float z = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
    return 0.5f * x * (2.0f - 2.0f / (sycl::native::exp(2.0f * z) + 1.0f));
}

// The F32 epilogue of vla_sycl_foldquant.cpp's launch_epilogue, in registers:
// ((acc * xs[m]) * ws[n]) (+ bias[n]) (+ res[m][n], or res + v * gate[n]), with
// xs the float at byte xs_off of blob row m. Contraction off, so the order and
// rounding are the reference's and the output is the same bit for bit.
struct Epi {
    const int8_t * blob;
    int            row_bytes, xs_off, M, N;
    const float *  ws, * bias, * res, * gate;
};
struct NoEpi {};

// r: res[m][n], loaded with the tile (unused without res).
inline float epi_value(const Epi & e, int32_t acc, float r, int m, int n) {
#pragma clang fp contract(off)
    const float xs = *reinterpret_cast<const float *>(e.blob + (size_t) m * e.row_bytes + e.xs_off);   // 4-aligned
    float v = ((float) acc * xs) * e.ws[n];
    if (e.bias) v = v + e.bias[n];
    if (e.res) v = e.gate ? r + v * e.gate[n] : v + r;
    return v;
}

// C (M,N) = A (M,K) s4 x B (N,K)^T s4: the s32 sums (NoEpi), or F32 through Epi.
// PF: prefetch distance in K tiles.
template <class Mma, int PF, class AT, class BT, class CT, class E>
void gemm_s32_device(AT const & A, BT const & B, CT & C, const E & e) {
    Mma mma{};
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    const int wg_m = int(item.get_group(1)), wg_n = int(item.get_group(0)), lid = int(item.get_local_id(0));
    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    Tensor cC = make_identity_tensor(C.shape());
    auto wg_tile = mma.tile_mnk();
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gC = local_tile(cC, wg_tile, make_coord(wg_m, wg_n, 0), Step<_1, _1, X>{});
    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);
    auto copy_c = make_block_2d_copy_D(mma, C);
    auto thr_mma = mma.get_slice(lid);
    auto thr_a = copy_a.get_slice(lid);
    auto thr_b = copy_b.get_slice(lid);
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tArA = thr_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_b.partition_sg_fragment_D(gB(_, _, 0));
    Tensor tAgA = thr_a.partition_S(gA);
    Tensor tBgB = thr_b.partition_S(gB);
    Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
    Tensor tCgC = thr_mma.partition_C(gC);
    auto pf_a = make_block_2d_prefetch(copy_a);
    auto pf_b = make_block_2d_prefetch(copy_b);
    auto tpa = pf_a.get_slice(lid).partition_S(gA);
    auto tpb = pf_b.get_slice(lid).partition_S(gB);
    const int k_tiles = ceil_div(shape<1>(A), get<2>(wg_tile));
    int kp = 0;
    clear(tCrC);
    CUTE_UNROLL
    for (; kp < PF; kp++) { prefetch(pf_a, tpa(_, _, _, kp)); prefetch(pf_b, tpb(_, _, _, kp)); }
    for (int kt = 0; kt < k_tiles; kt++, kp++) {
        barrier_arrive(ScopeWorkgroup);
        copy(copy_a, tAgA(_, _, _, kt), tArA);
        copy(copy_b, tBgB(_, _, _, kt), tBrB);
        prefetch(pf_a, tpa(_, _, _, kp));
        prefetch(pf_b, tpb(_, _, _, kp));
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);
        gemm(mma, tCrA, tCrB, tCrC);
        barrier_wait(ScopeWorkgroup);
    }
    if constexpr (std::is_same_v<E, NoEpi>) {
        copy(copy_c, tCrC, tCgC);
    } else {
        auto tCrO = make_fragment_like<float>(tCrC);
        // The residual tile as 2D block loads (rows past M read as zero), then
        // reordered into the accumulator's layout.
        auto tCrR = make_subgroup_tensor(make_fragment_like<float>(tCrC), layout(thr_mma.atom_partition_C(gC)));
        if (e.res) {
            auto R      = row_major(e.res, e.M, e.N, e.N);
            auto copy_r = make_block_2d_copy_C(mma, R);
            auto thr_r  = copy_r.get_slice(lid);
            auto tRrR   = thr_r.partition_sg_fragment_D(gC);
            copy(copy_r, thr_r.partition_S(gC), tRrR);
            reorder(tRrR, tCrR);
        }
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
            const auto c = tCgC(i);
            tCrO(i) = epi_value(e, tCrC(i), tCrR(i), cute::min(int(get<0>(c)), e.M - 1),
                                cute::min(int(get<1>(c)), e.N - 1));
        }
        copy(copy_c, tCrO, tCgC);
    }
}

// O (M,Nh) bf16 = gelu_tanh(g) * u, g = (acc_gate * xs[m]) * ws[n] (+ b), u likewise
// with the up rows n + Nh: both tiles of a column block in one work-group.
template <class Mma, int PF, class AT, class BT, class OT>
void gemm_glu_device(AT const & A, BT const & Bg, BT const & Bu, OT & O, const int8_t * blob, int row_bytes, int xs_off,
                     const float * ws, const float * bias, int M, int Nh) {
    Mma mma{};
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    const int wg_m = int(item.get_group(1)), wg_n = int(item.get_group(0)), lid = int(item.get_local_id(0));
    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(Bg.shape());
    Tensor cO = make_identity_tensor(O.shape());
    auto wg_tile = mma.tile_mnk();
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gO = local_tile(cO, wg_tile, make_coord(wg_m, wg_n, 0), Step<_1, _1, X>{});
    auto copy_a  = make_block_2d_copy_A(mma, A);
    auto copy_bg = make_block_2d_copy_B(mma, Bg);
    auto copy_bu = make_block_2d_copy_B(mma, Bu);
    auto copy_o  = make_block_2d_copy_D(mma, O);
    auto thr_mma = mma.get_slice(lid);
    auto thr_a  = copy_a.get_slice(lid);
    auto thr_bg = copy_bg.get_slice(lid);
    auto thr_bu = copy_bu.get_slice(lid);
    auto tCrA  = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrBg = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tCrBu = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tArA  = thr_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrBg = thr_bg.partition_sg_fragment_D(gB(_, _, 0));
    auto tBrBu = thr_bu.partition_sg_fragment_D(gB(_, _, 0));
    Tensor tAgA  = thr_a.partition_S(gA);
    Tensor tBgBg = thr_bg.partition_S(gB);
    Tensor tBgBu = thr_bu.partition_S(gB);
    Tensor accG = partition_fragment_C(mma, select<0, 1>(wg_tile));
    Tensor accU = partition_fragment_C(mma, select<0, 1>(wg_tile));
    Tensor tOgO = thr_mma.partition_C(gO);
    auto pf_a  = make_block_2d_prefetch(copy_a);
    auto pf_bg = make_block_2d_prefetch(copy_bg);
    auto pf_bu = make_block_2d_prefetch(copy_bu);
    auto tpa  = pf_a.get_slice(lid).partition_S(gA);
    auto tpbg = pf_bg.get_slice(lid).partition_S(gB);
    auto tpbu = pf_bu.get_slice(lid).partition_S(gB);
    const int k_tiles = ceil_div(shape<1>(A), get<2>(wg_tile));
    int kp = 0;
    clear(accG);
    clear(accU);
    CUTE_UNROLL
    for (; kp < PF; kp++) {
        prefetch(pf_a, tpa(_, _, _, kp));
        prefetch(pf_bg, tpbg(_, _, _, kp));
        prefetch(pf_bu, tpbu(_, _, _, kp));
    }
    for (int kt = 0; kt < k_tiles; kt++, kp++) {
        barrier_arrive(ScopeWorkgroup);
        copy(copy_a, tAgA(_, _, _, kt), tArA);
        copy(copy_bg, tBgBg(_, _, _, kt), tBrBg);
        copy(copy_bu, tBgBu(_, _, _, kt), tBrBu);
        prefetch(pf_a, tpa(_, _, _, kp));
        prefetch(pf_bg, tpbg(_, _, _, kp));
        prefetch(pf_bu, tpbu(_, _, _, kp));
        reorder(tArA, tCrA);
        reorder(tBrBg, tCrBg);
        reorder(tBrBu, tCrBu);
        gemm(mma, tCrA, tCrBg, accG);
        gemm(mma, tCrA, tCrBu, accU);
        barrier_wait(ScopeWorkgroup);
    }
    auto tOrO = make_fragment_like<bfloat16_t>(accG);
    CUTE_UNROLL
    for (int i = 0; i < size(accG); ++i) {
        const auto c = tOgO(i);
        const int m = cute::min(int(get<0>(c)), M - 1), n = cute::min(int(get<1>(c)), Nh - 1);
        const float xs = *reinterpret_cast<const float *>(blob + (size_t) m * row_bytes + xs_off);   // 4-aligned
        float g = ((float) accG(i) * xs) * ws[n];
        float u = ((float) accU(i) * xs) * ws[n + Nh];
        if (bias) { g = g + bias[n]; u = u + bias[n + Nh]; }
        tOrO(i) = bfloat16_t(gelu_tanh_dev(g) * u);
    }
    copy(copy_o, tOrO, tOgO);
}

template <class, class> class S4Gemm32Name;
template <class> class S4GemmGluName;

namespace syclex  = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

}  // namespace

namespace {
// Whether the device is a small Xe part (<= 64 vector engines, i.e. up to 8 Xe
// cores), which takes the *Xe4 tilings. VLA_S4_TILES=small|large overrides.
bool small_xe(sycl::queue & q) {
    static const int forced = [] {
        const char * e = std::getenv("VLA_S4_TILES");
        if (!e || !*e) return -1;
        return std::strcmp(e, "small") == 0 ? 1 : std::strcmp(e, "large") == 0 ? 0 : -1;
    }();
    if (forced >= 0) return forced == 1;
    return q.get_device().get_info<sycl::info::device::max_compute_units>() <= 64;
}

template <class Mma, int TM, int TN, int GRF, int PF, class OutT, class E, class Elt = int4_t>
void launch_s32(sycl::queue & q, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, const int8_t * w,
                int64_t N, OutT * out, const E & e) {
    constexpr int epb = std::is_same_v<Elt, int4_t> ? 2 : 1;   // elements per byte
    auto A = row_major(reinterpret_cast<Elt *>(const_cast<int8_t *>(blob)), (int) M, (int) K, (int) (row_bytes * epb));
    auto B = row_major(reinterpret_cast<Elt *>(const_cast<int8_t *>(w)), (int) N, (int) K, (int) K);
    auto C = row_major(out, (int) M, (int) N, (int) N);
    Mma mma{};
    const sycl::range<2> local = { (size_t) size(mma), 1 };
    const sycl::range<2> global = { local[0] * (size_t) ceil_div((int) N, TN), (size_t) ceil_div((int) M, TM) };
    syclex::properties props{ syclex::sub_group_size<16>, intelex::grf_size<GRF> };
    q.parallel_for<S4Gemm32Name<Mma, E>>(sycl::nd_range<2>(global, local), props, [=](auto) {
        auto Cm = C;
        gemm_s32_device<Mma, PF>(A, B, Cm, e);
    });
}

template <class OutT, class E>
bool launch_by_k(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, const int8_t * w,
                 int64_t N, OutT * out, const E & e) {
    if (K % 128 != 0 || N % 256 != 0 || (row_bytes % 16) != 0 || M <= 0) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    if (M <= 16 && !no_small_m()) launch_s32<MmaSmallM, 16, 64, 256, 3>(q, blob, M, K, row_bytes, w, N, out, e);
    else if (K >= 8192 && small_xe(q)) {
        // With the epilogue this tile spills ~44 registers (GRF 256): decline,
        // and the caller runs the s32 sums and the separate epilogue, same bits.
        if constexpr (std::is_same_v<E, NoEpi>) launch_s32<MmaLongKXe4, 192, 128, 256, 2>(q, blob, M, K, row_bytes, w, N, out, e);
        else return false;
    }
    else if (K >= 8192) launch_s32<MmaLongK, 128, 128, 256, 3>(q, blob, M, K, row_bytes, w, N, out, e);
    else           launch_s32<MmaShortK, 64, 256, 256, 3>(q, blob, M, K, row_bytes, w, N, out, e);
    return true;
}
}  // namespace

bool sycl_s4_gemm_s32(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, const int8_t * w,
                      int64_t N, int32_t * out) {
    return launch_by_k(queue, blob, M, K, row_bytes, w, N, out, NoEpi{});
}

bool sycl_s4_gemm_f32(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, int64_t xs_off,
                      const int8_t * w, int64_t N, const float * ws, const float * bias, const float * res,
                      const float * gate, float * out) {
    if ((xs_off % 4) != 0) return false;
    const Epi e{ blob, (int) row_bytes, (int) xs_off, (int) M, (int) N, ws, bias, res, gate };
    return launch_by_k(queue, blob, M, K, row_bytes, w, N, out, e);
}

namespace {
template <class Mma, int TM, int TN, int GRF, int PF>
void launch_glu(sycl::queue & q, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, int64_t xs_off,
                const int8_t * w, const float * ws, const float * bias, int64_t Nh, uint16_t * out_bf16) {
    auto A  = row_major(reinterpret_cast<int4_t *>(const_cast<int8_t *>(blob)), (int) M, (int) K, (int) (row_bytes * 2));
    auto Bg = row_major(reinterpret_cast<int4_t *>(const_cast<int8_t *>(w)), (int) Nh, (int) K, (int) K);
    auto Bu = row_major(reinterpret_cast<int4_t *>(const_cast<int8_t *>(w) + Nh * K / 2), (int) Nh, (int) K, (int) K);
    auto O  = row_major(reinterpret_cast<bfloat16_t *>(out_bf16), (int) M, (int) Nh, (int) Nh);
    Mma mma{};
    const sycl::range<2> local = { (size_t) size(mma), 1 };
    const sycl::range<2> global = { local[0] * (size_t) ceil_div((int) Nh, TN), (size_t) ceil_div((int) M, TM) };
    syclex::properties props{ syclex::sub_group_size<16>, intelex::grf_size<GRF> };
    const int iM = (int) M, iNh = (int) Nh, rb = (int) row_bytes, xo = (int) xs_off;
    q.parallel_for<S4GemmGluName<Mma>>(sycl::nd_range<2>(global, local), props, [=](auto) {
        auto Om = O;
        gemm_glu_device<Mma, PF>(A, Bg, Bu, Om, blob, rb, xo, ws, bias, iM, iNh);
    });
}
}  // namespace

bool sycl_s8_gemm_f32(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, int64_t xs_off,
                      const int8_t * w, int64_t N, const float * ws, const float * bias, const float * res,
                      const float * gate, float * out) {
    if (M <= 0 || M > 16 || K % 128 != 0 || N % 64 != 0 || (row_bytes % 16) != 0 || (xs_off % 4) != 0) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    const Epi e{ blob, (int) row_bytes, (int) xs_off, (int) M, (int) N, ws, bias, res, gate };
    launch_s32<MmaSmallM8, 16, 64, 256, 3, float, Epi, int8_t>(q, blob, M, K, row_bytes, w, N, out, e);
    return true;
}

bool sycl_s4_gemm_glu(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, int64_t xs_off,
                      const int8_t * w, const float * ws, const float * bias, int64_t Nh, uint16_t * out_bf16) {
    if (K % 128 != 0 || (row_bytes % 16) != 0 || (xs_off % 4) != 0 || M <= 0) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    const bool small_m = M <= 16 && !no_small_m();
    if (!small_m && small_xe(q) && Nh % GLU_N_XE4 == 0) {
        launch_glu<MmaHalfXe4, GLU_M_XE4, GLU_N_XE4, 128, 2>(q, blob, M, K, row_bytes, xs_off, w, ws, bias, Nh, out_bf16);
        return true;
    }
    if (Nh % GLU_N != 0) return false;
    if (small_m) launch_glu<MmaSmallM, 16, 64, 256, 3>(q, blob, M, K, row_bytes, xs_off, w, ws, bias, Nh, out_bf16);
    else         launch_glu<MmaHalf, GLU_M, GLU_N, 256, 3>(q, blob, M, K, row_bytes, xs_off, w, ws, bias, Nh, out_bf16);
    return true;
}

}  // namespace vla
