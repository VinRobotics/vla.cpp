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
// on sycl-tla's examples/cute/tutorial/xe_gemm.cpp main loop.
//
// A: nibble-packed activation blob rows (FoldQuant act layout: K/2 code bytes,
//    then the float row scale), pitch = blob row bytes.
// B: nibble-packed weight rows [N][K/2] (low nibble = even column).

#include "sycl/vla_sycl_s4gemm.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cute/tensor.hpp>

#include <cstring>

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
// The GEGLU kernel: two accumulators (gate, up) per work-group; 64 x 256 measured
// best among 5 tilings on the Arc B390 (561 x 2 x 16384 x 2048: 0.84 ms).
using SGL2x8  = Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>;
using MmaHalf = typename TiledMMAHelper<MMA_Atom<MmaOp>, Layout<Shape<_64, _256, _128>>, SGL2x8>::TiledMMA;
constexpr int GLU_M = 64, GLU_N = 256;

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

// C (M,N) s32 = A (M,K) s4 x B (N,K)^T s4.
template <class Mma, class AT, class BT, class CT>
void gemm_s32_device(AT const & A, BT const & B, CT & C) {
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
    for (; kp < 3; kp++) { prefetch(pf_a, tpa(_, _, _, kp)); prefetch(pf_b, tpb(_, _, _, kp)); }
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
    copy(copy_c, tCrC, tCgC);
}

// O (M,Nh) bf16 = gelu_tanh(g) * u, g = (acc_gate * xs[m]) * ws[n] (+ b), u likewise
// with the up rows n + Nh: both tiles of a column block in one work-group.
template <class AT, class BT, class OT>
void gemm_glu_device(AT const & A, BT const & Bg, BT const & Bu, OT & O, const int8_t * blob, int row_bytes, int xs_off,
                     const float * ws, const float * bias, int M, int Nh) {
    MmaHalf mma{};
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
    for (; kp < 3; kp++) {
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

template <class> class S4Gemm32Name;

template <class Mma, int TM, int TN>
void launch_s32(sycl::queue & q, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, const int8_t * w,
                int64_t N, int32_t * out);
template <class> class S4GemmGluName;

namespace syclex  = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

}  // namespace

namespace {
template <class Mma, int TM, int TN>
void launch_s32(sycl::queue & q, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, const int8_t * w,
                int64_t N, int32_t * out) {
    auto A = row_major(reinterpret_cast<int4_t *>(const_cast<int8_t *>(blob)), (int) M, (int) K, (int) (row_bytes * 2));
    auto B = row_major(reinterpret_cast<int4_t *>(const_cast<int8_t *>(w)), (int) N, (int) K, (int) K);
    auto C = row_major(out, (int) M, (int) N, (int) N);
    Mma mma{};
    const sycl::range<2> local = { (size_t) size(mma), 1 };
    const sycl::range<2> global = { local[0] * (size_t) ceil_div((int) N, TN), (size_t) ceil_div((int) M, TM) };
    syclex::properties props{ syclex::sub_group_size<16>, intelex::grf_size<256> };
    q.parallel_for<S4Gemm32Name<Mma>>(sycl::nd_range<2>(global, local), props, [=](auto) {
        auto Cm = C;
        gemm_s32_device<Mma>(A, B, Cm);
    });
}
}  // namespace

bool sycl_s4_gemm_s32(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, const int8_t * w,
                      int64_t N, int32_t * out) {
    if (K % 128 != 0 || N % 256 != 0 || (row_bytes % 16) != 0 || M <= 0) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    if (K >= 8192) launch_s32<MmaLongK, 128, 128>(q, blob, M, K, row_bytes, w, N, out);
    else           launch_s32<MmaShortK, 64, 256>(q, blob, M, K, row_bytes, w, N, out);
    return true;
}

bool sycl_s4_gemm_glu(void * queue, const int8_t * blob, int64_t M, int64_t K, int64_t row_bytes, int64_t xs_off,
                      const int8_t * w, const float * ws, const float * bias, int64_t Nh, uint16_t * out_bf16) {
    if (K % 128 != 0 || Nh % GLU_N != 0 || (row_bytes % 16) != 0 || (xs_off % 4) != 0 || M <= 0) return false;
    sycl::queue & q = *static_cast<sycl::queue *>(queue);
    auto A  = row_major(reinterpret_cast<int4_t *>(const_cast<int8_t *>(blob)), (int) M, (int) K, (int) (row_bytes * 2));
    auto Bg = row_major(reinterpret_cast<int4_t *>(const_cast<int8_t *>(w)), (int) Nh, (int) K, (int) K);
    auto Bu = row_major(reinterpret_cast<int4_t *>(const_cast<int8_t *>(w) + Nh * K / 2), (int) Nh, (int) K, (int) K);
    auto O  = row_major(reinterpret_cast<bfloat16_t *>(out_bf16), (int) M, (int) Nh, (int) Nh);
    MmaHalf mma{};
    const sycl::range<2> local = { (size_t) size(mma), 1 };
    const sycl::range<2> global = { local[0] * (size_t) ceil_div((int) Nh, GLU_N), (size_t) ceil_div((int) M, GLU_M) };
    syclex::properties props{ syclex::sub_group_size<16>, intelex::grf_size<256> };
    const int iM = (int) M, iNh = (int) Nh, rb = (int) row_bytes, xo = (int) xs_off;
    q.parallel_for<S4GemmGluName<void>>(sycl::nd_range<2>(global, local), props, [=](auto) {
        auto Om = O;
        gemm_glu_device(A, Bg, Bu, Om, blob, rb, xo, ws, bias, iM, iNh);
    });
    return true;
}

}  // namespace vla
