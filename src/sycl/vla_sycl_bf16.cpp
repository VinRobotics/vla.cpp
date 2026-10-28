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

// BF16 activation kernels for the SYCL backend, the twin of src/cuda/vla_cuda_bf16.cu.
//
// ggml-sycl already runs ADD, MUL and UNARY on BF16, so those are left to it.
// This file covers what it cannot: NORM, RMS_NORM and SCALE assert F32, and its
// BF16 MUL_MAT always writes an f32 dst. It is reached through the extension hook
// (scripts/patch_ggml_sycl_ext_hook.py) and uses only the public ggml header.
// Every entry point declines what it does not handle, so the F32 path is
// unchanged. Accumulation is f32; each op follows the CUDA formula and order of
// operations, so both backends stay in the same tolerance band
// (tests/test_bf16_sycl_ops.cpp).

#include "ggml.h"
#include "env_flag.h"
#include "sycl/vla_sycl_ext.h"

#include <sycl/sycl.hpp>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>
#include <unordered_map>
#include <mutex>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

// One work-group per row for the norms; also the flat block for scale. Matches
// the CUDA BLOCK so both backends split a row the same number of ways.
constexpr int WG = 256;

// VLA_SYCL_BF16_OFF=1 makes every entry point decline, which hands the graph
// back to stock ggml-sycl. NORM/RMS_NORM/SCALE then abort on their F32 asserts,
// so this is a bisecting tool for the MUL_MAT path, not a supported mode.
bool disabled() {
    static const bool v = vla::env_flag("VLA_SYCL_BF16_OFF");
    return v;
}

// ---------------------------------------------------------------------------
// oneDNN engine and primitive cache, bound to ggml-sycl's own queue: a
// dnnl::stream must wrap the queue its work is ordered against.
// ---------------------------------------------------------------------------

struct Ctx {
    dnnl::engine engine;
    dnnl::stream stream;

    explicit Ctx(sycl::queue & q) :
        engine(dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())),
        stream(dnnl::sycl_interop::make_stream(engine, q)) {}
};

// Keyed by queue so a second queue never reuses another queue's stream.
Ctx & ctx_for(sycl::queue & q) {
    static std::mutex                    mu;
    static std::map<sycl::queue *, Ctx*> cache;

    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(&q);
    if (it == cache.end()) {
        it = cache.emplace(&q, new Ctx(q)).first;
    }
    return *it->second;
}

using dt   = dnnl::memory::data_type;
using ddims = dnnl::memory::dims;

struct Key {
    long long M, N, K, batch, wbatch;
    bool      f32_out;

    bool operator<(const Key & o) const {
        if (f32_out != o.f32_out) return f32_out < o.f32_out;
        if (M != o.M) return M < o.M;
        if (N != o.N) return N < o.N;
        if (K != o.K) return K < o.K;
        if (batch != o.batch) return batch < o.batch;
        return wbatch < o.wbatch;
    }
};

struct Plan {
    dnnl::memory::desc a_md, b_md, c_md;
    dnnl::matmul       prim;
};

// ---------------------------------------------------------------------------
// mul_mat: BF16 x BF16 -> BF16
// ---------------------------------------------------------------------------
//
// ggml-sycl's bf16 path always writes f32. For a BF16 result (vla::mul_mat_t)
// with BF16 operands the bytes go to oneDNN as they are. src0 is (N, K)
// row-major, so describing it as {K, N} with strides {1, K} gives B^T without
// a transpose. Accumulation is f32, as in the CUDA twin.

bool mul_mat(ggml_tensor * dst, sycl::queue & q) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!src0 || !src1) return false;
    // BF16 x BF16 -> BF16, or -> F32 for a result that feeds F32 work (RoPE).
    if ((dst->type != GGML_TYPE_BF16 && dst->type != GGML_TYPE_F32) || src0->type != GGML_TYPE_BF16 ||
        src1->type != GGML_TYPE_BF16) {
        return false;
    }
    const bool f32_out = dst->type == GGML_TYPE_F32;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // src0 is either shared across the whole batch or batched 1:1 with src1
    const bool bcast = src0->ne[2] == 1 && src0->ne[3] == 1;
    const bool batch_ok = bcast || (src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]);
    if (!batch_ok) return false;

    const long long K     = src1->ne[0];
    const long long M     = src1->ne[1];
    const long long N     = src0->ne[1];
    const long long batch = src1->ne[2]*src1->ne[3];
    if (src0->ne[0] != K) return false;
    if (dst->ne[0] != N || dst->ne[1] != M) return false;

    // oneDNN broadcasts a batch through an extent of 1, not a zero stride.
    const long long wbatch = bcast ? 1 : batch;

    const Key key{ M, N, K, batch, wbatch, f32_out };

    static std::mutex          mu;
    static std::map<Key, Plan> cache;

    try {
        Ctx & c = ctx_for(q);

        const dnnl::memory::desc a_md({ batch, M, K }, dt::bf16, ddims{ M*K, K, 1 });
        const dnnl::memory::desc b_md({ wbatch, K, N }, dt::bf16, ddims{ N*K, 1, K });
        const dnnl::memory::desc c_md({ batch, M, N }, f32_out ? dt::f32 : dt::bf16, ddims{ M*N, N, 1 });

        const Plan * plan = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = cache.find(key);
            if (it == cache.end()) {
                dnnl::matmul::primitive_desc pd(c.engine, a_md, b_md, c_md);
                it = cache.emplace(key, Plan{ a_md, b_md, c_md, dnnl::matmul(pd) }).first;
            }
            plan = &it->second;
        }

        // oneDNN's handles are non-const; the primitive only reads A and B.
        dnnl::memory a_mem(plan->a_md, c.engine, src1->data);
        dnnl::memory b_mem(plan->b_md, c.engine, src0->data);
        dnnl::memory c_mem(plan->c_md, c.engine, dst->data);

        plan->prim.execute(c.stream,
                           { { DNNL_ARG_SRC, a_mem }, { DNNL_ARG_WEIGHTS, b_mem }, { DNNL_ARG_DST, c_mem } });
        return true;
    } catch (const dnnl::error & e) {
        // Declining would let ggml-sycl write f32 into a bf16 dst. Report the
        // real cause before the assert that follows.
        std::fprintf(stderr, "vla(sycl_bf16): matmul %lldx%lldx%lld (batch %lld) failed: %s\n",
                     M, N, K, batch, e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// fused linear (layers/fused_ew.h): act(x W^T + b) (+ res) as one matmul with
// its bias and post-ops
// ---------------------------------------------------------------------------

struct LinKey {
    long long M, N, K, x_ld, res_ld;
    int       bias, gelu, res_type, dst_type;
    auto      tie() const { return std::tie(M, N, K, x_ld, res_ld, bias, gelu, res_type, dst_type); }
    bool      operator<(const LinKey & o) const { return tie() < o.tie(); }
};

struct LinPlan {
    dnnl::memory::desc a_md, b_md, bias_md, c_md, r_md;
    dnnl::matmul       prim;
};

dt dnnl_type(int t) {
    return t == GGML_TYPE_F32 ? dt::f32 : t == GGML_TYPE_F16 ? dt::f16 : dt::bf16;
}

bool fused_linear(ggml_tensor * dst, const void * spec_v, sycl::queue & q) {
    const ggml_tensor * W   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * b   = dst->src[2];
    const ggml_tensor * res = dst->src[3];
    int32_t gelu = 0;
    std::memcpy(&gelu, (const char *) spec_v + sizeof(uint32_t), sizeof(gelu));

    const long long K = x->ne[0], M = x->ne[1], N = W->ne[1];
    const LinKey key{ M, N, K, (long long) (x->nb[1] / ggml_type_size(x->type)),
                      res ? (long long) (res->nb[1] / ggml_type_size(res->type)) : 0, b != nullptr, gelu,
                      res ? (int) res->type : -1, (int) dst->type };

    static std::mutex             mu;
    static std::map<LinKey, LinPlan> cache;
    try {
        Ctx & c = ctx_for(q);
        const LinPlan * plan = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = cache.find(key);
            if (it == cache.end()) {
                LinPlan p;
                p.a_md = dnnl::memory::desc({ M, K }, dt::bf16, ddims{ key.x_ld, 1 });
                p.b_md = dnnl::memory::desc({ K, N }, dt::bf16, ddims{ 1, K });
                p.c_md = dnnl::memory::desc({ M, N }, dnnl_type(dst->type), ddims{ N, 1 });
                if (b) p.bias_md = dnnl::memory::desc({ 1, N }, dt::f32, ddims{ N, 1 });
                dnnl::post_ops po;
                if (gelu) po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.0f, 0.0f);
                if (res) {
                    p.r_md = dnnl::memory::desc({ M, N }, dnnl_type(res->type), ddims{ key.res_ld, 1 });
                    po.append_binary(dnnl::algorithm::binary_add, p.r_md);
                }
                dnnl::primitive_attr attr;
                attr.set_post_ops(po);
                dnnl::matmul::primitive_desc pd = b ? dnnl::matmul::primitive_desc(c.engine, p.a_md, p.b_md, p.bias_md,
                                                                                 p.c_md, attr)
                                                    : dnnl::matmul::primitive_desc(c.engine, p.a_md, p.b_md, p.c_md, attr);
                p.prim = dnnl::matmul(pd);
                it = cache.emplace(key, p).first;
            }
            plan = &it->second;
        }
        std::unordered_map<int, dnnl::memory> args{
            { DNNL_ARG_SRC, dnnl::memory(plan->a_md, c.engine, x->data) },
            { DNNL_ARG_WEIGHTS, dnnl::memory(plan->b_md, c.engine, W->data) },
            { DNNL_ARG_DST, dnnl::memory(plan->c_md, c.engine, dst->data) } };
        if (b) args.insert({ DNNL_ARG_BIAS, dnnl::memory(plan->bias_md, c.engine, b->data) });
        if (res)
            args.insert({ DNNL_ARG_ATTR_MULTIPLE_POST_OP(gelu ? 1 : 0) | DNNL_ARG_SRC_1,
                          dnnl::memory(plan->r_md, c.engine, res->data) });
        plan->prim.execute(c.stream, args);
        return true;
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(sycl_bf16): fused linear %lldx%lldx%lld failed: %s\n", M, N, K, e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// W8A8 linear (layers/qlinear.h): per-token INT8 activations, one INT8 matmul
// with both scales, the bias and the post-ops
// ---------------------------------------------------------------------------

template <typename T> class k_quant_rows;
template <typename T> class k_quant_rows8;

// One work-group per row: max |x| -> scale, then the INT8 codes. Rows of a
// multiple of 8 with 8-element alignment go 8 at a time (16/32-byte loads).
template <typename T>
void quant_rows(sycl::queue & q, const ggml_tensor * x, int8_t * qx, float * sx) {
    const int64_t K = x->ne[0], M = x->ne[1];
    const size_t  xs = x->nb[1];
    const char *  xb = (const char *) x->data;
    const sycl::nd_range<1> nr(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG));
    if (K % 8 == 0 && (uintptr_t) xb % (8 * sizeof(T)) == 0 && xs % (8 * sizeof(T)) == 0) {
        using V = sycl::vec<T, 8>;
        q.parallel_for<k_quant_rows8<T>>(nr, [=](sycl::nd_item<1> it) {
            const int64_t m   = (int64_t) it.get_group(0);
            const int64_t lid = (int64_t) it.get_local_id(0);
            const V *     xr  = (const V *) (xb + m * xs);
            float amax = 0.0f;
            for (int64_t c = lid; c < K / 8; c += WG) {
                const V v = xr[c];
#pragma unroll
                for (int j = 0; j < 8; ++j) amax = sycl::fmax(amax, sycl::fabs((float) v[j]));
            }
            amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            const float sc  = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float inv = 1.0f / sc;
            auto *      qr  = (sycl::vec<int8_t, 8> *) (qx + m * K);
            for (int64_t c = lid; c < K / 8; c += WG) {
                const V v = xr[c];
                sycl::vec<int8_t, 8> o;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    o[j] = (int8_t) sycl::rint(sycl::fmin(127.0f, sycl::fmax(-127.0f, (float) v[j] * inv)));
                qr[c] = o;
            }
            if (lid == 0) sx[m] = sc;
        });
        return;
    }
    q.parallel_for<k_quant_rows<T>>(nr, [=](sycl::nd_item<1> it) {
        const int64_t m   = (int64_t) it.get_group(0);
        const int64_t lid = (int64_t) it.get_local_id(0);
        const T *     xr  = (const T *) (xb + m * xs);
        float amax = 0.0f;
        for (int64_t k = lid; k < K; k += WG) amax = sycl::fmax(amax, sycl::fabs((float) xr[k]));
        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        const float sc  = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / sc;
        int8_t *    qr  = qx + m * K;
        for (int64_t k = lid; k < K; k += WG)
            qr[k] = (int8_t) sycl::rint(sycl::fmin(127.0f, sycl::fmax(-127.0f, (float) xr[k] * inv)));
        if (lid == 0) sx[m] = sc;
    });
}

// Grown, never shrunk; the queue is in order, so one buffer serves every call.
void * q8_scratch(sycl::queue & q, size_t bytes) {
    static std::mutex mu;
    static void *     buf = nullptr;
    static size_t     cap = 0;
    std::lock_guard<std::mutex> lock(mu);
    if (bytes > cap) {
        if (buf) {
            q.wait();
            sycl::free(buf, q);
        }
        buf = sycl::malloc_device(bytes, q);
        cap = bytes;
    }
    return buf;
}

struct QKey {
    long long M, N, K, res_ld;
    int       bias, gelu, res_type, dst_type, gate, bits;
    long long group;
    auto      tie() const { return std::tie(M, N, K, res_ld, bias, gelu, res_type, dst_type, gate, bits, group); }
    bool      operator<(const QKey & o) const { return tie() < o.tie(); }
};

struct QPlan {
    dnnl::memory::desc a_md, b_md, bias_md, c_md, r_md, sa_md, sw_md, g_md;
    dnnl::matmul       prim;
    int                gate_idx = -1, res_idx = -1;  // post-op positions
};

bool qlinear(ggml_tensor * dst, const void * spec_v, sycl::queue & q) {
    const ggml_tensor * W   = dst->src[0];
    const ggml_tensor * sw  = dst->src[1];
    const ggml_tensor * x   = dst->src[2];
    const ggml_tensor * b   = dst->src[3];
    const ggml_tensor * res = dst->src[4];
    const ggml_tensor * gt  = dst->src[5];
    int32_t sp[3] = { 0, 8, 0 };  // QLinearSpec after the magic: gelu, bits, group
    std::memcpy(sp, (const char *) spec_v + sizeof(uint32_t), sizeof(sp));
    const int32_t   gelu = sp[0], bits = sp[1];
    const long long K = bits == 4 ? 2 * W->ne[0] : W->ne[0], N = W->ne[1];
    const long long G = bits == 4 ? sp[2] : K;
    const bool      blob = x->type == GGML_TYPE_I8;
    const long long M = blob ? x->ne[0] / (K + 4) : x->ne[1];

    int8_t * qx;
    float *  sx;
    if (blob) {
        // Quantized by its producer: M rows of codes, then the M scales.
        qx = (int8_t *) x->data;
        sx = (float *) ((int8_t *) x->data + M * K);
    } else {
        const size_t qbytes = ((size_t) (M * K) + 255) / 256 * 256;
        char *       scr    = (char *) q8_scratch(q, qbytes + (size_t) M * sizeof(float));
        qx = (int8_t *) scr;
        sx = (float *) (scr + qbytes);
        if (x->type == GGML_TYPE_BF16) quant_rows<bf16>(q, x, qx, sx);
        else                           quant_rows<float>(q, x, qx, sx);
    }

    const QKey key{ M, N, K, res ? (long long) (res->nb[1] / ggml_type_size(res->type)) : 0, b != nullptr, gelu,
                    res ? (int) res->type : -1, (int) dst->type, gt != nullptr, bits, G };
    static std::mutex            mu;
    static std::map<QKey, QPlan> cache;
    try {
        Ctx & c = ctx_for(q);
        const QPlan * plan = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = cache.find(key);
            if (it == cache.end()) {
                QPlan p;
                p.a_md  = dnnl::memory::desc({ M, K }, dt::s8, ddims{ K, 1 });
                p.b_md  = dnnl::memory::desc({ K, N }, bits == 4 ? dt::s4 : dt::s8, ddims{ 1, K });
                p.c_md  = dnnl::memory::desc({ M, N }, dnnl_type(dst->type), ddims{ N, 1 });
                p.sa_md = dnnl::memory::desc({ M }, dt::f32, ddims{ 1 });
                p.sw_md = dnnl::memory::desc({ (K / G) * N }, dt::f32, ddims{ 1 });
                if (b) p.bias_md = dnnl::memory::desc({ 1, N }, dt::f32, ddims{ N, 1 });
                dnnl::primitive_attr attr;
                // Per-token src scales (one group spanning K), per-channel weight scales.
                attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1), { 1, K }, dt::f32);
                if (G < K) attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { G, 1 }, dt::f32);
                else       attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
                dnnl::post_ops po;
                int            idx = 0;
                if (gelu) {
                    po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.0f, 0.0f);
                    ++idx;
                }
                if (gt) {
                    p.g_md = dnnl::memory::desc({ 1, N }, dt::f32, ddims{ N, 1 });
                    po.append_binary(dnnl::algorithm::binary_mul, p.g_md);
                    p.gate_idx = idx++;
                }
                if (res) {
                    p.r_md = dnnl::memory::desc({ M, N }, dnnl_type(res->type), ddims{ key.res_ld, 1 });
                    po.append_binary(dnnl::algorithm::binary_add, p.r_md);
                    p.res_idx = idx++;
                }
                attr.set_post_ops(po);
                dnnl::matmul::primitive_desc pd = b ? dnnl::matmul::primitive_desc(c.engine, p.a_md, p.b_md, p.bias_md,
                                                                                 p.c_md, attr)
                                                    : dnnl::matmul::primitive_desc(c.engine, p.a_md, p.b_md, p.c_md, attr);
                p.prim = dnnl::matmul(pd);
                it = cache.emplace(key, p).first;
            }
            plan = &it->second;
        }
        std::unordered_map<int, dnnl::memory> args{
            { DNNL_ARG_SRC, dnnl::memory(plan->a_md, c.engine, qx) },
            { DNNL_ARG_WEIGHTS, dnnl::memory(plan->b_md, c.engine, W->data) },
            { DNNL_ARG_DST, dnnl::memory(plan->c_md, c.engine, dst->data) },
            { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(plan->sa_md, c.engine, sx) },
            { DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(plan->sw_md, c.engine, sw->data) } };
        if (b) args.insert({ DNNL_ARG_BIAS, dnnl::memory(plan->bias_md, c.engine, b->data) });
        if (gt)
            args.insert({ DNNL_ARG_ATTR_MULTIPLE_POST_OP(plan->gate_idx) | DNNL_ARG_SRC_1,
                          dnnl::memory(plan->g_md, c.engine, gt->data) });
        if (res)
            args.insert({ DNNL_ARG_ATTR_MULTIPLE_POST_OP(plan->res_idx) | DNNL_ARG_SRC_1,
                          dnnl::memory(plan->r_md, c.engine, res->data) });
        plan->prim.execute(c.stream, args);
        return true;
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(sycl_bf16): W8A8 linear %lldx%lldx%lld failed: %s\n", M, N, K, e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// scale: dst = x*scale + bias
// ---------------------------------------------------------------------------

bool scale(ggml_tensor * dst, sycl::queue & q) {
    const ggml_tensor * src0 = dst->src[0];
    if (!src0) return false;
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;

    float s = 1.0f, b = 0.0f;
    std::memcpy(&s, (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&b, (const float *) dst->op_params + 1, sizeof(float));

    const int64_t n = ggml_nelements(dst);
    if (n <= 0) return true;

    const bf16 * x = (const bf16 *) src0->data;
    bf16 *       d = (bf16 *) dst->data;

    const size_t global = (size_t) ((n + WG - 1)/WG)*WG;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
                   [=](sycl::nd_item<1> it) {
                       const int64_t i = (int64_t) it.get_global_id(0);
                       if (i < n) d[i] = bf16(s*(float) x[i] + b);
                   });
    return true;
}

// ---------------------------------------------------------------------------
// norm / rms_norm: one work-group per row, float reduction
// ---------------------------------------------------------------------------
//
// Not bit-identical to CUDA (the reduction trees differ in shape), but the same
// formulas: rms uses rsqrt(mean(x^2) + eps), norm rsqrt(E[x^2] - E[x]^2 + eps).
// The latter is less stable than ggml's CPU (x - mean)^2, but it keeps the two
// GPU backends in agreement.

template <bool rms> class k_norm_bf16;

template <bool rms>
bool norm(ggml_tensor * dst, sycl::queue & q) {
    const ggml_tensor * src0 = dst->src[0];
    if (!src0) return false;
    if (dst->type != GGML_TYPE_BF16 || src0->type != GGML_TYPE_BF16) return false;
    // Rows must be dense; higher dims are handled by flattening into the row index.
    if (src0->nb[0] != ggml_type_size(src0->type)) return false;
    if (dst->nb[0]  != ggml_type_size(dst->type))  return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ncols > 0 ? ggml_nelements(src0)/ncols : 0;
    if (ncols <= 0 || nrows <= 0) return true;

    const bf16 * x = (const bf16 *) src0->data;
    bf16 *       d = (bf16 *) dst->data;

    q.parallel_for<k_norm_bf16<rms>>(
        sycl::nd_range<1>(sycl::range<1>((size_t) nrows*WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            const int64_t row = (int64_t) it.get_group(0);
            const int64_t lid = (int64_t) it.get_local_id(0);

            const bf16 * xr = x + row*ncols;
            bf16 *       dr = d + row*ncols;

            float sum = 0.0f, sumsq = 0.0f;
            for (int64_t c = lid; c < ncols; c += WG) {
                const float v = (float) xr[c];
                sumsq += v*v;
                if (!rms) sum += v;
            }

            auto g = it.get_group();
            if (rms) {
                const float ms  = sycl::reduce_over_group(g, sumsq, sycl::plus<float>())/(float) ncols;
                const float inv = sycl::rsqrt(ms + eps);
                for (int64_t c = lid; c < ncols; c += WG) dr[c] = bf16((float) xr[c]*inv);
            } else {
                // Two reductions, both reached by every work-item in the group:
                // the branch is on a template parameter, so it is uniform.
                const float mean   = sycl::reduce_over_group(g, sum,   sycl::plus<float>())/(float) ncols;
                const float meansq = sycl::reduce_over_group(g, sumsq, sycl::plus<float>())/(float) ncols;
                const float inv    = sycl::rsqrt(meansq - mean*mean + eps);
                for (int64_t c = lid; c < ncols; c += WG) dr[c] = bf16(((float) xr[c] - mean)*inv);
            }
        });
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// hook entry point
// ---------------------------------------------------------------------------

namespace vla {
// For vla_sycl_fused.cpp's fused linear node: the matmul and its epilogue.
bool sycl_fused_linear(ggml_tensor * dst, const void * spec, sycl::queue & q) {
    return fused_linear(dst, spec, q);
}
bool sycl_qlinear(ggml_tensor * dst, const void * spec, sycl::queue & q) {
    return qlinear(dst, spec, q);
}
bool sycl_flash_attn_ext(ggml_tensor * dst, sycl::queue & q);  // vla_sycl_attn.cpp
bool sycl_flash_attn_ext_supported(const ggml_tensor * dst);   // vla_sycl_attn.cpp
}

extern "C" bool vla_sycl_bf16_forward(ggml_tensor * dst, void * stream_v) {
    if (!dst || !stream_v || disabled()) return false;

    sycl::queue & q = *static_cast<sycl::queue *>(stream_v);

    // Flash attention is claimed for every activation dtype (vla_sycl_attn.cpp).
    if (dst->op == GGML_OP_FLASH_ATTN_EXT) return vla::sycl_flash_attn_ext(dst, q);

    // A BF16 x BF16 GEMM with an F32 result (vla::mm_f32) is ours as well.
    if (dst->op == GGML_OP_MUL_MAT && dst->type == GGML_TYPE_F32) return mul_mat(dst, q);

    // Otherwise only BF16 results can be ours.
    if (dst->type != GGML_TYPE_BF16) return false;

    switch (dst->op) {
        case GGML_OP_MUL_MAT:  return mul_mat(dst, q);
        case GGML_OP_SCALE:    return scale(dst, q);
        case GGML_OP_NORM:     return norm<false>(dst, q);
        case GGML_OP_RMS_NORM: return norm<true>(dst, q);
        // ADD, MUL and UNARY are absent: ggml-sycl handles BF16 for those.
        default: return false;
    }
}

// supports_op half of the hook: flash attention only. ggml-sycl already admits
// the BF16 ops above, but (b11223+) refuses FLASH_ATTN_EXT shapes this
// extension runs.
static bool vla_sycl_bf16_supports(const ggml_tensor * op) {
    return !disabled() && vla::sycl_flash_attn_ext_supported(op);
}

namespace vla {

// Called once, after the SYCL backend is up. Idempotent.
void sycl_register_bf16_ops() {
    sycl_ext_register(vla_sycl_bf16_forward, vla_sycl_bf16_supports);
}

}  // namespace vla
