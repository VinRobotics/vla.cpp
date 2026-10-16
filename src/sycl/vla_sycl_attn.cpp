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

// GGML_OP_FLASH_ATTN_EXT on oneDNN Graph's fused SDPA, through the same
// ggml-sycl forward hook as vla_sycl_bf16.cpp. oneDNN's SDPA micro-kernel does
// QK^T, softmax and PV in one pass with the scores on chip, where ggml-sycl's
// own FA kernel and the unfused graph both round-trip the score plane.
//
// Semantics, ggml's (ggml.h, ggml_flash_attn_ext):
//   q    [D, S,    H,    1]   any of F32/F16/BF16, arbitrary strides
//   k, v [D, S_kv, H_kv, 1]   H % H_kv == 0 (grouped-query)
//   mask [S_kv, S_pad, 1, 1]  F16, additive, rows >= S ignored; or null
//   dst  [D, H, S, 1]         F32 or BF16, contiguous (note H and S swap)
//   op_params: scale, max_bias, logit_softcap
//
// Precision: Q, K and V are converted to F16 on the way in, as ggml's own FA
// kernels do with K/V; scores, softmax and accumulation are F32. F16 over BF16
// for its three extra mantissa bits; post-norm activations fit its range.
//
// Declines (returns false, ggml runs its own kernel) for ALiBi (max_bias != 0),
// logit soft-capping, batch > 1 unless every operand is BF16 read in place, or
// a pattern oneDNN will not fuse.

#include "ggml.h"
#include "env_flag.h"

#include <sycl/sycl.hpp>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

namespace {

using lt   = dnnl::graph::logical_tensor;
using gdt  = lt::data_type;
using half = sycl::half;

void * graph_malloc(size_t size, size_t alignment, const void * dev, const void * ctx) {
    return sycl::aligned_alloc_device(alignment, size, *static_cast<const sycl::device *>(dev),
                                      *static_cast<const sycl::context *>(ctx));
}

void graph_free(void * ptr, const void *, const void * ctx, void * event) {
    if (event) static_cast<sycl::event *>(event)->wait();
    sycl::free(ptr, *static_cast<const sycl::context *>(ctx));
}

/// Engine + stream on ggml-sycl's queue, with a graph allocator for the SDPA
/// partitions' scratch. One per queue.
struct Ctx {
    dnnl::engine engine;
    dnnl::stream stream;
    explicit Ctx(sycl::queue & q) :
        engine(dnnl::graph::sycl_interop::make_engine_with_allocator(
            q.get_device(), q.get_context(),
            dnnl::graph::sycl_interop::make_allocator(graph_malloc, graph_free))),
        stream(dnnl::sycl_interop::make_stream(engine, q)) {}
};

Ctx & ctx_for(sycl::queue & q) {
    static std::mutex                     mu;
    static std::map<sycl::queue *, Ctx *> cache;
    std::lock_guard<std::mutex>           lock(mu);
    auto                                  it = cache.find(&q);
    if (it == cache.end()) it = cache.emplace(&q, new Ctx(q)).first;
    return *it->second;
}

// Operand strides in elements when it is read in place (an F16 tensor with
// dense rows); all zero when it goes through the F16 scratch converter.
struct Direct {
    long long s1 = 0, s2 = 0;  // row (S) and head (H) strides
    long long s3 = 0;          // batch stride (ne[3] > 1)
    bool      on() const { return s1 != 0; }
    auto      tie() const { return std::tie(s1, s2, s3); }
};

struct Key {
    long long B = 1;  // batch (ne[3]); > 1 only with all operands in place
    long long S, S_kv, H, H_kv, D, mask_ld;
    bool      has_mask;
    float     scale;
    Direct    dq, dk, dv;
    // All of Q, K, V and dst BF16: the kernel runs on BF16 in place and writes
    // dst directly.
    bool      bf = false;
    auto      tie() const {
        return std::tie(B, S, S_kv, H, H_kv, D, mask_ld, has_mask, scale, dq.s1, dq.s2, dq.s3, dk.s1, dk.s2, dk.s3,
                        dv.s1, dv.s2, dv.s3, bf);
    }
    bool      operator<(const Key & o) const { return tie() < o.tie(); }
};

struct Plan {
    bool                            ok = false;
    bool                            conv_q = true, conv_k = true, conv_v = true;
    lt                              q, k, v, scale, mask, out;
    dnnl::graph::compiled_partition cp;
    float *                         d_scale = nullptr;
};

Plan build(const Key & key, sycl::queue & q, Ctx & c) {
    Plan      p;
    const long long rep = key.H / key.H_kv, S = key.S, Skv = key.S_kv, D = key.D;
    size_t    id  = 0;

    // F16 operands: either the converter's scratch (q as [H][S][D], k and v as
    // [H_kv][S_kv][D]) or the ggml tensor itself, read in place with its own
    // row and head strides.
    const gdt ty = key.bf ? gdt::bf16 : gdt::f16;
    p.conv_q = !key.dq.on();
    p.conv_k = !key.dk.on();
    p.conv_v = !key.dv.on();
    const long long B = key.B;
    p.q = p.conv_q ? lt(id++, ty, lt::dims{1, key.H_kv, rep, S, D},
                        lt::dims{key.H * S * D, rep * S * D, S * D, D, 1})
                   : lt(id++, ty, lt::dims{B, key.H_kv, rep, S, D},
                        lt::dims{B > 1 ? key.dq.s3 : key.H * S * D, rep * key.dq.s2, key.dq.s2, key.dq.s1, 1});
    auto kv_lt = [&](const Direct & d) {
        return d.on() ? lt(id++, ty, lt::dims{B, key.H_kv, 1, Skv, D},
                           lt::dims{B > 1 ? d.s3 : key.H_kv * Skv * D, d.s2, d.s2, d.s1, 1})
                      : lt(id++, ty, lt::dims{1, key.H_kv, 1, Skv, D},
                           lt::dims{key.H_kv * Skv * D, Skv * D, Skv * D, D, 1});
    };
    p.k = kv_lt(key.dk);
    p.v = kv_lt(key.dv);
    p.scale = lt(id++, gdt::f32, lt::dims{1}, lt::layout_type::strided);
    // ggml's mask is [S_pad rows of S_kv], broadcast over heads; only the first
    // S rows are read.
    p.mask = lt(id++, gdt::f16, lt::dims{1, 1, 1, S, Skv},  // B == 1 whenever there is a mask
                lt::dims{S * key.mask_ld, S * key.mask_ld, S * key.mask_ld, key.mask_ld, 1});
    // Output in ggml's dst order [D, H, S] (element (h, s, d) at s*H*D + h*D + d)
    // but F16, into scratch: oneDNN fuses the pattern only when the output type
    // matches the inputs'; an F32 output silently becomes gemm + softmax + gemm.
    // The widening to the real dst type is a separate elementwise pass.
    p.out = lt(id++, ty, lt::dims{B, key.H_kv, rep, S, D},
               lt::dims{S * key.H * D, rep * D, D, key.H * D, 1});

    const lt::dims score{B, key.H_kv, rep, S, Skv};
    lt s0(id++, gdt::f32, score, lt::layout_type::strided);
    lt s1(id++, gdt::f32, score, lt::layout_type::strided);
    lt s2(id++, gdt::f32, score, lt::layout_type::strided);
    lt pr(id++, ty, score, lt::layout_type::strided);

    using op = dnnl::graph::op;
    op bmm1(id++, op::kind::MatMul, "qk");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({p.q, p.k});
    bmm1.add_outputs({s0});
    op mul(id++, op::kind::Multiply, "scale");
    mul.add_inputs({s0, p.scale});
    mul.add_outputs({s1});
    op add(id++, op::kind::Add, "mask");
    add.add_inputs({s1, p.mask});
    add.add_outputs({s2});
    op sm(id++, op::kind::SoftMax, "softmax");
    sm.set_attr<int64_t>(op::attr::axis, -1);
    // A fully masked row is -inf everywhere; inf_as_zero makes it zeros, which
    // is what ggml's kernels produce, instead of NaN.
    sm.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    sm.add_inputs({key.has_mask ? s2 : s1});
    sm.add_outputs({pr});
    op bmm2(id++, op::kind::MatMul, "pv");
    bmm2.add_inputs({pr, p.v});
    bmm2.add_outputs({p.out});

    try {
        dnnl::graph::graph g(dnnl::engine::kind::gpu);
        g.add_op(bmm1);
        g.add_op(mul);
        if (key.has_mask) g.add_op(add);
        g.add_op(sm);
        g.add_op(bmm2);
        g.finalize();
        auto parts = g.get_partitions();
        if (parts.size() != 1 || !parts[0].is_supported()) {
            std::fprintf(stderr, "vla(sycl): fused SDPA unavailable for S=%lld kv=%lld %lldq/%lldkv x%lld%s; "
                         "using ggml's flash attention\n", S, Skv, key.H, key.H_kv, D,
                         key.dq.on() || key.dk.on() || key.dv.on() ? " (in-place operands)" : "");
            return p;
        }
        std::vector<lt> ins = key.has_mask ? std::vector<lt>{p.q, p.k, p.scale, p.mask, p.v}
                                           : std::vector<lt>{p.q, p.k, p.scale, p.v};
        p.cp = parts[0].compile(ins, {p.out}, c.engine);
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(sycl): fused SDPA compile failed (%s); using ggml's flash attention\n",
                     e.what());
        return p;
    }
    p.d_scale = sycl::malloc_device<float>(1, q);
    q.memcpy(p.d_scale, &key.scale, sizeof(float)).wait();
    p.ok = true;
    return p;
}

/// Grown, never shrunk: one region each for the F16 q, k, v and output.
half * scratch(sycl::queue & q, int slot, size_t elems) {
    static std::mutex mu;
    static half *     buf[4] = {nullptr, nullptr, nullptr, nullptr};
    static size_t     cap[4] = {0, 0, 0, 0};
    std::lock_guard<std::mutex> lock(mu);
    if (elems > cap[slot]) {
        if (buf[slot]) {
            q.wait();
            sycl::free(buf[slot], q);
        }
        buf[slot] = sycl::malloc_device<half>(elems, q);
        cap[slot] = elems;
    }
    return buf[slot];
}

template <typename T> struct k_to_f16;
struct k_f16_q8 {};
struct k_from_f16 {};
struct k_from_f16_bf {};

template <typename T> inline float to_float(T v) { return (float) v; }

/// [D, S, H] tensor of any float type, any strides -> contiguous F16 [H][S][D].
/// One work-item per element, rows along the fast dimension so reads coalesce
/// whenever the source rows are dense.
template <typename T>
void to_f16_t(const ggml_tensor * t, half * dst, sycl::queue & q) {
    const long long D = t->ne[0], S = t->ne[1], H = t->ne[2];
    const size_t    nb1 = t->nb[1], nb2 = t->nb[2];
    const char *    src = static_cast<const char *>(t->data);
    const bool      dense = t->nb[0] == sizeof(T);
    const size_t    nb0 = t->nb[0];
    q.parallel_for<k_to_f16<T>>(sycl::range<2>((size_t) (S * H), (size_t) D), [=](sycl::item<2> it) {
        const long long r = (long long) it[0], d = (long long) it[1];
        const long long h = r / S, s = r - h * S;
        const char *    row = src + s * nb1 + h * nb2;
        const T         v = dense ? reinterpret_cast<const T *>(row)[d] : *reinterpret_cast<const T *>(row + d * nb0);
        dst[r * D + d] = (half) to_float(v);
    });
}

bool to_f16(const ggml_tensor * t, half * dst, sycl::queue & q) {
    if (t->type == GGML_TYPE_F32)      to_f16_t<float>(t, dst, q);
    else if (t->type == GGML_TYPE_F16) to_f16_t<half>(t, dst, q);
    else                               to_f16_t<sycl::ext::oneapi::bfloat16>(t, dst, q);
    return true;
}

/// In-place strides for an operand of type `want` (F16 or BF16) with dense
/// rows, else none.
Direct direct(const ggml_tensor * t, int want = GGML_TYPE_F16) {
    static const bool off = vla::env_flag("VLA_SYCL_SDPA_CONVERT");
    if (off || t->type != want || t->nb[0] != sizeof(half)) return {};
    if (t->nb[1] % sizeof(half) || t->nb[2] % sizeof(half)) return {};
    return Direct{ (long long) (t->nb[1] / sizeof(half)), (long long) (t->nb[2] / sizeof(half)),
                   (long long) (t->nb[3] / sizeof(half)) };
}

bool float_type(int t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16; }

}  // namespace

namespace vla {

// Every shape and type test the kernel below makes before it builds a plan. It is
// also the hook's supports_op answer for FLASH_ATTN_EXT: ggml-sycl (b11223+)
// refuses K and V of different types and strides not a multiple of 16, both
// normal here (F32 K from RoPE next to a BF16 V; a mask row of n_kv halves), and
// a refused node is scheduled onto the CPU before the compute hook sees it.
bool sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    static const bool off = vla::env_flag("VLA_SYCL_NO_SDPA");
    if (off || dst->op != GGML_OP_FLASH_ATTN_EXT) return false;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];
    if (!Q || !K || !V) return false;

    float max_bias = 0.0f, softcap = 0.0f;
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) return false;

    if (!float_type(Q->type) || !float_type(K->type) || !float_type(V->type)) return false;
    // I8: a q8 blob (layers/fused_ew.h) of the S rows of H*D, for an INT8 GEMM.
    const bool blob = dst->type == GGML_TYPE_I8;
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_BF16 && !blob) return false;
    if (blob && (dst->ne[0] != Q->ne[1] * (Q->ne[0] * Q->ne[2] + 4) || Q->ne[3] != 1 ||
                 (Q->ne[0] * Q->ne[2]) % 8)) return false;
    // A batch (ne[3] > 1) only on the in-place BF16 path: every operand BF16 with
    // dense rows, BF16 dst, no mask.
    if (Q->ne[3] != K->ne[3] || Q->ne[3] != V->ne[3]) return false;
    if (Q->ne[3] > 1) {
        for (const ggml_tensor * t : { Q, K, V })
            if (t->type != GGML_TYPE_BF16 || t->nb[0] != 2) return false;
        if (dst->type != GGML_TYPE_BF16 || M) return false;
    }
    if (M && (M->type != GGML_TYPE_F16 || M->ne[2] != 1 || M->ne[3] != 1 ||
              M->nb[0] != sizeof(half) || M->ne[1] < Q->ne[1])) return false;
    if (!ggml_is_contiguous(dst)) return false;

    const long long D = Q->ne[0], Skv = K->ne[1], Hkv = K->ne[2];
    return K->ne[0] == D && V->ne[0] == D && V->ne[1] == Skv && V->ne[2] == Hkv && Hkv != 0 &&
           Q->ne[2] % Hkv == 0;
}

bool sycl_flash_attn_ext(ggml_tensor * dst, sycl::queue & q) {
    if (!sycl_flash_attn_ext_supported(dst)) return false;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const long long D = Q->ne[0], S = Q->ne[1], H = Q->ne[2];
    const long long Skv = K->ne[1], Hkv = K->ne[2];

    Key key{1, S, Skv, H, Hkv, D, M ? (long long) (M->nb[1] / sizeof(half)) : Skv, M != nullptr,
            scale, direct(Q), direct(K), direct(V)};
    const long long B = Q->ne[3];
    // Multi-query attention (one K/V head, as in Gemma) with Q rows (s, h) laid
    // out densely, as RoPE writes them, is single-head attention over S*H query
    // rows, which tiles better than H short heads. dst [D, H, S] keeps the order.
    static const bool no_flat = vla::env_flag("VLA_SYCL_SDPA_NO_MQA_FLAT");
    if (!no_flat && Hkv == 1 && H > 1 && !M && key.dq.on() && key.dq.s2 == D && key.dq.s1 == H * D) {
        key.S  = S * H;
        key.H  = 1;
        key.dq = Direct{ D, S * H * D };
    }
    if (dst->type == GGML_TYPE_BF16 && !M) {
        const Direct bq = direct(Q, GGML_TYPE_BF16), bk = direct(K, GGML_TYPE_BF16), bv = direct(V, GGML_TYPE_BF16);
        static const bool no_bf = vla::env_flag("VLA_SYCL_SDPA_NO_BF16");
        if (B > 1 && !(bq.on() && bk.on() && bv.on())) return false;
        if ((!no_bf || B > 1) && bq.on() && bk.on() && bv.on()) {
            key.B  = B;
            key.dq = key.H == 1 && H > 1 ? Direct{ D, S * H * D } : bq;
            key.dk = bk;
            key.dv = bv;
            key.bf = true;
        }
    }

    Ctx & c = ctx_for(q);
    static std::mutex          mu;
    static std::map<Key, Plan> plans;
    Plan *                     p;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto                        it = plans.find(key);
        if (it == plans.end()) {
            Plan np = build(key, q, c);
            if (!np.ok && key.B == 1 && (key.dq.on() || key.dk.on() || key.dv.on())) {
                // oneDNN would not fuse with these strides: convert all three.
                Key ck = key;
                ck.dq = ck.dk = ck.dv = Direct{};
                ck.bf = false;
                ck.S  = S;  // the converter writes heads apart: no MQA flattening
                ck.H  = H;
                np = build(ck, q, c);
            }
            it = plans.emplace(key, np).first;
        }
        p = &it->second;
    }
    if (!p->ok) return false;

    // A BF16 plan writes dst itself.
    half * o16 = p->q.get_data_type() == gdt::bf16 ? static_cast<half *>(dst->data) : scratch(q, 3, (size_t) (D * S * H));
    half * q16 = p->conv_q ? scratch(q, 0, (size_t) (D * S * H)) : static_cast<half *>(Q->data);
    half * k16 = p->conv_k ? scratch(q, 1, (size_t) (D * Skv * Hkv)) : static_cast<half *>(K->data);
    half * v16 = p->conv_v ? scratch(q, 2, (size_t) (D * Skv * Hkv)) : static_cast<half *>(V->data);
    if (p->conv_q) to_f16(Q, q16, q);
    if (p->conv_k) to_f16(K, k16, q);
    if (p->conv_v) to_f16(V, v16, q);

    using dnnl::graph::tensor;
    tensor tq(p->q, c.engine, q16), tk(p->k, c.engine, k16), tv(p->v, c.engine, v16);
    tensor ts(p->scale, c.engine, p->d_scale), to(p->out, c.engine, o16);
    try {
        if (M) {
            tensor tm(p->mask, c.engine, M->data);
            p->cp.execute(c.stream, {tq, tk, ts, tm, tv}, {to});
        } else {
            p->cp.execute(c.stream, {tq, tk, ts, tv}, {to});
        }
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "vla(sycl): fused SDPA execute failed: %s\n", e.what());
        return false;
    }

    if (p->q.get_data_type() == gdt::bf16) return true;
    if (dst->type == GGML_TYPE_I8) {
        // Rows of H*D (one token each) straight to INT8 codes + row scales.
        const long long R = H * D;
        int8_t *        codes  = static_cast<int8_t *>(dst->data);
        float *         scales = reinterpret_cast<float *>(codes + S * R);
        constexpr int   WGS    = 256;
        q.parallel_for<k_f16_q8>(sycl::nd_range<1>(sycl::range<1>((size_t) S * WGS), sycl::range<1>(WGS)),
                                 [=](sycl::nd_item<1> it) {
            const long long s   = (long long) it.get_group(0);
            const int       tid = (int) it.get_local_id(0);
            using V8 = sycl::vec<half, 8>;
            const V8 * row = reinterpret_cast<const V8 *>(o16 + s * R);
            float amax = 0.0f;
            for (long long c = tid; c < R / 8; c += WGS) {
                const V8 v = row[c];
#pragma unroll
                for (int j = 0; j < 8; ++j) amax = sycl::fmax(amax, sycl::fabs((float) v[j]));
            }
            amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
            const float sc = amax > 0.0f ? amax / 127.0f : 1.0f, inv = 1.0f / sc;
            auto * out = reinterpret_cast<sycl::vec<int8_t, 8> *>(codes + s * R);
            for (long long c = tid; c < R / 8; c += WGS) {
                const V8 v = row[c];
                sycl::vec<int8_t, 8> o;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    o[j] = (int8_t) sycl::rint(sycl::fmin(127.0f, sycl::fmax(-127.0f, (float) v[j] * inv)));
                out[c] = o;
            }
            if (tid == 0) scales[s] = sc;
        });
        return true;
    }
    // Widen into the real dst, which is contiguous in the same order.
    const size_t n = (size_t) (D * S * H);
    if (dst->type == GGML_TYPE_F32) {
        float * o = static_cast<float *>(dst->data);
        q.parallel_for<k_from_f16>(sycl::range<1>(n), [=](sycl::id<1> i) { o[i] = (float) o16[i]; });
    } else {
        auto * o = static_cast<sycl::ext::oneapi::bfloat16 *>(dst->data);
        q.parallel_for<k_from_f16_bf>(sycl::range<1>(n), [=](sycl::id<1> i) {
            o[i] = sycl::ext::oneapi::bfloat16((float) o16[i]);
        });
    }
    return true;
}

}  // namespace vla
