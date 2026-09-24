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

// FoldQuant nodes on the CUDA backend. A site is two GGML_OP_CUSTOM nodes
// (layers/fq_linear.h) whose userdata points at an FqActSpec / FqGemmSpec; the
// magic word in that struct is how this handler recognises its nodes.
//
// Everything is validated before anything is launched. A node that fails the
// contract is declined (return false), which makes ggml abort loudly on the
// unsupported op instead of silently computing something else.
//
// VLA_FQ_CPU_REF=1 routes every node through the CPU reference on host copies
// of the device tensors: slow, but a byte-exact A/B against the kernels.

#include "foldquant.h"
#include <cstdlib>
#include <algorithm>
#include "foldquant_ref.h"
#include "env_flag.h"
#include "kernels/foldquant/fq_kernels.h"
#include "cuda/vla_cuda_ext.h"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Layout of the op_params ggml_custom_4d writes (ggml-impl.h, private). Pinned
// by tests/test_foldquant_cuda_op.cpp, which checks that the CPU op and this
// decoder see the same userdata.
struct custom_op_params_mirror {
    void * fun;
    int    n_tasks;
    void * userdata;
};

const void * node_userdata(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_CUSTOM) return nullptr;
    custom_op_params_mirror p;
    std::memcpy(&p, dst->op_params, sizeof(p));
    return p.userdata;
}

uint32_t magic_of(const void * ud) {
    uint32_t m = 0;
    if (ud) std::memcpy(&m, ud, sizeof(m));
    return m;
}

bool trace() {
    static const bool t = vla::env_flag("VLA_FQ_TRACE");
    return t;
}

bool cpu_ref() {
    static const bool r = vla::env_flag("VLA_FQ_CPU_REF");
    return r;
}

// --- CPU reference on host copies ------------------------------------------

struct HostCopy {
    ggml_tensor        t;
    std::vector<uint8_t> bytes;
};

bool host_copy(const ggml_tensor * src, HostCopy & out, cudaStream_t stream) {
    out.t = *src;
    out.bytes.resize(ggml_nbytes(src));
    if (cudaMemcpyAsync(out.bytes.data(), src->data, out.bytes.size(), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return false;
    out.t.data = out.bytes.data();
    return true;
}

template <typename Fn>
bool run_on_host(ggml_tensor * dst, int n_src, Fn fn, cudaStream_t stream) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "vla(fq): VLA_FQ_CPU_REF cannot run inside CUDA-graph capture; "
                             "set GGML_CUDA_DISABLE_GRAPHS=1 (foldquant_check_backend does so at load)\n");
        return false;
    }
    HostCopy srcs[GGML_MAX_SRC];
    ggml_tensor d = *dst;
    for (int i = 0; i < n_src; ++i) {
        if (!dst->src[i]) { d.src[i] = nullptr; continue; }
        if (!host_copy(dst->src[i], srcs[i], stream)) return false;
        d.src[i] = &srcs[i].t;
    }
    std::vector<uint8_t> out(ggml_nbytes(dst));
    d.data = out.data();
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
    fn(&d);
    return cudaMemcpyAsync(dst->data, out.data(), out.size(), cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

// VLA_FQ_CHECK=1: after each kernel, recompute the node with the CPU reference
// on host copies of the same inputs and report any byte difference. Slow;
// a diagnostic for a mismatch that the unit tests' shapes do not reproduce.
bool check_mode() {
    static const bool c = vla::env_flag("VLA_FQ_CHECK");
    return c;
}

template <typename Fn>
void check_against_host(ggml_tensor * dst, int n_src, Fn fn, cudaStream_t stream, size_t cmp_bytes_per_row, size_t row_bytes) {
    HostCopy srcs[GGML_MAX_SRC];
    ggml_tensor d = *dst;
    for (int i = 0; i < n_src; ++i) {
        if (!dst->src[i]) { d.src[i] = nullptr; continue; }
        if (!host_copy(dst->src[i], srcs[i], stream)) return;
        d.src[i] = &srcs[i].t;
    }
    std::vector<uint8_t> got(ggml_nbytes(dst)), want(ggml_nbytes(dst));
    if (cudaMemcpyAsync(got.data(), dst->data, got.size(), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    d.data = want.data();
    fn(&d);
    const int64_t rows = dst->ne[1];
    int64_t bad_rows = 0, first = -1;
    size_t first_off = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t * a = got.data() + (size_t) r * row_bytes;
        const uint8_t * b = want.data() + (size_t) r * row_bytes;
        for (size_t k = 0; k < cmp_bytes_per_row; ++k)
            if (a[k] != b[k]) { ++bad_rows; if (first < 0) { first = r; first_off = k; } break; }
    }
    if (bad_rows) {
        std::printf("vla(fq) CHECK %-40s ne=[%lld,%lld] MISMATCH rows %lld/%lld (first row %lld byte %zu)\n",
                    ggml_get_name(dst), (long long) dst->ne[0], (long long) dst->ne[1], (long long) bad_rows,
                    (long long) rows, (long long) first, first_off);
        if (dst->type == GGML_TYPE_F32) {
            const float * a = (const float *) (got.data() + (size_t) first * row_bytes);
            const float * b = (const float *) (want.data() + (size_t) first * row_bytes);
            std::printf("    kernel:"); for (int k = 0; k < 6; ++k) std::printf(" %.7g", a[k]); std::printf("\n");
            std::printf("    host  :"); for (int k = 0; k < 6; ++k) std::printf(" %.7g", b[k]); std::printf("\n");
        }
    }
    else
        std::printf("vla(fq) CHECK %-40s ne=[%lld,%lld] ok\n", ggml_get_name(dst), (long long) dst->ne[0], (long long) dst->ne[1]);
}

// Rows contiguous and 16-byte aligned, any row stride, higher dims packed: what
// the prologue reads without a ggml_cont in front of it.
// VLA_FQ_PREFETCH_MB: how much of the next site's weights a GEMM prefetches
// into L2. Off by default: on Orin (4 MB L2) a 2 MB prefetch made the GEMMs
// 12% slower in the model (extra DRAM traffic, no hit-rate gain).
int64_t prefetch_bytes() {
    static const int64_t b = [] {
        const char * e = std::getenv("VLA_FQ_PREFETCH_MB");
        return (e && *e) ? (int64_t) (std::atof(e) * 1024.0 * 1024.0) : (int64_t) 0;
    }();
    return b;
}

bool contiguous_f32_rows(const ggml_tensor * t, int64_t K) {
    return t && t->type == GGML_TYPE_F32 && t->ne[0] == K && t->nb[0] == sizeof(float) &&
           t->nb[1] % 16 == 0 && t->nb[2] == t->nb[1] * (size_t) t->ne[1] && t->nb[3] == t->nb[2] * (size_t) t->ne[2];
}

// --- the two nodes ----------------------------------------------------------

bool forward_act(ggml_tensor * dst, const vla::FqActSpec & s, cudaStream_t stream) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * g = nullptr, * as = nullptr;
    vla::fq_act_srcs(dst, s, &g, &as);
    const int64_t K  = s.K;
    const int64_t rb = vla::fq_act_row_bytes(K, s.abits);
    if (!contiguous_f32_rows(x, K) || dst->type != GGML_TYPE_I8 || dst->ne[0] != rb ||
        (as && !(as->type == GGML_TYPE_F32 && as->ne[0] == K)) ||
        (g  && !(g->type  == GGML_TYPE_F32 && g->ne[0]  == K)) ||
        (s.has_gamma && !g) || (s.has_ascale && !as)) {
        std::fprintf(stderr, "vla(fq): act node %s violates the contract (x %s ne0=%lld, dst ne0=%lld)\n",
                     ggml_get_name(dst), x ? ggml_type_name(x->type) : "null",
                     x ? (long long) x->ne[0] : 0ll, (long long) dst->ne[0]);
        return false;
    }
    const int64_t M = ggml_nelements(x) / K;
    if (trace())
        std::printf("vla(fq): act %s M=%lld K=%lld A%d rot%d%s%s\n", ggml_get_name(dst), (long long) M,
                    (long long) K, s.abits, s.rot_block, g ? " +rmsnorm" : "", as ? " +ascale" : "");

    if (cpu_ref())
        return run_on_host(dst, 3, [&](ggml_tensor * d) { vla::fq_act_cpu(d, 0, 1, (void *) &s); }, stream);

    vla::fq::ActArgs a;
    a.x = (const float *) x->data;
    a.x_stride = (int64_t) (x->nb[1] / sizeof(float));
    a.ascale = as ? (const float *) as->data : nullptr;
    a.gamma  = (s.has_gamma && g) ? (const float *) g->data : nullptr;
    a.blob = (int8_t *) dst->data;
    a.row_bytes = rb;
    a.M = M; a.K = K;
    a.abits = s.abits; a.rot_block = s.rot_block; a.fold_before = s.fold_before;
    a.clip = s.clip; a.eps = s.eps;
    a.inv_sqrt_bs = vla::fqref::inv_sqrt_block(s.rot_block > 1 ? s.rot_block : 1);
    const cudaError_t e = vla::fq::launch_act(a, stream);
    if (e == cudaErrorNotSupported)
        return run_on_host(dst, 3, [&](ggml_tensor * d) { vla::fq_act_cpu(d, 0, 1, (void *) &s); }, stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "vla(fq): act launch failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    if (check_mode())
        check_against_host(dst, 3, [&](ggml_tensor * d) { vla::fq_act_cpu(d, 0, 1, (void *) &s); }, stream,
                           (size_t) vla::fq_act_kpack(K, s.abits) + 4, (size_t) rb);
    return true;
}

bool forward_gemm(ggml_tensor * dst, const vla::FqGemmSpec & s, cudaStream_t stream) {
    const ggml_tensor * w  = dst->src[0];
    const ggml_tensor * xq = dst->src[1];
    const ggml_tensor * ws = dst->src[2];
    const ggml_tensor * b  = dst->src[3];
    const ggml_tensor * r  = dst->src[4];
    const int64_t K = s.K, N = s.N;
    const int64_t kpw = vla::fq_w_kpack(K, s.wbits);
    if (!w || w->type != GGML_TYPE_I8 || w->ne[0] != kpw || w->ne[1] != N || !ggml_is_contiguous(w) ||
        !xq || xq->type != GGML_TYPE_I8 || !ggml_is_contiguous(xq) ||
        !ws || ws->type != GGML_TYPE_F32 || ws->ne[0] != N ||
        (b && !(b->type == GGML_TYPE_F32 && b->ne[0] == N)) ||
        (r && !(r->type == GGML_TYPE_F32 && ggml_is_contiguous(r) && ggml_nelements(r) == ggml_nelements(dst) && r->ne[0] == N)) ||
        dst->type != GGML_TYPE_F32 || dst->ne[0] != N || !ggml_is_contiguous(dst)) {
        std::fprintf(stderr, "vla(fq): gemm node %s violates the contract\n", ggml_get_name(dst));
        return false;
    }
    const int abits = (xq->ne[0] == K + vla::FQ_ACT_TAIL) ? 8 : 4;
    if (xq->ne[0] != vla::fq_act_row_bytes(K, abits)) {
        std::fprintf(stderr, "vla(fq): gemm node %s: blob row of %lld bytes does not match K=%lld\n",
                     ggml_get_name(dst), (long long) xq->ne[0], (long long) K);
        return false;
    }
    const int64_t M = xq->ne[1];
    if (trace())
        std::printf("vla(fq): gemm %s M=%lld N=%lld K=%lld W%dA%d%s\n", ggml_get_name(dst), (long long) M,
                    (long long) N, (long long) K, s.wbits, abits, b ? " +bias" : "");

    if (cpu_ref())
        return run_on_host(dst, 5, [&](ggml_tensor * d) { vla::fq_gemm_cpu(d, 0, 1, (void *) &s); }, stream);

    vla::fq::GemmArgs g;
    g.w = (const int8_t *) w->data;
    g.blob = (const int8_t *) xq->data;
    g.wscale = (const float *) ws->data;
    g.bias = b ? (const float *) b->data : nullptr;
    g.res  = r ? (const float *) r->data : nullptr;
    if (s.next_w && s.next_w->data && prefetch_bytes() > 0) {
        g.pf = (const int8_t *) s.next_w->data;
        g.pf_bytes = std::min<int64_t>((int64_t) ggml_nbytes(s.next_w), prefetch_bytes());
    }
    g.y = (float *) dst->data;
    g.head_dim = s.head_dim; g.heads = s.heads; g.vmask = s.vmask;
    g.M = M; g.N = N; g.K = K;
    g.row_bytes = xq->ne[0];
    g.wbits = s.wbits; g.abits = abits;
    const cudaError_t e = vla::fq::launch_gemm(g, stream);
    if (e == cudaErrorNotSupported) {
        static bool warned = false;
        if (!warned) {
            std::printf("vla(fq): W%dA%d GEMM has no CUDA kernel yet; running the CPU reference for those sites\n",
                        s.wbits, abits);
            warned = true;
        }
        return run_on_host(dst, 5, [&](ggml_tensor * d) { vla::fq_gemm_cpu(d, 0, 1, (void *) &s); }, stream);
    }
    if (e != cudaSuccess) {
        std::fprintf(stderr, "vla(fq): gemm launch failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    if (check_mode())
        check_against_host(dst, 5, [&](ggml_tensor * d) { vla::fq_gemm_cpu(d, 0, 1, (void *) &s); }, stream,
                           (size_t) N * sizeof(float), (size_t) N * sizeof(float));
    return true;
}

}  // namespace

extern "C" bool vla_cuda_foldquant_forward(ggml_tensor * dst, void * stream_v) {
    if (!dst || dst->op != GGML_OP_CUSTOM)
        return false;
    const void * ud = node_userdata(dst);
    switch (magic_of(ud)) {
        case vla::FQ_ACT_MAGIC:  return forward_act (dst, *(const vla::FqActSpec  *) ud, (cudaStream_t) stream_v);
        case vla::FQ_GEMM_MAGIC: return forward_gemm(dst, *(const vla::FqGemmSpec *) ud, (cudaStream_t) stream_v);
        default: return false;
    }
}

namespace vla {

void cuda_register_foldquant_ops() {
    cuda_ext_add_handler(vla_cuda_foldquant_forward);
}

}  // namespace vla
