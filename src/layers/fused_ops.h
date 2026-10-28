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

// Fused CUDA nodes (GGML_OP_CUSTOM). Each replaces a chain of ggml ops that
// the CUDA backend runs as separate kernels, each reading and writing the
// whole activation:
//
//   qkv_heads      bias + Q/K/V split + optional 2D RoPE + head layout
//   heads_rope     bias + NEOX RoPE + head layout, with an optional cached prefix
//   qk_norm_rope   per-head rms_norm * w + IMROPE + head layout
//   ln_affine      norm(x) * w + b
//   ln_modulate    n + n*sc + sh, n = norm(x)
//   ln_plain       norm(x)
//   rms_mul        rms_norm(x) * w
//   rms_modulate   n + n*sc + sh, n = rms_norm(x)
//   bias_res       r + (y + b)
//   bias_gelu      gelu(y + b), tanh form
//   bias_gelu_erf  gelu_erf(y + b)
//   bias_gate_res  r + (y + b)*g
//   gate_res       r + y*g
//   silu_mul       silu(g) * u
//   gelu_mul       gelu(g) * u
//   glu_act        gelu(g) * u, silu(g) * u or gelu(g), quantized as a FoldQuant
//                  act node (layers/fq_linear.h)
//   norm_act       ln_modulate, a plain LayerNorm or rms_modulate, quantized the
//                  same way
//   patchify       HWC images -> normalized Conv3d patches in merge order
//   gemm_views     BF16 W x, one GEMM per view, as ggml-cuda issues it
//   attn_exact     softmax(scale * Q K^T + mask) V
//
// The CUDA kernels (src/cuda/vla_cuda_fused.cu) give the same bits as the ggml
// chains they replace; gemm_views and attn_exact check this per shape at first
// use. A node that feeds only a BF16 GEMM may write BF16, the type ggml rounds
// the operand to anyway. The CPU functions keep a graph valid on other
// backends; models emit these nodes on CUDA only.

#pragma once

#include "env_flag.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <vector>

namespace vla {

constexpr uint32_t FUSED_MAGIC = 0x31534656u;  // "VFS1"

// Set once the CUDA handler for these nodes is registered (enable_fused_ops);
// generic helpers such as mm_act() only emit fused nodes then.
inline bool & fused_ops_active() {
    static bool on = false;
    return on;
}

// The flag above is set for the process, but one process can also hold models
// on other backends: a node is fused only where its weight lives on CUDA.
inline bool fused_ops_for(const ggml_tensor * w) {
    if (!fused_ops_active() || !w || !w->buffer || ggml_backend_buffer_is_host(w->buffer))
        return false;
    return std::strncmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(w->buffer)), "CUDA", 4) == 0;
}

enum class FusedKind : int32_t { QkvHeads = 1, BiasRes, BiasGelu, LnAffine, LnModulate, LnPlain, RmsMul, SiluMul,
                                 Patchify, QkNormRope, GemmViews, AttnExact, GeluMul, RmsModulate, GateRes,
                                 HeadsRope, BiasGeluErf, BiasGateRes, GluAct, NormAct };

// glu_act's activation: ggml's geglu_split, silu(g) * u (ggml_silu then ggml_mul), ggml_gelu.
enum class FusedGlu : int32_t { GeluMul, SiluMul, Gelu };

// Lives as node userdata, so it must outlive every graph: specs are interned
// for the life of the process (a model emits a handful of distinct ones).
struct FusedSpec {
    uint32_t  magic = FUSED_MAGIC;
    FusedKind kind  = FusedKind::BiasRes;
    // qkv_heads: n_parts of width hd*heads in the input rows; part p is written
    // as [hd, S, heads, nv] or, if bit p of vt_mask is set, [S, hd, heads, nv];
    // parts with bit p of rope_mask set get the 2D RoPE (cos/sin are [hd, S]).
    int32_t n_parts = 0, hd = 0, heads = 0, S = 0, nv = 0, vt_mask = 0, rope_mask = 0;
    float   eps = 0.0f;
    int32_t out_bf16 = 0;   // bias_gelu / norms: result stored as BF16
    // qkv_heads: which parts this node writes (0: all), packed in part order,
    // and stored F16 (what flash attention casts K/V to) instead of F32.
    int32_t part_mask = 0, out_f16 = 0;
    // patchify: image side, patch, temporal copies, spatial merge; per-channel mean/std
    int32_t side = 0, patch = 0, temporal = 0, merge = 0;
    float   mean[3] = {0, 0, 0}, std[3] = {1, 1, 1};
    // qk_norm_rope (hd, heads, S = tokens, eps, out_f16 above): IMROPE sections
    // and frequency base; ext_factor 0, freq_scale 1, attn_factor 1 only.
    int32_t sect[4] = {0, 0, 0, 0};
    float   freq_base = 0.0f;
    // glu_act: the activation, and the FqActSpec of the site it feeds (no gamma;
    // u in src[1] unless Gelu, ascale in src[2]). norm_act: the norm
    // (LnModulate, LnPlain or RmsModulate, eps above), the same act fields;
    // p1/p2 in src[1..2] unless LnPlain, ascale in src[3].
    FusedGlu  glu  = FusedGlu::GeluMul;
    FusedKind norm = FusedKind::LnPlain;
    int32_t abits = 0, rot_block = 0, fold_before = 0;
    float   clip = 0.0f, inv_sqrt_bs = 0.0f;
};

inline const FusedSpec * fused_intern(const FusedSpec & s) {
    static std::mutex            mu;
    static std::deque<FusedSpec> pool;
    std::lock_guard<std::mutex> lk(mu);
    for (const FusedSpec & p : pool)
        if (std::memcmp(&p, &s, sizeof(s)) == 0)
            return &p;
    pool.push_back(s);
    return &pool.back();
}

namespace fused_cpu {

inline float silu(float x) {
    return x/(1.0f+std::exp(-x));
}

inline float gelu(float x) {
    const float GELU_COEF_A = 0.044715f, SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f*x*(1.0f+std::tanh(SQRT_2_OVER_PI*x*(1.0f+GELU_COEF_A*x*x)));
}

inline void norm_row(const float * x, float * n, int64_t ncols, float eps) {
    double s = 0.0, s2 = 0.0;
    for (int64_t i=0; i<ncols; ++i) { s += x[i]; s2 += (double) x[i]*x[i]; }
    const float mean = (float) (s/ncols), var = (float) (s2/ncols) - mean*mean;
    const float inv = 1.0f/std::sqrt(var+eps);
    for (int64_t i=0; i<ncols; ++i) n[i] = (x[i]-mean)*inv;
}

inline void store(ggml_tensor * dst, int64_t i, float v) {
    if (dst->type == GGML_TYPE_BF16)     ((ggml_bf16_t *) dst->data)[i] = ggml_fp32_to_bf16(v);
    else if (dst->type == GGML_TYPE_F16) ((ggml_fp16_t *) dst->data)[i] = ggml_fp32_to_fp16(v);
    else                                 ((float *) dst->data)[i] = v;
}

inline int packed_part(const FusedSpec & s, int p) {
    if (!s.part_mask) return p;
    if (!((s.part_mask >> p) & 1)) return -1;
    int k = 0;
    for (int j=0; j<p; ++j) k += (s.part_mask >> j) & 1;
    return k;
}

inline void run(ggml_tensor * dst, int ith, int nth, void * ud) {
    if (ith != 0) return;
    (void) nth;
    const FusedSpec & s = *(const FusedSpec *) ud;
    float * out = (float *) dst->data;
    const ggml_tensor * a = dst->src[0];
    const float * x = (const float *) a->data;
    switch (s.kind) {
        case FusedKind::QkvHeads: {
            (void) out;
            const float * b  = dst->src[1] ? (const float *) dst->src[1]->data : nullptr;
            const float * cs = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;
            const float * sn = dst->src[3] ? (const float *) dst->src[3]->data : nullptr;
            const int64_t D = (int64_t) s.hd*s.heads, W = D*s.n_parts, half = s.hd/2, part = D*s.S*s.nv;
            for (int64_t p=0; p<s.n_parts; ++p)
            if (packed_part(s, (int) p) >= 0)
            for (int64_t v=0; v<s.nv; ++v)
            for (int64_t h=0; h<s.heads; ++h)
            for (int64_t t=0; t<s.S; ++t)
            for (int64_t d=0; d<s.hd; ++d) {
                const int64_t c = p*D+h*s.hd+d, row = (t+(int64_t) s.S*v)*W;
                float val = x[row+c]+(b ? b[c] : 0.0f);
                if ((s.rope_mask >> p) & 1) {
                    const int64_t pc = d < half ? c+half : c-half;
                    const float pv = x[row+pc]+(b ? b[pc] : 0.0f);
                    const float rot = d < half ? -pv : pv;
                    val = val*cs[d+s.hd*t]+rot*sn[d+s.hd*t];
                }
                const int64_t o = ((s.vt_mask >> p) & 1) ? t+s.S*(d+s.hd*(h+s.heads*v)) : d+s.hd*(t+s.S*(h+s.heads*v));
                store(dst, packed_part(s, (int) p)*part+o, val);
            }
            return;
        }
        case FusedKind::BiasRes: {
            const float * b = (const float *) dst->src[1]->data, * r = (const float *) dst->src[2]->data;
            const int64_t N = a->ne[0], n = ggml_nelements(a);
            for (int64_t i=0; i<n; ++i) out[i] = r[i]+(x[i]+b[i%N]);
            return;
        }
        case FusedKind::BiasGelu: {
            const float * b = (const float *) dst->src[1]->data;
            const int64_t N = a->ne[0], n = ggml_nelements(a);
            for (int64_t i=0; i<n; ++i) store(dst, i, gelu(x[i]+b[i%N]));
            return;
        }
        case FusedKind::Patchify: {
            const int64_t ps = s.patch, tps = s.temporal, m = s.merge, gw = s.side/ps, S = gw*gw, pf = 3*tps*ps*ps;
            const int64_t nv = a->ne[3];
            for (int64_t v=0; v<nv; ++v)
            for (int64_t si=0; si<S; ++si) {
                int64_t t = si; const int64_t wj = t%m; t /= m; const int64_t wi = t%m; t /= m;
                const int64_t bc = t%(gw/m), br = t/(gw/m), row = br*m+wi, col = bc*m+wj;
                for (int64_t ch=0; ch<3; ++ch)
                for (int64_t ph=0; ph<ps; ++ph)
                for (int64_t pw=0; pw<ps; ++pw) {
                    const int64_t px = ((v*s.side+row*ps+ph)*s.side+col*ps+pw)*3+ch;
                    const float raw = a->type == GGML_TYPE_I8 ? ((const uint8_t *) a->data)[px]/255.0f : x[px];
                    const float val = (raw-s.mean[ch])/s.std[ch];
                    for (int64_t tt=0; tt<tps; ++tt)
                        out[(v*S+si)*pf+ch*tps*ps*ps+tt*ps*ps+ph*ps+pw] = val;
                }
            }
            return;
        }
        case FusedKind::QkNormRope: {
            const float * w = (const float *) dst->src[1]->data;
            const int32_t * pos = (const int32_t *) dst->src[2]->data;
            const int64_t hd = s.hd, half = hd/2, T = s.S, sd = s.sect[0]+s.sect[1]+s.sect[2]+s.sect[3];
            const float theta_scale = std::pow(s.freq_base, -2.0f/(float) hd);
            std::vector<float> n((size_t) hd);
            for (int64_t t=0; t<T; ++t)
            for (int64_t h=0; h<s.heads; ++h) {
                const float * xr = x+(t*s.heads+h)*hd;
                double s2 = 0.0;
                for (int64_t d=0; d<hd; ++d) s2 += (double) xr[d]*xr[d];
                const float scale = 1.0f/std::sqrt((float) (s2/hd)+s.eps);
                for (int64_t d=0; d<hd; ++d) n[d] = scale*xr[d]*w[d];
                for (int64_t j=0; j<half; ++j) {
                    const int64_t sec = j%sd;
                    int axis = 3;
                    if      (sec%3 == 1 && sec < 3*s.sect[1]) axis = 1;
                    else if (sec%3 == 2 && sec < 3*s.sect[2]) axis = 2;
                    else if (sec%3 == 0 && sec < 3*s.sect[0]) axis = 0;
                    const float theta = (float) pos[t+T*axis]*std::pow(theta_scale, (float) j);
                    const float c = std::cos(theta), sn = std::sin(theta), x0 = n[j], x1 = n[j+half];
                    store(dst, j+hd*(t+T*h), x0*c-x1*sn);
                    store(dst, j+half+hd*(t+T*h), x0*sn+x1*c);
                }
            }
            return;
        }
        case FusedKind::GemmViews: {
            const ggml_tensor * W = dst->src[0], * X = dst->src[1];
            const int64_t K = W->ne[0], N = W->ne[1], T = X->ne[1];
            const ggml_bf16_t * w = (const ggml_bf16_t *) W->data;
            auto xv = [&](int64_t i) {
                return X->type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(((const ggml_bf16_t *) X->data)[i])
                                                 : ggml_bf16_to_fp32(ggml_fp32_to_bf16(((const float *) X->data)[i]));
            };
            for (int64_t t=0; t<T; ++t)
                for (int64_t n=0; n<N; ++n) {
                    float acc = 0.0f;
                    for (int64_t k=0; k<K; ++k)
                        acc += ggml_bf16_to_fp32(w[n*K+k])*xv(t*K+k);
                    out[t*N+n] = acc;
                }
            return;
        }
        case FusedKind::AttnExact: {
            const float * K = (const float *) dst->src[1]->data, * V = (const float *) dst->src[2]->data;
            const ggml_tensor * M = dst->src[3];
            const int64_t hd = s.hd, H = s.heads, Tq = s.S, Tkv = dst->src[1]->ne[1], r = H/dst->src[1]->ne[2];
            std::vector<float> p((size_t) Tkv);
            for (int64_t h=0; h<H; ++h)
            for (int64_t q=0; q<Tq; ++q) {
                float mx = -INFINITY, sum = 0.0f;
                const int64_t hk = h/r;
                for (int64_t k=0; k<Tkv; ++k) {
                    float d = 0.0f;
                    for (int64_t i=0; i<hd; ++i) d = std::fma(K[i+hd*(k+Tkv*hk)], x[i+hd*(q+Tq*h)], d);
                    const char * mrow = M ? (const char *) M->data+q*M->nb[1] : nullptr;
                    const float m = !M ? 0.0f : M->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) mrow)[k])
                                                                         : ((const float *) mrow)[k];
                    p[k] = d*s.eps+m;
                    mx = std::max(mx, p[k]);
                }
                for (int64_t k=0; k<Tkv; ++k) { p[k] = std::exp(p[k]-mx); sum += p[k]; }
                const int64_t nv = s.nv > 0 ? s.nv : 1, hh = h%(H/nv), v = h/(H/nv);
                for (int64_t i=0; i<hd; ++i) {
                    float acc = 0.0f;
                    for (int64_t k=0; k<Tkv; ++k) acc = std::fma(V[k+Tkv*(i+hd*hk)], p[k]*(1.0f/sum), acc);
                    store(dst, i+hd*hh+hd*(H/nv)*(q+Tq*v), acc);
                }
            }
            return;
        }
        case FusedKind::BiasGeluErf: {
            const float * b = (const float *) dst->src[1]->data;
            const int64_t N = a->ne[0], n = ggml_nelements(a);
            for (int64_t i=0; i<n; ++i) {
                const float v = x[i]+b[i%N];
                store(dst, i, 0.5f*v*(1.0f+std::erf(v*0.70710678118654752440f)));
            }
            return;
        }
        case FusedKind::BiasGateRes: {
            const float * b = (const float *) dst->src[1]->data, * g = (const float *) dst->src[2]->data;
            const float * r = (const float *) dst->src[3]->data;
            const int64_t N = a->ne[0], n = ggml_nelements(a);
            for (int64_t i=0; i<n; ++i) out[i] = r[i]+(x[i]+b[i%N])*g[i%N];
            return;
        }
        case FusedKind::GluAct:   // built with their own CPU functions (layers/fq_linear.h)
        case FusedKind::NormAct:
            return;
        case FusedKind::GeluMul: {
            const float * u = (const float *) dst->src[1]->data;
            for (int64_t i=0; i<ggml_nelements(a); ++i) store(dst, i, gelu(x[i])*u[i]);
            return;
        }
        case FusedKind::GateRes: {
            const float * g = (const float *) dst->src[1]->data, * r = (const float *) dst->src[2]->data;
            const int64_t N = a->ne[0], n = ggml_nelements(a);
            for (int64_t i=0; i<n; ++i) out[i] = r[i]+x[i]*g[i%N];
            return;
        }
        case FusedKind::RmsModulate: {
            const float * sc = (const float *) dst->src[1]->data, * sh = (const float *) dst->src[2]->data;
            const int64_t N = a->ne[0], rows = ggml_nrows(a);
            for (int64_t r=0; r<rows; ++r) {
                double s2 = 0.0;
                for (int64_t i=0; i<N; ++i) s2 += (double) x[r*N+i]*x[r*N+i];
                const float scale = 1.0f/std::sqrt((float) (s2/N)+s.eps);
                for (int64_t i=0; i<N; ++i) {
                    const float n = scale*x[r*N+i];
                    store(dst, r*N+i, (n+n*sc[i])+sh[i]);
                }
            }
            return;
        }
        case FusedKind::HeadsRope: {
            const int32_t * pos = dst->src[1] ? (const int32_t *) dst->src[1]->data : nullptr;
            const float * P = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;
            const float * bb = dst->src[3] ? (const float *) dst->src[3]->data : nullptr;
            const int64_t hd = s.hd, half = hd/2, H = s.heads, T = s.S;
            const int64_t Tp = P ? ggml_nelements(dst->src[2])/(hd*H) : 0, Tt = Tp+T;
            const float theta_scale = std::pow(s.freq_base, -2.0f/(float) hd);
            for (int64_t t=0; t<Tt; ++t)
            for (int64_t h=0; h<H; ++h)
            for (int64_t d=0; d<hd; ++d) {
                float val;
                if (t < Tp) {
                    val = P[d+hd*(h+H*t)];
                } else {
                    const float * xr = x+hd*(h+H*(t-Tp));
                    auto xb = [&](int64_t i) { return xr[i]+(bb ? bb[h*hd+i] : 0.0f); };
                    val = xb(d);
                    if (pos) {
                        const int64_t j = d%half;
                        const float theta = (float) pos[t-Tp]*std::pow(theta_scale, (float) j);
                        const float c = std::cos(theta), sn = std::sin(theta), x0 = xb(j), x1 = xb(j+half);
                        val = d < half ? x0*c-x1*sn : x0*sn+x1*c;
                    }
                }
                store(dst, s.vt_mask ? t+Tt*(d+hd*h) : d+hd*(t+Tt*h), val);
            }
            return;
        }
        case FusedKind::SiluMul: {
            const float * u = (const float *) dst->src[1]->data;
            for (int64_t i=0; i<ggml_nelements(a); ++i) store(dst, i, silu(x[i])*u[i]);
            return;
        }
        case FusedKind::RmsMul: {
            const float * w = (const float *) dst->src[1]->data;
            const int64_t N = a->ne[0], rows = ggml_nrows(a);
            for (int64_t r=0; r<rows; ++r) {
                double s2 = 0.0;
                for (int64_t i=0; i<N; ++i) s2 += (double) x[r*N+i]*x[r*N+i];
                const float scale = 1.0f/std::sqrt((float) (s2/N)+s.eps);
                for (int64_t i=0; i<N; ++i) store(dst, r*N+i, scale*x[r*N+i]*w[i]);
            }
            return;
        }
        case FusedKind::LnAffine:
        case FusedKind::LnModulate:
        case FusedKind::LnPlain: {
            const float * p1 = dst->src[1] ? (const float *) dst->src[1]->data : nullptr;
            const float * p2 = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;
            const int64_t N = a->ne[0], rows = ggml_nrows(a);
            std::vector<float> o((size_t) N);
            for (int64_t r=0; r<rows; ++r) {
                norm_row(x+r*N, o.data(), N, s.eps);
                for (int64_t i=0; i<N; ++i)
                    store(dst, r*N+i, s.kind == FusedKind::LnAffine   ? o[i]*p1[i]+p2[i]
                                    : s.kind == FusedKind::LnModulate ? (o[i]+o[i]*p1[i])+p2[i] : o[i]);
            }
            return;
        }
    }
}

}  // namespace fused_cpu

inline ggml_tensor * fused_node(ggml_context * C, ggml_tensor ** args, int n_args, int64_t ne0, int64_t ne1,
                                int64_t ne2, int64_t ne3, const FusedSpec & s) {
    const ggml_type t = s.out_bf16 ? GGML_TYPE_BF16 : s.out_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    return ggml_custom_4d(C, t, ne0, ne1, ne2, ne3, args, n_args, fused_cpu::run, 1, (void *) fused_intern(s));
}

// y: [n_parts*hd*heads, S*nv] GEMM output without bias. Returns the packed parts;
// fused_part() views one of them.
inline ggml_tensor * fused_qkv_heads(ggml_context * C, ggml_tensor * y, ggml_tensor * bias, ggml_tensor * cos_t,
                                     ggml_tensor * sin_t, int64_t n_parts, int64_t hd, int64_t heads, int64_t S,
                                     int64_t nv, int vt_mask, int rope_mask, int part_mask = 0, bool out_f16 = false) {
    GGML_ASSERT(ggml_is_contiguous(y) && y->ne[0] == n_parts*hd*heads && ggml_nelements(y) == y->ne[0]*S*nv);
    FusedSpec s; s.kind = FusedKind::QkvHeads; s.n_parts = (int32_t) n_parts; s.hd = (int32_t) hd; s.heads = (int32_t) heads;
    s.S = (int32_t) S; s.nv = (int32_t) nv; s.vt_mask = vt_mask; s.rope_mask = rope_mask;
    s.part_mask = part_mask; s.out_f16 = out_f16;
    int n_out = 0;
    for (int p=0; p<n_parts; ++p) n_out += part_mask ? (part_mask >> p) & 1 : 1;
    ggml_tensor * args[4] = { y, bias, cos_t, sin_t };
    return fused_node(C, args, 4, hd*S, heads*nv, n_out, 1, s);
}

// part: index among the parts the node writes (packed).
inline ggml_tensor * fused_part(ggml_context * C, ggml_tensor * qkv, int part, bool transposed) {
    const FusedSpec & s = *(const FusedSpec *) ((const void * const *) qkv->op_params)[2];
    const size_t es = ggml_element_size(qkv), off = (size_t) part*s.hd*s.S*s.heads*s.nv*es;
    if (transposed)
        return ggml_view_4d(C, qkv, s.S, s.hd, s.heads, s.nv, s.S*es, (size_t) s.S*s.hd*es, (size_t) s.S*s.hd*s.heads*es, off);
    return ggml_view_4d(C, qkv, s.hd, s.S, s.heads, s.nv, s.hd*es, (size_t) s.hd*s.S*es, (size_t) s.hd*s.S*s.heads*es, off);
}

inline ggml_tensor * fused_bias_res(ggml_context * C, ggml_tensor * y, ggml_tensor * b, ggml_tensor * r) {
    GGML_ASSERT(ggml_is_contiguous(y) && ggml_is_contiguous(r) && ggml_are_same_shape(y, r) && b->ne[0] == y->ne[0]);
    FusedSpec s; s.kind = FusedKind::BiasRes;
    ggml_tensor * args[3] = { y, b, r };
    return fused_node(C, args, 3, y->ne[0], y->ne[1], y->ne[2], y->ne[3], s);
}

inline ggml_tensor * fused_bias_gelu(ggml_context * C, ggml_tensor * y, ggml_tensor * b, bool out_bf16 = false) {
    GGML_ASSERT(ggml_is_contiguous(y) && b->ne[0] == y->ne[0]);
    FusedSpec s; s.kind = FusedKind::BiasGelu; s.out_bf16 = out_bf16;
    ggml_tensor * args[2] = { y, b };
    return fused_node(C, args, 2, y->ne[0], y->ne[1], y->ne[2], y->ne[3], s);
}

// LayerNorm with affine (w, b), with an adaLN modulation (n + n*sc + sh), or
// plain (p1 = p2 = null).
inline ggml_tensor * fused_ln(ggml_context * C, ggml_tensor * x, ggml_tensor * p1, ggml_tensor * p2, float eps,
                              bool modulate, bool out_bf16 = false) {
    GGML_ASSERT(ggml_is_contiguous(x) && (!p1 || (ggml_is_contiguous(p1) && ggml_is_contiguous(p2) &&
                ggml_nelements(p1) == x->ne[0] && ggml_nelements(p2) == x->ne[0])));
    FusedSpec s; s.eps = eps; s.out_bf16 = out_bf16;
    s.kind = !p1 ? FusedKind::LnPlain : modulate ? FusedKind::LnModulate : FusedKind::LnAffine;
    ggml_tensor * args[3] = { x, p1, p2 };
    return fused_node(C, args, 3, x->ne[0], x->ne[1], x->ne[2], x->ne[3], s);
}

// img: [3, side, side, nv] HWC pixels, F32 in [0, 1] or I8 holding uint8.
// Returns [3*temporal*patch^2, n_patches*nv], the tower's t_patches.
inline ggml_tensor * fused_patchify(ggml_context * C, ggml_tensor * img, int64_t patch, int64_t temporal, int64_t merge,
                                    const float mean[3], const float std_[3]) {
    GGML_ASSERT(ggml_is_contiguous(img) && img->ne[0] == 3 && img->ne[1] == img->ne[2] &&
                (img->type == GGML_TYPE_F32 || img->type == GGML_TYPE_I8));
    FusedSpec s; s.kind = FusedKind::Patchify; s.side = (int32_t) img->ne[1]; s.patch = (int32_t) patch;
    s.temporal = (int32_t) temporal; s.merge = (int32_t) merge;
    for (int c=0; c<3; ++c) { s.mean[c] = mean[c]; s.std[c] = std_[c]; }
    const int64_t gw = img->ne[1]/patch;
    ggml_tensor * args[1] = { img };
    return fused_node(C, args, 1, 3*temporal*patch*patch, gw*gw*img->ne[3], 1, 1, s);
}

// x: [hd, heads, T] projection output; returns [hd, T, heads] for attention.
inline ggml_tensor * fused_qk_norm_rope(ggml_context * C, ggml_tensor * x, ggml_tensor * w, ggml_tensor * pos, float eps,
                                        const int sections[4], float freq_base, bool out_f16) {
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_nelements(w) == x->ne[0] && pos->type == GGML_TYPE_I32 &&
                ggml_nelements(pos) == 4*x->ne[2] && x->ne[0]%64 == 0);
    FusedSpec s; s.kind = FusedKind::QkNormRope; s.hd = (int32_t) x->ne[0]; s.heads = (int32_t) x->ne[1];
    s.S = (int32_t) x->ne[2]; s.eps = eps; s.out_f16 = out_f16; s.freq_base = freq_base;
    for (int i=0; i<4; ++i) s.sect[i] = sections[i];
    ggml_tensor * args[3] = { x, w, pos };
    return fused_node(C, args, 3, x->ne[0], x->ne[2], x->ne[1], 1, s);
}

// W [K, N] BF16, x [K, S*nv] BF16 or F32 -> [N, S*nv] F32, one GEMM per view.
// An F32 x is rounded to BF16 inside the op, as ggml's own BF16 matmul does.
inline ggml_tensor * fused_gemm_views(ggml_context * C, ggml_tensor * W, ggml_tensor * x, int64_t S, int64_t nv) {
    GGML_ASSERT(W->type == GGML_TYPE_BF16 && (x->type == GGML_TYPE_BF16 || x->type == GGML_TYPE_F32) &&
                ggml_is_contiguous(W) && ggml_is_contiguous(x) &&
                W->ne[0] == x->ne[0] && x->ne[1] == S*nv && ggml_nrows(x) == S*nv);
    FusedSpec s; s.kind = FusedKind::GemmViews; s.S = (int32_t) S; s.nv = (int32_t) nv;
    ggml_tensor * args[2] = { W, x };
    // Pass all four slots: fused_node checks src[0..3]; the CUDA handler
    // accepts BF16 sources for this kind.
    return fused_node(C, args, 2, W->ne[1], S*nv, 1, 1, s);
}

// Q [hd, Tq, H], K [hd, Tkv, Hkv], V^T [Tkv, hd, Hkv] (contiguous F32), H a
// multiple of Hkv (query head h reads K/V head h/(H/Hkv), as ggml's broadcast
// mul_mat). mask: [Tkv, >= Tq] F32 or F16, added to the scaled scores like
// soft_max_ext's. Returns [hd*H, Tq], BF16 when it feeds a BF16 projection.
// nv > 1: the H batched matrices are heads x views (view-major), and the rows
// are laid out [hd*H/nv, Tq*nv] like attention_rows().
inline ggml_tensor * fused_attn_exact(ggml_context * C, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * Vt, float scale,
                                      bool out_bf16, int64_t nv = 1, ggml_tensor * mask = nullptr) {
    GGML_ASSERT(ggml_is_contiguous(Q) && ggml_is_contiguous(K) && ggml_is_contiguous(Vt) &&
                Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 && Vt->type == GGML_TYPE_F32 &&
                K->ne[0] == Q->ne[0] && Q->ne[2]%K->ne[2] == 0 && Vt->ne[0] == K->ne[1] && Vt->ne[1] == K->ne[0] &&
                Vt->ne[2] == K->ne[2] && K->ne[1] <= 4096);
    GGML_ASSERT(!mask || ((mask->type == GGML_TYPE_F32 || mask->type == GGML_TYPE_F16) && ggml_is_contiguous_rows(mask) &&
                          mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 && mask->ne[3] == 1));
    FusedSpec s; s.kind = FusedKind::AttnExact; s.hd = (int32_t) Q->ne[0]; s.S = (int32_t) Q->ne[1];
    s.heads = (int32_t) Q->ne[2]; s.eps = scale; s.out_bf16 = out_bf16; s.nv = (int32_t) nv;
    GGML_ASSERT(nv >= 1 && Q->ne[2]%nv == 0 && (nv == 1 || (!mask && K->ne[2] == Q->ne[2])));
    ggml_tensor * args[4] = { Q, K, Vt, mask };
    return fused_node(C, args, 4, Q->ne[0]*Q->ne[2]/nv, Q->ne[1]*nv, 1, 1, s);
}

inline ggml_tensor * fused_rms_mul(ggml_context * C, ggml_tensor * x, ggml_tensor * w, float eps, bool out_bf16) {
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(w) && ggml_nelements(w) == x->ne[0]);
    FusedSpec s; s.kind = FusedKind::RmsMul; s.eps = eps; s.out_bf16 = out_bf16;
    ggml_tensor * args[2] = { x, w };
    return fused_node(C, args, 2, x->ne[0], x->ne[1], x->ne[2], x->ne[3], s);
}

inline ggml_tensor * fused_silu_mul(ggml_context * C, ggml_tensor * g, ggml_tensor * u, bool out_bf16) {
    GGML_ASSERT(ggml_is_contiguous(g) && ggml_is_contiguous(u) && ggml_are_same_shape(g, u));
    FusedSpec s; s.kind = FusedKind::SiluMul; s.out_bf16 = out_bf16;
    ggml_tensor * args[2] = { g, u };
    return fused_node(C, args, 2, g->ne[0], g->ne[1], g->ne[2], g->ne[3], s);
}

// ggml's geglu_split: gelu (tanh form) of g times u.
inline ggml_tensor * fused_gelu_mul(ggml_context * C, ggml_tensor * g, ggml_tensor * u, bool out_bf16) {
    GGML_ASSERT(ggml_is_contiguous(g) && ggml_is_contiguous(u) && ggml_are_same_shape(g, u));
    FusedSpec s; s.kind = FusedKind::GeluMul; s.out_bf16 = out_bf16;
    ggml_tensor * args[2] = { g, u };
    return fused_node(C, args, 2, g->ne[0], g->ne[1], g->ne[2], g->ne[3], s);
}

// r + y*g (g broadcast over rows): a gated residual, ggml_add(r, ggml_mul(y, g)).
inline ggml_tensor * fused_gate_res(ggml_context * C, ggml_tensor * y, ggml_tensor * g, ggml_tensor * r) {
    GGML_ASSERT(ggml_is_contiguous(y) && ggml_is_contiguous(r) && ggml_are_same_shape(y, r) &&
                ggml_is_contiguous(g) && ggml_nelements(g) == y->ne[0]);
    FusedSpec s; s.kind = FusedKind::GateRes;
    ggml_tensor * args[3] = { y, g, r };
    return fused_node(C, args, 3, y->ne[0], y->ne[1], y->ne[2], y->ne[3], s);
}

// ggml_gelu_erf(ggml_add(y, b)).
inline ggml_tensor * fused_bias_gelu_erf(ggml_context * C, ggml_tensor * y, ggml_tensor * b, bool out_bf16 = false) {
    GGML_ASSERT(ggml_is_contiguous(y) && b->ne[0] == y->ne[0]);
    FusedSpec s; s.kind = FusedKind::BiasGeluErf; s.out_bf16 = out_bf16;
    ggml_tensor * args[2] = { y, b };
    return fused_node(C, args, 2, y->ne[0], y->ne[1], y->ne[2], y->ne[3], s);
}

// r + (y + b)*g (b, g broadcast over rows): a bias, a layer scale and the
// residual, ggml_add(r, ggml_mul(ggml_add(y, b), g)).
inline ggml_tensor * fused_bias_gate_res(ggml_context * C, ggml_tensor * y, ggml_tensor * b, ggml_tensor * g, ggml_tensor * r) {
    GGML_ASSERT(ggml_is_contiguous(y) && ggml_is_contiguous(r) && ggml_are_same_shape(y, r) &&
                ggml_nelements(b) == y->ne[0] && ggml_is_contiguous(g) && ggml_nelements(g) == y->ne[0]);
    FusedSpec s; s.kind = FusedKind::BiasGateRes;
    ggml_tensor * args[4] = { y, b, g, r };
    return fused_node(C, args, 4, y->ne[0], y->ne[1], y->ne[2], y->ne[3], s);
}

// adaRMS: n + n*sc + sh with n = ggml_rms_norm(x), in ggml's op order.
inline ggml_tensor * fused_rms_modulate(ggml_context * C, ggml_tensor * x, ggml_tensor * sc, ggml_tensor * sh, float eps,
                                        bool out_bf16) {
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(sc) && ggml_is_contiguous(sh) &&
                ggml_nelements(sc) == x->ne[0] && ggml_nelements(sh) == x->ne[0]);
    FusedSpec s; s.kind = FusedKind::RmsModulate; s.eps = eps; s.out_bf16 = out_bf16;
    ggml_tensor * args[3] = { x, sc, sh };
    return fused_node(C, args, 3, x->ne[0], x->ne[1], x->ne[2], x->ne[3], s);
}

// Per-head attention operand from x [hd*heads, T]: [hd, Tp+T, heads], or
// transposed [Tp+T, hd, heads]. The first Tp tokens come from P (cached,
// already rotated); x gets bias b and ggml's NEOX RoPE at pos when given.
// Replaces ggml's add + rope + concat + permute + cont.
inline ggml_tensor * fused_heads_rope(ggml_context * C, ggml_tensor * x, ggml_tensor * pos, ggml_tensor * P, int64_t hd,
                                      int64_t heads, float freq_base, bool transposed, ggml_tensor * b = nullptr) {
    GGML_ASSERT(ggml_is_contiguous(x) && x->type == GGML_TYPE_F32 && ggml_nelements(x)%(hd*heads) == 0 && hd%2 == 0);
    const int64_t T = ggml_nelements(x)/(hd*heads), Tp = P ? ggml_nelements(P)/(hd*heads) : 0;
    GGML_ASSERT(!pos || (pos->type == GGML_TYPE_I32 && ggml_nelements(pos) == T));
    GGML_ASSERT(!P || (ggml_is_contiguous(P) && P->type == GGML_TYPE_F32 && ggml_nelements(P) == Tp*hd*heads));
    GGML_ASSERT(!b || (ggml_is_contiguous(b) && b->type == GGML_TYPE_F32 && ggml_nelements(b) == hd*heads));
    FusedSpec s; s.kind = FusedKind::HeadsRope; s.hd = (int32_t) hd; s.heads = (int32_t) heads; s.S = (int32_t) T;
    s.vt_mask = transposed ? 1 : 0; s.rope_mask = pos ? 1 : 0; s.freq_base = freq_base;
    ggml_tensor * args[4] = { x, pos, P, b };
    return transposed ? fused_node(C, args, 4, Tp+T, hd, heads, 1, s) : fused_node(C, args, 4, hd, Tp+T, heads, 1, s);
}

// A GEMM operand may be produced as BF16 only when the weight is BF16: cuBLAS
// then runs on it unchanged, exactly as on ggml's own conversion of it.
inline bool bf16_operand(const ggml_tensor * W) {
    return W->type == GGML_TYPE_BF16;
}

// The same for an operand of `rows` rows (see cublas_rows).
inline bool bf16_operand(const ggml_tensor * W, int64_t rows) {
    return W->type == GGML_TYPE_BF16 && rows > 16;
}

// The attention output [hd, T, heads, nv] permuted to rows of heads*hd and
// stored BF16 for the output projection: one copy kernel instead of ggml's
// cont + convert.
inline ggml_tensor * heads_to_rows_bf16(ggml_context * C, ggml_tensor * kqv, int64_t dim, int64_t T, int64_t nv) {
    ggml_tensor * p = ggml_permute(C, kqv, 0, 2, 1, 3);
    return ggml_reshape_2d(C, ggml_cpy(C, p, ggml_new_tensor_4d(C, GGML_TYPE_BF16, p->ne[0], p->ne[1], p->ne[2], p->ne[3])),
                           dim, T*nv);
}

// layers/attn.h attention() with its output in rows of dim, as BF16 when it
// feeds a BF16 projection.
inline ggml_tensor * attention_rows(ggml_context * C, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V, float scale,
                                    int64_t dim, int64_t T, int64_t nv, bool out_bf16, ggml_tensor * mask = nullptr) {
    ggml_tensor * kq = ggml_mul_mat(C, K, Q);
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    ggml_tensor * kqv = ggml_mul_mat(C, V, ggml_soft_max_ext(C, kq, mask, scale, 0.0f));
    if (out_bf16)
        return heads_to_rows_bf16(C, kqv, dim, T, nv);
    return ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, kqv, 0, 2, 1, 3)), dim, T*nv);
}

// attention_rows() through fused_attn_exact when the operands allow it
// (contiguous F32, V transposed, heads x views batched as one dim);
// VLA_ATTN_EXACT=0 keeps the ggml chain everywhere.
inline bool attn_exact_enabled() {
    static const bool on = env_flag("VLA_ATTN_EXACT", true);
    return on;
}

inline ggml_tensor * attention_rows_exact(ggml_context * C, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * Vt, float scale,
                                          int64_t dim, int64_t T, int64_t nv, bool out_bf16) {
    const int64_t hd = Q->ne[0], heads = Q->ne[2]*Q->ne[3];
    if (!attn_exact_enabled() || !ggml_is_contiguous(Q) || !ggml_is_contiguous(K) || !ggml_is_contiguous(Vt) ||
        Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F32 || Vt->type != GGML_TYPE_F32 || K->ne[1] > 4096 ||
        K->ne[2]*K->ne[3] != heads || Vt->ne[0] != K->ne[1] || Vt->ne[1] != hd || hd*heads != dim*nv)
        return attention_rows(C, Q, K, Vt, scale, dim, T, nv, out_bf16);
    return fused_attn_exact(C, ggml_reshape_3d(C, Q, hd, Q->ne[1], heads), ggml_reshape_3d(C, K, hd, K->ne[1], heads),
                            ggml_reshape_3d(C, Vt, Vt->ne[0], hd, heads), scale, out_bf16, nv);
}

// ggml-cuda multiplies a BF16 weight by an F32 operand of up to 16 rows with
// its own kernels (mul_mat_vec_f for one row, mul_mat_f up to 16 on Ampere),
// and by more rows with cuBLAS - the call gemm_exact reproduces. A fused path
// may hand a GEMM a BF16 operand only where ggml would have used cuBLAS.
inline bool cublas_rows(int64_t rows) {
    return rows > 16;
}

// A BF16-weight GEMM through fused_gemm_views (one view): cuBLASLt with the
// fastest algorithm that reproduces ggml's cublasGemmEx bit for bit, or that
// call itself. Only used on the fused (CUDA) paths, and only where ggml uses
// cuBLAS (cublas_rows) on a 2D operand: a batched one is ggml's batched cuBLAS
// call, which one GEMM over all rows does not reproduce. VLA_GEMM_EXACT=0
// keeps ggml_mul_mat.
inline ggml_tensor * mm_exact(ggml_context * C, ggml_tensor * W, ggml_tensor * x) {
    static const bool on = env_flag("VLA_GEMM_EXACT", true);
    if (x->type == GGML_TYPE_BF16)
        GGML_ASSERT(cublas_rows(ggml_nrows(x)) && "a BF16 operand where ggml would not use cuBLAS");
    if (!on || W->type != GGML_TYPE_BF16 || !ggml_is_contiguous(W) || !ggml_is_contiguous(x) ||
        (x->type != GGML_TYPE_BF16 && x->type != GGML_TYPE_F32) || x->ne[0] != W->ne[0] ||
        x->ne[2]*x->ne[3] != 1 || !cublas_rows(x->ne[1]))
        return ggml_mul_mat(C, W, x);
    return fused_gemm_views(C, W, x, x->ne[1], 1);
}

}  // namespace vla
