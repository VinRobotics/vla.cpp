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

// ACT (LeRobot policies/act, from Zhao et al. 2023): a ResNet over each camera,
// a transformer encoder over [latent, state, image features], and a decoder
// whose chunk_size zero queries cross-attend the encoder output. No language
// and no denoising loop. At inference the VAE latent is zeros, so the latent
// token is its projection's bias and the VAE encoder is not converted.
//
// The ResNet runs at whatever size the cameras send, like the PyTorch policy,
// over all views as one batch; the graph is cached per input size.

#include "arch.h"
#include "backend.h"
#include "gguf.h"
#include "gguf_reader.h"
#include "loader.h"
#include "model.h"
#include "options.h"
#include "scratch_ctx.h"
#include "layers/attn.h"
#include "layers/conv.h"
#include "layers/ffn.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "modules/preprocess.h"

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace vla {
namespace {

constexpr float kLnEps   = 1e-5f;  // nn.LayerNorm default
constexpr float kNormEps = 1e-8f;  // LeRobot's NormalizerProcessorStep: (x - mean) / (std + eps)

struct ConvW {
    ggml_tensor *w = nullptr, *b = nullptr;
};

struct BlockW {
    ConvW conv1, conv2, down;
    int   stride = 1;
};

// q and k share their input (tokens plus position), so they are one GEMM.
struct SelfAttnW {
    ggml_tensor *qk_w, *qk_b, *v_w, *v_b, *o_w, *o_b;
};

struct CrossAttnW {
    ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
};

struct EncLayerW {
    SelfAttnW   attn;
    ggml_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b, *ln1_w, *ln1_b, *ln2_w, *ln2_b;
};

struct DecLayerW {
    SelfAttnW   attn;
    CrossAttnW  cross;
    ggml_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b, *ln1_w, *ln1_b, *ln2_w, *ln2_b, *ln3_w, *ln3_b;
};

// torchvision ResNet output size of a 3x3 (or 7x7) conv / pool at stride 2.
int64_t down2(int64_t n, int64_t k, int64_t pad) {
    return (n + 2*pad - k) / 2 + 1;
}

// ggml-cuda's direct convolution is an implicit GEMM only where it has MMA
// (Turing and newer). Before that, Volta's Jetson Xavier included, it falls back
// to a naive kernel several times slower than im2col.
bool cuda_conv_has_mma() {
#ifdef GGML_USE_CUDA
    int major = 0, minor = 0;
    const int dev = backend_device_index();
    return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
           major*10 + minor >= 75;
#else
    return false;
#endif
}

}  // namespace

struct ActModelArch : public ModelArchBase {
    ActModelArch() : ModelArchBase(Arch::ACT) {}
    ~ActModelArch() override {
        graph.release();
        if (weight_buf)  ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend)     ggml_backend_free(backend);
    }

    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             mt          = GGML_TYPE_F32;
    // CUDA (Turing+) runs the ResNet as ggml's direct convolution on F16
    // kernels: an implicit GEMM on tensor cores with F32 accumulation and no
    // im2col buffer. Elsewhere the kernels stay F32 and go through im2col, which
    // is the faster of the two on CPU.
    bool                  conv_direct = false;

    int64_t dim = 512, heads = 8, ff = 3200, enc_layers = 4, dec_layers = 1, chunk = 100;
    int64_t state_dim = 0, action_dim = 0, n_views = 1, img_h = 480, img_w = 640;
    bool    pre_norm = false, gelu = false;
    std::vector<int64_t> blocks;
    std::string cameras;  // the training feature names, in the order views must arrive

    ConvW               stem;
    std::vector<BlockW> res;
    ggml_tensor *img_proj_w = nullptr, *img_proj_b = nullptr;
    ggml_tensor *latent_tok = nullptr, *state_w = nullptr, *state_b = nullptr;
    std::vector<EncLayerW> enc;
    ggml_tensor *enc_norm_w = nullptr, *enc_norm_b = nullptr;
    std::vector<DecLayerW> dec;
    ggml_tensor *dec_norm_w = nullptr, *dec_norm_b = nullptr, *dec_pos = nullptr;
    ggml_tensor *head_w = nullptr, *head_b = nullptr;

    std::vector<float> pos_1d;  // [n_1d, dim], host copy for the encoder position table
    // img_std and the other stds already carry LeRobot's eps.
    std::vector<float> img_mean, img_std, state_mean, state_std, action_mean, action_std;

    struct Key {
        int64_t h = -1, w = -1;
        bool operator==(const Key & o) const { return h == o.h && w == o.w; }
    };
    struct IO {
        ggml_tensor *pixels = nullptr, *state = nullptr, *enc_pos = nullptr, *actions = nullptr;
    };
    graph_cache<Key, IO> graph;
    std::vector<float>   pixels;  // host staging for io.pixels, kept so a call does not fault in fresh pages

    int64_t n_1d() const { return state_dim > 0 ? 2 : 1; }
    static void feature_size(int64_t h, int64_t w, int64_t & fh, int64_t & fw) {
        fh = down2(down2(h, 7, 3), 3, 1);  // conv1, maxpool
        fw = down2(down2(w, 7, 3), 3, 1);
        for (int i = 0; i < 3; ++i) {      // layer2..4 open with a stride-2 block
            fh = down2(fh, 3, 1);
            fw = down2(fw, 3, 1);
        }
    }

    ggml_cgraph *      build(ggml_context * C, IO & io, int64_t h, int64_t w) const;
    std::vector<float> enc_pos_table(int64_t fh, int64_t fw) const;

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

// [w, h, c, view] to [ow, oh, oc, view]. The kernel's type picks the path; see
// ActModelArch::conv_direct.
ggml_tensor * conv2d(ggml_context * C, const ConvW & cw, ggml_tensor * x, int stride, int pad) {
    ggml_tensor * y = cw.w->type == GGML_TYPE_F16
        ? ggml_conv_2d_direct(C, cw.w, x, stride, stride, pad, pad, 1, 1)
        : conv_2d_f32(C, cw.w, x, stride, pad);
    return ggml_add(C, y, ggml_reshape_4d(C, cw.b, 1, 1, cw.b->ne[0], 1));
}

// [D, T] attention with separate query and key/value sources. q_in and k_in
// already carry their position embeddings; v_in does not.
ggml_tensor * mha(ggml_context * C, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                  int64_t hd, int64_t heads, int64_t Tq, int64_t Tk) {
    const float scale = 1.0f / std::sqrt((float) hd);
    if (flash_attn_enabled()) {
        auto h = [&](ggml_tensor * p, int64_t T) {
            return ggml_permute(C, ggml_reshape_3d(C, ggml_cont(C, p), hd, heads, T), 0, 2, 1, 3);
        };
        return flash_attention(C, h(q, Tq), h(k, Tk), h(v, Tk), nullptr, scale);
    }
    return ggml_reshape_2d(C, attention(C, to_heads(C, ggml_cont(C, q), hd, heads, Tq),
                                        to_heads(C, ggml_cont(C, k), hd, heads, Tk),
                                        to_heads_v(C, ggml_cont(C, v), hd, heads, Tk),
                                        nullptr, scale, hd*heads, Tq), hd*heads, Tq);
}

ggml_tensor * self_attn(ggml_context * C, const SelfAttnW & a, ggml_tensor * x, ggml_tensor * pos,
                        int64_t dim, int64_t heads, int64_t T) {
    ggml_tensor * qk = linear(C, a.qk_w, a.qk_b, ggml_add(C, x, pos));
    ggml_tensor * q  = ggml_view_2d(C, qk, dim, T, qk->nb[1], 0);
    ggml_tensor * k  = ggml_view_2d(C, qk, dim, T, qk->nb[1], (size_t) dim*ggml_element_size(qk));
    ggml_tensor * v  = linear(C, a.v_w, a.v_b, x);
    return linear(C, a.o_w, a.o_b, mha(C, q, k, v, dim / heads, heads, T, T));
}

ggml_tensor * cross_attn(ggml_context * C, const CrossAttnW & a, ggml_tensor * x, ggml_tensor * x_pos,
                         ggml_tensor * mem, ggml_tensor * mem_pos, int64_t dim, int64_t heads,
                         int64_t Tq, int64_t Tk) {
    ggml_tensor * q = linear(C, a.q_w, a.q_b, ggml_add(C, x, x_pos));
    ggml_tensor * k = linear(C, a.k_w, a.k_b, ggml_add(C, mem, mem_pos));
    ggml_tensor * v = linear(C, a.v_w, a.v_b, mem);
    return linear(C, a.o_w, a.o_b, mha(C, q, k, v, dim / heads, heads, Tq, Tk));
}

}  // namespace

ggml_cgraph * ActModelArch::build(ggml_context * C, IO & io, int64_t h, int64_t w) const {
    int64_t fh = 0, fw = 0;
    feature_size(h, w, fh, fw);
    const int64_t NF = fh*fw, T = n_1d() + n_views*NF;
    auto ffn = [&](ggml_tensor * f1w, ggml_tensor * f1b, ggml_tensor * f2w, ggml_tensor * f2b, ggml_tensor * x) {
        return gelu ? ffn_gelu_erf(C, f1w, f1b, f2w, f2b, x) : ffn_relu(C, f1w, f1b, f2w, f2b, x);
    };

    // ResNet over the views as one batch: [w, h, c, view]. The stem's ReLU
    // commutes with the max pool, so it runs on the pooled map, a quarter the size.
    io.pixels = ggml_new_tensor_4d(C, GGML_TYPE_F32, w, h, 3, n_views);
    ggml_set_input(io.pixels);
    ggml_tensor * x = ggml_pool_2d(C, conv2d(C, stem, io.pixels, 2, 3), GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1);
    x = ggml_relu(C, x);
    for (const BlockW & b : res) {
        ggml_tensor * y = ggml_relu(C, conv2d(C, b.conv1, x, b.stride, 1));
        y = conv2d(C, b.conv2, y, 1, 1);
        ggml_tensor * skip = b.down.w ? conv2d(C, b.down, x, b.stride, 0) : x;
        x = ggml_relu(C, ggml_add(C, y, skip));
    }
    // 1x1 projection to the model width, tokens in (view h w) order. Only this
    // last, small feature map is transposed to channels-first.
    const int64_t fc = x->ne[2];
    ggml_tensor * f = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, x, NF, fc, n_views), 1, 0, 2, 3));
    ggml_tensor * img = linear(C, img_proj_w, img_proj_b, ggml_reshape_2d(C, f, fc, NF*n_views));

    ggml_tensor * tok = ggml_reshape_2d(C, latent_tok, dim, 1);
    if (state_dim > 0) {
        io.state = ggml_new_tensor_1d(C, GGML_TYPE_F32, state_dim);
        ggml_set_input(io.state);
        tok = ggml_concat(C, tok, ggml_reshape_2d(C, linear(C, state_w, state_b, io.state), dim, 1), 1);
    }
    tok = ggml_concat(C, tok, img, 1);

    // Uploaded once per graph, so it is an output too: gallocr never reuses an
    // output's memory for a later node.
    io.enc_pos = ggml_new_tensor_2d(C, GGML_TYPE_F32, dim, T);
    ggml_set_input(io.enc_pos);
    ggml_set_output(io.enc_pos);
    for (const EncLayerW & l : enc) {
        if (pre_norm) {
            ggml_tensor * n = layer_norm(C, tok, l.ln1_w, l.ln1_b, kLnEps);
            tok = ggml_add(C, tok, self_attn(C, l.attn, n, io.enc_pos, dim, heads, T));
            n   = layer_norm(C, tok, l.ln2_w, l.ln2_b, kLnEps);
            tok = ggml_add(C, tok, ffn(l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b, n));
        } else {
            tok = layer_norm(C, ggml_add(C, tok, self_attn(C, l.attn, tok, io.enc_pos, dim, heads, T)),
                             l.ln1_w, l.ln1_b, kLnEps);
            tok = layer_norm(C, ggml_add(C, tok, ffn(l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b, tok)),
                             l.ln2_w, l.ln2_b, kLnEps);
        }
    }
    if (enc_norm_w)
        tok = layer_norm(C, tok, enc_norm_w, enc_norm_b, kLnEps);

    // Decoder: the queries start at zero and carry only their position.
    ggml_tensor * a = ggml_scale(C, dec_pos, 0.0f);
    for (const DecLayerW & l : dec) {
        if (pre_norm) {
            ggml_tensor * n = layer_norm(C, a, l.ln1_w, l.ln1_b, kLnEps);
            a = ggml_add(C, a, self_attn(C, l.attn, n, dec_pos, dim, heads, chunk));
            n = layer_norm(C, a, l.ln2_w, l.ln2_b, kLnEps);
            a = ggml_add(C, a, cross_attn(C, l.cross, n, dec_pos, tok, io.enc_pos, dim, heads, chunk, T));
            n = layer_norm(C, a, l.ln3_w, l.ln3_b, kLnEps);
            a = ggml_add(C, a, ffn(l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b, n));
        } else {
            a = layer_norm(C, ggml_add(C, a, self_attn(C, l.attn, a, dec_pos, dim, heads, chunk)),
                           l.ln1_w, l.ln1_b, kLnEps);
            a = layer_norm(C, ggml_add(C, a, cross_attn(C, l.cross, a, dec_pos, tok, io.enc_pos, dim, heads, chunk, T)),
                           l.ln2_w, l.ln2_b, kLnEps);
            a = layer_norm(C, ggml_add(C, a, ffn(l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b, a)), l.ln3_w, l.ln3_b, kLnEps);
        }
    }
    a = layer_norm(C, a, dec_norm_w, dec_norm_b, kLnEps);
    io.actions = linear(C, head_w, head_b, a);
    ggml_set_output(io.actions);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 4096, false);
    ggml_build_forward_expand(gf, io.actions);
    return gf;
}

// Encoder positions: the learned 1D rows for the latent and state tokens, then
// ACTSinusoidalPositionEmbedding2d over the feature grid, the same for every
// camera. Channels are [y | x], each half interleaving sin and cos of the
// normalized coordinate (index+1)/(n+eps)*2pi over periods 10000^(2*(i/2)/half).
std::vector<float> ActModelArch::enc_pos_table(int64_t fh, int64_t fw) const {
    const int64_t half = dim / 2, NF = fh*fw, T = n_1d() + n_views*NF;
    std::vector<float> table((size_t) dim*T, 0.0f);
    std::copy(pos_1d.begin(), pos_1d.end(), table.begin());
    const float two_pi = 6.28318530717958647692f, eps = 1e-6f;
    std::vector<float> inv((size_t) half);
    for (int64_t i = 0; i < half; ++i)
        inv[(size_t) i] = std::pow(10000.0f, (float) (2*(i/2)) / (float) half);
    std::vector<float> grid((size_t) dim*NF);
    for (int64_t y = 0; y < fh; ++y)
        for (int64_t x = 0; x < fw; ++x) {
            const float yr = (float) (y + 1) / ((float) fh + eps) * two_pi;
            const float xr = (float) (x + 1) / ((float) fw + eps) * two_pi;
            float * row = grid.data() + (size_t) (y*fw + x)*dim;
            for (int64_t i = 0; i < half; ++i) {
                row[i]        = (i % 2 == 0) ? std::sin(yr / inv[(size_t) i]) : std::cos(yr / inv[(size_t) i]);
                row[half + i] = (i % 2 == 0) ? std::sin(xr / inv[(size_t) i]) : std::cos(xr / inv[(size_t) i]);
            }
        }
    for (int64_t v = 0; v < n_views; ++v)
        std::copy(grid.begin(), grid.end(), table.begin() + (size_t) (n_1d() + v*NF)*dim);
    return table;
}

namespace {

bool load_config(const gguf_reader & g, ActModelArch & m) {
    auto U = [&](const char * k, int64_t & dst) {
        char key[96];
        std::snprintf(key, sizeof(key), "act.%s", k);
        if (g.has(key))
            dst = (int64_t) g.u32(key);
    };
    int64_t pre = 0, gelu = 0;
    U("dim_model", m.dim);              U("n_heads", m.heads);           U("dim_feedforward", m.ff);
    U("n_encoder_layers", m.enc_layers); U("n_decoder_layers", m.dec_layers);
    U("chunk_size", m.chunk);           U("state_dim", m.state_dim);     U("action_dim", m.action_dim);
    U("num_views", m.n_views);          U("image_height", m.img_h);      U("image_width", m.img_w);
    U("pre_norm", pre);                 U("gelu", gelu);
    m.pre_norm = pre != 0;
    m.gelu     = gelu != 0;

    const int64_t kb = gguf_find_key(g.gctx, "act.backbone_blocks");
    if (kb < 0 || gguf_get_kv_type(g.gctx, kb) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g.gctx, kb) != GGUF_TYPE_INT32 ||
        gguf_get_arr_n(g.gctx, kb) != 4) {
        std::fprintf(stderr, "vla(act): missing or malformed act.backbone_blocks\n");
        return false;
    }
    const int32_t * nb = (const int32_t *) gguf_get_arr_data(g.gctx, kb);
    m.blocks.assign(nb, nb + 4);

    const int64_t kc = gguf_find_key(g.gctx, "act.cameras");
    if (kc >= 0 && gguf_get_kv_type(g.gctx, kc) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g.gctx, kc) == GGUF_TYPE_STRING)
        for (size_t i = 0; i < gguf_get_arr_n(g.gctx, kc); ++i)
            m.cameras += std::string(i ? ", " : "") + gguf_get_arr_str(g.gctx, kc, i);

    bool ok = m.state_dim >= 0 && m.dim % m.heads == 0 && m.dim % 4 == 0;
    for (int64_t v : { m.dim, m.heads, m.ff, m.enc_layers, m.dec_layers, m.chunk, m.action_dim, m.n_views,
                       m.blocks[0], m.blocks[1], m.blocks[2], m.blocks[3] })
        ok = ok && v >= 1;
    if (!ok) {
        std::fprintf(stderr, "vla(act): inconsistent dimensions in GGUF metadata\n");
        return false;
    }
    return true;
}

SelfAttnW load_self_attn(WeightLoader & L, const std::string & p) {
    SelfAttnW a{};
    a.qk_w = L.fuse_gemm((p + "_qk.weight").c_str(), { p + "_q.weight", p + "_k.weight" });
    a.qk_b = L.fuse_f32((p + "_qk.bias").c_str(), { p + "_q.bias", p + "_k.bias" });
    a.v_w  = L.gemm("%s_v.weight", p.c_str());
    a.v_b  = L.f32("%s_v.bias", p.c_str());
    a.o_w  = L.gemm("%s_o.weight", p.c_str());
    a.o_b  = L.f32("%s_o.bias", p.c_str());
    return a;
}

bool load_weights(ActModelArch & m, gguf_reader & g) {
    ggml_init_params wp = { (size_t) 4*1024*1024, nullptr, true };
    m.ctx_weights = ggml_init(wp);
    if (!m.ctx_weights)
        return false;
    WeightLoader L("act", g, m.ctx_weights, m.mt);

    const ggml_type kt = m.conv_direct ? GGML_TYPE_F16 : GGML_TYPE_F32;
    auto conv = [&](const std::string & p) {
        return ConvW{ L.typed(kt, "%s.weight", p.c_str()), L.f32("%s.bias", p.c_str()) };
    };
    m.stem = conv("bb.conv1");
    for (int64_t li = 0; li < 4; ++li)
        for (int64_t bi = 0; bi < m.blocks[(size_t) li]; ++bi) {
            const std::string p = "bb.layer" + std::to_string(li + 1) + "." + std::to_string(bi);
            BlockW b;
            b.stride = (li > 0 && bi == 0) ? 2 : 1;
            b.conv1  = conv(p + ".conv1");
            b.conv2  = conv(p + ".conv2");
            if (gguf_find_tensor(g.gctx, (p + ".down.weight").c_str()) >= 0)
                b.down = conv(p + ".down");
            m.res.push_back(b);
        }
    m.img_proj_w = L.gemm("img_proj.weight");
    m.img_proj_b = L.f32("img_proj.bias");

    m.latent_tok = L.f32("latent_tok");
    if (m.state_dim > 0) {
        m.state_w = L.gemm("state_proj.weight");
        m.state_b = L.f32("state_proj.bias");
    }

    m.enc.resize((size_t) m.enc_layers);
    for (int64_t i = 0; i < m.enc_layers; ++i) {
        EncLayerW & e = m.enc[(size_t) i];
        const std::string p = "enc.blk." + std::to_string(i);
        e.attn  = load_self_attn(L, p + ".attn");
        e.fc1_w = L.gemm("%s.fc1.weight", p.c_str());
        e.fc1_b = L.f32("%s.fc1.bias", p.c_str());
        e.fc2_w = L.gemm("%s.fc2.weight", p.c_str());
        e.fc2_b = L.f32("%s.fc2.bias", p.c_str());
        e.ln1_w = L.f32("%s.ln1.weight", p.c_str());
        e.ln1_b = L.f32("%s.ln1.bias", p.c_str());
        e.ln2_w = L.f32("%s.ln2.weight", p.c_str());
        e.ln2_b = L.f32("%s.ln2.bias", p.c_str());
    }
    if (m.pre_norm) {
        m.enc_norm_w = L.f32("enc.norm.weight");
        m.enc_norm_b = L.f32("enc.norm.bias");
    }

    m.dec.resize((size_t) m.dec_layers);
    for (int64_t i = 0; i < m.dec_layers; ++i) {
        DecLayerW & d = m.dec[(size_t) i];
        const std::string p = "dec.blk." + std::to_string(i);
        d.attn = load_self_attn(L, p + ".attn");
        for (const char * part : { "q", "k", "v", "o" }) {
            ggml_tensor * w = L.gemm("%s.cross_%s.weight", p.c_str(), part);
            ggml_tensor * b = L.f32("%s.cross_%s.bias", p.c_str(), part);
            switch (part[0]) {
                case 'q': d.cross.q_w = w; d.cross.q_b = b; break;
                case 'k': d.cross.k_w = w; d.cross.k_b = b; break;
                case 'v': d.cross.v_w = w; d.cross.v_b = b; break;
                default:  d.cross.o_w = w; d.cross.o_b = b; break;
            }
        }
        d.fc1_w = L.gemm("%s.fc1.weight", p.c_str());
        d.fc1_b = L.f32("%s.fc1.bias", p.c_str());
        d.fc2_w = L.gemm("%s.fc2.weight", p.c_str());
        d.fc2_b = L.f32("%s.fc2.bias", p.c_str());
        d.ln1_w = L.f32("%s.ln1.weight", p.c_str());
        d.ln1_b = L.f32("%s.ln1.bias", p.c_str());
        d.ln2_w = L.f32("%s.ln2.weight", p.c_str());
        d.ln2_b = L.f32("%s.ln2.bias", p.c_str());
        d.ln3_w = L.f32("%s.ln3.weight", p.c_str());
        d.ln3_b = L.f32("%s.ln3.bias", p.c_str());
    }
    m.dec_norm_w = L.f32("dec.norm.weight");
    m.dec_norm_b = L.f32("dec.norm.bias");
    m.dec_pos    = L.f32("dec.pos");
    m.head_w     = L.gemm("action_head.weight");
    m.head_b     = L.f32("action_head.bias");

    if (!L.upload(m.backend, &m.weight_buf))
        return false;

    if (m.stem.w->ne[2] != 3 || m.img_proj_w->ne[0] != m.res.back().conv2.w->ne[3] ||
        m.img_proj_w->ne[1] != m.dim || m.dec_pos->ne[0] != m.dim || m.dec_pos->ne[1] != m.chunk ||
        m.head_w->ne[1] != m.action_dim || m.enc[0].fc1_w->ne[1] != m.ff ||
        (m.state_w && m.state_w->ne[0] != m.state_dim)) {
        std::fprintf(stderr, "vla(act): tensor shapes disagree with GGUF metadata\n");
        return false;
    }

    // Host-side tables: the 1D position rows and the normalization stats.
    m.pos_1d = g.read_f32("pos_1d");
    m.img_mean = g.read_f32("image_mean");
    m.img_std  = g.read_f32("image_std");
    m.action_mean = g.read_f32("action_mean");
    m.action_std  = g.read_f32("action_std");
    if (m.state_dim > 0) {
        m.state_mean = g.read_f32("state_mean");
        m.state_std  = g.read_f32("state_std");
    }
    // Folded in once, so every use divides by the stored std.
    for (std::vector<float> * sd : { &m.img_std, &m.state_std, &m.action_std })
        for (float & s : *sd)
            s += kNormEps;
    if (m.pos_1d.size() != (size_t) (m.n_1d()*m.dim) || m.img_mean.size() != (size_t) (3*m.n_views) ||
        m.img_std.size() != m.img_mean.size() || m.action_mean.size() != (size_t) m.action_dim ||
        m.action_std.size() != (size_t) m.action_dim || m.state_mean.size() != (size_t) m.state_dim ||
        m.state_std.size() != (size_t) m.state_dim) {
        std::fprintf(stderr, "vla(act): position or normalization tables disagree with GGUF metadata\n");
        return false;
    }
    return true;
}

}  // namespace

std::unique_ptr<ModelArchBase> act_create(const std::string& mmproj_path,
                                          const std::string& ckpt_path,
                                          const std::string&,
                                          const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(act): note - mmproj '%s' is ignored (vision is baked into the GGUF)\n",
                    mmproj_path.c_str());

    auto m = std::make_unique<ActModelArch>();
    m->mt = opts.weight_dtype.value_or(GGML_TYPE_F32);

    gguf_reader g("act");
    if (!g.open(ckpt_path))
        return nullptr;
    if (!g.has("act.architecture")) {
        std::fprintf(stderr, "vla(act): %s is not an ACT GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, *m))
        return nullptr;

    const Backend b = backend_init("vla(act)", m->n_threads);
    if (!b.handle)
        return nullptr;
    m->backend = b.handle;
    // An explicit --weight-dtype f32 keeps the convolutions F32 too.
    m->conv_direct = b.is_cuda && cuda_conv_has_mma() && opts.weight_dtype.value_or(GGML_TYPE_F16) != GGML_TYPE_F32;

    if (!load_weights(*m, g))
        return nullptr;

    int64_t fh = 0, fw = 0;
    ActModelArch::feature_size(m->img_h, m->img_w, fh, fw);
    m->cfg.n_img           = m->n_views * fh * fw;
    m->cfg.n_lang          = 0;
    m->cfg.n_state         = m->state_dim > 0 ? 1 : 0;
    m->cfg.n_suffix        = m->chunk;
    m->cfg.hidden          = m->dim;
    m->cfg.max_state_dim   = m->state_dim;
    m->cfg.real_state_dim  = m->state_dim;
    m->cfg.max_action_dim  = m->action_dim;
    m->cfg.real_action_dim = m->action_dim;

    std::printf("vla(act): weights resident %.1f MiB (%s, %s convs) - ResNet x%lld views at %lldx%lld, "
                "%lld+%lld layers, dim %lld, chunk %lld\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0), dtype_name(m->mt),
                m->conv_direct ? "f16 direct" : "f32 im2col",
                (long long) m->n_views, (long long) m->img_h, (long long) m->img_w, (long long) m->enc_layers, (long long) m->dec_layers, (long long) m->dim, (long long) m->chunk);
    if (!m->cameras.empty())
        std::printf("vla(act): send the views in this order: %s\n", m->cameras.c_str());
    return m;
}

std::vector<float> ActModelArch::predict(const Inputs& in) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    stats = Stats{};

    if (in.precomputed_img_emb) {
        std::fprintf(stderr, "vla(act): precomputed_img_emb is not supported; pass raw images\n");
        return {};
    }
    if (!in.images || in.n_images != n_views) {
        std::fprintf(stderr, "vla(act): expected %lld views, got %d\n", (long long) n_views, in.n_images);
        return {};
    }
    const int64_t h = in.images[0].h, w = in.images[0].w;
    for (int64_t i = 0; i < n_views; ++i) {
        const ImageView & iv = in.images[i];
        if (!iv.data || iv.w != w || iv.h != h || w < 32 || h < 32) {
            std::fprintf(stderr, "vla(act): view %lld is %dx%d; all views must share one size of at least 32x32\n",
                         (long long) i, iv.w, iv.h);
            return {};
        }
    }
    if (state_dim > 0 && !in.state) {
        std::fprintf(stderr, "vla(act): the checkpoint needs a %lld-dim state\n", (long long) state_dim);
        return {};
    }

    bool fresh = false;
    const size_t arena = ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(4096, false);
    if (!graph.ensure(backend, Key{h, w}, arena,
                      [&](ggml_context * C, IO & io) { fresh = true; return build(C, io, h, w); })) {
        std::fprintf(stderr, "vla(act): graph build/alloc failed\n");
        return {};
    }
    IO & io = graph.io();
    if (fresh) {
        int64_t fh = 0, fw = 0;
        feature_size(h, w, fh, fw);
        const std::vector<float> table = enc_pos_table(fh, fw);
        ggml_backend_tensor_set(io.enc_pos, table.data(), 0, ggml_nbytes(io.enc_pos));
    }

    // HWC to [w, h, c, view], normalized per camera with the dataset stats.
    pixels.resize((size_t) w*h*3*n_views);
    for (int64_t v = 0; v < n_views; ++v)
        image_to_chw(in.images[v], &img_mean[(size_t) v*3], &img_std[(size_t) v*3], pixels.data() + (size_t) v*3*h*w);
    ggml_backend_tensor_set(io.pixels, pixels.data(), 0, ggml_nbytes(io.pixels));
    if (state_dim > 0) {
        std::vector<float> s((size_t) state_dim);
        for (int64_t i = 0; i < state_dim; ++i)
            s[(size_t) i] = (in.state[i] - state_mean[(size_t) i]) / state_std[(size_t) i];
        ggml_backend_tensor_set(io.state, s.data(), 0, ggml_nbytes(io.state));
    }

    const auto tc = clock::now();
    graph_unique_names(graph.graph());
    if (ggml_backend_graph_compute(backend, graph.graph()) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(act): compute failed\n");
        return {};
    }
    std::vector<float> out((size_t) (chunk*action_dim));
    ggml_backend_tensor_get(io.actions, out.data(), 0, out.size()*sizeof(float));
    for (int64_t t = 0; t < chunk; ++t)
        for (int64_t j = 0; j < action_dim; ++j) {
            float & a = out[(size_t) (t*action_dim + j)];
            a = a * action_std[(size_t) j] + action_mean[(size_t) j];
        }

    stats.ms_inference = std::chrono::duration<float, std::milli>(clock::now() - tc).count();
    stats.ms_total     = std::chrono::duration<float, std::milli>(clock::now() - t0).count();
    return out;
}

}  // namespace vla
