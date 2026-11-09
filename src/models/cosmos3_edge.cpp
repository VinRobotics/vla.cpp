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

// Cosmos3-Edge-Policy (LIBERO): a world-action model. One 28-layer Mixture-of-Transformers backbone (Nemotron 3
// Dense, two streams) denoises 4 latent video frames and a 16-step action chunk together, conditioned on the
// Wan2.2 VAE latent of the current frame and on the prompt, with 30 UniPC steps. lerobot's
// policies/cosmos3/modeling_cosmos3.py is the reference; this file follows its inference path:
//
//   VAE encoder     canvas (one frame, pixel-unshuffled to 12 channels) -> normalized latent, 50 patches of 192
//   prefix          the understanding stream over the prompt, once: its keys (normed and rotated for the
//                   generation queries) and values of every layer are kept for the denoising loop
//   step x 30       the generation stream over [video patches | actions]: timestep embedding, 28 layers whose
//                   attention reads [prompt keys | its own keys], then the velocities of latents and actions
//   solver          UniPC (diffusers' UniPCMultistepScheduler: flow prediction, order 2, bh2, corrector) on the
//                   host, in f32 like the reference
//   output          the 16x10 chunk unnormalized with the dataset quantiles: [dpos(3), rot6d(6), gripper]
//
// Input contract (docs/MODELS.md): two camera views in the order [third person, wrist], composed into the
// 192x320 canvas here exactly as lerobot's Cosmos3VideoCanvasProcessorStep does (or one canvas the caller
// composed); the prompt as token ids, which src/models/cosmos3_edge_prompt.cpp builds from the instruction
// (vla-cli --text); `Inputs::noise` the 16x10 action noise (the video noise is drawn here from a seed that
// follows it). cosmos3_libero_action (the prompt file) decodes the chunk to LIBERO commands.

#include "arch.h"
#include "backend.h"
#include "gguf_reader.h"
#include "loader.h"
#include "model.h"
#include "options.h"
#include "scratch_ctx.h"
#include "layers/attn.h"
#include "layers/conv.h"
#include "layers/norm.h"
#include "layers/rope.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vla {
namespace {

constexpr const char * kTag = "vla(cosmos3_edge)";

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------------------------------------
// The canvas: lerobot's Cosmos3VideoCanvasProcessorStep
// ------------------------------------------------------------------------------------------------

// How the policy's cameras fill the canvas, from the GGUF (scripts/convert_cosmos3_edge_to_gguf.py).
struct Layout {
    int n_views = 2, view_h = 256, view_w = 256;  // every view resized to view_h x view_w, composed side by side
    int canvas_h = 192, canvas_w = 320;           // the canvas of the resolution tier
    int content_h = 160, content_w = 320;         // the composed frame fit into it (never enlarged), top left
};

// torch's antialiased resampling (aten UpSampleKernel, align_corners=False): per output index the input
// range and its normalized filter weights. Float arithmetic as in aten, so uint8 results agree.
struct AaTaps {
    std::vector<int>   xmin, xsize;
    std::vector<float> w;  // [out][maxk]
    int                maxk = 0;
};

float triangle(float x) {
    x = std::fabs(x);
    return x < 1.f ? 1.f - x : 0.f;
}

// Keys' cubic, a = -0.5, written as aten's cubic_convolution1/2 (the same value as the textbook form, but not
// the same float rounding).
float cubic(float x) {
    const float a = -0.5f;
    x             = std::fabs(x);
    if (x < 1.f)
        return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f;
    if (x < 2.f)
        return ((a * x - 5.f * a) * x + 8.f * a) * x - 4.f * a;
    return 0.f;
}

// The types are aten's (_compute_indices_min_size_weights_aa): float values, but sums with the double literals
// 0.5 and 1.0 are done in double, then narrowed.
AaTaps aa_taps(int in, int out, bool bicubic) {
    AaTaps t;
    const float  scale   = (float) in / (float) out;
    const double interp  = bicubic ? 4.0 : 2.0;
    const float  support = scale >= 1.f ? (float) ((interp * 0.5) * (double) scale) : (float) (interp * 0.5);
    const float  invscal = scale >= 1.f ? (float) (1.0 / (double) scale) : 1.f;
    t.maxk               = (int) std::ceil(support) * 2 + 1;
    t.xmin.resize((size_t) out);
    t.xsize.resize((size_t) out);
    t.w.assign((size_t) out * (size_t) t.maxk, 0.f);
    for (int i = 0; i < out; i++) {
        const float center = (float) ((double) scale * ((double) i + 0.5));
        const int   lo     = std::max((int) ((double) (center - support) + 0.5), 0);
        const int   hi     = std::min((int) ((double) (center + support) + 0.5), in);
        const int   n      = hi - lo;
        float *     w      = &t.w[(size_t) i * (size_t) t.maxk];
        float       total  = 0.f;
        for (int j = 0; j < n; j++) {
            const float d = (float) (j + lo) - center;
            const float x = (float) (((double) d + 0.5) * (double) invscal);
            w[j]          = bicubic ? cubic(x) : triangle(x);
            total += w[j];
        }
        if (total != 0.f)
            for (int j = 0; j < n; j++)
                w[j] /= total;
        t.xmin[(size_t) i]  = lo;
        t.xsize[(size_t) i] = n;
    }
    return t;
}

// planes: [n_planes][h][w] floats -> [n_planes][oh][ow]; width first, then height, as aten does.
std::vector<float> resize_aa(const std::vector<float> & src, int planes, int h, int w, int oh, int ow, bool bicubic) {
    const AaTaps tw = aa_taps(w, ow, bicubic);
    const AaTaps th = aa_taps(h, oh, bicubic);
    std::vector<float> tmp((size_t) planes * (size_t) h * (size_t) ow);
    for (int p = 0; p < planes; p++)
        for (int y = 0; y < h; y++) {
            const float * row = &src[((size_t) p * h + y) * (size_t) w];
            float *       dst = &tmp[((size_t) p * h + y) * (size_t) ow];
            for (int x = 0; x < ow; x++) {
                const float * k = &tw.w[(size_t) x * (size_t) tw.maxk];
                float         s = 0.f;
                for (int j = 0; j < tw.xsize[(size_t) x]; j++)
                    s = std::fmaf(row[tw.xmin[(size_t) x] + j], k[j], s);
                dst[x] = s;
            }
        }
    std::vector<float> out((size_t) planes * (size_t) oh * (size_t) ow);
    for (int p = 0; p < planes; p++)
        for (int y = 0; y < oh; y++) {
            const float * k = &th.w[(size_t) y * (size_t) th.maxk];
            for (int x = 0; x < ow; x++) {
                float s = 0.f;
                for (int j = 0; j < th.xsize[(size_t) y]; j++)
                    s = std::fmaf(tmp[((size_t) p * h + (size_t) (th.xmin[(size_t) y] + j)) * (size_t) ow + (size_t) x], k[j], s);
                out[((size_t) p * oh + y) * (size_t) ow + (size_t) x] = s;
            }
        }
    return out;
}

// One camera view as CHW floats in [0, 1] at the view size (bilinear with antialias and uint8 rounding when it
// has another size, as the processor's prepare_view does).
bool view_chw(const ImageView & v, int view_h, int view_w, std::vector<float> & chw) {
    if (!v.data || v.w <= 0 || v.h <= 0)
        return false;
    const size_t n = (size_t) v.w * (size_t) v.h;
    chw.resize(3 * n);
    for (size_t i = 0; i < n; i++)
        for (int c = 0; c < 3; c++) {
            const float x = v.format == PixelFormat::U8 ? (float) ((const uint8_t *) v.data)[i * 3 + (size_t) c] / 255.f
                                                         : ((const float *) v.data)[i * 3 + (size_t) c];
            chw[(size_t) c * n + i] = x;
        }
    if (v.w != view_w || v.h != view_h) {
        std::vector<float> scaled(chw.size());
        for (size_t i = 0; i < chw.size(); i++)
            scaled[i] = chw[i] * 255.f;
        std::vector<float> r = resize_aa(scaled, 3, v.h, v.w, view_h, view_w, false);
        for (float & x : r)
            x = std::min(std::max(std::nearbyint(x), 0.f), 255.f) / 255.f;
        chw = std::move(r);
    }
    return true;
}

// The views -> the float canvas [3][canvas_h][canvas_w] in uint8 levels.
bool make_canvas(const ImageView * images, int n_images, const Layout & L, std::vector<float> & canvas) {
    // A caller that composes the canvas itself (lerobot's processor in Python) sends it as the one image.
    if (n_images == 1 && images[0].data && images[0].w == L.canvas_w && images[0].h == L.canvas_h) {
        const size_t n = (size_t) L.canvas_w * (size_t) L.canvas_h;
        canvas.resize(3 * n);
        for (size_t i = 0; i < n; i++)
            for (size_t c = 0; c < 3; c++) {
                const size_t k = i * 3 + c;
                canvas[c * n + i] = images[0].format == PixelFormat::U8
                                        ? (float) ((const uint8_t *) images[0].data)[k]
                                        : std::nearbyint(((const float *) images[0].data)[k] * 255.f);
            }
        return true;
    }
    if (n_images != L.n_views) {
        std::fprintf(stderr, "%s: expected %d views [third person, wrist] or one %dx%d canvas, got %d images\n", kTag,
                     L.n_views, L.canvas_w, L.canvas_h, n_images);
        return false;
    }
    std::vector<std::vector<float>> views((size_t) L.n_views);
    for (int i = 0; i < L.n_views; i++)
        if (!view_chw(images[i], L.view_h, L.view_w, views[(size_t) i])) {
            std::fprintf(stderr, "%s: view %d is empty\n", kTag, i);
            return false;
        }
    // Side by side, to uint8 by truncation (the processor does (frame * 255).clamp(0, 255).to(uint8)).
    const int fw = L.view_w * L.n_views, fh = L.view_h;
    std::vector<float> frame((size_t) 3 * (size_t) fh * (size_t) fw);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < fh; y++)
            for (int v = 0; v < L.n_views; v++)
                for (int x = 0; x < L.view_w; x++) {
                    const float f = views[(size_t) v][((size_t) c * fh + y) * (size_t) L.view_w + (size_t) x] * 255.f;
                    frame[((size_t) c * fh + y) * (size_t) fw + (size_t) (v * L.view_w + x)] =
                        std::trunc(std::min(std::max(f, 0.f), 255.f));
                }
    // Bicubic with antialias into the content area (rounded to even, as torch.round), then the canvas filled
    // to the right and below by reflection (edge replication when the padding is wider than the content).
    std::vector<float> content = frame;
    if (L.content_h != fh || L.content_w != fw) {
        content = resize_aa(frame, 3, fh, fw, L.content_h, L.content_w, true);
        for (float & x : content)
            x = std::nearbyint(std::min(std::max(x, 0.f), 255.f));
    }
    const bool replicate = L.canvas_w - L.content_w >= L.content_w || L.canvas_h - L.content_h >= L.content_h;
    auto src = [&](int i, int n) {  // the content index a canvas index reads
        if (i < n)
            return i;
        return replicate ? n - 1 : 2 * (n - 1) - i;  // reflect without repeating the edge
    };
    canvas.assign((size_t) 3 * (size_t) L.canvas_h * (size_t) L.canvas_w, 0.f);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < L.canvas_h; y++) {
            const int sy = src(y, L.content_h);
            for (int x = 0; x < L.canvas_w; x++)
                canvas[((size_t) c * L.canvas_h + y) * (size_t) L.canvas_w + (size_t) x] =
                    content[((size_t) c * L.content_h + (size_t) sy) * (size_t) L.content_w + (size_t) src(x, L.content_w)];
        }
    return true;
}

// ------------------------------------------------------------------------------------------------
// UniPC: diffusers' UniPCMultistepScheduler as the policy configures it (flow prediction and flow sigmas, bh2,
// solver order 2, predict_x0, lower_order_final, the corrector on). f32 data, f64 coefficients, on the host.
// ------------------------------------------------------------------------------------------------

struct UniPC {
    std::vector<float>   sigmas;     // [n + 1], the last one 0
    std::vector<int64_t> timesteps;  // [n], the integer timesteps the network is told

    // The native sigma schedule: num_steps noise levels evenly spaced from 1 - 1/num_train_timesteps (as a
    // float32, like the native sampler) down to 0, shifted to shift*s / (1 + (shift-1)*s); the timesteps are
    // 1000*s truncated to integers (the scheduler stores them as int64).
    UniPC(int n, int train_steps, double shift) {
        const double sigma_max = (double) (float) (1.0 - 1.0 / (double) train_steps);
        sigmas.resize((size_t) n + 1);
        timesteps.resize((size_t) n);
        for (int i = 0; i < n; i++) {
            const double s  = sigma_max + (0.0 - sigma_max) * ((double) i / (double) n);  // np.linspace(sigma_max, 0, n + 1)[i]
            const double ss = shift * s / (1.0 + (shift - 1.0) * s);
            sigmas[(size_t) i]    = (float) ss;
            timesteps[(size_t) i] = (int64_t) (ss * (double) train_steps);
        }
        sigmas[(size_t) n] = 0.f;
    }

    // alpha = 1 - sigma (flow), lambda = log(alpha) - log(sigma).
    double lambda(int i) const {
        const double s = sigmas[(size_t) i];
        return std::log(1.0 - s) - std::log(s);
    }

    struct Stream {
        std::vector<float> m0, m1;       // model_outputs[-1], [-2]: the x0 predictions of the last two steps
        std::vector<float> last_sample;  // the sample the last predictor started from
        int  lower_order_nums = 0, this_order = 1;
        bool have_last = false;
    };

    // One scheduler.step() on a stream: v is the network's velocity at x, the sample of step i; x becomes the
    // sample of step i + 1. UniC corrects x with the new model output first (from step 1 on), then UniP predicts.
    void step(int i, Stream & S, const std::vector<float> & v, std::vector<float> & x) const {
        const size_t N = x.size();
        const int    n = (int) timesteps.size();
        std::vector<float> x0(N);  // convert_model_output: x0 = x - sigma_t * v
        for (size_t k = 0; k < N; k++)
            x0[k] = x[k] - sigmas[(size_t) i] * v[k];

        if (i > 0 && S.have_last) {  // multistep_uni_c_bh_update(x0, last_sample, x, this_order)
            const double sigma_t = sigmas[(size_t) i], sigma_s0 = sigmas[(size_t) i - 1];
            const double alpha_t = 1.0 - sigma_t;
            const double h = lambda(i) - lambda(i - 1), hh = -h;
            const double h_phi_1 = std::expm1(hh), B_h = std::expm1(hh);
            // rhos_c: R rhos = b for the order; order 1 is the fixed 1/2.
            double r_m1 = 0.0, r_t = 0.5, rk = 1.0;
            if (S.this_order == 2) {
                rk = (lambda(i - 2) - lambda(i - 1)) / h;
                double h_phi_k = h_phi_1 / hh - 1.0;
                const double b0 = h_phi_k / B_h;  // factorial 1
                h_phi_k = h_phi_k / hh - 1.0 / 2.0;
                const double b1 = h_phi_k * 2.0 / B_h;  // factorial 2
                // R = [[1, 1], [rk, 1]]
                r_m1 = (b0 - b1) / (1.0 - rk);
                r_t  = b0 - r_m1;
            }
            const double c_last = sigma_t / sigma_s0;
            for (size_t k = 0; k < N; k++) {
                const double m0 = S.m0[k];
                double corr = r_t * ((double) x0[k] - m0);
                if (S.this_order == 2)
                    corr += r_m1 * (((double) S.m1[k] - m0) / rk);
                x[k] = (float) (c_last * (double) S.last_sample[k] - alpha_t * h_phi_1 * m0 - alpha_t * B_h * corr);
            }
        }
        // The model-output history, then the order of this step's predictor (lower_order_final, warm-up).
        S.m1 = std::move(S.m0);
        S.m0 = std::move(x0);
        S.this_order  = std::min(std::min(2, n - i), S.lower_order_nums + 1);
        S.last_sample = x;
        S.have_last   = true;

        // multistep_uni_p_bh_update(x, this_order): from sigma[i] to sigma[i + 1].
        const double sigma_t = sigmas[(size_t) i + 1], sigma_s0 = sigmas[(size_t) i];
        if (sigma_t <= 0.0) {
            // The last step lands on sigma 0: its lambda is infinite, sigma_t/sigma_s0 is 0, h_phi_1 and B_h are -1
            // and alpha_t is 1, so the update is x = m0, the x0 prediction (lower_order_final makes this step
            // order 1 whenever there are two steps or more).
            for (size_t k = 0; k < N; k++)
                x[k] = S.m0[k];
        } else {
            const double alpha_t = 1.0 - sigma_t;
            const double lambda_t = std::log(1.0 - sigma_t) - std::log(sigma_t);
            const double h = lambda_t - lambda(i), hh = -h;
            const double h_phi_1 = std::expm1(hh), B_h = std::expm1(hh);
            const double c_x = sigma_t / sigma_s0;
            double rk = 1.0;
            if (S.this_order == 2)
                rk = (lambda(i - 1) - lambda(i)) / h;
            for (size_t k = 0; k < N; k++) {
                const double m0 = S.m0[k];
                double xt = c_x * (double) x[k] - alpha_t * h_phi_1 * m0;
                if (S.this_order == 2)
                    xt -= alpha_t * B_h * 0.5 * (((double) S.m1[k] - m0) / rk);
                x[k] = (float) xt;
            }
        }
        if (S.lower_order_nums < 2)
            S.lower_order_nums++;
    }
};

// ------------------------------------------------------------------------------------------------
// Weights
// ------------------------------------------------------------------------------------------------

struct ConvW {
    ggml_tensor *w = nullptr, *b = nullptr;
};

struct ResW {
    ggml_tensor *norm1 = nullptr, *norm2 = nullptr;
    ConvW        conv1, conv2, shortcut;  // shortcut.w null when in_dim == out_dim
};

struct DownW {
    std::vector<ResW> res;
    ConvW             down;           // ZeroPad2d((0, 1, 0, 1)) + Conv2d(dim, dim, 3, stride 2); none on the last block
    ggml_tensor *     avg = nullptr;  // AvgDown3D of the shortcut as a strided constant kernel; null when identity
    int               factor_t = 1, factor_s = 1;
};

struct VaeW {
    ConvW              conv_in;
    std::vector<DownW> down;
    ResW               mid0, mid1;
    ggml_tensor *      attn_norm = nullptr;
    ConvW              qkv, proj;  // 1x1
    ggml_tensor *      norm_out = nullptr;
    ConvW              conv_out, quant;  // quant 1x1
};

struct StreamW {
    ggml_tensor *attn_norm = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *ffn_norm = nullptr, *up = nullptr, *down = nullptr;
};

struct LayerW {
    StreamW       und, gen;
    ggml_tensor * k_norm_for_gen = nullptr;              // the understanding keys' norm for the generation queries
    ggml_tensor * q_norm = nullptr, * k_norm = nullptr;  // the generation stream's QK norms
};

bool cuda_conv_has_mma() {
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)
    int major = 0, minor = 0;
    const int dev = backend_device_index();
    return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
           major * 10 + minor >= 75;
#else
    return false;
#endif
}

// ------------------------------------------------------------------------------------------------
// The architecture
// ------------------------------------------------------------------------------------------------

struct Cosmos3EdgeModelArch : public ModelArchBase {
    Cosmos3EdgeModelArch() : ModelArchBase(Arch::COSMOS3_EDGE) {}
    ~Cosmos3EdgeModelArch() override;

    ggml_backend_t backend   = nullptr;
    int            n_threads = default_cpu_threads();
    ggml_type      mt        = GGML_TYPE_BF16;  // GEMM weights
    // CUDA (Turing+) runs the VAE as ggml's direct convolution on F16 kernels (implicit GEMM on tensor cores,
    // F32 accumulation); elsewhere the kernels stay F32 and go through im2col, as act.cpp does.
    bool           conv_direct = false;

    // --- sizes, from the GGUF ---
    int64_t hidden = 2048, inter = 9216, n_layers = 28, n_heads = 16, n_kv_heads = 8, head_dim = 128;
    float   rms_eps = 1e-5f, rope_theta = 1e8f;
    int     mrope[3] = { 24, 20, 20 };
    int64_t time_dim = 256;
    float   timestep_scale = 0.001f;
    int64_t margin = 15000;  // temporal_modality_margin: the generation stream starts this far after the text
    double  base_fps = 24.0, fps = 20.0;
    int     steps = 30, train_steps = 1000;
    double  shift = 5.0;
    int64_t chunk = 16, adim = 10;
    int64_t z_dim = 48, vae_patch = 2, spatial = 16, temporal = 4, latent_patch = 2;
    int64_t latent_h = 10, latent_w = 20;  // the content's latent grid (the canvas's is cropped to it)
    int64_t frames = 5, ppf = 50, n_video = 250, patch_dim = 192;  // latent frames, patches per frame
    int64_t max_prompt = 512;
    Layout  layout;
    std::vector<float> lat_mean, lat_std, q01, q99;

    // --- weights ---
    ggml_context *        ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context *        ctx_aux = nullptr;  // the AvgDown kernels, made here
    ggml_backend_buffer_t buf_aux = nullptr;
    ggml_tensor *         tok_embd = nullptr;
    std::vector<LayerW>   layers;
    ggml_tensor *         gen_out_norm = nullptr;
    ggml_tensor *vae2llm_w = nullptr, *vae2llm_b = nullptr, *llm2vae_w = nullptr, *llm2vae_b = nullptr;
    ggml_tensor *a2l_w = nullptr, *a2l_b = nullptr, *l2a_w = nullptr, *l2a_b = nullptr, *act_embed = nullptr;
    ggml_tensor *te1_w = nullptr, *te1_b = nullptr, *te2_w = nullptr, *te2_b = nullptr;
    VaeW vae;

    // --- graphs ---
    struct VaeIO {
        ggml_tensor *px = nullptr, *lat = nullptr;
    };
    struct PrefixIO {
        ggml_tensor *ids = nullptr, *cos = nullptr, *sin = nullptr, *mask = nullptr;
    };
    struct StepIO {
        ggml_tensor *xv = nullptr, *xa = nullptr, *tsin = nullptr;
        ggml_tensor *vv = nullptr, *va = nullptr;
    };
    graph_cache<int, VaeIO>        vae_graph;
    graph_cache<int64_t, PrefixIO> prefix_graph;  // keyed by the prompt length
    graph_cache<int64_t, StepIO>   step_graph;
    // What every step reads and the prefix sets up, in a buffer of its own so the graph allocator never reuses it
    // (a graph input's memory is free after its last use): the understanding stream's keys and values of every
    // layer, [head_dim, n_text, n_kv_heads], and the generation stream's rotation tables [head_dim, n_gen], whose
    // positions start after the prompt. Re-made when the prompt length changes.
    struct KV {
        ggml_context *             ctx = nullptr;
        ggml_backend_buffer_t      buf = nullptr;
        int64_t                    n   = 0;
        std::vector<ggml_tensor *> k, v;
        ggml_tensor *              cos = nullptr, *sin = nullptr;
    } kv;

    uint64_t calls = 0;

    bool load_config(gguf_reader & g);
    bool load_weights(gguf_reader & g);
    bool make_avg_kernels();
    bool ensure_kv(int64_t n);

    ggml_cgraph * build_vae(ggml_context * C, VaeIO & io) const;
    ggml_cgraph * build_prefix(ggml_context * C, PrefixIO & io, int64_t n) const;
    ggml_cgraph * build_step(ggml_context * C, StepIO & io, int64_t n) const;

    // Graph pieces.
    ggml_tensor * lin(ggml_context * C, ggml_tensor * W, ggml_tensor * b, ggml_tensor * x) const;
    ggml_tensor * conv(ggml_context * C, const ConvW & cw, ggml_tensor * x, int stride, int pad) const;
    ggml_tensor * conv1x1(ggml_context * C, const ConvW & cw, ggml_tensor * x) const;
    ggml_tensor * chan_rms(ggml_context * C, ggml_tensor * x, ggml_tensor * gamma) const;
    ggml_tensor * resnet(ggml_context * C, const ResW & r, ggml_tensor * x) const;
    ggml_tensor * vae_attention(ggml_context * C, ggml_tensor * x) const;
    ggml_tensor * attend(ggml_context * C, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                         int64_t n_q) const;

    // Host tables.
    void rope_text(int64_t n, std::vector<float> & cs, std::vector<float> & sn) const;
    void rope_gen(int64_t n_text, std::vector<float> & cs, std::vector<float> & sn) const;
    void time_sinusoid(int64_t t, std::vector<float> & out) const;

    std::vector<float> predict(const Inputs & in) override;
};

Cosmos3EdgeModelArch::~Cosmos3EdgeModelArch() {
    vae_graph.release();
    prefix_graph.release();
    step_graph.release();
    if (kv.buf)
        ggml_backend_buffer_free(kv.buf);
    if (kv.ctx)
        ggml_free(kv.ctx);
    if (buf_aux)
        ggml_backend_buffer_free(buf_aux);
    if (ctx_aux)
        ggml_free(ctx_aux);
    if (buf_w)
        ggml_backend_buffer_free(buf_w);
    if (ctx_w)
        ggml_free(ctx_w);
    if (backend)
        ggml_backend_free(backend);
}

bool Cosmos3EdgeModelArch::load_config(gguf_reader & g) {
    auto u = [&](const std::string & k) { return (int64_t) g.u32(("cosmos3_edge." + k).c_str()); };
    auto f = [&](const std::string & k) { return g.f32(("cosmos3_edge." + k).c_str()); };
    n_layers = u("n_layers"); hidden = u("hidden"); inter = u("intermediate");
    n_heads = u("n_heads"); n_kv_heads = u("n_kv_heads"); head_dim = u("head_dim");
    mrope[0] = (int) u("mrope_section_t"); mrope[1] = (int) u("mrope_section_h"); mrope[2] = (int) u("mrope_section_w");
    time_dim = u("time_embed_dim"); margin = u("temporal_modality_margin");
    steps = (int) u("num_steps"); train_steps = (int) u("num_train_timesteps");
    chunk = u("chunk_size"); adim = u("action_dim"); latent_patch = u("latent_patch_size");
    z_dim = u("vae.z_dim"); vae_patch = u("vae.patch_size");
    spatial = u("vae.scale_factor_spatial"); temporal = u("vae.scale_factor_temporal");
    latent_h = u("latent_h"); latent_w = u("latent_w"); max_prompt = u("prompt_len");
    layout.n_views = (int) u("n_views"); layout.view_h = (int) u("view_h"); layout.view_w = (int) u("view_w");
    layout.canvas_h = (int) u("canvas_h"); layout.canvas_w = (int) u("canvas_w");
    layout.content_h = (int) u("content_h"); layout.content_w = (int) u("content_w");
    rms_eps = f("rms_eps"); rope_theta = f("rope_theta"); timestep_scale = f("timestep_scale");
    base_fps = f("base_fps"); fps = f("fps"); shift = f("sampling_shift");

    const int64_t n_blocks = u("vae.n_blocks"), n_res = u("vae.num_res_blocks");
    if (n_layers <= 0 || hidden <= 0 || inter <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 ||
        n_heads % n_kv_heads != 0 || head_dim % 2 != 0 || time_dim <= 0 || time_dim % 2 != 0 || steps <= 0 ||
        train_steps <= 0 || chunk <= 0 || adim <= 0 || z_dim <= 0 || vae_patch != 2 || latent_patch != 2 ||
        spatial <= 0 || temporal <= 0 || latent_h <= 0 || latent_w <= 0 || n_blocks <= 0 || n_res <= 0 ||
        max_prompt <= 0 || layout.n_views <= 0 || layout.view_h <= 0 || layout.view_w <= 0 || layout.canvas_h <= 0 ||
        layout.canvas_w <= 0 || layout.content_h > layout.canvas_h || layout.content_w > layout.canvas_w ||
        !(rms_eps > 0.f) || !(rope_theta > 0.f) || !(fps > 0.0) || !(shift > 0.0) ||
        mrope[0] + mrope[1] + mrope[2] != head_dim / 2) {
        std::fprintf(stderr, "%s: the GGUF has no valid sizes (converted with an older script?)\n", kTag);
        return false;
    }
    if (layout.canvas_h % spatial != 0 || layout.canvas_w % spatial != 0 || layout.content_h / spatial != latent_h ||
        layout.content_w / spatial != latent_w || latent_h % latent_patch != 0 || latent_w % latent_patch != 0 ||
        chunk % temporal != 0) {
        std::fprintf(stderr, "%s: canvas %dx%d / content %dx%d do not fit the VAE's grid\n", kTag, layout.canvas_h,
                     layout.canvas_w, layout.content_h, layout.content_w);
        return false;
    }
    frames    = 1 + chunk / temporal;
    ppf       = (latent_h / latent_patch) * (latent_w / latent_patch);
    n_video   = frames * ppf;
    patch_dim = latent_patch * latent_patch * z_dim;

    vae.down.resize((size_t) n_blocks);
    for (int64_t b = 0; b < n_blocks; b++) {
        DownW & d  = vae.down[(size_t) b];
        d.res.resize((size_t) n_res);
        d.factor_t = (int) u("vae.down." + std::to_string(b) + ".factor_t");
        d.factor_s = (int) u("vae.down." + std::to_string(b) + ".factor_s");
        if (d.factor_t < 1 || d.factor_s < 1) {
            std::fprintf(stderr, "%s: VAE block %lld has no factors\n", kTag, (long long) b);
            return false;
        }
    }

    lat_mean = g.read_f32("cosmos3_edge.latents_mean");
    lat_std  = g.read_f32("cosmos3_edge.latents_std");
    q01      = g.read_f32("cosmos3_edge.action_q01");
    q99      = g.read_f32("cosmos3_edge.action_q99");
    if ((int64_t) lat_mean.size() != z_dim || (int64_t) lat_std.size() != z_dim || (int64_t) q01.size() != adim ||
        (int64_t) q99.size() != adim) {
        std::fprintf(stderr, "%s: latent statistics or action quantiles disagree with the sizes\n", kTag);
        return false;
    }
    for (float s : lat_std)
        if (!(s > 0.f)) {
            std::fprintf(stderr, "%s: a latent std is not positive\n", kTag);
            return false;
        }
    return true;
}

bool Cosmos3EdgeModelArch::load_weights(gguf_reader & g) {
    const int64_t n_tensors = gguf_get_n_tensors(g.gctx);
    ggml_init_params p = { (size_t) (n_tensors + 16) * ggml_tensor_overhead(), nullptr, true };
    ctx_w = ggml_init(p);
    if (!ctx_w)
        return false;
    WeightLoader wl("cosmos3_edge", g, ctx_w, mt);
    const ggml_type ct = conv_direct ? GGML_TYPE_F16 : GGML_TYPE_F32;

    tok_embd = wl.gemm("token_embd.weight");
    layers.resize((size_t) n_layers);
    for (int64_t i = 0; i < n_layers; i++) {
        LayerW & L = layers[(size_t) i];
        auto stream = [&](StreamW & s, const char * pfx) {
            s.attn_norm = wl.f32("blk.%lld.%sattn_norm.weight", (long long) i, pfx);
            s.q         = wl.gemm("blk.%lld.%sattn_q.weight", (long long) i, pfx);
            s.k         = wl.gemm("blk.%lld.%sattn_k.weight", (long long) i, pfx);
            s.v         = wl.gemm("blk.%lld.%sattn_v.weight", (long long) i, pfx);
            s.o         = wl.gemm("blk.%lld.%sattn_output.weight", (long long) i, pfx);
            s.ffn_norm  = wl.f32("blk.%lld.%sffn_norm.weight", (long long) i, pfx);
            s.up        = wl.gemm("blk.%lld.%sffn_up.weight", (long long) i, pfx);
            s.down      = wl.gemm("blk.%lld.%sffn_down.weight", (long long) i, pfx);
        };
        stream(L.und, "");
        stream(L.gen, "gen.");
        L.k_norm_for_gen = wl.f32("blk.%lld.attn_k_norm_gen.weight", (long long) i);
        L.q_norm         = wl.f32("blk.%lld.gen.attn_q_norm.weight", (long long) i);
        L.k_norm         = wl.f32("blk.%lld.gen.attn_k_norm.weight", (long long) i);
    }
    gen_out_norm = wl.f32("gen.output_norm.weight");
    vae2llm_w = wl.gemm("vae2llm.weight");  vae2llm_b = wl.f32("vae2llm.bias");
    llm2vae_w = wl.gemm("llm2vae.weight");  llm2vae_b = wl.f32("llm2vae.bias");
    a2l_w = wl.gemm("action2llm.weight");   a2l_b = wl.f32("action2llm.bias");
    l2a_w = wl.gemm("llm2action.weight");   l2a_b = wl.f32("llm2action.bias");
    act_embed = wl.f32("action_modality_embed.weight");
    te1_w = wl.f32("time_embedder.linear_1.weight"); te1_b = wl.f32("time_embedder.linear_1.bias");
    te2_w = wl.f32("time_embedder.linear_2.weight"); te2_b = wl.f32("time_embedder.linear_2.bias");

    auto cw = [&](ConvW & c, const std::string & name, bool required = true) {
        c.w = required ? wl.typed(ct, "%s.weight", name.c_str()) : wl.opt_typed(ct, "%s.weight", name.c_str());
        c.b = c.w ? wl.f32("%s.bias", name.c_str()) : nullptr;
    };
    auto res = [&](ResW & r, const std::string & name) {
        r.norm1 = wl.f32("%s.norm1.gamma", name.c_str());
        r.norm2 = wl.f32("%s.norm2.gamma", name.c_str());
        cw(r.conv1, name + ".conv1");
        cw(r.conv2, name + ".conv2");
        cw(r.shortcut, name + ".shortcut", false);
    };
    cw(vae.conv_in, "vae.conv_in");
    for (size_t b = 0; b < vae.down.size(); b++) {
        for (size_t r = 0; r < vae.down[b].res.size(); r++)
            res(vae.down[b].res[r], "vae.down." + std::to_string(b) + ".res." + std::to_string(r));
        cw(vae.down[b].down, "vae.down." + std::to_string(b) + ".down", false);
    }
    res(vae.mid0, "vae.mid.res.0");
    vae.attn_norm = wl.f32("vae.mid.attn.norm.gamma");
    cw(vae.qkv, "vae.mid.attn.qkv");
    cw(vae.proj, "vae.mid.attn.proj");
    res(vae.mid1, "vae.mid.res.1");
    vae.norm_out = wl.f32("vae.norm_out.gamma");
    cw(vae.conv_out, "vae.conv_out");
    cw(vae.quant, "vae.quant_conv");

    if (!wl.ok() || !wl.upload(backend, &buf_w))
        return false;

    // Shape checks that the graphs rely on.
    if (tok_embd->ne[0] != hidden || layers[0].und.q->ne[1] != n_heads * head_dim ||
        layers[0].und.k->ne[1] != n_kv_heads * head_dim || layers[0].und.up->ne[1] != inter ||
        vae2llm_w->ne[0] != patch_dim || vae2llm_w->ne[1] != hidden || llm2vae_w->ne[1] != patch_dim ||
        a2l_w->ne[0] != adim || a2l_w->ne[1] != hidden || l2a_w->ne[1] != adim || te1_w->ne[0] != time_dim ||
        vae.conv_in.w->ne[2] != 3 * vae_patch * vae_patch || vae.quant.w->ne[1] != 2 * z_dim) {
        std::fprintf(stderr, "%s: tensor shapes disagree with the GGUF metadata\n", kTag);
        return false;
    }
    for (size_t b = 0; b < vae.down.size(); b++) {
        const DownW & d = vae.down[b];
        if ((d.factor_s > 1) != (d.down.w != nullptr)) {
            std::fprintf(stderr, "%s: VAE block %zu: the downsampler and its factor disagree\n", kTag, b);
            return false;
        }
    }
    return make_avg_kernels();
}

// The shortcut of a Wan2.2 residual down block is AvgDown3D: the clip is frame- and pixel-unshuffled (a first
// frame has zero frames in front of it) and groups of the resulting channels are averaged into the block's
// output channels. On one frame that is a strided convolution over the block's input with a constant kernel of
// 0 and 1/g, built here. A block that keeps its size and width (the last) is the identity and gets no kernel.
bool Cosmos3EdgeModelArch::make_avg_kernels() {
    struct Pending {
        ggml_tensor *      t;
        std::vector<float> k;
    };
    std::vector<Pending> pending;
    const ggml_type ct = conv_direct ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_init_params p = { (vae.down.size() + 2) * ggml_tensor_overhead(), nullptr, true };
    ctx_aux = ggml_init(p);
    if (!ctx_aux)
        return false;
    for (size_t b = 0; b < vae.down.size(); b++) {
        DownW &       d      = vae.down[b];
        const int64_t c_in   = d.res[0].conv1.w->ne[2];
        const int64_t c_out  = d.res.back().conv2.w->ne[3];
        const int     ft     = d.factor_t, fs = d.factor_s;
        const int64_t factor = (int64_t) ft * fs * fs;
        if (fs == 1 && ft == 1 && c_in == c_out)
            continue;  // identity
        if ((c_in * factor) % c_out != 0) {
            std::fprintf(stderr, "%s: AvgDown3D of %lld x %lld channels into %lld does not divide\n", kTag,
                         (long long) c_in, (long long) factor, (long long) c_out);
            return false;
        }
        const int64_t g = c_in * factor / c_out;
        Pending pend;
        pend.t = ggml_new_tensor_4d(ctx_aux, ct, fs, fs, c_in, c_out);
        ggml_format_name(pend.t, "vae.avg_down.%zu", b);
        pend.k.assign((size_t) (fs * fs * c_in * c_out), 0.f);
        for (int64_t o = 0; o < c_out; o++)
            for (int64_t c = 0; c < c_in; c++)
                for (int hy = 0; hy < fs; hy++)
                    for (int wx = 0; wx < fs; wx++) {
                        // The unshuffled channel of (c, the frame itself = tap ft-1, hy, wx), as
                        // x.view(B, C, T', ft, H', fs, W', fs).permute(0, 1, 3, 5, 7, 2, 4, 6) orders them.
                        const int64_t e = ((c * ft + (ft - 1)) * fs + hy) * fs + wx;
                        if (e / g == o)
                            pend.k[(size_t) (((o * c_in + c) * fs + hy) * fs + wx)] = 1.f / (float) g;
                    }
        d.avg = pend.t;
        pending.push_back(std::move(pend));
    }
    if (pending.empty())
        return true;
    buf_aux = alloc_weights(ctx_aux, backend);
    if (!buf_aux)
        return false;
    for (Pending & pend : pending) {
        if (ct == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(pend.k.size());
            ggml_fp32_to_fp16_row(pend.k.data(), h.data(), (int64_t) h.size());
            ggml_backend_tensor_set(pend.t, h.data(), 0, ggml_nbytes(pend.t));
        } else {
            ggml_backend_tensor_set(pend.t, pend.k.data(), 0, ggml_nbytes(pend.t));
        }
    }
    return true;
}

bool Cosmos3EdgeModelArch::ensure_kv(int64_t n) {
    if (kv.ctx && kv.n == n)
        return true;
    if (kv.buf)
        ggml_backend_buffer_free(kv.buf);
    if (kv.ctx)
        ggml_free(kv.ctx);
    kv = KV{};
    ggml_init_params p = { (size_t) (2 * n_layers + 4) * ggml_tensor_overhead(), nullptr, true };
    kv.ctx = ggml_init(p);
    if (!kv.ctx)
        return false;
    kv.cos = ggml_new_tensor_2d(kv.ctx, GGML_TYPE_F32, head_dim, n_video + chunk);
    kv.sin = ggml_new_tensor_2d(kv.ctx, GGML_TYPE_F32, head_dim, n_video + chunk);
    ggml_set_name(kv.cos, "kv.cos_gen");
    ggml_set_name(kv.sin, "kv.sin_gen");
    kv.k.resize((size_t) n_layers);
    kv.v.resize((size_t) n_layers);
    for (int64_t i = 0; i < n_layers; i++) {
        kv.k[(size_t) i] = ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F32, head_dim, n, n_kv_heads);
        kv.v[(size_t) i] = ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F32, head_dim, n, n_kv_heads);
        ggml_format_name(kv.k[(size_t) i], "kv.k.%lld", (long long) i);
        ggml_format_name(kv.v[(size_t) i], "kv.v.%lld", (long long) i);
    }
    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buf)
        return false;
    std::vector<float> cs, sn;
    rope_gen(n, cs, sn);
    ggml_backend_tensor_set(kv.cos, cs.data(), 0, ggml_nbytes(kv.cos));
    ggml_backend_tensor_set(kv.sin, sn.data(), 0, ggml_nbytes(kv.sin));
    kv.n = n;
    return true;
}

// ------------------------------------------------------------------------------------------------
// Graph pieces
// ------------------------------------------------------------------------------------------------

ggml_tensor * Cosmos3EdgeModelArch::lin(ggml_context * C, ggml_tensor * W, ggml_tensor * b, ggml_tensor * x) const {
    ggml_tensor * y = ggml_mul_mat(C, W, x);
    if (mm_prec_f32_enabled())
        ggml_prec_set_acc(y, GGML_PREC_F32);
    return b ? ggml_add(C, y, b) : y;
}

// x [W, H, C, 1] -> [OW, OH, OC, 1] with the bias.
ggml_tensor * Cosmos3EdgeModelArch::conv(ggml_context * C, const ConvW & cw, ggml_tensor * x, int stride, int pad) const {
    ggml_tensor * y = cw.w->type == GGML_TYPE_F16 ? ggml_conv_2d_direct(C, cw.w, x, stride, stride, pad, pad, 1, 1)
                                                  : conv_2d_f32(C, cw.w, x, stride, pad);
    return cw.b ? ggml_add(C, y, ggml_reshape_4d(C, cw.b, 1, 1, cw.b->ne[0], 1)) : y;
}

// A 1x1 convolution stored as a linear map [IC, OC].
ggml_tensor * Cosmos3EdgeModelArch::conv1x1(ggml_context * C, const ConvW & cw, ggml_tensor * x) const {
    const ConvW k4 = { ggml_reshape_4d(C, cw.w, 1, 1, cw.w->ne[0], cw.w->ne[1]), cw.b };
    return conv(C, k4, x, 1, 0);
}

// WanRMS_norm over the channels of x [W, H, C, 1]: F.normalize(x, dim=channels) * sqrt(C) * gamma, which is
// x / rms(x) * gamma. (F.normalize clamps the norm at 1e-12; a channel vector of zeros does not occur.)
ggml_tensor * Cosmos3EdgeModelArch::chan_rms(ggml_context * C, ggml_tensor * x, ggml_tensor * gamma) const {
    ggml_tensor * t = ggml_cont(C, ggml_permute(C, x, 1, 2, 0, 3));  // [C, W, H]
    t = ggml_mul(C, ggml_rms_norm(C, t, 1e-24f), gamma);
    return ggml_cont(C, ggml_permute(C, t, 2, 0, 1, 3));  // [W, H, C]
}

// WanResidualBlock on one frame: norm, SiLU, conv, norm, SiLU, conv, plus the (1x1) shortcut.
ggml_tensor * Cosmos3EdgeModelArch::resnet(ggml_context * C, const ResW & r, ggml_tensor * x) const {
    ggml_tensor * h = r.shortcut.w ? conv1x1(C, r.shortcut, x) : x;
    ggml_tensor * y = conv(C, r.conv1, ggml_silu(C, chan_rms(C, x, r.norm1)), 1, 1);
    y               = conv(C, r.conv2, ggml_silu(C, chan_rms(C, y, r.norm2)), 1, 1);
    return ggml_add(C, y, h);
}

// WanAttentionBlock: single-head attention over the frame's positions, the channels as the head.
ggml_tensor * Cosmos3EdgeModelArch::vae_attention(ggml_context * C, ggml_tensor * x) const {
    const int64_t W = x->ne[0], H = x->ne[1], Ch = x->ne[2], N = W * H;
    ggml_tensor * qkv = conv1x1(C, vae.qkv, chan_rms(C, x, vae.attn_norm));  // [W, H, 3 Ch]
    auto part = [&](int i) {  // channel block i as [Ch, N]: one row per position
        ggml_tensor * v = ggml_view_3d(C, qkv, W, H, Ch, qkv->nb[1], qkv->nb[2], (size_t) i * (size_t) Ch * qkv->nb[2]);
        return ggml_cont(C, ggml_transpose(C, ggml_reshape_2d(C, ggml_cont(C, v), N, Ch)));
    };
    ggml_tensor * q = part(0), *k = part(1), *v = part(2);
    // Always the explicit form: a head of 640 is outside the flash-attention kernels' head sizes.
    ggml_tensor * kq = ggml_mul_mat(C, k, q);  // [N_k, N_q]
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(C, kq, nullptr, 1.0f / std::sqrt((float) Ch), 0.0f);
    ggml_tensor * o = ggml_mul_mat(C, ggml_cont(C, ggml_transpose(C, v)), kq);  // [Ch, N_q]
    ggml_prec_set_acc(o, GGML_PREC_F32);
    o = ggml_reshape_3d(C, ggml_cont(C, ggml_transpose(C, ggml_reshape_2d(C, o, Ch, N))), W, H, Ch);
    return ggml_add(C, conv1x1(C, vae.proj, o), x);
}

// q [head_dim, n_q, n_heads], k and v [head_dim, n_k, n_kv_heads] (the query heads of a key head read it, as
// grouped-query attention does) -> [hidden, n_q]. mask [n_k, n_q] F32 or null.
ggml_tensor * Cosmos3EdgeModelArch::attend(ggml_context * C, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                           ggml_tensor * mask, int64_t n_q) const {
    const float scale = 1.0f / std::sqrt((float) head_dim);
    if (flash_attn_enabled()) {
        ggml_tensor * m16 = mask ? ggml_cast(C, mask, GGML_TYPE_F16) : nullptr;
        return ggml_reshape_2d(C, flash_attention(C, q, fa_kv(C, k), fa_kv(C, v), m16, scale), hidden, n_q);
    }
    ggml_tensor * vt = ggml_cont(C, ggml_permute(C, v, 1, 0, 2, 3));  // [n_k, head_dim, n_kv_heads]
    return ggml_reshape_2d(C, attention(C, q, k, vt, mask, scale, hidden, n_q), hidden, n_q);
}

// ------------------------------------------------------------------------------------------------
// Graphs
// ------------------------------------------------------------------------------------------------

// The Wan2.2 encoder on one frame. px: the canvas in [-1, 1], pixel-unshuffled to [W/2, H/2, 12] (host).
// lat: the mean of the posterior, [W/16, H/16, z_dim], before normalization and cropping (host).
ggml_cgraph * Cosmos3EdgeModelArch::build_vae(ggml_context * C, VaeIO & io) const {
    const int64_t W0 = layout.canvas_w / vae_patch, H0 = layout.canvas_h / vae_patch;
    io.px = ggml_new_tensor_4d(C, GGML_TYPE_F32, W0, H0, 3 * vae_patch * vae_patch, 1);
    ggml_set_input(io.px);

    ggml_tensor * x = conv(C, vae.conv_in, io.px, 1, 1);
    for (const DownW & d : vae.down) {
        ggml_tensor * x_in = x;
        for (const ResW & r : d.res)
            x = resnet(C, r, x);
        if (d.down.w)
            x = conv(C, d.down, ggml_pad(C, x, 1, 1, 0, 0), 2, 0);
        ggml_tensor * sc = d.avg ? conv(C, ConvW{ d.avg, nullptr }, x_in, d.factor_s, 0) : x_in;
        x = ggml_add(C, x, sc);
    }
    x = resnet(C, vae.mid0, x);
    x = vae_attention(C, x);
    x = resnet(C, vae.mid1, x);
    x = conv(C, vae.conv_out, ggml_silu(C, chan_rms(C, x, vae.norm_out)), 1, 1);  // [w, h, 2 z]
    x = conv1x1(C, vae.quant, x);
    // DiagonalGaussianDistribution.mode(): the first z_dim channels.
    io.lat = ggml_cont(C, ggml_view_3d(C, x, x->ne[0], x->ne[1], z_dim, x->nb[1], x->nb[2], 0));
    ggml_set_output(io.lat);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 4096, false);
    ggml_build_forward_expand(gf, io.lat);
    return gf;
}

// The understanding stream over the n prompt tokens (causal), writing every layer's keys as the generation
// queries read them (their own norm, then RoPE) and values into kv.
ggml_cgraph * Cosmos3EdgeModelArch::build_prefix(ggml_context * C, PrefixIO & io, int64_t n) const {
    io.ids  = ggml_new_tensor_1d(C, GGML_TYPE_I32, n);
    io.cos  = ggml_new_tensor_2d(C, GGML_TYPE_F32, head_dim, n);
    io.sin  = ggml_new_tensor_2d(C, GGML_TYPE_F32, head_dim, n);
    io.mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, n, n);
    for (ggml_tensor * t : { io.ids, io.cos, io.sin, io.mask })
        ggml_set_input(t);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_tensor * h  = ggml_get_rows(C, tok_embd, io.ids);  // [hidden, n] f32
    for (int64_t i = 0; i < n_layers; i++) {
        const LayerW & L = layers[(size_t) i];
        ggml_tensor * x = rms_norm(C, h, L.und.attn_norm, rms_eps);
        ggml_tensor * q = to_heads(C, lin(C, L.und.q, nullptr, x), head_dim, n_heads, n);
        ggml_tensor * k = to_heads(C, lin(C, L.und.k, nullptr, x), head_dim, n_kv_heads, n);
        ggml_tensor * v = to_heads(C, lin(C, L.und.v, nullptr, x), head_dim, n_kv_heads, n);
        // What the generation stream reads of this layer.
        ggml_tensor * k_gen = rope_2d(C, rms_norm(C, k, L.k_norm_for_gen, rms_eps), io.cos, io.sin);
        ggml_build_forward_expand(gf, ggml_cpy(C, k_gen, kv.k[(size_t) i]));
        ggml_build_forward_expand(gf, ggml_cpy(C, v, kv.v[(size_t) i]));
        // Nemotron's understanding attention has no QK norm.
        q = rope_2d(C, q, io.cos, io.sin);
        k = rope_2d(C, k, io.cos, io.sin);
        h = ggml_add(C, h, lin(C, L.und.o, nullptr, attend(C, q, k, v, io.mask, n)));
        x = rms_norm(C, h, L.und.ffn_norm, rms_eps);
        h = ggml_add(C, h, lin(C, L.und.down, nullptr, ggml_sqr(C, ggml_relu(C, lin(C, L.und.up, nullptr, x)))));
    }
    return gf;
}

// One network call of the generation stream: [video patches | actions] at the timestep, attending to
// [prompt | itself] in every layer, to the velocities of the video patches and the actions.
ggml_cgraph * Cosmos3EdgeModelArch::build_step(ggml_context * C, StepIO & io, int64_t n) const {
    (void) n;  // the prompt length enters through kv
    const int64_t n_gen = n_video + chunk;
    io.xv   = ggml_new_tensor_2d(C, GGML_TYPE_F32, patch_dim, n_video);
    io.xa   = ggml_new_tensor_2d(C, GGML_TYPE_F32, adim, chunk);
    io.tsin = ggml_new_tensor_1d(C, GGML_TYPE_F32, time_dim);
    for (ggml_tensor * t : { io.xv, io.xa, io.tsin })
        ggml_set_input(t);

    // TimestepEmbedding of the sinusoid (f32): linear, SiLU, linear.
    ggml_tensor * temb = lin(C, te2_w, te2_b, ggml_silu(C, lin(C, te1_w, te1_b, io.tsin)));  // [hidden]
    // Video patches: vae2llm, plus the timestep embedding on the noisy frames (not on the condition frame).
    ggml_tensor * ve   = lin(C, vae2llm_w, vae2llm_b, io.xv);  // [hidden, n_video]
    ggml_tensor * cond = ggml_view_2d(C, ve, hidden, ppf, ve->nb[1], 0);
    ggml_tensor * rest = ggml_add(C, ggml_view_2d(C, ve, hidden, n_video - ppf, ve->nb[1], (size_t) ppf * ve->nb[1]), temb);
    ve = ggml_concat(C, ggml_cont(C, cond), rest, 1);
    // Actions: action2llm of the embodiment, the modality embedding, the timestep embedding.
    ggml_tensor * ae = ggml_add(C, ggml_add(C, lin(C, a2l_w, a2l_b, io.xa), act_embed), temb);  // [hidden, chunk]
    ggml_tensor * h  = ggml_concat(C, ve, ae, 1);  // [hidden, n_gen]

    for (int64_t i = 0; i < n_layers; i++) {
        const LayerW & L = layers[(size_t) i];
        ggml_tensor * x = rms_norm(C, h, L.gen.attn_norm, rms_eps);
        // QK norm per head, then heads, then RoPE.
        auto heads = [&](ggml_tensor * p, ggml_tensor * norm, int64_t nh) {
            ggml_tensor * t = ggml_reshape_3d(C, p, head_dim, nh, n_gen);
            if (norm)
                t = rms_norm(C, t, norm, rms_eps);
            return ggml_cont(C, ggml_permute(C, t, 0, 2, 1, 3));  // [head_dim, n_gen, nh]
        };
        ggml_tensor * q = rope_2d(C, heads(lin(C, L.gen.q, nullptr, x), L.q_norm, n_heads), kv.cos, kv.sin);
        ggml_tensor * k = rope_2d(C, heads(lin(C, L.gen.k, nullptr, x), L.k_norm, n_kv_heads), kv.cos, kv.sin);
        ggml_tensor * v = heads(lin(C, L.gen.v, nullptr, x), nullptr, n_kv_heads);
        ggml_tensor * K = ggml_concat(C, kv.k[(size_t) i], k, 1);  // [head_dim, n + n_gen, n_kv_heads]
        ggml_tensor * V = ggml_concat(C, kv.v[(size_t) i], v, 1);
        h = ggml_add(C, h, lin(C, L.gen.o, nullptr, attend(C, q, K, V, nullptr, n_gen)));
        x = rms_norm(C, h, L.gen.ffn_norm, rms_eps);
        h = ggml_add(C, h, lin(C, L.gen.down, nullptr, ggml_sqr(C, ggml_relu(C, lin(C, L.gen.up, nullptr, x)))));
    }
    h = rms_norm(C, h, gen_out_norm, rms_eps);
    ggml_tensor * hv = ggml_view_2d(C, h, hidden, n_video, h->nb[1], 0);
    ggml_tensor * ha = ggml_view_2d(C, h, hidden, chunk, h->nb[1], (size_t) n_video * h->nb[1]);
    io.vv = lin(C, llm2vae_w, llm2vae_b, hv);  // [patch_dim, n_video]
    io.va = lin(C, l2a_w, l2a_b, ha);          // [adim, chunk]
    ggml_set_output(io.vv);
    ggml_set_output(io.va);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_build_forward_expand(gf, io.vv);
    ggml_build_forward_expand(gf, io.va);
    return gf;
}

// ------------------------------------------------------------------------------------------------
// Host tables
// ------------------------------------------------------------------------------------------------

// The understanding stream's positions are 0..n-1 on all three M-RoPE axes, so its rotation is the plain one.
// Float arithmetic as the reference: inv_freq = 1 / theta^(2j/d) in f32, angle = position * inv_freq in f32.
void Cosmos3EdgeModelArch::rope_text(int64_t n, std::vector<float> & cs, std::vector<float> & sn) const {
    const int64_t half = head_dim / 2;
    cs.resize((size_t) (n * head_dim));
    sn.resize((size_t) (n * head_dim));
    for (int64_t s = 0; s < n; s++)
        for (int64_t j = 0; j < half; j++) {
            const float inv = 1.0f / std::pow(rope_theta, (float) (2 * j) / (float) head_dim);
            const float a   = (float) s * inv;
            cs[(size_t) (s * head_dim + j)] = cs[(size_t) (s * head_dim + half + j)] = (float) std::cos((double) a);
            sn[(size_t) (s * head_dim + j)] = sn[(size_t) (s * head_dim + half + j)] = (float) std::sin((double) a);
        }
}

// The generation stream's M-RoPE positions (Cosmos3VFMNetwork.get_rope_index): it starts
// temporal_modality_margin after the last text position; latent frame j sits at j * (base_fps / temporal) *
// (temporal / fps) on the time axis and on its patch grid on the height and width axes; action i at
// (i + 1) * frame_step / temporal with zero height and width. The three axes interleave over the frequencies
// as Qwen3-VL's rotary embedding does ([T H W T H W ... T T] over mrope_section).
void Cosmos3EdgeModelArch::rope_gen(int64_t n_text, std::vector<float> & cs, std::vector<float> & sn) const {
    const int64_t half = head_dim / 2, n_gen = n_video + chunk;
    const int64_t grid_w = latent_w / latent_patch;
    const float   gen_start  = (float) (n_text - 1) + 1.f + (float) margin;
    const float   frame_step = (float) (base_fps / (double) temporal) * (float) ((double) temporal / fps);
    std::vector<float> pos(3 * (size_t) n_gen);
    for (int64_t t = 0; t < n_video; t++) {
        const int64_t fr = t / ppf, p = t % ppf;
        pos[3 * (size_t) t + 0] = gen_start + (float) fr * frame_step;
        pos[3 * (size_t) t + 1] = (float) (p / grid_w);
        pos[3 * (size_t) t + 2] = (float) (p % grid_w);
    }
    for (int64_t i = 0; i < chunk; i++) {
        const int64_t t = n_video + i;
        pos[3 * (size_t) t + 0] = gen_start + (float) (i + 1) * (frame_step / (float) temporal);
        pos[3 * (size_t) t + 1] = pos[3 * (size_t) t + 2] = 0.f;
    }
    cs.resize((size_t) (n_gen * head_dim));
    sn.resize((size_t) (n_gen * head_dim));
    for (int64_t s = 0; s < n_gen; s++)
        for (int64_t j = 0; j < half; j++) {
            int axis = 0;
            if (j % 3 == 1 && j < 3 * mrope[1])
                axis = 1;
            else if (j % 3 == 2 && j < 3 * mrope[2])
                axis = 2;
            const float inv = 1.0f / std::pow(rope_theta, (float) (2 * j) / (float) head_dim);
            const float a   = pos[3 * (size_t) s + (size_t) axis] * inv;
            cs[(size_t) (s * head_dim + j)] = cs[(size_t) (s * head_dim + half + j)] = (float) std::cos((double) a);
            sn[(size_t) (s * head_dim + j)] = sn[(size_t) (s * head_dim + half + j)] = (float) std::sin((double) a);
        }
}

// SinusoidalTimesteps of t * timestep_scale: cos features then sin, frequencies exp(-ln(10000) j / half).
void Cosmos3EdgeModelArch::time_sinusoid(int64_t t, std::vector<float> & out) const {
    const int64_t half = time_dim / 2;
    const float   ts   = (float) t * timestep_scale;
    out.resize((size_t) time_dim);
    for (int64_t j = 0; j < half; j++) {
        const float freq = std::exp(-std::log(10000.f) * (float) j / (float) half);
        const float a    = ts * freq;
        out[(size_t) j]          = (float) std::cos((double) a);
        out[(size_t) (half + j)] = (float) std::sin((double) a);
    }
}

// ------------------------------------------------------------------------------------------------
// predict
// ------------------------------------------------------------------------------------------------

std::vector<float> Cosmos3EdgeModelArch::predict(const Inputs & in) {
    const double t0 = now_ms();
    stats           = Stats{};
    if (in.precomputed_img_emb) {
        std::fprintf(stderr, "%s: precomputed_img_emb is not supported; pass the camera views\n", kTag);
        return {};
    }

    // --- inputs -----------------------------------------------------------------------------------
    std::vector<float> canvas;
    if (!make_canvas(in.images, in.n_images, layout, canvas))
        return {};
    if (!in.lang_tokens || in.n_lang <= 0) {
        std::fprintf(stderr, "%s: no prompt tokens (build them with the lerobot prompt and tokenizer steps)\n", kTag);
        return {};
    }
    std::vector<int32_t> ids;
    for (int i = 0; i < in.n_lang; i++) {
        if (in.attention_mask && i < in.attention_mask_n && in.attention_mask[i] == 0)
            continue;  // padding: only the real tokens go in, in order
        ids.push_back(in.lang_tokens[i]);
    }
    const int64_t n = (int64_t) ids.size();
    if (n == 0 || n > max_prompt) {
        std::fprintf(stderr, "%s: prompt of %lld tokens; 1 to %lld are taken\n", kTag, (long long) n, (long long) max_prompt);
        return {};
    }

    // Action noise from the caller; the video noise is drawn here from a seed that follows it.
    std::vector<float> an((size_t) (chunk * adim), 0.f);
    uint64_t           seed = 1469598103934665603ull;  // FNV-1a over the action noise, else a fixed seed
    if (in.noise) {
        std::memcpy(an.data(), in.noise, sizeof(float) * an.size());
        const auto * bytes = (const uint8_t *) in.noise;
        for (size_t i = 0; i < sizeof(float) * an.size(); i++)
            seed = (seed ^ bytes[i]) * 1099511628211ull;
    } else {
        // No noise given: sample it like the other archs do, from VLA_COSMOS3_SEED when set (reproducible).
        std::random_device rd;
        const char *       env = std::getenv("VLA_COSMOS3_SEED");
        std::mt19937_64    rng(env ? (uint64_t) std::strtoull(env, nullptr, 10) + calls : ((uint64_t) rd() << 32) ^ rd());
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float & x : an)
            x = nd(rng);
        seed = rng();  // the video noise follows the action noise just drawn
    }
    std::vector<float> vn((size_t) ((frames - 1) * ppf * patch_dim));
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float & x : vn)
            x = nd(rng);
    }
    calls++;

    // --- the VAE: canvas -> latent of the condition frame -----------------------------------------------
    const double  t1 = now_ms();
    const int64_t W0 = layout.canvas_w / vae_patch, H0 = layout.canvas_h / vae_patch;
    {
        const size_t arena = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
        if (!vae_graph.ensure(backend, 0, arena, [&](ggml_context * C, VaeIO & io) { return build_vae(C, io); })) {
            std::fprintf(stderr, "%s: VAE graph build/alloc failed\n", kTag);
            return {};
        }
        // [-1, 1], then Wan's patchify: channel (c * p + r) * p + q holds pixel (2h + q, 2w + r).
        std::vector<float> px((size_t) (W0 * H0 * 3 * vae_patch * vae_patch));
        for (int64_t c = 0; c < 3; c++)
            for (int64_t q = 0; q < vae_patch; q++)
                for (int64_t r = 0; r < vae_patch; r++) {
                    const int64_t ch = (c * vae_patch + r) * vae_patch + q;
                    for (int64_t y = 0; y < H0; y++)
                        for (int64_t x = 0; x < W0; x++)
                            px[(size_t) ((ch * H0 + y) * W0 + x)] =
                                canvas[(size_t) ((c * layout.canvas_h + (vae_patch * y + q)) * layout.canvas_w + vae_patch * x + r)] / 127.5f - 1.f;
                }
        VaeIO & io = vae_graph.io();
        ggml_backend_tensor_set(io.px, px.data(), 0, ggml_nbytes(io.px));
        graph_unique_names(vae_graph.graph());
        if (ggml_backend_graph_compute(backend, vae_graph.graph()) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s: VAE compute failed\n", kTag);
            return {};
        }
    }
    // The latent [w, h, z], normalized with the VAE's statistics, cropped to the content's grid, patchified
    // ("cthpwq->thwpqc"): patch (h, w) holds (p, q, c) = latent[c, 2h + p, 2w + q].
    std::vector<float> xv((size_t) (n_video * patch_dim));
    {
        const VaeIO & io = vae_graph.io();
        const int64_t lw = io.lat->ne[0], lh = io.lat->ne[1];
        if (lw < latent_w || lh < latent_h || io.lat->ne[2] != z_dim) {
            std::fprintf(stderr, "%s: the VAE produced a %lldx%lld latent, the content needs %lldx%lld\n", kTag,
                         (long long) lh, (long long) lw, (long long) latent_h, (long long) latent_w);
            return {};
        }
        std::vector<float> lat((size_t) (lw * lh * z_dim));
        ggml_backend_tensor_get(io.lat, lat.data(), 0, ggml_nbytes(io.lat));
        const int64_t grid_w = latent_w / latent_patch;
        for (int64_t c = 0; c < z_dim; c++)
            for (int64_t y = 0; y < latent_h; y++)
                for (int64_t x = 0; x < latent_w; x++) {
                    const float   z     = (lat[(size_t) ((c * lh + y) * lw + x)] - lat_mean[(size_t) c]) / lat_std[(size_t) c];
                    const int64_t patch = (y / latent_patch) * grid_w + x / latent_patch;
                    const int64_t d     = ((y % latent_patch) * latent_patch + x % latent_patch) * z_dim + c;
                    xv[(size_t) (patch * patch_dim + d)] = z;
                }
    }
    // The 4 predicted frames start as pure noise (noise level 1), the condition frame stays clean.
    std::memcpy(&xv[(size_t) (ppf * patch_dim)], vn.data(), sizeof(float) * vn.size());
    const double t2 = now_ms();

    // --- the prompt: the understanding stream's keys and values -------------------------------------------
    {
        if (!ensure_kv(n)) {
            std::fprintf(stderr, "%s: cannot allocate the prompt cache\n", kTag);
            return {};
        }
        const size_t arena = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
        bool fresh = false;
        if (!prefix_graph.ensure(backend, n, arena, [&](ggml_context * C, PrefixIO & io) { fresh = true; return build_prefix(C, io, n); })) {
            std::fprintf(stderr, "%s: prefix graph build/alloc failed\n", kTag);
            return {};
        }
        if (fresh)  // the step graph reads kv, which was re-made with the prefix
            step_graph.release();
        PrefixIO & io = prefix_graph.io();
        std::vector<float> cs, sn, mask((size_t) (n * n), 0.f);
        rope_text(n, cs, sn);
        for (int64_t q = 0; q < n; q++)
            for (int64_t k = q + 1; k < n; k++)
                mask[(size_t) (q * n + k)] = -INFINITY;
        ggml_backend_tensor_set(io.ids, ids.data(), 0, ggml_nbytes(io.ids));
        ggml_backend_tensor_set(io.cos, cs.data(), 0, ggml_nbytes(io.cos));
        ggml_backend_tensor_set(io.sin, sn.data(), 0, ggml_nbytes(io.sin));
        ggml_backend_tensor_set(io.mask, mask.data(), 0, ggml_nbytes(io.mask));
        graph_unique_names(prefix_graph.graph());
        if (ggml_backend_graph_compute(backend, prefix_graph.graph()) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s: prefix compute failed\n", kTag);
            return {};
        }
    }
    const double t3 = now_ms();

    // --- the denoising loop ------------------------------------------------------------------------------
    {
        const size_t arena = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
        if (!step_graph.ensure(backend, n, arena, [&](ggml_context * C, StepIO & io) { return build_step(C, io, n); })) {
            std::fprintf(stderr, "%s: step graph build/alloc failed\n", kTag);
            return {};
        }
    }
    StepIO & io = step_graph.io();
    const UniPC   solver(steps, train_steps, shift);
    UniPC::Stream sv, sa;
    std::vector<float> xa = an, vv((size_t) (n_video * patch_dim)), va((size_t) (chunk * adim)), tsin;
    for (int k = 0; k < steps; k++) {
        time_sinusoid(solver.timesteps[(size_t) k], tsin);
        ggml_backend_tensor_set(io.xv, xv.data(), 0, ggml_nbytes(io.xv));
        ggml_backend_tensor_set(io.xa, xa.data(), 0, ggml_nbytes(io.xa));
        ggml_backend_tensor_set(io.tsin, tsin.data(), 0, ggml_nbytes(io.tsin));
        graph_unique_names(step_graph.graph());
        if (ggml_backend_graph_compute(backend, step_graph.graph()) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s: step %d compute failed\n", kTag, k);
            return {};
        }
        ggml_backend_tensor_get(io.vv, vv.data(), 0, ggml_nbytes(io.vv));
        ggml_backend_tensor_get(io.va, va.data(), 0, ggml_nbytes(io.va));
        // mask_velocity: the condition frame has no velocity.
        std::fill(vv.begin(), vv.begin() + (ptrdiff_t) (ppf * patch_dim), 0.f);
        solver.step(k, sv, vv, xv);
        solver.step(k, sa, va, xa);
    }
    const double t4 = now_ms();

    // The chunk in dataset units: the unnormalizer's (x + 1) / 2 * (q99 - q01) + q01.
    std::vector<float> actions((size_t) (chunk * adim));
    for (int64_t t = 0; t < chunk; t++)
        for (int64_t j = 0; j < adim; j++) {
            float denom = q99[(size_t) j] - q01[(size_t) j];
            if (denom == 0.f)
                denom = 1e-8f;
            actions[(size_t) (t * adim + j)] = (xa[(size_t) (t * adim + j)] + 1.f) * denom / 2.f + q01[(size_t) j];
        }

    stats.ms_vision    = (float) (t2 - t1);
    stats.ms_prefill   = (float) (t3 - t2);
    stats.ms_denoise   = (float) (t4 - t3);
    stats.ms_inference = stats.ms_prefill + stats.ms_denoise;
    stats.ms_total     = (float) (now_ms() - t0);
    return actions;
}

}  // namespace

std::unique_ptr<ModelArchBase> cosmos3_edge_create(const std::string & mmproj_path, const std::string & ckpt_path,
                                                   const std::string &, const Options & opts) {
    if (!mmproj_path.empty())
        std::printf("%s: note - mmproj '%s' is ignored (the VAE is baked into the GGUF)\n", kTag, mmproj_path.c_str());
    if (opts.act_dtype.value_or(GGML_TYPE_F32) != GGML_TYPE_F32) {
        std::fprintf(stderr, "%s: --act-dtype bf16 is not supported; the activations stay F32\n", kTag);
        return nullptr;
    }

    auto m = std::make_unique<Cosmos3EdgeModelArch>();
    m->mt  = opts.weight_dtype.value_or(default_weight_dtype(GGML_TYPE_BF16));
    gguf_reader g("cosmos3_edge");
    if (!g.open(ckpt_path))
        return nullptr;
    if (!g.has("cosmos3_edge.architecture")) {
        std::fprintf(stderr, "%s: %s is not a Cosmos3-Edge GGUF\n", kTag, ckpt_path.c_str());
        return nullptr;
    }
    if (!m->load_config(g) || !resolve_num_steps("cosmos3_edge", opts, m->steps))
        return nullptr;

    const Backend b = backend_init(kTag, m->n_threads);
    if (!b.handle)
        return nullptr;
    m->backend = b.handle;
    // An explicit --weight-dtype f32 keeps the convolutions F32 too.
    m->conv_direct = b.is_cuda && cuda_conv_has_mma() && opts.weight_dtype.value_or(GGML_TYPE_F16) != GGML_TYPE_F32;
    if (opts.flash_attn.has_value())
        set_flash_attn(*opts.flash_attn);

    if (!m->load_weights(g))
        return nullptr;

    Config & c        = m->cfg;
    c.n_img           = m->n_video;  // the video tokens of the generation stream
    c.n_lang          = m->max_prompt;
    c.n_state         = 0;
    c.n_prefix        = m->max_prompt;
    c.n_suffix        = m->chunk;
    c.n_full          = c.n_prefix + c.n_img + c.n_suffix;
    c.hidden          = m->hidden;
    c.expert_h        = m->hidden;
    c.intermediate    = m->inter;
    c.expert_inter    = m->inter;
    c.n_q_heads       = m->n_heads;
    c.n_kv_heads      = m->n_kv_heads;
    c.head_dim        = m->head_dim;
    c.q_full_dim      = m->n_heads * m->head_dim;
    c.kv_full_dim     = m->n_kv_heads * m->head_dim;
    c.n_layers        = m->n_layers;
    c.max_state_dim   = 0;
    c.real_state_dim  = 0;
    c.max_action_dim  = m->adim;  // the model's 64 padded channels are dropped in the GGUF
    c.real_action_dim = m->adim;
    c.num_steps       = m->steps;
    c.rms_eps         = m->rms_eps;
    c.norm_eps        = m->rms_eps;
    c.rope_n_dims     = (int) m->head_dim;
    c.rope_freq_base  = m->rope_theta;
    c.denormalized    = true;  // dataset units: [dpos(3), rot6d(6), gripper]

    std::printf("%s: weights resident %.1f MiB (%s GEMMs, %s convs) - %lld MoT layers, hidden %lld, canvas %dx%d, "
                "prompt <= %lld tokens, %d UniPC steps, chunk %lldx%lld\n",
                kTag, ggml_backend_buffer_get_size(m->buf_w) / (1024.0 * 1024.0), dtype_name(m->mt),
                m->conv_direct ? "f16 direct" : "f32 im2col", (long long) m->n_layers, (long long) m->hidden,
                m->layout.canvas_h, m->layout.canvas_w, (long long) m->max_prompt, m->steps, (long long) m->chunk,
                (long long) m->adim);
    std::printf("%s: send %d views in this order: third person, wrist (or one %dx%d canvas); the instruction as "
                "--text (ASCII; needs the tokenizer in the GGUF) or as token ids; noise = %lldx%lld floats\n",
                kTag, m->layout.n_views, m->layout.canvas_w, m->layout.canvas_h, (long long) m->chunk, (long long) m->adim);
    return m;
}

}  // namespace vla
