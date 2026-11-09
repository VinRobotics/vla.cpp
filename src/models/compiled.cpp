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

// A model compiled by ggml-compiler: the GGUF carries the weights, the op
// graphs and a `vla` serving manifest (compiled_manifest.h). The runtime
// (libggml_compiler_rt, built against this tree's ggml) runs the program on
// the backend backend.h picks; this file is only the glue the manifest
// describes: image resize and normalization, token padding, state
// normalization, noise, and action un-normalization.

#include "arch.h"
#include "backend.h"
#include "models/compiled_manifest.h"
#include "modules/preprocess.h"

#include "ggml_compiler_rt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vla {

namespace {

const char * kTag = "vla(compiled)";

int64_t numel(const ggml_compiler_tensor_info & info) {
    int64_t n = 1;
    for (int i = 0; i < info.ndim; ++i)
        n *= info.shape[i];
    return n;
}

std::string shape_str(const ggml_compiler_tensor_info & info) {
    std::string s = "[";
    for (int i = 0; i < info.ndim; ++i)
        s += (i ? ", " : "") + std::to_string((long long) info.shape[i]);
    return s + "]";
}

// PyTorch F.interpolate(mode="bilinear", align_corners=False, antialias=False)
// of an interleaved RGB view into [oh, ow, 3] floats in [0, 1].
void resize_bilinear_hwc(const ImageView & v, int64_t oh, int64_t ow, std::vector<float> & out) {
    out.resize((size_t) oh*ow*3);
    const int64_t ih = v.h, iw = v.w;
    const float sy = (float) ih / (float) oh, sx = (float) iw / (float) ow;
    auto px = [&](int64_t y, int64_t x, int c) -> float {
        const size_t i = ((size_t) y*iw + x)*3 + c;
        return v.format == PixelFormat::U8 ? ((const uint8_t *) v.data)[i]/255.0f : ((const float *) v.data)[i];
    };
    for (int64_t y = 0; y < oh; ++y) {
        const float fy = std::max(0.f, ((float) y + 0.5f)*sy - 0.5f);
        const int64_t y0 = std::min<int64_t>((int64_t) fy, ih - 1), y1 = std::min<int64_t>(y0 + 1, ih - 1);
        const float ly = fy - (float) y0;
        for (int64_t x = 0; x < ow; ++x) {
            const float fx = std::max(0.f, ((float) x + 0.5f)*sx - 0.5f);
            const int64_t x0 = std::min<int64_t>((int64_t) fx, iw - 1), x1 = std::min<int64_t>(x0 + 1, iw - 1);
            const float lx = fx - (float) x0;
            float * dst = &out[((size_t) y*ow + x)*3];
            for (int c = 0; c < 3; ++c) {
                const float top = px(y0, x0, c)*(1.f - lx) + px(y0, x1, c)*lx;
                const float bot = px(y1, x0, c)*(1.f - lx) + px(y1, x1, c)*lx;
                dst[c] = top*(1.f - ly) + bot*ly;
            }
        }
    }
}

class CompiledArch final : public ModelArchBase {
public:
    CompiledArch() : ModelArchBase(Arch::COMPILED) {}
    ~CompiledArch() override {
        if (model)
            ggml_compiler_free(model);  // frees the backend it was loaded on too
    }

    ggml_compiler_model * model = nullptr;
    compiled::Manifest    mf;
    std::unordered_map<std::string, ggml_compiler_tensor_info> inputs;  // the default entry's
    ggml_compiler_tensor_info out_info{};
    std::mt19937          rng{std::random_device{}()};

    // Scratch, kept across calls.
    std::vector<float>    f32;
    std::vector<float>    hwc;
    std::vector<uint16_t> f16;
    std::vector<int32_t>  i32;

    // Upload n floats (or ints) as input `name`, converting to the dtype the
    // program declares. The count must match the input exactly.
    template <typename T>
    bool set(const std::string & name, const T * data, int64_t n);

    std::vector<float> predict(const Inputs & in) override;
};

template <typename T>
bool CompiledArch::set(const std::string & name, const T * data, int64_t n) {
    const auto it = inputs.find(name);
    if (it == inputs.end()) {
        std::fprintf(stderr, "%s: no program input '%s'\n", kTag, name.c_str());
        return false;
    }
    const ggml_compiler_tensor_info & info = it->second;
    if (numel(info) != n) {
        std::fprintf(stderr, "%s: input '%s' is %s (%lld values), got %lld\n", kTag, name.c_str(),
                     shape_str(info).c_str(), (long long) numel(info), (long long) n);
        return false;
    }
    const void * src = nullptr;
    size_t nbytes = 0;
    switch (info.dtype) {
        case GGML_COMPILER_DTYPE_F32:
            if constexpr (std::is_same_v<T, float>) {
                src = data;
            } else {
                f32.resize((size_t) n);
                for (int64_t i = 0; i < n; ++i) f32[(size_t) i] = (float) data[i];
                src = f32.data();
            }
            nbytes = (size_t) n*sizeof(float);
            break;
        case GGML_COMPILER_DTYPE_F16:
            f16.resize((size_t) n);
            for (int64_t i = 0; i < n; ++i) f16[(size_t) i] = ggml_fp32_to_fp16((float) data[i]);
            src = f16.data();
            nbytes = (size_t) n*sizeof(uint16_t);
            break;
        case GGML_COMPILER_DTYPE_I32:
            if constexpr (std::is_same_v<T, int32_t>) {
                src = data;
            } else {
                i32.resize((size_t) n);
                for (int64_t i = 0; i < n; ++i) i32[(size_t) i] = (int32_t) std::lrint(data[i]);
                src = i32.data();
            }
            nbytes = (size_t) n*sizeof(int32_t);
            break;
    }
    if (!src || nbytes != info.nbytes) {
        std::fprintf(stderr, "%s: input '%s' has an unsupported dtype\n", kTag, name.c_str());
        return false;
    }
    if (ggml_compiler_set_input(model, name.c_str(), src, nbytes) != 0) {
        std::fprintf(stderr, "%s: set_input('%s'): %s\n", kTag, name.c_str(), ggml_compiler_last_error());
        return false;
    }
    return true;
}

std::vector<float> CompiledArch::predict(const Inputs & in) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    stats = Stats{};

    if (in.precomputed_img_emb) {
        std::fprintf(stderr, "%s: precomputed_img_emb is not supported; pass raw images\n", kTag);
        return {};
    }

    // Cameras, in manifest order. A missing camera is allowed only when the
    // program has a mask for it.
    const int n_cams = (int) mf.images.size();
    if (in.n_images > n_cams) {
        std::fprintf(stderr, "%s: the program takes %d camera view(s), got %d\n", kTag, n_cams, in.n_images);
        return {};
    }
    for (int v = 0; v < n_cams; ++v) {
        const compiled::ImageSpec & spec = mf.images[(size_t) v];
        const bool present = v < in.n_images && in.images && in.images[v].data;
        f32.assign((size_t) 3*spec.h*spec.w, 0.f);
        if (present) {
            const ImageView & iv = in.images[v];
            if (iv.w == spec.w && iv.h == spec.h) {
                image_to_chw(iv, spec.mean, spec.std_, f32.data());
            } else if (spec.resize) {
                resize_bilinear_hwc(iv, spec.h, spec.w, hwc);
                const ImageView r{hwc.data(), (int) spec.w, (int) spec.h, PixelFormat::F32_RGB_01};
                image_to_chw(r, spec.mean, spec.std_, f32.data());
            } else {
                std::fprintf(stderr, "%s: view %d is %dx%d; '%s' was compiled for %lldx%lld (fit: exact)\n", kTag, v,
                             iv.w, iv.h, spec.input.c_str(), (long long) spec.w, (long long) spec.h);
                return {};
            }
        } else if (spec.mask.empty()) {
            std::fprintf(stderr, "%s: the program takes %d camera view(s) and '%s' has no mask, got %d\n", kTag,
                         n_cams, spec.input.c_str(), in.n_images);
            return {};
        }
        // set<float> converts through f16/i32, never through f32, so the image
        // can stay in f32.
        if (!set(spec.input, f32.data(), (int64_t) 3*spec.h*spec.w))
            return {};
        if (!spec.mask.empty()) {
            const float on = present ? 1.f : 0.f;
            if (!set(spec.mask, &on, 1))
                return {};
        }
    }

    // Instruction.
    if (mf.lang) {
        const compiled::LangSpec & l = *mf.lang;
        if (!in.lang_tokens || in.n_lang <= 0) {
            std::fprintf(stderr, "%s: the program takes an instruction (up to %lld tokens); none given\n", kTag,
                         (long long) l.max_len);
            return {};
        }
        if (in.n_lang > l.max_len) {
            std::fprintf(stderr, "%s: %d language tokens exceed the compiled maximum %lld\n", kTag, in.n_lang,
                         (long long) l.max_len);
            return {};
        }
        std::vector<int32_t> tok((size_t) l.max_len, l.pad_id);
        std::copy(in.lang_tokens, in.lang_tokens + in.n_lang, tok.begin());
        if (!set(l.tokens, tok.data(), l.max_len))
            return {};
        if (!l.mask.empty()) {
            std::vector<float> mask((size_t) l.max_len, 0.f);
            std::fill(mask.begin(), mask.begin() + in.n_lang, 1.f);
            if (!set(l.mask, mask.data(), l.max_len))
                return {};
        }
    } else if (in.n_lang > 0) {
        std::fprintf(stderr, "%s: this program takes no instruction; drop the language tokens\n", kTag);
        return {};
    }

    // State.
    if (mf.state) {
        const compiled::StateSpec & s = *mf.state;
        if (!in.state) {
            std::fprintf(stderr, "%s: the program takes a %lld-dim state\n", kTag, (long long) s.real_dim);
            return {};
        }
        std::vector<float> st((size_t) s.dim, 0.f);
        for (int64_t i = 0; i < s.real_dim; ++i)
            st[(size_t) i] = s.norm.normalize(in.state[i], (size_t) i);
        if (!set(s.input, st.data(), s.dim))
            return {};
    }

    // Noise: the caller's, or N(0, 1).
    if (!mf.noise.empty()) {
        const int64_t n = numel(inputs.at(mf.noise));
        std::vector<float> z((size_t) n);
        if (in.noise) {
            std::copy(in.noise, in.noise + n, z.begin());
        } else {
            std::normal_distribution<float> normal(0.f, 1.f);
            for (auto & x : z) x = normal(rng);
        }
        if (!set(mf.noise, z.data(), n))
            return {};
    }

    const auto tc = clock::now();
    if (ggml_compiler_run(model) != 0) {
        std::fprintf(stderr, "%s: run failed: %s\n", kTag, ggml_compiler_last_error());
        return {};
    }

    const compiled::ActionSpec & a = mf.actions;
    const int64_t n_out = a.chunk*a.dim;
    std::vector<float> out((size_t) n_out);
    if (out_info.dtype == GGML_COMPILER_DTYPE_F32) {
        if (ggml_compiler_get_output(model, a.output.c_str(), out.data(), out_info.nbytes) != 0) {
            std::fprintf(stderr, "%s: get_output('%s'): %s\n", kTag, a.output.c_str(), ggml_compiler_last_error());
            return {};
        }
    } else {
        f16.resize((size_t) n_out);
        if (ggml_compiler_get_output(model, a.output.c_str(), f16.data(), out_info.nbytes) != 0) {
            std::fprintf(stderr, "%s: get_output('%s'): %s\n", kTag, a.output.c_str(), ggml_compiler_last_error());
            return {};
        }
        for (int64_t i = 0; i < n_out; ++i)
            out[(size_t) i] = ggml_fp16_to_fp32(f16[(size_t) i]);
    }
    for (int64_t t = 0; t < a.chunk; ++t)
        for (int64_t j = 0; j < a.real_dim; ++j) {
            float & y = out[(size_t) (t*a.dim + j)];
            y = a.norm.denormalize(y, (size_t) j);
        }

    ggml_compiler_stats st{};
    ggml_compiler_last_stats(model, &st);
    stats.ms_inference = (float) st.ms_total;
    (void) tc;
    stats.ms_total = std::chrono::duration<float, std::milli>(clock::now() - t0).count();
    return out;
}

}  // namespace

std::unique_ptr<ModelArchBase> compiled_create(const std::string & mmproj_path, const std::string & ckpt_path,
                                               const std::string & config_path, const Options & opts) {
    (void) config_path;
    if (!mmproj_path.empty())
        std::printf("%s: ignoring --mmproj; the program bundles its vision tower\n", kTag);
    if (opts.weight_dtype || opts.act_dtype) {
        std::fprintf(stderr, "%s: weight and activation dtypes are fixed at compile time; use a ggml-compiler "
                             "dtype policy instead of --weight-dtype/--act-dtype\n", kTag);
        return nullptr;
    }

    auto m = std::make_unique<CompiledArch>();

    const Backend b = backend_init(kTag, default_cpu_threads());
    if (!b.handle)
        return nullptr;
    // The runtime owns the backend from here on, failure included.
    m->model = ggml_compiler_load_on(ckpt_path.c_str(), b.handle, nullptr, 0);
    if (!m->model) {
        std::fprintf(stderr, "%s: %s\n", kTag, ggml_compiler_last_error());
        return nullptr;
    }

    if (ggml_compiler_n_symbols(m->model) > 0) {
        std::fprintf(stderr, "%s: the program has symbolic dims; serving them is not supported yet, compile with "
                             "fixed shapes\n", kTag);
        return nullptr;
    }
    const int n_entries = ggml_compiler_n_entries(m->model);
    if (n_entries > 1)
        std::printf("%s: %d entry points; serving the default, '%s'\n", kTag, n_entries,
                    ggml_compiler_entry_name(m->model, 0));

    const char * meta = ggml_compiler_metadata(m->model, "ggml_compiler.meta");
    std::string err;
    if (!compiled::parse_manifest(meta ? meta : "{}", m->mf, err)) {
        std::fprintf(stderr, "%s: %s: %s\n", kTag, ckpt_path.c_str(), err.c_str());
        return nullptr;
    }

    // The manifest must cover the default entry's inputs exactly, and name a
    // real output of the right size.
    for (int i = 0; i < ggml_compiler_entry_n_inputs(m->model, 0); ++i) {
        ggml_compiler_tensor_info info{};
        ggml_compiler_entry_input_info(m->model, 0, i, &info);
        m->inputs[info.name] = info;
    }
    std::vector<std::string> named = m->mf.input_names();
    for (const auto & n : named) {
        if (!m->inputs.count(n)) {
            std::fprintf(stderr, "%s: the manifest names input '%s', which the program does not have\n", kTag,
                         n.c_str());
            return nullptr;
        }
    }
    for (const auto & [n, info] : m->inputs) {
        if (std::find(named.begin(), named.end(), n) == named.end()) {
            std::fprintf(stderr, "%s: program input '%s' %s has no role in the manifest\n", kTag, n.c_str(),
                         shape_str(info).c_str());
            return nullptr;
        }
    }
    bool have_out = false;
    for (int i = 0; i < ggml_compiler_entry_n_outputs(m->model, 0); ++i) {
        ggml_compiler_tensor_info info{};
        ggml_compiler_entry_output_info(m->model, 0, i, &info);
        if (m->mf.actions.output == info.name) {
            m->out_info = info;
            have_out = true;
        }
    }
    if (!have_out) {
        std::fprintf(stderr, "%s: the manifest's actions output '%s' is not a program output\n", kTag,
                     m->mf.actions.output.c_str());
        return nullptr;
    }
    if (numel(m->out_info) != m->mf.actions.chunk*m->mf.actions.dim) {
        std::fprintf(stderr, "%s: output '%s' is %s, not chunk %lld x dim %lld\n", kTag, m->mf.actions.output.c_str(),
                     shape_str(m->out_info).c_str(), (long long) m->mf.actions.chunk, (long long) m->mf.actions.dim);
        return nullptr;
    }
    if (m->out_info.dtype != GGML_COMPILER_DTYPE_F32 && m->out_info.dtype != GGML_COMPILER_DTYPE_F16) {
        std::fprintf(stderr, "%s: output '%s' is not a float tensor\n", kTag, m->mf.actions.output.c_str());
        return nullptr;
    }
    for (const auto & im : m->mf.images) {
        const ggml_compiler_tensor_info & info = m->inputs.at(im.input);
        if (numel(info) != 3*im.h*im.w) {
            std::fprintf(stderr, "%s: input '%s' is %s, not an image of 3 x %lld x %lld\n", kTag, im.input.c_str(),
                         shape_str(info).c_str(), (long long) im.h, (long long) im.w);
            return nullptr;
        }
    }

    Config & c = m->cfg;
    c.n_lang          = m->mf.lang ? m->mf.lang->max_len : 0;
    c.n_state         = m->mf.state ? 1 : 0;
    c.n_suffix        = m->mf.actions.chunk;
    c.max_state_dim   = m->mf.state ? m->mf.state->dim : 0;
    c.real_state_dim  = m->mf.state ? m->mf.state->real_dim : 0;
    c.max_action_dim  = m->mf.actions.dim;
    c.real_action_dim = m->mf.actions.real_dim;
    c.denormalized    = true;

    std::printf("%s: %s on %s; %zu camera(s), %s%s%schunk %lld x %lld action dims\n", kTag,
                ggml_compiler_version(), ggml_compiler_backend_name(m->model), m->mf.images.size(),
                m->mf.lang ? "instruction, " : "", m->mf.state ? "state, " : "", m->mf.noise.empty() ? "" : "noise, ",
                (long long) m->mf.actions.chunk, (long long) m->mf.actions.real_dim);
    for (size_t v = 0; v < m->mf.images.size(); ++v) {
        const auto & im = m->mf.images[v];
        std::printf("%s: view %zu '%s' %lldx%lld%s\n", kTag, v, im.input.c_str(), (long long) im.w, (long long) im.h,
                    im.resize ? " (other sizes resized)" : "");
    }
    return m;
}

}  // namespace vla
