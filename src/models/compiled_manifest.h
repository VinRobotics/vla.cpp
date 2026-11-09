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

// The `vla` serving manifest a ggml-compiler GGUF carries in its metadata
// (ggml_compiler.meta -> "vla"): which program input is which camera, the
// instruction, the state and the noise, which output is the action chunk, and
// the preprocessing of each. Version 1 is specified in ggml-compiler's
// docs/SERVING.md. Header-only and runtime-free so it can be unit-tested.

#pragma once

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vla {
namespace compiled {

constexpr int kManifestVersion = 1;

// LeRobot's normalizers: (x - mean) / (std + eps), or 2 (x - min) / (max - min + eps) - 1.
struct Norm {
    enum Kind { NONE, MEAN_STD, MIN_MAX };
    Kind               kind = NONE;
    std::vector<float> a, b;  // mean/std or min/max, real_dim values
    float              eps  = 1e-8f;

    float normalize(float x, size_t i) const {
        switch (kind) {
            case MEAN_STD: return (x - a[i]) / (b[i] + eps);
            case MIN_MAX:  return 2.f*(x - a[i]) / (b[i] - a[i] + eps) - 1.f;
            default:       return x;
        }
    }
    float denormalize(float y, size_t i) const {
        switch (kind) {
            case MEAN_STD: return y*(b[i] + eps) + a[i];
            case MIN_MAX:  return (y + 1.f)*0.5f*(b[i] - a[i] + eps) + a[i];
            default:       return y;
        }
    }
};

struct ImageSpec {
    std::string input;         // [1, 3, h, w] float input
    std::string mask;          // optional [1] input: 1 when the camera was sent
    int64_t     h = 0, w = 0;
    bool        resize = false;  // "fit": "resize" scales another size bilinearly; "exact" refuses it
    float       mean[3] = {0.f, 0.f, 0.f};
    float       std_[3] = {1.f, 1.f, 1.f};
};

struct LangSpec {
    std::string tokens;        // [1, max_len] int input
    std::string mask;          // optional [1, max_len] 0/1 input
    int64_t     max_len = 0;
    int32_t     pad_id  = 0;
};

struct StateSpec {
    std::string input;         // [1, dim] float input
    int64_t     dim = 0, real_dim = 0;
    Norm        norm;
};

struct ActionSpec {
    std::string output;        // [1, chunk, dim] float output
    int64_t     chunk = 0, dim = 0, real_dim = 0;
    Norm        norm;
};

struct Manifest {
    int                      version = 0;
    std::vector<ImageSpec>   images;
    std::optional<LangSpec>  lang;
    std::optional<StateSpec> state;
    std::string              noise;  // input name; empty when the program takes none
    ActionSpec               actions;

    // Every program input the manifest fills, in host order.
    std::vector<std::string> input_names() const {
        std::vector<std::string> n;
        for (const auto & im : images) {
            n.push_back(im.input);
            if (!im.mask.empty()) n.push_back(im.mask);
        }
        if (lang) {
            n.push_back(lang->tokens);
            if (!lang->mask.empty()) n.push_back(lang->mask);
        }
        if (state) n.push_back(state->input);
        if (!noise.empty()) n.push_back(noise);
        return n;
    }
};

namespace detail {

using json = nlohmann::json;

inline const json & need(const json & j, const char * key, const char * where) {
    const auto it = j.find(key);
    if (it == j.end())
        throw std::runtime_error(std::string(where) + " lacks '" + key + "'");
    return *it;
}

inline std::string str(const json & j, const char * key, const char * where) {
    const json & v = need(j, key, where);
    if (!v.is_string() || v.get<std::string>().empty())
        throw std::runtime_error(std::string(where) + "." + key + " must be a non-empty string");
    return v.get<std::string>();
}

inline std::string opt_str(const json & j, const char * key) {
    const auto it = j.find(key);
    return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
}

inline int64_t positive(const json & j, const char * key, const char * where) {
    const json & v = need(j, key, where);
    if (!v.is_number_integer() || v.get<int64_t>() <= 0)
        throw std::runtime_error(std::string(where) + "." + key + " must be a positive integer");
    return v.get<int64_t>();
}

inline std::vector<float> floats(const json & j, const char * key, const char * where, size_t n) {
    const json & v = need(j, key, where);
    if (!v.is_array() || v.size() != n)
        throw std::runtime_error(std::string(where) + "." + key + " must hold " + std::to_string(n) + " numbers");
    std::vector<float> out;
    out.reserve(n);
    for (const auto & x : v) {
        if (!x.is_number())
            throw std::runtime_error(std::string(where) + "." + key + " must hold numbers");
        out.push_back(x.get<float>());
    }
    return out;
}

inline Norm norm(const json & j, const char * where, size_t n) {
    Norm out;
    const std::string kind = opt_str(j, "norm").empty() ? "none" : opt_str(j, "norm");
    if (kind == "none")
        return out;
    if (kind == "mean_std") {
        out.kind = Norm::MEAN_STD;
        out.a = floats(j, "mean", where, n);
        out.b = floats(j, "std", where, n);
    } else if (kind == "min_max") {
        out.kind = Norm::MIN_MAX;
        out.a = floats(j, "min", where, n);
        out.b = floats(j, "max", where, n);
    } else {
        throw std::runtime_error(std::string(where) + ".norm '" + kind + "' is not mean_std, min_max or none");
    }
    if (const auto it = j.find("eps"); it != j.end()) {
        if (!it->is_number())
            throw std::runtime_error(std::string(where) + ".eps must be a number");
        out.eps = it->get<float>();
    }
    return out;
}

inline void real_dims(const json & j, const char * where, int64_t & dim, int64_t & real_dim) {
    dim = positive(j, "dim", where);
    real_dim = j.contains("real_dim") ? positive(j, "real_dim", where) : dim;
    if (real_dim > dim)
        throw std::runtime_error(std::string(where) + ".real_dim exceeds dim");
}

}  // namespace detail

// Parses the whole ggml_compiler.meta JSON and validates its "vla" object.
// Returns false with err set when there is no usable manifest.
inline bool parse_manifest(const std::string & meta_json, Manifest & out, std::string & err) {
    using namespace detail;
    try {
        const json meta = json::parse(meta_json);
        const auto it = meta.find("vla");
        if (it == meta.end() || !it->is_object())
            throw std::runtime_error("metadata has no 'vla' object: the program was compiled without a serving manifest "
                                     "(Program(metadata={'vla': ggml_compiler.vla.manifest(...)}))");
        const json & v = *it;
        Manifest m;
        const json & ver = need(v, "version", "vla");
        if (!ver.is_number_integer() || ver.get<int>() != kManifestVersion)
            throw std::runtime_error("vla.version " + ver.dump() + " is not " + std::to_string(kManifestVersion) +
                                     ", the only version this build reads");
        m.version = kManifestVersion;

        const json & images = need(v, "images", "vla");
        if (!images.is_array())
            throw std::runtime_error("vla.images must be an array");
        for (size_t i = 0; i < images.size(); ++i) {
            const std::string where = "vla.images[" + std::to_string(i) + "]";
            const json & im = images[i];
            if (!im.is_object())
                throw std::runtime_error(where + " must be an object");
            ImageSpec s;
            s.input = str(im, "input", where.c_str());
            s.mask  = opt_str(im, "mask");
            s.h     = positive(im, "height", where.c_str());
            s.w     = positive(im, "width", where.c_str());
            const std::string layout = opt_str(im, "layout").empty() ? "NCHW" : opt_str(im, "layout");
            if (layout != "NCHW")
                throw std::runtime_error(where + ".layout '" + layout + "' is not NCHW");
            const std::string fit = opt_str(im, "fit").empty() ? "exact" : opt_str(im, "fit");
            if (fit == "resize")
                s.resize = true;
            else if (fit != "exact")
                throw std::runtime_error(where + ".fit '" + fit + "' is not exact or resize");
            const std::vector<float> mean = im.contains("mean") ? floats(im, "mean", where.c_str(), 3)
                                                                : std::vector<float>{0.f, 0.f, 0.f};
            const std::vector<float> std_ = im.contains("std") ? floats(im, "std", where.c_str(), 3)
                                                               : std::vector<float>{1.f, 1.f, 1.f};
            for (int c = 0; c < 3; ++c) {
                if (std_[(size_t) c] == 0.f)
                    throw std::runtime_error(where + ".std has a zero");
                s.mean[c] = mean[(size_t) c];
                s.std_[c] = std_[(size_t) c];
            }
            m.images.push_back(std::move(s));
        }

        if (const auto l = v.find("lang"); l != v.end() && !l->is_null()) {
            if (!l->is_object())
                throw std::runtime_error("vla.lang must be an object");
            LangSpec s;
            s.tokens  = str(*l, "tokens", "vla.lang");
            s.mask    = opt_str(*l, "mask");
            s.max_len = positive(*l, "max_len", "vla.lang");
            if (const auto p = l->find("pad_id"); p != l->end()) {
                if (!p->is_number_integer())
                    throw std::runtime_error("vla.lang.pad_id must be an integer");
                s.pad_id = p->get<int32_t>();
            }
            m.lang = std::move(s);
        }

        if (const auto st = v.find("state"); st != v.end() && !st->is_null()) {
            if (!st->is_object())
                throw std::runtime_error("vla.state must be an object");
            StateSpec s;
            s.input = str(*st, "input", "vla.state");
            real_dims(*st, "vla.state", s.dim, s.real_dim);
            s.norm = norm(*st, "vla.state", (size_t) s.real_dim);
            m.state = std::move(s);
        }

        if (const auto n = v.find("noise"); n != v.end() && !n->is_null()) {
            if (!n->is_object())
                throw std::runtime_error("vla.noise must be an object");
            m.noise = str(*n, "input", "vla.noise");
        }

        const json & a = need(v, "actions", "vla");
        if (!a.is_object())
            throw std::runtime_error("vla.actions must be an object");
        m.actions.output = str(a, "output", "vla.actions");
        m.actions.chunk  = positive(a, "chunk", "vla.actions");
        real_dims(a, "vla.actions", m.actions.dim, m.actions.real_dim);
        m.actions.norm = norm(a, "vla.actions", (size_t) m.actions.real_dim);

        std::vector<std::string> names = m.input_names();
        std::sort(names.begin(), names.end());
        for (size_t i = 1; i < names.size(); ++i)
            if (names[i] == names[i-1])
                throw std::runtime_error("vla names input '" + names[i] + "' more than once");

        out = std::move(m);
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

}  // namespace compiled
}  // namespace vla
