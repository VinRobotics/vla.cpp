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

// The `vla` serving manifest parser: a full manifest, the defaults, the
// normalizers' round trip, and the errors a bad manifest must name.

#include "models/compiled_manifest.h"

#include <cmath>
#include <cstdio>
#include <string>

using vla::compiled::Manifest;
using vla::compiled::Norm;
using vla::compiled::parse_manifest;

static int g_fail = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_fail;                                                                    \
        }                                                                                \
    } while (0)

static bool parses(const std::string & vla_json, Manifest & m, std::string & err) {
    return parse_manifest("{\"other\": 1, \"vla\": " + vla_json + "}", m, err);
}

static bool rejects(const std::string & vla_json, const char * needle) {
    Manifest m;
    std::string err;
    if (parses(vla_json, m, err)) {
        std::fprintf(stderr, "FAIL: accepted %s\n", vla_json.c_str());
        return false;
    }
    if (err.find(needle) == std::string::npos) {
        std::fprintf(stderr, "FAIL: error '%s' lacks '%s'\n", err.c_str(), needle);
        return false;
    }
    return true;
}

static const char * kFull = R"({
  "version": 1,
  "images": [
    {"input": "images.0", "height": 512, "width": 512, "layout": "NCHW", "fit": "exact",
     "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "mask": "img_masks.0"},
    {"input": "images.1", "height": 240, "width": 320, "fit": "resize"}
  ],
  "lang": {"tokens": "lang_tokens", "mask": "lang_masks", "max_len": 48, "pad_id": 7},
  "state": {"input": "state", "dim": 32, "real_dim": 2, "norm": "mean_std", "mean": [1, 2], "std": [3, 4], "eps": 0.5},
  "noise": {"input": "noise"},
  "actions": {"output": "x", "chunk": 50, "dim": 32, "real_dim": 2, "norm": "min_max", "min": [0, -1], "max": [1, 1]}
})";

static const char * kAct = R"({
  "version": 1,
  "images": [{"input": "front", "height": 480, "width": 640}],
  "state": {"input": "state", "dim": 6, "norm": "none"},
  "actions": {"output": "actions", "chunk": 100, "dim": 6}
})";

int main() {
    {
        Manifest m;
        std::string err;
        CHECK(parses(kFull, m, err));
        if (!err.empty()) std::fprintf(stderr, "%s\n", err.c_str());
        CHECK(m.version == 1);
        CHECK(m.images.size() == 2);
        CHECK(m.images[0].input == "images.0" && m.images[0].mask == "img_masks.0");
        CHECK(m.images[0].h == 512 && m.images[0].w == 512 && !m.images[0].resize);
        CHECK(m.images[0].mean[1] == 0.5f && m.images[0].std_[2] == 0.5f);
        CHECK(m.images[1].h == 240 && m.images[1].w == 320 && m.images[1].resize && m.images[1].mask.empty());
        CHECK(m.images[1].mean[0] == 0.f && m.images[1].std_[0] == 1.f);
        CHECK(m.lang && m.lang->tokens == "lang_tokens" && m.lang->mask == "lang_masks");
        CHECK(m.lang && m.lang->max_len == 48 && m.lang->pad_id == 7);
        CHECK(m.state && m.state->dim == 32 && m.state->real_dim == 2);
        CHECK(m.state && m.state->norm.kind == Norm::MEAN_STD && m.state->norm.eps == 0.5f);
        CHECK(m.state && std::fabs(m.state->norm.normalize(8.f, 1) - (8.f - 2.f)/(4.f + 0.5f)) < 1e-6f);
        CHECK(m.noise == "noise");
        CHECK(m.actions.output == "x" && m.actions.chunk == 50 && m.actions.dim == 32 && m.actions.real_dim == 2);
        CHECK(m.actions.norm.kind == Norm::MIN_MAX);
        // min_max: y = 2 (x - min) / (max - min + eps) - 1 and back.
        const float x = 0.25f;
        const float y = m.actions.norm.normalize(x, 1);
        CHECK(std::fabs(y - (2.f*(0.25f + 1.f)/(2.f + 1e-8f) - 1.f)) < 1e-6f);
        CHECK(std::fabs(m.actions.norm.denormalize(y, 1) - x) < 1e-6f);
        const std::vector<std::string> names = m.input_names();
        const std::vector<std::string> want = {"images.0", "img_masks.0", "images.1", "lang_tokens",
                                               "lang_masks", "state", "noise"};
        CHECK(names == want);
    }
    {
        Manifest m;
        std::string err;
        CHECK(parses(kAct, m, err));
        CHECK(!m.lang && m.noise.empty());
        CHECK(m.state && m.state->real_dim == 6 && m.state->norm.kind == Norm::NONE);
        CHECK(m.state && m.state->norm.normalize(3.f, 0) == 3.f);
        CHECK(m.actions.real_dim == 6 && m.actions.norm.kind == Norm::NONE);
        CHECK(m.actions.norm.denormalize(2.f, 5) == 2.f);
    }
    {
        // mean_std round trip with LeRobot's eps.
        Norm n;
        n.kind = Norm::MEAN_STD;
        n.a = {10.f};
        n.b = {2.f};
        const float y = n.normalize(14.f, 0);
        CHECK(std::fabs(y - 4.f/(2.f + 1e-8f)) < 1e-6f);
        CHECK(std::fabs(n.denormalize(y, 0) - 14.f) < 1e-5f);
    }
    {
        Manifest m;
        std::string err;
        CHECK(!parse_manifest("{\"eos\": [1]}", m, err) && err.find("no 'vla'") != std::string::npos);
        CHECK(!parse_manifest("not json", m, err));
    }
    CHECK(rejects(R"({"version": 2, "images": [], "actions": {"output": "a", "chunk": 1, "dim": 1}})", "version"));
    CHECK(rejects(R"({"version": 1, "actions": {"output": "a", "chunk": 1, "dim": 1}})", "images"));
    CHECK(rejects(R"({"version": 1, "images": []})", "actions"));
    CHECK(rejects(R"({"version": 1, "images": [{"input": "i", "height": 0, "width": 4}],
                     "actions": {"output": "a", "chunk": 1, "dim": 1}})", "height"));
    CHECK(rejects(R"({"version": 1, "images": [{"input": "i", "height": 4, "width": 4, "fit": "pad"}],
                     "actions": {"output": "a", "chunk": 1, "dim": 1}})", "fit"));
    CHECK(rejects(R"({"version": 1, "images": [{"input": "i", "height": 4, "width": 4, "layout": "NHWC"}],
                     "actions": {"output": "a", "chunk": 1, "dim": 1}})", "layout"));
    CHECK(rejects(R"({"version": 1, "images": [{"input": "i", "height": 4, "width": 4, "mean": [0.5]}],
                     "actions": {"output": "a", "chunk": 1, "dim": 1}})", "3 numbers"));
    CHECK(rejects(R"({"version": 1, "images": [], "state": {"input": "s", "dim": 2, "real_dim": 3},
                     "actions": {"output": "a", "chunk": 1, "dim": 1}})", "real_dim"));
    CHECK(rejects(R"({"version": 1, "images": [], "state": {"input": "s", "dim": 2, "norm": "mean_std", "mean": [0, 0]},
                     "actions": {"output": "a", "chunk": 1, "dim": 1}})", "std"));
    CHECK(rejects(R"({"version": 1, "images": [], "state": {"input": "s", "dim": 2, "norm": "quantile"},
                     "actions": {"output": "a", "chunk": 1, "dim": 1}})", "norm"));
    CHECK(rejects(R"({"version": 1, "images": [], "actions": {"output": "a", "chunk": 1, "dim": 2, "norm": "mean_std",
                     "mean": [0, 0], "std": [1]}})", "std"));
    CHECK(rejects(R"({"version": 1, "images": [{"input": "x", "height": 4, "width": 4}],
                     "state": {"input": "x", "dim": 2}, "actions": {"output": "a", "chunk": 1, "dim": 1}})",
                  "more than once"));
    CHECK(rejects(R"({"version": 1, "images": [], "lang": {"tokens": "t", "max_len": 0},
                     "actions": {"output": "a", "chunk": 1, "dim": 1}})", "max_len"));

    if (g_fail) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("compiled_manifest: ok\n");
    return 0;
}
