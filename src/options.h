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

// Runtime knobs that trade precision for speed. Only options a model
// actually consumes belong here: an advertised flag nothing reads is how a
// benchmark ends up measuring a configuration nobody asked for.
//
// Thread count, solver steps, GR00T embodiment and un-normalisation key stay
// environment variables (VLA_N_THREADS, VLA_NUM_STEPS, VLA_GR00T_EMBODIMENT,
// VLA_*_UNNORM_KEY) until the loaders read them from here.
//
// Runtime knobs that trade precision for speed. Every field is unset by
// default; a model reads one with value_or() so its own default stays at its
// own call site, and both branches are always compiled.
//
// The fastest setting differs per architecture and some of them change
// numerics, so none of this can be decided at build time.

#pragma once

#include "ggml.h"

#include <optional>
#include <string>

namespace vla {

struct Options {
    std::optional<ggml_type>   weight_dtype;
    std::optional<ggml_type>   act_dtype;
    std::optional<bool>        flash_attn;
    std::optional<bool>        mm_prec_f32;

    // Consumes argv[i] (and its value) if it names an option. Returns false
    // with err set on a bad value; leaves i untouched and err empty when the
    // argument is not ours.
    bool parse_arg(int argc, char ** argv, int & i, std::string & err);

    // Fails if a caller still sets one of the env switches these replaced.
    static bool reject_retired_env(std::string & err);

    // Merges the "runtime" object of a policy config.json, if present.
    bool load_json(const std::string & path, std::string & err);

    static const char * usage();
};

const char * dtype_name(ggml_type t);

// Flash attention is decided once per loaded model but read from graph builders
// several call levels down, so it is held here rather than threaded through
// every signature.
void set_flash_attn(bool on);
bool flash_attn_enabled();

void set_mm_prec_f32(bool on);
bool mm_prec_f32_enabled();

}
