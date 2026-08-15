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

#include "options.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace vla {

namespace {

bool parse_dtype(const std::string & v, ggml_type & out) {
    if (v == "f32"  || v == "fp32")  {
        out = GGML_TYPE_F32;
        return true;
    }
    if (v == "bf16" || v == "bfp16") {
        out = GGML_TYPE_BF16;
        return true;
    }
    return false;
}

bool parse_bool(const std::string & v, bool & out) {
    if (v == "1" || v == "true"  || v == "on"  || v == "yes") {
        out = true;
        return true;
    }
    if (v == "0" || v == "false" || v == "off" || v == "no")  {
        out = false;
        return true;
    }
    return false;
}


}

const char * dtype_name(ggml_type t) {
    return t == GGML_TYPE_BF16 ? "bf16" : "f32";
}

namespace {
bool g_flash_attn  = false;
bool g_mm_prec_f32 = true;
}

void set_flash_attn(bool on) {
    g_flash_attn = on;
}
bool flash_attn_enabled()    {
    return g_flash_attn;
}

void set_mm_prec_f32(bool on)  {
    g_mm_prec_f32 = on;
}
bool mm_prec_f32_enabled()     {
    return g_mm_prec_f32;
}

const char * Options::usage() {
    return "  --weight-dtype f32|bf16   resident dtype for GEMM weights\n"
           "  --act-dtype f32|bf16      activation dtype (needs CUDA and bf16 weights)\n"
           "  --flash-attn [0|1]        flash attention; faster, changes numerics\n"
           "  --mm-prec default|f32     matmul accumulation precision\n";
}

bool Options::parse_arg(int argc, char ** argv, int & i, std::string & err) {
    const std::string a = argv[i];

    auto next = [&](std::string & v) -> bool {
        if (i+1 >= argc) {
            err = a+" needs a value";
            return false;
        }
        v = argv[++i];
        return true;
    };

    if (a == "--weight-dtype" || a == "--act-dtype") {
        std::string v;
        if (!next(v))
            return false;

        ggml_type t;
        if (!parse_dtype(v, t)) {
            err = a+": expected f32 or bf16, got '"+v+"'";
            return false;
        }
        if (a == "--weight-dtype")
            weight_dtype = t;
        else
            act_dtype    = t;
        return true;
    }

    if (a == "--flash-attn") {
        bool v = true;
        if (i+1 < argc && argv[i+1][0] != '-' && parse_bool(argv[i+1], v))
            ++i;
        flash_attn = v;
        return true;
    }

    if (a == "--mm-prec") {
        std::string v;
        if (!next(v))
            return false;
        if (v == "default")  {
            mm_prec_f32 = false;
            return true;
        }
        if (v == "f32")      {
            mm_prec_f32 = true;
            return true;
        }
        err = "--mm-prec: expected default or f32, got '"+v+"'";
        return false;
    }

    err.clear();
    return false;
}

// These moved to CLI flags. Leaving them silently ignored would let a
// benchmark script measure a configuration it did not ask for.
bool Options::reject_retired_env(std::string & err) {
    static const char * const retired[][2] = {
        {"VLA_GR00T_BF16_WEIGHTS",     "--weight-dtype bf16"},
        {"VLA_JEPA_BF16_WEIGHTS",      "--weight-dtype bf16"},
        {"VLA_BITVLA_BF16_WEIGHTS",    "--weight-dtype bf16"},
        {"VLA_PI0_F32_WEIGHTS",        "--weight-dtype f32"},
        {"VLA_PI05_F32_WEIGHTS",       "--weight-dtype f32"},
        {"VLA_EVO1_F32_WEIGHTS",       "--weight-dtype f32"},
        {"VLA_ADAPTER_F32_WEIGHTS",    "--weight-dtype f32"},
        {"VLA_OPENVLA_OFT_F32_WEIGHTS","--weight-dtype f32"},
        {"VLA_PI0_BF16_ACT",           "--act-dtype bf16"},
        {"VLA_EVO1_BF16_ACT",          "--act-dtype bf16"},
        {"VLA_SMOLVLA_FA",             "--flash-attn"},
        {"VLA_PI0_FA",                 "--flash-attn"},
        {"VLA_EVO1_FA",                "--flash-attn"},
        {"VLA_FLASH_ATTN",             "--flash-attn"},
        {"VLA_MM_PREC",                "--mm-prec"},
        {"VLA_WEIGHT_DTYPE",           "--weight-dtype"},
    };

    for (const auto & r : retired)
        if (std::getenv(r[0])) {
            err = std::string(r[0])+" is no longer read; pass "+r[1]+" instead";
            return false;
        }
    return true;
}

bool Options::load_json(const std::string & path, std::string & err) {
    if (path.empty())
        return true;

    std::ifstream f(path);
    if (!f)
        return true;

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception & e) {
        err = std::string("config json: ")+e.what();
        return false;
    }
    if (!j.contains("runtime") || !j["runtime"].is_object())
        return true;
    const nlohmann::json & r = j["runtime"];

    try {
        ggml_type t;
        if (r.contains("weight_dtype") && parse_dtype(r["weight_dtype"].get<std::string>(), t))
            weight_dtype = t;
        if (r.contains("act_dtype")    && parse_dtype(r["act_dtype"].get<std::string>(),    t))
            act_dtype    = t;
        if (r.contains("flash_attn"))
            flash_attn  = r["flash_attn"].get<bool>();
        if (r.contains("mm_prec"))
            mm_prec_f32 = r["mm_prec"].get<std::string>() == "f32";
    } catch (const std::exception & e) {
        err = std::string("config json runtime: ")+e.what();
        return false;
    }
    return true;
}

}
