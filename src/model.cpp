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

#include "arch.h"
#include "model.h"

#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

namespace vla {

struct Model {
    std::unique_ptr<ModelArchBase> impl;
};

namespace {

bool ends_with_gguf(const std::string& p) {
    if (p.size() < 5) return false;
    return std::strcmp(p.c_str() + p.size() - 5, ".gguf") == 0;
}

bool detect_arch_gguf(const std::string& path, Arch* out) {
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx      = nullptr;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), p);
    if (!gctx) return false;

    auto try_str = [&](const char * key, std::string& val) -> bool {
        const int64_t kid = gguf_find_key(gctx, key);
        if (kid < 0) return false;
        const char * s = gguf_get_val_str(gctx, kid);
        if (!s) return false;
        val = s;
        return true;
    };

    bool ok = false;
    std::string arch_str;
    if (try_str("general.architecture",  arch_str) ||
        try_str("smolvla.architecture",  arch_str) ||
        try_str("pi0.architecture",      arch_str) ||
        try_str("pi05.architecture",     arch_str) ||
        try_str("evo1.architecture",     arch_str) ||
        try_str("gr00t_n1_5.architecture", arch_str) ||
        try_str("gr00t_n1_6.architecture", arch_str) ||
        try_str("gr00t_n1_7.architecture", arch_str) ||
        try_str("bitvla.architecture",     arch_str) ||
        try_str("vla_adapter.architecture", arch_str)) {
        if      (arch_str == "smolvla")    { *out = Arch::SMOLVLA;    ok = true; }
        else if (arch_str == "pi0")        { *out = Arch::PI0;        ok = true; }
        else if (arch_str == "pi05")       { *out = Arch::PI05;       ok = true; }
        else if (arch_str == "evo1")       { *out = Arch::EVO1;       ok = true; }
        else if (arch_str == "gr00t_n1_5") { *out = Arch::GR00T_N1_5; ok = true; }
        else if (arch_str == "gr00t_n1_6") { *out = Arch::GR00T_N1_6; ok = true; }
        else if (arch_str == "gr00t_n1_7") { *out = Arch::GR00T_N1_7; ok = true; }
        else if (arch_str == "bitvla")     { *out = Arch::BITVLA;     ok = true; }
        else if (arch_str == "vla_adapter"){ *out = Arch::VLA_ADAPTER;ok = true; }
    }

    gguf_free(gctx);
    return ok;
}

bool detect_arch_safetensors(const std::string& path, Arch* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
    if (!f || header_size == 0 || header_size > (1u << 28)) return false;
    std::string header(header_size, '\0');
    f.read(header.data(), header_size);
    if (!f) return false;

    if (header.find("vlm_with_expert.vlm.") != std::string::npos) {
        *out = Arch::SMOLVLA;
        return true;
    }
    if (header.find("paligemma_with_expert.paligemma.") != std::string::npos) {

        if (header.find("action_time_mlp_in") != std::string::npos) {
            *out = Arch::PI0;
        } else if (header.find("time_mlp_in") != std::string::npos) {
            *out = Arch::PI05;
        } else {

            return false;
        }
        return true;
    }
    return false;
}

}

bool detect_arch_from_ckpt(const std::string& ckpt_path, Arch* out) {
    if (!out) return false;
    if (ends_with_gguf(ckpt_path)) return detect_arch_gguf(ckpt_path, out);
    return detect_arch_safetensors(ckpt_path, out);
}

Model* model_load(const std::string& mmproj_path, const std::string& ckpt_path,
                  const std::string& config_path) {
    Arch arch;
    if (!detect_arch_from_ckpt(ckpt_path, &arch)) {
        std::fprintf(stderr,
                     "vla: cannot detect architecture from %s "
                     "(unrecognised GGUF KV / safetensors namespace)\n",
                     ckpt_path.c_str());
        return nullptr;
    }

    std::unique_ptr<ModelArchBase> impl;
    switch (arch) {
        case Arch::SMOLVLA:
            std::printf("vla: arch = smolvla\n");
            impl = smolvla_create(mmproj_path, ckpt_path, config_path);
            break;
        case Arch::PI0:
            std::printf("vla: arch = pi0\n");
            impl = pi0_create(mmproj_path, ckpt_path, config_path);
            break;
        case Arch::PI05:
            std::fprintf(stderr,
                         "vla: arch = pi05 - not yet implemented "
                         "(optional companion to pi0)\n");
            return nullptr;
        case Arch::EVO1:
            std::printf("vla: arch = evo1\n");
            impl = evo1_create(mmproj_path, ckpt_path, config_path);
            break;
        case Arch::GR00T_N1_5:
            std::printf("vla: arch = gr00t_n1_5\n");
            impl = gr00t_n1_5_create(mmproj_path, ckpt_path, config_path);
            break;
        case Arch::GR00T_N1_6:
            std::printf("vla: arch = gr00t_n1_6\n");
            impl = gr00t_n1_6_create(mmproj_path, ckpt_path, config_path);
            break;
        case Arch::GR00T_N1_7:
            std::printf("vla: arch = gr00t_n1_7\n");
            impl = gr00t_n1_7_create(mmproj_path, ckpt_path, config_path);
            break;
        case Arch::BITVLA:
            std::printf("vla: arch = bitvla\n");
            impl = bitvla_create(mmproj_path, ckpt_path, config_path);
            break;
        case Arch::VLA_ADAPTER:
            std::printf("vla: arch = vla_adapter\n");
            impl = vla_adapter_create(mmproj_path, ckpt_path, config_path);
            break;
    }
    if (!impl) return nullptr;

    auto* m = new Model();
    m->impl = std::move(impl);
    return m;
}

void model_free(Model* m) {
    delete m;
}

const Config& model_config(const Model* m) {
    return m->impl->cfg;
}

const Stats& last_stats(const Model* m) {
    return m->impl->stats;
}

std::vector<float> predict(Model* m, const Inputs& in) {
    return m->impl->predict(in);
}

}
