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

#include "tokenizer.h"

#ifdef VLA_USE_SPM
#include "sentencepiece_processor.h"
#endif

#include <cctype>
#include <cmath>
#include <cstdio>

namespace vla {
namespace {

const void * kv_array(const gguf_reader& g, const char * arch, const std::string& key,
                      gguf_type type, size_t& n) {
    const int64_t id = gguf_find_key(g.gctx, key.c_str());
    if (id < 0) {
        std::fprintf(stderr, "vla(%s): missing metadata %s\n", arch, key.c_str());
        return nullptr;
    }
    if (gguf_get_kv_type(g.gctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g.gctx, id) != type) {
        std::fprintf(stderr, "vla(%s): %s is not a %s array\n", arch, key.c_str(), gguf_type_name(type));
        return nullptr;
    }
    n = gguf_get_arr_n(g.gctx, id);
    return gguf_get_arr_data(g.gctx, id);
}

bool read_f32_array(const gguf_reader& g, const char * arch, const std::string& key, std::vector<float>& out) {
    size_t n = 0;
    const float * data = (const float *) kv_array(g, arch, key, GGUF_TYPE_FLOAT32, n);
    if (!data)
        return false;
    out.assign(data, data+n);
    return true;
}

}  // namespace

#ifdef VLA_USE_SPM
bool spm_encode(const gguf_reader& g, const char * arch, const std::string& text,
                std::vector<int32_t>& ids, bool add_bos) {
    size_t n = 0;
    const void * data = kv_array(g, arch, std::string(arch) + ".tokenizer.spm_model", GGUF_TYPE_UINT8, n);
    if (!data)
        return false;

    sentencepiece::SentencePieceProcessor sp;
    const auto status = sp.LoadFromSerializedProto(absl::string_view((const char *) data, n));
    if (!status.ok()) {
        std::fprintf(stderr, "vla(%s): sentencepiece LoadFromSerializedProto failed: %s\n",
                     arch, status.ToString().c_str());
        return false;
    }

    const std::vector<int> enc = sp.EncodeAsIds(text);
    ids.clear();
    if (add_bos)
        ids.push_back((int32_t) sp.bos_id());
    ids.insert(ids.end(), enc.begin(), enc.end());
    return true;
}

bool has_spm_tokenizer(const std::string& ckpt_path, const char * arch) {
    if (ckpt_path.size() < 5 || ckpt_path.compare(ckpt_path.size()-5, 5, ".gguf") != 0)
        return false;
    gguf_reader g{arch};
    return g.open(ckpt_path) && g.has((std::string(arch) + ".tokenizer.spm_model").c_str());
}
#else
bool spm_encode(const gguf_reader&, const char * arch, const std::string&, std::vector<int32_t>&, bool) {
    std::fprintf(stderr, "vla(%s): built without SentencePiece (VLA_SPM=OFF)\n", arch);
    return false;
}

bool has_spm_tokenizer(const std::string&, const char *) {
    return false;
}
#endif

bool prompt_text(const char * arch, const std::string& text, const std::vector<float>& state,
                 const std::vector<float>& q01, const std::vector<float>& q99, std::string& out) {
    const std::string a = arch;
    if (a == "pi0") {
        out = text;
        if (out.empty() || out.back() != '\n')
            out += '\n';
        return true;
    }
    if (a == "openvla_oft") {
        std::string lower = text;
        for (char& c : lower)
            c = (char) std::tolower((unsigned char) c);
        out = "In: What action should the robot take to " + lower + "?\nOut:";
        return true;
    }
    if (a != "pi05") {
        std::fprintf(stderr, "vla(%s): no --text prompt template for this arch\n", arch);
        return false;
    }

    if (q01.empty() || q01.size() != q99.size()) {
        std::fprintf(stderr, "vla(pi05): state q01/q99 are missing or differ in length\n");
        return false;
    }
    if (state.size() < q01.size()) {
        std::fprintf(stderr, "vla(pi05): --state has %zu values, the stats describe %zu\n",
                     state.size(), q01.size());
        return false;
    }
    std::string bins;
    for (size_t i=0; i<q01.size(); ++i) {
        const float x = 2.0f*(state[i]-q01[i])/(q99[i]-q01[i])-1.0f;
        int bin = std::isnan(x) ? 255 : -1;
        for (int b=0; b<256; ++b)
            if (-1.0+b/128.0 <= (double) x)
                bin = b;
        bins += (i ? " " : "") + std::to_string(bin);
    }
    const size_t first = text.find_first_not_of(" \t\n\r\f\v");
    std::string cleaned = first == std::string::npos
        ? std::string() : text.substr(first, text.find_last_not_of(" \t\n\r\f\v")-first+1);
    for (char& c : cleaned)
        if (c == '_' || c == '\n')
            c = ' ';
    out = "Task: " + cleaned + ", State: " + bins + ";\nAction: ";
    return true;
}

bool tokenize_prompt(const std::string& ckpt_path, const char * arch, const std::string& text,
                     const std::vector<float>& state, std::vector<int32_t>& ids,
                     std::vector<int32_t>& attention_mask) {
    gguf_reader g{arch};
    if (!g.open(ckpt_path))
        return false;
    const std::string a = arch;

    if (a == "octo") {
        if (!spm_encode(g, arch, text, ids))
            return false;
        const uint32_t eos_id     = g.has("octo.tokenizer.eos_id") ? g.u32("octo.tokenizer.eos_id") : 1;
        const uint32_t pad_id     = g.has("octo.tokenizer.pad_id") ? g.u32("octo.tokenizer.pad_id") : 0;
        const int64_t  max_length = g.has("octo.tokens.language") ? g.u32("octo.tokens.language") : 16;
        if (max_length != 16) {
            std::fprintf(stderr, "vla(octo): octo.tokens.language=%lld, the T5 encoder is built for 16\n",
                         (long long) max_length);
            return false;
        }
        if ((int64_t) ids.size() > max_length-1)
            ids.resize((size_t) (max_length-1));
        ids.push_back((int32_t) eos_id);
        attention_mask.assign(ids.size(), 1);
        ids.resize((size_t) max_length, (int32_t) pad_id);
        attention_mask.resize((size_t) max_length, 0);
        return true;
    }

    std::vector<float> q01, q99;
    if (a == "pi05" && (!read_f32_array(g, arch, "pi05.state.q01", q01) ||
                        !read_f32_array(g, arch, "pi05.state.q99", q99))) {
        std::fprintf(stderr, "vla(pi05): the state goes into the prompt; add its stats with "
                             "scripts/add_tokenizer_to_gguf.py\n");
        return false;
    }
    std::string prompt;
    if (!prompt_text(arch, text, state, q01, q99, prompt) || !spm_encode(g, arch, prompt, ids, true))
        return false;
    const uint32_t max_length = g.u32((a + ".tokenizer_max_length").c_str());
    if (max_length > 0 && ids.size() > max_length)
        ids.resize(max_length);
    if (a == "openvla_oft" && ids.back() != 29871)
        ids.push_back(29871);
    return true;
}

}  // namespace vla
