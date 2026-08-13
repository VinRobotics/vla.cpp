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

// Resolves -hf user/repo[:file.gguf] to a local path, shelling out to the hf
// CLI on a miss.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace vla {

// Repo ids reach a shell command, so reject anything outside this set.
inline bool hf_token_ok(const std::string & s, bool allow_slash) {
    if (s.empty() || s.size() > 200)
        return false;
    // A leading '/' would make fs::path join replace the cache root instead of
    // extending it, putting the download anywhere on disk.
    if (s.front() == '-' || s.front() == '/' || s.find("..") != std::string::npos)
        return false;
    for (const char c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-' ||
                        (allow_slash && c == '/');
        if (!ok)
            return false;
    }
    return true;
}

inline std::string hf_cache_root() {
    if (const char * e = std::getenv("VLA_CACHE"); e && *e)
        return e;
    if (const char * h = std::getenv("HOME");     h && *h)
        return std::string(h) + "/.cache/vla";
    return ".vla-cache";
}

// Largest non-mmproj .gguf in dir, or "".
inline std::string hf_pick_gguf(const std::filesystem::path & dir, const std::string & want) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::string best;
    uintmax_t best_size = 0;
    for (fs::recursive_directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
        if (!it->is_regular_file(ec) || it->path().extension() != ".gguf")
            continue;
        const std::string name = it->path().filename().string();
        if (!want.empty()) {
            if (name == want)
                return it->path().string();
            continue;
        }
        if (name.rfind("mmproj", 0) == 0)
            continue;
        const uintmax_t sz = it->file_size(ec);
        if (ec)
            continue;
        if (sz > best_size) {
            best_size = sz;
            best = it->path().string();
        }
    }
    return best;
}

// Returns "" and explains on stderr.
inline std::string hf_resolve(const std::string & spec) {
    namespace fs = std::filesystem;

    const size_t colon = spec.find(':');
    const std::string repo = spec.substr(0, colon);
    const std::string file = (colon == std::string::npos) ? "" : spec.substr(colon + 1);

    if (repo.find('/') == std::string::npos || !hf_token_ok(repo, true) ||
        (!file.empty() && !hf_token_ok(file, false))) {
        std::fprintf(stderr, "vla: -hf expects user/repo[:file.gguf], got '%s'\n", spec.c_str());
        return "";
    }

    const fs::path dir = fs::path(hf_cache_root()) / repo;

    std::error_code ec;
    if (fs::is_directory(dir, ec)) {
        const std::string hit = hf_pick_gguf(dir, file);
        if (!hit.empty())
            return hit;
    }

    fs::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "vla: cannot create %s: %s\n", dir.string().c_str(), ec.message().c_str());
        return "";
    }

    // The cache root reaches the shell too, and it comes from VLA_CACHE or HOME.
    // A single quote in either would close the quoting and run the rest.
    if (dir.string().find('\'') != std::string::npos) {
        std::fprintf(stderr, "vla: refusing a cache path containing a quote: %s\n", dir.string().c_str());
        return "";
    }

    std::string cmd = "hf download " + repo;
    if (!file.empty())
        cmd += " " + file;
    cmd += " --local-dir '" + dir.string() + "'";
    std::fprintf(stderr, "vla: %s\n", cmd.c_str());

    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr,
                     "vla: download failed (exit %d). Install the CLI with\n"
                     "     pip install -U \"huggingface_hub[cli]\"\n", rc);
        return "";
    }

    const std::string hit = hf_pick_gguf(dir, file);
    if (hit.empty())
        std::fprintf(stderr, "vla: no .gguf under %s after download\n", dir.string().c_str());
    return hit;
}

}  // namespace vla
