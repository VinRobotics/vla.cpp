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

// Resolves -hf user/repo[:file.gguf|:tag] to a local path, shelling out to the
// hf CLI on a miss.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

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
    for (const char * v : {"HOME", "USERPROFILE"})
        if (const char * h = std::getenv(v); h && *h)
            return std::string(h) + "/.cache/vla";
    return ".vla-cache";
}

inline std::string hf_lower(std::string s) {
    for (char & c : s)
        c = (char) std::tolower((unsigned char) c);
    return s;
}

inline std::vector<std::string> hf_local_ggufs(const std::filesystem::path & dir) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec))
        if (it->is_regular_file(ec) && it->path().extension() == ".gguf")
            out.push_back(it->path().lexically_relative(dir).generic_string());
    std::sort(out.begin(), out.end());
    return out;
}

inline bool hf_remote_ggufs(const std::string & repo, std::vector<std::string> & out) {
    const std::string cmd = "hf download " + repo + " --include \"*.gguf\" --dry-run -q";
    std::fprintf(stderr, "vla: %s\n", cmd.c_str());
#ifdef _WIN32
    FILE * fp = _popen(cmd.c_str(), "r");
#else
    FILE * fp = popen(cmd.c_str(), "r");
#endif
    if (!fp)
        return false;
    char buf[1024];
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string s = buf;
        while (!s.empty() && std::isspace((unsigned char) s.back()))
            s.pop_back();
        if (std::filesystem::path(s).extension() == ".gguf" && hf_token_ok(s, true))
            out.push_back(s);
    }
#ifdef _WIN32
    return _pclose(fp) == 0;
#else
    return pclose(fp) == 0;
#endif
}

inline std::vector<std::string> hf_match(const std::vector<std::string> & files, const std::string & want) {
    for (const std::string & f : files)
        if (f == want)
            return {f};
    std::vector<std::string> out;
    const std::string w = hf_lower(want);
    const bool named = std::filesystem::path(want).extension() == ".gguf";
    for (const std::string & f : files) {
        const std::string base = f.substr(f.find_last_of('/')+1);
        if (named ? base == want : (base.rfind("mmproj", 0) != 0 && hf_lower(f).find(w) != std::string::npos))
            out.push_back(f);
    }
    return out;
}

inline void hf_list(const std::string & repo, const char * why, const std::vector<std::string> & files) {
    std::fprintf(stderr, "vla: %s %s; pick one with -hf %s:<file>\n", repo.c_str(), why, repo.c_str());
    for (const std::string & f : files)
        std::fprintf(stderr, "     %s\n", f.c_str());
}

// Returns "" and explains on stderr.
inline std::string hf_resolve(const std::string & spec) {
    namespace fs = std::filesystem;

    const size_t colon = spec.find(':');
    const std::string repo = spec.substr(0, colon);
    const std::string want = (colon == std::string::npos) ? "" : spec.substr(colon+1);

    if (repo.find('/') == std::string::npos || !hf_token_ok(repo, true) ||
        (colon != std::string::npos && !hf_token_ok(want, true))) {
        std::fprintf(stderr, "vla: -hf expects user/repo[:file.gguf|:tag], got '%s'\n", spec.c_str());
        return "";
    }

    const fs::path dir = fs::path(hf_cache_root())/repo;

    std::vector<std::string> hits = hf_match(hf_local_ggufs(dir), want);
    if (hits.size() == 1)
        return (dir/hits[0]).string();
    if (hits.size() > 1) {
        hf_list(repo, "has several cached GGUFs", hits);
        return "";
    }

    // The cache root reaches the shell too, and it comes from VLA_CACHE, HOME or USERPROFILE.
    // A quote in any of them would close the quoting and run the rest.
#ifdef _WIN32
    const char * bad = "\"%";
    const std::string q = "\"";
#else
    const char * bad = "'";
    const std::string q = "'";
#endif
    if (dir.string().find_first_of(bad) != std::string::npos) {
        std::fprintf(stderr, "vla: refusing a cache path containing %s: %s\n", bad, dir.string().c_str());
        return "";
    }

    std::string file = want;
    if (fs::path(want).extension() != ".gguf") {
        std::vector<std::string> remote;
        if (!hf_remote_ggufs(repo, remote)) {
            std::fprintf(stderr,
                         "vla: cannot list %s. Name the file with -hf %s:<file.gguf>, or update the CLI with\n"
                         "     pip install -U \"huggingface_hub[cli]\"\n", repo.c_str(), repo.c_str());
            return "";
        }
        if (remote.empty()) {
            std::fprintf(stderr, "vla: %s has no GGUF\n", repo.c_str());
            return "";
        }
        hits = hf_match(remote, want);
        if (hits.empty()) {
            hf_list(repo, "has no GGUF matching that tag", remote);
            return "";
        }
        if (hits.size() > 1) {
            hf_list(repo, "has several GGUFs", hits);
            return "";
        }
        file = hits[0];
    }

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "vla: cannot create %s: %s\n", dir.string().c_str(), ec.message().c_str());
        return "";
    }

    const std::string cmd = "hf download " + repo + " " + file + " --local-dir " + q + dir.string() + q;
    std::fprintf(stderr, "vla: %s\n", cmd.c_str());

    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr,
                     "vla: download failed (exit %d). The file is the path inside the repo;\n"
                     "     install the CLI with pip install -U \"huggingface_hub[cli]\"\n", rc);
        return "";
    }

    const fs::path got = dir/file;
    if (!fs::is_regular_file(got, ec)) {
        std::fprintf(stderr, "vla: no %s after download\n", got.string().c_str());
        return "";
    }
    return got.string();
}

}  // namespace vla
