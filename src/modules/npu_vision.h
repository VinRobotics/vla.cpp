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

// Prototype: one camera view's vision encoder on the NPU (docs/backend/ptl_npu.md).
//
// A model with >= 2 views sends its last view's preprocessed CHW image to
// scripts/npu_vision_worker.py, which serves that model's vision graph (built
// from the GGUF by scripts/build_vision_ir.py) on the OpenVINO NPU, and encodes
// the other views on its own backend meanwhile. Wire format, native endianness:
// request = u32 n_floats + floats (3 x size x size), reply = u32 n_floats +
// floats (n_img_tokens x hidden), or n_floats = 0 on failure. Any error means
// the caller encodes that view itself.

#pragma once

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace vla {

#ifdef _WIN32
// No Unix-socket client on Windows: report failure, so the caller encodes the
// view itself.
inline bool npu_vision_roundtrip(const char *, const std::vector<float> &, float *, size_t) {
    return false;
}
#else
inline bool npu_vision_roundtrip(const char * sock_path, const std::vector<float> & chw, float * out,
                                 size_t out_floats) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    // A worker that died must not take this process with it (SIGPIPE on the
    // write), and one that stalls must not hold the request: either way the
    // caller encodes the view itself. A view takes ~0.1 s on the NPU.
#ifdef MSG_NOSIGNAL
    const int send_flags = MSG_NOSIGNAL;
#else
    const int send_flags = 0;
#endif
#ifdef SO_NOSIGPIPE
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    const timeval tv{2, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock_path);
    auto io_all = [&](bool wr, void * p, size_t n) {
        char * c = static_cast<char *>(p);
        while (n) {
            const ssize_t k = wr ? ::send(fd, c, n, send_flags) : ::read(fd, c, n);
            if (k <= 0) return false;
            c += k;
            n -= (size_t) k;
        }
        return true;
    };
    bool ok = ::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
    uint32_t n = (uint32_t) chw.size();
    ok = ok && io_all(true, &n, sizeof(n)) && io_all(true, const_cast<float *>(chw.data()), chw.size() * sizeof(float));
    uint32_t m = 0;
    ok = ok && io_all(false, &m, sizeof(m)) && m == out_floats && io_all(false, out, out_floats * sizeof(float));
    ::close(fd);
    return ok;
}
#endif

}  // namespace vla
