#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Four small fixes to the fetched ggml OpenVINO backend.

ggml-openvino is written against llama.cpp's graphs: one decoder-only
transformer, one position input, an F16 KV cache. vla.cpp drives it with vision
towers and action experts instead, which is legal ggml but nothing the backend
has seen. Each hunk below is a place where an llama.cpp-shaped assumption is
narrower than the ggml contract. They are what carries a vla.cpp vision tower
through translation; see docs/backend/ov.md for what still does not.

  1. ggml-openvino-extra.cpp - pick the *Intel* OpenCL platform.
     `GGML_OPENVINO_DEVICE=GPU` builds an OpenVINO remote context on an OpenCL
     queue and takes the first platform the ICD loader reports. With more than
     one runtime installed (an NVIDIA card next to the Intel iGPU, POCL,
     Rusticl) that is whichever `/etc/OpenCL/vendors/*.icd` sorted first, and
     the GPU plugin only accepts an Intel context -- it aborts at startup with
     "Incompatible OpenCL runtime: program is not in expected ELF format".
     Selecting by `CL_PLATFORM_VENDOR` leaves single-runtime boxes unchanged.

  2. ggml-decoder.cpp - narrow RESHAPE op_case 3.
     Case 3 is the KV-cache flatten, `[512,1024,1,1] -> [1,524288,1,1]`, and it
     emits a shape with `-1` in dim 2 and 1 in dim 3. Its guard only tests
     `src->ne[0]*ne[1]*ne[2] == node->ne[1]`, which also matches the kernel
     reshape inside `ggml_conv_2d` (`[16,16,3,768] -> [768,768]`) and mangles
     it. The real case always has `node->ne[0] == 1`; requiring that sends the
     conv kernel to case 6, the plain reshape.

  3. openvino/op/flash_attn_ext.cpp - convert K/V to F16 with Q.
     The translator converts Q, the mask and the scale to F16 because
     llama.cpp's KV cache already is. vla.cpp keeps K/V in F32, and OpenVINO's
     SDPA rejects mixed input types ("Mixed input types are not supported").
     Converting K/V too matches the precision the translator already chose.

  4. utils.cpp - make the naive-path graph-size threshold settable.
     Graphs under 20 nodes bypass the LLM decoder and translate literally, with
     static shapes and no KV-cache inference. That literal path is the one that
     suits a vision tower, but a vision tower is ~450 nodes. The constant
     becomes `GGML_OPENVINO_NAIVE_GRAPH_SIZE`; src/backend.h defaults it high
     for vla.cpp and an explicit setting still wins.

Idempotent - re-running on a patched tree is a no-op, so a reconfigure that
re-populates the FetchContent source dir is safe either way.

Usage: scripts/patch_ggml_openvino.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp:"

HELPER = """// vla.cpp: select the Intel OpenCL platform. With several OpenCL runtimes
// installed the first platform is not always Intel's, and the OpenVINO GPU
// plugin only accepts an Intel context. Cached: the ICD list cannot change
// under a running process.
static cl_platform_id ggml_openvino_get_intel_platform() {
    static cl_platform_id platform = nullptr;
    static bool searched = false;
    if (searched) {
        return platform;
    }
    searched = true;

    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        return nullptr;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return nullptr;
    }

    for (cl_platform_id p : platforms) {
        char vendor[256] = "";
        if (clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr) != CL_SUCCESS) {
            continue;
        }
        if (strstr(vendor, "Intel") != nullptr) {
            platform = p;
            break;
        }
    }
    return platform;
}

"""

USM_LOOKUP = """        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (%s_fn) clGetExtensionFunctionAddressForPlatform(platform, "%s");
"""

USM_LOOKUP_NEW = """        cl_platform_id platform = ggml_openvino_get_intel_platform();
        if (platform != nullptr) {
            fn = (%s_fn) clGetExtensionFunctionAddressForPlatform(platform, "%s");
"""

# file -> [(anchor, replacement), ...]. Every anchor must match exactly once.
EDITS = {
    "ggml/src/ggml-openvino/ggml-openvino-extra.cpp": [
        ("#include <optional>\n", "#include <optional>\n#include <vector>\n"),
        ("void ggml_openvino_device_config::init() {", HELPER + "void ggml_openvino_device_config::init() {"),
        (
            """        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL platform: %d\\n", err);
            return;
        }
""",
            """        cl_int err;
        cl_platform_id platform = ggml_openvino_get_intel_platform();
        if (platform == nullptr) {
            GGML_LOG_ERROR("Failed to find an Intel OpenCL platform\\n");
            return;
        }
""",
        ),
        (USM_LOOKUP % (("clEnqueueMemFillINTEL",) * 2), USM_LOOKUP_NEW % (("clEnqueueMemFillINTEL",) * 2)),
        (USM_LOOKUP % (("clEnqueueMemcpyINTEL",) * 2), USM_LOOKUP_NEW % (("clEnqueueMemcpyINTEL",) * 2)),
    ],
    "ggml/src/ggml-openvino/ggml-decoder.cpp": [
        (
            """        } else if (src->ne[0] * src->ne[1] * src->ne[2] == node->ne[1]) {
            op_case = 3;""",
            """            // vla.cpp: case 3 is the KV-cache flatten, whose result is always
            // [1, n, 1, 1]. Without the ne[0] test it also swallows the kernel
            // reshape ggml_conv_2d emits and rewrites it to the wrong shape.
        } else if (src->ne[0] * src->ne[1] * src->ne[2] == node->ne[1] && node->ne[0] == 1) {
            op_case = 3;""",
        ),
    ],
    "ggml/src/ggml-openvino/openvino/op/flash_attn_ext.cpp": [
        (
            """    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);""",
            """    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    // vla.cpp: Q, the mask and the scale below are all forced to F16 because
    // llama.cpp's KV cache already is. K/V that arrive as F32 have to come along
    // or SDPA rejects the mix.
    if (k.get_element_type() != ov::element::f16) {
        k = std::make_shared<ov::op::v0::Convert>(k, ov::element::f16);
    }
    if (v.get_element_type() != ov::element::f16) {
        v = std::make_shared<ov::op::v0::Convert>(v, ov::element::f16);
    }""",
        ),
    ],
    "ggml/src/ggml-openvino/utils.cpp": [
        (
            """bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;""",
            """bool is_naive(ggml_cgraph * cgraph) {
    // vla.cpp: the literal translation path suits any graph that is not a
    // decoder-only LLM, so let the caller raise the bar it is chosen under.
    static const int naive_graph_size_threshold = [] {
        const char * env = getenv("GGML_OPENVINO_NAIVE_GRAPH_SIZE");
        return (env && *env) ? atoi(env) : 20;
    }();""",
        ),
    ],
}


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    for rel, edits in EDITS.items():
        src = root / rel
        if not src.is_file():
            print(f"patch_ggml_openvino: {src} not found", file=sys.stderr)
            return 1
        text = src.read_text()
        if MARKER in text:
            print(f"patch_ggml_openvino: {rel} already patched")
            continue
        for anchor, replacement in edits:
            n = text.count(anchor)
            if n != 1:
                print(f"patch_ggml_openvino: anchor matched {n} times in {rel}, expected 1:\n{anchor}",
                      file=sys.stderr)
                return 1
            text = text.replace(anchor, replacement)
        src.write_text(text)
        print(f"patch_ggml_openvino: patched {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
