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

"""Add one extension hook to the fetched ggml SYCL backend.

The SYCL counterpart of scripts/patch_ggml_cuda_ext_hook.py. Every kernel behind
the hook is ordinary in-tree code that depends only on the public ggml header,
so a GIT_TAG bump cannot break it:

  * src/sycl/vla_sycl_foldquant.cpp - FoldQuant. A FoldQuant GGUF builds two
    GGML_OP_CUSTOM nodes per INT8/INT4 site, and ggml runs GGML_OP_CUSTOM on the
    CPU backend only.
  * src/sycl/vla_sycl_bf16.cpp - the BF16 ops ggml-sycl cannot run. NORM,
    RMS_NORM and SCALE assert F32 outright, and ggml_sycl_op_mul_mat_sycl hands
    oneDNN a to_dt<float>() dst unconditionally (for a BF16 dst that is both
    wrong and twice the bytes the allocator reserved). ADD/MUL and the unary
    ops already have BF16 instantiations, so they are left to ggml.

Both register through src/sycl/vla_sycl_ext.cpp, which owns the two pointers
below and asks each extension in turn. There is no way to register kernels for
built-in ops from outside, so the backend has to offer one place where an
external implementation gets first refusal.

What it changes
---------------
  1. ggml/src/ggml-sycl/ggml-sycl.cpp: two exported function pointers, null by
     default: the compute hook and a supports_op answer for the nodes it
     claims. The first is called at the top of ggml_sycl_compute_forward with
     ctx.stream(), the same in-order queue the surrounding ops use, so the
     extension needs no synchronisation of its own. Returning false means "not
     mine", and ggml runs the op as usual. The second is called at the top of
     the supports_op switch: final for GGML_OP_CUSTOM, and for a built-in op a
     yes admits what ggml-sycl's own gate refuses (e.g. FLASH_ATTN_EXT with
     mixed K/V types or strides that are not multiples of 16, which the oneDNN
     SDPA path in src/sycl/vla_sycl_attn.cpp takes).
  2. ggml/src/ggml-sycl/fusion.cpp: the RMS_NORM+MUL(+ADD) and RMS_NORM+SCALE
     fusion checks GGML_ASSERT F32 rather than declining, so a BF16 rms_norm
     would abort before dispatch. Those asserts become a return, as the checks
     below them already do for unsupported operand types.

With the pointers left null this is a no-op, so a hooked ggml behaves
identically to a stock one.

Usage: scripts/patch_ggml_sycl_ext_hook.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp: SYCL extension hook"

# (relative path, [(anchor, replacement, expected anchor count)])
EDITS = [
    (
        "ggml/src/ggml-sycl/ggml-sycl.cpp",
        [
            (
                """static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\\n", dst->name, ggml_op_name(dst->op));
    if (!g_sycl_loaded) return false;
""",
                """// vla.cpp: SYCL extension hook. Null until src/sycl/vla_sycl_ext.cpp installs its
// dispatcher (FoldQuant, BF16 ops). The void* is the ggml_backend_sycl_context's
// sycl::queue*, so an extension submits into the same in-order queue.
extern "C" {
typedef bool (*ggml_sycl_ext_forward_t)(struct ggml_tensor * dst, void * queue);
__attribute__((visibility("default"))) ggml_sycl_ext_forward_t ggml_sycl_ext_forward = nullptr;
typedef bool (*ggml_sycl_ext_supports_t)(const struct ggml_tensor * op);
__attribute__((visibility("default"))) ggml_sycl_ext_supports_t ggml_sycl_ext_supports = nullptr;
}

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\\n", dst->name, ggml_op_name(dst->op));
    if (!g_sycl_loaded) return false;
    if (ggml_sycl_ext_forward && ggml_sycl_ext_forward(dst, (void *) ctx.stream())) {
        return true;
    }
""",
                1,
            ),
            (
                """static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
""",
                """static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // vla.cpp: SYCL extension hook - the nodes the extension claims. For a
    // GGML_OP_CUSTOM its answer is final. For a built-in op a yes overrides the
    // switch below, whose kernels may not take operand types or strides the
    // extension does (FLASH_ATTN_EXT), and a no falls through to it.
    if (ggml_sycl_ext_supports && ggml_sycl_ext_supports(op)) {
        return true;
    }
    if (op->op == GGML_OP_CUSTOM) {
        return false;
    }
""",
                1,
            ),
        ],
    ),
    (
        "ggml/src/ggml-sycl/fusion.cpp",
        [
            (
                """        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
""",
                """        // vla.cpp: SYCL extension hook - decline instead of aborting, so a BF16
        // rms_norm falls through to the unfused path (and then to the hook).
        if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32) {
            return false;
        }
""",
                2,
            ),
        ],
    ),
]


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not (src / "ggml/src/ggml-sycl/ggml-sycl.cpp").exists():
        raise SystemExit(f"not a llama.cpp source tree: {src}")

    for rel, edits in EDITS:
        path = src / rel
        text = path.read_text()
        if MARKER in text:
            continue  # idempotent: re-configure over an already-patched tree
        for old, new, want in edits:
            n = text.count(old)
            if n != want:
                raise SystemExit(
                    f"{path}: anchor found {n} times, expected {want}. The pinned llama.cpp "
                    f"probably moved; re-check this anchor against the new tag.\n"
                    f"---\n{old[:400]}\n---"
                )
            text = text.replace(old, new)
        path.write_text(text)


if __name__ == "__main__":
    main()
