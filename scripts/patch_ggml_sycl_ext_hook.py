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

The FoldQuant kernels for Intel GPUs live in src/sycl/vla_sycl_foldquant.cpp as
ordinary in-tree code; this is the ggml side, the SYCL counterpart of
scripts/patch_ggml_cuda_ext_hook.py. A FoldQuant GGUF builds two
GGML_OP_CUSTOM nodes per INT8/INT4 site, and ggml runs GGML_OP_CUSTOM on the
CPU backend only, so the SYCL backend has to offer one place where an external
implementation gets first refusal.

What it changes (ggml/src/ggml-sycl/ggml-sycl.cpp only)
--------------------------------------------------------
  1. Two exported function pointers, null by default: the compute hook and a
     supports_op answer for the nodes it claims.
  2. One call to the first at the top of ggml_sycl_compute_forward. Returning
     false means "not mine", and ggml runs the op exactly as before.
  3. One call to the second at the top of the supports_op switch, for
     GGML_OP_CUSTOM only.

With the pointers left null this is a no-op, so a hooked ggml behaves
identically to a stock one.

Usage: scripts/patch_ggml_sycl_ext_hook.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp: SYCL extension hook"

HOOK_DECL = (
    """static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\\n", dst->name, ggml_op_name(dst->op));
    if (!g_sycl_loaded) return false;
""",
    """// vla.cpp: SYCL extension hook. Null unless vla::sycl_register_foldquant_ops()
// ran; see src/sycl/vla_sycl_foldquant.cpp, which holds every kernel behind it.
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
)

SUPPORTS = (
    """static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
""",
    """static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // vla.cpp: SYCL extension hook - the custom nodes the extension claims.
    if (op->op == GGML_OP_CUSTOM) {
        return ggml_sycl_ext_supports && ggml_sycl_ext_supports(op);
    }
""",
)


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    path = src / "ggml/src/ggml-sycl/ggml-sycl.cpp"
    if not path.exists():
        raise SystemExit(f"not a llama.cpp source tree: {src}")

    text = path.read_text()
    if MARKER in text:
        return  # idempotent: re-configure over an already-patched tree

    for old, new in (HOOK_DECL, SUPPORTS):
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"{path}: anchor found {n} times, expected 1. The pinned llama.cpp "
                f"probably moved; re-check this anchor against the new tag.\n"
                f"---\n{old[:400]}\n---"
            )
        text = text.replace(old, new)

    path.write_text(text)


if __name__ == "__main__":
    main()
