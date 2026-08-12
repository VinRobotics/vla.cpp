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

"""Add one extension hook to the fetched ggml CUDA backend.

This is the entire ggml modification. The kernels it enables live in
src/cuda/vla_cuda_bf16.cu as ordinary in-tree code, depending only on the public
ggml header, so a GIT_TAG bump cannot break them.

Why a hook is needed at all
---------------------------
ggml's CUDA backend has no BF16 instantiation for add/mul/unary and asserts F32
in norm/rms_norm/scale, and ggml_cuda_mul_mat_cublas writes F32 into dst->data
unconditionally (for a BF16 dst that is both wrong and twice the bytes the
allocator reserved). There is no way to register kernels for built-in ops from
outside: GGML_OP_CUSTOM is CPU-only. So the backend has to offer one place where
an external implementation gets first refusal.

What it changes (ggml/src/ggml-cuda/ggml-cuda.cu only)
------------------------------------------------------
  1. An exported function pointer, null by default.
  2. One call to it at the top of ggml_cuda_compute_forward. Returning false
     means "not mine", and ggml runs the op exactly as before.
  3. The RMS_NORM+MUL fusion check GGML_ASSERTs F32 rather than declining, so a
     BF16 rms_norm aborts the process before dispatch is ever reached. Those two
     asserts become a return, which is what the surrounding checks already do
     for every other unsupported type.

With the pointer left null this is a no-op, so an unpatched-but-hooked ggml
behaves identically to a stock one.

Usage: scripts/patch_ggml_cuda_ext_hook.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp: CUDA extension hook"

HOOK_DECL = (
    """static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {""",
    """// vla.cpp: CUDA extension hook. Null unless vla::cuda_register_bf16_ops() ran;
// see src/cuda/vla_cuda_bf16.cu, which holds every kernel behind it.
extern "C" {
typedef bool (*ggml_cuda_ext_forward_t)(struct ggml_tensor * dst, void * stream);
__attribute__((visibility("default"))) ggml_cuda_ext_forward_t ggml_cuda_ext_forward = nullptr;
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    if (ggml_cuda_ext_forward && ggml_cuda_ext_forward(dst, (void *) ctx.stream())) {
        return true;
    }

    switch (dst->op) {""",
)

# A BF16 rms_norm reaching this assert kills the process, and GGML_ASSERT is not
# compiled out in Release. Declining the fusion is what the checks immediately
# below already do for an unsupported mul/add type.
FUSION_GUARD = (
    """        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);""",
    """        // vla.cpp: CUDA extension hook - decline instead of aborting, so a BF16
        // rms_norm falls through to the unfused path (and then to the hook).
        if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32) {
            return false;
        }""",
)


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    path = src / "ggml/src/ggml-cuda/ggml-cuda.cu"
    if not path.exists():
        raise SystemExit(f"not a llama.cpp source tree: {src}")

    text = path.read_text()
    if MARKER in text:
        return  # idempotent: re-configure over an already-patched tree

    for old, new in (HOOK_DECL, FUSION_GUARD):
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
