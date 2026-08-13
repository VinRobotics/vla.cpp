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
  1. Two exported function pointers, null by default.
  2. One call to the first at the top of ggml_cuda_compute_forward. Returning
     false means "not mine", and ggml runs the op exactly as before.
  3. The RMS_NORM+MUL fusion check GGML_ASSERTs F32 rather than declining, so a
     BF16 rms_norm aborts the process before dispatch is ever reached. Those two
     asserts become a return, which is what the surrounding checks already do
     for every other unsupported type.
  4. One call to the second in the ADD/MUL fusion branch of ggml_cuda_try_fuse.
     Fusion happens in ggml_backend_cuda_graph_compute, upstream of
     ggml_cuda_compute_forward, so the hook in (2) never sees a fused node --
     and ggml_cuda_op_fused_binbcast_impl handles F32/F16 only and GGML_ABORTs
     on BF16. Without this the choice is a crash or no fusion at all for BF16
     activations, and the unfused path costs ~18 ms/call on evo1.

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

// Same contract for a fused ADD/MUL run: dst carries src[0] plus n_fuse addends
// in src[1..n_fuse], all sharing one layout, and dst->data is the final output.
typedef bool (*ggml_cuda_ext_fused_binbcast_t)(struct ggml_tensor * dst, int n_fuse, void * stream);
__attribute__((visibility("default"))) ggml_cuda_ext_fused_binbcast_t ggml_cuda_ext_fused_binbcast = nullptr;
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

# The fused ADD/MUL run is assembled here and handed to a kernel that supports
# F32/F16 only. Offer it to the extension first; declining costs one null check.
FUSED_BINBCAST_GUARD = (
    """            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }""",
    """            // vla.cpp: CUDA extension hook - first refusal on the fused node.
            if (!(ggml_cuda_ext_fused_binbcast &&
                  ggml_cuda_ext_fused_binbcast(&fused_node, n_fuse, (void *) cuda_ctx->stream()))) {
                if (node->op == GGML_OP_ADD) {
                    ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
                } else {
                    ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
                }
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

    for old, new in (HOOK_DECL, FUSION_GUARD, FUSED_BINBCAST_GUARD):
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
