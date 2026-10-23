# CUDA: the bit-exact fast path

On CUDA, at the default precision (F32 activations, the checkpoint's BF16 or
F32 weights, no flash attention), every architecture runs a faster path that
returns **the same bits** as the plain ggml graph. Nothing about the model's
arithmetic changes: no lower precision, no reordered sums. Success rates
measured at the defaults therefore carry over unchanged.

On a Jetson AGX Orin it cuts mean latency by 23-40% for every arch except
Octo-Small, which is host-bound, and adds 0 to 207 MiB of device memory
([numbers](../benchmark/jetson-agx-orin.md#against-1012c05)).

The speed comes from three places:

- **Fused nodes** (`src/layers/fused_ops.h`, `src/cuda/vla_cuda_fused.cu`).
  A chain of ggml ops that the CUDA backend would run as separate kernels, each
  reading and writing the whole activation, becomes one `GGML_OP_CUSTOM` node:
  LayerNorm or RMSNorm with its affine or modulation, bias + activation,
  bias + residual (with or without a gate), the Q/K/V split with RoPE into the
  attention layout, the SiLU or GELU gate. A node that feeds only a BF16 GEMM
  writes its output as BF16, which ggml would round to anyway.
- **`gemm_exact`** (`mm_exact`). A BF16-weight GEMM runs on the fastest
  cuBLASLt algorithm, or `gemm_skinny` kernel, that reproduces ggml's own
  `cublasGemmEx` call bit for bit. Only where ggml itself would call cuBLAS
  (more than 16 operand rows).
- **`attn_exact`**. Attention on a fused kernel where it reproduces ggml's
  chain (two batched cuBLAS GEMMs around `soft_max_f32`): an FFMA kernel
  (path A), the same cuBLAS calls with the softmax and the permute fused
  (path B), or a TF32 tensor-core kernel (path C).

BitVLA runs its own CUDA pipeline (`src/kernels/bitvla/`) rather than ggml; there
the ternary GEMM moves to `mma.sync` tensor-core instructions with fused
epilogues, and the row kernels (softmax, norms with their activation
quantization, RoPE) are rewritten. Its products are int8 times a ternary
weight summed in int32, which is exact in any order.

## How the output stays the same

- **Elementwise and norm kernels** round to F32 wherever ggml does, with
  `__fmul_rn` / `__fadd_rn` so the compiler cannot contract two ops into one
  FMA, and use FMA exactly where ggml-cuda's compiled kernels do. The norm and
  softmax kernels keep ggml-cuda's reduction tree: the same block size, the same
  strided per-thread sums, the same shuffle and shared-memory stages.
- **GEMMs and attention** are chosen per shape at first use, on random data:
  every candidate runs next to ggml's own call and is kept only if the outputs
  are identical, on two operand sets. Among the identical ones the fastest is
  timed and kept. Without an identical candidate the node issues ggml's call
  itself. The choice happens once per shape, outside CUDA graph capture.
- **Extra memory** is per shape and kept for the process: the attention
  scratch for path B, a BF16 staging buffer where an F32 operand feeds a BF16
  GEMM, and the cuBLASLt workspace when the chosen algorithm needs one.
- **Multi-view inputs** keep the original graph's GEMM shapes. Where ggml
  multiplies the views as one 3D activation (a batched cuBLAS call), the fused
  graph does the same. Where the original runs one view per graph (GR00T N1.7's
  tower), `fused_gemm_views` issues one GEMM per view. A single GEMM over all
  views would pick other kernels and other bits.

## Coverage

| Architecture | What runs fused |
|---|---|
| GR00T N1.5, N1.6 | SigLIP tower (one view; with several, ggml's graph), Qwen3 LM, N1.5's VL self-attention, DiT |
| GR00T N1.7, VLA-JEPA | Qwen3-VL tower (all views inside the main graph for N1.7), Qwen3 LM, DiT |
| π0, π0.5 | SigLIP tower, Gemma prefix and action expert (RoPE, MQA, adaRMS, gated residuals) |
| SmolVLA | SigLIP tower (attention over 1024 tokens on path C), BF16 GEMMs through `gemm_exact` |
| Evo-1 | InternViT, Qwen2 LM, DiT |
| OpenVLA-OFT, VLA-Adapter | DINOv2 + SigLIP towers and projector, Llama-2 / Qwen2.5 LM |
| TurboVLA | DINOv3 layers, fusion layers, text enhancer and ACT decoder epilogues |
| Octo-Small | Block transformer, diffusion head epilogues |
| BitVLA | Ternary GEMM, epilogues, row kernels, image embeddings kept on the device |

ACT has no fused path.

## Switches

The defaults are the verified configuration. The switches exist for A/B checks.

| Variable | Default | Effect |
|---|---|---|
| `VLA_FUSED_OPS` | on | `0` runs the plain ggml graph. Evo-1 also takes `vit`, `lm` or `dit`, GR00T N1.7 `vit` or `dit`, to limit the fused paths to one part. |
| `VLA_GEMM_EXACT` | on | `0`: BF16 GEMMs stay `ggml_mul_mat`. |
| `VLA_GEMM_LT`, `VLA_GEMM_SKINNY` | on | `0` drops the cuBLASLt or the `gemm_skinny` candidates. |
| `VLA_ATTN_EXACT` | on | `0` keeps ggml's attention chain. `VLA_ATTN_EXACT_FUSED=0` and `VLA_ATTN_TF32=0` drop paths A and C. |
| `VLA_GR00T_FUSED_VIT` | on with the vision fused ops | GR00T N1.7: `0` runs the vision tower as its own graph. |
| `VLA_BITVLA_MMA` | on | `0` runs BitVLA's original kernels. |
| `VLA_BITVLA_DEVICE_EMBEDS` | on | BitVLA: `0` copies the image embeddings through the host. |
| `VLA_GEMM_CACHE=file` | off | Extra cuBLASLt configurations to try, one per line. `VLA_GEMM_SEARCH=1` tries every configuration (slow) and appends the fastest identical ones to the file. Entries carry the cuBLASLt version and are ignored after an upgrade. |
| `VLA_EXACT_TRACE` | off | `1` prints each shape's GEMM and attention choice, `2` every candidate. |

## Verifying

- `ctest -R fused_cuda_ops` runs every fused node against the ggml chain it
  replaces and compares the bytes.
- `tests/predict_check.cpp` (see [CONTRIBUTING.md](../../CONTRIBUTING.md)) on any
  checkpoint: the output must not change with `VLA_FUSED_OPS=0`.

The elementwise and norm kernels reproduce ggml-cuda's kernels as compiled from
the pinned llama.cpp, and have no run-time check. After a llama.cpp bump, run
the two checks above before trusting the fast path; the GEMM and attention
choices re-check themselves at first use.
