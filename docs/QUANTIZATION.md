# FoldQuant GGUF: the file contract

vla.cpp runs two kinds of quantized checkpoint:

| | Stock repack | FoldQuant |
|---|---|---|
| Producer | `scripts/quantize_gguf.py` | `scripts/convert_quantized_model_to_gguf.py` from a FoldQuantVLA quantized model (calibrated), `scripts/foldquant_fake_export.py` (uncalibrated) |
| Weights | ggml `Q8_0` / `Q4_0` blocks (block-32 absmax) | INT8 or INT4 codes, per-output-row scale, block-Hadamard-rotated frame, SmoothQuant folded in |
| Activations | float | dynamic per-token INT8 (INT4 in phase 3) |
| Executed by | `ggml_mul_mat` dequantizing at compute | in-tree integer kernels (`src/kernels/foldquant/`) or the CPU reference |
| Backends | all | integer kernels on CUDA, exact reference on CPU, OpenVINO ops on the OpenVINO CPU/GPU plugins; every other backend reads the sites back as float weights (see [Other backends](#other-backends)) |

This page is the canonical description of the FoldQuant file and of the
arithmetic the runtime performs on it. The converters and vla.cpp's loader
and kernels are all written against it; `scripts/inspect_gguf_quant.py`
checks a file against it.

## The file

A FoldQuant GGUF is the file the architecture's converter
(`scripts/convert_<arch>_to_gguf.py`) would write - same tensor names, same
`<arch>.*` keys, BF16 elsewhere - except that each quantized **site** replaces
its float weight with:

| Tensor | Type, ggml `ne` | Meaning |
|---|---|---|
| `<site>.weight` | `I8`, `(K, N)`; INT4: `(K/2, N)` nibble-packed | integer codes, output row `n` contiguous along `k`, in the rotated (and SmoothQuant-folded) frame |
| `<site>.wscale` | `F32 (N)` | per-output-row weight scale |
| `<site>.ascale` | `F32 (K)`, optional | the static SmoothQuant vector `s` the activation is **divided** by; action-module sites only |
| `<site>.bias` | unchanged | |
| norm gammas | unchanged names, **folded** values | the LLM's `attn_norm` / `ffn_norm` already carry SmoothQuant |

`ascale` is the shipped static vector. The dynamic per-token activation
scale is computed at runtime and never stored.

INT4 nibbles: byte `b` of a row holds column `2b` in the low nibble and
`2b+1` in the high nibble, two's complement, values in `[-7, 7]`.

Sites of GR00T N1.5 / N1.6 / N1.7 (KV prefix `gr00t_n1_5` / `gr00t_n1_6` / `gr00t_n1_7`):

- LLM: `vlm.blk.{i}.{attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down}`
- DiT: `aex.dit.{i}.{attn_q, attn_k, attn_v, attn_o, ff0, ff2}`

Sites of pi0.5 (KV prefix `pi05`):

- PaliGemma prefix (LLM recipe): `vlm.blk.{i}.{attn_q, attn_k, attn_v, attn_o,
  ffn_gate, ffn_up, ffn_down}`. The RMSNorm and its folded gamma ride in the
  q/k/v and gate/up act nodes. vla.cpp loads Gemma norms as `1 + w`, so the
  exporter writes `attn_norm` / `ffn_norm` as F32 holding the folded gamma
  minus one.
- Gemma action expert (action recipe): `aex.blk.{i}.{attn_q, attn_k, attn_v,
  attn_o, ffn_gate, ffn_up, ffn_down}`, each with an `ascale` (its input comes
  out of AdaRMS, so there is no gamma to fold into); q/k/v and gate/up share
  one vector per group. The expert's residuals are gated, so its o / down
  GEMMs do not take the fused residual.

q/k/v (and gate/up) are separate tensors with their own `wscale`. Projections
that read the same input share one input transform, so where the loader fuses
them the codes and `wscale` concatenate along `N` and any `ascale` must be
identical across the group - the exporter writes the same vector under each
name, the loader asserts equality and keeps one. The fused groups are the DiT's
`Wqkv` on self-attention blocks and `Wkv` on cross-attention blocks; on a
cross block `attn_q` reads the hidden state while `attn_k`/`attn_v` read the
VL encoder, so `q` carries its own `ascale` (and a different `K`) there.
`scripts/inspect_gguf_quant.py` infers the split from `K`. ViT, VLSA, adaLN, the state/action
encoders and decoders stay float in every scheme.

`K` and `N` must be multiples of 64 at every site. A site whose shape does not
qualify must be left float by the exporter.

## Metadata

All keys are prefixed `<arch>.quant.` (e.g. `gr00t_n1_7.quant.method`).

| Key | Type | Meaning |
|---|---|---|
| `method` | str | `"foldquant"` - the presence test |
| `applied_at` | str | producer (`"foldquant"`, `"foldquant_fake_export"`) |
| `scheme_llm`, `scheme_action` | str | the producer's scheme keys (FoldQuantVLA `w8a8_sr`, `w4a4_shg`, ...), for logs and provenance |
| `llm_weight_bits`, `llm_act_bits` | u32 | 8 or 4 |
| `llm_rot_block_size` | u32 | nominal Hadamard block (64) |
| `action_weight_bits`, `action_act_bits`, `action_rot_block_size` | u32 | same for the action module |
| `action_fold_order` | str | `"before"`: divide by `ascale` before the butterfly; `"after"`: after |
| `act_clip_ratio` | f32 | activation clip, INT4 activations only (LLM) |
| `site_bits` | str | LLM per-site overrides, e.g. `"o:8,down:8"`; keys `qkv \| o \| gateup \| down`; absent keys inherit the module widths |
| `provenance` | str | free text: the quantized model it came from, its base checkpoint and digest |

The rotation block is nominal: both sides narrow it per site to the largest
power of two that divides `K` (`rotation_block_for`), 1 meaning no rotation.

## The arithmetic

Per site, on a token row `x[K]` in fp32, in this order:

1. `[gamma]` RMSNorm with the folded gamma: `y = (x * rstd) * gamma`,
   `rstd = 1 / sqrt(mean(x^2) + eps)`. LLM q/k/v and gate/up sites; the norm is
   fused into the activation node. `attn_o` / `ffn_down` take the raw residual
   stream and skip this step; DiT sites take the adaLN / LayerNorm output.
2. `[fold_order = before, ascale]` `y = y / ascale`
3. `[rot_block > 1]` in-place block butterfly on every `rot_block` elements:
   stages `h = 1, 2, 4, ...`, pairs `(i, i+h)` become `(a+b, a-b)`, then
   `y *= 1/sqrt(rot_block)` (a host-computed float constant).
4. `[fold_order = after, ascale]` `y = y / ascale`
5. `scale = max(clip * amax(|y|) / qmax, 1e-12)`, `inv = 1 / scale`,
   `q = clamp(rint(y * inv), -qmax, qmax)`, `qmax = 127` (INT8) or `7` (INT4),
   `rint` rounding half to even. The reciprocal multiply is what FoldQuant's
   TensorRT kernels do (`rmsnorm_per_row_quant_cuda.cu`), so engine and runtime
   round alike; PyTorch's `x / scale` emulation can differ by one code at a
   rounding boundary.
6. `acc[n] = sum_k q[k] * w[n][k]` in int32; `out[n] = ((float) acc * scale) * wscale[n] (+ bias[n])`.

Weights were prepared offline as `W' = W . H_block^T` (plus the SmoothQuant
fold) and rounded per output row with `wscale = amax_row / qmax`, so `W' y'`
equals `W x` up to quantization.

`tests/foldquant_gemm_check` (CUDA builds, not ctest) times the prologue and
GEMM at GR00T shapes against a cuBLAS BF16 GEMM of the same shape, and
`--stress` re-runs the FFN shapes hundreds of times hashing the outputs, which
is how a kernel race would show. Do not run any of these, or `vla_predict_check`,
while a build relinks `libvla_core.so` / `libggml-cuda.so`: a process that has
the old library mapped executes a mix of old and new code and its output is
garbage that looks exactly like a race.

`src/foldquant_ref.cpp` is this list written out in the kernels' operation order
(per-thread strided partial sums for `mean(x^2)`, a fixed reduction tree, IEEE
sqrt and division, no FMA contraction). It is the one translation unit built
with `-ffp-contract=off`; keep the bodies there rather than in the header, since
an inline copy compiled into a model file picks up the default contraction and
the linker keeps whichever copy it likes. The CPU backend runs it as is; the CUDA
kernels (`src/kernels/foldquant/`, compiled with `-fmad=false`) reproduce it bit
for bit - `tests/test_foldquant_cuda_op.cpp` checks that on isolated nodes and
`VLA_FQ_CHECK=1` re-checks every node inside a real model run. `scripts/foldquant_ref.py`
is the same arithmetic in numpy; `tests/py/test_foldquant_ref.py` pins it to the
C++ test's golden checksum.

## In the graph

Each site is two `GGML_OP_CUSTOM` nodes (`src/layers/fq_linear.h`):

- `fq_act(x[, gamma][, ascale]) -> I8 [K_pack + 16, T]`: per row, the codes
  (`K` bytes, or `K/2` nibble-packed) followed by the fp32 per-token scale at
  byte `K_pack`. Sources are packed without holes (the spec's `has_gamma` /
  `has_ascale` say which follow `x`). It is an ordinary gallocr intermediate,
  shared by every projection that reads the same input (q/k/v, gate/up).
- `fq_gemm(w, blob, wscale[, bias][, residual]) -> F32 [N, T]`. The fifth
  source is the F32 tensor the model would add right after the GEMM (o_proj and
  down/ff2 residuals); the epilogue adds it, one float add, so the result is
  what `ggml_add` would give. When the spec carries a head layout
  (`fq_set_heads`: DiT q/k/v, cross-attention k/v, LLM v) the epilogue writes
  each projection straight in the layout the attention reads, `[hd, T, heads]`
  for Q/K or `[T, hd, heads]` for V, and the model takes views instead of
  permute copies; the numbers are the same, only their addresses move.

The CPU backend executes the custom function. On CUDA the same nodes are
claimed by the extension hook (`src/cuda/vla_cuda_foldquant.cu`, registered by
`foldquant_check_backend` at load) through the magic word in the node's
userdata; a node that violates the contract is declined, and ggml then aborts
on the unsupported op rather than computing something else.

### OpenVINO

On ggml's OpenVINO backend the two nodes are translated into OpenVINO ops
(`src/openvino/foldquant_ov.cpp`, compiled into the backend; the hook that
registers it for `GGML_OP_CUSTOM` is hunk 14 of `scripts/patch_ggml_openvino.py`).
`fq_act` becomes the reference's arithmetic in OpenVINO ops (RMSNorm with the
folded gamma, the ascale divide, the block rotation as a MatMul with the
normalised Hadamard matrix, `scale = max(clip * amax / qmax, 1e-12)`,
`clamp(round_half_even(y * (1 / scale)))`) and carries the codes and the
per-token scale as floats to `fq_gemm`, which multiplies them with the weight
kept as an `i8` or `i4` constant (the W4 nibble bytes are reinterpreted in place;
OpenVINO's `i4` has the same low-nibble-first order) dequantized by `wscale` in
the decompression pattern the plugins keep compressed. The head-laid-out
epilogue is off there (`FqModuleSpec::no_heads`), since a translated graph has
no raw layout for the views to read.

That runs on the CPU and GPU plugins (`GGML_OPENVINO_DEVICE=CPU|GPU`). The NPU
compiler accepts no such graph, so on the NPU the file uses dequant mode below,
as `VLA_FQ_DEQUANT=1` does anywhere. `tests/test_foldquant_ov_op.cpp` runs one
site through the backend against the CPU reference: every code agrees and the
outputs match to under 1e-6 relative, on an Intel CPU and an Arc iGPU. A whole
model follows the CUDA integer path to 1.00000 action cosine at W8A8; at W4A4 a
code that sits on a rounding tie in one place and not the other (the reductions
run in a different float order) flips by one, and with 15 activation levels
those flips compound through the layers and the denoise steps to about 0.999.

### SYCL

On ggml's SYCL backend (Intel GPUs) the nodes are claimed through an extension
hook like CUDA's (`scripts/patch_ggml_sycl_ext_hook.py` adds it to ggml-sycl;
`src/sycl/vla_sycl_foldquant.cpp` registers the kernels at load). `fq_act`
follows the CPU reference's reduction tree: lane `l` of a 32-wide sub-group owns
the 64-element chunks `l, l+32, ...` of a row, and the sum of squares and the
amax are xor butterflies over the sub-group. `fq_gemm` runs oneDNN's int8 matmul
(INT4 weights as oneDNN `s4`, INT4 activations unpacked to `s8`) into an int32
buffer, then the reference's epilogue. Built without oneDNN (`GGML_SYCL_DNN=OFF`),
or with `VLA_FQ_SYCL_GEMM=native`, it uses its own GEMV for up to 32 tokens and
an XMX `joint_matrix` kernel above that; both are much slower.

The integer sums are exact in any order, so the output is bit-identical to the CPU
reference as long as the float steps are: the source is built with
`-ffp-contract=off`, and the device image carries
`-cl-fp32-correctly-rounded-divide-sqrt` for the driver's JIT, which otherwise
approximates division and sqrt and moves codes across rounding ties.
`tests/test_foldquant_sycl_op.cpp` checks every activation byte and every output
bit against the reference, for every bit width, on every GEMM path, at
production shapes up to K = 16384. With a GPU it goes through ggml's SYCL
backend; without one (CI, or `ONEAPI_DEVICE_SELECTOR=opencl:cpu`) it hands the
nodes straight to the extension hook on the OpenCL CPU device, since ggml-sycl
will not start on a CPU. A whole π0.5 model follows the CUDA integer
path to 0.99999 action cosine at W8A8, 0.9997 at W4A4 with INT8 o/down and 0.9991
at W4A4 (FoldQuantVLA's calibrated LIBERO checkpoints). Both GPUs are exact to the same reference, so the gap comes from the
float layers between the sites (the bf16 model's actions differ by up to 7e-4
between them), which land some codes on the other side of a rounding tie; with
15 activation levels those flips compound, as on OpenVINO.

### Other backends

Metal, Vulkan, Hexagon and OpenCL (and the OpenVINO NPU) have no
implementation of the two custom nodes, and vla.cpp drives a single backend with
no per-op fallback.
There `foldquant_check_backend` switches the file to dequant mode: every site is
registered with the loader as a float GEMM weight, rebuilt at upload in the
resident type (`--weight-dtype`) from its codes, `wscale` and `ascale`, and the
arch takes its stock float path. The activation path above is
`x' = R(x / a)` (`fold_order = before`) or `R(x) / a` (`after`), with `R` the
block-normalised Sylvester-Hadamard butterfly, which is symmetric and
orthonormal; so `W_deq x' = (W_deq R diag(1/a)) x` or `(W_deq diag(1/a) R) x`, and
row `n` of the float weight is `R(w_n) / a` or `R(w_n / a)` (`fq_dequant_rows`).
The LLM sites' SmoothQuant vector is already folded into the norm gains the file
carries, which the float path reads as its norm weights.

What that runs is weight-only quantization: the weights keep FoldQuant's
rounding, the activations stay float (no per-token quantization, no INT4 clip),
so the actions are close to, not bit-identical with, the integer path, and
memory and speed are those of the bf16 GGUF. `VLA_FQ_DEQUANT=1` selects the same
mode on CUDA or CPU (on a CPU it is much faster than the exact reference).
`tests/test_foldquant_dequant.cpp` checks the rebuilt weights against the dense
product.

Environment switches: `VLA_FQ_CHECK=1` recomputes every node with the CPU
reference after its kernel and reports mismatches; `VLA_FQ_CPU_REF=1` runs the CPU reference on host copies
of every node (a byte-exact A/B against the kernels; it disables ggml's CUDA
graphs, whose capture cannot contain the host round trip); `VLA_FQ_TRACE=1`
prints each node's shape once per graph build. A/B switches for the graph-level
optimisations, all bit-identical either way: `VLA_FQ_NO_FUSE=1` keeps the
residual add as a separate node, `VLA_FQ_NO_HEADS=1` keeps the permute copies,
`VLA_FQ_PREFETCH_MB=<mb>` makes each GEMM prefetch that much of the next site's
weights into L2 (opt-in; measured slower on Orin).

## Producing a file

- The family converters (`scripts/convert_<arch>_to_gguf.py`) expose
  `convert(ckpt, out, *, writer_factory, ...)`: a writer factory from
  `scripts/gguf_quant_writer.py` turns their output into a FoldQuant file.
  `convert_pi05_to_gguf.py` also takes an OpenPI-converted checkpoint with
  `--config-json` (the lerobot policy fields) and an OpenPI `norm_stats.json`.
- From a quantized model (a quantized policy saved as data: the calibration result and
  every site's integer codes, next to the base checkpoint), with no calibration and
  nothing re-rounded:
  `python scripts/convert_quantized_model_to_gguf.py --quantized-model <model dir> --out model.gguf`.
  FoldQuantVLA writes one in two formats, and the converter takes both (told apart by the
  manifest's `format`, not its file name):
  - FoldQuantVLA's quantized checkpoint (`foldquant.quantized_checkpoint`, format
    `foldquant-quantized-checkpoint` v1): the base checkpoint in which every quantized
    projection's `.weight` is replaced by `.qweight` (int8 `(N, K)`, or int4 nibble-packed
    `(N, K/2)` uint8, low nibble = even column) and `.weight_scale` (fp32 per row), the
    SmoothQuant vectors as `foldquant.<module>.sq.<site>` tensors, and `foldquant_quant.json`
    naming, per site, the checkpoint keys it covers in row order. The directory is its own
    base checkpoint (`--ckpt` is refused). The family converter reads it through a base view
    in which each missing `.weight` is `qweight x weight_scale`; sites are then written as
    codes. A W4A4 DiT also stores its adaLN projections as INT4; vla.cpp keeps adaLN in float,
    so those are written from the dequantized weight, with the scale rounded to bf16 first as
    the TensorRT `AdaLNModInt4` plugin bakes it. Every quantized projection must end up as a
    GGUF site or as that adaLN, or the conversion fails.
  - FoldQuantVLA's earlier fake-quant state (`foldquant.fakequant`, format
    `foldquant-quant-state`: `foldquant_quant.json` + `quant_state.safetensors` beside the
    base checkpoint).

  For both FoldQuantVLA formats the LLM norm gains are folded as FoldQuant's emitter folds
  them (`(w [+ 1 for Gemma]) / s` in fp32 on the gain in the export dtype, rounded once), and
  `--check-onnx <arm>/onnx` byte-compares every site and gain against the plugin ONNX graphs
  the TensorRT engines were built from and fails on any difference.

  The recorded codes and row scales go into the GGUF as they are (FoldQuant's INT4 layout
  is this page's) and each action site ships its SmoothQuant vector as `.ascale`.

  | family | LLM schemes | action schemes | checked |
  |---|---|---|---|
  | GR00T N1.5 / N1.6 / N1.7 | `w8a8_sr`, `w4a4_srg` (+ `site_bits`) | `w8a8_sh`, `w4a4_shg` | fake-quant states `n15_fq`, `n16_fq_clean`, `fq_w8v2`, `fq_w4v2` (LIBERO) and N1.7 SO101 `w8a8` / `w4a4` / `w4a4_od8`: every site and gain byte-identical to the TensorRT graphs; N1.7 SO101 `w8a8` run on the real arm. Quantized checkpoints: unit-tested |
  | pi0.5 | `w8a8_sr`, `w4a4_srg` (+ `site_bits`, activation clip) | `w8a8_sh`, `w4a4_sh`, `w4a4_shg` | FoldQuantVLA `pi05_fq` (LIBERO, `w8a8_sr` + `w4a4_shg`) byte-identical to its TensorRT graphs; action cosine median 0.99991 vs 0.99986 for the engines |
  | any | | `*_sr` (dense learned rotation) | refused: vla.cpp runs the fixed butterfly only |

  In the earlier FoldQuantVLA state, GPTQ (INT4) DiT sites are recorded by name and
  round-to-nearest (INT8) DiT packs by call order (`dit.rtn/{n}`); the converter walks the latter in the emitter's
  per-block order (cross: q, o, kv, ffn0, ffn2; self: qkv, o, ffn0, ffn2), checks every
  shape, and `--check-onnx` confirms the mapping byte for byte. The DiT's adaLN projections
  stay float in vla.cpp, so that format's adaLN packs are not needed. A Hub-style state without the
  base checkpoint files takes `--ckpt`. Action `sq_fold_order` must be `before` (the only order the
  runtime implements); a learned per-site activation clip has no metadata slot yet.

- Uncalibrated, for kernel bring-up and benchmarks:
  `python scripts/foldquant_fake_export.py --in n17-bf16.gguf --out n17-fq.gguf`.
- Check: `python scripts/inspect_gguf_quant.py n17-fq.gguf` (exit 1 on a violation).

## Phases

1. W8A8, butterfly rotation, GR00T N1.6 / N1.7 (this page).
2. W4A8: INT4 codes (`*_weight_bits = 4`, `ne0 = K/2`), INT8 activations; the
   same file layout otherwise. The CUDA GEMM runs the CPU reference for these
   sites until the nibble-unpacking kernel lands.
3. W4A4: INT4 activations with `act_clip_ratio`.
4. Other families: GR00T N1.5 and pi0.5 are wired; pi0, SmolVLA and Evo-1 are not
   (per-module validation taps and dense rotations are also still open).
