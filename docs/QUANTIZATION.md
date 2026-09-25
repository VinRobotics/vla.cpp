# FoldQuant GGUF: the contract between VLA-OPT and vla.cpp

vla.cpp runs two kinds of quantized checkpoint:

| | Stock repack | FoldQuant |
|---|---|---|
| Producer | `scripts/quantize_gguf.py` | `vla-opt build --target vlacpp` (calibrated), `scripts/foldquant_fake_export.py` (uncalibrated) |
| Weights | ggml `Q8_0` / `Q4_0` blocks (block-32 absmax) | INT8 or INT4 codes, per-output-row scale, block-Hadamard-rotated frame, SmoothQuant folded in |
| Activations | float | dynamic per-token INT8 (INT4 in phase 3) |
| Executed by | `ggml_mul_mat` dequantizing at compute | in-tree integer kernels (`src/kernels/foldquant/`) or the CPU reference |
| Backends | all | CUDA, CPU (others refuse the file at load) |

This page is the canonical description of the FoldQuant file and of the
arithmetic the runtime performs on it. VLA-OPT's exporter and vla.cpp's loader
and kernels are both written against it; `scripts/inspect_gguf_quant.py`
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
| `applied_at` | str | producer (`"vla-opt"`, `"foldquant_fake_export"`) |
| `scheme_llm`, `scheme_action` | str | VLA-OPT scheme keys, for logs and provenance |
| `llm_weight_bits`, `llm_act_bits` | u32 | 8 or 4 |
| `llm_rot_block_size` | u32 | nominal Hadamard block (64) |
| `action_weight_bits`, `action_act_bits`, `action_rot_block_size` | u32 | same for the action module |
| `action_fold_order` | str | `"before"`: divide by `ascale` before the butterfly; `"after"`: after |
| `act_clip_ratio` | f32 | activation clip, INT4 activations only (LLM) |
| `site_bits` | str | LLM per-site overrides, e.g. `"o:8,down:8"`; keys `qkv \| o \| gateup \| down`; absent keys inherit the module widths |
| `provenance` | str | free text: VLA-OPT version and commit, preset id and hash, calibration manifest hash |

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
   `rint` rounding half to even. The reciprocal multiply is what VLA-OPT's
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
on the unsupported op rather than computing something else. Other backends
refuse the file at load: there is no per-op fallback in vla.cpp.

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

- Calibrated (SmoothQuant, GPTQ, the arms measured in VLA-OPT):
  `vla-opt build --policy-type <groot_n1_5|groot_n1_6|groot_n1_7|pi05> --target vlacpp --config <family>/vlacpp/<preset> ...`
  writes `weights/model.gguf` inside the Policy Artifact. The converters
  (`scripts/convert_<arch>_to_gguf.py`) expose `convert(ckpt, out, *, writer_factory, ...)`
  for that; `convert_pi05_to_gguf.py` also takes an OpenPI-converted checkpoint
  with `--config-json` (the lerobot policy fields) and an OpenPI `norm_stats.json`.
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
