# OpenVINO backend - progress report

Status of branch `backend/ov` at `7c2c89e` plus the working-tree changes below,
llama.cpp pinned at `b10729`. Reference doc: [ov.md](ov.md).

Measured on an Intel Core Ultra X7 358H (Panther Lake), Arc B390 iGPU, AI Boost
NPU, Ubuntu 24.04, OpenVINO 2026.2.1. Every fidelity number here - CPU plugin,
iGPU and NPU - was taken on `b10729`. Only the latency table in [ov.md](ov.md)
still dates from `b10331`.

## Where it stands

All ten architectures that can reach this backend are inside the accuracy bar on
the OpenVINO CPU plugin, and nine of the ten on the iGPU as well. The three that used to be wrong
on the GPU - VLA-JEPA, GR00T N1.7 and π0 - are fixed.

| Arch | CPU plugin | iGPU | NPU |
|---|---:|---:|---:|
| Evo-1 | 2.2e-6 | 6.0e-4 | compiler rejects |
| VLA-Adapter | 3.9e-6 | 5.3e-3 | compiler rejects |
| OpenVLA-OFT | 3.9e-6 | 2.2e-3 | compiler rejects |
| π0.5 | 6.1e-5 | 7.6e-4 | 1.6e-3 |
| VLA-JEPA | 7.7e-5 | 2.6e-3 | returns NaN |
| GR00T N1.7 | 3.9e-4 | 2.6e-3 | NPUW throws |
| π0 | 5.8e-4 | 6.6e-5 | **1.7e0 - wrong** |
| GR00T N1.5 | 6.0e-4 | 1.6e-3 | NPUW throws |
| GR00T N1.6 | 1.1e-3 | 1.4e-3 | NPUW throws |
| SmolVLA | 1.4e-3 | 2.6e-3 | 1.1e-2 |
| BitVLA | pins to CPU by design | - | - |

The bar is 2.9e-3, the figure the SYCL backend is held to. Each number is
max\|delta\| against whichever CPU-backend reference is tighter for that arch (see
[Baselines](#baselines)); GR00T N1.5 and SmolVLA land closer to BF16, the rest to
F32. The iGPU computes in F16 and is legitimately looser: VLA-Adapter (5.3e-3)
is the one row outside the bar there, on a peak of 0.62, while being 3.9e-6 on the
CPU plugin. Judge translation fidelity on the CPU plugin and treat the GPU as a
separate precision target. π0 is tighter on the GPU than on the CPU plugin because
it is the one arch that runs the GPU at F32 - see below.

## Fixed this round

**The position-input fix had gone silently dead at `b10729`.** SmolVLA and π0.5
returned `action_len=0` and a shape-inference failure:

```text
Multiply (VariadicSplit[1]:f32[1,113,5,32], Multiply[0]:f32[1,50,1,32])
Argument shapes are inconsistent.
```

113 is SmolVLA's prefix (64 image + 48 language + 1), 50 its suffix. One cos/sin
table, built from one position input, was being applied to both.

The cause is worth recording because the failure mode is invisible. `b10729`
moved the graph-input naming out of `GgmlOvDecoder::get_graph_input_ov_name()` -
the member the patch guards - into a new free function
`get_tensor_graph_input_ov_name()` at `ggml-decoder.cpp:191`, which hardcodes
`return "inp_pos";`. `compute_model_inputs()` and `set_input_output()` call the
free one; the patched member has zero call sites. **The anchor still matched, the
hunk applied cleanly, and the fix stopped doing anything.** Both functions are
guarded now.

A hunk that applies is not a hunk that runs. On every `VLA_LLAMA_TAG` bump, check
that each patched function still has a live caller.

That prompted an audit of the other twelve hunks for the same failure mode:
eleven live, one dead, none refuted by a three-way adversarial check. The dead
one was the imrope shared sin/cos hunk - benign, because its only caller has been
commented out upstream (`// This optimization is error-prone`) since `b10331` and
the per-op path in `translate_rope()` already passes the flag. Retired; two new
ones landed later in the round, leaving thirteen.

The audit also caught `scripts/upstream_split.py` addressing hunks by position in
the edit list. Adding a hunk to the front of a file's list silently handed every
later hunk to the wrong branch - two were swapped - while its own coverage count
still read 29/29, because each index was still used exactly once. Hunks are now
addressed by a unique substring of their anchor, which fails loudly instead.

## What it took, in total

Two changes in vla.cpp, both ordinary correctness fixes invisible on the other
backends:

- **Weight buffers tagged** `..._WEIGHTS` instead of the default `ANY`, which
  ggml-openvino reads as "KV cache". One call site in `src/loader.cpp`.
- **Graph tensors given unique names.** ggml derives a result's name from its
  source, so unnamed intermediates collide; ggml-openvino keys its translation
  map on those names and silently merges them. `vla::graph_unique_names` at each
  `ggml_backend_graph_compute` site; compiles to nothing off OpenVINO.

Plus one default set in `backend_init` (`GGML_OPENVINO_NAIVE_GRAPH_SIZE`, so
non-LLM graphs take the literal translation path), and thirteen fixes to the
fetched ggml OpenVINO backend applied by `scripts/patch_ggml_openvino.py` at
configure time. The load-bearing ones:

| Fix | Assumption it breaks |
|---|---|
| Two elementwise adds never stacked on a GEMM | the plugin folds both into the GEMM and drops the second operand |
| PERMUTE op_case 2 requires a ROPE | any permute of a view is a rope'd query |
| GELU translated as tanh, not erf | ggml's `ggml_gelu` is the exact erf form |
| Position inputs keyed per tensor | a graph has exactly one position input |
| Folded weights padded to full rank | a 2-D weight is only ever a GEMM operand |
| Naive-path graph cache | (speed) that path recompiled on every graph_compute |
| GPU inference precision exposed | F16 is always the right trade on the GPU |

## Fixed: the three archs that were wrong on the iGPU

Two different bugs wearing the same symptom.

**VLA-JEPA and GR00T N1.7: two elementwise adds stacked on a GEMM.** The GPU
plugin folds elementwise ops into the preceding GEMM as post-ops. Given
`ADD(ADD(residual, GEMM), graph_input)` it folds both and the second operand is
silently lost - the result equals the inner add, as though the outer one never
ran. A llama.cpp graph never builds that chain; a VLA does, wherever a tower's
features are added on top of an FFN residual.

Bisected on VLA-JEPA with `GGML_OPENVINO_DEBUG_NODE`, which materialises an
intermediate as an extra `ov::Result` - the only way to observe an interior tensor
here. Its ViT and DiT graphs matched the CPU plugin to 0.2%; the VLM prefill was
already wrong at the end of layer 0; a binary search inside that layer landed on
`node_35`, the FFN residual add under the deepstack add. Only the first three
layers carry a deepstack add, which is why only three nodes mattered.

Addition is associative, so re-hang the outer add on the inner one's non-GEMM
operand and the GEMM keeps a single post-op. VLA-JEPA 5.4e-1 -> 2.6e-3, GR00T N1.7
1.9e0 -> 2.6e-3, CPU plugin unchanged.

Two false starts worth recording. The first attempt tested `inner->src[0]` for the
MUL_MAT, but ggml puts it in `src[1]`, so the guard never fired and the experiment
read as "re-association does not help" when it had simply not run. And a probe
showing the deepstack input bound to different data on each device was an artifact
of the probe itself: a debug `Result` over a Parameter aliases the ggml buffer,
which gallocr had already reused. `ds_pad[0]` is identical on both devices.

**π0: F16 compounding, not a translation bug.** Its error was almost entirely one
element - step 44, dim 6, off by 1.69 while every other value was within 0.013.
Dim 6 is the gripper, a saturating ±1 channel: the CPU flips it at step 44 and the
GPU at step 45. Split by channel, the continuous dims 0-5 are 4.0e-2 and the
gripper alone produces the 1.69.

The cause is that π0 unrolls its whole 10-step denoise loop inside a single
6863-node graph, so the GPU plugin's F16 arithmetic compounds across every step
with nothing to reset it. `GGML_OPENVINO_GPU_PRECISION=f32` puts π0 at 6.6e-5.
It costs about 3x (383 ms -> 1,170 ms), so `backend_init` defaults it for π0 alone,
matched on the exact tag `vla(pi0)` - `vla(pi05)` contains that string and π0.5
neither needs nor gets it, which was checked by measurement.

## Also found, not fixed

`r_ctx->device` is the default `"CPU"` on a GPU run. `ov_runtime_context` is
constructed with `device("CPU")` and `get_ov_runtime_context_ptr()` sets it from
`ggml_openvino_get_device_name()`, yet `naive_compute()` observes `"CPU"` while
`ggml_openvino_get_device_name()` returns `"GPU"` and a remote context exists.
Two decisions read that stale string:

- the `ExecutionMode` hint is applied to the CPU plugin while the model compiles
  for the GPU through the remote context, so upstream's ACCURACY/PERFORMANCE
  choice never reaches the GPU at all;
- `manual_gqa_enabled` defaults to `device == "GPU"`, which is therefore always
  false on a GPU run.

Neither causes the failures above (both were tested directly), so this was left
alone rather than changed blind. It is a genuine upstream defect and worth a
separate report.

## Baselines

OpenVINO folds BF16 weights in as constants and executes them at F32, while
ggml's CPU backend keeps them BF16. Comparing against the default reference
charges the backend for a precision *upgrade* - it made Evo-1 look like 2.7e-3
when it is 2.2e-6. The BF16 and F32 references bracket the answer and which is
tighter is arch-dependent, so report both. For scale, the CPU backend's own
output moves 2.0e-3 (SmolVLA) or 1.1e-2 (VLA-JEPA) from flipping that one flag.

Compare with a guard. Four wrong conclusions in this project came from diffing
against a file that was missing or empty, whose signature is
`max|delta| ~= peak|reference|`. `cmpf.sh` refuses to compare unless both files
exist with equal, non-zero value counts.

## Known issues

- **Do not set `GGML_OPENVINO_CACHE_DIR`.** OpenVINO's on-disk blob cache returns
  a graph that computes the wrong thing on reload - cold run correct, next run
  wrong, nothing logged. `backend_init` clears it and says so.
- **NPU accepts three of ten archs and only two are correct.** Compiler
  alignment rejections (Evo-1, VLA-Adapter), all-NaN output (VLA-JEPA), and one
  NPUW partitioning assertion shared by all three GR00T models - the earlier note
  that N1.5 and N1.6 failed *differently* was wrong; at `b10729` all three report
  `NPUW: Assertion all_ok failed` at `partitioning.cpp:1350`. None are vla.cpp's
  doing. π0 runs but is wrong there, and unlike on the GPU it cannot be fixed:
  forcing F32 makes the NPU refuse to compile the model.
- **SmolVLA's `VLA_TIMING=phase` path is wrong under OpenVINO** on every device.
  The default path that `vla-server` and `vla-cli` use is correct.
- **NPU needs two extra setup steps** beyond the driver: `libze1`, and
  `ZE_ENABLE_ALT_DRIVERS` pointing at `libze_intel_npu.so.1`.
