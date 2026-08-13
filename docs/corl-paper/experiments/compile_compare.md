# Inference latency — vla.cpp vs PyTorch (eager / `torch.compile`)

- Generated: 2026-08-13
- Branch: `fix/bf16-fusion` (`4d29e95`), llama.cpp pinned at `b10331`
- Suite: `libero_object`, task 0
- Hardware: 1x RTX 3090 (24 GB), Intel i7-14700F (20C/28T)
- Harness: `eval/run_latency_serial.sh` -> `eval/run_latency_compare.sh`

**This revision replaces an earlier one measured on branch `corl`. It is not a
re-run of the same experiment.** Three defects in that document's method were
found while reproducing it, and each changed a conclusion rather than a digit;
they are written up in [What the previous revision got wrong](#what-the-previous-revision-got-wrong)
because the corrections matter more than the new numbers.

Server-side inference time per prediction. The vla.cpp server reports its own
latency per response; the PyTorch server times `select_action` in-process with
CUDA synchronisation on both sides. Neither includes ZeroMQ transport or image
serialisation. Both stacks run `n_action_steps = 1`, so every timed call is a
real forward pass rather than an action-queue pop. Warmup is excluded.

**Every model was measured alone.** One model, one variant, one GPU, nothing
else on the machine. The previous revision used a two-lane driver that ran two
sweeps concurrently across both GPUs, which matters more than it sounds — see
[Why serial](#why-serial).

## Results

| Model | vla.cpp | PyTorch eager | compile (default) | compile (reduce-overhead) | best PyTorch | vla.cpp vs best |
|---|---:|---:|---:|---:|---:|---:|
| `smolvla` | **51.1**¹ | 164.4 | 64.0 | 59.3 | 59.3 | **1.16x faster** |
| `pi0` | **94.2**² | 102.0 | 102.0 | fails³ | 102.0 | **1.08x faster** |
| `evo1` | **137.5**² | 151.1 | 140.9 | 140.5 | 140.5 | **1.02x faster** |
| `gr00t_n1_5` | **68.7** | 94.3 | 88.6 | 87.9 | 87.9 | **1.28x faster** |
| `gr00t_n1_6` | **54.4** | 65.4 | 59.9 | fails³ | 59.9 | **1.10x faster** |
| `gr00t_n1_7` | **53.8** | 62.4 | 59.5 | fails³ | 59.5 | **1.11x faster** |
| `bitvla` | **52.1** | 319.7⁴ | 94.6⁴ | 94.0⁴ | 94.0 | **1.80x faster** |

Milliseconds, lower is better. `vla.cpp vs best` is best-PyTorch / vla.cpp, so
>1x means vla.cpp wins. **vla.cpp is faster on 7/7.**

¹ needs `VLA_MM_PREC=default`. This is not a tuning flag — it is the
configuration the shipped accuracy number was measured under. See
[smolvla and the precision that was never applied](#smolvla-and-the-precision-that-was-never-applied).
² needs `VLA_*_BF16_ACT=1` + `VLA_*_FA=1`. BF16 activations are bit-exact;
flash attention is not. See [Switches](#switches).
³ `reduce-overhead` (CUDA graphs) does not run on these: a tensor escapes the
compiled region and the graph memory pool overwrites it (`pi0`: a tensor in
lerobot's `sample_actions`; `gr00t_n1_6`: Eagle SigLIP2 caches `freqs_cis` as a
module attribute), and `gr00t_n1_7` aborts during capture. Fixing these means
editing the reference implementations, so "best PyTorch" is not drawn from an
equal menu across rows.
⁴ `bitvla`'s PyTorch side needs OpenVLA-OFT's `prismatic` and BitVLA's own
transformers fork, which cannot share the `pt_ref` venv, so it runs on
`eval/run_bitvla_compile_compare.sh`. Those three figures are carried over
unchanged from the previous revision; nothing in this branch touches the
PyTorch side. The vla.cpp figure is fresh.

### Sample sizes

PyTorch columns are n=300; vla.cpp is n=200. **Every vla.cpp figure comes from
one binary** (`4d29e95`) — the four rows this branch does not touch
(`gr00t_*`, `bitvla`) were re-measured on it to confirm, and moved by at most
0.6 ms. Spread makes the differing n immaterial: every vla.cpp row has p95
within 2% of its median.

### Distribution (median / p95, ms)

| Model | vla.cpp | eager | compile (default) | compile (reduce-overhead) |
|---|---|---|---|---|
| `smolvla` | 51.0 / 51.9 | 164.0 / 166.4 | 63.4 / 67.4 | 59.0 / 61.1 |
| `pi0` | 94.1 / 95.2 | 102.0 / 103.7 | 102.0 / 103.4 | — |
| `evo1` | 137.6 / 138.4 | 150.8 / 153.8 | 140.4 / 143.8 | 139.8 / 143.4 |
| `gr00t_n1_5` | 68.6 / 69.3 | 82.8 / 138.9 | 80.6 / 129.2 | 79.7 / 124.6 |
| `gr00t_n1_6` | 54.4 / 54.8 | 65.3 / 66.8 | 59.8 / 60.8 | — |
| `gr00t_n1_7` | 53.8 / 54.2 | 62.1 / 64.0 | 58.4 / 59.4 | — |
| `bitvla` | 52.1 / 52.8 | — | — | — |

vla.cpp's p95 sits within 2% of its median on every row. PyTorch's `gr00t_n1_5`
is the extreme case in the other direction: 82.8 median against 138.9 p95. For a
closed-loop controller the tail is usually what binds, so the mean-based table
understates vla.cpp's practical position — and on `gr00t_n1_5` it overstates
PyTorch's, because that mean is inflated by a tail rather than describing a
typical call.

## What the previous revision got wrong

### 1. `gr00t_n1_5`'s win was mostly PyTorch's host-side jitter

The old revision called 1.20x on `gr00t_n1_5` a decisive kernel win. It is not a
kernel result. Instrumenting the reference's `select_action` in three phases
(`eval/pytorch_ref/policies/gr00t_n15/__init__.py`) over 300 calls:

| phase | mean | med | p95 | min | max |
|---|---:|---:|---:|---:|---:|
| host preprocessing (lerobot pipeline) | 21.3 | 14.8 | **60.7** | 10.6 | **71.3** |
| model forward | 53.6 | 52.4 | 56.7 | 51.5 | 58.9 |
| D2H + unnormalize | 11.0 | 11.0 | 11.1 | 10.8 | 11.2 |

The model is steady to +/-4 ms. All of the variance is host-side Python,
swinging 10.6 -> 71.3 ms per call, which `torch.compile` cannot touch — and
indeed `fwd` is 53.6 eager, 54.3 compiled, 54.5 with CUDA graphs. **Compile does
nothing to this model's forward.** The 100.5 -> 94.7 "compile win" the old
revision reported was preprocessing noise.

On model time alone (`fwd + post` ~= 65 ms) PyTorch is *faster* than vla.cpp's
69.3. vla.cpp wins end-to-end because its preprocessing is ~5 ms of C++ against
lerobot's 10-71 ms of Python. That is a real deployment property and worth
having, but it is not the claim the old table made.

`gr00t_n1_5`'s PyTorch numbers also move 6-7 ms run to run on an idle machine
with identical code, for the same reason. Do not read any single one of them
too precisely, including the ones in this table.

### 2. Every profile was taken at the wrong granularity

The old revision's kernel breakdown — *"only 53.6% of GPU time is in GEMM or
attention"*, and the three causes built on it — was measured with `nsys`
defaulting to `--cuda-graph-trace=graph`. That reports each CUDA **graph launch
as one entry** instead of itemising the kernels inside it. Every model here
replays CUDA graphs, so those profiles saw a small fraction of the real work: on
`evo1`, summing the reported kernels gives ~160 ms across 60 requests against
~8.8 s of actual GPU time — about 2%.

**Treat that section of the old document as unusable, not merely imprecise.**
The corrected profile (`--cuda-graph-trace=node`) is what the `evo1` work below
is based on. Whether cause #2 (fusion, ~15%) and cause #3 (layout copies, 5.2%)
survive re-derivation is untested; they were sized from the bad profile.

### 3. The baselines came from a different branch, not an earlier commit

The old figures were measured on branch `corl`, whose `src/` differs from `main`
across 30 files: a pre-refactor `smolvla.cpp`, no `src/cuda/vla_cuda_bf16.cu`
(so BF16 support lived inside a multi-file ggml patch), and llama.cpp pinned at
**b9866** rather than b10331. So "post-merge regression" comparisons against it
were really `corl` vs `main`.

Rebuilding `corl` reproduces its numbers on today's machine (`smolvla` 55.7 vs
55.8 published; 67.7 vs 68.5), so the old figures were sound *for that tree*.
They were just never a baseline for this one.

## smolvla and the precision that was never applied

`smolvla` measured 8.3 ms slower on `main` than on `corl`, which looked like a
regression and was reported as one. It is not.

`smolvla` stamps `GGML_PREC_F32` on **every** tower weight matmul, via `mm_w()`
— unlike every other arch here, which sets it only on the `kq` attention-score
product where F32 accumulation buys softmax stability. llama.cpp b9866 silently
ignored that request for BF16 matmuls; b10331 honours it, routing all those
GEMMs onto the F32 cuBLAS path.

The control is decisive — dropping the request changes nothing on the old build
and everything on the new one:

| `smolvla`, FA on | F32 prec requested | BF16 prec (`VLA_MM_PREC=default`) | cost of honouring |
|---|---:|---:|---:|
| llama b9866 | 55.3 | 55.7 | **0.0 — ignored** |
| llama b10331 | 63.2 | **51.1** | 12.1 |

So `main` is not slower; it is doing arithmetic we asked for and never previously
received. Two consequences:

- The published 55.8 ms **and its 96/100 success rate** were both obtained at
  BF16 GEMM precision. The old document describes a configuration that build
  could not deliver.
- Today's default is therefore the *slower and unvalidated* one, and
  `VLA_MM_PREC=default` is the *faster and already-measured* one. Flipping the
  default returns to the validated configuration rather than departing from it.

**Not yet flipped.** It is still a numerics change, and the argument rests on
inferring what an old build did rather than on a direct measurement. One
100-episode SR arm at `VLA_MM_PREC=default` on this build (~2 h, expect ~96/100)
would settle it. Until then the table's 51.1 requires the flag.

## `evo1`: what moving kernels in-house cost, and getting it back

Commit `5d9ec9e` (pre-dating this branch) moved BF16 activation support out of
ggml's own `binbcast.cu`/`norm.cu` and behind a single extension hook, with the
kernels reimplemented in `src/cuda/vla_cuda_bf16.cu`. That had three effects, all
discovered here:

| state | `evo1` BF16+FA |
|---|---:|
| on merged `main`, before this branch | **crash** |
| fusion declined (`GGML_CUDA_DISABLE_FUSION`) | 169.5 |
| fused-add hook (`ad9a56b`) | 166.6 |
| row-addressed kernels (`0daf584`) | 146.9 |
| 8-wide vectorized rows (`4d29e95`) | **137.5** |
| `corl` at b9866, via ggml's own BF16 kernels | 142.3 |

**The crash.** ggml fuses runs of elementwise nodes in
`ggml_backend_cuda_graph_compute`, *upstream* of `ggml_cuda_compute_forward` —
so the hook never got first refusal on a fused node, and
`ggml_cuda_op_fused_binbcast_impl` handles F32/F16 only and `GGML_ABORT`s on
BF16. `ad9a56b` adds one more exported pointer beside `ggml_cuda_ext_forward`
and one guarded call site in the fusion branch; the fused BF16 kernel lives in
`src/cuda/`. The llama.cpp patch remains three hunks.

**The slowness, which was the larger problem.** Restoring fusion bought only
~3 ms. The real cost was the replacement kernels: `k_bin_bcast_bf16` recovered
`(i0,i1,i2,i3)` from a flat index with a 64-bit division and three modulos *per
element*, where ggml derives them from a 3D grid. `0daf584` addresses rows via
`blockIdx.y/z` (−19.7 ms); `4d29e95` then widens the contiguous case to 8 BF16
per thread via one `uint4` (−9.4 ms). Both are **bit-identical** to the kernel
they replace — max |diff| 0.0 over 20 steps against `VLA_BF16_FLAT=1`, actions
compared with pinned flow-matching noise (`eval/compare_act_dtype.py`).

`pi0` shares these kernels and moved 104.2 -> 96.7 -> **94.2** on the same two
commits.

**The lesson is about ownership, not kernels.** Whatever ggml owns improves
without us; whatever we take in-house we maintain against upstream's rate. We
spent this work re-deriving indexing and vectorization that `binbcast.cu`
already had.

## Bumping llama.cpp is not free in one direction

Same vla.cpp source, only `VLA_LLAMA_TAG` differing (`b9866` was `corl`'s pin):

| model | b9866 | b10331 | change |
|---|---:|---:|---:|
| `gr00t_n1_6` | 57.0 | 54.1 | −2.9 (−5.1%) |
| `gr00t_n1_5` | 71.7 | 69.3 | −2.4 (−3.3%) |
| `gr00t_n1_7` | 55.5 | 53.8 | −1.7 (−3.1%) |
| `evo1` | 148.6 | 146.9 | −1.7 (−1.1%) |
| `pi0` | 97.1 | 96.7 | −0.4 (−0.4%) |
| `bitvla` | 51.8 | 51.8 | **0.0** |
| `smolvla` | 55.3 | 63.8 | +8.5 (precision, above) |

Five models got faster for nothing. **`bitvla`'s zero is the control**: its hot
path is our own `bitvla_cuda_kernels`, not ggml's, so it is insulated from ggml
kernel work — and it is the one row that did not move. That is evidence the
gains are ggml's kernels rather than anything environmental.

Which upstream commits did it is unknown; b9866 -> b10331 is ~460 tags and only
`smolvla` was profiled across them. The policy conclusion is not "pin forward",
it is **bump regularly and re-measure every model**, because direction is not
uniform and one row moved 8 ms against us while five moved 2-3 ms for us.

`VLA_LLAMA_TAG` (CMake cache) and `SERVER_BIN` (harness) exist so two builds can
be compared without editing tracked files.

## Recommended flags

### Build — one configuration for every model

There is no per-model build flag. vla.cpp exposes exactly two CMake options of
its own (`VLA_LLAMA_TAG`, `VLA_BUILD_TESTS`); everything performance-relevant
comes from ggml and is already the default.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86      # 86 = RTX 3090 (sm_86)
cmake --build build -j"$(nproc)"
```

Defaults worth knowing are already on and should stay on: `GGML_CUDA_GRAPHS=ON`
(capture/replay — every model here replays, see
[the audit](#what-the-previous-revision-got-wrong)), `GGML_CUDA_FA=ON`,
`GGML_CUDA_FORCE_MMQ=OFF`, `GGML_CUDA_FORCE_CUBLAS=OFF`. Set
`CMAKE_CUDA_ARCHITECTURES` to your own SM version; a mismatch costs a JIT compile
on first launch.

Leave `VLA_LLAMA_TAG` at its default `b10331`. It exists to bisect upstream, not
to tune — and pinning back is the wrong move even though it would win 8 ms on
`smolvla`, because it loses 2-3 ms each on five other models
([why](#bumping-llamacpp-is-not-free-in-one-direction)).

**Rebuilding across `5d9ec9e` with a warm `build/` fails**, because
`build/_deps/llama-src` still carries the older multi-file ggml patch and the
current single-hook patch cannot find its anchor. The script is idempotent for an
already-hooked tree but not for a differently-patched one. Fix:

```sh
git -C build/_deps/llama-src checkout HEAD -- ggml/src/ggml-cuda/ggml-cuda.cu
```

### Run — per model

Fastest measured configuration for each. `VLA_POLICY_DIR`/`HF_HOME` are omitted;
set them as your deployment needs.

| Model | run flags | ms | bit-exact vs default? |
|---|---|---:|---|
| `smolvla` | `VLA_SMOLVLA_FA=1 VLA_MM_PREC=default` | 51.1 | no — both change numerics |
| `pi0` | `VLA_PI0_BF16_ACT=1 VLA_PI0_FA=1` | 94.2 | BF16 yes, FA no |
| `evo1` | `VLA_EVO1_BF16_ACT=1 VLA_EVO1_FA=1` | 137.5 | BF16 yes, FA no |
| `gr00t_n1_5` | `VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=new_embodiment` | 68.7 | yes |
| `gr00t_n1_6` | `VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=libero_panda` | 54.4 | yes |
| `gr00t_n1_7` | `VLA_GR00T_BF16_WEIGHTS=1` | 53.8 | yes |
| `bitvla` | *(none)* | 52.1 | yes |

`VLA_GR00T_EMBODIMENT` is a correctness flag, not a performance one — it selects
the normalization statistics, and the wrong value produces wrong actions rather
than slower ones.

### If you want speed without changing numerics

Every flash-attention flag and `smolvla`'s `VLA_MM_PREC=default` alter results;
the rest do not. Dropping the non-exact ones:

| Model | bit-exact flags | ms | cost vs fastest |
|---|---|---:|---|
| `smolvla` | *(none)* | 74.6 | +23.5 |
| `pi0` | `VLA_PI0_BF16_ACT=1` | 96.7 | +2.5 |
| `evo1` | `VLA_EVO1_BF16_ACT=1` | ~147 | +9.5 |
| `gr00t_n1_5/6/7`, `bitvla` | as above — already exact | — | 0 |

At bit-exact settings vla.cpp still wins six of seven; only `smolvla` flips
(74.6 against 59.3). Note the asymmetry in what "exact" is worth: it costs
`smolvla` 31% and `pi0` under 3%, because `pi0`'s 256-token tower barely
benefits from flash attention in the first place.

`evo1`'s bit-exact figure is interpolated from its pre-vectorization measurement
rather than measured on `4d29e95`; the vectorized kernels are bit-exact, so only
the FA component is being removed.

### Debug and bisect flags

Not for production; they exist to make regressions findable.

| Flag | effect |
|---|---|
| `VLA_BF16_FLAT=1` | scalar BF16 elementwise path instead of vectorized; bit-identical, so an A/B isolates indexing bugs from arithmetic ones |
| `VLA_BITVLA_NARROW_GEMM=1` | pre-retiling ternary GEMM |
| `GGML_CUDA_DISABLE_FUSION=1` | stock ggml switch; declines elementwise fusion |
| `VLA_MM_PREC=default` | listed above, but also the control for the precision finding |
| `SERVER_BIN=<path>` | harness override to measure a second build |
| `VLA_LLAMA_TAG=<tag>` | CMake cache, to build against another llama.cpp |

All boolean flags read their **value** since `1efb7a3`; before that, `=0` turned
them on.

## Switches

| Model | switches for the table figure | bit-exact? |
|---|---|---|
| `smolvla` | `VLA_SMOLVLA_FA=1`, `VLA_MM_PREC=default` | FA no; MM_PREC no (but see above) |
| `pi0` | `VLA_PI0_BF16_ACT=1`, `VLA_PI0_FA=1` | BF16 yes, FA no |
| `evo1` | `VLA_EVO1_BF16_ACT=1`, `VLA_EVO1_FA=1` | BF16 yes, FA no |
| `gr00t_n1_5/6/7` | `VLA_GR00T_BF16_WEIGHTS=1` | yes |
| `bitvla` | none | yes |

`VLA_GR00T_BF16_WEIGHTS` is **not** a shipping default: `gr00tn1d*.cpp` selects
F32 when it is unset, and `eval/run_latency_compare.sh` exports `=1` for the
GR00T models. So launching `vla-server` by hand without it measures a different
precision than the table. What the F32 default costs is unmeasured.

**All boolean switches now read their value.** Until `1efb7a3` they tested only
presence, so `VLA_EVO1_FA=0` *enabled* flash attention. That silently invalidates
any A/B done by setting a switch to zero, and it cost one run in this session
before being found. 20 switches across 12 files were converted; `0`, `false`,
`off`, `no` and empty are now false.

### Flash attention is the one unsettled trade

ggml's CUDA flash attention computes K/V at **F16 regardless of input type**
(`fattn.cu:247` reinterprets an F32 K/V tensor as F16), so there is no
full-precision FA path on this backend. Passing F32 K/V produced byte-identical
actions, which is how this was established.

The gain tracks tokens² because score-matrix traffic dominates: ~1024-token
towers gain 15-25%, `pi0`'s 256-token tower ~3%. Against that, two models showed
a ~4-5 pp SR drop in the same direction (`evo1` 97->92, `smolvla` 96->92,
`pi0` 88->89 i.e. none) — but the `evo1` half **did not reproduce** in a later
same-binary 2x2 (96 with, 96 without), and at n=100 with p~0.95 a 4 pp effect is
~1.3 SE, inconclusive either way.

So FA ships **opt-in** not because it is known-bad but because a measured 15-25%
is being traded against an unmeasured accuracy risk, and for a robot policy that
is the wrong direction to leave uncertainty. Settling it needs ~1,450 episodes
per arm (~7 h) on `evo1` and `smolvla`, which is where the gain lives; `pi0`'s
3% does not justify any risk.

One untested opportunity: `gr00t_n1_7`'s vision tower has its own
`VLA_FLASH_ATTN` switch (`src/models/qwen3vl_vit.h`), also default off, so its
53.8 ms was measured without FA. Its 64 merged tokens suggest little to gain.

## Why serial

The previous revision's driver dealt models across both GPUs and ran two lanes
concurrently, so every figure was measured with a second full sweep — another
server, another LIBERO sim — competing for the same 20 cores. That is not a
detail: the PyTorch rows spend 10-70 ms per call in host-side Python
(`gr00t_n1_5`, above) which absorbs CPU contention directly.

It also produced a mismatch inside one row: `gr00t_n1_5`'s published vla.cpp
figure came from the contended two-lane run while its PyTorch figures came from
a later solo re-run — a comparison that ran *against* vla.cpp. `eval/run_latency_serial.sh`
runs one model, one variant, one GPU at a time.

## Open

1. **SR arm at `VLA_MM_PREC=default`** for `smolvla` (~2 h). Gates flipping the
   default, worth 19% on that row.
2. **Powered SR for flash attention** on `evo1` and `smolvla`, ~1,450 eps/arm.
   The weakest evidence in this document.
3. **Re-derive causes #2 and #3** with `--cuda-graph-trace=node`. The old
   sizings came from graph-granularity profiles and cannot be trusted.
4. **`evo1`'s remaining elementwise cost.** A corrected profile still puts our
   BF16 kernels at ~17 ms/call of 137.5 (`k_bin_bcast` 2,276 launches/call for
   the largest). Much of that is launch count, not per-launch efficiency, so the
   next lever is fewer nodes or more fusion rather than faster kernels.
5. **`bitvla` builds five graphs per `predict()`** with no caching, the only arch
   without it. Whether that costs anything is unmeasured — `smolvla` turned out
   to be replaying CUDA graphs fine, so this is less promising than it looks.
