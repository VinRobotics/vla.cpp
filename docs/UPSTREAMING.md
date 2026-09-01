# Upstreaming the ggml-openvino fixes

`scripts/patch_ggml_openvino.py` rewrites nine files inside the fetched llama.cpp
tree at configure time. Every fix in it is a generic ggml-openvino defect, not a
vla.cpp workaround: each narrows an assumption that fits llama.cpp's one
decoder-only graph but is stricter than the ggml contract, or fills a gap in the
op table. None of them needs to live here.

Landing them upstream deletes the patch step, lets vla.cpp build against stock
llama.cpp, and fixes the same bugs for everyone else translating a graph that is
not a decoder-only LLM - whisper.cpp, embedding models, any vision tower.

## The series

A local clone with one branch per PR, each a single commit on `master`:

```
~/llama.cpp-upstream
```

Regenerate it after editing the patch script:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp-upstream
python3 scripts/upstream_split.py
```

| Branch | What it fixes | Value to upstream |
|---|---|---|
| `openvino-naive-cache` | the naive path re-converts and re-compiles the model on every `graph_compute` | **highest**: 22.7 s to 1.4 s on a SmolVLA graph. Also replaces `graph_key` with a shape-aware `naive_key` for that cache, since a node count plus two names collides |
| `openvino-multiple-inp-pos` | all ROPE position inputs are renamed to one `inp_pos` parameter | **high**: any graph with more than one position tensor fails shape inference today |
| `openvino-view-input-rank` | a folded 2-D weight is sliced at full ggml rank | high: `Axis 2 out of the tensor rank range [-2, 1]` on any fused QKV weight |
| `openvino-concat-rank` | CONCAT cannot broadcast rank | high: same rank-2 constants, different op |
| `openvino-reshape-op-case` | the KV-cache-flatten guard also swallows `ggml_conv_2d`'s kernel reshape | medium |
| `openvino-sdpa-kv-f16` | K/V stay F32 while Q is converted | medium: `Mixed input types are not supported` |
| `openvino-naive-graph-size-env` | the 20-node naive threshold is a `constexpr` | medium: exposes `GGML_OPENVINO_NAIVE_GRAPH_SIZE` |
| `openvino-gelu-modes` | ggml's tanh `GELU` is mapped onto ov's erf default, and `GELU_ERF` has no entry at all | **high**: wrong activation on every GELU node. Fixing it moved GR00T N1.5 and VLA-JEPA inside the accuracy bar |
| `openvino-intel-opencl-platform` | the GPU remote context takes the first OpenCL platform | low, but a hard startup abort when it bites |
| `openvino-imrope-sections` | the IMROPE sector cycle ignores `sections` | low, no measured output change |
| `openvino-gemm-double-eltwise` | the GPU plugin folds two chained elementwise adds into the preceding GEMM and silently drops the second operand | **highest**: wrong output on GPU with nothing logged, on any graph that adds a tower's features on top of an FFN residual |
| `openvino-gpu-precision-env` | the GPU plugin's f16 default compounds through a long serial chain in one graph | medium: exposes `GGML_OPENVINO_GPU_PRECISION`, default unchanged |
| `openvino-permute-op-case` | PERMUTE `op_case` 2 is reached by any permute over a view of a non-leaf, not just llama.cpp's rope'd query | **high**: a DiT cross-attention V comes out with its elements rearranged and nothing is reported |

`RELU`, `NEG` and `SQR` were in the patch too. Upstream added all three in
`b10729`, so they are not in the series.

## Before submitting

Read `CONTRIBUTING.md` in llama.cpp. The parts that bite:

- **One PR per feature.** Hence one branch each; do not squash them.
- **AI usage must be disclosed**, and using AI to write the PR text itself is
  prohibited outright. Undisclosed use risks a ban. Write the PR bodies yourself.
- **A modified operator needs `test-backend-ops`.** That covers
  `openvino-gelu-modes`, `openvino-sdpa-kv-f16`, `openvino-concat-rank`,
  `openvino-view-input-rank` and `openvino-reshape-op-case`. The others touch the
  session and cache layers, which `test-backend-ops` does not reach; those need a
  before/after run on a real graph in the PR body.
- **A bug fix needs a reproducible case that fails before and passes after.**
  Each commit message already names the error string or the timing it fixes.
- New contributors should keep one PR open at a time. `openvino-naive-cache` is
  the one to lead with.

## Not yet written

**Key the translation map on the tensor, not its name.** `ggml-decoder.cpp` keys
everything on `node->name`. ggml permits duplicate names - it derives a result's
name from its source, so an unnamed `ggml_reshape_2d` is called `" (reshaped)"` -
and llama.cpp only escapes this because it labels every node it builds. Two nodes
sharing a name silently become one and the graph wires the wrong tensor into the
next op.

`vla::graph_unique_names` in `src/backend.h` works around it from the outside, at
29 call sites. De-duplicating at graph ingest inside `ggml_decoder` would be a few
lines, fix it for every caller, and let vla.cpp delete the helper and all 29
calls. Worth writing; not in the series because it is new code rather than a fix
already proven on hardware.

## Status

The thirteen branches carry code that has run on an Intel Core Ultra X7 358H (Arc
B390 iGPU, AI Boost NPU) through vla.cpp's own OpenVINO builds - see
`docs/backend/ov.md` for what that covered. They have **not** been compiled from
these branches: no OpenVINO runtime is installed on the machine that split them,
so `ggml-openvino` does not build there. Build and run each branch before opening
a PR.
