# Contributing

## Build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DVLA_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

First-party code must compile clean under `-Wall -Wextra`. Warnings from
`build/_deps` are upstream and not your problem.

## Proving a change is numerically neutral

Refactors, performance work and dependency bumps must not move the output.
`vla_predict_check` feeds fixed images, tokens, state and noise, then prints the
action chunk:

```bash
VLA_IMG_SIZE=224 ./build/tests/vla_predict_check model.gguf "" 1 > before.txt
# ... make the change, rebuild ...
VLA_IMG_SIZE=224 ./build/tests/vla_predict_check model.gguf "" 1 > after.txt
diff before.txt after.txt
```

Any difference is a bug unless the change is meant to alter numerics, in which
case say so in the commit message and back it with a LIBERO sweep.

`VLA_IMG_SIZE` must match the model or `predict` returns empty: 512 for
SmolVLA, 448 for Evo-1, 256 for GR00T N1.7 and VLA-JEPA, 224 for the rest. Other
knobs: `VLA_BENCH_ITERS` (timing), `VLA_TIMING=phase`, `VLA_EXTRA_TOKEN` /
`VLA_EXTRA_COUNT` (VLA-JEPA needs its `<embodied>` tokens), `VLA_N_THREADS`,
`VLA_DEVICE`.

Checkpoints are at [huggingface.co/vrfai](https://huggingface.co/vrfai), or let
the binaries fetch them:

```bash
./build/vla-cli -hf vrfai/smolvla-libero-gguf --image assets/front.jpg --tokens 1,100,2
```

## Adding an architecture

Six sites, all mechanical. `smolvla` is the reference for a two-file (mmproj +
ckpt) model, `bitvla` for a vision-baked one.

1. `src/arch.h` - add to `enum class Arch`.
2. `src/arch.h` - declare `<name>_create(mmproj_path, ckpt_path, config_path)`.
3. `src/model.cpp` - add `<name>.architecture` to the `try_str` list in
   `detect_arch_gguf`.
4. `src/model.cpp` - map the string to the enum in the same function.
5. `src/model.cpp` - add a `case` to the `model_load` switch.
6. `CMakeLists.txt` - add `src/models/<name>.cpp` to `vla_core`.

Then write `src/models/<name>.cpp`. Before adding a helper, check
`src/models/`: `gguf_reader.h` (tensor and KV reads), `vision_common.h`
(preprocessing, pixel shuffle), `dual_tower.h` (DINOv2 + SigLIP),
`qwen3vl_vit.h` (Qwen3-VL tower), `layers/embed.h` (time embeddings, causal
mask), `scratch_ctx.h` (compute context reuse), `backend.h` (accelerator
selection).

Your loader must fail rather than return a half-built model: check every tensor
lookup, and check `real_*_dim <= max_*_dim` (`config_is_sane` in `src/model.cpp`
does this for all archs).

A converter goes in `scripts/convert_<name>_to_gguf.py`, and its tensor-name
remap should be covered by `tests/py/test_converters.py`.

## Ports of upstream quirks

Some references do surprising things, and we match them because the weights were
trained that way. Three so far: VLA-Adapter's RoPE pairs a half-split frequency
table with an interleaved rotation, OpenVLA-OFT's LM attention is bidirectional,
and BitVLA's is too. Each carries a comment naming the reference file and lines.

If you find something that looks wrong, diff against the reference before
changing it, and leave a comment with the line numbers so the next reader does
not have to.

## Commits and pull requests

One logical change per commit, imperative subject, no trailing period. Say what
you verified: which archs, which backends, whether the numeric output moved.
