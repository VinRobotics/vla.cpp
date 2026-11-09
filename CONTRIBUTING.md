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
SmolVLA, 448 for Evo-1, 256 for GR00T N1.7 and VLA-JEPA, 256 with 2 views for
TurboVLA and Cosmos3 Policy, 224 for the rest. Other knobs: `VLA_BENCH_ITERS` (timing),
`VLA_TIMING=phase`, `VLA_EXTRA_TOKEN` / `VLA_EXTRA_COUNT` (VLA-JEPA needs its
`<embodied>` tokens; Octo needs exactly 16 tokens, so
`VLA_EXTRA_TOKEN=0 VLA_EXTRA_COUNT=10`), `VLA_OCTO_UNNORM_DATASET` (Octo, e.g.
`libero_object`), `VLA_N_THREADS`, `VLA_DEVICE`.

Checkpoints are at [huggingface.co/vrfai](https://huggingface.co/vrfai), or let
the binaries fetch them:

```bash
./build/vla-cli -hf vrfai/smolvla-libero-gguf --image assets/front.jpg --tokens 1,100,2
```

## Benchmarking new hardware

New models are not hand-written into this tree (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#adding-an-architecture)); new
devices are. A benchmark report is one file,
`docs/benchmark/<device>.md`, written from
[docs/benchmark/TEMPLATE.md](docs/benchmark/TEMPLATE.md). The template fixes
the section and column order so that reports compare cell for cell; this is
the procedure that fills it.

1. **Build once, at one commit.** Configure a Release build of `main` with the
   backend's flags and run the whole round from it. If the code changes, start
   the round over. The report records the short SHA, the llama.cpp tag
   (`scripts/llama_tag.sh`), the cmake flags including `GGML_NATIVE`, the
   driver or toolkit version and the compiler (`scripts/print_versions.sh`
   prints a block), the OS and kernel, and the power state: limits, governor
   or profile, AC or battery, `nvpmodel` on Jetson.

2. **Fetch the checkpoints.** Every report runs the `vrfai/*` GGUFs in the
   template's "Model configuration" table, at the views, input size, token
   count and steps given there. `-hf vrfai/<repo>` fetches them; check each
   file's sha256 against the one Hugging Face lists, so every device runs
   identical weights.

3. **Measure the defaults.** One process per model, run three times:

   ```bash
   ./build/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
   ```

   Report the process with the lowest mean, copying `min`, `mean`, `p50`,
   `p90` and `vision` as printed (`--markdown` prints the row). Memory is the
   peak over the three processes, taken with the probe the template names
   for the backend: `nvidia-smi --query-compute-apps` sampled every 0.5 s on
   discrete CUDA, DRM `fdinfo` on Intel GPUs and NPUs, peak RSS where the
   device shares the process's memory (Jetson, Apple silicon, CPU). Per-model
   extras: `--extra-token 151697 --extra-count 32` for VLA-JEPA,
   `VLA_GR00T_EMBODIMENT` for GR00T (`new_embodiment`, `libero_panda`,
   `libero_sim` for N1.5, N1.6, N1.7), `VLA_OCTO_UNNORM_DATASET=libero_object`
   for Octo, `--size 640 --height 480` for ACT.

4. **Screen the runtime flags.** For each model, run 2 warmups and 5 reps
   per flag set from the template's list (`--flash-attn`, `--mm-prec default`,
   `--weight-dtype bf16|f16`, their combinations, `--act-dtype bf16` on CUDA
   only, plus any backend-specific candidate). A set replaces the defaults
   only if its mean is more than 3% lower in the screen and again under the
   full protocol of step 3. The screen means go in the collapsed table; the
   winner goes in "Fastest configuration".

5. **Account for every model.** A model with no latency row gets a bullet
   under "Not run": `N/A` when the arch has no path on the backend, "does
   not fit" with the quoted allocation failure, "not attempted" with both
   sizes when the weights plainly exceed the device's memory, or the error
   it printed. Anything a reader needs to interpret the numbers (a power
   limit that puts `min` well below `p50`, a required environment variable,
   a row whose output is known to be wrong) goes under "Notes".

6. **Write it up.** Copy the template, fill every `<placeholder>`, delete the
   HTML comments, and end with a "Reproducing" block that gives the cmake
   line and one `vla-bench` row. Never name a host, a user or an IP address.
   Add the device to the summary table in
   [docs/benchmark/README.md](docs/benchmark/README.md): the `min ms` at the
   defaults per model, `N/A`, `-` or `n/r` with the meanings the README
   gives, and a sentence in its intro if the round ran at a commit the other
   reports did not.

A new backend, as opposed to a new device on a backend the tree already
supports, also needs a numeric check: run `vla_predict_check` on fixed inputs
against the CPU build and report the differences, as
[docs/backend/rocm.md](docs/backend/rocm.md) does.

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
