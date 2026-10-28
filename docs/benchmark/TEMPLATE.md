<!--
Template for one device across all models. Copy it to docs/benchmark/<device>.md,
fill every <placeholder>, and delete these comments. Keep the section order and
the column order, so that reports compare cell for cell.

Measure every report in a round at one commit. If the code changes, re-measure.
Never name a host, a user or an IP address anywhere in a report.
-->

# <Device name> (<Backend>)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

<!-- Optional: one short paragraph when one build covers several backends
     (e.g. NPU with CPU fallback, or OpenVINO GPU and NPU), saying how each
     section selects its device. -->

## Test setup

| | |
|---|---|
| Device | <Accelerator or CPU model, memory size and type, architecture (e.g. sm_86)> |
| Host | <Host CPU model (cores, threads), RAM>. Drop this row for a CPU-only report |
| Power | <Power limits, governor or power profile, AC or battery. For Jetson: the nvpmodel mode> |
| Threads | <CPU threads used, e.g. "16 (the default: all hardware threads, capped at 16)">. CPU reports only |
| OS | <OS and version, kernel> |
| Driver / toolkit | <Driver version / CUDA, oneAPI, OpenVINO or SDK version, compiler> |
| Commit | `<short sha>` (<branch or PR>) |
| llama.cpp | `<tag>` |
| Date | <YYYY-MM-DD HH:MM> to <HH:MM> (UTC<offset>) |
| Build | `<cmake flags>` (`GGML_NATIVE=<ON/OFF>`) |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

<!-- One sentence saying what the memory columns measure on this device. Pick
     the probe that matches the backend:
     - CUDA (discrete): per-process device memory from
       `nvidia-smi --query-compute-apps`, sampled every 0.5 s, plus peak host RSS.
     - Jetson and Apple silicon: peak RSS (`getrusage`). The GPU buffers are
       mapped into the process, so RSS is its whole footprint.
     - Intel discrete GPU (i915 or xe): `drm-total-local0` from the process's
       DRM `fdinfo`, plus peak host RSS.
     - Intel iGPU and NPU: the DRM `fdinfo` memory regions (`drm-total-gtt`,
       `-system` and `-stolen` for xe, `drm-total-memory` for intel_vpu), plus
       peak RSS.
     - Windows: peak working set and peak private commit (`GetProcessMemoryInfo`).
     - CPU: peak RSS (`getrusage`). -->
**<Memory column>** is <what it measures and how it was sampled>.

## Model configuration

Each model runs at its native view count and input size, with weights as
shipped. Checkpoints are the `vrfai/*` GGUFs; where a repo ships several, we use
the `libero_object` variant. Settings not listed are the defaults.

| Model | Checkpoint | Views | Input | Lang tokens | Action chunk | Action dim | Action head | Steps |
|---|---|--:|--:|--:|--:|--:|---|--:|
| SmolVLA | `smolvla-libero-gguf` | 2 | 512 | 16 | 50 | 7 | flow matching | 10 |
| π0 | `pi0-libero-finetuned-v044-gguf` | 2 | 224 | 16 | 50 | 7 | flow matching | 10 |
| π0.5 | `pi05-libero-gguf` | 2 | 224 | 16 | 50 | 7 | flow matching | 10 |
| GR00T N1.5 | `gr00tn1d5-libero-object-gguf` | 1 | 224 | 16 | 16 | 32 | flow matching (DiT) | 4 |
| GR00T N1.6 | `gr00tn1d6-libero-gguf` | 1 | 224 | 16 | 50 | 128 | flow matching (DiT) | 4 |
| GR00T N1.7 | `gr00tn1d7-libero-gguf` | 1 | 256 | 16 | 40 | 132 | flow matching (DiT) | 4 |
| VLA-JEPA | `vla-jepa-libero` | 1 | 256 | 16 + 32 | 7 | 7 | flow matching (DiT) | 4 |
| Evo-1 | `evo1-libero-gguf` | 1 | 448 | 16 | 50 | 7 | flow matching | 32 |
| BitVLA | `bitvla-libero-gguf` (int2) | 1 | 224 | 16 | 8 | 7 | parallel decoding | 1 |
| VLA-Adapter | `vla-adapter-libero-gguf` | 1 | 224 | 16 | 8 | 7 | bridge-attention policy | 1 |
| OpenVLA-OFT | `openvla-oft-libero-gguf` | 1 | 224 | 16 | 8 | 7 | L1 regression head | 1 |
| Octo-Small | `octo-small-libero-gguf` | 2 | 256 + 128 | 16 | 4 | 7 | DDPM diffusion | 20 |
| TurboVLA | `turbovla-libero-gguf` | 2 | 256 | 16 | 12 | 7 | single-pass decoder | 1 |

- **Lang tokens** is the synthetic prompt length. VLA-JEPA adds its 32
  `<embodied>` tokens with `--extra-token 151697 --extra-count 32`.
- **Action dim** is the checkpoint's padded width. LIBERO uses the first 7 dims.
- **GR00T** needs its embodiment set in the environment:
  `VLA_GR00T_EMBODIMENT=new_embodiment` for N1.5, `libero_panda` for N1.6 and
  `libero_sim` for N1.7. **Octo** needs `VLA_OCTO_UNNORM_DATASET=libero_object`.
- **Octo** takes a 256 px primary view and a 128 px wrist view.

<!-- Check each file's sha256 against the one Hugging Face lists, so every device
     runs identical weights. -->

## Latency and memory

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

<!-- One row per model that ran at the defaults, sorted by min ms. Run each
     process as:
       vla-bench --ckpt <gguf> --images <views> --size <input> --warmup 3 --reps 20 [extra]
     three times. Keep the process with the lowest mean, and report its
     min/mean/p50/p90/vision exactly as printed (one decimal). Report the
     largest memory reading of the three processes. Keep the memory columns
     that match the probe above. -->

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | <Memory column> MiB | <Memory column> MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <Model> | <views> | <input> | <min> | <mean> | <p50> | <p90> | <vision or -> | <MiB> | <MiB> |

### Fastest configuration

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults;
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--weight-dtype bf16` and `--weight-dtype f16`, each alone and with
  `--flash-attn`;
- `--act-dtype bf16 --flash-attn`, for π0 and Evo-1 on CUDA only.

<!-- Delete the --act-dtype line off CUDA. Add a line for any backend-specific
     candidate, e.g. `--flash-attn 0` on Hexagon, where flash attention is on
     by default. -->

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention, bf16 activations and lower-precision weights can move the
action chunk. Success rates measured at the defaults therefore do not carry over
to these rows.

<!-- Sorted by min ms. When the defaults win, write *(defaults)*, repeat the
     default row's numbers and put "-" under "vs defaults". Otherwise
     "vs defaults" is the change in mean, e.g. -24%. -->

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | <Memory column> MiB | <Memory column> MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| <Model> | `<flags>` or *(defaults)* | <min> | <mean> | <p50> | <p90> | <vision or -> | <MiB> | <MiB> | <-N% or -> |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| <Model> | <mean> | <mean> | <mean> | <mean> | <mean> | <mean> | <mean> | <mean> |

</details>

<!-- For a report with several backends (e.g. NPU and CPU), repeat the
     "Latency and memory" and "Fastest configuration" sections once per backend,
     named "## Latency and memory: <backend>" and "### Fastest configuration:
     <backend>". Explain the flag screen once, under
     "### How the fastest configuration is picked". -->

## Not run

<!-- One bullet per model with no row above, giving the reason and, where there
     is one, the error it printed. Examples:
     - N/A: the arch has no path on this backend, e.g. BitVLA's int2-packed GGUF
       off CUDA.
     - Does not fit: loading ran out of memory. Quote the allocation failure.
     - Not attempted: the weights plainly exceed the device's memory. Give both
       sizes.
     - Fails: anything else. Quote the error. -->

- **<Model>**: <reason>

  ```text
  <error lines, with paths trimmed>
  ```

## Notes

<!-- Optional. Anything a reader needs to interpret the numbers, e.g. power
     limits that make min run well below p50, environment variables the backend
     requires, or rows whose output is known to be wrong. -->

## Reproducing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release <backend flags>
cmake --build build -j

# one row, for example SmolVLA
./build/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
