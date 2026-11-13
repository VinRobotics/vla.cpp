# Qualcomm Dragonwing IQ-10: Hexagon NPU

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

The graph runs on the Hexagon NPU, and ops the NPU rejects run on the CPU
through vla.cpp's fallback wrapper.

## Test setup

| | |
|---|---|
| Device | Qualcomm Dragonwing IQ-10 (IQX EVK): QCS8797, 18 Oryon cores, Hexagon v81 NPU (one of its four CDSP sessions in use), 60.8 GiB RAM visible |
| Threads | 16 (the default: all hardware threads, capped at 16), for the NPU's CPU fallback |
| Power | CPU governor `performance` |
| OS | Ubuntu 24.04.4 LTS, kernel 6.6.110 |
| Toolchain | `ghcr.io/snapdragon-toolchain/arm64-linux:v0.7`: clang 21.1.8, Hexagon SDK 6.6.0.0 (tools 19.0.07) |
| Commit | `f0b12ae` |
| llama.cpp | `b11223` |
| Date | 2026-11-09 17:58 to 18:28 (UTC+07) |
| Build | `scripts/build_linux_hexagon.sh` (servers included); the HTP skel processed for the board and installed in `/vendor/dsp/cdsp0..3` |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

**Peak RSS** is the process's peak resident set (`getrusage`, via
`/usr/bin/time`). It leaves out the weights, which live in FastRPC buffers
mapped for the NPU. **System RAM** is the largest drop in the system's
`MemAvailable` while the process ran, sampled every 0.5 s; it counts those
buffers. Nothing else ran on the board.

## Model configuration

Each model runs at its native view count and input size, with weights as
shipped. Checkpoints are the `vrfai/*` GGUFs, and ACT's is
`ravediamond/act-aloha-sim-transfer-cube-gguf`; where a repo ships several, we
use the `libero_object` variant. Settings not listed are the defaults.

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
| ACT | `act-aloha-sim-transfer-cube-gguf` | 1 | 640 x 480 | - | 100 | 14 | transformer decoder | 1 |

- **Lang tokens** is the synthetic prompt length. VLA-JEPA adds its 32
  `<embodied>` tokens with `--extra-token 151697 --extra-count 32`.
- **Action dim** is the checkpoint's padded width. LIBERO uses the first 7 dims.
- **GR00T** needs its embodiment set in the environment:
  `VLA_GR00T_EMBODIMENT=new_embodiment` for N1.5, `libero_panda` for N1.6 and
  `libero_sim` for N1.7. **Octo** needs `VLA_OCTO_UNNORM_DATASET=libero_object`.
- **Octo** takes a 256 px primary view and a 128 px wrist view.
- **ACT** is an ALOHA checkpoint (sim transfer cube): no language, a 14-dim
  bimanual state and action, run with `--images 1 --size 640 --height 480`.

## Latency and memory

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 81.2 | 85.6 | 83.6 | 91.1 | 21.9 | 252 | 743 |
| ACT | 1 | 640 x 480 | 147.7 | 153.1 | 152.0 | 157.7 | - | 194 | 470 |
| TurboVLA | 2 | 256 | 184.0 | 191.9 | 191.4 | 197.1 | - | 249 | 1047 |
| VLA-JEPA | 1 | 256 | 490.2 | 498.4 | 497.9 | 505.6 | 199.0 | 184 | 4012 |
| GR00T N1.5 | 1 | 224 | 524.7 | 525.9 | 525.7 | 527.8 | 138.6 | 162 | 3644 |
| SmolVLA | 2 | 512 | 583.5 | 584.4 | 584.2 | 585.2 | 229.9 | 145 | 1194 |
| VLA-Adapter | 1 | 224 | 662.0 | 668.3 | 666.0 | 677.9 | 257.9 | 872 | 3600 |
| GR00T N1.6 | 1 | 224 | 731.7 | 733.7 | 733.1 | 735.1 | 194.5 | 274 | 5691 |
| GR00T N1.7 | 1 | 256 | 739.2 | 742.7 | 741.9 | 746.2 | 197.7 | 191 | 5049 |
| π0.5 | 2 | 224 | 1103.9 | 1106.8 | 1106.1 | 1109.9 | 277.2 | 285 | 5790 |
| OpenVLA-OFT | 1 | 224 | 1121.4 | 1123.0 | 1121.9 | 1126.0 | 262.4 | 844 | 15323 |
| π0 | 2 | 224 | 1127.8 | 1131.2 | 1130.8 | 1134.6 | 276.9 | 285 | 5531 |
| Evo-1 | 1 | 448 | 1650.2 | 1658.4 | 1656.5 | 1666.6 | 821.1 | 319 | 1553 |

### Fastest configuration

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults (F16 weights, the Hexagon build's default);
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--flash-attn 0`, because flash attention is on by default in Hexagon builds.

The weight dtype stays F16 in every set. `--weight-dtype f16` is the default,
and `--weight-dtype bf16` is left out: the NPU has no BF16 kernels, so every
matmul would run on the CPU (SmolVLA: 18.6 s against 0.58 s, mean, 5 reps).

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention and the matmul precision can move the action chunk. Success
rates measured at the defaults therefore do not carry over to these rows. For
VLA-JEPA and GR00T N1.7, `--flash-attn` puts their LM back on flash attention,
which the defaults leave off because it is outside the fidelity bar against a
CPU-only build: VLA-JEPA 6.2e-3 on a served LIBERO observation (2.7e-3 on fixed
inputs), N1.7 6.2e-3 on fixed inputs (2.7e-3 served).

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 81.2 | 85.6 | 83.6 | 91.1 | 21.9 | 252 | 743 | - |
| ACT | *(defaults)* | 147.7 | 153.1 | 152.0 | 157.7 | - | 194 | 470 | - |
| TurboVLA | *(defaults)* | 184.0 | 191.9 | 191.4 | 197.1 | - | 249 | 1047 | - |
| VLA-JEPA | `--flash-attn --mm-prec default` | 428.6 | 435.0 | 434.1 | 442.6 | 197.6 | 184 | 3971 | -12.7% |
| GR00T N1.5 | *(defaults)* | 524.7 | 525.9 | 525.7 | 527.8 | 138.6 | 162 | 3644 | - |
| SmolVLA | *(defaults)* | 583.5 | 584.4 | 584.2 | 585.2 | 229.9 | 145 | 1194 | - |
| VLA-Adapter | *(defaults)* | 662.0 | 668.3 | 666.0 | 677.9 | 257.9 | 872 | 3600 | - |
| GR00T N1.7 | `--flash-attn --mm-prec default` | 714.1 | 719.2 | 718.5 | 725.5 | 196.6 | 191 | 5081 | -3.2% |
| GR00T N1.6 | *(defaults)* | 731.7 | 733.7 | 733.1 | 735.1 | 194.5 | 274 | 5691 | - |
| π0.5 | *(defaults)* | 1103.9 | 1106.8 | 1106.1 | 1109.9 | 277.2 | 285 | 5790 | - |
| OpenVLA-OFT | *(defaults)* | 1121.4 | 1123.0 | 1121.9 | 1126.0 | 262.4 | 844 | 15323 | - |
| π0 | *(defaults)* | 1127.8 | 1131.2 | 1130.8 | 1134.6 | 276.9 | 285 | 5531 | - |
| Evo-1 | *(defaults)* | 1650.2 | 1658.4 | 1656.5 | 1666.6 | 821.1 | 319 | 1553 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|
| Octo-Small | 84.5 | 84.9 | 86.0 | 87.9 | 83.1 |
| ACT | 155.5 | 155.2 | 158.4 | 147.8 | 176.9 |
| TurboVLA | 191.1 | 193.3 | 188.9 | 191.5 | 252.8 |
| VLA-JEPA | 501.1 | 441.9 | 504.7 | 439.9 | 598.9 |
| GR00T N1.5 | 525.8 | 525.2 | 526.4 | 525.6 | 809.1 |
| SmolVLA | 584.3 | 584.8 | 585.6 | 584.5 | 2145.5 |
| VLA-Adapter | 677.1 | 667.2 | 673.1 | 674.4 | 950.4 |
| GR00T N1.6 | 739.0 | 736.8 | 737.4 | 741.2 | 897.7 |
| GR00T N1.7 | 749.3 | 727.4 | 744.9 | 723.3 | 851.0 |
| π0.5 | 1125.3 | 1114.0 | 1133.7 | 1115.2 | 2053.2 |
| OpenVLA-OFT | 1127.0 | 1124.8 | 1128.0 | 1121.5 | 1397.1 |
| π0 | 1113.4 | 1113.5 | 1113.9 | 1113.7 | 2070.2 |
| Evo-1 | 1657.1 | 1657.0 | 1656.2 | 1662.9 | 3106.6 |

</details>

## Q8_0 weights

This section runs GGUFs that `scripts/quantize_gguf.py --type Q8_0` made from
the checkpoints above (ggml's block-32 Q8_0 weights). The setup is the one above,
except:

| | |
|---|---|
| Commit | `71cdb09` |
| Date | 2026-11-10 to 2026-11-13 (UTC+07) |
| Runtime flags | defaults |

A Q8_0 level puts more of the model in Q8_0 than the level above it. Embeddings,
the output head, norms, patch embeddings and position tables stay float at every
level. Weights not in Q8_0 load as they do in the F16 configuration
(`--weight-dtype f16`, the default).

| Q8_0 level | `quantize_gguf.py` flags | Weights in Q8_0 |
|---|---|---|
| F16 | - | none: the shipped file |
| base | `--type Q8_0` | the 2-D weights outside the vision tower and the action head |
| base + vision, unpadded | `--type Q8_0 --vision` | base, plus the vision tower |
| base + vision | `--type Q8_0 --vision --pad-m32` | base, plus the vision tower with FFN widths that are not a multiple of 32 zero-padded (SigLIP's 4304 to 4320 in π0, π0.5, GR00T N1.5 and N1.6, VLA-Adapter and OpenVLA-OFT) |
| base + vision + DiT | `--type Q8_0 --vision --pad-m32 --dit` | base + vision, plus the DiT's weight matrices except adaLN and the time embedding |
| *level*, *n* kept F16 (*kind* list) | the level's flags plus `--keep-float REGEX` | the level, except *n* weights that `REGEX` keeps F16 |

Each configuration runs six checks: max|Δ| of the action chunk against the
CPU-only build at F32, on *fixed*, the fixed inputs of
[Fidelity](../backend/hexagon.md#fidelity), and on *obs0* to *obs4*, five LIBERO
observations served as in [Serving](../backend/hexagon.md#serving). obs*k*
shifts the views' color pattern and draws their noise from seed *k*, moves the
end-effector by 0.02·*k* on each axis and takes task prompt *k* mod 4 of four
LIBERO prompts; obs0 is the observation of
[Serving](../backend/hexagon.md#serving). A check passes when max|Δ| is at most
its threshold: 2.9e-3, or 1.1 times the F16 max|Δ| on that check where that is
above 2.9e-3. For each check, max|Δ| divided by the threshold gives a ratio; a
configuration's score is its largest ratio, so it passes all checks at a score
of 1 or less.

The `--keep-float` lists come from a greedy search on one Q8_0 level. Groups of
blocks (a quarter of a stack, attention or FFN), most sensitive first, and then
single blocks are kept F16 until the score is at most 1; the F16 blocks are then
returned to Q8_0 while the score stays at most 1. The list's
*kind* names the checks it was searched on: *fixed-input* (fixed), *obs0*,
*fixed-input + obs0* (both lists' weights kept F16), and *six-check*. A
six-check list is searched on fixed, obs0, obs1 and obs2, then checked on obs3
and obs4; an observation that fails joins the search, once. A six-check search
runs on the level with the most in Q8_0 first, gets 1.75 h per level, and moves
to the next level only when it finds no list. A level is not searched when all
of it in Q8_0 scores above 10. Mean latency and System RAM follow the TEMPLATE
protocol.

### Configuration per model

Per model, the fastest Q8_0 configuration that passes all checks and has a
recorded `--keep-float` list, against F16.

- *Weights in Q8_0* counts the weights in Q8_0 out of those the level can put in
  Q8_0; *n kept F16* is the number `--keep-float` keeps out.
- A model fails when no configuration passes; its row gives the best score its
  searches reached, or, where no search ran, the score with all of base in Q8_0.

| Model | Q8_0 result | Q8_0 configuration | Weights in Q8_0 | F16 ms | Q8_0 ms | vs F16 | F16 System RAM MiB | Q8_0 System RAM MiB | vs F16 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| π0 | passes all checks | base + vision, 8 kept F16 | 281 of 289 | 1123.3 | 937.7 | -16.5% | 5662 | 3326 | -41.3% |
| π0.5 | passes all checks | base + vision, 181 kept F16 | 108 of 289 | 1107.8 | 1061.2 | -4.2% | 5980 | 5095 | -14.8% |
| Octo-Small | passes all checks | base + vision, 4 kept F16 | 119 of 123 | 84.1 | 81.5 | -3.1% | 725 | 407 | -43.9% |
| ACT | passes all checks | base | 35 of 35 | 152.5 | 150.9 | -1.0% | 469 | 405 | -13.6% |
| TurboVLA | passes all checks | base, 12 kept F16 | 121 of 133 | 190.2 | 189.7 | -0.3% | 1079 | 881 | -18.4% |
| Evo-1 | passes all checks | base, 66 kept F16 | 34 of 100 | 1659.0 | 1657.4 | -0.1% | 1582 | 1494 | -5.6% |
| OpenVLA-OFT | fails: best score 1.03 | - | - | 1125.2 | - | - | 15266 | - | - |
| VLA-Adapter | fails: best score 1.03 | - | - | 669.2 | - | - | 3569 | - | - |
| GR00T N1.5 | fails: best score 1.23 | - | - | 529.3 | - | - | 3659 | - | - |
| GR00T N1.6 | fails: best score 1.33 | - | - | 682.6 | - | - | 4929 | - | - |
| GR00T N1.7 | fails: best score 1.40 | - | - | 744.9 | - | - | 5108 | - | - |
| SmolVLA | fails: not searched, score 44 with all of base in Q8_0 | - | - | 584.0 | - | - | 1226 | - | - |
| VLA-JEPA | fails: not searched, score 36 with all of base in Q8_0 | - | - | 496.8 | - | - | 4018 | - | - |

Flags for `python scripts/quantize_gguf.py --in <checkpoint> --out <file>`:

```bash
# π0
--type Q8_0 --vision --pad-m32 --keep-float '^vlm\.blk\.[04]\.attn_[qkvo]\.'
# π0.5
--type Q8_0 --vision --pad-m32 --keep-float '^vit\.blk\.(0|2|3|4|5|6|14|17|18|19|20)\.|^vit\.blk\.(7|8|9|12|13|21|23|24|25)\.attn_|^vlm\.blk\.(9|10|11|12|13|14)\.|^vlm\.blk\.(0|1|2|3|4|15|16)\.ffn_|^vlm\.blk\.[5-8]\.attn_'
# Octo-Small
--type Q8_0 --vision --pad-m32 --keep-float '^octo\.blk\.(1\.attn_(qkv|o)|6\.ffn_(up|down))\.'
# ACT
--type Q8_0
# TurboVLA
--type Q8_0 --keep-float '^text\.encoder\.layer\.[02]\.'
# Evo-1
--type Q8_0 --keep-float '^vlm\.blk\.([2-9]|10)\.|^vlm\.blk\.13\.ffn_'
```

### Per model

Each model has two tables. The first gives every Q8_0 configuration measured,
with mean latency, System RAM, max|Δ| on each of the six checks, and how many
checks pass. The second gives every six-check search on the model, in the order
run: the score with all of the level in Q8_0, the search, the check on obs3 and
obs4, the search again when that check fails, the result and the hours it took.
A search that finds no list gives its best score.

#### Octo-Small

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 84.1 | - | 725 | - | 5.0e-4 | 1.4e-4 | 4.7e-4 | 3.3e-4 | 1.4e-4 | 6.3e-4 | 6 / 6 |
| base | 81.6 | -3.0% | 405 | -44.1% | 2.5e-3 | 2.0e-3 | 4.4e-3 | 2.4e-3 | 1.8e-3 | 4.9e-3 | 4 / 6 |
| base + vision | 83.5 | -0.7% | 402 | -44.6% | 1.9e-3 | 1.9e-3 | 4.2e-3 | 3.2e-3 | 1.7e-3 | 4.9e-3 | 3 / 6 |
| base + vision, 14 kept F16 (six-check list) | 83.8 | -0.4% | 425 | -41.4% | 2.7e-3 | 1.9e-3 | 2.4e-3 | 2.0e-3 | 7.5e-4 | 1.5e-3 | 6 / 6 |
| base + vision, 4 kept F16 (six-check list) | 81.5 | -3.1% | 407 | -43.9% | 1.0e-3 | 1.7e-3 | 2.7e-3 | 1.6e-3 | 1.2e-3 | 2.5e-3 | 6 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 1.68 | list found (score ≤ 1) | obs4 fails: 3.6e-3 > threshold | list found (score ≤ 1) | passes: 14 weights kept F16 | 0.5 |
| base + vision | 1.68 | list found (score ≤ 1) | obs4 fails: 4.4e-3 > threshold | list found (score ≤ 1) | passes: 4 weights kept F16 | 0.5 |

The 14-weight list comes from the first search, which kept a group F16 even when
that raised the score; the 4-weight list comes from the second, which keeps a
group F16 only when that lowers the score.

#### ACT

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 152.5 | - | 469 | - | 6.1e-4 | 1.0e-3 | 9.0e-4 | 5.7e-4 | 7.2e-4 | 7.6e-4 | 6 / 6 |
| base | 150.9 | -1.0% | 405 | -13.6% | 2.1e-3 | 2.4e-3 | 2.7e-3 | 2.0e-3 | 2.3e-3 | 2.0e-3 | 6 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base | 0.93 | not searched: all in Q8_0 passes | - | - | passes: no weight kept F16 | - |

ACT takes no prompt: its obs*k* shifts the view and the 14-dim state the same
way. ACT has no vision tower to put in Q8_0 (its ResNet stays float), so base is
its only level.

#### TurboVLA

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 190.2 | - | 1079 | - | 3.7e-4 | 8.5e-4 | crash | crash | crash | 6.3e-4 | 3 / 3 |
| base | 186.5 | -1.9% | 814 | -24.6% | 1.3e-2 | 4.2e-3 | crash | crash | crash | 1.7e-3 | 1 / 3 |
| base + vision | 185.7 | -2.4% | 485 | -55.1% | 1.4e-2 | 8.3e-3 | crash | crash | crash | 5.7e-3 | 0 / 3 |
| base + vision, 32 kept F16 (fixed-input list) | 185.8 | -2.3% | 611 | -43.4% | 2.6e-3 | 6.9e-3 | crash | crash | crash | 4.2e-3 | 1 / 3 |
| base, 12 kept F16 (six-check list) | 189.7 | -0.3% | 881 | -18.4% | 1.5e-3 | 2.7e-3 | crash | crash | crash | 1.6e-3 | 3 / 3 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 4.86 | no list within 1.75 h (best 1.02) | - | - | no list | 1.8 |
| base | 4.48 | list found on fixed, obs0, obs4 (score ≤ 1) | not run: crash | - | passes: 12 weights kept F16 | 0.1 |

obs1 to obs3 crash the Hexagon server at F16 too (`ggml-hex: HTP0 synchronize
failed : dsp-error NO-SUPPORT` in `TurboVlaModelArch::predict`). TurboVLA
therefore has three checks (fixed, obs0, obs4), and its six-check list searches
on those three.

#### VLA-JEPA

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 496.8 | - | 4018 | - | 2.1e-3 | 2.2e-3 | 1.1e-3 | 1.4e-3 | 2.8e-3 | 3.4e-4 | 6 / 6 |
| base | 432.1 | -13.0% | 2718 | -32.4% | 1.4e-2 | 1.0e-1 | 1.8e-2 | 6.4e-2 | 6.9e-2 | 9.9e-3 | 0 / 6 |
| base + vision | 411.6 | -17.1% | 2350 | -41.5% | 1.5e-2 | 1.6e-1 | 1.5e-2 | 5.3e-2 | 5.4e-2 | 1.0e-2 | 0 / 6 |
| base + vision + DiT | 393.5 | -20.8% | 2288 | -43.1% | 1.3e-2 | 1.3e-1 | 1.6e-2 | 5.5e-2 | 5.5e-2 | 1.0e-2 | 0 / 6 |
| base + vision + DiT, 233 kept F16 (fixed-input list) | 461.4 | -7.1% | 3614 | -10.1% | 2.9e-3 | 8.3e-3 | 2.5e-3 | 5.8e-3 | 3.3e-3 | 2.5e-3 | 3 / 6 |
| base + vision + DiT, 143 kept F16 (obs0 list) | 436.6 | -12.1% | 3117 | -22.4% | 8.4e-3 | 2.1e-3 | 3.8e-3 | 6.1e-3 | 2.2e-2 | 2.9e-3 | 2 / 6 |
| base + vision + DiT, 244 kept F16 (fixed-input + obs0 list) | 461.6 | -7.1% | 3679 | -8.4% | 3.7e-3 | 5.1e-3 | 2.8e-3 | 3.2e-3 | 3.1e-3 | 2.4e-3 | 2 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision + DiT | 43.20 | not searched: score above 10 | - | - | no list | - |
| base + vision | 54.82 | not searched: score above 10 | - | - | no list | - |
| base | 35.67 | not searched: score above 10 | - | - | no list | - |

#### GR00T N1.5

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 529.3 | - | 3659 | - | 1.9e-3 | 1.2e-2 | 4.1e-3 | 3.8e-3 | 3.1e-2 | 1.2e-2 | 6 / 6 |
| base | 500.2 | -5.5% | 3148 | -14.0% | 5.1e-3 | 8.0e-2 | 2.1e-2 | 4.3e-2 | 9.8e-2 | 6.5e-2 | 0 / 6 |
| base + vision | 481.1 | -9.1% | 2754 | -24.7% | 7.2e-3 | 4.6e-2 | 4.9e-2 | 1.2e-1 | 1.4e-1 | 5.2e-2 | 0 / 6 |
| base + vision + DiT | 399.2 | -24.6% | 2339 | -36.1% | 8.4e-3 | 3.6e-2 | 9.7e-2 | 1.3e-1 | 1.6e-1 | 6.5e-2 | 0 / 6 |
| base + vision + DiT, 191 kept F16 (fixed-input list) | 449.0 | -15.2% | 3061 | -16.3% | 2.7e-3 | 2.8e-2 | 3.5e-2 | 5.1e-2 | 5.4e-2 | 6.2e-2 | 1 / 6 |
| base + vision + DiT, 223 kept F16 (obs0 list) | 472.9 | -10.7% | 3172 | -13.3% | 6.8e-3 | 9.1e-3 | 2.7e-2 | 1.2e-2 | 3.0e-2 | 1.6e-2 | 2 / 6 |
| base + vision + DiT, 272 kept F16 (fixed-input + obs0 list) | 491.2 | -7.2% | 3413 | -6.7% | 3.3e-3 | 1.1e-2 | 1.1e-2 | 1.2e-2 | 3.5e-2 | 1.7e-2 | 1 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 29.16 | no list: no group or block brings the score below 1.77 | - | - | no list | 3.0 |
| base | 10.44 | list found (score ≤ 1) | both fail: 4.0e-2, 3.2e-2 > threshold | no list (best 1.23) | no list | 1.8 |
| base + vision + DiT | 30.86 | no list within 4 h (best 1.84) | - | - | no list | 4.1 |

Searched with 4 h per level, before levels scoring above 10 were skipped.

#### SmolVLA

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 584.0 | - | 1226 | - | 7.4e-4 | 1.9e-3 | 2.9e-2 | 1.9e-1 | 5.6e-2 | 2.5e-1 | 6 / 6 |
| base | 569.0 | -2.6% | 919 | -25.0% | 1.2e-2 | 1.3e-2 | 1.4e+0 | 1.3e-1 | 2.2e-2 | 5.3e-1 | 2 / 6 |
| base + vision | 561.0 | -3.9% | 848 | -30.8% | 3.4e-2 | 1.6e-2 | 1.7e+0 | 1.1e-1 | 8.5e-2 | 4.4e-1 | 1 / 6 |
| base + vision, 58 kept F16 (fixed-input list) | 564.2 | -3.4% | 896 | -26.9% | 2.0e-3 | 1.6e-2 | 1.6e+0 | 2.4e-1 | 2.6e-1 | 9.6e-1 | 1 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 53.74 | not searched: score above 10 | - | - | no list | - |
| base | 44.12 | not searched: score above 10 | - | - | no list | - |

On obs1 to obs4 the gripper channel switches sign within the chunk, and a shift
in when it switches sets max|Δ|, at F16 too.

#### VLA-Adapter

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 669.2 | - | 3569 | - | 3.9e-4 | 3.9e-4 | 1.2e-3 | 6.6e-4 | 4.7e-4 | 3.3e-4 | 6 / 6 |
| base | 653.5 | -2.3% | 3200 | -10.3% | 4.3e-3 | 6.3e-3 | 1.4e-2 | 9.6e-3 | 8.0e-3 | 8.2e-3 | 0 / 6 |
| base + vision | 623.4 | -6.8% | 2555 | -28.4% | 3.1e-3 | 6.0e-3 | 1.5e-2 | 8.9e-3 | 7.7e-3 | 7.1e-3 | 0 / 6 |
| base + vision, 8 kept F16 (fixed-input list) | 622.7 | -6.9% | 2516 | -29.5% | 2.8e-3 | 6.3e-3 | 1.4e-2 | 6.9e-3 | 8.5e-3 | 6.4e-3 | 1 / 6 |
| base + vision, 40 kept F16 (obs0 list) | 627.7 | -6.2% | 2494 | -30.1% | 2.9e-3 | 2.9e-3 | 1.5e-2 | 6.6e-3 | 8.0e-3 | 5.3e-3 | 2 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 5.24 | no list within 1.75 h (best 1.03) | - | - | no list | 1.8 |
| base | 4.90 | list found (score ≤ 1) | both fail: 4.3e-3, 5.0e-3 > threshold | score 0.82 only with every LM block kept F16; time ran out returning blocks to Q8_0 | no list | 1.8 |

#### GR00T N1.6

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 682.6 | - | 4929 | - | 1.3e-3 | 1.0e-2 | 1.2e-3 | 1.2e-3 | 1.1e-3 | 1.8e-3 | 6 / 6 |
| base | 644.5 | -5.6% | 3959 | -19.7% | 7.2e-3 | 3.4e-2 | 8.1e-3 | 9.2e-3 | 1.5e-2 | 8.3e-3 | 0 / 6 |
| base + vision, unpadded | 817.3 | +19.7% | 3738 | -24.2% | 7.1e-3 | 2.5e-2 | - | - | - | - | 0 / 2 |
| base + vision | 622.5 | -8.8% | 3588 | -27.2% | 8.0e-3 | 2.9e-2 | 2.1e-2 | 1.0e-2 | 2.5e-2 | 2.2e-2 | 0 / 6 |
| base + vision + DiT | 462.2 | -32.3% | 2777 | -43.7% | 8.1e-3 | 2.9e-2 | 2.0e-2 | 1.0e-2 | 2.5e-2 | 2.2e-2 | 0 / 6 |
| base + vision + DiT, 57 kept F16 (fixed-input list) | 482.3 | -29.3% | 3076 | -37.6% | 2.8e-3 | 3.6e-2 | 1.7e-2 | 1.2e-2 | 2.2e-2 | 2.2e-2 | 1 / 6 |
| base + vision + DiT, 3 kept F16 (obs0 list) | 464.0 | -32.0% | 2800 | -43.2% | 7.6e-3 | 9.2e-3 | 1.2e-2 | 1.1e-2 | 1.6e-2 | 2.8e-2 | 1 / 6 |
| base + vision + DiT, 60 kept F16 (fixed-input + obs0 list) | 480.7 | -29.6% | 3114 | -36.8% | 3.0e-3 | 8.6e-3 | 7.1e-3 | 7.9e-3 | 1.3e-2 | 3.3e-2 | 1 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision + DiT | 8.63 | no list within 1.75 h (best 1.68) | - | - | no list | 1.8 |
| base + vision | 8.57 | no list within 1.75 h (best 1.33) | - | - | no list | 1.8 |
| base | 5.12 | list found (score ≤ 1) | both fail: 4.5e-3, 4.7e-3 > threshold | no list within 1.75 h (best 1.55) | no list | 1.8 |

*base + vision, unpadded* ran fixed and obs0 only.

#### GR00T N1.7

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 744.9 | - | 5108 | - | 2.0e-3 | 2.0e-3 | 2.8e-3 | 1.3e-3 | 1.5e-3 | 6.2e-4 | 6 / 6 |
| base | 703.5 | -5.6% | 4356 | -14.7% | 1.1e-2 | 2.0e-2 | 1.0e-2 | 2.2e-2 | 1.7e-2 | 2.2e-2 | 0 / 6 |
| base + vision | 688.4 | -7.6% | 3996 | -21.8% | 2.0e-2 | 2.2e-2 | 1.1e-2 | 1.7e-2 | 1.9e-2 | 1.8e-2 | 0 / 6 |
| base + vision + DiT | 536.1 | -28.0% | 3151 | -38.3% | 2.8e-2 | 2.1e-2 | 1.0e-2 | 1.7e-2 | 1.7e-2 | 1.7e-2 | 0 / 6 |
| base + vision + DiT, 185 kept F16 (fixed-input list) | 606.0 | -18.6% | 4051 | -20.7% | 2.6e-3 | 5.3e-3 | 5.6e-3 | 5.0e-3 | 2.0e-3 | 2.2e-3 | 3 / 6 |
| base + vision + DiT, 157 kept F16 (obs0 list) | 579.8 | -22.2% | 3870 | -24.2% | 4.2e-2 | 2.8e-3 | 3.6e-3 | 1.8e-3 | 1.6e-3 | 3.3e-3 | 3 / 6 |
| base + vision + DiT, 233 kept F16 (fixed-input + obs0 list) | 614.4 | -17.5% | 4249 | -16.8% | 3.9e-3 | 6.5e-3 | 3.0e-3 | 3.0e-3 | 1.4e-3 | 1.2e-3 | 3 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision + DiT | 9.76 | no list within 1.75 h (best 1.40) | - | - | no list | 1.8 |
| base + vision | 7.48 | no list within 1.75 h (best 2.07) | - | - | no list | 1.8 |
| base | 7.65 | no list: no block brings the score below 1.45 | - | - | no list | 0.6 |

#### π0.5

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 1107.8 | - | 5980 | - | 4.8e-4 | 3.7e-4 | 6.0e-4 | 5.5e-3 | 8.9e-4 | 3.9e-4 | 6 / 6 |
| base | 971.0 | -12.3% | 3997 | -33.2% | 4.9e-4 | 1.9e-3 | 5.0e-3 | 1.7e-2 | 3.9e-3 | 3.5e-3 | 2 / 6 |
| base + vision, unpadded | 1320.2 | +19.2% | 3760 | -37.1% | 8.5e-4 | 2.7e-3 | - | - | - | - | 2 / 2 |
| base + vision | 941.3 | -15.0% | 3676 | -38.5% | 9.2e-4 | 2.8e-3 | 8.0e-3 | 1.7e-2 | 1.0e-2 | 4.9e-3 | 2 / 6 |
| base + vision, 109 kept F16 (six-check list) | 1041.3 | -6.0% | 4785 | -20.0% | 7.9e-4 | 1.4e-3 | 2.7e-3 | 4.7e-3 | 2.7e-3 | 1.2e-3 | 6 / 6 |
| base + vision, 181 kept F16 (six-check list) | 1061.2 | -4.2% | 5095 | -14.8% | 6.0e-4 | 8.5e-4 | 2.8e-3 | 6.0e-3 | 2.5e-3 | 1.1e-3 | 6 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 3.53 | list found (score ≤ 1) | both fail: 8.9e-3, 3.5e-3 > threshold | list found (score ≤ 1) | passes: 109 weights kept F16 | 4.8 |
| base + vision | 3.53 | list found (score ≤ 1) | obs3 fails: 6.4e-3 > threshold | list found (score ≤ 1) | passes: 181 weights kept F16 | 5.6 |
| base | 2.75 | not searched: base + vision has a list | - | - | - | - |

*base + vision, unpadded* ran fixed and obs0 only. Both π0.5 searches ran
without the time limit. The 109-weight list comes from the first search, which
kept a group F16 even when that raised the score; that list was not recorded, so
the configuration per model uses the 181-weight list from the second search.

#### π0

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 1123.3 | - | 5662 | - | 7.7e-4 | 5.8e-4 | 4.7e-4 | 5.1e-4 | 5.8e-4 | 6.1e-4 | 6 / 6 |
| base | 978.8 | -12.9% | 3723 | -34.2% | 1.8e-2 | 1.7e-3 | 6.2e-4 | 6.6e-4 | 8.8e-4 | 1.4e-3 | 5 / 6 |
| base + vision | 937.6 | -16.5% | 3337 | -41.1% | 1.6e-2 | 1.7e-3 | 5.6e-4 | 7.2e-4 | 5.4e-4 | 1.5e-3 | 5 / 6 |
| base, 8 kept F16 (fixed-input list) | 978.8 | -12.9% | 3711 | -34.5% | 1.4e-3 | 5.2e-4 | 6.1e-4 | 6.6e-4 | 5.3e-4 | 7.4e-4 | 6 / 6 |
| base + vision, 8 kept F16 (six-check list) | 937.7 | -16.5% | 3326 | -41.3% | 1.0e-3 | 5.6e-4 | 6.1e-4 | 6.7e-4 | 4.6e-4 | 1.0e-3 | 6 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 5.55 | list found (score ≤ 1) | both pass: 4.6e-4, 1.0e-3 | not needed | passes: 8 weights kept F16 | 0.8 |
| base | 6.21 | not searched: base + vision has a list | - | - | - | - |

#### OpenVLA-OFT

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 1125.2 | - | 15266 | - | 3.1e-4 | 7.5e-4 | 5.6e-4 | 4.3e-4 | 6.8e-4 | 5.6e-4 | 6 / 6 |
| base | 845.7 | -24.8% | 9403 | -38.4% | 5.9e-4 | 3.5e-3 | 5.1e-3 | 4.4e-3 | 4.9e-3 | 3.8e-3 | 1 / 6 |
| base + vision | 812.5 | -27.8% | 8821 | -42.2% | 7.9e-4 | 3.2e-3 | 5.8e-3 | 4.5e-3 | 3.9e-3 | 6.0e-3 | 1 / 6 |
| base + vision, 12 kept F16 (obs0 list) | 821.4 | -27.0% | 8959 | -41.3% | 7.9e-4 | 2.5e-3 | 4.3e-3 | 3.4e-3 | 2.8e-3 | 5.4e-3 | 3 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 2.08 | no list within 1.75 h (best 1.35) | - | - | no list | 1.8 |
| base | 1.76 | list found (score ≤ 1) | obs3 fails: 3.4e-3 > threshold | no list within 1.75 h (best 1.03) | no list | 1.8 |

#### Evo-1

| Q8_0 configuration | mean ms | vs F16 | System RAM MiB | vs F16 | max\|Δ\| fixed | max\|Δ\| obs0 | max\|Δ\| obs1 | max\|Δ\| obs2 | max\|Δ\| obs3 | max\|Δ\| obs4 | checks passed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 1659.0 | - | 1582 | - | 6.2e-4 | 1.1e-3 | 9.7e-4 | 4.6e-4 | 4.5e-4 | 9.8e-4 | 6 / 6 |
| base | 1655.4 | -0.2% | 1459 | -7.8% | 5.4e-3 | 1.2e-2 | 7.2e-3 | 4.0e-3 | 3.4e-3 | 6.1e-3 | 0 / 6 |
| base + vision | 1638.8 | -1.2% | 1121 | -29.1% | 6.3e-3 | 1.4e-2 | 5.9e-3 | 4.3e-3 | 4.8e-3 | 6.4e-3 | 0 / 6 |
| base + vision, 25 kept F16 (fixed-input list) | 1634.5 | -1.5% | 1171 | -26.0% | 2.4e-3 | 1.1e-2 | 5.8e-3 | 3.3e-3 | 4.7e-3 | 3.6e-3 | 1 / 6 |
| base, 66 kept F16 (six-check list) | 1657.4 | -0.1% | 1494 | -5.6% | 1.4e-3 | 2.1e-3 | 2.5e-3 | 2.7e-3 | 2.2e-3 | 2.3e-3 | 6 / 6 |

| Q8_0 level | Score, all of the level in Q8_0 | Search on fixed, obs0, obs1, obs2 | Check on obs3, obs4 | Search again on all six | Result | Hours |
|---|--:|---|---|---|---|--:|
| base + vision | 4.74 | time ran out at 1.75 h with a list at score 0.95 not yet refined | - | - | no list | 1.8 |
| base | 4.23 | list found (score ≤ 1) | both pass: 2.2e-3, 2.3e-3 | not needed | passes: 66 weights kept F16 | 1.3 |

### HMX use and the CPU-only build

A per-op profile of the base, base + vision and base + vision + DiT files puts
every Q8_0 matmul on the HMX. The loader zero-pads a 4304-wide FFN to 4320 at
load only when its weights are float, so without `--pad-m32` the vision tower's
FFNs of π0.5 and GR00T N1.6 keep their 4304 width (rows *base + vision,
unpadded*).

Fixed-input max|Δ| of base, against the CPU-only build at F32, run on the NPU
and on the CPU-only build:

| Model | NPU | CPU-only build |
|---|--:|--:|
| π0 | 1.8e-2 | 1.9e-2 |
| π0.5 | 4.9e-4 | 1.1e-3 |
| SmolVLA | 1.2e-2 | 1.8e-2 |
| Evo-1 | 5.4e-3 | 7.3e-3 |
| GR00T N1.5 | 5.1e-3 | 1.2e-2 |
| GR00T N1.6 | 7.2e-3 | 1.2e-2 |
| GR00T N1.7 | 1.1e-2 | 4.4e-2 |
| VLA-JEPA | 1.4e-2 | 2.7e-2 |
| VLA-Adapter | 4.4e-3 | 6.2e-3 |
| TurboVLA | 1.3e-2 | 2.1e-2 |

## Not run

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only
  CUDA builds load it.

## Notes

The HTP skel has to be processed for this board and installed in
`/vendor/dsp/cdsp0..3`: its FastRPC ignores `ADSP_LIBRARY_PATH`. See
[backend/hexagon.md](../backend/hexagon.md#linux-iq-10).

GR00T N1.6 runs one view here, as the configuration above fixes. With two views
its SigLIP2 tower's patch matmul and flash attention, batched over the views,
still run on the CPU; see
[backend/hexagon.md](../backend/hexagon.md#op-coverage-measured).

GR00T N1.6's row runs its SigLIP2 tower with F32 weights and its first three
blocks' matmuls on the CPU, which keeps it within the fidelity bar when served;
the F32 tower is also why its System RAM is higher than N1.5's.

These rows time the defaults; they do not vouch for accuracy. On a served LIBERO
observation, GR00T N1.5 lands above the 2.9e-3 fidelity bar, closer to the
reference than the CPU at F16; see
[backend/hexagon.md](../backend/hexagon.md#serving).

## Reproducing

```bash
scripts/build_linux_hexagon.sh
# copy build-lnx-htp/pkg to the board, install its HTP skel (see backend/hexagon.md), then on the board:
export LD_LIBRARY_PATH=$PWD/pkg/lib

# one row, for example SmolVLA
./pkg/bin/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
