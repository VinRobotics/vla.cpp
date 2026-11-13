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

These rows run GGUFs that `scripts/quantize_gguf.py --type Q8_0` made from the
checkpoints above (ggml's block-32 Q8_0 weights). The setup is the one above,
except:

| | |
|---|---|
| Commit | `71cdb09` |
| Date | 2026-11-10 to 2026-11-13 (UTC+07) |
| Runtime flags | defaults |

Each configuration packs the weights of the one before it, plus:

| Config | `quantize_gguf.py` flags | Packs |
|---|---|---|
| F16 | - | none: the shipped file, loaded as F16 (`--weight-dtype f16`, the default) |
| default | `--type Q8_0` | the 2-D weights outside the vision tower and the action head. Embeddings, the output head, norms, patch embeddings and position tables stay float in every configuration |
| vision, unpadded | `--vision` | the vision tower |
| vision | `--vision --pad-m32` | the vision tower, with FFN widths that are not a multiple of 32 zero-padded (SigLIP's 4304 to 4320 in π0, π0.5, GR00T N1.5 and N1.6, VLA-Adapter and OpenVLA-OFT) |
| vision + DiT | `--vision --pad-m32 --dit` | the DiT's weight matrices, except adaLN and the time embedding |
| ..., keep-float (*kind*, *n*) | the flags of the named configuration plus `--keep-float REGEX` | that configuration with *n* weights left float |

Each configuration runs six checks, max|Δ| of the action chunk against the
CPU-only build at F32: *fixed*, on fixed inputs as in
[Fidelity](../backend/hexagon.md#fidelity), and *obs0* to *obs4*, five LIBERO
observations served as in [Serving](../backend/hexagon.md#serving). obs*k* shifts
the views' color pattern and draws their noise from seed *k*, moves the
end-effector by 0.02·*k* on each axis and takes task prompt *k* mod 4 of four
LIBERO prompts. A check passes when max|Δ| is at most 2.9e-3, or at most 1.1
times F16 where F16 is above 2.9e-3. A configuration's score is the largest
ratio of a check's max|Δ| to its threshold; it passes at 1 or less.

The `--keep-float` lists come from a greedy search over the weights a level
packs: groups of blocks (a quarter of a stack, attention or FFN), most sensitive
first, then single blocks are left float until the score is at most 1, and float
blocks are then packed back while it stays there. *fixed* and *served* lists
search on one check, fixed or obs0, and *union* leaves both float. *all checks*
lists search on fixed and obs0 to obs2, then must pass obs3 and obs4; an
observation that fails joins the search, once. Each level gets 1.75 h. A lower
level is searched only when the level above finds no list, and none is searched
when packing every weight scores above 10. Mean latency and System RAM follow the
TEMPLATE protocol.

### Configuration per model

The fastest Q8_0 configuration that passes all checks, against F16. *Q8_0
weights* counts the weights packed out of those the level can pack; the rest
stay float. Where no configuration passes, the table gives the best search
score, or the score with every weight packed where the search was skipped.

| Model | Q8_0 level | Q8_0 weights | F16 ms | Q8_0 ms | vs F16 | F16 RAM MiB | Q8_0 RAM MiB | vs F16 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| π0 | vision | 281 of 289 | 1123.3 | 937.7 | -16.5% | 5662 | 3326 | -41.3% |
| π0.5 | vision | 108 of 289 | 1107.8 | 1061.2 | -4.2% | 5980 | 5095 | -14.8% |
| Octo-Small | vision | 119 of 123 | 84.1 | 81.5 | -3.1% | 725 | 407 | -43.9% |
| ACT | default | 35 of 35 | 152.5 | 150.9 | -1.0% | 469 | 405 | -13.6% |
| TurboVLA | default | 121 of 133 | 190.2 | 189.7 | -0.3% | 1079 | 881 | -18.4% |
| Evo-1 | default | 34 of 100 | 1659.0 | 1657.4 | -0.1% | 1582 | 1494 | -5.6% |
| OpenVLA-OFT | none passes (best 1.03) | - | 1125.2 | - | - | 15266 | - | - |
| VLA-Adapter | none passes (best 1.03) | - | 669.2 | - | - | 3569 | - | - |
| GR00T N1.5 | none passes (best 1.23) | - | 529.3 | - | - | 3659 | - | - |
| GR00T N1.6 | none passes (best 1.33) | - | 682.6 | - | - | 4929 | - | - |
| GR00T N1.7 | none passes (best 1.40) | - | 744.9 | - | - | 5108 | - | - |
| SmolVLA | none passes (fully packed: 44) | - | 584.0 | - | - | 1226 | - | - |
| VLA-JEPA | none passes (fully packed: 36) | - | 496.8 | - | - | 4018 | - | - |

Flags for `python scripts/quantize_gguf.py --in <checkpoint> --out <file>`:

```bash
# Octo-Small
--type Q8_0 --vision --pad-m32 --keep-float '^octo\.blk\.(1\.attn_(qkv|o)|6\.ffn_(up|down))\.'
# ACT
--type Q8_0
# TurboVLA
--type Q8_0 --keep-float '^text\.encoder\.layer\.[02]\.'
# π0.5
--type Q8_0 --vision --pad-m32 --keep-float '^vit\.blk\.(0|2|3|4|5|6|14|17|18|19|20)\.|^vit\.blk\.(7|8|9|12|13|21|23|24|25)\.attn_|^vlm\.blk\.(9|10|11|12|13|14)\.|^vlm\.blk\.(0|1|2|3|4|15|16)\.ffn_|^vlm\.blk\.[5-8]\.attn_'
# π0
--type Q8_0 --vision --pad-m32 --keep-float '^vlm\.blk\.[04]\.attn_[qkvo]\.'
# Evo-1
--type Q8_0 --keep-float '^vlm\.blk\.([2-9]|10)\.|^vlm\.blk\.13\.ffn_'
```

### Per model

#### Octo-Small

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 84.1 | - | 725 | - | 5.0e-4 | 1.4e-4 | 4.7e-4 | 3.3e-4 | 1.4e-4 | 6.3e-4 | 6 / 6 |
| default | 81.6 | -3.0% | 405 | -44.1% | 2.5e-3 | 2.0e-3 | 4.4e-3 | 2.4e-3 | 1.8e-3 | 4.9e-3 | 4 / 6 |
| vision | 83.5 | -0.7% | 402 | -44.6% | 1.9e-3 | 1.9e-3 | 4.2e-3 | 3.2e-3 | 1.7e-3 | 4.9e-3 | 3 / 6 |
| vision, keep-float (all checks, 4) | 81.5 | -3.1% | 407 | -43.9% | 1.0e-3 | 1.7e-3 | 2.7e-3 | 1.6e-3 | 1.2e-3 | 2.5e-3 | 6 / 6 |

#### ACT

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 152.5 | - | 469 | - | 6.1e-4 | 1.0e-3 | 9.0e-4 | 5.7e-4 | 7.2e-4 | 7.6e-4 | 6 / 6 |
| default | 150.9 | -1.0% | 405 | -13.6% | 2.1e-3 | 2.4e-3 | 2.7e-3 | 2.0e-3 | 2.3e-3 | 2.0e-3 | 6 / 6 |

ACT takes no prompt: its obs*k* shifts the view and the 14-dim state the same
way. Packing every weight passes all six checks; no list was searched.

#### TurboVLA

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 190.2 | - | 1079 | - | 3.7e-4 | 8.5e-4 | crash | crash | crash | 6.3e-4 | 3 / 3 |
| default | 186.5 | -1.9% | 814 | -24.6% | 1.3e-2 | 4.2e-3 | crash | crash | crash | 1.7e-3 | 1 / 3 |
| vision | 185.7 | -2.4% | 485 | -55.1% | 1.4e-2 | 8.3e-3 | crash | crash | crash | 5.7e-3 | 0 / 3 |
| vision, keep-float (fixed, 32) | 185.8 | -2.3% | 611 | -43.4% | 2.6e-3 | 6.9e-3 | crash | crash | crash | 4.2e-3 | 1 / 3 |
| default, keep-float (all checks, 12) | 189.7 | -0.3% | 881 | -18.4% | 1.5e-3 | 2.7e-3 | crash | crash | crash | 1.6e-3 | 3 / 3 |

obs1 to obs3 crash the Hexagon server at F16 too (`ggml-hex: HTP0 synchronize
failed : dsp-error NO-SUPPORT` in `TurboVlaModelArch::predict`), so TurboVLA has
three checks, and its *all checks* list searches on all three. At the vision
level no list passed within the time limit (best score 1.02).

#### VLA-JEPA

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 496.8 | - | 4018 | - | 2.1e-3 | 2.2e-3 | 1.1e-3 | 1.4e-3 | 2.8e-3 | 3.4e-4 | 6 / 6 |
| default | 432.1 | -13.0% | 2718 | -32.4% | 1.4e-2 | 1.0e-1 | 1.8e-2 | 6.4e-2 | 6.9e-2 | 9.9e-3 | 0 / 6 |
| vision | 411.6 | -17.1% | 2350 | -41.5% | 1.5e-2 | 1.6e-1 | 1.5e-2 | 5.3e-2 | 5.4e-2 | 1.0e-2 | 0 / 6 |
| vision + DiT | 393.5 | -20.8% | 2288 | -43.1% | 1.3e-2 | 1.3e-1 | 1.6e-2 | 5.5e-2 | 5.5e-2 | 1.0e-2 | 0 / 6 |
| vision + DiT, keep-float (fixed, 233) | 461.4 | -7.1% | 3614 | -10.1% | 2.9e-3 | 8.3e-3 | 2.5e-3 | 5.8e-3 | 3.3e-3 | 2.5e-3 | 3 / 6 |
| vision + DiT, keep-float (served, 143) | 436.6 | -12.1% | 3117 | -22.4% | 8.4e-3 | 2.1e-3 | 3.8e-3 | 6.1e-3 | 2.2e-2 | 2.9e-3 | 2 / 6 |
| vision + DiT, keep-float (union, 244) | 461.6 | -7.1% | 3679 | -8.4% | 3.7e-3 | 5.1e-3 | 2.8e-3 | 3.2e-3 | 3.1e-3 | 2.4e-3 | 2 / 6 |

Packing every weight scores 43 (vision + DiT), 55 (vision) and 36 (default); no
list was searched.

#### GR00T N1.5

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 529.3 | - | 3659 | - | 1.9e-3 | 1.2e-2 | 4.1e-3 | 3.8e-3 | 3.1e-2 | 1.2e-2 | 6 / 6 |
| default | 500.2 | -5.5% | 3148 | -14.0% | 5.1e-3 | 8.0e-2 | 2.1e-2 | 4.3e-2 | 9.8e-2 | 6.5e-2 | 0 / 6 |
| vision | 481.1 | -9.1% | 2754 | -24.7% | 7.2e-3 | 4.6e-2 | 4.9e-2 | 1.2e-1 | 1.4e-1 | 5.2e-2 | 0 / 6 |
| vision + DiT | 399.2 | -24.6% | 2339 | -36.1% | 8.4e-3 | 3.6e-2 | 9.7e-2 | 1.3e-1 | 1.6e-1 | 6.5e-2 | 0 / 6 |
| vision + DiT, keep-float (fixed, 191) | 449.0 | -15.2% | 3061 | -16.3% | 2.7e-3 | 2.8e-2 | 3.5e-2 | 5.1e-2 | 5.4e-2 | 6.2e-2 | 1 / 6 |
| vision + DiT, keep-float (served, 223) | 472.9 | -10.7% | 3172 | -13.3% | 6.8e-3 | 9.1e-3 | 2.7e-2 | 1.2e-2 | 3.0e-2 | 1.6e-2 | 2 / 6 |
| vision + DiT, keep-float (union, 272) | 491.2 | -7.2% | 3413 | -6.7% | 3.3e-3 | 1.1e-2 | 1.1e-2 | 1.2e-2 | 3.5e-2 | 1.7e-2 | 1 / 6 |

Searched with 4 h per level, before levels scoring above 10 were skipped; no
list passed. Best scores: 1.84 (vision + DiT), 1.77 (vision), 1.23 (default).

#### SmolVLA

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 584.0 | - | 1226 | - | 7.4e-4 | 1.9e-3 | 2.9e-2 | 1.9e-1 | 5.6e-2 | 2.5e-1 | 6 / 6 |
| default | 569.0 | -2.6% | 919 | -25.0% | 1.2e-2 | 1.3e-2 | 1.4e+0 | 1.3e-1 | 2.2e-2 | 5.3e-1 | 2 / 6 |
| vision | 561.0 | -3.9% | 848 | -30.8% | 3.4e-2 | 1.6e-2 | 1.7e+0 | 1.1e-1 | 8.5e-2 | 4.4e-1 | 1 / 6 |
| vision, keep-float (fixed, 58) | 564.2 | -3.4% | 896 | -26.9% | 2.0e-3 | 1.6e-2 | 1.6e+0 | 2.4e-1 | 2.6e-1 | 9.6e-1 | 1 / 6 |

Packing every weight scores 54 (vision) and 44 (default); no list was searched.
On obs1 to obs4 the gripper channel switches sign within the chunk, and a shift
in when it switches sets max|Δ|, at F16 too.

#### VLA-Adapter

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 669.2 | - | 3569 | - | 3.9e-4 | 3.9e-4 | 1.2e-3 | 6.6e-4 | 4.7e-4 | 3.3e-4 | 6 / 6 |
| default | 653.5 | -2.3% | 3200 | -10.3% | 4.3e-3 | 6.3e-3 | 1.4e-2 | 9.6e-3 | 8.0e-3 | 8.2e-3 | 0 / 6 |
| vision | 623.4 | -6.8% | 2555 | -28.4% | 3.1e-3 | 6.0e-3 | 1.5e-2 | 8.9e-3 | 7.7e-3 | 7.1e-3 | 0 / 6 |
| vision, keep-float (fixed, 8) | 622.7 | -6.9% | 2516 | -29.5% | 2.8e-3 | 6.3e-3 | 1.4e-2 | 6.9e-3 | 8.5e-3 | 6.4e-3 | 1 / 6 |
| vision, keep-float (served, 40) | 627.7 | -6.2% | 2494 | -30.1% | 2.9e-3 | 2.9e-3 | 1.5e-2 | 6.6e-3 | 8.0e-3 | 5.3e-3 | 2 / 6 |

No list passed within the time limit. At the vision level the best score is
1.03. At the default level the retry on all six checks reached 0.82 only with
every LM block float, and the time limit ran out while blocks were packed back.

#### GR00T N1.6

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 682.6 | - | 4929 | - | 1.3e-3 | 1.0e-2 | 1.2e-3 | 1.2e-3 | 1.1e-3 | 1.8e-3 | 6 / 6 |
| default | 644.5 | -5.6% | 3959 | -19.7% | 7.2e-3 | 3.4e-2 | 8.1e-3 | 9.2e-3 | 1.5e-2 | 8.3e-3 | 0 / 6 |
| vision, unpadded | 817.3 | +19.7% | 3738 | -24.2% | 7.1e-3 | 2.5e-2 | - | - | - | - | 0 / 2 |
| vision | 622.5 | -8.8% | 3588 | -27.2% | 8.0e-3 | 2.9e-2 | 2.1e-2 | 1.0e-2 | 2.5e-2 | 2.2e-2 | 0 / 6 |
| vision + DiT | 462.2 | -32.3% | 2777 | -43.7% | 8.1e-3 | 2.9e-2 | 2.0e-2 | 1.0e-2 | 2.5e-2 | 2.2e-2 | 0 / 6 |
| vision + DiT, keep-float (fixed, 57) | 482.3 | -29.3% | 3076 | -37.6% | 2.8e-3 | 3.6e-2 | 1.7e-2 | 1.2e-2 | 2.2e-2 | 2.2e-2 | 1 / 6 |
| vision + DiT, keep-float (served, 3) | 464.0 | -32.0% | 2800 | -43.2% | 7.6e-3 | 9.2e-3 | 1.2e-2 | 1.1e-2 | 1.6e-2 | 2.8e-2 | 1 / 6 |
| vision + DiT, keep-float (union, 60) | 480.7 | -29.6% | 3114 | -36.8% | 3.0e-3 | 8.6e-3 | 7.1e-3 | 7.9e-3 | 1.3e-2 | 3.3e-2 | 1 / 6 |

*vision, unpadded* ran fixed and obs0 only. No list passed within the time
limit. Best scores: 1.68 (vision + DiT), 1.33 (vision), 1.55 (default).

#### GR00T N1.7

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 744.9 | - | 5108 | - | 2.0e-3 | 2.0e-3 | 2.8e-3 | 1.3e-3 | 1.5e-3 | 6.2e-4 | 6 / 6 |
| default | 703.5 | -5.6% | 4356 | -14.7% | 1.1e-2 | 2.0e-2 | 1.0e-2 | 2.2e-2 | 1.7e-2 | 2.2e-2 | 0 / 6 |
| vision | 688.4 | -7.6% | 3996 | -21.8% | 2.0e-2 | 2.2e-2 | 1.1e-2 | 1.7e-2 | 1.9e-2 | 1.8e-2 | 0 / 6 |
| vision + DiT | 536.1 | -28.0% | 3151 | -38.3% | 2.8e-2 | 2.1e-2 | 1.0e-2 | 1.7e-2 | 1.7e-2 | 1.7e-2 | 0 / 6 |
| vision + DiT, keep-float (fixed, 185) | 606.0 | -18.6% | 4051 | -20.7% | 2.6e-3 | 5.3e-3 | 5.6e-3 | 5.0e-3 | 2.0e-3 | 2.2e-3 | 3 / 6 |
| vision + DiT, keep-float (served, 157) | 579.8 | -22.2% | 3870 | -24.2% | 4.2e-2 | 2.8e-3 | 3.6e-3 | 1.8e-3 | 1.6e-3 | 3.3e-3 | 3 / 6 |
| vision + DiT, keep-float (union, 233) | 614.4 | -17.5% | 4249 | -16.8% | 3.9e-3 | 6.5e-3 | 3.0e-3 | 3.0e-3 | 1.4e-3 | 1.2e-3 | 3 / 6 |

No list passed within the time limit. Best scores: 1.40 (vision + DiT), 2.07
(vision), 1.45 (default).

#### π0.5

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 1107.8 | - | 5980 | - | 4.8e-4 | 3.7e-4 | 6.0e-4 | 5.5e-3 | 8.9e-4 | 3.9e-4 | 6 / 6 |
| default | 971.0 | -12.3% | 3997 | -33.2% | 4.9e-4 | 1.9e-3 | 5.0e-3 | 1.7e-2 | 3.9e-3 | 3.5e-3 | 2 / 6 |
| vision, unpadded | 1320.2 | +19.2% | 3760 | -37.1% | 8.5e-4 | 2.7e-3 | - | - | - | - | 2 / 2 |
| vision | 941.3 | -15.0% | 3676 | -38.5% | 9.2e-4 | 2.8e-3 | 8.0e-3 | 1.7e-2 | 1.0e-2 | 4.9e-3 | 2 / 6 |
| vision, keep-float (all checks, 181) | 1061.2 | -4.2% | 5095 | -14.8% | 6.0e-4 | 8.5e-4 | 2.8e-3 | 6.0e-3 | 2.5e-3 | 1.1e-3 | 6 / 6 |

*vision, unpadded* ran fixed and obs0 only. The *all checks* list was searched
without the time limit, in 5.6 h; the default level was not searched.

#### π0

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 1123.3 | - | 5662 | - | 7.7e-4 | 5.8e-4 | 4.7e-4 | 5.1e-4 | 5.8e-4 | 6.1e-4 | 6 / 6 |
| default | 978.8 | -12.9% | 3723 | -34.2% | 1.8e-2 | 1.7e-3 | 6.2e-4 | 6.6e-4 | 8.8e-4 | 1.4e-3 | 5 / 6 |
| vision | 937.6 | -16.5% | 3337 | -41.1% | 1.6e-2 | 1.7e-3 | 5.6e-4 | 7.2e-4 | 5.4e-4 | 1.5e-3 | 5 / 6 |
| default, keep-float (fixed, 8) | 978.8 | -12.9% | 3711 | -34.5% | 1.4e-3 | 5.2e-4 | 6.1e-4 | 6.6e-4 | 5.3e-4 | 7.4e-4 | 6 / 6 |
| vision, keep-float (all checks, 8) | 937.7 | -16.5% | 3326 | -41.3% | 1.0e-3 | 5.6e-4 | 6.1e-4 | 6.7e-4 | 4.6e-4 | 1.0e-3 | 6 / 6 |

The default level was not searched.

#### OpenVLA-OFT

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 1125.2 | - | 15266 | - | 3.1e-4 | 7.5e-4 | 5.6e-4 | 4.3e-4 | 6.8e-4 | 5.6e-4 | 6 / 6 |
| default | 845.7 | -24.8% | 9403 | -38.4% | 5.9e-4 | 3.5e-3 | 5.1e-3 | 4.4e-3 | 4.9e-3 | 3.8e-3 | 1 / 6 |
| vision | 812.5 | -27.8% | 8821 | -42.2% | 7.9e-4 | 3.2e-3 | 5.8e-3 | 4.5e-3 | 3.9e-3 | 6.0e-3 | 1 / 6 |
| vision, keep-float (served, 12) | 821.4 | -27.0% | 8959 | -41.3% | 7.9e-4 | 2.5e-3 | 4.3e-3 | 3.4e-3 | 2.8e-3 | 5.4e-3 | 3 / 6 |

No list passed within the time limit. Best scores: 1.35 (vision), 1.03
(default).

#### Evo-1

| Config | mean ms | vs F16 | System RAM MiB | vs F16 | fixed | obs0 | obs1 | obs2 | obs3 | obs4 | passes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F16 | 1659.0 | - | 1582 | - | 6.2e-4 | 1.1e-3 | 9.7e-4 | 4.6e-4 | 4.5e-4 | 9.8e-4 | 6 / 6 |
| default | 1655.4 | -0.2% | 1459 | -7.8% | 5.4e-3 | 1.2e-2 | 7.2e-3 | 4.0e-3 | 3.4e-3 | 6.1e-3 | 0 / 6 |
| vision | 1638.8 | -1.2% | 1121 | -29.1% | 6.3e-3 | 1.4e-2 | 5.9e-3 | 4.3e-3 | 4.8e-3 | 6.4e-3 | 0 / 6 |
| vision, keep-float (fixed, 25) | 1634.5 | -1.5% | 1171 | -26.0% | 2.4e-3 | 1.1e-2 | 5.8e-3 | 3.3e-3 | 4.7e-3 | 3.6e-3 | 1 / 6 |
| default, keep-float (all checks, 66) | 1657.4 | -0.1% | 1494 | -5.6% | 1.4e-3 | 2.1e-3 | 2.5e-3 | 2.7e-3 | 2.2e-3 | 2.3e-3 | 6 / 6 |

At the vision level a list scored 0.95 on the search set, but the time limit ran
out before it was refined and checked on obs3 and obs4.

A per-op profile of the default, vision and vision + DiT files puts every Q8_0
matmul on the HMX. The loader zero-pads a 4304-wide FFN to 4320 at load only
when its weights are float, so without `--pad-m32` the vision tower's FFNs of
π0.5 and GR00T N1.6 keep their 4304 width (rows *vision, unpadded*).

Fixed-input deviation of the default configuration on the NPU and on the
CPU-only build, against the CPU-only build at F32:

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
