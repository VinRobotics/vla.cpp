# LIBERO task success

Each model below runs closed loop in LIBERO through `vla-server`, driven by
[`eval/run_libero.sh`](../../eval/run_libero.sh): 10 tasks and 20 episodes per
task, so 200 episodes per suite. An episode that hits the step limit counts as
a failure.

## Test setup

| | |
|---|---|
| Device | NVIDIA GeForce RTX 3090, 24 GB, CUDA 12.8 (the build in [rtx-3090.md](rtx-3090.md)) |
| Commit | `c93ca0a` (branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 11:28 to 16:53 (UTC+07) |
| Checkpoints | the `vrfai/*` GGUFs listed in [rtx-3090.md](rtx-3090.md#model-configuration); BitVLA and GR00T N1.7 use the per-suite checkpoint |
| Runtime flags | defaults (GR00T in BF16, as shipped); diffusion noise not seeded |
| Simulator | LIBERO with robosuite 1.4.0 and MuJoCo 3.9.0, 500-step limit in every suite |
| Replay | actions executed from each predicted chunk before the next query, the `run_libero.sh` default |

## LIBERO-Object, all models

| Model | Replay | Successes | Success rate |
|---|--:|--:|--:|
| BitVLA      |  8 | 200/200 | 100.0% |
| TurboVLA    | 12 | 200/200 | 100.0% |
| VLA-JEPA    |  7 | 200/200 | 100.0% |
| Evo-1       | 14 | 197/200 |  98.5% |
| OpenVLA-OFT |  8 | 197/200 |  98.5% |
| GR00T N1.7  | 16 | 196/200 |  98.0% |
| π0.5        | 10 | 195/200 |  97.5% |
| GR00T N1.5  | 16 | 194/200 |  97.0% |
| VLA-Adapter |  8 | 191/200 |  95.5% |
| SmolVLA     | 10 | 179/200 |  89.5% |
| GR00T N1.6  | 16 | 173/200 |  86.5% |
| π0          | 50 | 144/200 |  72.0% |
| Octo-Small  |  4 |  16/200 |   8.0% |

With 200 episodes, a rate near 95% carries a 95% confidence interval of about
±3 points, so gaps of a few points between models are not significant.

Most replay counts match the model's upstream LIBERO eval. Three do not.
SmolVLA replays 10 of its 50-step chunk, the best setting in its paper's
ablation, where upstream replans every step. π0 replays the whole chunk, as
its LeRobot config does, where openpi's LIBERO eval replans every 5. GR00T
replays 16, where Isaac-GR00T uses 8 (1 for N1.5). π0 at 72% and Octo at 8%
are well below published results for these checkpoints and have not been
investigated yet.

## Four suites

| Model | Spatial | Object | Goal | Long (LIBERO-10) | Average |
|---|--:|--:|--:|--:|--:|
| BitVLA     | 95.5% | 100.0% | 91.5% | 89.5% | 94.1% |
| GR00T N1.7 | 95.5% |  98.0% | 97.5% | 90.0% | 95.3% |

Same replay as above: 8 for BitVLA, 16 for GR00T N1.7.

Success rate belongs to the checkpoint, not the engine;
`vla_predict_check` in [CONTRIBUTING.md](../../CONTRIBUTING.md) is how a
change is shown to leave it alone.
