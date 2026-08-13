# SmolVLA on the UR10e through vla.cpp - CUDA vs CPU

Two sessions, 40 rollouts, 2026-08-07. Same checkpoint, same GGUF, same client,
same bridge, same host. The only difference is which `vla-server` binary answered:

| Session | Engine | Data | Labels |
|---|---|---|---|
| G | `build/ReleaseCUDA/vla-server` | `rollouts_vlacpp_gpu/` | operator, 20/20 |
| C | `build/ReleaseCPU/vla-server`, `VLA_SMOLVLA_FA=1 VLA_WEIGHT_DTYPE=f32` | `rollouts_vlacpp_cpu1/` | operator 19/20, ep20 from video |

Host: i9-14900HX + RTX 5070 Laptop (Blackwell, 8 GB). Client, bridge and engine
all on this box - `server: 127.0.0.1:8791` in every `metadata.json`, so unlike the
earlier M4 / Ryzen sessions there is no network in the loop. Every number below
comes from the `states.npz` / `metadata.json` files and the recorded video.

## Summary

**Latency is the whole story, and it is a 19x gap.** 111 ms per query on CUDA
against 2164 ms on CPU. The engine is 24x apart (88 ms vs 2142 ms); everything
around it - the bridge preprocessing and the loopback socket - is identical to
within 1 ms. Both are metronome-stable across a full session: 3.7 - 4.9% spread
in episode means, no thermal drift, zero dropped ticks, zero server errors over
1243 queries.

**What that buys on the robot is the control rate: 17.7 Hz against 7.1 Hz.** On
CUDA the client spends 8% of the session blocked on the server; on CPU it spends
62%. At `--exec-horizon 25` a chunk cycle is 26 ticks on CUDA (1.30 s, of which
0.11 s is dwell) and 68 ticks on CPU (3.40 s, of which 2.16 s is dwell). CUDA is
the first configuration on this robot that gets close to the 20 Hz the policy was
trained at.

**Success rate: 15/20 (75%) on CUDA, 12/20 (60%) on CPU.** The difference is not
significant (Fisher p = 0.50; Wilson 95% CIs 53-89% and 39-78%, heavily
overlapping). Do not read a quality gap into it.

**The failure is the same one documented in `REPORT_SMOLVLA.md`, unchanged.** All
13 failures across both sessions are a failed *release*: the policy grasps the cup
and carries it to the basket in 40 of 40 episodes, then in the failures holds it
just outside the rim and oscillates there until the operator stops the run. Grasp
and transport never failed on either backend.

**One earlier conclusion does not replicate.** `REPORT_SMOLVLA.md` reported that
doubling latency (1.18 s -> 2.34 s) cost 19% more steps to reach the grasp. Here
latency goes up 19x and steps-to-grasp does not move: 335 +- 26 on CUDA against
329 +- 41 on CPU (Welch t = 0.57). See
[Latency and policy behaviour](#latency-and-policy-behaviour).

## Headline comparison

| Metric | G: vla.cpp CUDA | C: vla.cpp CPU | Ratio |
|---|---|---|---|
| Success | 15/20 (75%) | 12/20 (60%) | Fisher p = 0.50 |
| Server round trip, mean | **111.4 ms** | **2164.4 ms** | 19.4x |
| Engine inference, mean | 88.3 ms | 2142.0 ms | 24.3x |
| Round trip p95 / p99 / max | 126 / 148 / 152 ms | 2301 / 2424 / 2492 ms | |
| Round-trip spread (episode means) | 109.4 - 114.8 ms (4.9%) | 2131 - 2211 ms (3.7%) | |
| Coefficient of variation | 0.069 | 0.038 | |
| Wall clock blocked on the server | **8.0%** | **62.1%** | |
| Effective control rate | **17.68 Hz** | **7.11 Hz** (nominal 20 Hz) | 2.49x |
| Chunk cycle, p50 | 1300 ms (26 ticks) | 3400 ms (68 ticks) | |
| Steps to first grasp | 335 +- 26 | 329 +- 41 | t = 0.57, n.s. |
| Time to first grasp | 17.6 +- 1.3 s | 45.5 +- 5.5 s | 2.6x |
| Steps per successful episode | 835 (1.16x a demo) | 857 (1.19x a demo) | |
| Session wall time | 14.8 min | 35.2 min | 2.4x |
| Steps / queries recorded | 15,681 / 637 | 15,030 / 606 | |
| Dropped recording ticks | 0 | 0 | |
| Slew-limited commands | 0.08% | 0.04% | |
| Stop reason | `interrupted` x20 | `interrupted` x20 | |

## Latency

### Where the round trip goes

Each query is measured at three points: `latency_ms` by the client (request sent ->
chunk received), `handling_ms` by the bridge (`vlacpp_policy_server.py`, the whole
request), `inference_ms` by the engine (`vla-server`, `latency_ms_total`).
Subtracting gives the loopback socket and the bridge's own preprocessing.

| Stage | CUDA mean | CPU mean | What it is |
|---|---|---|---|
| Engine inference | **88.3 ms** | **2142.0 ms** | SigLIP tower + prefill + 10 flow-matching steps, inside `vla-server` |
| Bridge preprocessing | 21.1 ms | 20.6 ms | resize-with-pad 480x640 -> 512x512 x2, tokenizer lookup, protobuf + ZMQ hop |
| Pickle loopback | 2.0 ms | 1.8 ms | client <-> bridge over 127.0.0.1, 1.84 MB of raw frames |
| **Round trip** | **111.4 ms** | **2164.4 ms** | |
| Inference share of round trip | 79.3% | 99.0% | |

The bridge costs the same 21 ms either way - it is numpy and a dict lookup on the
host CPU, unaffected by the backend. On CUDA that is a fifth of the whole query
and the largest remaining target; on CPU it is 1% and irrelevant.

### Distribution

| Percentile | CUDA round trip | CUDA engine | CPU round trip | CPU engine |
|---|---|---|---|---|
| min | 102.7 | 82.0 | 1934.4 | 1911.7 |
| p50 | 109.5 | 86.7 | 2151.1 | 2128.9 |
| p90 | 116.8 | 90.7 | 2270.4 | 2246.7 |
| p95 | 126.0 | 102.4 | 2300.7 | 2276.4 |
| p99 | 147.8 | 123.7 | 2423.9 | 2401.1 |
| max | 152.0 | 130.5 | 2491.9 | 2466.5 |
| sd | 7.7 | 6.9 | 82.5 | 82.1 |

All in ms, 637 and 606 queries. The CPU path is *relatively* tighter (CV 0.038 vs
0.069) but its absolute jitter is 11x larger: a p99 outlier costs 36 ms on CUDA
and 260 ms on CPU, which is 5 extra ticks of dwell.

**CUDA has a warm-up query, CPU does not.** The first query of each episode
averages 120.5 ms against 87.3 ms for the rest (+38%); the CPU first query is
2134 ms against 2142 ms, i.e. nothing. That is the CUDA graph / kernel path
re-warming after the bridge has been idle between episodes. It costs 33 ms once
per episode and is not worth chasing.

**Neither backend degraded over a session.** CUDA episode means: first five 88.6
ms, last five 88.3 ms. CPU: first five 2135 ms, last five 2155 ms. Full-session
spread 2.7 ms (3.1%) and 79.7 ms (3.7%). No thermal ramp on either, over 15 and
35 minutes of continuous load on a laptop.

### What the client does with it

The client ticks at 20 Hz, executes `exec_horizon = 25` actions from each chunk,
then issues the next query and **blocks** on it. Ticks are quantized to 50 ms, so
the cycle rounds up:

| | CUDA | CPU |
|---|---|---|
| Execute 25 actions | 1250 ms | 1250 ms |
| Blocked on the query | 111 ms | 2164 ms |
| Chunk cycle, p50 | **1300 ms** = 26 ticks | **3400 ms** = 68 ticks |
| Chunk cycle, mean | 1383 ms | 3464 ms |
| Dwell fraction | 4% of the cycle | 64% of the cycle |
| Predicted rate | 18.4 Hz | 7.3 Hz |
| Measured session rate | 17.68 Hz | 7.11 Hz |

On CUDA the pause between chunks is one tick. The arm looks continuous. On CPU it
is 43 ticks of standing still per 25 of motion.

### Observation staleness

`img_age` is recorded after the tick's work, so on the tick that carries a query
it reads the frame age *plus* the query duration. Netting that out, the frame
actually sent to the policy is ~17 ms old on both paths - the cameras run at
~30 Hz and both backends sample equally fresh.

What differs is how old that observation is by the time each action derived from it
reaches the arm:

| | CUDA | CPU |
|---|---|---|
| Age at the 1st action of the chunk | 128 ms | 2181 ms |
| Age at the 25th action of the chunk | 1378 ms | 3431 ms |
| Worst observed | 185 ms | 2516 ms |

The whole chunk on CUDA is acted on fresher than the *first* action on CPU.

### The `VLA_SMOLVLA_FA=1` gain, on the robot

`rollouts_vlacpp_cpu/` is an earlier CPU session on the same host without flash
attention in the vision tower. Same client, same GGUF:

| CPU session | Engine mean | Engine p50 | Round trip mean | Episode-mean spread | Rate |
|---|---|---|---|---|---|
| `rollouts_vlacpp_cpu` (no FA) | 2350 ms | 2371 ms | 2374 ms | 16.4% | 6.69 Hz |
| `rollouts_vlacpp_cpu1` (FA) | **2142 ms** | 2129 ms | **2164 ms** | 3.7% | **7.11 Hz** |

-8.8% on the engine, +6% control rate, and the run-to-run spread collapses from
16.4% to 3.7%. This confirms the bench number in `VLACPP.md` (~2.3 s -> ~2.0 s) on
the robot. `VLA_SMOLVLA_FA=1` should stay in the documented CPU command.

## Success rate and labels

| | CUDA | CPU |
|---|---|---|
| Operator labels | 20 of 20 | 19 of 20 |
| Pass | 15 | 11 labeled + ep20 |
| Success rate | 15/20 = 75% | 12/20 = 60% |
| Wilson 95% CI | 53 - 89% | 39 - 78% |

**On the label count.** The supplied CPU list has 19 entries against 20 rollouts.
Aligning it to the first 19 rollouts in timestamp order is self-consistent: the
stall detector (below) fires on exactly the 8 episodes marked `F` and on none of
the 11 marked `P`, with no misalignment possible. That leaves
`rollout_20260807_102344` unlabeled. Its last frame shows the blue cup sitting in
the basket with the gripper open above it, and its signals match the pass group
exactly (idle 0.1 s, no stall) - so it is scored as a pass, and both numbers are
given above. `figures_vlacpp/end_cpu_ep18_fail_vs_ep20_pass.jpg` is that frame
next to a confirmed failure.

**The stall detector from `REPORT_SMOLVLA.md` holds on both sessions.** Rule:
*measured gripper closed at the end AND no gripper command for more than 10 s*. It
agrees with the operator on **39 of 39** labeled episodes here - 13/13 failures,
0 false alarms on 26 successes - and it now stands at 29/29 failures with 0 false
alarms over 79 episodes and four backends.

Margin note: on CUDA the pass/fail idle gap is 6.4 s vs 13.3 s, against 2.5 s vs
28.7 s on CPU. Episodes are 2.5x shorter in wall clock, so the 10 s threshold has
less headroom. It still separates cleanly, but if the control rate rises further,
count the idle in policy steps rather than seconds.

## Per-episode results

`1st grasp` = step of the first gripper CLOSE. `Last grip` = time of the last
gripper command. `Idle` = seconds from that command to the operator's stop.
`Drift` = median over the episode of `max|action - measured joints|`.

### Session G - vla.cpp CUDA (`rollouts_vlacpp_gpu/`)

| Ep | Result | Rollout | Steps | Wall s | Q | RT ms | Engine ms | 1st grasp | Last grip s | Idle s | Drift rad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pass | `085124` | 756 | 44.4 | 31 | 111.8 | 87.8 | 320 | 43.6 | 0.2 | 0.0055 |
| 2 | pass | `085238` | 761 | 44.5 | 31 | 113.2 | 88.8 | 318 | 43.8 | 0.3 | 0.0059 |
| 3 | pass | `085358` | 766 | 44.7 | 31 | 113.3 | 88.9 | 311 | 42.4 | 1.8 | 0.0060 |
| 4 | **fail** | `085501` | 639 | 34.4 | 26 | 114.8 | 89.8 | 338 | 17.8 | 16.6 | 0.0066 |
| 5 | pass | `085601` | 940 | 55.3 | 38 | 112.1 | 87.7 | 346 | 53.6 | 1.7 | 0.0044 |
| 6 | **fail** | `085712` | 575 | 33.1 | 23 | 111.5 | 89.1 | 333 | 17.8 | 15.3 | 0.0070 |
| 7 | pass | `085816` | 828 | 47.4 | 34 | 110.0 | 87.5 | 340 | 45.6 | 1.9 | 0.0046 |
| 8 | pass | `085919` | 735 | 41.5 | 30 | 109.8 | 87.5 | 338 | 35.1 | 6.4 | 0.0057 |
| 9 | pass | `090022` | 835 | 47.8 | 34 | 109.4 | 87.4 | 319 | 46.3 | 1.5 | 0.0045 |
| 10 | pass | `090128` | 1053 | 57.2 | 43 | 111.5 | 88.4 | 339 | 56.0 | 0.0 | 0.0037 |
| 11 | pass | `090242` | 811 | 46.7 | 33 | 111.5 | 88.8 | 416 | 45.1 | 1.6 | 0.0049 |
| 12 | **fail** | `090346` | 545 | 29.5 | 22 | 112.0 | 89.5 | 308 | 16.2 | 13.3 | 0.0081 |
| 13 | pass | `090435` | 917 | 50.9 | 37 | 111.5 | 88.7 | 363 | 48.1 | 1.8 | 0.0043 |
| 14 | pass | `090548` | 869 | 48.5 | 35 | 110.3 | 87.8 | 312 | 45.5 | 2.9 | 0.0051 |
| 15 | pass | `090653` | 747 | 42.1 | 30 | 110.5 | 88.3 | 326 | 39.0 | 3.1 | 0.0057 |
| 16 | pass | `090803` | 764 | 44.2 | 31 | 111.8 | 88.7 | 316 | 42.3 | 1.9 | 0.0055 |
| 17 | pass | `090903` | 791 | 45.7 | 32 | 112.6 | 89.5 | 315 | 45.4 | 0.2 | 0.0055 |
| 18 | **fail** | `091021` | 733 | 40.3 | 30 | 110.8 | 88.4 | 313 | 17.4 | 23.0 | 0.0060 |
| 19 | pass | `091201` | 953 | 52.9 | 39 | 109.4 | 87.0 | 363 | 52.8 | 0.0 | 0.0046 |
| 20 | **fail** | `091311` | 663 | 35.6 | 27 | 111.0 | 88.2 | 362 | 18.9 | 16.6 | 0.0066 |

### Session C - vla.cpp CPU (`rollouts_vlacpp_cpu1/`)

| Ep | Result | Rollout | Steps | Wall s | Q | RT ms | Engine ms | 1st grasp | Last grip s | Idle s | Drift rad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pass | `094404` | 750 | 105.2 | 30 | 2194.6 | 2171.7 | 257 | 103.1 | 1.7 | 0.0052 |
| 2 | **fail** | `094602` | 575 | 79.1 | 23 | 2130.9 | 2109.0 | 361 | 49.5 | 29.0 | 0.0064 |
| 3 | **fail** | `094741` | 525 | 76.2 | 21 | 2136.9 | 2114.5 | 268 | 45.6 | 28.7 | 0.0082 |
| 4 | **fail** | `094917` | 575 | 79.2 | 23 | 2151.9 | 2130.2 | 334 | 46.7 | 32.3 | 0.0074 |
| 5 | pass | `095054` | 802 | 112.3 | 33 | 2172.7 | 2150.0 | 284 | 109.9 | 2.4 | 0.0049 |
| 6 | pass | `095305` | 705 | 99.0 | 29 | 2180.9 | 2158.6 | 338 | 98.8 | 0.1 | 0.0058 |
| 7 | pass | `095459` | 900 | 122.5 | 36 | 2138.4 | 2116.6 | 291 | 122.1 | 0.3 | 0.0041 |
| 8 | pass | `095720` | 900 | 125.6 | 36 | 2149.5 | 2127.3 | 295 | 123.8 | 0.2 | 0.0046 |
| 9 | pass | `095941` | 875 | 121.5 | 35 | 2159.5 | 2137.0 | 390 | 119.5 | 0.2 | 0.0045 |
| 10 | pass | `100204` | 1025 | 144.8 | 41 | 2146.2 | 2123.7 | 398 | 142.7 | 0.4 | 0.0037 |
| 11 | pass | `100446` | 877 | 123.6 | 36 | 2176.2 | 2153.6 | 313 | 121.1 | 2.5 | 0.0042 |
| 12 | **fail** | `100713` | 600 | 83.3 | 24 | 2189.2 | 2166.5 | 384 | 53.5 | 29.4 | 0.0063 |
| 13 | **fail** | `100858` | 580 | 81.4 | 24 | 2158.1 | 2135.9 | 363 | 50.2 | 31.2 | 0.0064 |
| 14 | pass | `101100` | 1025 | 146.8 | 41 | 2177.8 | 2155.2 | 354 | 145.1 | 0.5 | 0.0039 |
| 15 | **fail** | `101344` | 575 | 80.7 | 23 | 2135.6 | 2113.1 | 292 | 49.7 | 30.0 | 0.0062 |
| 16 | pass | `101525` | 766 | 108.4 | 31 | 2177.6 | 2154.7 | 307 | 106.7 | 1.7 | 0.0055 |
| 17 | pass | `101736` | 800 | 112.8 | 32 | 2168.5 | 2146.1 | 340 | 111.7 | 0.3 | 0.0051 |
| 18 | **fail** | `101952` | 725 | 105.6 | 29 | 2158.9 | 2136.6 | 323 | 72.2 | 32.4 | 0.0051 |
| 19 | **fail** | `102158` | 555 | 80.2 | 23 | 2211.4 | 2188.7 | 315 | 47.6 | 32.6 | 0.0070 |
| 20 | pass* | `102344` | 895 | 124.3 | 36 | 2169.4 | 2146.7 | 366 | 124.2 | 0.1 | 0.0043 |

\* not in the supplied label list; scored from the video and the signals.

## The failure mode

Unchanged from `REPORT_SMOLVLA.md`: transport succeeds, release does not. In all
13 failures the last gripper command is a CLOSE issued long before the operator
stopped, the measured gripper reads 0.000, and the last frame shows the cup still
clamped in the gripper next to the basket.
`figures_vlacpp/end_side_gpu.jpg` and `figures_vlacpp/end_side_cpu.jpg` are 5x4
contact sheets of the last frame of every episode, green border = pass.

**End pose against the 44 training demonstrations.** Release pose = last frame of
a success, stuck pose = last frame of a failure; `z` is in demonstration sd:

| Joint | Train release | sd | Release G / z | Release C / z | Stuck G / z | Stuck C / z |
|---|---|---|---|---|---|---|
| shoulder_pan | -1.0391 | 0.019 | -1.0580 / -1.0 | -1.0645 / -1.3 | -1.0806 / -2.2 | -1.1296 / **-4.8** |
| shoulder_lift | -1.9178 | 0.026 | -1.8723 / +1.8 | -1.8801 / +1.5 | -1.8591 / +2.3 | -1.8266 / **+3.6** |
| elbow | -1.5323 | 0.031 | -1.5754 / -1.4 | -1.5838 / -1.7 | -1.5369 / -0.1 | -1.5447 / -0.4 |
| wrist_1 | -1.2655 | 0.025 | -1.2683 / -0.1 | -1.2521 / +0.5 | -1.3191 / -2.1 | -1.3445 / **-3.2** |
| wrist_2 | +1.5713 | 0.0001 | +1.5713 | +1.5712 | +1.5713 | +1.5712 |
| wrist_3 | -2.6082 | 0.019 | -2.6284 / -1.1 | -2.6346 / -1.4 | -2.6468 / -2.0 | -2.6983 / **-4.7** |

RMS z over the five live joints: release 1.21 (G) and 1.34 (C), stuck 1.94 (G) and
3.68 (C). Successful releases land inside the demonstrations' spread on both
backends; the stuck pose is outside it, and further out on CPU. `wrist_2` is dead,
as `DATASET.md` documents.

The two backends' *release* poses agree to 0.016 rad (0.9 deg) - the policy stops
in the same place regardless of engine. The *stuck* poses agree only to 0.051 rad,
because the CPU stuck pose is more extreme; with n=5 and n=8 that is one cluster
being sampled differently, not a second failure mode.

**The stall is an oscillation, and its size scales with the control rate.** Joint
travel over the last 20 s:

| | Failures | Successes |
|---|---|---|
| CUDA | 1.81 - 2.36 rad (mean 2.10) | 0.83 - 2.04 rad (mean 1.51) |
| CPU | 1.05 - 1.22 rad (mean 1.14) | 0.27 - 0.52 rad (mean 0.35) |

On both, the failures move more, and the motion is a coupled
`shoulder_pan` / `wrist_3` sweep of nearly equal amplitude (CPU failures: 0.45 and
0.45 rad of the 1.14 total). But **the separation only holds on CPU**. On CUDA the
ranges overlap - at 17.7 Hz a successful episode also covers 1.5 rad in its last
20 s simply because it is still executing 350 actions in that window. Travel is
not a rate-portable failure signal; the gripper-state detector is.

## Latency and policy behaviour

`REPORT_SMOLVLA.md` found that going from 1.18 s to 2.34 s per query cost **+19%
steps to first grasp** (384 -> 459, Welch t = 5.8) and read it as closed-loop
degradation. This pair does not reproduce it, at a much larger latency ratio:

| Session | Engine | Round trip | Steps to first grasp |
|---|---|---|---|
| G, vla.cpp CUDA | CUDA | 111 ms | **335 +- 26** |
| C, vla.cpp CPU | CPU | 2164 ms | **329 +- 41** |
| | | | Welch t = 0.57, df = 32, n.s. |

For context, all five sessions on this task:

| Session | Server | Round trip | Steps to first grasp | Rate |
|---|---|---|---|---|
| `rollouts_vlacpp_gpu` | vla.cpp CUDA | 111 ms | 335 +- 26 | 17.68 Hz |
| `rollouts_m4` | vla.simd, M4, over LAN | 1177 ms | 384 +- 37 | 9.74 Hz |
| `rollouts_vlacpp_cpu1` | vla.cpp CPU | 2164 ms | 329 +- 41 | 7.11 Hz |
| `rollouts_vlacpp_cpu` | vla.cpp CPU, no FA | 2374 ms | 364 +- 44 | 6.69 Hz |
| `rollouts_ryzen7` | vla.simd, Ryzen, over LAN | 2343 ms | 459 +- 44 | 6.73 Hz |

Steps-to-grasp is not monotone in latency: the two slowest sessions sit at 364 and
459, and the fastest sits at 335, between them. Three sessions at 2.1 - 2.4 s span
329 to 459 steps. Whatever produced the M4 -> Ryzen difference, it is not the
latency alone - cup placement and engine differ too - and the +19% figure should
not be carried forward as a latency law. **Within a clean single-variable
comparison, latency changes the wall clock and not the action count.**

Steps per successful episode tell the same story: 835 on CUDA and 857 on CPU
(1.16x and 1.19x a 721-step demonstration), against 951 and 1073 in the earlier
sessions. Both vla.cpp sessions are closer to the demonstrations than either
vla.simd session, on either backend.

## Infrastructure

| Metric | CUDA | CPU |
|---|---|---|
| Episodes / steps / queries | 20 / 15,681 / 637 | 20 / 15,030 / 606 |
| Recorded wall time | 14.8 min | 35.2 min |
| Blocked on the server | 71 s of 887 s = 8.0% | 1312 s of 2113 s = 62.1% |
| Effective control rate | 17.68 Hz | 7.11 Hz |
| Dropped recording ticks | 0 | 0 |
| Ticks with step period > 100 ms | 4.31% (1 per chunk) | 4.22% (1 per chunk) |
| Slew-limited commands | 0.08% | 0.04% |
| Stop reason | `interrupted` x20 | `interrupted` x20 |

Zero `server_error` stops across 1243 queries. The bridge held for 50 minutes of
continuous serving across two backends with no restart. Step period is 49.99 /
50.00 ms at p50 and 50.44 / 50.39 ms at p95 on the ticks that do not carry a
query, so the 20 Hz timer itself is exact - the entire rate loss is the blocking
query.

## Caveats

- **n=20 per backend, one scene, one operator, one day.** 75% and 60% have Wilson
  intervals of 53-89% and 39-78%. Fisher p = 0.50. The sessions are consistent
  with equal success rates and with a real 15-point gap; neither is established.
- **The CPU session has 19 operator labels for 20 rollouts.** Ep20's label comes
  from the video and the signals, both unambiguous, but it is not the operator's.
- **Stop times are the operator's.** Episodes end on Ctrl+C, so step counts and
  durations partly measure operator patience. "Failures are shorter" is close to
  circular; the stall-then-abort structure is the content.
- **The two sessions ran back to back, CUDA first (08:51-09:13) then CPU
  (09:44-10:23).** Any scene drift over those 90 minutes is confounded with the
  backend. The cup start pose in the contact sheets looks stable, but it was not
  measured.
- **Latency is only compared at two points.** With n=2 backends, "no effect on
  steps-to-grasp" is a failure to reproduce, not a demonstration of no effect.
- **Numerics are not compared here.** `RUN_SMOLVLA.md` covers that: CUDA bf16 is
  6.3e-04 rad from torch fp32, CPU f32 is 1.6e-05. Both are far tighter than the
  torch GPU server this robot has been running, so neither session's behaviour
  should be attributed to engine arithmetic.
- Nothing here measures grasp force or whether the cup was held securely.

## Recommendations

1. **Make vla.cpp/CUDA the default server on this host.** 111 ms per query, 1.1 GB
   VRAM, 17.7 Hz on the robot, stable across a full session. It is 2x faster than
   the torch server for half the VRAM, three orders of magnitude closer to torch
   fp32, and the first configuration that gets near the trained 20 Hz.
2. **Keep `VLA_SMOLVLA_FA=1` in the CPU command.** -8.8% engine time and a 4x
   reduction in run-to-run spread, measured on the robot.
3. **CPU is a fallback, and vla.simd is the better one.** 2.16 s per query against
   vla.simd's ~1.0 s. Use the vla.cpp CPU path when the box has no GPU *and* the
   single-runtime property matters more than 2x latency.
4. **The next latency win on CUDA is the bridge, not the engine.** 21 ms of
   resize-with-pad and tokenization is 19% of the query. The tokenization is
   constant per session and already cacheable (`--tokens`); the two resizes are
   the rest. Moving them into the engine, or into the client where the frames are
   already in memory, would take the query under 95 ms.
5. **Prefetching is now worth it on CUDA and only on CUDA.** At 111 ms of
   inference against 1250 ms of execution per chunk, a background thread would
   remove the dwell entirely and reach the full 20 Hz. On CPU, prefetching alone
   does not help - 2.16 s does not fit in 1.25 s - it needs `--exec-horizon 50`
   too.
6. **Ship the stall detector.** 39/39 on this pair, 29/29 failures and 0 false
   alarms over 79 episodes and four backends. Count the idle in policy steps, not
   seconds, so the threshold survives the higher control rate.
7. **The release is still the bug, and speed does not fix it.** 13 failures across
   both sessions, all at the basket, all with grasp and transport intact, at 17.7
   Hz and at 7.1 Hz alike. The fix is demonstrations that release from a range of
   poses around the basket - the gap `DATASET.md` calls out as "no failures and no
   recoveries" - not a faster server.

## Reproducing these numbers

```bash
python3 - <<'EOF'
import json, numpy as np
from pathlib import Path
for root in ("rollouts_vlacpp_gpu", "rollouts_vlacpp_cpu1"):
    lat, inf, hd, steps, wall = [], [], [], 0, 0.0
    for d in sorted(Path(root).glob("rollout_*")):
        m = json.loads((d/"metadata.json").read_text())
        z = np.load(d/"states.npz")
        lat.append(z["query_latency_ms"]); inf.append(z["query_inference_ms"])
        hd.append(z["query_handling_ms"])
        steps += m["steps"]; wall += m["duration_s"]
        sc, t, g = z["sent_close"], z["t"], z["state_grip"]
        fl = np.where(np.diff(sc.astype(int)) != 0)[0]
        idle = t[-1] - t[fl[-1]]
        print(root, d.name[8:], m["steps"],
              "rt %.0fms" % m["timings_ms"]["server_roundtrip"]["mean"],
              "eng %.0fms" % m["timings_ms"]["server_inference"]["mean"],
              "grasp@%d" % z["step"][fl[0]], "idle %.0fs" % idle,
              "STALLED" if (g[-1] < 0.5 and idle > 10) else "ok")
    lat, inf, hd = map(np.concatenate, (lat, inf, hd))
    print(f"{root}: rt {lat.mean():.1f} (p95 {np.percentile(lat,95):.1f}) "
          f"engine {inf.mean():.1f}  bridge {(hd-inf).mean():.1f}  "
          f"loopback {(lat-hd).mean():.1f}  blocked {100*lat.sum()/1000/wall:.1f}%  "
          f"rate {steps/wall:.2f} Hz\n")
EOF
```

Figures in `figures_vlacpp/` (the rollout directories are owned by another user
and not writable): `end_side_gpu.jpg`, `end_side_cpu.jpg` (5x4 contact sheets of
every last frame, green border = pass) and
`end_cpu_ep18_fail_vs_ep20_pass.jpg`.
