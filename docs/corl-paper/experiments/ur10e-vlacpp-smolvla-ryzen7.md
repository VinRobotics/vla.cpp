# SmolVLA on the UR10e through vla.cpp - Ryzen 7 ultrabook / CPU

One session, 10 rollouts, 2026-08-08 09:17:39 - 09:47:36. Same checkpoint, same
GGUF, same client and same bridge as every other vla.cpp session; the engine is
`vla-server` on the ultrabook's CPU.

| | |
|---|---|
| Engine | `vla-server`, CPU backend, `VLA_SMOLVLA_FA=1 VLA_WEIGHT_DTYPE=f32` |
| Box | Ryzen 7 PRO 6850HS ultrabook, 8 cores / 16 threads, 15 GB, `192.168.56.11` |
| Data | `rollouts_ryzen7_vlacpp/` |
| Labels | operator, 10/10 |
| Topology | client (container, this host) -> bridge + engine (ultrabook), over the wired segment |

**This is the same physical box as session B**, the vla.simd Ryzen session in
`REPORT_SMOLVLA.md` (`rollouts_ryzen7_vlasimd/`, 2026-07-29). Only its IP changed,
`192.168.56.71` -> `192.168.56.11`. That makes this the first pair in the whole
series that swaps the *engine* with the hardware held fixed. Setup and the parity
result are in RUN_SMOLVLA.md, "On the ultrabook (CPU)".

**Ten episodes, not twenty.** Every success-rate statement rests on n = 10. The
latency numbers come from 297 queries and are solid; the success rate is not.

## Summary

**vla.cpp is 1.56x slower than vla.simd on this box, and that is the expected
trade.** 3653 ms per query against 2343 ms, 4.98 Hz against 6.73 Hz. vla.cpp buys
one runtime and one GGUF across CPU, CUDA and Metal at roughly twice the CPU
latency; this session prices that trade on identical hardware instead of across
boxes.

**The package power limit is visible in the robot data, not just on the bench.**
The first query of every episode averages 2551 ms and the rest average 3493 ms -
the first query is **27% faster**, in ten out of ten episodes. This is the
opposite sign from CUDA, where the first query is 38% *slower* (graph warm-up).
Here the box has been idle between episodes, so it starts each one with a full
boost budget and spends it in the first query. Nothing warms up; something runs
out.

**The robot's duty cycle is worth 19%.** Back-to-back on the bench this engine
sustains 4247 ms. In the session, with 1.25 s of arm motion between queries, it
averages 3461 ms. The idle gap partially refills the boost budget every cycle. A
benchmark that hammers the server understates the robot number by a fifth, and one
that fires a single query overstates it by a third.

**Success rate: 7/10 (70%),** the highest of the three CPU sessions, and not
distinguishable from any of them: Fisher p = 0.70 against both 12/20 sessions,
p = 1.00 against CUDA's 15/20. Wilson 95% CI is 40 - 89%. Do not read a quality
gap into it.

**The failure is the documented release failure, unchanged.** All 10 episodes
grasp the cup and carry it to the basket. The 3 failures then hold it outside the
rim and oscillate until the operator stops the run, idle 46 - 56 s against 0.1 -
0.5 s on the passes. Grasp and transport did not fail once.

**The "latency costs steps" law is now dead on its own hardware.** On this box,
vla.simd at 2343 ms took 459 +- 44 steps to first grasp; vla.cpp at 3653 ms takes
**348 +- 26** (Welch t = -8.7, df = 27). The engine got 1.56x slower and the
policy reached the grasp in 111 fewer steps. Whatever produced session B's slow
grasp, it was not the latency. See
[Latency and policy behaviour](#latency-and-policy-behaviour).

## Headline comparison

Session B is `rollouts_ryzen7_vlasimd/` on **this same box**. G and C are from
`UR-SMOLVLA-VLA-CPP.md` (engine on the client's box), M from
`UR-SMOLVLA-VLA-CPP-MAC.md`.

| Metric | R: vla.cpp Ryzen CPU | B: vla.simd, same box | C: vla.cpp host CPU | M: vla.cpp Metal | G: vla.cpp CUDA |
|---|---|---|---|---|---|
| Episodes | 10 | 20 | 20 | 10 | 20 |
| Success | 7/10 (70%) | 12/20 (60%) | 12/20 (60%) | 6/10 (60%) | 15/20 (75%) |
| Server round trip, mean | **3652.9 ms** | 2342.6 ms | 2164.4 ms | 582.0 ms | 111.4 ms |
| Engine inference, mean | 3461.2 ms | n/a | 2142.0 ms | 402.3 ms | 88.3 ms |
| Bridge preprocessing | 29.7 ms | n/a | 20.6 ms | 19.6 ms | 21.1 ms |
| Network | **162.0 ms** | in the 2343 | 1.8 ms (loopback) | 160.1 ms | 2.0 ms (loopback) |
| Round trip p95 / p99 / max | 3817 / 4034 / 4578 ms | | 2301 / 2424 / 2492 ms | 634 / 638 / 651 ms | 126 / 148 / 152 ms |
| Round-trip spread (episode means) | 3605 - 3738 ms (3.6%) | | 2131 - 2211 ms (3.7%) | 576 - 598 ms (3.8%) | 109 - 115 ms (4.9%) |
| Coefficient of variation | 0.056 | | 0.038 | 0.055 | 0.069 |
| Wall clock blocked on the server | **73.2%** | 63.3% | 62.1% | 30.0% | 8.0% |
| Effective control rate | **4.98 Hz** | 6.73 Hz | 7.11 Hz | 12.76 Hz | 17.68 Hz |
| Chunk cycle, p50 | 4900 ms (98 ticks) | | 3400 ms (68 ticks) | 1800 ms (36 ticks) | 1300 ms (26 ticks) |
| Steps to first grasp | **348 +- 26** | 459 +- 44 | 329 +- 41 | 374 +- 44 | 335 +- 26 |
| Steps per successful episode | 786 (1.09x a demo) | | 857 (1.19x) | 782 (1.08x) | 835 (1.16x) |
| Session wall time | 24.7 min | 46.3 min | 35.2 min | 9.5 min | 14.8 min |
| Steps / queries recorded | 7,379 / 297 | 18,726 / 751 | 15,030 / 606 | 7,244 / 293 | 15,681 / 637 |
| Dropped recording ticks | 0 | | 0 | 0 | 0 |
| Slew-limited commands | 0% | | 0.04% | 0% | 0.08% |
| Stop reason | `interrupted` x10 | | `interrupted` x20 | `interrupted` x10 | `interrupted` x20 |

## Latency

### Where the round trip goes

Three measurement points: `latency_ms` by the client (request sent -> chunk
received), `handling_ms` by the bridge (the whole request), `inference_ms` by the
engine. The differences give the network and the bridge's own preprocessing.

| Stage | Mean | Share | What it is |
|---|---|---|---|
| Engine (`vla-server`, CPU f32) | **3461.2 ms** | 94.7% | SigLIP tower + prefill + 10 flow-matching steps |
| Bridge (`handling - inference`) | 29.7 ms | 0.8% | resize-with-pad to 512x512 x2, tokenizer lookup, protobuf, ZMQ on loopback |
| Network (`latency - handling`) | 162.0 ms | 4.4% | 1.8 MB of raw frames out, the pickled chunk back, on 100BASE-TX |
| **Client round trip** | **3652.9 ms** | | |

The engine is 95% of the query here, against 79% on CUDA. Everything outside it is
noise at this speed: the same 162 ms of cable that costs the Mac 27% of its query
costs this box 4%.

### The network term

162.0 ms mean, p95 162.6 ms, max 178.2 ms over 297 queries - the same wire, and
the same stability, the Mac session measured (160.1 ms). Both ends of this segment
report 100BASE-TX. A gigabit switch would remove ~145 ms of it, which here is 4%
of the query and not worth doing for this path alone. It is worth doing for the
Metal path, where the same 145 ms is a quarter of the query.

### Distribution

| | Round trip | Engine |
|---|---|---|
| min | 2696.3 | 2508.4 |
| p50 | 3683.5 | 3492.2 |
| p95 | 3817.1 | 3623.1 |
| p99 | 4033.5 | |
| max | 4578.0 | 4380.5 |
| mean | 3652.9 | 3461.2 |
| sd | 205.9 | |
| CV | 0.056 | |

All in ms, 297 queries. The mins are the per-episode first queries, not outliers -
see below.

### The power limit, measured on the robot

**The first query of each episode is the fastest one in it, every time:**

| | Engine | Round trip |
|---|---|---|
| First query of an episode (n = 10) | **2551 ms** | 2746 ms |
| Every other query (n = 287) | **3493 ms** | 3684 ms |
| Difference | **-27%** | -25% |

The per-episode firsts span 2508 - 2606 ms, a 4% spread across ten episodes. This
is not noise and it is not a cache: it is the 28 W package starting each episode
with an unspent boost budget after ~20 s of operator idle between runs.

Within an episode it decays over the first three or four queries and then flattens:

| Query # in episode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Engine, mean ms | 2551 | 3349 | 3423 | 3459 | 3460 | 3525 | 3622 | 3478 |

**This is the opposite sign from CUDA.** There the first query of an episode is
38% *slower* (120.5 ms against 87.3 ms) because the graph and kernels re-warm
after an idle gap. Here idle makes the box *faster*. Two different machines, two
opposite first-query artifacts, and on this one it is worth 0.9 s.

**No session-level drift.** Episode-mean engine time runs 3414 - 3540 ms with no
trend (ep 1 is the highest at 3540, ep 10 is 3437), and the package sits at
64 - 68 C under load against a ~95 C Tjmax. The limit is power, not heat, and it
is reached within one episode rather than over the session.

### Bench against robot

The bench numbers in RUN_SMOLVLA.md were taken back to back, each query issued the
moment the last chunk arrived. The robot leaves 1.25 s of arm motion between them:

| Duty cycle | Engine | Round trip |
|---|---|---|
| Back to back (bench, 34 queries) | 4247 ms | 4444 ms |
| Robot, `--exec-horizon 25` (297 queries) | **3461 ms** | **3653 ms** |
| Single query into an idle box | 2551 ms | 2746 ms |

A 1.7x spread on the same binary, the same GGUF and the same box, decided entirely
by how much idle time sits between queries. **Quote the duty-cycle-matched
number.** Neither the hammered bench nor the single shot describes the robot.

### What the client does with it

The client ticks at 20 Hz, executes `exec_horizon = 25` actions from each chunk,
then issues the next query and blocks on it:

| | Measured |
|---|---|
| Execute 25 actions | 1250 ms |
| Blocked on the query | 3653 ms |
| Chunk cycle, p50 | **4900 ms** = 98 ticks |
| Chunk cycle, mean | 4939 ms |
| Dwell fraction | 75% of the cycle |
| Predicted rate | 5.10 Hz |
| Measured session rate | **4.98 Hz** |

Three ticks of standing still for every one of motion. This is the slowest of the
five backends and it looks it: the arm moves for 1.25 s and then holds for 3.7 s.

### Observation staleness

The frame the policy actually sees is ~17 ms old (`img_age` p50), the same as
every other session - the cameras run at ~30 Hz and the sampling is unaffected by
the backend. What differs is how stale it is by the time the actions derived from
it reach the arm:

| | This session | CUDA |
|---|---|---|
| Age at the 1st action of the chunk | 3670 ms | 128 ms |
| Age at the 25th action | 4920 ms | 1378 ms |

The whole CUDA chunk is acted on ~2.7x fresher than this session's *freshest*
action.

## Success rate and labels

Operator labels, in order:

```
P F P P P P P F P F
```

7 passes (ep 1, 3, 4, 5, 6, 7, 9), 3 failures (ep 2, 8, 10). **7/10 = 70%,**
Wilson 95% CI 40 - 89%.

| Session | Engine | RT | Success | Wilson 95% CI |
|---|---|---|---|---|
| R (this) | vla.cpp CPU, Ryzen 7 | 3653 ms | 7/10 (70%) | 40-89% |
| B | vla.simd, **same box** | 2343 ms | 12/20 (60%) | 39-78% |
| C | vla.cpp CPU, host | 2164 ms | 12/20 (60%) | 39-78% |
| M | vla.cpp Metal, M4 | 582 ms | 6/10 (60%) | 31-83% |
| G | vla.cpp CUDA | 111 ms | 15/20 (75%) | 53-89% |
| A | vla.simd, M4 | 1177 ms | 12/20 (60%) | 39-78% |

| Comparison | Fisher exact |
|---|---|
| R vs B (vla.simd, same box) | p = 0.70 |
| R vs C (host CPU) | p = 0.70 |
| R vs G (CUDA) | p = 1.00 |
| R vs M (Metal) | p = 1.00 |

70% is the best number any CPU session has produced and it means nothing on its
own: with 10 episodes the interval runs from 40% to 89%, and it overlaps every
other session in the series. Six sessions spanning 111 ms to 3653 ms per query
have now all landed between 60% and 75%.

## Per-episode results

`1st grasp` = step of the first gripper CLOSE. `Last grip` = time of the last
gripper command. `Idle` = seconds from that command to the operator's stop.
`Drift` = median over the episode of `max|action - measured joints|`.

| Ep | Result | Rollout | Steps | Wall s | Q | RT ms | Engine ms | Net ms | 1st grasp | Last grip s | Idle s | Drift rad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pass | `091739` | 800 | 162.9 | 32 | 3738.0 | 3540.4 | 162.5 | 384 | 160.0 | 0.1 | 0.0045 |
| 2 | **fail** | `092042` | 600 | 119.7 | 24 | 3650.1 | 3455.3 | 161.9 | 362 | 72.0 | 46.2 | 0.0063 |
| 3 | pass | `092315` | 700 | 141.1 | 28 | 3641.1 | 3450.8 | 161.8 | 307 | 138.1 | 0.4 | 0.0063 |
| 4 | pass | `092618` | 775 | 155.0 | 31 | 3617.3 | 3426.6 | 161.9 | 331 | 150.8 | 0.5 | 0.0053 |
| 5 | pass | `092921` | 832 | 167.8 | 34 | 3604.9 | 3414.4 | 161.9 | 321 | 167.5 | 0.2 | 0.0046 |
| 6 | pass | `093240` | 775 | 155.8 | 31 | 3688.0 | 3497.3 | 161.9 | 339 | 153.0 | 0.4 | 0.0047 |
| 7 | pass | `093553` | 761 | 151.7 | 31 | 3641.9 | 3451.5 | 162.0 | 364 | 151.0 | 0.4 | 0.0053 |
| 8 | **fail** | `093853` | 650 | 130.0 | 26 | 3668.9 | 3478.1 | 161.9 | 384 | 82.4 | 46.9 | 0.0061 |
| 9 | pass | `094135` | 861 | 173.2 | 35 | 3648.9 | 3458.3 | 162.0 | 343 | 172.5 | 0.5 | 0.0047 |
| 10 | **fail** | `094530` | 625 | 125.7 | 25 | 3627.9 | 3437.1 | 162.1 | 340 | 66.9 | 55.6 | 0.0066 |

Drift is 0.0045 - 0.0066 rad, the same band as every other session, so the
controller is tracking the policy exactly as well here as it does at 17.7 Hz.

## The failure mode

Unchanged: transport succeeds, release does not. In all 3 failures the last
gripper command is a CLOSE issued long before the operator stopped, the measured
gripper reads 0.000, and the last frame shows the cup still clamped next to the
basket.

**The stall detector agrees with the operator on 10 of 10.** Rule: *measured
gripper closed at the end AND no gripper command for more than 10 s*. Its margin
here is the widest in the series - failures idle 46.2, 46.9 and 55.6 s while
passes idle 0.1 - 0.5 s, so the 10 s threshold has two orders of magnitude of
headroom on both sides. Across six sessions and 89 episodes it now stands at
**32/32 failures caught, 0 false alarms**.

**Joint travel separates cleanly again.** Over the last 20 s:

| | Failures | Successes |
|---|---|---|
| This session | 0.64 - 0.77 rad (mean 0.70) | 0.21 - 0.40 rad (mean 0.29) |
| Host CPU (C) | 1.05 - 1.22 rad (mean 1.14) | 0.27 - 0.52 rad (mean 0.35) |
| CUDA (G) | 1.81 - 2.36 rad (mean 2.10) | 0.83 - 2.04 rad (mean 1.51) |

The separation holds on both slow backends and collapses on CUDA, which is the
rate artifact `UR-SMOLVLA-VLA-CPP.md` called out: at 17.7 Hz a *successful*
episode also covers a lot of ground in its last 20 s, because it is still
executing 350 actions in that window. At 5 Hz it is executing 100. Travel is not a
rate-portable signal; the gripper-state detector is.

**End pose against the 44 training demonstrations.** Release pose = last frame of
a success, stuck pose = last frame of a failure. Training statistics quoted from
`UR-SMOLVLA-VLA-CPP.md`:

| Joint | Train release | sd | Release / z | Stuck / z |
|---|---|---|---|---|
| shoulder_pan | -1.0391 | 0.019 | -1.0624 / -1.2 | -1.0760 / -1.9 |
| shoulder_lift | -1.9178 | 0.026 | -1.8654 / +2.0 | -1.8460 / **+2.8** |
| elbow | -1.5323 | 0.031 | -1.5816 / -1.6 | -1.5300 / +0.1 |
| wrist_1 | -1.2655 | 0.025 | -1.2689 / -0.1 | -1.3405 / **-3.0** |
| wrist_2 | +1.5713 | 0.0001 | +1.5712 | +1.5713 |
| wrist_3 | -2.6082 | 0.019 | -2.6336 / -1.3 | -2.6438 / -1.9 |

RMS z over the five live joints: **release 1.41, stuck 2.19**. Successful releases
land inside the demonstrations' spread; the stuck pose sits outside it, driven by
`wrist_1` at z = -3.0. The two poses agree to 0.072 rad, so this is one cluster
sampled two ways, not a second failure mode. `wrist_2` is dead, as `DATASET.md`
documents.

## Latency and policy behaviour

This session settles a question the earlier reports left open.

`REPORT_SMOLVLA.md` compared the M4 (1.18 s per query, 384 +- 37 steps to grasp)
against this same Ryzen box running vla.simd (2.34 s, 459 +- 44) and read the
+19% as closed-loop degradation from latency. `UR-SMOLVLA-VLA-CPP.md` then failed
to reproduce it across CUDA and the host CPU and warned the figure should not be
carried forward. This session tests it **on the box that produced it**:

| Session | Box | Engine | Round trip | Steps to first grasp |
|---|---|---|---|---|
| B | Ryzen 7 6850HS | vla.simd | 2343 ms | 459 +- 44 |
| R (this) | **same box** | vla.cpp CPU | **3653 ms** | **348 +- 26** |
| | | | | Welch t = -8.7, df = 27 |

**The engine got 1.56x slower and the policy reached the grasp 111 steps
earlier.** With the hardware, the cell, the client and the checkpoint all held
fixed, latency moved the wrong way for the hypothesis and the effect is large and
significant. The +19% law does not survive.

All six sessions, ordered by round trip:

| Session | Server | Round trip | Steps to first grasp | Rate |
|---|---|---|---|---|
| `rollouts_vlacpp_gpu` | vla.cpp CUDA | 111 ms | 335 +- 26 | 17.68 Hz |
| `rollouts_m4_vlacpp1` | vla.cpp Metal, M4 | 582 ms | 374 +- 44 | 12.76 Hz |
| `rollouts_m4_vlasimd` | vla.simd, M4 | 1177 ms | 384 +- 37 | 9.74 Hz |
| `rollouts_vlacpp_cpu1` | vla.cpp CPU, host | 2164 ms | 329 +- 41 | 7.11 Hz |
| `rollouts_ryzen7_vlasimd` | vla.simd, Ryzen | 2343 ms | 459 +- 44 | 6.73 Hz |
| `rollouts_ryzen7_vlacpp` | vla.cpp CPU, Ryzen | **3653 ms** | **348 +- 26** | 4.98 Hz |

Steps-to-grasp spans 329 to 459 with no relationship to latency, and the slowest
session in the series sits in the middle of the range. Against this session,
CUDA (t = +1.3) and the host CPU (t = +1.5) are indistinguishable across a 33x
latency ratio; only the two vla.simd sessions are slower to grasp.

What that leaves is an engine-shaped residual: the two highest steps-to-grasp
numbers are the two vla.simd sessions, and the four vla.cpp sessions sit at
329 - 374 regardless of backend or speed. That is consistent with the SigLIP
position-id gotcha `VLACPP.md` documents, which the vla.simd path hit by a
different mechanism and which shifts actions by ~2e-02 rad while still looking
plausible. **This session does not establish that** - the two runs are 10 days
apart with no scene measurement between them - but it is the hypothesis worth
testing next, and it is cheap to test: re-run session B's conversion through the
current parity check.

Steps per successful episode tell the same story: 786 here (1.09x a 721-step
demonstration), the second-tightest in the series behind Metal's 782, against
857 on the host CPU and 835 on CUDA.

## Infrastructure

| Metric | Value |
|---|---|
| Episodes / steps / queries | 10 / 7,379 / 297 |
| Recorded wall time | 24.7 min |
| Blocked on the server | 1085 s of 1483 s = 73.2% |
| Effective control rate | 4.98 Hz |
| Dropped recording ticks | 0 |
| Step period, non-query ticks | p50 50.00 ms, p95 50.21 ms |
| Ticks with step period > 100 ms | 4.08% (one per chunk) |
| Slew-limited commands | 0 |
| Server errors | 0 |
| Stop reason | `interrupted` x10 |

Zero errors across 297 queries. The engine and the bridge ran for 30 minutes of
continuous serving with no restart, on a laptop, over the wired segment. The 20 Hz
timer itself is exact - 50.00 ms at p50 on ticks that do not carry a query - so
the entire rate loss is the blocking query.

## Caveats

- **n = 10, one scene, one operator, one day.** 70% has a Wilson interval of
  40 - 89% and is consistent with every other session in the series. Nothing here
  ranks backends by success.
- **The comparison against session B is 10 days apart.** Same box and same
  checkpoint, but the cup start pose was not measured on either day, and
  `REPORT_SMOLVLA.md` already flags B's labels as inferred rather than the
  operator's. The steps-to-grasp gap is large enough (t = -8.7) that scene drift
  is an unlikely full explanation, but it is not excluded.
- **Stop times are the operator's.** Episodes end on Ctrl+C, so step counts and
  durations partly measure operator patience. "Failures are shorter" is close to
  circular; the stall-then-abort structure is the content.
- **The power-limit numbers depend on the duty cycle,** which depends on
  `--exec-horizon` and on how long the operator waits between episodes. At
  `--exec-horizon 50` the gaps are longer and the engine should sit somewhere
  between the 3461 ms measured here and the 2551 ms single-shot figure. That was
  not measured.
- **The bench numbers in the duty-cycle table are mine, not the session's,** taken
  the same day on the same binary and GGUF through the same bridge, but with a
  synthetic client on real frames rather than the robot.
- **Numerics are not compared here.** The parity result for this box is in
  RUN_SMOLVLA.md: 5.8e-05 rad against torch fp32 with the f32 towers this session
  ran, which is tighter than any other spot in the series.
- Nothing here measures grasp force or whether the cup was held securely.

## Recommendations

1. **Use `--exec-horizon 50` on this box.** At 25 the arm dwells 75% of the
   session; at 50 the same query cost is amortised over 2.5 s of motion, which
   takes the rate from 5.0 Hz to ~7.2 Hz for free. This is the one configuration
   change that matters here.
2. **Keep vla.simd as the fast CPU path on this box.** 2343 ms against 3653 ms on
   identical hardware. Reach for vla.cpp here when the single-runtime, single-GGUF
   property is worth 1.56x, which is a real reason - it is the same binary and the
   same file that run on CUDA and Metal.
3. **Quote duty-cycle-matched latency for power-limited boxes.** The same engine
   measures 2551, 3461 or 4247 ms depending only on the gap between queries. Bench
   this class of machine the way the robot will drive it, or the number is
   fiction.
4. **Re-test session B's conversion.** The four vla.cpp sessions cluster at
   329 - 374 steps to grasp and the two vla.simd sessions sit at 384 and 459. Run
   the current parity check against the vla.simd Ryzen setup before attributing
   any of that to the policy.
5. **Ship the stall detector.** 10/10 here, 32/32 failures with 0 false alarms
   over 89 episodes and five backends. Count the idle in policy steps rather than
   seconds so the threshold survives across a 3.5x range of control rates.
6. **The release is still the bug.** Three failures, all at the basket, all with
   grasp and transport intact - now demonstrated at 4.98 Hz as well as at 7.1 and
   17.7 Hz. The fix is demonstrations that release from a range of poses around
   the basket, the gap `DATASET.md` calls out, not a faster server.

## Reproducing these numbers

```bash
python3 - <<'EOF'
import json, numpy as np
from pathlib import Path
root, lab = "rollouts_ryzen7_vlacpp", list("PFPPPPPFPF")
lat, inf, hd, steps, wall, first = [], [], [], 0, 0.0, []
for d, l in zip(sorted(Path(root).glob("rollout_*")), lab):
    m = json.loads((d/"metadata.json").read_text())
    z = np.load(d/"states.npz")
    lat.append(z["query_latency_ms"]); inf.append(z["query_inference_ms"])
    hd.append(z["query_handling_ms"]); first.append(z["query_inference_ms"][0])
    steps += m["steps"]; wall += m["duration_s"]
    sc, t, g = z["sent_close"], z["t"], z["state_grip"]
    fl = np.where(np.diff(sc.astype(int)) != 0)[0]
    idle = t[-1] - t[fl[-1]]
    print(root, d.name[8:], l, m["steps"],
          "rt %.0fms" % m["timings_ms"]["server_roundtrip"]["mean"],
          "eng %.0fms" % m["timings_ms"]["server_inference"]["mean"],
          "grasp@%d" % z["step"][fl[0]], "idle %.0fs" % idle,
          "STALLED" if (g[-1] < 0.5 and idle > 10) else "ok")
rest = np.concatenate([a[1:] for a in inf])
lat, inf, hd = map(np.concatenate, (lat, inf, hd))
print(f"{root}: rt {lat.mean():.1f} (p95 {np.percentile(lat,95):.1f}) "
      f"engine {inf.mean():.1f}  bridge {(hd-inf).mean():.1f}  "
      f"net {(lat-hd).mean():.1f}  blocked {100*lat.sum()/1000/wall:.1f}%  "
      f"rate {steps/wall:.2f} Hz")
print(f"first query of an episode {np.mean(first):.0f} ms vs rest {rest.mean():.0f} ms "
      f"({100*(np.mean(first)-rest.mean())/rest.mean():+.0f}%)")
EOF
```
