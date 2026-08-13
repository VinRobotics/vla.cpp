# SmolVLA on the UR10e through vla.cpp - Apple M4 / Metal

One session, 10 rollouts, 2026-08-08 07:39:56 - 07:54:21. Same checkpoint, same
GGUF, same client and same bridge as the CUDA and CPU sessions in
`UR-SMOLVLA-VLA-CPP.md`; the engine is `vla-server` on the Mac Mini M4's GPU.

| | |
|---|---|
| Engine | `vla-server`, Metal backend, `VLA_WEIGHT_DTYPE=f32` |
| Box | Mac Mini M4, 10 cores, 24 GB unified, `192.168.56.91` |
| Data | `rollouts_m4_vlacpp1/` |
| Labels | operator, 10/10 |
| Topology | client (container, this host) -> bridge + engine (Mac), over the wired segment |

Unlike sessions G and C, the server is **not** on the client's box: every
`metadata.json` says `server: 192.168.56.91:8791`, so the LAN is in the loop.
Setup and the Metal parity result are in RUN_SMOLVLA.md,
"On the Mac Mini (Metal)". Every number below comes from the `states.npz` /
`metadata.json` files and the recorded video.

**Ten episodes, not twenty.** Every success-rate statement here rests on n = 10,
which is half the sample of the sessions it is compared against. The latency
numbers come from 293 queries and are solid; the success rate is not.

## Summary

**Metal makes the Mac Mini twice the policy server vla.simd made it.** 582 ms per
query against 1177 ms for vla.simd on the same box, and 402 ms of that is the
engine. Dwell between chunks halves, the control rate goes from 9.7 Hz to
12.8 Hz, and the arm spends 30% of the session waiting instead of 46%.

**A quarter of the round trip is cable, not compute.** 160.1 ms of the 582.0 ms is
the network, and it is the most predictable number in the session: p95 161.7 ms,
max 162.9 ms over 293 queries. That is two raw 640x480 frames, 1.8 MB, on a
**100BASE-TX** segment. On gigabit the same query lands at ~440 ms. The engine is
now small enough that the wire is the second-biggest term in the loop.

**Success rate: 6/10 (60%),** the same as vla.simd on this box, the same as
vla.cpp on the host CPU, and the same as the Ryzen. Against the CUDA session's
75%, Fisher p = 0.43. Nothing here separates Metal from any other way of running
this policy.

**The failure is the documented release failure, unchanged.** All 10 episodes
grasp the cup and carry it to the basket. The 4 failures then hold it just outside
the rim and oscillate there until the operator stops the run, idle 16.9 - 21.4 s.
Grasp and transport did not fail once.

**Latency still does not predict success.** This session is 5.2x slower than the
CUDA one and scores within noise of it; steps to first grasp is 374 +- 44 against
CUDA's 335 +- 26. Four backends spanning 111 ms to 2343 ms per query have now all
landed between 60% and 75%.

## Headline comparison

Sessions G and C are from `UR-SMOLVLA-VLA-CPP.md` (same host, engine on the
client's box). Session A is `rollouts_m4/`, vla.simd on **this same Mac**, from
`REPORT_SMOLVLA.md`.

| Metric | M: vla.cpp Metal | G: vla.cpp CUDA | C: vla.cpp CPU | A: vla.simd M4 |
|---|---|---|---|---|
| Episodes | 10 | 20 | 20 | 20 |
| Success | 6/10 (60%) | 15/20 (75%) | 12/20 (60%) | 12/20 (60%) |
| Server round trip, mean | 582.0 ms | 111.4 ms | 2164.4 ms | 1176.6 ms |
| Engine inference, mean | 402.3 ms | 88.3 ms | 2142.0 ms | n/a |
| Bridge preprocessing | 19.6 ms | 21.1 ms | 20.6 ms | n/a |
| Network | **160.1 ms** | 2.0 ms (loopback) | 1.8 ms (loopback) | in the 1177 |
| Round trip p95 / p99 / max | 634 / 638 / 651 ms | 126 / 148 / 152 ms | 2301 / 2424 / 2492 ms | |
| Round-trip spread (episode means) | 575.7 - 597.8 ms (3.8%) | 109.4 - 114.8 ms (4.9%) | 2131 - 2211 ms (3.7%) | 6% |
| Coefficient of variation | 0.055 | 0.069 | 0.038 | |
| Wall clock blocked on the server | 30.0% | 8.0% | 62.1% | 46.1% |
| Effective control rate | 12.76 Hz | 17.68 Hz | 7.11 Hz | 9.74 Hz |
| Chunk cycle, p50 | 1800 ms | 1300 ms | 3400 ms | |
| Steps to first grasp | 374 +- 44 | 335 +- 26 | 329 +- 41 | 384 +- 37 |
| Steps per successful episode | 782 (1.08x a demo) | 835 (1.16x) | 857 (1.19x) | 951 (1.32x) |
| Session wall time | 9.5 min | 14.8 min | 35.2 min | 28.3 min |
| Steps / queries recorded | 7,244 / 293 | 15,681 / 637 | 15,030 / 606 | |
| Dropped recording ticks | 0 | 0 | 0 | |
| Slew-limited commands | 0 | 0.08% | 0.04% | |
| Stop reason | `interrupted` x10 | `interrupted` x20 | `interrupted` x20 | |

## Latency

### Where the round trip goes

Three measurement points: `latency_ms` by the client (request sent -> chunk
received), `handling_ms` by the bridge (the whole request), `inference_ms` by the
engine. The differences give the network and the bridge's own preprocessing.

| Stage | Mean | What it is |
|---|---|---|
| Engine (`vla-server`, Metal) | 402.3 ms | vision tower + prefill + denoise on the M4 GPU |
| Bridge (`handling - inference`) | 19.6 ms | resize-with-pad to 512x512 x2, tokenizer lookup, protobuf, ZMQ on loopback |
| Network (`latency - handling`) | 160.1 ms | 1.8 MB of raw frames out, the pickled chunk back, on 100BASE-TX |
| **Client round trip** | **582.0 ms** | |

### The network term

| | Value |
|---|---|
| Mean | 160.1 ms |
| p95 / max | 161.7 / 162.9 ms |
| Spread over 293 queries | ~3 ms |

This is a wire, not a queue: 1.8 MB of uncompressed frames at 100 Mbit is ~150 ms
of serialisation, and nothing in the session perturbs it. Both ends report
100BASE-TX (`ifconfig en0 | grep media` on the Mac,
`cat /sys/class/net/enp2s0/speed` here). A gigabit switch and cable removes ~145 ms
of it, taking this configuration from 582 ms to ~440 ms and the control rate from
12.8 Hz to about 15 Hz - no code, and it helps the vla.simd path by exactly the
same amount.

### Distribution

| | Value |
|---|---|
| Round trip mean / p50 | 582.0 / 559.1 ms |
| p95 / p99 / max | 633.9 / 638.0 / 651.1 ms |
| min | 549.8 ms |
| Coefficient of variation | 0.055 |
| Engine mean / p50 / p95 / max | 402.3 / 388.4 / 439.9 / 462.8 ms |

Stable across the session: episode-mean round trip stays inside
575.7 - 597.8 ms (3.8%) and episode-mean engine drifts 399.1 -> 414.7 ms, a ~4%
rise over 9.5 minutes that is consistent with the fanless M4 warming up. No
thermal cliff, no dropped ticks, no slew-limited commands and no server errors
across 293 queries.

### What the client does with it

At `--exec-horizon 25` the client executes 25 actions (1.25 s at 20 Hz) and then
blocks on the next chunk. Measured chunk cycle is 1800 ms p50: 1250 ms of motion
plus ~0.58 s of dwell. That is 30.0% of the session spent waiting, against 8.0% on
CUDA and 46.1% for vla.simd on this same Mac.

The arm visibly pauses between chunks, at half the vla.simd pause. It is the best
this cell does without the workstation GPU.

## Success rate and labels

Operator labels, in order:

```
P F F P P F P P F P
```

6 passes (ep 1, 4, 5, 7, 8, 10), 4 failures (ep 2, 3, 6, 9). **6/10 = 60%**,
Wilson 95% CI 31 - 83%.

| Session | Engine | RT | Success | Wilson 95% CI |
|---|---|---|---|---|
| M (this) | vla.cpp Metal, M4 | 582 ms | 6/10 (60%) | 31-83% |
| G | vla.cpp CUDA | 111 ms | 15/20 (75%) | 53-89% |
| C | vla.cpp CPU | 2164 ms | 12/20 (60%) | 39-78% |
| A | vla.simd, same M4 | 1177 ms | 12/20 (60%) | 39-78% |
| B | vla.simd, Ryzen 7 | 2343 ms | 12/20 (60%) | 39-78% |

| Comparison | Fisher exact |
|---|---|
| M vs G (CUDA) | p = 0.43 |
| M vs C (host CPU) | p = 1.00 |
| M vs A (vla.simd on this same Mac) | p = 1.00 |

Metal lands exactly on the 60% that three of the four earlier sessions produced.
The CI is wide - 31 to 83% on ten episodes - so this is evidence of *no
difference*, not evidence of equality; it would take a much longer session to
separate 60% from 75%.

## Per-episode results

`1st grasp` = step of the first gripper CLOSE. `Last grip` = time of the last
gripper command. `Idle` = seconds from that command to the operator's stop.
`Drift` = median over the episode of `max|action - measured joints|`.

| Ep | Result | Rollout | Steps | Wall s | Q | RT ms | Engine ms | Net ms | 1st grasp | Last grip s | Idle s | Drift rad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pass | `073956` | 675 | 53.6 | 27 | 575.7 | 399.1 | 160.6 | 320 | 51.6 | 1.6 | 0.0058 |
| 2 | **fail** | `074209` | 590 | 46.0 | 24 | 580.3 | 401.7 | 160.3 | 372 | 29.1 | 16.9 | 0.0064 |
| 3 | **fail** | `074325` | 589 | 46.0 | 24 | 582.2 | 402.1 | 159.9 | 309 | 26.1 | 19.8 | 0.0069 |
| 4 | pass | `074452` | 752 | 59.4 | 31 | 580.6 | 400.8 | 160.2 | 411 | 57.0 | 2.3 | 0.0053 |
| 5 | pass | `074624` | 850 | 64.9 | 34 | 579.4 | 400.7 | 157.1 | 387 | 64.6 | 0.3 | 0.0048 |
| 6 | **fail** | `074813` | 725 | 59.8 | 29 | 580.8 | 400.2 | 160.6 | 449 | 39.5 | 20.0 | 0.0055 |
| 7 | pass | `074941` | 758 | 59.4 | 31 | 584.4 | 404.4 | 160.6 | 365 | 59.3 | 0.1 | 0.0063 |
| 8 | pass | `075115` | 872 | 68.2 | 35 | 577.8 | 398.5 | 160.5 | 420 | 64.2 | 3.9 | 0.0046 |
| 9 | **fail** | `075300` | 646 | 50.1 | 26 | 579.2 | 399.9 | 160.5 | 348 | 28.7 | 21.4 | 0.0060 |
| 10 | pass | `075421` | 787 | 60.3 | 32 | 597.8 | 414.7 | 160.8 | 355 | 59.3 | 0.5 | 0.0053 |


## The failure mode

Contact sheet of every last frame, green = pass:
`figures_vlacpp/end_side_m4_metal.jpg`. Fail and pass side by side at full
resolution: `figures_vlacpp/end_m4_metal_ep09_fail_vs_ep10_pass.jpg`.

**Grasp and transport never failed.** All 10 episodes closed the gripper on the
cup - first close at step 374 +- 44, about 19 s of policy time - and carried it to
the basket. Nothing failed before the release, which is what every earlier session
on this robot found.

**All 4 failures are the same release stall.** The arm arrives holding the cup,
hovers at or just outside the rim, and oscillates there until the operator stops
the run. The telemetry signature:

| Signal | Failures (4) | Successes (6) |
|---|---|---|
| Idle after the last gripper command | 16.9 - 21.4 s | 0.1 - 3.9 s |
| Gripper state at the end | closed, all 4 | open, or closing back within ~1 s of a release |
| Steps per episode | 638 +- 64 | 782 +- 72 |

The detector from `REPORT_SMOLVLA.md` - gripper closed at the end **and** idle
> 10 s - fires on all 4 failures with 0 false alarms on the 6 successes. The idle
gap is the cleanest separator in the data: 16.9 s at the low end of the failures
against 3.9 s at the high end of the successes, with nothing in between.

**Drift does not separate them this time.** Successes span 0.0046 - 0.0063 rad and
failures 0.0055 - 0.0069, and ep7 (pass, 0.0063) sits above ep6 (fail, 0.0055).
`REPORT_SMOLVLA.md` already flagged drift as a signal that looks like a perfect
classifier on one session and stops being one on the next; this session is another
instance. Use the idle gap.

## Latency and policy behaviour

Two things are worth reading off the comparison table.

**Steps to first grasp does not track latency.** 374 +- 44 here at 582 ms per
query, against 335 +- 26 at 111 ms and 329 +- 41 at 2164 ms. The 2.3x spread in
approach length across sessions has no relationship to the 19x spread in server
latency.

**Successful episodes are the shortest recorded on this robot** - 782 steps, 1.08x
the 721-step average demo, against 1.16x on CUDA, 1.19x on the host CPU and 1.32x
for vla.simd on this same Mac. On ten episodes that is 6 successes' worth of
evidence and could easily be scene luck, so it is an observation, not a claim.

What the latency does buy is the dwell: 0.58 s between chunks instead of vla.simd's
1.18 s on the same hardware. At `--exec-horizon 25` that is 0.58 s of pause per
1.25 s of motion - visible, but far from the vla.simd stop-and-go.

## Infrastructure

- The Mac is offline - one interface on the robot segment, no default route - so
  the engine was built there from a source tree, a pinned llama.cpp and a set of
  wheels staged over rsync by `scripts/deploy_vlacpp_mac.sh`. Details in
  RUN_SMOLVLA.md, "On the Mac Mini (Metal)".
- `vla-server` ran the Metal backend with `VLA_WEIGHT_DTYPE=f32`: 1447.5 MiB of
  unified memory for the towers, against 775.6 MiB at the bf16 default. f32 is
  required on this backend - bf16 fails the parity check at 2.0e-03 rad on the
  joints where f32 gives 8.3e-05.
- `metadata.json` does not record the dtype, but the timings confirm it: the
  minimum engine time over 293 queries is 379.9 ms and p05 is 382.2 ms, against an
  idle-box benchmark of 363 ms for bf16 and 388 - 396 ms for f32. The session never
  approached the bf16 floor.
- The bridge ran with `--tokens 18188,614,260,7118,198`, which pins the
  instruction and keeps transformers off the Mac. Every `metadata.json` handshake
  confirms `task: "pick up the cup"`.
- The client was unchanged: `--server 192.168.56.91:8791`, `--exec-horizon 25`.
- Image age at request time is p50 17.1 ms, the same as every other session: the
  160 ms of network happens *after* the frame is grabbed, so it does not make the
  observation the policy sees any staler.

## Caveats

- **n = 10.** Wilson 95% CI on 6/10 is 31 - 83%. This session can tell you Metal is
  not obviously broken; it cannot rank Metal against CUDA.
- **Labels are the operator's**, applied live during the run.
- **The engine dtype is inferred, not recorded.** The timing argument is strong but
  indirect. Worth passing the engine's backend and dtype through the bridge
  handshake into `metadata.json`.
- **Session A (vla.simd on this Mac) is from 2026-07-29**, so the same-box
  comparison spans a scene reset and ten days.
- The network term is specific to this cabling. Re-measure after any switch or
  cable change: `latency_ms - handling_ms` in any rollout gives it directly.

## Recommendations

1. **Put the segment on gigabit.** 145 ms of the 582 ms round trip is
   serialisation on a 100 Mbit link. It costs a cable and a switch port, needs no
   code, and helps the vla.simd path identically.
2. **Keep `VLA_WEIGHT_DTYPE=f32` on Metal.** It costs 33 ms and 672 MiB and buys
   24x on joint parity; the bf16 default fails the check. See RUN_SMOLVLA.md.
3. **Run 20 episodes next time**, and interleave them with a CUDA block in the
   same sitting if the goal is to compare backends rather than to check one works.
4. **Record the backend and dtype in `metadata.json`.** The bridge already knows
   both from the engine banner.
5. **The release is still the whole problem.** Four backends spanning 111 ms to
   2343 ms per query, and every failure in every session is the release. Grasp and
   transport are solved; no inference hardware has moved the number, and none
   will.

## Reproducing these numbers

```bash
python3 - <<'EOF'
import json, numpy as np
from pathlib import Path
LAB = "P F F P P F P P F P".split()
lat=[];hd=[];inf=[];steps=0;wall=0.0
for d,l in zip(sorted(Path("rollouts_m4_vlacpp1").glob("rollout_*")), LAB):
    m=json.loads((d/"metadata.json").read_text()); z=np.load(d/"states.npz")
    lat.append(z["query_latency_ms"]); hd.append(z["query_handling_ms"])
    inf.append(z["query_inference_ms"])
    steps+=m["steps"]; wall+=m["duration_s"]
    fl=np.where(np.diff(z["sent_close"].astype(int))!=0)[0]
    idle=z["t"][-1]-z["t"][fl[-1]]
    print(d.name[-6:], l, m["steps"], "rt %.0f" % m["timings_ms"]["server_roundtrip"]["mean"],
          "grasp@%d" % z["step"][fl[0]], "idle %.0fs" % idle,
          "STALL" if (z["state_grip"][-1] < 0.5 and idle > 10) else "")
lat,hd,inf = map(np.concatenate,(lat,hd,inf))
print(f"n={len(lat)} rt {lat.mean():.1f} (p95 {np.percentile(lat,95):.1f}) "
      f"engine {inf.mean():.1f} bridge {(hd-inf).mean():.1f} net {(lat-hd).mean():.1f} "
      f"blocked {100*lat.sum()/1000/wall:.1f}% rate {steps/wall:.2f} Hz "
      f"SR {LAB.count('P')}/{len(LAB)}")
EOF
```

Figures in `figures_vlacpp/` (the rollout directories are owned by another user
and not writable): `end_side_m4_metal.jpg` (2x5 contact sheet of every last frame,
green border = pass) and `end_m4_metal_ep09_fail_vs_ep10_pass.jpg`.
