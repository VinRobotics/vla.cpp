# vla.cpp cross-platform CI

Regression gate for `vla.cpp` across the three target platforms - RTX 3090
(`rtx3090`), Jetson Orin Nano (`orin`), Apple M4 (`m4`). On every push to `dev`,
each platform's `vla-server` is evaluated in LIBERO and gated on success rate
(client side) + latency / memory (server side).

Machines are referred to by **role** (`orchestrator`) and **platform key**; real
hostnames / IPs live only in the gitignored `ci/config/hosts.env`.

## Topology

The orchestrator runs the GitHub Actions runner and one LIBERO client per
platform; each platform is a remote server reached over the LAN.

```
   ┌──────────────────── orchestrator ────────────────┐
   │  GH Actions runner + LIBERO client + vla-ci-ctl  │
   └──────┬──────────────────┬──────────────────┬─────┘
 ctrl+data│        ctrl+data │        ctrl+data │   (LAN, two ports each)
   ┌──────▼───────┐  ┌───────▼──────┐  ┌─────────▼────┐
   │ rtx3090 srv  │  │  orin srv    │  │   m4 srv     │
   │ vla-ci-agent │  │ vla-ci-agent │  │ vla-ci-agent │
   │ + vla-server │  │ + vla-server │  │ + vla-server │
   └──────────────┘  └──────────────┘  └──────────────┘
```

## Evaluation matrix

| Platform | Models | Suites |
|---|---|---|
| `rtx3090` | all 10 (smolvla, pi0, pi05, bitvla, evo1, vla_adapter, openvla_oft, gr00t_n1_5/6/7) | `libero_object`; **+ spatial/object/goal/10** for bitvla & gr00t_n1_7 |
| `orin` | smolvla, pi0, bitvla, evo1, gr00t_n1_5 | `libero_object` |
| `m4` | smolvla, pi0, gr00t_n1_7 | `libero_object` |

Every cell is **10 tasks × 1 episode**.

## Gating

- **SR** (client side) - must be **≥ 0.5× `sr_reported`** of the baseline
  (`sr_tolerance` in `ci/baselines/<platform>.json` overrides 0.5; **> 0** where a
  model has no `sr_reported`).
- **Server latency & memory** - must be **≤ 1.10× baseline**
  (`ci/baselines/<platform>.json`). M4 memory has no baseline (recorded, not gated).

Per-platform verdict → `outputs/ci/<platform>/verdict.{json,md}`; combined →
`outputs/ci/report.{md,json}`.

## Setup & usage

### 1. Install the control protocol (on the orchestrator **and** every server)

```bash
cmake -S ci/agent -B ci/agent/build -DCMAKE_BUILD_TYPE=Release && cmake --build ci/agent/build -j
```

Builds `vla-ci-agent` (runs on each server) and `vla-ci-ctl` (runs on the
orchestrator). Needs only protobuf + ZeroMQ (same as `vla-server`).

### 2. Configure hosts

```bash
cp ci/config/hosts.env.example ci/config/hosts.env
$EDITOR ci/config/hosts.env     # LAN IPs, ctrl/data ports, repo paths, MODELS_ROOT
```

### 3. Bring up servers + agents, check with ping

On each server (its own git checkout, with `vla-server` already built), run the
agent as a service (systemd / launchd / nohup). It binds 127.0.0.1 unless given
`--bind`:

```bash
ci/agent/build/vla-ci-agent --bind 'tcp://*:5600' [--token "$VLA_CI_TOKEN"]
```

From the orchestrator, confirm each server is reachable:

```bash
ci/agent/build/vla-ci-ctl --endpoint tcp://<server-lan-ip>:5600 ping
```

Build `vla-server` on the servers (each builds its own checkout, once per commit):

```bash
bash ci/build_servers.sh                 # all servers, in parallel (uses *_CMAKE_FLAGS)
# or one:  vla-ci-ctl --endpoint tcp://<ip>:5600 build --cwd <server-repo> --flags "<cmake flags>"
```

### 4. Run the three platforms simultaneously

```bash
bash ci/orchestrate.sh all
```

Verifies the tested machines are on the same commit, then sweeps all three
platforms in parallel (one LIBERO client per platform on the orchestrator).
Single platform: `bash ci/orchestrate.sh rtx3090|orin|m4`.

### 5. Aggregate report

`orchestrate.sh all` writes the combined report after the sims finish:

```bash
cat outputs/ci/report.md
```

It collects each server's latency / memory + client SR + commit into one
PASS/FAIL summary. To re-gate an existing sweep without re-running it:

```bash
python ci/check_thresholds.py --platform rtx3090 \
    --sweep outputs/ci/rtx3090 --baseline ci/baselines/rtx3090.json
```

## CI trigger

`.github/workflows/vla-ci.yml` runs `ci/orchestrate.sh all` on a self-hosted
runner labelled `vla-ci-orchestrator` on every push to `dev` (or by hand via
workflow_dispatch), and uploads `outputs/ci/` as an artifact. It sets
`VLA_CI_EXPECTED_COMMIT` to the pushed commit, so every server first fetches that
commit from its `origin`, checks it out and rebuilds (`ci/build_servers.sh`); the
gate fails if a server is on any other commit. The runner's `.env` must set
`VLA_CI_HOSTS_ENV` to a hosts.env outside the workspace (checkout wipes ignored
files).

It does not run on `pull_request`: a PR run takes its workflow file from the PR's
merge commit, so a fork could drop any guard in it and reach the LAN agents. Put
the orchestrator runner in an org runner group whose workflow access is restricted
to `VinRobotics/vla.cpp/.github/workflows/vla-ci.yml@refs/heads/dev`. A path-only
restriction is not enough, because a fork edits the same path. With that group,
a manual dispatch gets a runner only when run from `dev`.
