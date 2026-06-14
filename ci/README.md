# vla.cpp cross-platform CI

Per-PR behavioural + performance regression gate for `vla.cpp`, run across the
three hardware platforms the project targets: a **consumer GPU** (RTX 3060), a
**Jetson** (Orin Nano), and **Apple Silicon** (M4).

## Goal

When a PR from `dev` targets `main`, evaluate the codebase on all three
platforms, refer to [Evaluation Matrix](#evaluation-matrix).

Every `(platform, model)` runs **10 tasks × 1 episode**. Additionally, on
**rtx3060**, **BitVLA** and **GR00T-N1.7** also run a **4-suite** set (10×1 each).

Recorded per run: **success rate** (client side), **latency** + **used memory**
(server side). SR must be **positive**; latency and memory must be **in range (or
better)** vs the previously reported results.

Host identities are configured in `ci/config/hosts.env`.
This README and all scripts refer to machines by **role** (`orchestrator`)
and **platform key** (`rtx3060` / `orin` / `m4`).

## Topology

One **orchestrator** host runs the GitHub Actions self-hosted runner and the
LIBERO client for every platform. Each platform is a **remote server** - only
`vla-server` + the GGUF weights live on the target; the orchestrator connects out
to it over the LAN.

```
   ┌──────────────────── orchestrator ────────────────────┐
   │  GH Actions runner + LIBERO client + vla-ci-ctl      │
   └──────┬──────────────────┬──────────────────┬─────────┘
 ctrl+data│        ctrl+data │        ctrl+data │   (LAN, two ports each)
   ┌──────▼───────┐  ┌───────▼──────┐  ┌─────────▼────┐
   │ rtx3060 srv  │  │  orin srv    │  │   m4 srv     │
   │ vla-ci-agent │  │ vla-ci-agent │  │ vla-ci-agent │
   │ + vla-server │  │ + vla-server │  │ + vla-server │
   └──────────────┘  └──────────────┘  └──────────────┘
```

### Networking - both planes on the LAN

The control plane is a small purpose-built agent.
Each server runs **two** ZMQ listeners, both dialled at the server's LAN
IP (`*_SERVER_HOST`) on different ports:

- **Control plane** (`*_CTRL_PORT`): `vla-ci-ctl` ⇄ `vla-ci-agent` - push code,
  build, spawn/stop the server, fetch logs.
- **Data plane** (`*_SERVER_PORT`): the LIBERO client ⇄ `vla-server` on every
  call.

`*_SERVER_HOST` **is a local-subnet IP** (e.g. `192.168.1.*`).
Both daemons bind `tcp://*:<port>` (all interfaces), so they already listen on the LAN IP; you
only point the orchestrator at the right one.

### Control plane - `vla-ci-agent`

A tiny C++ daemon on each server speaks a protobuf-over-ZMQ
REQ/REP protocol (the same stack `vla-server` already uses). It is built
standalone from [`ci/agent/`](agent/):

- **`vla-ci-agent`** runs on each server; ops: `ping`, `put` (upload + extract a
  gzip-tar of the PR checkout, with rsync-`--delete`-style prune that protects
  `build/`, `third_party/`, sim venvs…), `exec` (run the build, stream exit code
  + output), `spawn`/`stop` (detached server lifecycle by name), `get` (fetch a
  log). Protocol: [`ci/agent/vla_ctl.proto`](agent/vla_ctl.proto).
- **`vla-ci-ctl`** runs on the orchestrator; `run_remote.sh` calls it in place of
  ssh/rsync/scp.

Optional shared-secret auth via `VLA_CI_TOKEN` (`--token`). The agent runs
arbitrary commands by design - bind it to the LAN.

```bash
# build the agent tools (orchestrator AND each server - stdlib of the project: protobuf + ZeroMQ)
cmake -S ci/agent -B ci/agent/build -DCMAKE_BUILD_TYPE=Release && cmake --build ci/agent/build -j

# on each server, run the daemon (systemd / launchd / nohup), e.g.:
ci/agent/build/vla-ci-agent --bind 'tcp://*:5600' [--token "$VLA_CI_TOKEN"]
```

This is a one-time bootstrap per server (like installing `sshd`): copy/build the
agent once, run it as a service. Subsequent PRs reuse it.

## Evaluation Matrix

| Platform | Server | Models | Suites |
|---|---|---|---|
| `rtx3060` | RTX 3060 (CUDA sm_86) | smolvla, pi0, bitvla, evo1, gr00t_n1_5, gr00t_n1_6, gr00t_n1_7 | `libero_object` (all) **+** spatial/object/goal/10 for **bitvla** and **gr00t_n1_7** |
| `orin` | Jetson Orin Nano (Tegra sm_87) | smolvla, pi0, bitvla, evo1, gr00t_n1_5 | `libero_object` |
| `m4` | Apple M4 (Metal) | smolvla, pi0, gr00t_n1_7 | `libero_object` |

Every cell is **10 tasks × 1 episode**. Gating on **server-side** metrics (server
`total` ms, per-PID server memory) means the orchestrator↔server ZMQ transport
(recorded in `client/call`, not gated) does **not** invalidate the committed
baselines.

## Metrics & gating

For each `(platform, model, suite)` the run records:

- **SR** (client side) - `successes / n_episodes`, parsed from each
  `summary.txt`. **Gate: SR > 0.**
- **Server latency** - `total` ms per `vla-server` call, parsed from
  `_server_logs/<arch>.<suite>.log` (`vla-server: rid=… total=… ms vision=… inf=… other=…`).
  Per-suite models (**bitvla**, **gr00t_n1_7**) serve each suite from its own
  checkpoint, so one log per suite; the gate takes the sample-weighted mean.
  **Gate: ≤ 1.10× baseline.**
- **Server memory** - peak, sampled by a sidecar while `vla-server` is alive:
  - `rtx3060` → **peak VRAM** (`nvidia-smi --query-compute-apps`).
  - `orin` → **sys Δ** (`MemTotal − MemAvailable` rise). On Tegra the iGPU shares
    system RAM and is invisible to VmHWM / nvidia-smi, so the system-used delta
    is the only faithful figure.
  - `m4` → best-effort RSS + system-used (no `/proc`, no `nvidia-smi`).
    **Record-only - no baseline, not gated.**
  - **Gate (rtx3060, orin): ≤ 1.10× baseline.**

Baselines live in [`ci/baselines/`](baselines/) as JSON (one per platform key,
transcribed from the reports) so the gate is a pure data comparison. Update them
by hand if the committed reports change.

## Setup & usage

```bash
# 1. One-time per server: build + run the control-plane agent (see "Control plane").
cmake -S ci/agent -B ci/agent/build -DCMAKE_BUILD_TYPE=Release && cmake --build ci/agent/build -j
#   ...then on each server:  ci/agent/build/vla-ci-agent --bind 'tcp://*:5600'

# 2. One-time: create your machine config (gitignored).
cp ci/config/hosts.env.example ci/config/hosts.env
$EDITOR ci/config/hosts.env          # fill LAN IPs, ctrl/data ports, repo paths, MODELS_ROOT, build flags

# 3. One platform end-to-end (put→build→serve→client→gate):
bash ci/orchestrate.sh rtx3060
bash ci/orchestrate.sh orin
bash ci/orchestrate.sh m4

# 4. Just re-check an existing sweep against baselines:
python ci/check_thresholds.py --platform rtx3060 \
    --sweep outputs/ci/rtx3060 --baseline ci/baselines/rtx3060.json
```

Results land under `outputs/ci/<platform>/` (client `summary.txt` per task,
`_server_logs/<arch>.<suite>.log`, `_server_logs/<arch>.<suite>.mem.json`) plus a
`verdict.json` / `verdict.md` from the gate. CI uploads these as artifacts and
fails the job on any gate violation.

`run_remote.sh` (via `vla-ci-ctl`) `put`s the PR checkout to each server
(excluding `build/`, `third_party/`, `outputs/`, the sim venvs), `exec`s the
`vla-server` build there with the platform's CMake flags (`*_CMAKE_FLAGS` in
`hosts.env`), then per model `spawn`s `remote_server.sh`, drives the client, and
`get`s the server log + `mem.json` back so the gate sees one consistent tree.

## Prerequisites by role

- **orchestrator** (runner + all clients): GitHub Actions self-hosted runner with
  label `vla-ci-orchestrator`, the repo checkout, the LIBERO venv
  (`eval/sim/libero/setup_libero.sh`), the GR00T `dataset_statistics.json` files
  at `ORCH_MODELS_ROOT` (client-side un-normalisation - tiny, copy from any
  server), and the built **`vla-ci-ctl`** (`cmake -S ci/agent -B ci/agent/build`).
  No SSH needed. It does **not** build or run `vla-server`.
- **each server**: the **`vla-ci-agent`** running (protobuf + ZeroMQ to build it),
  plus the platform toolchain + GGUF weights:
  - **rtx3060**: CUDA toolchain (sm_86) + weights for all 7 archs.
  - **orin**: CUDA/Tegra toolchain (sm_87) + weights for smolvla, pi0, bitvla,
    evo1, gr00t_n1_5.
  - **m4**: toolchain with Metal ([docs/backend/metal.md](../docs/backend/metal.md))
    + weights for smolvla, pi0, gr00t_n1_7.

`vla-server` itself is (re)built from the PR code by `run_remote.sh`, so servers
need only the toolchain + weights + the running agent, not a prebuilt `vla-server`.
