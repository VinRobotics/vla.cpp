# Simulator evaluation

The eval scaffold under [`eval/`](../eval/) runs `vla-server` against two
simulators end-to-end: LIBERO and SimplerEnv. To run it in containers instead,
see [DOCKER.md](DOCKER.md).

## Install simulators

Each setup script bootstraps an isolated Python 3.10 `uv` venv next to itself and
clones the upstream sim repo. Both require [`uv`](https://github.com/astral-sh/uv)
on `PATH`.

### LIBERO

```bash
bash eval/sim/libero/setup_libero.sh
```

Clones LIBERO into [`eval/sim/libero/LIBERO/`](../eval/sim/libero/LIBERO),
creates `eval/sim/libero/libero_uv/.venv/`, and pins compatible versions of
torch, lerobot, transformers, and gymnasium.

### SimplerEnv

```bash
bash eval/sim/simpler/setup_SimplerEnv.sh
```

Clones SimplerEnv (and its nested `ManiSkill2_real2sim`) into
[`eval/sim/simpler/SimplerEnv/`](../eval/sim/simpler/SimplerEnv), creates
`eval/sim/simpler/simpler_uv/.venv/`.

## Running the client

[`eval/client/`](../eval/client/) drives `vla-server` directly over the protobuf
protocol. Start the server first, see [USAGE.md](USAGE.md#vla-server).

```bash
export VLA_GGUF=models/smolvla/smolvla-libero.gguf   # the checkpoint to serve
export VLA_ARCH=smolvla                              # client-side arch preset, see --help
```

### LIBERO

With `vla-server` already running:

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate
python eval/client/run_sim_client_direct.py \
    --task libero_object --task-id 0 --n-episodes 1 \
    --output-dir /tmp/libero_outputs \
    --arch "$VLA_ARCH"
```

The GR00T models need two extras:

- client side: `--stats-json /path/to/dataset_statistics.json`
- server side: `VLA_GR00T_EMBODIMENT` (`new_embodiment` for N1.5, `libero_panda`
  for N1.6, `libero_sim` for N1.7).

### SimplerEnv

So far only **GR00T-N1.6** is wired (the `gr00t-n1d6-bridge` checkpoint with the
`oxe_widowx` embodiment). Start `vla-server` with the `oxe_widowx` embodiment:

```bash
VLA_GR00T_EMBODIMENT=oxe_widowx \
    ./build/vla-server "$GR00T_N1D6_GGUF"
```

Then drive it from the SimplerEnv venv:

```bash
source eval/sim/simpler/simpler_uv/.venv/bin/activate
python eval/client/run_simpler_client_direct.py \
    --arch gr00t_n1_6 \
    --task-id oxe_widowx/widowx_spoon_on_towel --n-episodes 1 \
    --embodiment oxe_widowx --image-size 252 \
    --stats-json "$VLA_STATS_JSON"
```

## Task success

Success rate belongs to the checkpoint, not the engine; `vla_predict_check` in
[CONTRIBUTING.md](../CONTRIBUTING.md) is how a change is shown to leave it
alone. Measured LIBERO success rates are in [CHANGELOG.md](../CHANGELOG.md).
