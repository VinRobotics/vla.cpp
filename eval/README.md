# Evaluation

The eval scaffold drives `vla-server` against two simulators end-to-end over the ZeroMQ +
protobuf protocol. The C++ server does all model inference on CPU/GPU; the Python client only runs
the simulator and the per-arch normalisation, so it stays on CPU.

## Install simulators

Each setup script bootstraps an isolated Python 3.10 [`uv`](https://github.com/astral-sh/uv) venv
next to itself and clones the upstream sim. Requires `uv` on `PATH`.

### LIBERO

```bash
bash eval/sim/libero/setup_libero.sh
```

Clones LIBERO into `eval/sim/libero/LIBERO/`, creates `eval/sim/libero/libero_uv/.venv/`, pins
compatible torch / lerobot / transformers / gymnasium (and `mujoco==2.3.2`, required by
robosuite 1.4.0), and seeds `~/.libero/config.yaml` non-interactively.

### SimplerEnv

```bash
bash eval/sim/simpler/setup_SimplerEnv.sh
```

Clones SimplerEnv (and its nested `ManiSkill2_real2sim`) into `eval/sim/simpler/SimplerEnv/`,
creates `eval/sim/simpler/simpler_uv/.venv/`, and pins its ManiSkill2 + SimplerEnv editable
installs.

## Run an episode (LIBERO)

Start the server, then drive it from the LIBERO venv:

```bash
./build/vla-server "$VLA_GGUF"                 # terminal 1

# terminal 2
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
    --arch "$VLA_ARCH" \
    --task libero_object --task-id 0 --n-episodes 1 \
    --output-dir /tmp/libero_outputs
```

Notes:

- `--arch` must match the served GGUF (see the model table in the top-level README).
- **Rendering + torch on one box**: run the client with `MUJOCO_GL=egl` and
  `CUDA_VISIBLE_DEVICES=0`. robosuite's EGL renderer needs a valid device index (an empty
  `CUDA_VISIBLE_DEVICES` breaks it); the client's torch stays on CPU, so it does not need a
  torch build matching the GPU's compute capability. The server (a separate process) uses the
  GPU for inference.
- **π0** uses the gated `google/paligemma-3b-pt-224` tokenizer. Run `huggingface-cli login` and
  accept the licence, or point `--tokenizer` at a local copy.
- **GR00T** arches need `--stats-json <ckpt>/dataset_statistics.json` and an embodiment selected
  server-side via `VLA_GR00T_EMBODIMENT` (`new_embodiment` for N1.5, `libero_panda` for N1.6,
  `libero_sim` for N1.7), plus `VLA_GR00T_BF16_WEIGHTS=1` to fit an 8 GB card.

To sweep every model over `libero_object` tasks 0–9, use `eval/run_libero.sh -i <MODELS_ROOT>`.

## Run an episode (SimplerEnv)

So far only **GR00T-N1.6** is wired (the `gr00t-n1d6-bridge` checkpoint with the `oxe_widowx`
embodiment). Serve it, then drive from the SimplerEnv venv:

```bash
VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=oxe_widowx \
    ./build/vla-server "$GR00T_N1D6_GGUF"

eval/sim/simpler/simpler_uv/.venv/bin/python eval/client/run_simpler_client_direct.py \
    --arch gr00t_n1_6 \
    --task-id oxe_widowx/widowx_spoon_on_towel --n-episodes 1 \
    --stats-json "$VLA_STATS_JSON" \
    --embodiment oxe_widowx --image-size 252
```

`$VLA_STATS_JSON` is the `statistics.json` beside the bridge GGUF. The default 224-px GGUF
mis-localises on WidowX (≈20% success); the 252-px build is required.

## Reports

`eval/collect_libero_results.py` / `collect_simpler_results.py` aggregate per-episode outputs into
the markdown reports under [`reports/`](reports/); `scripts/print_versions.sh` emits the
reproducibility block (host, toolchain, GGUF hashes) for each.
