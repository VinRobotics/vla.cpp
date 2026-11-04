# Using the binaries

How to drive `vla-cli` and `vla-server`, and the flags and environment variables
they share. The README's [Quickstart](../README.md#quickstart) covers the
first run.

## `vla-cli`

`vla-cli` runs a single prediction without a server or simulator: give it a model,
an image, and an instruction, and it prints the action chunk. Handy for
smoke-testing a GGUF or scripting a quick inference.

```bash
./build/vla-cli -hf vrfai/smolvla-libero-gguf \
    --image assets/front.jpg --text "pick up the black bowl" --pretty
```

`--text` builds the same prompt the eval client sends for that arch.
It is tokenized in-process when the GGUF carries a SentencePiece
tokenizer (Octo, or pi0, pi0.5 and OpenVLA-OFT after
`scripts/add_tokenizer_to_gguf.py --in model.gguf --out model-tok.gguf`).
Otherwise it calls `scripts/tokenize_prompt.py` with the tokenizer the
architecture was trained on, looking in `scripts/` next to `vla-cli`, then
`share/vla`, then the source tree (`VLA_PYTHON` picks the interpreter,
`VLA_TOKENIZE_SCRIPT` overrides the script). Pass `--tokens 1,100,200,2` instead
if you already have ids.
`--pretty` prints one action row per line;
`--state` sets proprioception (defaults to zeros). pi0.5 puts the state into its
prompt, so pi0.5 `--text` needs `--state`.

## Fetching checkpoints with `-hf`

`-hf` takes `user/repo`, `user/repo:path/in/repo.gguf`, or `user/repo:tag`,
where the tag is any part of the file path (case-insensitive, like `:Q8_0`).
The BitVLA and GR00T N1.7 repos hold one GGUF per LIBERO suite. When more than
one file matches, `-hf` lists them and stops, so pick one:

```bash
./build/vla-cli -hf vrfai/gr00tn1d7-libero-gguf:libero_object/gr00tn1d7-libero-object.gguf ...
./build/vla-cli -hf vrfai/gr00tn1d7-libero-gguf:object ...
```

Checkpoints are cached under `$VLA_CACHE` (default `~/.cache/vla`).

## `vla-server`

`vla-server` loads the model once at startup and answers ZeroMQ REQ/REP requests
synchronously.

```bash
./build/vla-server "$VLA_GGUF"
```

When ready, the server prints:

```
vla-server: bound to tcp://*:5555. ready.
```

Use `--bind` to change the address and port. Stop the server with `Ctrl-C`.
`vla-server` also takes `-hf user/repo[:file.gguf|:tag]` in place of a checkpoint path.

Clients: the LIBERO and SimplerEnv runners in [EVAL.md](EVAL.md), and the
real-robot client in the README's
[Rollout on a real robot](../README.md#rollout-on-a-real-robot).

### Prefix conditioning across chunks (SmolVLA)

For asynchronous execution, a request can carry the part of the previous chunk
that has not run yet, so the new chunk starts where the old one is going instead
of jumping. Off by default; without it the output is bit-identical.

| field | where | meaning |
|---|---|---|
| `prev_actions` | `PredictRequest` field 9, `vla_inputs.prev_actions` / `n_prev_actions`, Python `predict(prev_actions=...)` | the not-yet-executed rest of the previous chunk, robot units (same as the output), whole rows of `max_action_dim`, at most `chunk_size` rows |
| `inference_delay` | `PredictRequest` field 10, `vla_inputs.inference_delay`, Python `predict(inference_delay=...)` | actions that run while this chunk is computed (d) |

At every Euler step the predicted clean chunk is pulled onto `prev_actions`:
`x1 = x_t - t v`, `v <- v - gw(t) W (Y - x1)`, with `Y` the rows normalised like the
output (zero past the last row), and `W` and `gw(t)` LeRobot's `RTCProcessor` prefix
weights and guidance weight (`gw` clamped by `max_guidance_weight`). It is
gradient-free: the error is used directly, the default of huggingface/lerobot#4945,
instead of the vector-Jacobian product, which would need backward passes through
the action expert and backward rules ggml lacks for flash attention and `concat`.
It adds a few elementwise ops per step. The behaviour is checked against LeRobot's
code in `tests/test_prefix_cond.cpp`; smoothness at the chunk switch on recorded
episodes and task success on hardware are not measured for this version.

Alignment: row `i` of `prev_actions` is the action for the same timestep as row `i`
of the new chunk (LeRobot's `prev_chunk_left_over`), `inference_delay` is how many
of those rows will already have run when the new chunk arrives. The weights are
never sent per request. They are load-time options, the `rtc_config` object of a
`config.json` (the model directory's for safetensors; for a GGUF, which carries its
own config, the file given as `--config` / `config_path`), with LeRobot's `RTCConfig`
names and defaults. The engine prints the values in use at load time:

| `rtc_config` key | default | meaning |
|---|---|---|
| `execution_horizon` | 10 | rows over which the weights decay; cut to the length of `prev_actions` |
| `max_guidance_weight` | 10.0 | clamp of `gw(t)` |
| `prefix_attention_schedule` | `LINEAR` | `ZEROS`, `ONES`, `LINEAR` or `EXP` |

SmolVLA only. Any other architecture refuses the input instead of ignoring it:
`VLA_ERR_UNSUPPORTED` from `vla_predict`, `NotImplementedError` in Python, an error
response from `vla-server`.

## Runtime flags

The same on `vla-server`, `vla-cli` and `vla-bench` (`--help` for the full
list). On `vla-server` and `vla-cli`, the `runtime` block of a `--config` JSON
sets them too, and the command line wins. The fastest configuration per model,
with measured latency and success rate, is in [CHANGELOG.md](../CHANGELOG.md);
per-device fastest flags are in [benchmark/](benchmark/).

- `--weight-dtype f32|bf16|f16` - resident dtype for GEMM weights.
- `--act-dtype f32|bf16` - activation dtype; bf16 is π0 and Evo-1 only and
  needs CUDA and bf16 weights.
- `--flash-attn` - faster on the larger towers, but changes numerics.
- `--mm-prec default|f32` - matmul accumulation precision.
- `--num-steps N` - flow-matching solver steps for π0, π0.5, SmolVLA, Evo-1,
  GR00T and VLA-JEPA (default: the checkpoint's). The other archs refuse it.

## Environment variables

These apply to every arch:

- `VLA_N_THREADS` - CPU backend thread count, default core count capped at 16.
- `VLA_DEVICE` - GPU ordinal for CUDA and SYCL builds, default 0.
- `VLA_CACHE` - where `-hf` stores checkpoints, default `~/.cache/vla`.

Checkpoints that carry stats for several datasets need the one to un-normalize
with, for example `VLA_OCTO_UNNORM_DATASET=libero_object` for the Octo LIBERO
GGUF, which ships four. GR00T needs `VLA_GR00T_EMBODIMENT`, see
[EVAL.md](EVAL.md).

## Benchmarking

`vla-bench` times `predict()` in-process on synthetic inputs: engine only, no
transport, no simulator, no claim about task success.

```bash
./build/vla-bench -hf vrfai/smolvla-libero-gguf --images 2 --size 512 --markdown
```

Results per device are in [benchmark/](benchmark/).
