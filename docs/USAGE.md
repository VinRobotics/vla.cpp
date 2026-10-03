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

`vla-server` loads the model once at startup and answers ZeroMQ requests carrying
the protobuf messages in `src/serving/vla.proto`.

```bash
./build/vla-server "$VLA_GGUF"
```

When ready, the server prints:

```
vla-server: bound to tcp://*:5555. ready.
```

Use `--bind` to change the address and port. Stop the server with `Ctrl-C`.
`vla-server` also takes `-hf user/repo[:file.gguf|:tag]` in place of a checkpoint path.

### Asynchronous clients

The socket is a ZeroMQ ROUTER, and prediction runs on its own thread. A REQ
client sees what it always did: one request, one reply. A DEALER client can
keep several requests in flight, and the server keeps receiving while the model
runs, so a robot's control loop never has to wait for a prediction to send the
next observation. Replies carry the request's `request_id`, which is how a
client matches a chunk to the observation it came from.

`--queue` says what happens to requests that arrive while a prediction is
running:

- `latest` (default): one pending request per client. A newer request from the
  same client replaces the pending one, which is answered at once with
  `error="superseded"`. The model therefore always works on the freshest
  observation each robot has sent, and several robots sharing one server are
  served in turn.
- `fifo`: every request is served in arrival order, for benchmarks that want
  throughput rather than freshness.

`latency_ms_queue` in the reply is how long the request waited for the predict
thread. A malformed request is rejected on the socket thread, so an error comes
back within milliseconds even mid-prediction.

Clients: the LIBERO and SimplerEnv runners in [EVAL.md](EVAL.md), and the
real-robot client in the README's
[Rollout on a real robot](../README.md#rollout-on-a-real-robot).

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
