# Multimodal chat example — SmolVLM2-500M-Instruct

A minimal **streaming image+text chat** client for `vlm-server`, the llama.cpp +
libmtmd chat runtime (`src/vlm/engine.cpp`) behind a ZMQ daemon. Send text and
images, get a streamed reply. The design rationale lives in
[docs/VLM-SERVER.md](../../docs/VLM-SERVER.md); this README is how to **run** it,
plus the validation numbers for the SmolVLM2-500M-Instruct setup.

```
examples/chat/
├── vlm_chat_client.py   streaming DEALER chat client (interactive REPL + --once)
├── vlm_pb2.py           generated proto binding (regen below if vlm.proto changes)
└── README.md            this file
```

All commands below are run **from the repo root**.

---

## 1. Build the GGUFs (one-time)

Both GGUFs come straight from `HuggingFaceTB/SmolVLM2-500M-Instruct` via
llama.cpp's own `convert_hf_to_gguf.py` (no custom converter): one pass for the
text LM, one for the vision tower (mmproj). Conversion is CPU-only.

Dependencies — system `python3` already had `torch` / `transformers` /
`safetensors`; the only addition:

```bash
pip install --user sentencepiece     # used by convert_hf_to_gguf's tokenizer step
# the `gguf` module is NOT pip-installed — it's taken from the vendored gguf-py
# via PYTHONPATH (below), so it matches our llama.cpp snapshot exactly.
```

Fetch the weights (or point `$SRC` at the HF cache snapshot you already have,
e.g. `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/<rev>`):

```bash
huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Instruct \
    --local-dir /tmp/smolvlm2-500m-instruct
```

Convert:

```bash
SRC=/tmp/smolvlm2-500m-instruct
OUT=~/data/$USER/smolvlm2-500m-instruct-gguf
mkdir -p "$OUT"
CONV="PYTHONPATH=third_party/llama.cpp/gguf-py python3 third_party/llama.cpp/convert_hf_to_gguf.py"

# text LM  ->  smolvlm2-500m-instruct-f16.gguf  (819 MB)
eval $CONV "$SRC" --outtype f16 \
    --outfile "$OUT/smolvlm2-500m-instruct-f16.gguf"

# vision tower  ->  mmproj-smolvlm2-500m-instruct-f16.gguf  (199 MB)
#   --mmproj exports ONLY the SigLIP encoder + connector (prefixes "mmproj-").
eval $CONV "$SRC" --mmproj \
    --outfile "$OUT/mmproj-smolvlm2-500m-instruct-f16.gguf"
```

The LM conversion resolves the hparams (hidden 960, ff 2560, 15 heads / 5 KV,
rope θ 1e5) and bakes the SmolVLM2 chat template into the GGUF KV store; the
`--mmproj` pass writes 198 vision tensors. Sanity-check the pair with
`llama-mtmd-cli`:

```bash
./build-cuda/bin/llama-mtmd-cli \
    -m "$OUT/smolvlm2-500m-instruct-f16.gguf" \
    --mmproj "$OUT/mmproj-smolvlm2-500m-instruct-f16.gguf" \
    --image third_party/llama.cpp/tools/mtmd/test-1.jpeg -p "Describe this image." --temp 0
```

> Shortcut: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` has **byte-identical
> weights** (same `model.safetensors` hash, config and chat template), and
> `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF` is a pre-converted equivalent
> (ships f16 + Q8_0 for both the LM and the mmproj). Either is a drop-in
> replacement for this whole step — just take the mmproj from the same repo. We
> convert our own only to keep the mmproj version-matched to the vendored
> `clip`/`mtmd` we link against.

---

## 2. Start the server

`vlm-server` positional args are `<mmproj.gguf> <lm.gguf>` (use `build-cuda` —
the CPU build is ~100× slower).

```bash
OUT=~/data/$USER/smolvlm2-500m-instruct-gguf
./build-cuda/vlm-server --bind tcp://*:5567 \
    "$OUT/mmproj-smolvlm2-500m-instruct-f16.gguf" \
    "$OUT/smolvlm2-500m-instruct-f16.gguf"
# flags: -c N_CTX (4096) | --ngl N (999=all GPU) | --no-mmproj-gpu
```

---

## 3. Chat

The client needs `pyzmq` + `protobuf`. If `src/serving/vlm.proto` changes,
regenerate the binding next to the client:

```bash
protoc --proto_path=src/serving --python_out=examples/chat src/serving/vlm.proto
```

**Interactive REPL** (streams tokens live; stateless multi-turn — the client
resends the full history each turn):

```bash
python examples/chat/vlm_chat_client.py --addr tcp://localhost:5567
  > /image path/to/photo.jpg     attach an image to the next turn
  > what is in this picture?
  > /system You are terse.        set a system prompt
  > /clear                        reset the conversation
  > /quit
```

**One-shot** (good for scripting):

```bash
python examples/chat/vlm_chat_client.py --once \
    --image third_party/llama.cpp/tools/mtmd/test-1.jpeg \
    -p "Describe this image in detail." -n 200 --temp 0
```

Useful flags: `--temp` (0 = greedy), `--top-p`, `--top-k`, `-n/--max-tokens`,
`--seed`, `--no-stream` (one-shot single reply), `--quiet`. Images attach to the
turn they are sent with and are **not** resent on later turns, so follow-ups rely
on the text history rather than re-seeing prior images.

---

## Other models

The server and client are **model-agnostic**: any VLM the vendored llama.cpp +
libmtmd supports works by swapping the two GGUFs — no rebuild, no code change.
Verified with **Qwen3-VL-2B-Instruct** via ggml-org's pre-built Q8_0 GGUF:

```bash
OUT=~/data/$USER/qwen3-vl-2b-instruct-gguf
hf download ggml-org/Qwen3-VL-2B-Instruct-GGUF \
    Qwen3-VL-2B-Instruct-Q8_0.gguf mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf --local-dir "$OUT"
./build-cuda/vlm-server --bind tcp://*:5568 \
    "$OUT/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf" "$OUT/Qwen3-VL-2B-Instruct-Q8_0.gguf"
python examples/chat/vlm_chat_client.py --addr tcp://localhost:5568
```

Qwen3-VL is more accurate than SmolVLM2-500M and far more token-efficient on
images (~23 image tokens vs SmolVLM2's ~1100 from tiling), at ~3.4 GB VRAM (Q8).
The same recipe applies to other mtmd-supported VLMs (InternVL, Gemma 3, …).
