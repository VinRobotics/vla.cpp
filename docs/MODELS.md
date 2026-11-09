# Converting and quantizing models

Each model ships as a single self-contained GGUF; the published ones are listed in
the README's [support matrix](../README.md#support-matrix) and the
[vrfai collection](https://huggingface.co/collections/vrfai/vlacpp-model-bundles).

## Conversion

To convert a HuggingFace safetensors checkpoint yourself, [`scripts/`](../scripts/)
has a converter per arch. Set up its venv:

```bash
python3 -m venv .venv-converter
source .venv-converter/bin/activate
pip install -e ".[convert]"
```

Then run any of the per-arch converters (`--help` for the full flag list):

```bash
python scripts/convert_smolvla_to_gguf.py \
    --ckpt /path/to/smolvla-libero \
    --out  /path/to/smolvla-libero-bf16.gguf
```

## Quantization

Most shipped GGUFs are BF16. π0.5, Octo and TurboVLA ship F32, and GR00T N1.5
and N1.6 are mostly F32. `scripts/quantize_gguf.py` repacks the LM-backbone weight
matrices to a smaller type and copies everything else unchanged. The loader keeps
the packed weights, so the file loads and runs like the original.

```bash
python scripts/quantize_gguf.py --in model-bf16.gguf --out model-q8_0.gguf --type Q8_0
```

On CPU and CUDA the packed matmuls do not dequantize to float first. ggml
quantizes the activations to 8 bits and runs integer dot products on the blocks
(`vec_dot_q8_0_q8_0` on CPU, the MMQ kernels on CUDA), so a Q8_0 LM is int8
compute, not BF16.

`Q8_0` is near-lossless and roughly halves the LM against BF16. `Q4_0` is 4-bit for
a bigger cut (`--type` also takes `Q4_1`, `Q5_0`, `Q5_1`).
Embeddings, the output head, norms and the action expert stay float; pass `--vision` to
pack the vision tower too (smaller, but more accuracy loss).

### FoldQuant W8A8 / W4A4

The stock repack keeps the action head float and rounds the LM weights block by
block with no calibration. A FoldQuant GGUF ships the language backbone and the
action head as INT8 or INT4 codes in a Hadamard-rotated, SmoothQuant-folded frame
with per-row scales, calibrated by FoldQuantVLA; vla.cpp quantizes the activations
per token (to 4 bits for W4A4) and runs the GEMMs on the integer tensor cores. It loads like any other checkpoint on every backend: CUDA and SYCL (Intel GPUs) run
integer kernels, CPU the exact reference and OpenVINO (CPU and GPU) the same
arithmetic as OpenVINO ops with INT8/INT4 weight constants; the others read the sites back as float
weights with FoldQuant's rounding (weight-only quantization, bf16-GGUF speed); the format
and the arithmetic are in [QUANTIZATION.md](QUANTIZATION.md).

A calibrated arm saved as a quantized model by
[FoldQuantVLA](https://github.com/VinRobotics/FoldQuantVLA) converts to that file with
no calibration and nothing re-rounded (GR00T N1.5 / N1.6 / N1.7 and π0.5):

```bash
python scripts/convert_quantized_model_to_gguf.py --quantized-model <quantized model dir> --out model-fq.gguf
./build/vla-server model-fq.gguf --bind tcp://*:5556

# uncalibrated stand-in for bring-up and benchmarks (rotation + per-row INT8, no SmoothQuant)
python scripts/foldquant_fake_export.py --in model-bf16.gguf --out model-fq.gguf
python scripts/inspect_gguf_quant.py model-fq.gguf
./build/vla-bench --ckpt model-fq.gguf --images 1 --size 256 --tokens 16
```

## Cosmos3-Edge (world-action model)

Cosmos3-Edge (`lerobot/Cosmos3-Edge-Policy-LIBERO`) is a world-action model: a 28-layer Mixture-of-Transformers
backbone (Nemotron 3 Dense, an understanding stream over the prompt and a generation stream over
`[video patches | actions]`) denoises 4 latent video frames and a 16-step action chunk together, conditioned
on the Wan2.2 VAE latent of the current frame, with 30 UniPC steps. `src/models/cosmos3_edge.cpp` follows
lerobot's `policies/cosmos3/modeling_cosmos3.py`: the VAE encoder and the backbone are ggml graphs (the
understanding stream runs once, its keys and values are kept for the 30 steps), the UniPC solver
(diffusers' `UniPCMultistepScheduler`, flow prediction, order 2, corrector) runs on the host in f32 as the
reference does. Convert from the checkpoint directory:

```bash
# the VAE encoder (one frame: the 3-D kernels' last temporal tap), the 28 layers of both streams, the
# projections of domain 5 (LIBERO), the latent statistics, the action quantiles, the canvas layout, the caption
# settings and the tokenizer. The decoder, the SigLIP tower and the language head stay out.
python scripts/convert_cosmos3_edge_to_gguf.py --ckpt <checkpoint dir> --out cosmos3-edge-libero-bf16.gguf
./build/vla-cli --ckpt cosmos3-edge-libero-bf16.gguf --image third_person.png --image wrist.png \
    --text "turn on the stove" --libero --pretty
```

**Inputs.** Two camera views in the order `[third person, wrist]`, of any size: each is resized to 256x256 with
antialias, composed side by side and fit into the 192x320 canvas the way lerobot's
`Cosmos3VideoCanvasProcessorStep` does (the layout is in the GGUF). Or one 320x192 image that the caller
composed itself. The instruction goes in as `--text`, which builds the JSON caption and the chat template of
`Cosmos3PromptProcessorStep` and tokenizes it with the model's byte-level BPE from the GGUF; or as `--tokens`
ids from lerobot. The text must be ASCII; anything else is refused. `noise` is the 16x10 action noise (random
if omitted, or `VLA_COSMOS3_SEED`); the video noise is drawn here from a seed that follows it.

**Output.** The 16x10 chunk in dataset units, `[dpos(3), rot6d(6), gripper]`. `vla-cli --libero` prints the
LIBERO command `[dpos(3), axis-angle(3), gripper]` instead (lerobot's `Cosmos3LiberoActionProcessorStep`);
through `libvla` the caller decodes the rot6d. GEMM weights are the checkpoint's bf16 (`--weight-dtype` casts
them at load); on CUDA the VAE runs as f16 direct convolutions, elsewhere as f32 im2col. `--flash-attn`,
`--mm-prec-f32` and `--num-steps` work; `--act-dtype bf16` is refused. The engine is written for this policy's
setup: mode `wam`, no robot state, no classifier-free guidance, one or two cameras of the 256 resolution tier;
the converter refuses a config that differs.

**Serving.** `vla-server` needs no change for this model: it hands the images, token ids and noise of a request to
`predict()` whatever the architecture is, and Cosmos3-Edge fits its checks (no state, a 16x10 noise, a 16x10
response; the start-up warm-up of two 224x224 views and 32 made-up tokens runs). The server has no text field, so
the client sends token ids. `eval/client/vla_cpp_client.py` has a `cosmos3_edge` preset that builds them (the caption
and chat template of lerobot's prompt step, with the policy's tokenizer, and the caption settings read from the
server's GGUF with `--gguf`, the same `cosmos3_edge.caption.*` keys `vla-cli --text` uses; without `--gguf` it assumes
the LIBERO checkpoint's), sends the two 256x256 views as float RGB, and decodes the 16x10 rot6d chunk it gets back
to 16 LIBERO commands:

```bash
./build/vla-server cosmos3-edge-bf16.gguf                      # -DVLA_BUILD_SERVER=ON
python eval/client/run_sim_client_direct.py --arch cosmos3_edge --n-action-steps 16 \
    --gguf cosmos3-edge-bf16.gguf --tokenizer <checkpoint dir>/tokenizer ...   # nvidia/Cosmos3-Edge if reachable
```

`eval/run_libero.sh -m cosmos3_edge` runs the whole flow (server, LIBERO client, stop) from a models directory that
holds `cosmos3-edge-libero-gguf/cosmos3-edge-libero-bf16.gguf`; a missing directory is skipped, so `-m all` is
unaffected. The client needs a Python with LIBERO and a transformers that reads the policy's tokenizer:
`COSMOS3_EDGE_PYTHON` picks it (default: the LIBERO venv of `eval/sim/libero/setup_libero.sh`, which pins an older
transformers), `COSMOS3_EDGE_TOKENIZER` the local `tokenizer/` directory.

**What was checked** (bf16 GGUF; CPU and CUDA on an RTX 3060):

- *The engine*, against lerobot's policy run in bf16 (the configuration of its LIBERO numbers) on the same views,
  tokens and noise: on a 152-token prompt the 16x10 chunk agrees to a scaled error of 2.2e-3 on CUDA and 2.6e-3
  on CPU, on a 150-token prompt to 1.8e-2 (CUDA). The bf16 reference carries rounding of that order itself (its
  latent differs from the engine's f32 one by 1e-2), so the second figure is not yet separated into engine and
  reference error; an f32 reference run is what settles it. Stage by stage at the first step, the prompt keys
  and values, the generation stream's embeddings, layer 0's queries and output and both velocities agree to
  5e-3 to 1e-2, the reference's precision. The host UniPC port agrees with diffusers' scheduler to 5e-7 on a
  synthetic velocity field. One chunk takes 4.3 s on the 3060 (VAE 0.11 s, prompt 0.11 s, 30 steps 4.1 s) and
  74 to 103 s on a 12-thread CPU.
- *The prompt.* For the 40 LIBERO tasks and five edge cases the caption is byte-identical to lerobot's and
  the ids to Hugging Face's, and 550 random ASCII and special-token texts tokenize identically
  (`test_cosmos3_edge_prompt`, data in `tests/data`). llama.cpp's BPE (`b11223`, pre-tokenizer `tekken`, which uses
  a rewritten approximation of the model's regex) gave the same ids on 607 of 607 texts, so the in-tree tokenizer
  was cross-checked against both. It implements the ASCII letter classes only, hence the ASCII limit.
- *The action decode.* rot6d to axis-angle matches lerobot's to 7e-7 on 62 rows.
- *The canvas.* The C++ canvas matches torch's CPU result to within one pixel of 184,320 on random images. It
  differs from the canvas lerobot makes on a GPU in 33 to 38 pixels, by one level each, as torch's own CPU and
  CUDA results do; the actions on the two canvases differ by less than the engine-reference gap above.
- *The server* (CPU, an earlier revision of the engine). A request for one case returned exactly what a direct
  `libvla` call returned on the same inputs, the client built the same 150 token ids as the case's, its numpy
  rot6d decode matched lerobot's to 7e-7, and `get_action` returned a whole chunk of 16 finite LIBERO commands.
  Through `eval/run_libero.sh -m cosmos3_edge -s libero_spatial` (task 0, one episode, CPU) the episode was
  solved. The engine has been rewritten since; the server path itself is unchanged.

Not checked with the hand-written engine: the closed loop on LIBERO (an earlier revision of this port, with
graphs generated from the same PyTorch path, solved 19 of 20 libero_spatial episodes and reached 96.45% over all
of LIBERO against 96.6% for the PyTorch policy in bf16); `--flash-attn` (the VAE's 640-wide attention head is kept
on the explicit path because the CUDA kernels have no such head size, but the remaining path was not compared);
ROCm, SYCL, Metal, OpenVINO and Hexagon; `vla-server` on a GPU.
