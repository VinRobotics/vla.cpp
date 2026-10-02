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
per token (to 4 bits for W4A4) and runs the GEMMs on the integer tensor cores. It loads like any other checkpoint on the CUDA and CPU backends (other backends
refuse it); the format and the arithmetic are in [QUANTIZATION.md](QUANTIZATION.md).

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
