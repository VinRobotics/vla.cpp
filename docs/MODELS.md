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
