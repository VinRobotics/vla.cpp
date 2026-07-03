# Known issues

## Evo-1 scores 0% on LIBERO despite the 94.5% benchmark (NEEDS DOUBLE-CHECK)

The released `vrfai/evo1-libero-gguf` scores **0%** on `libero_object` (checked on
task_0/1/2, RTX 5090) while the README benchmark reports **94.5%**.

Evidence this is a real, pre-existing regression, not a harness or config artifact:
- **smolvla scores 100%** on the same harness, build, and task, so the eval path works.
- evo1 `predict_check` is bit-identical on llama.cpp b9860 and b9866, so the bump did
  not change evo1 numerics.
- The model loads and emits finite, non-degenerate actions (~13 ms/step); episodes
  just never complete the task.
- Client wiring looks right: image_size 448, InternVL3 tokenizer, a dedicated evo1
  path, and a token-count guard that fails loudly on mismatch rather than scoring 0.

Most likely the uploaded `evo1-libero.gguf` diverged from the checkpoint that produced
the 94.5% number, or the evo1 loader/converter regressed before it was benchmarked.
Do not blind-fix the flow-matching base noise or attention masking: the base is
N(0,1), and switching it to the reference's uniform[-1,1] also gives 0%.

Double-check: diff the released GGUF's tensors and config against a fresh
`convert_evo1_to_gguf.py` from `MINT-SJTU/Evo1_LIBERO`, and compare one denoise step's
`v_t` against the Evo-1 reference on identical inputs.

Status: the N(0,1) base noise and the DiT context mask are in the tree, so the code
path matches the reference. The tensor/config diff above is the open step and needs the
upstream safetensors checkpoint, which is not bundled here.
