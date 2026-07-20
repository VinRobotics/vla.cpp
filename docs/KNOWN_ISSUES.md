# Known issues

## Evo-1 scores 0% on LIBERO despite the 94.5% benchmark (RESOLVED)

Evo-1 scored 0% on `libero_object` while the README benchmark reports 94.5%, and the
released `vrfai/evo1-libero-gguf` was suspected. The real cause was the base action
noise in `predict`: it sampled from N(0,1), but Evo-1 is trained with uniform[-1,1)
noise. Commit `2efef28f` switches the fallback sampler in `src/models/evo1.cpp` to
uniform[-1,1), which resolves the issue.
