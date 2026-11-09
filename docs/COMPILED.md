# Serving a model compiled by ggml-compiler

vla.cpp no longer adds hand-written architectures for every new policy. A
model the engine does not know is compiled with
[ggml-compiler](https://github.com/khanhnd61-vr/ggml-compiler) into one GGUF
that holds its weights, its op graphs and a **serving manifest**, and
`vla-server`, `vla-cli`, `vla-bench` and `libvla` run that GGUF like any
built-in model. No vla.cpp change, no plugin binary, no model-specific code.

- [How it works](#how-it-works)
- [Compiling a policy](#compiling-a-policy)
- [Running it](#running-it)
- [The manifest](#the-manifest)
- [Build](#build)
- [Limits of this first version](#limits-of-this-first-version)

## How it works

`model_load` detects `general.architecture = "ggml_compiler"` and dispatches to
the `compiled` arch (`src/models/compiled.cpp`). That arch:

1. picks the backend the way every arch does (`src/backend.h`: the compiled-in
   accelerator, else CPU) and loads the program onto it through ggml-compiler's
   runtime, `libggml_compiler_rt`, which this tree builds against its own ggml
   (one ggml per process, so every backend vla.cpp has is available to compiled
   models);
2. reads the `vla` manifest from the GGUF's metadata and checks it against the
   program: every input has a role, the actions output exists and has
   `chunk x dim` values;
3. fills `Config` from the manifest, so `vla-server`'s request validation, the
   protobuf, the C API and the Python bindings work unchanged;
4. on each `predict`, does only what the manifest describes: image resize and
   normalization, token padding, state normalization, noise sampling, the run,
   and action un-normalization.

The runtime checks every node of every graph against the backend at load, so an
op the backend cannot run is a named load error, not a wrong action.

## Compiling a policy

In a ggml-compiler checkout, following its
[export guide](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/EXPORT.md)
and [serving guide](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/SERVING.md):

```bash
uv run ggml-compiler compile examples/act/export.py:program --out out/act --emit gguf
uv run ggml-compiler verify out/act/model.gguf --backend cpu
```

The export script attaches the manifest with `Program(metadata={"vla": ...})`,
built by `ggml_compiler.vla`; `compile` refuses a manifest that does not cover
the program's inputs. Reduced-precision weights come from a ggml-compiler dtype
policy (`--policy`), not from `--weight-dtype`, which the compiled arch rejects.

## Running it

```bash
./build/vla-cli --ckpt out/act/model.gguf --image front.jpg --image wrist.jpg --state 0,0,0,0,0,0 --pretty
./build/vla-server out/act/model.gguf
```

The load prints what the manifest declares:

```
vla: arch = compiled (ggml-compiler)
vla(compiled): ggml-compiler 0.1.0 on CPU; 2 camera(s), state, chunk 100 x 6 action dims
vla(compiled): view 0 'observation.images.front' 640x480
vla(compiled): view 1 'observation.images.wrist' 640x480
```

Cameras go in manifest order. A compiled program carries no tokenizer: pass
`--tokens` to `vla-cli`, and token ids in `PredictRequest.lang_tokens` as every
client already does. Images arrive at the compiled size unless the manifest
says `fit: resize`.

## The manifest

Version 1, stored as JSON under `vla` in the GGUF key `ggml_compiler.meta`.
The full reference is ggml-compiler's
[SERVING.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/SERVING.md#manifest-reference);
in short:

| Role | Program tensor | vla.cpp does |
|---|---|---|
| `images[i]` | `[1, 3, h, w]` float | request view i, scaled to [0, 1], `(x - mean) / std` per channel, CHW. `fit: exact` refuses another size, `resize` scales bilinearly. An optional `mask` input is 1 when the view was sent. |
| `lang` | `tokens` `[1, max_len]` int, optional `mask` | token ids padded with `pad_id`; `Config::n_lang = max_len` |
| `state` | `[1, dim]` float | `real_dim` values normalized (`mean_std`: LeRobot's `(x - mean) / (std + eps)`; `min_max`; or `none`), the rest zero |
| `noise` | any float input | the request's noise, or N(0, 1) |
| `actions` | `[1, chunk, dim]` float | un-normalized over `real_dim` columns; `Config::n_suffix = chunk`, `max_action_dim = dim` |

Inputs are converted to the dtype the program declares (f32, f16 or i32), so a
0/1 mask may be either float or int in the program.

## Build

`VLA_COMPILED` (default ON) fetches ggml-compiler's `runtime/` at the ref in
`VLA_GGML_COMPILER_TAG` and builds it against this tree's ggml with
`GGML_COMPILED_EXTERNAL_GGML`; the runtime refuses a llama.cpp tag other than
the one it was tested with, which is the tree's `VLA_LLAMA_TAG`. `-DVLA_COMPILED=OFF`
drops the dependency; a compiled GGUF then fails to load with a message saying
so. To build against a local ggml-compiler checkout:

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_GGML_COMPILER=/path/to/ggml-compiler
```

Unit test: `tests/test_compiled_manifest.cpp` (ctest `compiled_manifest`).

## Limits of this first version

- Programs with symbolic dims are refused; compile with fixed shapes and pad.
- Only the default entry point is served; a multi-entry program (a KV-cached
  VLM) runs its first entry.
- No tokenizer and no prompt template: clients send token ids.
- `precomputed_img_emb` and `encode_images` are not supported.
- `--weight-dtype`, `--act-dtype` and `--num-steps` do not apply; those are
  compile-time choices.
- Image fitting is exact or bilinear resize; letterboxing (LeRobot's
  `resize_with_pad`) is not implemented yet.
- vla.cpp's fused CUDA and FoldQuant paths reach ggml through its extension
  hooks and are not used by compiled graphs.
