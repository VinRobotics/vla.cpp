# vla-cpp

Python bindings for [vla.cpp](https://github.com/VinRobotics/vla.cpp) over its
C ABI (`include/vla.h`). No PyTorch at inference time.

```python
import vla_cpp

model = vla_cpp.load("smolvla-libero.gguf")
actions = model.predict(frame_hwc_uint8, tokens=[1, 100, 200, 2])
```

`predict` returns `[rows, max_action_dim]`; only the first
`model.config.real_action_dim` columns carry values. `model.config.denormalized`
says whether they are already in world units.

## Finding the library

Wheels bundle `libvla.so`. From a source checkout, build it and point at it:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target vla
VLA_LIBRARY=build/libvla.so LD_LIBRARY_PATH=build:build/bin python your_script.py
```

`LD_LIBRARY_PATH` is needed because `libvla.so` links `libvla_core.so` and the
ggml libraries from the same build tree.

## API

| | |
|---|---|
| `load(ckpt, mmproj=None, config=None)` | `mmproj` only for SmolVLA, pi0, pi0.5 |
| `Model.predict(images, tokens, state=None, noise=None, ...)` | `images` is one HWC array or a sequence |
| `Model.config` | resolved hyper-parameters |
| `Model.last_stats()` | per-phase timings, needs `timing=TIMING_PHASE` |
| `Model.close()` | or use as a context manager |

Output is bit-identical to `vla-cli` on the same inputs.
