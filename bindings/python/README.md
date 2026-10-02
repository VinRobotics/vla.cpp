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

## Installing

From a checkout of the repo:

```bash
pip install ./bindings/python
```

This builds `libvla.so` with CMake and puts it inside the package, so nothing
else needs to be on the library path. You need a C++17 compiler and network
access (CMake fetches llama.cpp). The build is CPU only (Metal on macOS). It
uses `GGML_NATIVE=OFF`, so the wheel runs on other machines (on x86 it needs
AVX2).
`pip wheel ./bindings/python -w dist` gives you the wheel file. With
`python -m build`, pass `--wheel`: the sdist does not include the C++ sources.

## Using your own build

Point `VLA_LIBRARY` at a `libvla.so` from a normal CMake build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target vla
VLA_LIBRARY=build/libvla.so python your_script.py
```

`cmake --install build --prefix <dir>` gives a relocatable copy that does not
need the build tree. The libraries go to `<dir>/lib`, or `<dir>/lib64` on
Fedora and RHEL.

## API

| | |
|---|---|
| `load(ckpt_path, mmproj_path=None, config_path=None)` | every arch ignores `mmproj_path`; the `runtime` block of `config_path` sets the precision options |
| `Model.predict(images, tokens, state=None, noise=None, ...)` | `images` is one HWC array or a sequence |
| `Model.config` | resolved hyper-parameters |
| `Model.last_stats()` | per-phase timings, needs `timing=TIMING_PHASE` |
| `Model.close()` | or use as a context manager |

Output is bit-identical to `vla-cli` on the same inputs.
