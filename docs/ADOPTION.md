# Adoption notes

The engine works; the gap is distribution. Roughly in priority order.

1. **C ABI.** Done: `include/vla.h` and `libvla`. `src/model.h` is C++ only, so
   without it nothing outside C++ can link the engine.

2. **Python bindings.** Done: `bindings/python`. Robotics runs on Python; the
   only other way in is the ZeroMQ server plus a hand-written client.

3. **Prebuilt binaries.** `.github/workflows/build.yml` already builds
   `vla-server`, `vlm-server` and `vla-cli` and uploads nothing. Tagged
   artifacts (linux x86-64 CPU/CUDA, linux aarch64 Jetson, macOS arm64 Metal)
   plus a published Docker image remove the build step.

4. **One-command model fetch.** Today: install `huggingface_hub`, run
   `hf download`, pass a path. llama.cpp solved this with `-hf user/repo`. The
   GGUFs are already on the Hub under [`vrfai`](https://huggingface.co/vrfai).

5. **Reproducible benchmarks.** The README latency table has no in-repo source
   and disagrees with `ci/baselines/rtx3090.json`. A `vla-bench` that emits the
   table, with quantization and memory columns, makes it checkable.

6. **Contributor path.** Adding an architecture touches six sites, none written
   down: the `Arch` enum and factory in `src/arch.h`, the key list, string map
   and switch in `src/model.cpp`, and `CMakeLists.txt`.

None of these change inference behaviour.
