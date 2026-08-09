# Adoption notes

The engine works; the gap is distribution.

Done:

1. **C ABI.** `include/vla.h` and `libvla`. `src/model.h` is C++ only, so
   without it nothing outside C++ can link the engine.
2. **Python bindings.** `bindings/python`, ctypes over the ABI.
3. **Prebuilt binaries.** `.github/workflows/release.yml` publishes
   linux-x86_64 (CPU and CUDA), linux-aarch64 (CPU, for Jetson-class boards),
   macos-arm64-metal and a Docker image on tag.
4. **One-command model fetch.** `-hf user/repo[:file.gguf]` on `vla-cli`,
   `vla-server` and `vla-bench`, cached under `$VLA_CACHE`.
5. **Reproducible benchmarks.** `vla-bench` emits the README table rows.
6. **Contributor path.** `CONTRIBUTING.md` has the six-site walkthrough for
   adding an architecture, plus issue and PR templates.
7. **Instruction in, action out.** `vla-cli --text` tokenizes with the
   architecture's own tokenizer, so the quickstart no longer needs raw ids.

Left:

- **Jetson CUDA binaries.** The aarch64 job is CPU only: the hosted arm64 image
  carries no CUDA, so a Jetson GPU build still happens on the device.
- **PyPI.** The wheel is built from `bindings/python` but nothing publishes it.
- **`ci/baselines/rtx3090.json`** still disagrees with the README table, which is
  now RTX 5090 numbers from `vla-bench`. Re-record the baselines on one machine.
- **Success rates.** The README table comes from a May 2026 RTX 3060 sweep and
  covers seven of the eleven archs. A fresh sweep would cover the rest.

None of these change inference behaviour.
