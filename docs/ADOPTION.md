# Adoption notes

The engine works; the gap is distribution.

Done:

1. **C ABI.** `include/vla.h` and `libvla`. `src/model.h` is C++ only, so
   without it nothing outside C++ can link the engine.
2. **Python bindings.** `bindings/python`, ctypes over the ABI.
   `pip install ./bindings/python` builds a self-contained wheel with
   scikit-build-core.
3. **Prebuilt binaries.** `.github/workflows/release.yml` publishes on tag:
   linux-x86_64 (CPU, CUDA 12.8, CUDA 13.4), linux-aarch64 (CPU, and CUDA 13.4
   for Orin, Thor and DGX Spark), macos-arm64-metal, and a Docker image. The
   tarballs now run off the build machine: shared libraries ship next to the
   binaries with an `$ORIGIN` rpath, builds use `GGML_NATIVE=OFF`, and CI runs
   `vla-cli --help` with the build tree moved away. `vla-server` still needs
   `libzmq5` from the system.
4. **Install rules.** `cmake --install` gives a relocatable tree with a
   `$ORIGIN/../lib` rpath.
5. **One-command model fetch.** `-hf user/repo[:path/file.gguf|:tag]` on
   `vla-cli`, `vla-server` and `vla-bench`, cached under `$VLA_CACHE`. A repo
   with several GGUFs lists them instead of guessing.
6. **Reproducible benchmarks.** `vla-bench` emits the README table rows.
7. **Contributor path.** `CONTRIBUTING.md` has the six-site walkthrough for
   adding an architecture, plus issue and PR templates.
8. **Instruction in, action out.** `vla-cli --text` builds each arch's real
   prompt. Octo carries its tokenizer in the GGUF, and
   `scripts/add_tokenizer_to_gguf.py` adds one to pi0, pi0.5 and OpenVLA-OFT, so
   those need no Python at run time. The other archs call
   `scripts/tokenize_prompt.py`.

Left:

- **Jetson on JetPack 6.** The tarballs are built on Ubuntu 24.04 and its glibc,
  and the aarch64 CUDA one needs a CUDA 13 driver, so JetPack 6 Orins still
  build on the device.
- **PyPI.** The wheel builds from `bindings/python`, but nothing publishes it.
- **Hugging Face library registration.** The `vrfai` model cards set
  `library_name: vla.cpp`, but vla.cpp is not registered in huggingface.js, so
  the Hub shows no "Use this model" snippet.
- **Windows zips and a Homebrew formula.** Windows builds from source only, and
  there is no brew tap, though the install rules now make one possible.
- **`ci/baselines/rtx3090.json`** still disagrees with the README table, which is
  now RTX 5090 numbers from `vla-bench`. Re-record the baselines on one machine.
- **Success rates.** The README table comes from a May 2026 RTX 3060 sweep and
  covers seven of the thirteen archs, and π0, SmolVLA and GR00T N1.7 have moved
  since. A fresh sweep would cover the rest.

None of these change inference behaviour.
