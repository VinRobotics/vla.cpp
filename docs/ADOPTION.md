# Adoption notes

vla.cpp is technically solid (7 architectures, self-contained GGUFs, CUDA + Jetson,
real benchmarks). The gap to llama.cpp-style reach is mostly distribution, not code.
Ordered by leverage:

1. **Publish on GitHub.** The repo lives on Bitbucket (`bitbucket.org/vinrobotics/vla.cpp`)
   while the README already links a `github.com/VinRobotics/vla.cpp` URL. llama.cpp's reach
   came from GitHub visibility, issues, and PRs. A public GitHub mirror is the single biggest
   lever; nothing else here matters as much.

2. **Ship the models.** All seven GGUFs are already published under
   [`vrfai`](https://huggingface.co/vrfai) on the Hub - the README's "coming soon" rows are
   stale (now fixed). Keep the model table pointing at the real repos so the policies are
   one `hf download` away.

3. **Cut releases.** `v0.1.0` is the first tag (see `CHANGELOG.md`). Tagged releases +
   changelog give users something to pin and cite.

4. **Rotate the committed credential.** The local `.git/config` remote URL embeds an
   access token (`https://<token>@bitbucket.org/...`). It is never pushed (git config is not
   tracked), so this is hygiene, not a live leak - but rotate it and use a credential helper
   or SSH remote instead of an inline token.

5. **Lower the build bar (optional).** A CUDA `Dockerfile` now exists; publishing a prebuilt
   image (and, later, macOS/Metal or ROCm backends) removes the from-source step that stops
   most drive-by users.

None of these change inference behaviour; they change who can find and run it.
