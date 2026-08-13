import os
from typing import Any
import torch
import numpy as np

from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors

from policies.torch_compile import maybe_compile


class PI0PolicyPipeline:
    arch_type: str = "pi0"
    # Some PI0 finetunes only store lm_head for tied language embeddings.
    # Non-strict load avoids hard-failing when embed_tokens is omitted.

    def __init__(
        self,
        model_id: str = "lerobot/pi0_libero_finetuned_v044",
        device: torch.device | str = None,
        n_action_steps: int | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.device = device

        # Pin the device on the config *before* construction: some policies (e.g.
        # PI0Policy) call `self.model.to(config.device)` inside __init__, so a later
        # `.to(device)` is too late — the model is already on the config's default
        # ("cuda"), which OOMs on small GPUs.
        self._policy = PI0Policy.from_pretrained(
            model_id, device=device
        ).eval()
        # Chunk replay. lerobot policies queue `config.n_action_steps` actions
        # from each predicted chunk and pop one per `select_action`, which is
        # exactly vla.cpp's client-side --n-action-steps. Override it so both
        # stacks run the same control cadence.
        if n_action_steps is not None:
            print(
                f"[pi0] n_action_steps: {self._policy.config.n_action_steps} -> {n_action_steps}",
                flush=True,
            )
            self._policy.config.n_action_steps = n_action_steps

        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config, model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        # PaliGemma ties `language_model.embed_tokens.weight` with `lm_head.weight`.
        # Some finetuned checkpoints (e.g. ours) save only the lm_head copy, so
        # lerobot's safetensors loader leaves embed_tokens at random init — which
        # silently produces garbage actions. Re-tie here.
        paligemma = self._policy.model.paligemma_with_expert.paligemma
        embed = paligemma.model.language_model.embed_tokens.weight
        lm_head = paligemma.lm_head.weight
        if embed.shape == lm_head.shape and not torch.equal(embed, lm_head):
            paligemma.model.language_model.embed_tokens.weight = paligemma.lm_head.weight
            print("PI0PolicyPipeline: retied paligemma embed_tokens.weight <- lm_head.weight", flush=True)

        # Optional torch.compile of the inference hot path — the PaliGemma +
        # action-expert stack, run once for the prefix KV cache and once per
        # flow-matching denoise step. Compiled *after* the re-tie above so the
        # weight surgery happens on the eager module.
        self._policy.model.paligemma_with_expert = maybe_compile(
            self._policy.model.paligemma_with_expert, tag="pi0.paligemma_with_expert",
        )

        # VLA_PROFILE=1 splits the forward into prefix (vision tower + PaliGemma
        # prefill) vs the flow-matching denoise loop, so the reference can be
        # compared against vla.cpp's own vision/inference breakdown. CUDA is
        # async, so each stage is synchronized before being timed.
        if os.environ.get("VLA_PROFILE") == "1":
            self._install_profiling()

    def _install_profiling(self) -> None:
        import time
        model = self._policy.model
        pwe = model.paligemma_with_expert
        state = {"prefix": 0.0, "denoise": 0.0, "steps": 0,
                 "vit": 0.0, "vit_calls": 0, "fwd": 0.0, "fwd_calls": 0}

        def timed(name, fn, counter=None):
            def wrapper(*a, **kw):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out = fn(*a, **kw)
                torch.cuda.synchronize()
                state[name] += time.perf_counter() - t0
                if counter:
                    state[counter] += 1
                return out
            return wrapper

        model.embed_prefix = timed("prefix", model.embed_prefix)
        model.denoise_step = timed("denoise", model.denoise_step, "steps")
        # embed_image is the SigLIP tower; forward is the shared PaliGemma+expert
        # stack, called once for the prefix KV cache and once per denoise step.
        pwe.embed_image = timed("vit", pwe.embed_image, "vit_calls")
        pwe.forward = timed("fwd", pwe.forward, "fwd_calls")
        self._prof_state = state

    def reset(self):
        self._policy.reset()
        return {}

    def select_action(self, observations: dict[str, Any]) -> np.ndarray:
        # Convert any numpy arrays in observations to torch tensors
        for key in observations:
            if isinstance(observations[key], np.ndarray):
                observations[key] = torch.from_numpy(observations[key])

        prof = getattr(self, "_prof_state", None)
        if prof is not None:
            import time as _t
            for k in prof:
                prof[k] = 0 if isinstance(prof[k], int) else 0.0
            torch.cuda.synchronize(); _t0 = _t.perf_counter()

        # Run the policy
        batch = self._preprocess(observations)
        if prof is not None:
            torch.cuda.synchronize(); _t1 = _t.perf_counter()
        with torch.inference_mode():
            pred_action = self._policy.select_action(batch)
        pred_action = self._postprocess(pred_action)

        if prof is not None:
            torch.cuda.synchronize()
            total = _t.perf_counter() - _t0
            pre = _t1 - _t0
            # forward() is called once for the prefix KV cache and once per
            # denoise step; the denoise calls are already counted separately, so
            # attribute the remainder of forward() time to the prefill.
            prefill = prof["fwd"] - prof["denoise"] if prof["fwd"] > prof["denoise"] else prof["fwd"]
            print(f"[pi0-prof] total={1000*total:.1f} preprocess={1000*pre:.1f} "
                  f"vit={1000*prof['vit']:.1f}({prof['vit_calls']}) "
                  f"embed_prefix={1000*prof['prefix']:.1f} "
                  f"fwd_total={1000*prof['fwd']:.1f}({prof['fwd_calls']}) "
                  f"prefill~{1000*prefill:.1f} "
                  f"denoise={1000*prof['denoise']:.1f}({prof['steps']})", flush=True)

        # Return the action as a numpy array
        return pred_action[0].cpu().numpy()

    def get_arch(self) -> dict[str, str]:
        return {"arch": self.arch_type}
