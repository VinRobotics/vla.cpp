from typing import Any
import statistics
import time

import torch
import numpy as np

from policies.gr00t_n15.groot.modeling_groot import GrootPolicy
from policies.torch_compile import maybe_compile
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors


class GR00TN15PolicyPipeline:
    arch_type: str = "gr00t-n15"

    def __init__(
        self,
        model_id: str = "liorbenhorin-nv/groot-libero_object-64_40000",
        device: torch.device | str = None,
        n_action_steps: int | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.device = device
        self._pre_ms: list[float] = []
        self._fwd_ms: list[float] = []
        self._post_ms: list[float] = []

        # The 2.4B-param model loads in fp32 by default (~9.6 GB) and lerobot's
        # from_pretrained moves it straight to config.device — that OOMs an 8 GB
        # GPU. Load on CPU, then cast to bf16 (this model already computes under
        # torch.autocast(bf16), so bf16-resident weights are the same precision
        # at ~4.8 GB) and move to the target device. Pinning the config to "cpu"
        # also avoids the transient full-checkpoint-on-GPU spike during loading.
        cfg = PreTrainedConfig.from_pretrained(model_id)
        cfg.device = "cpu"
        self._policy = GrootPolicy.from_pretrained(
            model_id, config=cfg, strict=False
        ).eval()
        self._policy = self._policy.to(dtype=torch.bfloat16, device=device)
        self._policy.config.device = str(device)

        # Chunk replay. lerobot policies queue `config.n_action_steps` actions
        # from each predicted chunk and pop one per `select_action`, which is
        # exactly vla.cpp's client-side --n-action-steps. Override it so both
        # stacks run the same control cadence.
        if n_action_steps is not None:
            print(
                f"[gr00t_n15] n_action_steps: {self._policy.config.n_action_steps} -> {n_action_steps}",
                flush=True,
            )
            self._policy.config.n_action_steps = n_action_steps
            # GrootPolicy.__init__ already built _action_queue as
            # deque(maxlen=config.n_action_steps) with the checkpoint's value,
            # and its select_action extends that deque with the *full* horizon
            # (unlike lerobot's pi0, which slices to n_action_steps first). So
            # the override alone does not take effect — the stale maxlen keeps
            # queueing actions and most select_action calls become cheap pops.
            # reset() rebuilds the deque against the value we just set.
            self._policy.reset()
            print(
                f"[gr00t_n15] action queue maxlen: {self._policy._action_queue.maxlen}",
                flush=True,
            )

        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config, model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        # Optional torch.compile of the Eagle backbone + flow-matching head.
        # GrootPolicy is a thin lerobot wrapper (queueing, normalization) around
        # _groot_model, which is where every FLOP is spent — so that submodule
        # is the compile target, not the wrapper.
        self._policy._groot_model = maybe_compile(
            self._policy._groot_model, tag="gr00t_n15._groot_model",
        )

    def reset(self):
        self._policy.reset()
        return {}

    def select_action(self, observations: dict[str, Any]) -> np.ndarray:
        # Phase split, printed periodically below. No CUDA syncs here on
        # purpose: adding them would serialize work that normally overlaps and
        # inflate the total this is meant to explain. `pre` is pure host work
        # so it is exact; `fwd` only counts kernel launches, and the GPU wait
        # lands in `post`, whose first act is a device->host copy.
        t0 = time.perf_counter()

        # Convert any numpy arrays in observations to torch tensors
        for key in observations:
            if isinstance(observations[key], np.ndarray):
                observations[key] = torch.from_numpy(observations[key])

        # Run the policy
        batch = self._preprocess(observations)
        t1 = time.perf_counter()
        with torch.inference_mode():
            pred_action = self._policy.select_action(batch)
        t2 = time.perf_counter()
        pred_action = self._postprocess(pred_action)

        # Return the action as a numpy array
        out = pred_action[0].cpu().numpy()
        t3 = time.perf_counter()

        self._pre_ms.append(1000.0 * (t1 - t0))
        self._fwd_ms.append(1000.0 * (t2 - t1))
        self._post_ms.append(1000.0 * (t3 - t2))
        n = len(self._pre_ms)
        if n % 50 == 0:
            def q(xs):
                s = sorted(xs)
                return (statistics.fmean(s), s[len(s) // 2], s[int(0.95 * (len(s) - 1))], s[0], s[-1])
            for name, xs in (("pre", self._pre_ms), ("fwd", self._fwd_ms), ("post", self._post_ms)):
                m, md, p95, lo, hi = q(xs[-50:])
                print(f"[gr00t_n15] call={n} {name:4s} mean={m:7.2f} med={md:7.2f} "
                      f"p95={p95:7.2f} min={lo:7.2f} max={hi:7.2f}", flush=True)
        return out

    def get_arch(self) -> dict[str, str]:
        return {"arch": self.arch_type}
