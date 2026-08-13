from typing import Any
import torch
import numpy as np

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

from policies.torch_compile import maybe_compile


class SmolVLAPolicyPipeline:
    arch_type: str = "smolvla"

    def __init__(
        self,
        model_id: str = "HuggingFaceVLA/smolvla_libero",
        device: torch.device | str = None,
        n_action_steps: int | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.device = device

        self._policy = SmolVLAPolicy.from_pretrained(
            model_id, device=device
        ).eval()

        # Chunk replay. lerobot policies queue `config.n_action_steps` actions
        # from each predicted chunk and pop one per `select_action`, which is
        # exactly vla.cpp's client-side --n-action-steps. Override it so both
        # stacks run the same control cadence.
        if n_action_steps is not None:
            print(
                f"[smolvla] n_action_steps: {self._policy.config.n_action_steps} -> {n_action_steps}",
                flush=True,
            )
            self._policy.config.n_action_steps = n_action_steps

        # Optional torch.compile of the core transformer (the SmolVLMWithExpert
        # forward is the inference hot path — called once for the prefix KV
        # cache and once per flow-matching denoise step). See policies/
        # torch_compile.py for the env-var contract.
        self._policy.model.vlm_with_expert = maybe_compile(
            self._policy.model.vlm_with_expert, tag="smolvla.vlm_with_expert",
        )

        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config, model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

    def reset(self):
        self._policy.reset()
        return {}

    def select_action(self, observations: dict[str, Any]) -> np.ndarray:
        # Convert any numpy arrays in observations to torch tensors
        for key in observations:
            if isinstance(observations[key], np.ndarray):
                observations[key] = torch.from_numpy(observations[key])

        # Run the policy
        batch = self._preprocess(observations)
        with torch.inference_mode():
            pred_action = self._policy.select_action(batch)
        pred_action = self._postprocess(pred_action)

        # Return the action as a numpy array
        return pred_action[0].cpu().numpy()

    def get_arch(self) -> dict[str, str]:
        return {"arch": self.arch_type}
