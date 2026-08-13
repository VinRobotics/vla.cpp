from typing import Any
import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from contextlib import nullcontext
from torchvision.transforms import ToTensor
from huggingface_hub import snapshot_download

from policies import POLICY_DIR
from policies.evo1.policy import EVO1Policy
from policies.evo1.normalizer import EVO1Normalizer
from policies.torch_compile import maybe_compile


class EVO1PolicyPipeline:
    arch_type: str = "evo1"

    def __init__(
        self, 
        model_id: str = "MINT-SJTU/Evo1_LIBERO",
        device: str | torch.device | None = None,
        n_action_steps: int | None = None,
    ) -> None:
        model_name = model_id.split("/")[-1]
        ckpt_dir = POLICY_DIR / model_name
        if not ckpt_dir.exists():
            ckpt_dir = snapshot_download(
                repo_id=model_id,
                local_dir=str(ckpt_dir),
                local_dir_use_symlinks=False
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Chunk replay. EVO1Policy keeps its own deque and re-predicts once
        # n_action_steps of the horizon-50 chunk have been consumed — the same
        # cadence as vla.cpp's client-side --n-action-steps.
        self._policy, self._normalizer = self.load_policy_and_normalizer(
            ckpt_dir, self.device, n_action_steps
        )
        if n_action_steps is not None:
            print(f"[evo1] n_action_steps={self._policy.n_action_steps}", flush=True)
        self.to_tensor = ToTensor()

        # Optional torch.compile. Evo-1 splits cleanly into two hot spots: the
        # InternVL3 embedder (one forward per prediction) and the flow-matching
        # action head (num_inference_timesteps=50 forwards per prediction). The
        # head is the dominant cost and the better compile target, but both are
        # wrapped so the compiled variant is not measuring a half-compiled model.
        self._policy.action_head = maybe_compile(
            self._policy.action_head, tag="evo1.action_head",
        )
        # The embedder is driven through submodule calls — extract_feature()
        # invokes vision_model(...) and the wrapper then calls language_model(...)
        # directly — so InternVL3's own forward() never runs. Compiling the top
        # level module would be a silent no-op; these two are the calls that
        # actually execute.
        self._policy.embedder.model.vision_model = maybe_compile(
            self._policy.embedder.model.vision_model, tag="evo1.embedder.vision_model",
        )
        self._policy.embedder.model.language_model = maybe_compile(
            self._policy.embedder.model.language_model, tag="evo1.embedder.language_model",
        )

    def get_arch(self) -> dict[str, str]:
        return {"arch": self.arch_type}
    
    def reset(self):
        self._policy.reset()
        return {}

    def select_action(self, observations: dict[str, Any]) -> np.ndarray:
        images = [self.decode_image_from_list(img) for img in observations["image"]]
        assert len(images) == 3, "Must provide exactly 3 images."

        state = torch.as_tensor(
            observations["state"], dtype=torch.float32, device=self.device
        )
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[1] < 24:
            state = torch.cat([
                state, torch.zeros((1, 24 - state.shape[1]), device=self.device)
            ], dim=1)
        norm_state = self._normalizer.normalize_state(state).to(dtype=torch.float32)
        
        prompt = observations["prompt"]
        image_mask = torch.tensor(observations["image_mask"], dtype=torch.int32, device=self.device)
        action_mask = torch.tensor([observations["action_mask"]],dtype=torch.int32, device=self.device)

        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available() and str(self.device).startswith("cuda")
            else nullcontext()
        )

        with autocast_context:
            action = self._policy.run_inference(
                images=images,
                image_mask=image_mask,
                prompt=prompt,
                state_input=norm_state,
                action_mask=action_mask
            )
            action = self._normalizer.denormalize_action(action.view(-1))[0]
        return action.cpu().numpy()
    
    def decode_image_from_list(
        self, img_list: list[np.ndarray]
    ) -> torch.Tensor:
        img_array = np.array(img_list, dtype=np.uint8)
        img = cv2.resize(img_array, (448, 448))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        return self.to_tensor(pil).to(self.device)

    @staticmethod
    def load_policy_and_normalizer(
        ckpt_dir: str, device: str | torch.device, n_action_steps: int | None = None
    ) -> tuple[EVO1Policy, EVO1Normalizer]:
        config = json.load(open(os.path.join(ckpt_dir, "config.json")))
        stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

        config["finetune_vlm"] = False
        config["finetune_action_head"] = False
        config["num_inference_timesteps"] = 32

        policy = EVO1Policy(config, n_action_steps=n_action_steps).eval()
        ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

        checkpoint = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
        policy.load_state_dict(checkpoint["module"], strict=True)
        policy = policy.to(device)

        normalizer = EVO1Normalizer(stats)
        return policy, normalizer
