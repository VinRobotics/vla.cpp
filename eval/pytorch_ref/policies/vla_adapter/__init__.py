"""VLA-Adapter golden PyTorch policy pipeline (self-contained; no dependency on
`third_party/VLA-Adapter`). Mirrors the `policies/evo1` convention:
`get_arch` / `reset` / `select_action`. The model lives in `model.py` and is
numerically pinned to the upstream `predict_action` (verify_ref.py, ~1e-6).
"""

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from policies.vla_adapter.model import VLAAdapter, PROPRIO_DIM

# use_minivlm chat-template prompt (run_libero_eval / openvla_utils).
PROMPT_TPL = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
    "assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to "
    "{instruction}?<|im_end|>\n<|im_start|>assistant\n"
)
# per-tower normalisation (preprocessor_config.json): DINOv2 (imagenet) + SigLIP (0.5).
DINO_MEAN = np.array([0.484375, 0.455078125, 0.40625], np.float32)
DINO_STD = np.array([0.228515625, 0.2236328125, 0.224609375], np.float32)
SIG_MEAN = np.array([0.5, 0.5, 0.5], np.float32)
SIG_STD = np.array([0.5, 0.5, 0.5], np.float32)
IMG = 224


class VLAAdapterPolicyPipeline:
    arch_type: str = "vla_adapter"

    def __init__(self, ckpt_dir: str = "/home/khanhnd61/data/VLA-Adapter/LIBERO-Object-Pro",
                 device: str | torch.device | None = None, dtype: torch.dtype | None = None,
                 unnorm_key: str = "libero_object_no_noops") -> None:
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # match the real deployment: action head runs bf16 on GPU; f32 on CPU.
        self.dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.unnorm_key = unnorm_key
        self.model = VLAAdapter.from_checkpoint(ckpt_dir, device=self.device, dtype=self.dtype)
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir)

    def get_arch(self) -> dict[str, str]:
        return {"arch": self.arch_type}

    def reset(self):
        return {}

    @staticmethod
    def _to_hwc_u8(img) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 3:          # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:                        # float [0,1] -> u8
            img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return img

    def _preprocess_view(self, img) -> np.ndarray:
        """RGB image -> [6, 224, 224] = [dino3, siglip3] (resize 224 + center_crop 0.9)."""
        hwc = self._to_hwc_u8(img)
        if hwc.shape[:2] != (IMG, IMG):
            hwc = np.array(Image.fromarray(hwc, "RGB").resize((IMG, IMG), Image.LANCZOS), np.uint8)
        # center_crop crop_scale=0.9 (prepare_images_for_vla, cfg.center_crop=True)
        s = 0.9 ** 0.5
        nh, nw = int(round(IMG * s)), int(round(IMG * s))
        oh, ow = (IMG - nh) // 2, (IMG - nw) // 2
        hwc = np.array(Image.fromarray(hwc[oh:oh + nh, ow:ow + nw], "RGB").resize((IMG, IMG), Image.BICUBIC), np.uint8)
        x = hwc.astype(np.float32) / 255.0
        dino = ((x - DINO_MEAN) / DINO_STD).transpose(2, 0, 1)   # [3,224,224]
        sig = ((x - SIG_MEAN) / SIG_STD).transpose(2, 0, 1)
        return np.concatenate([dino, sig], axis=0)               # [6,224,224]

    @torch.no_grad()
    def select_action(self, observations: dict[str, Any]) -> np.ndarray:
        views = observations.get("image") or [observations[k] for k in ("full_image", "wrist_image") if k in observations]
        pixel = np.concatenate([self._preprocess_view(v) for v in views], axis=0)[None]   # [1, 6*n, 224, 224]
        pixel = torch.from_numpy(pixel)

        state = observations.get("state", observations.get("observation.state"))
        proprio = np.asarray(state.numpy() if isinstance(state, torch.Tensor) else state, np.float32).reshape(-1)[:PROPRIO_DIM]

        task = observations.get("prompt", observations.get("task", ""))
        if isinstance(task, bytes):
            task = task.decode()
        ids = self.tok(PROMPT_TPL.format(instruction=task.lower()), add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long)

        actions, _ = self.model.predict_action(input_ids, pixel, proprio, unnorm_key=self.unnorm_key)
        actions = np.asarray(actions, np.float32)                # [chunk, 7]
        # openvla gripper post-process: binarize sign(2x-1) then invert.
        actions[:, -1] = np.sign(2.0 * actions[:, -1] - 1.0) * -1.0
        return actions
