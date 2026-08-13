import os
from types import SimpleNamespace
from typing import List, Union, Tuple
from PIL import Image

import torch
import torch.nn as nn
from collections import deque

from policies.evo1.model.internvl3.internvl3_embedder import InternVL3Embedder
from policies.evo1.model.action_head.flow_matching import FlowmatchingActionHead


class EVO1Policy(nn.Module):
    def __init__(self, config: dict, n_action_steps: int = None):
        super().__init__() 
        self.config = config
        self._device = config.get("device", "cuda")
        self.return_cls_only = config.get("return_cls_only", False)
        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        self.embedder = InternVL3Embedder(model_name=vlm_name, device=self._device)

        action_head_type = config.get("action_head", "flowmatching").lower()
        
        if action_head_type == "flowmatching":
           
            horizon = config.get("action_horizon", config.get("horizon", 16))
            per_action_dim = config.get("per_action_dim", 7)
            action_dim = horizon * per_action_dim
            
            config["horizon"] = horizon
            config["per_action_dim"] = per_action_dim
            config["action_dim"] = action_dim
            
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            self.action_head = FlowmatchingActionHead(config=SimpleNamespace(
                embed_dim=config.get("embed_dim", 896),    
                hidden_dim=config.get("hidden_dim", 1024),
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                state_dim=config.get("state_dim", 7),
                state_hidden_dim=config.get("state_hidden_dim", 1024),
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 8),
                dropout=config.get("dropout", 0.0),
                num_inference_timesteps=config.get("num_inference_timesteps", 50),
                num_categories=config.get("num_categories", 1)
            )).to(self._device)
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")
        
        self.n_action_steps = n_action_steps or config.get("n_action_steps", 14)
        assert self.n_action_steps <= self.horizon, "n_action_steps must be less than or equal to the horizon length."
        self.reset()
    
    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = deque(maxlen=self.horizon)

    def get_vl_embeddings(
        self,
        images: List[Image.Image],
        image_mask: torch.Tensor,  
        prompt: str = "",
        return_cls_only: Union[bool, None] = None
    ) -> torch.Tensor:
        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image (PIL.Image). Got `images=None` or empty list.")
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")
        
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if actions_gt is None:
            return self.action_head.get_action(fused_tokens, state=state, action_mask=action_mask, embodiment_id=embodiment_ids)
        else:
            return self.action_head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids)

    def forward(
        self, 
        fused_tokens: torch.Tensor, 
        state: torch.Tensor = None, 
        actions_gt: torch.Tensor = None, 
        action_mask: torch.Tensor = None, 
        embodiment_ids: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if not self._check_get_actions_condition():
            return self._queues.popleft()
        # VLA_PROFILE=1 splits the forward into its two hot stages (InternVL3
        # embedder vs flow-matching action head) so the PyTorch reference can be
        # compared against vla.cpp's own vision/inference breakdown. CUDA is
        # async, so each stage is synchronized before it is timed.
        _prof = os.environ.get("VLA_PROFILE") == "1"
        if _prof:
            import time as _t
            torch.cuda.synchronize(); _t0 = _t.perf_counter()
        fused_tokens = self.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=return_cls_only
        )
        if _prof:
            torch.cuda.synchronize(); _t1 = _t.perf_counter()
        state_tensor = self.prepare_state(state_input)
        actions = self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)
        if _prof:
            torch.cuda.synchronize(); _t2 = _t.perf_counter()
            print(f"[evo1-prof] embedder={1000*(_t1-_t0):.1f}ms "
                  f"action_head={1000*(_t2-_t1):.1f}ms", flush=True)
        actions = actions.reshape(-1, self.horizon, self.per_action_dim)
        self._queues.extend(actions.transpose(0, 1))
        return self._queues.popleft()
    
    def _check_get_actions_condition(self) -> bool:
        action_length = len(self._queues)
        return (
            (action_length == (self.horizon - self.n_action_steps)) or \
            (action_length == 0)
        )

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        print(f"Freezing {name} parameters...")
        for p in module.parameters():
            p.requires_grad = False

    def set_finetune_flags(self) -> None:
        config = self.config  
        if not config.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")
        else:
            print("Finetuning VLM (InternVL3)...")

        if not config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")
