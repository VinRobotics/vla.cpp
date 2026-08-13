"""Gr00t N1.7 policy pipeline for inference.

This module provides the GR00TN17PolicyPipeline class that mirrors the surface
of GR00TN16PolicyPipeline but targets the Gr00tN1d7 model (Cosmos-Reason2-2B
Qwen3-VL backbone + AlternateVLDiT flow-matching head). Like the N1.6 vendor,
all NVIDIA Isaac-GR00T sources needed at inference time live under this
package — no external `gr00t` package install is required.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from policies.action_chunk import ActionChunkQueue
from policies.torch_compile import maybe_compile
from policies.gr00t_n17.data.embodiment_tags import (
    FINETUNE_ONLY_TAGS,
    POSTTRAIN_TAGS,
    EmbodimentTag,
)
from policies.gr00t_n17.data.interfaces import BaseProcessor
from policies.gr00t_n17.data.types import MessageType, VLAStepData


class GR00TN17PolicyPipeline:
    arch_type: str = "gr00t"

    def __init__(
        self,
        embodiment_tag: str | EmbodimentTag,
        model_path: str,
        *,
        device: int | str | None = None,
        strict: bool = True,
        n_action_steps: int = 1,
    ):
        # Import this to register Gr00tN1d7 / Gr00tN1d7Processor with the
        # transformers Auto* registries before AutoModel / AutoProcessor fire.
        import policies.gr00t_n17.model  # noqa: F401

        model_dir = Path(model_path)
        device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = AutoModel.from_pretrained(model_dir)
        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # Solver-step override, mirroring vla.cpp's VLA_NUM_STEPS (see
        # src/models/gr00tn1d7.cpp) so one variable sets T on both stacks.
        # The denoise loop reads `self.num_inference_timesteps` off the head
        # instance, NOT off config, so assigning to config here would be inert.
        ns = os.environ.get("VLA_NUM_STEPS")
        if ns:
            n = int(ns)
            if n < 1:
                raise ValueError(f"VLA_NUM_STEPS must be >= 1 (got {n})")
            head = model.action_head
            print(
                f"[gr00t_n17] VLA_NUM_STEPS override: "
                f"num_inference_timesteps {head.num_inference_timesteps} -> {n}",
                flush=True,
            )
            head.num_inference_timesteps = n
            head.config.num_inference_timesteps = n  # keep config in sync for logging

        # Optional torch.compile. As in N1.6, inference runs through
        # `get_action` rather than `forward`, so that bound method is the
        # compile target; kept on the pipeline to leave the HF module untouched.
        self._get_action = maybe_compile(model.get_action, tag="gr00t_n17.get_action")

        # Training saves processor files under a "processor/" subdirectory, but
        # AutoProcessor expects them at the model root. Fall back to the
        # subdirectory when the root lacks a processor_config.json.
        processor_dir = (
            model_dir / "processor"
            if (model_dir / "processor").is_dir()
            and not (model_dir / "processor_config.json").exists()
            else model_dir
        )
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(processor_dir)
        self.processor.eval()

        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        self.embodiment_tag = embodiment_tag

        all_modality_configs = self.processor.get_modality_configs()
        if self.embodiment_tag.value not in all_modality_configs:
            supported_lines = []
            for tag_value in sorted(all_modality_configs.keys()):
                enum_name = EmbodimentTag.reverse_lookup(tag_value)
                if enum_name != tag_value:
                    supported_lines.append(f"  {enum_name:30s} (--embodiment-tag {enum_name})")
                else:
                    supported_lines.append(f"  {tag_value:30s} (internal, no public enum)")
            supported_str = "\n".join(supported_lines)

            hint = ""
            if self.embodiment_tag in POSTTRAIN_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is a posttrain tag that requires "
                    f"a finetuned checkpoint, not the base model."
                )
            elif self.embodiment_tag in FINETUNE_ONLY_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is for finetuning custom robots."
                )

            raise ValueError(
                f"Embodiment tag '{self.embodiment_tag.name}' "
                f"(value='{self.embodiment_tag.value}') is not supported "
                f"by this checkpoint.\n\n"
                f"Supported tags in this checkpoint:\n{supported_str}"
                f"{hint}"
            )
        # `rl_info` is a training-only modality; drop it so inference checks stay tight.
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Some embodiments (e.g. OXE_DROID) define multiple language keys for
        # training-time augmentation (paraphrases). At inference we only use the first key.
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) >= 1, "At least one language key is required"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

        self.strict = strict

        # Chunk replay: hand out n_action_steps timesteps per forward, matching
        # vla.cpp's client-side --n-action-steps (see policies/action_chunk.py).
        self._chunk_queue = ActionChunkQueue(n_action_steps)

    def reset(self) -> dict[str, Any]:
        self._chunk_queue.clear()
        return {}

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._chunk_queue.empty:
            self._chunk_queue.fill(self._predict_action_chunk(observation))
        return self._chunk_queue.pop()

    def _predict_action_chunk(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.strict:
            self.check_observation(observation)

        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self._get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }

        if self.strict:
            self.check_action(action)

        return action

    def get_arch(self) -> dict[str, str]:
        return {"arch": self.arch_type}

    ####################################################################################################
    ######################################## HELPER METHODS ############################################
    ####################################################################################################
    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types."""
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        bs = -1

        # ===== VIDEO VALIDATION =====
        for video_key in self.modality_configs["video"].modality_keys:
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            batched_video = observation["video"][video_key]

            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        for state_key in self.modality_configs["state"].modality_keys:
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            batched_state = observation["state"][state_key]

            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        for language_key in self.modality_configs["language"].modality_keys:
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            batched_language: list[list[str]] = observation["language"][language_key]

            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            for batch_item in batched_language:
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types."""
        for action_key in self.modality_configs["action"].modality_keys:
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations."""
        unbatched_obs = []
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing."""
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype."""
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x
