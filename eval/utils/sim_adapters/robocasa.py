# Copyright 2026 VinRobotics - Apache-2.0
"""RoboCasa PandaOmron layouts and observation/action parsers."""

from collections.abc import Mapping
import json
from pathlib import Path
import re

import numpy as np

from utils.sim_adapters.base import BasePipelineAdapter

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
STATE_DIMS = {
    "base_position": 3,
    "base_rotation": 4,
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
}
SMOLVLA_STATE_KEYS = (
    "base_position", "base_rotation", "end_effector_position_relative",
    "end_effector_rotation_relative", "gripper_qpos",
)
SMOLVLA_CAMERA_MAP = {
    "observation.images.robot0_agentview_left": "observation.images.camera1",
    "observation.images.robot0_agentview_right": "observation.images.camera2",
    "observation.images.robot0_eye_in_hand": "observation.images.camera3",
}
GR00T_STATE_KEYS = (
    "end_effector_position_relative", "end_effector_rotation_relative",
    "gripper_qpos", "base_position", "base_rotation",
)
ACTION_DIMS = {
    "base_motion": 4,
    "control_mode": 1,
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
}
SMOLVLA_ACTION_KEYS = tuple(ACTION_DIMS)
GR00T_ACTION_KEYS = (
    "end_effector_position", "end_effector_rotation", "gripper_close",
    "base_motion", "control_mode",
)
GR00T_VIDEO_KEYS = ("res256_image_side_0", "res256_image_side_1", "res256_image_wrist_0")


def vector(value, dim: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (dim,) or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected {dim} finite values, got shape {result.shape}")
    return result


def rgb_image(value, label: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{label}: expected HWC uint8 RGB, got {image.shape}/{image.dtype}")
    return np.ascontiguousarray(image)


def flatten_state(state: Mapping, keys=SMOLVLA_STATE_KEYS) -> np.ndarray:
    return np.concatenate([vector(state[k], STATE_DIMS[k], f"state.{k}") for k in keys])


def split_action(action, keys=SMOLVLA_ACTION_KEYS) -> dict[str, np.ndarray]:
    action = vector(action, 12, "action")
    offset = 0
    result = {}
    for key in keys:
        dim = ACTION_DIMS[key]
        result[f"action.{key}"] = action[offset:offset + dim].copy()
        offset += dim
    return result


def validate_action(action: Mapping) -> dict[str, np.ndarray]:
    expected = {f"action.{key}" for key in ACTION_DIMS}
    if set(action) != expected:
        raise ValueError(f"Expected action fields {sorted(expected)}, got {sorted(action)}")
    return {f"action.{key}": vector(action[f"action.{key}"], dim, f"action.{key}").copy()
            for key, dim in ACTION_DIMS.items()}


def task_instruction(obs: Mapping, task_prompt: str) -> str:
    if task_prompt not in {"task_id", "language"}:
        raise ValueError("task_prompt must be task_id or language")
    key = "task_id" if task_prompt == "task_id" else "task_description"
    task = obs[key]
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"Policy requires a nonempty {key} string")
    return task


def smolvla_observation(obs: Mapping, *, input_features=None, rename_map=None, task_prompt="task_id") -> dict:
    """Format the checkpoint's inputs before batching/model preprocessing.

    Images are CHW float32 in [0, 1], state is raw float32 (16,), and task is
    text. The transport pads state/resizes images; C++ applies the GGUF's
    MEAN_STD state normalization and SigLIP [-1, 1] image normalization once.
    """
    if input_features is None:
        input_features = {
            "observation.state": {"type": "STATE", "shape": [16]},
            **{key: {"type": "VISUAL", "shape": [3, 256, 256]}
               for key in SMOLVLA_CAMERA_MAP.values()},
        }
    rename_map = SMOLVLA_CAMERA_MAP if rename_map is None else rename_map
    sources = {target: source.removeprefix("observation.images.")
               for source, target in rename_map.items()}
    # RoboCasaGymEnv already flips the raw MuJoCo images vertically. Applying
    # LIBERO's 180-degree rotation here would corrupt the trained camera views.
    result = {}
    for key, feature in input_features.items():
        if feature["type"] == "VISUAL":
            camera = sources[key]
            image = rgb_image(obs["pixels"][camera], camera)
            chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
            if chw.shape != tuple(feature["shape"]):
                raise ValueError(f"{key}: checkpoint expects {feature['shape']}, got {chw.shape}")
            result[key] = chw
    state = flatten_state(obs["robot_state"], SMOLVLA_STATE_KEYS)
    if state.shape != tuple(input_features["observation.state"]["shape"]):
        raise ValueError("Observation state does not match the SmolVLA checkpoint")
    result["observation.state"] = state
    # The released dataset uses CamelCase labels, not ep_meta['lang']; see
    # https://github.com/huggingface/lerobot/issues/4045.
    task = task_instruction(obs, task_prompt)
    result["task"] = task if task.endswith("\n") else task + "\n"
    return result


SMOLVLA_REPO = "lerobot/smolvla_robocasa"
SMOLVLA_REVISION = "f7af588d11e30b0069f200792fb285904b74c80b"
GR00T_EMBODIMENT = "robocasa_panda_omron"


def load_smolvla_metadata(policy_path=None):
    def metadata(name):
        if policy_path is not None:
            path = Path(policy_path) / name
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(SMOLVLA_REPO, name, revision=SMOLVLA_REVISION))
        return json.loads(path.read_text())

    config = metadata("config.json")
    steps = metadata("policy_preprocessor.json")["steps"]
    return config, {step["registry_name"]: step["config"] for step in steps}


def smolvla_client_options(config, processors, *, tokenizer=None, n_action_steps=50,
                           recv_timeout_ms=120_000):
    features = config["input_features"]
    tokenizer_config = processors["tokenizer_processor"]
    image_keys = [key for key, feature in features.items() if feature["type"] == "VISUAL"]
    if (config["type"] != "smolvla" or config["n_obs_steps"] != 1
            or features["observation.state"]["shape"] != [16]
            or config["output_features"]["action"]["shape"] != [12]
            or len(image_keys) != 3):
        raise ValueError("Expected the lerobot/smolvla_robocasa 16D-state/12D-action, three-camera contract")
    if not 1 <= n_action_steps <= config["chunk_size"]:
        raise ValueError(f"SmolVLA n_action_steps must be in [1, {config['chunk_size']}]")
    resize = config["resize_imgs_with_padding"]
    if resize[0] != resize[1]:
        raise ValueError("The C++ SmolVLA client requires a square vision input")
    return dict(
        arch="smolvla", tokenizer_name=tokenizer or tokenizer_config["tokenizer_name"],
        image_keys=image_keys, max_length=tokenizer_config["max_length"],
        max_state_dim=config["max_state_dim"], real_action_dim=12, image_size=resize[0],
        n_action_steps=n_action_steps, recv_timeout_ms=recv_timeout_ms,
    )


class SmolVLARoboCasaParser:
    """Checkpoint observation/action conversion; C++ owns normalization."""

    def __init__(self, config, processors, *, task_prompt="task_id"):
        if task_prompt not in {"task_id", "language"}:
            raise ValueError("task_prompt must be task_id or language")
        self.task_prompt = task_prompt
        self._input_features = config["input_features"]
        self._rename_map = processors["rename_observations_processor"]["rename_map"]
        self._chunk_shape = (config["chunk_size"], config["max_action_dim"])

    def parse_observation(self, observation):
        return smolvla_observation(
            observation,
            input_features=self._input_features,
            rename_map=self._rename_map,
            task_prompt=self.task_prompt
        )

    def validate_response(self, response):
        if ((response.chunk_size, response.action_dim) != self._chunk_shape
                or not np.isfinite(response.action_chunk).all()):
            raise ValueError(
                "Expected a finite SmolVLA response with shape (50, 32); check the served model"
            )

    def parse_action(self, action):
        return split_action(action)


class GR00TN16RoboCasaParser:
    """N1.6 PandaOmron observation/action conversion from checkpoint metadata.

    Layouts and the valid action horizon come from the matching checkpoint's
    processor_config.json. Only the PandaOmron absolute NON_EEF modalities are
    supported; reject incompatible fine-tunes rather than silently misdecode.
    """

    def __init__(self, *, stats_json, processor_config,
                 n_action_steps=8, image_size=224, task_prompt="language"):
        if task_prompt not in {"task_id", "language"}:
            raise ValueError("task_prompt must be task_id or language")
        self.task_prompt = task_prompt
        config = json.loads(Path(processor_config).read_text())["processor_kwargs"]
        modalities = config["modality_configs"][GR00T_EMBODIMENT]
        self._state_keys = tuple(modalities["state"]["modality_keys"])
        self._action_keys = tuple(modalities["action"]["modality_keys"])
        video_keys = tuple(modalities["video"]["modality_keys"])
        for name, keys, expected in (
            ("state", self._state_keys, GR00T_STATE_KEYS),
            ("action", self._action_keys, GR00T_ACTION_KEYS),
            ("video", video_keys, GR00T_VIDEO_KEYS),
        ):
            if len(keys) != len(expected) or set(keys) != set(expected):
                raise ValueError(f"Unsupported RoboCasa {name} modalities: {keys}")
        self._camera_order = tuple(CAMERAS[GR00T_VIDEO_KEYS.index(key)] for key in video_keys)
        for group in ("state", "action"):
            if modalities[group].get("mean_std_embedding_keys") or modalities[group].get("sin_cos_embedding_keys"):
                raise ValueError(f"Unsupported RoboCasa {group} normalization/encoding")
        for group in ("state", "video"):
            if modalities[group].get("delta_indices", [0]) != [0]:
                raise ValueError(f"RoboCasa GR00T supports one current {group} observation")
        for action_config in modalities["action"].get("action_configs") or []:
            if (action_config.get("rep"), action_config.get("type"), action_config.get("format")) != (
                    "ABSOLUTE", "NON_EEF", "DEFAULT"):
                raise ValueError("RoboCasa GR00T requires ABSOLUTE NON_EEF DEFAULT action modalities")
        indices = modalities["action"]["delta_indices"]
        if indices != list(range(16)):
            raise ValueError("RoboCasa GR00T N1.6 expects action indices 0..15")
        self._decode_horizon = len(indices)
        if not 1 <= n_action_steps <= self._decode_horizon:
            raise ValueError("GR00T n_action_steps must be in [1, 16]")
        if image_size < 28 or image_size % 28:
            raise ValueError("GR00T image_size must be a positive multiple of 28 matching the GGUF")
        if config["max_state_dim"] != 128 or config["max_action_dim"] != 128:
            raise ValueError("GR00T N1.6 requires max_state_dim=max_action_dim=128")
        if config.get("shortest_image_edge", 256) != 256:
            raise ValueError("RoboCasa GR00T expects shortest_image_edge=256")
        if config.get("image_crop_size") is not None or config.get("image_target_size") is not None:
            raise ValueError("RoboCasa GR00T expects the N1.6 variable-size image processor")
        self._crop_fraction = float(config.get("crop_fraction", 0.95))
        if not 0 < self._crop_fraction <= 1:
            raise ValueError("crop_fraction must be in (0, 1]")
        self._formalize_language = bool(config.get("formalize_language", True))
        self._clip_outliers = bool(config.get("clip_outliers", True))
        stats = json.loads(Path(stats_json).read_text())[GR00T_EMBODIMENT]
        lo_field, hi_field = ("q01", "q99") if config.get("use_percentiles", False) else ("min", "max")

        def bounds(group, keys, dims):
            lows, highs = [], []
            for key in keys:
                low = np.asarray(stats[group][key][lo_field], dtype=np.float32)
                high = np.asarray(stats[group][key][hi_field], dtype=np.float32)
                if (low.shape != (dims[key],) or high.shape != low.shape
                        or not np.isfinite(low).all() or not np.isfinite(high).all()
                        or np.any(high < low)):
                    raise ValueError(f"Invalid statistics for {group}.{key}")
                lows.append(low)
                highs.append(high)
            return np.concatenate(lows), np.concatenate(highs)

        self._state_min, self._state_max = bounds("state", self._state_keys, STATE_DIMS)
        self._action_min, self._action_max = bounds("action", self._action_keys, ACTION_DIMS)
        self.image_size = image_size
        self.n_action_steps = n_action_steps
        self.image_keys = [f"observation.images.{camera}" for camera in self._camera_order]

    def client_options(self, *, tokenizer, recv_timeout_ms=120_000):
        return dict(
            arch="gr00t_n1_6", tokenizer_name=tokenizer, image_size=self.image_size,
            max_state_dim=128, real_action_dim=12, image_keys=self.image_keys,
            n_action_steps=self.n_action_steps, recv_timeout_ms=recv_timeout_ms,
            stats_json=None,  # Normalization belongs to the parser for this embodiment.
        )

    def parse_observation(self, observation):
        from client.vla_cpp_client import VlaCppClient, _minmax_norm

        images = []
        for camera in self._camera_order:
            image = rgb_image(observation["pixels"][camera], camera)
            # NVIDIA's processor crops then restores shortest_image_edge=256
            # before Eagle resizes to the GGUF's vision resolution.
            image = VlaCppClient._gr00t_n16_image_transform(image, 256, self._crop_fraction)
            image = VlaCppClient._gr00t_n16_image_transform(image, self.image_size, 1.0)
            images.append(np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0)
        raw_state = flatten_state(observation["robot_state"], self._state_keys)
        state = _minmax_norm(raw_state, self._state_min, self._state_max,
                             ~np.isclose(self._state_max, self._state_min))
        if self._clip_outliers:
            state = np.clip(state, -1.0, 1.0)
        language = task_instruction(observation, self.task_prompt)
        if self._formalize_language:
            language = re.sub(r"[^\w\s]", "", language.lower())
        return {
            **dict(zip(self.image_keys, images)),
            "observation.state": state,
            "task": language,
        }

    def validate_response(self, response):
        if response.chunk_size < self._decode_horizon or response.action_dim != 128:
            raise ValueError("Expected a GR00T N1.6 response with at least 16 steps and 128 action channels")
        chunk = np.asarray(response.action_chunk, dtype=np.float32).reshape(
            response.chunk_size, response.action_dim)[:self._decode_horizon, :12]
        if not np.isfinite(chunk).all():
            raise ValueError("Server returned nonfinite GR00T actions")

    def parse_action(self, action):
        action = vector(action, 12, "action")
        decoded = ((np.clip(action, -1.0, 1.0) + 1.0) * 0.5
                   * (self._action_max - self._action_min) + self._action_min).astype(np.float32)
        return split_action(decoded, self._action_keys)


class RoboCasaSimAdapter(BasePipelineAdapter):
    """Apply a RoboCasa parser around the shared client, as in LIBEROSimAdapter."""

    def __init__(self, client, parser):
        super().__init__(client=client, parser=parser, arch=client.arch)

    @property
    def _step(self):
        return self._client._step

    def close(self):
        self._client.close()

    def get_action(self, obs):
        parsed_obs = self._parser.parse_observation(obs)
        action = self._client.get_action(parsed_obs)
        self._parser.validate_response(self._client._last_response)
        return self._parser.parse_action(action)
