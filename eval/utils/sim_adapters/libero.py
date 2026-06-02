# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any
import numpy as np
import torch
from tree import map_structure

from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

from utils.sim_adapters.base import BasePipelineAdapter

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

class LeRobotLIBEROParser:
    _preprocessor = PolicyProcessorPipeline(
        [LiberoProcessorStep()]
    )
    _postprocessor = PolicyProcessorPipeline()

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        parsed_obs = map_structure(lambda x: x[None] if isinstance(x, np.ndarray) else x, obs)
        parsed_obs = preprocess_observation(parsed_obs)
        parsed_obs = self._preprocessor(parsed_obs)
        parsed_obs = map_structure(
            lambda x: (x.numpy()[0] if isinstance(x, torch.Tensor) else x), parsed_obs
        )
        parsed_obs["task"] = obs.get("task_description", "")
        return parsed_obs

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        action_transition = {ACTION: torch.from_numpy(action[None])}
        action_transition = self._postprocessor(action_transition)
        action = action_transition[ACTION].cpu().numpy()[0]
        return action

class Evo1LIBEROParser:
    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        front_img = np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1])

        return {
            "image": [front_img, wrist_img, np.zeros_like(front_img)],
            "state": np.concatenate((
                obs["robot_state"]["eef"]["pos"],
                quat2axisangle(obs["robot_state"]["eef"]["quat"]),
                obs["robot_state"]["gripper"]["qpos"],
            )),
            "prompt": obs["task_description"],
            "image_mask": [1, 1, 0],
            "action_mask": [1] * 7 + [0] * 17,
        }

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action[:7], dtype=np.float32).copy()
        action[6] = -1.0 if action[6] > 0.5 else 1.0
        return action

    @staticmethod
    def encode_image_array(img_array: np.ndarray):
        return img_array.astype(np.uint8).tolist()

class GR00TN16LIBEROParser:
    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        pos = obs["robot_state"]["eef"]["pos"].astype(np.float32)
        angles = quat2axisangle(obs["robot_state"]["eef"]["quat"]).astype(np.float32)
        gripper = obs["robot_state"]["gripper"]["qpos"].astype(np.float32)

        return {
            "video": {
                "image": np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1])[None, None],
                "wrist_image": np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1])[None, None]
            },
            "state": {
                "x": pos[0].reshape(1, 1, 1),
                "y": pos[1].reshape(1, 1, 1),
                "z": pos[2].reshape(1, 1, 1),
                "roll": angles[0].reshape(1, 1, 1),
                "pitch": angles[1].reshape(1, 1, 1),
                "yaw": angles[2].reshape(1, 1, 1),
                "gripper": gripper.reshape(1, 1, 2)
            },
            "language": {
                "annotation.human.action.task_description": [[obs["task_description"]]]
            }
        }

    def parse_action(self, action: dict[str, Any]) -> np.ndarray:
        action = np.concatenate([
            action["x"][0, 0],
            action["y"][0, 0],
            action["z"][0, 0],
            action["roll"][0, 0],
            action["pitch"][0, 0],
            action["yaw"][0, 0],
            action["gripper"][0, 0]
        ], axis=0).astype(np.float32)

        orig_low, orig_high = 0.0, 1.0
        action[-1] = 2 * (action[-1] - orig_low) / (orig_high - orig_low) - 1
        action[-1] = np.sign(action[-1])

        action[-1] = action[-1] * -1.0
        return action

LIBERO_PARSER_REGISTRY = {
    "smolvla": LeRobotLIBEROParser,
    "pi0": LeRobotLIBEROParser,
    "gr00t-n15": LeRobotLIBEROParser,
    "evo1": Evo1LIBEROParser,
    "gr00t": GR00TN16LIBEROParser,
}

class LIBEROSimAdapter(BasePipelineAdapter):
    def __init__(self, client: Any):
        arch = client.get_arch()
        parser_cls = LIBERO_PARSER_REGISTRY.get(arch)
        if parser_cls is None:
            raise ValueError(f"No parser found for architecture {arch}")
        super().__init__(client=client, parser=parser_cls(), arch=arch)
