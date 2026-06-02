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

class BasePipelineAdapter:
    def __init__(self, client: Any = None):
        self._client = client

    def reset(self):
        return self._client.reset()

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        parsed_obs = self.parse_observation(obs)
        action = self._client.get_action(parsed_obs)
        parsed_action = self.parse_action(action)
        return parsed_action

    def get_action_from_queue(self) -> np.ndarray:
        action = self._client.get_action_from_queue()
        return self.parse_action(action)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LeRobotPipelineAdapter(BasePipelineAdapter):
    def __init__(self, client: Any = None):
        super().__init__(client)
        self._preprocessor = PolicyProcessorPipeline([LiberoProcessorStep()])
        self._postprocessor = PolicyProcessorPipeline([])

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

class Evo1PipelineAdapter(BasePipelineAdapter):
    def __init__(self, client: Any = None):
        super().__init__(client)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        front_img = np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1])

        return {
            "image": [front_img, wrist_img, np.zeros_like(front_img)],
            "state": np.concatenate((
                obs["robot_state"]["eef"]["pos"],
                self.quat2axisangle(obs["robot_state"]["eef"]["quat"]),
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

    @staticmethod
    def quat2axisangle(quat):
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

class Gr00tPipelineAdapter(BasePipelineAdapter):

    def __init__(self, client: Any = None):
        super().__init__(client)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:

        front = np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1]).astype(np.uint8)
        wrist = np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1]).astype(np.uint8)

        eef_pos = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float32)
        rpy = Evo1PipelineAdapter.quat2axisangle(
            np.array(obs["robot_state"]["eef"]["quat"], dtype=np.float64, copy=True)
        ).astype(np.float32)
        gripper = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float32)

        def _scalar_state(v: float) -> np.ndarray:

            return np.array([[[v]]], dtype=np.float32)

        return {
            "video.image":       front[None, None],
            "video.wrist_image": wrist[None, None],
            "state.x":       _scalar_state(eef_pos[0]),
            "state.y":       _scalar_state(eef_pos[1]),
            "state.z":       _scalar_state(eef_pos[2]),
            "state.roll":    _scalar_state(rpy[0]),
            "state.pitch":   _scalar_state(rpy[1]),
            "state.yaw":     _scalar_state(rpy[2]),
            "state.gripper": gripper.reshape(1, 1, -1).astype(np.float32),

            "task": (obs.get("task_description", ""),),
            "annotation.human.action.task_description": (obs.get("task_description", ""),),
        }

    def parse_action(self, action: np.ndarray) -> np.ndarray:

        action = np.asarray(action[:7], dtype=np.float32).copy()
        action[6] = -1.0 if action[6] > 0.5 else 1.0
        return action

class Gr00tN15PipelineAdapter(Gr00tPipelineAdapter):

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action[:7], dtype=np.float32).copy()
