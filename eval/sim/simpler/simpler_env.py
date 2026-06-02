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

import cv2
import numpy as np
from pathlib import Path
from transforms3d import euler as te, quaternions as tq

import imageio.v2 as imageio
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

import gymnasium as gym
from gymnasium.envs.registration import register

class GoogleFractalEnv(gym.Env):
    def __init__(
        self,
        env_name: str,
        image_size: tuple[int, int],
        output_video_dir: str | None = None,
        video_fps: int = 20,
    ):
        env = simpler_env.make(env_name)
        env._max_episode_steps = 1000

        self.env = env
        self._output_video_dir = Path(output_video_dir) if output_video_dir else None
        self._video_writer = None
        self._episode_index = 0
        self._video_fps = video_fps
        if self._output_video_dir is not None:
            self._output_video_dir.mkdir(parents=True, exist_ok=True)
        agent_space = env.observation_space["agent"]
        print("[SimplerEnv] agent space keys:", list(agent_space.spaces.keys()))

        obs_low = env.observation_space["agent"]["eef_pos"].low
        obs_high = env.observation_space["agent"]["eef_pos"].high
        self.observation_space = gym.spaces.Dict(
            {
                "video.image": gym.spaces.Box(
                    low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
                ),
                "state.x": gym.spaces.Box(
                    low=obs_low[0], high=obs_high[0], shape=(1,), dtype=np.float32
                ),
                "state.y": gym.spaces.Box(
                    low=obs_low[1], high=obs_high[1], shape=(1,), dtype=np.float32
                ),
                "state.z": gym.spaces.Box(
                    low=obs_low[2], high=obs_high[2], shape=(1,), dtype=np.float32
                ),

                "state.rx": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "state.ry": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "state.rz": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "state.rw": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "state.gripper": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        action_low = env.action_space.low
        action_high = env.action_space.high
        self.action_space = gym.spaces.Dict(
            {
                "action.x": gym.spaces.Box(
                    low=action_low[0], high=action_high[0], shape=(1,), dtype=np.float32
                ),
                "action.y": gym.spaces.Box(
                    low=action_low[1], high=action_high[1], shape=(1,), dtype=np.float32
                ),
                "action.z": gym.spaces.Box(
                    low=action_low[2], high=action_high[2], shape=(1,), dtype=np.float32
                ),
                "action.roll": gym.spaces.Box(
                    low=action_low[3], high=action_high[3], shape=(1,), dtype=np.float32
                ),
                "action.pitch": gym.spaces.Box(
                    low=action_low[4], high=action_high[4], shape=(1,), dtype=np.float32
                ),
                "action.yaw": gym.spaces.Box(
                    low=action_low[5], high=action_high[5], shape=(1,), dtype=np.float32
                ),
                "action.gripper": gym.spaces.Box(
                    low=action_low[6], high=action_high[6], shape=(1,), dtype=np.float32
                ),
            }
        )
        self.image_size = image_size
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        self.sticky_gripper_num_repeat = 15

    def reset(self, seed=None, options=None):
        self._finalize_episode_video()
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        observation, info = self.env.reset()
        observation = self._process_observation(observation)
        info["success"] = False
        self._append_video_frame(observation)
        return observation, info

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                self._postprocess_gripper(action["action.gripper"]),
            ],
            axis=0,
        )
        observation, reward, done, truncated, info = self.env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = done
        self._append_video_frame(observation)
        if done or truncated:
            self._finalize_episode_video()
        return observation, reward, done, truncated, info

    def _process_observation(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = obs["agent"]["eef_pos"]
        qunat_xyzw = np.roll(proprio[3:7], -1)
        gripper_closedness = 1 - proprio[7]
        return {
            "video.image": cv2.resize(img, (self.image_size[1], self.image_size[0])),
            "state.x": np.asarray([proprio[0]], dtype=np.float32),
            "state.y": np.asarray([proprio[1]], dtype=np.float32),
            "state.z": np.asarray([proprio[2]], dtype=np.float32),
            "state.rx": np.asarray([qunat_xyzw[0]], dtype=np.float32),
            "state.ry": np.asarray([qunat_xyzw[1]], dtype=np.float32),
            "state.rz": np.asarray([qunat_xyzw[2]], dtype=np.float32),
            "state.rw": np.asarray([qunat_xyzw[3]], dtype=np.float32),
            "state.gripper": np.asarray([gripper_closedness], dtype=np.float32),
            "annotation.human.action.task_description": (
                self.env.unwrapped.get_language_instruction()
            ),
        }

    def _postprocess_gripper(self, current_gripper_action: float) -> float:
        current_gripper_action = (current_gripper_action * 2) - 1
        relative_gripper_action = -current_gripper_action
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        return relative_gripper_action

    def _start_episode_video(self) -> None:
        if self._output_video_dir is None or self._video_writer is not None:
            return
        video_path = self._output_video_dir / f"episode_{self._episode_index:06d}.mp4"
        self._video_writer = imageio.get_writer(video_path.as_posix(), fps=self._video_fps)

    def _append_video_frame(self, obs: dict) -> None:
        if self._output_video_dir is None:
            return
        self._start_episode_video()
        frame = obs["video.image"]
        self._video_writer.append_data(frame)

    def _finalize_episode_video(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
            self._episode_index += 1

class WidowXBridgeEnv(gym.Env):
    def __init__(
        self,
        env_name: str,
        image_size: tuple[int, int],
        output_video_dir: str | None = None,
        video_fps: int = 20,
    ):
        env = simpler_env.make(env_name)
        env._max_episode_steps = 1000
        self.env = env
        self._output_video_dir = Path(output_video_dir) if output_video_dir else None
        self._video_writer = None
        self._episode_index = 0
        self._video_fps = video_fps
        if self._output_video_dir is not None:
            self._output_video_dir.mkdir(parents=True, exist_ok=True)
        obs_low = env.observation_space["agent"]["eef_pos"].low
        obs_high = env.observation_space["agent"]["eef_pos"].high
        self.observation_space = gym.spaces.Dict(
            {
                "video.image_0": gym.spaces.Box(
                    low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
                ),
                "state.x": gym.spaces.Box(
                    low=obs_low[0], high=obs_high[0], shape=(1,), dtype=np.float32
                ),
                "state.y": gym.spaces.Box(
                    low=obs_low[1], high=obs_high[1], shape=(1,), dtype=np.float32
                ),
                "state.z": gym.spaces.Box(
                    low=obs_low[2], high=obs_high[2], shape=(1,), dtype=np.float32
                ),

                "state.roll": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                "state.pitch": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                "state.yaw": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),

                "state.pad": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "state.gripper": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        action_low = env.action_space.low
        action_high = env.action_space.high
        self.action_space = gym.spaces.Dict(
            {
                "action.x": gym.spaces.Box(
                    low=action_low[0], high=action_high[0], shape=(1,), dtype=np.float32
                ),
                "action.y": gym.spaces.Box(
                    low=action_low[1], high=action_high[1], shape=(1,), dtype=np.float32
                ),
                "action.z": gym.spaces.Box(
                    low=action_low[2], high=action_high[2], shape=(1,), dtype=np.float32
                ),
                "action.roll": gym.spaces.Box(
                    low=action_low[3], high=action_high[3], shape=(1,), dtype=np.float32
                ),
                "action.pitch": gym.spaces.Box(
                    low=action_low[4], high=action_high[4], shape=(1,), dtype=np.float32
                ),
                "action.yaw": gym.spaces.Box(
                    low=action_low[5], high=action_high[5], shape=(1,), dtype=np.float32
                ),
                "action.gripper": gym.spaces.Box(
                    low=action_low[6], high=action_high[6], shape=(1,), dtype=np.float32
                ),
            }
        )
        self.image_size = image_size

        self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    def reset(self, seed=None, options=None):
        self._finalize_episode_video()
        observation, info = self.env.reset()
        observation = self._process_observation(observation)
        info["success"] = False
        self._append_video_frame(observation)
        return observation, info

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                self._postprocess_gripper(action["action.gripper"]),
            ],
            axis=0,
        )
        observation, reward, done, truncated, info = self.env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = done
        self._append_video_frame(observation)
        if done or truncated:
            self._finalize_episode_video()
        return observation, reward, done, truncated, info

    def _process_observation(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = obs["agent"]["eef_pos"]
        rm_bridge = tq.quat2mat(proprio[3:7])
        rpy_bridge_converted = te.mat2euler(rm_bridge @ self.default_rot.T)
        return {
            "video.image_0": cv2.resize(img, (self.image_size[1], self.image_size[0])),
            "state.x": np.asarray([proprio[0]], dtype=np.float32),
            "state.y": np.asarray([proprio[1]], dtype=np.float32),
            "state.z": np.asarray([proprio[2]], dtype=np.float32),
            "state.roll": np.asarray([rpy_bridge_converted[0]], dtype=np.float32),
            "state.pitch": np.asarray([rpy_bridge_converted[1]], dtype=np.float32),
            "state.yaw": np.asarray([rpy_bridge_converted[2]], dtype=np.float32),
            "state.pad": np.asarray([0.0], dtype=np.float32),
            "state.gripper": np.asarray([proprio[7]], dtype=np.float32),
            "annotation.human.action.task_description": (
                self.env.unwrapped.get_language_instruction()
            ),
        }

    def _postprocess_gripper(self, action):

        return 2.0 * (action > 0.5) - 1.0

    def _start_episode_video(self) -> None:
        if self._output_video_dir is None or self._video_writer is not None:
            return
        video_path = self._output_video_dir / f"episode_{self._episode_index:06d}.mp4"
        self._video_writer = imageio.get_writer(video_path.as_posix(), fps=self._video_fps)

    def _append_video_frame(self, obs: dict) -> None:
        if self._output_video_dir is None:
            return
        self._start_episode_video()
        frame = obs["video.image_0"]
        self._video_writer.append_data(frame)

    def _finalize_episode_video(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
            self._episode_index += 1

def register_simpler_envs():

    for env_name in [
        "google_robot_pick_coke_can",
        "google_robot_pick_object",
        "google_robot_move_near",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_place_in_closed_drawer",
    ]:
        register(
            id=f"oxe_google/{env_name}",
            entry_point="sim.simpler.simpler_env:GoogleFractalEnv",
            kwargs={
                "env_name": env_name,
                "image_size": (256, 320),
                "output_video_dir": None,
                "video_fps": 20
            },
        )

    for env_name in [
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket",
        "widowx_put_eggplant_in_sink",
        "widowx_open_drawer",
        "widowx_close_drawer",
    ]:
        register(
            id=f"oxe_widowx/{env_name}",
            entry_point="sim.simpler.simpler_env:WidowXBridgeEnv",
            kwargs={
                "env_name": env_name,
                "image_size": (256, 256),
                "output_video_dir": None,
                "video_fps": 20
            },
        )
