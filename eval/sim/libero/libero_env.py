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

from __future__ import annotations

import os

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.envs.registration import register

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:

    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams

def _get_suite(name: str) -> benchmark.Benchmark:

    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite

def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:

    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids

def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[i].problem_folder
        / task_suite.tasks[i].init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)
    return init_states

def get_libero_dummy_action():

    return [0, 0, 0, 0, 0, 0, -1]

ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 500,
    "libero_object": 500,
    "libero_goal": 500,
    "libero_10": 500,
    "libero_90": 500,
}

class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite_name: str,
        task_id: int,
        seed: int | None = 42,
        output_video_dir: str | None = None,
        video_fps: int = 20,
        video_view_mode: str = "multi-view",
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        render_mode: str = "rgb_array",
        observation_width: int = 360,
        observation_height: int = 360,
        init_states: bool = True,
        episode_index: int = 0,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
    ):
        super().__init__()
        self.task_suite = _get_suite(task_suite_name)
        self.task_id = task_id
        self.render_mode = render_mode
        self.seed = seed
        self.observation_width = observation_width
        self.observation_height = observation_height
        self._output_video_dir = Path(output_video_dir) if output_video_dir else None
        self._video_writer = None
        self._episode_index = 0
        self._video_fps = video_fps
        self._video_view_mode = video_view_mode
        self.init_states = init_states
        self.camera_name = _parse_camera_names(
            camera_name
        )
        if self._output_video_dir is not None:
            self._output_video_dir.mkdir(parents=True, exist_ok=True)
        if self._video_view_mode not in {"single-view", "multi-view"}:
            raise ValueError(
                f"video_view_mode must be one of ['single-view', 'multi-view'], got: {self._video_view_mode}"
            )

        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        self.episode_length = episode_length

        self._init_states = get_task_init_states(self.task_suite, self.task_id) if self.init_states else None
        self._reset_stride = 1

        self.init_state_id = self.episode_index

        self._env = self._make_envs_task(self.task_suite, self.task_id)
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, 500)
            if self.episode_length is None
            else self.episode_length
        )
        self._step_id = 0
        self.control_mode = control_mode
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(images),
                "robot_state": spaces.Dict(
                    {
                        "eef": spaces.Dict(
                            {
                                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                "quat": spaces.Box(
                                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                ),
                                "mat": spaces.Box(
                                    low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                ),
                            }
                        ),
                        "gripper": spaces.Dict(
                            {
                                "qpos": spaces.Box(
                                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                ),
                                "qvel": spaces.Box(
                                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                ),
                            }
                        ),
                        "joints": spaces.Dict(
                            {
                                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                            }
                        ),
                    }
                ),
                "task_description": spaces.Text(max_length=512),
            }
        )
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def render(self):
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        image = image[::-1, ::-1]
        return image

    def _resize_to_height(self, image: np.ndarray, target_h: int) -> np.ndarray:
        h, w, c = image.shape
        if h == target_h:
            return image
        scale = target_h / float(h)
        target_w = max(1, int(round(w * scale)))
        row_idx = np.linspace(0, h - 1, target_h).astype(np.int32)
        col_idx = np.linspace(0, w - 1, target_w).astype(np.int32)
        return image[row_idx][:, col_idx].reshape(target_h, target_w, c)

    def _compose_video_frame(self, obs: dict[str, Any]) -> np.ndarray:
        front = np.asarray(obs["pixels"]["image"][::-1, ::-1], dtype=np.uint8)
        if self._video_view_mode == "single-view" or "image2" not in obs["pixels"]:
            return front

        wrist = np.asarray(obs["pixels"]["image2"][::-1, ::-1], dtype=np.uint8)
        target_h = max(front.shape[0], wrist.shape[0])
        front = self._resize_to_height(front, target_h)
        wrist = self._resize_to_height(wrist, target_h)
        return np.concatenate([front, wrist], axis=1)

    def _start_episode_video(self) -> None:
        if self._output_video_dir is None or self._video_writer is not None:
            return
        video_path = self._output_video_dir / f"episode_{self._episode_index:06d}.mp4"
        self._video_writer = imageio.get_writer(video_path.as_posix(), fps=self._video_fps)

    def _append_video_frame(self, obs: dict[str, Any]) -> None:
        if self._output_video_dir is None:
            return
        self._start_episode_video()
        self._video_writer.append_data(self._compose_video_frame(obs))

    def _finalize_episode_video(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
            self._episode_index += 1

    def _make_envs_task(self, task_suite: Any, task_id: int = 0) -> OffScreenRenderEnv:
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            images[self.camera_name_mapping[camera_name]] = image

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")

        obs = {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,
                    "quat": eef_quat,
                    "mat": eef_mat,
                },
                "gripper": {
                    "qpos": gripper_qpos,
                    "qvel": gripper_qvel,
                },
                "joints": {
                    "pos": joint_pos,
                    "vel": joint_vel,
                },
            },
            "task_description": self.task_description,
        }

        if eef_pos is None or eef_quat is None or gripper_qpos is None:
            raise ValueError(
                f"Missing required robot state fields in raw observation. "
                f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                f"gripper_qpos={gripper_qpos is not None}"
            )
        return obs

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        record_video = kwargs.pop("_record_video", True)
        self._finalize_episode_video()
        seed = self.seed if seed is None else seed
        if seed is not None:
            self._env.seed(seed)
        self._step_id = 0
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
            self.init_state_id += self._reset_stride

        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        observation = self._format_raw_obs(raw_obs)
        if record_video:
            self._append_video_frame(observation)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        self._append_video_frame(observation)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self._finalize_episode_video()
            self.reset(_record_video=False)
        self._step_id += 1
        truncated = (self._step_id >= self._max_episode_steps)
        return observation, reward, terminated, truncated, info

    def close(self):
        self._finalize_episode_video()
        self._env.close()

def register_libero_envs() -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    for task_suite_name in [
        "libero_10",
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_90",
    ]:
        task_suite = benchmark_dict[task_suite_name]()
        for task_id in range(task_suite.get_num_tasks()):
            register(
                id=f"{task_suite_name}/task_{task_id}",
                entry_point="sim.libero.libero_env:LiberoEnv",
                kwargs={
                    "task_suite_name": task_suite_name,
                    "task_id": task_id,
                    "seed": 42,
                    "output_video_dir": None,
                    "video_fps": 20,
                    "video_view_mode": "multi-view",
                    "episode_length": None,
                    "camera_name": "agentview_image,robot0_eye_in_hand_image",
                    "render_mode": "rgb_array",
                    "observation_width": 256,
                    "observation_height": 256,
                    "init_states": True,
                    "episode_index": 0,
                    "camera_name_mapping": None,
                    "num_steps_wait": 10,
                    "control_mode": "relative",
                },
            )
