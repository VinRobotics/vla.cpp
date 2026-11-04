# Copyright 2026 VinRobotics - Apache-2.0
"""Gymnasium wrapper around the upstream RoboCasa365 PandaOmron environment."""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import imageio.v2 as imageio
import numpy as np

from utils.sim_adapters.robocasa import (
    ACTION_DIMS, CAMERAS, STATE_DIMS, rgb_image, validate_action, vector,
)


class RoboCasaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, task="CloseFridge", *, split="target", episode_length=None,
                 output_video_dir=None, video_fps=20, view_mode="multi-view", env_factory=None,
                 camera_size=256):
        super().__init__()
        if split not in {"target", "pretrain", "all"}:
            raise ValueError("split must be target, pretrain, or all")
        if episode_length is not None and episode_length < 1:
            raise ValueError("episode_length must be positive")
        if video_fps < 1 or view_mode not in {"single-view", "multi-view"}:
            raise ValueError("Invalid video FPS or view mode")
        if camera_size < 1:
            raise ValueError("camera_size must be positive")
        self.camera_size = camera_size
        if episode_length is None:
            from robocasa.utils.dataset_registry_utils import get_task_horizon
            episode_length = int(get_task_horizon(task))
        self.task = task
        self.split = split
        self.max_episode_steps = episode_length
        self._env_factory = env_factory
        self._env = None
        self._elapsed_steps = 0
        self._done = True
        self._obs = None
        self._video_dir = Path(output_video_dir) if output_video_dir else None
        self._writer = None
        self._episode_index = -1
        self._video_fps = video_fps
        self._view_mode = view_mode
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({camera: spaces.Box(0, 255, (256, 256, 3), np.uint8)
                                   for camera in CAMERAS}),
            "robot_state": spaces.Dict({key: spaces.Box(-np.inf, np.inf, (dim,), np.float32)
                                        for key, dim in STATE_DIMS.items()}),
            "task_id": spaces.Text(max_length=256),
            "task_description": spaces.Text(max_length=2048, charset=(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
                ",.\n\t[]{}()!?'_:-/\";=&%+"
            )),
        })
        self.action_space = spaces.Dict({
            f"action.{key}": spaces.Box(-1.0, 1.0, (dim,), np.float32)
            for key, dim in ACTION_DIMS.items()
        })

    def _ensure_env(self):
        if self._env is None:
            factory = self._env_factory
            if factory is None:
                from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
                factory = RoboCasaGymEnv
            self._env = factory(env_name=self.task, split=self.split,
                                camera_widths=self.camera_size, camera_heights=self.camera_size,
                                obj_registries=("lightwheel",))

    def _format(self, raw):
        pixels = {}
        for camera in CAMERAS:
            image = rgb_image(raw[f"video.{camera}"], camera)
            if image.shape != (self.camera_size, self.camera_size, 3):
                raise ValueError(f"{camera}: expected {self.camera_size}-pixel square render, got {image.shape}")
            if self.camera_size != 256:
                import cv2
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            pixels[camera] = image
        return {
            "pixels": pixels,
            "robot_state": {key: vector(raw[f"state.{key}"], dim, f"state.{key}")
                            for key, dim in STATE_DIMS.items()},
            "task_id": self.task,
            "task_description": str(raw["annotation.human.task_description"]),
        }

    def _finish_video(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def render(self):
        if self._obs is None:
            raise RuntimeError("reset() must precede render()")
        cameras = CAMERAS if self._view_mode == "multi-view" else CAMERAS[:1]
        return np.concatenate([self._obs["pixels"][camera] for camera in cameras], axis=1)

    def _record_frame(self):
        if self._video_dir is None:
            return
        if self._writer is None:
            self._video_dir.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                str(self._video_dir / f"episode_{self._episode_index:06d}.mp4"), fps=self._video_fps)
        self._writer.append_data(self.render())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._finish_video()
        self._ensure_env()
        raw, info = self._env.reset(seed=seed, options=options)
        self._obs = self._format(raw)
        self._elapsed_steps = 0
        self._done = False
        self._episode_index += 1
        self._record_frame()
        return self._obs, {**info, "is_success": False, "task": self.task}

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() is required before stepping a new episode")
        raw, reward, done, truncated, info = self._env.step(validate_action(action))
        self._elapsed_steps += 1
        success = bool(info.get("success", False))
        terminated = bool(done or success)
        truncated = bool(truncated or self._elapsed_steps >= self.max_episode_steps) and not terminated
        self._done = terminated or truncated
        self._obs = self._format(raw)
        self._record_frame()
        info = {**info, "is_success": success, "task": self.task}
        if self._done:
            self._finish_video()
        return self._obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self._finish_video()
        finally:
            if self._env is not None:
                self._env.close()
                self._env = None
            self._done = True
