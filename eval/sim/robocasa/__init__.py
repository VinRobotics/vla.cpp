# Copyright 2026 VinRobotics - Apache-2.0
"""RoboCasa365 evaluation support, isolated from the LIBERO dependency stack."""

import os

# Set these before importing MuJoCo / PyOpenGL. Respect explicit backend choices.
os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] in {"egl", "osmesa"}:
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
