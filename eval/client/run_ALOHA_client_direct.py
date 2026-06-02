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

"""
ALOHA real-robot deployment node.

Subscribes to ROS2 camera and joint-state topics, calls vla-server over ZMQ,
and publishes joint commands to the Interbotix ALOHA controllers.

Usage (GR00T-N1.6 / new_embodiment):
  python run_ALOHA_client_direct.py \
      --arch gr00t_n1_6 \
      --stats-json /path/to/statistics.json \
      --embodiment new_embodiment \
      --task "pick up the cup and place it on the plate" \
      --vla-addr tcp://localhost:5555

For dual-arm, add --dual-arm.
Supported arches: smolvla, pi0, bitvla, evo1, gr00t_n1_5, gr00t_n1_6, gr00t_n1_7
"""

import argparse
import datetime
import json
import logging
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client.vla_cpp_client import VlaCppClient, ARCH_PRESETS

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from cv_bridge import CvBridge

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARM_DOF    = 6
GRIPPER_DOF = 1
JOINT_DOF  = ARM_DOF + GRIPPER_DOF  # 7 per arm

# ---------------------------------------------------------------------------
# GR00T-N1.6 ALOHA client (new_embodiment: single_arm + gripper)
# ---------------------------------------------------------------------------

class AlohaGr00tN16Client:
    """
    Handles GR00T-N1.6 inference for ALOHA joint-space control.

    The 'new_embodiment' statistics format uses flat 'single_arm' (6D) and
    'gripper' (1D) keys rather than the per-axis EEF keys assumed by the
    generic VlaCppClient gr00t_n1_6 path.  This class loads those stats
    directly and owns state normalisation / action un-normalisation.
    """

    _N_TOK_PER_VIEW  = 81      # (252 // 14 // 2) ** 2 for image_size=252
    _IMG_PAD_ID      = 151669  # <IMG_CONTEXT> token id
    _CROP_FRACTION   = 0.95
    _ACTION_HORIZON  = 16
    _MAX_STATE_DIM   = 128

    def __init__(
        self,
        vla_addr: str,
        stats_json: str,
        embodiment: str = "new_embodiment",
        image_size: int = 252,
        n_action_steps: int = 1,
        recv_timeout_ms: int = 60_000,
        tokenizer: str | None = None,
    ):
        # Reuse VlaCppClient only for its ZMQ socket, protobuf module, and tokenizer.
        # Pass stats_json=None so it skips its own state normalisation.
        self._inner = VlaCppClient(
            vla_addr       = vla_addr,
            arch           = "gr00t_n1_6",
            tokenizer_name = tokenizer,
            image_size     = image_size,
            max_state_dim  = self._MAX_STATE_DIM,
            real_action_dim= JOINT_DOF,
            image_keys     = [],
            max_length     = 48,
            recv_timeout_ms= recv_timeout_ms,
            n_action_steps = 1,
            stats_json     = None,
            bitvla_unnorm_key = None,
        )

        blob = json.loads(Path(stats_json).read_text())
        if embodiment not in blob:
            raise KeyError(
                f"Embodiment '{embodiment}' not found in {stats_json}. "
                f"Available: {list(blob)}")
        st = blob[embodiment]

        # State normalisation: single_arm (6D) + gripper (1D) → [-1, 1]
        s_min = np.array(
            st["state"]["single_arm"]["min"] + st["state"]["gripper"]["min"],
            dtype=np.float32)
        s_max = np.array(
            st["state"]["single_arm"]["max"] + st["state"]["gripper"]["max"],
            dtype=np.float32)
        s_rng = s_max - s_min

        def _state_norm(x, mn=s_min, rng=s_rng):
            return np.clip(2.0 * (x - mn) / np.where(rng > 1e-8, rng, 1.0) - 1.0,
                           -1.0, 1.0).astype(np.float32)
        self._state_norm = _state_norm

        # Action un-normalisation: server outputs values in [-1, 1]
        a_min = np.array(
            st["action"]["single_arm"]["min"] + st["action"]["gripper"]["min"],
            dtype=np.float32)
        a_max = np.array(
            st["action"]["single_arm"]["max"] + st["action"]["gripper"]["max"],
            dtype=np.float32)
        a_rng = a_max - a_min

        def _action_unnorm(chunk, mn=a_min, rng=a_rng):
            norm = np.clip(chunk[..., :JOINT_DOF].astype(np.float32), -1.0, 1.0)
            return ((norm + 1.0) * 0.5 * rng + mn).astype(np.float32)
        self._action_unnorm = _action_unnorm

        # Recalculate n_tok_per_view from actual image_size
        self._n_tok_per_view = (image_size // 14 // 2) ** 2
        self.image_size = image_size
        self.n_action_steps = n_action_steps
        self._queue: deque = deque()

        print(
            f"AlohaGr00tN16Client: embodiment={embodiment}  "
            f"image_size={image_size}  tok/view={self._n_tok_per_view}  "
            f"n_action_steps={n_action_steps}  via {stats_json}",
            flush=True,
        )

    def reset(self):
        self._queue.clear()

    def get_action(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray,
                   left_state: np.ndarray, task: str) -> np.ndarray:
        """Returns (JOINT_DOF,) float32 joint positions for one step."""
        if not self._queue:
            chunk = self._predict_chunk(front_rgb, wrist_rgb, left_state, task)
            for row in chunk[: self.n_action_steps]:
                self._queue.append(row.copy())
        return self._queue.popleft()

    def predict_chunk(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray,
                      left_state: np.ndarray, task: str) -> np.ndarray:
        """Returns (ACTION_HORIZON, JOINT_DOF) float32 — full un-normalised chunk."""
        return self._predict_chunk(front_rgb, wrist_rgb, left_state, task)

    def _predict_chunk(self, front_rgb, wrist_rgb, left_state, task):
        c = self._inner

        # Image processing (pad→crop→resize, same as VlaCppClient._gr00t_n16_image_transform)
        front_u8 = VlaCppClient._gr00t_n16_image_transform(
            front_rgb, self.image_size, self._CROP_FRACTION)
        wrist_u8 = VlaCppClient._gr00t_n16_image_transform(
            wrist_rgb, self.image_size, self._CROP_FRACTION)

        # State normalisation
        state_raw = self._state_norm(left_state[:JOINT_DOF].astype(np.float32))
        state_padded = np.zeros(self._MAX_STATE_DIM, dtype=np.float32)
        state_padded[:state_raw.size] = state_raw

        # Tokenise
        language = re.sub(r"[^\w\s]", "", task.lower())
        def _img_block(n):
            return (f"<image {n}><img>"
                    + "<IMG_CONTEXT>" * self._n_tok_per_view
                    + "</img>")
        user_content = language + _img_block(1) + _img_block(2)
        conv = [{"role": "user", "content": user_content}]
        text = c.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        ids = c.tok(text, add_special_tokens=False)["input_ids"]

        n_slots = sum(1 for t in ids if t == self._IMG_PAD_ID)
        if n_slots != 2 * self._n_tok_per_view:
            raise RuntimeError(
                f"AlohaGr00tN16Client: expected {2 * self._n_tok_per_view} "
                f"<IMG_CONTEXT> slots in prompt, got {n_slots}. "
                f"Check tokenizer special tokens.")

        # Build protobuf request
        req = c.pb.PredictRequest()
        req.request_id = c._step
        c._step += 1

        for img_u8 in (front_u8, wrist_u8):
            img_f32 = np.ascontiguousarray(img_u8.astype(np.float32) / 255.0)
            ip = req.images.add()
            ip.encoding = c.pb.Image.F32_RGB_01
            ip.height   = img_f32.shape[0]
            ip.width    = img_f32.shape[1]
            ip.data     = img_f32.tobytes()

        req.lang_tokens.extend(int(t) for t in ids)
        req.state.extend(float(x) for x in state_padded)

        c.sock.send(req.SerializeToString())
        body = c.sock.recv()
        resp = c.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"vla-server error: {resp.error}")

        chunk = (np.array(resp.action_chunk, dtype=np.float32)
                   .reshape(resp.chunk_size, resp.action_dim))
        return self._action_unnorm(chunk[: self._ACTION_HORIZON])

    def close(self):
        self._inner.close()


# ---------------------------------------------------------------------------
# Generic per-arch obs building (smolvla / pi0 / bitvla / evo1 / gr00t_n1_{5,7})
# ---------------------------------------------------------------------------

def _hwc_to_chw_f32(img: np.ndarray, size: int) -> np.ndarray:
    r = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return np.transpose(r.astype(np.float32) / 255.0, (2, 0, 1))

def _hwc_u8(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)


def build_obs(arch, front_rgb, wrist_rgb, left_state, task,
              image_size, max_state_dim, right_state=None):
    if arch in ("smolvla", "pi0", "bitvla"):
        state = left_state.astype(np.float32)
        if right_state is not None:
            state = np.concatenate([state, right_state.astype(np.float32)])
        padded = np.zeros(max_state_dim, dtype=np.float32)
        padded[:state.size] = state
        return {
            "observation.images.image":  _hwc_to_chw_f32(front_rgb, image_size),
            "observation.images.image2": _hwc_to_chw_f32(wrist_rgb, image_size),
            "observation.state": padded,
            "task": task,
        }

    if arch == "evo1":
        state = left_state.astype(np.float32)
        padded = np.zeros(max_state_dim, dtype=np.float32)
        padded[:state.size] = state
        front_u8 = _hwc_u8(front_rgb, image_size)
        wrist_u8 = _hwc_u8(wrist_rgb, image_size)
        return {
            "image": [front_u8, wrist_u8, np.zeros_like(front_u8)],
            "state": padded,
            "prompt": task,
            "image_mask": [1, 1, 0],
        }

    if arch in ("gr00t_n1_5", "gr00t_n1_7"):
        def _s(v): return np.array([[[float(v)]]], dtype=np.float32)
        j = left_state.astype(np.float32)
        gripper = j[6:7] if j.size > 6 else np.zeros(1, dtype=np.float32)
        return {
            "video.image":       _hwc_u8(front_rgb, image_size)[None, None],
            "video.wrist_image": _hwc_u8(wrist_rgb, image_size)[None, None],
            "state.x":     _s(j[0]), "state.y":    _s(j[1]), "state.z":    _s(j[2]),
            "state.roll":  _s(j[3]), "state.pitch": _s(j[4]), "state.yaw":  _s(j[5]),
            "state.gripper": gripper.reshape(1, 1, -1),
            "task": task,
        }

    raise ValueError(f"Unknown arch: {arch!r}")


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class AlohaInferenceNode(Node):

    def __init__(self, args: argparse.Namespace):
        super().__init__("aloha_vla_inference")

        self.args          = args
        self.arch          = args.arch
        self.task          = args.task
        self.executed_length = args.executed_length
        self.smooth_step   = args.smooth_step
        self.dual_arm      = args.dual_arm

        self._setup_logger()

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.bridge = CvBridge()

        # Publishers
        self.left_arm_pub  = self.create_publisher(JointGroupCommand,  "/follower_left/commands/joint_group",  10)
        self.left_hand_pub = self.create_publisher(JointSingleCommand, "/follower_left/commands/joint_single", 10)
        if self.dual_arm:
            self.right_arm_pub  = self.create_publisher(JointGroupCommand,  "/follower_right/commands/joint_group",  10)
            self.right_hand_pub = self.create_publisher(JointSingleCommand, "/follower_right/commands/joint_single", 10)

        # Subscribers
        self.create_subscription(Image,      "/camera_high/camera/color/image_raw",       self._cb_front,      qos)
        self.create_subscription(Image,      "/camera_wrist_left/camera/color/image_raw", self._cb_wrist_left, qos)
        self.create_subscription(JointState, "/follower_left/joint_states",                self._cb_left_state, 10)
        if self.dual_arm:
            self.create_subscription(JointState, "/follower_right/joint_states", self._cb_right_state, 10)

        # Sensor buffers
        self.lock             = Lock()
        self.front_rgb        = None
        self.wrist_left_rgb   = None
        self.left_state       = None
        self.right_state      = None
        self.prev_left_state  = None
        self.prev_right_state = None

        # Build client
        image_size    = args.image_size    or ARCH_PRESETS[self.arch]["image_size"]
        max_state_dim = args.max_state_dim or ARCH_PRESETS[self.arch].get("max_state_dim", 32)
        self.image_size    = image_size
        self.max_state_dim = max_state_dim

        self._use_aloha_gr00t = (
            self.arch == "gr00t_n1_6"
            and args.stats_json is not None
            and args.embodiment in ("new_embodiment",)
        )

        if self._use_aloha_gr00t:
            self.client = AlohaGr00tN16Client(
                vla_addr       = args.vla_addr,
                stats_json     = args.stats_json,
                embodiment     = args.embodiment,
                image_size     = image_size,
                n_action_steps = args.n_action_steps,
                recv_timeout_ms= args.recv_timeout_ms,
                tokenizer      = args.tokenizer,
            )
        else:
            image_keys = (
                ["observation.images.image", "observation.images.image2"]
                if self.arch not in ("evo1", "gr00t_n1_5", "gr00t_n1_6", "gr00t_n1_7")
                else []
            )
            self.client = VlaCppClient(
                vla_addr          = args.vla_addr,
                arch              = self.arch,
                tokenizer_name    = args.tokenizer,
                image_size        = image_size,
                max_state_dim     = max_state_dim,
                real_action_dim   = JOINT_DOF * (2 if self.dual_arm else 1),
                image_keys        = image_keys,
                max_length        = args.max_length,
                recv_timeout_ms   = args.recv_timeout_ms,
                n_action_steps    = args.n_action_steps,
                stats_json        = args.stats_json,
                bitvla_unnorm_key = args.bitvla_unnorm_key,
            )
        self.client.reset()
        self.log.info(f"client ready  arch={self.arch}  addr={args.vla_addr}")

        self._end_action = True

        # Async inference: overlap next prediction with current chunk execution.
        # Only the _async_worker thread ever touches self.client / ZMQ socket.
        self._async_mode = args.async_infer
        if self._async_mode:
            self._async_trigger  = Event()   # set to request a new inference
            self._async_ready    = Event()   # set when the prefetched chunk is ready
            self._async_stop     = Event()   # set on shutdown to wake the worker
            self._async_lock     = Lock()
            self._async_chunk    = None      # prefetched raw chunk (HORIZON × action_dim)
            self._async_obs      = None      # obs snapshot captured at trigger time
            self._async_err: Exception | None = None
            self._async_cached   = None      # chunk ready for next _run_inference call
            self._async_worker   = Thread(target=self._async_infer_worker, daemon=True)
            self._async_worker.start()

        self._infer_thread = Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------

    def _setup_logger(self):
        log_dir = Path(os.path.expanduser("~/vla_aloha_logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log = logging.getLogger("aloha_inference")
        self.log.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
        fh = logging.FileHandler(log_dir / f"{ts}_aloha_inference.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        self.log.addHandler(fh)
        self.log.addHandler(sh)

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def _cb_front(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.front_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.log.warning(f"front cam: {e}")

    def _cb_wrist_left(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.wrist_left_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.log.warning(f"wrist-left cam: {e}")

    def _cb_left_state(self, msg: JointState):
        try:
            pos = np.array(msg.position[:JOINT_DOF], dtype=np.float32)
            with self.lock:
                self.left_state = pos
                if self.prev_left_state is None:
                    self.prev_left_state = pos[None]
        except Exception as e:
            self.log.warning(f"left joint state: {e}")

    def _cb_right_state(self, msg: JointState):
        try:
            pos = np.array(msg.position[:JOINT_DOF], dtype=np.float32)
            with self.lock:
                self.right_state = pos
                if self.prev_right_state is None:
                    self.prev_right_state = pos[None]
        except Exception as e:
            self.log.warning(f"right joint state: {e}")

    # ------------------------------------------------------------------
    # Async inference worker (only this thread touches self.client / ZMQ)
    # ------------------------------------------------------------------

    def _async_infer_worker(self):
        while not self._async_stop.is_set():
            # Wait for either a trigger or a stop signal.
            self._async_trigger.wait()
            if self._async_stop.is_set():
                break
            self._async_trigger.clear()

            with self._async_lock:
                obs_snap = self._async_obs

            if obs_snap is None:
                continue

            front, wrist, left_state, right_state = obs_snap
            t0 = time.time()
            try:
                chunk = self._get_chunk(front, wrist, left_state, right_state)
                err   = None
            except Exception as e:
                chunk = None
                err   = e

            with self._async_lock:
                self._async_chunk = chunk
                self._async_err   = err
            self._async_ready.set()
            self.log.debug(
                f"async prefetch {'ok' if err is None else 'err'}  "
                f"{(time.time()-t0)*1000:.1f} ms"
            )

    def _trigger_next_infer(self, predicted_left=None, predicted_right=None):
        """Snapshot current obs and kick the async worker.

        If predicted_left/right are given (predicted end-state of the current
        chunk) they replace the live sensor readings as the state input, giving
        the model a more accurate view of where the arm will be when the next
        chunk starts executing.
        """
        with self.lock:
            left_state  = predicted_left  if predicted_left  is not None else self.left_state
            right_state = predicted_right if predicted_right is not None else (
                self.right_state if self.dual_arm else None)
            obs_snap = (
                self.front_rgb,
                self.wrist_left_rgb,
                left_state,
                right_state,
            )
        with self._async_lock:
            self._async_obs   = obs_snap
            self._async_chunk = None
            self._async_err   = None
        self._async_ready.clear()
        self._async_trigger.set()

    def _collect_prefetched(self, timeout: float = 2.0):
        """
        Wait for the prefetched chunk.  On timeout, clear the cache so the
        next _run_inference falls back to a fresh synchronous request.
        Returns the raw chunk or raises on error.
        """
        if not self._async_ready.wait(timeout=timeout):
            self.log.warning(
                f"async prefetch timed out after {timeout:.1f}s — "
                "next call will re-trigger synchronously"
            )
            self._async_cached = None
            raise TimeoutError("async inference timed out")
        with self._async_lock:
            err   = self._async_err
            chunk = self._async_chunk
        if err is not None:
            raise err
        return chunk

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    def _infer_loop(self):
        time.sleep(4.0)
        while rclpy.ok():
            if self._end_action:
                self._run_inference()

    def _run_inference(self):
        self._end_action = False
        if self._async_mode:
            self._run_inference_async()
        else:
            self._run_inference_sync()

    # ------ synchronous path (original behaviour) -------------------------

    def _run_inference_sync(self):
        with self.lock:
            front       = self.front_rgb
            wrist       = self.wrist_left_rgb
            left_state  = self.left_state
            right_state = self.right_state if self.dual_arm else None
            prev_left   = self.prev_left_state                              # noqa: F841  snapshot for downstream consumers
            prev_right  = self.prev_right_state if self.dual_arm else None  # noqa: F841  snapshot for downstream consumers

        if front is None or wrist is None or left_state is None:
            self.log.info(
                f"Waiting for data  front={front is not None}"
                f"  wrist={wrist is not None}"
                f"  left_state={left_state is not None}"
            )
            self._end_action = True
            return

        if self.dual_arm and right_state is None:
            self.log.info("Waiting for right arm joint states...")
            self._end_action = True
            return

        t0 = time.time()
        try:
            chunk = self._get_chunk(front, wrist, left_state, right_state)
        except Exception as e:
            self.log.error(f"inference failed: {e}")
            self._end_action = True
            return
        self.log.info(f"inference {(time.time() - t0) * 1000:.1f} ms  chunk={chunk.shape}")

        self._execute_chunk(chunk)

    # ------ async path (overlap inference N+1 with execution of N) --------
    #
    # Timeline:
    #   obtain chunk N  →  trigger prefetch N+1  →  execute chunk N  →  collect N+1
    #
    # Obs snapshot for chunk N+1 is taken at trigger time (sensor-at-trigger).
    # Smoothing is applied here (at execution time) because it needs prev_state
    # which is only known after chunk N finishes.
    #
    # If inference is faster than execution the prefetch will be fully hidden.
    # If inference is slower you'll still block briefly at _collect_prefetched.

    def _run_inference_async(self):
        # ---- obtain chunk for this iteration ----
        if self._async_cached is not None:
            chunk = self._async_cached
            self._async_cached = None
            self.log.debug("async: using prefetched chunk")
        else:
            # First call or after a timeout fallback: trigger and wait.
            with self.lock:
                if self.front_rgb is None or self.wrist_left_rgb is None or self.left_state is None:
                    self.log.info(
                        f"Waiting for data  front={self.front_rgb is not None}"
                        f"  wrist={self.wrist_left_rgb is not None}"
                        f"  left_state={self.left_state is not None}"
                    )
                    self._end_action = True
                    return
                if self.dual_arm and self.right_state is None:
                    self.log.info("Waiting for right arm joint states...")
                    self._end_action = True
                    return

            t0 = time.time()
            self._trigger_next_infer()
            try:
                chunk = self._collect_prefetched()
            except Exception as e:
                self.log.error(f"inference failed: {e}")
                self._end_action = True
                return
            self.log.info(f"inference {(time.time()-t0)*1000:.1f} ms  chunk={chunk.shape}")

        # ---- trigger prefetch for the NEXT chunk (optionally delayed) ------
        # Delaying the trigger gives a fresher image snapshot: the arm has moved
        # closer to where it will be when chunk N+1 starts executing.
        # Optimal delay ≈ execution_ms - avg_inference_ms (leave ~50ms margin).
        # State input uses the predicted end-state of chunk N so the model sees
        # where the arm will actually be, independent of when the trigger fires.
        pred_left  = chunk[-1, :JOINT_DOF].copy()
        pred_right = (chunk[-1, JOINT_DOF:JOINT_DOF * 2].copy()
                      if self.dual_arm and chunk.shape[-1] >= JOINT_DOF * 2
                      else None)
        if self.args.async_trigger_delay_ms > 0:
            time.sleep(self.args.async_trigger_delay_ms / 1000.0)
        self._trigger_next_infer(predicted_left=pred_left, predicted_right=pred_right)

        # ---- execute current chunk ----------------------------------------
        self._execute_chunk(chunk)

        # ---- collect the prefetched chunk (should be ready by now) --------
        # expected_exec_ms = executed_length * smooth_step * 5 ms
        try:
            self._async_cached = self._collect_prefetched()
            self.log.debug("async: prefetch ready, cached for next step")
        except Exception as e:
            self.log.warning(f"async prefetch failed, will retry sync: {e}")
            self._async_cached = None

    # ------ shared execution helper ----------------------------------------

    def _execute_chunk(self, chunk: np.ndarray):
        with self.lock:
            prev_left  = self.prev_left_state
            prev_right = self.prev_right_state if self.dual_arm else None

        cur = prev_left[0] if prev_left is not None else np.zeros(JOINT_DOF, dtype=np.float32)
        if self.dual_arm:
            cur_r = prev_right[0] if prev_right is not None else np.zeros(JOINT_DOF, dtype=np.float32)
            cur   = np.concatenate([cur, cur_r])

        if self.smooth_step > 1:
            chunk = self._smooth_action(np.concatenate([cur[None], chunk], axis=0))

        t_exec = time.time()
        n_steps = min(self.executed_length * self.smooth_step, len(chunk))
        for i in range(n_steps):
            t_step = time.time()
            row = chunk[i]
            self._pub_left(row[:ARM_DOF], row[ARM_DOF])
            if self.dual_arm and row.size >= JOINT_DOF * 2:
                self._pub_right(row[JOINT_DOF:JOINT_DOF + ARM_DOF], row[JOINT_DOF + ARM_DOF])
            time.sleep(max(0.0, 1.0 / 200.0 - (time.time() - t_step)))

        with self.lock:
            last = chunk[n_steps - 1]
            self.prev_left_state = last[:JOINT_DOF][None]
            if self.dual_arm and last.size >= JOINT_DOF * 2:
                self.prev_right_state = last[JOINT_DOF:JOINT_DOF * 2][None]

        self.log.info(f"execution {(time.time() - t_exec) * 1000:.1f} ms  ({n_steps} steps)")
        self._end_action = True

    def _get_chunk(self, front, wrist, left_state, right_state) -> np.ndarray:
        """Returns (executed_length, action_dim) float32."""
        n = self.executed_length

        if self._use_aloha_gr00t:
            chunk = self.client.predict_chunk(front, wrist, left_state, self.task)
            return chunk[:n]

        # Standard VlaCppClient path: collect n steps from queue
        obs = build_obs(self.arch, front, wrist, left_state, self.task,
                        self.image_size, self.max_state_dim, right_state)
        rows = []
        for _ in range(n):
            rows.append(self.client.get_action(obs))
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _pub_left(self, arm: np.ndarray, gripper: float):
        arm_msg = JointGroupCommand()
        arm_msg.name = "arm"
        arm_msg.cmd  = arm.tolist()
        self.left_arm_pub.publish(arm_msg)

        grp_msg = JointSingleCommand()
        grp_msg.name = "gripper"
        grp_msg.cmd  = float(gripper)
        self.left_hand_pub.publish(grp_msg)
        self.get_logger().debug(f"left  arm={arm_msg.cmd}  gripper={grp_msg.cmd:.4f}")

    def _pub_right(self, arm: np.ndarray, gripper: float):
        arm_msg = JointGroupCommand()
        arm_msg.name = "arm"
        arm_msg.cmd  = arm.tolist()
        self.right_arm_pub.publish(arm_msg)

        grp_msg = JointSingleCommand()
        grp_msg.name = "gripper"
        grp_msg.cmd  = float(gripper)
        self.right_hand_pub.publish(grp_msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _smooth_action(self, actions: np.ndarray) -> np.ndarray:
        segments = []
        for i in range(len(actions) - 1):
            segments.append(np.linspace(actions[i], actions[i + 1],
                                        self.smooth_step, endpoint=True))
        return np.concatenate(segments, axis=0)

    def destroy_node(self):
        if self._async_mode:
            self._async_stop.set()
            self._async_trigger.set()   # wake the worker so it can exit
            self._async_worker.join(timeout=2.0)
        self._infer_thread.join(timeout=2.0)
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", choices=sorted(ARCH_PRESETS), default="gr00t_n1_6")
    parser.add_argument("--task", type=str,
        default="pick up the object and place it in the target location")
    parser.add_argument("--vla-addr", type=str, default="tcp://localhost:5555")
    parser.add_argument("--tokenizer",      type=str, default=None)
    parser.add_argument("--image-size",     type=int, default=None)
    parser.add_argument("--max-state-dim",  type=int, default=None)
    parser.add_argument("--max-length",     type=int, default=48)
    parser.add_argument("--recv-timeout-ms", type=int, default=60_000)
    parser.add_argument("--n-action-steps", type=int, default=16,
        help="Steps buffered from each chunk (default 16 = full GR00T-N1.6 horizon).")
    parser.add_argument("--executed-length", type=int, default=8,
        help="Steps to execute per inference call before re-querying.")
    parser.add_argument("--smooth-step", type=int, default=20,
        help="Interpolation sub-steps between actions (1 = no smoothing).")
    parser.add_argument("--dual-arm", action="store_true")
    parser.add_argument("--stats-json",       type=str, default=None)
    parser.add_argument("--bitvla-unnorm-key", type=str, default=None)
    parser.add_argument("--embodiment", type=str, default="new_embodiment",
        help="Embodiment key in statistics.json (default: new_embodiment).")
    parser.add_argument("--async-infer", action="store_true",
        help="Overlap next inference with current chunk execution to hide latency. "
             "Obs snapshot is taken at trigger time (sensor-at-trigger). "
             "With defaults (executed_length=8, smooth_step=20) execution takes ~860 ms; "
             "if inference is faster it will be fully hidden.")
    parser.add_argument("--async-trigger-delay-ms", type=int, default=0,
        help="Wait N ms into chunk execution before triggering the next prefetch. "
             "Gives a fresher obs snapshot; set to execution_ms - avg_inference_ms - 50. "
             "Example: --async-trigger-delay-ms 300 (for ~510ms inference, ~860ms execution). "
             "Only used with --async-infer.")

    args = parser.parse_args(sys.argv[1:])

    rclpy.init()
    node = AlohaInferenceNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
