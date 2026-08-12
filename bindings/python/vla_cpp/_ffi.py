"""ctypes declarations for libvla. Mirrors include/vla.h field for field."""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_void_p,
)

ABI_VERSION = 1

OK = 0
ERR_ARG = -1
ERR_PREDICT = -2
ERR_EXCEPTION = -3

PIXEL_U8 = 0
PIXEL_F32_RGB_01 = 1

TIMING_NONE = 0
TIMING_PHASE = 1


class Config(ctypes.Structure):
    _fields_ = [
        ("n_img", c_int64),
        ("n_lang", c_int64),
        ("n_state", c_int64),
        ("n_prefix", c_int64),
        ("n_suffix", c_int64),
        ("n_full", c_int64),
        ("hidden", c_int64),
        ("expert_h", c_int64),
        ("intermediate", c_int64),
        ("expert_inter", c_int64),
        ("n_q_heads", c_int64),
        ("n_kv_heads", c_int64),
        ("head_dim", c_int64),
        ("q_full_dim", c_int64),
        ("kv_full_dim", c_int64),
        ("n_layers", c_int64),
        ("self_attn_every_n", c_int32),
        ("max_state_dim", c_int64),
        ("max_action_dim", c_int64),
        ("real_state_dim", c_int64),
        ("real_action_dim", c_int64),
        ("norm_eps", c_float),
        ("min_period", c_double),
        ("max_period", c_double),
        ("num_steps", c_int32),
        ("rms_eps", c_float),
        ("rope_n_dims", c_int32),
        ("rope_mode", c_int32),
        ("rope_freq_base", c_float),
        ("denormalized", c_int32),
    ]


class Image(ctypes.Structure):
    _fields_ = [
        ("data", c_void_p),
        ("w", c_int32),
        ("h", c_int32),
        ("format", c_int32),
    ]


class Inputs(ctypes.Structure):
    _fields_ = [
        ("images", POINTER(Image)),
        ("n_images", c_int32),
        ("precomputed_img_emb", POINTER(c_float)),
        ("n_img_views", c_int32),
        ("lang_tokens", POINTER(c_int32)),
        ("n_lang", c_int32),
        ("state", POINTER(c_float)),
        ("noise", POINTER(c_float)),
        ("attention_mask", POINTER(c_int32)),
        ("attention_mask_n", c_int32),
        ("timing_detail", c_int32),
    ]


class Stats(ctypes.Structure):
    _fields_ = [
        ("ms_total", c_float),
        ("ms_vision", c_float),
        ("ms_inference", c_float),
        ("ms_prefill", c_float),
        ("ms_denoise", c_float),
    ]


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libvla.dylib"
    if sys.platform == "win32":
        return "vla.dll"
    return "libvla.so"


def _candidates() -> list[str]:
    name = _library_name()
    here = os.path.dirname(os.path.abspath(__file__))
    found = []
    env = os.environ.get("VLA_LIBRARY")
    if env:
        found.append(env)
    # Bundled next to the package (what a wheel ships), then a local build tree.
    found.append(os.path.join(here, name))
    found.append(os.path.join(here, "lib", name))
    found.append(name)  # fall through to the loader search path
    return found


def load_library() -> ctypes.CDLL:
    errors = []
    lib = None
    for path in _candidates():
        try:
            lib = ctypes.CDLL(path)
            break
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if lib is None:
        raise OSError(
            "could not load libvla. Set VLA_LIBRARY to its path, or build it with "
            "`cmake --build <build> --target vla`.\nTried:\n  " + "\n  ".join(errors)
        )

    lib.vla_abi_version.restype = c_int32
    lib.vla_abi_version.argtypes = []

    lib.vla_model_load.restype = c_void_p
    lib.vla_model_load.argtypes = [c_char_p, c_char_p, c_char_p]

    lib.vla_model_free.restype = None
    lib.vla_model_free.argtypes = [c_void_p]

    lib.vla_model_config.restype = c_int32
    lib.vla_model_config.argtypes = [c_void_p, POINTER(Config)]

    lib.vla_predict.restype = c_int32
    lib.vla_predict.argtypes = [c_void_p, POINTER(Inputs), POINTER(POINTER(c_float)), POINTER(c_int64)]

    lib.vla_free_actions.restype = None
    lib.vla_free_actions.argtypes = [POINTER(c_float)]

    lib.vla_last_stats.restype = c_int32
    lib.vla_last_stats.argtypes = [c_void_p, POINTER(Stats)]

    got = lib.vla_abi_version()
    if got != ABI_VERSION:
        raise RuntimeError(f"libvla ABI {got}, this package expects {ABI_VERSION}")
    return lib
