"""Python bindings for vla.cpp.

    import vla_cpp
    model = vla_cpp.load("smolvla-libero.gguf")
    actions = model.predict(image_hwc_uint8, tokens=[1, 100, 200, 2])

Actions are [rows, max_action_dim] float32; only the first
``model.config.real_action_dim`` columns carry values.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import POINTER, c_float, c_int32, c_int64
from typing import Sequence

from . import _ffi
from ._ffi import PIXEL_F32_RGB_01, PIXEL_U8, TIMING_NONE, TIMING_PHASE

__all__ = ["Model", "load", "PIXEL_U8", "PIXEL_F32_RGB_01", "TIMING_NONE", "TIMING_PHASE"]

_lib = None


def _lib_handle():
    global _lib
    if _lib is None:
        _lib = _ffi.load_library()
    return _lib


def _as_f32_array(values, length: int, name: str):
    """Accept a numpy array, a list, or None. Returns (ptr, keepalive)."""
    if values is None:
        return None, None
    buf = (c_float * length)()
    try:  # numpy fast path without importing numpy as a hard dependency
        mv = memoryview(values)
        if mv.format == "f" and mv.nbytes == length * 4 and mv.c_contiguous:
            ctypes.memmove(buf, (ctypes.c_char * mv.nbytes).from_buffer_copy(mv), mv.nbytes)
            return buf, buf
    except TypeError:
        pass
    seq = list(values)
    if len(seq) != length:
        raise ValueError(f"{name} has {len(seq)} values, model expects {length}")
    for i, v in enumerate(seq):
        buf[i] = float(v)
    return buf, buf


class Model:
    """A loaded checkpoint. Free it with ``close()`` or a ``with`` block."""

    def __init__(self, handle, lib):
        self._mu = threading.Lock()
        self._h = handle
        self._lib = lib
        cfg = _ffi.Config()
        rc = lib.vla_model_config(handle, ctypes.byref(cfg))
        if rc != _ffi.OK:
            raise RuntimeError(f"vla_model_config failed ({rc})")
        self.config = cfg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        with self._mu:
            if getattr(self, "_h", None):
                self._lib.vla_model_free(self._h)
                self._h = None

    def __del__(self):
        self.close()

    @staticmethod
    def _pack_images(images, pixel_format):
        views = images if isinstance(images, (list, tuple)) else [images]
        if not views:
            raise ValueError("at least one image is required")

        img_array = (_ffi.Image * len(views))()
        keep = []
        for i, im in enumerate(views):
            mv = memoryview(im)
            if not mv.c_contiguous:
                raise ValueError("image must be C-contiguous")
            shape = mv.shape
            if len(shape) != 3 or shape[2] != 3:
                raise ValueError(f"image must be HxWx3, got {shape}")
            want = "f" if pixel_format == PIXEL_F32_RGB_01 else "B"
            if mv.format != want:
                raise ValueError(f"image format {mv.format!r} does not match pixel_format (expected {want!r})")
            raw = (ctypes.c_char * mv.nbytes).from_buffer_copy(mv)
            keep.append(raw)
            img_array[i].data = ctypes.cast(raw, ctypes.c_void_p)
            img_array[i].h = int(shape[0])
            img_array[i].w = int(shape[1])
            img_array[i].format = int(pixel_format)
        return views, img_array, keep

    def encode_images(self, images, pixel_format: int = PIXEL_U8):
        """Run only the vision tower; returns float32 [n_views * tokens_per_view, hidden].

        Pass the result to predict(img_emb=..., n_img_views=len(images)), possibly
        on a model loaded by another process on another device. Not every arch
        supports it.
        """
        if not hasattr(self._lib, "vla_encode_images"):
            raise RuntimeError("this libvla has no vla_encode_images")
        views, img_array, keep = self._pack_images(images, pixel_format)
        out = POINTER(c_float)()
        n = c_int64()
        with self._mu:
            if self._h is None:
                raise RuntimeError("model is closed")
            rc = self._lib.vla_encode_images(self._h, img_array, len(views), ctypes.byref(out), ctypes.byref(n))
            if rc != _ffi.OK:
                raise RuntimeError(f"vla_encode_images failed ({rc})")
            try:
                import numpy as np
                flat = np.ctypeslib.as_array(out, shape=(n.value,)).copy()
            finally:
                self._lib.vla_free_actions(out)
        hidden = int(self.config.hidden) or 1
        return flat.reshape(-1, hidden)

    def predict(self, images, tokens: Sequence[int], state=None, noise=None,
                pixel_format: int = PIXEL_U8, timing: int = TIMING_NONE,
                img_emb=None, n_img_views: int = 0):
        """Run one forward pass.

        images: one HWC array, or a sequence of them for multi-view. uint8 RGB by
        default; pass pixel_format=PIXEL_F32_RGB_01 for float RGB in [0, 1].
        img_emb: optional output of encode_images (then images may be None, and
        n_img_views must give the number of views it holds).
        """
        if img_emb is not None:
            import numpy as np
            emb = np.ascontiguousarray(img_emb, dtype=np.float32)
            if n_img_views < 1:
                raise ValueError("n_img_views must be >= 1 with img_emb")
            views, img_array, keep = [], None, [emb]
        else:
            views, img_array, keep = self._pack_images(images, pixel_format)

        tok = list(tokens)
        if not tok:
            raise ValueError("tokens must not be empty")
        tok_buf = (c_int32 * len(tok))(*[int(t) for t in tok])

        state_ptr, state_keep = _as_f32_array(
            state if state is not None else [0.0] * int(self.config.max_state_dim),
            int(self.config.max_state_dim), "state")
        keep.append(state_keep)

        noise_len = int(self.config.max_action_dim) * int(self.config.n_suffix)
        noise_ptr, noise_keep = _as_f32_array(noise, noise_len, "noise")
        keep.append(noise_keep)

        cin = _ffi.Inputs()
        if img_emb is not None:
            cin.precomputed_img_emb = emb.ctypes.data_as(POINTER(c_float))
            cin.n_img_views = int(n_img_views)
        else:
            cin.images = img_array
            cin.n_images = len(views)
        cin.lang_tokens = tok_buf
        cin.n_lang = len(tok)
        if state_ptr is not None:
            cin.state = ctypes.cast(state_ptr, POINTER(c_float))
        if noise_ptr is not None:
            cin.noise = ctypes.cast(noise_ptr, POINTER(c_float))
        cin.timing_detail = int(timing)

        out = POINTER(c_float)()
        n = c_int64()
        with self._mu:
            if self._h is None:
                raise RuntimeError("model is closed")
            rc = self._lib.vla_predict(self._h, ctypes.byref(cin), ctypes.byref(out), ctypes.byref(n))
            if rc != _ffi.OK:
                raise RuntimeError(f"vla_predict failed ({rc})")
            try:
                flat = [out[i] for i in range(n.value)]
            finally:
                self._lib.vla_free_actions(out)

        cols = int(self.config.max_action_dim) or 1
        rows = len(flat) // cols if cols else len(flat)
        try:
            import numpy as np
            return np.asarray(flat, dtype="float32").reshape(rows, cols)
        except ImportError:
            return [flat[r * cols:(r + 1) * cols] for r in range(rows)]

    def last_stats(self) -> _ffi.Stats:
        st = _ffi.Stats()
        with self._mu:
            rc = self._lib.vla_last_stats(self._h, ctypes.byref(st))
        if rc != _ffi.OK:
            raise RuntimeError(f"vla_last_stats failed ({rc})")
        return st


def load(ckpt_path: str, mmproj_path: str | None = None, config_path: str | None = None) -> Model:
    """Load a checkpoint. mmproj_path is accepted and ignored; every arch bundles its vision tower."""
    lib = _lib_handle()
    handle = lib.vla_model_load(
        mmproj_path.encode() if mmproj_path else None,
        ckpt_path.encode(),
        config_path.encode() if config_path else None,
    )
    if not handle:
        raise RuntimeError(f"could not load {ckpt_path}")
    return Model(handle, lib)
