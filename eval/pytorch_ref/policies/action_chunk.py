"""Action-chunk replay queue, shared by the GR00T pipelines.

The GR00T N1.6 / N1.7 pipelines predict a whole chunk `[B, T, D]` per forward
but the LIBERO client re-queries every env step, so without a queue they
re-predict from scratch at every step (`n_action_steps == 1`) and throw away
`T - 1` of every chunk.

vla.cpp's client replays `n_action_steps` actions from each chunk before
re-querying `vla-server`. To make the PyTorch reference and the GGUF port
comparable on success rate, this queue reproduces that cadence on the PyTorch
side: predict a chunk, hand out the first `n_action_steps` timesteps one per
call, then predict again from the observation current at that moment.

Timesteps are kept in the pipeline's native `dict[str, ndarray[B, T, D]]` shape
(sliced to `T == 1`) so the client-side parsers stay unchanged.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class ActionChunkQueue:
    """Hands out one timestep per call from a predicted `[B, T, D]` chunk."""

    def __init__(self, n_action_steps: int = 1) -> None:
        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        self.n_action_steps = n_action_steps
        self._queue: deque[dict[str, np.ndarray]] = deque()

    def clear(self) -> None:
        self._queue.clear()

    @property
    def empty(self) -> bool:
        return not self._queue

    def fill(self, chunk: dict[str, Any]) -> None:
        """Split a `dict[str, ndarray[B, T, D]]` chunk into per-timestep dicts."""
        horizon = min(arr.shape[1] for arr in chunk.values())
        n = min(self.n_action_steps, horizon)
        for t in range(n):
            self._queue.append({k: v[:, t : t + 1] for k, v in chunk.items()})

    def pop(self) -> dict[str, np.ndarray]:
        return self._queue.popleft()
