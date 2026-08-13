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

"""Opt-in torch.compile wrapping, shared by every reference policy.

The latency benchmark serves an eager / compiled / graph-captured variant of the
same checkpoint from one code path, selected by environment variable so the
server entrypoints need no new flags:

    VLA_TORCH_COMPILE=1                                          -> enable
    VLA_TORCH_COMPILE_MODE=default|reduce-overhead|max-autotune  -> inductor mode

``reduce-overhead`` is the CUDA-graph ("graph-captured") path, which is usually
the one that matters here: these policies run a flow-matching denoise loop at
batch size 1, where per-kernel launch overhead dominates.

Compilation is deliberately *not* fullgraph. These backbones contain data
dependent control flow, so a fullgraph requirement would simply fail to load;
allowing graph breaks measures what a user would actually get.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import torch


def compile_enabled() -> bool:
    return os.environ.get("VLA_TORCH_COMPILE") == "1"


def compile_mode() -> str:
    return os.environ.get("VLA_TORCH_COMPILE_MODE", "default")


def _clone_outputs(out: Any) -> Any:
    """Deep-copy tensors out of the CUDA-graph memory pool.

    Container types must survive the round trip. transformers' ``ModelOutput``
    subclasses ``dict``, so rebuilding it as a plain dict silently strips
    attribute access and the caller dies on ``.last_hidden_state``; KV caches
    are plain objects holding tensor lists and need reaching into by attribute.
    """
    if isinstance(out, torch.Tensor):
        return out.clone()
    if isinstance(out, tuple):
        # namedtuples take positional args, plain tuples take an iterable.
        if hasattr(out, "_fields"):
            return type(out)(*(_clone_outputs(o) for o in out))
        return tuple(_clone_outputs(o) for o in out)
    if isinstance(out, list):
        return [_clone_outputs(o) for o in out]
    if isinstance(out, dict):
        cloned = {k: _clone_outputs(v) for k, v in out.items()}
        if type(out) is dict:
            return cloned
        # dict subclass (ModelOutput, ...): keep the class and its attributes.
        try:
            new = copy.copy(out)
            for k, v in cloned.items():
                new[k] = v
            return new
        except Exception:
            return cloned
    # transformers Cache objects hold their tensors in these two lists. The
    # prefix KV cache is exactly the tensor that outlives a denoise step, so
    # this branch is what makes CUDA graphs usable at all here.
    if hasattr(out, "key_cache") and hasattr(out, "value_cache"):
        try:
            new = copy.copy(out)
            new.key_cache = [_clone_outputs(t) for t in out.key_cache]
            new.value_cache = [_clone_outputs(t) for t in out.value_cache]
            return new
        except Exception:
            return out
    return out


class _CudaGraphSafe(torch.nn.Module):
    """Wrap a compiled module so its outputs survive the next invocation.

    The compiled module is registered as a normal submodule so the wrapper is
    transparent to ``.parameters()`` / ``.to()`` / ``.state_dict()`` — GR00T-N1.5
    does ``next(self.parameters()).device`` on the parent, which raises
    ``StopIteration`` if the wrapper hides its parameters. ``__getattr__``
    forwards anything else to the compiled module, which in turn forwards to the
    original, so attribute access on the wrapped model keeps working.
    """

    def __init__(self, compiled: Any):
        super().__init__()
        self.compiled = compiled

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        torch.compiler.cudagraph_mark_step_begin()
        return _clone_outputs(self.compiled(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Reach the submodule dict directly; going through self.compiled here
        # would recurse back into __getattr__ before registration completes.
        inner = object.__getattribute__(self, "_modules").get("compiled")
        if inner is None or name == "compiled":
            raise AttributeError(name)
        return getattr(inner, name)


def _cudagraph_safe(compiled: Any) -> Any:
    """Guard a ``reduce-overhead`` region against output-pool reuse.

    With CUDA graphs, a compiled region's output tensors live in a graph memory
    pool that the *next* invocation overwrites. These policies call the compiled
    backbone many times per prediction and hold on to earlier outputs — the
    prefix KV cache spans the entire denoise loop — so the pool gets clobbered
    mid-prediction and PyTorch raises. Marking the step boundary and cloning
    outputs is the remedy PyTorch documents for exactly this case.

    The clone costs a copy per call, so ``reduce-overhead`` numbers are "CUDA
    graphs as actually usable here" rather than a pure graph-replay lower bound.
    """
    if isinstance(compiled, torch.nn.Module):
        return _CudaGraphSafe(compiled)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        torch.compiler.cudagraph_mark_step_begin()
        return _clone_outputs(compiled(*args, **kwargs))

    return wrapper


def _tolerate_untraceable_ops() -> None:
    """Let dynamo fall back to eager on regions it cannot trace.

    GR00T-N1.7's Qwen3-VL vision tower calls the flash-attn custom op
    ``flash_attn::_flash_attn_varlen_forward``, which declares ``max_seqlen_q``
    as a SymInt while transformers passes a 0-d tensor. Dynamo's fake-tensor
    pass raises on it and, by default, that aborts compilation of the whole
    model. ``suppress_errors`` downgrades such failures to a graph break, so
    the untraceable region runs eagerly (identical kernel, identical numerics)
    and everything around it still compiles.

    The honest reading: a model that leans on this is only partially compiled,
    so its "compiled" latency may land close to eager. That is the real
    out-of-the-box behaviour, and the report says so.
    """
    try:
        import torch._dynamo as dynamo
    except ImportError:
        return
    dynamo.config.suppress_errors = True
    print("[torch.compile] dynamo.config.suppress_errors=True "
          "(untraceable regions fall back to eager)", flush=True)


def maybe_compile(obj: Any, *, tag: str) -> Any:
    """Return ``torch.compile(obj)`` when enabled, else ``obj`` unchanged.

    Accepts either an ``nn.Module`` or a bound method — the GR00T pipelines
    drive inference through ``model.get_action`` rather than ``forward``, so
    the compile target there is a method, not the module.
    """
    if not compile_enabled():
        return obj
    mode = compile_mode()
    print(f"[torch.compile] {tag}: mode={mode!r}", flush=True)
    _tolerate_untraceable_ops()
    compiled = torch.compile(obj, mode=mode)
    if mode == "reduce-overhead":
        compiled = _cudagraph_safe(compiled)
    return compiled


def compile_tag() -> str:
    """Short label for the active variant, for logs and result filenames."""
    return f"compile-{compile_mode()}" if compile_enabled() else "eager"
