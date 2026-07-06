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

"""Unit tests for the clip->vit tensor-name remap in the mmproj merge scripts.
A wrong mapping silently breaks the in-tree tower. remap() is pure, so the heavy
`gguf` import is stubbed."""

import importlib.util
import pathlib
import sys
import types


def _load(name):
    sys.modules.setdefault("gguf", types.ModuleType("gguf"))
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check(remap):
    # clip block names -> in-tree vit block names, with the attn/ffn sub-renames.
    assert remap("v.blk.5.attn_out.weight") == "vit.blk.5.attn_o.weight"
    assert remap("v.blk.0.ffn_up.weight") == "vit.blk.0.fc1.weight"
    assert remap("v.blk.11.ffn_down.bias") == "vit.blk.11.fc2.bias"
    # a passthrough sub-name keeps its suffix
    assert remap("v.blk.2.attn_q.weight") == "vit.blk.2.attn_q.weight"
    # fixed top-level names
    assert remap("v.patch_embd.weight") == "vit.patch_embd.weight"
    assert remap("v.position_embd.weight") == "vit.pos_embd"
    assert remap("v.post_ln.bias") == "vit.post_ln.bias"
    # unrelated tensors are dropped
    assert remap("some.audio.tensor") is None


def test_pi0_remap():
    m = _load("merge_pi0_mmproj_to_gguf")
    assert m.remap("mm.input_projection.weight") == "mm.proj.weight"
    _check(m.remap)


# NB: SmolVLA has no merge script - convert_smolvla_to_gguf.py bakes the vision
# tower straight from safetensors, so there is no clip->vit remap to test here.


if __name__ == "__main__":
    test_pi0_remap()
    print("test_converters: remap OK")
