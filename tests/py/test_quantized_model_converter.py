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

"""Unit tests for scripts/convert_quantized_model_to_gguf.py and scripts/gguf_quant_writer.py.

A tiny GR00T-shaped model is recorded three ways (a FoldQuantVLA quantized
checkpoint, an earlier FoldQuantVLA fake-quant state, a plugin ONNX graph pair)
and every route must produce the same GGUF sites. The end-to-end check against real exports runs through `--check-onnx`.
"""

import json
import pathlib
import sys
import types

import numpy as np
import pytest
import torch

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# test_converters.py installs stub numpy/torch/gguf/safetensors modules (it runs on bare
# Python in CI, one file per process); collected in the same pytest process, those stubs
# shadow the real packages this test needs.
if not hasattr(sys.modules.get("safetensors"), "__path__") and "safetensors" in sys.modules:
    pytest.skip("real safetensors/gguf are stubbed by another test module in this process; "
                "run this file on its own", allow_module_level=True)

import gguf  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402
from safetensors.torch import save_file as save_pt  # noqa: E402

import convert_quantized_model_to_gguf as cf  # noqa: E402
from gguf_quant_writer import QuantizingGGUFWriter, Site  # noqa: E402

# A two-layer Qwen3-shaped LLM (K=128, q 128 / kv 64, inner 128) over a two-block DiT
# (K=128, inner 64, encoder 256, ffn 256): block 0 cross-attention, block 1 self-attention.
K, KV, INNER, D, K_ENC, FF = 128, 64, 128, 64, 256, 256
LM_ROOT, AHK = "backbone.lm", "action_head"
RNG = np.random.default_rng(0)


def _i8(n, k):
    return RNG.integers(-127, 128, (n, k), dtype=np.int8)


def _u8(n, k):
    return RNG.integers(0, 256, (n, k // 2), dtype=np.uint8)


def _f32(n, lo=0.5, hi=2.0):
    return RNG.uniform(lo, hi, n).astype(np.float32)


def _bf16(n):
    return (torch.randn(n) * 0.1).to(torch.bfloat16)


class Tiny:
    """The recorded model: LLM sites int8, DiT sites int4, plus the checkpoint gains and SQ vectors."""

    def __init__(self):
        self.llm = {}
        for i in range(2):
            self.llm[f"L{i}_qkv"] = (_i8(K + 2 * KV, K), _f32(K + 2 * KV))
            self.llm[f"L{i}_o"] = (_i8(K, K), _f32(K))
            self.llm[f"L{i}_gateup"] = (_i8(2 * INNER, K), _f32(2 * INNER))
            self.llm[f"L{i}_down"] = (_i8(K, INNER), _f32(K))
        self.sq = {f"L{i}_{s}": _f32(INNER if s == "down" else K) for i in range(2) for s in ("qkv", "gateup", "down")}
        self.gains = {f"L{i}_{n}": _bf16(K) for i in range(2) for n in ("input_layernorm", "post_attention_layernorm")}
        self.dit = {
            "block0_q": (_u8(D, K), _f32(D)), "encoder": (_u8(2 * D, K_ENC), _f32(2 * D)),
            "block0_o": (_u8(K, D), _f32(K)), "block0_ffn0": (_u8(FF, K), _f32(FF)), "block0_ffn2": (_u8(K, FF), _f32(K)),
            "block1_qkv": (_u8(3 * D, K), _f32(3 * D)),
            "block1_o": (_u8(K, D), _f32(K)), "block1_ffn0": (_u8(FF, K), _f32(FF)), "block1_ffn2": (_u8(K, FF), _f32(K)),
        }
        self.adaln = [(_u8(2 * K, K), _f32(2 * K, 1e-3, 1e-2)) for _ in range(2)]
        self.dit_sq = {"block0_q": _f32(K), "encoder": _f32(K_ENC), "block0_o": _f32(D), "block0_ffn0": _f32(K), "block0_ffn2": _f32(FF),
                       "block1_qkv": _f32(K), "block1_o": _f32(D), "block1_ffn0": _f32(K), "block1_ffn2": _f32(FF)}

    def folded_gamma(self, i, key):
        norm = "input_layernorm" if key == "qkv" else "post_attention_layernorm"
        return (self.gains[f"L{i}_{norm}"].float() / torch.from_numpy(self.sq[f"L{i}_{key}"])).to(torch.bfloat16)

    # -- the base checkpoint ---------------------------------------------------------
    def write_checkpoint(self, root: pathlib.Path):
        t = {}
        for i in range(2):
            t[f"{LM_ROOT}.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(KV, K, dtype=torch.bfloat16)
            t[f"{LM_ROOT}.layers.{i}.input_layernorm.weight"] = self.gains[f"L{i}_input_layernorm"]
            t[f"{LM_ROOT}.layers.{i}.post_attention_layernorm.weight"] = self.gains[f"L{i}_post_attention_layernorm"]
            t[f"{AHK}.model.transformer_blocks.{i}.attn1.to_k.weight"] = torch.zeros(D, K if i else K_ENC, dtype=torch.bfloat16)
        root.mkdir(parents=True, exist_ok=True)
        save_pt(t, str(root / "model.safetensors"))
        return root

    # -- FoldQuantVLA quantized checkpoint (foldquant.quantized_checkpoint) --------------
    LLM_PROJ = {"qkv": [("self_attn.q_proj", K), ("self_attn.k_proj", KV), ("self_attn.v_proj", KV)],
                "o": [("self_attn.o_proj", K)], "gateup": [("mlp.gate_proj", INNER), ("mlp.up_proj", INNER)],
                "down": [("mlp.down_proj", K)]}

    def write_quantized_checkpoint(self, root: pathlib.Path, *, gptq_kv: bool = True, extra_sites=None):
        """The layout FoldQuantVLA's write_quantized_checkpoint produces: qweight/weight_scale in place of
        each quantized .weight, foldquant.<module>.sq.<site> vectors, sites naming checkpoint keys by rows."""
        t = {"other.weight": torch.ones(3, dtype=torch.bfloat16)}
        sites = {"llm": {}, "dit": {}}

        def put(module, key, bits, groups, codes, scale):
            params, r0 = [], 0
            for group in groups:
                keys = []
                for head, rows in group:
                    t[f"{head}.qweight"] = torch.from_numpy(codes[r0:r0 + rows].copy())
                    t[f"{head}.weight_scale"] = torch.from_numpy(scale[r0:r0 + rows].copy())
                    keys.append(f"{head}.weight")
                    r0 += rows
                params.append(keys)
            assert r0 == codes.shape[0]
            sites[module][key] = {"bits": bits, "params": params}

        for i in range(2):
            L = f"{LM_ROOT}.layers.{i}."
            t[L + "input_layernorm.weight"] = self.gains[f"L{i}_input_layernorm"]
            t[L + "post_attention_layernorm.weight"] = self.gains[f"L{i}_post_attention_layernorm"]
            for site, projs in self.LLM_PROJ.items():
                codes, scale = self.llm[f"L{i}_{site}"]
                put("llm", f"llm.rtn.L{i}_{site}", 8, [[(L + p, r) for p, r in projs]], codes, scale)
        for key, v in self.sq.items():
            t[f"foldquant.llm.sq.{key}"] = torch.from_numpy(v)
        B = f"{AHK}.model.transformer_blocks."
        put("dit", "dit.block0_q", 4, [[(B + "0.attn1.to_q", D)]], *self.dit["block0_q"])
        kv = [[(B + "0.attn1.to_k", D), (B + "0.attn1.to_v", D)]]
        put("dit", "dit.encoder" if gptq_kv else "dit.rtn.block0_kv", 4, kv, *self.dit["encoder"])
        put("dit", "dit.block1_qkv", 4, [[(B + f"1.attn1.to_{x}", D) for x in "qkv"]], *self.dit["block1_qkv"])
        for i in range(2):
            put("dit", f"dit.block{i}_o", 4, [[(B + f"{i}.attn1.to_out.0", K)]], *self.dit[f"block{i}_o"])
            put("dit", f"dit.block{i}_ffn0", 4, [[(B + f"{i}.ff.net.0.proj", FF)]], *self.dit[f"block{i}_ffn0"])
            put("dit", f"dit.block{i}_ffn2", 4, [[(B + f"{i}.ff.net.2", K)]], *self.dit[f"block{i}_ffn2"])
            put("dit", f"dit.rtn.block{i}_adaln", 4, [[(B + f"{i}.norm1.linear", 2 * K)]], *self.adaln[i])
        for key, v in self.dit_sq.items():
            t[f"foldquant.dit.sq.{key}"] = torch.from_numpy(v)
        for module, key, bits, groups, codes, scale in (extra_sites or []):
            put(module, key, bits, groups, codes, scale)
        root.mkdir(parents=True, exist_ok=True)
        save_pt(t, str(root / "model.safetensors"))
        manifest = {"format": cf.FQC_FORMAT, "format_version": 1, "family": "groot_n1_7", "base": {"model_id": "tiny"},
                    "weights": {"files": ["model.safetensors"], "sharded": False},
                    "modules": {"llm": {"scheme": "w8a8_sr", "config": {"bits": 8, "act_bits": 8, "rot_bs": 64, "act_clip": 1.0},
                                        "tensors": [f"sq/{k}" for k in self.sq], "sites": sites["llm"]},
                                "dit": {"scheme": "w4a4_shg", "config": {"params": {}},
                                        "tensors": [f"sq/{k}" for k in self.dit_sq], "sites": sites["dit"]}}}
        (root / cf.FQ_MANIFEST).write_text(json.dumps(manifest))
        return root

    # -- FoldQuantVLA state -----------------------------------------------------------
    def write_state(self, root: pathlib.Path, llm_scheme="w8a8_sr", dit_scheme="w4a4_shg"):
        t = {}
        for key, (codes, scale) in self.llm.items():
            t[f"llm/g/llm.rtn.{key}/0/codes"], t[f"llm/g/llm.rtn.{key}/0/scale"] = codes, scale
        for key, v in self.sq.items():
            t[f"llm/t/sq/{key}"] = v
        for key, (codes, scale) in self.dit.items():
            group = "dit.encoder" if key == "encoder" else f"dit.{key}"
            t[f"dit/g/{group}/0/codes"], t[f"dit/g/{group}/0/scale"] = codes, scale
        for key, v in self.dit_sq.items():
            t[f"dit/t/sq/{key}"] = v
        t["dit/g/dit.rtn/0/codes"], t["dit/g/dit.rtn/0/scale"] = _u8(2 * K, K), _f32(2 * K)   # adaLN, by call order
        self.write_checkpoint(root)
        save_file(t, str(root / cf.FQ_TENSORS))
        manifest = {"format": cf.FQ_FORMAT, "format_version": 2, "family": "groot_n1_7", "base": {"model_id": "tiny"},
                    "modules": {"llm": {"scheme": llm_scheme, "config": {"bits": 8, "act_bits": 8, "rot_bs": 64, "act_clip": 1.0}},
                                "dit": {"scheme": dit_scheme, "config": {"params": {}}}}}
        (root / cf.FQ_MANIFEST).write_text(json.dumps(manifest))
        return root

    def write_state_rtn(self, root: pathlib.Path):
        """The same DiT recorded the way the INT8 emitter records it: `dit.rtn/{n}` in call order."""
        t = {}
        for key, (codes, scale) in self.llm.items():
            t[f"llm/g/llm.rtn.{key}/0/codes"], t[f"llm/g/llm.rtn.{key}/0/scale"] = codes, scale
        for key, v in self.sq.items():
            t[f"llm/t/sq/{key}"] = v
        order = [("block0_q",), ("block0_o",), ("encoder",), ("block0_ffn0",), ("block0_ffn2",),
                 ("block1_qkv",), ("block1_o",), ("block1_ffn0",), ("block1_ffn2",)]
        for n, (key,) in enumerate(order):
            codes, scale = self.dit[key]
            t[f"dit/g/dit.rtn/{n}/codes"], t[f"dit/g/dit.rtn/{n}/scale"] = codes, scale
        for key, v in self.dit_sq.items():
            t[f"dit/t/sq/{key}"] = v
        self.write_checkpoint(root)
        save_file(t, str(root / cf.FQ_TENSORS))
        manifest = {"format": cf.FQ_FORMAT, "format_version": 2, "family": "groot_n1_7", "base": {"model_id": "tiny"},
                    "modules": {"llm": {"scheme": "w8a8_sr", "config": {"bits": 8, "act_bits": 8, "rot_bs": 64, "act_clip": 1.0}},
                                "dit": {"scheme": "w8a8_sh", "config": {"params": {}}}}}
        (root / cf.FQ_MANIFEST).write_text(json.dumps(manifest))
        return root

    # -- plugin ONNX graphs ------------------------------------------------------------
    def write_graphs(self, root: pathlib.Path):
        import onnx
        from onnx import helper as oh

        root.mkdir(parents=True, exist_ok=True)

        def graph(nodes, name):
            g = oh.make_graph(nodes, name, [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
                              [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])])
            onnx.save(oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid("trt.plugins", 1)]), str(root / f"{name}.onnx"))

        llm = []
        for i in range(2):
            for key, op in (("qkv", "FusedRmsNormLinearInt8"), ("o", "PerRowInt8LinearResidual"),
                            ("gateup", "FusedRmsNormLinearInt8"), ("down", "PerRowInt8LinearResidual")):
                codes, scale = self.llm[f"L{i}_{key}"]
                attrs = dict(K=codes.shape[1], N=codes.shape[0], rot_block_size=64, weight_i8=codes.tobytes(), weight_scale=scale.tobytes())
                if key in ("qkv", "gateup"):
                    attrs["gamma"] = self.folded_gamma(i, key).view(torch.int16).numpy().tobytes()
                llm.append(oh.make_node(op, ["x"], ["y"], name=cf.LLM_NODES[key].format(i=i), domain="trt.plugins", **attrs))
        graph(llm, "llm")

        def w(key, stem):
            codes, scale = self.dit[key]
            return {f"{stem}_i4": codes.tobytes(), f"{stem}_scale": scale.tobytes()}

        dit = [oh.make_node("EncoderPreQuantInt4", ["x"], ["y"], name="encoder_prequant_int4", domain="trt.plugins",
                            K_enc=K_ENC, act_scale_pre_enc=self.dit_sq["encoder"].tobytes(), block_size=64)]
        dit.append(oh.make_node("FusedCrossAttnFullInt4", ["x"], ["y"], name="block0_crossattn_int4", domain="trt.plugins",
                                K=K, K_enc=K_ENC, inner_dim=D, act_scale_pre_in=self.dit_sq["block0_q"].tobytes(),
                                act_scale_pre_o=self.dit_sq["block0_o"].tobytes(), **w("block0_q", "weight_q"),
                                **w("encoder", "weight_kv"), **w("block0_o", "weight_o")))
        dit.append(oh.make_node("FusedSelfAttnFullInt4", ["x"], ["y"], name="block1_selfattn_int4", domain="trt.plugins",
                                K=K, inner_dim=D, act_scale_pre_in=self.dit_sq["block1_qkv"].tobytes(),
                                act_scale_pre_o=self.dit_sq["block1_o"].tobytes(), **w("block1_qkv", "weight_qkv"), **w("block1_o", "weight_o")))
        for i in range(2):
            dit.append(oh.make_node("FusedFfnBlockInt4", ["x"], ["y"], name=f"block{i}_ffn_int4", domain="trt.plugins",
                                    K=K, inner_dim=FF, block_size=64, act_scale_pre0=self.dit_sq[f"block{i}_ffn0"].tobytes(),
                                    act_scale_pre2=self.dit_sq[f"block{i}_ffn2"].tobytes(),
                                    **w(f"block{i}_ffn0", "weight_proj0"), **w(f"block{i}_ffn2", "weight_proj2")))
        graph(dit, "dit")
        return root


CONV = types.SimpleNamespace(LM_ROOT=LM_ROOT, AHK=AHK)
FAM = cf.FAMILIES["groot_n1_7"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    base = tmp_path_factory.mktemp("tiny")
    t = Tiny()
    t.state = t.write_state(base / "state")
    t.ckpt = t.write_checkpoint(base / "ckpt")
    t.graphs = t.write_graphs(base / "onnx")
    t.fqc = t.write_quantized_checkpoint(base / "fqc")
    t.fqc_rtn = t.write_quantized_checkpoint(base / "fqc_rtn", gptq_kv=False)
    return t


def test_fold_gain_matches_the_emitter_rule() -> None:
    torch.manual_seed(0)
    w = (torch.randn(128) * 0.1).to(torch.bfloat16)
    s = np.random.default_rng(0).uniform(0.5, 2.0, 128).astype(np.float32)
    got = cf._fold_gain(w, s, gemma=True)
    want = ((w.float() + 1.0) / torch.from_numpy(s)).to(torch.bfloat16)   # the 1 in fp32, one rounding
    assert got.dtype == torch.bfloat16 and torch.equal(got, want)
    twice = ((w + 1.0).float() / torch.from_numpy(s)).to(torch.bfloat16)  # the 1 in bf16: two roundings
    assert not torch.equal(got, twice)
    f32 = cf._fold_gain(w.float(), s, gemma=False)
    assert f32.dtype == torch.float32 and torch.equal(f32, w.float() / torch.from_numpy(s))


def test_state_sites_map_by_name(tiny) -> None:
    src = cf.open_source(tiny.state)
    assert isinstance(src, cf.FoldQuantState) and src.family == "groot_n1_7"
    sites, overrides, kv, extra = cf.build_sites(src, FAM, cf.Checkpoint(tiny.ckpt), CONV)
    # 2 layers x 7 LLM projections + block0 (q,k,v,o,ff0,ff2) + block1 (q,k,v,o,ff0,ff2)
    assert len(sites) == 14 + 12 and len(overrides) == 4
    q, k, v = (sites[f"vlm.blk.0.attn_{x}.weight"] for x in "qkv")
    codes, scale = tiny.llm["L0_qkv"]
    assert np.array_equal(np.concatenate([q.codes, k.codes, v.codes]), codes) and np.array_equal(v.wscale, scale[-KV:])
    assert q.bits == 8 and q.ascale is None
    # cross block: kv from the encoder record, with the shared encoder SmoothQuant vector
    kk = sites["aex.dit.0.attn_k.weight"]
    assert kk.bits == 4 and np.array_equal(kk.codes, tiny.dit["encoder"][0][:D]) and np.array_equal(kk.ascale, tiny.dit_sq["encoder"])
    assert np.array_equal(sites["aex.dit.0.attn_q.weight"].ascale, tiny.dit_sq["block0_q"])
    # self block: q/k/v split from the merged record, sharing its vector
    vv = sites["aex.dit.1.attn_v.weight"]
    assert np.array_equal(vv.codes, tiny.dit["block1_qkv"][0][2 * D:]) and np.array_equal(vv.ascale, tiny.dit_sq["block1_qkv"])
    # folded gains: w / s in fp32, one bf16 rounding; GR00T files keep them as they are (no Gemma - 1)
    g = torch.from_numpy(overrides["vlm.blk.1.ffn_norm.weight"])
    assert torch.equal(g.to(torch.bfloat16), tiny.folded_gamma(1, "gateup"))
    assert kv["scheme_llm"] == ("str", "w8a8_sr") and kv["action_weight_bits"] == ("u32", 4) and kv["llm_weight_bits"] == ("u32", 8)
    assert kv["site_bits"] == ("str", "") and kv["applied_at"] == ("str", "foldquant")


def test_graphs_and_state_agree(tiny) -> None:
    src = cf.open_source(tiny.state)
    ckpt = cf.Checkpoint(tiny.ckpt)
    sites, _, _, extra = cf.build_sites(src, FAM, ckpt, CONV)
    n = cf.check_against_graphs(src, FAM, ckpt, CONV, sites, extra["gains"], tiny.graphs)
    assert n == len(sites)
    # a single flipped code is caught
    sites["aex.dit.1.ff2.weight"].codes[0, 0] ^= 0xFF
    with pytest.raises(SystemExit, match="ff2.weight codes"):
        cf.check_against_graphs(src, FAM, ckpt, CONV, sites, extra["gains"], tiny.graphs)


def test_call_order_dit_packs_map_by_emitter_order(tiny, tmp_path) -> None:
    t = Tiny()
    t.__dict__.update({k: v for k, v in tiny.__dict__.items() if k in ("llm", "sq", "gains", "dit", "dit_sq")})
    src = cf.open_source(t.write_state_rtn(tmp_path / "rtn"))
    ckpt = cf.Checkpoint(tiny.ckpt)
    sites, _, kv, extra = cf.build_sites(src, FAM, ckpt, CONV)
    ref, _, _, _ = cf.build_sites(cf.open_source(tiny.state), FAM, ckpt, CONV)
    assert set(sites) == set(ref)
    for name in sites:
        assert np.array_equal(sites[name].codes, ref[name].codes) and np.array_equal(sites[name].ascale, ref[name].ascale)
    assert kv["scheme_action"] == ("str", "w8a8_sh")
    n = cf.check_against_graphs(src, FAM, ckpt, CONV, sites, extra["gains"], tiny.graphs)
    assert n == len(sites)


def test_refusals(tiny, tmp_path) -> None:
    t = Tiny()
    with pytest.raises(SystemExit, match="dense learned rotation"):
        cf._check_schemes(cf.open_source(t.write_state(tmp_path / "sr", dit_scheme="w4a4_sr")), "dit")
    with pytest.raises(SystemExit, match="no vla.cpp kernel"):
        cf._check_schemes(cf.open_source(t.write_state(tmp_path / "plain", llm_scheme="w8a8")), "llm")
    with pytest.raises(SystemExit, match="not a FoldQuantVLA quantized model"):
        cf.open_source(tmp_path)
    src = cf.open_source(tiny.state)
    with pytest.raises(SystemExit, match="call order only"):
        src.site("dit", "block0_adaln")


@pytest.mark.parametrize("layout", ["fqc", "fqc_rtn"])
def test_quantized_checkpoint_sites_match_the_state(tiny, layout) -> None:
    root = getattr(tiny, layout)
    src = cf.open_source(root)
    assert isinstance(src, cf.FoldQuantCheckpoint) and src.family == "groot_n1_7"
    ckpt = cf.Checkpoint(root)
    sites, overrides, kv, extra = cf.build_sites(src, FAM, ckpt, CONV)
    ref, ref_over, ref_kv, _ = cf.build_sites(cf.open_source(tiny.state), FAM, cf.Checkpoint(tiny.ckpt), CONV)
    assert set(sites) == set(ref)
    for name in sites:
        a, b = sites[name], ref[name]
        assert a.bits == b.bits and np.array_equal(a.codes, b.codes) and np.array_equal(a.wscale, b.wscale), name
        assert (a.ascale is None and b.ascale is None) or np.array_equal(a.ascale, b.ascale), name
    for name in overrides:
        assert np.array_equal(overrides[name], ref_over[name]), name
    assert kv["scheme_llm"] == ref_kv["scheme_llm"] and kv["action_weight_bits"] == ("u32", 4)
    assert "adaLN dequantized from INT4" in kv["provenance"][1]
    src.check_accounting()                      # every projection is a site or the allowed float adaLN
    assert cf.check_against_graphs(src, FAM, ckpt, CONV, sites, extra["gains"], tiny.graphs) == len(sites)


def test_quantized_checkpoint_base_view(tiny) -> None:
    ckpt = cf.Checkpoint(tiny.fqc)
    keys = set(ckpt.keys())
    B = f"{AHK}.model.transformer_blocks."
    assert ckpt.is_quantized and "other.weight" in keys
    assert not any(k.endswith((".qweight", ".weight_scale")) or k.startswith("foldquant.") for k in keys)
    assert f"{LM_ROOT}.layers.0.self_attn.k_proj.weight" in keys and f"{B}0.norm1.linear.weight" in keys
    assert ckpt.shape(f"{LM_ROOT}.layers.0.self_attn.k_proj.weight") == (KV, K)
    assert ckpt.shape(f"{B}0.attn1.to_k.weight") == (D, K_ENC)            # int4: K from the packed width
    # adaLN: INT4 codes x the scale rounded to bf16 (what AdaLNModInt4 multiplies by), one bf16 rounding
    codes, scale = tiny.adaln[0]
    lo = (codes & 0xF).astype(np.int8)
    hi = (codes >> 4).astype(np.int8)
    q = np.stack([np.where(lo >= 8, lo - 16, lo), np.where(hi >= 8, hi - 16, hi)], -1).reshape(codes.shape[0], -1)
    s_bf16 = torch.from_numpy(scale).to(torch.bfloat16).float().numpy()
    want = torch.from_numpy(q.astype(np.float32) * s_bf16[:, None]).to(torch.bfloat16)
    got = ckpt.tensor(f"{B}0.norm1.linear.weight")
    assert got.dtype == torch.bfloat16 and torch.equal(got, want)
    m = ckpt.mapping((f"{AHK}.",))
    assert f"{B}1.norm1.linear.weight" in m and "other.weight" not in m and len(m) == len(list(m))
    h = ckpt.handle()
    assert h.get_slice(f"{LM_ROOT}.layers.0.mlp.gate_proj.weight").get_shape() == [INNER, K]


def test_base_view_points_the_converter_at_the_quantized_checkpoint(tiny, tmp_path) -> None:
    import safetensors

    ckpt = cf.Checkpoint(tiny.fqc)
    other = tmp_path / "other.safetensors"
    save_pt({"x": torch.zeros(2)}, str(other))
    conv = types.SimpleNamespace(load_safetensors=lambda path, keep=None: "real", safe_open=safetensors.safe_open)
    real_load, real_open = conv.load_safetensors, conv.safe_open
    with cf.base_view(conv, ckpt):
        assert isinstance(conv.load_safetensors(tiny.fqc), cf._TensorMap)
        assert conv.load_safetensors(tmp_path) == "real"
        assert isinstance(conv.safe_open(tiny.fqc / "model.safetensors", framework="pt"), cf._Handle)
        with conv.safe_open(str(other), framework="pt") as f:
            assert list(f.keys()) == ["x"]
    assert conv.load_safetensors is real_load and conv.safe_open is real_open
    plain = cf.Checkpoint(tiny.ckpt)                       # an ordinary checkpoint: nothing is patched
    with cf.base_view(conv, plain):
        assert conv.load_safetensors is real_load


def test_quantized_checkpoint_refusals(tiny, tmp_path) -> None:
    t = Tiny()
    t.__dict__.update({k: v for k, v in tiny.__dict__.items() if k in ("llm", "sq", "gains", "dit", "dit_sq", "adaln")})
    # a site vla.cpp has no mapping for
    root = t.write_quantized_checkpoint(tmp_path / "bad_site", extra_sites=[
        ("dit", "dit.block0_mystery", 4, [[(f"{AHK}.x", 64)]], _u8(64, 128), _f32(64))])
    with pytest.raises(SystemExit, match="no mapping for"):
        cf.open_source(root)
    # a quantized projection no GGUF site consumes (layer 7 has an `o` site but no layers otherwise)
    root = t.write_quantized_checkpoint(tmp_path / "orphan", extra_sites=[
        ("llm", "llm.rtn.L7_o", 8, [[(f"{LM_ROOT}.layers.7.self_attn.o_proj", K)]], _i8(K, K), _f32(K))])
    src = cf.open_source(root)
    cf.build_sites(src, FAM, cf.Checkpoint(root), CONV)
    with pytest.raises(SystemExit, match="neither a vla.cpp site"):
        src.check_accounting()
    # a newer format version, and an unknown FoldQuant format
    m = json.loads((tiny.fqc / cf.FQ_MANIFEST).read_text())
    for fmt, ver, match in ((cf.FQC_FORMAT, 2, "newer than this converter"), ("foldquant-something", 1, "not one this converter reads")):
        d = tmp_path / f"m_{ver}_{fmt}"
        d.mkdir()
        (d / "model.safetensors").symlink_to(tiny.fqc / "model.safetensors")
        (d / cf.FQ_MANIFEST).write_text(json.dumps({**m, "format": fmt, "format_version": ver}))
        with pytest.raises(SystemExit, match=match):
            cf.open_source(d)


def test_writer_round_trip(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "t.gguf"
    codes4 = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)   # K = 128, nibble-packed
    sites = {"blk.0.w.weight": Site(codes=codes4, wscale=np.full(64, 0.5, np.float32),
                                    ascale=np.full(128, 2.0, np.float32), bits=4)}
    overrides = {"blk.0.norm.weight": np.full(128, 0.25, np.float32)}
    w = QuantizingGGUFWriter(out, "t", sites=sites, overrides=overrides,
                             quant_kv={"method": ("str", "foldquant"), "llm_weight_bits": ("u32", 4)}, f32_overrides=True)
    w.add_tensor("blk.0.w.weight", np.zeros((64, 128), np.float32))
    w.add_tensor("blk.0.norm.weight", np.zeros(128, np.float32))
    w.add_tensor("other", np.ones(3, np.float32))
    assert w.unconsumed() == []
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    r = gguf.GGUFReader(str(out))
    t = {x.name: x for x in r.tensors}
    assert t["blk.0.w.weight"].tensor_type == gguf.GGMLQuantizationType.I8
    assert np.array_equal(np.asarray(t["blk.0.w.weight"].data).view(np.uint8).reshape(64, 64), codes4)
    assert np.allclose(t["blk.0.w.wscale"].data, 0.5) and np.allclose(t["blk.0.w.ascale"].data, 2.0)
    assert t["blk.0.norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32
    assert np.allclose(t["blk.0.norm.weight"].data, 0.25)
    assert "t.quant.method" in r.fields


def test_writer_rejects_shape_mismatch(tmp_path: pathlib.Path) -> None:
    sites = {"a.weight": Site(codes=np.zeros((64, 64), np.int8), wscale=np.ones(64, np.float32), ascale=None, bits=8)}
    w = QuantizingGGUFWriter(tmp_path / "t.gguf", "t", sites=sites, overrides={}, quant_kv={})
    with pytest.raises(ValueError, match="the site is"):
        w.add_tensor("a.weight", np.zeros((64, 128), np.float32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
