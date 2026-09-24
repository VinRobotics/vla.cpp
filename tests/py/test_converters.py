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

"""Unit tests for scripts/gguf_blocks.py, the tensor groups every converter
shares. A wrong source name or a reordered emit breaks whichever architectures
reuse the block, so each test pins both the checkpoint-side names it reads and
the GGUF-side names it writes, in order. torch / numpy / gguf / safetensors are
stubbed: the maps are pure name arithmetic and CI runs this on bare Python."""

import pathlib
import sys
import types

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _stub_deps():
    np = types.ModuleType("numpy")
    np.ndarray = type("ndarray", (), {})
    np.float32 = "f32"
    np.zeros = np.ones = np.asarray = np.ascontiguousarray = lambda *a, **k: None
    sys.modules.setdefault("numpy", np)

    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16, torch.uint16 = "f32", "bf16", "u16"
    torch.Tensor = type("Tensor", (), {})
    nn = types.ModuleType("torch.nn")
    nn.functional = types.ModuleType("torch.nn.functional")
    torch.nn = nn
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", nn)
    sys.modules.setdefault("torch.nn.functional", nn.functional)

    st = types.ModuleType("safetensors")
    st.safe_open = lambda *a, **k: None
    sys.modules.setdefault("safetensors", st)

    gguf = types.ModuleType("gguf")
    gguf.GGMLQuantizationType = types.SimpleNamespace(F32="F32", BF16="BF16", I8="I8")
    gguf.GGUFWriter = type("GGUFWriter", (), {})
    sys.modules.setdefault("gguf", gguf)


class _Tensor:
    dtype = "bf16"
    shape = (1,)

    def contiguous(self): return self
    def cpu(self): return self
    def view(self, *a): return self
    def numpy(self): return None
    def float(self): return _F32Tensor()
    def to(self, *a): return self
    def reshape(self, *a): return self
    def squeeze(self, *a): return self
    def clone(self): return self
    def __mul__(self, other): return self


class _F32Tensor(_Tensor):
    dtype = "f32"


class _Writer:
    """Records (name, dtype) in emit order; `srcs` records what the getter read."""

    def __init__(self):
        self.names = []
        self.dtypes = []

    def add_tensor(self, name, data, raw_shape=None, raw_dtype=None):
        self.names.append(name)
        self.dtypes.append(raw_dtype)


def _run(fn, *args, **kwargs):
    w = _Writer()
    srcs = []

    def g(name):
        srcs.append(name)
        return _Tensor()

    fn(w, g, *args, **kwargs)
    return w, srcs


_stub_deps()
sys.path.insert(0, str(SCRIPTS))
import gguf_blocks as B  # noqa: E402


def test_decoder_blocks():
    w, srcs = _run(B.write_decoder_blocks, "pfx", "vlm", 1)
    assert w.names == [
        "vlm.blk.0.attn_norm.weight", "vlm.blk.0.attn_q.weight", "vlm.blk.0.attn_k.weight",
        "vlm.blk.0.attn_v.weight", "vlm.blk.0.attn_o.weight", "vlm.blk.0.ffn_norm.weight",
        "vlm.blk.0.ffn_gate.weight", "vlm.blk.0.ffn_up.weight", "vlm.blk.0.ffn_down.weight",
    ]
    assert srcs[0] == "pfx.layers.0.input_layernorm.weight"
    assert srcs[1] == "pfx.layers.0.self_attn.q_proj.weight"
    assert srcs[-1] == "pfx.layers.0.mlp.down_proj.weight"
    # two layers stay grouped per layer, not per suffix
    w2, _ = _run(B.write_decoder_blocks, "pfx", "aex", 2)
    assert w2.names[9] == "aex.blk.1.attn_norm.weight"


def test_siglip_tower():
    w, srcs = _run(B.write_siglip_tower, "vis", 1)
    assert w.names == [
        "vit.patch_embd.weight", "vit.patch_embd.bias", "vit.pos_embd",
        "vit.blk.0.ln1.weight", "vit.blk.0.ln1.bias", "vit.blk.0.ln2.weight", "vit.blk.0.ln2.bias",
        "vit.blk.0.attn_q.weight", "vit.blk.0.attn_q.bias",
        "vit.blk.0.attn_k.weight", "vit.blk.0.attn_k.bias",
        "vit.blk.0.attn_v.weight", "vit.blk.0.attn_v.bias",
        "vit.blk.0.attn_o.weight", "vit.blk.0.attn_o.bias",
        "vit.blk.0.fc1.weight", "vit.blk.0.fc1.bias", "vit.blk.0.fc2.weight", "vit.blk.0.fc2.bias",
        "vit.post_ln.weight", "vit.post_ln.bias",
    ]
    assert srcs[0] == "vis.embeddings.patch_embedding.weight"
    assert srcs[3] == "vis.encoder.layers.0.layer_norm1.weight"
    assert srcs[-1] == "vis.post_layernorm.bias"
    # dtype-preserving by default; f32_norms casts norms and biases only
    assert set(w.dtypes) == {"BF16"}
    wf, _ = _run(B.write_siglip_tower, "vis", 1, f32_norms=True)
    by_name = dict(zip(wf.names, wf.dtypes))
    assert by_name["vit.blk.0.ln1.weight"] == "F32"
    assert by_name["vit.blk.0.attn_q.bias"] == "F32"
    assert by_name["vit.blk.0.attn_q.weight"] == "BF16"
    assert by_name["vit.blk.0.fc1.weight"] == "BF16"


def test_qwen3_lm():
    w, srcs = _run(B.write_qwen3_lm, "lm", 1)
    assert w.names == [
        "token_embd.weight", "vlm.output_norm.weight",
        "vlm.blk.0.attn_norm.weight", "vlm.blk.0.attn_q.weight", "vlm.blk.0.attn_k.weight",
        "vlm.blk.0.attn_v.weight", "vlm.blk.0.attn_o.weight",
        "vlm.blk.0.attn_q_norm.weight", "vlm.blk.0.attn_k_norm.weight",
        "vlm.blk.0.ffn_norm.weight", "vlm.blk.0.ffn_gate.weight",
        "vlm.blk.0.ffn_up.weight", "vlm.blk.0.ffn_down.weight",
    ]
    assert srcs[0] == "lm.embed_tokens.weight"
    assert srcs[1] == "lm.norm.weight"
    # GR00T-N1.5 falls back to lm_head when embed_tokens was pruned
    _, srcs2 = _run(B.write_qwen3_lm, "lm", 0, embd_key="other.lm_head.weight")
    assert srcs2[0] == "other.lm_head.weight"


def test_qwen3vl_vit():
    w, srcs = _run(B.write_qwen3vl_vit, "vis", 1, 1, 1024, 1536)
    assert w.names == [
        "vit.patch_embd.weight", "vit.patch_embd.bias", "vit.pos_embd",
        "vit.blk.0.ln1.weight", "vit.blk.0.ln1.bias", "vit.blk.0.ln2.weight", "vit.blk.0.ln2.bias",
        "vit.blk.0.attn_qkv.weight", "vit.blk.0.attn_qkv.bias",
        "vit.blk.0.attn_o.weight", "vit.blk.0.attn_o.bias",
        "vit.blk.0.fc1.weight", "vit.blk.0.fc1.bias", "vit.blk.0.fc2.weight", "vit.blk.0.fc2.bias",
        "vit.deepstack.0.norm.weight", "vit.deepstack.0.norm.bias",
        "vit.deepstack.0.fc1.weight", "vit.deepstack.0.fc1.bias",
        "vit.deepstack.0.fc2.weight", "vit.deepstack.0.fc2.bias",
        "vit.merger.norm.weight", "vit.merger.norm.bias",
        "vit.merger.fc1.weight", "vit.merger.fc1.bias",
        "vit.merger.fc2.weight", "vit.merger.fc2.bias",
    ]
    assert srcs[0] == "vis.patch_embed.proj.weight"
    assert srcs[3] == "vis.blocks.0.norm1.weight"
    assert "vis.deepstack_merger_list.0.linear_fc1.weight" in srcs
    assert srcs[-1] == "vis.merger.linear_fc2.bias"


def test_dit_and_vlsa_blocks():
    w, srcs = _run(B.write_dit_blocks, "ah.tb", "aex.dit", 1)
    assert w.names == [
        "aex.dit.0.adaln.weight", "aex.dit.0.adaln.bias",
        "aex.dit.0.attn_q.weight", "aex.dit.0.attn_q.bias",
        "aex.dit.0.attn_k.weight", "aex.dit.0.attn_k.bias",
        "aex.dit.0.attn_v.weight", "aex.dit.0.attn_v.bias",
        "aex.dit.0.attn_o.weight", "aex.dit.0.attn_o.bias",
        "aex.dit.0.ff0.weight", "aex.dit.0.ff0.bias",
        "aex.dit.0.ff2.weight", "aex.dit.0.ff2.bias",
    ]
    assert srcs[0] == "ah.tb.0.norm1.linear.weight"
    assert srcs[2] == "ah.tb.0.attn1.to_q.weight"
    assert srcs[8] == "ah.tb.0.attn1.to_out.0.weight"
    assert srcs[10] == "ah.tb.0.ff.net.0.proj.weight"
    assert srcs[12] == "ah.tb.0.ff.net.2.weight"

    # the VL self-attention block is the same shape with plain norms, not adaLN
    v, vsrcs = _run(B.write_vlsa_blocks, "ah.sa", "aex.vlsa", 1)
    assert v.names[:4] == ["aex.vlsa.0.norm1.weight", "aex.vlsa.0.norm1.bias",
                           "aex.vlsa.0.norm3.weight", "aex.vlsa.0.norm3.bias"]
    assert v.names[4:] == [n.replace("aex.dit.0", "aex.vlsa.0") for n in w.names[2:]]
    assert vsrcs[0] == "ah.sa.0.norm1.weight"


def test_gr00t_action_head_pieces():
    w, srcs = _run(B.write_gr00t_projectors, "action_head")
    assert w.names == [
        "aex.state_enc.l1.W", "aex.state_enc.l1.b", "aex.state_enc.l2.W", "aex.state_enc.l2.b",
        "aex.act_enc.W1.W", "aex.act_enc.W1.b", "aex.act_enc.W2.W", "aex.act_enc.W2.b",
        "aex.act_enc.W3.W", "aex.act_enc.W3.b",
        "aex.act_dec.l1.W", "aex.act_dec.l1.b", "aex.act_dec.l2.W", "aex.act_dec.l2.b",
    ]
    assert srcs[0] == "action_head.state_encoder.layer1.W"
    assert srcs[4] == "action_head.action_encoder.W1.W"

    t, tsrcs = _run(B.write_gr00t_time_embed, "action_head", "aex.dit")
    assert t.names == ["aex.dit.time_emb.l1.weight", "aex.dit.time_emb.l1.bias",
                       "aex.dit.time_emb.l2.weight", "aex.dit.time_emb.l2.bias"]
    assert tsrcs[0] == "action_head.model.timestep_encoder.timestep_embedder.linear_1.weight"

    p, psrcs = _run(B.write_gr00t_proj_out, "action_head", "aex.dit")
    assert p.names == ["aex.dit.proj_out1.weight", "aex.dit.proj_out1.bias",
                       "aex.dit.proj_out2.weight", "aex.dit.proj_out2.bias"]
    assert psrcs[0] == "action_head.model.proj_out_1.weight"


def test_prismatic_tower_and_lm():
    w, srcs = _run(B.write_prismatic_tower, 1, 1)
    assert w.names == [
        "vis.d.patch.weight", "vis.d.patch.bias", "vis.d.cls", "vis.d.reg", "vis.d.pos",
        "vis.d.blk.0.ln1.weight", "vis.d.blk.0.ln1.bias", "vis.d.blk.0.ln2.weight", "vis.d.blk.0.ln2.bias",
        "vis.d.blk.0.ls1", "vis.d.blk.0.ls2",
        "vis.d.blk.0.qkv.weight", "vis.d.blk.0.qkv.bias", "vis.d.blk.0.proj.weight", "vis.d.blk.0.proj.bias",
        "vis.d.blk.0.fc1.weight", "vis.d.blk.0.fc1.bias", "vis.d.blk.0.fc2.weight", "vis.d.blk.0.fc2.bias",
        "vis.s.patch.weight", "vis.s.patch.bias", "vis.s.pos",
        "vis.s.blk.0.ln1.weight", "vis.s.blk.0.ln1.bias", "vis.s.blk.0.ln2.weight", "vis.s.blk.0.ln2.bias",
        "vis.s.blk.0.qkv.weight", "vis.s.blk.0.qkv.bias", "vis.s.blk.0.proj.weight", "vis.s.blk.0.proj.bias",
        "vis.s.blk.0.fc1.weight", "vis.s.blk.0.fc1.bias", "vis.s.blk.0.fc2.weight", "vis.s.blk.0.fc2.bias",
        "vis.proj.fc1.weight", "vis.proj.fc1.bias", "vis.proj.fc2.weight", "vis.proj.fc2.bias",
        "vis.proj.fc3.weight", "vis.proj.fc3.bias",
    ]
    assert srcs[0] == "vision_backbone.featurizer.patch_embed.proj.weight"
    assert "vision_backbone.fused_featurizer.blocks.0.attn.qkv.weight" in srcs
    assert srcs[-1] == "projector.fc3.bias"
    # SigLIP half has no layer-scale, unlike the DINOv2 half
    assert not any(n.startswith("vis.s.blk.0.ls") for n in w.names)

    # OpenVLA-OFT's Llama has no q/k/v bias; VLA-Adapter's Qwen2 does
    o, _ = _run(B.write_prismatic_lm, 1)
    assert o.names == [
        "lm.blk.0.attn_norm.weight", "lm.blk.0.ffn_norm.weight",
        "lm.blk.0.attn_q.weight", "lm.blk.0.attn_k.weight", "lm.blk.0.attn_v.weight",
        "lm.blk.0.attn_o.weight", "lm.blk.0.ffn_gate.weight", "lm.blk.0.ffn_up.weight",
        "lm.blk.0.ffn_down.weight", "lm.output_norm.weight",
    ]
    q, qsrcs = _run(B.write_prismatic_lm, 1, qkv_bias=True)
    assert q.names[2:8] == [
        "lm.blk.0.attn_q.weight", "lm.blk.0.attn_q.bias",
        "lm.blk.0.attn_k.weight", "lm.blk.0.attn_k.bias",
        "lm.blk.0.attn_v.weight", "lm.blk.0.attn_v.bias",
    ]
    assert qsrcs[2] == "language_model.model.layers.0.self_attn.q_proj.weight"


class _SourceTensors(dict):
    """Return a tensor stub while recording every converter source lookup."""

    def __init__(self):
        self.keys_read = []

    def __getitem__(self, key):
        self.keys_read.append(key)
        return _Tensor()

    def __contains__(self, key):
        return True


def test_turbovla_converter_remap():
    import importlib

    T = importlib.import_module("convert_turbovla_to_gguf")
    tensors = _SourceTensors()
    dims = types.SimpleNamespace(
        vit_dim=1,
        vit_layers=1,
        text_layers=1,
        num_fusion_layers=1,
        num_text_layers=1,
        num_action_decoder_layers=1,
    )
    writer = _Writer()

    T.write_vision_encoder(writer, tensors, dims)
    T.write_text_encoder(writer, tensors, dims)
    T.write_vision_projection(writer, tensors)
    T.write_vision_language_interaction(writer, tensors, dims)
    T.write_action_decoder(writer, tensors, dims)
    T.write_state_projection(writer, tensors)
    T.write_view_embeddings(writer, tensors)

    assert writer.names == [
        "vit.cls_token", "vit.patch_embed.weight", "vit.patch_embed.bias", "vit.register_tokens",
        "vit.blk.0.attn_q.weight", "vit.blk.0.attn_q.bias", "vit.blk.0.attn_k.weight", "vit.blk.0.attn_k.bias",
        "vit.blk.0.attn_v.weight", "vit.blk.0.attn_v.bias", "vit.blk.0.attn_o.weight", "vit.blk.0.attn_o.bias",
        "vit.blk.0.ln1.weight", "vit.blk.0.ln1.bias", "vit.blk.0.ln2.weight", "vit.blk.0.ln2.bias",
        "vit.blk.0.fc1.weight", "vit.blk.0.fc1.bias", "vit.blk.0.fc2.weight", "vit.blk.0.fc2.bias",
        "text.embed.word_embeddings", "text.embed.position_embeddings", "text.embed.token_type_embeddings",
        "text.embed.LayerNorm.weight", "text.embed.LayerNorm.bias",
        "text.encoder.layer.0.attention.self.query.weight", "text.encoder.layer.0.attention.self.query.bias",
        "text.encoder.layer.0.attention.self.key.weight", "text.encoder.layer.0.attention.self.key.bias",
        "text.encoder.layer.0.attention.self.value.weight", "text.encoder.layer.0.attention.self.value.bias",
        "text.encoder.layer.0.attention.output.dense.weight", "text.encoder.layer.0.attention.output.dense.bias",
        "text.encoder.layer.0.attention.output.LayerNorm.weight", "text.encoder.layer.0.attention.output.LayerNorm.bias",
        "text.encoder.layer.0.intermediate.dense.weight", "text.encoder.layer.0.intermediate.dense.bias",
        "text.encoder.layer.0.output.dense.weight", "text.encoder.layer.0.output.dense.bias",
        "text.encoder.layer.0.output.LayerNorm.weight", "text.encoder.layer.0.output.LayerNorm.bias",
        "text_proj.weight", "text_proj.bias",
        "vit_proj.input_norm.weight", "vit_proj.input_norm.bias", "vit_proj.mlp.0.weight", "vit_proj.mlp.0.bias",
        "vit_proj.mlp.3.weight", "vit_proj.mlp.3.bias", "vit_proj.skip.weight", "vit_proj.output_norm.weight",
        "vit_proj.output_norm.bias",
        "vl_fusion.0.v_proj.weight", "vl_fusion.0.v_proj.bias", "vl_fusion.0.l_proj.weight", "vl_fusion.0.l_proj.bias",
        "vl_fusion.0.values_v.weight", "vl_fusion.0.values_v.bias", "vl_fusion.0.values_l.weight", "vl_fusion.0.values_l.bias",
        "vl_fusion.0.out_v.weight", "vl_fusion.0.out_v.bias", "vl_fusion.0.out_l.weight", "vl_fusion.0.out_l.bias",
        "vl_fusion.0.norm_v.weight", "vl_fusion.0.norm_v.bias", "vl_fusion.0.norm_l.weight", "vl_fusion.0.norm_l.bias",
        "vl_fusion.0.gamma_v", "vl_fusion.0.gamma_l",
        "vl_text.0.attn_qkv.weight", "vl_text.0.attn_qkv.bias", "vl_text.0.attn_o.weight", "vl_text.0.attn_o.bias",
        "vl_text.0.ln1.weight", "vl_text.0.ln1.bias", "vl_text.0.fc1.weight", "vl_text.0.fc1.bias",
        "vl_text.0.fc2.weight", "vl_text.0.fc2.bias", "vl_text.0.ln2.weight", "vl_text.0.ln2.bias",
        "act.q.weight", "act.dec.0.self_qkv.weight", "act.dec.0.self_qkv.bias", "act.dec.0.self_out.weight",
        "act.dec.0.self_out.bias", "act.dec.0.cross_qkv.weight", "act.dec.0.cross_qkv.bias",
        "act.dec.0.cross_out.weight", "act.dec.0.cross_out.bias", "act.dec.0.ln1.weight", "act.dec.0.ln1.bias",
        "act.dec.0.ln2.weight", "act.dec.0.ln2.bias", "act.dec.0.ln3.weight", "act.dec.0.ln3.bias",
        "act.dec.0.fc1.weight", "act.dec.0.fc1.bias", "act.dec.0.fc2.weight", "act.dec.0.fc2.bias",
        "act.proj.0.weight", "act.proj.0.bias", "act.proj.1.weight", "act.proj.1.bias", "act.proj.2.weight",
        "act.proj.2.bias", "state.proj.0.weight", "state.proj.0.bias", "state.proj.1.weight", "state.proj.1.bias",
        "state.proj.4.weight", "state.proj.4.bias", "state.proj.output_norm.weight", "state.proj.output_norm.bias",
        "state.proj.position", "view_emb",
    ]
    assert {
        "vision_encoder.backbone.embeddings.patch_embeddings.weight",
        "text_encoder.bert.encoder.layer.0.attention.self.query.weight",
        "vision_language_interaction.fusion_layers.0.attn.v_proj.weight",
        "vision_language_interaction.text_layers.0.self_attn.in_proj_weight",
        "action_head.decoder.decoder.layers.0.multihead_attn.in_proj_weight",
        "action_head.state_projection.net.4.weight",
        "view_embedding",
    } <= set(tensors.keys_read)


def test_every_converter_imports():
    # every converter must resolve against the two shared modules
    import importlib
    for path in sorted(SCRIPTS.glob("convert_*_to_gguf.py")):
        importlib.import_module(path.stem)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_converters: gguf_blocks maps OK")
