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

"""Cosmos3-Edge-Policy (lerobot/Cosmos3-Edge-Policy-LIBERO) -> vla.cpp GGUF.

Reads the checkpoint directory: config.json, model.safetensors, the postprocessor's unnormalizer stats and
tokenizer/. Only what inference needs is written:

- the Wan2.2 VAE encoder. Its causal 3-D convolutions see one frame behind two zero frames, so only the last
  temporal tap of each kernel reaches the output: that tap is written as a 2-D kernel. The decoder and the
  temporal convolutions of the downsamplers (skipped on a first frame) are left out;
- the 28 MoT decoder layers, both streams. The understanding stream's final norm, the language head, the
  SigLIP vision tower and its projector play no part in action inference and are left out;
- the projections between the backbone and the latents: vae2llm, llm2vae, and action2llm / llm2action of the
  config's embodiment (`domain_id`), cut to `action_dim` since the other channels are zero;
- the timestep embedding, in f32 as the policy keeps it;
- the latent statistics of the VAE, the action quantiles of the unnormalizer, the canvas layout of the policy's
  cameras, the caption settings of the prompt step and the byte-level BPE tokenizer (llama.cpp's
  `tokenizer.ggml.*` keys), so `vla-cli --text` builds the prompt itself (src/models/cosmos3_edge.cpp).

GEMM weights keep the checkpoint's bf16 (the engine casts them to its matmul type at load), norms and biases
are f32.

    python scripts/convert_cosmos3_edge_to_gguf.py --ckpt <checkpoint dir> --out cosmos3-edge-libero-bf16.gguf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from gguf_common import (
    add,
    add_array,
    add_f32,
    arg_parser,
    finish,
    kv_f32,
    kv_prefix,
    kv_u32,
    open_writer,
    read_json,
    require,
    resolve_out,
)

ARCH = "cosmos3_edge"
KV = kv_prefix(ARCH)
NET = "model.network."
LM = NET + "mot_backbone.model.language_model."

# Video canvases of the 256 tier by aspect ratio name, (width, height): lerobot's VIDEO_CANVAS_SIZES.
CANVAS_256 = {"1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320), "16,9": (320, 192), "9,16": (192, 320)}


def canvas_layout(config: dict) -> dict:
    """The canvas of the policy's cameras, as lerobot's Cosmos3VideoCanvasProcessorStep lays it out: every view
    at `view_size`, two side by side, fit into the closest canvas of the 256 tier without enlarging, then the
    latent grid of the content. One or two cameras are supported (three use another layout)."""
    cameras = [k for k, f in config["input_features"].items() if f["type"] == "VISUAL"]
    if len(cameras) not in (1, 2):
        raise SystemExit(f"{len(cameras)} cameras; the canvas layout is written for one or two")
    view_h, view_w = (int(x) for x in config["view_size"])
    if min(view_h, view_w) > 256:
        raise SystemExit(f"a {view_h}x{view_w} view needs a higher resolution tier than the 256 one written here")
    frame_h, frame_w = view_h, view_w * len(cameras)
    canvas_w, canvas_h = min(CANVAS_256.values(), key=lambda s: abs(frame_h / frame_w - s[1] / s[0]))
    scale = min(canvas_w / frame_w, canvas_h / frame_h, 1.0)
    content_h, content_w = int(scale * frame_h + 0.5), int(scale * frame_w + 0.5)
    spatial = int(config["vae"]["scale_factor_spatial"])
    ratio = next(name for name, size in CANVAS_256.items() if size == (canvas_w, canvas_h))
    return {
        "n_views": len(cameras), "view_h": view_h, "view_w": view_w,
        "canvas_h": canvas_h, "canvas_w": canvas_w, "content_h": content_h, "content_w": content_w,
        "latent_h": content_h // spatial, "latent_w": content_w // spatial, "aspect_ratio": ratio,
        "cameras": cameras,
    }


def conv_tap(w: torch.Tensor) -> torch.Tensor:
    """The 2-D kernel a causal 3-D convolution applies to a first frame: the input is padded with KT-1 zero frames
    in front, so of the kernel [OC, IC, KT, KH, KW] only tap KT-1 reaches that frame's output."""
    if w.dim() != 5:
        raise SystemExit(f"expected a 3-D kernel, got shape {tuple(w.shape)}")
    return w[:, :, -1].contiguous()


def add_tokenizer(writer, tok_dir: Path) -> None:
    """The BPE of tokenizer.json as llama.cpp's GGUF vocabulary keys (gpt2 model, "tekken" pre-tokenizer)."""
    d = json.loads((tok_dir / "tokenizer.json").read_text())
    cfg = json.loads((tok_dir / "tokenizer_config.json").read_text())
    m = d["model"]
    if m["type"] != "BPE" or not m.get("ignore_merges") or m.get("byte_fallback"):
        raise SystemExit("expected a byte-level BPE tokenizer with ignore_merges, as the Cosmos3 tokenizer is")
    vocab = m["vocab"]
    tokens = [None] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    if any(t is None for t in tokens):
        raise SystemExit("the vocabulary has gaps")
    types = [1] * len(tokens)  # llama.cpp: 1 normal, 2 unknown, 3 control, 4 user defined
    for a in d["added_tokens"]:
        if a["id"] >= len(tokens):  # the C++ reads the vocabulary's spellings; a token outside it has none
            raise SystemExit(f"added token {a['content']!r} (id {a['id']}) is not in the vocabulary of {len(tokens)}")
        types[a["id"]] = 3 if a["special"] else 4
    # The specials of tokenizer_config.json: a string, an AddedToken dict or null (a byte-level BPE needs no
    # unknown token); one that is not a token is left out rather than guessed.
    ids = dict(vocab)
    ids.update((a["content"], a["id"]) for a in d["added_tokens"])

    def special_id(name: str) -> int | None:
        t = cfg.get(name)
        if isinstance(t, dict):
            t = t.get("content")
        return ids.get(t) if isinstance(t, str) else None

    unk = special_id("unk_token")
    if unk is not None:
        types[unk] = 2
    merges = [" ".join(x) if isinstance(x, list) else x for x in m["merges"]]
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "tekken")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.token_type", types)
    writer.add_array("tokenizer.ggml.merges", merges)
    for key, name in (("bos_token_id", "bos_token"), ("eos_token_id", "eos_token"), ("unknown_token_id", "unk_token"),
                      ("padding_token_id", "pad_token")):
        i = special_id(name)
        if i is not None:
            writer.add_uint32(f"tokenizer.ggml.{key}", i)
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    print(f"tokenizer: {len(tokens)} tokens, {len(merges)} merges, {sum(t in (3, 4) for t in types)} special")


def load_quantiles(ckpt: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """q01 and q99 of the action from the postprocessor's unnormalizer (QUANTILES normalization)."""
    meta = read_json(ckpt / "policy_postprocessor.json")
    state_file = next((s.get("state_file") for s in meta.get("steps", [])
                       if s.get("registry_name") == "unnormalizer_processor"), None)
    if not state_file or not (ckpt / state_file).is_file():
        raise SystemExit("policy_postprocessor.json names no unnormalizer state file")
    with safe_open(str(ckpt / state_file), framework="pt") as f:
        keys = set(f.keys())
        if "action.q01" not in keys or "action.q99" not in keys:
            raise SystemExit(f"{state_file} has no action.q01/q99 (the policy normalizes actions by quantiles)")
        q01 = f.get_tensor("action.q01").float().numpy().reshape(-1)
        q99 = f.get_tensor("action.q99").float().numpy().reshape(-1)
    if q01.size != action_dim or q99.size != action_dim:
        raise SystemExit(f"action quantiles have {q01.size} values, the action has {action_dim}")
    return q01.astype(np.float32), q99.astype(np.float32)


class Weights:
    """The checkpoint's tensors by name, read one at a time; remembers which were used."""

    def __init__(self, path: Path):
        self.f = safe_open(str(path), framework="pt")
        self.keys = set(self.f.keys())
        self.used: set[str] = set()

    def __call__(self, name: str) -> torch.Tensor:
        if name not in self.keys:
            raise SystemExit(f"missing tensor {name}")
        self.used.add(name)
        return self.f.get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.keys


def check_config(config: dict) -> None:
    """The engine is written for the LIBERO policy's setup; refuse what it does not do."""
    if config.get("type") != "cosmos3_edge":
        raise SystemExit(f"config.json is a {config.get('type')!r} policy, not cosmos3_edge")
    text = config["mot_backbone"]["text_config"]
    if text["hidden_act"] != "relu2" or text["rope_parameters"]["rope_type"] != "default":
        raise SystemExit("the engine is written for Nemotron 3's relu^2 MLP and default M-RoPE")
    if text["joint_attn_implementation"] != "two_way":
        raise SystemExit("the engine is written for two_way joint attention")
    if config["mode"] != "wam" or config.get("use_state") or config["guidance_scale"] != 1.0 or config["n_obs_steps"] != 1:
        raise SystemExit("the engine is written for mode wam, no state, no classifier-free guidance and one observation step")
    if config["sampler"] != "unipc":
        raise SystemExit(f"sampler {config['sampler']!r}: the engine runs UniPC")
    if config["latent_patch_size"] != 2 or config["vae"]["patch_size"] != 2:
        raise SystemExit("the engine is written for a latent patch of 2 and a VAE pixel patch of 2")
    if config["vae"].get("is_residual") is not True:
        raise SystemExit("the engine is written for the Wan2.2 VAE (residual down blocks)")


def main() -> int:
    ap = arg_parser(ARCH, "checkpoint directory (config.json, model.safetensors, policy_postprocessor*, tokenizer/)",
                    description=__doc__)
    args = ap.parse_args()
    ckpt = args.ckpt.resolve()
    require(ckpt / "config.json", ckpt / "model.safetensors", ckpt / "tokenizer" / "tokenizer.json")
    out = args.out.resolve() if args.out else resolve_out(args, ckpt, ARCH)

    config = read_json(ckpt / "config.json")
    check_config(config)
    text = config["mot_backbone"]["text_config"]
    vae = config["vae"]
    action_dim = int(config["output_features"]["action"]["shape"][0])
    chunk = int(config["chunk_size"])
    layout = canvas_layout(config)
    n_layers = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    n_blocks = len(vae["dim_mult"])

    W = Weights(ckpt / "model.safetensors")
    writer = open_writer(out, ARCH)

    # --- metadata -------------------------------------------------------------------------------------------
    kv_u32(writer, KV, {
        "n_layers": n_layers, "hidden": hidden, "intermediate": text["intermediate_size"],
        "n_heads": text["num_attention_heads"], "n_kv_heads": text["num_key_value_heads"], "head_dim": text["head_dim"],
        "vocab_size": text["vocab_size"],
        "mrope_section_t": text["rope_parameters"]["mrope_section"][0],
        "mrope_section_h": text["rope_parameters"]["mrope_section"][1],
        "mrope_section_w": text["rope_parameters"]["mrope_section"][2],
        "time_embed_dim": config["timestep_embedding_dim"], "temporal_modality_margin": config["temporal_modality_margin"],
        "num_steps": config["num_inference_steps"], "num_train_timesteps": config["num_train_timesteps"],
        "chunk_size": chunk, "action_dim": action_dim, "max_action_dim": config["max_action_dim"],
        "domain_id": config["domain_id"], "latent_patch_size": config["latent_patch_size"],
        "vae.z_dim": vae["z_dim"], "vae.base_dim": vae["base_dim"], "vae.patch_size": vae["patch_size"],
        "vae.scale_factor_spatial": vae["scale_factor_spatial"], "vae.scale_factor_temporal": vae["scale_factor_temporal"],
        "vae.n_blocks": n_blocks, "vae.num_res_blocks": vae["num_res_blocks"],
        "n_views": layout["n_views"], "view_h": layout["view_h"], "view_w": layout["view_w"],
        "canvas_h": layout["canvas_h"], "canvas_w": layout["canvas_w"],
        "content_h": layout["content_h"], "content_w": layout["content_w"],
        "latent_h": layout["latent_h"], "latent_w": layout["latent_w"],
        "prompt_len": 512,  # the longest prompt the engine takes
    })
    # Each down block's shortcut averages 2x2 pixels (factor_s) and, where the block also halves time, pairs of
    # frames (factor_t); the last block keeps its size.
    temporal = list(vae["temperal_downsample"]) + [False] * n_blocks
    for i, mult in enumerate(vae["dim_mult"]):
        last = i == n_blocks - 1
        kv_u32(writer, KV, {f"vae.down.{i}.dim_mult": mult,
                            f"vae.down.{i}.factor_t": 2 if (temporal[i] and not last) else 1,
                            f"vae.down.{i}.factor_s": 1 if last else 2})
    kv_f32(writer, KV, {
        "rms_eps": text["rms_norm_eps"], "rope_theta": text["rope_parameters"]["rope_theta"],
        "timestep_scale": config["timestep_scale"], "base_fps": config["base_fps"], "fps": config["fps"],
        "sampling_shift": config["sampling_shift"],
    })
    writer.add_string(KV("caption.view_description"), str(config["view_description"]))
    writer.add_string(KV("caption.aspect_ratio"), layout["aspect_ratio"])
    writer.add_float32(KV("caption.fps"), float(config["fps"]))
    writer.add_uint32(KV("caption.use_state"), 1 if config.get("use_state") else 0)
    writer.add_string(KV("cameras"), ", ".join(layout["cameras"]))
    add_tokenizer(writer, ckpt / "tokenizer")

    add_array(writer, KV("latents_mean"), np.asarray(vae["latents_mean"], dtype=np.float32))
    add_array(writer, KV("latents_std"), np.asarray(vae["latents_std"], dtype=np.float32))
    q01, q99 = load_quantiles(ckpt, action_dim)
    add_array(writer, KV("action_q01"), q01)
    add_array(writer, KV("action_q99"), q99)

    # --- the backbone ----------------------------------------------------------------------------------------
    n = 0

    def put(name: str, t: torch.Tensor, f32: bool = False) -> None:
        nonlocal n
        (add_f32 if f32 else add)(writer, name, t)
        n += 1

    put("token_embd.weight", W(LM + "embed_tokens.weight"))
    for i in range(n_layers):
        p = f"{LM}layers.{i}."
        for stream, suffix in (("", ""), ("gen.", "_gen")):
            b = f"blk.{i}.{stream}"
            put(b + "attn_norm.weight", W(f"{p}input_layernorm{suffix}.weight"), f32=True)
            put(b + "ffn_norm.weight", W(f"{p}post_attention_layernorm{suffix}.weight"), f32=True)
            for ours, theirs in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")):
                put(f"{b}{ours}.weight", W(f"{p}self_attn.{theirs}{suffix}.weight"))
            put(b + "ffn_up.weight", W(f"{p}mlp{suffix}.up_proj.weight"))
            put(b + "ffn_down.weight", W(f"{p}mlp{suffix}.down_proj.weight"))
        # The generation stream's QK norms, and the norm of the understanding keys its queries read.
        put(f"blk.{i}.gen.attn_q_norm.weight", W(f"{p}self_attn.q_norm_gen.weight"), f32=True)
        put(f"blk.{i}.gen.attn_k_norm.weight", W(f"{p}self_attn.k_norm_gen.weight"), f32=True)
        put(f"blk.{i}.attn_k_norm_gen.weight", W(f"{p}self_attn.k_norm_und_for_gen.weight"), f32=True)
    put("gen.output_norm.weight", W(LM + "norm_gen.weight"), f32=True)

    # --- projections and the timestep embedding ----------------------------------------------------------
    put("vae2llm.weight", W(NET + "vae2llm.weight"))
    put("vae2llm.bias", W(NET + "vae2llm.bias"), f32=True)
    put("llm2vae.weight", W(NET + "llm2vae.weight"))
    put("llm2vae.bias", W(NET + "llm2vae.bias"), f32=True)
    domain = int(config["domain_id"])
    max_ad = int(config["max_action_dim"])
    # DomainAwareLinear computes x @ W + b per domain with W [in, out]; nn.Linear's [out, in] is its transpose.
    a2l = W(NET + "action2llm.fc.weight")[domain].view(max_ad, hidden)[:action_dim].t().contiguous()  # [hidden, A]
    put("action2llm.weight", a2l)
    put("action2llm.bias", W(NET + "action2llm.bias.weight")[domain], f32=True)
    l2a = W(NET + "llm2action.fc.weight")[domain].view(hidden, max_ad)[:, :action_dim].t().contiguous()  # [A, hidden]
    put("llm2action.weight", l2a)
    put("llm2action.bias", W(NET + "llm2action.bias.weight")[domain][:action_dim], f32=True)
    put("action_modality_embed.weight", W(NET + "action_modality_embed.weight"), f32=True)
    for k in ("linear_1", "linear_2"):
        put(f"time_embedder.{k}.weight", W(f"{NET}time_embedder.{k}.weight"), f32=True)
        put(f"time_embedder.{k}.bias", W(f"{NET}time_embedder.{k}.bias"), f32=True)

    # --- the VAE encoder ---------------------------------------------------------------------------------
    V = NET + "vae."

    def conv3(ours: str, theirs: str) -> None:
        put(ours + ".weight", conv_tap(W(theirs + ".weight")))
        put(ours + ".bias", W(theirs + ".bias"), f32=True)

    def norm(ours: str, theirs: str) -> None:
        put(ours + ".gamma", W(theirs + ".gamma").flatten(), f32=True)

    def conv1x1(ours: str, theirs: str) -> None:
        w = W(theirs + ".weight")  # [OC, IC, (1,) 1, 1]: a 1x1 convolution is a linear map
        put(ours + ".weight", w.reshape(w.shape[0], w.shape[1]).contiguous())
        put(ours + ".bias", W(theirs + ".bias"), f32=True)

    def resnet(ours: str, theirs: str) -> None:
        norm(ours + ".norm1", theirs + ".norm1")
        conv3(ours + ".conv1", theirs + ".conv1")
        norm(ours + ".norm2", theirs + ".norm2")
        conv3(ours + ".conv2", theirs + ".conv2")
        if W.has(theirs + ".conv_shortcut.weight"):
            conv1x1(ours + ".shortcut", theirs + ".conv_shortcut")

    conv3("vae.conv_in", V + "encoder.conv_in")
    for b in range(n_blocks):
        for r in range(int(vae["num_res_blocks"])):
            resnet(f"vae.down.{b}.res.{r}", f"{V}encoder.down_blocks.{b}.resnets.{r}")
        if W.has(f"{V}encoder.down_blocks.{b}.downsampler.resample.1.weight"):
            # ZeroPad2d((0, 1, 0, 1)) then Conv2d(dim, dim, 3, stride 2)
            put(f"vae.down.{b}.down.weight", W(f"{V}encoder.down_blocks.{b}.downsampler.resample.1.weight"))
            put(f"vae.down.{b}.down.bias", W(f"{V}encoder.down_blocks.{b}.downsampler.resample.1.bias"), f32=True)
    resnet("vae.mid.res.0", V + "encoder.mid_block.resnets.0")
    norm("vae.mid.attn.norm", V + "encoder.mid_block.attentions.0.norm")
    conv1x1("vae.mid.attn.qkv", V + "encoder.mid_block.attentions.0.to_qkv")
    conv1x1("vae.mid.attn.proj", V + "encoder.mid_block.attentions.0.proj")
    resnet("vae.mid.res.1", V + "encoder.mid_block.resnets.1")
    norm("vae.norm_out", V + "encoder.norm_out")
    conv3("vae.conv_out", V + "encoder.conv_out")
    conv1x1("vae.quant_conv", V + "quant_conv")

    skipped = sorted(k for k in W.keys if k not in W.used)
    heads = sorted({".".join(k.split(".")[:5]) for k in skipped})
    print(f"{n} tensors written; {len(skipped)} of the checkpoint's left out: {', '.join(heads)}")
    print(f"canvas {layout['canvas_h']}x{layout['canvas_w']}, content {layout['content_h']}x{layout['content_w']} of "
          f"{layout['n_views']} {layout['view_h']}x{layout['view_w']} views, latent grid {layout['latent_h']}x{layout['latent_w']}; "
          f"{config['num_inference_steps']} UniPC steps, chunk {chunk}x{action_dim}, domain {domain}")
    return finish(writer, out)


if __name__ == "__main__":
    sys.exit(main())
