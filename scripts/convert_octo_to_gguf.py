#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import gguf

ARCH = "octo"
MODEL_ID = "hf://rail-berkeley/octo-small-1.5"
# window_size for the rail-berkeley/octo-small-1.5 bridge pretrain checkpoint (the
# MODEL_ID default above). Used only as a last-resort fallback when converting that
# default checkpoint and its window_size can't be read back out of its own config
# (e.g. hf:// config.json fetch races) -- never used for a checkpoint passed via --ckpt.
DEFAULT_BRIDGE_WINDOW_SIZE = 2

OCTO_META: dict[str, Any] = {
    "architecture": "octo-small-1.5",
    "embedding_length": 384,
    "block_count": 12,
    "attention.head_count": 6,
    "feed_forward_length": 1536,
    "attention.layer_norm_eps": 1e-6,
    # action.horizon / action.dim / action.head_type are set from the checkpoint's own
    # config (model.config["model"]["heads"]["action"]) in main() -- they differ between
    # the diffusion libero checkpoints (horizon=4) and L1 pytorch checkpoints (horizon=20).
    "readout.count": 1,
    "tokens.primary": 256,
    "tokens.wrist": 64,
    "tokens.language": 16,
    "image.primary_size": 256,
    "image.wrist_size": 128,
    "diffusion.steps": 20,
    "diffusion.beta_schedule": "cosine",
    "diffusion.s": 0.008,
    "diffusion.max_action": 5.0,
    "diffusion.time_dim": 32,
    "diffusion.hidden": 256,
    "diffusion.num_blocks": 3,
}

# octo.action.head_type values, keyed by the PyTorch action-head class name recorded in
# a checkpoint's own config.json (model.config["model"]["heads"]["action"]["name"]).
HEAD_TYPE_BY_CLASS: dict[str, str] = {
    "L1ActionHeadPt": "l1",
    "MSEActionHeadPt": "mse",
    "DiffusionActionHeadPt": "diffusion",
    "UNetDDPMActionHeadPt": "diffusion",
}

IGNORED_PATTERNS = (
    # tied duplicate of hf_model.shared.weight (same underlying tensor, both names
    # appear in state_dict()); only shared.weight is mapped to octo.t5.tok_embd.weight.
    re.compile(r"module\.octo_transformer\.task_tokenizers\.language\.hf_model\.encoder\.embed_tokens\.weight"),
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"cannot JSON encode {type(obj).__name__}")


def _add_meta(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    full = f"octo.{key}"
    if isinstance(value, str):
        writer.add_string(full, value)
    elif isinstance(value, bool):
        writer.add_bool(full, value)
    elif isinstance(value, int):
        writer.add_uint32(full, value)
    elif isinstance(value, float):
        writer.add_float32(full, value)
    else:
        raise TypeError(f"unsupported metadata {full}={value!r}")


def _f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()


def _embed_tokenizer(writer: gguf.GGUFWriter, tokenizer_name: str = "t5-base") -> None:
    """Embed the raw T5 SentencePiece unigram model (spiece.model) as a UINT8 GGUF
    array (not a GGUF string: the serialized proto contains embedded NUL bytes,
    which would truncate a null-terminated-string read). google-t5/t5-base's
    tokenizer_name is "t5-base" (see octo-pytorch's octo_pretrain_config.py).
    """
    from huggingface_hub import hf_hub_download

    spm_path = hf_hub_download(tokenizer_name, "spiece.model")
    spm_bytes = Path(spm_path).read_bytes()
    writer.add_array("octo.tokenizer.spm_model", spm_bytes)
    # T5 unigram special tokens (fixed across all T5 SentencePiece vocabs): pad=0, eos=</s>=1.
    writer.add_uint32("octo.tokenizer.eos_id", 1)
    writer.add_uint32("octo.tokenizer.pad_id", 0)


def _strip_prefix(key: str) -> str:
    return key.removeprefix("module.")


def map_key(pt_key: str) -> str | None:
    k = _strip_prefix(pt_key)

    m = re.fullmatch(r"octo_transformer\.observation_tokenizers\.(primary|wrist)\.encoder_def\.layers\.(\d+)\.0\.(weight|bias)", k)
    if m:
        view, idx, leaf = m.groups()
        return f"octo.obs.{view}.stem.{idx}.conv.{leaf}"

    m = re.fullmatch(r"octo_transformer\.observation_tokenizers\.(primary|wrist)\.encoder_def\.layers\.(\d+)\.1\.(weight|bias)", k)
    if m:
        view, idx, leaf = m.groups()
        return f"octo.obs.{view}.stem.{idx}.gn.{leaf}"

    m = re.fullmatch(r"octo_transformer\.observation_tokenizers\.(primary|wrist)\.encoder_def\.embedding\.(weight|bias)", k)
    if m:
        view, leaf = m.groups()
        return f"octo.obs.{view}.patch_embd.{leaf}"

    m = re.fullmatch(r"octo_transformer\.obs_projections\.obs_(primary|wrist|proprio)_projection\.(weight|bias)", k)
    if m:
        view, leaf = m.groups()
        return f"octo.obs.{view}.proj.{leaf}"

    m = re.fullmatch(r"octo_transformer\.obs_(primary|wrist|proprio)_pos_embedding", k)
    if m:
        return f"octo.obs.{m.group(1)}.pos_embd"

    # LowdimObsTokenizerPt (proprio): fixed, non-trainable bin edges for the BinTokenizer
    # quantization -- not a learned weight, but still needed by the engine to reproduce
    # the same binning at inference time.
    if k == "octo_transformer.observation_tokenizers.proprio.thresholds":
        return "octo.obs.proprio.bin_thresholds"

    m = re.fullmatch(r"octo_transformer\.task_projections\.task_language_projection\.(weight|bias)", k)
    if m:
        return f"octo.task.language.proj.{m.group(1)}"
    if k == "octo_transformer.task_language_pos_embedding":
        return "octo.task.language.pos_embd"
    if k == "octo_transformer.readout_action_pos_embedding":
        return "octo.readout.action.pos_embd"

    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.layer_norm1\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.attn_norm.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.self_attention\.in_proj_(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.attn_qkv.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.self_attention\.out_proj\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.attn_o.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.layer_norm2\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.ffn_norm.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.mlp_block\.dense1\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.ffn_up.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\.(\d+)\.mlp_block\.dense2\.(weight|bias)", k)
    if m:
        return f"octo.blk.{m.group(1)}.ffn_down.{m.group(2)}"
    m = re.fullmatch(r"octo_transformer\.block_transformer\.transformer\.layer_norm\.(weight|bias)", k)
    if m:
        return f"octo.output_norm.{m.group(1)}"

    p = "heads.action.map_head."
    if k == p + "probe":
        return "octo.head.l1.map.probe"
    m = re.fullmatch(re.escape(p) + r"attention\.(in_proj_weight|in_proj_bias)", k)
    if m:
        leaf = "weight" if m.group(1) == "in_proj_weight" else "bias"
        return f"octo.head.l1.map.attn_qkv.{leaf}"
    m = re.fullmatch(re.escape(p) + r"attention\.out_proj\.(weight|bias)", k)
    if m:
        return f"octo.head.l1.map.attn_o.{m.group(1)}"
    m = re.fullmatch(re.escape(p) + r"layer_norm\.(weight|bias)", k)
    if m:
        return f"octo.head.l1.map.norm.{m.group(1)}"
    m = re.fullmatch(re.escape(p) + r"mlp_block\.dense1\.(weight|bias)", k)
    if m:
        return f"octo.head.l1.map.ffn_up.{m.group(1)}"
    m = re.fullmatch(re.escape(p) + r"mlp_block\.dense2\.(weight|bias)", k)
    if m:
        return f"octo.head.l1.map.ffn_down.{m.group(1)}"
    m = re.fullmatch(r"heads\.action\.mean_proj\.(weight|bias)", k)
    if m:
        return f"octo.head.l1.mean_proj.{m.group(1)}"

    p = "heads.action.diffusion_model."
    if k == p + "time_preprocess.w":
        return "octo.head.diffusion.time_fourier.weight"
    m = re.fullmatch(re.escape(p) + r"cond_encoder\.layers\.(0|2)\.(weight|bias)", k)
    if m:
        idx = "0" if m.group(1) == "0" else "1"
        return f"octo.head.diffusion.cond.{idx}.{m.group(2)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.linear1\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.in.{m.group(1)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.blocks\.(\d+)\.layer_norm\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.blk.{m.group(1)}.ln.{m.group(2)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.blocks\.(\d+)\.linear1\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.blk.{m.group(1)}.fc1.{m.group(2)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.blocks\.(\d+)\.linear2\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.blk.{m.group(1)}.fc2.{m.group(2)}"
    m = re.fullmatch(re.escape(p) + r"reverse_network\.linear2\.(weight|bias)", k)
    if m:
        return f"octo.head.diffusion.reverse.out.{m.group(1)}"

    # T5-base encoder (google-t5/t5-base, frozen; module.*.hf_model.* was skipped at M0).
    t5p = "octo_transformer.task_tokenizers.language.hf_model."
    if k == t5p + "shared.weight":
        return "octo.t5.tok_embd.weight"
    m = re.fullmatch(re.escape(t5p) + r"encoder\.block\.(\d+)\.layer\.0\.layer_norm\.weight", k)
    if m:
        return f"octo.t5.blk.{m.group(1)}.attn_norm.weight"
    m = re.fullmatch(re.escape(t5p) + r"encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.(q|k|v|o)\.weight", k)
    if m:
        return f"octo.t5.blk.{m.group(1)}.attn_{m.group(2)}.weight"
    m = re.fullmatch(re.escape(t5p) + r"encoder\.block\.0\.layer\.0\.SelfAttention\.relative_attention_bias\.weight", k)
    if m:
        return "octo.t5.blk.0.attn_rel_b.weight"
    m = re.fullmatch(re.escape(t5p) + r"encoder\.block\.(\d+)\.layer\.1\.layer_norm\.weight", k)
    if m:
        return f"octo.t5.blk.{m.group(1)}.ffn_norm.weight"
    m = re.fullmatch(re.escape(t5p) + r"encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.wi\.weight", k)
    if m:
        return f"octo.t5.blk.{m.group(1)}.ffn_up.weight"
    m = re.fullmatch(re.escape(t5p) + r"encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.wo\.weight", k)
    if m:
        return f"octo.t5.blk.{m.group(1)}.ffn_down.weight"
    if k == t5p + "encoder.final_layer_norm.weight":
        return "octo.t5.output_norm.weight"

    return None


def _finetune_config_window_size(finetune_cfg: dict) -> int | None:
    if "window_size" in finetune_cfg:
        return int(finetune_cfg["window_size"])
    return (
        finetune_cfg.get("dataset_kwargs", {})
        .get("traj_transform_kwargs", {})
        .get("window_size")
    )


def _resolve_window_size(model: Any, ckpt_arg: str | None, ckpt_path: str,
                          override: int | None) -> int:
    """window_size actually trained into `model`'s checkpoint -- NOT a fixed constant,
    since it differs between the rail-berkeley bridge pretrain (2) and LIBERO
    finetunes such as cyrusneary/octo-finetuned-libero (1).

    Priority: --window-size override > finetune_config.json next to the checkpoint
    > model.config["finetune_metadata"]["effective_window_size"] > model.config["window_size"]
    (all three read from the checkpoint's own config.json / finetune_config.json).
    finetune_config.json wins when present: it is the fully-resolved per-run training
    recipe (real dataset_dir, real dataset_kwargs_list, real save paths), whereas a
    checkpoint's saved config.json can retain the base architecture's window_size (the
    pos-embedding weight table's native shape, inherited unchanged from the
    octo-small-1.5 pretrain) even when the finetune's data pipeline only ever fed it
    fewer timesteps -- confirmed by hand for cyrusneary/octo-finetuned-libero/
    2025-06-20_..._175739: config.json says window_size=2, but finetune_config.json
    (matching every other fact about that run -- 4 LIBERO datasets, primary-only
    image_obs_keys, 60000 steps) says window_size=1 both at top level and under
    dataset_kwargs.traj_transform_kwargs.

    Native PyTorch checkpoints (OctoModelPt.load_pretrained, e.g. the aloha
    jitter2525 open-loop adapt run) have no finetune_config.json file at all, but
    carry the same kind of discrepancy inside their own config.json: top-level
    window_size=2 (inherited from the octo-small-1.5 pretrain this run was adapted
    from) vs. config["finetune_metadata"]["effective_window_size"]=1 (the actual
    window size this specific adapt run trained/evaluated with, logged by the
    training script's own flags snapshot). effective_window_size, when present,
    is therefore preferred over the bare top-level window_size for exactly the same
    reason finetune_config.json is preferred over it.

    If neither resolves AND a custom --ckpt was given, fail loudly rather than
    silently guessing -- only the unmodified default MODEL_ID (rail-berkeley bridge,
    which has no finetune_config.json or finetune_metadata) falls back to
    model.config, then to the known bridge constant.
    """
    if override is not None:
        return override

    if ckpt_arg is not None:
        finetune_cfg_path = Path(ckpt_path) / "finetune_config.json"
        if finetune_cfg_path.exists():
            finetune_cfg = json.loads(finetune_cfg_path.read_text())
            ws = _finetune_config_window_size(finetune_cfg)
            if ws is not None:
                return int(ws)

    cfg = getattr(model, "config", None)
    if isinstance(cfg, dict):
        effective_ws = (cfg.get("finetune_metadata") or {}).get("effective_window_size")
        if effective_ws is not None:
            return int(effective_ws)
        if "window_size" in cfg:
            return int(cfg["window_size"])

    if ckpt_arg is None:
        return DEFAULT_BRIDGE_WINDOW_SIZE

    raise SystemExit(
        f"cannot determine window_size for checkpoint {ckpt_path!r} from "
        "finetune_config.json or model.config; pass --window-size explicitly"
    )


def _head_type_from_class(head_class_name: str) -> str:
    try:
        return HEAD_TYPE_BY_CLASS[head_class_name]
    except KeyError:
        raise SystemExit(
            f"unrecognized action head class {head_class_name!r}; add it to "
            "HEAD_TYPE_BY_CLASS with its octo.action.head_type value"
        )


def _detect_ckpt_format(ckpt_arg: str | None, step: int | None) -> str:
    """Auto-detect checkpoint format for --ckpt-format=auto.

    PyTorch checkpoints saved via OctoModelPt.save_pretrained() lay out
    <ckpt>/config.json, <ckpt>/dataset_statistics.json, <ckpt>/<step>/weights.pth.
    JAX/Orbax checkpoints (the only kind load_pretrained_from_jax reads) never
    have a weights.pth. hf:// ids and the default MODEL_ID (rail-berkeley bridge
    pretrain) are always jax.
    """
    if ckpt_arg is None or ckpt_arg.startswith("hf://"):
        return "jax"
    ckpt_path = Path(ckpt_arg)
    if not ckpt_path.is_dir():
        return "jax"
    if step is not None:
        return "pytorch" if (ckpt_path / str(step) / "weights.pth").exists() else "jax"
    for sub in ckpt_path.iterdir():
        if sub.is_dir() and sub.name.isdigit() and (sub / "weights.pth").exists():
            return "pytorch"
    return "jax"


def _validate_required(mapped: dict[str, str], head_type: str, has_proprio: bool) -> list[str]:
    required: list[str] = []
    for view in ("primary", "wrist"):
        for i in range(4):
            for leaf in ("weight", "bias"):
                required.append(f"octo.obs.{view}.stem.{i}.conv.{leaf}")
                required.append(f"octo.obs.{view}.stem.{i}.gn.{leaf}")
        for leaf in ("weight", "bias"):
            required.append(f"octo.obs.{view}.patch_embd.{leaf}")
            required.append(f"octo.obs.{view}.proj.{leaf}")
        required.append(f"octo.obs.{view}.pos_embd")
    if has_proprio:
        # LowdimObsTokenizerPt has no conv stem/patch_embd (it's a BinTokenizer, not an
        # image encoder) -- only a projection, a pos embedding, and the bin thresholds.
        for leaf in ("weight", "bias"):
            required.append(f"octo.obs.proprio.proj.{leaf}")
        required.append("octo.obs.proprio.pos_embd")
        required.append("octo.obs.proprio.bin_thresholds")
    required += ["octo.task.language.proj.weight", "octo.task.language.proj.bias", "octo.task.language.pos_embd", "octo.readout.action.pos_embd"]
    for i in range(12):
        for stem in ("attn_norm", "attn_qkv", "attn_o", "ffn_norm", "ffn_up", "ffn_down"):
            for leaf in ("weight", "bias"):
                required.append(f"octo.blk.{i}.{stem}.{leaf}")
    if head_type == "diffusion":
        required += ["octo.head.diffusion.time_fourier.weight"]
        for i in range(2):
            for leaf in ("weight", "bias"):
                required.append(f"octo.head.diffusion.cond.{i}.{leaf}")
        for leaf in ("weight", "bias"):
            required.append(f"octo.head.diffusion.reverse.in.{leaf}")
            required.append(f"octo.head.diffusion.reverse.out.{leaf}")
        for i in range(3):
            for sub in ("ln", "fc1", "fc2"):
                for leaf in ("weight", "bias"):
                    required.append(f"octo.head.diffusion.reverse.blk.{i}.{sub}.{leaf}")
    elif head_type == "l1":
        required.append("octo.head.l1.map.probe")
        for leaf in ("weight", "bias"):
            required.append(f"octo.head.l1.map.attn_qkv.{leaf}")
            required.append(f"octo.head.l1.map.attn_o.{leaf}")
            required.append(f"octo.head.l1.map.norm.{leaf}")
            required.append(f"octo.head.l1.map.ffn_up.{leaf}")
            required.append(f"octo.head.l1.map.ffn_down.{leaf}")
            required.append(f"octo.head.l1.mean_proj.{leaf}")
    else:
        raise SystemExit(f"no required-tensor list for head_type {head_type!r}")
    required += ["octo.t5.tok_embd.weight", "octo.t5.blk.0.attn_rel_b.weight", "octo.t5.output_norm.weight"]
    for i in range(12):
        for stem in ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_norm", "ffn_up", "ffn_down"):
            required.append(f"octo.t5.blk.{i}.{stem}.weight")
    have = set(mapped.values())
    return [k for k in required if k not in have]


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Octo PyTorch state_dict to F32 GGUF.")
    ap.add_argument("--out", type=Path, default=Path("octo-small-1.5-f32.gguf"))
    ap.add_argument("--octo-root", type=Path, default=Path(__file__).resolve().parents[1] / "octo-pytorch")
    ap.add_argument("--allow-unmapped", action="store_true", help="write known mapped tensors and report unmapped keys instead of failing")
    ap.add_argument("--ckpt", type=str, default=None,
        help="path or HF id (hf://...) to a checkpoint dir to convert, e.g. an Octo "
             "finetune experiment dir with config.json/dataset_statistics.json/<step>/. "
             f"Default: {MODEL_ID!r} (the rail-berkeley bridge pretrain).")
    ap.add_argument("--step", type=int, default=None,
        help="checkpoint step to load from --ckpt (default: latest available step). "
             "Ignored/invalid when --ckpt is an hf:// id.")
    ap.add_argument("--window-size", type=int, default=None,
        help="override window_size written to octo.window_size GGUF meta; only needed "
             "if it can't be read from the checkpoint's config.json/finetune_config.json.")
    ap.add_argument("--ckpt-format", choices=("auto", "jax", "pytorch"), default="auto",
        help="checkpoint format to load --ckpt as: 'jax' (Orbax, via "
             "OctoModelPt.load_pretrained_from_jax -- the original/default path) or "
             "'pytorch' (native, via OctoModelPt.load_pretrained -- config.json + "
             "dataset_statistics.json + <step>/weights.pth). 'auto' (default) detects "
             "pytorch by the presence of <step>/weights.pth next to --ckpt; the default "
             f"{MODEL_ID!r} (no --ckpt) always resolves to 'jax'.")
    args = ap.parse_args()

    if args.octo_root.exists():
        sys.path.insert(0, str(args.octo_root))

    from octo.model.octo_model_pt import OctoModelPt

    model_id = args.ckpt if args.ckpt is not None else MODEL_ID
    ckpt_format = args.ckpt_format
    if ckpt_format == "auto":
        ckpt_format = _detect_ckpt_format(args.ckpt, args.step)
        print(f"--ckpt-format auto detected: {ckpt_format}")

    if ckpt_format == "pytorch":
        if args.ckpt is None:
            raise SystemExit("--ckpt-format pytorch requires --ckpt (a local PyTorch checkpoint dir)")
        print(f"loading {model_id} via OctoModelPt.load_pretrained (step={args.step}) ...")
        loaded = OctoModelPt.load_pretrained(model_id, step=args.step)
    else:
        print(f"loading {model_id} via OctoModelPt.load_pretrained_from_jax (step={args.step}) ...")
        loaded = OctoModelPt.load_pretrained_from_jax(model_id, step=args.step, skip_keys_regex=".*hf_model")
    m = loaded["octo_model"]
    sd = m.state_dict()

    window_size = _resolve_window_size(m, args.ckpt, model_id, args.window_size)
    print(f"window_size = {window_size} (from "
          f"{'--window-size override' if args.window_size is not None else 'checkpoint config'})")
    OCTO_META["window_size"] = window_size

    head_cfg = m.config["model"]["heads"]["action"]
    head_type = _head_type_from_class(head_cfg["name"])
    OCTO_META["action.head_type"] = head_type
    OCTO_META["action.horizon"] = int(head_cfg["kwargs"]["action_horizon"])
    OCTO_META["action.dim"] = int(head_cfg["kwargs"]["action_dim"])
    print(f"action head: {head_cfg['name']} -> head_type={head_type} "
          f"horizon={OCTO_META['action.horizon']} dim={OCTO_META['action.dim']}")

    has_proprio = "proprio" in m.config["model"]["observation_tokenizers"]
    print(f"has_proprio = {has_proprio} (from checkpoint config observation_tokenizers)")

    print("state_dict keys and shapes:")
    for key in sorted(sd):
        print(f"{key}\t{tuple(sd[key].shape)}\t{sd[key].dtype}")

    mapped: dict[str, str] = {}
    ignored: list[str] = []
    unmapped: list[str] = []
    for key, tensor in sd.items():
        if not tensor.is_floating_point():
            continue
        if any(pattern.match(key) for pattern in IGNORED_PATTERNS):
            ignored.append(key)
            continue
        dst = map_key(key)
        if dst is None:
            unmapped.append(key)
        elif dst in mapped.values():
            raise SystemExit(f"duplicate GGUF destination {dst} from {key}")
        else:
            mapped[key] = dst

    missing = _validate_required(mapped, head_type, has_proprio)
    if ignored:
        print("IGNORED STATE_DICT KEYS:")
        for key in sorted(ignored):
            print(f"  {key} {tuple(sd[key].shape)}")

    if unmapped or missing:
        print("TENSOR MAP REPORT:")
        if unmapped:
            print("unmapped state_dict keys:")
            for key in sorted(unmapped):
                print(f"  {key} {tuple(sd[key].shape)}")
        if missing:
            print("missing required GGUF tensors:")
            for key in missing:
                print(f"  {key}")
        if unmapped or missing:
            if not args.allow_unmapped:
                raise SystemExit("Octo tensor map is incomplete; re-run with --allow-unmapped only for investigation")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), arch=ARCH)
    for key, value in OCTO_META.items():
        _add_meta(writer, key, value)
    writer.add_string("octo.dataset_statistics", json.dumps(m.dataset_statistics, default=_json_default, sort_keys=True))
    _embed_tokenizer(writer)

    rows = []
    for src, dst in sorted(mapped.items(), key=lambda kv: kv[1]):
        tensor = sd[src]
        writer.add_tensor(dst, _f32(tensor), raw_dtype=gguf.GGMLQuantizationType.F32)
        rows.append({"state_dict": src, "gguf": dst, "shape": list(tensor.shape)})
        print(f"map {src} {tuple(tensor.shape)} -> {dst}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    report = args.out.with_suffix(args.out.suffix + ".tensor_map.json")
    report.write_text(json.dumps({"mapped": rows, "ignored": ignored, "unmapped": unmapped, "missing_required": missing}, indent=2), encoding="utf-8")
    print(f"done: {args.out} ({args.out.stat().st_size / (1024 * 1024):.1f} MiB)")
    print(f"tensor map: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
