"""Self-contained PyTorch re-implementation of VLA-Adapter (OpenHelix, arXiv
2509.09372) — the *inference* (L1-regression) path only, with ZERO dependency on
the `third_party/VLA-Adapter` (prismatic) package.

VLA-Adapter is a Prismatic-VLM:
    fused vision backbone  DINOv2-L/14-reg4  +  SigLIP-so400m/14   (timm)
    3-layer GELU projector  (2176 -> 8704 -> 896 -> 896)
    Qwen2.5-0.5B language model (24L causal)                       (HF transformers)
    action_queries Embedding (64 x 896)
    24-block Bridge-Attention L1-regression head  (hand-written here)
    8 -> 896 GELU proprio projector

It is NOT flow-matching / diffusion: one LM prefill + one head sweep. The whole
thing is consolidated from the upstream's scattered modeling_prismatic.py /
action_heads.py / projectors.py into one clean module, dropping all the
training / diffusion / FiLM / discrete-token cruft.

Numerically equivalent to the upstream `OpenVLAForActionPrediction.predict_action`
— pinned by `verify_ref.py` (matches the gold f32 reference to ~1e-6).

Loadable straight from the HF checkpoint dir:
    VLAAdapter.from_checkpoint("/home/khanhnd61/data/VLA-Adapter/LIBERO-Object-Pro")
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from timm.models.vision_transformer import LayerScale

# ── LIBERO constants (prismatic/vla/constants.py, LIBERO suite) ──
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
NUM_TOKENS = 64          # action-query tokens
STOP_INDEX = 2
LLM_DIM = 896
VISION_DIM = 1024 + 1152  # DINOv2 + SigLIP


# ════════════════════════════════════════════════════════════════════════
#   vision backbone (timm towers + 2nd-to-last-block extraction)
# ════════════════════════════════════════════════════════════════════════
def _unpack(fn):
    def wrap(*a, **k):
        r = fn(*a, **k)
        return r[0] if isinstance(r, tuple) else r
    return wrap


class PrismaticVisionBackbone(nn.Module):
    """DINOv2-L/14-reg4 (featurizer) ⊕ SigLIP-so400m/14 (fused_featurizer), each
    returning its 2nd-to-last block's patch tokens; concat on the feature dim."""

    def __init__(self) -> None:
        super().__init__()
        self.featurizer = self._make("vit_large_patch14_reg4_dinov2.lvd142m")
        self.fused_featurizer = self._make("vit_so400m_patch14_siglip_224")
        # LayerScale stores `gamma`; HF checkpoint stores `scale_factor` — patch so
        # both names work and the forward uses scale_factor.
        for m in self.modules():
            if isinstance(m, LayerScale):
                m.scale_factor = nn.Parameter(m.gamma.clone())
                m.forward = (lambda self, x: x * self.scale_factor).__get__(m, LayerScale)
                del m.gamma

    @staticmethod
    def _make(model_id: str) -> nn.Module:
        m = timm.create_model(model_id, pretrained=False, num_classes=0, img_size=224)
        nb = len(m.blocks)
        m.forward = _unpack(partial(m.get_intermediate_layers, n={nb - 2}))  # 2nd-to-last
        return m

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, 6*num_images, 224, 224] = per image [dino3 | siglip3]."""
        n_img = pixel_values.shape[1] // 6
        outs = []
        for img in torch.split(pixel_values, [6] * n_img, dim=1):
            dino, sig = torch.split(img, [3, 3], dim=1)
            outs.append(torch.cat([self.featurizer(dino), self.fused_featurizer(sig)], dim=2))
        return torch.cat(outs, dim=1)   # [B, 256*n_img, 2176]


class PrismaticProjector(nn.Module):
    """Fused 3-layer GELU MLP: 2176 -> 8704 -> 896 -> 896."""

    def __init__(self, vision_dim: int = VISION_DIM, llm_dim: int = LLM_DIM) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, 4 * vision_dim)
        self.fc2 = nn.Linear(4 * vision_dim, llm_dim)
        self.fc3 = nn.Linear(llm_dim, llm_dim)
        self.act_fn1, self.act_fn2 = nn.GELU(), nn.GELU()

    def forward(self, x):
        return self.fc3(self.act_fn2(self.fc2(self.act_fn1(self.fc1(x)))))


class ProprioProjector(nn.Module):
    """8 -> 896 GELU 896 -> 896."""

    def __init__(self, llm_dim: int = LLM_DIM, proprio_dim: int = PROPRIO_DIM) -> None:
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.act_fn1 = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act_fn1(self.fc1(x)))


# ════════════════════════════════════════════════════════════════════════
#   Bridge-Attention action head (MLPResNetBlock_Pro)
# ════════════════════════════════════════════════════════════════════════
def _rope_cos_sin(seq_len: int, dim: int, device, dtype, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    emb = torch.cat([torch.einsum("i,j->ij", t, inv_freq)] * 2, dim=-1)   # repeated-half
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(q, k, cos, sin):
    """Non-standard: repeated-half cos/sin table + INTERLEAVED rotate_half."""
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape_as(x)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class MLPResNetBlock_Pro(nn.Module):
    """One Bridge-Attention block: query (action tokens x) cross-attends to
    self(x) ⊕ adapter(action-queries + proprio) ⊕ task(vision) with separate K/V
    per source, interleaved RoPE (head_dim 112), and a tanh-gated task branch."""

    def __init__(self, dim: int = LLM_DIM, num_heads: int = 8) -> None:
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.q_proj = nn.Linear(dim, dim)
        self.k_self, self.v_self = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.k_adapter, self.v_adapter = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.k_task, self.v_task = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))
        # FiLM head is unused at inference but present in the checkpoint.
        self.film_gen = nn.Sequential(nn.Linear(dim, dim * 2))

    def _heads(self, t, B, L):
        return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, h_a, h_t, p):
        ratio_g = torch.tanh(self.gating_factor)
        h_adapter = torch.cat((h_a, p), dim=1)
        B, T, _ = x.shape
        K_a, K_t = h_adapter.size(1), h_t.size(1)

        q = self._heads(self.q_proj(x), B, T)
        k_tok, v_tok = self._heads(self.k_self(x), B, T), self._heads(self.v_self(x), B, T)
        k_ad, v_ad = self._heads(self.k_adapter(h_adapter), B, K_a), self._heads(self.v_adapter(h_adapter), B, K_a)
        k_ta, v_ta = self._heads(self.k_task(h_t), B, K_t), self._heads(self.v_task(h_t), B, K_t)

        dev, dt = x.device, x.dtype
        cm, sm = _rope_cos_sin(T, self.head_dim, dev, dt)
        q, k_tok = _apply_rope(q, k_tok, cm, sm)
        ca, sa = _rope_cos_sin(K_a, self.head_dim, dev, dt); _, k_ad = _apply_rope(k_ad, k_ad, ca, sa)
        ct, st = _rope_cos_sin(K_t, self.head_dim, dev, dt); _, k_ta = _apply_rope(k_ta, k_ta, ct, st)

        scores = torch.cat([
            torch.matmul(q, k_tok.transpose(-2, -1)),
            torch.matmul(q, k_ad.transpose(-2, -1)),
            torch.matmul(q, k_ta.transpose(-2, -1)) * ratio_g,
        ], dim=-1) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        v = torch.cat([v_tok, v_ad, v_ta], dim=2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.dim)
        out = self.o_proj(out)
        return self.ffn(out + x)   # x is REPLACED (no outer residual)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks=24, input_dim=LLM_DIM * ACTION_DIM, hidden_dim=LLM_DIM, output_dim=ACTION_DIM):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock_Pro(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_a, h_t, p):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for i, blk in enumerate(self.mlp_resnet_blocks):
            x = blk(x, h_a=h_a[:, i + 1], h_t=h_t[:, i + 1], p=p)   # block i uses LM hidden i+1
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    def __init__(self, num_task_tokens=512):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.model = MLPResNet()

    def predict_action(self, actions_hidden_states, proprio, proprio_projector):
        B = actions_hidden_states.shape[0]
        p = proprio_projector(proprio.reshape(B, -1).to(actions_hidden_states.dtype)).unsqueeze(1)
        h_t = actions_hidden_states[:, :, :self.num_task_tokens, :]          # task (vision)
        h_a = actions_hidden_states[:, :, self.num_task_tokens:, :]          # adapter (action queries)
        x = torch.zeros((B, NUM_ACTIONS_CHUNK, ACTION_DIM * LLM_DIM),
                        device=actions_hidden_states.device, dtype=actions_hidden_states.dtype)
        return self.model(x, h_a=h_a, h_t=h_t, p=p)


# ════════════════════════════════════════════════════════════════════════
#   top-level model + predict_action
# ════════════════════════════════════════════════════════════════════════
class VLAAdapter(nn.Module):
    """VLM (vision + projector + Qwen2.5 LM + action_queries) — the action head &
    proprio projector are held alongside but are logically separate checkpoints."""

    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        from transformers import Qwen2Config, Qwen2Model
        self.device_, self.dtype_ = device, dtype
        self.vision_backbone = PrismaticVisionBackbone()
        self.projector = PrismaticProjector()
        self.language_model = None   # set in from_checkpoint (needs config)
        self.action_queries = nn.Embedding(NUM_TOKENS, LLM_DIM)
        self.action_head = L1RegressionActionHead()
        self.proprio_projector = ProprioProjector()
        self.norm_stats: dict[str, Any] = {}

    # ---- loading ----
    @classmethod
    def from_checkpoint(cls, ckpt_dir: str | Path, device="cuda", dtype=torch.float32) -> "VLAAdapter":
        from transformers import Qwen2Config, Qwen2Model
        ckpt = Path(ckpt_dir)
        self = cls(device, dtype)
        cfg = Qwen2Config(**json.load(open(ckpt / "config.json"))["text_config"])
        cfg._attn_implementation = "eager"
        self.language_model = Qwen2Model(cfg)

        # VLM + vision + projector + action_queries from model.safetensors
        vlm_sd, dino_sd, sig_sd = {}, {}, {}
        with safe_open(str(ckpt / "model.safetensors"), "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k.startswith("vision_backbone.featurizer."):
                    # LayerScale was patched gamma->scale_factor in __init__, so load
                    # the checkpoint's `*.scale_factor` straight in (NO remap).
                    dino_sd[k[len("vision_backbone.featurizer."):]] = t
                elif k.startswith("vision_backbone.fused_featurizer."):
                    sig_sd[k[len("vision_backbone.fused_featurizer."):]] = t
                elif k.startswith("projector."):
                    vlm_sd["projector." + k[len("projector."):]] = t
                elif k.startswith("language_model.model."):
                    vlm_sd["language_model." + k[len("language_model.model."):]] = t
                elif k == "action_queries.weight":
                    vlm_sd["action_queries.weight"] = t
        self.vision_backbone.featurizer.load_state_dict(dino_sd, strict=False)
        self.vision_backbone.fused_featurizer.load_state_dict(sig_sd, strict=False)
        self.projector.load_state_dict({k[len("projector."):]: v for k, v in vlm_sd.items() if k.startswith("projector.")})
        self.language_model.load_state_dict({k[len("language_model."):]: v for k, v in vlm_sd.items() if k.startswith("language_model.")}, strict=False)
        self.action_queries.load_state_dict({"weight": vlm_sd["action_queries.weight"]})

        # action head + proprio projector from the sidecar .pt files
        ah = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in torch.load(ckpt / "action_head--checkpoint.pt", map_location="cpu", weights_only=False).items()}
        pp = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in torch.load(ckpt / "proprio_projector--checkpoint.pt", map_location="cpu", weights_only=False).items()}
        self.action_head.load_state_dict(ah)
        self.proprio_projector.load_state_dict(pp)

        self.norm_stats = json.load(open(ckpt / "dataset_statistics.json"))
        return self.to(device=device, dtype=dtype).eval()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # ---- inference ----
    @torch.no_grad()
    def predict_action(self, input_ids, pixel_values, proprio, unnorm_key=None) -> np.ndarray:
        """input_ids [1, P] (prompt only); pixel_values [1, 6*n_img, 224, 224];
        proprio [PROPRIO_DIM]. Returns unnormalized actions [NUM_ACTIONS_CHUNK, ACTION_DIM]."""
        dev = self.device_
        input_ids = input_ids.to(dev)
        pixel_values = pixel_values.to(dev, dtype=self.dtype_)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

        # append 64 action placeholders + stop token (their embeddings get replaced/used)
        ph = torch.ones((1, NUM_TOKENS), dtype=input_ids.dtype, device=dev)
        stop = torch.full((1, 1), STOP_INDEX, dtype=input_ids.dtype, device=dev)
        full_ids = torch.cat([input_ids, ph, stop], dim=-1)

        emb = self.get_input_embeddings()(full_ids)                      # [1, P+65, D]
        aq = self.action_queries.weight.unsqueeze(0).to(emb.dtype)       # [1, 64, D]
        emb = torch.cat([emb[:, :NUM_PROMPT_TOKENS + 1], aq, emb[:, -1:]], dim=1)  # replace the 64 action positions

        patches = self.projector(self.vision_backbone(pixel_values))     # [1, 256*n_img, D]
        NUM_PATCHES = patches.shape[1]

        # multimodal = [emb[:1], patches, emb[1:]]
        mm = torch.cat([emb[:, :1], patches, emb[:, 1:]], dim=1)
        out = self.language_model(inputs_embeds=mm, output_hidden_states=True, use_cache=False, return_dict=True)

        # build multi_layer_hidden_states: per layer, [task(NUM_PATCHES) ⊕ action(NUM_TOKENS)]
        ml = []
        for item in out.hidden_states:   # 25 entries (incl. post-final-norm at [-1])
            act = item[:, NUM_PATCHES + NUM_PROMPT_TOKENS: NUM_PATCHES + NUM_PROMPT_TOKENS + NUM_TOKENS, :]
            task = item[:, :NUM_PATCHES, :]
            ml.append(torch.cat([task, act], dim=1).unsqueeze(1))
        ml = torch.cat(ml, dim=1).to(torch.bfloat16 if self.dtype_ == torch.bfloat16 else torch.float32)

        proprio_t = torch.as_tensor(proprio, dtype=self.dtype_, device=dev)
        norm_actions = self.action_head.predict_action(ml, proprio_t, self.proprio_projector)
        norm_actions = norm_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).float().cpu().numpy()
        return self._unnormalize(norm_actions, unnorm_key), norm_actions

    def _unnormalize(self, normalized, unnorm_key=None):
        key = unnorm_key or next(iter(self.norm_stats))
        st = self.norm_stats[key]["action"]
        mask = np.array(st.get("mask", [True] * ACTION_DIM))
        lo, hi = np.array(st["q01"]), np.array(st["q99"])
        return np.where(mask, 0.5 * (normalized + 1) * (hi - lo + 1e-8) + lo, normalized)
