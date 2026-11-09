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

"""The PyTorch reference of one Cosmos3-Edge call, for tests/py/cosmos3_edge_parity.py.

Runs lerobot's Cosmos3 policy (lerobot with policies/cosmos3, commit 02655292 or later) on two camera views and
a task through its own processors, with seeded noise, and writes CASE.npz:

    views         f32 [2, 256, 256, 3] in [0, 1]   the camera views, as libvla takes them (PIXEL_F32_RGB_01)
    canvas        f32 [1, 3, 192, 320]              the canvas in uint8 levels, as the processor composed it
    input_ids     i64 [1, n]      attention_mask i64 [1, n]
    video_noise   f32 [200, 192]                    noise of the 4 predicted latent frames, 50 patches x (p, q, c)
    action_noise  f32 [16, 10]
    z0            f32 [50, 192]                     the normalized latent of the canvas, patchified
    actions_norm  f32 [16, 10]                      the sampled chunk in the normalized action space
    actions       f32 [16, 10]                      the chunk unnormalized with the postprocessor's quantiles

    python tests/py/cosmos3_edge_reference.py --ckpt <checkpoint dir> --lerobot-src <lerobot/src> \\
        --task "put the bowl on the plate" --seed 1 --out CASE.npz [--views a.png b.png] \\
        [--device cuda] [--dtype bf16|f32] [--slim]

`--slim` drops the modules action inference never runs (the SigLIP tower and its projector, the language head,
the VAE decoder) before moving the policy to the device: the bf16 policy then fits a 12 GB GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True, help="checkpoint directory (config.json, model.safetensors, tokenizer/)")
ap.add_argument("--lerobot-src", type=Path, default=None, help="lerobot/src to import from (else the installed lerobot)")
ap.add_argument("--task", required=True)
ap.add_argument("--seed", type=int, required=True, help="seed of the views (without --views) and of both noises")
ap.add_argument("--views", nargs=2, default=None, help="two PNGs [third person, wrist]; random 256x256 views without")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="bf16", choices=("bf16", "f32", "f32-stream"),
                help="f32-stream: f32 arithmetic on the checkpoint's bf16 values, one decoder layer cast at a time (the "
                     "arithmetic libvla does with bf16 weights, and it fits a 12 GB GPU with --slim)")
ap.add_argument("--slim", action="store_true")
args = ap.parse_args()

if args.lerobot_src is not None:
    sys.path.insert(0, str(args.lerobot_src))
    # lerobot.policies' __init__ imports every policy, some of which want a newer transformers than Cosmos3
    # needs; take the package without running it.
    import types
    pkg = types.ModuleType("lerobot.policies")
    pkg.__path__ = [str(args.lerobot_src / "lerobot" / "policies")]
    sys.modules["lerobot.policies"] = pkg

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3EdgePolicy, FlowState  # noqa: E402
from lerobot.policies.cosmos3.processor_cosmos3 import make_cosmos3_pre_post_processors_from_pretrained  # noqa: E402

device = torch.device(args.device)
cfg = PreTrainedConfig.from_pretrained(str(args.ckpt))
cfg.device = "cpu"  # load on the host, then move what is left
cfg.dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
cfg.gradient_checkpointing = False
cfg.compile_model = False
policy = Cosmos3EdgePolicy.from_pretrained(str(args.ckpt), config=cfg)
policy.eval()
net = policy.model.network
if args.slim:
    del net.mot_backbone.model.visual
    del net.mot_backbone.model.projector
    del net.mot_backbone.lm_head
    del net.vae.decoder
policy.to(device)
if args.dtype == "f32-stream":
    # Everything but the decoder layers in f32 for good; each layer is cast to f32 for its forward and back to
    # bf16 after (its values are bf16, so nothing is lost), and the token embeddings come out f32.
    from lerobot.policies.cosmos3.modeling_cosmos3 import MoTDecoderLayer
    lm = net.mot_backbone.model.language_model
    for module in (net.vae, net.vae2llm, net.llm2vae, net.action2llm, net.llm2action, net.action_modality_embed,
                   lm.norm, lm.norm_gen):
        module.float()
    embed_forward = lm.embed_tokens.forward
    lm.embed_tokens.forward = lambda ids: embed_forward(ids).float()
    layer_forward = MoTDecoderLayer.forward

    def streamed(self, *a, **kw):
        self.float()
        try:
            return layer_forward(self, *a, **kw)
        finally:
            self.to(torch.bfloat16)

    MoTDecoderLayer.forward = streamed
print(f"policy on {device} in {args.dtype}{' (slim)' if args.slim else ''}")

pre, _ = make_cosmos3_pre_post_processors_from_pretrained(
    cfg, str(args.ckpt), preprocessor_overrides={"device_processor": {"device": str(device)}},
    postprocessor_overrides={"device_processor": {"device": "cpu"}})

# --- the views and the batch ---------------------------------------------------------------------------------
g = torch.Generator().manual_seed(args.seed)
if args.views:
    from PIL import Image
    views = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
             for p in args.views]
else:
    views = [torch.rand(3, 256, 256, generator=g) for _ in range(2)]
cameras = list(cfg.image_features)
obs = {cameras[0]: views[0], cameras[1]: views[1], "observation.state": torch.zeros(8), "task": args.task}
batch = pre(obs)
pv, grid, spg = batch["pixel_values_videos_gen"], batch["video_grid_thw_gen"], batch["second_per_grid_ts_gen"]
ids, mask = batch["input_ids"], batch["attention_mask"]
n_text = int(mask.sum())
print(f"prompt {n_text} tokens; canvas {tuple(pv.shape)}, latent grid {grid[0].tolist()}")

# --- the noise, in libvla's layout -----------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
frames, lat_h, lat_w = (int(x) for x in grid[0].tolist())
patch = cfg.latent_patch_size
z_dim = net.vae.config.z_dim
ppf = (lat_h // patch) * (lat_w // patch)  # patches per latent frame
video_noise = rng.standard_normal(((frames - 1) * ppf, patch * patch * z_dim), dtype=np.float32)
action_noise = rng.standard_normal((cfg.chunk_size, cfg.original_action_dim), dtype=np.float32)
# The condition frame's noise is never read (its noise level is 0): zeros, then libvla's rows.
patches = torch.zeros(1, frames * ppf, patch * patch * z_dim)
patches[0, ppf:] = torch.from_numpy(video_noise)
vn = net.unpatchify(patches, torch.Size((1, z_dim, frames, lat_h, lat_w))).to(device)
an = torch.zeros(1, cfg.chunk_size, cfg.max_action_dim)
an[0, :, : cfg.original_action_dim] = torch.from_numpy(action_noise)
an = an.to(device)

# --- the reference -------------------------------------------------------------------------------------------
with torch.no_grad():
    z0 = net.encode_video(pv, grid)  # [1, C, 1, h, w] normalized
    z0_patches = net.patchify(z0)[0, 0].float().cpu().numpy()  # [ppf, p*p*C]
    actions_full, _ = policy.model.generate(
        input_ids=ids, pixel_values_videos_gen=pv, video_grid_thw_gen=grid, second_per_grid_ts_gen=spg,
        attention_mask=mask, mm_token_type_ids=batch.get("mm_token_type_ids"), noise=FlowState(vn, an))
actions_norm = actions_full[0, :, : cfg.original_action_dim].float().cpu().numpy()

# The postprocessor's unnormalizer: QUANTILES, (x + 1) / 2 * (q99 - q01) + q01.
post_meta = __import__("json").loads((args.ckpt / "policy_postprocessor.json").read_text())
state_file = next(s["state_file"] for s in post_meta["steps"] if s["registry_name"] == "unnormalizer_processor")
with safe_open(str(args.ckpt / state_file), framework="pt") as f:
    q01 = f.get_tensor("action.q01").float().numpy()
    q99 = f.get_tensor("action.q99").float().numpy()
denom = np.where(q99 - q01 == 0, 1e-8, q99 - q01)
actions = (actions_norm + 1.0) * denom / 2.0 + q01

canvas = ((pv[0, :, 0].float().cpu() + 1.0) * 127.5).round().numpy()[None]  # [1, 3, H, W] in levels
np.savez(args.out, views=np.stack([v.permute(1, 2, 0).contiguous().numpy() for v in views]), canvas=canvas,
         input_ids=ids.cpu().numpy(), attention_mask=mask.cpu().numpy(), video_noise=video_noise,
         action_noise=action_noise, z0=z0_patches, actions_norm=actions_norm, actions=actions.astype(np.float32))
print(f"wrote {args.out}: actions[0] = {np.array2string(actions[0], precision=4)}")
