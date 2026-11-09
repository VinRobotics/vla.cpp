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

"""Cosmos3-Edge through a running vla-server.

Start the server with the video noise of the case in a file, so that its answer is deterministic:

    VLA_COSMOS3_VIDEO_NOISE=vn.f32 ./build/vla-server --bind tcp://127.0.0.1:5599 cosmos3-edge-bf16-tok.gguf

then (needs pyzmq, protobuf, torch, transformers, numpy and protoc):

    VLA_LIBRARY=build/libvla.so python tests/py/cosmos3_edge_server_check.py --addr tcp://127.0.0.1:5599 \\
        --gguf cosmos3-edge-bf16-tok.gguf --inputs CASE.npz.inputs.npz --seed 1 \\
        --task "turn on the stove" --tokenizer <checkpoint>/tokenizer

CASE.npz.inputs.npz and --seed are those of tests/py/cosmos3_edge_parity.py. Checks, in order: (1) the answer of
the server to a raw request is bit-identical to a direct libvla call on the same inputs; (2) VlaCppClient's tokens
for the task are the ids of the case; (3) its rot6d decode matches lerobot's on tests/data; (4) VlaCppClient.get_action
runs 16 steps through the server and returns finite LIBERO commands.
"""
import argparse, os, sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "bindings" / "python"))
sys.path.insert(0, str(root / "eval" / "client"))

ap = argparse.ArgumentParser()
ap.add_argument("--addr", required=True)
ap.add_argument("--gguf", required=True)
ap.add_argument("--inputs", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--task", required=True)
ap.add_argument("--tokenizer", required=True)
args = ap.parse_args()
os.environ.setdefault("VLA_COSMOS3_VIDEO_NOISE", "")  # the direct call below needs the same file as the server's
if not os.environ["VLA_COSMOS3_VIDEO_NOISE"]:
    raise SystemExit("set VLA_COSMOS3_VIDEO_NOISE to the file the server was started with")

import torch
import vla_cpp
import vla_cpp_client as C

inp = np.load(args.inputs)
n = int(inp["attention_mask"].sum())
tokens = inp["input_ids"][0, :n].tolist()
noise = inp["action_noise"].astype(np.float32).ravel()
g = torch.Generator().manual_seed(args.seed)
views = [torch.rand(3, 256, 256, generator=g) for _ in range(2)]            # CHW in [0, 1], as the env gives them
views_hwc = [v.permute(1, 2, 0).contiguous().numpy() for v in views]
fails = 0

# (1) raw request vs a direct call
pb = C._load_pb()
import zmq
sock = zmq.Context.instance().socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 600_000)
sock.connect(args.addr)
req = pb.PredictRequest()
req.request_id = 1
for v in views_hwc:
    im = req.images.add()
    im.encoding = pb.Image.F32_RGB_01
    im.height, im.width = v.shape[:2]
    im.data = v.tobytes()
req.lang_tokens.extend(tokens)
req.noise.extend(noise.tolist())
sock.send(req.SerializeToString())
resp = pb.PredictResponse()
resp.ParseFromString(sock.recv())
assert not resp.error, resp.error
served = np.array(resp.action_chunk, dtype=np.float32).reshape(resp.chunk_size, resp.action_dim)
print(f"server: chunk {served.shape}, {resp.latency_ms_total:.0f} ms")

model = vla_cpp.load(args.gguf)
direct = model.predict(views_hwc, tokens, noise=noise, pixel_format=vla_cpp.PIXEL_F32_RGB_01)
same = np.array_equal(served, direct)
print(f"(1) server vs direct libvla call: bit-identical {same}, max|d| {np.abs(served - direct).max():.3e}")
fails += not same

# (2) the client's tokens, (3) its rot6d decode
client = C.VlaCppClient(args.addr, arch="cosmos3_edge", tokenizer_name=args.tokenizer, n_action_steps=16,
                        recv_timeout_ms=600_000, cosmos3_gguf=args.gguf)
mine = client._cosmos3_edge_prompt_ids(args.task)
ok_tok = mine == tokens
print(f"(2) client tokens for {args.task!r}: {len(mine)} ids, identical to the case's {ok_tok}")
fails += not ok_tok

rows = [l.rstrip("\n").split("\t") for l in open(root / "tests" / "data" / "cosmos3_edge_rot6d.tsv")]
x = np.array([[float(t) for t in r[0].split()] for r in rows], dtype=np.float32)
want = np.array([[float(t) for t in r[1].split()] for r in rows], dtype=np.float32)
got = np.concatenate([C._cosmos3_edge_libero(x[i:i + 1]) for i in range(len(x))])
d = float(np.abs(got - want).max())
print(f"(3) client rot6d decode vs lerobot on {len(x)} rows: max|d| {d:.2e}")
fails += not d < 3e-6

# (4) get_action through the server, a whole chunk
obs = {"observation.images.image": views[0], "observation.images.image2": views[1], "task": args.task}
acts = [client.get_action(obs) for _ in range(16)]
a = np.stack(acts)
ok_run = a.shape == (16, 7) and bool(np.isfinite(a).all()) and bool((np.abs(a[:, 6]) <= 1.0).all())
print(f"(4) get_action x16: {a.shape}, finite {np.isfinite(a).all()}, gripper in [-1, 1] {bool((np.abs(a[:, 6]) <= 1).all())}")
fails += not ok_run
print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
