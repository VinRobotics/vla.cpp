"""Run from repo root: .venv/bin/python research-notes/smolvla-audit-20261104/check_interfaces.py."""
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL = Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
os.environ['VLA_LIBRARY'] = str(ROOT / 'build-cuda/libvla.so')
sys.path.insert(0, str(ROOT / 'bindings/python'))
import vla_cpp

y, x, c = np.indices((512, 512, 3))
images = [((x + 2*y + 40*c + 17*v) & 255).astype(np.uint8) for v in range(2)]
tokens = [1, 100, 200, 300, 400, 2]
with vla_cpp.load(str(MODEL)) as model:
    cfg = model.config
    state = np.zeros(cfg.max_state_dim, dtype=np.float32)
    state[:cfg.real_state_dim] = np.float32(.01) * np.arange(1, cfg.real_state_dim+1, dtype=np.float32)
    noise = np.float32(.001) * ((np.arange(cfg.max_action_dim*cfg.n_suffix, dtype=np.uint64)*2654435761) % 1000).astype(np.float32) - np.float32(.5)
    actions = model.predict(images, tokens=tokens, state=state, noise=noise)
    np.save(OUT / 'binding-actions.npy', actions)
    print('binding shape:', actions.shape, 'finite:', bool(np.isfinite(actions).all()), flush=True)
    for name, _ in cfg._fields_:
        print('config', name, getattr(cfg, name), flush=True)

lines = (OUT / 'predict-default.out').read_text().splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith('action_len='))
reference = np.asarray([float(v) for v in lines[start+1:]], dtype=np.float32).reshape(actions.shape)
print('binding vs predict_check: max_abs =', float(np.max(np.abs(actions-reference))), 'bit_equal =', np.array_equal(actions.view(np.uint32), reference.view(np.uint32)), flush=True)

subprocess.run(['protoc', '--proto_path=src/serving', '--python_out='+str(OUT), 'src/serving/vla.proto'], cwd=ROOT, check=True)
import vla_pb2 as pb
req = pb.PredictRequest(request_id=1, lang_tokens=tokens, state=state.tolist(), noise=noise.tolist())
for image in images:
    req.images.add(encoding=pb.Image.RGB_U8, height=512, width=512, data=image.tobytes())
context = zmq.Context()
with (OUT / 'server.log').open('w') as log:
    server = subprocess.Popen([str(ROOT/'build-cuda/vla-server'), '--bind', 'tcp://127.0.0.1:15555', '--warmup', '0', str(MODEL)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while 'ready.' not in (OUT / 'server.log').read_text():
            if server.poll() is not None:
                raise RuntimeError('server exited; see server.log')
            if time.monotonic() > deadline:
                raise TimeoutError('server readiness timeout')
            time.sleep(.2)
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 120000)
        socket.connect('tcp://127.0.0.1:15555')
        try:
            socket.send(req.SerializeToString())
            response = pb.PredictResponse.FromString(socket.recv())
        finally:
            socket.close()
        if response.error:
            raise RuntimeError(response.error)
        received = np.asarray(response.action_chunk, dtype=np.float32).reshape(response.chunk_size, response.action_dim)
        np.save(OUT / 'server-actions.npy', received)
        print('server shape:', received.shape, 'finite:', bool(np.isfinite(received).all()), flush=True)
        print('server vs binding: max_abs =', float(np.max(np.abs(received-actions))), 'bit_equal =', np.array_equal(received.view(np.uint32), actions.view(np.uint32)), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        context.term()
