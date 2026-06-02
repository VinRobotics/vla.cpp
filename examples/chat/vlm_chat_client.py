#!/usr/bin/env python3
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

import argparse
import sys

import zmq

import vlm_pb2

def load_jpeg(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def build_request(rid, messages, images_jpeg, temp, top_p, top_k, max_tokens, seed, stream):
    req = vlm_pb2.ChatRequest()
    req.request_id = rid
    req.stream = stream
    for role, content in messages:
        m = req.messages.add()
        m.role = role
        m.content = content
    for jpeg in images_jpeg:
        im = req.images.add()
        im.encoding = vlm_pb2.Image.JPEG
        im.data = jpeg
    req.sampling.temperature = temp
    req.sampling.top_p = top_p
    req.sampling.top_k = top_k
    req.sampling.max_tokens = max_tokens
    req.sampling.seed = seed
    return req

def chat(sock, req, on_delta):

    sock.send(req.SerializeToString())
    parts = []
    while True:
        raw = sock.recv()
        sm = vlm_pb2.StreamMessage()
        sm.ParseFromString(raw)
        kind = sm.WhichOneof("kind")
        if kind == "delta":
            parts.append(sm.delta.delta)
            if on_delta:
                on_delta(sm.delta.delta)
        elif kind == "final":
            return sm.final, "".join(parts)
        else:
            return None, "".join(parts)

def make_socket(addr, timeout_ms):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(addr)
    return sock

def run_once(args, sock):
    messages = []
    if args.system:
        messages.append(("system", args.system))
    messages.append(("user", args.prompt))
    images = [load_jpeg(args.image)] if args.image else []

    def on_delta(piece):
        if not args.quiet:
            sys.stdout.write(piece)
            sys.stdout.flush()

    req = build_request(1, messages, images, args.temp, args.top_p, args.top_k,
                        args.max_tokens, args.seed, stream=not args.no_stream)
    final, streamed = chat(sock, req, on_delta if not args.no_stream else None)
    if args.no_stream:
        sys.stdout.write(final.text if final else "")
    sys.stdout.write("\n")
    if final is None:
        print("ERROR: no final frame", file=sys.stderr)
        return 1
    if final.finish_reason == "error":
        print(f"ERROR from server: {final.error}", file=sys.stderr)
        return 1
    print(
        f"[finish={final.finish_reason} prompt_tokens={final.prompt_tokens} "
        f"completion_tokens={final.completion_tokens} total={final.latency_ms_total:.0f}ms "
        f"prefill={final.latency_ms_prefill:.0f}ms decode={final.latency_ms_decode:.0f}ms]",
        file=sys.stderr,
    )
    return 0

def run_repl(args, sock):
    history = []
    if args.system:
        history.append(("system", args.system))
    pending_images = []
    if args.image:
        pending_images.append(load_jpeg(args.image))
        print(f"(loaded image {args.image} for the next turn)")

    print("vlm chat — commands: /image <path>, /system <text>, /clear, /quit\n")
    rid = 0
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/clear":
            history = [("system", args.system)] if args.system else []
            pending_images = []
            print("(conversation cleared)")
            continue
        if line.startswith("/image "):
            path = line[len("/image "):].strip()
            try:
                pending_images.append(load_jpeg(path))
                print(f"(loaded {path} for the next turn)")
            except OSError as e:
                print(f"(could not load image: {e})")
            continue
        if line.startswith("/system "):
            sys_text = line[len("/system "):].strip()
            history = [("system", sys_text)] + [t for t in history if t[0] != "system"]
            print("(system prompt set)")
            continue

        rid += 1
        messages = list(history) + [("user", line)]
        req = build_request(rid, messages, pending_images, args.temp, args.top_p,
                            args.top_k, args.max_tokens, args.seed, stream=True)

        def on_delta(piece):
            sys.stdout.write(piece)
            sys.stdout.flush()

        final, streamed = chat(sock, req, on_delta)
        print()
        if final is None or final.finish_reason == "error":
            err = final.error if final else "no final frame"
            print(f"(error: {err})", file=sys.stderr)
            continue

        reply = final.text or streamed
        history.append(("user", line))
        history.append(("assistant", reply))
        pending_images = []
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming chat client for vlm-server")
    ap.add_argument("--addr", default="tcp://localhost:5567")
    ap.add_argument("--once", action="store_true", help="send one prompt and exit (non-interactive)")
    ap.add_argument("-p", "--prompt", default="Describe this image in detail.")
    ap.add_argument("--image", help="image to attach to the first/only turn")
    ap.add_argument("--system", help="optional system prompt")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("-n", "--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stream", action="store_true", help="(--once) request a single non-streamed reply")
    ap.add_argument("--quiet", action="store_true", help="(--once) suppress live token printing")
    ap.add_argument("--timeout-ms", type=int, default=120000)
    args = ap.parse_args()

    sock = make_socket(args.addr, args.timeout_ms)
    try:
        return run_once(args, sock) if args.once else run_repl(args, sock)
    except zmq.error.Again:
        print("ERROR: timed out waiting for vlm-server", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
