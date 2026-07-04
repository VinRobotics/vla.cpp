// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vlm/engine.h"
#include "serving/vlm.pb.h"

#include <zmq.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <vector>

namespace {

std::atomic<bool> g_shutdown{false};
void on_signal(int) { g_shutdown.store(true, std::memory_order_relaxed); }

// Reject absurd image dimensions before any size arithmetic on untrusted input.
constexpr unsigned kMaxImageDim = 8192;

std::string make_error_stream(uint64_t request_id, const std::string & msg) {
    vlm_chat::StreamMessage sm;
    vlm_chat::ChatResponse * resp = sm.mutable_final();
    resp->set_request_id(request_id);
    resp->set_finish_reason("error");
    resp->set_error(msg);
    return sm.SerializeAsString();
}

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s [--bind ADDR] [-c N_CTX] [--ngl N] [--no-mmproj-gpu] <mmproj.gguf> <lm.gguf>\n"
        "  <mmproj.gguf>     vision-tower mmproj GGUF (e.g. mmproj-smolvlm2-500m-instruct-f16.gguf)\n"
        "  <lm.gguf>         text LM GGUF (e.g. smolvlm2-500m-instruct-f16.gguf)\n"
        "  --bind ADDR       ZMQ bind address (default: tcp://*:5567)\n"
        "  -c, --n-ctx N     context window (default: 4096)\n"
        "  --ngl N           LM layers to offload to GPU (default: 999 = all)\n"
        "  --no-mmproj-gpu   run the vision tower on CPU\n",
        prog);
}

}

int main(int argc, char ** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string bind_addr = "tcp://*:5567";
    vlm::LoadParams lp;
    std::vector<std::string> positionals;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bind" && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if ((a == "-c" || a == "--n-ctx") && i + 1 < argc) {
            lp.n_ctx = std::atoi(argv[++i]);
        } else if (a == "--ngl" && i + 1 < argc) {
            lp.n_gpu_layers = std::atoi(argv[++i]);
        } else if (a == "--no-mmproj-gpu") {
            lp.mmproj_use_gpu = false;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            positionals.push_back(std::move(a));
        }
    }
    if (positionals.size() != 2) {
        usage(argv[0]);
        return 1;
    }
    lp.mmproj_path = positionals[0];
    lp.model_path  = positionals[1];

    std::printf("vlm-server: loading model ...\n  mmproj: %s\n  lm:     %s\n",
                lp.mmproj_path.c_str(), lp.model_path.c_str());

    vlm::Engine engine;
    if (!engine.load(lp)) {
        std::fprintf(stderr, "vlm-server: engine load failed\n");
        return 1;
    }
    std::printf("vlm-server: loaded. n_ctx=%d ngl=%d\n", lp.n_ctx, lp.n_gpu_layers);

    zmq::context_t zctx( 1);
    zmq::socket_t  sock(zctx, zmq::socket_type::router);
    sock.set(zmq::sockopt::linger, 0);
    // cap inbound messages so one oversized request cannot exhaust memory.
    sock.set(zmq::sockopt::maxmsgsize, int64_t(256) * 1024 * 1024);
    sock.bind(bind_addr);
    std::printf("vlm-server: bound to %s. ready.\n", bind_addr.c_str());

    if (bind_addr.find("127.0.0.1") == std::string::npos &&
        bind_addr.find("localhost") == std::string::npos) {
        std::fprintf(stderr,
            "vlm-server: WARNING: bound to %s with no authentication. Any host that can\n"
            "            reach this address may submit requests. Restrict to a trusted\n"
            "            network, or bind tcp://127.0.0.1:PORT for local use.\n",
            bind_addr.c_str());
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    zmq::pollitem_t poll[] = {{ static_cast<void*>(sock), 0, ZMQ_POLLIN, 0 }};
    uint64_t served = 0;

    while (!g_shutdown.load(std::memory_order_relaxed)) {
        try {
            zmq::poll(poll, 1, std::chrono::milliseconds(200));
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR) continue;
            if (e.num() == ETERM) break;
            std::fprintf(stderr, "vlm-server: zmq error: %s\n", e.what());
            continue;
        }
        if (!(poll[0].revents & ZMQ_POLLIN)) continue;

        std::vector<std::string> env;
        std::string payload;
        bool recv_ok = true, have_payload = false;
        for (;;) {
            zmq::message_t part;
            try {
                auto rr = sock.recv(part, zmq::recv_flags::none);
                if (!rr) { recv_ok = false; break; }
            } catch (const zmq::error_t & e) {
                if (e.num() != EINTR)
                    std::fprintf(stderr, "vlm-server: zmq recv error: %s\n", e.what());
                recv_ok = false; break;
            }
            if (sock.get(zmq::sockopt::rcvmore)) {
                env.emplace_back(static_cast<const char*>(part.data()), part.size());
            } else {
                payload.assign(static_cast<const char*>(part.data()), part.size());
                have_payload = true;
                break;
            }
        }
        if (!recv_ok || !have_payload || env.empty()) continue;

        auto send_reply = [&](const std::string & body) {
            try {
                for (const auto & e : env) {
                    sock.send(zmq::buffer(e), zmq::send_flags::sndmore);
                }
                sock.send(zmq::buffer(body), zmq::send_flags::none);
            } catch (const zmq::error_t & e) {
                // A throw mid-multipart leaves the ROUTER frame open, which would
                // corrupt the next reply; shut down cleanly instead of crashing.
                std::fprintf(stderr, "vlm-server: reply send failed (%s); shutting down\n", e.what());
                g_shutdown.store(true, std::memory_order_relaxed);
            }
        };

        vlm_chat::ChatRequest req;
        if (!req.ParseFromString(payload)) {
            std::fprintf(stderr, "vlm-server: ChatRequest parse failed (size=%zu)\n", payload.size());
            send_reply(make_error_stream(0, "request parse failed"));
            continue;
        }
        const uint64_t rid = req.request_id();

        if (req.messages_size() < 1) {
            send_reply(make_error_stream(rid, "ChatRequest has no messages"));
            continue;
        }
        if (req.images_size() > 16) {
            send_reply(make_error_stream(rid, "too many image views (max 16)"));
            continue;
        }

        std::vector<vlm::Image> images;
        images.reserve(req.images_size());
        bool decode_ok = true;
        for (int v = 0; v < req.images_size(); ++v) {
            const vlm_chat::Image & im = req.images(v);
            vlm::Image out;
            if (im.encoding() == vlm_chat::Image::JPEG) {
                const auto & d = im.data();
                if (!engine.decode_image_buf(
                        reinterpret_cast<const uint8_t*>(d.data()), d.size(), out)) {
                    char buf[64]; std::snprintf(buf, sizeof(buf), "image[%d] JPEG decode failed", v);
                    send_reply(make_error_stream(rid, buf));
                    decode_ok = false; break;
                }
            } else if (im.encoding() == vlm_chat::Image::RGB_U8) {
                if (im.width() == 0 || im.height() == 0 ||
                    im.width() > kMaxImageDim || im.height() > kMaxImageDim) {
                    char buf[96]; std::snprintf(buf, sizeof(buf),
                        "image[%d] RGB_U8 dims %ux%u out of range (max %u)", v,
                        im.width(), im.height(), kMaxImageDim);
                    send_reply(make_error_stream(rid, buf));
                    decode_ok = false; break;
                }
                const size_t expected = size_t(3) * im.width() * im.height();
                if (im.data().size() != expected) {
                    char buf[96]; std::snprintf(buf, sizeof(buf),
                        "image[%d] RGB_U8 size %zu != 3*%u*%u", v,
                        im.data().size(), im.width(), im.height());
                    send_reply(make_error_stream(rid, buf));
                    decode_ok = false; break;
                }
                out.width  = im.width();
                out.height = im.height();
                out.rgb.assign(im.data().begin(), im.data().end());
            } else {
                char buf[64]; std::snprintf(buf, sizeof(buf), "image[%d] unknown encoding", v);
                send_reply(make_error_stream(rid, buf));
                decode_ok = false; break;
            }
            images.push_back(std::move(out));
        }
        if (!decode_ok) continue;

        std::vector<vlm::Message> messages;
        messages.reserve(req.messages_size());
        for (const auto & m : req.messages()) {
            messages.push_back({m.role(), m.content()});
        }

        vlm::SamplingParams sp;
        if (req.has_sampling()) {
            const auto & s = req.sampling();
            sp.temperature = s.temperature();
            sp.top_p       = s.top_p();
            sp.top_k       = s.top_k();
            // Cap client-requested length at the context window so a huge
            // max_tokens cannot monopolize the single-threaded decode loop.
            const uint32_t cap  = uint32_t(lp.n_ctx > 0 ? lp.n_ctx : 4096);
            const uint32_t want = s.max_tokens() > 0 ? s.max_tokens() : 256;
            sp.max_tokens = int(std::min<uint32_t>(want, cap));
            sp.seed        = s.seed() != 0 ? s.seed() : 0xFFFFFFFFu;
        }

        vlm::TokenCallback on_token = nullptr;
        if (req.stream()) {
            on_token = [&](const std::string & piece) -> bool {
                vlm_chat::StreamMessage sm;
                vlm_chat::ChatStreamDelta * d = sm.mutable_delta();
                d->set_request_id(rid);
                d->set_delta(piece);
                send_reply(sm.SerializeAsString());
                return true;
            };
        }

        const auto t0 = std::chrono::steady_clock::now();
        const vlm::ChatResult r = engine.chat(messages, images, sp, on_token);
        const float ms_total = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        vlm_chat::StreamMessage sm;
        vlm_chat::ChatResponse * resp = sm.mutable_final();
        resp->set_request_id(rid);
        resp->set_finish_reason(r.finish_reason);
        resp->set_text(r.text);
        resp->set_prompt_tokens(r.prompt_tokens);
        resp->set_completion_tokens(r.completion_tokens);
        resp->set_latency_ms_total(ms_total);
        resp->set_latency_ms_prefill(r.ms_prefill);
        resp->set_latency_ms_decode(r.ms_decode);
        if (r.finish_reason == "error") resp->set_error(r.error);
        send_reply(sm.SerializeAsString());

        ++served;
        std::printf("vlm-server: req #%llu rid=%llu finish=%s pt=%d ct=%d %.0fms%s\n",
                    (unsigned long long) served, (unsigned long long) rid,
                    r.finish_reason.c_str(), r.prompt_tokens, r.completion_tokens, ms_total,
                    req.stream() ? " (streamed)" : "");
    }

    std::printf("vlm-server: shutting down after %llu requests.\n",
                (unsigned long long) served);
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
