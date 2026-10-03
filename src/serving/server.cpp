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

#include "model.h"
#include "options.h"
#include "serving/hf_fetch.h"
#include "serving/request_queue.h"
#include "serving/vla.pb.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
// stb ships its full implementation here; silence its unused-function noise so
// our own -Wall -Wextra output stays meaningful.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "stb_image.h"
#pragma GCC diagnostic pop

#include <zmq.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_shutdown{false};

void on_signal(int) {
    g_shutdown.store(true, std::memory_order_relaxed);
}

// Reject absurd image dimensions before any size arithmetic, so an untrusted
// width or height cannot overflow size_t or truncate to a negative int.
constexpr unsigned kMaxImageDim = 8192;
constexpr size_t   kMaxTotalPixels = size_t(64) << 20;

bool decode_image(const vla::Image & img,
                  std::vector<uint8_t> & u8,
                  std::vector<float> & f32,
                  vla::ImageView & view) {
    if (img.encoding() == vla::Image::JPEG) {
        int w = 0, h = 0, ch = 0;
        const auto & data = img.data();
        if (data.size() > size_t(INT_MAX)) {
            std::fprintf(stderr, "vla-server: JPEG payload too large (%zu bytes)\n",
                         data.size());
            return false;
        }
        // Header first: stbi_load allocates 3*w*h before returning, so a small JPEG
        // declaring huge dimensions would allocate gigabytes before any check.
        if (!stbi_info_from_memory(
                reinterpret_cast<const unsigned char *>(data.data()),
                static_cast<int>(data.size()), &w, &h, &ch)) {
            std::fprintf(stderr, "vla-server: stbi_info_from_memory failed: %s\n",
                         stbi_failure_reason());
            return false;
        }
        if (w <= 0 || h <= 0 || w > int(kMaxImageDim) || h > int(kMaxImageDim)) {
            std::fprintf(stderr, "vla-server: JPEG dims %dx%d out of range (max %u)\n",
                         w, h, kMaxImageDim);
            return false;
        }
        unsigned char * px = stbi_load_from_memory(
            reinterpret_cast<const unsigned char *>(data.data()),
            static_cast<int>(data.size()),
            &w, &h, &ch,  3);
        if (!px) {
            std::fprintf(stderr, "vla-server: stbi_load_from_memory failed: %s\n",
                         stbi_failure_reason());
            return false;
        }
        if (w <= 0 || h <= 0 || w > int(kMaxImageDim) || h > int(kMaxImageDim)) {
            std::fprintf(stderr, "vla-server: decoded JPEG dims %dx%d out of range (max %u)\n",
                         w, h, kMaxImageDim);
            stbi_image_free(px);
            return false;
        }
        u8.assign(px, px+size_t(3)*w * h);
        stbi_image_free(px);
        view = { u8.data(), w, h, vla::PixelFormat::U8 };
        return true;
    } else if (img.encoding() == vla::Image::RGB_U8) {
        if (img.width() == 0 || img.height() == 0 ||
            img.width() > kMaxImageDim || img.height() > kMaxImageDim) {
            std::fprintf(stderr, "vla-server: RGB_U8 dims %ux%u out of range (max %u)\n",
                         img.width(), img.height(), kMaxImageDim);
            return false;
        }
        const size_t expected = size_t(3)*img.width()*img.height();
        if (img.data().size() != expected) {
            std::fprintf(stderr, "vla-server: RGB_U8 size %zu != 3*%u*%u = %zu\n",
                         img.data().size(), img.width(), img.height(), expected);
            return false;
        }
        u8.assign(reinterpret_cast<const uint8_t*>(img.data().data()),
                  reinterpret_cast<const uint8_t*>(img.data().data())+expected);
        view = { u8.data(), int(img.width()), int(img.height()), vla::PixelFormat::U8 };
        return true;
    } else if (img.encoding() == vla::Image::F32_RGB_01) {
        if (img.width() == 0 || img.height() == 0 ||
            img.width() > kMaxImageDim || img.height() > kMaxImageDim) {
            std::fprintf(stderr, "vla-server: F32_RGB_01 dims %ux%u out of range (max %u)\n",
                         img.width(), img.height(), kMaxImageDim);
            return false;
        }
        const size_t pixels   = size_t(3)*img.width()*img.height();
        const size_t expected = pixels * sizeof(float);
        if (img.data().size() != expected) {
            std::fprintf(stderr, "vla-server: F32_RGB_01 size %zu != 4*3*%u*%u = %zu\n",
                         img.data().size(), img.width(), img.height(), expected);
            return false;
        }

        f32.resize(pixels);
        std::memcpy(f32.data(), img.data().data(), expected);
        // State and noise are swept for NaN/Inf; pixels were not, so a bad pixel
        // came back out as a robot action.
        for (size_t i=0; i<pixels; ++i) {
            if (!std::isfinite(f32[i])) {
                std::fprintf(stderr, "vla-server: F32_RGB_01 pixel %zu is not finite\n", i);
                return false;
            }
        }
        view = { f32.data(), int(img.width()), int(img.height()),
                 vla::PixelFormat::F32_RGB_01 };
        return true;
    } else {
        std::fprintf(stderr, "vla-server: unknown image encoding %d\n",
                     int(img.encoding()));
        return false;
    }
}

std::string make_error_response(uint64_t request_id, const std::string & msg) {
    vla::PredictResponse resp;
    resp.set_request_id(request_id);
    resp.set_error(msg);
    return resp.SerializeAsString();
}

int find_non_finite(const float * data, int n) {
    for (int i=0; i<n; ++i) {
        if (!std::isfinite(data[i]))
            return i;
    }
    return -1;
}

using Clock = std::chrono::steady_clock;

// One validated request, decoded on the socket thread and predicted on the
// worker thread. The ImageViews point into u8_bufs / f32_bufs; moving a Job
// moves the vectors' heap buffers with it, so the views stay valid.
struct Job {
    // ROUTER routing frames: the peer identity, plus REQ's empty delimiter.
    // Echoed in front of the reply so the socket routes it back.
    std::vector<zmq::message_t> envelope;
    std::string                 key;      ///< Envelope bytes; one pending job per key in LATEST mode.
    uint64_t                    rid = 0;
    Clock::time_point           t_recv;

    std::vector<std::vector<uint8_t>> u8_bufs;
    std::vector<std::vector<float>>   f32_bufs;
    std::vector<vla::ImageView>       img_views;
    std::vector<float>                precomputed_emb;
    int                               precomputed_n_views = 0;
    bool                              use_precomputed = false;

    std::vector<int32_t> lang_tokens;
    std::vector<float>   state;
    std::vector<float>   noise;
    std::vector<int32_t> attn_mask;
};

// Checks the request against the model and fills the job. Returns an empty
// string on success, else the error to send back.
std::string build_job(const vla::PredictRequest & req, const vla::Config & cfg, Job & job) {
    char buf[192];
    if (req.images_size() < 1 && req.precomputed_img_emb_size() == 0) {
        return "PredictRequest must contain images or precomputed_img_emb";
    }
    if (req.images_size() > 16) {
        return "too many image views (max 16)";
    }
    if (req.lang_tokens_size() < 1 || req.lang_tokens_size() > int(cfg.n_lang)) {
        std::snprintf(buf, sizeof(buf), "lang_tokens length %d out of range [1, %lld]",
                      req.lang_tokens_size(), (long long) cfg.n_lang);
        return buf;
    }
    for (int t=0; t<req.lang_tokens_size(); ++t) {
        if (req.lang_tokens(t) < 0) {
            std::snprintf(buf, sizeof(buf), "lang_tokens[%d] = %d is negative", t, req.lang_tokens(t));
            return buf;
        }
    }
    if (req.state_size() != int(cfg.max_state_dim)) {
        std::snprintf(buf, sizeof(buf), "state length %d != expected %lld",
                      req.state_size(), (long long) cfg.max_state_dim);
        return buf;
    }
    const int expected_noise_n = int(cfg.n_suffix*cfg.max_action_dim);
    if (req.noise_size() != 0 && req.noise_size() != expected_noise_n) {
        std::snprintf(buf, sizeof(buf), "noise length %d != 0 or %d (chunk_size * action_dim)",
                      req.noise_size(), expected_noise_n);
        return buf;
    }

    job.use_precomputed = req.precomputed_img_emb_size() > 0;
    if (job.use_precomputed) {
        job.precomputed_n_views = static_cast<int>(req.precomputed_img_emb_n_views());
        const int64_t per_view = cfg.n_img*cfg.hidden;
        const int64_t expected = per_view * static_cast<int64_t>(job.precomputed_n_views);
        if (job.precomputed_n_views < 1 || job.precomputed_n_views > 16) {
            std::snprintf(buf, sizeof(buf), "precomputed_img_emb_n_views %d out of range [1, 16]",
                          job.precomputed_n_views);
            return buf;
        }
        if (static_cast<int64_t>(req.precomputed_img_emb_size()) != expected) {
            std::snprintf(buf, sizeof(buf),
                "precomputed_img_emb size %d != %lld (n_views=%d * n_img_per_view=%lld * hidden=%lld)",
                req.precomputed_img_emb_size(), (long long) expected, job.precomputed_n_views,
                (long long) cfg.n_img, (long long) cfg.hidden);
            return buf;
        }
        job.precomputed_emb.assign(req.precomputed_img_emb().begin(), req.precomputed_img_emb().end());
        const int bad = find_non_finite(job.precomputed_emb.data(),
                                        static_cast<int>(job.precomputed_emb.size()));
        if (bad >= 0) {
            std::snprintf(buf, sizeof(buf), "precomputed_img_emb[%d] = %g is not finite (NaN/Inf)",
                          bad, job.precomputed_emb[bad]);
            return buf;
        }
    } else {
        const int n_views = req.images_size();
        job.u8_bufs.resize(n_views);
        job.f32_bufs.resize(n_views);
        job.img_views.resize(n_views);
        size_t total_px = 0;
        for (int v=0; v<n_views; ++v) {
            if (!decode_image(req.images(v), job.u8_bufs[v], job.f32_bufs[v], job.img_views[v])) {
                std::snprintf(buf, sizeof(buf), "image[%d] decode failed", v);
                return buf;
            }
            if ((total_px += size_t(job.img_views[v].w)*size_t(job.img_views[v].h)) > kMaxTotalPixels) {
                return "images exceed the per-request pixel budget";
            }
        }
    }

    job.lang_tokens.assign(req.lang_tokens().begin(), req.lang_tokens().end());
    job.state.assign(req.state().begin(), req.state().end());
    {
        const int bad = find_non_finite(job.state.data(), static_cast<int>(job.state.size()));
        if (bad >= 0) {
            std::snprintf(buf, sizeof(buf), "state[%d] = %g is not finite (NaN/Inf)", bad, job.state[bad]);
            return buf;
        }
    }
    if (req.noise_size() == expected_noise_n) {
        job.noise.assign(req.noise().begin(), req.noise().end());
        const int bad = find_non_finite(job.noise.data(), static_cast<int>(job.noise.size()));
        if (bad >= 0) {
            std::snprintf(buf, sizeof(buf), "noise[%d] = %g is not finite (NaN/Inf)", bad, job.noise[bad]);
            return buf;
        }
    }
    if (req.attention_mask_size() > 0) {
        job.attn_mask.assign(req.attention_mask().begin(), req.attention_mask().end());
    }
    return "";
}

// Sends envelope frames followed by body on sock. The socket thread owns the
// ROUTER socket; the worker reaches it through an inproc PUSH that the socket
// thread forwards, so both ends use this.
bool send_multipart(zmq::socket_t & sock, std::vector<zmq::message_t> & envelope,
                    const std::string & body) {
    try {
        for (auto & frame : envelope) {
            sock.send(frame, zmq::send_flags::sndmore);
        }
        sock.send(zmq::buffer(body), zmq::send_flags::none);
        return true;
    } catch (const zmq::error_t & e) {
        std::fprintf(stderr, "vla-server: send failed: %s\n", e.what());
        return false;
    }
}

// Receives every frame of one multipart message. Returns false on no message.
bool recv_multipart(zmq::socket_t & sock, std::vector<zmq::message_t> & frames) {
    frames.clear();
    do {
        zmq::message_t frame;
        auto rr = sock.recv(frame, zmq::recv_flags::none);
        if (!rr)
            return !frames.empty();
        frames.push_back(std::move(frame));
    } while (sock.get(zmq::sockopt::rcvmore));
    return true;
}

std::string envelope_key(const std::vector<zmq::message_t> & envelope) {
    std::string key;
    for (const auto & f : envelope) {
        key.append(static_cast<const char*>(f.data()), f.size());
        key.push_back('\0');
    }
    return key;
}

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s [--bind ADDR] [--queue latest|fifo] [--timing-detail none|phase] "
        "[--config PATH] ([<mmproj.gguf>] <ckpt> | -hf user/repo[:file.gguf|:tag])\n"
        "  <mmproj.gguf>           ignored; every arch bundles its vision tower in the\n"
        "                          ckpt GGUF. Accepted so older command lines still work.\n"
        "  -hf                     HuggingFace repo, user/repo[:file.gguf|:tag]; downloaded\n"
        "                          on a miss and cached under $VLA_CACHE. Not combined\n"
        "                          with positional args.\n"
        "  <ckpt>                  SmolVLA .safetensors or .gguf, or any of the other\n"
        "                          supported architectures' .gguf; the architecture is\n"
        "                          auto-detected from the checkpoint.\n"
        "  --bind ADDR             ZMQ bind address (default: tcp://*:5555). The socket is\n"
        "                          a ROUTER: REQ clients get one reply per request as\n"
        "                          before; DEALER clients may keep several in flight.\n"
        "  --queue MODE            what to do with requests waiting for the predict\n"
        "                          thread (default: latest)\n"
        "                          'latest': one pending request per client; a newer one\n"
        "                                    replaces it and the old one is answered with\n"
        "                                    error=\"superseded\"\n"
        "                          'fifo'  : serve every request in arrival order\n"
        "  --timing-detail LEVEL   per-request timing breakdown (default: none)\n"
        "                          'none'  : single ms_inference\n"
        "                          'phase' : ms_prefill + ms_denoise broken out\n"
        "                          (only SmolVLA, BitVLA and VLA-JEPA report the split;\n"
        "                          the others report ms_inference only)\n"
        "%s"
        "  --config PATH           policy config.json. Its \"runtime\" object sets the\n"
        "                          runtime flags above for any arch; flags given on\n"
        "                          the command line win. SmolVLA safetensors also read\n"
        "                          the policy from it (default <dirname(ckpt)>/config.json).\n",
        prog, vla::Options::usage());
}

}

int main(int argc, char ** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

#ifdef _WIN32
    // The Windows CRT has no line buffering (_IOLBF means full there) and treats
    // a zero size as an invalid argument, which fails fast before main prints a
    // word. Unbuffered is the nearest match.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#else
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
#endif

    std::string bind_addr   = "tcp://*:5555";
    std::string mmproj_path;
    std::string ckpt_path;
    std::string hf_spec;
    std::string config_path;
    vla::TimingDetail timing_detail = vla::TimingDetail::NONE;
    vla::serving::QueueMode queue_mode = vla::serving::QueueMode::LATEST;
    vla::Options opts;
    std::string  opt_err;

    std::vector<std::string> positionals;
    for (int i=1; i<argc; ++i) {
        std::string a = argv[i];
        if (a == "--bind" && i+1 < argc) {
            bind_addr = argv[++i];
        } else if (a == "-hf" && i+1 < argc) {
            hf_spec = argv[++i];
        } else if (a == "--config" && i+1 < argc) {
            config_path = argv[++i];
        } else if (a == "--queue" && i+1 < argc) {
            const std::string v = argv[++i];
            if (!vla::serving::parse_queue_mode(v, queue_mode)) {
                std::fprintf(stderr, "vla-server: bad --queue value '%s'\n", v.c_str());
                usage(argv[0]);
                return 1;
            }
        } else if (a == "--timing-detail" && i+1 < argc) {
            const std::string v = argv[++i];
            if      (v == "none")
                timing_detail = vla::TimingDetail::NONE;
            else if (v == "phase") timing_detail = vla::TimingDetail::PHASE;
            else {
                std::fprintf(stderr, "vla-server: bad --timing-detail value '%s'\n", v.c_str());
                usage(argv[0]);
                return 1;
            }
        } else if (opts.parse_arg(argc, argv, i, opt_err)) {
            continue;
        } else if (!opt_err.empty()) {
            std::fprintf(stderr, "vla-server: %s\n", opt_err.c_str());
            usage(argv[0]);
            return 1;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else if (a.size() > 1 && a[0] == '-') {
            std::fprintf(stderr, "vla-server: unknown option '%s'\n", a.c_str());
            usage(argv[0]);
            return 1;
        } else {
            positionals.push_back(std::move(a));
        }
    }
    if (!hf_spec.empty()) {
        if (!positionals.empty()) {
            std::fprintf(stderr, "vla-server: pass <ckpt> or -hf, not both\n");
            usage(argv[0]);
            return 1;
        }
        ckpt_path = vla::hf_resolve(hf_spec);
        if (ckpt_path.empty())
            return 1;
    } else if (positionals.size() == 1) {
        ckpt_path = positionals[0];
    } else if (positionals.size() == 2) {
        mmproj_path = positionals[0];
        ckpt_path   = positionals[1];
    } else {
        std::fprintf(stderr,
                     "vla-server: expected -hf or [<mmproj.gguf>] <ckpt>, got %zu positional args\n",
                     positionals.size());
        usage(argv[0]);
        return 1;
    }

    std::printf("vla-server: loading model ...\n");
    if (!mmproj_path.empty()) {
        std::printf("  mmproj: %s\n", mmproj_path.c_str());
    }
    std::printf("  ckpt:   %s\n", ckpt_path.c_str());
    if (!config_path.empty()) {
        std::printf("  config: %s\n", config_path.c_str());
    }
    if (!opts.load_json(config_path, opt_err)) {
        std::fprintf(stderr, "vla-server: %s\n", opt_err.c_str());
        return 1;
    }

    vla::Model * model = vla::model_load(mmproj_path, ckpt_path, config_path, opts);
    if (!model) {
        std::fprintf(stderr, "vla-server: model_load failed\n");
        return 1;
    }
    const auto & cfg = vla::model_config(model);
    std::printf("vla-server: loaded. chunk_size=%lld  action_dim=%lld  "
                "n_lang=%lld  hidden=%lld  expert_h=%lld  timing_detail=%s  queue=%s\n",
                (long long) cfg.n_suffix, (long long) cfg.max_action_dim,
                (long long) cfg.n_lang, (long long) cfg.hidden, (long long) cfg.expert_h,
                timing_detail == vla::TimingDetail::PHASE ? "phase" : "none",
                vla::serving::queue_mode_name(queue_mode));

    zmq::context_t zctx(1);
    // ROUTER rather than REP: replies carry the peer's identity, so the socket
    // thread keeps receiving while the worker predicts, and a DEALER client may
    // have more than one request in flight. A REQ client sees the same one
    // request, one reply exchange as before.
    zmq::socket_t sock(zctx, zmq::socket_type::router);
    sock.set(zmq::sockopt::linger, 0);
    // 64 MiB is above any real request (16 views of 512x512 F32 RGB is ~50 MiB) and
    // low enough to bound protobuf's expansion during ParseFromArray.
    sock.set(zmq::sockopt::maxmsgsize, int64_t(64)*1024*1024);
    try {
        sock.bind(bind_addr);
    } catch (const zmq::error_t & e) {
        std::fprintf(stderr, "vla-server: bind %s: %s\n", bind_addr.c_str(), e.what());
        return 1;
    }

    // Replies travel worker -> socket thread over inproc; ZeroMQ sockets are not
    // shareable between threads.
    const char * reply_addr = "inproc://vla-server-replies";
    zmq::socket_t reply_pull(zctx, zmq::socket_type::pull);
    reply_pull.bind(reply_addr);

    std::printf("vla-server: bound to %s. ready.\n", bind_addr.c_str());

    if (bind_addr.find("127.0.0.1") == std::string::npos &&
        bind_addr.find("localhost") == std::string::npos) {
        std::fprintf(stderr,
            "vla-server: WARNING: bound to %s with no authentication. Any host that\n"
            "            can reach this address may submit requests. Restrict access to a\n"
            "            trusted network, or bind tcp://127.0.0.1:PORT for local use.\n",
            bind_addr.c_str());
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    vla::serving::RequestQueue<Job> queue(queue_mode);
    std::atomic<uint64_t> served{0};
    std::atomic<uint64_t> superseded{0};

    // Predict thread: the only caller of vla::predict, so the model needs no lock.
    std::thread worker([&] {
        zmq::socket_t reply_push(zctx, zmq::socket_type::push);
        reply_push.set(zmq::sockopt::linger, 0);
        reply_push.connect(reply_addr);

        Job job;
        while (queue.pop(job)) {
            if (g_shutdown.load(std::memory_order_relaxed)) {
                continue;  // drain without predicting; the socket thread is gone
            }
            const auto t_start = Clock::now();
            const float ms_queue = std::chrono::duration<float, std::milli>(t_start - job.t_recv).count();

            vla::Inputs in;
            if (job.use_precomputed) {
                in.precomputed_img_emb = job.precomputed_emb.data();
                in.n_img_views         = job.precomputed_n_views;
                in.images              = nullptr;
                in.n_images            = 0;
            } else {
                in.images              = job.img_views.data();
                in.n_images            = static_cast<int>(job.img_views.size());
                in.precomputed_img_emb = nullptr;
                in.n_img_views         = 0;
            }
            in.lang_tokens      = job.lang_tokens.data();
            in.n_lang           = static_cast<int>(job.lang_tokens.size());
            in.state            = job.state.data();
            in.noise            = job.noise.empty() ? nullptr : job.noise.data();
            in.attention_mask   = job.attn_mask.empty() ? nullptr : job.attn_mask.data();
            in.attention_mask_n = static_cast<int>(job.attn_mask.size());
            in.timing_detail    = timing_detail;

            std::vector<float> action_chunk = vla::predict(model, in);
            const auto & st = vla::last_stats(model);

            std::string body;
            if (action_chunk.empty()) {
                body = make_error_response(job.rid, "predict failed");
            } else {
                vla::PredictResponse resp;
                resp.set_request_id(job.rid);
                resp.mutable_action_chunk()->Reserve(static_cast<int>(action_chunk.size()));
                for (float v : action_chunk)
                    resp.add_action_chunk(v);
                resp.set_chunk_size(static_cast<uint32_t>(cfg.n_suffix));
                resp.set_action_dim(static_cast<uint32_t>(cfg.max_action_dim));
                resp.set_latency_ms_total(st.ms_total);
                resp.set_latency_ms_vision(st.ms_vision);
                resp.set_latency_ms_inference(st.ms_inference);
                resp.set_latency_ms_prefill(st.ms_prefill);
                resp.set_latency_ms_denoise(st.ms_denoise);
                resp.set_latency_ms_queue(ms_queue);
                body = resp.SerializeAsString();
            }
            send_multipart(reply_push, job.envelope, body);

            const uint64_t n = ++served;
            if (n%10 == 1) {
                const float ms_other = std::max(0.f, st.ms_total-st.ms_vision-st.ms_inference);
                if (timing_detail == vla::TimingDetail::PHASE) {
                    std::printf("vla-server: rid=%llu  served=%llu  superseded=%llu  queue=%.1f ms  "
                                "total=%.1f ms  vision=%.1f  inf=%.1f (prefill=%.1f + denoise=%.1f)  other=%.1f\n",
                                (unsigned long long) job.rid, (unsigned long long) n,
                                (unsigned long long) superseded.load(), ms_queue,
                                st.ms_total, st.ms_vision, st.ms_inference,
                                st.ms_prefill, st.ms_denoise, ms_other);
                } else {
                    std::printf("vla-server: rid=%llu  served=%llu  superseded=%llu  queue=%.1f ms  "
                                "total=%.1f ms  vision=%.1f  inf=%.1f  other=%.1f\n",
                                (unsigned long long) job.rid, (unsigned long long) n,
                                (unsigned long long) superseded.load(), ms_queue,
                                st.ms_total, st.ms_vision, st.ms_inference, ms_other);
                }
                std::fflush(stdout);
            }
        }
    });

    zmq::pollitem_t poll[] = {
        { static_cast<void*>(sock),       0, ZMQ_POLLIN, 0 },
        { static_cast<void*>(reply_pull), 0, ZMQ_POLLIN, 0 },
    };

    // Socket thread: receive, validate, enqueue; forward the worker's replies.
    std::vector<zmq::message_t> frames;
    while (!g_shutdown.load(std::memory_order_relaxed)) {
        try {
            zmq::poll(poll, 2, std::chrono::milliseconds(200));
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR)
                continue;
            if (e.num() == ETERM)
                break;
            std::fprintf(stderr, "vla-server: zmq error: %s\n", e.what());
            continue;
        }

        if (poll[1].revents & ZMQ_POLLIN) {
            try {
                // Forward every reply the worker has queued, not just one per poll.
                while ((reply_pull.get(zmq::sockopt::events) & ZMQ_POLLIN) &&
                       recv_multipart(reply_pull, frames)) {
                    zmq::message_t body = std::move(frames.back());
                    frames.pop_back();
                    // A peer that disconnected is unroutable; ROUTER drops the reply
                    // silently, which is what we want.
                    for (auto & f : frames)
                        sock.send(f, zmq::send_flags::sndmore);
                    sock.send(body, zmq::send_flags::none);
                }
            } catch (const zmq::error_t & e) {
                if (e.num() == ETERM)
                    break;
                std::fprintf(stderr, "vla-server: reply forward failed: %s\n", e.what());
            }
        }

        if (!(poll[0].revents & ZMQ_POLLIN))
            continue;

        try {
            if (!recv_multipart(sock, frames))
                continue;
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR)
                continue;
            if (e.num() == ETERM)
                break;
            std::fprintf(stderr, "vla-server: zmq error: %s\n", e.what());
            continue;
        }

        // ROUTER prepends the peer identity; REQ adds an empty delimiter after it.
        // Everything before the last frame is the envelope to echo.
        if (frames.size() < 2) {
            std::fprintf(stderr, "vla-server: dropped a request with no body\n");
            continue;
        }
        Job job;
        zmq::message_t req_msg = std::move(frames.back());
        frames.pop_back();
        job.envelope = std::move(frames);
        job.key      = envelope_key(job.envelope);
        job.t_recv   = Clock::now();
        frames.clear();

        vla::PredictRequest req;
        if (!req.ParseFromArray(req_msg.data(), static_cast<int>(req_msg.size()))) {
            std::fprintf(stderr, "vla-server: PredictRequest parse failed (size=%zu)\n",
                         req_msg.size());
            send_multipart(sock, job.envelope, make_error_response(0, "request parse failed"));
            continue;
        }
        job.rid = req.request_id();

        const std::string err = build_job(req, cfg, job);
        if (!err.empty()) {
            send_multipart(sock, job.envelope, make_error_response(job.rid, err));
            continue;
        }

        std::optional<Job> displaced;
        const auto rc = queue.push(std::move(job), [](const Job & j) -> const std::string & { return j.key; },
                                   displaced);
        if (rc == vla::serving::PushResult::REPLACED) {
            ++superseded;
            send_multipart(sock, displaced->envelope, make_error_response(displaced->rid, "superseded"));
        } else if (rc == vla::serving::PushResult::REJECTED) {
            send_multipart(sock, displaced->envelope, make_error_response(displaced->rid, "queue full"));
        }
    }

    std::printf("vla-server: shutting down (served %llu requests, superseded %llu)\n",
                (unsigned long long) served.load(), (unsigned long long) superseded.load());
    g_shutdown.store(true, std::memory_order_relaxed);
    queue.stop();
    worker.join();
    reply_pull.close();
    sock.close();
    zctx.close();
    vla::model_free(model);
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
