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
#include "serving/vla.pb.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include <zmq.hpp>

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::atomic<bool> g_shutdown{false};

void on_signal(int) { g_shutdown.store(true, std::memory_order_relaxed); }

bool decode_image(const vla::Image & img,
                  std::vector<uint8_t> & u8,
                  std::vector<float> & f32,
                  vla::ImageView & view) {
    if (img.encoding() == vla::Image::JPEG) {
        int w = 0, h = 0, ch = 0;
        const auto & data = img.data();
        unsigned char * px = stbi_load_from_memory(
            reinterpret_cast<const unsigned char *>(data.data()),
            static_cast<int>(data.size()),
            &w, &h, &ch,  3);
        if (!px) {
            std::fprintf(stderr, "vla-server: stbi_load_from_memory failed: %s\n",
                         stbi_failure_reason());
            return false;
        }
        u8.assign(px, px + size_t(3) * w * h);
        stbi_image_free(px);
        view = { u8.data(), w, h, vla::PixelFormat::U8 };
        return true;
    } else if (img.encoding() == vla::Image::RGB_U8) {
        const size_t expected = size_t(3) * img.width() * img.height();
        if (img.data().size() != expected) {
            std::fprintf(stderr, "vla-server: RGB_U8 size %zu != 3*%u*%u = %zu\n",
                         img.data().size(), img.width(), img.height(), expected);
            return false;
        }
        u8.assign(reinterpret_cast<const uint8_t*>(img.data().data()),
                  reinterpret_cast<const uint8_t*>(img.data().data()) + expected);
        view = { u8.data(), int(img.width()), int(img.height()), vla::PixelFormat::U8 };
        return true;
    } else if (img.encoding() == vla::Image::F32_RGB_01) {
        const size_t pixels   = size_t(3) * img.width() * img.height();
        const size_t expected = pixels * sizeof(float);
        if (img.data().size() != expected) {
            std::fprintf(stderr, "vla-server: F32_RGB_01 size %zu != 4*3*%u*%u = %zu\n",
                         img.data().size(), img.width(), img.height(), expected);
            return false;
        }

        f32.resize(pixels);
        std::memcpy(f32.data(), img.data().data(), expected);
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
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) return i;
    }
    return -1;
}

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s [--bind ADDR] [--timing-detail none|phase] [--config PATH] "
        "[<mmproj.gguf>] <ckpt>\n"
        "  <mmproj.gguf>           vision-tower mmproj GGUF (SigLIP / PaliGemma /\n"
        "                          connector). Required for SmolVLA, π0, Evo-1, GR00T.\n"
        "                          Omit for BitVLA - its vision tower is baked into\n"
        "                          the combined ckpt GGUF.\n"
        "  <ckpt>                  SmolVLA .safetensors or .gguf, or any of the other\n"
        "                          supported architectures' .gguf; the architecture is\n"
        "                          auto-detected from the checkpoint.\n"
        "  --bind ADDR             ZMQ bind address (default: tcp://*:5555)\n"
        "  --timing-detail LEVEL   per-request timing breakdown (default: none)\n"
        "                          'none'  : single ms_inference\n"
        "                          'phase' : ms_prefill + ms_denoise broken out\n"
        "                          (π0 currently reports only the combined ms_inference)\n"
        "  --config PATH           LeRobot policy config.json (SmolVLA safetensors only;\n"
        "                          ignored for GGUF checkpoints). If omitted, uses\n"
        "                          <dirname(ckpt)>/config.json.\n",
        prog);
}

}

int main(int argc, char ** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string bind_addr   = "tcp://*:5555";
    std::string mmproj_path;
    std::string ckpt_path;
    std::string config_path;
    vla::TimingDetail timing_detail = vla::TimingDetail::NONE;

    std::vector<std::string> positionals;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bind" && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--timing-detail" && i + 1 < argc) {
            const std::string v = argv[++i];
            if      (v == "none")  timing_detail = vla::TimingDetail::NONE;
            else if (v == "phase") timing_detail = vla::TimingDetail::PHASE;
            else {
                std::fprintf(stderr, "vla-server: bad --timing-detail value '%s'\n", v.c_str());
                usage(argv[0]);
                return 1;
            }
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            positionals.push_back(std::move(a));
        }
    }
    if (positionals.size() == 1) {
        ckpt_path = positionals[0];
    } else if (positionals.size() == 2) {
        mmproj_path = positionals[0];
        ckpt_path   = positionals[1];
    } else {
        std::fprintf(stderr,
                     "vla-server: expected 1 or 2 positional args "
                     "(<mmproj.gguf> <ckpt> for SmolVLA/π0/Evo-1/GR00T, "
                     "or just <ckpt> for BitVLA), got %zu\n",
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
    vla::Model * model = vla::model_load(mmproj_path, ckpt_path, config_path);
    if (!model) {
        std::fprintf(stderr, "vla-server: model_load failed\n");
        return 1;
    }
    const auto & cfg = vla::model_config(model);
    std::printf("vla-server: loaded. chunk_size=%lld  action_dim=%lld  "
                "n_lang=%lld  hidden=%lld  expert_h=%lld  timing_detail=%s\n",
                (long long) cfg.n_suffix, (long long) cfg.max_action_dim,
                (long long) cfg.n_lang, (long long) cfg.hidden, (long long) cfg.expert_h,
                timing_detail == vla::TimingDetail::PHASE ? "phase" : "none");

    zmq::context_t zctx( 1);
    zmq::socket_t  sock(zctx, zmq::socket_type::rep);
    sock.set(zmq::sockopt::linger, 0);
    sock.bind(bind_addr);
    std::printf("vla-server: bound to %s. ready.\n", bind_addr.c_str());

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    zmq::pollitem_t poll[] = {{ static_cast<void*>(sock), 0, ZMQ_POLLIN, 0 }};

    uint64_t served = 0;
    while (!g_shutdown.load(std::memory_order_relaxed)) {

        try {
            zmq::poll(poll, 1, std::chrono::milliseconds(200));
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR) continue;
            throw;
        }
        if (!(poll[0].revents & ZMQ_POLLIN)) continue;

        zmq::message_t req_msg;
        try {
            auto rr = sock.recv(req_msg, zmq::recv_flags::none);
            if (!rr) continue;
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR) continue;
            throw;
        }

        vla::PredictRequest req;
        if (!req.ParseFromArray(req_msg.data(), static_cast<int>(req_msg.size()))) {
            std::fprintf(stderr, "vla-server: PredictRequest parse failed (size=%zu)\n",
                         req_msg.size());
            const std::string body = make_error_response(0, "request parse failed");
            sock.send(zmq::buffer(body), zmq::send_flags::none);
            continue;
        }

        const uint64_t rid = req.request_id();

        if (req.images_size() < 1 && req.precomputed_img_emb_size() == 0) {
            const std::string body = make_error_response(rid,
                "PredictRequest must contain images or precomputed_img_emb");
            sock.send(zmq::buffer(body), zmq::send_flags::none);
            continue;
        }
        if (req.lang_tokens_size() < 1 || req.lang_tokens_size() > int(cfg.n_lang)) {
            char buf[128]; std::snprintf(buf, sizeof(buf),
                "lang_tokens length %d out of range [1, %lld]",
                req.lang_tokens_size(), (long long) cfg.n_lang);
            sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
            continue;
        }
        {
            bool tokens_ok = true;
            for (int t = 0; t < req.lang_tokens_size(); ++t) {
                if (req.lang_tokens(t) < 0) {
                    char buf[128]; std::snprintf(buf, sizeof(buf),
                        "lang_tokens[%d] = %d is negative", t, req.lang_tokens(t));
                    sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                    tokens_ok = false;
                    break;
                }
            }
            if (!tokens_ok) continue;
        }
        if (req.state_size() != int(cfg.max_state_dim)) {
            char buf[128]; std::snprintf(buf, sizeof(buf),
                "state length %d != expected %lld",
                req.state_size(), (long long) cfg.max_state_dim);
            sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
            continue;
        }
        const int expected_noise_n = int(cfg.n_suffix * cfg.max_action_dim);
        if (req.noise_size() != 0 && req.noise_size() != expected_noise_n) {
            char buf[128]; std::snprintf(buf, sizeof(buf),
                "noise length %d != 0 or %d (chunk_size * action_dim)",
                req.noise_size(), expected_noise_n);
            sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
            continue;
        }

        const bool use_precomputed = req.precomputed_img_emb_size() > 0;
        std::vector<float> precomputed_emb;
        int precomputed_n_views = 0;

        const int n_views = req.images_size();
        std::vector<std::vector<uint8_t>> u8_bufs (n_views);
        std::vector<std::vector<float>>   f32_bufs(n_views);
        std::vector<vla::ImageView>       img_views(n_views);

        if (use_precomputed) {
            precomputed_n_views = static_cast<int>(req.precomputed_img_emb_n_views());
            const int per_view = int(cfg.n_img * cfg.hidden);
            const int expected = per_view * precomputed_n_views;
            if (precomputed_n_views < 1 || req.precomputed_img_emb_size() != expected) {
                char buf[160]; std::snprintf(buf, sizeof(buf),
                    "precomputed_img_emb size %d != %d (n_views=%d * n_img_per_view=%lld * hidden=%lld)",
                    req.precomputed_img_emb_size(), expected, precomputed_n_views,
                    (long long) cfg.n_img, (long long) cfg.hidden);
                sock.send(zmq::buffer(make_error_response(rid, buf)),
                          zmq::send_flags::none);
                continue;
            }
            precomputed_emb.assign(req.precomputed_img_emb().begin(),
                                   req.precomputed_img_emb().end());
            const int bad = find_non_finite(precomputed_emb.data(),
                                            static_cast<int>(precomputed_emb.size()));
            if (bad >= 0) {
                char buf[128]; std::snprintf(buf, sizeof(buf),
                    "precomputed_img_emb[%d] = %g is not finite (NaN/Inf)",
                    bad, precomputed_emb[bad]);
                sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                continue;
            }
        } else {

            bool decode_ok = true;
            for (int v = 0; v < n_views; ++v) {
                if (!decode_image(req.images(v), u8_bufs[v], f32_bufs[v], img_views[v])) {
                    char buf[64]; std::snprintf(buf, sizeof(buf), "image[%d] decode failed", v);
                    sock.send(zmq::buffer(make_error_response(rid, buf)),
                              zmq::send_flags::none);
                    decode_ok = false;
                    break;
                }
            }
            if (!decode_ok) continue;
        }

        std::vector<int32_t> lang_tokens(req.lang_tokens().begin(), req.lang_tokens().end());
        std::vector<float>   state_vec(req.state().begin(), req.state().end());
        {
            const int bad = find_non_finite(state_vec.data(), static_cast<int>(state_vec.size()));
            if (bad >= 0) {
                char buf[128]; std::snprintf(buf, sizeof(buf),
                    "state[%d] = %g is not finite (NaN/Inf)", bad, state_vec[bad]);
                sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                continue;
            }
        }
        std::vector<float>   noise_vec;
        if (req.noise_size() == expected_noise_n) {
            noise_vec.assign(req.noise().begin(), req.noise().end());
            const int bad = find_non_finite(noise_vec.data(), static_cast<int>(noise_vec.size()));
            if (bad >= 0) {
                char buf[128]; std::snprintf(buf, sizeof(buf),
                    "noise[%d] = %g is not finite (NaN/Inf)", bad, noise_vec[bad]);
                sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                continue;
            }
        }

        vla::Inputs in;
        if (use_precomputed) {
            in.precomputed_img_emb = precomputed_emb.data();
            in.n_img_views         = precomputed_n_views;

            in.images              = nullptr;
            in.n_images            = 0;
        } else {
            in.images              = img_views.data();
            in.n_images            = n_views;
            in.precomputed_img_emb = nullptr;
            in.n_img_views         = 0;
        }
        std::vector<int32_t> attn_mask_vec;
        if (req.attention_mask_size() > 0) {
            attn_mask_vec.assign(req.attention_mask().begin(), req.attention_mask().end());
        }
        in.lang_tokens      = lang_tokens.data();
        in.n_lang           = static_cast<int>(lang_tokens.size());
        in.state            = state_vec.data();
        in.noise            = noise_vec.empty() ? nullptr : noise_vec.data();
        in.attention_mask   = attn_mask_vec.empty() ? nullptr : attn_mask_vec.data();
        in.attention_mask_n = static_cast<int>(attn_mask_vec.size());
        in.timing_detail    = timing_detail;

        std::vector<float> action_chunk = vla::predict(model, in);
        const auto & st = vla::last_stats(model);

        if (action_chunk.empty()) {
            sock.send(zmq::buffer(make_error_response(rid, "predict failed")),
                      zmq::send_flags::none);
            continue;
        }

        vla::PredictResponse resp;
        resp.set_request_id(rid);
        resp.mutable_action_chunk()->Reserve(static_cast<int>(action_chunk.size()));
        for (float v : action_chunk) resp.add_action_chunk(v);
        resp.set_chunk_size(static_cast<uint32_t>(cfg.n_suffix));
        resp.set_action_dim(static_cast<uint32_t>(cfg.max_action_dim));
        resp.set_latency_ms_total(st.ms_total);
        resp.set_latency_ms_vision(st.ms_vision);
        resp.set_latency_ms_inference(st.ms_inference);
        resp.set_latency_ms_prefill(st.ms_prefill);
        resp.set_latency_ms_denoise(st.ms_denoise);

        const std::string body = resp.SerializeAsString();
        sock.send(zmq::buffer(body), zmq::send_flags::none);

        ++served;
        if (served % 10 == 1) {
            const float ms_other = std::max(0.f,
                st.ms_total - st.ms_vision - st.ms_inference);
            if (timing_detail == vla::TimingDetail::PHASE) {
                std::printf("vla-server: rid=%llu  served=%llu  total=%.1f ms  "
                            "vision=%.1f  inf=%.1f (prefill=%.1f + denoise=%.1f)  other=%.1f\n",
                            (unsigned long long) rid, (unsigned long long) served,
                            st.ms_total, st.ms_vision, st.ms_inference,
                            st.ms_prefill, st.ms_denoise, ms_other);
            } else {
                std::printf("vla-server: rid=%llu  served=%llu  total=%.1f ms  "
                            "vision=%.1f  inf=%.1f  other=%.1f\n",
                            (unsigned long long) rid, (unsigned long long) served,
                            st.ms_total, st.ms_vision, st.ms_inference, ms_other);
            }
            std::fflush(stdout);
        }
    }

    std::printf("vla-server: shutting down (served %llu requests)\n",
                (unsigned long long) served);
    sock.close();
    zctx.close();
    vla::model_free(model);
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
