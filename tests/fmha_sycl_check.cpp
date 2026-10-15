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

// Drives the sycl-tla flash attention (src/sycl/vla_sycl_fmha.cpp) on random
// inputs over a list of shapes and compares each against a double-precision
// host reference. Not a ctest: it needs an Intel GPU.
//
//   fmha_sycl_check [q_std k_std]

#include "sycl/vla_sycl_fmha.h"

#include "ggml.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Dev {
    sycl::queue & q;
    std::vector<void *> ptrs;
    ~Dev() { for (void * p : ptrs) sycl::free(p, q); }
    float * upload(const std::vector<float> & h) {
        float * d = sycl::malloc_device<float>(h.size(), q);
        q.memcpy(d, h.data(), h.size() * sizeof(float)).wait();
        ptrs.push_back(d);
        return d;
    }
};

void set3(ggml_tensor & t, int64_t d, int64_t s, int64_t h) {
    t.type = GGML_TYPE_F32;
    t.ne[0] = d; t.ne[1] = s; t.ne[2] = h; t.ne[3] = 1;
    t.nb[0] = sizeof(float); t.nb[1] = d * sizeof(float); t.nb[2] = d * s * sizeof(float); t.nb[3] = t.nb[2] * h;
}

double run(sycl::queue & q, int64_t Sq, int64_t Skv, int64_t Hq, int64_t Hkv, float qs, float ks, bool & claimed) {
    const int64_t D = 256;
    std::mt19937 rng(1234 + (unsigned) (Sq * 7 + Skv));
    std::normal_distribution<float> nq(0.f, qs), nk(0.f, ks), nv(0.f, 1.f);
    std::vector<float> hq(D * Sq * Hq), hk(D * Skv * Hkv), hv(D * Skv * Hkv), ho(D * Hq * Sq);
    for (auto & x : hq) x = nq(rng);
    for (auto & x : hk) x = nk(rng);
    for (auto & x : hv) x = nv(rng);

    Dev dev{q, {}};
    ggml_tensor Q{}, K{}, V{}, O{};
    set3(Q, D, Sq, Hq);  Q.data = dev.upload(hq);
    set3(K, D, Skv, Hkv); K.data = dev.upload(hk);
    set3(V, D, Skv, Hkv); V.data = dev.upload(hv);
    set3(O, D, Hq, Sq);  O.data = dev.upload(ho);
    O.op = GGML_OP_FLASH_ATTN_EXT;
    O.src[0] = &Q; O.src[1] = &K; O.src[2] = &V;
    const float scale = 1.f / 16.f;
    std::memcpy(O.op_params, &scale, sizeof(float));
    std::strncpy(O.name, "test", sizeof(O.name) - 1);

    claimed = vla::sycl_fmha_forward(&O, &q);
    if (!claimed) return -1;
    q.memcpy(ho.data(), O.data, ho.size() * sizeof(float)).wait();

    double worst = 0;
    std::vector<double> sc(Skv), o(D);
    for (int64_t h = 0; h < Hq; ++h) {
        const int64_t g = h / (Hq / Hkv);
        for (int64_t s = 0; s < Sq; ++s) {
            double m = -1e300;
            for (int64_t t = 0; t < Skv; ++t) {
                double a = 0;
                for (int64_t d = 0; d < D; ++d) a += (double) hq[(h * Sq + s) * D + d] * hk[(g * Skv + t) * D + d];
                sc[t] = a * scale; m = std::max(m, sc[t]);
            }
            double z = 0;
            for (int64_t t = 0; t < Skv; ++t) { sc[t] = std::exp(sc[t] - m); z += sc[t]; }
            std::fill(o.begin(), o.end(), 0.0);
            for (int64_t t = 0; t < Skv; ++t)
                for (int64_t d = 0; d < D; ++d) o[d] += sc[t] / z * hv[(g * Skv + t) * D + d];
            for (int64_t d = 0; d < D; ++d)
                worst = std::max(worst, std::fabs(ho[(s * Hq + h) * D + d] - o[d]));
        }
    }
    return worst;
}

}  // namespace

int main(int argc, char ** argv) {
    const float qs = argc > 2 ? (float) std::atof(argv[1]) : 1.f;
    const float ks = argc > 2 ? (float) std::atof(argv[2]) : 1.f;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s  q_std %.2f k_std %.2f\n", q.get_device().get_info<sycl::info::device::name>().c_str(), qs, ks);
    const int64_t shapes[][4] = {
        {128, 128, 8, 1}, {512, 512, 8, 1}, {528, 528, 8, 1}, {528, 576, 8, 1},
        {576, 576, 8, 1}, {256, 528, 8, 1}, {50, 578, 8, 1}, {50, 576, 8, 1}, {128, 528, 8, 1},
        {528, 528, 8, 8}, {528, 528, 1, 1},
    };
    int bad = 0;
    for (const auto & s : shapes) {
        bool claimed = false;
        const double e = run(q, s[0], s[1], s[2], s[3], qs, ks, claimed);
        const bool ok = claimed && e < 2e-2;
        bad += !ok;
        std::printf("Sq %4lld Skv %4lld Hq %lld Hkv %lld : %s max|err| %.4g\n", (long long) s[0], (long long) s[1],
                    (long long) s[2], (long long) s[3], claimed ? (ok ? "ok  " : "BAD ") : "not claimed", e);
    }
    return bad ? 1 : 0;
}
