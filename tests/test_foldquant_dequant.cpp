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

// FoldQuant CPU reference ops, run as GGML_OP_CUSTOM nodes on the CPU backend
// with several threads, checked three ways:
// fq_dequant_rows (FoldQuant read back as a float weight, foldquant.h) against
// the dense product it stands for: W_deq . R . diag(1/a) (fold before) or
// W_deq . diag(1/a) . R (fold after), with R the block-diagonal normalised
// natural-order Sylvester-Hadamard matrix. Both widths, both fold orders, with
// and without ascale, and a K whose rotation block narrows.

#include "foldquant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vla;

namespace {

// Natural-order Sylvester-Hadamard entry (i, j) of size n: (-1)^popcount(i & j).
float hadamard(int i, int j) {
    return (__builtin_popcount((unsigned) (i & j)) & 1) ? -1.0f : 1.0f;
}

int run(int64_t K, int64_t N, int wbits, bool fold_before, bool with_ascale, std::mt19937 & rng) {
    const int rb = fq_rot_block_for(K, 64);
    const int qmax = wbits == 4 ? 7 : 127;
    std::uniform_int_distribution<int>   qd(-qmax, qmax);
    std::uniform_real_distribution<float> sd(0.001f, 0.02f), ad(0.5f, 2.0f);

    std::vector<int8_t> q((size_t) (N * K));
    for (auto & v : q) v = (int8_t) qd(rng);
    std::vector<float> ws((size_t) N), as((size_t) K);
    for (auto & v : ws) v = sd(rng);
    for (auto & v : as) v = ad(rng);

    // Codes as the GGUF stores them.
    const int64_t kp = fq_w_kpack(K, wbits);
    std::vector<int8_t> codes((size_t) (N * kp));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int8_t v = q[(size_t) (n * K + k)];
            if (wbits == 8) {
                codes[(size_t) (n * kp + k)] = v;
            } else {
                uint8_t & b = (uint8_t &) codes[(size_t) (n * kp + k / 2)];
                b = (k % 2 == 0) ? (uint8_t) ((b & 0xF0) | (v & 0xF)) : (uint8_t) ((b & 0x0F) | ((v & 0xF) << 4));
            }
        }

    std::vector<float> got((size_t) (N * K));
    fq_dequant_rows(codes.data(), ws.data(), with_ascale ? as.data() : nullptr, K, N, wbits, rb, fold_before, got.data());

    // Dense reference in double.
    const double norm = 1.0 / std::sqrt((double) rb);
    double max_err = 0.0, max_ref = 0.0;
    for (int64_t n = 0; n < N; ++n) {
        std::vector<double> w((size_t) K);
        for (int64_t k = 0; k < K; ++k) w[(size_t) k] = (double) q[(size_t) (n * K + k)] * ws[(size_t) n];
        if (with_ascale && !fold_before)
            for (int64_t k = 0; k < K; ++k) w[(size_t) k] /= as[(size_t) k];
        std::vector<double> r((size_t) K, 0.0);
        for (int64_t j = 0; j < K; ++j) {       // (w . R)[j] = sum_i w[i] R[i][j]
            const int64_t blk = j / rb * rb;
            double acc = 0.0;
            for (int64_t i = blk; i < blk + rb; ++i)
                acc += w[(size_t) i] * hadamard((int) (i - blk), (int) (j - blk)) * norm;
            r[(size_t) j] = rb > 1 ? acc : w[(size_t) j];
        }
        if (with_ascale && fold_before)
            for (int64_t k = 0; k < K; ++k) r[(size_t) k] /= as[(size_t) k];
        for (int64_t k = 0; k < K; ++k) {
            max_err = std::fmax(max_err, std::fabs(r[(size_t) k] - got[(size_t) (n * K + k)]));
            max_ref = std::fmax(max_ref, std::fabs(r[(size_t) k]));
        }
    }
    const bool ok = max_err <= 1e-5 * std::fmax(1.0, max_ref);
    std::printf("%s K=%lld rb=%d W%d fold_%s ascale=%d: max|err| %.3g (max|w| %.3g)\n", ok ? "ok  " : "FAIL",
                (long long) K, rb, wbits, fold_before ? "before" : "after", (int) with_ascale, max_err, max_ref);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(1234);
    int fails = 0;
    for (int64_t K : {128, 96})                 // 96 narrows the nominal 64-block to 32
        for (int wbits : {8, 4})
            for (bool before : {true, false})
                for (bool asc : {true, false})
                    fails += run(K, 64, wbits, before, asc, rng);
    std::printf("%s\n", fails ? "test_foldquant_dequant: FAILED" : "test_foldquant_dequant: all ok");
    return fails ? 1 : 0;
}
