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

// Pins the Qwen3-VL patch geometry. predict_check covers the graph builders.

#include "models/qwen3vl_vit.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vla;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// 4x4 grid, 2x2 merge: patches arrive grouped by block, not in raster order.
static void test_merge_block_coords() {
    std::vector<int64_t> row, col;
    merge_block_coords(4, 4, 2, row, col);
    CHECK(row.size() == 16 && col.size() == 16);
    const int64_t want_r[16] = {0,0,1,1, 0,0,1,1, 2,2,3,3, 2,2,3,3};
    const int64_t want_c[16] = {0,1,0,1, 2,3,2,3, 0,1,0,1, 2,3,2,3};
    for (int i = 0; i < 16; ++i) { CHECK(row[i] == want_r[i]); CHECK(col[i] == want_c[i]); }
}

// Half-split layout: the second half of the table repeats the first.
static void test_vit_rope_tables() {
    std::vector<int64_t> row = {0, 1}, col = {0, 2};
    std::vector<float> c, s;
    const int64_t hd = 8;
    vit_rope_tables(row, col, hd, 10000.0, c, s);
    CHECK((int64_t) c.size() == 2 * hd && (int64_t) s.size() == 2 * hd);
    for (int64_t p = 0; p < 2; ++p)
        for (int64_t i = 0; i < hd / 2; ++i) {
            CHECK(c[p * hd + i] == c[p * hd + hd / 2 + i]);
            CHECK(s[p * hd + i] == s[p * hd + hd / 2 + i]);
        }
    // Position 0 has zero angle on every frequency.
    for (int64_t i = 0; i < hd; ++i) { CHECK(c[i] == 1.0f); CHECK(s[i] == 0.0f); }
    // Row 1, col 2 with inv_freq[0] == 1: row half is cos(1), col half is cos(2).
    CHECK(std::fabs(c[hd + 0] - std::cos(1.0f)) < 1e-6f);
    CHECK(std::fabs(c[hd + hd / 4] - std::cos(2.0f)) < 1e-6f);
}

// Same grid as the source table is an identity resample.
static void test_interp_pos_embed_identity() {
    const int64_t side = 2, hidden = 2;
    std::vector<float> table = {0,10, 1,11, 2,12, 3,13};
    std::vector<int64_t> row, col;
    merge_block_coords(side, side, 1, row, col);
    std::vector<float> out;
    interp_pos_embed(table, side, hidden, row, col, side, side, out);
    CHECK((int64_t) out.size() == side * side * hidden);
    for (size_t s = 0; s < row.size(); ++s) {
        const int64_t src = row[s] * side + col[s];
        CHECK(std::fabs(out[s * hidden + 0] - table[src * hidden + 0]) < 1e-6f);
        CHECK(std::fabs(out[s * hidden + 1] - table[src * hidden + 1]) < 1e-6f);
    }
}

// Upsampling 2x2 to 3x3 puts the midpoint at the mean of the four corners.
static void test_interp_pos_embed_bilinear() {
    const int64_t side = 2, hidden = 1;
    std::vector<float> table = {0, 2, 4, 6};
    std::vector<int64_t> row, col;
    merge_block_coords(3, 3, 1, row, col);
    std::vector<float> out;
    interp_pos_embed(table, side, hidden, row, col, 3, 3, out);
    for (size_t s = 0; s < row.size(); ++s)
        if (row[s] == 1 && col[s] == 1) CHECK(std::fabs(out[s] - 3.0f) < 1e-6f);
}

// A wrong-sized view must be rejected before any pixel is read.
static void test_preprocess_rejects_bad_view() {
    std::vector<uint8_t> px(3 * 4 * 4, 0);
    std::vector<int64_t> row, col;
    merge_block_coords(2, 2, 1, row, col);
    std::vector<float> out;
    ImageView bad{px.data(), 3, 4, PixelFormat::U8};
    CHECK(!preprocess_image_patches("test", bad, 4, 2, 2, row, col, out));
    ImageView null{nullptr, 4, 4, PixelFormat::U8};
    CHECK(!preprocess_image_patches("test", null, 4, 2, 2, row, col, out));
}

// U8 128 maps to ~0 after the 0.5/0.5 normalization, and every temporal slice
// repeats the same value.
static void test_preprocess_values() {
    const int64_t side = 2, ps = 1, tps = 2;
    std::vector<uint8_t> px((size_t) 3 * side * side, 128);
    std::vector<int64_t> row, col;
    merge_block_coords(side, side, 1, row, col);
    std::vector<float> out;
    ImageView v{px.data(), (int) side, (int) side, PixelFormat::U8};
    CHECK(preprocess_image_patches("test", v, side, ps, tps, row, col, out));
    const int64_t pf = 3 * tps * ps * ps;
    CHECK((int64_t) out.size() == pf * (int64_t) row.size());
    for (float f : out) CHECK(std::fabs(f - (128.0f / 255.0f * 2.0f - 1.0f)) < 1e-6f);
}

int main() {
    test_merge_block_coords();
    test_vit_rope_tables();
    test_interp_pos_embed_identity();
    test_interp_pos_embed_bilinear();
    test_preprocess_rejects_bad_view();
    test_preprocess_values();
    if (fails == 0) std::printf("qwen3vl_vit: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
