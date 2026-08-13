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

#include "modules/siglip_vit.h"

#include "layers/norm.h"

#include <cstdio>

namespace vla {

void SigLipTower::declare(WeightLoader & L, const char * prefix, int64_t layers, bool patch_embd_is_gemm) {
    patch_w   = patch_embd_is_gemm ? L.gemm("%s.patch_embd.weight", prefix)
                                   : L.f32 ("%s.patch_embd.weight", prefix);
    patch_b   = L.f32("%s.patch_embd.bias", prefix);
    pos       = L.f32("%s.pos_embd",        prefix);
    post_ln_w = L.f32("%s.post_ln.weight",  prefix);
    post_ln_b = L.f32("%s.post_ln.bias",    prefix);

    char blk_prefix[192];
    std::snprintf(blk_prefix, sizeof(blk_prefix), "%s.blk", prefix);
    enc.declare(L, blk_prefix, layers);
}

ggml_tensor * SigLipTower::embed_conv(ggml_context * C, ggml_tensor * pixels, int64_t patch, int64_t grid) const {
    ggml_tensor * conv = ggml_conv_2d(C, patch_w, pixels, (int)patch, (int)patch, 0, 0, 1, 1);
    ggml_tensor * flat = ggml_cont(C, ggml_transpose(C, ggml_reshape_2d(C, conv, grid*grid, enc.cfg.hidden)));
    return ggml_add(C, ggml_add(C, flat, patch_b), pos);
}

ggml_tensor * SigLipTower::embed_patches(ggml_context * C, ggml_tensor * patches) const {
    return ggml_add(C, ggml_add(C, ggml_mul_mat(C, patch_w, patches), patch_b), pos);
}

ggml_tensor * SigLipTower::build(ggml_context * C, ggml_tensor * h, int64_t seq, int64_t nv) const {
    h = enc.build(C, h, seq, nv);
    return layer_norm(C, h, post_ln_w, post_ln_b, enc.cfg.ln_eps);
}

}
