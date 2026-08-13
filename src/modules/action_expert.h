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

// GR00T action expert: the per-embodiment MLPs that lift state and noisy
// actions into the DiT token stream and project the DiT output back to action
// space. Every projection is a cat_linear row selected by embodiment_id.

#pragma once

#include "loader.h"

#include "ggml.h"

#include <cstdint>

namespace vla {

struct ActionExpert {
    ggml_tensor *se_l1W = nullptr, *se_l1b = nullptr, *se_l2W = nullptr, *se_l2b = nullptr;
    ggml_tensor *ae_W1W = nullptr, *ae_W1b = nullptr, *ae_W2W = nullptr, *ae_W2b = nullptr;
    ggml_tensor *ae_W3W = nullptr, *ae_W3b = nullptr;
    ggml_tensor *ad_l1W = nullptr, *ad_l1b = nullptr, *ad_l2W = nullptr, *ad_l2b = nullptr;
    ggml_tensor *pos_embd = nullptr;

    int64_t embodiment_id = 0;

    // Reads "<prefix>.state_enc.*", "<prefix>.act_enc.*", "<prefix>.act_dec.*"
    // and "<prefix>.pos_embd".
    void declare(WeightLoader & L, const char * prefix);

    ggml_tensor * encode_state(ggml_context * C, ggml_tensor * state) const;

    ggml_tensor * encode_action(ggml_context * C, ggml_tensor * actions, ggml_tensor * tau,
                                int64_t embed_dim, int64_t horizon) const;

    ggml_tensor * decode(ggml_context * C, ggml_tensor * model_out) const;
};

}
