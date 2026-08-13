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

#include "modules/action_expert.h"

#include "layers/linear.h"

namespace vla {

void ActionExpert::declare(WeightLoader & L, const char * prefix) {
    se_l1W = L.f32("%s.state_enc.l1.W", prefix);
    se_l1b = L.f32("%s.state_enc.l1.b", prefix);
    se_l2W = L.f32("%s.state_enc.l2.W", prefix);
    se_l2b = L.f32("%s.state_enc.l2.b", prefix);

    ae_W1W = L.f32("%s.act_enc.W1.W", prefix);
    ae_W1b = L.f32("%s.act_enc.W1.b", prefix);
    ae_W2W = L.f32("%s.act_enc.W2.W", prefix);
    ae_W2b = L.f32("%s.act_enc.W2.b", prefix);
    ae_W3W = L.f32("%s.act_enc.W3.W", prefix);
    ae_W3b = L.f32("%s.act_enc.W3.b", prefix);

    ad_l1W = L.f32("%s.act_dec.l1.W", prefix);
    ad_l1b = L.f32("%s.act_dec.l1.b", prefix);
    ad_l2W = L.f32("%s.act_dec.l2.W", prefix);
    ad_l2b = L.f32("%s.act_dec.l2.b", prefix);

    pos_embd = L.f32("%s.pos_embd", prefix);
}

ggml_tensor * ActionExpert::encode_state(ggml_context * C, ggml_tensor * state) const {
    ggml_tensor * h = ggml_relu(C, cat_linear(C, se_l1W, se_l1b, embodiment_id, state));
    return cat_linear(C, se_l2W, se_l2b, embodiment_id, h);
}

ggml_tensor * ActionExpert::encode_action(ggml_context * C, ggml_tensor * actions, ggml_tensor * tau,
                                          int64_t embed_dim, int64_t horizon) const {
    ggml_tensor * a_emb = cat_linear(C, ae_W1W, ae_W1b, embodiment_id, actions);
    ggml_tensor * x_w2  = ggml_silu(C, cat_linear(C, ae_W2W, ae_W2b, embodiment_id, ggml_concat(C, a_emb, tau, 0)));
    ggml_tensor * pos   = ggml_view_2d(C, pos_embd, embed_dim, horizon, pos_embd->nb[1], 0);
    return ggml_add(C, cat_linear(C, ae_W3W, ae_W3b, embodiment_id, x_w2), pos);
}

ggml_tensor * ActionExpert::decode(ggml_context * C, ggml_tensor * model_out) const {
    ggml_tensor * h = ggml_relu(C, cat_linear(C, ad_l1W, ad_l1b, embodiment_id, model_out));
    return cat_linear(C, ad_l2W, ad_l2b, embodiment_id, h);
}

}
