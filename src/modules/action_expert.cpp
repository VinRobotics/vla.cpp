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

#include <cstdio>
#include <cstdlib>

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

ggml_tensor * ActionExpert::denoise(ggml_context * C, const DitHead & dit, bool interleave, int64_t every2,
                                    ggml_tensor * state, ggml_tensor * future, ggml_tensor * txt, ggml_tensor * img,
                                    ggml_tensor * x0, const std::vector<ggml_tensor *> & tau,
                                    const std::vector<ggml_tensor *> & tproj) const {
    const int64_t n_layers = dit.cfg.layers, AD = x0->ne[0], AH = x0->ne[1];
    ggml_tensor * state_features = encode_state(C, state);

    std::vector<ggml_tensor *> enc(n_layers, nullptr), Kc(n_layers, nullptr), Vc(n_layers, nullptr);
    for (int64_t i=0; i<n_layers; ++i) {
        if (interleave && (i%2 == 1))
            continue;
        enc[i] = (i%every2 == 0) ? txt : img;
        dit.kv(C, dit.blk[i], enc[i], &Kc[i], &Vc[i]);
    }

    const float dt = 1.0f/(float) tau.size();
    ggml_tensor * actions = x0;
    for (size_t s=0; s<tau.size(); ++s) {
        ggml_tensor * temb = dit.time_emb(C, tproj[s]);
        ggml_tensor * af   = encode_action(C, actions, tau[s], tau[s]->ne[0], AH);
        ggml_tensor * sa   = future ? ggml_concat(C, state_features, future, 1) : state_features;
        ggml_tensor * hh   = ggml_concat(C, sa, af, 1);

        for (int64_t i=0; i<n_layers; ++i)
            hh = dit.block(C, dit.blk[i], hh, temb, enc[i], Kc[i], Vc[i]);

        ggml_tensor * pred = decode(C, dit.proj_out(C, hh, temb));
        ggml_tensor * vel  = ggml_cont(C, ggml_view_2d(C, pred, AD, AH, pred->nb[1], (size_t)(pred->ne[1]-AH)*pred->nb[1]));
        actions = ggml_add(C, actions, ggml_scale(C, vel, dt));
    }
    return actions;
}

bool resolve_embodiment(const char * arch, const std::string & mapping, const char * default_tag,
                        int64_t max_id, int64_t & id) {
    auto lookup = [&](const char * key) -> long {
        const std::string k = std::string("\"")+key+"\"";
        size_t p = mapping.find(k);
        if (p == std::string::npos)
            return -1;
        p = mapping.find(':', p+k.size());
        if (p == std::string::npos)
            return -1;
        return std::strtol(mapping.c_str()+p+1, nullptr, 10);
    };

    if (default_tag) {
        const long d = lookup(default_tag);
        if (d >= 0)
            id = d;
    }
    if (const char * e = std::getenv("VLA_GR00T_EMBODIMENT")) {
        char * end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end && *end == '\0') {
            id = v;
        } else {
            const long t = lookup(e);
            if (t >= 0)
                id = t;
            else
                std::fprintf(stderr, "vla(%s): embodiment tag '%s' not in the GGUF embodiment mapping; using id %lld\n",
                             arch, e, (long long) id);
        }
    }
    if (id < 0 || id >= max_id) {
        std::fprintf(stderr, "vla(%s): embodiment id %lld out of range [0,%lld)\n", arch, (long long) id, (long long) max_id);
        return false;
    }
    return true;
}

}
