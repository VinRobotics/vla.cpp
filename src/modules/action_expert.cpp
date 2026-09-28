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

#include "backend.h"
#include "layers/linear.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace vla {

namespace {

bool read_slab(gguf_reader & g, const char * name, int64_t id, int64_t n, std::vector<float> & out) {
    const ggml_tensor * t = g.meta(name);
    const int64_t        tid = gguf_find_tensor(g.gctx, name);
    if (!t || tid < 0) {
        std::fprintf(stderr, "vla(%s): missing tensor %s\n", g.arch, name);
        return false;
    }
    if (id < 0 || id >= ggml_nelements(t)/n) {
        std::fprintf(stderr, "vla(%s): embodiment id %lld out of range for %s\n", g.arch, (long long) id, name);
        return false;
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_F16) {
        std::fprintf(stderr, "vla(%s): tensor %s unsupported type %d\n", g.arch, name, (int) t->type);
        return false;
    }
    const size_t es = ggml_type_size(t->type);
    std::vector<uint8_t> raw((size_t) n*es);
    if (vla_fseek64(g.fp, g.data_off+gguf_get_tensor_offset(g.gctx, tid)+(uint64_t) id*raw.size()) != 0 ||
        std::fread(raw.data(), 1, raw.size(), g.fp) != raw.size()) {
        std::fprintf(stderr, "vla(%s): read %s failed\n", g.arch, name);
        return false;
    }
    out.resize((size_t) n);
    if (t->type == GGML_TYPE_F32)
        std::memcpy(out.data(), raw.data(), raw.size());
    else if (t->type == GGML_TYPE_BF16)
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(raw.data()), out.data(), n);
    else
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(raw.data()), out.data(), n);
    return true;
}

}

ActionExpert::~ActionExpert() {
    if (buf)
        ggml_backend_buffer_free(buf);
    if (ctx)
        ggml_free(ctx);
}

bool ActionExpert::declare(WeightLoader & L, gguf_reader & g, ggml_backend_t backend, const char * prefix) {
    pos_embd = L.f32("%s.pos_embd", prefix);

    struct Proj { const char * name; ggml_tensor ** W; ggml_tensor ** b; };
    const Proj projs[] = {
        {"state_enc.l1", &se_l1W, &se_l1b}, {"state_enc.l2", &se_l2W, &se_l2b},
        {"act_enc.W1",   &ae_W1W, &ae_W1b}, {"act_enc.W2",   &ae_W2W, &ae_W2b}, {"act_enc.W3", &ae_W3W, &ae_W3b},
        {"act_dec.l1",   &ad_l1W, &ad_l1b}, {"act_dec.l2",   &ad_l2W, &ad_l2b},
    };

    ggml_init_params p = { 2*std::size(projs)*ggml_tensor_overhead(), nullptr, true };
    ctx = ggml_init(p);
    if (!ctx)
        return false;
    char wn[192], bn[192];
    for (const Proj & pr : projs) {
        std::snprintf(wn, sizeof(wn), "%s.%s.W", prefix, pr.name);
        std::snprintf(bn, sizeof(bn), "%s.%s.b", prefix, pr.name);
        const ggml_tensor * W = g.meta(wn);
        const ggml_tensor * b = g.meta(bn);
        if (!W || !b || ggml_n_dims(W) > 3 || ggml_n_dims(b) > 2 || b->ne[0] != W->ne[0] || b->ne[1] != W->ne[2]) {
            std::fprintf(stderr, "vla(%s): %s/%s missing or not a stacked [out,in,n] weight and [out,n] bias\n",
                         g.arch, wn, bn);
            return false;
        }
        *pr.W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W->ne[1], W->ne[0]);
        *pr.b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, W->ne[0]);
        ggml_set_name(*pr.W, wn);
        ggml_set_name(*pr.b, bn);
    }
    buf = alloc_weights(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "vla(%s): action expert buffer alloc failed\n", g.arch);
        return false;
    }

    std::vector<float> slab, t;
    for (const Proj & pr : projs) {
        const int64_t out = (*pr.W)->ne[1], in = (*pr.W)->ne[0];
        std::snprintf(wn, sizeof(wn), "%s.%s.W", prefix, pr.name);
        std::snprintf(bn, sizeof(bn), "%s.%s.b", prefix, pr.name);
        if (!read_slab(g, wn, embodiment_id, in*out, slab))
            return false;
        t.resize(slab.size());
        for (int64_t o=0; o<out; ++o)
            for (int64_t i=0; i<in; ++i)
                t[(size_t) (o*in+i)] = slab[(size_t) (i*out+o)];
        ggml_backend_tensor_set(*pr.W, t.data(), 0, ggml_nbytes(*pr.W));
        if (!read_slab(g, bn, embodiment_id, out, slab))
            return false;
        ggml_backend_tensor_set(*pr.b, slab.data(), 0, ggml_nbytes(*pr.b));
    }
    return true;
}

ggml_tensor * ActionExpert::encode_state(ggml_context * C, ggml_tensor * state) const {
    ggml_tensor * h = ggml_relu(C, linear(C, se_l1W, se_l1b, state));
    return linear(C, se_l2W, se_l2b, h);
}

ggml_tensor * ActionExpert::encode_action(ggml_context * C, ggml_tensor * actions, ggml_tensor * tau,
                                          int64_t embed_dim, int64_t horizon) const {
    ggml_tensor * a_emb = linear(C, ae_W1W, ae_W1b, actions);
    ggml_tensor * x_w2  = ggml_silu(C, linear(C, ae_W2W, ae_W2b, ggml_concat(C, a_emb, tau, 0)));
    ggml_tensor * pos   = ggml_view_2d(C, pos_embd, embed_dim, horizon, pos_embd->nb[1], 0);
    return ggml_add(C, linear(C, ae_W3W, ae_W3b, x_w2), pos);
}

ggml_tensor * ActionExpert::decode(ggml_context * C, ggml_tensor * model_out) const {
    ggml_tensor * h = ggml_relu(C, linear(C, ad_l1W, ad_l1b, model_out));
    return linear(C, ad_l2W, ad_l2b, h);
}

ggml_tensor * ActionExpert::denoise(ggml_context * C, const DitHead & dit, const FlowTimes & times, bool interleave, int64_t every2,
                                    ggml_tensor * state, ggml_tensor * future, ggml_tensor * txt, ggml_tensor * img,
                                    ggml_tensor * x0) const {
    const int64_t n_layers = dit.cfg.layers, AD = x0->ne[0], AH = x0->ne[1];
    ggml_tensor * state_features = encode_state(C, state);

    std::vector<ggml_tensor *> enc(n_layers, nullptr), Kc(n_layers, nullptr), Vc(n_layers, nullptr);
    for (int64_t i=0; i<n_layers; ++i) {
        if (interleave && (i%2 == 1))
            continue;
        enc[i] = (i%every2 == 0) ? txt : img;
        dit.kv(C, dit.blk[i], enc[i], &Kc[i], &Vc[i]);
    }

    const int64_t steps = (int64_t) times.tau.size();
    const float   dt    = 1.0f/(float) steps;
    ggml_tensor * actions = x0;
    for (int64_t s=0; s<steps; ++s) {
        ggml_tensor * tau = times.tau[(size_t) s];
        ggml_tensor * af  = encode_action(C, actions, tau, tau->ne[0], AH);
        ggml_tensor * sa  = future ? ggml_concat(C, state_features, future, 1) : state_features;
        ggml_tensor * hh  = ggml_concat(C, sa, af, 1);

        for (int64_t i=0; i<n_layers; ++i)
            hh = dit.block(C, dit.blk[i], hh, times.mod(C, s, i), enc[i], Kc[i], Vc[i]);

        ggml_tensor * pred = decode(C, dit.proj_out(C, hh, times.mod(C, s, n_layers)));
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
