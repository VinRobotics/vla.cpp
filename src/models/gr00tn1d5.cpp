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

#include "arch.h"
#include "options.h"
#include "backend.h"
#include "env_flag.h"
#include "gguf_reader.h"
#include "layers/embed.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "model.h"
#include "modules/action_expert.h"
#include "modules/dit_head.h"
#include "modules/encoder.h"
#include "modules/preprocess.h"
#include "modules/prompt.h"
#include "modules/qwen3_lm.h"
#include "modules/siglip_vit.h"
#include "scratch_ctx.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vla {

struct Gr00tN1d5ModelArch : public ModelArchBase {
    Gr00tN1d5ModelArch() : ModelArchBase(Arch::GR00T_N1_5) {}
    ~Gr00tN1d5ModelArch() override;

    // Opened once at load: reopening per predict re-parses the whole GGUF header.
    gguf_reader           io{"gr00tn1d5"};
    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;
    scratch_ctx           vision_scratch;

    struct MainKey {
        int64_t seq=-1, nsteps=-1;
        bool operator==(const MainKey & o) const {
            return seq==o.seq && nsteps==o.nsteps;
        }
    };
    struct MainIO {
        ggml_tensor *t_embeds=nullptr,*t_pos=nullptr,*t_lmmask=nullptr,*t_state=nullptr,*t_x0=nullptr,*actions=nullptr;
        std::vector<ggml_tensor*> t_tau, t_tproj;
    };
    graph_cache<MainKey, MainIO> main_graph;

    SigLipTower  vit;
    Qwen3LM      lm;
    EncStack     vlsa;
    ActionExpert aex;
    DitHead      dit;
    ggml_tensor *mm_W=nullptr, *mm_b=nullptr;
    ggml_tensor *vlln_w=nullptr, *vlln_b=nullptr;
    ggml_tensor *future_tokens=nullptr;

    int64_t vit_layers=27, vit_inter=4304, image_size=224, patch_size=14, n_img_tokens=256;
    int64_t lm_inter=6144, vocab=151680, image_token_index=151669;
    int64_t bb_embed_dim=2048, in_embed_dim=1536, dit_interleave=1, vlsa_layers=4;
    int64_t num_future=32, action_horizon=16, action_dim=32, max_state_dim=64;
    int64_t num_steps=4, num_buckets=1000, max_embodiments=32, max_seq_len=1024;
    float   vlln_eps=1e-5f;

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

bool load_config(const gguf_reader & g, Gr00tN1d5ModelArch & m, Config & cfg) {
    auto U  = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F  = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto fk = [&](const char * s) { static char b[64]; std::snprintf(b, sizeof(b), "gr00t_n1_5.%s", s); return b; };

    U(fk("vit_hidden"     ), m.vit.enc.cfg.hidden);
    U(fk("vit_layers"     ), m.vit_layers);
    U(fk("vit_heads"      ), m.vit.enc.cfg.heads);
    U(fk("vit_inter"      ), m.vit_inter);
    U(fk("image_size"     ), m.image_size);
    U(fk("patch_size"     ), m.patch_size);
    U(fk("n_img_tokens"   ), m.n_img_tokens);
    U(fk("lm_hidden"      ), m.lm.cfg.hidden);
    U(fk("lm_layers_used" ), m.lm.cfg.layers);
    U(fk("lm_q_heads"     ), m.lm.cfg.n_q);
    U(fk("lm_kv_heads"    ), m.lm.cfg.n_kv);
    U(fk("lm_head_dim"    ), m.lm.cfg.head_dim);
    U(fk("lm_inter"       ), m.lm_inter);
    U(fk("vocab_size"     ), m.vocab);
    U(fk("image_token_index"), m.image_token_index);
    U(fk("backbone_embedding_dim"), m.bb_embed_dim);
    U(fk("input_embedding_dim"   ), m.in_embed_dim);
    U(fk("dit_hidden"     ), m.dit.cfg.hidden);
    U(fk("dit_heads"      ), m.dit.cfg.heads);
    U(fk("dit_head_dim"   ), m.dit.cfg.head_dim);
    U(fk("dit_layers"     ), m.dit.cfg.layers);
    U(fk("dit_interleave" ), m.dit_interleave);
    U(fk("vlsa_layers"    ), m.vlsa_layers);
    U(fk("vlsa_heads"     ), m.vlsa.cfg.heads);
    U(fk("vlsa_head_dim"  ), m.vlsa.cfg.head_dim);
    U(fk("num_target_vision_tokens"), m.num_future);
    U(fk("action_horizon" ), m.action_horizon);
    U(fk("action_dim"     ), m.action_dim);
    U(fk("max_state_dim"  ), m.max_state_dim);
    U(fk("num_inference_timesteps"), m.num_steps);
    U(fk("num_timestep_buckets"   ), m.num_buckets);
    U(fk("max_num_embodiments"    ), m.max_embodiments);
    U(fk("max_seq_len"    ), m.max_seq_len);

    F(fk("vit_ln_eps"     ), m.vit.enc.cfg.ln_eps);
    F(fk("lm_rms_eps"     ), m.lm.cfg.rms_eps);
    F(fk("ln_eps"         ), m.dit.cfg.ln_eps);
    F(fk("norm_out_eps"   ), m.dit.cfg.norm_out_eps);
    F(fk("vlln_eps"       ), m.vlln_eps);

    if (g.has(fk("lm_rope_theta")))
        m.lm.cfg.rope.freq_base = (float) g.f64(fk("lm_rope_theta"));

    m.vit.enc.cfg.head_dim = m.vit.enc.cfg.hidden/m.vit.enc.cfg.heads;
    m.vlsa.cfg.hidden      = m.bb_embed_dim;
    m.vlsa.cfg.ln_eps      = m.dit.cfg.ln_eps;
    m.lm.cfg.rope.n_dims   = (int) m.lm.cfg.head_dim;

    m.aex.embodiment_id = 24;
    if (const char * e = std::getenv("VLA_GR00T_EMBODIMENT")) {
        char * end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end && *end == '\0') {
            m.aex.embodiment_id = (int64_t) v;
        } else {
            const std::string js  = g.str(fk("embodiment_tag_mapping"));
            const std::string key = std::string("\"")+e+"\":";
            const size_t p = js.find(key);
            if (p != std::string::npos)
                m.aex.embodiment_id = std::strtol(js.c_str()+p+key.size(), nullptr, 10);
            else std::fprintf(stderr, "vla(gr00tn1d5): embodiment tag '%s' not in embodiment_tag_mapping; using id %lld\n", e, (long long) m.aex.embodiment_id);
        }
    }
    if (m.aex.embodiment_id < 0 || m.aex.embodiment_id >= m.max_embodiments) {
        std::fprintf(stderr, "vla(gr00tn1d5): embodiment id %lld out of range [0,%lld)\n",
                     (long long) m.aex.embodiment_id, (long long) m.max_embodiments);
        return false;
    }

    cfg = Config{};
    cfg.n_img           = m.n_img_tokens;
    cfg.n_lang          = m.max_seq_len;
    cfg.n_state         = 1;
    cfg.n_suffix        = m.action_horizon;
    cfg.max_state_dim   = m.max_state_dim;
    cfg.max_action_dim  = m.action_dim;
    cfg.real_state_dim  = m.max_state_dim;
    cfg.real_action_dim = m.action_dim;
    cfg.hidden          = m.lm.cfg.hidden;
    cfg.n_q_heads       = m.lm.cfg.n_q;
    cfg.n_kv_heads      = m.lm.cfg.n_kv;
    cfg.head_dim        = m.lm.cfg.head_dim;
    cfg.n_layers        = m.lm.cfg.layers;
    cfg.num_steps       = (int) m.num_steps;
    cfg.rms_eps         = m.lm.cfg.rms_eps;
    cfg.rope_n_dims     = (int) m.lm.cfg.head_dim;
    cfg.rope_mode       = GGML_ROPE_TYPE_NEOX;
    cfg.rope_freq_base  = m.lm.cfg.rope.freq_base;
    // Raw output: this arch expects the client to apply the dataset statistics
    // (see the --stats-json flag in eval/client).
    cfg.denormalized    = false;
    cfg.norm_eps        = 1e-8f;
    return true;
}

}

Gr00tN1d5ModelArch::~Gr00tN1d5ModelArch() {
    if (weight_buf)
        ggml_backend_buffer_free(weight_buf);
    if (ctx_weights)
        ggml_free(ctx_weights);
    if (backend)
        ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> gr00t_n1_5_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string&,
                                                 const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(gr00tn1d5): note - mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<Gr00tN1d5ModelArch>();
    m->matmul_type           = opts.weight_dtype.value_or(GGML_TYPE_BF16);
    m->lm.cfg.rope.freq_base = 1000000.0f;

    if (!m->io.open(ckpt_path))
        return nullptr;
    gguf_reader & g = m->io;
    if (!g.has("gr00t_n1_5.architecture")) {
        std::fprintf(stderr, "vla(gr00tn1d5): %s is not a gr00t_n1_5 GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, *m, m->cfg))
        return nullptr;

    std::printf("vla(gr00tn1d5): vit=%lldd×%lldL×%lldh n_img_tok=%lld  lm=Qwen3 %lldd×%lldL (%lldq/%lldkv×%lld)  "
                "dit=%lldL×%lldh×%lld(inner %lld) interleave=%lld  vlsa=%lldL×%lldh×%lld  in_emb=%lld  horizon=%lld action_dim=%lld N_steps=%lld  embodiment=%lld  resident=%s\n",
                (long long) m->vit.enc.cfg.hidden, (long long) m->vit_layers, (long long) m->vit.enc.cfg.heads, (long long) m->n_img_tokens,
                (long long) m->lm.cfg.hidden, (long long) m->lm.cfg.layers, (long long) m->lm.cfg.n_q, (long long) m->lm.cfg.n_kv, (long long) m->lm.cfg.head_dim,
                (long long) m->dit.cfg.layers, (long long) m->dit.cfg.heads, (long long) m->dit.cfg.head_dim, (long long) m->dit.cfg.hidden, (long long) m->dit_interleave,
                (long long) m->vlsa_layers, (long long) m->vlsa.cfg.heads, (long long) m->vlsa.cfg.head_dim, (long long) m->in_embed_dim,
                (long long) m->action_horizon, (long long) m->action_dim, (long long) m->num_steps, (long long) m->aex.embodiment_id,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    const Backend b = backend_init("vla(gr00tn1d5)", m->n_threads);
    if (!b.handle)
        return nullptr;
    m->backend = b.handle;

    ggml_init_params wp = { (size_t) 32*1024*1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) {
        std::fprintf(stderr, "vla(gr00tn1d5): ggml_init(ctx_weights) failed\n");
        return nullptr;
    }

    WeightLoader L("gr00tn1d5", g, m->ctx_weights, m->matmul_type);

    m->vit.declare(L, "vit", m->vit_layers);
    m->mm_W = L.gemm("mm.fc.weight");
    m->mm_b = L.f32 ("mm.fc.bias");

    m->lm.declare(L, "vlm");

    m->vlln_w = L.f32("aex.vlln.weight");
    m->vlln_b = L.f32("aex.vlln.bias");
    m->vlsa.declare(L, "aex.vlsa", m->vlsa_layers, EncNames{"norm1", "norm3", "ff0", "ff2"});

    m->aex.declare(L, "aex");
    m->future_tokens = L.f32("aex.future_tokens");
    m->dit.declare(L, "aex.dit");

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;

    std::printf("vla(gr00tn1d5): weights resident in %.2f GiB (%s) - incl. SigLIP vision tower; embodiment id %lld\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0),
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16", (long long) m->aex.embodiment_id);
    return m;
}

std::vector<float> Gr00tN1d5ModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    const int64_t H   = lm.cfg.hidden;
    const int64_t K   = n_img_tokens;
    const int64_t E   = in_embed_dim;
    const int64_t AD  = action_dim;
    const int64_t AH  = action_horizon;
    const int64_t Nsa = 1+num_future+AH;

    int64_t n_views = 0;
    std::vector<float> img_emb_host;
    const float * img_emb_ptr = nullptr;

    if (in.precomputed_img_emb && in.n_img_views > 0) {
        n_views     = in.n_img_views;
        img_emb_ptr = in.precomputed_img_emb;
    } else if (in.images && in.n_images > 0) {
        n_views = in.n_images;
        img_emb_host.assign((size_t) n_views*K*H, 0.0f);

        ggml_context * VC = vision_scratch.reset((size_t) 64*1024*1024);
        if (!VC) { std::fprintf(stderr, "vla(gr00tn1d5): ggml_init(vision ctx) failed\n"); return {}; }

        const int64_t grid = image_size/patch_size;
        ggml_tensor * t_px = ggml_new_tensor_3d(VC, GGML_TYPE_F32, image_size, image_size, 3);
        ggml_set_input(t_px);

        ggml_tensor * h       = vit.build(VC, vit.embed_conv(VC, t_px, patch_size, grid), K);
        ggml_tensor * vit_emb = linear(VC, mm_W, mm_b, h);
        ggml_set_output(vit_emb);

        ggml_cgraph * vg = ggml_new_graph_custom(VC, 8192, false);
        ggml_build_forward_expand(vg, vit_emb);
        if (!vision_scratch.alloc(backend, vg)) { std::fprintf(stderr, "vla(gr00tn1d5): vision gallocr alloc failed\n"); return {}; }

        const auto tv0 = std::chrono::steady_clock::now();
        std::vector<float> chw;
        for (int64_t v = 0; v < n_views; ++v) {
            if (!preprocess_image_chw("gr00tn1d5", in.images[v], image_size, chw)) return {};
            ggml_backend_tensor_set(t_px, chw.data(), 0, ggml_nbytes(t_px));
            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(gr00tn1d5): vision compute failed\n");
                return {};
            }
            ggml_backend_tensor_get(vit_emb, img_emb_host.data()+v*K*H, 0, ggml_nbytes(vit_emb));
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-tv0).count();
        img_emb_ptr = img_emb_host.data();
    } else {
        std::fprintf(stderr, "vla(gr00tn1d5): no images and no precomputed_img_emb in the request\n");
        return {};
    }
    const int64_t n_img = n_views*K;

    Prompt prompt;
    if (!build_prompt("gr00tn1d5", in, n_img, (int32_t) image_token_index, max_seq_len, prompt)) return {};
    const int64_t SEQ = prompt.len();

    std::vector<float> inputs_embeds;
    if (!fetch_embeds("gr00tn1d5", io, prompt, img_emb_ptr, H, inputs_embeds)) return {};

    std::vector<float> x_init;
    init_noise(in, (size_t) AH*AD, x_init);

    const MainKey mkey{ SEQ, num_steps };
    const bool built = main_graph.ensure(backend, mkey, (size_t) 128*1024*1024,
                                         [&](ggml_context * C, MainIO & gio) -> ggml_cgraph * {
        ggml_tensor * t_embeds = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);           ggml_set_input(t_embeds);
        ggml_tensor * t_pos    = ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ);              ggml_set_input(t_pos);
        ggml_tensor * t_lmmask = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ);         ggml_set_input(t_lmmask);
        ggml_tensor * t_state  = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_state_dim, 1); ggml_set_input(t_state);
        ggml_tensor * t_x0     = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH);           ggml_set_input(t_x0);

        std::vector<ggml_tensor *> t_tau(num_steps), t_tproj(num_steps);
        for (int64_t s = 0; s < num_steps; ++s) {
            t_tau[s]   = ggml_new_tensor_2d(C, GGML_TYPE_F32, E, AH); ggml_set_input(t_tau[s]);
            t_tproj[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, 256);   ggml_set_input(t_tproj[s]);
        }

        ggml_tensor * eagle   = lm.build(C, t_embeds, t_pos, t_lmmask, SEQ);
        ggml_tensor * vl      = layer_norm(C, eagle, vlln_w, vlln_b, vlln_eps);
        ggml_tensor * vl_embs = vlsa.build(C, vl, SEQ);

        ggml_tensor * state_features = aex.encode_state(C, t_state);

        std::vector<ggml_tensor *> Kc(dit.cfg.layers, nullptr), Vc(dit.cfg.layers, nullptr);
        for (int64_t i = 0; i < dit.cfg.layers; ++i) {
            if (dit_interleave && (i%2 == 1))
                continue;
            dit.kv(C, dit.blk[i], vl_embs, &Kc[i], &Vc[i]);
        }

        const float dt = 1.0f/(float) num_steps;
        ggml_tensor * actions = t_x0;
        for (int64_t s = 0; s < num_steps; ++s) {
            ggml_tensor * temb = dit.time_emb(C, t_tproj[s]);
            ggml_tensor * af   = aex.encode_action(C, actions, t_tau[s], E, AH);
            ggml_tensor * hh   = ggml_concat(C, ggml_concat(C, state_features, future_tokens, 1), af, 1);

            for (int64_t i = 0; i < dit.cfg.layers; ++i) {
                ggml_tensor * enc = (dit_interleave && (i%2 == 1)) ? nullptr : vl_embs;
                hh = dit.block(C, dit.blk[i], hh, temb, enc, Kc[i], Vc[i]);
            }

            ggml_tensor * pred = aex.decode(C, dit.proj_out(C, hh, temb));
            ggml_tensor * vel  = ggml_cont(C, ggml_view_2d(C, pred, AD, AH, pred->nb[1], (size_t)(Nsa-AH)*pred->nb[1]));
            actions = ggml_add(C, actions, ggml_scale(C, vel, dt));
        }
        ggml_set_name(actions, "action_pred");
        ggml_set_output(actions);

        gio.t_embeds=t_embeds; gio.t_pos=t_pos; gio.t_lmmask=t_lmmask; gio.t_state=t_state;
        gio.t_x0=t_x0; gio.t_tau=t_tau; gio.t_tproj=t_tproj; gio.actions=actions;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf, actions);
        return gf;
    });
    if (!built) { std::fprintf(stderr, "vla(gr00tn1d5): main graph build failed\n"); return {}; }

    MainIO & gio = main_graph.io();

    ggml_backend_tensor_set(gio.t_embeds, inputs_embeds.data(), 0, ggml_nbytes(gio.t_embeds));

    std::vector<int32_t> pp(SEQ);
    for (int64_t i = 0; i < SEQ; ++i)
        pp[i] = (int32_t) i;
    ggml_backend_tensor_set(gio.t_pos, pp.data(), 0, ggml_nbytes(gio.t_pos));

    std::vector<float> mask;
    build_causal_mask(SEQ, mask);
    ggml_backend_tensor_set(gio.t_lmmask, mask.data(), 0, ggml_nbytes(gio.t_lmmask));

    std::vector<float> st(max_state_dim, 0.0f);
    for (int64_t i = 0; i < max_state_dim; ++i)
        st[i] = in.state ? in.state[i] : 0.0f;
    ggml_backend_tensor_set(gio.t_state, st.data(), 0, ggml_nbytes(gio.t_state));

    ggml_backend_tensor_set(gio.t_x0, x_init.data(), 0, ggml_nbytes(gio.t_x0));

    for (int64_t s = 0; s < num_steps; ++s) {
        const int64_t bucket = (int64_t) ((double) s/(double) num_steps*(double) num_buckets);
        std::vector<float> tau, tpr;
        action_sinusoid(bucket, E, AH, tau);
        timesteps_proj(bucket, tpr);
        ggml_backend_tensor_set(gio.t_tau[s],   tau.data(), 0, ggml_nbytes(gio.t_tau[s]));
        ggml_backend_tensor_set(gio.t_tproj[s], tpr.data(), 0, ggml_nbytes(gio.t_tproj[s]));
    }

    const auto tc0 = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, main_graph.graph());
    const auto tc1 = std::chrono::steady_clock::now();
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(gr00tn1d5): graph compute failed (%d)\n", (int) status);
        return {};
    }
    stats.ms_inference = std::chrono::duration<float, std::milli>(tc1-tc0).count();

    std::vector<float> out((size_t) AH*AD);
    ggml_backend_tensor_get(gio.actions, out.data(), 0, out.size()*sizeof(float));
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-t0).count();
    return out;
}

}
