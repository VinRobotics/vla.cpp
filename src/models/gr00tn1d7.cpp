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
#include "layers/norm.h"
#include "modules/action_expert.h"
#include "modules/dit_head.h"
#include "modules/encoder.h"
#include "modules/prompt.h"
#include "modules/qwen3_lm.h"
#include "options.h"
#include "model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "backend.h"
#include "env_flag.h"
#include "gguf_reader.h"
#include "scratch_ctx.h"
#include "layers/embed.h"
#include "modules/qwen3vl_vit.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace vla {

struct Gr00tN1d7ModelArch : public ModelArchBase {
    Gr00tN1d7ModelArch() : ModelArchBase(Arch::GR00T_N1_7) {}
    ~Gr00tN1d7ModelArch() override;

    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    graph_cache<int, VitIO> vision_graph;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;

    int64_t lm_hidden=2048, lm_layers=16, n_q=16, n_kv=8, lm_head_dim=128, lm_inter=6144, vocab=151936, image_token_index=151655;
    int64_t vlsa_layers=4, vlsa_heads=32, vlsa_head_dim=64, vlsa_ff_inner=8192;
    int64_t bb_embed_dim=2048, in_embed_dim=1536, dit_hidden=1536, dit_heads=32, dit_head_dim=48, dit_layers=32, dit_interleave=1, attend_text_every_n=2;
    int64_t action_horizon=40, action_dim=132, max_state_dim=132;
    int64_t num_steps=4, num_buckets=1000, max_embodiments=32, max_seq_len=1024;
    float   lm_rms_eps=1e-6f, lm_rope_base=5000000.0f;
    float   vlln_eps=1e-5f, vlsa_ln_eps=1e-5f, ln_eps=1e-5f, norm_out_eps=1e-6f;

    Qwen3VLTower vit;
    Qwen3LM      lm;
    EncStack     vlsa;
    ActionExpert aex;
    DitHead      dit;
    ggml_tensor *vlln_w=nullptr,*vlln_b=nullptr;

    FlowTimes          times;
    std::vector<float> c_mask; int64_t c_mask_seq = -1;
    gguf_reader        io{"gr00tn1d7"};

    struct MainKey {
        int64_t seq=-1, n_img=-1, seq_txt=-1, nsteps=-1; bool deepstack=false;
        bool operator==(const MainKey & o) const {
            return seq==o.seq && n_img==o.n_img && seq_txt==o.seq_txt &&
                   nsteps==o.nsteps && deepstack==o.deepstack;
        }
    };
    struct MainIO {
        ggml_tensor *t_embeds=nullptr,*t_pos=nullptr,*t_lmmask=nullptr,*t_state=nullptr,*t_x0=nullptr;
        ggml_tensor *t_ds[3]={nullptr,nullptr,nullptr};
        ggml_tensor *t_img_idx=nullptr,*t_txt_idx=nullptr,*actions=nullptr;
    };
    graph_cache<MainKey, MainIO> mg;

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

bool load_config(const gguf_reader & g, const Options & opts, Gr00tN1d7ModelArch & m, Config & cfg) {
    auto U = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto fk = [&](const char * s) { thread_local char b[64]; std::snprintf(b, sizeof(b), "gr00t_n1_7.%s", s); return b; };
    if (!m.vit.load_config("gr00tn1d7", g, "gr00t_n1_7"))
        return false;
    U(fk("lm_hidden"), m.lm_hidden); U(fk("lm_layers_used"), m.lm_layers); U(fk("lm_q_heads"), m.n_q); U(fk("lm_kv_heads"), m.n_kv);
    U(fk("lm_head_dim"), m.lm_head_dim); U(fk("lm_inter"), m.lm_inter); U(fk("vocab_size"), m.vocab); U(fk("image_token_index"), m.image_token_index);
    U(fk("vlsa_layers"), m.vlsa_layers); U(fk("vlsa_heads"), m.vlsa_heads); U(fk("vlsa_head_dim"), m.vlsa_head_dim); U(fk("vlsa_ff_inner"), m.vlsa_ff_inner);
    U(fk("backbone_embedding_dim"), m.bb_embed_dim); U(fk("input_embedding_dim"), m.in_embed_dim);
    U(fk("dit_hidden"), m.dit_hidden); U(fk("dit_heads"), m.dit_heads); U(fk("dit_head_dim"), m.dit_head_dim); U(fk("dit_layers"), m.dit_layers); U(fk("dit_interleave"), m.dit_interleave);
    U(fk("attend_text_every_n_blocks"), m.attend_text_every_n);
    U(fk("action_horizon"), m.action_horizon); U(fk("action_dim"), m.action_dim); U(fk("max_state_dim"), m.max_state_dim);
    U(fk("num_inference_timesteps"), m.num_steps); U(fk("num_timestep_buckets"), m.num_buckets); U(fk("max_num_embodiments"), m.max_embodiments); U(fk("max_seq_len"), m.max_seq_len);

    if (m.attend_text_every_n <= 0) {
        std::fprintf(stderr, "vla(gr00tn1d7): attend_text_every_n_blocks %lld must be positive\n", (long long) m.attend_text_every_n);
        return false;
    }

    if (!resolve_num_steps("gr00tn1d7", opts, m.num_steps))
        return false;
    F(fk("lm_rms_eps"), m.lm_rms_eps); F(fk("ln_eps"), m.ln_eps); F(fk("norm_out_eps"), m.norm_out_eps);
    F(fk("vlln_eps"), m.vlln_eps); F(fk("vlsa_ln_eps"), m.vlsa_ln_eps);
    if (g.has(fk("lm_rope_theta")))
        m.lm_rope_base = (float) g.f64(fk("lm_rope_theta"));

    m.lm.cfg.hidden      = m.lm_hidden;
    m.lm.cfg.layers      = m.lm_layers;
    m.lm.cfg.n_q         = m.n_q;
    m.lm.cfg.n_kv        = m.n_kv;
    m.lm.cfg.head_dim    = m.lm_head_dim;
    m.lm.cfg.inter       = m.lm_inter;
    m.lm.cfg.rms_eps     = m.lm_rms_eps;
    m.lm.cfg.flash_attn  = flash_attn_enabled();
    m.lm.cfg.rope.type   = GGML_ROPE_TYPE_IMROPE;
    m.lm.cfg.rope.n_dims = (int) m.lm_head_dim;
    m.lm.cfg.rope.freq_base  = m.lm_rope_base;
    m.lm.cfg.rope.sections[0]= 24;
    m.lm.cfg.rope.sections[1]= 20;
    m.lm.cfg.rope.sections[2]= 20;
    m.lm.cfg.rope.sections[3]= 0;

    m.vlsa.cfg.hidden     = m.bb_embed_dim;
    m.vlsa.cfg.heads      = m.vlsa_heads;
    m.vlsa.cfg.head_dim   = m.vlsa_head_dim;
    m.vlsa.cfg.ln_eps     = m.vlsa_ln_eps;
    m.vlsa.cfg.flash_attn = flash_attn_enabled();

    m.dit.cfg.hidden       = m.dit_hidden;
    m.dit.cfg.heads        = m.dit_heads;
    m.dit.cfg.head_dim     = m.dit_head_dim;
    m.dit.cfg.layers       = m.dit_layers;
    m.dit.cfg.ln_eps       = m.ln_eps;
    m.dit.cfg.norm_out_eps = m.norm_out_eps;

    m.aex.embodiment_id = 2;
    if (!resolve_embodiment("gr00tn1d7", g.str(fk("embodiment_id_mapping")), "libero_sim", m.max_embodiments, m.aex.embodiment_id))
        return false;

    cfg = Config{};
    cfg.n_img = m.vit.n_tokens();
    cfg.n_lang = m.max_seq_len; cfg.n_state = 1;
    cfg.n_suffix = m.action_horizon; cfg.max_state_dim = m.max_state_dim; cfg.max_action_dim = m.action_dim;
    cfg.real_state_dim = m.max_state_dim; cfg.real_action_dim = m.action_dim;
    cfg.hidden = m.lm_hidden; cfg.n_q_heads = m.n_q; cfg.n_kv_heads = m.n_kv; cfg.head_dim = m.lm_head_dim; cfg.n_layers = m.lm_layers;
    cfg.num_steps = (int) m.num_steps; cfg.rms_eps = m.lm_rms_eps;
    cfg.rope_n_dims = (int) m.lm_head_dim; cfg.rope_mode = GGML_ROPE_TYPE_NEOX; cfg.rope_freq_base = m.lm_rope_base;
    // Raw output: this arch expects the client to apply the dataset statistics
    // (see the --stats-json flag in eval/client).
    cfg.denormalized = false;
    cfg.norm_eps = 1e-8f;
    return true;
}

}

Gr00tN1d7ModelArch::~Gr00tN1d7ModelArch() {
    mg.release();
    if (weight_buf)
        ggml_backend_buffer_free(weight_buf);
    if (ctx_weights)
        ggml_free(ctx_weights);
    if (backend)
        ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> gr00t_n1_7_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string&,
                                                 const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(gr00tn1d7): note - mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<Gr00tN1d7ModelArch>();
    m->matmul_type = opts.weight_dtype.value_or(vla::default_weight_dtype(GGML_TYPE_BF16));

    if (!m->io.open(ckpt_path))
        return nullptr;
    gguf_reader & g = m->io;
    if (!g.has("gr00t_n1_7.architecture")) {
        std::fprintf(stderr, "vla(gr00tn1d7): %s is not a gr00t_n1_7 GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, opts, *m, m->cfg))
        return nullptr;
    FoldQuantSpec fq = foldquant_parse(g, "gr00t_n1_7");
    std::printf("vla(gr00tn1d7): vit=Qwen3-VL %lldd×%lldL×%lldh (Conv3d patch %lld², temporal %lld; learned pos %lld + 2D rope; deepstack@{%lld,%lld,%lld}; merge÷%lld)  "
                "lm=Qwen3-VL %lldd×%lldL (%lldq/%lldkv×%lld, θ=%g)  vlsa=%lldL×%lldh×%lld  dit=AlternateVLDiT %lldL×%lldh×%lld(inner %lld) attend_text_every_n=%lld  "
                "in_emb=%lld  horizon=%lld action_dim=%lld max_state=%lld N_steps=%lld  embodiment=%lld  resident=%s\n",
                (long long) m->vit.hidden, (long long) m->vit.layers, (long long) m->vit.heads, (long long) m->vit.patch, (long long) m->vit.temporal,
                (long long) m->vit.num_pos, (long long) m->vit.deepstack_idx[0], (long long) m->vit.deepstack_idx[1], (long long) m->vit.deepstack_idx[2], (long long) m->vit.merge,
                (long long) m->lm_hidden, (long long) m->lm_layers, (long long) m->n_q, (long long) m->n_kv, (long long) m->lm_head_dim, (double) m->lm_rope_base,
                (long long) m->vlsa_layers, (long long) m->vlsa_heads, (long long) m->vlsa_head_dim,
                (long long) m->dit_layers, (long long) m->dit_heads, (long long) m->dit_head_dim, (long long) m->dit_hidden, (long long) m->attend_text_every_n, (long long) m->in_embed_dim,
                (long long) m->action_horizon, (long long) m->action_dim, (long long) m->max_state_dim, (long long) m->num_steps, (long long) m->aex.embodiment_id,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    {
        const Backend b = backend_init("vla(gr00tn1d7)", m->n_threads);
        if (!b.handle) {
            return nullptr;
        }
        m->backend = b.handle;
        if (!foldquant_check_backend("vla(gr00tn1d7)", b, fq, opts.weight_dtype.has_value()))
            return nullptr;
    }

    ggml_init_params wp = { (size_t) 32*1024*1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) {
        std::fprintf(stderr, "vla(gr00tn1d7): ggml_init(ctx_weights) failed\n");
        return nullptr;
    }

    WeightLoader L("gr00tn1d7", g, m->ctx_weights, m->matmul_type);

    m->vit.declare(L, "vit");
    m->lm.declare(L, "vlm", fq.present ? &fq.llm : nullptr);

    m->vlln_w = L.f32("aex.vlln.weight");
    m->vlln_b = L.f32("aex.vlln.bias");
    m->vlsa.declare(L, "aex.vlsa", m->vlsa_layers, EncNames{"norm1", "norm3", "ff0", "ff2"});

    if (!m->aex.declare(L, g, m->backend, "aex"))
        return nullptr;
    m->dit.declare(L, "aex.dit", true, m->dit_interleave != 0, nullptr, fq.present ? &fq.action : nullptr);
    if (fq.present) {
        // Execution order of the FoldQuant GEMMs, so each one can prefetch the
        // next site's weights (cross-attention K/V run before the step loop and
        // are left out of the chain).
        std::vector<FqLinear *> order;
        for (auto & b : m->lm.blk)
            for (FqLinear * s : {&b.fq_q, &b.fq_k, &b.fq_v, &b.fq_o, &b.fq_gate, &b.fq_up, &b.fq_down}) order.push_back(s);
        for (auto & b : m->dit.blk)
            for (FqLinear * s : {&b.fq_qkv, &b.fq_q, &b.fq_o, &b.fq_ff0, &b.fq_ff2}) order.push_back(s);
        fq_link_prefetch(order);
    }

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;
    if (!m->times.build("gr00tn1d7", m->backend, m->dit, m->num_steps, m->num_buckets, m->in_embed_dim, m->action_horizon))
        return nullptr;

    std::printf("vla(gr00tn1d7): QKV-fused DiT (self Wqkv / cross Wkv)\n");
    std::printf("vla(gr00tn1d7): weights resident in %.2f GiB (%s) - incl. Qwen3-VL vision tower + deepstack + vl_self_attention; embodiment id %lld\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0),
                dtype_name(m->matmul_type), (long long) m->aex.embodiment_id);
    if (!m->vit.build_caches("gr00tn1d7", m->io))
        return nullptr;
    return m;
}

std::vector<float> Gr00tN1d7ModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    const int64_t H = lm_hidden;
    const int64_t K = vit.n_tokens();
    const int64_t AD = action_dim, AH = action_horizon;
    const bool    do_dump = (std::getenv("VLA_GR00T_N17_DUMP") != nullptr);

    int64_t n_views = 0;
    std::vector<float> img_emb_host, ds_host[3];
    const float * img_emb_ptr = nullptr;
    if (in.precomputed_img_emb && in.n_img_views > 0) {
        n_views = in.n_img_views; img_emb_ptr = in.precomputed_img_emb;
    } else if (in.images && in.n_images > 0) {
        n_views = in.n_images;
        const auto tv0 = std::chrono::steady_clock::now();
        const bool vok = vit.encode("gr00tn1d7", backend, vision_graph, in.images, n_views, nullptr, img_emb_host, ds_host);
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-tv0).count();
        if (!vok) return {};
        img_emb_ptr = img_emb_host.data();
    } else {
        std::fprintf(stderr, "vla(gr00tn1d7): no images and no precomputed_img_emb in the request\n"); return {};
    }
    const int64_t n_img = n_views * K;

    Prompt prompt;
    if (!build_prompt("gr00tn1d7", in, n_img, (int32_t) image_token_index, max_seq_len, prompt)) return {};
    const int64_t SEQ = prompt.len(), SEQ_TXT = prompt.n_text();

    std::vector<float> inputs_embeds;
    if (!fetch_embeds(io, prompt, img_emb_ptr, H, inputs_embeds)) return {};

    std::vector<int32_t> pp;
    if (!mrope_positions("gr00tn1d7", prompt.ids, (int32_t) image_token_index, vit.grid()/vit.merge, pp)) return {};

    std::vector<std::vector<float>> ds_pad(3);
    const bool inject_deepstack = !img_emb_host.empty();
    if (inject_deepstack) for (int j=0; j<3; ++j) {
        ds_pad[j].assign((size_t) SEQ * H, 0.0f);
        for (int64_t k=0; k<n_img; ++k) {
            std::memcpy(ds_pad[j].data()+(size_t) prompt.image_pos[k]*H,
                        ds_host[j].data()+(size_t) k * H, H * sizeof(float));
        }
    }

    std::vector<float> x_init;
    init_noise(in, (size_t) AH*AD, x_init);

    // On by default: 16% faster, bit-identical. Set VLA_GR00T_GRAPH_CACHE=0 to opt out.
    // Dumping adds graph outputs, so it always rebuilds.
    const char * gc = std::getenv("VLA_GR00T_GRAPH_CACHE");
    const bool use_cache = (!gc || std::strcmp(gc, "0") != 0) && !do_dump;
    if (!use_cache)
        mg.release();

    ggml_tensor * eagle = nullptr, * vl_embs = nullptr;
    std::vector<ggml_tensor*> lm_h_dump, vlsa_dump;
    const MainKey mkey{ SEQ, n_img, SEQ_TXT, num_steps, inject_deepstack };
    const size_t main_nodes = 65536 + (size_t) num_steps*64*(dit_layers+1);
    const bool built = mg.ensure(backend, mkey, main_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(main_nodes, false),
                                 [&](ggml_context * C, MainIO & gio) -> ggml_cgraph * {
    ggml_tensor * t_embeds = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);          ggml_set_input(t_embeds);
    ggml_tensor * t_pos    = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4*SEQ);         ggml_set_input(t_pos);
    ggml_tensor * t_lmmask = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ);        ggml_set_input(t_lmmask);
    ggml_tensor * t_state  = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_state_dim, 1);ggml_set_input(t_state);
    ggml_tensor * t_x0     = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH);          ggml_set_input(t_x0);
    ggml_tensor * t_ds[3] = {nullptr,nullptr,nullptr};
    if (inject_deepstack) for (int j=0; j<3; ++j) {
        t_ds[j] = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);
        ggml_set_input(t_ds[j]);
    }

    ggml_tensor * t_img_idx = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_img);    ggml_set_input(t_img_idx);
    ggml_tensor * t_txt_idx = (SEQ_TXT > 0)
        ? ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ_TXT) : nullptr;
    if (t_txt_idx)
        ggml_set_input(t_txt_idx);

    ggml_tensor * h = t_embeds;
    for (int64_t i=0; i<lm_layers; ++i) {
        h = lm.block(C, lm.blk[i], h, t_pos, t_lmmask, SEQ);
        if (inject_deepstack && i < 3)
            h = ggml_add(C, h, t_ds[i]);
        if (do_dump) {
            ggml_set_output(h);
            lm_h_dump.push_back(h);
        }
    }

    eagle = h;
    ggml_set_name(eagle, "eagle");

    vl_embs = layer_norm(C, eagle, vlln_w, vlln_b, vlln_eps);
    if (do_dump) {
        ggml_set_output(vl_embs);
        vlsa_dump.push_back(vl_embs);
    }
    for (int64_t i=0; i<vlsa_layers; ++i) {
        vl_embs = vlsa.block(C, vlsa.blk[i], vl_embs, SEQ);
        if (do_dump) {
            ggml_set_output(vl_embs);
            vlsa_dump.push_back(vl_embs);
        }
    }
    ggml_set_name(vl_embs, "vl_embs");

    ggml_tensor * vl_img = ggml_get_rows(C, vl_embs, t_img_idx);
    ggml_tensor * vl_txt = (t_txt_idx ? ggml_get_rows(C, vl_embs, t_txt_idx) : vl_img);

    ggml_tensor * actions = aex.denoise(C, dit, times, dit_interleave != 0, 2*attend_text_every_n, t_state, nullptr,
                                        vl_txt, vl_img, t_x0);
    ggml_set_name(actions, "action_pred"); ggml_set_output(actions);

    gio.t_embeds=t_embeds; gio.t_pos=t_pos; gio.t_lmmask=t_lmmask; gio.t_state=t_state; gio.t_x0=t_x0;
    gio.t_ds[0]=t_ds[0]; gio.t_ds[1]=t_ds[1]; gio.t_ds[2]=t_ds[2];
    gio.t_img_idx=t_img_idx; gio.t_txt_idx=t_txt_idx; gio.actions=actions;

    ggml_cgraph * gf = ggml_new_graph_custom(C, main_nodes, false);
    ggml_build_forward_expand(gf, actions);
    return gf;
    });
    if (!built) { std::fprintf(stderr, "vla(gr00tn1d7): main graph build failed\n"); return {}; }

    MainIO & gio = mg.io();
    ggml_cgraph * gf = mg.graph();

    ggml_backend_tensor_set(gio.t_embeds, inputs_embeds.data(), 0, ggml_nbytes(gio.t_embeds));
    ggml_backend_tensor_set(gio.t_pos, pp.data(), 0, ggml_nbytes(gio.t_pos));
    if (c_mask_seq != SEQ) {
        build_causal_mask(SEQ, c_mask);
        c_mask_seq = SEQ;
    }
    ggml_backend_tensor_set(gio.t_lmmask, c_mask.data(), 0, ggml_nbytes(gio.t_lmmask));
    {
        std::vector<float> st(max_state_dim, 0.0f);
        for (int64_t i=0; i<max_state_dim; ++i)
            st[i] = in.state ? in.state[i] : 0.0f;
        ggml_backend_tensor_set(gio.t_state, st.data(), 0, ggml_nbytes(gio.t_state));
    }
    ggml_backend_tensor_set(gio.t_x0, x_init.data(), 0, ggml_nbytes(gio.t_x0));
    if (inject_deepstack) for (int j=0; j<3; ++j) ggml_backend_tensor_set(gio.t_ds[j], ds_pad[j].data(), 0, ggml_nbytes(gio.t_ds[j]));
    ggml_backend_tensor_set(gio.t_img_idx, prompt.image_pos.data(), 0, ggml_nbytes(gio.t_img_idx));
    if (gio.t_txt_idx)
        ggml_backend_tensor_set(gio.t_txt_idx, prompt.text_pos.data(), 0, ggml_nbytes(gio.t_txt_idx));

    graph_unique_names(gf);
    // VLA_GRAPH_DEBUG=1: report what changes between two computes of the cached
    // graph, mirroring ggml-cuda's graph-reuse test (whole node struct + source
    // data pointers / shapes); any change there defeats CUDA-graph replay.
    if (env_flag("VLA_GRAPH_DEBUG")) {
        struct Prop { ggml_tensor t; const void * sp[GGML_MAX_SRC]; };
        static std::vector<Prop> prev;
        const int n = ggml_graph_n_nodes(gf);
        std::vector<Prop> cur((size_t) n);
        for (int i = 0; i < n; ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            std::memcpy(&cur[(size_t) i].t, t, sizeof(ggml_tensor));
            for (int j = 0; j < GGML_MAX_SRC; ++j) cur[(size_t) i].sp[j] = t->src[j] ? t->src[j]->data : nullptr;
        }
        if (prev.size() == cur.size()) {
            int changed = 0;
            for (int i = 0; i < n; ++i) {
                const ggml_tensor & a = prev[(size_t) i].t, & b = cur[(size_t) i].t;
                if (std::memcmp(&a, &b, sizeof(ggml_tensor)) != 0 || std::memcmp(prev[(size_t) i].sp, cur[(size_t) i].sp, sizeof(cur[(size_t) i].sp)) != 0) {
                    if (changed < 5)
                        std::printf("vla(graph-debug): node %d %s changed: data %p->%p op_params %d name %d flags %d extra %p->%p srcdata %d\n",
                                    i, ggml_get_name(&b), a.data, b.data, std::memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0,
                                    std::strcmp(a.name, b.name) != 0, a.flags != b.flags, a.extra, b.extra,
                                    std::memcmp(prev[(size_t) i].sp, cur[(size_t) i].sp, sizeof(cur[(size_t) i].sp)) != 0);
                    ++changed;
                }
            }
            std::printf("vla(graph-debug): %d of %d nodes changed since the previous compute\n", changed, n);
        } else if (!prev.empty()) {
            std::printf("vla(graph-debug): graph size changed %zu -> %zu\n", prev.size(), cur.size());
        }
        prev = std::move(cur);
    }
    const auto tc0 = std::chrono::steady_clock::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    const auto tc1 = std::chrono::steady_clock::now();
    if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(gr00tn1d7): graph compute failed (%d)\n", (int) st); mg.release(); return {}; }
    stats.ms_inference = std::chrono::duration<float, std::milli>(tc1-tc0).count();

    std::vector<float> out((size_t) AH * AD);
    ggml_backend_tensor_get(gio.actions, out.data(), 0, out.size()*sizeof(float));

    if (const char * dump = std::getenv("VLA_GR00T_N17_DUMP")) {
        auto dump_t = [&](const char * name, ggml_tensor * t) {
            const int64_t n0 = t->ne[0], n1 = t->ne[1];
            std::vector<float> buf((size_t) n0*n1);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size()*sizeof(float));
            char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump, name, (long long) n0, (long long) n1);
            FILE * fp = std::fopen(path, "wb");
            if (fp) {
                std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
                std::fclose(fp);
                std::fprintf(stderr, "vla(gr00tn1d7): dumped %s shape=(%lld,%lld) to %s\n", name, (long long) n1, (long long) n0, path);
            }
        };
        dump_t("eagle",  eagle);
        dump_t("vl_embs", vl_embs);

        for (size_t li=0; li<lm_h_dump.size(); ++li) {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "lm_h_%02zu", li);
            dump_t(nm, lm_h_dump[li]);
        }

        for (size_t vi=0; vi<vlsa_dump.size(); ++vi) {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "vlsa_%02zu", vi);
            dump_t(nm, vlsa_dump[vi]);
        }

        if (inject_deepstack) {
            auto dump_host = [&](const char * name, const float * data, int64_t n0, int64_t n1) {
                char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump, name, (long long) n0, (long long) n1);
                FILE * fp = std::fopen(path, "wb");
                if (fp) {
                    std::fwrite(data, sizeof(float), (size_t) n0*n1, fp);
                    std::fclose(fp);
                    std::fprintf(stderr, "vla(gr00tn1d7): dumped %s shape=(%lld,%lld) to %s\n", name, (long long) n1, (long long) n0, path);
                }
            };

            for (int64_t v=0; v<n_views; ++v) {
                char nm[32]; std::snprintf(nm, sizeof(nm), "ds0_view%lld", (long long) v);
                dump_host(nm, ds_host[0].data()+v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "ds1_view%lld", (long long) v);
                dump_host(nm, ds_host[1].data()+v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "ds2_view%lld", (long long) v);
                dump_host(nm, ds_host[2].data()+v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "vit_view%lld", (long long) v);
                dump_host(nm, img_emb_host.data()+v * K * H, H, K);
            }
        }
    }
    if (!use_cache)
        mg.release();
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-t0).count();
    return out;
}

}
