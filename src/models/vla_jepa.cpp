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
#include "model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "backend.h"
#include "gguf_reader.h"
#include "scratch_ctx.h"
#include "layers/embed.h"
#include "layers/ffn.h"
#include "modules/dit_head.h"
#include "modules/prompt.h"
#include "modules/qwen3_lm.h"
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

struct VlaJepaModelArch : public ModelArchBase {
    VlaJepaModelArch() : ModelArchBase(Arch::VLA_JEPA) {}
    ~VlaJepaModelArch() override;

    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    scratch_ctx           vision_scratch;
    struct LmKey {
        int64_t seq=-1, nfuture=-1;
        bool operator==(const LmKey & o) const {
            return seq==o.seq && nfuture==o.nfuture;
        }
    };
    struct LmIO {
        ggml_tensor *t_embeds=nullptr,*t_pos2=nullptr,*t_lmmask=nullptr,*t_emb_idx=nullptr;
        ggml_tensor *t_ds[3]={nullptr,nullptr,nullptr};
        ggml_tensor *eagle=nullptr,*conditioning=nullptr;
    };
    struct HeadKey {
        int64_t nsteps=-1;
        bool operator==(const HeadKey & o) const {
            return nsteps==o.nsteps;
        }
    };
    struct HeadIO {
        ggml_tensor *t_cond=nullptr,*t_state=nullptr,*t_x0=nullptr,*actions=nullptr;
        std::vector<ggml_tensor*> t_tau, t_tproj;
    };
    graph_cache<LmKey, LmIO>     lm_graph;
    graph_cache<HeadKey, HeadIO> head_graph;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;

    int64_t lm_hidden=2048, lm_layers=28, n_q=16, n_kv=8, lm_head_dim=128, lm_inter=6144, vocab=151936;
    int64_t image_token_index=151655, embodied_token_id=151697;

    int64_t dit_hidden=768, dit_heads=12, dit_head_dim=64, dit_layers=16, cross_dim=2048, output_dim=1024, time_proj_dim=256;
    int64_t action_dim=7, state_dim=8, action_horizon=7, num_future=32, num_steps=4, num_buckets=1000;
    float   lm_rms_eps=1e-6f, lm_rope_base=5000000.0f;
    float   dit_ln_eps=1e-5f, dit_norm_out_eps=1e-6f;

    Qwen3VLTower vit;
    Qwen3LM      lm;
    DitHead      dit;

    ggml_tensor *ae_l1W=nullptr,*ae_l1b=nullptr,*ae_l2W=nullptr,*ae_l2b=nullptr,*ae_l3W=nullptr,*ae_l3b=nullptr;
    ggml_tensor *se_l1W=nullptr,*se_l1b=nullptr,*se_l2W=nullptr,*se_l2b=nullptr;
    ggml_tensor *ad_l1W=nullptr,*ad_l1b=nullptr,*ad_l2W=nullptr,*ad_l2b=nullptr;
    ggml_tensor *future_tokens=nullptr,*pos_embd=nullptr;

    FlowTimes          times;
    std::vector<float> c_mask; int64_t c_mask_seq = -1;
    gguf_reader        io{"vla_jepa"};

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

bool load_config(const gguf_reader & g, VlaJepaModelArch & m, Config & cfg) {
    auto U = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto fk = [&](const char * s) { thread_local char b[64]; std::snprintf(b, sizeof(b), "vla_jepa.%s", s); return b; };
    if (!m.vit.load_config("vla_jepa", g, "vla_jepa"))
        return false;
    U(fk("lm_hidden"), m.lm_hidden); U(fk("lm_layers"), m.lm_layers); U(fk("lm_q_heads"), m.n_q); U(fk("lm_kv_heads"), m.n_kv);
    U(fk("lm_head_dim"), m.lm_head_dim); U(fk("lm_inter"), m.lm_inter); U(fk("vocab_size"), m.vocab);
    U(fk("image_token_index"), m.image_token_index); U(fk("embodied_action_token_id"), m.embodied_token_id);
    U(fk("dit_hidden"), m.dit_hidden); U(fk("dit_heads"), m.dit_heads); U(fk("dit_head_dim"), m.dit_head_dim); U(fk("dit_layers"), m.dit_layers);
    U(fk("cross_dim"), m.cross_dim); U(fk("output_dim"), m.output_dim); U(fk("time_proj_dim"), m.time_proj_dim);
    U(fk("action_dim"), m.action_dim); U(fk("state_dim"), m.state_dim); U(fk("action_horizon"), m.action_horizon);
    U(fk("num_future_tokens"), m.num_future); U(fk("num_inference_timesteps"), m.num_steps); U(fk("num_timestep_buckets"), m.num_buckets);
    env_num_steps("vla_jepa", m.num_steps);
    F(fk("lm_rms_eps"), m.lm_rms_eps); F(fk("dit_ln_eps"), m.dit_ln_eps); F(fk("dit_norm_out_eps"), m.dit_norm_out_eps);
    if (g.has(fk("lm_rope_theta")))
        m.lm_rope_base = (float) g.f64(fk("lm_rope_theta"));

    // timesteps_proj always emits 256 floats into the time-projection input.
    if (m.time_proj_dim != 256) {
        std::fprintf(stderr, "vla(vla_jepa): time_proj_dim %lld, expected 256\n", (long long) m.time_proj_dim);
        return false;
    }

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
    m.lm.cfg.rope.freq_base   = m.lm_rope_base;
    m.lm.cfg.rope.sections[0] = 24;
    m.lm.cfg.rope.sections[1] = 20;
    m.lm.cfg.rope.sections[2] = 20;
    m.lm.cfg.rope.sections[3] = 0;

    m.dit.cfg.hidden       = m.dit_hidden;
    m.dit.cfg.heads        = m.dit_heads;
    m.dit.cfg.head_dim     = m.dit_head_dim;
    m.dit.cfg.layers       = m.dit_layers;
    m.dit.cfg.ln_eps       = m.dit_ln_eps;
    m.dit.cfg.norm_out_eps = m.dit_norm_out_eps;

    cfg = Config{};
    cfg.n_img = m.vit.n_tokens();
    cfg.n_lang = 1024; cfg.n_state = 1;
    cfg.n_suffix = m.action_horizon; cfg.max_state_dim = m.state_dim; cfg.max_action_dim = m.action_dim;
    cfg.real_state_dim = m.state_dim; cfg.real_action_dim = m.action_dim;
    cfg.hidden = m.lm_hidden; cfg.n_q_heads = m.n_q; cfg.n_kv_heads = m.n_kv; cfg.head_dim = m.lm_head_dim; cfg.n_layers = m.lm_layers;
    cfg.num_steps = (int) m.num_steps; cfg.rms_eps = m.lm_rms_eps;
    cfg.rope_n_dims = (int) m.lm_head_dim; cfg.rope_mode = GGML_ROPE_TYPE_IMROPE; cfg.rope_freq_base = m.lm_rope_base;
    // Raw output: this arch expects the client to apply the dataset statistics
    // (see the --stats-json flag in eval/client).
    cfg.denormalized = false;
    cfg.norm_eps = 1e-8f;
    return true;
}

}

VlaJepaModelArch::~VlaJepaModelArch() {
    if (weight_buf)
        ggml_backend_buffer_free(weight_buf);
    if (ctx_weights)
        ggml_free(ctx_weights);
    if (backend)
        ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> vla_jepa_create(const std::string& mmproj_path,
                                               const std::string& ckpt_path,
                                               const std::string&,
                                               const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(vla_jepa): note - mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<VlaJepaModelArch>();
    m->matmul_type = opts.weight_dtype.value_or(vla::default_weight_dtype(GGML_TYPE_BF16));

    if (!m->io.open(ckpt_path))
        return nullptr;
    gguf_reader & g = m->io;
    if (!g.has("vla_jepa.architecture")) {
        std::fprintf(stderr, "vla(vla_jepa): %s is not a vla_jepa GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, *m, m->cfg))
        return nullptr;
    m->times.build(m->num_steps, m->num_buckets, m->dit_hidden, m->action_horizon);
    std::printf("vla(vla_jepa): vit=Qwen3-VL %lldd×%lldL (deepstack@{%lld,%lld,%lld}, merge÷%lld)  lm=Qwen3-VL %lldd×%lldL (%lldq/%lldkv×%lld, θ=%g)  "
                "dit-B %lldL×%lldh×%lld(inner %lld, cross %lld, out %lld)  horizon=%lld action_dim=%lld state_dim=%lld future=%lld N_steps=%lld  resident=%s\n",
                (long long) m->vit.hidden, (long long) m->vit.layers, (long long) m->vit.deepstack_idx[0], (long long) m->vit.deepstack_idx[1], (long long) m->vit.deepstack_idx[2], (long long) m->vit.merge,
                (long long) m->lm_hidden, (long long) m->lm_layers, (long long) m->n_q, (long long) m->n_kv, (long long) m->lm_head_dim, (double) m->lm_rope_base,
                (long long) m->dit_layers, (long long) m->dit_heads, (long long) m->dit_head_dim, (long long) m->dit_hidden, (long long) m->cross_dim, (long long) m->output_dim,
                (long long) m->action_horizon, (long long) m->action_dim, (long long) m->state_dim, (long long) m->num_future, (long long) m->num_steps,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    {
        const Backend b = backend_init("vla(vla_jepa)", m->n_threads);
        if (!b.handle) {
            return nullptr;
        }
        m->backend = b.handle;
    }

    ggml_init_params wp = { (size_t) 32*1024*1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) {
        std::fprintf(stderr, "vla(vla_jepa): ggml_init(ctx_weights) failed\n");
        return nullptr;
    }

    WeightLoader L("vla_jepa", g, m->ctx_weights, m->matmul_type);

    m->vit.declare(L, "vit");
    m->lm.declare(L, "vlm");

    m->ae_l1W = L.f32("ah.act_enc.l1.weight");   m->ae_l1b = L.f32("ah.act_enc.l1.bias");
    m->ae_l2W = L.f32("ah.act_enc.l2.weight");   m->ae_l2b = L.f32("ah.act_enc.l2.bias");
    m->ae_l3W = L.f32("ah.act_enc.l3.weight");   m->ae_l3b = L.f32("ah.act_enc.l3.bias");
    m->se_l1W = L.f32("ah.state_enc.l1.weight"); m->se_l1b = L.f32("ah.state_enc.l1.bias");
    m->se_l2W = L.f32("ah.state_enc.l2.weight"); m->se_l2b = L.f32("ah.state_enc.l2.bias");
    m->ad_l1W = L.f32("ah.act_dec.l1.weight");   m->ad_l1b = L.f32("ah.act_dec.l1.bias");
    m->ad_l2W = L.f32("ah.act_dec.l2.weight");   m->ad_l2b = L.f32("ah.act_dec.l2.bias");
    m->future_tokens = L.f32("ah.future_tokens");
    m->pos_embd      = L.f32("ah.pos_embd");

    m->dit.declare(L, "ah.dit", false, false, "ah");

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;

    std::printf("vla(vla_jepa): weights resident in %.2f GiB (%s)\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0), dtype_name(m->matmul_type));
    if (!m->vit.build_caches("vla_jepa", m->io))
        return nullptr;
    return m;
}

std::vector<float> VlaJepaModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    const int64_t H = lm_hidden, E = dit_hidden, AD = action_dim, AH = action_horizon, OUTD = output_dim;
    const int64_t K = vit.n_tokens(), n_patches = vit.grid()*vit.grid();
    const int64_t Nseq = 1+num_future+AH;
    const char * dump_prefix = std::getenv("VLA_JEPA_DUMP");

    auto dump_t = [&](const char * name, ggml_tensor * t) {
        if (!dump_prefix)
            return;
        const int64_t n0 = t->ne[0], n1 = t->ne[1];
        std::vector<float> buf((size_t) n0*std::max<int64_t>(1, n1));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size()*sizeof(float));
        char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump_prefix, name, (long long) n0, (long long) n1);
        FILE * fp = std::fopen(path, "wb"); if (fp) {
            std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
            std::fclose(fp);
        }
    };

    std::vector<float> x_init;
    init_noise(in, (size_t) AH*AD, x_init);

    std::vector<float> cond_host((size_t) H * num_future, 0.0f);
    const char * cond_file = std::getenv("VLA_JEPA_COND");
    if (cond_file) {
        FILE * fp = std::fopen(cond_file, "rb");
        if (!fp) { std::fprintf(stderr, "vla(vla_jepa): VLA_JEPA_COND open failed: %s\n", cond_file); return {}; }
        const size_t want = cond_host.size();
        if (std::fread(cond_host.data(), sizeof(float), want, fp) != want) { std::fprintf(stderr, "vla(vla_jepa): VLA_JEPA_COND short read\n"); std::fclose(fp); return {}; }
        std::fclose(fp);
        std::printf("vla(vla_jepa): conditioning injected from %s (action-head isolation)\n", cond_file);
    } else {
        if (in.precomputed_img_emb) {
            std::fprintf(stderr, "vla(vla_jepa): precomputed_img_emb is not supported. The V-JEPA tower also "
                                 "emits deepstack features that a single embedding buffer cannot carry; pass raw images.\n");
            return {};
        }
        int64_t n_views = in.n_images;
        if (n_views <= 0) { std::fprintf(stderr, "vla(vla_jepa): no images in the request\n"); return {}; }
        std::vector<float> img_emb_host, ds_host[3];

        std::vector<float> inj_patches; const char * patches_file = std::getenv("VLA_JEPA_PATCHES");
        if (patches_file) {
            FILE * fp = std::fopen(patches_file, "rb");
            if (!fp) { std::fprintf(stderr, "vla(vla_jepa): VLA_JEPA_PATCHES open failed\n"); return {}; }
            inj_patches.resize((size_t) n_views * n_patches * vit.patch_flat);
            if (std::fread(inj_patches.data(), sizeof(float), inj_patches.size(), fp) != inj_patches.size()) { std::fprintf(stderr, "vla(vla_jepa): VLA_JEPA_PATCHES short read\n"); std::fclose(fp); return {}; }
            std::fclose(fp);
            std::printf("vla(vla_jepa): pixel_values injected from %s\n", patches_file);
        }
        if (inj_patches.empty() && !in.images) { std::fprintf(stderr, "vla(vla_jepa): n_images=%d but the images pointer is null\n", in.n_images); return {}; }

        const auto tv0 = std::chrono::steady_clock::now();
        const bool vok = vit.encode("vla_jepa", backend, vision_scratch, in.images, n_views,
                                    inj_patches.empty() ? nullptr : inj_patches.data(), img_emb_host, ds_host);
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-tv0).count();
        if (!vok) return {};
        if (dump_prefix) for (int64_t v=0; v<n_views; ++v) {
            char path[1024]; std::snprintf(path, sizeof(path), "%s_vit_view%lld_%lldx%lld.f32", dump_prefix, (long long) v, (long long) H, (long long) K);
            FILE * fp = std::fopen(path, "wb"); if (fp) { std::fwrite(img_emb_host.data()+v * K * H, sizeof(float), (size_t) K * H, fp); std::fclose(fp); }
        }
        const int64_t n_img = n_views * K;

        Prompt prompt;
        if (!build_prompt("vla_jepa", in, n_img, (int32_t) image_token_index, INT64_MAX, prompt)) return {};
        const int64_t SEQ = prompt.len();

        std::vector<float> inputs_embeds;
        if (!fetch_embeds(io, prompt, img_emb_host.data(), H, inputs_embeds)) return {};

        std::vector<int32_t> emb_pos_idx;
        for (int64_t p=0; p<SEQ; ++p)
            if (prompt.ids[p] == (int32_t) embodied_token_id)
                emb_pos_idx.push_back((int32_t) p);
        if ((int64_t) emb_pos_idx.size() != num_future) { std::fprintf(stderr, "vla(vla_jepa): found %zu embodied tokens, expected %lld\n", emb_pos_idx.size(), (long long) num_future); return {}; }

        std::vector<int32_t> pp;
        if (!mrope_positions("vla_jepa", prompt.ids, (int32_t) image_token_index, vit.grid()/vit.merge, pp)) return {};

        std::vector<std::vector<float>> ds_pad(3);
        for (int j=0; j<3; ++j) {
            ds_pad[j].assign((size_t) SEQ * H, 0.0f);
            for (int64_t k=0; k<n_img; ++k)
                std::memcpy(ds_pad[j].data()+(size_t) prompt.image_pos[k]*H, ds_host[j].data()+(size_t) k * H, H * sizeof(float));
        }

        const LmKey lkey{ SEQ, num_future };
        const bool lm_built = lm_graph.ensure(backend, lkey, (size_t) 512*1024*1024,
                                              [&](ggml_context * C, LmIO & gio) -> ggml_cgraph * {
        ggml_tensor * t_embeds = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);   ggml_set_input(t_embeds);
        ggml_tensor * t_pos2   = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4*SEQ);  ggml_set_input(t_pos2);
        ggml_tensor * t_lmmask = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ); ggml_set_input(t_lmmask);
        ggml_tensor * t_emb_idx= ggml_new_tensor_1d(C, GGML_TYPE_I32, num_future); ggml_set_input(t_emb_idx);
        ggml_tensor * t_ds[3];
        for (int j=0; j<3; ++j) {
            t_ds[j] = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);
            ggml_set_input(t_ds[j]);
        }
        ggml_tensor * hh = t_embeds;
        for (int64_t i=0; i<lm_layers; ++i) {
            hh = lm.block(C, lm.blk[i], hh, t_pos2, t_lmmask, SEQ);
            if (i < 3)
                hh = ggml_add(C, hh, t_ds[i]);
        }
        ggml_tensor * eagle = hh;
        ggml_set_output(eagle);
        ggml_tensor * conditioning = ggml_get_rows(C, eagle, t_emb_idx);
        ggml_set_output(conditioning);
        gio.t_embeds=t_embeds; gio.t_pos2=t_pos2; gio.t_lmmask=t_lmmask; gio.t_emb_idx=t_emb_idx;
        gio.t_ds[0]=t_ds[0]; gio.t_ds[1]=t_ds[1]; gio.t_ds[2]=t_ds[2];
        gio.eagle=eagle; gio.conditioning=conditioning;

        ggml_cgraph * lg = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(lg, conditioning);
        return lg;
        });
        if (!lm_built) { std::fprintf(stderr, "vla(vla_jepa): LM graph build failed\n"); return {}; }

        LmIO & gio = lm_graph.io();
        ggml_cgraph * lg = lm_graph.graph();
        ggml_tensor * t_embeds = gio.t_embeds, * t_pos2 = gio.t_pos2, * t_lmmask = gio.t_lmmask;
        ggml_tensor * t_emb_idx = gio.t_emb_idx;
        ggml_tensor * t_ds[3] = { gio.t_ds[0], gio.t_ds[1], gio.t_ds[2] };
        ggml_tensor * eagle = gio.eagle, * conditioning = gio.conditioning;

        ggml_backend_tensor_set(t_embeds, inputs_embeds.data(), 0, ggml_nbytes(t_embeds));
        ggml_backend_tensor_set(t_pos2, pp.data(), 0, ggml_nbytes(t_pos2));
        if (c_mask_seq != SEQ) {
            build_causal_mask(SEQ, c_mask);
            c_mask_seq = SEQ;
        }
        ggml_backend_tensor_set(t_lmmask, c_mask.data(), 0, ggml_nbytes(t_lmmask));
        ggml_backend_tensor_set(t_emb_idx, emb_pos_idx.data(), 0, ggml_nbytes(t_emb_idx));
        for (int j=0; j<3; ++j)
            ggml_backend_tensor_set(t_ds[j], ds_pad[j].data(), 0, ggml_nbytes(t_ds[j]));

        const auto tp0 = std::chrono::steady_clock::now();
        graph_unique_names(lg);
        if (ggml_backend_graph_compute(backend, lg) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(vla_jepa): LM compute failed\n"); return {}; }
        stats.ms_prefill = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-tp0).count();
        if (dump_prefix) {
            dump_t("eagle", eagle);
            dump_t("conditioning", conditioning);
        }
        ggml_backend_tensor_get(conditioning, cond_host.data(), 0, cond_host.size()*sizeof(float));
    }

    // Dumping adds graph outputs, so it always rebuilds.
    std::vector<ggml_tensor *> step_seq, step_pred, step_vel, step_act;
    if (dump_prefix)
        head_graph.release();
    const HeadKey hkey{ num_steps };
    const size_t head_nodes = 8192 + (size_t) num_steps*64*(dit_layers+1);
    const bool head_built = head_graph.ensure(backend, hkey, head_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(head_nodes, false),
                                              [&](ggml_context * C, HeadIO & gio) -> ggml_cgraph * {
    ggml_tensor * t_cond  = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, num_future); ggml_set_input(t_cond);
    ggml_tensor * t_state = ggml_new_tensor_2d(C, GGML_TYPE_F32, state_dim, 1);  ggml_set_input(t_state);
    ggml_tensor * t_x0    = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH);        ggml_set_input(t_x0);
    std::vector<ggml_tensor *> t_tau(num_steps), t_tproj(num_steps);
    for (int64_t s=0; s<num_steps; ++s) {
        t_tau[s] = ggml_new_tensor_2d(C, GGML_TYPE_F32, E, AH);
        ggml_set_input(t_tau[s]);
        t_tproj[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, time_proj_dim);
        ggml_set_input(t_tproj[s]);
    }

    ggml_tensor * state_features = ffn_relu(C, se_l1W, se_l1b, se_l2W, se_l2b, t_state);
    ggml_tensor * future = future_tokens;
    const float dt = 1.0f/(float) num_steps;
    step_seq.assign(num_steps, nullptr); step_pred.assign(num_steps, nullptr);
    step_vel.assign(num_steps, nullptr);  step_act.assign(num_steps, nullptr);

    ggml_tensor * actions = t_x0;
    for (int64_t s=0; s<num_steps; ++s) {
        ggml_tensor * temb = dit.time_emb(C, t_tproj[s]);

        ggml_tensor * a_emb = ggml_add(C, ggml_mul_mat(C, ae_l1W, actions), ae_l1b);
        ggml_tensor * cat   = ggml_concat(C, a_emb, t_tau[s], 0);
        ggml_tensor * x2    = ggml_silu(C, ggml_add(C, ggml_mul_mat(C, ae_l2W, cat), ae_l2b));
        ggml_tensor * af    = ggml_add(C, ggml_mul_mat(C, ae_l3W, x2), ae_l3b);
        af = ggml_add(C, af, ggml_view_2d(C, pos_embd, E, AH, pos_embd->nb[1], 0));
        ggml_tensor * seq = ggml_concat(C, ggml_concat(C, state_features, future, 1), af, 1);
        step_seq[s] = seq;
        ggml_tensor * x = seq;
        for (int64_t i=0; i<dit_layers; ++i) {
            ggml_tensor * enc = (i%2 == 0) ? t_cond : nullptr;
            x = dit.block(C, dit.blk[i], x, temb, enc);
        }

        ggml_tensor * model_output = dit.proj_out(C, x, temb);
        step_pred[s] = model_output;

        ggml_tensor * last = ggml_cont(C, ggml_view_2d(C, model_output, OUTD, AH, model_output->nb[1], (size_t) (Nseq-AH)*model_output->nb[1]));
        ggml_tensor * vel = ffn_relu(C, ad_l1W, ad_l1b, ad_l2W, ad_l2b, last);
        step_vel[s] = vel;
        actions = ggml_add(C, actions, ggml_scale(C, vel, dt));
        step_act[s] = actions;
        if (dump_prefix) {
            ggml_set_output(step_seq[s]);
            ggml_set_output(step_pred[s]);
            ggml_set_output(step_vel[s]);
            ggml_set_output(step_act[s]);
        }
    }
    ggml_set_output(actions);
    gio.t_cond=t_cond; gio.t_state=t_state; gio.t_x0=t_x0; gio.actions=actions;
    gio.t_tau=t_tau; gio.t_tproj=t_tproj;

    ggml_cgraph * hg = ggml_new_graph_custom(C, head_nodes, false);
    ggml_build_forward_expand(hg, actions);
    if (dump_prefix) for (int64_t s=0; s<num_steps; ++s) {
        ggml_build_forward_expand(hg, step_seq[s]);
        ggml_build_forward_expand(hg, step_pred[s]);
        ggml_build_forward_expand(hg, step_vel[s]);
        ggml_build_forward_expand(hg, step_act[s]);
    }
    return hg;
    });
    if (!head_built) { std::fprintf(stderr, "vla(vla_jepa): head graph build failed\n"); return {}; }

    HeadIO & hio = head_graph.io();
    ggml_cgraph * hg = head_graph.graph();
    ggml_tensor * t_cond = hio.t_cond, * t_state = hio.t_state, * t_x0 = hio.t_x0, * actions = hio.actions;
    std::vector<ggml_tensor*> & t_tau = hio.t_tau; std::vector<ggml_tensor*> & t_tproj = hio.t_tproj;

    ggml_backend_tensor_set(t_cond, cond_host.data(), 0, ggml_nbytes(t_cond));
    {
        std::vector<float> st(state_dim, 0.0f);
        for (int64_t i=0; i<state_dim; ++i)
            st[i] = in.state ? in.state[i] : 0.0f;
        ggml_backend_tensor_set(t_state, st.data(), 0, ggml_nbytes(t_state));
    }
    ggml_backend_tensor_set(t_x0, x_init.data(), 0, ggml_nbytes(t_x0));
    times.upload(t_tau, t_tproj);

    const auto td0 = std::chrono::steady_clock::now();
    graph_unique_names(hg);
    if (ggml_backend_graph_compute(backend, hg) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(vla_jepa): head compute failed\n"); return {}; }
    stats.ms_denoise = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-td0).count();
    stats.ms_inference = stats.ms_prefill+stats.ms_denoise;

    if (dump_prefix) for (int64_t s=0; s<num_steps; ++s) {
        char nm[48];
        std::snprintf(nm, sizeof(nm), "step%lld_seq", (long long) s); dump_t(nm, step_seq[s]);
        std::snprintf(nm, sizeof(nm), "step%lld_dit_pred", (long long) s); dump_t(nm, step_pred[s]);
        std::snprintf(nm, sizeof(nm), "step%lld_velocity", (long long) s); dump_t(nm, step_vel[s]);
        std::snprintf(nm, sizeof(nm), "step%lld_actions", (long long) s); dump_t(nm, step_act[s]);
    }

    std::vector<float> out((size_t) AH * AD);
    ggml_backend_tensor_get(actions, out.data(), 0, out.size()*sizeof(float));
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-t0).count();
    return out;
}

}
