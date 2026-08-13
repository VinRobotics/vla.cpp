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
#include "layers/attn.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "modules/action_expert.h"
#include "modules/dit_head.h"
#include "modules/encoder.h"
#include "modules/qwen3_lm.h"
#include "options.h"
#include "model.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "backend.h"
#include "gguf.h"
#include "gguf_reader.h"
#include "scratch_ctx.h"
#include "layers/embed.h"
#include "modules/qwen3vl_vit.h"
#include "env_flag.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace vla {
namespace {


struct VlsaLayerW  { ggml_tensor *n1w,*n1b,*n3w,*n3b,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,*Wff0,*bff0,*Wff2,*bff2; };

}

struct Gr00tN1d7ModelArch : public ModelArchBase {
    Gr00tN1d7ModelArch() : ModelArchBase(Arch::GR00T_N1_7) {}
    ~Gr00tN1d7ModelArch() override;

    std::string           gguf_path;
    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    scratch_ctx           vision_scratch;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;

    int64_t vit_hidden=1024, vit_layers=24, vit_heads=16, vit_inter=4096;
    int64_t patch_size=16, temporal_patch=2, spatial_merge=2, vit_num_pos=2304, vit_patch_flat=1536, vit_merged_dim=4096;
    int64_t deepstack_idx[3] = {5, 11, 17};
    int64_t lm_hidden=2048, lm_layers=16, n_q=16, n_kv=8, lm_head_dim=128, lm_inter=6144, vocab=151936, image_token_index=151655;
    int64_t vlsa_layers=4, vlsa_heads=32, vlsa_head_dim=64, vlsa_ff_inner=8192;
    int64_t bb_embed_dim=2048, in_embed_dim=1536, dit_hidden=1536, dit_heads=32, dit_head_dim=48, dit_layers=32, dit_interleave=1, attend_text_every_n=2;
    int64_t action_horizon=40, action_dim=132, max_state_dim=132;
    int64_t num_steps=4, num_buckets=1000, max_embodiments=32, max_seq_len=1024;
    int64_t image_target_size=256;
    float   vit_ln_eps=1e-6f, vit_rope_base=10000.0f, lm_rms_eps=1e-6f, lm_rope_base=5000000.0f;
    float   vlln_eps=1e-5f, vlsa_ln_eps=1e-5f, ln_eps=1e-5f, norm_out_eps=1e-6f, connector_ln_eps=1e-6f;
    int64_t embodiment_id = 2;

    Qwen3VLTower vit;
    Qwen3LM      lm;
    EncStack     vlsa;
    ActionExpert aex;
    DitHead      dit;
    ggml_tensor *vlln_w=nullptr,*vlln_b=nullptr;

    bool                            caches_ready = false;
    std::vector<int64_t>            c_grow, c_gcol;
    std::vector<float>              c_rope_cos, c_rope_sin;
    std::vector<float>              c_pos_interp;
    std::vector<std::vector<float>> c_tau, c_tproj;
    std::vector<float>              c_mask; int64_t c_mask_seq = -1;
    gguf_reader                     io;
    bool build_caches();

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
        std::vector<ggml_tensor*> t_tau, t_tproj;
    };
    graph_cache<MainKey, MainIO> mg;

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

ggml_tensor * head_view(ggml_context * C, ggml_tensor * proj, int64_t hd, int64_t heads,
                        int64_t T, int64_t E, int nblk, int blk) {
    const size_t es = ggml_element_size(proj);
    return ggml_view_3d(C, proj, hd, heads, T, (size_t) hd * es, (size_t) nblk * E * es, (size_t) blk * E * es);
}





bool load_config(const gguf_reader & g, Gr00tN1d7ModelArch & m, Config & cfg) {
    auto U = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto fk = [&](const char * s) { static char b[64]; std::snprintf(b, sizeof(b), "gr00t_n1_7.%s", s); return b; };
    U(fk("vit_hidden"), m.vit_hidden); U(fk("vit_layers"), m.vit_layers); U(fk("vit_heads"), m.vit_heads); U(fk("vit_inter"), m.vit_inter);
    U(fk("patch_size"), m.patch_size); U(fk("temporal_patch_size"), m.temporal_patch); U(fk("spatial_merge_size"), m.spatial_merge);
    U(fk("vit_num_position_embeddings"), m.vit_num_pos); U(fk("vit_patch_flat"), m.vit_patch_flat); U(fk("vit_merged_dim"), m.vit_merged_dim);
    U(fk("deepstack_idx_0"), m.deepstack_idx[0]); U(fk("deepstack_idx_1"), m.deepstack_idx[1]); U(fk("deepstack_idx_2"), m.deepstack_idx[2]);
    U(fk("lm_hidden"), m.lm_hidden); U(fk("lm_layers_used"), m.lm_layers); U(fk("lm_q_heads"), m.n_q); U(fk("lm_kv_heads"), m.n_kv);
    U(fk("lm_head_dim"), m.lm_head_dim); U(fk("lm_inter"), m.lm_inter); U(fk("vocab_size"), m.vocab); U(fk("image_token_index"), m.image_token_index);
    U(fk("vlsa_layers"), m.vlsa_layers); U(fk("vlsa_heads"), m.vlsa_heads); U(fk("vlsa_head_dim"), m.vlsa_head_dim); U(fk("vlsa_ff_inner"), m.vlsa_ff_inner);
    U(fk("backbone_embedding_dim"), m.bb_embed_dim); U(fk("input_embedding_dim"), m.in_embed_dim);
    U(fk("dit_hidden"), m.dit_hidden); U(fk("dit_heads"), m.dit_heads); U(fk("dit_head_dim"), m.dit_head_dim); U(fk("dit_layers"), m.dit_layers); U(fk("dit_interleave"), m.dit_interleave);
    U(fk("attend_text_every_n_blocks"), m.attend_text_every_n);
    U(fk("action_horizon"), m.action_horizon); U(fk("action_dim"), m.action_dim); U(fk("max_state_dim"), m.max_state_dim);
    U(fk("num_inference_timesteps"), m.num_steps); U(fk("num_timestep_buckets"), m.num_buckets); U(fk("max_num_embodiments"), m.max_embodiments); U(fk("max_seq_len"), m.max_seq_len);
    U(fk("image_target_size"), m.image_target_size);

    // merge_block_coords only enumerates the patch grid exactly when the spatial
    // merge divides it; otherwise it emits rows past the position table.
    if (m.patch_size <= 0 || m.spatial_merge <= 0 || m.image_target_size % m.patch_size != 0 ||
        (m.image_target_size / m.patch_size) % m.spatial_merge != 0) {
        std::fprintf(stderr, "vla(gr00tn1d7): image %lld / patch %lld / merge %lld do not divide evenly\n",
                     (long long) m.image_target_size, (long long) m.patch_size, (long long) m.spatial_merge);
        return false;
    }

    if (const char * ns = std::getenv("VLA_NUM_STEPS")) {
        char * end = nullptr; long v = std::strtol(ns, &end, 10);
        if (end && *end == '\0' && v >= 1) { m.num_steps = (int64_t) v; std::fprintf(stderr, "vla(gr00tn1d7): VLA_NUM_STEPS override → num_steps=%lld\n", (long long) v); }
    }
    F(fk("vit_ln_eps"), m.vit_ln_eps); F(fk("lm_rms_eps"), m.lm_rms_eps); F(fk("ln_eps"), m.ln_eps); F(fk("norm_out_eps"), m.norm_out_eps);
    F(fk("vlln_eps"), m.vlln_eps); F(fk("vlsa_ln_eps"), m.vlsa_ln_eps); F(fk("connector_ln_eps"), m.connector_ln_eps); F(fk("vit_rope_theta"), m.vit_rope_base);
    if (g.has(fk("lm_rope_theta"))) m.lm_rope_base = (float) g.f64(fk("lm_rope_theta"));

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
    {
        const std::string js = g.str(fk("embodiment_id_mapping"));
        auto lookup = [&](const char * key) -> long {
            const std::string k = std::string("\"") + key + "\"";
            size_t p = js.find(k); if (p == std::string::npos) return -1;
            p = js.find(':', p + k.size()); if (p == std::string::npos) return -1;
            return std::strtol(js.c_str() + p + 1, nullptr, 10);
        };
        long ls = lookup("libero_sim"); if (ls >= 0) m.aex.embodiment_id = ls;
        if (const char * e = std::getenv("VLA_GR00T_EMBODIMENT")) {
            char * end = nullptr; long v = std::strtol(e, &end, 10);
            if (end && *end == '\0') m.aex.embodiment_id = v;
            else { long id = lookup(e); if (id >= 0) m.aex.embodiment_id = id; else std::fprintf(stderr, "vla(gr00tn1d7): embodiment tag '%s' not in embodiment_id_mapping; using id %lld\n", e, (long long) m.aex.embodiment_id); }
        }
    }
    if (m.aex.embodiment_id < 0 || m.aex.embodiment_id >= m.max_embodiments) { std::fprintf(stderr, "vla(gr00tn1d7): embodiment id %lld out of range [0,%lld)\n", (long long) m.aex.embodiment_id, (long long) m.max_embodiments); return false; }

    cfg = Config{};
    cfg.n_img = 64; cfg.n_lang = m.max_seq_len; cfg.n_state = 1;
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
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> gr00t_n1_7_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string&,
                                                 const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(gr00tn1d7): note - mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<Gr00tN1d7ModelArch>();
    m->gguf_path   = ckpt_path;
    m->matmul_type = opts.weight_dtype.value_or(GGML_TYPE_BF16);

    gguf_reader g("gr00tn1d7");
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has("gr00t_n1_7.architecture")) { std::fprintf(stderr, "vla(gr00tn1d7): %s is not a gr00t_n1_7 GGUF\n", ckpt_path.c_str()); return nullptr; }
    if (!load_config(g, *m, m->cfg)) return nullptr;
    std::printf("vla(gr00tn1d7): vit=Qwen3-VL %lldd×%lldL×%lldh (Conv3d patch %lld², temporal %lld; learned pos %lld + 2D rope; deepstack@{%lld,%lld,%lld}; merge÷%lld)  "
                "lm=Qwen3-VL %lldd×%lldL (%lldq/%lldkv×%lld, θ=%g)  vlsa=%lldL×%lldh×%lld  dit=AlternateVLDiT %lldL×%lldh×%lld(inner %lld) attend_text_every_n=%lld  "
                "in_emb=%lld  horizon=%lld action_dim=%lld max_state=%lld N_steps=%lld  embodiment=%lld  resident=%s\n",
                (long long) m->vit_hidden, (long long) m->vit_layers, (long long) m->vit_heads, (long long) m->patch_size, (long long) m->temporal_patch,
                (long long) m->vit_num_pos, (long long) m->deepstack_idx[0], (long long) m->deepstack_idx[1], (long long) m->deepstack_idx[2], (long long) m->spatial_merge,
                (long long) m->lm_hidden, (long long) m->lm_layers, (long long) m->n_q, (long long) m->n_kv, (long long) m->lm_head_dim, (double) m->lm_rope_base,
                (long long) m->vlsa_layers, (long long) m->vlsa_heads, (long long) m->vlsa_head_dim,
                (long long) m->dit_layers, (long long) m->dit_heads, (long long) m->dit_head_dim, (long long) m->dit_hidden, (long long) m->attend_text_every_n, (long long) m->in_embed_dim,
                (long long) m->action_horizon, (long long) m->action_dim, (long long) m->max_state_dim, (long long) m->num_steps, (long long) m->aex.embodiment_id,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    {
        const Backend b = backend_init("vla(gr00tn1d7)", m->n_threads);
        if (!b.handle) { return nullptr; }
        m->backend = b.handle;
    }

    ggml_init_params wp = { (size_t) 32*1024*1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) { std::fprintf(stderr, "vla(gr00tn1d7): ggml_init(ctx_weights) failed\n"); return nullptr; }

    WeightLoader L("gr00tn1d7", g, m->ctx_weights, m->matmul_type);

    m->vit.declare(L, "vit", m->vit_layers);
    m->lm.declare(L, "vlm");

    m->vlln_w = L.f32("aex.vlln.weight");
    m->vlln_b = L.f32("aex.vlln.bias");
    m->vlsa.declare(L, "aex.vlsa", m->vlsa_layers, EncNames{"norm1", "norm3", "ff0", "ff2"});

    m->aex.declare(L, "aex");
    m->dit.declare(L, "aex.dit", true, m->dit_interleave != 0);

    if (!L.upload(m->backend, &m->weight_buf)) return nullptr;

    std::printf("vla(gr00tn1d7): QKV-fused DiT (self Wqkv / cross Wkv)\n");
    std::printf("vla(gr00tn1d7): weights resident in %.2f GiB (%s) - incl. Qwen3-VL vision tower + deepstack + vl_self_attention; embodiment id %lld\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0),
                dtype_name(m->matmul_type), (long long) m->aex.embodiment_id);
    if (!m->build_caches()) { std::fprintf(stderr, "vla(gr00tn1d7): build_caches failed\n"); return nullptr; }
    return m;
}

bool Gr00tN1d7ModelArch::build_caches() {
    if (caches_ready) return true;
    const int64_t side = image_target_size, ps = patch_size, m2 = spatial_merge;
    const int64_t grid = side / ps;
    const int64_t hd_vit = vit_hidden / vit_heads;
    const int64_t num_side = (int64_t) std::lround(std::sqrt((double) vit_num_pos));
    const int64_t E = in_embed_dim, AH = action_horizon;

    merge_block_coords(grid, grid, m2, c_grow, c_gcol);
    vit_rope_tables(c_grow, c_gcol, hd_vit, (double) vit_rope_base, c_rope_cos, c_rope_sin);

    if (!io.open(gguf_path)) { std::fprintf(stderr, "vla(gr00tn1d7): build_caches: io.open(%s) failed\n", gguf_path.c_str()); return false; }
    std::vector<float> pos_table = io.read_f32("vit.pos_embd");
    if (pos_table.empty() || (int64_t) pos_table.size() != vit_num_pos * vit_hidden) {
        std::fprintf(stderr, "vla(gr00tn1d7): build_caches: vit.pos_embd unreadable\n"); return false;
    }
    interp_pos_embed(pos_table, num_side, vit_hidden, c_grow, c_gcol, grid, grid, c_pos_interp);

    c_tau.assign((size_t) num_steps, {}); c_tproj.assign((size_t) num_steps, {});
    for (int64_t s = 0; s < num_steps; ++s) {
        const int64_t bucket = (int64_t) ((double) s / (double) num_steps * (double) num_buckets);
        action_sinusoid(bucket, E, AH, c_tau[(size_t) s]);
        timesteps_proj(bucket, c_tproj[(size_t) s]);
    }
    caches_ready = true;
    return true;
}

std::vector<float> Gr00tN1d7ModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    const int64_t H = lm_hidden, E = in_embed_dim;
    const int64_t side = image_target_size;
    const int64_t ps = patch_size, m2 = spatial_merge;
    const int64_t grid = side / ps;
    const int64_t n_patches = grid * grid;
    const int64_t K = (grid / m2) * (grid / m2);
    const int64_t hd_vit = vit_hidden / vit_heads;
    const int64_t AD = action_dim, AH = action_horizon, Nsa = 1 + AH;
    const bool    do_dump = (std::getenv("VLA_GR00T_N17_DUMP") != nullptr);

    if (!caches_ready) { std::fprintf(stderr, "vla(gr00tn1d7): caches not ready\n"); return {}; }
    const std::vector<int64_t> & grow = c_grow, & gcol = c_gcol;
    const std::vector<float> & rope_cos = c_rope_cos, & rope_sin = c_rope_sin, & pos_interp = c_pos_interp;

    int64_t n_views = 0;
    std::vector<float> img_emb_host, ds_host[3];
    const float * img_emb_ptr = nullptr;
    if (in.precomputed_img_emb && in.n_img_views > 0) {
        n_views = in.n_img_views; img_emb_ptr = in.precomputed_img_emb;

        for (int j = 0; j < 3; ++j) ds_host[j].assign((size_t) n_views * K * H, 0.0f);
    } else if (in.images && in.n_images > 0) {
        n_views = in.n_images;
        img_emb_host.assign((size_t) n_views * K * H, 0.0f);
        for (int j = 0; j < 3; ++j) ds_host[j].assign((size_t) n_views * K * H, 0.0f);

        ggml_context * VC = vision_scratch.reset((size_t) 512 * 1024 * 1024);
        if (!VC) { std::fprintf(stderr, "vla(gr00tn1d7): ggml_init(vision ctx) failed\n"); return {}; }
        ggml_tensor * t_patches = ggml_new_tensor_2d(VC, GGML_TYPE_F32, vit_patch_flat, n_patches); ggml_set_input(t_patches);
        ggml_tensor * t_pos     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, vit_hidden, n_patches);     ggml_set_input(t_pos);
        ggml_tensor * t_cos     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, hd_vit, n_patches);          ggml_set_input(t_cos);
        ggml_tensor * t_sin     = ggml_new_tensor_2d(VC, GGML_TYPE_F32, hd_vit, n_patches);          ggml_set_input(t_sin);
        ggml_tensor * h = ggml_add(VC, ggml_add(VC, ggml_mul_mat(VC, vit.patch_w, t_patches), vit.patch_b), t_pos);

        ggml_set_output(h);
        ggml_tensor * stash[3] = {nullptr, nullptr, nullptr};
        for (int64_t i = 0; i < vit_layers; ++i) {
            h = build_vit_layer(VC, vit.blk[i], h, t_cos, t_sin, n_patches, vit_heads, hd_vit, vit_hidden, vit_ln_eps);
            ggml_set_output(h);
            for (int j = 0; j < 3; ++j) if (i == deepstack_idx[j]) stash[j] = h;
        }
        ggml_tensor * ds_out[3];
        for (int j = 0; j < 3; ++j) { ds_out[j] = build_merger(VC, vit.deepstack[j], stash[j] ? stash[j] : h, vit_hidden, m2, connector_ln_eps, false); ggml_set_output(ds_out[j]); }
        ggml_tensor * vit_embeds = build_merger(VC, vit.merger, h, vit_hidden, m2, connector_ln_eps, true);
        ggml_set_output(vit_embeds);
        ggml_cgraph * vg = ggml_new_graph_custom(VC, 16384, false);
        ggml_build_forward_expand(vg, vit_embeds);
        for (int j = 0; j < 3; ++j) ggml_build_forward_expand(vg, ds_out[j]);
        if (!vision_scratch.alloc(backend, vg)) { std::fprintf(stderr, "vla(gr00tn1d7): vision gallocr alloc failed\n"); return {}; }
        const auto tv0 = std::chrono::steady_clock::now();
        std::vector<float> patches;
        bool vok = true;
        for (int64_t v = 0; v < n_views && vok; ++v) {
            if (!preprocess_image_patches("gr00tn1d7", in.images[v], side, ps, temporal_patch, grow, gcol, patches)) { vok = false; break; }

            ggml_backend_tensor_set(t_pos, pos_interp.data(), 0, ggml_nbytes(t_pos));
            ggml_backend_tensor_set(t_cos, rope_cos.data(), 0, ggml_nbytes(t_cos));
            ggml_backend_tensor_set(t_sin, rope_sin.data(), 0, ggml_nbytes(t_sin));
            ggml_backend_tensor_set(t_patches, patches.data(), 0, ggml_nbytes(t_patches));
            if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(gr00tn1d7): vision compute failed\n"); vok = false; break; }
            ggml_backend_tensor_get(vit_embeds, img_emb_host.data() + v * K * H, 0, ggml_nbytes(vit_embeds));
            for (int j = 0; j < 3; ++j) ggml_backend_tensor_get(ds_out[j], ds_host[j].data() + v * K * H, 0, ggml_nbytes(ds_out[j]));
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tv0).count();
        if (!vok) return {};
        img_emb_ptr = img_emb_host.data();
    } else {
        std::fprintf(stderr, "vla(gr00tn1d7): no images and no precomputed_img_emb in the request\n"); return {};
    }
    const int64_t n_img = n_views * K;

    std::vector<int32_t> input_ids;
    int64_t n_img_slots = 0;
    for (int j = 0; j < in.n_lang; ++j) if (in.lang_tokens[j] == (int32_t) image_token_index) ++n_img_slots;
    if (n_img_slots == n_img) {
        input_ids.assign(in.lang_tokens, in.lang_tokens + in.n_lang);
    } else if (n_img_slots == 0) {
        input_ids.reserve(n_img + in.n_lang);
        for (int64_t i = 0; i < n_img; ++i) input_ids.push_back((int32_t) image_token_index);
        for (int j = 0; j < in.n_lang; ++j) input_ids.push_back(in.lang_tokens[j]);
    } else {
        std::fprintf(stderr, "vla(gr00tn1d7): lang_tokens has %lld image-token slots but n_img=%lld; expected 0 (v1 fallback) or %lld (chat-template path)\n",
                     (long long) n_img_slots, (long long) n_img, (long long) n_img);
        return {};
    }
    const int64_t SEQ = (int64_t) input_ids.size();
    if (SEQ > max_seq_len) { std::fprintf(stderr, "vla(gr00tn1d7): prompt too long (%lld > %lld)\n", (long long) SEQ, (long long) max_seq_len); return {}; }

    std::vector<float> inputs_embeds((size_t) SEQ * H);
    if (!io.fetch_rows_f32("token_embd.weight", input_ids, inputs_embeds.data(), H)) return {};
    {   int64_t k = 0;
        for (int64_t p = 0; p < SEQ; ++p) if (input_ids[p] == (int32_t) image_token_index) {
            if (k >= n_img) { std::fprintf(stderr, "vla(gr00tn1d7): more <image> tokens than ViT embeds\n"); return {}; }
            std::memcpy(inputs_embeds.data() + p * H, img_emb_ptr + k * H, H * sizeof(float)); ++k;
        }
    }

    std::vector<int32_t> image_pos_idx, text_pos_idx;
    image_pos_idx.reserve((size_t) n_img); text_pos_idx.reserve((size_t) (SEQ - n_img));
    for (int64_t p = 0; p < SEQ; ++p) {
        if (input_ids[p] == (int32_t) image_token_index) image_pos_idx.push_back((int32_t) p);
        else                                              text_pos_idx.push_back((int32_t) p);
    }
    const int64_t SEQ_TXT = (int64_t) text_pos_idx.size();
    if ((int64_t) image_pos_idx.size() != n_img) {
        std::fprintf(stderr, "vla(gr00tn1d7): internal: built %zu image positions, expected %lld\n", image_pos_idx.size(), (long long) n_img); return {};
    }

    std::vector<std::vector<float>> ds_pad(3);
    const bool inject_deepstack = (in.images && in.n_images > 0);
    if (inject_deepstack) for (int j = 0; j < 3; ++j) {
        ds_pad[j].assign((size_t) SEQ * H, 0.0f);
        for (int64_t k = 0; k < n_img; ++k) {
            std::memcpy(ds_pad[j].data() + (size_t) image_pos_idx[k] * H,
                        ds_host[j].data() + (size_t) k * H, H * sizeof(float));
        }
    }

    std::vector<float> x_init((size_t) AH * AD);
    if (in.noise) std::memcpy(x_init.data(), in.noise, x_init.size() * sizeof(float));
    else { std::mt19937 rng((uint32_t) std::chrono::steady_clock::now().time_since_epoch().count()); std::normal_distribution<float> nd(0.f, 1.f); for (auto & v : x_init) v = nd(rng); }

    // On by default: 16% faster, bit-identical. Set VLA_GR00T_GRAPH_CACHE=0 to opt out.
    // Dumping adds graph outputs, so it always rebuilds.
    const char * gc = std::getenv("VLA_GR00T_GRAPH_CACHE");
    const bool use_cache = (!gc || std::strcmp(gc, "0") != 0) && !do_dump;
    if (!use_cache) mg.release();

    ggml_tensor * eagle = nullptr, * vl_embs = nullptr;
    std::vector<ggml_tensor*> lm_h_dump, vlsa_dump;
    const MainKey mkey{ SEQ, n_img, SEQ_TXT, num_steps, inject_deepstack };
    const bool built = mg.ensure(backend, mkey, (size_t) 256 * 1024 * 1024,
                                 [&](ggml_context * C, MainIO & gio) -> ggml_cgraph * {
    ggml_tensor * t_embeds = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);          ggml_set_input(t_embeds);
    ggml_tensor * t_pos    = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4 * SEQ);         ggml_set_input(t_pos);
    ggml_tensor * t_lmmask = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ);        ggml_set_input(t_lmmask);
    ggml_tensor * t_state  = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_state_dim, 1);ggml_set_input(t_state);
    ggml_tensor * t_x0     = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH);          ggml_set_input(t_x0);
    ggml_tensor * t_ds[3] = {nullptr,nullptr,nullptr};
    if (inject_deepstack) for (int j = 0; j < 3; ++j) { t_ds[j] = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ); ggml_set_input(t_ds[j]); }

    ggml_tensor * t_img_idx = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_img);    ggml_set_input(t_img_idx);
    ggml_tensor * t_txt_idx = (SEQ_TXT > 0)
        ? ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ_TXT) : nullptr;
    if (t_txt_idx) ggml_set_input(t_txt_idx);
    std::vector<ggml_tensor *> t_tau(num_steps), t_tproj(num_steps);
    for (int64_t s = 0; s < num_steps; ++s) {
        t_tau[s]   = ggml_new_tensor_2d(C, GGML_TYPE_F32, E, AH); ggml_set_input(t_tau[s]);
        t_tproj[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, 256);   ggml_set_input(t_tproj[s]);
    }

    ggml_tensor * h = t_embeds;
    for (int64_t i = 0; i < lm_layers; ++i) {
        h = lm.block(C, lm.blk[i], h, t_pos, t_lmmask, SEQ);
        if (inject_deepstack && i < 3) h = ggml_add(C, h, t_ds[i]);
        if (do_dump) { ggml_set_output(h); lm_h_dump.push_back(h); }
    }

    eagle = h;
    ggml_set_name(eagle, "eagle"); ggml_set_output(eagle);

    vl_embs = ggml_add(C, ggml_mul(C, ggml_norm(C, eagle, vlln_eps), vlln_w), vlln_b);
    if (do_dump) { ggml_set_output(vl_embs); vlsa_dump.push_back(vl_embs); }
    for (int64_t i = 0; i < vlsa_layers; ++i) {
        vl_embs = vlsa.block(C, vlsa.blk[i], vl_embs, SEQ);
        if (do_dump) { ggml_set_output(vl_embs); vlsa_dump.push_back(vl_embs); }
    }
    ggml_set_name(vl_embs, "vl_embs"); ggml_set_output(vl_embs);

    ggml_tensor * vl_img = ggml_get_rows(C, vl_embs, t_img_idx);
    ggml_tensor * vl_txt = (t_txt_idx ? ggml_get_rows(C, vl_embs, t_txt_idx) : vl_img);

    ggml_tensor * state_features = cat_linear(C, aex.se_l2W, aex.se_l2b, aex.embodiment_id, ggml_relu(C, cat_linear(C, aex.se_l1W, aex.se_l1b, aex.embodiment_id, t_state)));

    const float dt = 1.0f / (float) num_steps;
    const int64_t every2 = 2 * attend_text_every_n;

    std::vector<ggml_tensor *> Kc(dit_layers, nullptr), Vc(dit_layers, nullptr);
    for (int64_t i = 0; i < dit_layers; ++i) {
        if (dit_interleave && (i % 2 == 1)) continue;
        ggml_tensor * enc = (i % every2 == 0) ? vl_txt : vl_img;
        dit.kv(C, dit.blk[i], enc, &Kc[i], &Vc[i]);
    }

    ggml_tensor * actions = t_x0;
    for (int64_t s = 0; s < num_steps; ++s) {
        ggml_tensor * temb = ggml_add(C, ggml_mul_mat(C, dit.te_l2W, ggml_silu(C, ggml_add(C, ggml_mul_mat(C, dit.te_l1W, t_tproj[s]), dit.te_l1b))), dit.te_l2b);
        ggml_tensor * a_emb = cat_linear(C, aex.ae_W1W, aex.ae_W1b, aex.embodiment_id, actions);
        ggml_tensor * x_w2  = ggml_silu(C, cat_linear(C, aex.ae_W2W, aex.ae_W2b, aex.embodiment_id, ggml_concat(C, a_emb, t_tau[s], 0)));
        ggml_tensor * af    = ggml_add(C, cat_linear(C, aex.ae_W3W, aex.ae_W3b, aex.embodiment_id, x_w2), ggml_view_2d(C, aex.pos_embd, E, AH, aex.pos_embd->nb[1], 0));
        ggml_tensor * sa = ggml_concat(C, state_features, af, 1);
        ggml_tensor * hh = sa;
        for (int64_t i = 0; i < dit_layers; ++i) {
            ggml_tensor * enc;
            if (dit_interleave && (i % 2 == 1)) enc = nullptr;
            else if (i % every2 == 0)           enc = vl_txt;
            else                                enc = vl_img;
            hh = dit.block(C, dit.blk[i], hh, temb, enc, Kc[i], Vc[i]);
        }
        ggml_tensor * po = ggml_add(C, ggml_mul_mat(C, dit.po1W, ggml_silu(C, temb)), dit.po1b);
        ggml_tensor * sh = ggml_view_1d(C, po, dit_hidden, 0), * sc = ggml_view_1d(C, po, dit_hidden, (size_t) dit_hidden * sizeof(float));
        ggml_tensor * hn = ggml_norm(C, hh, norm_out_eps);
        ggml_tensor * h_mod = ggml_add(C, ggml_add(C, hn, ggml_mul(C, hn, sc)), sh);
        ggml_tensor * model_output = ggml_add(C, ggml_mul_mat(C, dit.po2W, h_mod), dit.po2b);
        ggml_tensor * pred = cat_linear(C, aex.ad_l2W, aex.ad_l2b, aex.embodiment_id, ggml_relu(C, cat_linear(C, aex.ad_l1W, aex.ad_l1b, aex.embodiment_id, model_output)));
        ggml_tensor * vel  = ggml_cont(C, ggml_view_2d(C, pred, AD, AH, pred->nb[1], (size_t) (Nsa - AH) * pred->nb[1]));
        actions = ggml_add(C, actions, ggml_scale(C, vel, dt));
    }
    ggml_set_name(actions, "action_pred"); ggml_set_output(actions);

    gio.t_embeds=t_embeds; gio.t_pos=t_pos; gio.t_lmmask=t_lmmask; gio.t_state=t_state; gio.t_x0=t_x0;
    gio.t_ds[0]=t_ds[0]; gio.t_ds[1]=t_ds[1]; gio.t_ds[2]=t_ds[2];
    gio.t_img_idx=t_img_idx; gio.t_txt_idx=t_txt_idx; gio.t_tau=t_tau; gio.t_tproj=t_tproj; gio.actions=actions;

    ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
    ggml_build_forward_expand(gf, actions);
    return gf;
    });
    if (!built) { std::fprintf(stderr, "vla(gr00tn1d7): main graph build failed\n"); return {}; }

    MainIO & gio = mg.io();
    ggml_cgraph * gf = mg.graph();
    ggml_tensor * t_embeds = gio.t_embeds, * t_pos = gio.t_pos, * t_lmmask = gio.t_lmmask, * t_state = gio.t_state, * t_x0 = gio.t_x0;
    ggml_tensor * t_ds[3] = { gio.t_ds[0], gio.t_ds[1], gio.t_ds[2] };
    ggml_tensor * t_img_idx = gio.t_img_idx, * t_txt_idx = gio.t_txt_idx, * actions = gio.actions;
    std::vector<ggml_tensor*> & t_tau = gio.t_tau; std::vector<ggml_tensor*> & t_tproj = gio.t_tproj;

    ggml_backend_tensor_set(t_embeds, inputs_embeds.data(), 0, ggml_nbytes(t_embeds));
    {

        const int64_t llm_grid_h = image_target_size / patch_size / spatial_merge;
        const int64_t llm_grid_w = llm_grid_h;

        std::vector<int32_t> pp((size_t) 4 * SEQ, 0);
        int64_t st = 0, st_idx = 0;
        while (st < SEQ) {
            int64_t img_start = -1;
            for (int64_t i = st; i < SEQ; ++i) if (input_ids[i] == (int32_t) image_token_index) { img_start = i; break; }
            const int64_t text_end = (img_start < 0) ? SEQ : img_start;
            const int64_t text_len = text_end - st;
            for (int64_t i = 0; i < text_len; ++i) {
                const int32_t p = (int32_t) (i + st_idx);
                pp[0 * SEQ + (st + i)] = p;
                pp[1 * SEQ + (st + i)] = p;
                pp[2 * SEQ + (st + i)] = p;
            }
            if (img_start < 0) { st_idx += text_len; st = SEQ; break; }
            int64_t img_end = img_start;
            while (img_end < SEQ && input_ids[img_end] == (int32_t) image_token_index) ++img_end;
            const int64_t n_img_tokens = img_end - img_start;
            if (n_img_tokens % (llm_grid_h * llm_grid_w) != 0) {
                std::fprintf(stderr, "vla(gr00tn1d7): image run length %lld not a multiple of %lld (post-merge grid)\n",
                             (long long) n_img_tokens, (long long) (llm_grid_h * llm_grid_w));
                mg.release(); return {};
            }
            const int64_t this_t = n_img_tokens / (llm_grid_h * llm_grid_w);
            const int64_t image_offset = text_len + st_idx;
            for (int64_t tt = 0; tt < this_t; ++tt) {
                for (int64_t hh = 0; hh < llm_grid_h; ++hh) {
                    for (int64_t ww = 0; ww < llm_grid_w; ++ww) {
                        const int64_t k = (tt * llm_grid_h + hh) * llm_grid_w + ww;
                        const int64_t tok = img_start + k;
                        pp[0 * SEQ + tok] = (int32_t) (image_offset + tt);
                        pp[1 * SEQ + tok] = (int32_t) (image_offset + hh);
                        pp[2 * SEQ + tok] = (int32_t) (image_offset + ww);
                    }
                }
            }
            int64_t max_image_pos = this_t - 1;
            if (llm_grid_h - 1 > max_image_pos) max_image_pos = llm_grid_h - 1;
            if (llm_grid_w - 1 > max_image_pos) max_image_pos = llm_grid_w - 1;
            st_idx = image_offset + max_image_pos + 1;
            st = img_end;
        }

        std::memcpy(pp.data() + (size_t) 3 * SEQ, pp.data() + (size_t) 0 * SEQ, (size_t) SEQ * sizeof(int32_t));
        ggml_backend_tensor_set(t_pos, pp.data(), 0, ggml_nbytes(t_pos));
    }
    if (c_mask_seq != SEQ) { build_causal_mask(SEQ, c_mask); c_mask_seq = SEQ; }
    ggml_backend_tensor_set(t_lmmask, c_mask.data(), 0, ggml_nbytes(t_lmmask));
    { std::vector<float> st(max_state_dim, 0.0f); for (int64_t i = 0; i < max_state_dim; ++i) st[i] = in.state ? in.state[i] : 0.0f; ggml_backend_tensor_set(t_state, st.data(), 0, ggml_nbytes(t_state)); }
    ggml_backend_tensor_set(t_x0, x_init.data(), 0, ggml_nbytes(t_x0));
    if (inject_deepstack) for (int j = 0; j < 3; ++j) ggml_backend_tensor_set(t_ds[j], ds_pad[j].data(), 0, ggml_nbytes(t_ds[j]));
    ggml_backend_tensor_set(t_img_idx, image_pos_idx.data(), 0, ggml_nbytes(t_img_idx));
    if (t_txt_idx) ggml_backend_tensor_set(t_txt_idx, text_pos_idx.data(), 0, ggml_nbytes(t_txt_idx));
    for (int64_t s = 0; s < num_steps; ++s) {
        ggml_backend_tensor_set(t_tau[s],   c_tau[(size_t) s].data(),   0, ggml_nbytes(t_tau[s]));
        ggml_backend_tensor_set(t_tproj[s], c_tproj[(size_t) s].data(), 0, ggml_nbytes(t_tproj[s]));
    }

    const auto tc0 = std::chrono::steady_clock::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    const auto tc1 = std::chrono::steady_clock::now();
    if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "vla(gr00tn1d7): graph compute failed (%d)\n", (int) st); mg.release(); return {}; }
    stats.ms_inference = std::chrono::duration<float, std::milli>(tc1 - tc0).count();

    std::vector<float> out((size_t) AH * AD);
    ggml_backend_tensor_get(actions, out.data(), 0, out.size() * sizeof(float));

    if (const char * dump = std::getenv("VLA_GR00T_N17_DUMP")) {
        auto dump_t = [&](const char * name, ggml_tensor * t) {
            const int64_t n0 = t->ne[0], n1 = t->ne[1];
            std::vector<float> buf((size_t) n0 * n1);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump, name, (long long) n0, (long long) n1);
            FILE * fp = std::fopen(path, "wb");
            if (fp) { std::fwrite(buf.data(), sizeof(float), buf.size(), fp); std::fclose(fp); std::fprintf(stderr, "vla(gr00tn1d7): dumped %s shape=(%lld,%lld) to %s\n", name, (long long) n1, (long long) n0, path); }
        };
        dump_t("eagle",  eagle);
        dump_t("vl_embs", vl_embs);

        for (size_t li = 0; li < lm_h_dump.size(); ++li) { char nm[32]; std::snprintf(nm, sizeof(nm), "lm_h_%02zu", li); dump_t(nm, lm_h_dump[li]); }

        for (size_t vi = 0; vi < vlsa_dump.size(); ++vi) { char nm[32]; std::snprintf(nm, sizeof(nm), "vlsa_%02zu", vi); dump_t(nm, vlsa_dump[vi]); }

        if (inject_deepstack) {
            auto dump_host = [&](const char * name, const float * data, int64_t n0, int64_t n1) {
                char path[1024]; std::snprintf(path, sizeof(path), "%s_%s_%lldx%lld.f32", dump, name, (long long) n0, (long long) n1);
                FILE * fp = std::fopen(path, "wb");
                if (fp) { std::fwrite(data, sizeof(float), (size_t) n0 * n1, fp); std::fclose(fp); std::fprintf(stderr, "vla(gr00tn1d7): dumped %s shape=(%lld,%lld) to %s\n", name, (long long) n1, (long long) n0, path); }
            };

            for (int64_t v = 0; v < n_views; ++v) {
                char nm[32]; std::snprintf(nm, sizeof(nm), "ds0_view%lld", (long long) v);
                dump_host(nm, ds_host[0].data() + v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "ds1_view%lld", (long long) v);
                dump_host(nm, ds_host[1].data() + v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "ds2_view%lld", (long long) v);
                dump_host(nm, ds_host[2].data() + v * K * H, H, K);
                std::snprintf(nm, sizeof(nm), "vit_view%lld", (long long) v);
                dump_host(nm, img_emb_host.data() + v * K * H, H, K);
            }
        }
    }
    if (!use_cache) mg.release();
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}
