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
#include "layers/ffn.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "model.h"
#include "modules/action_expert.h"
#include "modules/dit_head.h"
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

struct Gr00tN1d6ModelArch : public ModelArchBase {
    Gr00tN1d6ModelArch() : ModelArchBase(Arch::GR00T_N1_6) {}
    ~Gr00tN1d6ModelArch() override;

    // Opened once at load: reopening per predict re-parses the whole GGUF header.
    gguf_reader           io{"gr00tn1d6"};
    ggml_backend_t        backend     = nullptr;
    int                   n_threads   = default_cpu_threads();
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_type             matmul_type = GGML_TYPE_F32;
    scratch_ctx           vision_scratch;
    scratch_ctx           merge_scratch;

    struct MainKey {
        int64_t seq=-1, n_img=-1, seq_txt=-1, nsteps=-1;
        bool operator==(const MainKey & o) const {
            return seq==o.seq && n_img==o.n_img && seq_txt==o.seq_txt && nsteps==o.nsteps;
        }
    };
    struct MainIO {
        ggml_tensor *t_embeds=nullptr,*t_pos=nullptr,*t_lmmask=nullptr,*t_state=nullptr,*t_x0=nullptr;
        ggml_tensor *t_img_idx=nullptr,*t_txt_idx=nullptr,*actions=nullptr;
        std::vector<ggml_tensor*> t_tau, t_tproj;
    };
    graph_cache<MainKey, MainIO> main_graph;

    SigLipTower  vit;
    Qwen3LM      lm;
    ActionExpert aex;
    DitHead      dit;
    ggml_tensor *mm_ln_w=nullptr, *mm_ln_b=nullptr;
    ggml_tensor *mm_fc1_w=nullptr, *mm_fc1_b=nullptr, *mm_fc2_w=nullptr, *mm_fc2_b=nullptr;
    ggml_tensor *vlln_w=nullptr, *vlln_b=nullptr;

    int64_t vit_layers=27, vit_inter=4304, image_size=224, patch_size=14;
    int64_t vit_num_patches=256, n_img_tokens=64, vit_pixel_shuffle=2, mlp_inner=4608;
    int64_t lm_inter=6144, vocab=151680, image_token_index=151669;
    int64_t bb_embed_dim=2048, in_embed_dim=1536, dit_interleave=1, attend_text_every_n=2;
    int64_t action_horizon=50, action_dim=128, max_state_dim=128;
    int64_t num_steps=4, num_buckets=1000, max_embodiments=32, max_seq_len=1024;
    float   vlln_eps=1e-5f, connector_ln_eps=1e-5f;

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

bool load_config(const gguf_reader & g, Gr00tN1d6ModelArch & m, Config & cfg) {
    auto U  = [&](const char * k, int64_t & dst) { if (g.has(k)) dst = (int64_t) g.u32(k); };
    auto F  = [&](const char * k, float & dst)   { if (g.has(k)) dst = g.f32(k); };
    auto fk = [&](const char * s) { static char b[64]; std::snprintf(b, sizeof(b), "gr00t_n1_6.%s", s); return b; };

    U(fk("vit_hidden"       ), m.vit.enc.cfg.hidden);
    U(fk("vit_layers"       ), m.vit_layers);
    U(fk("vit_heads"        ), m.vit.enc.cfg.heads);
    U(fk("vit_inter"        ), m.vit_inter);
    U(fk("image_size"       ), m.image_size);
    U(fk("patch_size"       ), m.patch_size);
    U(fk("vit_num_patches"  ), m.vit_num_patches);
    U(fk("n_img_tokens"     ), m.n_img_tokens);
    U(fk("vit_pixel_shuffle"), m.vit_pixel_shuffle);
    U(fk("mlp_connector_inner"), m.mlp_inner);
    U(fk("lm_hidden"        ), m.lm.cfg.hidden);
    U(fk("lm_layers_used"   ), m.lm.cfg.layers);
    U(fk("lm_q_heads"       ), m.lm.cfg.n_q);
    U(fk("lm_kv_heads"      ), m.lm.cfg.n_kv);
    U(fk("lm_head_dim"      ), m.lm.cfg.head_dim);
    U(fk("lm_inter"         ), m.lm_inter);
    U(fk("vocab_size"       ), m.vocab);
    U(fk("image_token_index"), m.image_token_index);
    U(fk("backbone_embedding_dim"), m.bb_embed_dim);
    U(fk("input_embedding_dim"   ), m.in_embed_dim);
    U(fk("dit_hidden"       ), m.dit.cfg.hidden);
    U(fk("dit_heads"        ), m.dit.cfg.heads);
    U(fk("dit_head_dim"     ), m.dit.cfg.head_dim);
    U(fk("dit_layers"       ), m.dit.cfg.layers);
    U(fk("dit_interleave"   ), m.dit_interleave);
    U(fk("attend_text_every_n_blocks"), m.attend_text_every_n);
    U(fk("action_horizon"   ), m.action_horizon);
    U(fk("action_dim"       ), m.action_dim);
    U(fk("max_state_dim"    ), m.max_state_dim);
    U(fk("num_inference_timesteps"), m.num_steps);
    U(fk("num_timestep_buckets"   ), m.num_buckets);
    U(fk("max_num_embodiments"    ), m.max_embodiments);
    U(fk("max_seq_len"      ), m.max_seq_len);

    F(fk("vit_ln_eps"       ), m.vit.enc.cfg.ln_eps);
    F(fk("lm_rms_eps"       ), m.lm.cfg.rms_eps);
    F(fk("ln_eps"           ), m.dit.cfg.ln_eps);
    F(fk("norm_out_eps"     ), m.dit.cfg.norm_out_eps);
    F(fk("vlln_eps"         ), m.vlln_eps);
    F(fk("connector_ln_eps" ), m.connector_ln_eps);

    if (g.has(fk("lm_rope_theta")))
        m.lm.cfg.rope.freq_base = (float) g.f64(fk("lm_rope_theta"));

    m.vit.enc.cfg.head_dim = m.vit.enc.cfg.hidden/m.vit.enc.cfg.heads;
    m.lm.cfg.rope.n_dims   = (int) m.lm.cfg.head_dim;

    m.aex.embodiment_id = 20;
    {
        const std::string js = g.str(fk("embodiment_id_mapping"));
        auto lookup = [&](const char * key) -> long {
            const std::string k = std::string("\"")+key+"\"";
            size_t p = js.find(k);
            if (p == std::string::npos)
                return -1;
            p = js.find(':', p+k.size());
            if (p == std::string::npos)
                return -1;
            return std::strtol(js.c_str()+p+1, nullptr, 10);
        };

        const long gr1 = lookup("gr1");
        if (gr1 >= 0)
            m.aex.embodiment_id = gr1;

        if (const char * e = std::getenv("VLA_GR00T_EMBODIMENT")) {
            char * end = nullptr;
            const long v = std::strtol(e, &end, 10);
            if (end && *end == '\0') {
                m.aex.embodiment_id = v;
            } else {
                const long id = lookup(e);
                if (id >= 0)
                    m.aex.embodiment_id = id;
                else std::fprintf(stderr, "vla(gr00tn1d6): embodiment tag '%s' not in embodiment_id_mapping; using id %lld\n", e, (long long) m.aex.embodiment_id);
            }
        }
    }
    if (m.aex.embodiment_id < 0 || m.aex.embodiment_id >= m.max_embodiments) {
        std::fprintf(stderr, "vla(gr00tn1d6): embodiment id %lld out of range [0,%lld)\n",
                     (long long) m.aex.embodiment_id, (long long) m.max_embodiments);
        return false;
    }

    // pixel_shuffle_back writes (grid/shuffle)^2 tokens into a buffer sized from
    // n_img_tokens, so the KV has to agree with the grid it is derived from.
    if (m.patch_size <= 0 || m.vit_pixel_shuffle <= 0 || m.image_size%m.patch_size != 0 ||
        (m.image_size/m.patch_size)%m.vit_pixel_shuffle != 0) {
        std::fprintf(stderr, "vla(gr00tn1d6): image %lld / patch %lld / shuffle %lld do not divide evenly\n",
                     (long long) m.image_size, (long long) m.patch_size, (long long) m.vit_pixel_shuffle);
        return false;
    }
    {
        const int64_t g2 = (m.image_size/m.patch_size)/m.vit_pixel_shuffle;
        if (m.n_img_tokens != g2*g2) {
            std::fprintf(stderr, "vla(gr00tn1d6): n_img_tokens %lld does not match the %lldx%lld shuffled grid\n",
                         (long long) m.n_img_tokens, (long long) g2, (long long) g2);
            return false;
        }
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

Gr00tN1d6ModelArch::~Gr00tN1d6ModelArch() {
    if (weight_buf)
        ggml_backend_buffer_free(weight_buf);
    if (ctx_weights)
        ggml_free(ctx_weights);
    if (backend)
        ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> gr00t_n1_6_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string&,
                                                 const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(gr00tn1d6): note - mmproj '%s' is ignored (the vision tower is bundled in the combined GGUF)\n", mmproj_path.c_str());

    auto m = std::make_unique<Gr00tN1d6ModelArch>();
    m->matmul_type           = opts.weight_dtype.value_or(GGML_TYPE_BF16);
    m->lm.cfg.rope.freq_base = 1000000.0f;

    if (!m->io.open(ckpt_path))
        return nullptr;
    gguf_reader & g = m->io;
    if (!g.has("gr00t_n1_6.architecture")) {
        std::fprintf(stderr, "vla(gr00tn1d6): %s is not a gr00t_n1_6 GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(g, *m, m->cfg))
        return nullptr;

    std::printf("vla(gr00tn1d6): vit=%lldd×%lldL×%lldh (Linear patch embed)  pixel_shuffle÷%lld ⇒ n_img_tok=%lld  mlp1=LN(%lld)→Linear→GELU→Linear  "
                "lm=Qwen3 %lldd×%lldL (%lldq/%lldkv×%lld)  dit=AlternateVLDiT %lldL×%lldh×%lld(inner %lld) attend_text_every_n=%lld  in_emb=%lld  "
                "horizon=%lld action_dim=%lld max_state=%lld N_steps=%lld  embodiment=%lld  resident=%s\n",
                (long long) m->vit.enc.cfg.hidden, (long long) m->vit_layers, (long long) m->vit.enc.cfg.heads, (long long) m->vit_pixel_shuffle, (long long) m->n_img_tokens, (long long) m->mlp_inner,
                (long long) m->lm.cfg.hidden, (long long) m->lm.cfg.layers, (long long) m->lm.cfg.n_q, (long long) m->lm.cfg.n_kv, (long long) m->lm.cfg.head_dim,
                (long long) m->dit.cfg.layers, (long long) m->dit.cfg.heads, (long long) m->dit.cfg.head_dim, (long long) m->dit.cfg.hidden, (long long) m->attend_text_every_n, (long long) m->in_embed_dim,
                (long long) m->action_horizon, (long long) m->action_dim, (long long) m->max_state_dim, (long long) m->num_steps, (long long) m->aex.embodiment_id,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    const Backend b = backend_init("vla(gr00tn1d6)", m->n_threads);
    if (!b.handle)
        return nullptr;
    m->backend = b.handle;

    ggml_init_params wp = { (size_t) 32*1024*1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) {
        std::fprintf(stderr, "vla(gr00tn1d6): ggml_init(ctx_weights) failed\n");
        return nullptr;
    }

    WeightLoader L("gr00tn1d6", g, m->ctx_weights, m->matmul_type);

    m->vit.declare(L, "vit", m->vit_layers, /*patch_embd_is_gemm=*/true);

    m->mm_ln_w  = L.f32 ("mm.ln.weight");
    m->mm_ln_b  = L.f32 ("mm.ln.bias");
    m->mm_fc1_w = L.gemm("mm.fc1.weight");
    m->mm_fc1_b = L.f32 ("mm.fc1.bias");
    m->mm_fc2_w = L.gemm("mm.fc2.weight");
    m->mm_fc2_b = L.f32 ("mm.fc2.bias");

    m->lm.declare(L, "vlm");

    m->vlln_w = L.f32("aex.vlln.weight");
    m->vlln_b = L.f32("aex.vlln.bias");

    m->aex.declare(L, "aex");
    m->dit.declare(L, "aex.dit");

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;

    std::printf("vla(gr00tn1d6): weights resident in %.2f GiB (%s) - incl. SigLIP2 vision tower; embodiment id %lld\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0),
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16", (long long) m->aex.embodiment_id);
    return m;
}

std::vector<float> Gr00tN1d6ModelArch::predict(const Inputs& in) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = Stats{};

    const int64_t H         = lm.cfg.hidden;
    const int64_t K         = n_img_tokens;
    const int64_t E         = in_embed_dim;
    const int64_t grid      = image_size/patch_size;
    const int64_t r         = vit_pixel_shuffle;
    const int64_t n_patches = grid*grid;
    const int64_t patch_dim = 3*patch_size*patch_size;
    const int64_t c4        = vit.enc.cfg.hidden*r*r;
    const int64_t AD        = action_dim;
    const int64_t AH        = action_horizon;
    const int64_t Nsa       = 1+AH;

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
        if (!VC) { std::fprintf(stderr, "vla(gr00tn1d6): ggml_init(vision ctx A) failed\n"); return {}; }

        ggml_tensor * t_patches = ggml_new_tensor_3d(VC, GGML_TYPE_F32, patch_dim, n_patches, n_views);
        ggml_set_input(t_patches);
        ggml_tensor * post_ln = vit.build(VC, vit.embed_patches(VC, t_patches), n_patches, n_views);
        ggml_set_output(post_ln);

        ggml_cgraph * vgA = ggml_new_graph_custom(VC, 8192, false);
        ggml_build_forward_expand(vgA, post_ln);
        if (!vision_scratch.alloc(backend, vgA)) { std::fprintf(stderr, "vla(gr00tn1d6): vision gallocr A alloc failed\n"); return {}; }

        ggml_context * MC = merge_scratch.reset((size_t) 16*1024*1024);
        if (!MC) { std::fprintf(stderr, "vla(gr00tn1d6): ggml_init(vision ctx B) failed\n"); return {}; }

        ggml_tensor * t_shuf = ggml_new_tensor_3d(MC, GGML_TYPE_F32, c4, K, n_views);
        ggml_set_input(t_shuf);
        ggml_tensor * mln = layer_norm(MC, t_shuf, mm_ln_w, mm_ln_b, connector_ln_eps);
        ggml_tensor * vit_embeds = ffn_gelu_erf(MC, mm_fc1_w, mm_fc1_b, mm_fc2_w, mm_fc2_b, mln);
        ggml_set_output(vit_embeds);

        ggml_cgraph * vgB = ggml_new_graph(MC);
        ggml_build_forward_expand(vgB, vit_embeds);
        if (!merge_scratch.alloc(backend, vgB)) { std::fprintf(stderr, "vla(gr00tn1d6): vision gallocr B alloc failed\n"); return {}; }

        const auto tv0 = std::chrono::steady_clock::now();
        std::vector<float> patches;
        std::vector<float> patches_all((size_t) patch_dim*n_patches*n_views);
        std::vector<float> post_ln_host((size_t) vit.enc.cfg.hidden*n_patches*n_views);
        std::vector<float> shuf_host((size_t) c4*K*n_views);

        bool vok = true;
        for (int64_t v = 0; v < n_views && vok; ++v) {
            if (!preprocess_image_patches("gr00tn1d6", in.images[v], image_size, patch_size, patches)) {
                vok = false;
                break;
            }
            std::memcpy(patches_all.data()+(size_t) v*patch_dim*n_patches, patches.data(), patches.size()*sizeof(float));
        }
        if (vok) {
            ggml_backend_tensor_set(t_patches, patches_all.data(), 0, ggml_nbytes(t_patches));
            if (ggml_backend_graph_compute(backend, vgA) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(gr00tn1d6): vision compute A failed\n");
                vok = false;
            }
        }
        if (vok) {
            ggml_backend_tensor_get(post_ln, post_ln_host.data(), 0, ggml_nbytes(post_ln));
            for (int64_t v = 0; v < n_views; ++v)
                pixel_shuffle_back(post_ln_host.data()+(size_t) v*vit.enc.cfg.hidden*n_patches, grid, vit.enc.cfg.hidden, r,
                                   shuf_host.data()+(size_t) v*c4*K);

            ggml_backend_tensor_set(t_shuf, shuf_host.data(), 0, ggml_nbytes(t_shuf));
            if (ggml_backend_graph_compute(backend, vgB) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(gr00tn1d6): vision compute B failed\n");
                vok = false;
            }
        }
        if (vok)
            ggml_backend_tensor_get(vit_embeds, img_emb_host.data(), 0, ggml_nbytes(vit_embeds));
        stats.ms_vision = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-tv0).count();
        if (!vok) return {};
        img_emb_ptr = img_emb_host.data();
    } else {
        std::fprintf(stderr, "vla(gr00tn1d6): no images and no precomputed_img_emb in the request\n");
        return {};
    }
    const int64_t n_img = n_views*K;

    Prompt prompt;
    if (!build_prompt("gr00tn1d6", in, n_img, (int32_t) image_token_index, max_seq_len, prompt)) return {};
    const int64_t SEQ     = prompt.len();
    const int64_t SEQ_TXT = prompt.n_text();

    std::vector<float> inputs_embeds;
    if (!fetch_embeds("gr00tn1d6", io, prompt, img_emb_ptr, H, inputs_embeds)) return {};

    std::vector<float> x_init;
    init_noise(in, (size_t) AH*AD, x_init);

    const MainKey mkey{ SEQ, n_img, SEQ_TXT, num_steps };
    const bool built = main_graph.ensure(backend, mkey, (size_t) 256*1024*1024,
                                         [&](ggml_context * C, MainIO & gio) -> ggml_cgraph * {
        ggml_tensor * t_embeds = ggml_new_tensor_2d(C, GGML_TYPE_F32, H, SEQ);           ggml_set_input(t_embeds);
        ggml_tensor * t_pos    = ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ);              ggml_set_input(t_pos);
        ggml_tensor * t_lmmask = ggml_new_tensor_2d(C, GGML_TYPE_F32, SEQ, SEQ);         ggml_set_input(t_lmmask);
        ggml_tensor * t_state  = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_state_dim, 1); ggml_set_input(t_state);
        ggml_tensor * t_x0     = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, AH);           ggml_set_input(t_x0);

        ggml_tensor * t_img_idx = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_img);
        ggml_set_input(t_img_idx);
        ggml_tensor * t_txt_idx = (SEQ_TXT > 0) ? ggml_new_tensor_1d(C, GGML_TYPE_I32, SEQ_TXT) : nullptr;
        if (t_txt_idx)
            ggml_set_input(t_txt_idx);

        std::vector<ggml_tensor *> t_tau(num_steps), t_tproj(num_steps);
        for (int64_t s = 0; s < num_steps; ++s) {
            t_tau[s]   = ggml_new_tensor_2d(C, GGML_TYPE_F32, E, AH); ggml_set_input(t_tau[s]);
            t_tproj[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, 256);   ggml_set_input(t_tproj[s]);
        }

        ggml_tensor * eagle   = lm.build(C, t_embeds, t_pos, t_lmmask, SEQ);
        ggml_tensor * vl_embs = layer_norm(C, eagle, vlln_w, vlln_b, vlln_eps);
        ggml_tensor * vl_img  = ggml_get_rows(C, vl_embs, t_img_idx);
        ggml_tensor * vl_txt  = t_txt_idx ? ggml_get_rows(C, vl_embs, t_txt_idx) : vl_img;

        ggml_tensor * state_features = aex.encode_state(C, t_state);

        const float   dt     = 1.0f/(float) num_steps;
        const int64_t every2 = 2*attend_text_every_n;

        std::vector<ggml_tensor *> Kc(dit.cfg.layers, nullptr), Vc(dit.cfg.layers, nullptr);
        for (int64_t i = 0; i < dit.cfg.layers; ++i) {
            if (dit_interleave && (i%2 == 1))
                continue;
            dit.kv(C, dit.blk[i], (i%every2 == 0) ? vl_txt : vl_img, &Kc[i], &Vc[i]);
        }

        ggml_tensor * actions = t_x0;
        for (int64_t s = 0; s < num_steps; ++s) {
            ggml_tensor * temb = dit.time_emb(C, t_tproj[s]);
            ggml_tensor * af   = aex.encode_action(C, actions, t_tau[s], E, AH);
            ggml_tensor * hh   = ggml_concat(C, state_features, af, 1);

            for (int64_t i = 0; i < dit.cfg.layers; ++i) {
                ggml_tensor * enc;
                if (dit_interleave && (i%2 == 1))
                    enc = nullptr;
                else if (i%every2 == 0)           enc = vl_txt;
                else
                    enc = vl_img;
                hh = dit.block(C, dit.blk[i], hh, temb, enc, Kc[i], Vc[i]);
            }

            ggml_tensor * pred = aex.decode(C, dit.proj_out(C, hh, temb));
            ggml_tensor * vel  = ggml_cont(C, ggml_view_2d(C, pred, AD, AH, pred->nb[1], (size_t)(Nsa-AH)*pred->nb[1]));
            actions = ggml_add(C, actions, ggml_scale(C, vel, dt));
        }
        ggml_set_name(actions, "action_pred");
        ggml_set_output(actions);

        gio.t_embeds=t_embeds; gio.t_pos=t_pos; gio.t_lmmask=t_lmmask; gio.t_state=t_state; gio.t_x0=t_x0;
        gio.t_img_idx=t_img_idx; gio.t_txt_idx=t_txt_idx; gio.t_tau=t_tau; gio.t_tproj=t_tproj; gio.actions=actions;

        ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
        ggml_build_forward_expand(gf, actions);
        return gf;
    });
    if (!built) { std::fprintf(stderr, "vla(gr00tn1d6): main graph build failed\n"); return {}; }

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
    ggml_backend_tensor_set(gio.t_img_idx, prompt.image_pos.data(), 0, ggml_nbytes(gio.t_img_idx));
    if (gio.t_txt_idx)
        ggml_backend_tensor_set(gio.t_txt_idx, prompt.text_pos.data(), 0, ggml_nbytes(gio.t_txt_idx));

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
        std::fprintf(stderr, "vla(gr00tn1d6): graph compute failed (%d)\n", (int) status);
        return {};
    }
    stats.ms_inference = std::chrono::duration<float, std::milli>(tc1-tc0).count();

    std::vector<float> out((size_t) AH*AD);
    ggml_backend_tensor_get(gio.actions, out.data(), 0, out.size()*sizeof(float));
    stats.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-t0).count();
    return out;
}

}
