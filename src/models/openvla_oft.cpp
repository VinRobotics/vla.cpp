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
#include "modules/preprocess.h"
#include "modules/dual_tower.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "backend.h"
#include "gguf_reader.h"
#include "scratch_ctx.h"
#include "layers/linear.h"
#include "layers/norm.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace vla {
namespace {

struct LMLayerW  { ggml_tensor *attn_norm,*Wq,*Wk,*Wv,*Wo,*ffn_norm,*Wg,*Wu,*Wd; };
struct HeadBlkW  { ggml_tensor *lnw,*lnb,*linw,*linb; };

}

struct OpenVlaOftModelArch : public ModelArchBase {
    OpenVlaOftModelArch() : ModelArchBase(Arch::OPENVLA_OFT) {}
    ~OpenVlaOftModelArch() override {
        if (weight_buf)
            ggml_backend_buffer_free(weight_buf);
        if (ctx_weights)
            ggml_free(ctx_weights);
        if (backend)
            ggml_backend_free(backend);
    }

    ggml_backend_t backend = nullptr;
    int n_threads = default_cpu_threads();
    ggml_context * ctx_weights = nullptr;
    graph_cache<int64_t, DualTower::VisIO> vision_graph;

    struct MainKey {
        int64_t seq=-1, n_views=-1, n_lang=-1;
        bool operator==(const MainKey & o) const {
            return seq==o.seq && n_views==o.n_views && n_lang==o.n_lang;
        }
    };
    struct MainIO {
        ggml_tensor *t_ids=nullptr,*t_state=nullptr,*t_proj=nullptr,*act0=nullptr,*t_pos=nullptr,*norm_actions=nullptr;
        bool consts=false;
    };
    graph_cache<MainKey, MainIO> main_graph;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_type mt = GGML_TYPE_BF16;

    int64_t lm_hidden=4096,lm_layers=32,n_q=32,n_kv=32,lm_head_dim=128,vocab=32064;
    float   lm_rope_base=1e4f, lm_rms_eps=1e-6f;
    int64_t chunk=8,action_dim=7,proprio_dim=8,head_blocks=2;
    float   head_ln_eps=1e-5f;
    int64_t stop_id=2;

    DualTower vis;
    ggml_tensor *token_embd,*lm_out_norm; std::vector<LMLayerW> lm;
    ggml_tensor *pp_fc1w,*pp_fc1b,*pp_fc2w,*pp_fc2b;
    ggml_tensor *h_ln1w,*h_ln1b,*h_fc1w,*h_fc1b,*h_ln2w,*h_ln2b,*h_fc2w,*h_fc2b; std::vector<HeadBlkW> hblk;

    Q99Stats act_stats;

    std::vector<float> predict(const Inputs& in) override;
};

std::unique_ptr<ModelArchBase> openvla_oft_create(const std::string& mmproj_path,
                                                  const std::string& ckpt_path,
                                                  const std::string&,
                                                  const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(openvla_oft): note - mmproj '%s' ignored (vision baked into combined GGUF)\n", mmproj_path.c_str());
    auto m = std::make_unique<OpenVlaOftModelArch>();
    m->mt = opts.weight_dtype.value_or(vla::default_weight_dtype(GGML_TYPE_BF16));

    gguf_reader g("openvla_oft");
    if (!g.open(ckpt_path))
        return nullptr;
    if (!g.has("openvla_oft.architecture")) {
        std::fprintf(stderr, "vla(openvla_oft): not an openvla_oft GGUF\n");
        return nullptr;
    }

    auto U=[&](const char*k,int64_t&d){ if(g.has(k)) d=(int64_t)g.u32(k); };
    auto F=[&](const char*k,float&d){ if(g.has(k)) d=g.f32(k); };
    m->vis.read_config(g, "openvla_oft");
    U("openvla_oft.lm.hidden",m->lm_hidden); U("openvla_oft.lm.layers",m->lm_layers);
    U("openvla_oft.lm.q_heads",m->n_q); U("openvla_oft.lm.kv_heads",m->n_kv); U("openvla_oft.lm.head_dim",m->lm_head_dim);
    U("openvla_oft.lm.vocab",m->vocab);
    F("openvla_oft.lm.rope_theta",m->lm_rope_base); F("openvla_oft.lm.rms_eps",m->lm_rms_eps);
    U("openvla_oft.action.chunk",m->chunk); U("openvla_oft.action.action_dim",m->action_dim);
    U("openvla_oft.action.proprio_dim",m->proprio_dim);
    U("openvla_oft.action.head_blocks",m->head_blocks); F("openvla_oft.action.head_ln_eps",m->head_ln_eps);
    // No empty_id: the reference zeroes the action-slot embeddings instead
    // (modeling_prismatic.py:891), which is what act0 below does.
    U("openvla_oft.tokens.stop_id",m->stop_id);
    if (m->vis.patch_size <= 0 || m->n_q <= 0) {
        std::fprintf(stderr, "vla(openvla_oft): bad geometry (patch_size %lld q_heads %lld)\n",
                     (long long) m->vis.patch_size, (long long) m->n_q);
        return nullptr;
    }
    if (m->lm_head_dim==0)
        m->lm_head_dim = m->lm_hidden/m->n_q;

    if (g.has("openvla_oft.statistics_json")) {
        if (!m->act_stats.parse(g.str("openvla_oft.statistics_json"), "VLA_OPENVLA_OFT_UNNORM_KEY", "openvla_oft", m->action_dim))
            {
                std::fprintf(stderr, "vla(openvla_oft): failed to parse statistics_json\n");
                return nullptr;
            }
        std::printf("vla(openvla_oft): unnorm suite = %s (q99 dim %zu)\n", m->act_stats.suite.c_str(), m->act_stats.q99.size());
    }

    {
        const Backend b = backend_init("vla(openvla_oft)", m->n_threads);
        if (!b.handle) {
            return nullptr;
        }
        m->backend = b.handle;
    }

    ggml_init_params wp = { (size_t)64*1024*1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    WeightLoader L("openvla_oft", g, m->ctx_weights, m->mt);
    auto mm  = [&](const char * n) { return L.gemm("%s", n); };
    auto f32 = [&](const char * n) { return L.f32("%s", n); };

    m->vis.declare(L);

    m->token_embd=mm("token_embd.weight"); m->lm_out_norm=f32("lm.output_norm.weight");
    m->lm.resize(m->lm_layers);
    for(int i=0;i<m->lm_layers;++i){ auto&w=m->lm[i]; char b[64];
        auto N=[&](const char*s){ std::snprintf(b,sizeof(b),"lm.blk.%d.%s",i,s); return (const char*)b; };
        w.attn_norm=f32(N("attn_norm.weight")); w.ffn_norm=f32(N("ffn_norm.weight"));
        w.Wq=mm(N("attn_q.weight")); w.Wk=mm(N("attn_k.weight")); w.Wv=mm(N("attn_v.weight")); w.Wo=mm(N("attn_o.weight"));
        w.Wg=mm(N("ffn_gate.weight")); w.Wu=mm(N("ffn_up.weight")); w.Wd=mm(N("ffn_down.weight")); }

    m->pp_fc1w=mm("aex.proprio.fc1.weight"); m->pp_fc1b=f32("aex.proprio.fc1.bias");
    m->pp_fc2w=mm("aex.proprio.fc2.weight"); m->pp_fc2b=f32("aex.proprio.fc2.bias");

    m->h_ln1w=f32("aex.head.ln1.weight"); m->h_ln1b=f32("aex.head.ln1.bias");
    m->h_fc1w=mm("aex.head.fc1.weight"); m->h_fc1b=f32("aex.head.fc1.bias");
    m->h_ln2w=f32("aex.head.ln2.weight"); m->h_ln2b=f32("aex.head.ln2.bias");
    m->h_fc2w=mm("aex.head.fc2.weight"); m->h_fc2b=f32("aex.head.fc2.bias");
    m->hblk.resize(m->head_blocks);
    for(int i=0;i<m->head_blocks;++i){ auto&w=m->hblk[i]; char b[64];
        auto N=[&](const char*s){ std::snprintf(b,sizeof(b),"aex.head.blk.%d.%s",i,s); return (const char*)b; };
        w.lnw=f32(N("ln.weight")); w.lnb=f32(N("ln.bias")); w.linw=mm(N("lin.weight")); w.linb=f32(N("lin.bias")); }

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;

    std::printf("vla(openvla_oft): weights resident %.2f GiB (%s) - DINOv2+SigLIP towers + Llama-2-7B + MLPResNet L1 head\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0), dtype_name(m->mt));

    m->cfg.n_suffix = m->chunk; m->cfg.max_action_dim = m->action_dim;
    m->cfg.real_action_dim = m->action_dim; m->cfg.real_state_dim = m->proprio_dim;
    m->cfg.max_state_dim = m->proprio_dim; m->cfg.n_img = m->vis.n_patches; m->cfg.hidden = m->lm_hidden;
    m->cfg.n_lang = 512;
    return m;
}

std::vector<float> OpenVlaOftModelArch::predict(const Inputs& in) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    stats = Stats{};
    const int64_t NP=vis.n_patches, HC=lm_hidden;
    const int64_t n_views = in.n_images;
    if (in.precomputed_img_emb) { std::fprintf(stderr, "vla(openvla_oft): precomputed_img_emb is not supported; the DINOv2+SigLIP tower is baked into the GGUF, pass raw images\n"); return {}; }
    if (in.n_lang < 1 || !in.lang_tokens) { std::fprintf(stderr, "vla(openvla_oft): need >=1 lang token\n"); return {}; }
    if (n_views < 1) { std::fprintf(stderr, "vla(openvla_oft): need >=1 image view\n"); return {}; }
    if (!in.images) { std::fprintf(stderr, "vla(openvla_oft): n_images=%d but the images pointer is null\n", in.n_images); return {}; }

    const int64_t NPATCH = NP * n_views;
    ggml_tensor*proj=nullptr;
    {
        const auto tv=clock::now();
        proj=vis.encode(backend,vision_graph,in,"openvla_oft");
        if(!proj)
            return {};
        stats.ms_vision = std::chrono::duration<float,std::milli>(clock::now()-tv).count();
    }

    const int64_t L = in.n_lang;
    // ggml_get_rows does not bound-check, so reject out-of-range tokens here.
    for (int64_t i=0; i<L; ++i)
        if (in.lang_tokens[i] < 0 || in.lang_tokens[i] >= vocab) {
            std::fprintf(stderr, "vla(openvla_oft): token %d out of vocab\n", in.lang_tokens[i]);
            return {};
        }
    if ((int64_t) stop_id < 0 || (int64_t) stop_id >= vocab) {
        std::fprintf(stderr, "vla(openvla_oft): stop_id %lld out of vocab\n", (long long) stop_id);
        return {};
    }
    const int64_t n_act = chunk * action_dim;
    const int64_t NUM_PATCHES = NPATCH+1;
    const int64_t NUM_PROMPT_TOKENS = L-1;
    const int64_t ACT_START = NUM_PATCHES+NUM_PROMPT_TOKENS;
    const int64_t SEQ = 1+NUM_PATCHES+(L-1)+n_act+1;
    const auto ti=clock::now();
    // LM + action head graph depends only on the sequence layout.
    const MainKey mkey{ SEQ, n_views, L };
    const bool built = main_graph.ensure(backend, mkey, (size_t)256*1024*1024,
                                         [&](ggml_context*C, MainIO & gio)->ggml_cgraph*{
    ggml_tensor*t_ids=ggml_new_tensor_1d(C,GGML_TYPE_I32,L+1); ggml_set_input(t_ids);
    ggml_tensor*emb=ggml_get_rows(C,token_embd,t_ids);
    if(emb->type!=GGML_TYPE_F32)
        emb=ggml_cast(C,emb,GGML_TYPE_F32);
    ggml_tensor*bos =ggml_cont(C,ggml_view_2d(C,emb,HC,1,emb->nb[1],0));
    ggml_tensor*rest=ggml_cont(C,ggml_view_2d(C,emb,HC,L-1,emb->nb[1],emb->nb[1]));
    ggml_tensor*stop=ggml_cont(C,ggml_view_2d(C,emb,HC,1,emb->nb[1],L*emb->nb[1]));

    ggml_tensor*t_state=ggml_new_tensor_1d(C,GGML_TYPE_F32,proprio_dim); ggml_set_input(t_state);
    ggml_tensor*pf=ggml_add(C,ggml_mul_mat(C,pp_fc1w,t_state),pp_fc1b); pf=ggml_gelu_erf(C,pf);
    pf=ggml_add(C,ggml_mul_mat(C,pp_fc2w,pf),pp_fc2b); ggml_tensor*pvec=ggml_reshape_2d(C,pf,HC,1);

    ggml_tensor*t_proj=ggml_new_tensor_2d(C,GGML_TYPE_F32,HC,NPATCH); ggml_set_input(t_proj);
    ggml_tensor*patches=ggml_concat(C,t_proj,pvec,1);

    ggml_tensor*act0=ggml_new_tensor_2d(C,GGML_TYPE_F32,HC,n_act); ggml_set_input(act0); ggml_set_output(act0);

    ggml_tensor*seq=ggml_concat(C,bos,patches,1);
    seq=ggml_concat(C,seq,rest,1);
    seq=ggml_concat(C,seq,act0,1);
    seq=ggml_concat(C,seq,stop,1);

    ggml_tensor*t_pos=ggml_new_tensor_1d(C,GGML_TYPE_I32,SEQ); ggml_set_input(t_pos); ggml_set_output(t_pos);
    const float lsc=1.0f/std::sqrt((float)lm_head_dim);
    ggml_tensor*x=seq;
    for(int i=0;i<lm_layers;++i){ const auto&l=lm[i];
        ggml_tensor*hn=ggml_mul(C,ggml_rms_norm(C,x,lm_rms_eps),l.attn_norm);
        ggml_tensor*qp=ggml_mul_mat(C,l.Wq,hn),*kp=ggml_mul_mat(C,l.Wk,hn),*vp=ggml_mul_mat(C,l.Wv,hn);
        ggml_tensor*qh=ggml_reshape_3d(C,qp,lm_head_dim,n_q,SEQ),*kh=ggml_reshape_3d(C,kp,lm_head_dim,n_kv,SEQ),*vh=ggml_reshape_3d(C,vp,lm_head_dim,n_kv,SEQ);
        ggml_tensor*qr=ggml_rope_ext(C,qh,t_pos,nullptr,(int)lm_head_dim,GGML_ROPE_TYPE_NEOX,0,lm_rope_base,1.0f,0.0f,1.0f,32.0f,1.0f);
        ggml_tensor*kr=ggml_rope_ext(C,kh,t_pos,nullptr,(int)lm_head_dim,GGML_ROPE_TYPE_NEOX,0,lm_rope_base,1.0f,0.0f,1.0f,32.0f,1.0f);
        ggml_tensor*Q=ggml_cont(C,ggml_permute(C,qr,0,2,1,3)),*K=ggml_cont(C,ggml_permute(C,kr,0,2,1,3)),*V=ggml_cont(C,ggml_permute(C,vh,1,2,0,3));
        ggml_tensor*kq=ggml_mul_mat(C,K,Q); ggml_prec_set_acc(kq,GGML_PREC_F32);
        // Unmasked on purpose: OpenVLA-OFT patches transformers to replace the
        // causal mask across the whole sequence (modeling_llama.py:719-723).
        ggml_tensor*aw=ggml_soft_max_ext(C,kq,nullptr,lsc,0.0f);
        ggml_tensor*kqv=ggml_mul_mat(C,V,aw);
        ggml_tensor*att=ggml_reshape_2d(C,ggml_cont(C,ggml_permute(C,kqv,0,2,1,3)),HC,SEQ);
        x=ggml_add(C,x,ggml_mul_mat(C,l.Wo,att));
        ggml_tensor*hn2=ggml_mul(C,ggml_rms_norm(C,x,lm_rms_eps),l.ffn_norm);
        ggml_tensor*gt=ggml_silu(C,ggml_mul_mat(C,l.Wg,hn2)),*ut=ggml_mul_mat(C,l.Wu,hn2);
        x=ggml_add(C,x,ggml_mul_mat(C,l.Wd,ggml_mul(C,gt,ut)));
    }
    ggml_tensor*hs=ggml_mul(C,ggml_rms_norm(C,x,lm_rms_eps),lm_out_norm);

    ggml_tensor*ah=ggml_cont(C,ggml_view_2d(C,hs,HC,n_act,hs->nb[1],ACT_START*hs->nb[1]));

    ggml_tensor*hin=ggml_reshape_2d(C,ah,action_dim*HC,chunk);

    ggml_tensor*hh=layer_norm(C,hin,h_ln1w,h_ln1b,head_ln_eps);
    hh=ggml_relu(C,linear(C,h_fc1w,h_fc1b,hh));
    for(int i=0;i<head_blocks;++i){ const auto&w=hblk[i];
        ggml_tensor*y=layer_norm(C,hh,w.lnw,w.lnb,head_ln_eps);
        y=ggml_relu(C,linear(C,w.linw,w.linb,y));
        hh=ggml_add(C,hh,y); }
    hh=layer_norm(C,hh,h_ln2w,h_ln2b,head_ln_eps);
    ggml_tensor*norm_actions=linear(C,h_fc2w,h_fc2b,hh); ggml_set_output(norm_actions);

    gio.t_ids=t_ids; gio.t_state=t_state; gio.t_proj=t_proj; gio.act0=act0;
    gio.t_pos=t_pos; gio.norm_actions=norm_actions;

    ggml_cgraph*gf=ggml_new_graph_custom(C,16384,false); ggml_build_forward_expand(gf,norm_actions);
    return gf;
    });
    if(!built){ std::fprintf(stderr,"vla(openvla_oft): main graph build failed\n"); return {}; }

    MainIO & gio = main_graph.io();
    ggml_cgraph * gf = main_graph.graph();
    ggml_tensor*t_ids=gio.t_ids,*t_state=gio.t_state,*t_proj=gio.t_proj;
    ggml_tensor*act0=gio.act0,*t_pos=gio.t_pos,*norm_actions=gio.norm_actions;

    { std::vector<int32_t> ids(L+1);
      for(int64_t i=0;i<L;++i)
          ids[i]=in.lang_tokens[i];
      ids[L]=(int32_t)stop_id;
      ggml_backend_tensor_set(t_ids,ids.data(),0,ggml_nbytes(t_ids)); }
    if(!ggml_are_same_shape(proj,t_proj)){ std::fprintf(stderr,"vla(openvla_oft): projector output does not match the LM width\n"); return {}; }
    ggml_backend_tensor_copy(proj,t_proj);
    if(!gio.consts){
        std::vector<int32_t> pp(SEQ);
        for(int64_t i=0;i<SEQ;++i)
            pp[i]=(int32_t)i;
        ggml_backend_tensor_set(t_pos,pp.data(),0,ggml_nbytes(t_pos));
        std::vector<float> z((size_t)HC*n_act,0.0f);
        ggml_backend_tensor_set(act0,z.data(),0,ggml_nbytes(act0));
        gio.consts=true;
    }
    { std::vector<float> sv(proprio_dim,0.0f); for(int64_t i=0;i<proprio_dim && in.state;++i) sv[i]=in.state[i];
      ggml_backend_tensor_set(t_state,sv.data(),0,ggml_nbytes(t_state)); }

    graph_unique_names(gf);
    if(ggml_backend_graph_compute(backend,gf)!=GGML_STATUS_SUCCESS){ std::fprintf(stderr,"vla(openvla_oft): main compute failed\n"); return {}; }
    std::vector<float> na((size_t)action_dim*chunk);
    ggml_backend_tensor_get(norm_actions,na.data(),0,na.size()*sizeof(float));
    stats.ms_inference = std::chrono::duration<float,std::milli>(clock::now()-ti).count();

    std::vector<float> out=act_stats.unnorm(na,chunk,action_dim,cfg.max_action_dim>0 ? cfg.max_action_dim : action_dim);
    stats.ms_total = std::chrono::duration<float,std::milli>(clock::now()-t0).count();
    return out;
}

}
