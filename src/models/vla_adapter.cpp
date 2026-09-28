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
#include "layers/attn.h"
#include "layers/embed.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "layers/rope.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace vla {
namespace {

struct LMLayerW  { ggml_tensor *attn_norm,*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*ffn_norm,*Wg,*Wu,*Wd; };
struct HeadBlkW  { ggml_tensor *Wq,*bq,*Wks,*bks,*Wvs,*bvs,*Wka,*bka,*Wva,*bva,*Wkt,*bkt,*Wvt,*bvt,*Wo,*bo,*flnw,*flnb,*flw,*flb; float rg; };

}

struct VlaAdapterModelArch : public ModelArchBase {
    VlaAdapterModelArch() : ModelArchBase(Arch::VLA_ADAPTER) {}
    ~VlaAdapterModelArch() override {
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
    scratch_ctx vision_scratch;

    struct MainKey {
        int64_t seq=-1, n_views=-1, nprompt=-1;
        bool operator==(const MainKey & o) const {
            return seq==o.seq && n_views==o.n_views && nprompt==o.nprompt;
        }
    };
    struct MainIO {
        ggml_tensor *t_ids=nullptr,*t_proj=nullptr,*t_pos=nullptr,*t_mask=nullptr;
        ggml_tensor *t_state=nullptr,*t_x0=nullptr,*norm_actions=nullptr;
        ggml_tensor *cT=nullptr,*sT=nullptr,*cA=nullptr,*sA=nullptr,*cK=nullptr,*sK=nullptr;
    };
    graph_cache<MainKey, MainIO> main_graph;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_type mt = GGML_TYPE_BF16;

    int64_t lm_hidden=896,lm_layers=24,n_q=14,n_kv=2,lm_head_dim=64,vocab=151936;
    float   lm_rope_base=1e6f, lm_rms_eps=1e-6f;
    int64_t chunk=8,action_dim=7,proprio_dim=8,num_tokens=64,head_blocks=24,head_heads=8,head_dim=112;
    float   head_rope_base=1e4f, head_ln_eps=1e-5f;
    int64_t stop_id=2;

    DualTower vis;
    ggml_tensor *token_embd,*action_queries,*lm_out_norm; std::vector<LMLayerW> lm;
    ggml_tensor *h_ln1w,*h_ln1b,*h_fc1w,*h_fc1b,*h_ln2w,*h_ln2b,*h_fc2w,*h_fc2b;
    ggml_tensor *pp_fc1w,*pp_fc1b,*pp_fc2w,*pp_fc2b; std::vector<HeadBlkW> hblk;

    Q99Stats act_stats;

    std::vector<float> predict(const Inputs& in) override;
};

std::unique_ptr<ModelArchBase> vla_adapter_create(const std::string& mmproj_path,
                                                  const std::string& ckpt_path,
                                                  const std::string&,
                                                  const Options& opts) {
    if (!mmproj_path.empty())
        std::printf("vla(vla_adapter): note - mmproj '%s' ignored (vision baked into combined GGUF)\n", mmproj_path.c_str());
    auto m = std::make_unique<VlaAdapterModelArch>();
    m->mt = opts.weight_dtype.value_or(vla::default_weight_dtype(GGML_TYPE_BF16));

    gguf_reader g("vla_adapter");
    if (!g.open(ckpt_path))
        return nullptr;
    if (!g.has("vla_adapter.architecture")) {
        std::fprintf(stderr, "vla(vla_adapter): not a vla_adapter GGUF\n");
        return nullptr;
    }

    auto U=[&](const char*k,int64_t&d){ if(g.has(k)) d=(int64_t)g.u32(k); };
    auto F=[&](const char*k,float&d){ if(g.has(k)) d=g.f32(k); };
    m->vis.read_config(g, "vla_adapter");
    U("vla_adapter.lm.hidden",m->lm_hidden); U("vla_adapter.lm.layers",m->lm_layers);
    U("vla_adapter.lm.q_heads",m->n_q); U("vla_adapter.lm.kv_heads",m->n_kv); U("vla_adapter.lm.head_dim",m->lm_head_dim);
    U("vla_adapter.lm.vocab",m->vocab);
    F("vla_adapter.lm.rope_theta",m->lm_rope_base); F("vla_adapter.lm.rms_eps",m->lm_rms_eps);
    U("vla_adapter.action.chunk",m->chunk); U("vla_adapter.action.action_dim",m->action_dim);
    U("vla_adapter.action.proprio_dim",m->proprio_dim); U("vla_adapter.action.num_tokens",m->num_tokens);
    U("vla_adapter.action.head_blocks",m->head_blocks); U("vla_adapter.action.head_heads",m->head_heads);
    U("vla_adapter.action.head_dim",m->head_dim); F("vla_adapter.action.head_rope_base",m->head_rope_base);
    F("vla_adapter.action.ln_eps",m->head_ln_eps); U("vla_adapter.tokens.stop_id",m->stop_id);

    // predict() taps one LM layer per head block, so head_blocks past lm_layers
    // would read off the end of the layer-output vector.
    if(m->lm_layers<1 || m->head_blocks<1 || m->head_blocks>m->lm_layers){
        std::fprintf(stderr,"vla(vla_adapter): head_blocks %lld outside [1, lm_layers %lld]\n",
                     (long long)m->head_blocks,(long long)m->lm_layers);
        return nullptr;
    }

    if (g.has("vla_adapter.statistics_json")) {
        if (!m->act_stats.parse(g.str("vla_adapter.statistics_json"), "VLA_ADAPTER_UNNORM_KEY", "vla_adapter", m->action_dim))
            {
                std::fprintf(stderr, "vla(vla_adapter): failed to parse statistics_json\n");
                return nullptr;
            }
        std::printf("vla(vla_adapter): unnorm suite = %s (q99 dim %zu)\n", m->act_stats.suite.c_str(), m->act_stats.q99.size());
    }

    {
        const Backend b = backend_init("vla(vla_adapter)", m->n_threads);
        if (!b.handle) {
            return nullptr;
        }
        m->backend = b.handle;
    }

    ggml_init_params wp = { (size_t)64*1024*1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    WeightLoader L("vla_adapter", g, m->ctx_weights, m->mt);
    auto mm  = [&](const char * n) { return L.gemm("%s", n); };
    auto f32 = [&](const char * n) { return L.f32("%s", n); };

    m->vis.declare(L);

    m->token_embd=mm("token_embd.weight"); m->action_queries=mm("action_queries.weight"); m->lm_out_norm=f32("lm.output_norm.weight");
    m->lm.resize(m->lm_layers);
    for(int i=0;i<m->lm_layers;++i){ auto&w=m->lm[i]; char b[64];
        auto N=[&](const char*s){ std::snprintf(b,sizeof(b),"lm.blk.%d.%s",i,s); return (const char*)b; };
        w.attn_norm=f32(N("attn_norm.weight")); w.ffn_norm=f32(N("ffn_norm.weight"));
        w.Wq=mm(N("attn_q.weight")); w.bq=f32(N("attn_q.bias")); w.Wk=mm(N("attn_k.weight")); w.bk=f32(N("attn_k.bias"));
        w.Wv=mm(N("attn_v.weight")); w.bv=f32(N("attn_v.bias")); w.Wo=mm(N("attn_o.weight"));
        w.Wg=mm(N("ffn_gate.weight")); w.Wu=mm(N("ffn_up.weight")); w.Wd=mm(N("ffn_down.weight")); }

    m->h_ln1w=f32("aex.head.ln1.weight"); m->h_ln1b=f32("aex.head.ln1.bias");
    m->h_fc1w=mm("aex.head.fc1.weight"); m->h_fc1b=f32("aex.head.fc1.bias");
    m->h_ln2w=f32("aex.head.ln2.weight"); m->h_ln2b=f32("aex.head.ln2.bias");
    m->h_fc2w=mm("aex.head.fc2.weight"); m->h_fc2b=f32("aex.head.fc2.bias");
    m->pp_fc1w=mm("aex.proprio.fc1.weight"); m->pp_fc1b=f32("aex.proprio.fc1.bias");
    m->pp_fc2w=mm("aex.proprio.fc2.weight"); m->pp_fc2b=f32("aex.proprio.fc2.bias");
    m->hblk.resize(m->head_blocks);
    for(int i=0;i<m->head_blocks;++i){ auto&w=m->hblk[i]; char b[64];
        auto N=[&](const char*s){ std::snprintf(b,sizeof(b),"aex.head.blk.%d.%s",i,s); return (const char*)b; };
        w.Wq=mm(N("q_proj.weight")); w.bq=f32(N("q_proj.bias")); w.Wks=mm(N("k_self.weight")); w.bks=f32(N("k_self.bias"));
        w.Wvs=mm(N("v_self.weight")); w.bvs=f32(N("v_self.bias")); w.Wka=mm(N("k_adapter.weight")); w.bka=f32(N("k_adapter.bias"));
        w.Wva=mm(N("v_adapter.weight")); w.bva=f32(N("v_adapter.bias")); w.Wkt=mm(N("k_task.weight")); w.bkt=f32(N("k_task.bias"));
        w.Wvt=mm(N("v_task.weight")); w.bvt=f32(N("v_task.bias")); w.Wo=mm(N("o_proj.weight")); w.bo=f32(N("o_proj.bias"));
        w.flnw=f32(N("ffn_ln.weight")); w.flnb=f32(N("ffn_ln.bias")); w.flw=mm(N("ffn_lin.weight")); w.flb=f32(N("ffn_lin.bias"));
        std::vector<float> gv=g.read_f32(N("gating"));
        if(gv.empty())
            return nullptr;
        w.rg=std::tanh(gv[0]); }

    if (!L.upload(m->backend, &m->weight_buf))
        return nullptr;

    std::printf("vla(vla_adapter): weights resident %.2f GiB (%s) - DINOv2+SigLIP towers + Qwen2.5-0.5B + Bridge head\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0), dtype_name(m->mt));

    m->cfg.n_suffix = m->chunk; m->cfg.max_action_dim = m->action_dim;
    m->cfg.real_action_dim = m->action_dim; m->cfg.real_state_dim = m->proprio_dim;
    m->cfg.max_state_dim = m->proprio_dim; m->cfg.n_img = m->vis.n_patches; m->cfg.hidden = m->lm_hidden;
    m->cfg.n_lang = 512;
    return m;
}

std::vector<float> VlaAdapterModelArch::predict(const Inputs& in) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    stats = Stats{};
    const int64_t NP=vis.n_patches, HC=lm_hidden, HD=head_dim, NH=head_heads;
    const int64_t n_views = in.n_images;
    if (in.precomputed_img_emb) { std::fprintf(stderr, "vla(vla_adapter): precomputed_img_emb is not supported; the DINOv2+SigLIP tower is baked into the GGUF, pass raw images\n"); return {}; }
    if (in.n_lang < 1 || !in.lang_tokens) { std::fprintf(stderr, "vla(vla_adapter): need >=1 lang token\n"); return {}; }
    if (n_views < 1) { std::fprintf(stderr, "vla(vla_adapter): need >=1 image view\n"); return {}; }
    if (!in.images) { std::fprintf(stderr, "vla(vla_adapter): n_images=%d but the images pointer is null\n", in.n_images); return {}; }

    std::vector<float> proj_host((size_t)HC*NP*n_views);
    {
        const auto tv=clock::now();
        if(!vis.encode(backend,vision_scratch,in,"vla_adapter",proj_host))
            return {};
        stats.ms_vision = std::chrono::duration<float,std::milli>(clock::now()-tv).count();
    }

    const int64_t NPROMPT = in.n_lang;
    // ggml_get_rows does not bound-check, so reject out-of-range tokens here.
    for (int64_t i=0; i<NPROMPT; ++i)
        if (in.lang_tokens[i] < 0 || in.lang_tokens[i] >= vocab) {
            std::fprintf(stderr, "vla(vla_adapter): token %d out of vocab\n", in.lang_tokens[i]);
            return {};
        }
    if ((int64_t) stop_id < 0 || (int64_t) stop_id >= vocab) {
        std::fprintf(stderr, "vla(vla_adapter): stop_id %lld out of vocab\n", (long long) stop_id);
        return {};
    }
    const int64_t NUM_PROMPT_TOKENS = NPROMPT-1;
    const int64_t NPATCH = NP * n_views;
    const int64_t SEQ = 1+NPATCH+(NPROMPT-1)+num_tokens+1;
    const auto ti=clock::now();
    // LM + action head graph depends only on the sequence layout.
    const MainKey mkey{ SEQ, n_views, NPROMPT };
    const bool built = main_graph.ensure(backend, mkey, (size_t)128*1024*1024,
                                         [&](ggml_context*C, MainIO & gio)->ggml_cgraph*{
    ggml_tensor*t_ids=ggml_new_tensor_1d(C,GGML_TYPE_I32,NPROMPT+num_tokens+1); ggml_set_input(t_ids);
    ggml_tensor*emb=ggml_get_rows(C,token_embd,t_ids);
    if(emb->type!=GGML_TYPE_F32)
        emb=ggml_cast(C,emb,GGML_TYPE_F32);
    ggml_tensor*aqf=action_queries->type==GGML_TYPE_F32?action_queries:ggml_cast(C,action_queries,GGML_TYPE_F32);
    ggml_tensor*pre=ggml_cont(C,ggml_view_2d(C,emb,HC,NPROMPT,emb->nb[1],0));
    ggml_tensor*stop=ggml_cont(C,ggml_view_2d(C,emb,HC,1,emb->nb[1],(NPROMPT+num_tokens)*emb->nb[1]));
    ggml_tensor*emb2=ggml_concat(C,ggml_concat(C,pre,aqf,1),stop,1);
    ggml_tensor*e0=ggml_cont(C,ggml_view_2d(C,emb2,HC,1,emb2->nb[1],0));
    ggml_tensor*erest=ggml_cont(C,ggml_view_2d(C,emb2,HC,NPROMPT+num_tokens,emb2->nb[1],emb2->nb[1]));
    ggml_tensor*t_proj=ggml_new_tensor_2d(C,GGML_TYPE_F32,HC,NPATCH); ggml_set_input(t_proj);
    ggml_tensor*mm_seq=ggml_concat(C,ggml_concat(C,e0,t_proj,1),erest,1);

    ggml_tensor*t_pos=ggml_new_tensor_1d(C,GGML_TYPE_I32,SEQ); ggml_set_input(t_pos);
    ggml_tensor*t_mask=ggml_new_tensor_2d(C,GGML_TYPE_F32,SEQ,SEQ); ggml_set_input(t_mask);
    const float lsc=1.0f/std::sqrt((float)lm_head_dim);
    std::vector<ggml_tensor*> lout(lm_layers); ggml_tensor*x=mm_seq;
    for(int i=0;i<lm_layers;++i){ const auto&l=lm[i];
        ggml_tensor*hn=ggml_mul(C,ggml_rms_norm(C,x,lm_rms_eps),l.attn_norm);
        ggml_tensor*qp=ggml_add(C,ggml_mul_mat(C,l.Wq,hn),l.bq),*kp=ggml_add(C,ggml_mul_mat(C,l.Wk,hn),l.bk),*vp=ggml_add(C,ggml_mul_mat(C,l.Wv,hn),l.bv);
        ggml_tensor*qh=ggml_reshape_3d(C,qp,lm_head_dim,n_q,SEQ),*kh=ggml_reshape_3d(C,kp,lm_head_dim,n_kv,SEQ),*vh=ggml_reshape_3d(C,vp,lm_head_dim,n_kv,SEQ);
        ggml_tensor*qr=ggml_rope_ext(C,qh,t_pos,nullptr,(int)lm_head_dim,GGML_ROPE_TYPE_NEOX,0,lm_rope_base,1.0f,0.0f,1.0f,32.0f,1.0f);
        ggml_tensor*kr=ggml_rope_ext(C,kh,t_pos,nullptr,(int)lm_head_dim,GGML_ROPE_TYPE_NEOX,0,lm_rope_base,1.0f,0.0f,1.0f,32.0f,1.0f);
        ggml_tensor*Q=ggml_cont(C,ggml_permute(C,qr,0,2,1,3)),*K=ggml_cont(C,ggml_permute(C,kr,0,2,1,3)),*V=ggml_cont(C,ggml_permute(C,vh,1,2,0,3));
        ggml_tensor*kq=ggml_mul_mat(C,K,Q); ggml_prec_set_acc(kq,GGML_PREC_F32);
        ggml_tensor*aw=ggml_soft_max_ext(C,kq,t_mask,lsc,0.0f);
        ggml_tensor*kqv=ggml_mul_mat(C,V,aw);
        ggml_tensor*att=ggml_reshape_2d(C,ggml_cont(C,ggml_permute(C,kqv,0,2,1,3)),HC,SEQ);
        x=ggml_add(C,x,ggml_mul_mat(C,l.Wo,att));
        ggml_tensor*hn2=ggml_mul(C,ggml_rms_norm(C,x,lm_rms_eps),l.ffn_norm);
        ggml_tensor*gt=ggml_silu(C,ggml_mul_mat(C,l.Wg,hn2)),*ut=ggml_mul_mat(C,l.Wu,hn2);
        x=ggml_add(C,x,ggml_mul_mat(C,l.Wd,ggml_mul(C,gt,ut))); lout[i]=x;
    }
    ggml_tensor*final_norm=ggml_mul(C,ggml_rms_norm(C,lout[lm_layers-1],lm_rms_eps),lm_out_norm);

    std::vector<ggml_tensor*> cond(head_blocks);
    for(int i=0;i<head_blocks-1;++i)
        cond[i]=lout[i];
    cond[head_blocks-1]=final_norm;

    ggml_tensor*t_state=ggml_new_tensor_1d(C,GGML_TYPE_F32,proprio_dim); ggml_set_input(t_state);
    ggml_tensor*pf=ggml_add(C,ggml_mul_mat(C,pp_fc1w,t_state),pp_fc1b); pf=ggml_gelu_erf(C,pf);
    pf=ggml_add(C,ggml_mul_mat(C,pp_fc2w,pf),pp_fc2b); ggml_tensor*pvec=ggml_reshape_2d(C,pf,HC,1);

    auto cs_tensor=[&](int64_t Lh)->std::pair<ggml_tensor*,ggml_tensor*>{
        ggml_tensor*cc=ggml_new_tensor_2d(C,GGML_TYPE_F32,HD,Lh); ggml_set_input(cc);
        ggml_tensor*ss=ggml_new_tensor_2d(C,GGML_TYPE_F32,HD,Lh); ggml_set_input(ss);
        return {cc,ss};
    };
    auto [cT,sT]=cs_tensor(chunk);
    auto [cA,sA]=cs_tensor(num_tokens+1);
    auto [cK,sK]=cs_tensor(NPATCH);

    ggml_tensor*t_x0=ggml_new_tensor_2d(C,GGML_TYPE_F32,action_dim*HC,chunk); ggml_set_input(t_x0);
    ggml_tensor*hx=ggml_relu(C,linear(C,h_fc1w,h_fc1b,layer_norm(C,t_x0,h_ln1w,h_ln1b,head_ln_eps)));
    const float hsc=1.0f/std::sqrt((float)HD);
    for(int i=0;i<head_blocks;++i){ const auto&w=hblk[i];

        ggml_tensor*ht=ggml_cont(C,ggml_view_2d(C,cond[i],HC,NPATCH,cond[i]->nb[1],0));
        ggml_tensor*ha=ggml_cont(C,ggml_view_2d(C,cond[i],HC,num_tokens,cond[i]->nb[1],(NPATCH+NUM_PROMPT_TOKENS)*cond[i]->nb[1]));
        ggml_tensor*had=ggml_concat(C,ha,pvec,1);
        auto qk=[&](ggml_tensor*p,ggml_tensor*cs,ggml_tensor*sn){ return rope_pairwise(C,to_heads(C,p,HD,NH,p->ne[1]),cs,sn,HD); };
        auto tov=[&](ggml_tensor*p){ return to_heads_v(C,p,HD,NH,p->ne[1]); };
        ggml_tensor*q=qk(linear(C,w.Wq,w.bq,hx),cT,sT);
        ggml_tensor*kse=qk(linear(C,w.Wks,w.bks,hx),cT,sT);
        ggml_tensor*vse=linear(C,w.Wvs,w.bvs,hx);
        ggml_tensor*kad=qk(linear(C,w.Wka,w.bka,had),cA,sA); ggml_tensor*vad=linear(C,w.Wva,w.bva,had);
        ggml_tensor*kta=qk(linear(C,w.Wkt,w.bkt,ht),cK,sK); ggml_tensor*vta=linear(C,w.Wvt,w.bvt,ht);
        ggml_tensor*Vs=tov(vse),*Va=tov(vad),*VT=tov(vta);
        ggml_tensor*ss2=ggml_mul_mat(C,kse,q),*sa=ggml_mul_mat(C,kad,q),*sr=ggml_mul_mat(C,kta,q);
        ggml_prec_set_acc(ss2,GGML_PREC_F32); ggml_prec_set_acc(sa,GGML_PREC_F32); ggml_prec_set_acc(sr,GGML_PREC_F32);
        ggml_tensor*st2=ggml_scale(C,sr,w.rg);
        ggml_tensor*scr=ggml_concat(C,ggml_concat(C,ss2,sa,0),st2,0);
        ggml_tensor*attn=ggml_soft_max_ext(C,scr,nullptr,hsc,0.0f);
        ggml_tensor*Vc=ggml_concat(C,ggml_concat(C,Vs,Va,0),VT,0);
        ggml_tensor*kqv=ggml_mul_mat(C,Vc,attn);
        ggml_tensor*mg=ggml_reshape_2d(C,ggml_cont(C,ggml_permute(C,kqv,0,2,1,3)),HC,chunk);
        ggml_tensor*res=ggml_add(C,linear(C,w.Wo,w.bo,mg),hx);
        hx=ggml_relu(C,linear(C,w.flw,w.flb,layer_norm(C,res,w.flnw,w.flnb,head_ln_eps)));
    }
    ggml_tensor*norm_actions=linear(C,h_fc2w,h_fc2b,layer_norm(C,hx,h_ln2w,h_ln2b,head_ln_eps)); ggml_set_output(norm_actions);

    gio.t_ids=t_ids; gio.t_proj=t_proj; gio.t_pos=t_pos; gio.t_mask=t_mask;
    gio.t_state=t_state; gio.t_x0=t_x0; gio.norm_actions=norm_actions;
    gio.cT=cT; gio.sT=sT; gio.cA=cA; gio.sA=sA; gio.cK=cK; gio.sK=sK;

    ggml_cgraph*gf=ggml_new_graph_custom(C,65536,false); ggml_build_forward_expand(gf,norm_actions);
    return gf;
    });
    if(!built){ std::fprintf(stderr,"vla(vla_adapter): main graph build failed\n"); return {}; }

    MainIO & gio = main_graph.io();
    ggml_cgraph * gf = main_graph.graph();
    ggml_tensor*t_ids=gio.t_ids,*t_proj=gio.t_proj,*t_pos=gio.t_pos,*t_mask=gio.t_mask;
    ggml_tensor*t_state=gio.t_state,*t_x0=gio.t_x0,*norm_actions=gio.norm_actions;
    ggml_tensor*cT=gio.cT,*sT=gio.sT,*cA=gio.cA,*sA=gio.sA,*cK=gio.cK,*sK=gio.sK;

    { std::vector<int32_t> ids(NPROMPT+num_tokens+1);
      for(int64_t i=0;i<NPROMPT;++i)
          ids[i]=in.lang_tokens[i];
      for(int64_t i=0;i<num_tokens;++i)
          ids[NPROMPT+i]=1;
      ids[NPROMPT+num_tokens]=(int32_t)stop_id;
      ggml_backend_tensor_set(t_ids,ids.data(),0,ggml_nbytes(t_ids)); }
    ggml_backend_tensor_set(t_proj,proj_host.data(),0,ggml_nbytes(t_proj));
    {
        std::vector<int32_t> pp(SEQ);
        for(int64_t i=0;i<SEQ;++i)
            pp[i]=(int32_t)i;
        ggml_backend_tensor_set(t_pos,pp.data(),0,ggml_nbytes(t_pos));
    }
    { std::vector<float> mk; build_causal_mask(SEQ, mk);
      ggml_backend_tensor_set(t_mask,mk.data(),0,ggml_nbytes(t_mask)); }
    { std::vector<float> sv(proprio_dim,0.0f); for(int64_t i=0;i<proprio_dim && in.state;++i) sv[i]=in.state[i];
      ggml_backend_tensor_set(t_state,sv.data(),0,ggml_nbytes(t_state)); }
    {
        std::vector<float> zx((size_t)action_dim*HC*chunk,0.0f);
        ggml_backend_tensor_set(t_x0,zx.data(),0,ggml_nbytes(t_x0));
    }

    auto fill_cs=[&](ggml_tensor*cc,ggml_tensor*ss,int64_t Lh){ std::vector<float> cb,sb; rope_pairwise_table(HD,Lh,head_rope_base,cb,sb);
        ggml_backend_tensor_set(cc,cb.data(),0,ggml_nbytes(cc)); ggml_backend_tensor_set(ss,sb.data(),0,ggml_nbytes(ss)); };
    fill_cs(cT,sT,chunk); fill_cs(cA,sA,num_tokens+1); fill_cs(cK,sK,NPATCH);

    graph_unique_names(gf);
    if(ggml_backend_graph_compute(backend,gf)!=GGML_STATUS_SUCCESS){ std::fprintf(stderr,"vla(vla_adapter): main compute failed\n"); return {}; }
    std::vector<float> na((size_t)action_dim*chunk);
    ggml_backend_tensor_get(norm_actions,na.data(),0,na.size()*sizeof(float));
    stats.ms_inference = std::chrono::duration<float,std::milli>(clock::now()-ti).count();

    std::vector<float> out=act_stats.unnorm(na,chunk,action_dim,cfg.max_action_dim>0 ? cfg.max_action_dim : action_dim);
    stats.ms_total = std::chrono::duration<float,std::milli>(clock::now()-t0).count();
    return out;
}

}
