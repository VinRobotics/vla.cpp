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
#include "model.h"
#include "vision_common.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include "gguf.h"
#include "models/gguf_reader.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace vla {
namespace {


bool parse_stats(const std::string & js, int64_t want, std::vector<float> & q01,
                 std::vector<float> & q99, std::vector<uint8_t> & mask, std::string & suite) {
    auto find_key = [&](size_t from, const std::string & key) -> size_t {
        const std::string pat = "\"" + key + "\"";
        return js.find(pat, from);
    };
    const char * env = std::getenv("VLA_OPENVLA_OFT_UNNORM_KEY");
    size_t suite_pos;
    if (env) { suite = env; suite_pos = find_key(0, suite); }
    else {
        size_t b = js.find('{'); size_t q = js.find('"', b);
        size_t qe = js.find('"', q + 1);
        suite = js.substr(q + 1, qe - q - 1); suite_pos = q;
    }
    if (suite_pos == std::string::npos) { std::fprintf(stderr, "vla(openvla_oft): suite '%s' not in stats\n", suite.c_str()); return false; }
    size_t act = find_key(suite_pos, "action");
    if (act == std::string::npos) return false;
    auto read_arr = [&](const std::string & key, std::vector<float> & out) -> bool {
        size_t k = find_key(act, key); if (k == std::string::npos) return false;
        size_t lb = js.find('[', k); size_t rb = js.find(']', lb);
        if (lb == std::string::npos || rb == std::string::npos) return false;
        out.clear(); size_t p = lb + 1;
        while (p < rb) {
            while (p < rb && (js[p] == ',' || js[p] == ' ' || js[p] == '\n' || js[p] == '\t' || js[p] == '\r')) ++p;
            if (p >= rb) break;
            bool t = (js.compare(p, 4, "true") == 0), f = (js.compare(p, 5, "false") == 0);
            if (t || f) { out.push_back(t ? 1.0f : 0.0f); p += t ? 4 : 5; }
            else { out.push_back(std::strtof(js.c_str() + p, nullptr)); while (p < rb && js[p] != ',') ++p; }
        }
        return true;
    };
    std::vector<float> mk;
    if (!read_arr("q01", q01) || !read_arr("q99", q99)) return false;
    if (!read_arr("mask", mk)) mk.assign(want, 1.0f);
    mask.assign(mk.size(), 1); for (size_t i = 0; i < mk.size(); ++i) mask[i] = mk[i] != 0.0f ? 1 : 0;
    return (int64_t) q01.size() == want && (int64_t) q99.size() == want;
}

struct ViTLayerW { ggml_tensor *n1w,*n1b,*n2w,*n2b,*ls1,*ls2,*Wqkv,*bqkv,*Wproj,*bproj,*Wfc1,*bfc1,*Wfc2,*bfc2; };
struct LMLayerW  { ggml_tensor *attn_norm,*Wq,*Wk,*Wv,*Wo,*ffn_norm,*Wg,*Wu,*Wd; };
struct HeadBlkW  { ggml_tensor *lnw,*lnb,*linw,*linb; };

}

struct OpenVlaOftModelArch : public ModelArchBase {
    OpenVlaOftModelArch() : ModelArchBase(Arch::OPENVLA_OFT) {}
    ~OpenVlaOftModelArch() override {
        if (weight_buf)  ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend)     ggml_backend_free(backend);
    }

    ggml_backend_t backend = nullptr;
    bool is_gpu = false; int n_threads = default_cpu_threads();
    ggml_context * ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_type mt = GGML_TYPE_BF16;

    int64_t d_hidden=1024,d_layers=23,d_heads=16,d_head_dim=64,d_inter=4096;
    int64_t s_hidden=1152,s_layers=26,s_heads=16,s_head_dim=72,s_inter=4304;
    int64_t image_size=224,patch_size=14,n_patches=256,proj_mid=8704,vdim=2176,num_images=2;
    float   vit_ln_eps=1e-6f;
    int64_t lm_hidden=4096,lm_layers=32,n_q=32,n_kv=32,lm_head_dim=128,lm_inter=11008,vocab=32064;
    float   lm_rope_base=1e4f, lm_rms_eps=1e-6f;
    int64_t chunk=8,action_dim=7,proprio_dim=8,head_hidden=4096,head_blocks=2;
    float   head_ln_eps=1e-5f;
    int64_t stop_id=2,empty_id=29871;

    ggml_tensor *d_patch_w,*d_patch_b,*d_cls,*d_reg,*d_pos; std::vector<ViTLayerW> dvit;
    ggml_tensor *s_patch_w,*s_patch_b,*s_pos;               std::vector<ViTLayerW> svit;
    ggml_tensor *pj_fc1w,*pj_fc1b,*pj_fc2w,*pj_fc2b,*pj_fc3w,*pj_fc3b;
    ggml_tensor *token_embd,*lm_out_norm; std::vector<LMLayerW> lm;
    ggml_tensor *pp_fc1w,*pp_fc1b,*pp_fc2w,*pp_fc2b;
    ggml_tensor *h_ln1w,*h_ln1b,*h_fc1w,*h_fc1b,*h_ln2w,*h_ln2b,*h_fc2w,*h_fc2b; std::vector<HeadBlkW> hblk;

    std::vector<float> q01,q99; std::vector<uint8_t> unnorm_mask; std::string suite;

    std::vector<float> predict(const Inputs& in) override;
};

namespace {

static ggml_tensor * LN(ggml_context*C, ggml_tensor*x, ggml_tensor*w, ggml_tensor*b, float eps){ return ggml_add(C,ggml_mul(C,ggml_norm(C,x,eps),w),b); }

static ggml_tensor* vit_block(ggml_context*C, const ViTLayerW&w, ggml_tensor*x, int64_t N, int64_t hidden, int64_t heads, int64_t hd, float eps, bool ls){
    const float sc=1.0f/std::sqrt((float)hd);
    ggml_tensor*xn=LN(C,x,w.n1w,w.n1b,eps);
    ggml_tensor*qkv=ggml_add(C,ggml_mul_mat(C,w.Wqkv,xn),w.bqkv);
    ggml_tensor*q=ggml_cont(C,ggml_view_2d(C,qkv,hidden,N,qkv->nb[1],0*hidden*sizeof(float)));
    ggml_tensor*k=ggml_cont(C,ggml_view_2d(C,qkv,hidden,N,qkv->nb[1],1*hidden*sizeof(float)));
    ggml_tensor*v=ggml_cont(C,ggml_view_2d(C,qkv,hidden,N,qkv->nb[1],2*hidden*sizeof(float)));
    ggml_tensor*Q=ggml_cont(C,ggml_permute(C,ggml_reshape_3d(C,q,hd,heads,N),0,2,1,3));
    ggml_tensor*K=ggml_cont(C,ggml_permute(C,ggml_reshape_3d(C,k,hd,heads,N),0,2,1,3));
    ggml_tensor*V=ggml_cont(C,ggml_permute(C,ggml_reshape_3d(C,v,hd,heads,N),1,2,0,3));
    ggml_tensor*kq=ggml_mul_mat(C,K,Q); ggml_mul_mat_set_prec(kq,GGML_PREC_F32);
    ggml_tensor*aw=ggml_soft_max_ext(C,kq,nullptr,sc,0.0f);
    ggml_tensor*kqv=ggml_mul_mat(C,V,aw);
    ggml_tensor*att=ggml_reshape_2d(C,ggml_cont(C,ggml_permute(C,kqv,0,2,1,3)),hidden,N);
    ggml_tensor*ao=ggml_add(C,ggml_mul_mat(C,w.Wproj,att),w.bproj);
    x=ggml_add(C,x,ls?ggml_mul(C,ao,w.ls1):ao);
    ggml_tensor*xn2=LN(C,x,w.n2w,w.n2b,eps);
    ggml_tensor*h=ggml_add(C,ggml_mul_mat(C,w.Wfc1,xn2),w.bfc1); h=ggml_gelu_erf(C,h);
    h=ggml_add(C,ggml_mul_mat(C,w.Wfc2,h),w.bfc2);
    return ggml_add(C,x,ls?ggml_mul(C,h,w.ls2):h);
}

static ggml_tensor* tower(ggml_context*C, ggml_tensor*pix, ggml_tensor*pw, ggml_tensor*pb, ggml_tensor*pos,
                          ggml_tensor*cls, ggml_tensor*reg, const std::vector<ViTLayerW>&blk,
                          int64_t hidden, int64_t heads, int64_t hd, int64_t inter, int64_t patch, float eps, bool prefix){
    (void)inter;
    const int64_t NP=256, nprefix=prefix?5:0, N=NP+nprefix;
    ggml_tensor*conv=ggml_conv_2d(C,pw,pix,patch,patch,0,0,1,1);
    ggml_tensor*pt=ggml_cont(C,ggml_transpose(C,ggml_reshape_2d(C,conv,NP,hidden)));
    pt=ggml_add(C,pt,pb); pt=ggml_add(C,pt,pos);
    ggml_tensor*x=pt;
    if(prefix){ ggml_tensor*tok=ggml_concat(C,ggml_reshape_2d(C,cls,hidden,1),reg,1); x=ggml_concat(C,tok,pt,1); }
    for(size_t i=0;i<blk.size();++i) x=vit_block(C,blk[i],x,N,hidden,heads,hd,eps,prefix);
    if(prefix) x=ggml_cont(C,ggml_view_2d(C,x,hidden,NP,x->nb[1],nprefix*x->nb[1]));
    return x;
}

}

std::unique_ptr<ModelArchBase> openvla_oft_create(const std::string& mmproj_path,
                                                  const std::string& ckpt_path,
                                                  const std::string& ) {
    if (!mmproj_path.empty())
        std::printf("vla(openvla_oft): note - mmproj '%s' ignored (vision baked into combined GGUF)\n", mmproj_path.c_str());
    auto m = std::make_unique<OpenVlaOftModelArch>();
    m->mt = std::getenv("VLA_OPENVLA_OFT_F32_WEIGHTS") ? GGML_TYPE_F32 : GGML_TYPE_BF16;

    gguf_reader g("openvla_oft");
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has("openvla_oft.architecture")) { std::fprintf(stderr, "vla(openvla_oft): not an openvla_oft GGUF\n"); return nullptr; }

    auto U=[&](const char*k,int64_t&d){ if(g.has(k)) d=(int64_t)g.u32(k); };
    auto F=[&](const char*k,float&d){ if(g.has(k)) d=g.f32(k); };
    U("openvla_oft.vit.dino.hidden",m->d_hidden); U("openvla_oft.vit.dino.layers",m->d_layers);
    U("openvla_oft.vit.dino.heads",m->d_heads); U("openvla_oft.vit.dino.head_dim",m->d_head_dim); U("openvla_oft.vit.dino.inter",m->d_inter);
    U("openvla_oft.vit.sig.hidden",m->s_hidden); U("openvla_oft.vit.sig.layers",m->s_layers);
    U("openvla_oft.vit.sig.heads",m->s_heads); U("openvla_oft.vit.sig.head_dim",m->s_head_dim); U("openvla_oft.vit.sig.inter",m->s_inter);
    U("openvla_oft.vit.image_size",m->image_size); U("openvla_oft.vit.patch_size",m->patch_size);
    U("openvla_oft.vit.n_patches",m->n_patches); F("openvla_oft.vit.ln_eps",m->vit_ln_eps);
    U("openvla_oft.vit.proj_mid",m->proj_mid); U("openvla_oft.vit.vdim",m->vdim); U("openvla_oft.vit.num_images",m->num_images);
    U("openvla_oft.lm.hidden",m->lm_hidden); U("openvla_oft.lm.layers",m->lm_layers);
    U("openvla_oft.lm.q_heads",m->n_q); U("openvla_oft.lm.kv_heads",m->n_kv); U("openvla_oft.lm.head_dim",m->lm_head_dim);
    U("openvla_oft.lm.inter",m->lm_inter); U("openvla_oft.lm.vocab",m->vocab);
    F("openvla_oft.lm.rope_theta",m->lm_rope_base); F("openvla_oft.lm.rms_eps",m->lm_rms_eps);
    U("openvla_oft.action.chunk",m->chunk); U("openvla_oft.action.action_dim",m->action_dim);
    U("openvla_oft.action.proprio_dim",m->proprio_dim); U("openvla_oft.action.head_hidden",m->head_hidden);
    U("openvla_oft.action.head_blocks",m->head_blocks); F("openvla_oft.action.head_ln_eps",m->head_ln_eps);
    U("openvla_oft.tokens.stop_id",m->stop_id); U("openvla_oft.tokens.empty_id",m->empty_id);
    if (m->lm_head_dim==0) m->lm_head_dim = m->lm_hidden / m->n_q;

    if (g.has("openvla_oft.statistics_json")) {
        if (!parse_stats(g.str("openvla_oft.statistics_json"), m->action_dim, m->q01, m->q99, m->unnorm_mask, m->suite))
            { std::fprintf(stderr, "vla(openvla_oft): failed to parse statistics_json\n"); return nullptr; }
        std::printf("vla(openvla_oft): unnorm suite = %s (q99 dim %zu)\n", m->suite.c_str(), m->q99.size());
    }

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) { m->is_gpu=true; std::printf("vla(openvla_oft): backend = CUDA (device 0)\n"); }
#elif defined(GGML_USE_METAL)
    m->backend = ggml_backend_metal_init();
    if (m->backend) { m->is_gpu=true; std::printf("vla(openvla_oft): backend = Metal\n"); }
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(openvla_oft): cpu backend init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(openvla_oft): backend = CPU (%d threads)\n", m->n_threads);
    }

    ggml_init_params wp = { (size_t)64*1024*1024, nullptr, true };
    m->ctx_weights = ggml_init(wp);
    ggml_context * W = m->ctx_weights;
    bool ok = true;
    auto mk=[&](const char*name, ggml_type ty)->ggml_tensor*{ const ggml_tensor*gt=g.meta(name);
        if(!gt){ std::fprintf(stderr,"vla(openvla_oft): missing %s\n",name); ok=false; return nullptr; }
        ggml_tensor*t=ggml_new_tensor(W,g.resident_type(gt,ty),ggml_n_dims(gt),gt->ne); ggml_set_name(t,name); return t; };
    auto mm=[&](const char*n){ return mk(n,m->mt); };
    auto f32=[&](const char*n){ return mk(n,GGML_TYPE_F32); };

    m->d_patch_w=mk("vis.d.patch.weight",GGML_TYPE_F32); m->d_patch_b=f32("vis.d.patch.bias");
    m->d_cls=f32("vis.d.cls"); m->d_reg=f32("vis.d.reg"); m->d_pos=f32("vis.d.pos");
    m->dvit.resize(m->d_layers);
    for(int i=0;i<m->d_layers;++i){ auto&w=m->dvit[i]; char b[64];
        auto N=[&](const char*s){ std::snprintf(b,sizeof(b),"vis.d.blk.%d.%s",i,s); return (const char*)b; };
        w.n1w=f32(N("ln1.weight")); w.n1b=f32(N("ln1.bias")); w.n2w=f32(N("ln2.weight")); w.n2b=f32(N("ln2.bias"));
        w.ls1=f32(N("ls1")); w.ls2=f32(N("ls2")); w.Wqkv=mm(N("qkv.weight")); w.bqkv=f32(N("qkv.bias"));
        w.Wproj=mm(N("proj.weight")); w.bproj=f32(N("proj.bias")); w.Wfc1=mm(N("fc1.weight")); w.bfc1=f32(N("fc1.bias"));
        w.Wfc2=mm(N("fc2.weight")); w.bfc2=f32(N("fc2.bias")); }

    m->s_patch_w=mk("vis.s.patch.weight",GGML_TYPE_F32); m->s_patch_b=f32("vis.s.patch.bias"); m->s_pos=f32("vis.s.pos");
    m->svit.resize(m->s_layers);
    for(int i=0;i<m->s_layers;++i){ auto&w=m->svit[i]; char b[64];
        auto N=[&](const char*s){ std::snprintf(b,sizeof(b),"vis.s.blk.%d.%s",i,s); return (const char*)b; };
        w.n1w=f32(N("ln1.weight")); w.n1b=f32(N("ln1.bias")); w.n2w=f32(N("ln2.weight")); w.n2b=f32(N("ln2.bias"));
        w.ls1=nullptr; w.ls2=nullptr; w.Wqkv=mm(N("qkv.weight")); w.bqkv=f32(N("qkv.bias"));
        w.Wproj=mm(N("proj.weight")); w.bproj=f32(N("proj.bias")); w.Wfc1=mm(N("fc1.weight")); w.bfc1=f32(N("fc1.bias"));
        w.Wfc2=mm(N("fc2.weight")); w.bfc2=f32(N("fc2.bias")); }

    m->pj_fc1w=mm("vis.proj.fc1.weight"); m->pj_fc1b=f32("vis.proj.fc1.bias");
    m->pj_fc2w=mm("vis.proj.fc2.weight"); m->pj_fc2b=f32("vis.proj.fc2.bias");
    m->pj_fc3w=mm("vis.proj.fc3.weight"); m->pj_fc3b=f32("vis.proj.fc3.bias");

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
    if(!ok){ std::fprintf(stderr,"vla(openvla_oft): weight setup failed\n"); return nullptr; }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if(!m->weight_buf){ std::fprintf(stderr,"vla(openvla_oft): alloc_ctx_tensors failed (OOM?)\n"); return nullptr; }
    for(ggml_tensor*t=ggml_get_first_tensor(W); t; t=ggml_get_next_tensor(W,t)){
        std::vector<uint8_t> bytes=g.read_convert(ggml_get_name(t),t->type);
        if(bytes.empty()||bytes.size()!=ggml_nbytes(t)){ std::fprintf(stderr,"vla(openvla_oft): load %s (%zu vs %zu)\n",ggml_get_name(t),bytes.size(),ggml_nbytes(t)); return nullptr; }
        ggml_backend_tensor_set(t,bytes.data(),0,bytes.size());
    }
    std::printf("vla(openvla_oft): weights resident %.2f GiB (%s) - DINOv2+SigLIP towers + Llama-2-7B + MLPResNet L1 head\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0), m->mt==GGML_TYPE_F32?"F32":"BF16");

    m->cfg.n_suffix = m->chunk; m->cfg.max_action_dim = m->action_dim;
    m->cfg.real_action_dim = m->action_dim; m->cfg.real_state_dim = m->proprio_dim;
    m->cfg.max_state_dim = m->proprio_dim; m->cfg.n_img = m->n_patches; m->cfg.hidden = m->lm_hidden;
    m->cfg.n_lang = 512;
    return m;
}

namespace {

void normalize_tower(const ImageView& v, int64_t S, const float mean[3], const float std_[3], std::vector<float>& out){
    out.assign((size_t)3*S*S,0.0f);
    for(int64_t h=0;h<S;++h) for(int64_t w=0;w<S;++w) for(int64_t c=0;c<3;++c){
        float px = (v.format==PixelFormat::U8) ? ((const uint8_t*)v.data)[(h*S+w)*3+c]/255.0f
                                               : ((const float*)v.data)[(h*S+w)*3+c];
        out[c*S*S+h*S+w] = (px-mean[c])/std_[c];
    }
}
}

std::vector<float> OpenVlaOftModelArch::predict(const Inputs& in) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    stats = Stats{};
    const int64_t S=image_size, NP=n_patches, HC=lm_hidden;
    const int64_t n_views = in.n_images;
    if (in.precomputed_img_emb) { std::fprintf(stderr, "vla(openvla_oft): precomputed_img_emb is not supported; the DINOv2+SigLIP tower is baked into the GGUF, pass raw images\n"); return {}; }
    if (n_views < 1) { std::fprintf(stderr, "vla(openvla_oft): need >=1 image view\n"); return {}; }
    if (!in.images) { std::fprintf(stderr, "vla(openvla_oft): n_images=%d but the images pointer is null\n", in.n_images); return {}; }
    // towers read S*S*3 per view; reject any view that is not exactly SxS.
    for (int64_t v = 0; v < n_views; ++v) {
        const ImageView& iv = in.images[v];
        if (!view_is_side(iv.data, iv.w, iv.h, S)) {
            std::fprintf(stderr, "vla(openvla_oft): image view %lld is %dx%d, expected %lldx%lld\n",
                         (long long) v, iv.w, iv.h, (long long) S, (long long) S);
            return {};
        }
    }
    static const float DMEAN[3]={0.484375f,0.455078125f,0.40625f}, DSTD[3]={0.228515625f,0.2236328125f,0.224609375f};
    static const float SMEAN[3]={0.5f,0.5f,0.5f}, SSTD[3]={0.5f,0.5f,0.5f};

    const int64_t NPATCH = NP * n_views;
    std::vector<float> proj_host((size_t)HC*NPATCH);
    {
        const auto tv=clock::now();
        ggml_init_params vp={(size_t)64*1024*1024,nullptr,true}; ggml_context*C=ggml_init(vp);
        std::vector<ggml_tensor*> px_d(n_views), px_s(n_views), cmb(n_views);
        for(int v=0; v<n_views; ++v){
            px_d[v]=ggml_new_tensor_3d(C,GGML_TYPE_F32,S,S,3); ggml_set_input(px_d[v]);
            px_s[v]=ggml_new_tensor_3d(C,GGML_TYPE_F32,S,S,3); ggml_set_input(px_s[v]);
            ggml_tensor*pd=tower(C,px_d[v],d_patch_w,d_patch_b,d_pos,d_cls,d_reg,dvit,d_hidden,d_heads,d_head_dim,d_inter,patch_size,vit_ln_eps,true);
            ggml_tensor*ps=tower(C,px_s[v],s_patch_w,s_patch_b,s_pos,nullptr,nullptr,svit,s_hidden,s_heads,s_head_dim,s_inter,patch_size,vit_ln_eps,false);
            cmb[v]=ggml_concat(C,pd,ps,0);
        }
        ggml_tensor*allp=cmb[0]; for(int v=1;v<n_views;++v) allp=ggml_concat(C,allp,cmb[v],1);
        ggml_tensor*ph=ggml_add(C,ggml_mul_mat(C,pj_fc1w,allp),pj_fc1b); ph=ggml_gelu_erf(C,ph);
        ph=ggml_add(C,ggml_mul_mat(C,pj_fc2w,ph),pj_fc2b); ph=ggml_gelu_erf(C,ph);
        ggml_tensor*proj=ggml_add(C,ggml_mul_mat(C,pj_fc3w,ph),pj_fc3b); ggml_set_output(proj);
        ggml_cgraph*vg=ggml_new_graph_custom(C,16384,false); ggml_build_forward_expand(vg,proj);
        ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if(!ga||!ggml_gallocr_alloc_graph(ga,vg)){ std::fprintf(stderr,"vla(openvla_oft): vision gallocr failed\n"); if(ga)ggml_gallocr_free(ga); ggml_free(C); return {}; }
        std::vector<float> dbuf, sbuf;
        for(int v=0;v<n_views;++v){
            normalize_tower(in.images[v],S,DMEAN,DSTD,dbuf); ggml_backend_tensor_set(px_d[v],dbuf.data(),0,ggml_nbytes(px_d[v]));
            normalize_tower(in.images[v],S,SMEAN,SSTD,sbuf); ggml_backend_tensor_set(px_s[v],sbuf.data(),0,ggml_nbytes(px_s[v]));
        }
        if(ggml_backend_graph_compute(backend,vg)!=GGML_STATUS_SUCCESS){ std::fprintf(stderr,"vla(openvla_oft): vision compute failed\n"); ggml_gallocr_free(ga); ggml_free(C); return {}; }
        ggml_backend_tensor_get(proj,proj_host.data(),0,proj_host.size()*sizeof(float));
        ggml_gallocr_free(ga); ggml_free(C);
        stats.ms_vision = std::chrono::duration<float,std::milli>(clock::now()-tv).count();
    }

    const int64_t L = in.n_lang;
    // ggml_get_rows does not bound-check, so reject out-of-range tokens here.
    for (int64_t i = 0; i < L; ++i)
        if (in.lang_tokens[i] < 0 || in.lang_tokens[i] >= vocab) {
            std::fprintf(stderr, "vla(openvla_oft): token %d out of vocab\n", in.lang_tokens[i]);
            return {};
        }
    if ((int64_t) stop_id < 0 || (int64_t) stop_id >= vocab) {
        std::fprintf(stderr, "vla(openvla_oft): stop_id %lld out of vocab\n", (long long) stop_id);
        return {};
    }
    const int64_t n_act = chunk * action_dim;
    const int64_t NUM_PATCHES = NPATCH + 1;
    const int64_t NUM_PROMPT_TOKENS = L - 1;
    const int64_t ACT_START = NUM_PATCHES + NUM_PROMPT_TOKENS;
    const int64_t SEQ = 1 + NUM_PATCHES + (L-1) + n_act + 1;
    const auto ti=clock::now();
    ggml_init_params mp={(size_t)256*1024*1024,nullptr,true}; ggml_context*C=ggml_init(mp);

    ggml_tensor*t_ids=ggml_new_tensor_1d(C,GGML_TYPE_I32,L+1); ggml_set_input(t_ids);
    ggml_tensor*emb=ggml_get_rows(C,token_embd,t_ids);
    if(emb->type!=GGML_TYPE_F32) emb=ggml_cast(C,emb,GGML_TYPE_F32);
    ggml_tensor*bos =ggml_cont(C,ggml_view_2d(C,emb,HC,1,emb->nb[1],0));
    ggml_tensor*rest=ggml_cont(C,ggml_view_2d(C,emb,HC,L-1,emb->nb[1],emb->nb[1]));
    ggml_tensor*stop=ggml_cont(C,ggml_view_2d(C,emb,HC,1,emb->nb[1],L*emb->nb[1]));

    ggml_tensor*t_state=ggml_new_tensor_1d(C,GGML_TYPE_F32,proprio_dim); ggml_set_input(t_state);
    ggml_tensor*pf=ggml_add(C,ggml_mul_mat(C,pp_fc1w,t_state),pp_fc1b); pf=ggml_gelu_erf(C,pf);
    pf=ggml_add(C,ggml_mul_mat(C,pp_fc2w,pf),pp_fc2b); ggml_tensor*pvec=ggml_reshape_2d(C,pf,HC,1);

    ggml_tensor*t_proj=ggml_new_tensor_2d(C,GGML_TYPE_F32,HC,NPATCH); ggml_set_input(t_proj);
    ggml_tensor*patches=ggml_concat(C,t_proj,pvec,1);

    ggml_tensor*act0=ggml_new_tensor_2d(C,GGML_TYPE_F32,HC,n_act); ggml_set_input(act0);

    ggml_tensor*seq=ggml_concat(C,bos,patches,1);
    seq=ggml_concat(C,seq,rest,1);
    seq=ggml_concat(C,seq,act0,1);
    seq=ggml_concat(C,seq,stop,1);

    ggml_tensor*t_pos=ggml_new_tensor_1d(C,GGML_TYPE_I32,SEQ); ggml_set_input(t_pos);
    const float lsc=1.0f/std::sqrt((float)lm_head_dim);
    ggml_tensor*x=seq;
    for(int i=0;i<lm_layers;++i){ const auto&l=lm[i];
        ggml_tensor*hn=ggml_mul(C,ggml_rms_norm(C,x,lm_rms_eps),l.attn_norm);
        ggml_tensor*qp=ggml_mul_mat(C,l.Wq,hn),*kp=ggml_mul_mat(C,l.Wk,hn),*vp=ggml_mul_mat(C,l.Wv,hn);
        ggml_tensor*qh=ggml_reshape_3d(C,qp,lm_head_dim,n_q,SEQ),*kh=ggml_reshape_3d(C,kp,lm_head_dim,n_kv,SEQ),*vh=ggml_reshape_3d(C,vp,lm_head_dim,n_kv,SEQ);
        ggml_tensor*qr=ggml_rope_ext(C,qh,t_pos,nullptr,(int)lm_head_dim,GGML_ROPE_TYPE_NEOX,0,lm_rope_base,1.0f,0.0f,1.0f,32.0f,1.0f);
        ggml_tensor*kr=ggml_rope_ext(C,kh,t_pos,nullptr,(int)lm_head_dim,GGML_ROPE_TYPE_NEOX,0,lm_rope_base,1.0f,0.0f,1.0f,32.0f,1.0f);
        ggml_tensor*Q=ggml_cont(C,ggml_permute(C,qr,0,2,1,3)),*K=ggml_cont(C,ggml_permute(C,kr,0,2,1,3)),*V=ggml_cont(C,ggml_permute(C,vh,1,2,0,3));
        ggml_tensor*kq=ggml_mul_mat(C,K,Q); ggml_mul_mat_set_prec(kq,GGML_PREC_F32);
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

    ggml_tensor*hh=LN(C,hin,h_ln1w,h_ln1b,head_ln_eps);
    hh=ggml_relu(C,ggml_add(C,ggml_mul_mat(C,h_fc1w,hh),h_fc1b));
    for(int i=0;i<head_blocks;++i){ const auto&w=hblk[i];
        ggml_tensor*y=LN(C,hh,w.lnw,w.lnb,head_ln_eps);
        y=ggml_relu(C,ggml_add(C,ggml_mul_mat(C,w.linw,y),w.linb));
        hh=ggml_add(C,hh,y); }
    hh=LN(C,hh,h_ln2w,h_ln2b,head_ln_eps);
    ggml_tensor*norm_actions=ggml_add(C,ggml_mul_mat(C,h_fc2w,hh),h_fc2b); ggml_set_output(norm_actions);

    ggml_cgraph*gf=ggml_new_graph_custom(C,16384,false); ggml_build_forward_expand(gf,norm_actions);
    ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!ga||!ggml_gallocr_alloc_graph(ga,gf)){ std::fprintf(stderr,"vla(openvla_oft): main gallocr failed\n"); if(ga)ggml_gallocr_free(ga); ggml_free(C); return {}; }

    { std::vector<int32_t> ids(L+1);
      for(int64_t i=0;i<L;++i) ids[i]=in.lang_tokens[i];
      ids[L]=(int32_t)stop_id;
      ggml_backend_tensor_set(t_ids,ids.data(),0,ggml_nbytes(t_ids)); }
    ggml_backend_tensor_set(t_proj,proj_host.data(),0,ggml_nbytes(t_proj));
    { std::vector<int32_t> pp(SEQ); for(int64_t i=0;i<SEQ;++i)pp[i]=(int32_t)i; ggml_backend_tensor_set(t_pos,pp.data(),0,ggml_nbytes(t_pos)); }
    { std::vector<float> sv(proprio_dim,0.0f); for(int64_t i=0;i<proprio_dim && in.state;++i) sv[i]=in.state[i];
      ggml_backend_tensor_set(t_state,sv.data(),0,ggml_nbytes(t_state)); }
    { std::vector<float> z((size_t)HC*n_act,0.0f); ggml_backend_tensor_set(act0,z.data(),0,ggml_nbytes(act0)); }

    if(ggml_backend_graph_compute(backend,gf)!=GGML_STATUS_SUCCESS){ std::fprintf(stderr,"vla(openvla_oft): main compute failed\n"); ggml_gallocr_free(ga); ggml_free(C); return {}; }
    std::vector<float> na((size_t)action_dim*chunk);
    ggml_backend_tensor_get(norm_actions,na.data(),0,na.size()*sizeof(float));
    ggml_gallocr_free(ga); ggml_free(C);
    stats.ms_inference = std::chrono::duration<float,std::milli>(clock::now()-ti).count();

    const int64_t Wd = cfg.max_action_dim>0 ? cfg.max_action_dim : action_dim;
    std::vector<float> out((size_t)chunk*Wd,0.0f);
    for(int64_t c=0;c<chunk;++c) for(int64_t d=0;d<action_dim;++d){ float v=na[c*action_dim+d];
        bool msk = (d<(int64_t)unnorm_mask.size()) ? unnorm_mask[d]!=0 : true;
        out[c*Wd+d] = (msk && !q01.empty()) ? 0.5f*(v+1.0f)*(q99[d]-q01[d]+1e-8f)+q01[d] : v; }
    stats.ms_total = std::chrono::duration<float,std::milli>(clock::now()-t0).count();
    return out;
}

}
