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

// DINOv2 + SigLIP dual vision tower and q01/q99 action stats, shared by
// OpenVLA-OFT and VLA-Adapter.
// DINOv2 passes prefix=true (CLS + 4 register tokens, dropped after the blocks)
// and uses LayerScale; SigLIP passes prefix=false.

#pragma once

#include "backend.h"
#include "gguf_reader.h"
#include "layers/attn.h"
#include "layers/ffn.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "loader.h"
#include "model.h"
#include "modules/preprocess.h"
#include "scratch_ctx.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace vla {

struct ViTLayerW { ggml_tensor *n1w,*n1b,*n2w,*n2b,*ls1,*ls2,*Wqkv,*bqkv,*Wproj,*bproj,*Wfc1,*bfc1,*Wfc2,*bfc2; };

// DINOv2 carries CLS + 4 register tokens and LayerScale; SigLIP carries
// neither, so its ls1/ls2 stay null and vit_block is told to skip them.
struct DualTower {
    int64_t d_hidden=1024,d_layers=23,d_heads=16,d_head_dim=64;
    int64_t s_hidden=1152,s_layers=26,s_heads=16,s_head_dim=72;
    int64_t image_size=224,patch_size=14,n_patches=256;
    float   ln_eps=1e-6f;

    ggml_tensor *d_patch_w=nullptr,*d_patch_b=nullptr,*d_cls=nullptr,*d_reg=nullptr,*d_pos=nullptr;
    ggml_tensor *s_patch_w=nullptr,*s_patch_b=nullptr,*s_pos=nullptr;
    std::vector<ViTLayerW> dvit, svit;
    ggml_tensor *pj_fc1w=nullptr,*pj_fc1b=nullptr,*pj_fc2w=nullptr,*pj_fc2b=nullptr,*pj_fc3w=nullptr,*pj_fc3b=nullptr;

    void read_config(const gguf_reader & g, const std::string & arch) {
        auto U=[&](const char*k,int64_t&d){ const std::string key=arch+".vit."+k; if(g.has(key.c_str())) d=(int64_t)g.u32(key.c_str()); };
        U("dino.hidden",d_hidden); U("dino.layers",d_layers); U("dino.heads",d_heads); U("dino.head_dim",d_head_dim);
        U("sig.hidden",s_hidden); U("sig.layers",s_layers); U("sig.heads",s_heads); U("sig.head_dim",s_head_dim);
        U("image_size",image_size); U("patch_size",patch_size); U("n_patches",n_patches);
        const std::string eps=arch+".vit.ln_eps";
        if(g.has(eps.c_str())) ln_eps=g.f32(eps.c_str());
    }

    void declare(WeightLoader & L) {
        auto blocks = [&](std::vector<ViTLayerW> & v, const char * pre, int64_t n, bool layer_scale) {
            v.resize(n);
            for (int64_t i=0; i<n; ++i) {
                ViTLayerW & w = v[i];
                w.n1w   = L.f32 ("%s.blk.%lld.ln1.weight",  pre, (long long)i);
                w.n1b   = L.f32 ("%s.blk.%lld.ln1.bias",    pre, (long long)i);
                w.n2w   = L.f32 ("%s.blk.%lld.ln2.weight",  pre, (long long)i);
                w.n2b   = L.f32 ("%s.blk.%lld.ln2.bias",    pre, (long long)i);
                w.ls1   = layer_scale ? L.f32("%s.blk.%lld.ls1", pre, (long long)i) : nullptr;
                w.ls2   = layer_scale ? L.f32("%s.blk.%lld.ls2", pre, (long long)i) : nullptr;
                w.Wqkv  = L.gemm("%s.blk.%lld.qkv.weight",  pre, (long long)i);
                w.bqkv  = L.f32 ("%s.blk.%lld.qkv.bias",    pre, (long long)i);
                w.Wproj = L.gemm("%s.blk.%lld.proj.weight", pre, (long long)i);
                w.bproj = L.f32 ("%s.blk.%lld.proj.bias",   pre, (long long)i);
                w.Wfc1  = L.gemm("%s.blk.%lld.fc1.weight",  pre, (long long)i);
                w.bfc1  = L.f32 ("%s.blk.%lld.fc1.bias",    pre, (long long)i);
                w.Wfc2  = L.gemm("%s.blk.%lld.fc2.weight",  pre, (long long)i);
                w.bfc2  = L.f32 ("%s.blk.%lld.fc2.bias",    pre, (long long)i);
            }
        };

        d_patch_w = L.f32("vis.d.patch.weight");
        d_patch_b = L.f32("vis.d.patch.bias");
        d_cls     = L.f32("vis.d.cls");
        d_reg     = L.f32("vis.d.reg");
        d_pos     = L.f32("vis.d.pos");
        blocks(dvit, "vis.d", d_layers, true);

        s_patch_w = L.f32("vis.s.patch.weight");
        s_patch_b = L.f32("vis.s.patch.bias");
        s_pos     = L.f32("vis.s.pos");
        blocks(svit, "vis.s", s_layers, false);

        pj_fc1w = L.gemm("vis.proj.fc1.weight"); pj_fc1b = L.f32("vis.proj.fc1.bias");
        pj_fc2w = L.gemm("vis.proj.fc2.weight"); pj_fc2b = L.f32("vis.proj.fc2.bias");
        pj_fc3w = L.gemm("vis.proj.fc3.weight"); pj_fc3b = L.f32("vis.proj.fc3.bias");
    }

    struct VisIO { std::vector<ggml_tensor*> px_d, px_s; ggml_tensor*proj=nullptr; };
    ggml_tensor* encode(ggml_backend_t backend, graph_cache<int64_t,VisIO> & cache, const Inputs & in, const char * tag) const;
};

inline ggml_tensor* vit_block(ggml_context*C, const ViTLayerW&w, ggml_tensor*x, int64_t N, int64_t hidden, int64_t heads, int64_t hd, float eps, bool ls){
    ggml_tensor*qkv=linear(C,w.Wqkv,w.bqkv,layer_norm(C,x,w.n1w,w.n1b,eps));
    ggml_tensor*Q=ggml_cont(C,ggml_permute(C,head_view(C,qkv,hd,heads,N,hidden,3,0),0,2,1,3));
    ggml_tensor*K=ggml_cont(C,ggml_permute(C,head_view(C,qkv,hd,heads,N,hidden,3,1),0,2,1,3));
    ggml_tensor*V=ggml_cont(C,ggml_permute(C,head_view(C,qkv,hd,heads,N,hidden,3,2),1,2,0,3));
    ggml_tensor*ao=linear(C,w.Wproj,w.bproj,attention(C,Q,K,V,nullptr,1.0f/std::sqrt((float)hd),hidden,N));
    x=ggml_add(C,x,ls?ggml_mul(C,ao,w.ls1):ao);
    ggml_tensor*h=ffn_gelu_erf(C,w.Wfc1,w.bfc1,w.Wfc2,w.bfc2,layer_norm(C,x,w.n2w,w.n2b,eps));
    return ggml_add(C,x,ls?ggml_mul(C,h,w.ls2):h);
}

inline ggml_tensor* tower(ggml_context*C, ggml_tensor*pix, ggml_tensor*pw, ggml_tensor*pb, ggml_tensor*pos,
                          ggml_tensor*cls, ggml_tensor*reg, const std::vector<ViTLayerW>&blk,
                          int64_t hidden, int64_t heads, int64_t hd, int64_t patch, float eps, bool prefix){
    ggml_tensor*conv=ggml_conv_2d(C,pw,pix,patch,patch,0,0,1,1);
    // Patch count from the conv, not a constant: both callers run 224/14 today,
    // and a different input size would otherwise reshape into the wrong grid.
    const int64_t NP=conv->ne[0]*conv->ne[1], nprefix=prefix?5:0, N=NP+nprefix;
    ggml_tensor*pt=ggml_cont(C,ggml_transpose(C,ggml_reshape_2d(C,conv,NP,hidden)));
    pt=ggml_add(C,pt,pb); pt=ggml_add(C,pt,pos);
    ggml_tensor*x=pt;
    if(prefix){
        ggml_tensor*tok=ggml_concat(C,ggml_reshape_2d(C,cls,hidden,1),reg,1);
        x=ggml_concat(C,tok,pt,1);
    }
    for(size_t i=0;i<blk.size();++i)
        x=vit_block(C,blk[i],x,N,hidden,heads,hd,eps,prefix);
    if(prefix)
        x=ggml_cont(C,ggml_view_2d(C,x,hidden,NP,x->nb[1],nprefix*x->nb[1]));
    return x;
}

inline ggml_tensor* DualTower::encode(ggml_backend_t backend, graph_cache<int64_t,VisIO> & cache, const Inputs & in, const char * tag) const {
    const int64_t S=image_size, n_views=in.n_images;
    // towers read S*S*3 per view; reject any view that is not exactly SxS.
    for (int64_t v=0; v<n_views; ++v) {
        const ImageView& iv = in.images[v];
        if (!view_is_side(iv.data, iv.w, iv.h, S)) {
            std::fprintf(stderr, "vla(%s): image view %lld is %dx%d, expected %lldx%lld\n",
                         tag, (long long) v, iv.w, iv.h, (long long) S, (long long) S);
            return nullptr;
        }
    }
    // ImageNet constants as bf16 rounds them (0.485 -> 0.484375). The reference
    // preprocesses in bf16, so these are the values it actually sees.
    static const float DMEAN[3]={0.484375f,0.455078125f,0.40625f}, DSTD[3]={0.228515625f,0.2236328125f,0.224609375f};

    const size_t max_nodes=(size_t)64*(d_layers+s_layers+1)*n_views+1024;
    const bool built=cache.ensure(backend,n_views,ggml_tensor_overhead()*max_nodes+ggml_graph_overhead_custom(max_nodes,false),
                                  [&](ggml_context*C, VisIO&io)->ggml_cgraph*{
        io.px_d.resize(n_views); io.px_s.resize(n_views);
        std::vector<ggml_tensor*> cmb(n_views);
        for(int v=0; v<n_views; ++v){
            io.px_d[v]=ggml_new_tensor_3d(C,GGML_TYPE_F32,S,S,3); ggml_set_input(io.px_d[v]);
            io.px_s[v]=ggml_new_tensor_3d(C,GGML_TYPE_F32,S,S,3); ggml_set_input(io.px_s[v]);
            ggml_tensor*pd=tower(C,io.px_d[v],d_patch_w,d_patch_b,d_pos,d_cls,d_reg,dvit,d_hidden,d_heads,d_head_dim,patch_size,ln_eps,true);
            ggml_tensor*ps=tower(C,io.px_s[v],s_patch_w,s_patch_b,s_pos,nullptr,nullptr,svit,s_hidden,s_heads,s_head_dim,patch_size,ln_eps,false);
            cmb[v]=ggml_concat(C,pd,ps,0);
        }
        ggml_tensor*allp=cmb[0]; for(int v=1;v<n_views;++v) allp=ggml_concat(C,allp,cmb[v],1);
        ggml_tensor*ph=ggml_gelu_erf(C,linear(C,pj_fc1w,pj_fc1b,allp));
        ph=ggml_gelu_erf(C,linear(C,pj_fc2w,pj_fc2b,ph));
        io.proj=linear(C,pj_fc3w,pj_fc3b,ph); ggml_set_output(io.proj);
        ggml_cgraph*vg=ggml_new_graph_custom(C,max_nodes,false); ggml_build_forward_expand(vg,io.proj);
        return vg;
    });
    if(!built){ std::fprintf(stderr,"vla(%s): vision gallocr failed\n",tag); return nullptr; }
    VisIO&io=cache.io();
    std::vector<float> dbuf, sbuf;
    for(int v=0;v<n_views;++v){
        preprocess_image_chw(tag,in.images[v],S,DMEAN,DSTD,dbuf); ggml_backend_tensor_set(io.px_d[v],dbuf.data(),0,ggml_nbytes(io.px_d[v]));
        preprocess_image_chw(tag,in.images[v],S,sbuf); ggml_backend_tensor_set(io.px_s[v],sbuf.data(),0,ggml_nbytes(io.px_s[v]));
    }
    graph_unique_names(cache.graph());
    if(ggml_backend_graph_compute(backend,cache.graph())!=GGML_STATUS_SUCCESS){ std::fprintf(stderr,"vla(%s): vision compute failed\n",tag); return nullptr; }
    return io.proj;
}

struct Q99Stats {
    std::vector<float> q01, q99;
    std::vector<uint8_t> mask;
    std::string suite;

    bool parse(const std::string & js, const char * env_key, const char * tag, int64_t want) {
        auto find_key = [&](size_t from, const std::string & key) -> size_t {
            const std::string pat = "\"" + key + "\"";
            return js.find(pat, from);
        };
        const char * env = std::getenv(env_key);
        size_t suite_pos;
        if (env) {
            suite = env;
            suite_pos = find_key(0, suite);
        }
        else {
            size_t b = js.find('{'); size_t q = js.find('"', b);
            size_t qe = js.find('"', q+1);
            suite = js.substr(q+1, qe-q-1); suite_pos = q;
        }
        if (suite_pos == std::string::npos) {
            std::fprintf(stderr, "vla(%s): suite '%s' not in stats\n", tag, suite.c_str());
            return false;
        }
        size_t act = find_key(suite_pos, "action");
        if (act == std::string::npos)
            return false;
        auto read_arr = [&](const std::string & key, std::vector<float> & out) -> bool {
            size_t k = find_key(act, key); if (k == std::string::npos) return false;
            size_t lb = js.find('[', k); size_t rb = js.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos)
                return false;
            out.clear(); size_t p = lb+1;
            while (p < rb) {
                while (p < rb && (js[p] == ',' || js[p] == ' ' || js[p] == '\n' || js[p] == '\t' || js[p] == '\r'))
                    ++p;
                if (p >= rb)
                    break;
                bool t = (js.compare(p, 4, "true") == 0), f = (js.compare(p, 5, "false") == 0);
                if (t || f) {
                    out.push_back(t ? 1.0f : 0.0f);
                    p += t ? 4 : 5;
                }
                else {
                    out.push_back(std::strtof(js.c_str()+p, nullptr));
                    while (p < rb && js[p] != ',')
                        ++p;
                }
            }
            return true;
        };
        std::vector<float> mk;
        if (!read_arr("q01", q01) || !read_arr("q99", q99))
            return false;
        if (!read_arr("mask", mk))
            mk.assign(want, 1.0f);
        mask.assign(mk.size(), 1); for (size_t i=0; i<mk.size(); ++i) mask[i] = mk[i] != 0.0f ? 1 : 0;
        return (int64_t) q01.size() == want && (int64_t) q99.size() == want;
    }

    std::vector<float> unnorm(const std::vector<float> & na, int64_t chunk, int64_t dim, int64_t W) const {
        std::vector<float> out((size_t)chunk*W,0.0f);
        for(int64_t c=0;c<chunk;++c) for(int64_t d=0;d<dim;++d){ float v=na[c*dim+d];
            bool msk = (d<(int64_t)mask.size()) ? mask[d]!=0 : true;
            out[c*W+d] = (msk && !q01.empty()) ? 0.5f*(v+1.0f)*(q99[d]-q01[d]+1e-8f)+q01[d] : v; }
        return out;
    }
};

}  // namespace vla
