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

// DINOv2 + SigLIP dual vision tower, shared by OpenVLA-OFT and VLA-Adapter.
// DINOv2 passes prefix=true (CLS + 4 register tokens, dropped after the blocks)
// and uses LayerScale; SigLIP passes prefix=false.

#pragma once

#include "ggml.h"
#include "model.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace vla {

struct ViTLayerW { ggml_tensor *n1w,*n1b,*n2w,*n2b,*ls1,*ls2,*Wqkv,*bqkv,*Wproj,*bproj,*Wfc1,*bfc1,*Wfc2,*bfc2; };

inline ggml_tensor * LN(ggml_context*C, ggml_tensor*x, ggml_tensor*w, ggml_tensor*b, float eps){ return ggml_add(C,ggml_mul(C,ggml_norm(C,x,eps),w),b); }

inline ggml_tensor* vit_block(ggml_context*C, const ViTLayerW&w, ggml_tensor*x, int64_t N, int64_t hidden, int64_t heads, int64_t hd, float eps, bool ls){
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

inline ggml_tensor* tower(ggml_context*C, ggml_tensor*pix, ggml_tensor*pw, ggml_tensor*pb, ggml_tensor*pos,
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

// HWC to CHW planar with per-channel mean/std: ImageNet for DINOv2, 0.5 for SigLIP.
inline void normalize_tower(const ImageView& v, int64_t S, const float mean[3], const float std_[3], std::vector<float>& out){
    out.assign((size_t)3*S*S,0.0f);
    for(int64_t h=0;h<S;++h) for(int64_t w=0;w<S;++w) for(int64_t c=0;c<3;++c){
        float px = (v.format==PixelFormat::U8) ? ((const uint8_t*)v.data)[(h*S+w)*3+c]/255.0f
                                               : ((const float*)v.data)[(h*S+w)*3+c];
        out[c*S*S+h*S+w] = (px-mean[c])/std_[c];
    }
}

}  // namespace vla
