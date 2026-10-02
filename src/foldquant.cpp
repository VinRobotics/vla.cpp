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

#include "foldquant.h"

#include "backend.h"
#include "env_flag.h"
#include "gguf_reader.h"
#include "loader.h"
#include "cuda/vla_cuda_ops.h"
#include "foldquant_ref.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace vla {

namespace {

constexpr size_t NAME_CAP = 256;

std::string key(const char * prefix, const char * k) {
    return std::string(prefix) + ".quant." + k;
}

// "o:8,down:8" -> {o: 8, down: 8}
std::map<std::string, int> parse_site_bits(const std::string & s) {
    std::map<std::string, int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const size_t c = item.find(':');
        if (c == std::string::npos || c == 0) continue;
        out[item.substr(0, c)] = std::atoi(item.c_str() + c + 1);
    }
    return out;
}

void parse_module(const gguf_reader & g, const char * prefix, const char * mod, FqModuleSpec & m) {
    const std::string p = std::string(prefix) + ".quant." + mod + "_";
    auto u32 = [&](const char * k, int def) {
        const std::string kk = p + k;
        return g.has(kk.c_str()) ? (int) g.u32(kk.c_str()) : def;
    };
    m.wbits     = u32("weight_bits",    m.wbits);
    m.abits     = u32("act_bits",       m.abits);
    m.rot_block = u32("rot_block_size", m.rot_block);
    m.scheme    = g.str(key(prefix, (std::string("scheme_") + mod).c_str()).c_str());
}

}  // namespace

int FqModuleSpec::wbits_for(const char * site_key) const {
    const auto it = site_bits.find(site_key);
    return it != site_bits.end() ? it->second : wbits;
}

int FqModuleSpec::abits_for(const char * site_key) const {
    const auto it = site_bits.find(site_key);
    return it != site_bits.end() ? it->second : abits;
}

int fq_rot_block_for(int64_t K, int nominal) {
    int bs = nominal;
    while (bs > 1 && (K % bs) != 0)
        bs >>= 1;
    return bs < 2 ? 1 : bs;
}

bool foldquant_present(const gguf_reader & g, const char * prefix) {
    const std::string k = key(prefix, "method");
    return g.has(k.c_str()) && g.str(k.c_str()) == "foldquant";
}

FoldQuantSpec foldquant_parse(const gguf_reader & g, const char * prefix) {
    FoldQuantSpec fq;
    if (!foldquant_present(g, prefix))
        return fq;
    fq.present    = true;
    fq.method     = g.str(key(prefix, "method").c_str());
    fq.applied_at = g.str(key(prefix, "applied_at").c_str());
    fq.provenance = g.str(key(prefix, "provenance").c_str());

    parse_module(g, prefix, "llm",    fq.llm);
    parse_module(g, prefix, "action", fq.action);

    const std::string fo = g.str(key(prefix, "action_fold_order").c_str());
    fq.action.fold_before = (fo == "before");
    fq.llm.fold_before    = false;   // the LLM never ships an ascale

    const std::string clipk = key(prefix, "act_clip_ratio");
    if (g.has(clipk.c_str())) {
        const float c = g.f32(clipk.c_str());
        if (c > 0.0f) fq.llm.clip = c;
    }
    fq.llm.site_bits = parse_site_bits(g.str(key(prefix, "site_bits").c_str()));

    std::printf("vla(%s): FoldQuant GGUF (%s): llm=%s W%dA%d rot%d%s  action=%s W%dA%d rot%d fold=%s%s\n",
                prefix, fq.applied_at.c_str(),
                fq.llm.scheme.c_str(), fq.llm.wbits, fq.llm.abits, fq.llm.rot_block,
                fq.llm.site_bits.empty() ? "" : " (site_bits)",
                fq.action.scheme.c_str(), fq.action.wbits, fq.action.abits, fq.action.rot_block,
                fq.action.fold_before ? "before" : "after",
                fq.llm.clip != 1.0f ? " clip" : "");
    if (!fq.provenance.empty())
        std::printf("vla(%s): FoldQuant provenance: %s\n", prefix, fq.provenance.c_str());
    return fq;
}

bool foldquant_check_backend(const char * tag, const Backend & b, FoldQuantSpec & fq, bool weight_dtype_set) {
    if (!fq.present)
        return true;
    const char * name = b.handle ? ggml_backend_name(b.handle) : "";
    const bool is_cpu = std::strcmp(name, "CPU") == 0;
    if ((!b.is_cuda && !is_cpu) || env_flag("VLA_FQ_DEQUANT")) {
        // No FoldQuant kernels on this backend (or asked for): every site is read
        // back as a float weight in the resident GEMM type and runs the stock path.
        fq.llm.dequant = fq.action.dequant = true;
        std::printf("%s: FoldQuant sites read back as float weights on '%s' (%s): weights keep the "
                    "W%d/W%d rounding, activations stay float\n",
                    tag, name, (!b.is_cuda && !is_cpu) ? "no integer kernels on this backend" : "VLA_FQ_DEQUANT=1",
                    fq.llm.wbits, fq.action.wbits);
        return true;
    }
    if (weight_dtype_set)
        std::printf("%s: --weight-dtype applies to the float tensors; FoldQuant sites stay INT%d/INT%d\n",
                    tag, fq.llm.wbits, fq.action.wbits);
    if (b.is_cuda) {
        if (env_flag("VLA_FQ_CPU_REF")) {
            // The staging shim synchronizes the stream, which CUDA-graph
            // capture forbids; ggml reads this switch before its first compute.
            setenv_default("GGML_CUDA_DISABLE_GRAPHS", "1");
            std::printf("%s: VLA_FQ_CPU_REF=1 - FoldQuant nodes run the CPU reference on host copies "
                        "(CUDA graphs disabled)\n", tag);
        }
        cuda_register_foldquant_ops();
        std::printf("%s: FoldQuant CUDA kernels registered\n", tag);
    } else {
        std::printf("%s: FoldQuant on the CPU reference path (exact, slow)\n", tag);
    }
    return true;
}

void fq_dequant_rows(const int8_t * codes, const float * wscale, const float * ascale,
                     int64_t K, int64_t N, int wbits, int rot_block, bool fold_before, float * out) {
    // The activation path is x' = R(x / a) (fold before) or R(x) / a (after), with R
    // the block-normalised Sylvester-Hadamard butterfly, which is symmetric and
    // orthonormal; so W x' = (W R diag(1/a)) x or (W diag(1/a) R) x, and row n of
    // the float weight is R(w_n) / a or R(w_n / a).
    const int64_t     kpack = fq_w_kpack(K, wbits);
    const float       inv   = rot_block > 1 ? fqref::inv_sqrt_block(rot_block) : 1.0f;
    std::vector<int8_t> q((size_t) K);
    for (int64_t n = 0; n < N; ++n) {
        const int8_t * row = codes + (size_t) n * kpack;
        if (wbits == 4)
            fqref::unpack_nibbles(row, K, q.data());
        else
            std::copy(row, row + K, q.begin());
        float * w = out + (size_t) n * K;
        for (int64_t k = 0; k < K; ++k)
            w[k] = (float) q[(size_t) k] * wscale[n];
        if (ascale && !fold_before)
            for (int64_t k = 0; k < K; ++k) w[k] = w[k] / ascale[k];
        if (rot_block > 1)
            fqref::fwht_row(w, K, rot_block, inv);
        if (ascale && fold_before)
            for (int64_t k = 0; k < K; ++k) w[k] = w[k] / ascale[k];
    }
}

namespace {

// Dequant mode: register `<site>.weight` with the loader as a float [K, N] weight
// rebuilt from the codes, wscale and ascale. Declares nothing itself.
bool register_dequant(WeightLoader & L, const FqModuleSpec & mod, const char * site_key, const std::string & site) {
    const std::string wname = site + ".weight";
    const ggml_tensor * meta = L.reader().meta(wname.c_str());
    const int wbits = mod.wbits_for(site_key);
    if (wbits != 8 && wbits != 4) {
        std::fprintf(stderr, "vla: %s: unsupported FoldQuant weight width %d\n", site.c_str(), wbits);
        L.fail("FoldQuant widths");
        return false;
    }
    const int64_t K = wbits == 4 ? 2 * meta->ne[0] : meta->ne[0];
    const int64_t N = meta->ne[1];
    const std::string sname = site + ".wscale", aname = site + ".ascale";
    const ggml_tensor * sm = L.reader().meta(sname.c_str());
    const ggml_tensor * am = L.reader().meta(aname.c_str());
    if (ggml_n_dims(meta) != 2 || K % 64 != 0 || !sm || sm->ne[0] != N || (am && am->ne[0] != K)) {
        std::fprintf(stderr, "vla: %s: malformed FoldQuant site (K=%lld N=%lld, wscale %s, ascale %s)\n",
                     site.c_str(), (long long) K, (long long) N, sm ? "ok" : "missing", am ? "present" : "absent");
        L.fail("FoldQuant site");
        return false;
    }
    const int  rot    = fq_rot_block_for(K, mod.rot_block);
    const bool before = mod.fold_before;
    gguf_reader & g = L.reader();
    const bool has_a = am != nullptr;
    L.as_float(wname, K, N, [&g, wname, sname, aname, has_a, K, N, wbits, rot, before](std::vector<float> & out) {
        const std::vector<uint8_t> codes = g.read_convert(wname.c_str(), GGML_TYPE_I8);
        const std::vector<float>   ws    = g.read_f32(sname.c_str());
        const std::vector<float>   as    = has_a ? g.read_f32(aname.c_str()) : std::vector<float>();
        if ((int64_t) codes.size() != fq_w_kpack(K, wbits) * N || (int64_t) ws.size() != N ||
            (has_a && (int64_t) as.size() != K))
            return false;
        out.assign((size_t) (K * N), 0.0f);
        fq_dequant_rows((const int8_t *) codes.data(), ws.data(), has_a ? as.data() : nullptr,
                        K, N, wbits, rot, before, out.data());
        return true;
    });
    return true;
}

bool fill_specs(WeightLoader & L, const FqModuleSpec & mod, const char * site_key, const char * site,
                ggml_tensor * gamma, float eps, FqLinear & r) {
    const int wbits = mod.wbits_for(site_key);
    const int abits = mod.abits_for(site_key);
    if ((wbits != 8 && wbits != 4) || (abits != 8 && abits != 4)) {
        std::fprintf(stderr, "vla: %s: unsupported FoldQuant widths W%dA%d\n", site, wbits, abits);
        L.fail("FoldQuant widths");
        return false;
    }
    const int64_t K = wbits == 4 ? 2 * r.w->ne[0] : r.w->ne[0];
    const int64_t N = r.w->ne[1];
    if (ggml_n_dims(r.w) != 2 || K % 64 != 0 || N % 64 != 0) {
        std::fprintf(stderr, "vla: %s: FoldQuant site needs 2-D weight with K%%64==0 and N%%64==0, got K=%lld N=%lld\n",
                     site, (long long) K, (long long) N);
        L.fail("FoldQuant shape");
        return false;
    }
    if (r.wscale->ne[0] != N || ggml_n_dims(r.wscale) != 1) {
        std::fprintf(stderr, "vla: %s.wscale must be F32[%lld]\n", site, (long long) N);
        L.fail("FoldQuant wscale");
        return false;
    }
    if (r.ascale && (r.ascale->ne[0] != K || ggml_n_dims(r.ascale) != 1)) {
        std::fprintf(stderr, "vla: %s.ascale must be F32[%lld]\n", site, (long long) K);
        L.fail("FoldQuant ascale");
        return false;
    }
    if (gamma && gamma->ne[0] != K) {
        std::fprintf(stderr, "vla: %s: fused norm gamma has %lld entries, K=%lld\n",
                     site, (long long) gamma->ne[0], (long long) K);
        L.fail("FoldQuant gamma");
        return false;
    }
    r.gamma           = gamma;
    r.act.K           = K;
    r.act.abits       = abits;
    r.act.rot_block   = fq_rot_block_for(K, mod.rot_block);
    if (r.act.rot_block > 64) {
        std::fprintf(stderr, "vla: %s: rotation block %d exceeds the 64-element chunk the kernels rotate\n",
                     site, r.act.rot_block);
        L.fail("FoldQuant rot_block");
        return false;
    }
    r.act.fold_before = mod.fold_before;
    r.act.has_gamma   = gamma != nullptr;
    r.act.has_ascale  = r.ascale != nullptr;
    r.act.clip        = abits == 4 ? mod.clip : 1.0f;
    r.act.eps         = eps;
    r.gemm.K          = K;
    r.gemm.N          = N;
    r.gemm.wbits      = wbits;
    return true;
}

}  // namespace

FqLinear fq_declare_linear(WeightLoader & L, const FqModuleSpec & mod, const char * site_key,
                           bool has_bias, ggml_tensor * gamma, float eps, const char * site_fmt, ...) {
    char site[NAME_CAP];
    va_list ap;
    va_start(ap, site_fmt);
    const int n = std::vsnprintf(site, sizeof(site), site_fmt, ap);
    va_end(ap);
    FqLinear r;
    if (n < 0 || (size_t) n >= sizeof(site)) {
        L.fail("FoldQuant site name too long");
        return r;
    }

    const std::string wname = std::string(site) + ".weight";
    const ggml_tensor * meta = L.reader().meta(wname.c_str());
    if (!meta || meta->type != GGML_TYPE_I8)
        return r;   // not a FoldQuant site: caller declares it as a float GEMM
    if (mod.dequant) {
        register_dequant(L, mod, site_key, site);
        return r;   // the caller's float declare reads the rebuilt weight
    }

    r.w      = L.typed(GGML_TYPE_I8, "%s.weight", site);
    r.wscale = L.f32("%s.wscale", site);
    r.ascale = L.opt_typed(GGML_TYPE_F32, "%s.ascale", site);
    r.bias   = has_bias ? L.f32("%s.bias", site) : nullptr;
    if (!r.w || !r.wscale || (has_bias && !r.bias)) {
        r.w = nullptr;
        return r;
    }
    if (!fill_specs(L, mod, site_key, site, gamma, eps, r))
        r.w = nullptr;
    return r;
}

void fq_set_heads(FqLinear & s, int head_dim, int heads, uint32_t vmask) {
    static const bool off = [] { const char * e = std::getenv("VLA_FQ_NO_HEADS"); return e && *e && *e != '0'; }();
    if (!s.w || off) return;
    const int64_t parts = s.gemm.N / ((int64_t) head_dim * heads);
    if (parts < 1 || parts > 32 || s.gemm.N % ((int64_t) head_dim * heads) != 0) return;
    s.gemm.head_dim = head_dim;
    s.gemm.heads    = heads;
    s.gemm.vmask    = vmask;
}

void fq_link_prefetch(const std::vector<FqLinear *> & order) {
    FqLinear * prev = nullptr;
    for (FqLinear * s : order) {
        if (!s || !s->w) continue;
        if (prev) prev->gemm.next_w = s->w;
        prev = s;
    }
}

FqLinear fq_declare_fused(WeightLoader & L, const FqModuleSpec & mod, const char * site_key,
                          bool has_bias, const std::string & out_base, const std::vector<std::string> & sites) {
    FqLinear r;
    if (sites.empty())
        return r;

    std::vector<std::string> ws, ss, bs;
    for (const std::string & s : sites) {
        const std::string wname = s + ".weight";
        const ggml_tensor * meta = L.reader().meta(wname.c_str());
        if (!meta || meta->type != GGML_TYPE_I8) {
            if (&s != &sites[0]) {
                std::fprintf(stderr, "vla: %s: FoldQuant sites fused as %s must all be INT8 or all float\n",
                             s.c_str(), out_base.c_str());
                L.fail("FoldQuant fused group");
            }
            return r;
        }
        ws.push_back(wname);
        ss.push_back(s + ".wscale");
        bs.push_back(s + ".bias");
    }
    if (mod.dequant) {
        // Each site's ascale goes into its own rows, so they need not agree here.
        for (const std::string & s : sites)
            if (!register_dequant(L, mod, site_key, s))
                break;
        return r;   // the caller's fuse_gemm reads the rebuilt weights
    }

    r.w      = L.fuse_typed(GGML_TYPE_I8, (out_base + ".w").c_str(), ws);
    r.wscale = L.fuse_f32((out_base + ".wscale").c_str(), ss);
    r.bias   = has_bias ? L.fuse_f32((out_base + ".b").c_str(), bs) : nullptr;
    if (!r.w || !r.wscale || (has_bias && !r.bias)) {
        r.w = nullptr;
        return r;
    }

    // The fused group shares one input transform, so the SQ vectors must agree.
    const std::string a0 = sites[0] + ".ascale";
    const bool has_a0 = L.reader().meta(a0.c_str()) != nullptr;
    for (size_t i = 1; i < sites.size(); ++i) {
        const std::string ai = sites[i] + ".ascale";
        const bool has_ai = L.reader().meta(ai.c_str()) != nullptr;
        if (has_ai != has_a0) {
            std::fprintf(stderr, "vla: %s: ascale present on some fused sites but not others\n", out_base.c_str());
            L.fail("FoldQuant fused ascale");
            r.w = nullptr;
            return r;
        }
        if (has_a0) {
            const std::vector<float> v0 = L.reader().read_f32(a0.c_str());
            const std::vector<float> vi = L.reader().read_f32(ai.c_str());
            if (v0.empty() || v0 != vi) {
                std::fprintf(stderr, "vla: %s and %s differ; fused sites must share one SQ vector\n",
                             a0.c_str(), ai.c_str());
                L.fail("FoldQuant fused ascale");
                r.w = nullptr;
                return r;
            }
        }
    }
    r.ascale = has_a0 ? L.typed(GGML_TYPE_F32, "%s", a0.c_str()) : nullptr;

    if (!fill_specs(L, mod, site_key, out_base.c_str(), nullptr, 0.0f, r))
        r.w = nullptr;
    return r;
}

}  // namespace vla
