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

#include "cosmos3_edge_prompt.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace vla {

namespace {

constexpr const char * kConcatViewTemplate = "This video contains concatenated views from multiple camera perspectives.";

// Chat template of the understanding stream (the Cosmos3 Edge template: an empty system turn, one user turn,
// the generation prompt with an empty think block) followed by the end-of-turn token and the token that starts
// the generation stream, as Cosmos3TokenizerProcessorStep.encode builds it. The pieces around the caption.
constexpr const char * kPromptHead = "\n<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
constexpr const char * kPromptTail = "<|im_end|>\n<|im_start|>assistant\n<think>\n<|im_end|><|vision_start|>";

// ---- ASCII classes of the pre-tokenizer ------------------------------------------------------------
inline bool is_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
inline bool is_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
inline bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool is_letter(unsigned char c) { return is_upper(c) || is_lower(c); }
inline bool is_nl(unsigned char c) { return c == '\r' || c == '\n'; }
inline bool is_space(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

// The length of the pre-token that starts at `i`, for the Split regex of the tokenizer:
//   [^\r\n\p{L}\p{N}]?[\p{Lu}...]*[\p{Ll}...]+  |  [^\r\n\p{L}\p{N}]?[\p{Lu}...]+[\p{Ll}...]*  |  \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n/]*  |  \s*[\r\n]+  |  \s+(?!\S)  |  \s+
// restricted to ASCII, where the letter classes are A-Z and a-z. The first alternative that matches wins.
size_t pretoken_len(const std::string & t, size_t i) {
    const size_t n = t.size();
    auto at = [&](size_t k) { return (unsigned char) t[k]; };

    // 1 and 2: an optional leading symbol, then capitals and lower case letters.
    for (int alt = 1; alt <= 2; alt++) {
        for (int with_prefix = 1; with_prefix >= 0; with_prefix--) {
            size_t p = i;
            if (with_prefix) {
                if (p >= n || is_nl(at(p)) || is_letter(at(p)) || is_digit(at(p)))
                    continue;
                p++;
            }
            size_t u = p;
            while (u < n && is_upper(at(u)))
                u++;
            size_t l = u;
            while (l < n && is_lower(at(l)))
                l++;
            if (alt == 1 && l > u)
                return l - i;                       // upper* lower+
            if (alt == 2 && u > p)
                return l - i;                       // upper+ lower*
        }
    }
    // 3: one digit.
    if (is_digit(at(i)))
        return 1;
    // 4: an optional space, symbols, then newlines and slashes.
    {
        size_t p = i;
        if (at(p) == ' ')
            p++;
        size_t s = p;
        while (s < n && !is_space(at(s)) && !is_letter(at(s)) && !is_digit(at(s)))
            s++;
        if (s > p) {
            while (s < n && (is_nl(at(s)) || at(s) == '/'))
                s++;
            return s - i;
        }
    }
    // 5, 6 and 7: white space. \s*[\r\n]+ ends at the last line break of the run.
    size_t run = i;
    while (run < n && is_space(at(run)))
        run++;
    if (run == i)
        return 1;  // not reachable for ASCII; keeps the loop moving
    for (size_t k = run; k > i; k--)
        if (is_nl(at(k - 1)))
            return k - i;
    if (run == n)
        return run - i;                              // \s+(?!\S) at the end of the text
    if (run - i >= 2)
        return run - i - 1;                          // \s+(?!\S) leaves the last space for the next token
    return run - i;                                  // \s+
}

// GPT-2's bytes_to_unicode: printable bytes stand for themselves, the others take 256 + n in order.
void byte_level_table(std::string out[256], int cp_of_byte[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        const int  cp        = printable ? b : 256 + n++;
        cp_of_byte[b]        = cp;
        std::string s;
        if (cp < 0x80) {
            s += (char) cp;
        } else {  // cp < 0x800 here
            s += (char) (0xC0 | (cp >> 6));
            s += (char) (0x80 | (cp & 0x3F));
        }
        out[b] = s;
    }
}

std::string json_string(const std::string & s) {  // json.dumps(s) for ASCII s
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c >= 0x20 && c <= 0x7e) {
                    o += (char) c;
                } else {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                }
        }
    }
    return o + "\"";
}

std::string strip(const std::string & s) {  // str.strip() for ASCII
    size_t a = 0, b = s.size();
    while (a < b && is_space((unsigned char) s[a]))
        a++;
    while (b > a && is_space((unsigned char) s[b - 1]))
        b--;
    return s.substr(a, b - a);
}

bool ascii(const std::string & s) {
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return c < 0x80; });
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------------------------------

bool Cosmos3Prompt::load(const gguf_reader & g) {
    gguf_context * ctx = g.gctx;
    const int64_t  kt  = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    const int64_t  kg  = gguf_find_key(ctx, "tokenizer.ggml.token_type");
    const int64_t  km  = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (kt < 0 || kg < 0 || km < 0 || !g.has("cosmos3_edge.caption.view_description")) {
        std::fprintf(stderr, "vla(cosmos3_edge): this GGUF has no tokenizer; convert with --policy-dir, or pass --tokens\n");
        return false;
    }
    if (gguf_get_arr_type(ctx, kt) != GGUF_TYPE_STRING || gguf_get_arr_type(ctx, km) != GGUF_TYPE_STRING ||
        gguf_get_arr_type(ctx, kg) != GGUF_TYPE_INT32) {
        std::fprintf(stderr, "vla(cosmos3_edge): the tokenizer arrays have unexpected types\n");
        return false;
    }
    const size_t n = gguf_get_arr_n(ctx, kt);
    if (n == 0 || gguf_get_arr_n(ctx, kg) != n) {
        std::fprintf(stderr, "vla(cosmos3_edge): the token and token-type arrays disagree\n");
        return false;
    }
    int cp_of_byte[256];
    byte_level_table(byte_str_, cp_of_byte);
    std::unordered_map<int, int> byte_of_cp;
    for (int b = 0; b < 256; b++)
        byte_of_cp[cp_of_byte[b]] = b;

    const int32_t * types = (const int32_t *) gguf_get_arr_data(ctx, kg);
    tokens_.resize(n);
    vocab_.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        tokens_[i] = gguf_get_arr_str(ctx, kt, i);
        vocab_.emplace(tokens_[i], (int32_t) i);
        // llama.cpp's token types: 3 control, 4 user defined. Both are matched as such in the text.
        if (types[i] == 3 || types[i] == 4) {
            std::string raw;  // back from the byte-level spelling to bytes
            bool        ok = true;
            for (size_t k = 0; k < tokens_[i].size();) {
                const unsigned char c = (unsigned char) tokens_[i][k];
                int cp, len;
                if (c < 0x80) {
                    cp = c; len = 1;
                } else if ((c & 0xE0) == 0xC0 && k + 1 < tokens_[i].size()) {
                    cp = ((c & 0x1F) << 6) | ((unsigned char) tokens_[i][k + 1] & 0x3F); len = 2;
                } else {
                    ok = false;
                    break;
                }
                const auto it = byte_of_cp.find(cp);
                if (it == byte_of_cp.end()) { ok = false; break; }
                raw += (char) it->second;
                k += (size_t) len;
            }
            if (ok && !raw.empty())
                specials_.emplace_back(raw, (int32_t) i);
        }
    }
    std::stable_sort(specials_.begin(), specials_.end(),
                     [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });

    const size_t nm = gguf_get_arr_n(ctx, km);
    rank_.reserve(nm * 2);
    for (size_t i = 0; i < nm; i++)
        rank_.emplace(gguf_get_arr_str(ctx, km, i), (int32_t) i);

    view_description_ = g.str("cosmos3_edge.caption.view_description");
    aspect_ratio_     = g.str("cosmos3_edge.caption.aspect_ratio");
    fps_              = g.f32("cosmos3_edge.caption.fps");
    use_state_        = g.u32("cosmos3_edge.caption.use_state") != 0;
    chunk_            = (int) g.u32("cosmos3_edge.chunk_size");
    canvas_h_         = (int) g.u32("cosmos3_edge.canvas_h");
    canvas_w_         = (int) g.u32("cosmos3_edge.canvas_w");
    if (fps_ <= 0.0 || chunk_ <= 0 || aspect_ratio_.empty()) {
        std::fprintf(stderr, "vla(cosmos3_edge): the caption settings of the GGUF are incomplete\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------
// The caption and the chat text
// ---------------------------------------------------------------------------------------------------

std::string Cosmos3Prompt::caption(const std::string & task_in) const {
    std::string task = strip(task_in);
    if (task.empty())
        return "";
    if (task.back() != '.' && task.back() != '!' && task.back() != '?')
        task += ".";
    const double duration = (double) (chunk_ + 1) / fps_;
    const long   rounded  = (long) std::nearbyint(duration);  // Python's round(): half to even
    char         time[48];
    std::snprintf(time, sizeof(time), "0:00-%ld:%02ld", rounded / 60, rounded % 60);

    std::string framing = kConcatViewTemplate;
    const std::string vd = strip(view_description_);
    if (!vd.empty())
        framing += " " + vd;

    std::string o = "{\"cinematography\": {\"framing\": " + json_string(framing) + "}, \"actions\": [{\"time\": " +
                    json_string(time) + ", \"description\": " + json_string(task) + ", \"idle_frame\": " +
                    json_string("0 out of " + std::to_string(chunk_ + (use_state_ ? 1 : 0)) + ".") + "}], \"duration\": " +
                    json_string(std::to_string((long) duration) + "s") + ", \"fps\": " + std::to_string((long) fps_) +
                    ".0, \"resolution\": {\"H\": " + std::to_string(canvas_h_) + ", \"W\": " + std::to_string(canvas_w_) +
                    "}, \"aspect_ratio\": " + json_string(aspect_ratio_) + "}";
    return o;
}

std::string Cosmos3Prompt::chat_text(const std::string & task) const {
    return std::string(kPromptHead) + caption(task) + kPromptTail;
}

// ---------------------------------------------------------------------------------------------------
// BPE
// ---------------------------------------------------------------------------------------------------

bool Cosmos3Prompt::bpe_segment(const std::string & text, std::vector<int32_t> & ids, std::string & err) const {
    if (!ascii(text)) {
        err = "non-ASCII text: this tokenizer implements the ASCII letter classes only";
        return false;
    }
    for (size_t i = 0; i < text.size();) {
        const size_t len = pretoken_len(text, i);
        std::string  word;  // the pre-token in the byte-level alphabet
        for (size_t k = i; k < i + len; k++)
            word += byte_str_[(unsigned char) text[k]];
        i += len;

        // ignore_merges: a pre-token that is a token is not merged.
        const auto whole = vocab_.find(word);
        if (whole != vocab_.end()) {
            ids.push_back(whole->second);
            continue;
        }
        // Merges by rank, the leftmost pair of the lowest rank first (the order of a priority queue).
        std::vector<std::string> sym;
        for (size_t k = 0; k < word.size();) {
            const size_t cl = ((unsigned char) word[k] < 0x80) ? 1 : 2;  // byte-level characters are 1 or 2 bytes
            sym.push_back(word.substr(k, cl));
            k += cl;
        }
        for (;;) {
            int32_t best = INT32_MAX;
            size_t  at   = 0;
            for (size_t k = 0; k + 1 < sym.size(); k++) {
                const auto it = rank_.find(sym[k] + " " + sym[k + 1]);
                if (it != rank_.end() && it->second < best) {
                    best = it->second;
                    at   = k;
                }
            }
            if (best == INT32_MAX)
                break;
            sym[at] += sym[at + 1];
            sym.erase(sym.begin() + (long) at + 1);
        }
        for (const std::string & s : sym) {
            const auto it = vocab_.find(s);
            if (it == vocab_.end()) {
                err = "a symbol is not in the vocabulary: " + s;
                return false;
            }
            ids.push_back(it->second);
        }
    }
    return true;
}

bool Cosmos3Prompt::encode(const std::string & text, std::vector<int32_t> & ids, std::string & err) const {
    size_t start = 0, i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (c == '<' || c == '[') {
            const std::pair<std::string, int32_t> * hit = nullptr;
            for (const auto & sp : specials_)  // longest first
                if (text.compare(i, sp.first.size(), sp.first) == 0) {
                    hit = &sp;
                    break;
                }
            if (hit) {
                if (i > start && !bpe_segment(text.substr(start, i - start), ids, err))
                    return false;
                ids.push_back(hit->second);
                i += hit->first.size();
                start = i;
                continue;
            }
        }
        i++;
    }
    return start >= text.size() || bpe_segment(text.substr(start), ids, err);
}

bool Cosmos3Prompt::tokens(const std::string & task, std::vector<int32_t> & ids, std::string & err) const {
    ids.clear();
    // The caption is json.dumps of the task, which escapes non-ASCII as UTF-16 \u sequences: refuse it here
    // instead of escaping bytes wrongly.
    if (!ascii(task)) {
        err = "non-ASCII instruction: only ASCII text is supported";
        return false;
    }
    return encode(chat_text(task), ids, err);
}

// ---------------------------------------------------------------------------------------------------
// rot6d -> axis-angle
// ---------------------------------------------------------------------------------------------------

namespace {

// Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations; eigenvalues descending, the
// eigenvectors as the columns of v.
void jacobi3(double a[3][3], double w[3], double v[3][3]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v[i][j] = i == j ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 50; sweep++) {
        const double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
        if (off < 1e-300)
            break;
        for (int p = 0; p < 2; p++)
            for (int q = p + 1; q < 3; q++) {
                if (std::fabs(a[p][q]) < 1e-300)
                    continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t     = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; k++) {  // columns p and q of a
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++) {  // rows p and q
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++) {
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
    }
    int order[3] = { 0, 1, 2 };
    std::sort(order, order + 3, [&](int x, int y) { return a[x][x] > a[y][y]; });
    double v2[3][3];
    for (int j = 0; j < 3; j++) {
        w[j] = a[order[j]][order[j]];
        for (int i = 0; i < 3; i++)
            v2[i][j] = v[i][order[j]];
    }
    std::memcpy(v, v2, sizeof(v2));
}

}  // namespace

void cosmos3_libero_action(const float * a, float * out) {
    const double f[3] = { a[3], a[4], a[5] }, s[3] = { a[6], a[7], a[8] };
    const double c[3] = { f[1] * s[2] - f[2] * s[1], f[2] * s[0] - f[0] * s[2], f[0] * s[1] - f[1] * s[0] };
    double       m[3][3];  // columns: first, second, their cross product
    for (int r = 0; r < 3; r++) {
        m[r][0] = f[r];
        m[r][1] = s[r];
        m[r][2] = c[r];
    }
    // The nearest rotation (the SVD projection onto SO(3)): U V^T, with the last singular direction flipped
    // if that is a reflection. Singular values and V from the eigenproblem of M^T M.
    double mtm[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mtm[i][j] = m[0][i] * m[0][j] + m[1][i] * m[1][j] + m[2][i] * m[2][j];
    double w[3], v[3][3];
    jacobi3(mtm, w, v);

    double u[3][3] = {};
    const double smax = std::sqrt(std::max(w[0], 0.0));
    bool         ok   = smax > 0.0;
    for (int j = 0; j < 3 && ok; j++) {
        const double sj = std::sqrt(std::max(w[j], 0.0));
        if (sj <= 1e-12 * smax) {
            if (j == 0) { ok = false; break; }
            // A vanishing singular value: complete U to an orthonormal frame.
            if (j == 1) {  // pick any vector orthogonal to u0
                const double ax = std::fabs(u[0][0]), ay = std::fabs(u[1][0]), az = std::fabs(u[2][0]);
                double t[3] = { 0, 0, 0 };
                t[ax <= ay && ax <= az ? 0 : (ay <= az ? 1 : 2)] = 1.0;
                const double d = t[0] * u[0][0] + t[1] * u[1][0] + t[2] * u[2][0];
                double x[3] = { t[0] - d * u[0][0], t[1] - d * u[1][0], t[2] - d * u[2][0] };
                const double nx = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
                for (int r = 0; r < 3; r++) u[r][1] = x[r] / nx;
            } else {
                u[0][2] = u[1][0] * u[2][1] - u[2][0] * u[1][1];
                u[1][2] = u[2][0] * u[0][1] - u[0][0] * u[2][1];
                u[2][2] = u[0][0] * u[1][1] - u[1][0] * u[0][1];
            }
            continue;
        }
        for (int r = 0; r < 3; r++)
            u[r][j] = (m[r][0] * v[0][j] + m[r][1] * v[1][j] + m[r][2] * v[2][j]) / sj;
    }
    double R[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    if (ok) {
        auto compose = [&]() {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    R[i][j] = u[i][0] * v[j][0] + u[i][1] * v[j][1] + u[i][2] * v[j][2];
        };
        compose();
        const double det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
                           R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
        if (det < 0) {
            for (int r = 0; r < 3; r++) u[r][2] = -u[r][2];
            compose();
        }
    }
    // matrix_to_axis_angle (angles below pi), in double as lerobot does.
    const double skew[3] = { R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1] };
    const double sin_a   = std::sqrt(skew[0] * skew[0] + skew[1] * skew[1] + skew[2] * skew[2]) / 2.0;
    const double cos_a   = (R[0][0] + R[1][1] + R[2][2] - 1.0) / 2.0;
    const double angle   = std::atan2(sin_a, cos_a);
    const double scale   = sin_a > 1e-8 ? angle / std::max(sin_a, 1e-8) : 1.0;

    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    for (int r = 0; r < 3; r++)
        out[3 + r] = (float) (skew[r] * scale / 2.0);
    out[6] = std::min(1.0f, std::max(-1.0f, 1.0f - 2.0f * a[9]));
}

}  // namespace vla
