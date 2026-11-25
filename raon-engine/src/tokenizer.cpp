#include "tokenizer.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>

#include "gguf.h"
#include "unicode_tables.h"

namespace raon {
namespace {

enum : uint16_t { F_NUMBER = 0x2, F_LETTER = 0x4, F_MARK = 0x10, F_WS = 0x100 };

// Per-code-point flags, built once (as llama.cpp's unicode_cpt_flags_array).
const std::vector<uint16_t>& cpt_flags() {
    static const std::vector<uint16_t> t = [] {
        std::vector<uint16_t> f(0x110000, 1);
        const size_t n = sizeof(uni::k_flags) / sizeof(uni::k_flags[0]);
        for (size_t i = 1; i < n; i++)
            for (uint32_t c = uni::k_flags[i - 1].start; c < uni::k_flags[i].start; c++) f[c] = (uint16_t)uni::k_flags[i - 1].flags;
        for (uint32_t c : uni::k_whitespace) f[c] |= F_WS;
        return f;
    }();
    return t;
}
inline uint16_t flags_of(uint32_t c) { return c < 0x110000 ? cpt_flags()[c] : 1; }

uint32_t ccc_of(uint32_t c) {
    const auto* b = uni::k_ccc;
    const auto* e = b + sizeof(uni::k_ccc) / sizeof(uni::k_ccc[0]);
    const auto* it = std::lower_bound(b, e, c, [](const uni::Ccc& x, uint32_t v) { return x.cpt < v; });
    return it != e && it->cpt == c ? it->cls : 0;
}
const uni::Dec* dec_of(uint32_t c) {
    const auto* b = uni::k_dec;
    const auto* e = b + sizeof(uni::k_dec) / sizeof(uni::k_dec[0]);
    const auto* it = std::lower_bound(b, e, c, [](const uni::Dec& x, uint32_t v) { return x.cpt < v; });
    return it != e && it->cpt == c ? it : nullptr;
}
uint32_t compose(uint32_t first, uint32_t second) {
    if (first >= 0x1100 && first < 0x1100 + 19 && second >= 0x1161 && second < 0x1161 + 21)
        return 0xAC00 + ((first - 0x1100) * 21 + second - 0x1161) * 28;
    if (first >= 0xAC00 && first < 0xAC00 + 11172 && (first - 0xAC00) % 28 == 0 && second > 0x11A7 && second < 0x11A7 + 28)
        return first + second - 0x11A7;
    const uint64_t key = (uint64_t(first) << 32) | second;
    const auto* b = uni::k_comp;
    const auto* e = b + sizeof(uni::k_comp) / sizeof(uni::k_comp[0]);
    const auto* it = std::lower_bound(b, e, key, [](const uni::Comp& x, uint64_t v) { return x.pair < v; });
    return it != e && it->pair == key ? it->cpt : 0;
}

// llama.cpp unicode_cpts_normalize_nfc, including its exact loop structure.
std::vector<uint32_t> nfc(const std::vector<uint32_t>& cpts) {
    std::vector<uint32_t> d, pending;
    d.reserve(cpts.size());
    for (uint32_t c0 : cpts) {
        pending.push_back(c0);
        while (!pending.empty()) {
            uint32_t c = pending.back();
            pending.pop_back();
            if (c >= 0xAC00 && c < 0xAC00 + 11172) {
                const uint32_t i = c - 0xAC00;
                d.push_back(0x1100 + i / 588);
                d.push_back(0x1161 + (i % 588) / 28);
                if (i % 28) d.push_back(0x11A7 + i % 28);
                continue;
            }
            const uni::Dec* x = dec_of(c);
            if (!x) { d.push_back(c); continue; }
            if (x->b) pending.push_back(x->b);
            pending.push_back(x->a);
        }
    }
    for (size_t i = 0; i < d.size();) {
        if (ccc_of(d[i]) == 0) ++i;
        const size_t begin = i;
        while (i < d.size() && ccc_of(d[i]) != 0) ++i;
        std::stable_sort(d.begin() + begin, d.begin() + i, [](uint32_t a, uint32_t b) { return ccc_of(a) < ccc_of(b); });
    }
    std::vector<uint32_t> r;
    r.reserve(d.size());
    size_t starter = 0;
    uint32_t prev = 0;
    for (uint32_t c : d) {
        const uint32_t cur = ccc_of(c);
        const uint32_t comp = !r.empty() && (prev == 0 || prev < cur) ? compose(r[starter], c) : 0;
        if (comp) {
            r[starter] = comp;
        } else {
            if (cur == 0) starter = r.size();
            r.push_back(c);
            prev = cur;
        }
    }
    return r;
}

// llama.cpp unicode_cpts_from_utf8: an invalid sequence becomes U+FFFD and consumes one byte.
std::vector<uint32_t> utf8_to_cpts(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    const size_t n = s.size();
    size_t o = 0;
    auto cont = [&](size_t i) { return i < n && ((uint8_t)s[i] & 0xC0) == 0x80; };
    while (o < n) {
        const uint8_t c = s[o];
        uint32_t r = 0;
        int len = 0;
        if (!(c & 0x80)) { r = c; len = 1; }
        else if (!(c & 0x40)) len = 0;
        else if (!(c & 0x20)) {
            if (cont(o + 1)) { r = ((c & 0x1f) << 6) | (s[o + 1] & 0x3f); len = r < 0x80 ? 0 : 2; }
        } else if (!(c & 0x10)) {
            if (cont(o + 1) && cont(o + 2)) {
                r = ((c & 0x0f) << 12) | ((s[o + 1] & 0x3f) << 6) | (s[o + 2] & 0x3f);
                len = r < 0x800 ? 0 : 3;
            }
        } else if (!(c & 0x08)) {
            if (cont(o + 1) && cont(o + 2) && cont(o + 3)) {
                r = ((c & 0x07) << 18) | ((s[o + 1] & 0x3f) << 12) | ((s[o + 2] & 0x3f) << 6) | (s[o + 3] & 0x3f);
                len = (r < 0x10000 || r > 0x10ffff) ? 0 : 4;
            }
        }
        if (len == 0) { out.push_back(0xFFFD); o += 1; }
        else { out.push_back(r); o += len; }
    }
    return out;
}

void cpt_to_utf8(uint32_t c, std::string& out) {
    if (c <= 0x7f) out += (char)c;
    else if (c <= 0x7ff) { out += (char)(0xc0 | (c >> 6)); out += (char)(0x80 | (c & 0x3f)); }
    else if (c <= 0xffff) {
        out += (char)(0xe0 | (c >> 12)); out += (char)(0x80 | ((c >> 6) & 0x3f)); out += (char)(0x80 | (c & 0x3f));
    } else {
        out += (char)(0xf0 | (c >> 18)); out += (char)(0x80 | ((c >> 12) & 0x3f));
        out += (char)(0x80 | ((c >> 6) & 0x3f)); out += (char)(0x80 | (c & 0x3f));
    }
}

// llama.cpp unicode_regex_split_custom_raon: lengths (in code points) of the pre-tokenizer chunks.
std::vector<size_t> raon_split(const std::vector<uint32_t>& cpts) {
    std::vector<size_t> result;
    const size_t end = cpts.size();
    auto flags = [&](size_t p) -> uint16_t { return p < end ? flags_of(cpts[p]) : 0; };
    auto cjk = [&](size_t p) {
        return p < end && ((cpts[p] >= 0x4e00 && cpts[p] <= 0x9fa5) || (cpts[p] >= 0x3040 && cpts[p] <= 0x30ff));
    };
    auto letter = [&](size_t p) { return (flags(p) & F_LETTER) && !cjk(p); };
    for (size_t pos = 0; pos < end;) {
        const size_t start = pos;
        size_t p = pos + (cpts[pos] == ' ');
        if (cjk(p)) {
            do { ++p; } while (cjk(p));
            pos = p;
        } else if (letter(p)) {
            while (true) {
                ++p;
                while (flags(p) & F_MARK) ++p;
                if (letter(p)) continue;
                if (p < end && cpts[p] == ' ' && letter(p + 1)) { ++p; continue; }
                break;
            }
            pos = p;
        } else if (flags(pos) & F_NUMBER) {
            p = pos;
            while (flags(p) & F_NUMBER) ++p;
            const size_t first = (p - pos) % 3;
            pos += first ? first : 3;
        } else {
            auto punct = [&](size_t q) {
                const uint16_t f = flags(q);
                return q < end && !(f & F_WS) && !(f & F_LETTER) && !(f & F_NUMBER);
            };
            if (punct(p)) {
                do { ++p; } while (punct(p));
                while (p < end && (cpts[p] == '\r' || cpts[p] == '\n' || cpts[p] == '/')) ++p;
                pos = p;
            } else {
                p = pos;
                size_t last_nl = pos;
                while (flags(p) & F_WS) {
                    if (cpts[p] == '\r' || cpts[p] == '\n') last_nl = p + 1;
                    ++p;
                }
                if (last_nl > pos) pos = last_nl;
                else if (p > pos) pos = p < end && p - pos > 1 ? p - 1 : p;
                else ++pos;
            }
        }
        result.push_back(pos - start);
    }
    return result;
}

// GPT-2 byte <-> unicode mapping.
void byte_map(uint32_t to_cpt[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        to_cpt[b] = keep ? b : 256 + n++;
    }
}

}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
    GGUF g(path);
    if (g.s("tokenizer.ggml.model") != "gpt2") throw std::runtime_error("tokenizer: expected gpt2 (byte-level BPE)");
    if (g.has("tokenizer.ggml.pre") && g.s("tokenizer.ggml.pre") != "raon")
        throw std::runtime_error("tokenizer: pre-tokenizer '" + g.s("tokenizer.ggml.pre") + "' is not supported (raon only)");
    const auto& toks = g.get("tokenizer.ggml.tokens").arr;
    const auto& types = g.get("tokenizer.ggml.token_type").arr;
    const auto& merges = g.get("tokenizer.ggml.merges").arr;
    eos_ = (int)g.u("tokenizer.ggml.eos_token_id");
    if (g.has("tokenizer.ggml.padding_token_id")) pad_ = (int)g.u("tokenizer.ggml.padding_token_id");
    if (g.has("tokenizer.chat_template")) chat_template_ = g.s("tokenizer.chat_template");

    uint32_t to_cpt[256];
    byte_map(to_cpt);
    std::unordered_map<uint32_t, int> cpt_to_byte;
    for (int b = 0; b < 256; b++) cpt_to_byte[to_cpt[b]] = b;

    const int V = (int)toks.size();
    texts_.resize(V);
    pieces_.resize(V);
    special_.assign(V, 0);
    text_to_id_.reserve(V * 2);
    for (int i = 0; i < V; i++) {
        texts_[i] = toks[i].s;
        text_to_id_.emplace(texts_[i], i);
        const int type = (int)types[i].u;   // 1 normal, 3 control, 4 user-defined, 2 unknown, 6 byte
        if (type == 3 || type == 4 || type == 2) {
            special_[i] = 1;
            specials_.push_back(i);
            pieces_[i] = texts_[i];
        } else {
            std::string bytes;
            for (uint32_t c : utf8_to_cpts(texts_[i])) {
                auto it = cpt_to_byte.find(c);
                if (it != cpt_to_byte.end()) bytes += (char)it->second;
                else cpt_to_utf8(c, bytes);   // not a byte-level token: keep the text
            }
            pieces_[i] = bytes;
        }
    }
    std::stable_sort(specials_.begin(), specials_.end(), [&](int a, int b) { return texts_[a].size() > texts_[b].size(); });
    for (int id : specials_) all_lt_ = all_lt_ && !texts_[id].empty() && texts_[id][0] == '<';

    for (int b = 0; b < 256; b++) {
        std::string s;
        cpt_to_utf8(to_cpt[b], s);
        byte_tok_[b] = find(s);
        if (byte_tok_[b] < 0) throw std::runtime_error("tokenizer: byte token missing for byte " + std::to_string(b));
    }
    // Merges as token-id pairs. llama.cpp merges strings; ids are equivalent as long as every merge's parts and
    // result are tokens, which is checked here.
    merges_.reserve(merges.size() * 2);
    size_t bad = 0;
    for (size_t r = 0; r < merges.size(); r++) {
        const std::string& m = merges[r].s;
        const size_t sp = m.find(' ', 1);
        if (sp == std::string::npos) { bad++; continue; }
        const std::string a = m.substr(0, sp), b = m.substr(sp + 1);
        const int ia = find(a), ib = find(b), iab = find(a + b);
        if (ia < 0 || ib < 0 || iab < 0) { bad++; continue; }
        merges_.emplace((uint64_t(ia) << 32) | (uint32_t)ib, std::make_pair((int)r, iab));
    }
    if (bad) throw std::runtime_error("tokenizer: " + std::to_string(bad) + " merges reference non-tokens");
}

int Tokenizer::find(const std::string& text) const {
    auto it = text_to_id_.find(text);
    return it == text_to_id_.end() ? -1 : it->second;
}

void Tokenizer::bpe_word(const std::vector<uint8_t>& bytes, std::vector<int>& out) const {
    struct Sym { int id, prev, next; };
    struct Big { int rank, left, right, lid, rid, merged; };
    auto worse = [](const Big& l, const Big& r) { return l.rank > r.rank || (l.rank == r.rank && l.left > r.left); };
    std::vector<Sym> sym(bytes.size());
    for (size_t i = 0; i < bytes.size(); i++) sym[i] = {byte_tok_[bytes[i]], (int)i - 1, i + 1 == bytes.size() ? -1 : (int)i + 1};
    std::priority_queue<Big, std::vector<Big>, decltype(worse)> q(worse);
    auto add = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        auto it = merges_.find((uint64_t(sym[l].id) << 32) | (uint32_t)sym[r].id);
        if (it == merges_.end()) return;
        q.push({it->second.first, l, r, sym[l].id, sym[r].id, it->second.second});
    };
    for (int i = 1; i < (int)sym.size(); i++) add(i - 1, i);
    while (!q.empty()) {
        const Big b = q.top();
        q.pop();
        Sym& L = sym[b.left];
        Sym& R = sym[b.right];
        if (L.id < 0 || R.id < 0 || L.id != b.lid || R.id != b.rid || L.next != b.right) continue;   // stale
        L.id = b.merged;
        R.id = -1;
        L.next = R.next;
        if (R.next >= 0) sym[R.next].prev = b.left;
        add(L.prev, b.left);
        add(b.left, L.next);
    }
    for (int i = 0; i >= 0 && i < (int)sym.size(); i = sym[i].next) out.push_back(sym[i].id);
}

void Tokenizer::encode_fragment(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    const std::vector<uint32_t> cpts = nfc(utf8_to_cpts(text));
    std::vector<uint8_t> bytes;
    std::string word;
    size_t at = 0;
    for (size_t len : raon_split(cpts)) {
        word.clear();
        for (size_t i = at; i < at + len; i++) cpt_to_utf8(cpts[i], word);
        at += len;
        bytes.assign(word.begin(), word.end());
        bpe_word(bytes, out);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int> out;
    if (!parse_special) { encode_fragment(text, out); return out; }
    size_t frag = 0, i = 0;
    while (i < text.size()) {
        int hit = -1;
        if (!all_lt_ || text[i] == '<') {   // all special tokens of this vocab start with '<'
            for (int id : specials_) {
                const std::string& s = texts_[id];
                if (s.size() && text.compare(i, s.size(), s) == 0) { hit = id; break; }
            }
        }
        if (hit < 0) { i++; continue; }
        encode_fragment(text.substr(frag, i - frag), out);
        out.push_back(hit);
        i += texts_[hit].size();
        frag = i;
    }
    encode_fragment(text.substr(frag), out);
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids)
        if (id >= 0 && id < n_vocab()) s += pieces_[id];
    return s;
}

std::string Utf8Stream::push(const std::string& bytes) {
    pending += bytes;
    // find the start of a trailing incomplete sequence
    size_t n = pending.size(), cut = n;
    for (size_t k = 1; k <= 4 && k <= n; k++) {
        const uint8_t c = pending[n - k];
        if ((c & 0xC0) == 0x80) continue;   // continuation byte
        const size_t need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        if (need > k) cut = n - k;
        break;
    }
    std::string out = pending.substr(0, cut);
    pending.erase(0, cut);
    return out;
}

}  // namespace raon
