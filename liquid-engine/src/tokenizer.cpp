#include "tokenizer.h"

#include <algorithm>
#include <list>
#include <queue>
#include <stdexcept>

#include "gguf.h"
#include "unicode_data.h"

namespace {

// ---- unicode helpers (flags identical to llama.cpp's unicode_cpt_flags) ----
enum : uint16_t { F_NUMBER = 0x2, F_LETTER = 0x4, F_WHITESPACE = 0x100 };

uint16_t cpt_flags(uint32_t c) {
    // binary search last range start <= c
    int lo = 0, hi = kUniRanges - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (kUniRangeStart[mid] <= c) lo = mid; else hi = mid - 1;
    }
    uint16_t f = c <= 0x10FFFF ? kUniRangeFlags[lo] : 0x1;
    if (c < 0x3001) {
        for (int i = 0; i < kUniWhitespaceN; i++) if (kUniWhitespace[i] == c) { f |= F_WHITESPACE; break; }
    }
    return f;
}

std::string cpt_to_utf8(uint32_t c) {
    std::string s;
    if (c < 0x80) s += (char)c;
    else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
    else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
    return s;
}

int utf8_len(uint8_t c) {
    static const int lut[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return lut[c >> 4];
}

// decode utf8 like llama.cpp (invalid sequences -> U+FFFD, consuming one byte)
std::vector<uint32_t> utf8_to_cpts(const std::string& s, std::vector<size_t>* byte_off = nullptr) {
    std::vector<uint32_t> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        if (byte_off) byte_off->push_back(i);
        uint8_t c = s[i];
        int len = 1; uint32_t cp = 0xFFFD;
        if (c < 0x80) { cp = c; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < n && (s[i + 1] & 0xC0) == 0x80) { cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F); len = 2; }
        else if ((c & 0xF0) == 0xE0 && i + 2 < n && (s[i + 1] & 0xC0) == 0x80 && (s[i + 2] & 0xC0) == 0x80) {
            cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F); len = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < n && (s[i + 1] & 0xC0) == 0x80 && (s[i + 2] & 0xC0) == 0x80 && (s[i + 3] & 0xC0) == 0x80) {
            cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F); len = 4;
        }
        out.push_back(cp);
        i += len;
    }
    if (byte_off) byte_off->push_back(n);
    return out;
}

uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

}  // namespace

Tokenizer::Tokenizer(const gguf::File& f) {
    if (f.get("tokenizer.ggml.model").str != "gpt2") throw std::runtime_error("expected a gpt2 (byte-level BPE) tokenizer");
    tokens_ = f.get("tokenizer.ggml.tokens").arr_str;
    for (double t : f.get("tokenizer.ggml.token_type").arr_num) types_.push_back((int)t);
    tok2id_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); i++) tok2id_.emplace(tokens_[i], (int)i);
    const auto& merges = f.get("tokenizer.ggml.merges").arr_str;
    merge_rank_.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); i++) {
        const std::string& m = merges[i];
        size_t p = m.find(' ', 1);
        if (p == std::string::npos) continue;
        merge_rank_.emplace(m.substr(0, p) + '\x1f' + m.substr(p + 1), (int)i);
    }
    if (f.has("tokenizer.ggml.bos_token_id")) bos = (int)f.u("tokenizer.ggml.bos_token_id");
    if (f.has("tokenizer.ggml.eos_token_id")) eos = (int)f.u("tokenizer.ggml.eos_token_id");
    if (f.has("tokenizer.ggml.padding_token_id")) pad = (int)f.u("tokenizer.ggml.padding_token_id");

    // GPT-2 bytes_to_unicode
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = keep ? (uint32_t)b : 256u + n++;
        byte_enc_[b] = cpt_to_utf8(cp);
        byte_dec_[cp] = (uint8_t)b;
    }
    // special tokens: CONTROL(3), USER_DEFINED(4), UNKNOWN(2); longest first (llama.cpp cache_special_tokens)
    for (size_t i = 0; i < types_.size(); i++)
        if (types_[i] == 2 || types_[i] == 3 || types_[i] == 4) specials_.push_back((int)i);
    std::stable_sort(specials_.begin(), specials_.end(), [&](int a, int b) { return tokens_[a].size() > tokens_[b].size(); });
}

// Port of llama.cpp unicode_regex_split_custom_llama3 (one fragment, no byte encoding).
void Tokenizer::split_words(const std::string& text, std::vector<std::string>& words) const {
    std::vector<size_t> boff;
    std::vector<uint32_t> cpts = utf8_to_cpts(text, &boff);
    const size_t n = cpts.size();
    std::vector<uint16_t> fl(n);
    for (size_t i = 0; i < n; i++) fl[i] = cpt_flags(cpts[i]);
    const uint32_t OOR = 0xFFFFFFFF;
    auto cpt = [&](size_t p) -> uint32_t { return p < n ? cpts[p] : OOR; };
    auto flags = [&](size_t p) -> uint16_t { return p < n ? fl[p] : 0; };
    size_t prev_end = 0;
    auto add = [&](size_t end) -> size_t {
        size_t len = end - prev_end;
        if (len > 0) {
            const size_t b0 = boff[prev_end], b1 = boff[end];
            std::string w;
            w.reserve((b1 - b0) * 2);
            for (size_t b = b0; b < b1; b++) w += byte_enc_[(uint8_t)text[b]];
            words.push_back(std::move(w));
        }
        prev_end = end;
        return len;
    };
    auto is_l = [](uint16_t f) { return (f & F_LETTER) != 0; };
    auto is_n = [](uint16_t f) { return (f & F_NUMBER) != 0; };
    auto is_w = [](uint16_t f) { return (f & F_WHITESPACE) != 0; };

    for (size_t pos = 0; pos < n;) {
        const uint32_t c = cpt(pos);
        const uint16_t f = flags(pos);
        if (c == '\'' && pos + 1 < n) {
            uint32_t c1 = ascii_lower(cpt(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { pos += add(pos + 2); continue; }
            if (pos + 2 < n) {
                uint32_t c2 = ascii_lower(cpt(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { pos += add(pos + 3); continue; }
            }
        }
        if (!(c == '\r' || c == '\n' || is_n(f))) {
            if (is_l(f) || is_l(flags(pos + 1))) {
                pos++;
                while (is_l(flags(pos))) pos++;
                add(pos);
                continue;
            }
        }
        if (is_n(f)) {
            size_t ini = pos;
            while (is_n(flags(pos))) {
                if (++pos - ini >= 3) { add(pos); ini = pos; }
            }
            add(pos);
            continue;
        }
        uint16_t f2 = (c == ' ' ? flags(pos + 1) : f);
        if (!(is_w(f2) || is_l(f2) || is_n(f2)) && f) {
            pos += (c == ' ');
            while (!(is_w(f2) || is_l(f2) || is_n(f2)) && f2) f2 = flags(++pos);
            uint32_t c2 = cpt(pos);
            while (c2 == '\r' || c2 == '\n') c2 = cpt(++pos);
            add(pos);
            continue;
        }
        size_t nws = 0, last_rn = 0;
        while (is_w(flags(pos + nws))) {
            uint32_t c2 = cpt(pos + nws);
            if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
            nws++;
        }
        if (last_rn > 0) { pos = last_rn; add(pos); continue; }
        if (nws > 1 && cpt(pos + nws) != OOR) { pos += nws - 1; add(pos); continue; }
        if (nws > 0) { pos += nws; add(pos); continue; }
        add(++pos);  // no match: single code point
    }
}

void Tokenizer::bpe_word(const std::string& w, std::vector<int>& out) const {
    auto it = tok2id_.find(w);
    if (it != tok2id_.end()) { out.push_back(it->second); return; }  // ignore_merges
    struct Sym { int prev, next; size_t off, n; };
    std::vector<Sym> sy;
    for (size_t off = 0; off < w.size();) {
        size_t len = std::min((size_t)utf8_len((uint8_t)w[off]), w.size() - off);
        int idx = (int)sy.size();
        sy.push_back({idx - 1, off + len == w.size() ? -1 : idx + 1, off, len});
        off += len;
    }
    struct Big { int left, right, rank; size_t size; };
    auto cmp = [](const Big& a, const Big& b) { return a.rank > b.rank || (a.rank == b.rank && a.left > b.left); };
    std::priority_queue<Big, std::vector<Big>, decltype(cmp)> q(cmp);
    std::string key;
    auto add_bigram = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        key.assign(w, sy[l].off, sy[l].n);
        key += '\x1f';
        key.append(w, sy[r].off, sy[r].n);
        auto m = merge_rank_.find(key);
        if (m == merge_rank_.end()) return;
        q.push({l, r, m->second, sy[l].n + sy[r].n});
    };
    for (size_t i = 1; i < sy.size(); i++) add_bigram((int)i - 1, (int)i);
    while (!q.empty()) {
        Big b = q.top(); q.pop();
        Sym& L = sy[b.left]; Sym& R = sy[b.right];
        if (L.n == 0 || R.n == 0) continue;
        if (L.n + R.n != b.size || L.next != b.right) continue;  // stale
        L.n += R.n; R.n = 0;
        L.next = R.next;
        if (R.next >= 0) sy[R.next].prev = b.left;
        add_bigram(L.prev, b.left);
        add_bigram(b.left, L.next);
    }
    for (int i = 0; i != -1; i = sy[i].next) {
        if (sy[i].n == 0) continue;
        auto t = tok2id_.find(w.substr(sy[i].off, sy[i].n));
        if (t != tok2id_.end()) { out.push_back(t->second); continue; }
        // byte fallback: each encoded char is one byte token
        for (size_t o = sy[i].off; o < sy[i].off + sy[i].n;) {
            size_t len = utf8_len((uint8_t)w[o]);
            auto bt = tok2id_.find(w.substr(o, len));
            if (bt != tok2id_.end()) out.push_back(bt->second);
            o += len;
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    // fragments: (is_token, token id or text span)
    struct Frag { int tok; size_t off, len; };
    std::list<Frag> frags{{-1, 0, text.size()}};
    for (int sid : specials_) {
        if (!parse_special && types_[sid] != 4) continue;
        const std::string& st = tokens_[sid];
        if (st.empty()) continue;
        for (auto fit = frags.begin(); fit != frags.end(); ++fit) {
            if (fit->tok >= 0) continue;
            size_t base = fit->off, end = fit->off + fit->len;
            size_t p = text.find(st, base);
            if (p == std::string::npos || p + st.size() > end) continue;
            // split this fragment at every occurrence
            auto ins = std::next(fit);
            size_t cur = base;
            while (p != std::string::npos && p + st.size() <= end) {
                if (p > cur) frags.insert(ins, {-1, cur, p - cur});
                frags.insert(ins, {sid, p, st.size()});
                cur = p + st.size();
                p = text.find(st, cur);
            }
            if (cur < end) frags.insert(ins, {-1, cur, end - cur});
            fit = frags.erase(fit);
            fit = std::prev(fit);  // loop ++ moves past the inserted pieces' start; re-scan is harmless
        }
    }
    std::vector<int> out;
    std::vector<std::string> words;
    for (const auto& fr : frags) {
        if (fr.tok >= 0) { out.push_back(fr.tok); continue; }
        words.clear();
        split_words(text.substr(fr.off, fr.len), words);
        for (const auto& w : words) bpe_word(w, out);
    }
    return out;
}

std::string Tokenizer::decode(int id, bool show_special) const {
    if (id < 0 || id >= (int)tokens_.size()) return "";
    const int ty = types_[id];
    if (ty == 3) return show_special ? tokens_[id] : "";
    if (ty == 4) return tokens_[id];
    if (ty == 5) return "";
    std::string out;
    const std::string& t = tokens_[id];
    std::vector<uint32_t> cps = utf8_to_cpts(t);
    for (uint32_t c : cps) {
        auto it = byte_dec_.find(c);
        if (it != byte_dec_.end()) out += (char)it->second;
        else out += cpt_to_utf8(c);
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool show_special) const {
    std::string s;
    for (int i : ids) s += decode(i, show_special);
    return s;
}
