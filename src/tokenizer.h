// Raon-1 nano tokenizer: byte-level BPE (GGUF tokenizer.ggml.*), NFC normalisation, the `raon` pre-tokenizer split,
// merge-rank BPE (no whole-word shortcut) and longest-first special-token matching. Ported from KRAFTON llama.cpp
// (src/llama-vocab.cpp, src/unicode.cpp) and checked token-for-token against it (tests/tokenizer_check.py).
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace raon {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& gguf_path);

    // parse_special: text matching a special token ("<|im_start|>", "<think>", ...) becomes that token.
    std::vector<int> encode(const std::string& text, bool parse_special = true) const;
    // Bytes of one token (special tokens give their text). May be a partial UTF-8 sequence.
    const std::string& piece(int id) const { return pieces_[id]; }
    std::string decode(const std::vector<int>& ids) const;

    int n_vocab() const { return (int)pieces_.size(); }
    int eos() const { return eos_; }
    int pad() const { return pad_; }
    bool is_special(int id) const { return id >= 0 && id < n_vocab() && special_[id]; }
    int find(const std::string& text) const;   // exact token text (as stored in the GGUF), -1 if absent
    const std::string& chat_template() const { return chat_template_; }

private:
    void encode_fragment(const std::string& text, std::vector<int>& out) const;
    void bpe_word(const std::vector<uint8_t>& bytes, std::vector<int>& out) const;

    std::vector<std::string> texts_;    // token text as stored (byte-encoded for normal tokens)
    std::vector<std::string> pieces_;   // decoded bytes
    std::vector<char> special_;
    std::unordered_map<std::string, int> text_to_id_;
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;   // (left id << 32 | right id) -> (rank, merged id)
    int byte_tok_[256];
    std::vector<int> specials_;         // control token ids, longest text first
    int eos_ = -1, pad_ = -1;
    bool all_lt_ = true;
    std::string chat_template_;
};

// Holds back the bytes of an incomplete UTF-8 sequence at the end of a stream of token pieces.
struct Utf8Stream {
    std::string pending;
    std::string push(const std::string& bytes);   // returns the complete prefix
    std::string flush() { std::string s; s.swap(pending); return s; }
};

}  // namespace raon
