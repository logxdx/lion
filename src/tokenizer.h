// Byte-level BPE tokenizer for LFM2/LFM3 GGUF vocabularies (tokenizer.ggml.pre = "lfm2"):
// Llama-3 pre-tokenizer regex, ignore_merges, special tokens parsed from text.
// Behaviour mirrors llama.cpp's llm_tokenizer_bpe so token ids match exactly.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gguf { class File; }

class Tokenizer {
public:
    explicit Tokenizer(const gguf::File& f);

    // parse_special: match control/user-defined token text (e.g. "<|im_start|>") as single tokens
    std::vector<int> encode(const std::string& text, bool parse_special = true) const;
    // bytes of one token; control tokens render as "" unless show_special
    std::string decode(int id, bool show_special = false) const;
    std::string decode(const std::vector<int>& ids, bool show_special = false) const;

    int n_vocab() const { return (int)tokens_.size(); }
    int bos = -1, eos = -1, pad = -1;
    int id(const std::string& tok) const { auto it = tok2id_.find(tok); return it == tok2id_.end() ? -1 : it->second; }
    bool is_control(int id) const { return id >= 0 && id < (int)types_.size() && types_[id] == 3; }

private:
    std::vector<std::string> tokens_;
    std::vector<int> types_;
    std::unordered_map<std::string, int> tok2id_;
    std::unordered_map<std::string, int> merge_rank_;  // key: left + '\x1f' + right (encoded strings never contain 0x1f)
    std::vector<int> specials_;                        // control/user-defined ids, longest text first
    std::string byte_enc_[256];                        // GPT-2 byte -> utf8 of mapped codepoint
    std::unordered_map<uint32_t, uint8_t> byte_dec_;   // mapped codepoint -> byte

    void bpe_word(const std::string& word_enc, std::vector<int>& out) const;
    void split_words(const std::string& text, std::vector<std::string>& words) const;
};
