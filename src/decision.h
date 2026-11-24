// Decision models (LiquidAI d1-2B, GGUF lfm3.decision.type = "lfm3-d1"): llama.cpp's /v1/systemone API.
//
// A request asks typed questions about a state (text or JSON) and optional images / videos:
//   {"state": ..., "files": [data URLs], "questions": {id: {"type": "noul"|"choice"|"score", "instructions", "criteria"}}}
// Each question is answered from the logits of its label tokens at the end of its prompt (yes/no, option codes,
// score digits), softmax at the model's calibrated temperature. The prompt is the model's systemone template; as
// llama.cpp b12674 does, the questions share one forward pass: the common prefix (media, state, "QUESTION:")
// once, then each question's text, and in the global attention layers a question does not see the other questions.
// Request parsing, escaping, rendering, labels and answers follow llama.cpp (tools/server/server-decision.cpp).
#pragma once
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

class Tokenizer;
namespace gguf { class File; }

namespace d1 {

using ojson = nlohmann::ordered_json;  // request key order matters (options, legend, answers)

enum class QType { Choice, Score, Noul };

struct Option {
    std::string key;
    ojson description;
};

struct Question {
    std::string id;
    QType type = QType::Noul;
    ojson instructions;
    std::vector<Option> options;            // choice: criteria order; score: levels 0..n-1; noul: [true, false]
    std::vector<std::string> labels;        // per option: the code shown in the prompt (choice) / the key
    std::vector<std::vector<int>> groups;   // per option: its label token ids (the logit is their max)
};

constexpr int MAX_PACK = 16;      // questions per forward pass (llama.cpp's limit for packed questions)
constexpr int MAX_MEDIA = 8;      // images + videos per request

class Context {
public:
    Context(const Tokenizer& tk, const gguf::File& f);

    // request -> questions (labels resolved); std::invalid_argument with llama.cpp's messages
    std::vector<Question> parse_questions(const ojson& body) const;
    // media data URLs (files, images, then image_url / video_url parts of a chat-message state) and the state
    // without them
    ojson parse_state(const ojson& body, std::vector<std::string>& media) const;

    // prompt pieces: the shared prefix (media markers given as text) and one question's own text
    std::string prefix(const ojson& state, const std::string& media_marks) const;
    std::string question_text(const Question& q) const;
    // before each packed question after the first: "QUESTION:\n" with a state, nothing without (as b12674)
    static const char* separator(bool has_state) { return has_state ? "QUESTION:\n" : ""; }

    // the answer object from the options' logits
    ojson answer(const Question& q, const std::vector<float>& logits, bool has_media) const;

    std::string d1_break(const std::string& text) const;  // control-token spellings in user text broken by U+200B

private:
    const Tokenizer& tk_;
    std::vector<std::string> spellings_;  // control tokens (>= 2 bytes), longest first
    std::string first_chars_;
    std::map<std::string, float> temps_;  // lfm3.decision.temperature.<name>
    void labels(Question& q) const;
    float temperature(const Question& q, bool has_media) const;
    ojson escape(const ojson& v, int depth = 0) const;
};

// Python json.dumps(v, ensure_ascii=False, indent, separators) — what the template's tojson prints
std::string tojson(const ojson& v, int indent, const std::string& item_sep, const std::string& key_sep);

}  // namespace d1
