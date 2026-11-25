// Raon-1 nano chat format: the GGUF chat template ported to C++ (checked byte-for-byte against KRAFTON llama-server
// /apply-template by tests/template_check.py), and a streaming parser for what the model writes back
// (reasoning in <think>, content, Qwen3-coder XML tool calls).
#pragma once
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace raon::chat {

using json = nlohmann::ordered_json;

struct TemplateOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    bool truncate_history_thinking = false;
};

// Renders an OpenAI-style message list (+ optional tools). Throws std::invalid_argument on input the template rejects
// (non-text content, unknown roles, no messages).
std::string render(const json& messages, const json& tools, const TemplateOptions& opt);

// Python-style json.dumps (ensure_ascii=False, separators ", " and ": ", key order kept), as the template's tojson.
std::string tojson(const json& v);

struct ToolCall {
    std::string name;
    json arguments = json::object();
};

// Incremental parser of the generated text. Feed decoded text as it arrives; it returns what can be shown so far.
// Text that may be the beginning of a tag ("</thi") is held back until it is resolved.
class OutputParser {
public:
    // in_reasoning: the prompt ended inside <think> (thinking enabled). tools: the request's tools (empty = do not
    // parse tool calls).
    OutputParser(bool in_reasoning, const json& tools);

    struct Delta {
        std::string reasoning, content;
        std::vector<ToolCall> calls;   // completed tool calls
        bool empty() const { return reasoning.empty() && content.empty() && calls.empty(); }
    };
    Delta push(const std::string& text);
    Delta finish();   // end of generation: flush held-back text

    const std::string& reasoning() const { return all_reasoning_; }
    const std::string& content() const { return all_content_; }
    const std::vector<ToolCall>& calls() const { return all_calls_; }

private:
    enum Mode { REASONING, CONTENT, TOOL };
    void step(Delta& d, bool final);
    ToolCall parse_call(const std::string& body) const;
    Mode mode_;
    bool parse_tools_;
    json tools_;
    std::string buf_;
    bool at_start_ = true;        // strip leading newlines of the current section
    std::string ws_hold_;         // trailing whitespace held back in the current section
    std::string all_reasoning_, all_content_;
    std::vector<ToolCall> all_calls_;
};

}  // namespace raon::chat
