// LFM3 chat template (port of the GGUF's jinja template) and model-output parsing for the OpenAI API.
#pragma once
#include <string>
#include <vector>

#include "json.hpp"

namespace chat {

using json = nlohmann::ordered_json;

struct TemplateOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    bool preserve_thinking = false;
};

// json.dumps(x, ensure_ascii=False) as used by the template's tojson: insertion order, ", " and ": " separators
std::string tojson(const json& v);

// Render OpenAI-style messages (+ optional tools array) exactly like the GGUF chat template.
// Tool-call arguments given as JSON strings are parsed into objects first (like llama-server).
std::string render(const json& messages, const json& tools, const TemplateOptions& opt);

struct ToolCall {
    std::string name;
    json arguments = json::object();
};

// Incremental parser of generated text: splits reasoning (before </think>), content and <tool_call> blocks.
class OutputParser {
public:
    // in_reasoning: the prompt ended with "<think>" (thinking enabled), so output starts inside the reasoning block
    OutputParser(bool in_reasoning, const json& tools) : reasoning_(in_reasoning), tools_(tools) {}
    // feed newly generated text; appends the deltas that are safe to emit
    void feed(const std::string& text, std::string& reasoning_delta, std::string& content_delta,
              std::vector<ToolCall>& calls);
    // end of generation: flush held-back text
    void finish(std::string& reasoning_delta, std::string& content_delta, std::vector<ToolCall>& calls);
    const std::string& reasoning() const { return all_reasoning_; }
    const std::string& content() const { return all_content_; }
    const std::vector<ToolCall>& tool_calls() const { return all_calls_; }

private:
    bool reasoning_;
    json tools_;
    std::string buf_;  // text not yet classified
    bool in_call_ = false;
    std::string all_reasoning_, all_content_;
    std::vector<ToolCall> all_calls_;
    void emit(std::string& rd, std::string& cd, std::vector<ToolCall>& calls, bool final);
    ToolCall parse_call(const std::string& body) const;
};

}  // namespace chat
