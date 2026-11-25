// Render conversations from a JSON file with the C++ template port (for comparison with llama-server's
// /apply-template) and run output-parser cases.  usage: test_chat <cases.json>  -> prints JSON lines
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../src/chat.h"

using chat::json;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s cases.json\n", argv[0]); return 2; }
    std::ifstream f(argv[1]);
    std::stringstream ss;
    ss << f.rdbuf();
    json cases = json::parse(ss.str());
    for (const auto& c : cases) {
        chat::TemplateOptions o;
        if (c.contains("chat_template_kwargs")) {
            const auto& k = c["chat_template_kwargs"];
            if (k.contains("enable_thinking")) o.enable_thinking = k["enable_thinking"].get<bool>();
            if (k.contains("preserve_thinking")) o.preserve_thinking = k["preserve_thinking"].get<bool>();
        }
        json out;
        try {
            out["prompt"] = chat::render(c["messages"], c.contains("tools") ? c["tools"] : json(), o);
        } catch (const std::exception& e) {
            out["error"] = e.what();
        }
        std::cout << out.dump() << "\n";
    }
    // parser self-test (streaming one character at a time must equal one-shot)
    json tools = json::parse(R"([{"type":"function","function":{"name":"bash","parameters":{"type":"object",
        "properties":{"command":{"type":"string"},"timeout":{"type":"integer"}}}}}])");
    const std::string gen = "Let me look.\nI'll run ls.</think>\n\nSure.<tool_call>\n<function=bash>\n<parameter=command>\n"
                            "ls -la 123\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>";
    std::string r1, c1, r2, c2;
    std::vector<chat::ToolCall> k1, k2;
    chat::OutputParser a(true, tools), b(true, tools);
    a.feed(gen, r1, c1, k1);
    a.finish(r1, c1, k1);
    for (char ch : gen) b.feed(std::string(1, ch), r2, c2, k2);
    b.finish(r2, c2, k2);
    const bool ok = r1 == r2 && c1 == c2 && k1.size() == 1 && k2.size() == 1 && k1[0].name == "bash" &&
                    k1[0].arguments.dump() == k2[0].arguments.dump() && r1 == "Let me look.\nI'll run ls." && c1 == "Sure." &&
                    k1[0].arguments["command"] == "ls -la 123" && k1[0].arguments["timeout"] == 30;
    fprintf(stderr, "parser: reasoning=%s content=%s call=%s %s -> %s\n", json(r1).dump().c_str(), json(c1).dump().c_str(),
            k1.empty() ? "-" : k1[0].name.c_str(), k1.empty() ? "" : k1[0].arguments.dump().c_str(), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
