// Built-in web_search / web_fetch tools, served the way llama-server serves its built-in tools (get_info, read_file, ...):
//   GET  /tools  -> [{display_name, tool, type: "server", permissions, uses_cwd, definition}]
//   POST /tools  {tool, params} -> {"plain_text_response": ...} | {"error": ...}
// so the llama.cpp web UI (raon-server) lists them next to its own tools and enables them by default.
// The tools are the pi extensions' code (tools/webtools), run as `node tools.ts call <name>` per call.
#pragma once
#include <functional>
#include <string>

#include "nlohmann/json.hpp"

class WebTools {
public:
    // dir holds tools.ts, pdf and node_modules. Loads the tool list once; on failure enabled() is false.
    WebTools(std::string dir, std::string node = "node", int timeout_s = 120);

    bool enabled() const { return enabled_; }
    const std::string& error() const { return error_; }
    const nlohmann::ordered_json& list() const { return list_; }
    bool has(const std::string& name) const;

    // Runs one tool call. should_stop (may be null) is polled; once it returns true the child is killed.
    // Returns the tool's JSON reply, or {"error": ...} on spawn failure / timeout / bad output.
    nlohmann::ordered_json call(const std::string& name, const nlohmann::ordered_json& params,
                                const std::function<bool()>& should_stop = nullptr) const;

private:
    struct RunResult { int status = -1; bool timed_out = false, stopped = false; std::string out, err; };
    RunResult run(const std::vector<std::string>& args, const std::string& input, int timeout_s,
                  const std::function<bool()>& should_stop) const;

    std::string dir_, node_;
    int timeout_s_;
    bool enabled_ = false;
    std::string error_;
    nlohmann::ordered_json list_ = nlohmann::ordered_json::array();
};
