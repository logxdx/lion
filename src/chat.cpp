#include "chat.h"

#include <cstring>
#include <stdexcept>

namespace chat {

// ---------------------------------------------------------------------------------------------- tojson
static void dump_into(const json& v, std::string& out) {
    switch (v.type()) {
        case json::value_t::object: {
            out += '{';
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) out += ", ";
                first = false;
                out += json(it.key()).dump();
                out += ": ";
                dump_into(it.value(), out);
            }
            out += '}';
            break;
        }
        case json::value_t::array: {
            out += '[';
            for (size_t i = 0; i < v.size(); i++) {
                if (i) out += ", ";
                dump_into(v[i], out);
            }
            out += ']';
            break;
        }
        default:
            out += v.dump(-1, ' ', false, json::error_handler_t::replace);
    }
}

std::string tojson(const json& v) {
    std::string s;
    dump_into(v, s);
    return s;
}

// ---------------------------------------------------------------------------------------------- helpers
static bool truthy(const json& v) {
    if (v.is_null()) return false;
    if (v.is_string()) return !v.get_ref<const std::string&>().empty();
    if (v.is_array() || v.is_object()) return !v.empty();
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number_integer()) return v.get<long long>() != 0;
    if (v.is_number_float()) return v.get<double>() != 0.0;
    return true;
}
static const json& getk(const json& o, const char* k) {
    static const json null_v;
    if (!o.is_object()) return null_v;
    auto it = o.find(k);
    return it == o.end() ? null_v : *it;
}
static bool has(const json& o, const char* k) { return o.is_object() && o.contains(k); }
static std::string trim(const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

static std::string render_value(const json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_null()) return "null";
    if (v.is_object() || v.is_array()) return tojson(v);
    return v.dump();
}

static std::string item_content(const json& item) {
    if (item.is_string()) return item.get<std::string>();
    if (item.is_object()) {
        const json& ty = getk(item, "type");
        if (ty == "image") return "<|vision_start|><|image|><|vision_end|>";
        if (ty == "video") return "<|vision_start|><|video|><|vision_end|>";
        if (ty == "text") {
            const json& t = getk(item, "text");
            return truthy(t) ? (t.is_string() ? t.get<std::string>() : tojson(t)) : "";
        }
    }
    return tojson(item);
}

static std::string parse_content(const json& c) {
    if (c.is_string()) return c.get<std::string>();
    if (c.is_object()) return item_content(c);
    if (c.is_array()) {
        std::string r;
        for (const auto& it : c) r += item_content(it);
        return r;
    }
    if (!c.is_null()) return tojson(c);
    return "";
}

static json parse_args(const json& a) {
    if (a.is_string()) {
        const std::string t = trim(a.get<std::string>());
        if (t.empty() || t == "{}" || t == "null") return json::object();
        try {
            return json::parse(t);
        } catch (...) {
            throw std::runtime_error("tool call arguments are not valid JSON");
        }
    }
    return a;
}

static std::string render_tool_calls(const json& calls) {
    std::string s;
    bool first = true;
    for (const auto& tc : calls) {
        if (!first) s += "\n\n";
        first = false;
        const json& fn = has(tc, "function") ? tc["function"] : tc;
        const json args = parse_args(getk(fn, "arguments"));
        s += "<tool_call>\n<function=" + getk(fn, "name").get<std::string>() + ">\n";
        if (args.is_object())
            for (auto it = args.begin(); it != args.end(); ++it)
                s += "<parameter=" + it.key() + ">\n" + render_value(it.value()) + "\n</parameter>\n";
        s += "</function>\n</tool_call>";
    }
    return s;
}

static std::string tool_to_json(const json& tool) {
    const json& fn = has(tool, "function") ? tool["function"] : tool;
    std::string s = "{";
    bool first = true;
    for (auto it = fn.begin(); it != fn.end(); ++it) {
        if (it.key() == "defer_loading") continue;
        if (!first) s += ", ";
        first = false;
        s += json(it.key()).dump() + ": " + tojson(it.value());
    }
    return s + "}";
}

static bool is_tool_reference(const json& item) { return item.is_object() && getk(item, "type") == "tool_reference"; }

static std::string referenced_tool_to_json(const json& tools, const json& name) {
    if (tools.is_array())
        for (const auto& t : tools) {
            const json& fn = has(t, "function") ? t["function"] : t;
            if (getk(fn, "name") == name) return tool_to_json(fn);
        }
    throw std::runtime_error("tool_reference to an unknown tool");
}

static std::string render_tool_response(const json& content, const json& tools) {
    if (content.is_string() && content.get<std::string>().find("<tool_response>") != std::string::npos)
        return content.get<std::string>();
    std::string s = "<tool_response>\n";
    if (content.is_string()) s += content.get<std::string>();
    else if (content.is_object()) s += parse_content(content);
    else if (content.is_array()) {
        for (size_t i = 0; i < content.size(); i++) {
            const json& item = content[i];
            if (i) s += "\n";
            if (is_tool_reference(item)) {
                if (i == 0 || !is_tool_reference(content[i - 1])) s += "<additional_tools>\n";
                s += referenced_tool_to_json(tools, getk(item, "name"));
                if (i + 1 == content.size() || !is_tool_reference(content[i + 1])) s += "\n</additional_tools>";
            } else if (item.is_object() && getk(item, "result").is_object() && has(item["result"], "tools")) {
                s += "<additional_tools>\n";
                const json& tl = item["result"]["tools"];
                if (tl.is_string()) s += tl.get<std::string>();
                else if (tl.is_object()) s += tool_to_json(tl);
                else
                    for (size_t j = 0; j < tl.size(); j++) s += (j ? "\n" : "") + tool_to_json(tl[j]);
                s += "\n</additional_tools>";
            } else {
                s += item_content(item);
            }
        }
    } else if (!content.is_null()) {
        s += tojson(content);
    }
    return s + "\n</tool_response>";
}

// ---------------------------------------------------------------------------------------------- render
std::string render(const json& messages_in, const json& tools, const TemplateOptions& opt) {
    std::string out = "<|startoftext|>";
    std::string system_prompt;
    json messages = messages_in.is_array() ? messages_in : json::array();
    if (!messages.empty() && getk(messages[0], "role") == "system") {
        if (truthy(getk(messages[0], "content"))) system_prompt = parse_content(messages[0]["content"]);
        messages.erase(messages.begin());
    }
    if (tools.is_array() && !tools.empty()) {
        std::vector<std::string> lines;
        for (const auto& t : tools) {
            const json& fn = has(t, "function") ? t["function"] : t;
            const bool dyn = (fn.is_object() && truthy(getk(fn, "defer_loading"))) || truthy(getk(t, "defer_loading"));
            if (!dyn) lines.push_back(tool_to_json(fn));
        }
        if (!lines.empty()) {
            std::string joined;
            for (size_t i = 0; i < lines.size(); i++) joined += (i ? "\n" : "") + lines[i];
            system_prompt += (system_prompt.empty() ? "" : "\n\n") + std::string("<tools>\n") + joined + "\n</tools>";
        }
    }
    if (!system_prompt.empty()) out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    int last_user = -1;
    for (size_t i = 0; i < messages.size(); i++)
        if (getk(messages[i], "role") == "user") last_user = (int)i;

    json carry = json::array();
    for (size_t i = 0; i < messages.size(); i++) {
        const json& m = messages[i];
        const std::string role = getk(m, "role").is_string() ? m["role"].get<std::string>() : "";
        const json* next = i + 1 < messages.size() ? &messages[i + 1] : nullptr;
        const bool merge = role == "tool" && !truthy(getk(m, "additional_tools")) && next &&
                           getk(*next, "role") == "tool" && !truthy(getk(*next, "additional_tools")) &&
                           truthy(getk(m, "tool_call_id")) && getk(*next, "tool_call_id") == getk(m, "tool_call_id");
        json items = json::array();
        if (role == "tool" && !truthy(getk(m, "additional_tools"))) {
            const json& c = getk(m, "content");
            if (c.is_string()) { if (!c.get<std::string>().empty()) items.push_back({{"type", "text"}, {"text", c}}); }
            else if (c.is_object()) items.push_back(c);
            else if (c.is_array()) for (const auto& x : c) items.push_back(x);
            else if (!c.is_null()) items.push_back(c);
        }
        if (merge) {
            for (auto& x : items) carry.push_back(x);
            continue;
        }
        out += "<|im_start|>" + role + "\n";
        if (role == "assistant") {
            const json* thinking = nullptr;
            if (has(m, "thinking")) thinking = &m["thinking"];
            else if (has(m, "reasoning")) thinking = &m["reasoning"];
            else if (has(m, "reasoning_content")) thinking = &m["reasoning_content"];
            const bool keep = opt.preserve_thinking || (int)i > last_user;
            const json& raw = getk(m, "content");
            const std::string raw_str = raw.is_string() ? raw.get<std::string>() : "";
            if (thinking && keep) out += "<think>" + (truthy(*thinking) ? parse_content(*thinking) : "") + "</think>";
            else if (keep && raw_str.find("</think>") == std::string::npos) out += "<think></think>";
            std::string content = truthy(raw) ? parse_content(raw) : "";
            if (!keep) {
                const size_t p = content.rfind("</think>");
                if (p != std::string::npos) content = trim(content.substr(p + 8));
                content = "<think></think>" + content;
            }
            out += content;
            if (truthy(getk(m, "tool_calls"))) out += render_tool_calls(m["tool_calls"]);
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            if (truthy(getk(m, "additional_tools"))) {
                const json& at = m["additional_tools"];
                const json& src = (at.is_object() || at.is_array()) ? at : getk(m, "content");
                if (src.is_string() && src.get<std::string>().find("<tool_response>") != std::string::npos) out += src.get<std::string>();
                else if (src.is_string() && src.get<std::string>().find("<additional_tools>") != std::string::npos)
                    out += "<tool_response>\n" + src.get<std::string>() + "\n</tool_response>";
                else if (truthy(src)) {
                    out += "<tool_response>\n<additional_tools>\n";
                    if (src.is_string()) out += src.get<std::string>();
                    else if (src.is_object()) out += tool_to_json(src);
                    else for (size_t j = 0; j < src.size(); j++) out += (j ? "\n" : "") + tool_to_json(src[j]);
                    out += "\n</additional_tools>\n</tool_response>";
                }
            } else if (!carry.empty()) {
                for (auto& x : items) carry.push_back(x);
                out += render_tool_response(carry, tools);
                carry = json::array();
            } else {
                const json& c = getk(m, "content");
                out += render_tool_response(c.is_null() ? json("") : c, tools);
            }
            out += "<|im_end|>\n";
        } else {
            if (truthy(getk(m, "content"))) out += parse_content(m["content"]);
            out += "<|im_end|>\n";
        }
    }
    if (opt.add_generation_prompt) out += opt.enable_thinking ? "<|im_start|>assistant\n<think>" : "<|im_start|>assistant\n<think></think>";
    return out;
}

// ---------------------------------------------------------------------------------------------- output parser
static size_t partial_suffix(const std::string& buf, const char* tag) {
    // length of the longest suffix of buf that is a proper prefix of tag
    const size_t tl = strlen(tag);
    for (size_t k = std::min(buf.size(), tl - 1); k > 0; k--)
        if (buf.compare(buf.size() - k, k, tag, k) == 0) return k;
    return 0;
}

ToolCall OutputParser::parse_call(const std::string& body) const {
    ToolCall c;
    const std::string t = trim(body);
    if (!t.empty() && t[0] == '{') {  // JSON-style call
        try {
            json j = json::parse(t);
            c.name = getk(j, "name").is_string() ? j["name"].get<std::string>() : "";
            json a = has(j, "arguments") ? j["arguments"] : getk(j, "parameters");
            c.arguments = a.is_string() ? json::parse(a.get<std::string>()) : (a.is_object() ? a : json::object());
            return c;
        } catch (...) {}
    }
    size_t p = t.find("<function=");
    if (p == std::string::npos) return c;
    size_t q = t.find('>', p);
    if (q == std::string::npos) return c;
    c.name = trim(t.substr(p + 10, q - p - 10));
    json props;
    if (tools_.is_array())
        for (const auto& tl : tools_) {
            const json& fn = has(tl, "function") ? tl["function"] : tl;
            if (getk(fn, "name") == c.name) props = getk(getk(fn, "parameters"), "properties");
        }
    size_t pos = q + 1;
    while ((p = t.find("<parameter=", pos)) != std::string::npos) {
        q = t.find('>', p);
        if (q == std::string::npos) break;
        const std::string key = trim(t.substr(p + 11, q - p - 11));
        size_t e = t.find("</parameter>", q);
        if (e == std::string::npos) e = t.find("</function>", q);
        if (e == std::string::npos) e = t.size();
        std::string val = t.substr(q + 1, e - q - 1);
        if (!val.empty() && val.front() == '\n') val.erase(0, 1);
        if (!val.empty() && val.back() == '\n') val.pop_back();
        const json& ty = getk(getk(props, key.c_str()), "type");
        if (ty == "string") c.arguments[key] = val;
        else {
            try {
                c.arguments[key] = json::parse(val);
            } catch (...) {
                c.arguments[key] = val;
            }
        }
        pos = e + 1;
    }
    return c;
}

void OutputParser::emit(std::string& rd, std::string& cd, std::vector<ToolCall>& calls, bool final) {
    for (;;) {
        if (reasoning_) {
            const size_t p = buf_.find("</think>");
            if (p != std::string::npos) {
                rd += buf_.substr(0, p);
                buf_.erase(0, p + 8);
                reasoning_ = false;
                continue;
            }
            const size_t keep = final ? 0 : partial_suffix(buf_, "</think>");
            rd += buf_.substr(0, buf_.size() - keep);
            buf_.erase(0, buf_.size() - keep);
            return;
        }
        if (in_call_) {
            const size_t p = buf_.find("</tool_call>");
            if (p != std::string::npos) {
                ToolCall c = parse_call(buf_.substr(0, p));
                buf_.erase(0, p + 12);
                in_call_ = false;
                if (!c.name.empty()) calls.push_back(c);
                continue;
            }
            if (final) {  // unterminated call: try to salvage it, else return the text
                ToolCall c = parse_call(buf_);
                if (!c.name.empty()) calls.push_back(c);
                else cd += "<tool_call>" + buf_;
                buf_.clear();
                in_call_ = false;
            }
            return;
        }
        const size_t p = buf_.find("<tool_call>");
        if (p != std::string::npos) {
            cd += buf_.substr(0, p);
            buf_.erase(0, p + 11);
            in_call_ = true;
            continue;
        }
        const size_t keep = final ? 0 : partial_suffix(buf_, "<tool_call>");
        cd += buf_.substr(0, buf_.size() - keep);
        buf_.erase(0, buf_.size() - keep);
        return;
    }
}

void OutputParser::feed(const std::string& text, std::string& rd, std::string& cd, std::vector<ToolCall>& calls) {
    buf_ += text;
    const size_t r0 = rd.size(), c0 = cd.size(), k0 = calls.size();
    emit(rd, cd, calls, false);
    // content: drop the whitespace the model puts right after </think> (before any visible content)
    if (all_content_.empty() && cd.size() > c0) {
        const size_t a = cd.find_first_not_of(" \t\r\n", c0);
        cd.erase(c0, (a == std::string::npos ? cd.size() : a) - c0);
    }
    all_reasoning_ += rd.substr(r0);
    all_content_ += cd.substr(c0);
    all_calls_.insert(all_calls_.end(), calls.begin() + k0, calls.end());
}

void OutputParser::finish(std::string& rd, std::string& cd, std::vector<ToolCall>& calls) {
    const size_t r0 = rd.size(), c0 = cd.size(), k0 = calls.size();
    emit(rd, cd, calls, true);
    all_reasoning_ += rd.substr(r0);
    all_content_ += cd.substr(c0);
    all_calls_.insert(all_calls_.end(), calls.begin() + k0, calls.end());
}

}  // namespace chat
