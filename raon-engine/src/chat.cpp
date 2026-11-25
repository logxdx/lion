#include "chat.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace raon::chat {
namespace {

// jinja |trim and str.strip(): ASCII isspace
std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && isspace((unsigned char)s[b])) b++;
    while (e > b && isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
}
std::string strip_chars(const std::string& s, const char* chars, bool left, bool right) {
    size_t b = 0, e = s.size();
    const std::string c(chars);
    if (left) while (b < e && c.find(s[b]) != std::string::npos) b++;
    if (right) while (e > b && c.find(s[e - 1]) != std::string::npos) e--;
    return s.substr(b, e - b);
}
bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
bool ends_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Python float repr: shortest round-trip digits; scientific when exponent < -4 or >= 16; ".0" on integral values.
std::string float_repr(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v < 0 ? "-Infinity" : "Infinity";
    char buf[64];
    int prec = 1;
    for (; prec <= 17; prec++) {
        snprintf(buf, sizeof buf, "%.*e", prec - 1, v);
        if (strtod(buf, nullptr) == v) break;
    }
    // buf = d.ddddde[+-]XX ; extract digits and exponent
    std::string s(buf);
    const size_t epos = s.find('e');
    const int exp = atoi(s.c_str() + epos + 1);
    std::string mant = s.substr(0, epos);
    const bool neg = mant[0] == '-';
    if (neg) mant.erase(0, 1);
    std::string digits;
    for (char c : mant) if (c != '.') digits += c;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    std::string out;
    if (exp < -4 || exp >= 16) {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    } else if (exp >= 0) {
        if ((int)digits.size() <= exp + 1) out = digits + std::string(exp + 1 - digits.size(), '0') + ".0";
        else out = digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
    } else {
        out = "0." + std::string(-exp - 1, '0') + digits;
    }
    return neg ? "-" + out : out;
}

void dump(const json& v, std::string& o) {
    switch (v.type()) {
        case json::value_t::null: o += "null"; break;
        case json::value_t::boolean: o += v.get<bool>() ? "true" : "false"; break;
        case json::value_t::number_integer: o += std::to_string(v.get<int64_t>()); break;
        case json::value_t::number_unsigned: o += std::to_string(v.get<uint64_t>()); break;
        case json::value_t::number_float: o += float_repr(v.get<double>()); break;
        case json::value_t::string: {
            o += '"';
            for (char c : v.get_ref<const std::string&>()) {
                switch (c) {
                    case '"': o += "\\\""; break;
                    case '\\': o += "\\\\"; break;
                    case '\b': o += "\\b"; break;
                    case '\f': o += "\\f"; break;
                    case '\n': o += "\\n"; break;
                    case '\r': o += "\\r"; break;
                    case '\t': o += "\\t"; break;
                    default:
                        if ((unsigned char)c < 0x20) {
                            char b[8];
                            snprintf(b, sizeof b, "\\u%04x", (unsigned char)c);
                            o += b;
                        } else {
                            o += c;
                        }
                }
            }
            o += '"';
            break;
        }
        case json::value_t::array: {
            o += '[';
            bool first = true;
            for (const auto& x : v) { if (!first) o += ", "; first = false; dump(x, o); }
            o += ']';
            break;
        }
        case json::value_t::object: {
            o += '{';
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) o += ", ";
                first = false;
                dump(json(it.key()), o);
                o += ": ";
                dump(it.value(), o);
            }
            o += '}';
            break;
        }
        default: o += "null";
    }
}

// render_content macro
std::string content_text(const json& msg) {
    if (!msg.is_object() || !msg.contains("content")) return "";
    const json& c = msg["content"];
    if (c.is_string()) return c.get<std::string>();
    if (c.is_null()) return "";
    if (c.is_array()) {
        std::string s;
        for (const auto& item : c) {
            if (item.is_object() && item.contains("text")) {
                const auto& t = item["text"];
                s += t.is_string() ? t.get<std::string>() : tojson(t);
            } else {
                throw std::invalid_argument("raon-1-nano supports text content only.");
            }
        }
        return s;
    }
    throw std::invalid_argument("Unexpected content type.");
}

std::string role_of(const json& m) {
    if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) throw std::invalid_argument("message without a role");
    return m["role"].get<std::string>();
}

// tool_call.arguments may arrive as a JSON string (OpenAI) or an object; the template iterates it as a mapping.
json arguments_of(const json& fn) {
    if (!fn.contains("arguments")) return json();
    const json& a = fn["arguments"];
    if (a.is_string()) {
        json p = json::parse(a.get<std::string>(), nullptr, false);
        if (p.is_discarded()) throw std::invalid_argument("tool_call arguments are not valid JSON");
        return p;
    }
    return a;
}

std::string assistant_body(const json& msg, const std::string& content, const std::string& reasoning, bool preserve) {
    std::string o;
    if (preserve && !trim(reasoning).empty()) o += "<think>\n" + trim(reasoning) + "\n</think>\n\n";
    else o += "<think></think>\n\n";
    o += content;
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
        bool first = true;
        for (const auto& tc0 : msg["tool_calls"]) {
            const json& tc = tc0.contains("function") ? tc0["function"] : tc0;
            const std::string name = tc.value("name", "");
            if (first) o += !trim(content).empty() ? "\n\n<tool_call>\n<function=" + name + ">\n" : "<tool_call>\n<function=" + name + ">\n";
            else o += "\n<tool_call>\n<function=" + name + ">\n";
            first = false;
            const json args = arguments_of(tc);
            if (args.is_object()) {
                for (auto it = args.begin(); it != args.end(); ++it) {
                    o += "<parameter=" + it.key() + ">\n";
                    o += it.value().is_string() ? it.value().get<std::string>() : tojson(it.value());
                    o += "\n</parameter>\n";
                }
            }
            o += "</function>\n</tool_call>";
        }
    }
    o += "<|im_end|>";
    return o;
}

const char* k_tools_rules =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

}  // namespace

std::string tojson(const json& v) {
    std::string o;
    dump(v, o);
    return o;
}

std::string render(const json& messages, const json& tools, const TemplateOptions& opt) {
    if (!messages.is_array() || messages.empty()) throw std::invalid_argument("No messages provided.");
    std::string o;
    const bool has_tools = tools.is_array() && !tools.empty();
    if (has_tools) {
        o += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& t : tools) o += "\n" + tojson(t);
        o += "\n</tools>";
        o += k_tools_rules;
        if (role_of(messages[0]) == "system") {
            const std::string c = trim(content_text(messages[0]));
            if (!c.empty()) o += "\n\n" + c;
        }
        o += "<|im_end|>\n";
    } else if (role_of(messages[0]) == "system") {
        o += "<|im_start|>system\n" + trim(content_text(messages[0])) + "<|im_end|>\n";
    }
    // last real user query (for truncate_history_thinking)
    const int n = (int)messages.size();
    int last_query = n - 1;
    for (int i = n - 1; i >= 0; i--) {
        if (role_of(messages[i]) != "user") continue;
        const std::string c = trim(content_text(messages[i]));
        if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) { last_query = i; break; }
    }
    for (int i = 0; i < n; i++) {
        const json& m = messages[i];
        const std::string role = role_of(m);
        std::string content = trim(content_text(m));
        if (role == "system") {
            if (i != 0) o += "<|im_start|>system\n" + content + "<|im_end|>\n";
        } else if (role == "user") {
            o += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string reasoning;
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                reasoning = m["reasoning_content"].get<std::string>();
            } else if (starts_with(content, "<think>") && content.find("</think>") != std::string::npos) {
                const size_t k = content.find("</think>");
                reasoning = strip_chars(content.substr(7, k - 7), "\n", true, true);
                content = strip_chars(content.substr(k + 8), "\n", true, false);
            }
            o += "<|im_start|>assistant\n";
            const bool preserve = !opt.truncate_history_thinking || i > last_query;
            o += assistant_body(m, content, reasoning, preserve);
            o += "\n";
        } else if (role == "tool") {
            if (i > 0 && role_of(messages[i - 1]) != "tool") o += "<|im_start|>user";
            o += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i == n - 1 || role_of(messages[i + 1]) != "tool") o += "<|im_end|>\n";
        } else {
            throw std::invalid_argument("Unexpected message role.");
        }
    }
    if (opt.add_generation_prompt) {
        o += "<|im_start|>assistant\n";
        o += opt.enable_thinking ? "<think>\n" : "<think></think>\n\n";
    }
    return o;
}

// ---------------------------------------------------------------- output parser

static const std::string k_end_think = "</think>";
static const std::string k_call_open = "<tool_call>";
static const std::string k_call_close = "</tool_call>";

OutputParser::OutputParser(bool in_reasoning, const json& tools)
    : mode_(in_reasoning ? REASONING : CONTENT), parse_tools_(tools.is_array() && !tools.empty()), tools_(tools) {}

// Length of the longest suffix of s that is a proper prefix of tag.
static size_t partial_tag(const std::string& s, const std::string& tag) {
    for (size_t k = std::min(s.size(), tag.size() - 1); k > 0; k--)
        if (s.compare(s.size() - k, k, tag, 0, k) == 0) return k;
    return 0;
}

// Emits section text: strips leading newlines at the start of a section (and all leading whitespace in content),
// and holds back trailing whitespace until more text follows (sections are trimmed, as the template renders them).
static void emit(std::string text, bool& at_start, std::string& ws_hold, std::string& out, std::string& all, bool content) {
    if (at_start) {
        size_t b = 0;
        while (b < text.size() && (content ? isspace((unsigned char)text[b]) : text[b] == '\n')) b++;
        text.erase(0, b);
        if (text.empty()) return;
        at_start = false;
    }
    size_t e = text.size();
    while (e > 0 && isspace((unsigned char)text[e - 1])) e--;
    if (e == 0) { ws_hold += text; return; }
    std::string s = ws_hold + text.substr(0, e);
    ws_hold = text.substr(e);
    out += s;
    all += s;
}

void OutputParser::step(Delta& d, bool final) {
    while (true) {
        if (mode_ == REASONING) {
            const size_t k = buf_.find(k_end_think);
            if (k != std::string::npos) {
                emit(buf_.substr(0, k), at_start_, ws_hold_, d.reasoning, all_reasoning_, false);
                buf_.erase(0, k + k_end_think.size());
                mode_ = CONTENT;
                at_start_ = true;
                ws_hold_.clear();
                continue;
            }
            const size_t keep = final ? 0 : partial_tag(buf_, k_end_think);
            emit(buf_.substr(0, buf_.size() - keep), at_start_, ws_hold_, d.reasoning, all_reasoning_, false);
            buf_.erase(0, buf_.size() - keep);
            return;
        }
        if (mode_ == CONTENT) {
            const size_t k = parse_tools_ ? buf_.find(k_call_open) : std::string::npos;
            if (k != std::string::npos) {
                emit(buf_.substr(0, k), at_start_, ws_hold_, d.content, all_content_, true);
                ws_hold_.clear();
                buf_.erase(0, k + k_call_open.size());
                mode_ = TOOL;
                continue;
            }
            const size_t keep = (final || !parse_tools_) ? 0 : partial_tag(buf_, k_call_open);
            emit(buf_.substr(0, buf_.size() - keep), at_start_, ws_hold_, d.content, all_content_, true);
            buf_.erase(0, buf_.size() - keep);
            return;
        }
        // TOOL
        const size_t k = buf_.find(k_call_close);
        if (k == std::string::npos) {
            if (final) {   // unterminated call: try to parse what is there
                if (buf_.find("<function=") != std::string::npos) {
                    ToolCall c = parse_call(buf_);
                    d.calls.push_back(c);
                    all_calls_.push_back(c);
                }
                buf_.clear();
            }
            return;
        }
        ToolCall c = parse_call(buf_.substr(0, k));
        d.calls.push_back(c);
        all_calls_.push_back(c);
        buf_.erase(0, k + k_call_close.size());
        mode_ = CONTENT;   // further text: more calls or (unexpected) content
    }
}

ToolCall OutputParser::parse_call(const std::string& body) const {
    ToolCall c;
    size_t p = body.find("<function=");
    if (p == std::string::npos) return c;
    size_t q = body.find('>', p);
    if (q == std::string::npos) return c;
    c.name = trim(body.substr(p + 10, q - p - 10));
    // parameter types from the tool schema
    json props = json::object();
    for (const auto& t : tools_) {
        const json& fn = t.contains("function") ? t["function"] : t;
        if (fn.value("name", "") == c.name && fn.contains("parameters") && fn["parameters"].contains("properties"))
            props = fn["parameters"]["properties"];
    }
    size_t at = q + 1;
    while (true) {
        const size_t a = body.find("<parameter=", at);
        if (a == std::string::npos) break;
        const size_t b = body.find('>', a);
        if (b == std::string::npos) break;
        const std::string key = trim(body.substr(a + 11, b - a - 11));
        size_t e = body.find("</parameter>", b);
        const size_t next = body.find("<parameter=", b);
        if (e == std::string::npos || (next != std::string::npos && next < e)) e = next != std::string::npos ? next : body.find("</function>", b);
        if (e == std::string::npos) e = body.size();
        std::string v = body.substr(b + 1, e - b - 1);
        if (!v.empty() && v[0] == '\n') v.erase(0, 1);
        if (!v.empty() && v.back() == '\n') v.pop_back();
        std::string type;
        if (props.contains(key) && props[key].is_object() && props[key].contains("type") && props[key]["type"].is_string())
            type = props[key]["type"].get<std::string>();
        json val = v;
        if (type != "string") {   // numbers, booleans, objects, arrays (or no schema): JSON if it parses
            json pv = json::parse(v, nullptr, false);
            if (!pv.is_discarded() && (type.empty() ? !pv.is_string() : true)) val = pv;
        }
        c.arguments[key] = val;
        at = e;
    }
    return c;
}

OutputParser::Delta OutputParser::push(const std::string& text) {
    Delta d;
    buf_ += text;
    step(d, false);
    return d;
}

OutputParser::Delta OutputParser::finish() {
    Delta d;
    step(d, true);
    return d;
}

}  // namespace raon::chat
