// lfm3-d1 decisions (see decision.h). Line references: llama.cpp tools/server/server-decision.cpp (v0.8.0 / b12674).
#include "decision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "gguf.h"
#include "tokenizer.h"

namespace d1 {

static const int RECURSIVE_LIMIT = 128;

// ------------------------------------------------------------------------------------------------
// tojson: Python json.dumps (ensure_ascii=False); floats as Python repr
// ------------------------------------------------------------------------------------------------
static std::string py_float(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
    // shortest digits that read back the same
    char buf[64];
    int prec = 1;
    for (; prec <= 17; prec++) {
        snprintf(buf, sizeof buf, "%.*e", prec - 1, v);
        if (strtod(buf, nullptr) == v) break;
    }
    // buf = [-]d.ddde[+-]XX
    std::string s = buf;
    const bool neg = s[0] == '-';
    if (neg) s.erase(0, 1);
    const size_t e = s.find('e');
    std::string digits = s.substr(0, e);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
    const int exp = atoi(s.c_str() + e + 1);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    std::string out;
    if (exp >= -4 && exp < 16) {  // repr: fixed notation
        if (exp >= 0) {
            if ((int)digits.size() <= exp + 1) out = digits + std::string(exp + 1 - digits.size(), '0') + ".0";
            else out = digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
        } else {
            out = "0." + std::string(-exp - 1, '0') + digits;
        }
    } else {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    }
    return neg ? "-" + out : out;
}

static void py_str(std::string& o, const std::string& s) {
    o += '"';
    for (char c : s) {
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
}

static void py_dump(std::string& o, const ojson& v, int indent, int level, const std::string& isep, const std::string& ksep) {
    auto nl = [&](int lv) {
        if (indent >= 0) {
            o += '\n';
            o.append((size_t)indent * lv, ' ');
        }
    };
    switch (v.type()) {
        case ojson::value_t::null: o += "null"; break;
        case ojson::value_t::boolean: o += v.get<bool>() ? "true" : "false"; break;
        case ojson::value_t::number_integer: o += std::to_string(v.get<int64_t>()); break;
        case ojson::value_t::number_unsigned: o += std::to_string((int64_t)v.get<uint64_t>()); break;
        case ojson::value_t::number_float: o += py_float(v.get<double>()); break;
        case ojson::value_t::string: py_str(o, v.get<std::string>()); break;
        case ojson::value_t::array: {
            o += '[';
            if (!v.empty()) {
                size_t i = 0;
                for (const auto& x : v) {
                    nl(level + 1);
                    py_dump(o, x, indent, level + 1, isep, ksep);
                    if (++i < v.size()) o += isep;
                }
                nl(level);
            }
            o += ']';
            break;
        }
        case ojson::value_t::object: {
            o += '{';
            if (!v.empty()) {
                size_t i = 0;
                for (const auto& [k, x] : v.items()) {
                    nl(level + 1);
                    py_str(o, k);
                    o += ksep;
                    py_dump(o, x, indent, level + 1, isep, ksep);
                    if (++i < v.size()) o += isep;
                }
                nl(level);
            }
            o += '}';
            break;
        }
        default: o += "null";
    }
}

std::string tojson(const ojson& v, int indent, const std::string& item_sep, const std::string& key_sep) {
    std::string o;
    py_dump(o, v, indent, 0, item_sep, key_sep);
    return o;
}

static std::string tojson_compact(const ojson& v) { return tojson(v, -1, ", ", ": "); }

// Python truthiness of a template value
static bool truthy(const ojson& v) {
    switch (v.type()) {
        case ojson::value_t::null: return false;
        case ojson::value_t::boolean: return v.get<bool>();
        case ojson::value_t::number_integer: return v.get<int64_t>() != 0;
        case ojson::value_t::number_unsigned: return v.get<uint64_t>() != 0;
        case ojson::value_t::number_float: return v.get<double>() != 0;
        case ojson::value_t::string: return !v.get<std::string>().empty();
        case ojson::value_t::array:
        case ojson::value_t::object: return !v.empty();
        default: return false;
    }
}

// ------------------------------------------------------------------------------------------------
// context
// ------------------------------------------------------------------------------------------------
Context::Context(const Tokenizer& tk, const gguf::File& f) : tk_(tk) {
    // init_quoted_spellings(CONTROL, 2): control token texts of >= 2 bytes, longest first (ties: token id order)
    for (int id = 0; id < tk.n_vocab(); id++)
        if (tk.is_control(id)) {
            const std::string t = tk.decode(id, true);
            if (t.size() >= 2) spellings_.push_back(t);
        }
    std::stable_sort(spellings_.begin(), spellings_.end(), [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    for (const auto& s : spellings_)
        if (first_chars_.find(s[0]) == std::string::npos) first_chars_ += s[0];
    // temperatures: read the way llama.cpp does, through "%f" text (std::to_string(float)) and strtof
    const std::string pre = "lfm3.decision.temperature.";
    for (const auto& [k, v] : f.kv)
        if (k.rfind(pre, 0) == 0) {
            char b[64];
            snprintf(b, sizeof b, "%f", (double)(float)v.num);
            const float t = strtof(b, nullptr);
            if (t <= 0) throw std::runtime_error("invalid decision temperature: " + k + " = " + b);
            temps_[k.substr(pre.size())] = t;
        }
}

std::string Context::d1_break(const std::string& text) const {
    static const std::string zwsp = "\xE2\x80\x8B";
    std::string cur = text;
    for (bool changed = true; changed;) {  // again for a match that overlapped the one broken before it
        changed = false;
        std::string out;
        for (size_t i = 0; i < cur.size();) {
            const std::string* hit = nullptr;
            if (first_chars_.find(cur[i]) != std::string::npos)
                for (const auto& s : spellings_)
                    if (cur.compare(i, s.size(), s) == 0) { hit = &s; break; }
            if (!hit) { out += cur[i++]; continue; }
            const unsigned char c = (*hit)[0];
            const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            out += hit->substr(0, n) + zwsp + hit->substr(n);
            i += hit->size();
            changed = true;
        }
        cur = std::move(out);
    }
    return cur;
}

ojson Context::escape(const ojson& v, int depth) const {
    if (depth >= RECURSIVE_LIMIT)
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    if (v.is_string()) return d1_break(v.get<std::string>());
    if (v.is_array()) {
        ojson out = ojson::array();
        for (const auto& x : v) out.push_back(escape(x, depth + 1));
        return out;
    }
    if (v.is_object()) {
        ojson out = ojson::object();
        for (const auto& [k, x] : v.items()) out[d1_break(k)] = escape(x, depth + 1);
        return out;
    }
    return v;
}

// ------------------------------------------------------------------------------------------------
// request
// ------------------------------------------------------------------------------------------------
std::vector<Question> Context::parse_questions(const ojson& body) const {
    if (!body.is_object() || !body.contains("state")) throw std::invalid_argument("\"state\" must be provided");
    if (!body.contains("questions") || !body["questions"].is_object() || body["questions"].empty())
        throw std::invalid_argument("\"questions\" must be a non-empty object");
    std::vector<Question> out;
    for (const auto& [id, q] : body["questions"].items()) {
        auto err = [&id = id](const std::string& m) { return std::invalid_argument("questions." + id + ": " + m); };
        if (!q.is_object()) throw err("must be an object");
        const ojson instructions = q.contains("instructions") ? q["instructions"] : ojson();
        if (instructions.is_null()) throw err("\"instructions\" must be provided");
        if (instructions.is_object() && instructions.contains("type") && instructions["type"] == "input_audio")
            throw err("spoken instructions need a model with audio input");
        Question qu;
        qu.id = id;
        qu.instructions = instructions;
        const std::string type = q.contains("type") && q["type"].is_string() ? q["type"].get<std::string>() : "";
        const ojson criteria = q.contains("criteria") ? q["criteria"] : ojson();
        if (type == "choice") {
            qu.type = QType::Choice;
            if (!criteria.is_object() || criteria.empty()) throw err("\"criteria\" must be a non-empty object");
            for (const auto& [k, d] : criteria.items()) qu.options.push_back({k, d});
        } else if (type == "score") {
            qu.type = QType::Score;
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10)
                throw err("\"criteria\" must be an array of 2 to 10 levels");
            for (size_t i = 0; i < criteria.size(); i++) qu.options.push_back({std::to_string(i), criteria[i]});
        } else if (type == "noul") {
            qu.type = QType::Noul;
            if (!criteria.is_null() && !criteria.is_object()) throw err("\"criteria\" must be an object");
            for (const char* k : {"true", "false"})  // true first
                qu.options.push_back({k, criteria.is_object() && criteria.contains(k) ? criteria[k] : ojson()});
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
        }
        if (qu.options.size() > 255)
            throw err("too many options (" + std::to_string(qu.options.size()) + "), this model supports at most 255");
        labels(qu);
        out.push_back(std::move(qu));
    }
    return out;
}

static void check_data_url(const ojson& url, std::vector<std::string>& media) {
    if (!url.is_string() || url.get<std::string>().rfind("data:", 0) != 0)
        throw std::invalid_argument("media must be data URLs (data:image/..., data:audio/... or data:video/...;base64,...)");
    if ((int)media.size() >= MAX_MEDIA)
        throw std::invalid_argument("too many images, the maximum is " + std::to_string(MAX_MEDIA));
    const std::string u = url.get<std::string>();
    const size_t c = u.find(',');
    if (c == std::string::npos || u.find(',', c + 1) != std::string::npos) throw std::invalid_argument("Invalid uri-encoded base64 value");
    const std::string head = u.substr(0, c);
    if (head.rfind("data:image/", 0) != 0 && head.rfind("data:video/", 0) != 0 && head.rfind("data:audio/", 0) != 0)
        throw std::invalid_argument("Invalid uri format: " + head);
    if (head.size() < 6 || head.compare(head.size() - 6, 6, "base64") != 0) throw std::invalid_argument("uri must be base64 encoded");
    if (head.rfind("data:audio/", 0) == 0) throw std::invalid_argument("audio input is not supported by this model");
    media.push_back(u);
}

ojson Context::parse_state(const ojson& body, std::vector<std::string>& media) const {
    if (body.contains("videos") && !body["videos"].is_null() && !body["videos"].empty())
        throw std::invalid_argument("\"videos\" is not supported");
    for (const char* key : {"files", "images"}) {  // "images" is an alias of "files"
        if (!body.contains(key) || body[key].is_null()) continue;
        if (!body[key].is_array()) throw std::invalid_argument(std::string("\"") + key + "\" must be an array");
        for (const auto& u : body[key]) check_data_url(u, media);
    }
    const ojson& state = body["state"];
    const bool wrapped = state.is_object() && state.contains("messages");
    const ojson& messages = wrapped ? state["messages"] : state;
    if (!messages.is_array()) return state;
    // chat messages: the image / video parts leave the content and become media
    ojson msgs = ojson::array();
    for (const auto& m : messages) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) { msgs.push_back(m); continue; }
        ojson content = ojson::array();
        for (const auto& part : m["content"]) {
            const std::string ty = part.is_object() && part.contains("type") && part["type"].is_string() ? part["type"].get<std::string>() : "";
            if (ty == "image_url" && part.contains("image_url")) {
                const ojson& iu = part["image_url"];
                check_data_url(iu.is_object() && iu.contains("url") ? iu["url"] : iu, media);
            } else if (ty == "video_url" && part.contains("video_url")) {
                const ojson& vu = part["video_url"];
                check_data_url(vu.is_object() && vu.contains("url") ? vu["url"] : vu, media);
            } else if (ty == "input_audio" && part.contains("input_audio")) {
                throw std::invalid_argument("audio input is not supported by this model");
            } else {
                content.push_back(part);
            }
        }
        ojson mo = m;
        mo["content"] = content;
        msgs.push_back(mo);
    }
    if (!wrapped) return msgs;
    ojson so = state;
    so["messages"] = msgs;
    return so;
}

// ------------------------------------------------------------------------------------------------
// prompt (the systemone template, split into the shared prefix and the question text)
// ------------------------------------------------------------------------------------------------
std::string Context::prefix(const ojson& state, const std::string& media_marks) const {
    std::string o = "<|startoftext|><|im_start|>user\n" + media_marks;
    if (!state.is_null()) {
        const ojson s = escape(state);
        o += s.is_string() ? s.get<std::string>() : tojson(s, 2, ",", ": ");
        o += "\n\n\nQUESTION:\n";
    }
    return o;
}

std::string Context::question_text(const Question& q) const {
    auto text = [](const ojson& v) { return v.is_string() ? v.get<std::string>() : tojson_compact(v); };
    std::string o = text(escape(q.instructions));
    const size_t n = q.options.size();
    if (q.type == QType::Choice) {
        o += "\n\nOptions:\n";
        for (size_t i = 0; i < n; i++) {
            const ojson d = escape(q.options[i].description);
            o += q.labels[i] + " ";
            if (truthy(d)) {
                o += text(d);
            } else {
                std::string k = d1_break(q.options[i].key);
                std::replace(k.begin(), k.end(), '_', ' ');
                o += k;
            }
            if (i + 1 < n) o += "\n";
        }
        o += "\n\nReply with the option code only.";
    } else if (q.type == QType::Noul) {
        bool any = false;
        for (const auto& op : q.options) any |= !op.description.is_null();
        if (any)
            for (const auto& op : q.options) {
                o += op.key == "true" ? "\nYes: " : "\nNo: ";
                o += op.description.is_null() ? std::string("None") : text(escape(op.description));
            }
        o += "\n\nReply with yes or no only.";
    } else {
        o += "\n\n";
        for (const auto& op : q.options) o += op.key + " " + text(escape(op.description)) + "\n";
        o += "\nReply with a single digit 0-" + std::to_string(n - 1) + " only.";
    }
    return o + "<|im_end|>\n<|im_start|>assistant\n<think></think>";
}

// ------------------------------------------------------------------------------------------------
// labels (d1_labels)
// ------------------------------------------------------------------------------------------------
void Context::labels(Question& q) const {
    auto single = [&](const std::vector<std::string>& forms) {
        std::vector<int> out;
        for (const auto& f : forms) {
            const auto t = tk_.encode(f, false);
            if (t.size() == 1 && std::find(out.begin(), out.end(), t[0]) == out.end()) out.push_back(t[0]);
        }
        return out;
    };
    if (q.type != QType::Choice) {
        for (const auto& op : q.options) {
            std::vector<int> g = q.type == QType::Score ? single({op.key})
                                 : op.key == "true"     ? single({"yes", "Yes", "YES"})
                                                        : single({"no", "No", "NO"});
            if (g.empty()) throw std::runtime_error("decision label is not a single token: " + op.key);
            q.labels.push_back(op.key);
            q.groups.push_back(g);
        }
        return;
    }
    const size_t n = q.options.size();
    bool letters = true;
    for (const auto& op : q.options) letters = letters && op.key.size() == 1 && std::isalpha((unsigned char)op.key[0]);
    std::vector<std::string> codes;
    for (size_t i = 0; i < n; i++) {
        if (letters) codes.push_back(q.options[i].key);
        else if (i < 26) codes.push_back(std::string(1, (char)('A' + i)));
        else codes.push_back(std::string{(char)('A' + (i - 26) / 26), (char)('A' + (i - 26) % 26)});
    }
    std::vector<std::string> pool;
    for (char c = 'A'; c <= 'Z'; c++) pool.push_back(std::string(1, c));
    for (int i = 0; i < 100; i++) {
        char b[8];
        snprintf(b, sizeof b, "%02d", i);
        pool.push_back(b);
    }
    for (char c = 'a'; c <= 'z'; c++) pool.push_back(std::string(1, c));
    for (int i = 0; i < 200; i++) pool.push_back("#" + std::to_string(i));
    for (char a = 'A'; a <= 'Z'; a++)
        for (char b = 'A'; b <= 'Z'; b++) pool.push_back(std::string{a, b});
    std::vector<int> used;
    auto take = [&](const std::string& code) {
        const auto t = tk_.encode(code, false);
        if (t.size() != 1 || std::find(used.begin(), used.end(), t[0]) != used.end()) return false;
        used.push_back(t[0]);
        std::vector<int> g = {t[0]};
        for (int x : single({" " + code}))
            if (x != t[0]) g.push_back(x);
        q.labels.push_back(code);
        q.groups.push_back(g);
        return true;
    };
    for (const auto& code : codes) {
        bool ok = take(code);
        for (size_t i = 0; !ok && i < pool.size(); i++) ok = take(pool[i]);
        if (!ok) throw std::invalid_argument("no single-token label left for " + std::to_string(n) + " options");
    }
}

// ------------------------------------------------------------------------------------------------
// answers
// ------------------------------------------------------------------------------------------------
float Context::temperature(const Question& q, bool has_media) const {
    const size_t n = q.options.size();
    const std::string tn = q.type == QType::Choice ? "choice" : q.type == QType::Score ? "score" : "noul";
    const std::string bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    float t = 1.0f;
    if (auto it = temps_.find(tn + "." + bucket); it != temps_.end()) t = it->second;
    else if (auto it2 = temps_.find(tn); it2 != temps_.end()) t = it2->second;
    if (has_media)
        if (auto it = temps_.find("image"); it != temps_.end()) t = it->second;
    return t;
}

ojson Context::answer(const Question& q, const std::vector<float>& s, bool has_media) const {
    const size_t n = q.options.size();
    for (float v : s)
        if (std::isnan(v) || v == INFINITY) throw std::runtime_error("the model could not evaluate the decision");
    const float temp = temperature(q, has_media);
    std::vector<double> lg(s.begin(), s.end()), p(n);
    const double mx = *std::max_element(lg.begin(), lg.end());
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        p[i] = std::exp((lg[i] - mx) / temp);
        sum += p[i];
    }
    for (size_t i = 0; i < n; i++) p[i] = 0.0 + p[i] / sum / 1.0;
    ojson a = {{"type", q.type == QType::Choice ? "choice" : q.type == QType::Score ? "score" : "noul"}};
    if (q.type == QType::Noul) {
        for (size_t i = 0; i < n; i++)
            if (q.options[i].key == "true") a["noul"] = p[i];
        return a;
    }
    ojson probs = ojson::object();
    for (size_t i = 0; i < n; i++) probs[q.options[i].key] = p[i];
    const double conf = *std::max_element(p.begin(), p.end());
    if (q.type == QType::Choice) {
        const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        a["choice"] = q.options[best].key;
        a["probabilities"] = probs;
        a["confidence"] = conf;
        return a;
    }
    ojson legend = ojson::object();
    for (size_t i = 0; i < n; i++) legend[q.options[i].key] = q.options[i].description;
    double mass = 0;  // the median level: the first whose cumulative probability reaches one half
    size_t level = n - 1;
    for (size_t i = 0; i < n && mass < 0.5; i++) {
        mass += p[i];
        level = i;
    }
    a["score"] = (double)level;
    a["legend"] = legend;
    a["probabilities"] = probs;
    a["confidence"] = conf;
    return a;
}

}  // namespace d1
