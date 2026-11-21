// liquid-server: OpenAI-compatible HTTP server for liquid-engine.
// One GPU thread (scheduler) owns the engine: request queue -> slots with prefix reuse -> chunked prefill
// interleaved with batched decode. HTTP worker threads (cpp-httplib) render templates, stream SSE, parse output.
#include <algorithm>
#include <cstdarg>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"
#include "chat.h"
#include "engine.h"
#include "gguf.h"
#include "prompt_cache.h"
#include "sampling_defaults.h"
#include "tokenizer.h"
#include "web_tools.h"

using chat::json;
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// ================================================================================================
// request / events
// ================================================================================================
struct Request {
    uint64_t id = 0;
    std::vector<int> prompt;
    le::SamplingParams sp;
    std::vector<int> stop_tokens;
    int max_tokens = 0;
    // results (scheduler -> HTTP thread)
    std::mutex m;
    std::condition_variable cv;
    std::vector<int> tokens;      // generated tokens (appended by the scheduler)
    bool finished = false;
    std::string finish_reason;    // "stop" | "length" | "error"
    std::string error;
    std::atomic<bool> cancelled{false};
    // timings
    clk::time_point t_arrive, t_admit, t_first;
    double prefill_ms = 0, decode_ms = 0;
    int n_cached = 0;             // prompt tokens served from a cache tier (no prefill)
    double restore_ms = 0;        // time to load them from the RAM/disk tier
    std::string cache_tier;       // "", "vram", "ram", "disk"
};

// ================================================================================================
// scheduler
// ================================================================================================
struct SchedConfig {
    int idle_chunk = 4096;   // prompt tokens per feed() when nothing is decoding
    int busy_chunk = 1024;   // ... while others decode (one decode step per chunk); swept: bench/server_load.py stall
    int max_queue = 64;
    int checkpoint_every = 16384;  // extra checkpoints inside long prompts
    int max_checkpoints = 6;
};

class Scheduler {
public:
    Scheduler(le::Engine& e, const Tokenizer& tk, SchedConfig cfg, PromptCache* cache)
        : e_(e), tk_(tk), cfg_(cfg), cache_(cache), slots_(e.n_slots()) {
        for (int i = 0; i < (int)slots_.size(); i++) slots_[i].id = i;
        th_ = std::thread([this] { loop(); });
    }
    ~Scheduler() { stop(); }
    void stop() {
        {
            std::lock_guard<std::mutex> l(qm_);
            if (stopping_) return;
            stopping_ = true;
        }
        qcv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    // returns false when the queue is full
    bool submit(const std::shared_ptr<Request>& r) {
        {
            std::lock_guard<std::mutex> l(qm_);
            if ((int)queue_.size() >= cfg_.max_queue) return false;
            queue_.push_back(r);
        }
        qcv_.notify_all();
        return true;
    }
    json stats() {
        std::lock_guard<std::mutex> l(stat_m_);
        return stats_;
    }

private:
    enum State { FREE, PREFILL, DECODE };
    struct Slot {
        int id = 0;
        State st = FREE;
        std::shared_ptr<Request> req;
        std::vector<int> tokens;  // consumed tokens == KV positions [0, n_past)
        int pending = -1;
        std::vector<std::pair<int, le::Engine::Checkpoint>> cps;  // (position, checkpoint)
        uint64_t last_use = 0;
        clk::time_point t_prefill_start;
    };
    le::Engine& e_;
    const Tokenizer& tk_;
    SchedConfig cfg_;
    PromptCache* cache_;
    std::vector<Slot> slots_;
    std::thread th_;
    std::mutex qm_;
    std::condition_variable qcv_;
    std::deque<std::shared_ptr<Request>> queue_;
    bool stopping_ = false;
    uint64_t tick_ = 0;
    std::mutex stat_m_;
    json stats_ = json::object();

    static int common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
        size_t k = 0;
        while (k < a.size() && k < b.size() && a[k] == b[k]) k++;
        return (int)k;
    }
    // best resumable position of slot s for prompt p (must leave >= 1 prompt token to become pending)
    int usable(const Slot& s, const std::vector<int>& p) const {
        const int L = std::min(common_prefix(s.tokens, p), (int)p.size() - 1);
        if (L <= 0) return 0;
        if (L == (int)s.tokens.size()) return L;  // the live state is exactly at the end of the shared prefix
        int best = 0;
        for (auto& c : s.cps)
            if (c.first <= L) best = std::max(best, c.first);
        return best;
    }
    void finish(Slot& s, const char* reason, const std::string& err = "") {
        auto r = s.req;
        {
            std::lock_guard<std::mutex> l(r->m);
            r->finished = true;
            r->finish_reason = reason;
            r->error = err;
        }
        r->cv.notify_all();
        s.req.reset();
        s.st = FREE;
        s.last_use = ++tick_;
    }
    void add_checkpoint(Slot& s) {
        const int pos = (int)s.tokens.size();
        for (auto& c : s.cps)
            if (c.first == pos) return;
        le::Engine::Checkpoint cp;
        e_.save_checkpoint(s.id, cp);
        s.cps.emplace_back(pos, std::move(cp));
        if ((int)s.cps.size() > cfg_.max_checkpoints) s.cps.erase(s.cps.begin());
    }
    // move a slot's contents to the RAM/disk tier before reuse for an unrelated prompt
    void evict(Slot& s) {
        if (cache_ && !s.cps.empty()) {
            auto& last = s.cps.back();  // the most useful resume point (usually end of the last prompt)
            cache_->store(e_, s.id, std::vector<int>(s.tokens.begin(), s.tokens.begin() + last.first), last.second);
        }
        s.tokens.clear();
        s.cps.clear();
        s.pending = -1;
    }

    void admit() {
        for (;;) {
            std::shared_ptr<Request> r;
            {
                std::lock_guard<std::mutex> l(qm_);
                if (queue_.empty()) return;
                bool any_free = false;
                for (auto& s : slots_) any_free |= s.st == FREE;
                if (!any_free) return;
                r = queue_.front();
                queue_.pop_front();
            }
            if (r->cancelled) { finish_unslotted(r, "stop"); continue; }
            r->t_admit = clk::now();
            const auto& p = r->prompt;
            if ((int)p.size() + 1 >= e_.max_ctx()) { finish_unslotted(r, "error", "prompt exceeds the context size"); continue; }
            // pick a slot: longest reusable prefix among free slots, else the least recently used one
            int best = -1, best_pos = 0;
            for (auto& s : slots_) {
                if (s.st != FREE) continue;
                const int u = usable(s, p);
                if (best < 0 || u > best_pos || (u == best_pos && s.last_use < slots_[best].last_use)) { best = s.id; best_pos = u; }
            }
            Slot& s = slots_[best];
            int start = best_pos;
            std::string tier = start > 0 ? "vram" : "";
            // the RAM/disk tier may hold a longer prefix than any slot
            int cache_n = 0;
            const uint64_t cid = cache_ ? cache_->lookup(p, start, cache_n) : 0;
            if (cid) {
                evict(s);  // the slot's own contents go down a tier before being overwritten
                e_.reset(s.id);
                const auto t0 = clk::now();
                if (cache_->restore(e_, s.id, cid, tier)) {
                    s.tokens.assign(p.begin(), p.begin() + cache_n);
                    le::Engine::Checkpoint cp;
                    e_.save_checkpoint(s.id, cp);  // the restored state is this slot's first checkpoint
                    s.cps.emplace_back(cache_n, std::move(cp));
                    start = cache_n;
                    r->restore_ms = ms_since(t0);
                    log("req %llu: restored %d prompt tokens from the %s tier in %.1f ms", (unsigned long long)r->id, cache_n,
                        tier.c_str(), ms_since(t0));
                } else {
                    e_.reset(s.id);
                    start = 0;
                    tier.clear();
                }
            } else if (start == 0) {
                evict(s);
                e_.reset(s.id);
            } else if (start != (int)s.tokens.size()) {
                // rewind to a checkpoint inside the shared prefix
                for (auto& c : s.cps)
                    if (c.first == start) { e_.load_checkpoint(s.id, c.second); break; }
                s.tokens.resize(start);
                s.cps.erase(std::remove_if(s.cps.begin(), s.cps.end(), [&](auto& c) { return c.first > start; }), s.cps.end());
            }
            r->n_cached = start;
            r->cache_tier = tier;
            s.req = r;
            s.st = PREFILL;
            s.t_prefill_start = clk::now();
            e_.set_sampling(s.id, r->sp, r->stop_tokens);
        }
    }
    void finish_unslotted(const std::shared_ptr<Request>& r, const char* reason, const std::string& err = "") {
        {
            std::lock_guard<std::mutex> l(r->m);
            r->finished = true;
            r->finish_reason = reason;
            r->error = err;
        }
        r->cv.notify_all();
    }

    void prefill_one(bool decoders_active) {
        Slot* s = nullptr;
        for (auto& x : slots_)
            if (x.st == PREFILL && (!s || x.req->t_admit < s->req->t_admit)) s = &x;
        if (!s) return;
        auto& r = *s->req;
        if (r.cancelled) { finish(*s, "stop"); return; }
        const int target = (int)r.prompt.size() - 1;  // the last prompt token becomes the pending token
        const int fed = (int)s->tokens.size();
        int n = std::min(decoders_active ? cfg_.busy_chunk : cfg_.idle_chunk, target - fed);
        // stop at the next periodic checkpoint boundary
        const int next_cp = (fed / cfg_.checkpoint_every + 1) * cfg_.checkpoint_every;
        if (fed + n > next_cp && next_cp < target) n = next_cp - fed;
        if (n > 0) {
            try {
                e_.feed(s->id, r.prompt.data() + fed, n);
            } catch (const std::exception& ex) {
                finish(*s, "error", ex.what());
                return;
            }
            s->tokens.insert(s->tokens.end(), r.prompt.begin() + fed, r.prompt.begin() + fed + n);
            if ((int)s->tokens.size() % cfg_.checkpoint_every == 0 && (int)s->tokens.size() < target) add_checkpoint(*s);
        }
        if ((int)s->tokens.size() == target) {
            add_checkpoint(*s);  // end-of-prompt state: reused by the next turn of the conversation
            s->pending = r.prompt.back();
            e_.set_pending(s->id, s->pending);
            r.prefill_ms = ms_since(s->t_prefill_start);
            s->st = DECODE;
        }
    }

    void decode_all() {
        std::vector<int> ids;
        for (auto& s : slots_) {
            if (s.st != DECODE) continue;
            if (s.req->cancelled) { finish(s, "stop"); continue; }
            ids.push_back(s.id);
        }
        if (ids.empty()) return;
        std::vector<int> toks;
        std::vector<char> done;
        const auto t0 = clk::now();
        try {
            e_.decode_step(ids, toks, done);
        } catch (const std::exception& ex) {
            for (int id : ids) finish(slots_[id], "error", ex.what());
            return;
        }
        const double dt = ms_since(t0);
        for (size_t i = 0; i < ids.size(); i++) {
            Slot& s = slots_[ids[i]];
            auto& r = *s.req;
            s.tokens.push_back(s.pending);  // the pending token was consumed by this step
            s.pending = toks[i];
            r.decode_ms += dt;
            bool stop = done[i] != 0;
            {
                std::lock_guard<std::mutex> l(r.m);
                if (r.tokens.empty()) r.t_first = clk::now();
                r.tokens.push_back(toks[i]);
            }
            r.cv.notify_all();
            if (stop) finish(s, "stop");
            else if ((int)r.tokens.size() >= r.max_tokens) finish(s, "length");
            else if ((int)s.tokens.size() + 2 >= e_.max_ctx()) finish(s, "length");
        }
    }

    void loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> l(qm_);
                bool busy = false;
                for (auto& s : slots_) busy |= s.st != FREE;
                qcv_.wait_for(l, std::chrono::milliseconds(busy ? 0 : 50), [&] { return stopping_ || !queue_.empty() || busy; });
                if (stopping_) break;
            }
            admit();
            bool decoders = false, prefills = false;
            for (auto& s : slots_) { decoders |= s.st == DECODE; prefills |= s.st == PREFILL; }
            if (prefills) prefill_one(decoders);
            decode_all();
            update_stats();
        }
        for (auto& s : slots_)
            if (s.req) finish(s, "error", "server shutting down");
        if (cache_) for (auto& s : slots_) evict(s);
        std::lock_guard<std::mutex> l(qm_);
        for (auto& r : queue_) finish_unslotted(r, "error", "server shutting down");
        queue_.clear();
    }
    void update_stats() {
        json j;
        int busy = 0;
        json sl = json::array();
        for (auto& s : slots_) {
            busy += s.st != FREE;
            sl.push_back({{"id", s.id}, {"state", s.st == FREE ? "idle" : s.st == PREFILL ? "prefill" : "decode"},
                          {"n_past", s.tokens.size()}, {"checkpoints", s.cps.size()}});
        }
        size_t qn;
        {
            std::lock_guard<std::mutex> l(qm_);
            qn = queue_.size();
        }
        j["slots"] = sl;
        j["busy_slots"] = busy;
        j["queued"] = qn;
        if (cache_) j["cache"] = cache_->stats();
        std::lock_guard<std::mutex> l(stat_m_);
        stats_ = j;
    }

public:
    static void log(const char* fmt, ...) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        fprintf(stderr, "[liquid-server] %s\n", buf);
    }
};

// ================================================================================================
// HTTP / OpenAI layer
// ================================================================================================
struct ServerConfig {
    std::string model_alias = "lfm3";
    le::SamplingParams sp;   // defaults (model card)
    int max_tokens = 4096;   // default when the request has none
    bool reasoning_preserve = false;
};

static std::string rand_id(const char* prefix) {
    static std::atomic<uint64_t> ctr{0};
    static const uint64_t seed = std::random_device{}();
    char b[64];
    snprintf(b, sizeof b, "%s%016llx", prefix, (unsigned long long)(seed ^ (++ctr * 0x9E3779B97F4A7C15ull)));
    return b;
}

// incremental UTF-8-safe detokenizer
struct Detok {
    const Tokenizer* tk;
    std::string hold;
    std::string push(int id) {
        hold += tk->decode(id, false);
        size_t n = hold.size(), cut = n;
        for (size_t k = 1; k <= std::min<size_t>(3, n); k++) {
            const unsigned char ch = hold[n - k];
            if ((ch & 0xC0) == 0x80) continue;
            const int len = ch >= 0xF0 ? 4 : ch >= 0xE0 ? 3 : ch >= 0xC0 ? 2 : 1;
            if ((int)k < len) cut = n - k;
            break;
        }
        std::string out = hold.substr(0, cut);
        hold.erase(0, cut);
        return out;
    }
    std::string flush() { std::string o; o.swap(hold); return o; }
};

struct ApiServer {
    le::Engine& eng;
    const Tokenizer& tk;
    Scheduler& sched;
    ServerConfig cfg;
    std::atomic<uint64_t> next_id{1};

    static json err_json(const std::string& msg, const char* type = "invalid_request_error") {
        return {{"error", {{"message", msg}, {"type", type}}}};
    }

    void parse_sampling(const json& b, le::SamplingParams& sp, int& max_tokens) {
        sp = cfg.sp;
        auto num = [&](const char* k, auto& dst) {
            if (b.contains(k) && b[k].is_number()) dst = b[k].get<std::decay_t<decltype(dst)>>();
        };
        num("temperature", sp.temperature);
        num("top_p", sp.top_p);
        num("top_k", sp.top_k);
        num("min_p", sp.min_p);
        num("repetition_penalty", sp.repeat_penalty);  // transformers / vLLM name
        num("repeat_penalty", sp.repeat_penalty);
        num("repeat_last_n", sp.repeat_last_n);
        if (b.contains("seed") && b["seed"].is_number_integer()) sp.seed = b["seed"].get<uint64_t>();
        else sp.seed = std::random_device{}() * 0x9E3779B97F4A7C15ull;
        max_tokens = cfg.max_tokens;
        if (b.contains("max_completion_tokens") && b["max_completion_tokens"].is_number_integer()) max_tokens = b["max_completion_tokens"];
        else if (b.contains("max_tokens") && b["max_tokens"].is_number_integer()) max_tokens = b["max_tokens"];
        if (max_tokens <= 0) max_tokens = cfg.max_tokens;
    }
    static std::vector<std::string> parse_stop(const json& b) {
        std::vector<std::string> v;
        if (b.contains("stop")) {
            if (b["stop"].is_string()) v.push_back(b["stop"]);
            else if (b["stop"].is_array())
                for (auto& s : b["stop"]) if (s.is_string() && !s.get<std::string>().empty()) v.push_back(s);
        }
        return v;
    }

    // Wait for new tokens of r; returns false when the request finished and all tokens were consumed.
    static bool next_tokens(Request& r, size_t& seen, std::vector<int>& out) {
        std::unique_lock<std::mutex> l(r.m);
        r.cv.wait_for(l, std::chrono::milliseconds(200), [&] { return r.finished || r.tokens.size() > seen; });
        out.assign(r.tokens.begin() + seen, r.tokens.end());
        seen = r.tokens.size();
        return !(r.finished && out.empty());
    }

    json usage_json(const Request& r, int completion) const {
        return {{"prompt_tokens", r.prompt.size()}, {"completion_tokens", completion},
                {"total_tokens", (int)r.prompt.size() + completion},
                {"prompt_tokens_details", {{"cached_tokens", r.n_cached}}}};
    }
    json timings_json(const Request& r, int completion) const {
        const double queue_ms = std::chrono::duration<double, std::milli>(r.t_admit - r.t_arrive).count();
        const int fed = (int)r.prompt.size() - r.n_cached;
        return {{"queue_ms", queue_ms}, {"prompt_n", fed}, {"prompt_ms", r.prefill_ms},
                {"prompt_per_second", r.prefill_ms > 0 ? fed / (r.prefill_ms / 1e3) : 0.0}, {"cache_n", r.n_cached},
                {"cache_tier", r.cache_tier}, {"restore_ms", r.restore_ms}, {"predicted_n", completion}, {"predicted_ms", r.decode_ms},
                {"predicted_per_second", r.decode_ms > 0 ? completion / (r.decode_ms / 1e3) : 0.0}};
    }

    // shared generation driver: calls emit(reasoning, content, calls, final, finish_reason) as text arrives
    template <class Emit>
    void drive(const std::shared_ptr<Request>& r, bool chat_mode, bool in_reasoning, const json& tools,
               const std::vector<std::string>& stops, Emit&& emit) {
        Detok dt{&tk, ""};
        chat::OutputParser parser(in_reasoning, tools);
        std::string text_all;   // completions mode / stop-string scan
        size_t seen = 0, sent_upto = 0;
        std::vector<int> fresh;
        bool stopped_by_string = false;
        while (next_tokens(*r, seen, fresh)) {
            std::string piece;
            for (int t : fresh) piece += dt.push(t);
            if (piece.empty()) continue;
            text_all += piece;
            size_t limit = text_all.size();
            if (!stops.empty()) {  // stop strings: cut there and cancel generation
                size_t cut = std::string::npos;
                for (auto& s : stops) {
                    const size_t p = text_all.find(s, sent_upto > s.size() ? sent_upto - s.size() : 0);
                    if (p != std::string::npos) cut = std::min(cut, p);
                }
                if (cut != std::string::npos) { limit = cut; stopped_by_string = true; r->cancelled = true; }
                else {  // hold back a possible partial stop string
                    size_t hold = 0;
                    for (auto& s : stops)
                        for (size_t k = std::min(s.size() - 1, text_all.size()); k > hold; k--)
                            if (text_all.compare(text_all.size() - k, k, s, 0, k) == 0) { hold = k; break; }
                    limit = text_all.size() - hold;
                }
            }
            if (limit > sent_upto) {
                const std::string delta = text_all.substr(sent_upto, limit - sent_upto);
                sent_upto = limit;
                std::string rd, cd;
                std::vector<chat::ToolCall> calls;
                if (chat_mode) parser.feed(delta, rd, cd, calls);
                else cd = delta;
                if (!emit(rd, cd, calls, false, "")) { r->cancelled = true; break; }
            }
            if (stopped_by_string) break;
        }
        // drain: wait for the scheduler to finish this request (cancel acknowledged)
        {
            std::unique_lock<std::mutex> l(r->m);
            r->cv.wait(l, [&] { return r->finished; });
        }
        std::string rd, cd;
        std::vector<chat::ToolCall> calls;
        if (!stopped_by_string) {
            const std::string tail = dt.flush();
            if (!tail.empty()) {
                if (chat_mode) parser.feed(tail, rd, cd, calls);
                else cd += tail;
            }
        }
        if (chat_mode) parser.finish(rd, cd, calls);
        std::string reason = stopped_by_string ? "stop" : r->finish_reason;
        if (chat_mode && !parser.tool_calls().empty() && reason == "stop") reason = "tool_calls";
        emit(rd, cd, calls, true, reason);
    }

    static json tool_calls_json(const std::vector<chat::ToolCall>& calls, size_t base_index, bool with_index) {
        json arr = json::array();
        for (size_t i = 0; i < calls.size(); i++) {
            json c = {{"id", rand_id("call_")}, {"type", "function"},
                      {"function", {{"name", calls[i].name}, {"arguments", calls[i].arguments.dump()}}}};
            if (with_index) c["index"] = base_index + i;
            arr.push_back(c);
        }
        return arr;
    }

    void handle(const httplib::Request& hreq, httplib::Response& res, bool chat_mode) {
        json b;
        try {
            b = json::parse(hreq.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(err_json(std::string("invalid JSON: ") + e.what()).dump(), "application/json");
            return;
        }
        auto r = std::make_shared<Request>();
        r->id = next_id++;
        r->t_arrive = clk::now();
        parse_sampling(b, r->sp, r->max_tokens);
        const auto stops = parse_stop(b);
        json tools = json::array();
        bool in_reasoning = false;
        try {
            if (chat_mode) {
                if (!b.contains("messages") || !b["messages"].is_array()) throw std::runtime_error("'messages' is required");
                if (b.contains("tools") && b["tools"].is_array() && b.value("tool_choice", json("auto")) != "none") tools = b["tools"];
                chat::TemplateOptions o;
                o.preserve_thinking = cfg.reasoning_preserve;
                if (b.contains("chat_template_kwargs") && b["chat_template_kwargs"].is_object()) {
                    const auto& k = b["chat_template_kwargs"];
                    if (k.contains("enable_thinking") && k["enable_thinking"].is_boolean()) o.enable_thinking = k["enable_thinking"];
                    if (k.contains("preserve_thinking") && k["preserve_thinking"].is_boolean()) o.preserve_thinking = k["preserve_thinking"];
                }
                if (b.contains("reasoning_effort") && b["reasoning_effort"] == "none") o.enable_thinking = false;
                in_reasoning = o.enable_thinking;
                r->prompt = tk.encode(chat::render(b["messages"], tools, o));
            } else {
                const json& p = b.value("prompt", json(""));
                if (p.is_string()) r->prompt = tk.encode(p.get<std::string>());
                else if (p.is_array() && !p.empty() && p[0].is_number_integer())
                    for (auto& x : p) r->prompt.push_back(x.get<int>());
                else throw std::runtime_error("'prompt' must be a string or an array of token ids");
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(err_json(e.what()).dump(), "application/json");
            return;
        }
        if (r->prompt.empty()) { res.status = 400; res.set_content(err_json("empty prompt").dump(), "application/json"); return; }
        r->max_tokens = std::min(r->max_tokens, eng.max_ctx() - (int)r->prompt.size() - 2);
        if (r->max_tokens <= 0) {
            res.status = 400;
            res.set_content(err_json("prompt (" + std::to_string(r->prompt.size()) + " tokens) does not fit the context").dump(), "application/json");
            return;
        }
        r->stop_tokens = {tk.eos};
        const int eot = tk.id("<|endoftext|>");
        if (eot >= 0 && eot != tk.eos) r->stop_tokens.push_back(eot);
        if (b.value("ignore_eos", false)) r->stop_tokens.clear();  // benchmark mode (llama-server compatible)
        if (!sched.submit(r)) {
            res.status = 503;
            res.set_content(err_json("server busy: request queue is full", "server_error").dump(), "application/json");
            return;
        }
        const std::string id = rand_id(chat_mode ? "chatcmpl-" : "cmpl-");
        const long created = (long)time(nullptr);
        const bool stream = b.value("stream", false);
        const bool include_usage = b.contains("stream_options") && b["stream_options"].is_object() &&
                                   b["stream_options"].value("include_usage", false);

        if (!stream) {
            std::string reasoning, content, reason;
            std::vector<chat::ToolCall> calls;
            drive(r, chat_mode, in_reasoning, tools, stops,
                  [&](const std::string& rd, const std::string& cd, const std::vector<chat::ToolCall>& cs, bool fin,
                      const std::string& why) {
                      reasoning += rd; content += cd;
                      calls.insert(calls.end(), cs.begin(), cs.end());
                      if (fin) reason = why;
                      return true;
                  });
            if (reason == "error") {
                res.status = 500;
                res.set_content(err_json(r->error, "server_error").dump(), "application/json");
                return;
            }
            const int ntok = (int)r->tokens.size();
            json out;
            if (chat_mode) {
                json msg = {{"role", "assistant"}, {"content", content}};
                if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
                if (!calls.empty()) msg["tool_calls"] = tool_calls_json(calls, 0, false);
                out = {{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", cfg.model_alias},
                       {"choices", json::array({{{"index", 0}, {"message", msg}, {"finish_reason", reason}}})}};
            } else {
                out = {{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", cfg.model_alias},
                       {"choices", json::array({{{"index", 0}, {"text", content}, {"finish_reason", reason}}})}};
            }
            out["usage"] = usage_json(*r, ntok);
            out["timings"] = timings_json(*r, ntok);
            res.set_content(out.dump(), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [this, r, id, created, chat_mode, in_reasoning, tools, stops,
                                                               include_usage](size_t, httplib::DataSink& sink) {
            auto send = [&](const json& j) {
                const std::string s = "data: " + j.dump() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            auto chunk = [&](const json& delta, const json& finish) {
                json c = {{"id", id}, {"object", chat_mode ? "chat.completion.chunk" : "text_completion"}, {"created", created},
                          {"model", cfg.model_alias}};
                if (chat_mode) c["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}});
                else c["choices"] = json::array({{{"index", 0}, {"text", delta.value("content", "")}, {"finish_reason", finish}}});
                return send(c);
            };
            bool ok = !chat_mode || chunk({{"role", "assistant"}, {"content", nullptr}}, nullptr);
            size_t ncalls = 0;
            if (ok)
                drive(r, chat_mode, in_reasoning, tools, stops,
                      [&](const std::string& rd, const std::string& cd, const std::vector<chat::ToolCall>& cs, bool fin,
                          const std::string& why) {
                          if (!ok) return false;
                          if (!rd.empty()) ok = ok && chunk({{"reasoning_content", rd}}, nullptr);
                          if (!cd.empty()) ok = ok && chunk({{"content", cd}}, nullptr);
                          if (!cs.empty()) {
                              ok = ok && chunk({{"tool_calls", tool_calls_json(cs, ncalls, true)}}, nullptr);
                              ncalls += cs.size();
                          }
                          if (fin && ok) {
                              if (why == "error") ok = send(err_json(r->error, "server_error"));
                              else {
                                  json last = {{"id", id}, {"object", chat_mode ? "chat.completion.chunk" : "text_completion"},
                                               {"created", created}, {"model", cfg.model_alias}};
                                  json ch = {{"index", 0}, {"finish_reason", why}};
                                  if (chat_mode) ch["delta"] = json::object();
                                  else ch["text"] = "";
                                  last["choices"] = json::array({ch});
                                  last["timings"] = timings_json(*r, (int)r->tokens.size());
                                  ok = send(last);
                                  if (ok && include_usage) {
                                      json u = {{"id", id}, {"object", chat_mode ? "chat.completion.chunk" : "text_completion"},
                                                {"created", created}, {"model", cfg.model_alias}, {"choices", json::array()},
                                                {"usage", usage_json(*r, (int)r->tokens.size())}};
                                      ok = send(u);
                                  }
                              }
                          }
                          return ok;  // false: client gone -> generation is cancelled
                      });
            if (ok) {
                const char* done = "data: [DONE]\n\n";
                sink.write(done, strlen(done));
            }
            sink.done();
            return true;
        });
    }
};

// ================================================================================================
// main
// ================================================================================================
static std::atomic<httplib::Server*> g_srv{nullptr};
static void on_signal(int) {
    if (auto* s = g_srv.load()) s->stop();
}

static void usage() {
    fprintf(stderr,
            "usage: liquid-server -m model.gguf [options]\n"
            "  --host H / --port P        listen address (default 127.0.0.1:8080)\n"
            "  -np N / --slots N          concurrent sequences (default 4, max %d)\n"
            "  -c N                       context per sequence (default 65536)\n"
            "  --kv q8|f16                KV cache type (default q8)\n"
            "  --alias NAME               model id reported by the API (default lfm3)\n"
            "  --max-tokens N             default max_tokens (default 4096)\n"
            "  --temp F --top-k N --top-p F --min-p F --repeat-penalty F --repeat-last-n N\n"
            "                             override the GGUF's sampling defaults (LFM3: 0.1 / 50 / off / off / 1.1 /\n"
            "                             whole context); per request: same names, plus repetition_penalty\n"
            "  --reasoning-preserve       keep reasoning of earlier assistant turns in the prompt\n"
            "  --busy-chunk N             prompt tokens per step while others decode (default 1024; 512 = smoother streams, slower new prompts)\n"
            "  --idle-chunk N             prompt tokens per step otherwise (default 4096)\n"
            "  --max-queue N              queued requests before 503 (default 64)\n"
            "  --cache-ram-mib N          RAM prompt-cache tier (default 4096, 0 = off)\n"
            "  --cache-disk-mib N         disk prompt-cache tier (default 16384, 0 = off)\n"
            "  --cache-dir DIR            disk tier directory (default $LIQUID_CACHE_DIR, else ~/.cache/liquid-engine);\n"
            "                             ~/.cache/liquid-engine is also the fallback when DIR is missing or dies\n"
            "  --threads N                HTTP worker threads (default 2*slots+16)\n"
            "  --tools-dir DIR            built-in tools web_search / web_fetch / parse_document, served at /tools like\n"
            "                             llama-server's (default <binary>/../tools/webtools)\n"
            "  --no-tools                 do not serve /tools\n"
            "  --node PATH                node binary for the tools (default node)\n"
            "  --tool-timeout S           per tool call (default 300)\n"
            "  --ui-dir DIR               llama.cpp web UI served at / (default <binary dir>/webui; make webui)\n"
            "  --no-ui                    API only\n",
            le::MAX_SLOTS);
}

int main(int argc, char** argv) {
    std::string model, host = "127.0.0.1", cache_dir;
    int port = 8080, slots = 4, ctx = 65536, threads = 0;
    long cache_ram = 4096, cache_disk = 16384;
    bool kv_q8 = true, tools_on = true;
    std::string tools_dir, node_bin = "node", ui_dir;
    bool ui_on = true;
    int tool_timeout = 300;
    ServerConfig scfg;
    SchedConfig sch;
    SamplingOverrides ov;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::stoi(next());
        else if (a == "-np" || a == "--slots") slots = std::stoi(next());
        else if (a == "-c") ctx = std::stoi(next());
        else if (a == "--kv") kv_q8 = next() != "f16";
        else if (a == "--alias") scfg.model_alias = next();
        else if (a == "--max-tokens") scfg.max_tokens = std::stoi(next());
        else if (a == "--temp") ov.temp = std::stof(next());
        else if (a == "--top-k") ov.top_k = std::stoi(next());
        else if (a == "--top-p") ov.top_p = std::stof(next());
        else if (a == "--min-p") ov.min_p = std::stof(next());
        else if (a == "--repeat-penalty") ov.penalty = std::stof(next());
        else if (a == "--repeat-last-n") ov.last_n = std::stoi(next());
        else if (a == "--reasoning-preserve") scfg.reasoning_preserve = true;
        else if (a == "--busy-chunk") sch.busy_chunk = std::stoi(next());
        else if (a == "--idle-chunk") sch.idle_chunk = std::stoi(next());
        else if (a == "--max-queue") sch.max_queue = std::stoi(next());
        else if (a == "--cache-ram-mib") cache_ram = std::stol(next());
        else if (a == "--cache-disk-mib") cache_disk = std::stol(next());
        else if (a == "--cache-dir") cache_dir = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--tools-dir") tools_dir = next();
        else if (a == "--no-tools") tools_on = false;
        else if (a == "--node") node_bin = next();
        else if (a == "--tool-timeout") tool_timeout = std::stoi(next());
        else if (a == "--ui-dir") ui_dir = next();
        else if (a == "--no-ui") ui_on = false;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    if (model.empty()) { usage(); return 1; }
    if (tools_dir.empty()) {
        std::error_code ec;
        auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
        tools_dir = (exe.parent_path().parent_path() / "tools" / "webtools").string();
    }
    // Disk-tier directories in order: --cache-dir, else $LIQUID_CACHE_DIR; then the default ~/.cache/liquid-engine,
    // used whenever the first one is missing or dies (PromptCache probes each with a test write).
    std::vector<std::string> cache_dirs;
    {
        const char* home = getenv("HOME");
        const std::string fallback = std::string(home ? home : ".") + "/.cache/liquid-engine";
        const char* env_dir = getenv("LIQUID_CACHE_DIR");
        if (cache_dir.empty() && env_dir && *env_dir) cache_dir = env_dir;
        if (!cache_dir.empty() && cache_dir != fallback) cache_dirs.push_back(cache_dir);
        cache_dirs.push_back(fallback);
    }
    try {
        std::unique_ptr<Tokenizer> tk;
        std::string chat_template;
        {
            gguf::File f(model);
            tk = std::make_unique<Tokenizer>(f);
            apply_gguf_sampling(f, scfg.sp);
            if (f.has("tokenizer.chat_template")) chat_template = f.get("tokenizer.chat_template").str;
        }
        ov.apply(scfg.sp);
        le::Engine eng(model, ctx, true, false, kv_q8, slots);
        std::unique_ptr<PromptCache> cache;
        if (cache_ram > 0 || cache_disk > 0)
            cache = std::make_unique<PromptCache>(eng, (size_t)cache_ram << 20, (size_t)cache_disk << 20, cache_dirs, model);
        Scheduler sched(eng, *tk, sch, cache.get());
        ApiServer api{eng, *tk, sched, scfg};

        httplib::Server srv;
        const int nthreads = threads > 0 ? threads : 2 * slots + 16;
        srv.new_task_queue = [nthreads] { return new httplib::ThreadPool(nthreads); };
        srv.set_read_timeout(600, 0);
        srv.set_write_timeout(600, 0);
        srv.set_keep_alive_max_count(1000);
        srv.set_keep_alive_timeout(30);
        srv.set_payload_max_length(256ull << 20);
        srv.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            std::string msg = "internal error";
            try { if (ep) std::rethrow_exception(ep); } catch (const std::exception& e) { msg = e.what(); } catch (...) {}
            res.status = 500;
            res.set_content(ApiServer::err_json(msg, "server_error").dump(), "application/json");
        });
        srv.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& r) { api.handle(q, r, true); });
        srv.Post("/chat/completions", [&](const httplib::Request& q, httplib::Response& r) { api.handle(q, r, true); });
        srv.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& r) { api.handle(q, r, false); });
        srv.Post("/completions", [&](const httplib::Request& q, httplib::Response& r) { api.handle(q, r, false); });
        auto models = [&](const httplib::Request&, httplib::Response& r) {
            json m = {{"object", "list"}, {"data", json::array({{{"id", scfg.model_alias}, {"object", "model"}, {"created", 0},
                                                                  {"owned_by", "liquid-engine"},
                                                                  {"meta", {{"n_ctx", eng.max_ctx()}, {"n_slots", eng.n_slots()}}}}})}};
            r.set_content(m.dump(), "application/json");
        };
        srv.Get("/v1/models", models);
        srv.Get("/models", models);
        srv.Get("/health", [](const httplib::Request&, httplib::Response& r) { r.set_content("{\"status\":\"ok\"}", "application/json"); });
        srv.Get("/props", [&](const httplib::Request&, httplib::Response& r) {
            const auto& sp = scfg.sp;
            auto r6 = [](float x) { return std::round((double)x * 1e6) / 1e6; };  // 0.1f -> 0.1, not 0.10000000149
            json params = {{"temperature", r6(sp.temperature)}, {"top_k", sp.top_k}, {"top_p", r6(sp.top_p)},
                           {"min_p", r6(sp.min_p)}, {"repeat_penalty", r6(sp.repeat_penalty)}, {"repeat_last_n", sp.repeat_last_n},
                           {"max_tokens", scfg.max_tokens}};
            json dgs = params;  // flat keys (liquid-server's own) plus llama-server's {n_ctx, params} read by the web UI
            dgs["n_ctx"] = eng.max_ctx();
            dgs["params"] = params;
            json p = {{"model", scfg.model_alias}, {"n_ctx", eng.max_ctx()}, {"n_slots", eng.n_slots()},
                      {"kv_cache", eng.kv_q8() ? "q8_0" : "f16"}, {"reasoning_preserve", scfg.reasoning_preserve},
                      {"default_generation_settings", dgs},
                      // llama-server /props fields used by the llama.cpp web UI
                      {"model_alias", scfg.model_alias}, {"model_path", model}, {"total_slots", eng.n_slots()},
                      {"modalities", {{"vision", false}, {"audio", false}, {"video", false}}},
                      {"chat_template", chat_template}, {"build_info", "liquid-engine"},
                      {"cors_proxy_enabled", false}, {"ui_settings", json::object()}};
            r.set_content(p.dump(), "application/json");
        });
        srv.Get("/slots", [&](const httplib::Request&, httplib::Response& r) { r.set_content(sched.stats().dump(), "application/json"); });
        // Built-in tools, llama-server's /tools contract (see web_tools.h); the web UI lists and calls them like get_info
        std::unique_ptr<WebTools> wtools;
        if (tools_on) {
            wtools = std::make_unique<WebTools>(tools_dir, node_bin, tool_timeout);
            if (wtools->enabled()) {
                std::string names;
                for (const auto& t : wtools->list()) names += (names.empty() ? "" : ", ") + t.value("tool", std::string());
                Scheduler::log("tools: %s (%s)", names.c_str(), tools_dir.c_str());
            } else {
                Scheduler::log("tools disabled: %s", wtools->error().c_str());
            }
        }
        auto tools_off = [&](httplib::Response& r) {
            if (wtools && wtools->enabled()) return false;
            r.status = 403;  // what llama-server answers without --tools; the UI then hides server tools
            r.set_content(ApiServer::err_json("server tools are disabled", "permission_error").dump(), "application/json");
            return true;
        };
        srv.Get("/tools", [&](const httplib::Request&, httplib::Response& r) {
            if (!tools_off(r)) r.set_content(wtools->list().dump(), "application/json");
        });
        srv.Post("/tools", [&](const httplib::Request& q, httplib::Response& r) {
            if (tools_off(r)) return;
            json b = json::parse(q.body, nullptr, false);
            if (!b.is_object() || !b.contains("tool") || !b["tool"].is_string()) {
                r.status = 400;
                r.set_content(ApiServer::err_json("expected {\"tool\": name, \"params\": {...}}", "invalid_request_error").dump(),
                              "application/json");
                return;
            }
            std::string name = b["tool"].get<std::string>();
            if (!wtools->has(name) || b.value("stream", false)) {
                r.status = 404;
                std::string msg = wtools->has(name) ? "tool \"" + name + "\" does not support stream = true"
                                                    : "unknown tool \"" + name + "\"";
                r.set_content(ApiServer::err_json(msg, "invalid_request_error").dump(), "application/json");
                return;
            }
            json params = b.contains("params") && b["params"].is_object() ? b["params"] : json::object();
            params.erase("cwd");  // like llama-server: the working directory comes only from the x-tool-cwd header
            if (q.has_header("x-tool-cwd")) params["cwd"] = q.get_header_value("x-tool-cwd");
            auto t0 = std::chrono::steady_clock::now();
            json out = wtools->call(name, params, q.is_connection_closed);
            Scheduler::log("tool %s: %s in %.0f ms", name.c_str(), out.contains("error") ? "error" : "ok",
                           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            r.set_content(out.dump(), "application/json");
        });
        srv.Post("/tokenize", [&](const httplib::Request& q, httplib::Response& r) {
            json b = json::parse(q.body, nullptr, false);
            json out = {{"tokens", b.is_object() && b.contains("content") ? json(tk->encode(b["content"].get<std::string>())) : json::array()}};
            r.set_content(out.dump(), "application/json");
        });

        // llama.cpp web UI (prebuilt bundle, `make webui`). Hash routing + relative asset paths: serving the directory
        // at / is all it needs. Cache headers as llama-server sends them.
        if (ui_on) {
            if (ui_dir.empty()) {
                std::error_code ec;
                ui_dir = (std::filesystem::read_symlink("/proc/self/exe", ec).parent_path() / "webui").string();
            }
            if (std::filesystem::exists(std::filesystem::path(ui_dir) / "index.html") && srv.set_mount_point("/", ui_dir)) {
                srv.set_file_extension_and_mimetype_mapping("webmanifest", "application/manifest+json");
                srv.set_file_extension_and_mimetype_mapping("mjs", "application/javascript");
                srv.set_file_request_handler([](const httplib::Request& q, httplib::Response& r) {
                    const std::string& p = q.path;
                    const bool revalidate = p == "/" || p == "/index.html" || p == "/service-worker.js" || p == "/sw.js" ||
                                            p == "/manifest.webmanifest" || p == "/build.json" || p == "/_app/version.json";
                    r.set_header("Cache-Control", revalidate ? "no-cache" : "public, max-age=31536000, immutable");
                    if (p == "/" || p == "/index.html") {
                        r.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                        r.set_header("Cross-Origin-Opener-Policy", "same-origin");
                    }
                });
                Scheduler::log("web UI: %s", ui_dir.c_str());
            } else {
                Scheduler::log("web UI not found in %s (run: make webui), serving the API only", ui_dir.c_str());
            }
        }
        // llama-server endpoints the UI probes; answering them keeps its console clean
        srv.Post("/v1/streams/lookup", [](const httplib::Request&, httplib::Response& r) { r.set_content("[]", "application/json"); });

        g_srv = &srv;
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        Scheduler::log("listening on http://%s:%d (%d slots x %d ctx, %s KV, %d HTTP threads)", host.c_str(), port, slots,
                       eng.max_ctx(), kv_q8 ? "q8" : "f16", nthreads);
        if (!srv.listen(host, port)) {
            fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
            return 1;
        }
        g_srv = nullptr;
        Scheduler::log("shutting down");
        sched.stop();
        if (cache) cache->flush();
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
