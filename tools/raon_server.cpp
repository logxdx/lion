// raon-server: OpenAI-compatible HTTP server for Raon-1 nano on raon-engine.
//
//   raon-server -m model.gguf [--slots 4] [--ctx 131072] [--port 8080] [--host 127.0.0.1]
//
// One scheduler thread owns the engine (all CUDA work happens there) and runs continuous batching: each iteration
// runs at most one prefill chunk of one request, then one batched decode step over every generating request. HTTP
// threads do templating, tokenization, detokenization, stop strings, output parsing and SSE.
//
// Prompt reuse: a slot keeps the tokens its state covers. A new request takes the idle slot with the longest reusable
// prefix: either the slot's whole history, or the checkpoint of the recurrent state saved where its last chat prompt's
// generation prefix begins (which is where the next turn of the same conversation diverges).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>

#include "chat.h"
#include "engine.h"
#include "httplib/httplib.h"
#include "tokenizer.h"

using namespace raon;
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
static std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

struct Options {
    std::string model, alias = "raon-1-nano", host = "127.0.0.1";
    int port = 8080, slots = 4, ctx = 131072, threads = 32;
    int chunk_busy = 2048;    // prefill tokens per iteration while other requests are generating
    float prefill_share = 0.5f;   // share of GPU time given to prefill while other requests are generating
    int chunk_idle = 8192;    // ... while none is (the engine splits it into its own chunks)
    int checkpoints = 4;      // recurrent-state checkpoints per slot (~19 MB each); 0 disables
    SamplingParams defaults;  // generation_config.json of Raon-1 nano: temperature 0.1, presence_penalty 1.5
};

struct TokEvent {
    int tok;
    float logprob;
    std::vector<std::pair<int, float>> top;
};

struct Req {
    std::string id;
    std::vector<int> prompt;
    int ckpt_pos = -1;        // save the recurrent state after this many prompt tokens (-1: no checkpoint)
    SamplingParams sp;
    std::vector<std::pair<int, float>> bias;
    uint64_t seed = 0;
    int max_tokens = 0;
    Clock::time_point t_arrive = Clock::now();
    std::atomic<bool> cancel{false};
    std::string cancel_reason = "stop";
    // scheduler -> HTTP thread
    std::mutex mu;
    std::condition_variable cv;
    std::deque<TokEvent> ev;
    bool done = false;
    std::string finish, error;
    int n_reused = 0, n_gen = 0, slot = -1;
    double prompt_ms = 0, gen_ms = 0, queue_ms = 0;
};
using ReqP = std::shared_ptr<Req>;

// ---------------------------------------------------------------- scheduler

class Scheduler {
public:
    explicit Scheduler(const Options& o, const Tokenizer& tok) : opt_(o), tok_(tok) {}
    void start() {
        th_ = std::thread([this] { run(); });
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return ready_ || !load_error_.empty(); });
        if (!load_error_.empty()) throw std::runtime_error(load_error_);
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    void submit(const ReqP& r) {
        { std::lock_guard<std::mutex> lk(mu_); queue_.push_back(r); }
        cv_.notify_all();
    }
    void wake() { cv_.notify_all(); }
    json slots_json() {
        std::lock_guard<std::mutex> lk(info_mu_);
        return info_;
    }
    int ctx() const { return opt_.ctx; }

private:
    enum St { IDLE, PREFILL, DECODE };
    struct Slot {
        int id = 0;
        St st = IDLE;
        ReqP req;
        std::vector<int> cached;      // tokens the state covers (positions 0..n-1 of the latent cache)
        struct Ckpt { std::vector<int> toks; bool valid = false; uint64_t used = 0; };
        std::vector<Ckpt> ck;         // saved recurrent states and the tokens each covers
        int last = -1;                // sampled token not yet fed
        std::unordered_map<int, int> counts;
        std::vector<std::pair<int, int>> counts_v;
        std::mt19937_64 rng;
        uint64_t seq = 0, last_use = 0;
        Clock::time_point t0, t1;
    };

    static bool is_prefix(const std::vector<int>& a, const std::vector<int>& b) {   // a is a prefix of b
        return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
    }

    void publish() {
        json a = json::array();
        for (const Slot& s : slots_)
            a.push_back({{"id", s.id}, {"state", s.st == IDLE ? "idle" : s.st == PREFILL ? "prefill" : "generating"},
                         {"cached_tokens", s.cached.size()}, {"checkpoints", [&] { json c = json::array(); for (const auto& k : s.ck) if (k.valid) c.push_back(k.toks.size()); return c; }()},
                         {"request", s.req ? s.req->id : ""}});
        std::lock_guard<std::mutex> lk(info_mu_);
        info_ = a;
    }

    void finish(Slot& s, const std::string& reason) {
        ReqP r = s.req;
        {
            std::lock_guard<std::mutex> lk(r->mu);
            r->done = true;
            r->finish = reason;
            if (s.st == DECODE) r->gen_ms = ms_since(s.t1);
        }
        r->cv.notify_all();
        fprintf(stderr, "raon-server: %s slot %d: prompt %zu (reused %d) %.0f ms%s, generated %d in %.0f ms (%.1f tok/s), %s\n",
                r->id.c_str(), s.id, r->prompt.size(), r->n_reused, r->prompt_ms,
                r->prompt_ms > 0 ? (" (" + std::to_string((int)((r->prompt.size() - r->n_reused) / (r->prompt_ms / 1000))) + " tok/s)").c_str() : "",
                r->n_gen, r->gen_ms, r->gen_ms > 0 ? r->n_gen / (r->gen_ms / 1000) : 0.0, reason.c_str());
        s.req.reset();
        s.st = IDLE;
        s.last_use = ++use_clock_;
    }

    void emit(Slot& s, const SampleOut& o) {
        Req& r = *s.req;
        const bool eos = o.token == tok_.eos() || o.token == tok_.pad();
        if (!eos) {
            {
                std::lock_guard<std::mutex> lk(r.mu);
                r.ev.push_back({o.token, o.logprob, o.top});
                r.n_gen++;
            }
            r.cv.notify_all();
            s.counts[o.token]++;
        }
        s.last = o.token;
        if (eos) { finish(s, "stop"); return; }
        if (r.n_gen >= r.max_tokens || (int)s.cached.size() + 1 >= opt_.ctx) finish(s, "length");
    }

    SampleIn sample_in(Slot& s) {
        SampleIn in;
        in.p = s.req->sp;
        const SamplingParams& p = in.p;
        if (p.presence_penalty != 0 || p.frequency_penalty != 0 || p.repetition_penalty != 1) {
            s.counts_v.assign(s.counts.begin(), s.counts.end());
            in.counts = &s.counts_v;
        }
        if (!s.req->bias.empty()) in.bias = &s.req->bias;
        in.u = std::uniform_real_distribution<double>(0.0, 1.0)(s.rng);
        return in;
    }

    void admit() {
        while (true) {
            ReqP r;
            {
                std::lock_guard<std::mutex> lk(mu_);
                while (!queue_.empty() && queue_.front()->cancel) {
                    ReqP c = queue_.front();
                    queue_.pop_front();
                    { std::lock_guard<std::mutex> lk2(c->mu); c->done = true; c->finish = "cancelled"; }
                    c->cv.notify_all();
                }
                if (queue_.empty()) return;
                r = queue_.front();
            }
            // the idle slot with the longest reusable prefix, else the least recently used idle slot
            int best = -1, best_reuse = -1, best_ck = -1, lru = -1;
            for (Slot& s : slots_) {
                if (s.st != IDLE) continue;
                if (lru < 0 || s.last_use < slots_[lru].last_use) lru = s.id;
                int reuse = 0, ck = -1;
                if (s.cached.size() < r->prompt.size() && is_prefix(s.cached, r->prompt)) reuse = (int)s.cached.size();
                for (int k = 0; k < (int)s.ck.size(); k++) {
                    const auto& c = s.ck[k];
                    if (c.valid && (int)c.toks.size() > reuse && c.toks.size() < r->prompt.size() && is_prefix(c.toks, r->prompt)) {
                        reuse = (int)c.toks.size();
                        ck = k;
                    }
                }
                if (reuse > best_reuse) { best = s.id; best_reuse = reuse; best_ck = ck; }
            }
            if (lru < 0) return;   // no idle slot
            {
                std::lock_guard<std::mutex> lk(mu_);
                queue_.pop_front();
            }
            Slot& s = slots_[best_reuse > 0 ? best : lru];
            const int reuse = best_reuse > 0 ? best_reuse : 0;
            if (reuse == 0) {
                eng_->reset(s.id);
                s.cached.clear();
            } else if (best_ck >= 0) {
                eng_->load_state(s.id, best_ck);
                s.cached = s.ck[best_ck].toks;
                s.ck[best_ck].used = ++use_clock_;
            }
            // a checkpoint stays valid while the latent-cache positions it covers are unchanged: the tokens it covers
            // must be a prefix of what this request keeps (the reused part plus the new prompt)
            for (auto& c : s.ck)
                if (c.valid && !is_prefix(c.toks, r->prompt)) {
                    const size_t lcp = std::mismatch(c.toks.begin(), c.toks.end(), r->prompt.begin(), r->prompt.begin() + std::min(c.toks.size(), r->prompt.size())).first - c.toks.begin();
                    if ((int)lcp < (int)c.toks.size()) c.valid = false;
                }
            s.req = r;
            s.st = PREFILL;
            s.counts.clear();
            s.rng.seed(r->seed);
            s.seq = ++seq_;
            s.t0 = Clock::now();
            r->n_reused = reuse;
            r->slot = s.id;
            r->queue_ms = ms_since(r->t_arrive);
        }
    }

    void run() {
        try {
            eng_ = std::make_unique<Engine>(opt_.model, opt_.slots, opt_.ctx, true);
            if (opt_.checkpoints > 0) eng_->set_checkpoints(opt_.checkpoints);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(mu_);
            load_error_ = e.what();
            cv_.notify_all();
            return;
        }
        slots_.resize(opt_.slots);
        for (int i = 0; i < opt_.slots; i++) {
            slots_[i].id = i;
            slots_[i].ck.resize(opt_.checkpoints);
        }
        publish();
        {
            std::lock_guard<std::mutex> lk(mu_);
            ready_ = true;
        }
        cv_.notify_all();
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] {
                    if (stop_) return true;
                    if (!queue_.empty()) return true;
                    for (const Slot& s : slots_) if (s.st != IDLE) return true;
                    return false;
                });
                if (stop_) return;
            }
            try {
                iterate();
            } catch (const std::exception& e) {
                fprintf(stderr, "raon-server: engine error: %s\n", e.what());
                for (Slot& s : slots_)
                    if (s.req) {
                        { std::lock_guard<std::mutex> lk(s.req->mu); s.req->error = e.what(); }
                        finish(s, "error");
                        s.cached.clear();
                        for (auto& c : s.ck) c.valid = false;
                    }
            }
            publish();
        }
    }

    void iterate() {
        for (Slot& s : slots_)
            if (s.req && s.req->cancel) finish(s, s.req->cancel_reason);
        admit();
        // one prefill chunk: the oldest request still in prefill. While others are generating, prefill gets
        // `prefill_share` of the time: after a chunk that took t, decode-only steps run for t * (1 - share) / share.
        Slot* pf = nullptr;
        bool decoding = false;
        for (Slot& s : slots_) {
            if (s.st == PREFILL && (!pf || s.seq < pf->seq)) pf = &s;
            decoding |= s.st == DECODE;
        }
        if (pf && decoding && Clock::now() < next_prefill_) pf = nullptr;
        if (pf) {
            const auto tp0 = Clock::now();
            Slot& s = *pf;
            Req& r = *s.req;
            const int n0 = (int)s.cached.size(), P = (int)r.prompt.size();
            int end = std::min(P, n0 + (decoding ? opt_.chunk_busy : opt_.chunk_idle));
            const bool ck = !s.ck.empty() && r.ckpt_pos > n0 && r.ckpt_pos < P;
            if (ck && r.ckpt_pos < end) end = r.ckpt_pos;
            std::vector<Row> rows;
            rows.reserve(end - n0);
            for (int i = n0; i < end; i++) rows.push_back({s.id, i, r.prompt[i], i == P - 1});
            std::vector<SampleIn> in;
            if (end == P) in.push_back(sample_in(s));
            const auto out = eng_->step(rows, in);
            s.cached.insert(s.cached.end(), r.prompt.begin() + n0, r.prompt.begin() + end);
            if (ck && end == r.ckpt_pos) {   // into a free or the least recently used checkpoint
                int k = 0;
                for (int j = 0; j < (int)s.ck.size(); j++) {
                    if (!s.ck[j].valid) { k = j; break; }
                    if (s.ck[j].used < s.ck[k].used) k = j;
                }
                eng_->save_state(s.id, k);
                s.ck[k].toks.assign(r.prompt.begin(), r.prompt.begin() + end);
                s.ck[k].valid = true;
                s.ck[k].used = ++use_clock_;
            }
            if (end == P) {
                r.prompt_ms = ms_since(s.t0);
                s.t1 = Clock::now();
                s.st = DECODE;
                emit(s, out[0]);
            }
            const float sh = std::min(1.f, std::max(0.05f, opt_.prefill_share));
            next_prefill_ = Clock::now() + std::chrono::duration_cast<Clock::duration>((Clock::now() - tp0) * ((1.f - sh) / sh));
        }
        // one decode step over every generating request
        std::vector<Row> rows;
        std::vector<SampleIn> in;
        std::vector<Slot*> who;
        for (Slot& s : slots_) {
            if (s.st != DECODE) continue;
            if (s.req->cancel) { finish(s, s.req->cancel_reason); continue; }
            rows.push_back({s.id, (int)s.cached.size(), s.last, true});
            in.push_back(sample_in(s));
            who.push_back(&s);
        }
        if (rows.empty()) return;
        const auto out = eng_->step(rows, in);
        for (size_t i = 0; i < who.size(); i++) {
            Slot& s = *who[i];
            s.cached.push_back(s.last);
            emit(s, out[i]);
        }
    }

    Options opt_;
    const Tokenizer& tok_;
    std::unique_ptr<Engine> eng_;
    std::vector<Slot> slots_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<ReqP> queue_;
    bool stop_ = false, ready_ = false;
    std::string load_error_;
    uint64_t seq_ = 0, use_clock_ = 0;
    Clock::time_point next_prefill_ = Clock::now();
    std::mutex info_mu_;
    json info_ = json::array();
};

// ---------------------------------------------------------------- request parsing

struct BadRequest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

static std::string new_id(const char* prefix) {
    static std::atomic<uint64_t> n{0};
    static const uint64_t salt = std::random_device{}();
    char b[64];
    snprintf(b, sizeof b, "%s-%08llx%06llx", prefix, (unsigned long long)(salt & 0xffffffffu), (unsigned long long)++n);
    return b;
}

static void parse_sampling(const json& b, const Options& opt, Req& r) {
    SamplingParams p = opt.defaults;
    auto num = [&](const char* k, float& v) {
        if (b.contains(k) && !b[k].is_null()) {
            if (!b[k].is_number()) throw BadRequest(std::string(k) + " must be a number");
            v = b[k].get<float>();
        }
    };
    num("temperature", p.temperature);
    num("top_p", p.top_p);
    num("min_p", p.min_p);
    num("presence_penalty", p.presence_penalty);
    num("frequency_penalty", p.frequency_penalty);
    num("repetition_penalty", p.repetition_penalty);
    if (b.contains("top_k") && b["top_k"].is_number()) p.top_k = std::max(0, b["top_k"].get<int>());
    if (p.top_p <= 0.f || p.top_p > 1.f) throw BadRequest("top_p must be in (0, 1]");
    if (p.repetition_penalty <= 0.f) throw BadRequest("repetition_penalty must be > 0");
    p.n_logprobs = 0;
    r.sp = p;
    if (b.contains("logit_bias") && b["logit_bias"].is_object())
        for (auto it = b["logit_bias"].begin(); it != b["logit_bias"].end(); ++it)
            r.bias.push_back({std::stoi(it.key()), it.value().get<float>()});
    r.seed = b.contains("seed") && b["seed"].is_number_integer() ? b["seed"].get<uint64_t>() : std::random_device{}();
}

static std::vector<std::string> parse_stop(const json& b) {
    std::vector<std::string> st;
    if (!b.contains("stop") || b["stop"].is_null()) return st;
    if (b["stop"].is_string()) st.push_back(b["stop"].get<std::string>());
    else if (b["stop"].is_array())
        for (const auto& x : b["stop"]) if (x.is_string() && !x.get<std::string>().empty()) st.push_back(x.get<std::string>());
    return st;
}

// ---------------------------------------------------------------- response streaming

// Consumes token events of one request: detokenizes, applies stop strings, parses chat output. Used by both the
// streaming and the non-streaming paths.
class Collector {
public:
    Collector(const Tokenizer& tok, ReqP r, std::vector<std::string> stops, bool chat, bool thinking, const json& tools)
        : tok_(tok), r_(std::move(r)), stops_(std::move(stops)), chat_(chat), parser_(thinking, tools) {}

    struct Out {
        chat::OutputParser::Delta d;
        std::string text;                          // completions: raw text delta
        std::vector<TokEvent> toks;
        bool done = false;
        std::string finish;
    };

    // Waits for new events (up to `wait`), returns what can be shown.
    Out next(std::chrono::milliseconds wait) {
        Out o;
        std::deque<TokEvent> ev;
        bool done = false;
        std::string fin, err;
        {
            std::unique_lock<std::mutex> lk(r_->mu);
            r_->cv.wait_for(lk, wait, [&] { return !r_->ev.empty() || r_->done; });
            ev.swap(r_->ev);
            done = r_->done;
            fin = r_->finish;
            err = r_->error;
        }
        for (const TokEvent& e : ev) {
            if (stopped_) break;
            o.toks.push_back(e);
            feed(u8_.push(tok_.piece(e.tok)), o);
        }
        if (stopped_) {
            if (!finished_) {
                finished_ = true;
                r_->cancel_reason = "stop";
                r_->cancel = true;
                flush(o);
                o.done = true;
                o.finish = "stop";
            }
            return o;
        }
        if (done) {
            finished_ = true;
            feed(u8_.flush(), o);
            held_out(o, held_);
            held_.clear();
            flush(o);
            o.done = true;
            o.finish = err.empty() ? fin : "error";
            error_ = err;
        }
        return o;
    }
    const std::string& error() const { return error_; }
    const chat::OutputParser& parser() const { return parser_; }

private:
    void feed(const std::string& text, Out& o) {
        if (text.empty()) return;
        held_ += text;
        for (const std::string& s : stops_) {
            const size_t k = held_.find(s);
            if (k != std::string::npos) {
                held_out(o, held_.substr(0, k));
                held_.clear();
                stopped_ = true;
                return;
            }
        }
        size_t keep = 0;   // longest suffix that may start a stop string
        for (const std::string& s : stops_)
            for (size_t k = std::min(held_.size(), s.size() - 1); k > keep; k--)
                if (held_.compare(held_.size() - k, k, s, 0, k) == 0) { keep = k; break; }
        held_out(o, held_.substr(0, held_.size() - keep));
        held_.erase(0, held_.size() - keep);
    }
    void held_out(Out& o, const std::string& text) {
        if (text.empty()) return;
        if (!chat_) { o.text += text; return; }
        auto d = parser_.push(text);
        o.d.reasoning += d.reasoning;
        o.d.content += d.content;
        for (auto& c : d.calls) o.d.calls.push_back(c);
    }
    void flush(Out& o) {
        if (!chat_) return;
        auto d = parser_.finish();
        o.d.reasoning += d.reasoning;
        o.d.content += d.content;
        for (auto& c : d.calls) o.d.calls.push_back(c);
    }

    const Tokenizer& tok_;
    ReqP r_;
    std::vector<std::string> stops_;
    bool chat_;
    chat::OutputParser parser_;
    Utf8Stream u8_;
    std::string held_, error_;
    bool stopped_ = false, finished_ = false;
};

static json bytes_of(const std::string& s) {
    json a = json::array();
    for (unsigned char c : s) a.push_back((int)c);
    return a;
}

static json logprobs_json(const Tokenizer& tok, const std::vector<TokEvent>& toks, int n_top) {
    json content = json::array();
    for (const TokEvent& e : toks) {
        json tops = json::array();
        for (int j = 0; j < (int)e.top.size() && j < n_top; j++)
            tops.push_back({{"token", tok.piece(e.top[j].first)}, {"logprob", e.top[j].second}, {"bytes", bytes_of(tok.piece(e.top[j].first))}});
        content.push_back({{"token", tok.piece(e.tok)}, {"logprob", e.logprob}, {"bytes", bytes_of(tok.piece(e.tok))}, {"top_logprobs", tops}});
    }
    return {{"content", content}};
}

static json tool_calls_json(const std::vector<chat::ToolCall>& calls, int base, std::vector<std::string>& ids, bool with_index) {
    json a = json::array();
    for (size_t i = 0; i < calls.size(); i++) {
        ids.push_back(new_id("call"));
        json c;
        if (with_index) c["index"] = base + (int)i;
        c["id"] = ids.back();
        c["type"] = "function";
        c["function"] = {{"name", calls[i].name}, {"arguments", chat::tojson(calls[i].arguments)}};
        a.push_back(c);
    }
    return a;
}

static json usage_json(const Req& r) {
    return {{"prompt_tokens", r.prompt.size()}, {"completion_tokens", r.n_gen}, {"total_tokens", r.prompt.size() + r.n_gen},
            {"prompt_tokens_details", {{"cached_tokens", r.n_reused}}}};
}
static json timings_json(const Req& r) {
    const int pn = (int)r.prompt.size() - r.n_reused;
    return {{"queue_ms", r.queue_ms}, {"prompt_n", pn}, {"prompt_ms", r.prompt_ms},
            {"prompt_per_second", r.prompt_ms > 0 ? pn / (r.prompt_ms / 1000) : 0.0}, {"cached_n", r.n_reused},
            {"predicted_n", r.n_gen}, {"predicted_ms", r.gen_ms},
            {"predicted_per_second", r.gen_ms > 0 ? r.n_gen / (r.gen_ms / 1000) : 0.0}, {"slot", r.slot}};
}

// ---------------------------------------------------------------- main

static httplib::Server* g_svr = nullptr;

static void usage() {
    fprintf(stderr,
            "usage: raon-server -m model.gguf [options]\n"
            "  --slots N            concurrent sequences (default 4)\n"
            "  --ctx N              context per sequence in tokens (default 131072)\n"
            "  --host H / --port P  listen address (default 127.0.0.1:8080)\n"
            "  --alias NAME         model name reported by the API (default raon-1-nano)\n"
            "  --chunk-busy N       prefill tokens per iteration while others generate (default 2048)\n"
            "  --prefill-share F    share of GPU time for prefill while others generate (default 0.5)\n"
            "  --checkpoints N      recurrent-state checkpoints per slot for prompt reuse (default 4, ~19 MB each; 0 = off)\n"
            "  --temp T --top-p P --top-k K --min-p M --presence-penalty X --frequency-penalty X --repeat-penalty X\n"
            "                       default sampling (default: temp 0.1, presence 1.5 from generation_config.json)\n"
            "  --threads N          HTTP worker threads (default 32)\n");
}

int main(int argc, char** argv) {
    Options opt;
    opt.defaults.temperature = 0.1f;
    opt.defaults.presence_penalty = 1.5f;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") opt.model = next();
        else if (a == "--slots") opt.slots = std::stoi(next());
        else if (a == "--ctx") opt.ctx = std::stoi(next());
        else if (a == "--host") opt.host = next();
        else if (a == "--port") opt.port = std::stoi(next());
        else if (a == "--alias") opt.alias = next();
        else if (a == "--chunk-busy") opt.chunk_busy = std::stoi(next());
        else if (a == "--prefill-share") opt.prefill_share = std::stof(next());
        else if (a == "--checkpoints") opt.checkpoints = std::max(0, std::stoi(next()));
        else if (a == "--threads") opt.threads = std::stoi(next());
        else if (a == "--temp") opt.defaults.temperature = std::stof(next());
        else if (a == "--top-p") opt.defaults.top_p = std::stof(next());
        else if (a == "--top-k") opt.defaults.top_k = std::stoi(next());
        else if (a == "--min-p") opt.defaults.min_p = std::stof(next());
        else if (a == "--presence-penalty") opt.defaults.presence_penalty = std::stof(next());
        else if (a == "--frequency-penalty") opt.defaults.frequency_penalty = std::stof(next());
        else if (a == "--repeat-penalty") opt.defaults.repetition_penalty = std::stof(next());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(); return 1; }
    }
    if (opt.model.empty()) { usage(); return 1; }

    Tokenizer tok(opt.model);
    Scheduler sched(opt, tok);
    sched.start();
    const long created = (long)time(nullptr);

    httplib::Server svr;
    g_svr = &svr;
    svr.new_task_queue = [&] { return new httplib::ThreadPool(opt.threads); };
    auto send_json = [](httplib::Response& res, const json& j, int status = 200) {
        res.status = status;
        res.set_content(dump(j), "application/json");
    };
    auto send_error = [&](httplib::Response& res, int status, const std::string& msg) {
        send_json(res, {{"error", {{"message", msg}, {"type", status == 400 ? "invalid_request_error" : "server_error"}, {"code", status}}}}, status);
    };

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) { send_json(res, {{"status", "ok"}}); });
    svr.Get("/slots", [&](const httplib::Request&, httplib::Response& res) { send_json(res, sched.slots_json()); });
    auto models = [&](const httplib::Request&, httplib::Response& res) {
        send_json(res, {{"object", "list"},
                        {"data", json::array({{{"id", opt.alias}, {"object", "model"}, {"created", created}, {"owned_by", "raon-engine"},
                                               {"meta", {{"n_ctx", opt.ctx}, {"n_slots", opt.slots}, {"n_vocab", tok.n_vocab()}}}}})}});
    };
    svr.Get("/v1/models", models);
    svr.Get("/models", models);

    svr.Post("/tokenize", [&](const httplib::Request& req, httplib::Response& res) {
        json b = json::parse(req.body, nullptr, false);
        if (b.is_discarded() || !b.contains("content") || !b["content"].is_string()) return send_error(res, 400, "expected {\"content\": string}");
        send_json(res, {{"tokens", tok.encode(b["content"].get<std::string>(), b.value("parse_special", true))}});
    });
    svr.Post("/detokenize", [&](const httplib::Request& req, httplib::Response& res) {
        json b = json::parse(req.body, nullptr, false);
        if (b.is_discarded() || !b.contains("tokens") || !b["tokens"].is_array()) return send_error(res, 400, "expected {\"tokens\": [ids]}");
        send_json(res, {{"content", tok.decode(b["tokens"].get<std::vector<int>>())}});
    });

    // Builds a request; throws BadRequest.
    struct Prepared {
        ReqP r;
        std::vector<std::string> stops;
        bool stream = false, include_usage = false, thinking = true, chat = true;
        int n_top = 0;
        bool logprobs = false;
        json tools;
    };
    auto prepare = [&](const json& b, bool chat_ep) -> Prepared {
        Prepared p;
        p.chat = chat_ep;
        p.r = std::make_shared<Req>();
        Req& r = *p.r;
        r.id = new_id(chat_ep ? "chatcmpl" : "cmpl");
        if (b.contains("n") && b["n"].is_number() && b["n"].get<int>() != 1) throw BadRequest("only n = 1 is supported");
        if (b.contains("response_format") && b["response_format"].is_object() &&
            b["response_format"].value("type", "text") != "text")
            throw BadRequest("response_format other than text is not supported");
        if (chat_ep) {
            if (!b.contains("messages")) throw BadRequest("missing messages");
            chat::TemplateOptions to;
            json kw = b.value("chat_template_kwargs", json::object());
            if (b.contains("enable_thinking") && b["enable_thinking"].is_boolean()) kw["enable_thinking"] = b["enable_thinking"];
            to.enable_thinking = kw.value("enable_thinking", true);
            to.truncate_history_thinking = kw.value("truncate_history_thinking", false);
            p.thinking = to.enable_thinking;
            p.tools = b.value("tools", json());
            if (b.contains("tool_choice") && b["tool_choice"].is_string() && b["tool_choice"] == "none") p.tools = json();
            std::string prompt;
            try {
                prompt = chat::render(b["messages"], p.tools, to);
                r.prompt = tok.encode(prompt);
                // checkpoint where the generation prefix starts: the next turn of this conversation diverges there
                to.add_generation_prompt = false;
                const std::vector<int> head = tok.encode(chat::render(b["messages"], p.tools, to));
                if (head.size() < r.prompt.size() && std::equal(head.begin(), head.end(), r.prompt.begin())) r.ckpt_pos = (int)head.size();
            } catch (const std::invalid_argument& e) {
                throw BadRequest(e.what());
            } catch (const json::exception& e) {
                throw BadRequest(std::string("bad messages: ") + e.what());
            }
        } else {
            if (!b.contains("prompt")) throw BadRequest("missing prompt");
            const json& pr = b["prompt"];
            if (pr.is_string()) r.prompt = tok.encode(pr.get<std::string>());
            else if (pr.is_array() && !pr.empty() && pr[0].is_number_integer()) r.prompt = pr.get<std::vector<int>>();
            else if (pr.is_array() && pr.size() == 1 && pr[0].is_string()) r.prompt = tok.encode(pr[0].get<std::string>());
            else throw BadRequest("prompt must be a string or an array of token ids");
            for (int t : r.prompt) if (t < 0 || t >= tok.n_vocab()) throw BadRequest("token id out of range");
        }
        if (r.prompt.empty()) throw BadRequest("empty prompt");
        if ((int)r.prompt.size() >= sched.ctx()) throw BadRequest("prompt of " + std::to_string(r.prompt.size()) + " tokens does not fit the context of " + std::to_string(sched.ctx()));
        parse_sampling(b, opt, r);
        int mt = sched.ctx() - (int)r.prompt.size();
        for (const char* k : {"max_completion_tokens", "max_tokens", "n_predict"})
            if (b.contains(k) && b[k].is_number_integer() && b[k].get<int>() > 0) { mt = std::min(mt, b[k].get<int>()); break; }
        r.max_tokens = mt;
        p.stops = parse_stop(b);
        p.stream = b.value("stream", false);
        p.include_usage = b.contains("stream_options") && b["stream_options"].is_object() && b["stream_options"].value("include_usage", false);
        if (chat_ep) {
            p.logprobs = b.value("logprobs", false);
            if (p.logprobs) p.n_top = std::min(20, std::max(0, b.value("top_logprobs", 0)));
        } else if (b.contains("logprobs") && b["logprobs"].is_number_integer()) {
            p.logprobs = true;
            p.n_top = std::min(20, std::max(0, b["logprobs"].get<int>()));
        }
        r.sp.n_logprobs = p.n_top;
        return p;
    };

    auto handle = [&](const httplib::Request& hreq, httplib::Response& res, bool chat_ep) {
        json b = json::parse(hreq.body, nullptr, false);
        if (b.is_discarded() || !b.is_object()) return send_error(res, 400, "invalid JSON body");
        Prepared p;
        try {
            p = prepare(b, chat_ep);
        } catch (const BadRequest& e) {
            return send_error(res, 400, e.what());
        }
        ReqP r = p.r;
        sched.submit(r);
        auto col = std::make_shared<Collector>(tok, r, p.stops, chat_ep, p.thinking, p.tools);
        const std::string model = opt.alias;
        const std::string object = chat_ep ? "chat.completion" : "text_completion";

        if (!p.stream) {
            std::vector<TokEvent> toks;
            std::string text, finish;
            while (true) {
                auto o = col->next(std::chrono::milliseconds(500));
                for (auto& t : o.toks) toks.push_back(t);
                text += o.text;
                if (o.done) { finish = o.finish; break; }
            }
            if (finish == "error") return send_error(res, 500, col->error());
            json choice = {{"index", 0}};
            if (chat_ep) {
                const auto& ps = col->parser();
                json msg = {{"role", "assistant"}, {"content", ps.content().empty() && !ps.calls().empty() ? json() : json(ps.content())}};
                if (!ps.reasoning().empty()) msg["reasoning_content"] = ps.reasoning();
                if (!ps.calls().empty()) {
                    std::vector<std::string> ids;
                    msg["tool_calls"] = tool_calls_json(ps.calls(), 0, ids, false);
                    if (finish == "stop") finish = "tool_calls";
                }
                choice["message"] = msg;
                choice["logprobs"] = p.logprobs ? logprobs_json(tok, toks, p.n_top) : json();
            } else {
                choice["text"] = text;
                choice["logprobs"] = json();
            }
            choice["finish_reason"] = finish;
            send_json(res, {{"id", r->id}, {"object", object}, {"created", (long)time(nullptr)}, {"model", model},
                            {"choices", json::array({choice})}, {"usage", usage_json(*r)}, {"timings", timings_json(*r)}});
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        auto state = std::make_shared<std::pair<bool, int>>(false, 0);   // (role sent, tool calls sent)
        res.set_chunked_content_provider("text/event-stream", [=, &tok, &sched](size_t, httplib::DataSink& sink) {
            auto send = [&](const json& j) {
                const std::string s = "data: " + dump(j) + "\n\n";
                return sink.write(s.data(), s.size());
            };
            auto chunk = [&](const json& delta_or_text, const json& finish, const json& lp) {
                json c = {{"index", 0}};
                if (chat_ep) c["delta"] = delta_or_text;
                else c["text"] = delta_or_text;
                c["logprobs"] = lp;
                c["finish_reason"] = finish;
                return json{{"id", r->id}, {"object", chat_ep ? "chat.completion.chunk" : "text_completion"},
                            {"created", (long)time(nullptr)}, {"model", model}, {"choices", json::array({c})}};
            };
            if (chat_ep && !state->first) {
                state->first = true;
                if (!send(chunk({{"role", "assistant"}, {"content", nullptr}}, nullptr, nullptr))) { r->cancel_reason = "cancelled"; r->cancel = true; return false; }
            }
            auto o = col->next(std::chrono::milliseconds(200));
            const json lp = p.logprobs && !o.toks.empty() ? (chat_ep ? logprobs_json(tok, o.toks, p.n_top) : json()) : json();
            bool ok = true;
            if (chat_ep) {
                json delta = json::object();
                if (!o.d.reasoning.empty()) delta["reasoning_content"] = o.d.reasoning;
                if (!o.d.content.empty()) delta["content"] = o.d.content;
                if (!o.d.calls.empty()) {
                    std::vector<std::string> ids;
                    delta["tool_calls"] = tool_calls_json(o.d.calls, state->second, ids, true);
                    state->second += (int)o.d.calls.size();
                }
                if (!delta.empty() || !lp.is_null()) ok = send(chunk(delta, nullptr, lp));
            } else if (!o.text.empty()) {
                ok = send(chunk(o.text, nullptr, nullptr));
            }
            if (ok && o.done) {
                if (o.finish == "error") {
                    send({{"error", {{"message", col->error()}, {"type", "server_error"}}}});
                } else {
                    std::string fin = o.finish;
                    if (chat_ep && fin == "stop" && state->second > 0) fin = "tool_calls";
                    send(chunk(chat_ep ? json::object() : json(""), fin, nullptr));
                    if (p.include_usage)
                        send({{"id", r->id}, {"object", chat_ep ? "chat.completion.chunk" : "text_completion"}, {"created", (long)time(nullptr)},
                              {"model", model}, {"choices", json::array()}, {"usage", usage_json(*r)}, {"timings", timings_json(*r)}});
                }
                const std::string d = "data: [DONE]\n\n";
                sink.write(d.data(), d.size());
                sink.done();
                return true;
            }
            if (!ok || !sink.is_writable()) {
                r->cancel_reason = "cancelled";
                r->cancel = true;
                sched.wake();
                return false;
            }
            return true;
        }, [r, &sched](bool) {   // connection closed: stop generating
            if (!r->done) { r->cancel_reason = "cancelled"; r->cancel = true; sched.wake(); }
        });
    };
    svr.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, true); });
    svr.Post("/chat/completions", [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, true); });
    svr.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, false); });
    svr.Post("/completions", [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, false); });
    svr.Post("/apply-template", [&](const httplib::Request& req, httplib::Response& res) {
        json b = json::parse(req.body, nullptr, false);
        if (b.is_discarded()) return send_error(res, 400, "invalid JSON body");
        chat::TemplateOptions to;
        json kw = b.value("chat_template_kwargs", json::object());
        to.enable_thinking = kw.value("enable_thinking", true);
        to.truncate_history_thinking = kw.value("truncate_history_thinking", false);
        try {
            send_json(res, {{"prompt", chat::render(b.value("messages", json()), b.value("tools", json()), to)}});
        } catch (const std::exception& e) {
            send_error(res, 400, e.what());
        }
    });

    signal(SIGINT, [](int) { if (g_svr) g_svr->stop(); });
    signal(SIGTERM, [](int) { if (g_svr) g_svr->stop(); });
    fprintf(stderr, "raon-server: %d slots x %d tokens, listening on http://%s:%d\n", opt.slots, opt.ctx, opt.host.c_str(), opt.port);
    if (!svr.listen(opt.host, opt.port)) {
        fprintf(stderr, "raon-server: cannot listen on %s:%d\n", opt.host.c_str(), opt.port);
        sched.stop();
        return 1;
    }
    sched.stop();
    return 0;
}
