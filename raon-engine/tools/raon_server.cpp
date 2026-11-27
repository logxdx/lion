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
#include <filesystem>
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
#include "web_tools.h"

using namespace raon;
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
static std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

struct Options {
    std::string model, alias = "raon-1-nano", host = "127.0.0.1";
    int port = 8080, slots = 4, ctx = 131072, threads = 32;
    int chunk_busy = 2048;    // prefill tokens per iteration while other requests are generating (mixed: the largest)
    float prefill_share = 0.5f;   // --no-mixed: share of GPU time given to prefill while other requests are generating
    // mixed steps: while others generate, a prompt chunk and their decode rows run as one batch (the weights are read
    // once). The chunk is sized so that a step takes about mixed_budget x a decode step (measured), within
    // [mixed_min, chunk_busy]. mixed = false: alternate prompt chunks and decode steps by prefill_share.
    bool mixed = true;
    float mixed_budget = 1.5f;
    int mixed_min = 32;
    int chunk_idle = 4096;    // ... while none is (= the engine chunk: one progress update per chunk)
    int checkpoints = 4;      // recurrent-state checkpoints per slot (~19 MB each); 0 disables
    SamplingParams defaults;  // generation_config.json of Raon-1 nano: temperature 0.1, presence_penalty 1.5
    std::string ui_dir;       // web UI bundle (default: <binary dir>/webui, built by `make webui`)
    bool ui = true;
    std::string tools_dir, node = "node";   // server tools (default: <binary dir>/../tools/webtools)
    bool tools = true;
    int tool_timeout = 300;   // seconds per tool call
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
    std::atomic<int> n_done{0};           // prompt tokens in the slot state so far (reused + prefilled)
    std::atomic<bool> admitted{false}, generating{false};
    Clock::time_point t_admit, t_gen0;    // set before admitted / generating become true
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
    std::string vram() { std::lock_guard<std::mutex> lk(info_mu_); return vram_; }
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
            r->t_admit = Clock::now();
            r->n_done = reuse;
            r->admitted = true;
        }
    }

    void run() {
        try {
            eng_ = std::make_unique<Engine>(opt_.model, opt_.slots, opt_.ctx, true, opt_.checkpoints);
            vram_ = eng_->vram_report();
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

    // after prompt rows [n0, end) of s were fed: checkpoint at ckpt_pos, DECODE (first token: *first) at the end
    void after_prefill(Slot& s, int n0, int end, bool ck, const SampleOut* first) {
        Req& r = *s.req;
        const int P = (int)r.prompt.size();
        s.cached.insert(s.cached.end(), r.prompt.begin() + n0, r.prompt.begin() + end);
        r.n_done = end;
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
            if (r.max_tokens == 0) {   // prefill only (n_predict = 0): the prompt is in the state and checkpoints
                finish(s, "length");
            } else {
                s.t1 = Clock::now();
                r.t_gen0 = s.t1;
                r.generating = true;
                s.st = DECODE;
                emit(s, *first);
            }
        }
    }
    // end of the next prompt chunk of s from n0, at most `chunk` rows, cut at the checkpoint position
    int chunk_end(const Slot& s, int n0, int chunk, bool& ck) const {
        const Req& r = *s.req;
        const int P = (int)r.prompt.size();
        int end = std::min(P, n0 + chunk);
        ck = !s.ck.empty() && r.ckpt_pos > n0 && r.ckpt_pos < P;
        if (ck && r.ckpt_pos < end) end = r.ckpt_pos;
        return end;
    }
    // decode rows of every generating request (cancelled ones are finished here)
    void decode_rows(std::vector<Row>& rows, std::vector<SampleIn>& in, std::vector<Slot*>& who) {
        for (Slot& s : slots_) {
            if (s.st != DECODE) continue;
            if (s.req->cancel) { finish(s, s.req->cancel_reason); continue; }
            rows.push_back({s.id, (int)s.cached.size(), s.last, true});
            in.push_back(sample_in(s));
            who.push_back(&s);
        }
    }
    // prompt chunk for a mixed step: the decode step's time budget left over, at the measured cost per prompt row
    int mixed_chunk(int nd) const {
        const double dec = dec_ms_[std::min(nd, 64)] > 0 ? dec_ms_[std::min(nd, 64)] : 20.0;
        const double per_row = row_ms_ > 0 ? row_ms_ : 0.1;
        int c = (int)((opt_.mixed_budget - 1.0) * dec / per_row);
        c = std::max(opt_.mixed_min, std::min(c, opt_.chunk_busy));
        // prompt + decode rows fill whole 128-row GEMM tiles: up to 128 rows cost about the same as 32 (the weight
        // read dominates), and one row past a multiple adds a whole tile (measured: --bench-mixed)
        const int rows = std::min(std::max(128, (c + nd) / 128 * 128), eng_->chunk_rows() / 128 * 128);
        return rows - nd;
    }

    void iterate() {
        for (Slot& s : slots_)
            if (s.req && s.req->cancel) finish(s, s.req->cancel_reason);
        admit();
        Slot* pf = nullptr;
        bool decoding = false;
        for (Slot& s : slots_) {
            if (s.st == PREFILL && (!pf || s.seq < pf->seq)) pf = &s;
            decoding |= s.st == DECODE;
        }
        // the mixed chunk is sized from the decode step time: measure it (a decode-only step) when it is unknown for
        // this batch size, and refresh it every 16 mixed steps (the decoders' contexts grow during a long prompt)
        int ndec = 0;
        for (const Slot& s : slots_) ndec += s.st == DECODE;
        const bool sample_decode = dec_ms_[std::min(ndec, 64)] <= 0 || mixed_run_ >= 16;
        if (pf && decoding && opt_.mixed && sample_decode) pf = nullptr;
        if (pf && decoding && opt_.mixed && mixed_chunk(ndec) <= 16) pf = nullptr, mixed_run_ = 0;   // no room: decode
        if (pf && decoding && opt_.mixed) {
            // one mixed step: a prompt chunk of the oldest request in prefill + every generating request's next token
            std::vector<Row> drows;
            std::vector<SampleIn> din;
            std::vector<Slot*> who;
            decode_rows(drows, din, who);
            Slot& s = *pf;
            Req& r = *s.req;
            const int n0 = (int)s.cached.size(), P = (int)r.prompt.size();
            bool ck = false;
            const int end = chunk_end(s, n0, mixed_chunk((int)drows.size()), ck);
            std::vector<Row> rows;
            std::vector<SampleIn> in;
            rows.reserve(end - n0 + drows.size());
            for (int i = n0; i < end; i++) rows.push_back({s.id, i, r.prompt[i], i == P - 1});
            if (end == P) in.push_back(sample_in(s));
            rows.insert(rows.end(), drows.begin(), drows.end());
            in.insert(in.end(), din.begin(), din.end());
            const auto t0 = Clock::now();
            const auto out = eng_->step(rows, in);
            const double dt = ms_since(t0);
            const double dec = drows.empty() ? 0 : dec_ms_[std::min((int)drows.size(), 64)];
            if (dec > 0 && end > n0) {
                const double per_row = std::max(1e-4, (dt - dec) / (end - n0));
                row_ms_ = row_ms_ > 0 ? 0.7 * row_ms_ + 0.3 * per_row : per_row;
            }
            mixed_run_++;
            const int k = end == P ? 1 : 0;
            after_prefill(s, n0, end, ck, k ? &out[0] : nullptr);
            for (size_t i = 0; i < who.size(); i++) {
                who[i]->cached.push_back(who[i]->last);
                emit(*who[i], out[k + i]);
            }
            return;
        }
        // one prefill chunk: the oldest request still in prefill. While others are generating (--no-mixed), prefill
        // gets `prefill_share` of the time: after a chunk that took t, decode-only steps run for t * (1 - share) / share.
        if (pf && decoding && (opt_.mixed || Clock::now() < next_prefill_)) pf = nullptr;
        if (pf) {
            const auto tp0 = Clock::now();
            Slot& s = *pf;
            Req& r = *s.req;
            const int n0 = (int)s.cached.size(), P = (int)r.prompt.size();
            bool ck = false;
            const int end = chunk_end(s, n0, decoding ? opt_.chunk_busy : opt_.chunk_idle, ck);
            std::vector<Row> rows;
            rows.reserve(end - n0);
            for (int i = n0; i < end; i++) rows.push_back({s.id, i, r.prompt[i], i == P - 1});
            std::vector<SampleIn> in;
            if (end == P) in.push_back(sample_in(s));
            const auto out = eng_->step(rows, in);
            after_prefill(s, n0, end, ck, end == P ? &out[0] : nullptr);
            const float sh = std::min(1.f, std::max(0.05f, opt_.prefill_share));
            next_prefill_ = Clock::now() + std::chrono::duration_cast<Clock::duration>((Clock::now() - tp0) * ((1.f - sh) / sh));
        }
        // one decode step over every generating request
        std::vector<Row> rows;
        std::vector<SampleIn> in;
        std::vector<Slot*> who;
        decode_rows(rows, in, who);
        if (rows.empty()) return;
        const auto t0 = Clock::now();
        const auto out = eng_->step(rows, in);
        const double dt = ms_since(t0);
        // the first step of a batch size captures its CUDA graph: not a sample
        const int bs = std::min((int)rows.size(), 64);
        if (dec_seen_[bs]++) {
            double& e = dec_ms_[bs];
            e = e > 0 ? 0.8 * e + 0.2 * dt : dt;
        }
        mixed_run_ = 0;
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
    double dec_ms_[65] = {};   // decode step time by batch size (EMA), for sizing mixed chunks
    double row_ms_ = 0;        // per prompt row in a mixed step, beyond the decode rows (EMA)
    int dec_seen_[65] = {};    // decode steps seen per batch size
    int mixed_run_ = 0;        // mixed steps since the last decode-only step
    std::mutex info_mu_;
    json info_ = json::array();
    std::string vram_;
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
        if (!lead_.empty() && !ev.empty()) {
            held_out(o, lead_);
            lead_.clear();
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
    // text placed before the generated output (e.g. "<think>\n" when reasoning stays inline); shown with the next delta
    void lead(const std::string& text) { lead_ += text; }

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
    std::string held_, error_, lead_;
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
// live: while generating, predicted_ms runs from the first token (the scheduler fills gen_ms only at the end)
static json timings_json(const Req& r, bool live = false, int n_gen = -1) {
    const int pn = (int)r.prompt.size() - r.n_reused;
    const int ng = n_gen >= 0 ? n_gen : r.n_gen;
    const double gms = live && r.generating ? ms_since(r.t_gen0) : r.gen_ms;
    return {{"queue_ms", r.queue_ms}, {"prompt_n", pn}, {"prompt_ms", r.prompt_ms},
            {"prompt_per_second", r.prompt_ms > 0 ? pn / (r.prompt_ms / 1000) : 0.0}, {"cache_n", r.n_reused},
            {"cached_n", r.n_reused}, {"predicted_n", ng}, {"predicted_ms", gms},
            {"predicted_per_second", gms > 0 ? ng / (gms / 1000) : 0.0}, {"slot", r.slot}};
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
            "  --no-mixed           while others generate, alternate prompt chunks and decode steps (by --prefill-share)\n"
            "                       instead of running the prompt chunk and the decode rows as one batch (default)\n"
            "  --mixed-budget F     mixed step time target as a multiple of a decode step (default 1.5; lower = faster\n"
            "                       streams while a prompt is fed, slower prompt)\n"
            "  --mixed-min N        smallest prompt chunk of a mixed step (default 32)\n"
            "  --prefill-share F    --no-mixed: share of GPU time for prefill while others generate (default 0.5)\n"
            "  --checkpoints N      recurrent-state checkpoints per slot for prompt reuse (default 4, ~19 MB each; 0 = off)\n"
            "  --temp T --top-p P --top-k K --min-p M --presence-penalty X --frequency-penalty X --repeat-penalty X\n"
            "                       default sampling (default: temp 0.1, presence 1.5 from generation_config.json)\n"
            "  --threads N          HTTP worker threads (default 32)\n"
            "  --ui-dir DIR         web UI served at / (default <binary dir>/webui; build it with: make webui)\n"
            "  --no-ui              serve the API only\n"
            "  --tools-dir DIR      server tools web_search / web_fetch / pdf, served at /tools like llama-server's\n"
            "                       built-in tools (default <binary dir>/../tools/webtools; node deps: make webtools)\n"
            "  --no-tools           do not serve /tools\n"
            "  --node PATH          node binary for the tools (default node)\n"
            "  --tool-timeout S     per tool call (default 300)\n");
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
        else if (a == "--no-mixed") opt.mixed = false;
        else if (a == "--mixed-budget") opt.mixed_budget = std::stof(next());
        else if (a == "--mixed-min") opt.mixed_min = std::stoi(next());
        else if (a == "--checkpoints") opt.checkpoints = std::max(0, std::stoi(next()));
        else if (a == "--threads") opt.threads = std::stoi(next());
        else if (a == "--ui-dir") opt.ui_dir = next();
        else if (a == "--no-ui") opt.ui = false;
        else if (a == "--tools-dir") opt.tools_dir = next();
        else if (a == "--no-tools") opt.tools = false;
        else if (a == "--node") opt.node = next();
        else if (a == "--tool-timeout") opt.tool_timeout = std::stoi(next());
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
    // the configuration this server runs with, and its VRAM use
    svr.Get("/props", [&](const httplib::Request&, httplib::Response& res) {
        auto r6 = [](float x) { return std::round((double)x * 1e6) / 1e6; };   // 0.1f -> 0.1
        const SamplingParams& d = opt.defaults;
        json params = {{"temperature", r6(d.temperature)}, {"top_k", d.top_k}, {"top_p", r6(d.top_p)}, {"min_p", r6(d.min_p)},
                       {"presence_penalty", r6(d.presence_penalty)}, {"frequency_penalty", r6(d.frequency_penalty)},
                       {"repeat_penalty", r6(d.repetition_penalty)}, {"max_tokens", -1}};
        json dgs = params;
        dgs["n_ctx"] = opt.ctx;
        dgs["params"] = params;
        send_json(res, {// llama-server /props fields used by the web UI
                        {"default_generation_settings", dgs}, {"model_alias", opt.alias}, {"model_path", opt.model},
                        {"total_slots", opt.slots}, {"n_ctx", opt.ctx},
                        {"modalities", {{"vision", false}, {"audio", false}, {"video", false}}},
                        {"chat_template", tok.chat_template()}, {"build_info", "raon-engine"},
                        {"cors_proxy_enabled", false}, {"ui_settings", json::object()},
                        // raon-server's own
                        {"model", opt.alias}, {"alias", opt.alias}, {"slots", opt.slots}, {"ctx", opt.ctx},
                        {"chunk_busy", opt.chunk_busy}, {"prefill_share", opt.prefill_share}, {"checkpoints", opt.checkpoints},
                        {"defaults", {{"temperature", opt.defaults.temperature}, {"top_p", opt.defaults.top_p},
                                      {"top_k", opt.defaults.top_k}, {"min_p", opt.defaults.min_p},
                                      {"presence_penalty", opt.defaults.presence_penalty},
                                      {"frequency_penalty", opt.defaults.frequency_penalty},
                                      {"repetition_penalty", opt.defaults.repetition_penalty}}},
                        {"vram", sched.vram()}});
    });
    svr.Get("/slots", [&](const httplib::Request&, httplib::Response& res) { send_json(res, sched.slots_json()); });
    auto models = [&](const httplib::Request&, httplib::Response& res) {
        send_json(res, {{"object", "list"},
                        {"data", json::array({{{"id", opt.alias}, {"object", "model"}, {"created", created}, {"owned_by", "raon-engine"},
                                               {"status", {{"value", "loaded"}}},
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
        bool parse_reasoning = true;      // reasoning_format "none": thinking stays inline in content
        bool progress = false, timings_per_token = false;
        double ping_s = 0;                // sse_ping_interval: SSE comment while nothing else is sent
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
        // the first of these that is given: >= 0 caps the completion (0 = prefill only), < 0 = up to the context
        int mt = sched.ctx() - (int)r.prompt.size();
        for (const char* k : {"max_completion_tokens", "max_tokens", "n_predict"})
            if (b.contains(k) && b[k].is_number_integer()) {
                if (b[k].get<int>() >= 0) mt = std::min(mt, b[k].get<int>());
                break;
            }
        r.max_tokens = mt;
        p.parse_reasoning = !(b.contains("reasoning_format") && b["reasoning_format"] == "none");
        p.progress = b.value("return_progress", false);
        p.timings_per_token = b.value("timings_per_token", false);
        if (b.contains("sse_ping_interval") && b["sse_ping_interval"].is_number()) p.ping_s = b["sse_ping_interval"].get<double>();
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
        auto col = std::make_shared<Collector>(tok, r, p.stops, chat_ep, p.thinking && p.parse_reasoning, p.tools);
        if (chat_ep && p.thinking && !p.parse_reasoning) col->lead("<think>\n");   // the prompt opened the block
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
        struct Live { int progress = -1; bool tokens = false; Clock::time_point last = Clock::now(); };
        auto live = std::make_shared<Live>();
        res.set_chunked_content_provider("text/event-stream", [=, &tok, &sched](size_t, httplib::DataSink& sink) {
            auto send = [&](const json& j) {
                const std::string s = "data: " + dump(j) + "\n\n";
                live->last = Clock::now();
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
            live->tokens |= !o.toks.empty();
            // prompt progress (llama-server's prompt_progress) until the first token
            if (p.progress && !live->tokens && r->admitted && r->n_done != live->progress) {
                live->progress = r->n_done;
                json pp = {{"total", (int)r->prompt.size()}, {"cache", r->n_reused}, {"processed", live->progress},
                           {"time_ms", ms_since(r->t_admit)}};
                if (!send({{"id", r->id}, {"object", chat_ep ? "chat.completion.chunk" : "text_completion"},
                           {"created", (long)time(nullptr)}, {"model", model}, {"choices", json::array()}, {"prompt_progress", pp}})) {
                    r->cancel_reason = "cancelled";
                    r->cancel = true;
                    sched.wake();
                    return false;
                }
            }
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
                if (!delta.empty() || !lp.is_null()) {
                    json c = chunk(delta, nullptr, lp);
                    if (p.timings_per_token) c["timings"] = timings_json(*r, true, r->n_gen);
                    ok = send(c);
                }
            } else if (!o.text.empty()) {
                json c = chunk(o.text, nullptr, nullptr);
                if (p.timings_per_token) c["timings"] = timings_json(*r, true, r->n_gen);
                ok = send(c);
            }
            if (ok && !o.done && p.ping_s > 0 &&
                std::chrono::duration<double>(Clock::now() - live->last).count() >= p.ping_s) {
                live->last = Clock::now();
                ok = sink.write(": ping\n\n", 8);
            }
            if (ok && o.done) {
                if (o.finish == "error") {
                    send({{"error", {{"message", col->error()}, {"type", "server_error"}}}});
                } else {
                    std::string fin = o.finish;
                    if (chat_ep && fin == "stop" && state->second > 0) fin = "tool_calls";
                    json last = chunk(chat_ep ? json::object() : json(""), fin, nullptr);
                    last["timings"] = timings_json(*r);   // as llama-server: the web UI shows speed from it
                    send(last);
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

    // llama-server endpoints the web UI calls: no model catalog / downloads, nothing to load or unload, no stream
    // resumption; answering keeps its console clean
    auto ok_json = [&](const httplib::Request&, httplib::Response& res) { send_json(res, {{"success", true}}); };
    svr.Get("/v1/catalog.json", [&](const httplib::Request&, httplib::Response& res) { send_json(res, json::array()); });
    svr.Post("/v1/streams/lookup", [&](const httplib::Request&, httplib::Response& res) { send_json(res, json::array()); });
    // Server tools, llama-server's /tools contract (see web_tools.h): the web UI lists them next to its own tools and
    // calls them through POST /tools when the model asks for them
    std::unique_ptr<WebTools> wtools;
    std::string tools_state = "off (--no-tools)";
    if (opt.tools) {
        std::string dir = opt.tools_dir;
        if (dir.empty()) {
            std::error_code ec;
            dir = (std::filesystem::read_symlink("/proc/self/exe", ec).parent_path().parent_path() / "tools" / "webtools").string();
        }
        wtools = std::make_unique<WebTools>(dir, opt.node, opt.tool_timeout);
        if (wtools->enabled()) {
            tools_state.clear();
            for (const auto& t : wtools->list()) tools_state += (tools_state.empty() ? "" : ", ") + t.value("tool", std::string());
            tools_state += " (" + dir + ")";
        } else {
            tools_state = "disabled: " + wtools->error();
        }
    }
    auto tools_off = [&](httplib::Response& res) {
        if (wtools && wtools->enabled()) return false;
        // what llama-server answers without --tools; the UI then hides server tools
        send_json(res, {{"error", {{"message", "server tools are disabled"}, {"type", "permission_error"}, {"code", 403}}}}, 403);
        return true;
    };
    svr.Get("/tools", [&](const httplib::Request&, httplib::Response& res) {
        if (!tools_off(res)) send_json(res, wtools->list());
    });
    svr.Post("/tools", [&](const httplib::Request& q, httplib::Response& res) {
        if (tools_off(res)) return;
        json b = json::parse(q.body, nullptr, false);
        if (!b.is_object() || !b.contains("tool") || !b["tool"].is_string())
            return send_error(res, 400, "expected {\"tool\": name, \"params\": {...}}");
        const std::string name = b["tool"].get<std::string>();
        if (!wtools->has(name)) return send_error(res, 404, "unknown tool \"" + name + "\"");
        if (b.value("stream", false)) return send_error(res, 404, "tool \"" + name + "\" does not support stream = true");
        json params = b.contains("params") && b["params"].is_object() ? b["params"] : json::object();
        params.erase("cwd");   // like llama-server: the working directory comes only from the x-tool-cwd header
        if (q.has_header("x-tool-cwd")) params["cwd"] = q.get_header_value("x-tool-cwd");
        const auto t0 = Clock::now();
        json out = wtools->call(name, params, q.is_connection_closed);
        fprintf(stderr, "raon-server: tool %s: %s in %.0f ms\n", name.c_str(), out.contains("error") ? "error" : "ok", ms_since(t0));
        send_json(res, out);
    });
    svr.Post("/models/load", ok_json);
    svr.Post("/models/unload", ok_json);
    svr.Get("/models/sse", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [](size_t, httplib::DataSink& sink) {
            for (int i = 0; i < 6; i++) {   // ~60 s of keep-alives, then the UI reconnects
                if (!sink.write(":\n\n", 3)) return false;
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
            sink.done();
            return true;
        });
    });

    // Web UI: llama.cpp's chat UI (prebuilt bundle, rebranded; `make webui`). Hash routing and relative asset paths, so
    // serving the directory at / is all it needs; GET requests for files it does not contain fall through to the API.
    std::string ui_state = "off (--no-ui)";
    if (opt.ui) {
        std::string dir = opt.ui_dir;
        if (dir.empty()) {
            std::error_code ec;
            dir = (std::filesystem::read_symlink("/proc/self/exe", ec).parent_path() / "webui").string();
        }
        if (std::filesystem::exists(std::filesystem::path(dir) / "index.html") && svr.set_mount_point("/", dir)) {
            svr.set_file_extension_and_mimetype_mapping("webmanifest", "application/manifest+json");
            svr.set_file_extension_and_mimetype_mapping("mjs", "application/javascript");
            // content-named bundle files are cached for good; index, icons, manifest and service worker revalidate
            svr.set_file_request_handler([](const httplib::Request& q, httplib::Response& r) {
                const bool immutable = q.path.rfind("/_app/immutable/", 0) == 0;
                r.set_header("Cache-Control", immutable ? "public, max-age=31536000, immutable" : "no-cache");
            });
            ui_state = dir;
        } else {
            ui_state = "not found in " + dir + " (run: make webui); API only";
        }
    }

    signal(SIGINT, [](int) { if (g_svr) g_svr->stop(); });
    signal(SIGTERM, [](int) { if (g_svr) g_svr->stop(); });
    fprintf(stderr, "raon-server: %d slots x %d tokens, chunk-busy %d, prefill-share %.2f, checkpoints %d, defaults: temp %.2f "
                    "top_p %.2f top_k %d min_p %.2f presence %.2f frequency %.2f repeat %.2f; listening on http://%s:%d\n",
            opt.slots, opt.ctx, opt.chunk_busy, opt.prefill_share, opt.checkpoints, opt.defaults.temperature, opt.defaults.top_p,
            opt.defaults.top_k, opt.defaults.min_p, opt.defaults.presence_penalty, opt.defaults.frequency_penalty,
            opt.defaults.repetition_penalty, opt.host.c_str(), opt.port);
    fprintf(stderr, "raon-server: web UI: %s\n", ui_state.c_str());
    fprintf(stderr, "raon-server: tools: %s\n", tools_state.c_str());
    if (!svr.listen(opt.host, opt.port)) {
        fprintf(stderr, "raon-server: cannot listen on %s:%d\n", opt.host.c_str(), opt.port);
        sched.stop();
        return 1;
    }
    sched.stop();
    return 0;
}
