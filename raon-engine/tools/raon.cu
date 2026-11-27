// raon CLI (v1): run token ids through the engine, greedy decode, print JSON lines with top-k log-probs.
//   raon -m model.gguf --tokens 1,2,3 [-n 32] [--top 5] [--ctx 8192]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "engine.h"

static std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    std::string cur;
    for (char ch : s + ",") {
        if (ch == ',' || ch == ' ' || ch == '\n' || ch == '[' || ch == ']') {
            if (!cur.empty()) v.push_back(atoi(cur.c_str()));
            cur.clear();
        } else {
            cur += ch;
        }
    }
    return v;
}

static void emit(int step, const float* lg, int V, int top, int chosen) {
    float mx = -INFINITY;
    for (int i = 0; i < V; i++) mx = std::max(mx, lg[i]);
    double sum = 0;
    for (int i = 0; i < V; i++) sum += std::exp((double)lg[i] - mx);
    const double lse = mx + std::log(sum);
    std::vector<int> idx;
    for (int k = 0; k < top; k++) {
        int best = -1;
        for (int i = 0; i < V; i++) {
            bool used = false;
            for (int j : idx) used |= j == i;
            if (!used && (best < 0 || lg[i] > lg[best])) best = i;
        }
        idx.push_back(best);
    }
    printf("{\"step\":%d,\"token\":%d,\"logprob\":%.6f,\"top\":[", step, chosen, lg[chosen] - lse);
    for (int k = 0; k < top; k++) printf("%s[%d,%.6f]", k ? "," : "", idx[k], lg[idx[k]] - lse);
    printf("]}\n");
    fflush(stdout);
}

int main(int argc, char** argv) {
    std::string model, toks_s;
    int n = 16, top = 5, ctx = 8192, slots = 1, depth = 0, steps = 32, prefill_n = 0;
    bool bench = false, sample = false, test_mixed = false, bench_mixed = false;
    int mixed_chunk = 256, pf_depth = 0;
    std::vector<int> chunks = {32, 64, 128, 256, 512, 1024, 2048};
    float b_temp = 0.8f, b_top_p = 0.95f;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model = next();
        else if (a == "--tokens") toks_s = next();
        else if (a == "--tokens-file") { std::ifstream f(next()); std::stringstream ss; ss << f.rdbuf(); toks_s = ss.str(); }
        else if (a == "-n") n = atoi(next().c_str());
        else if (a == "--top") top = atoi(next().c_str());
        else if (a == "--ctx") ctx = atoi(next().c_str());
        else if (a == "--bench") bench = true;
        else if (a == "--temp") b_temp = atof(next().c_str());
        else if (a == "--top-p") b_top_p = atof(next().c_str());
        else if (a == "--sample") sample = true;   // bench: sample on the GPU (Engine::step), no logits copy
        else if (a == "--slots") slots = atoi(next().c_str());
        else if (a == "--depth") depth = atoi(next().c_str());
        else if (a == "--steps") steps = atoi(next().c_str());
        else if (a == "--prefill") prefill_n = atoi(next().c_str());
        // mixed steps (decode rows inside prefill chunks):
        //   --test-mixed [--chunk C] [--prefill P] [-n G]   fed slot must match a plain prefill exactly
        //   --bench-mixed --slots N --depth D [--pf-depth Q] [--chunks a,b,..]
        else if (a == "--test-mixed") test_mixed = true;
        else if (a == "--bench-mixed") bench_mixed = true;
        else if (a == "--chunk") mixed_chunk = atoi(next().c_str());
        else if (a == "--pf-depth") pf_depth = atoi(next().c_str());
        else if (a == "--chunks") chunks = parse_ids(next());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (test_mixed && !model.empty()) {
        // slots 0 and 1 decode (greedy) while slot 2's prompt is fed in mixed chunks of `chunk` rows; reference: slot 2
        // fed alone in chunks of the same size, then greedy. The fed slot runs the same kernels on the same chunk
        // boundaries, so its logits and tokens must be IDENTICAL; the decode rows run k_mhc + the GEMMs instead of the
        // small-batch mHC + GEMVs, so on this (numerically chaotic) model they may part ways: reported, not failed.
        const int C = mixed_chunk, P = prefill_n > 0 ? prefill_n : 3000, G = n;
        if (C <= 16 || (P % C > 0 && P % C <= 16)) {
            fprintf(stderr, "--test-mixed: every prompt chunk must have more than 16 rows (--chunk > 16, prefill %% chunk "
                            "0 or > 16), else it takes the decode path and differs from the reference by design\n");
            return 1;
        }
        raon::Engine eng(model, 3, P + G + 1024);
        const int V = eng.cfg().n_vocab;
        auto synth = [](int n_, uint32_t seed) {
            std::vector<int> t(n_);
            for (auto& x : t) { seed = seed * 1664525u + 1013904223u; x = 1000 + (int)(seed % 200000u); }
            return t;
        };
        const std::vector<int> pa = synth(500, 1), pb = synth(700, 2), pc = synth(P, 3);
        auto argmax = [&](const float* lg) { int b = 0; for (int i = 1; i < V; i++) if (lg[i] > lg[b]) b = i; return b; };
        auto feed_chunks = [&](int slot, const std::vector<int>& t) {   // chunks of C rows, logits of the last row
            std::vector<float> lg;
            for (int p0 = 0; p0 < (int)t.size(); p0 += C) {
                std::vector<raon::Row> rows;
                for (int i = p0; i < std::min((int)t.size(), p0 + C); i++) rows.push_back({slot, i, t[i], i + 1 == (int)t.size()});
                auto o = eng.forward(rows);
                if (!o.empty()) lg = o;
            }
            return lg;
        };
        auto greedy = [&](int slot, int pos, int tok, int count, std::vector<int>& out) {
            for (int i = 0; i < count; i++) {
                auto lg = eng.forward({{slot, pos++, tok, true}});
                tok = argmax(lg.data());
                out.push_back(tok);
            }
        };
        // references, each alone in slot 0
        std::vector<int> ref_a, ref_b, ref_c;
        eng.reset(0); { auto lg = feed_chunks(0, pa); ref_a.push_back(argmax(lg.data())); greedy(0, 500, ref_a[0], G - 1, ref_a); }
        eng.reset(0); { auto lg = feed_chunks(0, pb); ref_b.push_back(argmax(lg.data())); greedy(0, 700, ref_b[0], G - 1, ref_b); }
        eng.reset(0);
        const std::vector<float> lg_c_ref = feed_chunks(0, pc);
        ref_c.push_back(argmax(lg_c_ref.data()));
        greedy(0, P, ref_c[0], G - 1, ref_c);
        // mixed
        for (int s = 0; s < 3; s++) eng.reset(s);
        std::vector<int> got_a, got_b, got_c;
        { auto lg = feed_chunks(0, pa); got_a.push_back(argmax(lg.data())); }
        { auto lg = feed_chunks(1, pb); got_b.push_back(argmax(lg.data())); }
        int pos_a = 500, pos_b = 700, steps_mixed = 0;
        std::vector<float> lg_c;
        for (int p0 = 0; p0 < P; p0 += C) {
            std::vector<raon::Row> rows;
            const int end = std::min(P, p0 + C);
            for (int i = p0; i < end; i++) rows.push_back({2, i, pc[i], i + 1 == P});
            rows.push_back({0, pos_a++, got_a.back(), true});
            rows.push_back({1, pos_b++, got_b.back(), true});
            auto o = eng.forward(rows);
            int k = 0;
            if (end == P) { lg_c.assign(o.begin(), o.begin() + V); k = 1; }
            got_a.push_back(argmax(o.data() + (size_t)k * V));
            got_b.push_back(argmax(o.data() + (size_t)(k + 1) * V));
            steps_mixed++;
        }
        got_c.push_back(argmax(lg_c.data()));
        greedy(2, P, got_c[0], G - 1, got_c);
        double dmax = 0;
        for (int i = 0; i < V; i++) dmax = std::max(dmax, (double)std::fabs(lg_c[i] - lg_c_ref[i]));
        auto cmp = [](const std::vector<int>& a, const std::vector<int>& b) {
            size_t k = 0;
            while (k < a.size() && k < b.size() && a[k] == b[k]) k++;
            return k;
        };
        const size_t na = std::min(got_a.size(), ref_a.size()), nb_ = std::min(got_b.size(), ref_b.size());
        const size_t ka = cmp(got_a, ref_a), kb = cmp(got_b, ref_b), kc = cmp(got_c, ref_c);
        printf("slot 2 (fed in %d mixed chunks of %d): last-prompt logits max |diff| %.3g, %zu tokens, first mismatch at %zu -> %s\n",
               steps_mixed, C, dmax, ref_c.size(), kc, kc == ref_c.size() && dmax == 0 ? "IDENTICAL" : "DIFF");
        printf("slot 0 (decoding during them): %zu tokens, first mismatch at %zu -> %s\n", na, ka, ka == na ? "IDENTICAL" : "DIFF");
        printf("slot 1 (decoding during them): %zu tokens, first mismatch at %zu -> %s\n", nb_, kb, kb == nb_ ? "IDENTICAL" : "DIFF");
        const bool ok = kc == ref_c.size() && dmax == 0;
        printf("%s\n", ok ? "PASS" : "FAIL (the fed slot must match exactly)");
        return ok ? 0 : 1;
    }
    if (bench_mixed && !model.empty()) {
        // `slots` streams decode at `depth` (synthetic caches) while one more slot is fed chunks at `pf_depth`. Per chunk
        // size, interleaved: decode = one decode step, chunk = the prompt chunk alone, mixed = both in one step.
        // The server used to alternate chunk and decode steps; a mixed step replaces both with one pass.
        int cmax = 0;
        for (int c : chunks) cmax = std::max(cmax, c);
        raon::Engine eng(model, slots + 1, std::max(depth, pf_depth) + cmax + steps + 64);
        for (int s = 0; s < slots; s++) eng.bench_fill(s, depth);
        const int ps = slots;
        eng.bench_fill(ps, pf_depth);
        raon::SampleIn si;
        si.p.temperature = 0.f;
        std::vector<int> pt(cmax);
        uint32_t x = 777;
        for (auto& t : pt) { x = x * 1664525u + 1013904223u; t = 1000 + (int)(x % 200000u); }
        int dpos = depth;
        auto decode_rows = [&]() {
            std::vector<raon::Row> r;
            for (int s = 0; s < slots; s++) r.push_back({s, dpos, 1000 + 37 * s, true});
            dpos++;
            if (dpos >= depth + steps) dpos = depth;
            return r;
        };
        auto prompt_rows = [&](int C) {
            std::vector<raon::Row> r;
            for (int i = 0; i < C; i++) r.push_back({ps, pf_depth + i, pt[i], false});
            return r;
        };
        auto timed = [&](const std::vector<raon::Row>& rows, int nlog) {
            const auto a = std::chrono::steady_clock::now();
            eng.step(rows, std::vector<raon::SampleIn>(nlog, si));
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        };
        const auto w0 = std::chrono::steady_clock::now();   // warm-up >= 1.5 s (clocks ramp up; graphs get captured)
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 1.5) timed(decode_rows(), slots);
        printf("%d decoding slots at depth %d, prompt at depth %d; VRAM %s\n", slots, depth, pf_depth, eng.vram_report().c_str());
        printf("| chunk | decode ms | chunk ms | mixed ms | alternating: decode t/s/stream, prefill t/s | mixed: decode t/s/stream, prefill t/s |\n");
        printf("|---|---|---|---|---|---|\n");
        for (int C : chunks) {
            double td = 0, tc = 0, tm = 0;
            const int reps = 5;
            for (int r = 0; r <= reps; r++) {
                const double d = timed(decode_rows(), slots);
                const double c = timed(prompt_rows(C), 0);
                auto mrows = prompt_rows(C - slots);   // C rows in all: whole 128-row GEMM tiles
                for (auto& row : decode_rows()) mrows.push_back(row);
                const double m = timed(mrows, slots);
                if (r == 0) continue;
                td += d / reps; tc += c / reps; tm += m / reps;
            }
            printf("| %d | %.2f | %.2f | %.2f | %.1f, %.0f | %.1f, %.0f |\n", C, td, tc, tm, 1e3 / (tc + td), C * 1e3 / (tc + td),
                   1e3 / tm, (C - slots) * 1e3 / tm);
            fflush(stdout);
        }
        return 0;
    }
    if (bench && prefill_n > 0 && !model.empty()) {
        // prefill benchmark: prefill_n synthetic tokens into slot 0, timed per 16k-token segment
        raon::Engine eng(model, 1, prefill_n + 64);
        std::vector<int> toks(prefill_n);
        uint32_t x = 12345;
        for (auto& t : toks) { x = x * 1664525u + 1013904223u; t = 1000 + (int)(x % 200000u); }
        // warm-up: >= 1.5 s of prefill chunks (GPU clocks ramp up from idle)
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 1.5) {
            eng.reset(0);
            eng.prefill(0, std::vector<int>(toks.begin(), toks.begin() + std::min(prefill_n, 2048)));
        }
        eng.reset(0);
        const int SEG = 16384;
        double total = 0;
        for (int p0 = 0; p0 < prefill_n; p0 += SEG) {
            const int n = std::min(SEG, prefill_n - p0);
            auto a = std::chrono::steady_clock::now();
            eng.prefill(0, std::vector<int>(toks.begin() + p0, toks.begin() + p0 + n), p0);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            total += s;
            printf("prefill [%6d, %6d): %.2f s, %.0f tok/s\n", p0, p0 + n, s, n / s);
            fflush(stdout);
        }
        printf("prefill %d tokens: %.1f s, %.0f tok/s average\n", prefill_n, total, prefill_n / total);
        return 0;
    }
    if (bench && !model.empty()) {
        // decode benchmark: `slots` streams at context `depth` (synthetic cache), `steps` batched decode steps
        raon::Engine eng(model, slots, depth + steps + 64);
        fprintf(stderr, "GPU memory: %.0f MiB (weights + states + caches)\n", eng.gpu_bytes() / 1048576.0);
        for (int s = 0; s < slots; s++) eng.bench_fill(s, depth);
        std::vector<double> ms;
        raon::SampleIn si;
        si.p.temperature = b_temp;
        si.p.top_p = b_top_p;
        int full = 0, picks = 0;
        auto run = [&](const std::vector<raon::Row>& rows) {
            if (sample)
                for (const auto& o : eng.step(rows, std::vector<raon::SampleIn>(rows.size(), si))) { full += o.full_copy; picks++; }
            else eng.forward(rows);
        };
        // warm-up: >= 1.5 s of steps at the same positions (GPU clocks ramp up; graphs get captured)
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 1.5) {
            std::vector<raon::Row> rows;
            for (int s = 0; s < slots; s++) rows.push_back({s, depth, 1000 + 37 * s, true});
            run(rows);
        }
        for (int i = 0; i < steps; i++) {
            std::vector<raon::Row> rows;
            for (int s = 0; s < slots; s++) rows.push_back({s, depth + i, 1000 + 37 * s + i, true});
            auto a = std::chrono::steady_clock::now();
            run(rows);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        printf("bench slots=%d depth=%d: step %.2f ms (median of %d), %.1f tok/s per stream, %.1f tok/s total\n", slots,
               depth, med, (int)ms.size(), 1000.0 / med, slots * 1000.0 / med);
        if (sample) printf("sampling: %d of %d picks needed more than the top candidates\n", full, picks);
        return 0;
    }
    if (model.empty() || toks_s.empty()) {
        fprintf(stderr, "usage: raon -m model.gguf --tokens 1,2,3 [-n 16] [--top 5] [--ctx 8192]\n");
        return 1;
    }
    const std::vector<int> toks = parse_ids(toks_s);
    raon::Engine eng(model, 1, ctx);
    const int V = eng.cfg().n_vocab;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> lg = eng.prefill(0, toks);
    auto t1 = std::chrono::steady_clock::now();
    int pos = (int)toks.size();
    double dec_s = 0;
    for (int s = 0; s < n; s++) {
        int best = 0;
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        emit(s, lg.data(), V, top, best);
        if (s + 1 == n) break;
        auto a = std::chrono::steady_clock::now();
        lg = eng.forward({{0, pos++, best, true}});
        dec_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    }
    const double pf = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "prefill %zu tokens %.3f s (%.1f t/s), decode %d tokens %.3f s (%.1f t/s)\n", toks.size(), pf,
            toks.size() / pf, n - 1, dec_s, n > 1 ? (n - 1) / dec_s : 0.0);
    return 0;
}
