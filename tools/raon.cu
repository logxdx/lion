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
    int n = 16, top = 5, ctx = 8192, slots = 1, depth = 0, steps = 32;
    bool bench = false;
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
        else if (a == "--slots") slots = atoi(next().c_str());
        else if (a == "--depth") depth = atoi(next().c_str());
        else if (a == "--steps") steps = atoi(next().c_str());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (bench && !model.empty()) {
        // decode benchmark: `slots` streams at context `depth` (synthetic cache), `steps` batched decode steps
        raon::Engine eng(model, slots, depth + steps + 64);
        fprintf(stderr, "GPU memory: %.0f MiB (weights + states + caches)\n", eng.gpu_bytes() / 1048576.0);
        for (int s = 0; s < slots; s++) eng.bench_fill(s, depth);
        std::vector<double> ms;
        // warm-up: >= 1.5 s of steps at the same positions (GPU clocks ramp up; graphs get captured)
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 1.5) {
            std::vector<raon::Row> rows;
            for (int s = 0; s < slots; s++) rows.push_back({s, depth, 1000 + 37 * s, true});
            eng.forward(rows);
        }
        for (int i = 0; i < steps; i++) {
            std::vector<raon::Row> rows;
            for (int s = 0; s < slots; s++) rows.push_back({s, depth + i, 1000 + 37 * s + i, true});
            auto a = std::chrono::steady_clock::now();
            eng.forward(rows);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        printf("bench slots=%d depth=%d: step %.2f ms (median of %d), %.1f tok/s per stream, %.1f tok/s total\n", slots,
               depth, med, (int)ms.size(), 1000.0 / med, slots * 1000.0 / med);
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
