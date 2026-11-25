// Checks Engine::step's sampler against a plain full-vocabulary reference of llama.cpp's chain on the same logits and
// the same uniform draw: for many (params, u) the token must be identical and logprobs must agree.
//   build/sampling_check model.gguf
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

#include "engine.h"

using namespace raon;

static int reference(const std::vector<float>& l0, const SamplingParams& p, double u, const std::vector<std::pair<int, int>>& counts,
                     float* logprob) {
    std::vector<float> l = l0;
    for (auto [t, c] : counts) {
        float x = l[t];
        if (p.repetition_penalty != 1.f) x = x > 0 ? x / p.repetition_penalty : x * p.repetition_penalty;
        l[t] = x - (p.presence_penalty + p.frequency_penalty * c);
    }
    const int V = (int)l.size();
    std::vector<int> ord(V);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return l[a] > l[b]; });
    double s = 0;
    for (int i = 0; i < V; i++) s += std::exp((double)l[ord[i]] - l[ord[0]]);
    const double lse = l[ord[0]] + std::log(s);
    auto done = [&](int k) { *logprob = (float)(l[ord[k]] - lse); return ord[k]; };
    if (p.temperature <= 0 || p.top_k == 1) return done(0);
    int keep = V;
    if (p.top_k > 0) keep = std::min(keep, p.top_k);
    if (p.top_p < 1) {
        double z = 0;
        for (int i = 0; i < keep; i++) z += std::exp((double)l[ord[i]] - l[ord[0]]);
        double cum = 0;
        for (int i = 0; i < keep; i++) {
            cum += std::exp((double)l[ord[i]] - l[ord[0]]) / z;
            if (cum >= p.top_p) { keep = i + 1; break; }
        }
    }
    if (p.min_p > 0) {
        const double thr = l[ord[0]] + std::log((double)p.min_p);
        int i = 0;
        while (i < keep && l[ord[i]] >= thr) i++;
        keep = std::max(i, 1);
    }
    double z = 0;
    for (int i = 0; i < keep; i++) z += std::exp(((double)l[ord[i]] - l[ord[0]]) / p.temperature);
    double cum = 0;
    for (int i = 0; i < keep; i++) {
        cum += std::exp(((double)l[ord[i]] - l[ord[0]]) / p.temperature) / z;
        if (cum > u) return done(i);
    }
    return done(keep - 1);
}

int main(int argc, char** argv) {
    Engine eng(argv[1], 2, 4096, false);
    const std::vector<int> prompt = {259201, 1839, 10, 207686, 44, 167405, 245521, 63629, 46, 259202, 10, 259201, 3115, 10, 259213, 10};
    std::vector<Row> rows;
    for (size_t i = 0; i < prompt.size(); i++) rows.push_back({0, (int)i, prompt[i], i + 1 == prompt.size()});
    eng.reset(0);
    const std::vector<float> logits = eng.forward(rows);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(0, 1);
    const std::vector<std::pair<int, int>> counts = {{198, 3}, {11, 1}, {262, 2}};
    struct Case { const char* name; SamplingParams p; bool pen; };
    std::vector<Case> cases;
    auto add = [&](const char* nm, float T, int k, float tp, float mp, float pres = 0, float freq = 0, float rep = 1) {
        SamplingParams p;
        p.temperature = T; p.top_k = k; p.top_p = tp; p.min_p = mp;
        p.presence_penalty = pres; p.frequency_penalty = freq; p.repetition_penalty = rep; p.n_logprobs = 5;
        cases.push_back({nm, p, pres != 0 || freq != 0 || rep != 1});
    };
    add("greedy", 0, 0, 1, 0);
    add("T=1 full vocab", 1, 0, 1, 0);
    add("T=0.7", 0.7f, 0, 1, 0);
    add("T=1.5 full vocab", 1.5f, 0, 1, 0);
    add("top_k 40", 1, 40, 1, 0);
    add("top_k 500", 1, 500, 1, 0);
    add("top_p 0.9", 1, 0, 0.9f, 0);
    add("top_p 0.99 T=2", 2, 0, 0.99f, 0);
    add("min_p 0.05", 1, 0, 1, 0.05f);
    add("min_p 0.001 T=1.3", 1.3f, 0, 1, 0.001f);
    add("k20 p0.95 m0.02", 0.8f, 20, 0.95f, 0.02f);
    add("presence 1.5 T=0.1", 0.1f, 0, 1, 0, 1.5f);
    add("freq+rep", 1, 0, 0.95f, 0, 0.5f, 0.3f, 1.2f);
    int bad = 0, total = 0, full = 0;
    for (const Case& c : cases) {
        const int trials = c.p.temperature <= 0 ? 1 : 24;
        for (int t = 0; t < trials; t++) {
            SampleIn in;
            in.p = c.p;
            in.u = U(rng);
            if (c.pen) in.counts = &counts;
            std::vector<Row> r2 = rows;
            for (auto& r : r2) r.slot = 1;
            eng.reset(1);
            const SampleOut o = eng.step(r2, {in})[0];
            float ref_lp = 0;
            const int ref = reference(logits, c.p, in.u, c.pen ? counts : std::vector<std::pair<int, int>>{}, &ref_lp);
            total++;
            full += o.full_copy;
            if (o.token != ref || std::fabs(o.logprob - ref_lp) > 1e-3f) {
                bad++;
                printf("MISMATCH %-22s u=%.6f: got %d (lp %.5f) ref %d (lp %.5f)\n", c.name, in.u, o.token, o.logprob, ref, ref_lp);
            }
        }
    }
    printf("%d/%d picks identical to the full-vocabulary reference (%d needed the whole row)\n", total - bad, total, full);
    return bad != 0;
}
