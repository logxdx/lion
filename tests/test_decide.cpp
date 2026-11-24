// Engine::decide() harness for decision models (d1-2B): one packed forward pass, label logits per question.
// usage: test_decide model.gguf spec.json
//   spec: {"pieces": [[group, "text"], ...], "labels": [["yes", "Yes"], ["no", "No"], ...]}
//   group 0 = shared prefix, k = question k (each contiguous, increasing); the text of each piece is tokenized on
//   its own (special tokens parsed). Prints, per question (its last token), the max logit of each label set and the
//   top 5 tokens.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "../src/engine.h"
#include "../src/gguf.h"
#include "../src/json.hpp"
#include "../src/tokenizer.h"

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf spec.json\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[2]);
    std::stringstream ss;
    ss << in.rdbuf();
    const json spec = json::parse(ss.str());
    gguf::File f(argv[1]);
    Tokenizer tk(f);
    le::Engine eng(argv[1], 8192, false, false, true, 1);
    if (!eng.decision()) { fprintf(stderr, "not a decision model\n"); return 1; }
    std::vector<int> toks, grp, outs;
    for (const auto& p : spec["pieces"]) {
        const int g = p[0].get<int>();
        const auto t = tk.encode(p[1].get<std::string>());
        toks.insert(toks.end(), t.begin(), t.end());
        grp.insert(grp.end(), t.size(), g);
    }
    for (size_t i = 0; i < toks.size(); i++)
        if (grp[i] > 0 && (i + 1 == toks.size() || grp[i + 1] != grp[i])) outs.push_back((int)i);
    if (spec.contains("out")) {  // explicit output positions (-1: the last token)
        outs.clear();
        for (const auto& o : spec["out"]) outs.push_back(o.get<int>() < 0 ? (int)toks.size() + o.get<int>() : o.get<int>());
    }
    std::vector<std::vector<int>> label_ids;
    for (const auto& set : spec.value("labels", json::array())) {
        std::vector<int> ids;
        for (const auto& s : set) {
            const auto t = tk.encode(s.get<std::string>(), false);
            if (t.size() == 1) ids.push_back(t[0]);
        }
        label_ids.push_back(ids);
    }
    for (int i = 1; i < spec.value("reps", 1); i++) eng.decide(toks, grp, nullptr, outs);  // warm-up runs
    auto t0 = std::chrono::steady_clock::now();
    const auto lg = eng.decide(toks, grp, nullptr, outs);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("tokens %zu, questions %zu, decide %.1f ms\n", toks.size(), outs.size(), ms);
    const int V = eng.config().n_vocab;
    for (size_t q = 0; q < outs.size(); q++) {
        const float* l = lg.data() + q * V;
        printf("q%zu (pos %d):", q + 1, outs[q]);
        for (const auto& ids : label_ids) {
            float m = -1e30f;
            for (int id : ids) m = std::max(m, l[id]);
            printf(" %.4f", m);
        }
        std::vector<int> idx(V);
        for (int i = 0; i < V; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        printf("  | top:");
        for (int i = 0; i < 5; i++) printf(" %s(%.2f)", json(tk.decode(idx[i], true)).dump().c_str(), l[idx[i]]);
        printf("\n");
    }
    return 0;
}
