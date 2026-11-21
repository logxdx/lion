// Step-level engine API + state transfer tests (run: build/test_engine_api <model.gguf>).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/engine.h"
#include "../src/gguf.h"
#include "../src/tokenizer.h"

using namespace le;

static std::vector<int> run_steps(Engine& e, int slot, int n) {
    std::vector<int> out, toks;
    std::vector<char> done;
    for (int i = 0; i < n; i++) {
        e.decode_step({slot}, toks, done);
        out.push_back(toks[0]);
        if (done[0]) break;
    }
    return out;
}
static int common(const std::vector<int>& a, const std::vector<int>& b) {
    size_t k = 0;
    while (k < a.size() && k < b.size() && a[k] == b[k]) k++;
    return (int)k;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
    std::unique_ptr<Tokenizer> tk;
    { gguf::File f(argv[1]); tk = std::make_unique<Tokenizer>(f); }
    Engine e(argv[1], 8192, true, false, true, 4);
    SamplingParams g; g.temperature = 0; g.repeat_penalty = 1.1f;  // penalty on: history must be restored too
    for (int s = 0; s < 4; s++) e.set_sampling(s, g, {tk->eos});
    std::string text = "<|startoftext|><|im_start|>user\nWrite a detailed explanation of how TCP congestion control works, "
                       "covering slow start, congestion avoidance, fast retransmit and fast recovery.";
    for (int i = 0; i < 40; i++) text += " Include concrete numeric examples.";
    text += "<|im_end|>\n<|im_start|>assistant\n<think></think>";
    const std::vector<int> P = tk->encode(text);
    const int n = (int)P.size(), N = 96;
    int fails = 0;
    auto check = [&](const char* name, const std::vector<int>& a, const std::vector<int>& b) {
        const int k = common(a, b);
        const bool ok = k == (int)a.size() && a.size() == b.size();
        fails += !ok;
        printf("%-58s %s (%d/%zu tokens match)\n", name, ok ? "IDENTICAL" : "DIFF", k, a.size());
    };

    // reference: the existing prefill + generate path
    e.reset();
    e.prefill(P, 0);
    const std::vector<int> ref = e.generate(N, g, {tk->eos});

    // 1) feed + decode_step
    e.reset(0);
    e.feed(0, P.data(), n - 1);
    e.set_pending(0, P[n - 1]);
    Engine::Checkpoint cp;
    e.save_checkpoint(0, cp);
    std::vector<uint8_t> kv(e.kv_row_bytes() * (n - 1));
    e.save_kv(0, 0, n - 1, kv.data());
    const std::vector<int> a = run_steps(e, 0, N);
    check("1. feed + decode_step == prefill + generate", ref, a);

    // 2) KV + checkpoint + history moved to another slot
    e.reset(1);
    e.load_kv(1, 0, n - 1, kv.data());
    e.load_checkpoint(1, cp);
    e.write_history(1, 0, P.data(), n - 1);
    e.set_pending(1, P[n - 1]);
    check("2. state copied to host and restored into slot 1", ref, run_steps(e, 1, N));

    // 3) rewind slot 0 to its checkpoint after it generated further
    e.load_checkpoint(0, cp);
    e.set_pending(0, P[n - 1]);
    check("3. same-slot rewind to the end-of-prompt checkpoint", ref, run_steps(e, 0, N));

    // 4) mid-prompt checkpoint (prefix reuse with a different continuation boundary)
    const int h = n / 2;
    e.reset(2);
    e.feed(2, P.data(), h);
    Engine::Checkpoint mid;
    e.save_checkpoint(2, mid);
    std::vector<uint8_t> kvh(e.kv_row_bytes() * h);
    e.save_kv(2, 0, h, kvh.data());
    e.reset(3);
    e.load_kv(3, 0, h, kvh.data());
    e.load_checkpoint(3, mid);
    e.write_history(3, 0, P.data(), h);
    e.feed(3, P.data() + h, n - 1 - h);
    e.set_pending(3, P[n - 1]);
    const std::vector<int> d = run_steps(e, 3, N);
    const int k = common(ref, d);
    printf("%-58s %s (%d/%zu tokens match; prefill chunk boundary moved)\n",
           "4. mid-prompt checkpoint + feed the rest", k == (int)ref.size() ? "IDENTICAL" : "DIFF", k, ref.size());

    // 5) batched decode_step over 4 slots == each alone (all slots resumed at end of prompt)
    for (int s = 0; s < 4; s++) {
        e.reset(s);
        e.load_kv(s, 0, n - 1, kv.data());
        e.load_checkpoint(s, cp);
        e.write_history(s, 0, P.data(), n - 1);
        e.set_pending(s, P[n - 1]);
    }
    std::vector<std::vector<int>> outs(4);
    std::vector<int> toks;
    std::vector<char> done;
    for (int i = 0; i < N; i++) {
        e.decode_step({0, 1, 2, 3}, toks, done);
        for (int s = 0; s < 4; s++) outs[s].push_back(toks[s]);
    }
    for (int s = 0; s < 4; s++) outs[s].resize(ref.size());
    check("5. 4 restored slots decoded in one batch", ref, outs[3]);
    printf("prompt %d tokens, checkpoint %zu KiB, KV %zu B/token\n%s\n", n, cp.bytes() / 1024, e.kv_row_bytes(),
           fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
