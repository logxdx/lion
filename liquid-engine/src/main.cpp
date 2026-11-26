// liquid: CLI for liquid-engine (LFM3-1.7B on CUDA).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "engine.h"
#include "gguf.h"
#include "sampling_defaults.h"
#include "tokenizer.h"
#include "video.h"
#include "vision.h"

using namespace le;

static void usage() {
    fprintf(stderr,
            "usage: liquid -m model.gguf [options]\n"
            "  -p TEXT            user prompt (chat template applied)\n"
            "  -f FILE            read the prompt from a file\n"
            "  -s TEXT            system prompt\n"
            "  -i                 interactive chat\n"
            "  --raw              feed the prompt verbatim (no chat template)\n"
            "  --no-think         disable thinking (<think></think> prefix)\n"
            "  --mmproj F --image F   vision (LFM3-VL): attach images (repeatable) to the first user message\n"
            "  --video F          video (LFM3-VL, needs --mmproj; ffmpeg + ffprobe on PATH): 2 fps, repeatable;\n"
            "                     attached after the images\n"
            "  -n N               max new tokens (default 2048)\n"
            "  -c N               context size (default 32768)\n"
            "  --temp F --top-k N --top-p F --min-p F --repeat-penalty F --repeat-last-n N --seed N\n"
            "                     sampling; defaults come from the GGUF (LFM3: temp 0.1, top-k 50, repeat-penalty\n"
            "                     1.1; top-p/min-p off; penalty window = whole context, -1)\n"
            "  --greedy           temperature 0, no penalty\n"
            "  --bench            benchmark: --pp N prompt tokens, --tg N generated tokens, --reps N\n"
            "  --parallel N       with --bench: N concurrent sequences (batched decode), aggregate t/s\n"
            "  --test-multi N     check: N prompts decoded together == each decoded alone (greedy)\n"
            "  --test-mixed       check: 2 slots decoding while a 3rd prompt is fed through mixed steps vs each alone\n"
            "  --bench-mixed      --parallel N slots decoding at depth --pp while a prompt (at depth --pf-depth) is fed:\n"
            "                     decode step, prefill chunk and mixed step times per chunk size (--chunks a,b,..)\n"
            "  --tokenize TEXT    print token ids\n"
            "  --dump-logits      after the prompt, print the top-10 next-token logits\n"
            "  --seq-prefill      process the prompt token by token (debug)\n"
            "  --kv q8|f16        KV cache type (default q8: int8 + f16 scale per 32; f16 = most accurate)\n"
            "  --profile          per-kernel decode timing (use with --pp N for context length)\n"
            "  --exact            reference mode: f32 activations, exact dequantization (slow, debug)\n"
            "  --show-special     print control tokens\n"
            "  -q                 quiet (no stats)\n");
}

struct Msg { std::string role, content; };

static std::string render_chat(const std::vector<Msg>& msgs, const std::string& system, bool think, bool add_gen) {
    std::string s = "<|startoftext|>";
    if (!system.empty()) s += "<|im_start|>system\n" + system + "<|im_end|>\n";
    int last_user = -1;
    for (int i = 0; i < (int)msgs.size(); i++) if (msgs[i].role == "user") last_user = i;
    for (int i = 0; i < (int)msgs.size(); i++) {
        const Msg& m = msgs[i];
        s += "<|im_start|>" + m.role + "\n";
        if (m.role == "assistant") {
            std::string c = m.content;
            if (i < last_user) {  // past turns: thinking stripped
                size_t p = c.rfind("</think>");
                if (p != std::string::npos) {
                    c = c.substr(p + 8);
                    size_t a = c.find_first_not_of(" \n\t\r");
                    c = a == std::string::npos ? "" : c.substr(a);
                    size_t e = c.find_last_not_of(" \n\t\r");
                    c = e == std::string::npos ? "" : c.substr(0, e + 1);
                }
                c = "<think></think>" + c;
            }
            s += c + "<|im_end|>\n";
        } else {
            s += m.content + "<|im_end|>\n";
        }
    }
    if (add_gen) s += think ? "<|im_start|>assistant\n<think>" : "<|im_start|>assistant\n<think></think>";
    return s;
}

struct StreamCtx { const Tokenizer* tok; bool show_special; std::string text; std::string pending_utf8; };

static bool on_tok(int id, void* ud) {
    auto* c = (StreamCtx*)ud;
    std::string piece = c->tok->decode(id, c->show_special);
    c->text += piece;
    // hold back incomplete UTF-8 sequences
    c->pending_utf8 += piece;
    size_t n = c->pending_utf8.size(), cut = n;
    for (size_t k = 1; k <= std::min<size_t>(3, n); k++) {
        unsigned char ch = c->pending_utf8[n - k];
        if ((ch & 0xC0) == 0x80) continue;
        int len = ch >= 0xF0 ? 4 : ch >= 0xE0 ? 3 : ch >= 0xC0 ? 2 : 1;
        if ((int)k < len) cut = n - k;
        break;
    }
    fwrite(c->pending_utf8.data(), 1, cut, stdout);
    fflush(stdout);
    c->pending_utf8.erase(0, cut);
    return true;
}

int main(int argc, char** argv) {
    std::string model, prompt, system, tok_text, mmproj;
    std::vector<std::string> image_paths, video_paths;
    bool interactive = false, raw = false, think = true, bench = false, dump_logits = false, quiet = false;
    bool seq_prefill = false, show_special = false, do_tokenize = false, exact = false, profile = false;
    bool kv_q8 = true;  // default q8_0 KV: faster decode at every context, half the memory (see README)
    int n_predict = 2048, ctx = 32768, pp = 512, tg = 128, reps = 3, parallel = 1, test_multi = 0;
    bool test_mixed = false, bench_mixed = false;
    int pf_depth = 0;
    std::vector<int> chunks = {32, 64, 128, 256, 512, 1024};
    SamplingParams sp;      // filled from the GGUF's general.sampling.* keys, then the flags below
    SamplingOverrides ov;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "-p") prompt = next();
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--image") image_paths.push_back(next());
        else if (a == "--video") video_paths.push_back(next());
        else if (a == "-f") {
            FILE* fp = fopen(next().c_str(), "rb");
            if (!fp) { perror("prompt file"); return 1; }
            char buf[65536]; size_t r;
            while ((r = fread(buf, 1, sizeof buf, fp)) > 0) prompt.append(buf, r);
            fclose(fp);
        }
        else if (a == "-s") system = next();
        else if (a == "-i") interactive = true;
        else if (a == "--raw") raw = true;
        else if (a == "--no-think") think = false;
        else if (a == "-n") n_predict = std::stoi(next());
        else if (a == "-c") ctx = std::stoi(next());
        else if (a == "--temp") ov.temp = std::stof(next());
        else if (a == "--top-k") ov.top_k = std::stoi(next());
        else if (a == "--top-p") ov.top_p = std::stof(next());
        else if (a == "--min-p") ov.min_p = std::stof(next());
        else if (a == "--repeat-penalty") ov.penalty = std::stof(next());
        else if (a == "--repeat-last-n") ov.last_n = std::stoi(next());
        else if (a == "--seed") sp.seed = std::stoull(next());
        else if (a == "--greedy") { ov.temp = 0; ov.penalty = 1.0f; }
        else if (a == "--bench") bench = true;
        else if (a == "--pp") pp = std::stoi(next());
        else if (a == "--tg") tg = std::stoi(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--parallel" || a == "-np") parallel = std::stoi(next());
        else if (a == "--test-multi") test_multi = std::stoi(next());
        else if (a == "--test-mixed") test_mixed = true;
        else if (a == "--bench-mixed") bench_mixed = true;
        else if (a == "--pf-depth") pf_depth = std::stoi(next());
        else if (a == "--chunks") {
            chunks.clear();
            std::string v = next();
            for (size_t i = 0; i < v.size();) {
                size_t j = v.find(',', i);
                if (j == std::string::npos) j = v.size();
                chunks.push_back(std::stoi(v.substr(i, j - i)));
                i = j + 1;
            }
        }
        else if (a == "--tokenize") { do_tokenize = true; tok_text = next(); }
        else if (a == "--dump-logits") dump_logits = true;
        else if (a == "--seq-prefill") seq_prefill = true;
        else if (a == "--exact") exact = true;
        else if (a == "--profile") profile = true;
        else if (a == "--kv") {
            const std::string v = next();
            if (v == "q8" || v == "q8_0") kv_q8 = true;
            else if (v == "f16") kv_q8 = false;
            else { fprintf(stderr, "--kv must be f16 or q8\n"); return 1; }
        }
        else if (a == "--show-special") show_special = true;
        else if (a == "-q") quiet = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    if (model.empty()) { usage(); return 1; }
    if (exact) kv_q8 = false;  // the f32 reference keeps an f16 KV cache
    if (sp.seed == 0) sp.seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();

    try {
        std::unique_ptr<Tokenizer> tokp;
        {
            gguf::File f(model);
            tokp = std::make_unique<Tokenizer>(f);
            apply_gguf_sampling(f, sp);
        }
        ov.apply(sp);
        const Tokenizer& tk = *tokp;
        if (do_tokenize) {
            auto ids = tk.encode(tok_text);
            for (size_t i = 0; i < ids.size(); i++) printf("%s%d", i ? " " : "", ids[i]);
            printf("\n");
            return 0;
        }
        std::vector<int> stop = {tk.eos};
        int eot = tk.id("<|endoftext|>");
        if (eot >= 0 && eot != tk.eos) stop.push_back(eot);

        if (profile) {
            Engine eng(model, std::max(ctx, pp + 64), !quiet, false, kv_q8);
            std::vector<int> toks(pp);
            for (int i = 0; i < pp; i++) toks[i] = 1000 + (i * 7919) % 100000;
            eng.profile_prefill(std::min(pp, 32768));
            eng.prefill(toks);
            eng.profile_decode(21);
            return 0;
        }
        if (test_multi > 0) {
            // N different prompts decoded in one batch must give exactly the tokens each gives alone
            const int N = test_multi;
            Engine eng(model, std::max(ctx, 4096), !quiet, false, kv_q8, N);
            SamplingParams g; g.temperature = 0; g.repeat_penalty = 1.0f;
            const char* topics[8] = {"the history of the printing press", "how vaccines train the immune system",
                                     "why the sky is blue", "the rules of chess", "how a compiler works",
                                     "the water cycle", "black holes", "the French revolution"};
            std::vector<std::vector<int>> prompts(N);
            for (int i = 0; i < N; i++) {
                // different lengths per slot: exercises per-slot positions and the SWA ring
                std::string q = std::string("Explain ") + topics[i % 8] + " in detail." + std::string(i * 40, ' ');
                for (int r = 0; r < i * 30; r++) q += " Please be thorough.";
                const std::vector<Msg> one = {{"user", q}};
                prompts[i] = tk.encode(render_chat(one, "", false, true));
            }
            std::vector<int> slots(N);
            for (int i = 0; i < N; i++) { slots[i] = i; eng.prefill(prompts[i], i); }
            auto batched = eng.generate_multi(slots, tg, g, {});
            int fails = 0;
            for (int i = 0; i < N; i++) {
                eng.reset(0);
                eng.prefill(prompts[i], 0);
                auto alone = eng.generate(tg, g, {});
                size_t k = 0;
                while (k < alone.size() && k < batched[i].size() && alone[k] == batched[i][k]) k++;
                const bool ok = k == alone.size() && alone.size() == batched[i].size();
                fails += !ok;
                printf("slot %d: prompt %zu tok, batched %zu / alone %zu tokens, first mismatch at %zu -> %s\n", i,
                       prompts[i].size(), batched[i].size(), alone.size(), k, ok ? "IDENTICAL" : "DIFF");
            }
            printf("%s\n", fails ? "FAIL" : "PASS");
            return fails ? 1 : 0;
        }
        if (test_mixed) {
            // slots 0 and 1 decode while slot 2's prompt is fed in mixed steps (chunks of 256), then all three decode.
            // Reference: each prompt alone (prefill + greedy generate). The fed slot goes through the same kernels as a
            // plain prefill, so it must be IDENTICAL; the decode rows run the prefill GEMMs instead of the decode GEMVs,
            // so they may part ways at a near-tie (reported; judge it against the logit gap).
            Engine eng(model, std::max(ctx, 8192), !quiet, false, kv_q8, 3);
            SamplingParams g; g.temperature = 0; g.repeat_penalty = 1.0f;
            const char* qs[3] = {"Explain how a compiler works, step by step.", "Write a short story about a lighthouse keeper.",
                                 "Summarise the following notes and list the open questions."};
            std::vector<std::vector<int>> prompts(3);
            for (int i = 0; i < 3; i++) {
                std::string q = qs[i];
                if (i == 2)
                    for (int r = 0; r < 120; r++)
                        q += " Note " + std::to_string(r) + ": the meeting moved the launch by a week because the vendor missed the certification window.";
                prompts[i] = tk.encode(render_chat(std::vector<Msg>{{"user", q}}, "", false, true));
            }
            const int n_gen = tg;
            std::vector<std::vector<int>> ref(3);
            for (int i = 0; i < 3; i++) {
                eng.reset(0);
                eng.prefill(prompts[i], 0);
                ref[i] = eng.generate(n_gen, g, {});
            }
            eng.reset();
            std::vector<std::vector<int>> got(3);
            for (int i = 0; i < 3; i++) eng.set_sampling(i, g, {});
            for (int i = 0; i < 2; i++) {
                eng.feed(i, prompts[i].data(), (int)prompts[i].size() - 1);
                eng.set_pending(i, prompts[i].back());
            }
            const auto& P = prompts[2];
            const int target = (int)P.size() - 1;
            int fed = 0, mixed_steps = 0;
            std::vector<int> ids = {0, 1}, toks;
            std::vector<char> done;
            while (fed < target) {
                const int n = std::min(256, target - fed);
                eng.mixed_step(2, P.data() + fed, n, ids, toks, done);
                fed += n;
                mixed_steps++;
                for (int k = 0; k < 2; k++) got[k].push_back(toks[k]);
            }
            eng.set_pending(2, P.back());
            ids = {0, 1, 2};
            while ((int)got[2].size() < n_gen) {
                eng.decode_step(ids, toks, done);
                for (int k = 0; k < 3; k++) got[k].push_back(toks[k]);
            }
            int fails = 0;
            for (int i = 0; i < 3; i++) {
                const size_t n = std::min(got[i].size(), ref[i].size());
                size_t k = 0;
                while (k < n && got[i][k] == ref[i][k]) k++;
                const bool same = k == n;
                if (i == 2 && !same) fails++;
                printf("slot %d (%s, prompt %zu tok): %zu tokens compared, first mismatch at %zu -> %s\n", i,
                       i == 2 ? "fed in mixed steps" : "decoding during them", prompts[i].size(), n, k,
                       same ? "IDENTICAL" : "DIFF");
            }
            printf("%d mixed steps; %s\n", mixed_steps, fails ? "FAIL (the fed slot must match exactly)" : "PASS");
            return fails ? 1 : 0;
        }
        if (bench_mixed) {
            // N slots decode at depth pp; slot N is fed chunks of C tokens from depth pf_depth. Per C, interleaved:
            //   decode = decode_step alone, chunk = feed(C) alone, mixed = mixed_step(C).
            // The server used to alternate chunk + decode; a mixed step replaces both with one pass.
            const int N = std::max(1, parallel);
            int cmax = 0;
            for (int c : chunks) cmax = std::max(cmax, c);
            Engine eng(model, std::max(ctx, std::max(pp, pf_depth) + cmax + 4096), !quiet, false, kv_q8, N + 1);
            SamplingParams g; g.temperature = 0; g.repeat_penalty = 1.0f;
            std::vector<int> ids(N);
            for (int i = 0; i < N; i++) {
                ids[i] = i;
                eng.set_sampling(i, g, {});
                std::vector<int> t(pp);
                for (int j = 0; j < pp; j++) t[j] = 1000 + ((j + 7 * i) * 7919) % 100000;
                eng.feed(i, t.data(), pp - 1);
                eng.set_pending(i, t.back());
            }
            const int ps = N;
            std::vector<int> pt(pf_depth + cmax + 1);
            for (size_t j = 0; j < pt.size(); j++) pt[j] = 1000 + ((j + 13) * 7919) % 100000;
            if (pf_depth > 0) eng.feed(ps, pt.data(), pf_depth);
            Engine::Checkpoint cp;
            eng.save_checkpoint(ps, cp);  // every chunk starts from depth pf_depth
            std::vector<int> toks;
            std::vector<char> done;
            using clk = std::chrono::steady_clock;
            auto ms = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };
            for (int w = 0; w < 30; w++) eng.decode_step(ids, toks, done);  // warm-up (clocks ramp)
            const std::string mname = model.substr(model.find_last_of('/') + 1);
            printf("%s, %d decoding slots at depth %d, prompt at depth %d, VRAM %.0f MiB\n", mname.c_str(), N, pp, pf_depth,
                   eng.gpu_bytes() / 1048576.0);
            printf("| chunk | decode ms | chunk ms | mixed ms | alternating: decode t/s/stream, prefill t/s | mixed: decode t/s/stream, prefill t/s |\n");
            printf("|---|---|---|---|---|---|\n");
            for (int C : chunks) {
                double td = 0, tc = 0, tm = 0;
                for (int r = 0; r < reps + 1; r++) {
                    auto t0 = clk::now();
                    eng.decode_step(ids, toks, done);
                    const double d = ms(t0);
                    eng.load_checkpoint(ps, cp);
                    t0 = clk::now();
                    eng.feed(ps, pt.data() + pf_depth, C);
                    const double c = ms(t0);
                    eng.load_checkpoint(ps, cp);
                    t0 = clk::now();
                    eng.mixed_step(ps, pt.data() + pf_depth, C, ids, toks, done);
                    const double m = ms(t0);
                    eng.load_checkpoint(ps, cp);
                    if (r == 0) continue;
                    td += d / reps; tc += c / reps; tm += m / reps;
                }
                printf("| %d | %.2f | %.2f | %.2f | %.1f, %.0f | %.1f, %.0f |\n", C, td, tc, tm, 1e3 / (tc + td),
                       C * 1e3 / (tc + td), 1e3 / tm, C * 1e3 / tm);
            }
            return 0;
        }
        if (bench && parallel > 1) {
            const int N = parallel;
            Engine eng(model, std::max(ctx, pp + tg + 16), !quiet, false, kv_q8, N);
            SamplingParams g; g.temperature = 0; g.repeat_penalty = 1.0f;
            std::vector<int> slots(N);
            for (int i = 0; i < N; i++) slots[i] = i;
            std::vector<double> pps, tgs;
            for (int r = 0; r < reps + 1; r++) {
                eng.reset();
                eng.timing = Timing{};
                for (int i = 0; i < N; i++) {
                    std::vector<int> toks(pp);
                    for (int j = 0; j < pp; j++) toks[j] = 1000 + ((j + 7 * i) * 7919) % 100000;
                    eng.prefill(toks, i);
                }
                const double pms = eng.timing.prefill_ms;
                eng.timing.decode_ms = 0; eng.timing.decode_tokens = 0;
                eng.generate_multi(slots, tg, g, {});
                if (r == 0) continue;
                pps.push_back(eng.timing.prefill_tokens / (pms / 1e3));
                tgs.push_back(eng.timing.decode_tokens / (eng.timing.decode_ms / 1e3));
            }
            auto mean = [](const std::vector<double>& v) { double m = 0; for (double x : v) m += x; return m / v.size(); };
            const std::string mname = model.substr(model.find_last_of('/') + 1);
            printf("| model | streams | ctx/stream | prefill t/s | decode aggregate t/s | decode per stream t/s | GPU MiB |\n");
            printf("|---|---|---|---|---|---|---|\n");
            printf("| %s | %d | %d | %.1f | %.1f | %.1f | %.0f |\n", mname.c_str(), N, pp - 1, mean(pps), mean(tgs),
                   mean(tgs) / N, eng.gpu_bytes() / 1048576.0);
            return 0;
        }
        if (bench) {
            Engine eng(model, std::max(ctx, pp + tg + 16), !quiet, false, kv_q8);
            SamplingParams g; g.temperature = 0; g.repeat_penalty = 1.0f;
            std::vector<int> toks(pp);
            for (int i = 0; i < pp; i++) toks[i] = 1000 + (i * 7919) % 100000;
            printf("| model | test | t/s |\n|---|---|---|\n");
            std::vector<double> pps, tgs;
            for (int r = 0; r < reps + 1; r++) {
                eng.reset();
                eng.timing = Timing{};
                eng.prefill(toks);
                eng.generate(tg, g, {});  // no stop tokens
                if (r == 0) continue;     // warmup
                pps.push_back(eng.timing.prefill_tokens / (eng.timing.prefill_ms / 1e3));
                tgs.push_back(eng.timing.decode_tokens / (eng.timing.decode_ms / 1e3));
            }
            auto stat = [](std::vector<double> v) {
                double m = 0; for (double x : v) m += x; m /= v.size();
                double s = 0; for (double x : v) s += (x - m) * (x - m); s = std::sqrt(s / v.size());
                char buf[64]; snprintf(buf, sizeof buf, "%.1f ± %.1f", m, s); return std::string(buf);
            };
            printf("| %s | pp%d | %s |\n", model.substr(model.find_last_of('/') + 1).c_str(), pp - 1, stat(pps).c_str());
            printf("| %s | tg%d | %s |\n", model.substr(model.find_last_of('/') + 1).c_str(), tg, stat(tgs).c_str());
            return 0;
        }

        Engine eng(model, ctx, !quiet, exact, kv_q8);
        // vision: images go into the first user message as <|vision_start|><|image|><|vision_end|> each
        std::unique_ptr<le::Vision> vis;
        std::vector<le::Image> images;
        std::vector<le::Video> videos;
        size_t next_image = 0, next_video = 0;
        const int image_tok = tk.id("<|image|>"), video_tok = tk.id("<|video|>");
        const int vstart_tok = tk.id("<|vision_start|>"), vend_tok = tk.id("<|vision_end|>");
        if (!image_paths.empty() || !video_paths.empty()) {
            if (mmproj.empty()) { fprintf(stderr, "--image / --video need --mmproj\n"); return 1; }
            vis = std::make_unique<le::Vision>(mmproj, !quiet);
            for (const auto& ip : image_paths) {
                FILE* fp = fopen(ip.c_str(), "rb");
                if (!fp) { fprintf(stderr, "cannot open %s\n", ip.c_str()); return 1; }
                std::vector<uint8_t> bytes;
                uint8_t buf[65536];
                size_t r;
                while ((r = fread(buf, 1, sizeof buf, fp)) > 0) bytes.insert(bytes.end(), buf, buf + r);
                fclose(fp);
                le::Image im;
                std::string err;
                if (!le::decode_image(bytes.data(), bytes.size(), im, err)) { fprintf(stderr, "%s: %s\n", ip.c_str(), err.c_str()); return 1; }
                images.push_back(std::move(im));
            }
            for (const auto& vp : video_paths) {
                le::Video v;
                std::string err;
                if (video_tok < 0) { fprintf(stderr, "this model has no <|video|> token\n"); return 1; }
                if (!le::decode_video_file(vp, v, err)) { fprintf(stderr, "%s: %s\n", vp.c_str(), err.c_str()); return 1; }
                if (!quiet)
                    fprintf(stderr, "[video] %s: %dx%d, %zu frames -> %d pairs at %dx%d\n", vp.c_str(), v.src_w, v.src_h,
                            v.frames.size(), v.n_pairs(), v.frames[0].w, v.frames[0].h);
                videos.push_back(std::move(v));
            }
            std::string marks;
            for (size_t i = 0; i < images.size(); i++) marks += "<|vision_start|><|image|><|vision_end|>";
            for (size_t i = 0; i < videos.size(); i++) marks += "<|vision_start|><|video|><|vision_end|>";
            prompt = marks + prompt;
        }
        // prefill that replaces each <|image|> token by the image's embeddings and each <|video|> token by
        // ("<t seconds>" <|vision_start|> pair embeddings <|vision_end|>) per frame pair (same pending-token rule as
        // prefill)
        auto prefill_mm = [&](const std::vector<int>& toks) {
            if (!vis || (std::find(toks.begin(), toks.end(), image_tok) == toks.end() &&
                         (video_tok < 0 || std::find(toks.begin(), toks.end(), video_tok) == toks.end()))) {
                eng.prefill(toks);
                return;
            }
            std::vector<int> all;
            if (eng.pending(0) >= 0) all.push_back(eng.pending(0));
            all.insert(all.end(), toks.begin(), toks.end() - 1);
            std::vector<int> run;
            auto flush = [&]() { if (!run.empty()) { eng.feed(0, run.data(), (int)run.size()); run.clear(); } };
            for (int t : all) {
                if (t == video_tok && video_tok >= 0) {
                    if (next_video >= videos.size()) throw std::runtime_error("more <|video|> tokens than videos");
                    le::Video& v = videos[next_video++];
                    for (int g = 0; g < v.n_pairs(); g++) {
                        const auto st = tk.encode(v.stamps[g]);
                        run.insert(run.end(), st.begin(), st.end());
                        run.push_back(vstart_tok);
                        flush();
                        const le::Image* f1 = 2 * g + 1 < (int)v.frames.size() ? &v.frames[2 * g + 1] : nullptr;
                        int n = 0;
                        const float* e = vis->encode_pair(v.frames[2 * g], f1, n);
                        if (!quiet) fprintf(stderr, "[vision] video %zu %s: %d tokens, %.1f ms\n", next_video, v.stamps[g].c_str(), n, vis->last_ms());
                        eng.feed_embd(0, e, n, image_tok);
                        run.push_back(vend_tok);
                    }
                    continue;
                }
                if (t != image_tok) { run.push_back(t); continue; }
                flush();
                if (next_image >= images.size()) throw std::runtime_error("more <|image|> tokens than images");
                int n = 0;
                const float* e = vis->encode(images[next_image++], n);
                if (!quiet) fprintf(stderr, "[vision] image %zu: %d tokens, %.1f ms\n", next_image, n, vis->last_ms());
                eng.feed_embd(0, e, n, image_tok);
            }
            flush();
            eng.set_pending(0, toks.back());
            eng.sync();
        };
        eng.use_batched_prefill = !seq_prefill && !exact;  // exact reference: every token via the f32 path

        if (dump_logits) {
            const std::vector<Msg> one = {{"user", prompt}};
            std::string text = raw ? prompt : render_chat(one, system, think, true);
            auto ids = tk.encode(text);
            eng.prefill(ids);
            SamplingParams g; g.temperature = 0; g.repeat_penalty = 1.0f;
            auto out = eng.generate(1, g, {});
            auto lg = eng.logits();
            std::vector<int> idx(lg.size());
            for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
            std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            double mx = lg[idx[0]], se = 0;
            for (float v : lg) se += std::exp((double)v - mx);
            printf("n_prompt=%zu next=%d\n", ids.size(), out.empty() ? -1 : out[0]);
            for (int i = 0; i < 20; i++)
                printf("%d\t%.4f\t%.6f\t%s\n", idx[i], lg[idx[i]], std::exp(lg[idx[i]] - mx) / se, tk.decode(idx[i], true).c_str());
            return 0;
        }

        std::vector<Msg> msgs;
        std::vector<int> ctx_tokens;  // tokens in the engine (consumed + pending)
        auto run_turn = [&](const std::string& user) {
            std::vector<int> ids;
            if (raw) ids = tk.encode(user);
            else {
                msgs.push_back({"user", user});
                ids = tk.encode(render_chat(msgs, system, think, true));
            }
            bool is_prefix = !raw && ctx_tokens.size() <= ids.size() &&
                             std::equal(ctx_tokens.begin(), ctx_tokens.end(), ids.begin());
            std::vector<int> feed;
            if (is_prefix && !ctx_tokens.empty()) feed.assign(ids.begin() + ctx_tokens.size(), ids.end());
            else { eng.reset(); ctx_tokens.clear(); feed = ids; }
            eng.timing = Timing{};
            prefill_mm(feed);
            ctx_tokens.insert(ctx_tokens.end(), feed.begin(), feed.end());
            StreamCtx sc{&tk, show_special, "", ""};
            if (!raw && think) { printf("<think>"); fflush(stdout); }
            auto out = eng.generate(n_predict, sp, stop, on_tok, &sc);
            fwrite(sc.pending_utf8.data(), 1, sc.pending_utf8.size(), stdout);
            printf("\n");
            ctx_tokens.insert(ctx_tokens.end(), out.begin(), out.end());
            if (!raw) {
                std::string content = (think ? "<think>" : "<think></think>") + sc.text;
                msgs.push_back({"assistant", content});
                // what the template renders after an assistant turn: <|im_end|>\n
                if (out.empty() || out.back() != tk.eos) ctx_tokens.push_back(-1);  // force re-render mismatch
            }
            if (!quiet) {
                const auto& t = eng.timing;
                fprintf(stderr, "[liquid] prompt %d tok %.1f ms (%.1f t/s) | gen %d tok %.1f ms (%.2f t/s) | ctx %d\n",
                        t.prefill_tokens, t.prefill_ms, t.prefill_tokens / (t.prefill_ms / 1e3 + 1e-9),
                        t.decode_tokens, t.decode_ms, t.decode_tokens / (t.decode_ms / 1e3 + 1e-9), eng.n_past() + 1);
            }
        };

        if (!interactive) {
            if (prompt.empty()) { usage(); return 1; }
            run_turn(prompt);
            return 0;
        }
        std::string line;
        while (true) {
            printf("\n> ");
            fflush(stdout);
            if (!std::getline(std::cin, line)) break;
            if (line == "/exit" || line == "/quit") break;
            if (line == "/reset") { msgs.clear(); ctx_tokens.clear(); eng.reset(); continue; }
            if (line.empty()) continue;
            run_turn(line);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
