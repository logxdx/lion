// raon-tok: tokenize a file with the engine's tokenizer and print the ids, for checks against llama-tokenize.
//   raon-tok -m model.gguf -f text.txt [--no-special] [--decode] [--time]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "tokenizer.h"

int main(int argc, char** argv) {
    std::string model, file;
    bool special = true, decode = false, timing = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-m") model = argv[++i];
        else if (a == "-f") file = argv[++i];
        else if (a == "--no-special") special = false;
        else if (a == "--decode") decode = true;
        else if (a == "--time") timing = true;
    }
    if (model.empty() || file.empty()) { fprintf(stderr, "usage: raon-tok -m model.gguf -f file [--no-special]\n"); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    raon::Tokenizer tok(model);
    auto t1 = std::chrono::steady_clock::now();
    std::ifstream in(file, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    auto ids = tok.encode(text, special);
    auto t2 = std::chrono::steady_clock::now();
    printf("[");
    for (size_t i = 0; i < ids.size(); i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n");
    if (decode) printf("roundtrip %s\n", tok.decode(ids) == text ? "exact" : "differs");
    if (timing)
        fprintf(stderr, "load %.0f ms, encode %zu bytes -> %zu tokens in %.1f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(), text.size(), ids.size(),
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    return 0;
}
