// Compare src/vision.cu embeddings with llama.cpp's (tests/vl_ref_embed output).
// usage: test_vision mmproj.gguf image ref.bin [ref.bin is optional: without it, only encodes and times]
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "../src/vision.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s mmproj.gguf image [ref.bin]\n", argv[0]);
        return 2;
    }
    le::Vision v(argv[1]);
    std::ifstream in(argv[2], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    printf("read %zu bytes\n", bytes.size());
    le::Image im;
    std::string err;
    if (!le::decode_image(bytes.data(), bytes.size(), im, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    int tw, th, n = 0;
    v.target_size(im.w, im.h, tw, th);
    std::vector<float> e = v.encode_host(im, n);
    auto t0 = std::chrono::steady_clock::now();
    v.encode_host(im, n);  // second run: warm timing
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("image %dx%d -> %dx%d, %d tokens, encode %.1f ms GPU, %.1f ms with resize\n", im.w, im.h, tw, th, n,
           v.last_ms(), wall);
    if (const char* sp = getenv("VT_SAVE")) {  // same format as vl_ref_embed
        FILE* f = fopen(sp, "wb");
        int hdr[2] = {n, 2048};
        fwrite(hdr, 4, 2, f);
        fwrite(e.data(), 4, e.size(), f);
        fclose(f);
    }
    if (argc < 4) return 0;
    std::ifstream rf(argv[3], std::ios::binary);
    int hdr[2];
    rf.read((char*)hdr, 8);
    if (hdr[0] != n || hdr[1] != 2048) { printf("FAIL token count: ref %d x %d, ours %d\n", hdr[0], hdr[1], n); return 1; }
    std::vector<float> r((size_t)n * 2048);
    rf.read((char*)r.data(), r.size() * 4);
    double worst = 1, sum_cos = 0, num = 0, den = 0, maxabs = 0;
    int worst_t = 0;
    for (int t = 0; t < n; t++) {
        double d = 0, a = 0, b = 0;
        for (int i = 0; i < 2048; i++) {
            const double x = e[(size_t)t * 2048 + i], y = r[(size_t)t * 2048 + i];
            d += x * y; a += x * x; b += y * y;
            num += (x - y) * (x - y); den += y * y;
            maxabs = std::fmax(maxabs, std::fabs(x - y));
        }
        const double c = d / std::sqrt(a * b + 1e-30);
        sum_cos += c;
        if (c < worst) { worst = c; worst_t = t; }
    }
    if (getenv("VT_GRID")) {  // per-token cosine on the merged grid: digit = floor((1 - cos) * 1000), capped at 9
        const int mx = tw / 32, my = th / 32;
        for (int y = 0; y < my; y++) {
            for (int x = 0; x < mx; x++) {
                const int t = y * mx + x;
                double d = 0, a = 0, b = 0;
                for (int i = 0; i < 2048; i++) {
                    const double p = e[(size_t)t * 2048 + i], q = r[(size_t)t * 2048 + i];
                    d += p * q; a += p * p; b += q * q;
                }
                const int k = (int)std::fmin(9, (1 - d / std::sqrt(a * b)) * 1000);
                putchar('0' + k);
            }
            putchar('\n');
        }
    }
    const double rel = std::sqrt(num / den);
    printf("cosine mean %.6f, worst %.6f (token %d), rel L2 %.4f, max |diff| %.4f\n", sum_cos / n, worst, worst_t, rel,
           maxabs);
    // llama.cpp CPU vs its own CUDA backend on test-1.jpeg: cos mean 0.99852, worst 0.911, rel L2 0.053
    const bool ok = sum_cos / n >= 0.997 && rel <= 0.08;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
