// Video path check: decode (src/video.cpp) + pair encoding (src/vision.cu) vs llama.cpp's pair embeddings
// (tests/vl_ref_video output). usage: test_video mmproj.gguf video [ref_pairs.bin]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include <cuda_runtime.h>

#include "../src/video.h"
#include "../src/vision.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s mmproj.gguf video [ref_pairs.bin]\n", argv[0]);
        return 2;
    }
    le::Video v;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!le::decode_video_file(argv[2], v, err)) { fprintf(stderr, "decode: %s\n", err.c_str()); return 1; }
    const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("source %dx%d, %d frames at %.3f fps -> %zu frames %dx%d in %.0f ms\n", v.src_w, v.src_h, v.src_frames, v.src_fps,
           v.frames.size(), v.frames[0].w, v.frames[0].h, dec_ms);
    printf("indices:");
    for (int i : v.indices) printf(" %d", i);
    printf("\nstamps:");
    for (auto& s : v.stamps) printf(" %s", s.c_str());
    printf("\n");
    if (const char* dir = getenv("DUMP_FRAMES")) {  // DUMP_FRAMES=dir: frames as dir/<source index + 1>.ppm
        for (size_t i = 0; i < v.frames.size(); i++) {
            char path[512];
            snprintf(path, sizeof path, "%s/%03d.ppm", dir, v.indices[i] + 1);
            FILE* o = fopen(path, "wb");
            if (!o) continue;
            fprintf(o, "P6\n%d %d\n255\n", v.frames[i].w, v.frames[i].h);
            fwrite(v.frames[i].rgb.data(), 1, v.frames[i].rgb.size(), o);
            fclose(o);
        }
    }

    le::Vision vis(argv[1], false);
    std::vector<std::vector<float>> pairs;
    double enc_ms = 0;
    for (int g = 0; g < v.n_pairs(); g++) {
        const le::Image* f1 = 2 * g + 1 < (int)v.frames.size() ? &v.frames[2 * g + 1] : nullptr;
        int n = 0;
        const float* d = vis.encode_pair(v.frames[2 * g], f1, n);
        enc_ms += vis.last_ms();
        std::vector<float> h((size_t)n * 2048);
        cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost);
        pairs.push_back(std::move(h));
    }
    printf("%d pairs x %zu tokens, encode %.1f ms total\n", v.n_pairs(), pairs[0].size() / 2048, enc_ms);

    auto fetch = [](const float* d, int n) {
        std::vector<float> h((size_t)n * 2048);
        cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost);
        return h;
    };
    auto cmp = [](const std::vector<float>& e, const std::vector<float>& r, double& mean, double& worst, double& rel) {
        const int n = (int)(r.size() / 2048);
        if (n == 0 || e.size() != r.size()) { mean = worst = rel = NAN; return; }
        double sum = 0, num = 0, den = 0;
        worst = 1;
        for (int t = 0; t < n; t++) {
            double d = 0, a = 0, b = 0;
            for (int i = 0; i < 2048; i++) {
                const double x = e[(size_t)t * 2048 + i], y = r[(size_t)t * 2048 + i];
                d += x * y; a += x * x; b += y * y;
                num += (x - y) * (x - y); den += y * y;
            }
            const double c = d / std::sqrt(a * b + 1e-30);
            sum += c;
            worst = std::fmin(worst, c);
        }
        mean = sum / n;
        rel = std::sqrt(num / den);
    };
    bool ok = true;
    {  // identity: a pair of identical frames == the still image, conv(f,k0) + conv(f,k1) == conv(f,k0+k1). Exact up to
       // f32 rounding at the patch embedding (checked there, via the encoder's LIQUID_VDUMP stage dump); downstream the
       // int8 activation quantization amplifies that rounding to ~3% rel L2 at the output (measured), so the end-to-end
       // comparison only gets a loose bound.
        char tmpl[] = "/tmp/liquid-vtest-XXXXXX";
        if (!mkdtemp(tmpl)) { perror("mkdtemp"); return 1; }
        const std::string dir = tmpl, pfile = dir + "/patchifier_out.bin";
        auto patch_out = [&]() {
            std::vector<float> h;
            FILE* pf = fopen(pfile.c_str(), "rb");
            int hdr[2];
            if (pf && fread(hdr, 4, 2, pf) == 2) {
                h.resize((size_t)hdr[0] * hdr[1]);
                if (fread(h.data(), 4, h.size(), pf) != h.size()) h.clear();
            }
            if (pf) fclose(pf);
            return h;
        };
        setenv("LIQUID_VDUMP", dir.c_str(), 1);
        int n1 = 0, n2 = 0;
        const float* d1 = vis.encode(v.frames[0], n1);
        const auto still = fetch(d1, n1);
        const auto p1 = patch_out();
        const float* d2 = vis.encode_pair(v.frames[0], &v.frames[0], n2);
        const auto same = fetch(d2, n2);
        const auto p2 = patch_out();
        unsetenv("LIQUID_VDUMP");
        std::filesystem::remove_all(dir);
        double num = 0, den = 0;
        for (size_t i = 0; i < p1.size() && p1.size() == p2.size(); i++) {
            num += (double)(p2[i] - p1[i]) * (p2[i] - p1[i]);
            den += (double)p1[i] * p1[i];
        }
        const double prel = p1.empty() || p1.size() != p2.size() ? NAN : std::sqrt(num / den);
        double mean, worst, rel;
        cmp(same, still, mean, worst, rel);
        const bool iok = n1 == n2 && prel <= 1e-5 && rel <= 0.15;  // NaN fails; a real error is ~0.6 (swapped frames)
        ok &= iok;
        printf("identity (frame 0 twice vs still image): patch embedding rel L2 %.2e, output rel L2 %.4f %s\n", prel, rel,
               iok ? "ok" : "FAIL");
    }
    if (argc < 4) return ok ? 0 : 1;

    FILE* f = fopen(argv[3], "rb");
    int nref = 0;
    if (!f || fread(&nref, 4, 1, f) != 1) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
    if (nref != v.n_pairs()) { printf("FAIL pair count: ref %d, ours %d\n", nref, v.n_pairs()); return 1; }
    std::vector<float> ref0;
    for (int g = 0; g < nref; g++) {
        int hdr[2];
        if (fread(hdr, 4, 2, f) != 2) return 1;
        std::vector<float> r((size_t)hdr[0] * hdr[1]);
        if (fread(r.data(), 4, r.size(), f) != r.size()) return 1;
        const auto& e = pairs[g];
        if (e.size() != r.size()) { printf("FAIL pair %d: %zu vs %zu values\n", g, e.size(), r.size()); return 1; }
        double mean, worst, rel;
        cmp(e, r, mean, worst, rel);
        // int8-activation encoder vs llama.cpp f32 CPU: content-dependent. The still-image path on single frames of
        // this clip measures mean 0.9967 .. 0.9992, rel L2 0.033 .. 0.074, so the same band applies per pair.
        const bool pok = mean >= 0.995 && rel <= 0.09;
        ok &= pok;
        printf("pair %d (%s): cosine mean %.6f worst %.6f rel L2 %.4f %s\n", g, v.stamps[g].c_str(), mean, worst, rel,
               pok ? "ok" : "FAIL");
        if (g == 0) ref0 = std::move(r);
    }
    fclose(f);
    if (v.frames.size() >= 2) {  // frame order matters: swapped frames must land clearly further from the reference
        int n = 0;
        const float* d = vis.encode_pair(v.frames[1], &v.frames[0], n);
        const auto sw = fetch(d, n);
        double m0, w0, r0, m1, w1, r1;
        cmp(pairs[0], ref0, m0, w0, r0);
        cmp(sw, ref0, m1, w1, r1);
        const bool sok = r1 > 1.5 * r0;
        ok &= sok;
        printf("order: pair 0 rel L2 %.4f in order, %.4f swapped %s\n", r0, r1, sok ? "ok" : "FAIL");
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
