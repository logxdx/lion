// k_mhc vs the split small-batch version (k_mhc_a + k_mhc_b) on identical random inputs: every output compared.
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc tests/mhc_check.cu -o build/mhc_check && build/mhc_check
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernels.cuh"

using namespace raon;

static float frand() { return (rand() % 20001 - 10000) * 1e-4f; }
template <class T>
static T* up(const std::vector<T>& v) {
    T* d;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T>
static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}
static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double md = 0, mv = 0;
    for (size_t i = 0; i < a.size(); i++) { md = fmax(md, fabs(a[i] - b[i])); mv = fmax(mv, fabs(a[i])); }
    return mv > 0 ? md / mv : md;
}

int main() {
    const int R = 4, E = 1536, HC = 4, MIX = 17;
    std::vector<float> X((size_t)R * HC * E), A((size_t)R * E), aw(E), Y((size_t)R * E), post(R * HC), comb(R * 16), base(MIX),
        chart(MIX * 16 + 32), nw(E);
    std::vector<half> fn((size_t)MIX * HC * E);
    for (auto& v : X) v = frand() * 3;
    for (auto& v : A) v = frand();
    for (auto& v : aw) v = 1 + frand() * 0.1f;
    for (auto& v : Y) v = frand();
    for (auto& v : post) v = frand();
    for (auto& v : comb) v = frand() * 0.5f;
    for (auto& v : base) v = frand();
    for (auto& v : chart) v = frand();
    for (auto& v : nw) v = 1 + frand() * 0.1f;
    for (auto& v : fn) v = __float2half(frand() * 0.05f);
    int fails = 0;
    for (int mode = 0; mode < 3; mode++) {   // 0: add (normed) + pre; 1: post + pre; 2: post + stream sum (int8), no pre
        float* out[2][6];
        int8_t* q[2][2];
        for (int v = 0; v < 2; v++) {
            MhcArgs a{};
            a.X = up(X); a.R = R;
            if (mode == 0) { a.A = up(A); a.aw = up(aw); }
            if (mode >= 1) a.Y = up(Y);
            a.post = up(post); a.comb = up(comb);
            if (mode == 2) {
                a.S = up(std::vector<float>((size_t)R * E)); a.sscale = 1.f;
                cudaMalloc(&a.Sq, R * E); a.Ss = up(std::vector<float>(R * E / 32));
            } else {
                a.fn = up(fn); a.base = up(base); a.chart = up(chart); a.nw = up(nw);
                a.H = up(std::vector<float>((size_t)R * E));
                cudaMalloc(&a.Hq, R * E); a.Hs = up(std::vector<float>(R * E / 32));
            }
            a.hc_eps = 1e-5f; a.eps = 1e-6f; a.mag = 1.f;
            if (v == 0) {
                k_mhc<<<R, 192>>>(a);
            } else {
                float* part;
                cudaMalloc(&part, R * MHC_NB * 18 * 4);
                k_mhc_a<<<dim3(R, MHC_NB), 192>>>(a, part);
                if (a.fn) k_mhc_b<<<R, 192>>>(a, part);
            }
            cudaDeviceSynchronize();
            out[v][0] = a.X; out[v][1] = a.post; out[v][2] = a.comb; out[v][3] = a.H; out[v][4] = a.S; out[v][5] = a.Hs ? a.Hs : a.Ss;
            q[v][0] = a.Hq; q[v][1] = a.Sq;
        }
        const char* names[6] = {"X", "post", "comb", "H", "S", "Hs/Ss"};
        const size_t ns[6] = {(size_t)R * HC * E, (size_t)R * HC, (size_t)R * 16, (size_t)R * E, (size_t)R * E, (size_t)R * E / 32};
        printf("mode %d:", mode);
        for (int k = 0; k < 6; k++) {
            if (!out[0][k]) continue;
            const double d = rel(down(out[0][k], ns[k]), down(out[1][k], ns[k]));
            printf("  %s %.2g", names[k], d);
            fails += d > 1e-4;
        }
        for (int k = 0; k < 2; k++) {
            if (!q[0][k]) continue;
            auto a8 = down(q[0][k], (size_t)R * E), b8 = down(q[1][k], (size_t)R * E);
            int diff = 0;
            for (size_t i = 0; i < a8.size(); i++) diff += a8[i] != b8[i];
            printf("  %s int8 differ %d/%zu", k ? "Sq" : "Hq", diff, a8.size());
            fails += diff > (int)a8.size() / 100;
        }
        printf("\n");
    }
    printf(fails ? "FAIL\n" : "ok\n");
    return fails != 0;
}
