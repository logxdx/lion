// k_rda_scan against an f64 CPU reference of the gated delta rule on random inputs: outputs and final state.
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc tests/scan_check.cu -o build/scan_check && build/scan_check [T]
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

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 64, H = 16, D = 128, C = H * D;
    std::vector<float> q((size_t)T * C), k((size_t)T * C), v((size_t)T * C), g((size_t)T * C), b(T * H), S0((size_t)H * D * D);
    auto l2 = [&](std::vector<float>& x) {
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++) {
                double ss = 0;
                for (int i = 0; i < D; i++) ss += x[(size_t)t * C + h * D + i] * x[(size_t)t * C + h * D + i];
                for (int i = 0; i < D; i++) x[(size_t)t * C + h * D + i] /= sqrt(ss);
            }
    };
    for (auto& x : q) x = frand();
    for (auto& x : k) x = frand();
    l2(q); l2(k);
    for (auto& x : v) x = frand() * 2;
    for (auto& x : g) x = -5.f * (0.5f + 0.5f * frand()) * 0.2f;   // log-decay in (-2, 0]
    for (auto& x : b) x = 0.5f + 0.5f * frand();
    for (auto& x : S0) x = frand() * 0.3f;
    Seg seg{0, 0, T};
    Seg* dseg = up(std::vector<Seg>{seg});
    float *dq = up(q), *dk = up(k), *dv = up(v), *dg = up(g), *db = up(b);
    std::vector<float> outs[1], states[1];
    for (int ver = 0; ver < 1; ver++) {
        float* dS = up(S0);
        float* dout = up(std::vector<float>((size_t)T * C));
        k_rda_scan<<<dim3(1, H, 4), 128>>>(dseg, dq, dk, dv, dg, db, dS, 0, H, dout);
        cudaDeviceSynchronize();
        outs[ver] = down(dout, (size_t)T * C);
        states[ver] = down(dS, (size_t)H * D * D);
    }
    // f64 reference
    std::vector<double> S(S0.begin(), S0.end()), ref((size_t)T * C);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++) {
            double* Sh = &S[(size_t)h * D * D];
            const size_t o = (size_t)t * C + h * D;
            for (int i = 0; i < D; i++)
                for (int j = 0; j < D; j++) Sh[i * D + j] *= exp((double)g[o + i]);
            for (int j = 0; j < D; j++) {
                double kv = 0;
                for (int i = 0; i < D; i++) kv += Sh[i * D + j] * k[o + i];
                const double dl = b[t * H + h] * (v[o + j] - kv);
                double acc = 0;
                for (int i = 0; i < D; i++) { Sh[i * D + j] += k[o + i] * dl; acc += Sh[i * D + j] * q[o + i]; }
                ref[o + j] = acc / sqrt((double)D);
            }
        }
    auto err = [&](const std::vector<float>& a, const std::vector<double>& r) {
        double md = 0, mv = 0;
        for (size_t i = 0; i < a.size(); i++) { md = fmax(md, fabs(a[i] - r[i])); mv = fmax(mv, fabs(r[i])); }
        return md / mv;
    };
    printf("k_rda_scan: out max rel err vs f64 %.3g, state %.3g\n", err(outs[0], ref), err(states[0], S));
    return 0;
}
