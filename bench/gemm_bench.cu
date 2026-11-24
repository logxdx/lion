// Prefill GEMM microbenchmark: the engine's k_gemm (src/gemm.cuh) on the LFM3-1.7B layer shapes, random data.
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc bench/gemm_bench.cu -o build/gemm_bench && build/gemm_bench [q8|q6] [T...]
// v1 = launch_gemm_ref (the original kernel), v2 = launch_gemm (what the engine runs); outputs compared byte for byte.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "gemm.cuh"

using namespace le;

static void* dalloc(size_t n, int fill = -1) {
    void* p = nullptr;
    cudaMalloc(&p, n);
    std::vector<uint8_t> h(n);
    for (size_t i = 0; i < n; i++) h[i] = fill >= 0 ? (uint8_t)fill : (uint8_t)(rand() & 0xff);
    cudaMemcpy(p, h.data(), n, cudaMemcpyHostToDevice);
    return p;
}

static QWeight make_w(int type, int N, int K) {
    QWeight w;
    w.type = type; w.N = N; w.K = K;
    if (type == W_Q8) {
        w.q = (uint8_t*)dalloc((size_t)N * K);
        std::vector<half> d((size_t)N * K / 32, __float2half(0.01f));
        cudaMalloc(&w.d, d.size() * 2);
        cudaMemcpy(w.d, d.data(), d.size() * 2, cudaMemcpyHostToDevice);
    } else {
        w.q = (uint8_t*)dalloc((size_t)N * K / 2);
        w.h = (uint8_t*)dalloc((size_t)N * K / 4);
        w.sc = (int8_t*)dalloc((size_t)N * K / 16, 3);
        std::vector<half> d((size_t)N * K / 256, __float2half(0.01f));
        cudaMalloc(&w.d, d.size() * 2);
        cudaMemcpy(w.d, d.data(), d.size() * 2, cudaMemcpyHostToDevice);
    }
    return w;
}

template <int K, int EPI>
static void launch(int v, const GemmArgs& g) {  // 1: the original kernel, 2: what the engine launches
    if (v == 1) launch_gemm_ref<K, EPI>(g, 0);
    else launch_gemm<K, EPI>(g, 0);
}

template <int K, int EPI>
static double time_gemm(GemmArgs g, int reps, int v) {
    for (int i = 0; i < 3; i++) launch<K, EPI>(v, g);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) launch<K, EPI>(v, g);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "CUDA error\n"); exit(1); }
    return ms / reps;
}

int main(int argc, char** argv) {
    const int type = argc > 1 && !strcmp(argv[1], "q6") ? W_Q6 : W_Q8;
    std::vector<int> Ts;
    for (int i = 2; i < argc; i++) Ts.push_back(atoi(argv[i]));
    if (Ts.empty()) Ts = {361, 512, 4096};
    const int TMAX = 4096;
    int8_t* A = (int8_t*)dalloc((size_t)TMAX * 7680);
    std::vector<float> ad((size_t)TMAX * 7680 / 32, 0.01f);
    float* Ad;
    cudaMalloc(&Ad, ad.size() * 4);
    cudaMemcpy(Ad, ad.data(), ad.size() * 4, cudaMemcpyHostToDevice);
    float* out;
    cudaMalloc(&out, (size_t)TMAX * 15360 * 4);
    int8_t* yq;
    float* yd;
    cudaMalloc(&yq, (size_t)TMAX * 7680);
    cudaMalloc(&yd, (size_t)TMAX * 7680 / 32 * 4);

    struct Shape { const char* name; int N, K, epi; };
    const Shape shapes[] = {{"in_proj 2048->6144", 6144, 2048, GE_STORE},
                            {"out     2048->2048", 2048, 2048, GE_STORE},
                            {"gate_up 2048->15360", 15360, 2048, GE_SWIGLU},
                            {"down    7680->2048", 2048, 7680, GE_STORE}};
    printf("%s weights\n", type == W_Q8 ? "Q8_0" : "Q6_K");
    for (int T : Ts) {
        double tot_ms = 0, tot_ops = 0;
        for (const auto& s : shapes) {
            QWeight w = make_w(type, s.N, s.K);
            GemmArgs g{};
            g.A = A; g.Ad = Ad; g.T = T; g.w = w; g.out = out; g.yq = yq; g.yd = yd;
            auto timed = [&](int v) {
                if (s.K == 2048 && s.epi == GE_STORE) return time_gemm<2048, GE_STORE>(g, 20, v);
                if (s.K == 2048) return time_gemm<2048, GE_SWIGLU>(g, 20, v);
                return time_gemm<7680, GE_STORE>(g, 20, v);
            };
            // output of each kernel (bytes), compared
            auto output = [&](int v) {
                cudaMemset(out, 0, (size_t)TMAX * 15360 * 4);
                cudaMemset(yq, 0, (size_t)TMAX * 7680);
                cudaMemset(yd, 0, (size_t)TMAX * 7680 / 32 * 4);
                if (s.K == 2048 && s.epi == GE_STORE) launch<2048, GE_STORE>(v, g);
                else if (s.K == 2048) launch<2048, GE_SWIGLU>(v, g);
                else launch<7680, GE_STORE>(v, g);
                std::vector<uint8_t> h;
                if (s.epi == GE_STORE) {
                    h.resize((size_t)T * s.N * 4);
                    cudaMemcpy(h.data(), out, h.size(), cudaMemcpyDeviceToHost);
                } else {
                    h.resize((size_t)T * s.N / 2 + (size_t)T * s.N / 64 * 4);
                    cudaMemcpy(h.data(), yq, (size_t)T * s.N / 2, cudaMemcpyDeviceToHost);
                    cudaMemcpy(h.data() + (size_t)T * s.N / 2, yd, (size_t)T * s.N / 64 * 4, cudaMemcpyDeviceToHost);
                }
                return h;
            };
            std::vector<int> vs = {1};
            if (type == W_Q8) vs = {1, 2};
            const auto ref = output(1);
            std::vector<double> ms(vs.size(), 0.0);
            for (int r = 0; r < 2; r++)  // interleaved: thermal drift hits all variants alike
                for (size_t i = 0; i < vs.size(); i++) ms[i] += timed(vs[i]) / 2;
            const double ops = 2.0 * T * s.N * s.K;
            printf("T=%5d  %-20s", T, s.name);
            for (size_t i = 0; i < vs.size(); i++)
                printf(" | v%d %6.1f TOPS%s", vs[i], ops / ms[i] / 1e9, i && output(vs[i]) != ref ? " DIFF" : "");
            printf("\n");
            const double best = *std::min_element(ms.begin() + (vs.size() > 1), ms.end());
            tot_ms += best;
            tot_ops += ops;
            cudaFree(w.q); cudaFree(w.d); if (w.h) cudaFree(w.h); if (w.sc) cudaFree(w.sc);
        }
        printf("T=%5d  all four (newest)    %8.3f ms  %6.1f TOPS\n\n", T, tot_ms, tot_ops / tot_ms / 1e9);
    }
    return 0;
}
