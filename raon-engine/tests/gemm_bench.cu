// Prefill Q8_0 GEMM check + benchmark: k_gemm_q8 / k_gemm_q8v<KS> vs the GEMV reference (same int8 activations).
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc tests/gemm_bench.cu -o build/gemm_bench && build/gemm_bench [T]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gemm.cuh"
#include "kernels.cuh"

using namespace raon;

template <class F>
static float timeit(F f, int reps) {
    for (int i = 0; i < 5; i++) f();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 2048;
    struct S { const char* name; int N, K; } shapes[] = {
        {"ffn_up 1536->12288", 12288, 1536}, {"ffn_down 6144->1536", 1536, 6144}, {"rda_qkv 1536->8192", 8192, 1536},
        {"rda_out 2048->1536", 1536, 2048},  {"q_b 512->6144", 6144, 512},        {"mla_out 4096->1536", 1536, 4096},
        {"q_a 1536->512", 512, 1536},        {"ple_gate 1536->256", 256, 1536},   {"attn_gate 1536->4096", 4096, 1536}};
    CK(cudaFuncSetAttribute(k_gemm_q8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(gq8::Smem)));
    CK(cudaFuncSetAttribute(k_gemm_q8v<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(GQ8V<2>)));
    CK(cudaFuncSetAttribute(k_gemm_q8v<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(GQ8V<4>)));
    for (auto& s : shapes) {
        const int N = s.N, K = s.K;
        std::vector<int8_t> hw((size_t)N * K);
        std::vector<half> hd((size_t)N * K / 32);
        std::vector<float> hx((size_t)T * K);
        for (auto& v : hw) v = (int8_t)(rand() % 255 - 127);
        for (auto& v : hd) v = __float2half(0.001f + (rand() % 100) * 1e-5f);
        for (auto& v : hx) v = (rand() % 2001 - 1000) * 1e-3f;
        int8_t *wq, *xq; half* wd; float *x, *xs, *y1, *y2;
        CK(cudaMalloc(&wq, hw.size())); CK(cudaMalloc(&wd, hd.size() * 2)); CK(cudaMalloc(&x, hx.size() * 4));
        CK(cudaMalloc(&xq, (size_t)T * K)); CK(cudaMalloc(&xs, (size_t)T * K / 32 * 4));
        CK(cudaMalloc(&y1, (size_t)T * N * 4)); CK(cudaMalloc(&y2, (size_t)T * N * 4));
        CK(cudaMemcpy(wq, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(wd, hd.data(), hd.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        k_quantize<<<(T * K / 32 * 32 + 255) / 256, 256>>>(x, K, K, T, xq, xs);
        for (int b0 = 0; b0 < T; b0 += 8)
            k_gemv_q8<8><<<(N + 3) / 4, 128>>>(wq, wd, N, K, xq + (size_t)b0 * K, xs + (size_t)b0 * K / 32,
                                              y1 + (size_t)b0 * N, N);
        CK(cudaDeviceSynchronize());
        std::vector<float> r1((size_t)T * N), r2((size_t)T * N);
        CK(cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost));
        dim3 grid((T + 127) / 128, N / 128);
        printf("T=%d %-22s", T, s.name);
        for (int v : {0, 2, 4}) {
            if (v && K % (32 * v)) { printf(" | KS=%d n/a", v); continue; }
            auto launch = [&]() {
                if (v == 0) k_gemm_q8<<<grid, gq8::THREADS, sizeof(gq8::Smem)>>>(xq, xs, wq, wd, T, N, K, y2, N);
                else if (v == 2) k_gemm_q8v<2><<<grid, 256, sizeof(GQ8V<2>)>>>(xq, xs, wq, wd, T, N, K, y2, N);
                else k_gemm_q8v<4><<<grid, 256, sizeof(GQ8V<4>)>>>(xq, xs, wq, wd, T, N, K, y2, N);
            };
            launch();
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(r2.data(), y2, r2.size() * 4, cudaMemcpyDeviceToHost));
            double maxd = 0, maxv = 0;
            for (size_t i = 0; i < r1.size(); i++) { maxd = fmax(maxd, fabs(r1[i] - r2[i])); maxv = fmax(maxv, fabs(r1[i])); }
            const float ms = timeit(launch, 30);
            printf(" | %s %5.1f TOPS%s", v == 0 ? "classic" : (v == 2 ? "KS=2" : "KS=4"), 2.0 * T * N * K / (ms * 1e-3) / 1e12,
                   maxd / maxv > 1e-5 ? " MISMATCH" : "");
        }
        printf("\n");
        cudaFree(wq); cudaFree(wd); cudaFree(x); cudaFree(xq); cudaFree(xs); cudaFree(y1); cudaFree(y2);
    }
    return 0;
}
