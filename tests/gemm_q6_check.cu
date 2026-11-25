// k_gemm_q6 vs k_gemv_q6 (same int8 activations) on Raon shapes; timing.
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc tests/gemm_q6_check.cu -o build/gemm_q6_check && build/gemm_q6_check [T]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gemm.cuh"
#include "kernels.cuh"

using namespace raon;

template <class Tp>
static Tp* up(const std::vector<Tp>& v) {
    Tp* d;
    CK(cudaMalloc(&d, v.size() * sizeof(Tp)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(Tp), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 77;
    struct S { const char* name; int N, K; } shapes[] = {
        {"ffn_up 1536->12288", 12288, 1536}, {"ffn_down 6144->1536", 1536, 6144}, {"rda_qkv 1536->8192", 8192, 1536},
        {"rda_out 2048->1536", 1536, 2048},  {"q_b 512->6144", 6144, 512},        {"ple_proj 256->1536", 1536, 256},
        {"mla_out 4096->1536", 1536, 4096},  {"q_a 1536->512", 512, 1536},        {"ple_gate 1536->256", 256, 1536}};
    CK(cudaFuncSetAttribute(k_gemm_q6, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(gq6::Smem)));
    for (auto& s : shapes) {
        const int N = s.N, K = s.K;
        const size_t NK = (size_t)N * K;
        std::vector<uint8_t> ql(NK / 2), qh(NK / 4);
        std::vector<int8_t> sc(NK / 16);
        std::vector<half> d(NK / 256);
        std::vector<float> hx((size_t)T * K);
        for (auto& v : ql) v = (uint8_t)rand();
        for (auto& v : qh) v = (uint8_t)rand();
        for (auto& v : sc) v = (int8_t)(rand() % 255 - 127);
        for (auto& v : d) v = __float2half(1e-4f + (rand() % 100) * 1e-6f);
        for (auto& v : hx) v = (rand() % 2001 - 1000) * 1e-3f;
        uint8_t *dl = up(ql), *dh = up(qh);
        int8_t* dsc = up(sc);
        half* dd = up(d);
        float* x = up(hx);
        int8_t* xq; float *xs, *y1, *y2;
        CK(cudaMalloc(&xq, (size_t)T * K)); CK(cudaMalloc(&xs, (size_t)T * K / 32 * 4));
        CK(cudaMalloc(&y1, (size_t)T * N * 4)); CK(cudaMalloc(&y2, (size_t)T * N * 4));
        k_quantize<<<(T * K / 32 * 32 + 255) / 256, 256>>>(x, K, K, T, xq, xs);
        for (int b0 = 0; b0 < T; b0++)
            k_gemv_q6<1><<<(N + 3) / 4, 128>>>((const int8_t*)dl, dh, dsc, dd, N, K, xq + (size_t)b0 * K,
                                              xs + (size_t)b0 * K / 32, y1 + (size_t)b0 * N, N);
        dim3 grid((T + 127) / 128, N / 128);
        k_gemm_q6<<<grid, gq6::THREADS, sizeof(gq6::Smem)>>>(xq, xs, dl, dh, dsc, dd, T, N, K, y2, N);
        CK(cudaDeviceSynchronize());
        std::vector<float> r1((size_t)T * N), r2((size_t)T * N);
        CK(cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(r2.data(), y2, r2.size() * 4, cudaMemcpyDeviceToHost));
        double maxd = 0, maxv = 0;
        for (size_t i = 0; i < r1.size(); i++) { maxd = fmax(maxd, fabs(r1[i] - r2[i])); maxv = fmax(maxv, fabs(r1[i])); }
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 10; i++) k_gemm_q6<<<grid, gq6::THREADS, sizeof(gq6::Smem)>>>(xq, xs, dl, dh, dsc, dd, T, N, K, y2, N);
        cudaEventRecord(a);
        for (int i = 0; i < 30; i++) k_gemm_q6<<<grid, gq6::THREADS, sizeof(gq6::Smem)>>>(xq, xs, dl, dh, dsc, dd, T, N, K, y2, N);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        ms /= 30;
        printf("T=%d %-22s max|diff| %.3g (max|y| %.3g, rel %.2g)  %.3f ms %.1f TOPS\n", T, s.name, maxd, maxv,
               maxd / maxv, ms, 2.0 * T * N * K / (ms * 1e-3) / 1e12);
        cudaFree(dl); cudaFree(dh); cudaFree(dsc); cudaFree(dd); cudaFree(x); cudaFree(xq); cudaFree(xs); cudaFree(y1);
        cudaFree(y2);
    }
    return 0;
}
