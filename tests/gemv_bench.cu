// GEMV microbenchmark: achieved DRAM bandwidth of the decode kernels on Raon-1 nano shapes.
// Weights rotate through a 640 MB pool (>> 24 MB L2), filled on the GPU; ~0.3 s of warm-up before timing.
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc tests/gemv_bench.cu -o build/gemv_bench && build/gemv_bench
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernels.cuh"

using namespace raon;

__global__ void k_fill(uint8_t* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = (uint8_t)(x & 0x3f);  // small values: scales stay finite
    }
}

static uint8_t* pool;
static const size_t POOL = 640ull << 20;

template <int B>
static double run(int type, int N, int K) {
    const size_t wq = (size_t)N * K, wqh = wq / 4, wsc = wq / 16, wd = wq / 32 * 2;
    const size_t one = type == 0 ? wq + wd : wq / 2 + wqh + wsc + wd;
    const int ncopy = (int)std::max<size_t>(1, POOL / (one + 4096));
    int8_t* xq; float *xs, *y;
    CK(cudaMalloc(&xq, (size_t)B * K));
    CK(cudaMalloc(&xs, (size_t)B * K / 32 * 4));
    CK(cudaMalloc(&y, (size_t)B * N * 4));
    CK(cudaMemset(xq, 1, (size_t)B * K));
    CK(cudaMemset(xs, 0, (size_t)B * K / 32 * 4));
    auto launch = [&](int i) {
        uint8_t* p = pool + (size_t)(i % ncopy) * ((one + 4095) / 4096 * 4096);
        const int blocks = (N + 3) / 4;
        if (type == 0) {
            k_gemv_q8<B><<<blocks, 128>>>((const int8_t*)p, (const half*)(p + wq), N, K, xq, xs, y, N);
        } else {
            k_gemv_q6<B><<<blocks, 128>>>((const int8_t*)p, p + wq / 2, (const int8_t*)(p + wq / 2 + wqh),
                                          (const half*)(p + wq / 2 + wqh + wsc), N, K, xq, xs, y, N);
        }
    };
    // warm-up ~0.3 s
    auto t0 = std::chrono::steady_clock::now();
    int i = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.3) {
        launch(i++);
        if (i % 16 == 0) cudaDeviceSynchronize();
    }
    const int reps = std::max(20, i);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int j = 0; j < reps; j++) launch(i + j);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    CK(cudaGetLastError());
    cudaFree(xq); cudaFree(xs); cudaFree(y);
    return (double)one / (ms / reps * 1e-3) / 1e9;
}

int main() {
    CK(cudaMalloc(&pool, POOL));
    k_fill<<<1024, 256>>>(pool, POOL);
    CK(cudaDeviceSynchronize());
    struct S { const char* name; int N, K; } shapes[] = {
        {"ffn_up 1536->12288", 12288, 1536}, {"ffn_down 6144->1536", 1536, 6144}, {"rda_qkv 1536->8192", 8192, 1536},
        {"rda_out 2048->1536", 1536, 2048},  {"ple_proj 256->1536", 1536, 256},    {"lm_head 1536->259840", 259840, 1536}};
    for (int type = 0; type < 2; type++) {
        printf("%s (DRAM 233 GB/s)\n", type == 0 ? "Q8_0" : "Q6_K");
        for (auto& s : shapes)
            printf("  %-22s B=1 %6.1f  B=2 %6.1f  B=4 %6.1f  B=6 %6.1f GB/s\n", s.name, run<1>(type, s.N, s.K),
                   run<2>(type, s.N, s.K), run<4>(type, s.N, s.K), run<6>(type, s.N, s.K));
    }
    return 0;
}
