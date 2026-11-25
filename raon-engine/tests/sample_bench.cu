// Time of the GPU sampling passes (k_topk1 + k_topk2) on random logits rows of the Raon vocabulary.
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc tests/sample_bench.cu -o build/sample_bench && build/sample_bench
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sampling.cuh"

using namespace raon;

int main() {
    const int V = 259840;
    for (int nout : {1, 4, 6}) {
        std::vector<float> h((size_t)nout * V), t(nout, 0.8f);
        for (auto& x : h) x = (rand() % 200000) * 1e-4f - 10.f;
        float *lg, *temp, *ck, *stats, *ok, *olse;
        int *ci, *oi;
        cudaMalloc(&lg, h.size() * 4);
        cudaMemcpy(lg, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&temp, nout * 4);
        cudaMemcpy(temp, t.data(), nout * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&ck, (size_t)nout * smp::CK_PER_ROW * 4);
        cudaMalloc(&ci, (size_t)nout * smp::CK_PER_ROW * 4);
        cudaMalloc(&stats, (size_t)nout * smp::CHUNKS * 3 * 4);
        cudaMalloc(&ok, (size_t)nout * smp::SAMPLE_K * 4);
        cudaMalloc(&oi, (size_t)nout * smp::SAMPLE_K * 4);
        cudaMalloc(&olse, nout * 2 * 4);
        cudaFuncSetAttribute(smp::k_topk2, cudaFuncAttributeMaxDynamicSharedMemorySize, smp::K2_SMEM);
        auto run = [&] {
            smp::k_topk1<<<dim3(smp::CHUNKS, nout), smp::K1_THREADS>>>(lg, V, temp, ck, ci, stats);
            smp::k_topk2<<<nout, 1024, smp::K2_SMEM>>>(ck, ci, stats, temp, ok, oi, olse);
        };
        for (int i = 0; i < 50; i++) run();
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < 200; i++) run();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        printf("rows %d: %.1f us per sampling pass (k_topk1 + k_topk2)\n", nout, ms / 200 * 1000);
    }
    return 0;
}
