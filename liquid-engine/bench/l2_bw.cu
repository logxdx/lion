// L2 read bandwidth: every SM streams a buffer that fits in L2 (after a first pass), 16-byte loads.
//   nvcc -O3 -arch=sm_89 bench/l2_bw.cu -o build/l2_bw && build/l2_bw
#include <cstdio>
#include <cuda_runtime.h>

__global__ void k_read(const int4* __restrict__ p, size_t n, int reps, int* out) {
    int4 acc = make_int4(0, 0, 0, 0);
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int r = 0; r < reps; r++)
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) {
            int4 v;
            asm volatile("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) out[0] = 1;
}

int main() {
    int* out;
    cudaMalloc(&out, 4);
    for (size_t mb : {4, 8, 16, 64, 256}) {
        const size_t bytes = mb << 20, n = bytes / 16;
        int4* p;
        cudaMalloc(&p, bytes);
        cudaMemset(p, 1, bytes);
        const int reps = mb <= 16 ? 50 : 5;
        for (int blocks : {20 * 4, 20 * 8}) {
            k_read<<<blocks, 256>>>(p, n, 1, out);  // warm (and fill L2)
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a);
            k_read<<<blocks, 256>>>(p, n, reps, out);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            printf("%4zu MB buffer, %3d CTAs x 256: %7.1f GB/s\n", mb, blocks, (double)bytes * reps / ms / 1e6);
        }
        cudaFree(p);
    }
    return 0;
}
