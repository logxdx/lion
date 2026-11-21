// Peak tensor throughput microbenchmark (register-only mma loops) for the instruction shapes the
// engine uses. build: nvcc -O3 -arch=sm_89 bench/mma_peak.cu -o build/mma_peak
#include <cstdio>
#include <cstdint>

template <int KIND>
__global__ void k(int* out, int iters) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
    uint32_t b0 = threadIdx.x * 11u, b1 = threadIdx.x * 13u;
    int c[8][4] = {};
    float f[8][4] = {};
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (KIND == 0)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            else if (KIND == 1)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                             : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                             : "r"(a[0]), "r"(a[1]), "r"(b0));
            else
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(f[j][0]), "+f"(f[j][1]), "+f"(f[j][2]), "+f"(f[j][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
        }
    }
    int s = 0;
    for (int j = 0; j < 8; j++) s += c[j][0] + c[j][3] + (int)f[j][0];
    if (s == 0x7fffffff) out[0] = s;
}

int main() {
    int* o;
    cudaMalloc(&o, 4);
    const char* names[3] = {"int8 m16n8k32", "int8 m16n8k16", "f16 m16n8k16 (f32 acc)"};
    const double macs[3] = {16 * 8 * 32, 16 * 8 * 16, 16 * 8 * 16};
    for (int kind = 0; kind < 3; kind++) {
        for (int warps : {4, 8, 16}) {
            const int iters = 20000, blocks = 20 * 4;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            auto launch = [&]() {
                if (kind == 0) k<0><<<blocks, warps * 32>>>(o, iters);
                else if (kind == 1) k<1><<<blocks, warps * 32>>>(o, iters);
                else k<2><<<blocks, warps * 32>>>(o, iters);
            };
            launch();
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            const double ops = 2.0 * macs[kind] * 8 * iters * blocks * warps;
            printf("%-24s warps/CTA %2d: %6.1f TOPS\n", names[kind], warps, ops / (ms * 1e-3) / 1e12);
        }
    }
}
