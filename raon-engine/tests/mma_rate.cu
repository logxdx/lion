// Tensor-core rates on this GPU for the P.V candidates: f16 m16n8k16 with f16 / f32 accumulate, e4m3 m16n8k32 with
// f16 / f32 accumulate, and int8 m16n8k32 for reference. Dependent chains are split over 8 accumulators per warp.
//   nvcc -O3 -std=c++17 -arch=sm_89 tests/mma_rate.cu -o build/mma_rate && build/mma_rate
#include <cstdio>
#include <cstdint>

template <int KIND>
__global__ void k_rate(int iters, uint32_t* out) {
    uint32_t a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 0x3c3c3c3cu, a2 = a0 + 7, a3 = a1 + 9, b0 = a0 * 3, b1 = a1 * 5;
    uint32_t h[8][2] = {};
    float f[8][4] = {};
    int s[8][4] = {};
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
            if (KIND == 0)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(h[k][0]), "+r"(h[k][1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            if (KIND == 1)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(f[k][0]), "+f"(f[k][1]), "+f"(f[k][2]), "+f"(f[k][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            if (KIND == 2)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(h[k][0]), "+r"(h[k][1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            if (KIND == 3)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(f[k][0]), "+f"(f[k][1]), "+f"(f[k][2]), "+f"(f[k][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            if (KIND == 4)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(s[k][0]), "+r"(s[k][1]), "+r"(s[k][2]), "+r"(s[k][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    uint32_t acc = 0;
    for (int k = 0; k < 8; k++) acc += h[k][0] ^ h[k][1] ^ __float_as_uint(f[k][0] + f[k][3]) ^ (uint32_t)s[k][0];
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int KIND>
static void run(const char* name, int kdepth) {
    uint32_t* out;
    cudaMalloc(&out, 160 * 256 * 4);
    const int iters = 4000;
    k_rate<KIND><<<160, 256>>>(200, out);   // warm-up
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int r = 0; r < 5; r++) {
        cudaEventRecord(a);
        k_rate<KIND><<<160, 256>>>(iters, out);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        best = ms < best ? ms : best;
    }
    const double flops = 2.0 * 16 * 8 * kdepth * 8.0 * iters * (160 * 256 / 32);
    printf("%-34s %6.1f TFLOPS\n", name, flops / (best * 1e-3) / 1e12);
    cudaFree(out);
}

int main() {
    run<0>("f16 m16n8k16, f16 accumulate", 16);
    run<1>("f16 m16n8k16, f32 accumulate", 16);
    run<2>("e4m3 m16n8k32, f16 accumulate", 32);
    run<3>("e4m3 m16n8k32, f32 accumulate", 32);
    run<4>("int8 m16n8k32, s32 accumulate", 32);
    return 0;
}
