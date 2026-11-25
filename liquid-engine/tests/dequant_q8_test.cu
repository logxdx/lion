// Unit test: prmt-based int8 -> f16 dequantisation (prefill flash attention) must be bit-identical to
// __float2half((float)x * d) for every int8 x and a sweep of f16 scales.
// build: nvcc -O3 -arch=sm_89 --use_fast_math tests/dequant_q8_test.cu -o build/dequant_q8_test
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

__global__ void k(const float* scales, int ns, unsigned long long* mism, unsigned short* ex) {
    const int si = blockIdx.x, x0 = threadIdx.x * 4 - 128;  // 64 threads x 4 values = all 256
    if (si >= ns) return;
    const float d = __half2float(__float2half(scales[si]));
    uint32_t w = 0;
    for (int j = 0; j < 4; j++) w |= (uint32_t)(uint8_t)(int8_t)(x0 + j) << (8 * j);
    const half2 d2 = __float2half2_rn(d);
    const half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
    const uint32_t x = w ^ 0x80808080u;
    uint32_t a = __byte_perm(x, 0x64646464u, 0x4140), b = __byte_perm(x, 0x64646464u, 0x4342);
    const half2 ha = __hmul2(__hsub2(*(const half2*)&a, bias), d2);
    const half2 hb = __hmul2(__hsub2(*(const half2*)&b, bias), d2);
    const half got[4] = {__low2half(ha), __high2half(ha), __low2half(hb), __high2half(hb)};
    for (int j = 0; j < 4; j++) {
        const half ref = __float2half((float)(x0 + j) * d);
        if (__half_as_ushort(ref) != __half_as_ushort(got[j])) {
            if (atomicAdd(mism, 1ull) == 0) { ex[0] = (unsigned short)(x0 + j); ex[1] = __half_as_ushort(ref); ex[2] = __half_as_ushort(got[j]); ex[3] = si; }
        }
    }
}

int main() {
    const int ns = 4096;
    float hs[ns];
    for (int i = 0; i < ns; i++) hs[i] = 1e-6f * powf(1.0045f, (float)i);  // 1e-6 .. ~1e2
    float* ds; unsigned long long* dm; unsigned short* de;
    cudaMalloc(&ds, sizeof hs); cudaMalloc(&dm, 8); cudaMalloc(&de, 8);
    cudaMemcpy(ds, hs, sizeof hs, cudaMemcpyHostToDevice);
    cudaMemset(dm, 0, 8);
    k<<<ns, 64>>>(ds, ns, dm, de);
    unsigned long long m; unsigned short e[4];
    cudaMemcpy(&m, dm, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(e, de, 8, cudaMemcpyDeviceToHost);
    printf("checked %d values, mismatches: %llu\n", ns * 256, m);
    if (m) printf("first: x=%d ref=0x%04x got=0x%04x scale=%g\n", (int)(short)e[0], e[1], e[2], hs[e[3]]);
    return m ? 1 : 0;
}
