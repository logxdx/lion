// k_gemm_f16 vs k_gemv_f16 (f32 activations) on Raon's F16 shapes, incl. grouped per-head ones.
//   nvcc -O3 -std=c++17 -arch=sm_89 -Isrc tests/gemm_f16_check.cu -o build/gemm_f16_check && build/gemm_f16_check
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gemm.cuh"
#include "kernels.cuh"

using namespace raon;

int main() {
    const int T = 77;
    struct S { const char* name; int N, K, ldx, rpg, gstride; } shapes[] = {
        {"hc_fn 6144->17", 17, 6144, 6144, 0, 0},       {"ssm_f_a 1536->160", 160, 1536, 1536, 0, 0},
        {"ssm_f_b 128->2048 (ldx 160)", 2048, 128, 160, 0, 0}, {"kv_a 1536->576", 576, 1536, 1536, 0, 0},
        {"k_b grouped 32x(128->512)", 16384, 128, 6144, 512, 192},
        {"v_b grouped 32x(512->128)", 4096, 512, 16384, 128, 512}, {"ple_proj 1536->6144", 6144, 1536, 1536, 0, 0}};
    for (auto& s : shapes) {
        const int N = s.N, K = s.K, rpg = s.rpg ? s.rpg : N;
        std::vector<half> hw((size_t)N * K);
        std::vector<float> hx((size_t)T * s.ldx);
        for (auto& v : hw) v = __float2half((rand() % 2001 - 1000) * 1e-4f);
        for (auto& v : hx) v = (rand() % 2001 - 1000) * 1e-3f;
        half* w; float *x, *y1, *y2;
        CK(cudaMalloc(&w, hw.size() * 2)); CK(cudaMalloc(&x, hx.size() * 4));
        CK(cudaMalloc(&y1, (size_t)T * N * 4)); CK(cudaMalloc(&y2, (size_t)T * N * 4));
        CK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        for (int b0 = 0; b0 < T; b0 += 1)
            k_gemv_f16<1><<<(N + 3) / 4, 128>>>(w, N, K, x + (size_t)b0 * s.ldx, s.ldx, rpg, s.gstride, y1 + (size_t)b0 * N, N);
        dim3 grid((T + 63) / 64, (N + 63) / 64);
        k_gemm_f16<<<grid, gf16::THREADS>>>(x, s.ldx, rpg, s.gstride, w, T, N, K, y2, N);
        CK(cudaDeviceSynchronize());
        std::vector<float> r1((size_t)T * N), r2((size_t)T * N);
        CK(cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(r2.data(), y2, r2.size() * 4, cudaMemcpyDeviceToHost));
        double maxd = 0, maxv = 0;
        for (size_t i = 0; i < r1.size(); i++) { maxd = fmax(maxd, fabs(r1[i] - r2[i])); maxv = fmax(maxv, fabs(r1[i])); }
        printf("%-30s max|diff| %.3g  (max|y| %.3g, rel %.2g)\n", s.name, maxd, maxv, maxd / maxv);
        cudaFree(w); cudaFree(x); cudaFree(y1); cudaFree(y2);
    }
    return 0;
}
