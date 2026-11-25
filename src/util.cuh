#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, \
                    cudaGetErrorString(e_));                                                         \
            abort();                                                                                 \
        }                                                                                            \
    } while (0)

namespace raon {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum; red must hold blockDim.x/32 floats; result broadcast to all threads
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.f;
    return warp_sum(t);
}
__device__ __forceinline__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : -INFINITY;
    return warp_max(t);
}
// exact int32 -> float for |c| < 2^22 with full-rate integer + float adds (I2F runs at 16/clk/SM on sm_89, FADD at 128)
__device__ __forceinline__ float i2f_exact(int c) { return __int_as_float(c + 0x4B400000) - 12582912.f; }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.f + tanhf(0.79788456080286535588f * x * (1.f + 0.044715f * x * x)));
}

}  // namespace raon
