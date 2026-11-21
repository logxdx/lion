// Device kernels for liquid-engine (decode path + weight repacking + sampling).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "engine.h"

namespace le {

// ----------------------------------------------------------------------------------------------
// small helpers
// ----------------------------------------------------------------------------------------------
__device__ __forceinline__ int4 ld_stream(const void* p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ int2 ld_stream2(const void* p) {
    int2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ uint16_t ld_stream_u16(const void* p) {
    uint16_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// block-wide sum for blockDim.x == NT (multiple of 32); red must hold 32 floats
template <int NT>
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    v = (l < NT / 32) ? red[l] : 0.f;
    return warp_sum(v);
}

// Element index -> position in the "half-block interleaved" activation layout used in shared
// memory (and for Q8 weights): per chunk of 32 groups (1024 elems) the first 16 bytes of every
// group come first, then the second 16 bytes. Lane i then reads group i with two
// conflict-free/coalesced 16-byte accesses.
template <int K>
__device__ __forceinline__ int perm_idx(int e) {
    const int g = e >> 5, c = g >> 5, gl = g & 31, hf = (e >> 4) & 1, p = e & 15;
    const int Gc = (K / 32 - c * 32) < 32 ? (K / 32 - c * 32) : 32;
    return c * 1024 + hf * Gc * 16 + gl * 16 + p;
}
__device__ __forceinline__ int perm_idx_rt(int e, int K) {
    const int g = e >> 5, c = g >> 5, gl = g & 31, hf = (e >> 4) & 1, p = e & 15;
    const int Gc = min(32, K / 32 - c * 32);
    return c * 1024 + hf * Gc * 16 + gl * 16 + p;
}

}  // namespace le
