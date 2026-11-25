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

// q8_0 KV rows: 128 int8 + 4 f16 scales (one per 32 dims), same block format as ggml q8_0.
// A warp holds one row as 4 values per lane (dims lane*4..lane*4+3); 8 lanes form a block.
__device__ __forceinline__ void store_kv_row4(const float* x, void* data, half* scales, size_t row, int lane, bool q8) {
    if (!q8) {
        half* dst = (half*)data + row * 128 + lane * 4;
        *(half2*)dst = __floats2half2_rn(x[0], x[1]);
        *(half2*)(dst + 2) = __floats2half2_rn(x[2], x[3]);
        return;
    }
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
    uint32_t pk = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) pk |= (uint32_t)(uint8_t)(int8_t)roundf(x[j] * id) << (8 * j);
    *(uint32_t*)((int8_t*)data + row * 128 + lane * 4) = pk;
    if ((lane & 7) == 0) scales[row * 4 + (lane >> 3)] = __float2half(d);
}

// L2 prefetch of the next kernel's weights: first `bytes` of every row of up to 3 arrays
struct Prefetch {
    const uint8_t* p[3];
    int stride[3], bytes[3];
    int rows;
};
__device__ __forceinline__ void l2_prefetch(const Prefetch& f, int part, int nparts) {
    if (!f.rows) return;
    const int r0 = (int)((long)part * f.rows / nparts), r1 = (int)((long)(part + 1) * f.rows / nparts);
#pragma unroll
    for (int a = 0; a < 3; a++) {
        if (!f.p[a] || f.bytes[a] <= 0) continue;
        const int lpr = (f.bytes[a] + 127) >> 7;
        const int L = (r1 - r0) * lpr;
        for (int i = threadIdx.x; i < L; i += blockDim.x) {
            const int row = r0 + i / lpr, off = (i % lpr) << 7;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(f.p[a] + (size_t)row * f.stride[a] + off));
        }
    }
}

}  // namespace le
