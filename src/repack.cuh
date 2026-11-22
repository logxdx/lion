// GGUF block formats -> engine weight layouts, on the GPU at load time (engine.cu, vision.cu).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#include "engine.h"
#include "kernels.cuh"

namespace le {

struct RowMap { int tile_in, tile_out, off; };
__device__ __forceinline__ int map_row(RowMap m, int r) { return (r / m.tile_in) * m.tile_out + m.off + r % m.tile_in; }

// Q8_0 block: half d; int8 qs[32]  (34 bytes)
static __global__ void repack_q8(const uint8_t* __restrict__ src, int nrows, int K, QWeight dst, RowMap m) {
    const int NG = K / 32;
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)nrows * NG) return;
    const int r = gid / NG, g = gid % NG;
    const uint8_t* b = src + ((size_t)r * NG + g) * 34;
    const int dr = map_row(m, r);
    uint16_t dbits = b[0] | (b[1] << 8);
    dst.d[(size_t)dr * NG + g] = __ushort_as_half(dbits);
    uint8_t* q = dst.q + (size_t)dr * K;
    for (int j = 0; j < 32; j++) q[perm_idx_rt(g * 32 + j, K)] = b[2 + j];
}

// Q6_K block: uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d  (210 bytes)
// engine layout per 32-weight group g:  lo[16] byte j = w[j]&15 | (w[j+16]&15)<<4
//   hi[8]: two u32 words (w[0..15], w[16..31]); byte j bits 2m..2m+1 = w[4m+j]>>4
//   sc[2]: the two 16-element sub-scales;  d: one half per 256 weights
static __global__ void repack_q6(const uint8_t* __restrict__ src, int nrows, int K, QWeight dst, RowMap m) {
    const int NB = K / 256;
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)nrows * NB) return;
    const int r = gid / NB, bi = gid % NB;
    const uint8_t* b = src + ((size_t)r * NB + bi) * 210;
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* scales = (const int8_t*)(b + 192);
    uint16_t dbits = b[208] | (b[209] << 8);
    uint8_t w[256];
    for (int n = 0; n < 2; n++) {
        const uint8_t* l_ = ql + 64 * n;
        const uint8_t* h_ = qh + 32 * n;
        for (int l = 0; l < 32; l++) {
            w[128 * n + l] = (l_[l] & 0xF) | (((h_[l] >> 0) & 3) << 4);
            w[128 * n + l + 32] = (l_[l + 32] & 0xF) | (((h_[l] >> 2) & 3) << 4);
            w[128 * n + l + 64] = (l_[l] >> 4) | (((h_[l] >> 4) & 3) << 4);
            w[128 * n + l + 96] = (l_[l + 32] >> 4) | (((h_[l] >> 6) & 3) << 4);
        }
    }
    const int dr = map_row(m, r);
    dst.d[(size_t)dr * NB + bi] = __ushort_as_half(dbits);
    for (int gg = 0; gg < 8; gg++) {
        const int g = bi * 8 + gg;
        const uint8_t* ww = w + gg * 32;
        uint8_t* lo = dst.q + (size_t)dr * (K / 2) + g * 16;
        for (int j = 0; j < 16; j++) lo[j] = (ww[j] & 0xF) | ((ww[j + 16] & 0xF) << 4);
        uint8_t* hi = dst.h + (size_t)dr * (K / 4) + g * 8;
        for (int hw = 0; hw < 2; hw++)
            for (int j = 0; j < 4; j++) {
                uint8_t v = 0;
                for (int mm = 0; mm < 4; mm++) v |= ((ww[hw * 16 + 4 * mm + j] >> 4) & 3) << (2 * mm);
                hi[hw * 4 + j] = v;
            }
        dst.sc[(size_t)dr * (K / 16) + g * 2 + 0] = scales[gg * 2 + 0];
        dst.sc[(size_t)dr * (K / 16) + g * 2 + 1] = scales[gg * 2 + 1];
    }
}

}  // namespace le
