// int8 tensor-core GEMM over the repacked Q8_0/Q6_K weights (+ the PTX helpers it and the attention kernels use).
// Header-only (inline device functions and templates): included by prefill.cuh (decoder) and vision.cu (encoder).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#include "engine.h"

namespace le {

// ------------------------------------------------------------------------------------------------
// PTX helpers
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* s, const void* g, bool pred = true) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(s)), "l"(g), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async8(void* s, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(s)), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_s8_k32(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}
__device__ __forceinline__ void mma_s8_k16(int* c, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// f16-accumulate variant (2x tensor rate on GeForce); used for the per-tile P.V partial sums only
__device__ __forceinline__ void mma_f16_acc16(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#ifndef FA_PV_F16
#define FA_PV_F16 1
#endif

// exact int32 -> float for |x| < 2^22 using the FP pipe instead of the quarter-rate I2F
__device__ __forceinline__ float i2f_magic(int x) { return __int_as_float(x + 0x4B400000) - 12582912.0f; }
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    half2 h = __floats2half2_rn(a, b);
    return *(uint32_t*)&h;
}


// ------------------------------------------------------------------------------------------------
// int8 tensor-core GEMM: Y[T][N] = A[T][K] (int8, per-32 scales) x W[N][K]^T (repacked Q8/Q6)
// CTA tile 128 tokens x 128 rows, k-tile 128 (4 groups), 8 warps (4 along M x 2 along N),
// warp tile 32 x 64, double-buffered cp.async. grid.x = M tiles (fastest: weights stay in L2).
// ------------------------------------------------------------------------------------------------
enum { GE_STORE = 0, GE_SWIGLU = 1 };
constexpr int GB_M = 128, GB_N = 128, GB_K = 128;

struct GemmArgs {
    const int8_t* A; const float* Ad; int T;
    QWeight w;
    float* out;                 // GE_STORE: [T][N]
    int8_t* yq; float* yd;      // GE_SWIGLU: [T][N/2] int8 + [T][N/64]
};

template <int WT>
struct GemmSmem;
template <>
struct GemmSmem<W_Q8> {
    int8_t a[2][GB_M * GB_K];
    int8_t b[2][GB_N * GB_K];
    float ad[2][GB_M * 4];
    half bd[2][GB_N * 4];
    float ws[4][GB_N];          // weight scales of the current k tile, [group][row]
};
template <>
struct GemmSmem<W_Q6> {
    int8_t a[2][GB_M * GB_K];
    uint8_t lo[2][GB_N * 64];
    uint8_t hi[2][GB_N * 32];
    int8_t sc[2][GB_N * 8];
    float ad[2][GB_M * 4];
    int8_t b[GB_N * GB_K];      // unpacked (q-32) tile, same swizzled layout as the Q8 tile
    float ws[8][GB_N];          // sub-scale * d, [sub-block][row]
};

template <int WT, int K>
__device__ __forceinline__ void gemm_load_stage(GemmSmem<WT>& sm, int st, const GemmArgs& g, int m0, int n0, int kt) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; i++) {  // A: 128 rows x 8 chunks
        const int idx = tid + i * 256, r = idx >> 3, c = idx & 7;
        const int t = m0 + r;
        const bool ok = t < g.T;
        const int8_t* src = g.A + (size_t)(ok ? t : 0) * K + kt * GB_K + c * 16;
        cp_async16(sm.a[st] + r * GB_K + ((c ^ (r & 7)) * 16), src, ok);
    }
    if (tid < 128) {
        const int t = m0 + tid;
        const bool ok = t < g.T;
        cp_async16(sm.ad[st] + tid * 4, g.Ad + (size_t)(ok ? t : 0) * (K / 32) + kt * 4, ok);
    }
    if constexpr (WT == W_Q8) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * 256, r = idx >> 3, c = idx & 7;
            const int grp = kt * 4 + (c >> 1), hf = c & 1;
            const int ch = grp >> 5, gl = grp & 31;
            const int Gc = min(32, K / 32 - ch * 32);
            const uint8_t* src = g.w.q + (size_t)(n0 + r) * K + ch * 1024 + hf * Gc * 16 + gl * 16;
            cp_async16(sm.b[st] + r * GB_K + ((c ^ (r & 7)) * 16), src);
        }
        if (tid < 128) cp_async8(sm.bd[st] + tid * 4, g.w.d + (size_t)(n0 + tid) * (K / 32) + kt * 4);
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int idx = tid + i * 256, r = idx >> 2, c = idx & 3;
            cp_async16(sm.lo[st] + r * 64 + c * 16, g.w.q + (size_t)(n0 + r) * (K / 2) + kt * 64 + c * 16);
        }
        {
            const int r = tid >> 1, c = tid & 1;
            cp_async16(sm.hi[st] + r * 32 + c * 16, g.w.h + (size_t)(n0 + r) * (K / 4) + kt * 32 + c * 16);
        }
        if (tid < 128) cp_async8(sm.sc[st] + tid * 8, g.w.sc + (size_t)(n0 + tid) * (K / 16) + kt * 8);
    }
}

// per k tile, after the stage landed: weight scales -> float [group][row]; Q6: unpack to int8 tile
template <int WT, int K>
__device__ __forceinline__ void gemm_prepare_stage(GemmSmem<WT>& sm, int st, const GemmArgs& g, int n0, int kt) {
    const int tid = threadIdx.x;
    if constexpr (WT == W_Q8) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int idx = tid + i * 256, r = idx & 127, grp = idx >> 7;
            sm.ws[grp][r] = __half2float(sm.bd[st][r * 4 + grp]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int idx = tid + i * 256, r = idx >> 2, j = idx & 3;
            const uint4 l = *(const uint4*)(sm.lo[st] + r * 64 + j * 16);
            const uint2 h = *(const uint2*)(sm.hi[st] + r * 32 + j * 8);
            const uint32_t lw[4] = {l.x, l.y, l.z, l.w};
            uint32_t q0[4], q1[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                q0[m] = __vsub4((lw[m] & 0x0F0F0F0F) | (((h.x >> (2 * m)) & 0x03030303) << 4), 0x20202020);
                q1[m] = __vsub4(((lw[m] >> 4) & 0x0F0F0F0F) | (((h.y >> (2 * m)) & 0x03030303) << 4), 0x20202020);
            }
            const int c0 = 2 * j, c1 = 2 * j + 1;
            *(uint4*)(sm.b + r * GB_K + ((c0 ^ (r & 7)) * 16)) = make_uint4(q0[0], q0[1], q0[2], q0[3]);
            *(uint4*)(sm.b + r * GB_K + ((c1 ^ (r & 7)) * 16)) = make_uint4(q1[0], q1[1], q1[2], q1[3]);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * 256, r = idx & 127, sub = idx >> 7;
            const float d = __half2float(g.w.d[(size_t)(n0 + r) * (K / 256) + (kt >> 1)]);
            sm.ws[sub][r] = (float)sm.sc[st][r * 8 + sub] * d;
        }
    }
}

template <int WT, int K, int EPI>
__global__ void __launch_bounds__(256, 1) k_gemm(GemmArgs g) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    GemmSmem<WT>& sm = *reinterpret_cast<GemmSmem<WT>*>(smem_raw);
    const int m0 = blockIdx.x * GB_M, n0 = blockIdx.y * GB_N;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, tig = lane & 3;
    const int wm = (warp & 3) * 32, wn = (warp >> 2) * 64;
    constexpr int KT = K / GB_K;

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    gemm_load_stage<WT, K>(sm, 0, g, m0, n0, 0);
    cp_commit();
    for (int kt = 0; kt < KT; kt++) {
        const int st = kt & 1;
        if (kt + 1 < KT) gemm_load_stage<WT, K>(sm, st ^ 1, g, m0, n0, kt + 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        gemm_prepare_stage<WT, K>(sm, st, g, n0, kt);
        __syncthreads();
        const int8_t* Bt;
        if constexpr (WT == W_Q8) Bt = sm.b[st];
        else Bt = sm.b;
#pragma unroll
        for (int grp = 0; grp < 4; grp++) {
            uint32_t af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int q = lane >> 3, r = wm + mi * 16 + (lane & 7) + (q & 1) * 8;
                const int c = grp * 2 + (q >> 1);
                ldsm_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3], sm.a[st] + r * GB_K + ((c ^ (r & 7)) * 16));
            }
            uint32_t bf[8][2];
#pragma unroll
            for (int np = 0; np < 4; np++) {
                const int q = lane >> 3, r = wn + np * 16 + (lane & 7) + (q >> 1) * 8;
                const int c = grp * 2 + (q & 1);
                ldsm_x4(bf[2 * np][0], bf[2 * np][1], bf[2 * np + 1][0], bf[2 * np + 1][1], Bt + r * GB_K + ((c ^ (r & 7)) * 16));
            }
            float xs[2][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                xs[mi][0] = sm.ad[st][(wm + mi * 16 + g8) * 4 + grp];
                xs[mi][1] = sm.ad[st][(wm + mi * 16 + g8 + 8) * 4 + grp];
            }
            if constexpr (WT == W_Q8) {
                int c[2][8][4];
#pragma unroll
                for (int ni = 0; ni < 8; ni++)
#pragma unroll
                    for (int mi = 0; mi < 2; mi++) mma_s8_k32(c[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
#pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    const float2 w = *(const float2*)&sm.ws[grp][wn + ni * 8 + tig * 2];
#pragma unroll
                    for (int mi = 0; mi < 2; mi++) {
                        acc[mi][ni][0] += i2f_magic(c[mi][ni][0]) * (xs[mi][0] * w.x);
                        acc[mi][ni][1] += i2f_magic(c[mi][ni][1]) * (xs[mi][0] * w.y);
                        acc[mi][ni][2] += i2f_magic(c[mi][ni][2]) * (xs[mi][1] * w.x);
                        acc[mi][ni][3] += i2f_magic(c[mi][ni][3]) * (xs[mi][1] * w.y);
                    }
                }
            } else {
#pragma unroll
                for (int np = 0; np < 4; np++) {
                    int c0[2][2][4], c1[2][2][4];
#pragma unroll
                    for (int h = 0; h < 2; h++)
#pragma unroll
                        for (int mi = 0; mi < 2; mi++) {
                            mma_s8_k16(c0[mi][h], af[mi][0], af[mi][1], bf[np * 2 + h][0]);
                            mma_s8_k16(c1[mi][h], af[mi][2], af[mi][3], bf[np * 2 + h][1]);
                        }
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int ni = np * 2 + h, col = wn + ni * 8 + tig * 2;
                        const float2 s0 = *(const float2*)&sm.ws[grp * 2][col];
                        const float2 s1 = *(const float2*)&sm.ws[grp * 2 + 1][col];
#pragma unroll
                        for (int mi = 0; mi < 2; mi++) {
                            acc[mi][ni][0] += xs[mi][0] * (s0.x * i2f_magic(c0[mi][h][0]) + s1.x * i2f_magic(c1[mi][h][0]));
                            acc[mi][ni][1] += xs[mi][0] * (s0.y * i2f_magic(c0[mi][h][1]) + s1.y * i2f_magic(c1[mi][h][1]));
                            acc[mi][ni][2] += xs[mi][1] * (s0.x * i2f_magic(c0[mi][h][2]) + s1.x * i2f_magic(c1[mi][h][2]));
                            acc[mi][ni][3] += xs[mi][1] * (s0.y * i2f_magic(c0[mi][h][3]) + s1.y * i2f_magic(c1[mi][h][3]));
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (EPI == GE_STORE) {
        const int N = g.w.N;
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int t = m0 + wm + mi * 16 + g8 + h * 8;
                if (t >= g.T) continue;
#pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    const int col = n0 + wn + ni * 8 + tig * 2;
                    if (col < N) *(float2*)(g.out + (size_t)t * N + col) = make_float2(acc[mi][ni][h * 2], acc[mi][ni][h * 2 + 1]);
                }
            }
    } else {
        // stage the 128x128 f32 tile through shared memory, then SwiGLU + quantise per 32 outputs
        float* tile = reinterpret_cast<float*>(smem_raw);
        constexpr int LD = GB_N + 4;
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = wm + mi * 16 + g8 + h * 8;
#pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    const int c = wn + ni * 8 + tig * 2;
                    tile[r * LD + c] = acc[mi][ni][h * 2];
                    tile[r * LD + c + 1] = acc[mi][ni][h * 2 + 1];
                }
            }
        __syncthreads();
        // 128 tokens x 2 pairs x 4 threads (8 outputs each) = 1024 items
        const int NO = g.w.N / 2;
        for (int it = threadIdx.x; it < GB_M * 8; it += 256) {
            const int r = it >> 3, pair = (it >> 2) & 1, q = it & 3;
            const int t = m0 + r;
            float y[8], amax = 0;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float gv = tile[r * LD + pair * 64 + q * 8 + i];
                const float uv = tile[r * LD + pair * 64 + 32 + q * 8 + i];
                y[i] = gv / (1.f + __expf(-gv)) * uv;
                amax = fmaxf(amax, fabsf(y[i]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
            const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
            if (t < g.T) {
                const int ob = (n0 / 64) + pair;  // output 32-block index
                uint32_t lo = 0, hi = 0;
#pragma unroll
                for (int i = 0; i < 4; i++) lo |= (uint32_t)(uint8_t)(int8_t)roundf(y[i] * id) << (8 * i);
#pragma unroll
                for (int i = 0; i < 4; i++) hi |= (uint32_t)(uint8_t)(int8_t)roundf(y[4 + i] * id) << (8 * i);
                *(uint2*)(g.yq + (size_t)t * NO + ob * 32 + q * 8) = make_uint2(lo, hi);
                if (q == 0) g.yd[(size_t)t * (NO / 32) + ob] = d;
            }
        }
    }
}

template <int K, int EPI>
inline void launch_gemm(const GemmArgs& g, cudaStream_t s) {
    dim3 grid((g.T + GB_M - 1) / GB_M, (g.w.N + GB_N - 1) / GB_N);
    constexpr size_t ep = 128 * 132 * 4;
    if (g.w.type == W_Q8) {
        constexpr size_t sm = sizeof(GemmSmem<W_Q8>) > ep ? sizeof(GemmSmem<W_Q8>) : ep;
        static bool init = false;
        if (!init) { cudaFuncSetAttribute(k_gemm<W_Q8, K, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); init = true; }
        k_gemm<W_Q8, K, EPI><<<grid, 256, sm, s>>>(g);
    } else {
        constexpr size_t sm = sizeof(GemmSmem<W_Q6>) > ep ? sizeof(GemmSmem<W_Q6>) : ep;
        static bool init = false;
        if (!init) { cudaFuncSetAttribute(k_gemm<W_Q6, K, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); init = true; }
        k_gemm<W_Q6, K, EPI><<<grid, 256, sm, s>>>(g);
    }
}

}  // namespace le
