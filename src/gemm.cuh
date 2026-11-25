// Prefill GEMMs on tensor cores: Y[T][N] = X[T][K] . W[N][K]^T with X quantized to int8 per 32 (xq, xs).
//
// Q8_0 weights: int8 mma m16n8k32, one k-step = one q8 block: the int32 product of the step is scaled by
// (activation scale of the row) x (weight scale of the column) and accumulated in f32.
// Block tile 128 tokens x 128 outputs, 8 warps (4 along tokens x 2 along outputs, warp tile 32 x 64), 3-stage cp.async.
#pragma once
#include <cstdint>

#include "util.cuh"

namespace raon {

namespace gq8 {
constexpr int BM = 128, BN = 128, BK = 32, ST = 3, THREADS = 256;
constexpr int P = BK + 16;  // smem row pitch (bytes): 48 -> ldmatrix rows land on distinct banks
struct Smem {
    int8_t a[ST][BM * P];
    int8_t b[ST][BN * P];
    float as[ST][BM];
    float bs[ST][BN];
};
}  // namespace gq8

__device__ __forceinline__ void g_cp16(void* smem, const void* gmem, bool pred) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void g_cp4(void* smem, const void* gmem, bool pred) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void g_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void g_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void g_ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void g_mma_s8(int* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

// grid (ceil(T/128), N/128), block 256. Requires N % 128 == 0, K % 32 == 0.
__global__ void __launch_bounds__(gq8::THREADS, 2)
k_gemm_q8(const int8_t* __restrict__ xq, const float* __restrict__ xs, const int8_t* __restrict__ wq,
          const half* __restrict__ wd, int T, int N, int K, float* __restrict__ y, int ldy) {
    using namespace gq8;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nkb = K / BK;
    const int wm = warp >> 1, wn = warp & 1;  // warp tile: tokens wm*32.., outputs wn*64..

    auto issue = [&](int kb, int st) {
        if (kb < nkb) {
            // A: 128 rows x 32 bytes = 256 x 16B; B: same
            {
                const int row = tid >> 1, half_ = tid & 1;
                const int t = m0 + row;
                g_cp16(&S.a[st][row * P + half_ * 16], xq + (size_t)(t < T ? t : 0) * K + kb * BK + half_ * 16, t < T);
                g_cp16(&S.b[st][row * P + half_ * 16], wq + (size_t)(n0 + row) * K + kb * BK + half_ * 16, true);
            }
            if (tid < BM) {
                const int t = m0 + tid;
                g_cp4(&S.as[st][tid], xs + (size_t)(t < T ? t : 0) * (K / 32) + kb, t < T);
            } else {
                const int n = tid - BM;
                S.bs[st][n] = __half2float(wd[(size_t)(n0 + n) * (K / 32) + kb]);  // plain load (half -> float)
            }
        }
        g_commit();
    };

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;

    for (int s = 0; s < ST - 1; s++) issue(s, s);
    const int g = lane >> 2, t4 = lane & 3;
    for (int kb = 0; kb < nkb; kb++) {
        const int st = kb % ST;
        g_wait<ST - 2>();
        __syncthreads();
        issue(kb + ST - 1, (kb + ST - 1) % ST);
        uint32_t a[2][4], b[8][2];
#pragma unroll
        for (int i = 0; i < 2; i++)
            g_ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3],
                      &S.a[st][(wm * 32 + i * 16 + (lane & 15)) * P + (lane >> 4) * 16]);
#pragma unroll
        for (int j = 0; j < 8; j += 2)  // x4 = two n8 tiles (rows n, n+8) x (k 0-15, 16-31)
            g_ldsm_x4(b[j][0], b[j][1], b[j + 1][0], b[j + 1][1],
                      &S.b[st][(wn * 64 + j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * P + ((lane >> 3) & 1) * 16]);
        float ra[2][2], cb[8][2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            ra[i][0] = S.as[st][wm * 32 + i * 16 + g];
            ra[i][1] = S.as[st][wm * 32 + i * 16 + g + 8];
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            cb[j][0] = S.bs[st][wn * 64 + j * 8 + 2 * t4];
            cb[j][1] = S.bs[st][wn * 64 + j * 8 + 2 * t4 + 1];
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int c[4];
                g_mma_s8(c, a[i][0], a[i][1], a[i][2], a[i][3], b[j][0], b[j][1]);
                acc[i][j][0] += (float)c[0] * (ra[i][0] * cb[j][0]);
                acc[i][j][1] += (float)c[1] * (ra[i][0] * cb[j][1]);
                acc[i][j][2] += (float)c[2] * (ra[i][1] * cb[j][0]);
                acc[i][j][3] += (float)c[3] * (ra[i][1] * cb[j][1]);
            }
    }
    // store
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int t = m0 + wm * 32 + i * 16 + g, n = n0 + wn * 64 + j * 8 + 2 * t4;
            if (t < T) *(float2*)&y[(size_t)t * ldy + n] = make_float2(acc[i][j][0], acc[i][j][1]);
            if (t + 8 < T) *(float2*)&y[(size_t)(t + 8) * ldy + n] = make_float2(acc[i][j][2], acc[i][j][3]);
        }
}

// ---------------------------------------------------------------- Q6_K weights
// Q6_K: 6-bit values with an int8 scale per 16 and a half scale per 256. A 32-wide k-step (one activation block) is
// one (half, r) slice of a 256-block: 32 contiguous low-nibble bytes + 32 contiguous high-bit bytes, two sub-scales.
// Per k-step: two int8 mma m16n8k16 (one per sub-block), combined in int32 with the sub-scales, then one float FMA.
namespace gq6 {
constexpr int BM = 128, BN = 128, BK = 32, ST = 2, THREADS = 256;
constexpr int P = BK + 16;
struct Smem {
    int8_t a[ST][BM * P];
    uint8_t ql[ST][BN * BK];
    uint8_t qh[ST][BN * BK];
    float as[ST][BM];
    float bd[ST][BN];
    int8_t bsc[ST][BN * 2];
    int8_t b[BN * P];  // unpacked weights of the current k-step
};
}  // namespace gq6

__device__ __forceinline__ void g_mma_s8k16(int* c, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// grid (ceil(T/128), N/128), block 256. W planes as repacked by the loader: ql [N][K/2], qh [N][K/4], sc [N][K/16],
// d [N][K/256], GGUF block order. Requires N % 128 == 0, K % 256 == 0.
__global__ void __launch_bounds__(gq6::THREADS, 2)
k_gemm_q6(const int8_t* __restrict__ xq, const float* __restrict__ xs, const uint8_t* __restrict__ wl,
          const uint8_t* __restrict__ wh, const int8_t* __restrict__ wsc, const half* __restrict__ wd, int T, int N, int K,
          float* __restrict__ y, int ldy) {
    using namespace gq6;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nkb = K / BK;
    const int wm = warp >> 1, wn = warp & 1;

    auto issue = [&](int kb, int st) {
        if (kb < nkb) {
            const int blk = kb >> 3, mm = kb & 7, hf = mm >> 2, r = mm & 3;
            const int row = tid >> 1, part = tid & 1;
            const int t = m0 + row;
            g_cp16(&S.a[st][row * P + part * 16], xq + (size_t)(t < T ? t : 0) * K + kb * BK + part * 16, t < T);
            const int n = n0 + row;
            g_cp16(&S.ql[st][row * BK + part * 16], wl + (size_t)n * (K / 2) + blk * 128 + hf * 64 + (r & 1) * 32 + part * 16, true);
            g_cp16(&S.qh[st][row * BK + part * 16], wh + (size_t)n * (K / 4) + blk * 64 + hf * 32 + part * 16, true);
            if (tid < BM) {
                const int tt = m0 + tid;
                g_cp4(&S.as[st][tid], xs + (size_t)(tt < T ? tt : 0) * (K / 32) + kb, tt < T);
            } else {
                const int nn = tid - BM;
                const size_t rn = (size_t)(n0 + nn);
                S.bd[st][nn] = __half2float(wd[rn * (K / 256) + blk]);
                const int si = blk * 16 + hf * 8 + 2 * r;
                S.bsc[st][2 * nn] = wsc[rn * (K / 16) + si];
                S.bsc[st][2 * nn + 1] = wsc[rn * (K / 16) + si + 1];
            }
        }
        g_commit();
    };

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;

    issue(0, 0);
    const int g = lane >> 2, t4 = lane & 3;
    for (int kb = 0; kb < nkb; kb++) {
        const int st = kb & 1;
        g_wait<0>();
        __syncthreads();
        issue(kb + 1, st ^ 1);
        // unpack this k-step's weights: 128 rows x 32 values; thread: row tid/2, 16 values
        {
            const int r = (kb & 7) & 3;
            const int row = tid >> 1, part = tid & 1;
            const uint4 lo = *(const uint4*)&S.ql[st][row * BK + part * 16];
            const uint4 hi = *(const uint4*)&S.qh[st][row * BK + part * 16];
            const int ns = r >= 2 ? 4 : 0, hs = 2 * r;
            uint4 o;
            o.x = (uint32_t)__vsubss4((int)(((lo.x >> ns) & 0x0F0F0F0Fu) | (((hi.x >> hs) & 0x03030303u) << 4)), 0x20202020);
            o.y = (uint32_t)__vsubss4((int)(((lo.y >> ns) & 0x0F0F0F0Fu) | (((hi.y >> hs) & 0x03030303u) << 4)), 0x20202020);
            o.z = (uint32_t)__vsubss4((int)(((lo.z >> ns) & 0x0F0F0F0Fu) | (((hi.z >> hs) & 0x03030303u) << 4)), 0x20202020);
            o.w = (uint32_t)__vsubss4((int)(((lo.w >> ns) & 0x0F0F0F0Fu) | (((hi.w >> hs) & 0x03030303u) << 4)), 0x20202020);
            *(uint4*)&S.b[row * P + part * 16] = o;
        }
        __syncthreads();
        uint32_t a[2][4], b[8][2];
#pragma unroll
        for (int i = 0; i < 2; i++)
            g_ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3],
                      &S.a[st][(wm * 32 + i * 16 + (lane & 15)) * P + (lane >> 4) * 16]);
#pragma unroll
        for (int j = 0; j < 8; j += 2)
            g_ldsm_x4(b[j][0], b[j][1], b[j + 1][0], b[j + 1][1],
                      &S.b[(wn * 64 + j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * P + ((lane >> 3) & 1) * 16]);
        float ra[2][2], cd[8][2];
        int s0[8][2], s1[8][2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            ra[i][0] = S.as[st][wm * 32 + i * 16 + g];
            ra[i][1] = S.as[st][wm * 32 + i * 16 + g + 8];
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 2; q++) {
                const int nn = wn * 64 + j * 8 + 2 * t4 + q;
                cd[j][q] = S.bd[st][nn];
                s0[j][q] = S.bsc[st][2 * nn];
                s1[j][q] = S.bsc[st][2 * nn + 1];
            }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int cl[4], ch[4];
                g_mma_s8k16(cl, a[i][0], a[i][1], b[j][0]);  // k 0-15: sub-block 0
                g_mma_s8k16(ch, a[i][2], a[i][3], b[j][1]);  // k 16-31: sub-block 1
                acc[i][j][0] += (float)(cl[0] * s0[j][0] + ch[0] * s1[j][0]) * (ra[i][0] * cd[j][0]);
                acc[i][j][1] += (float)(cl[1] * s0[j][1] + ch[1] * s1[j][1]) * (ra[i][0] * cd[j][1]);
                acc[i][j][2] += (float)(cl[2] * s0[j][0] + ch[2] * s1[j][0]) * (ra[i][1] * cd[j][0]);
                acc[i][j][3] += (float)(cl[3] * s0[j][1] + ch[3] * s1[j][1]) * (ra[i][1] * cd[j][1]);
            }
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int t = m0 + wm * 32 + i * 16 + g, n = n0 + wn * 64 + j * 8 + 2 * t4;
            if (t < T) *(float2*)&y[(size_t)t * ldy + n] = make_float2(acc[i][j][0], acc[i][j][1]);
            if (t + 8 < T) *(float2*)&y[(size_t)(t + 8) * ldy + n] = make_float2(acc[i][j][2], acc[i][j][3]);
        }
}

// ---------------------------------------------------------------- F16 weights
// Y[T][N] = X[T][K] . W[N][K]^T, X f32 (converted to f16 while staged), W f16, f32 accumulate.
// Grouped: output column n reads X + (n / rpg) * gstride (per-head inputs); a 64-wide N tile never crosses a group.
// Block tile 64 x 64, 4 warps (2 x 2, warp tile 32 x 32), K step 32, double-buffered.
namespace gf16 {
constexpr int BM = 64, BN = 64, BK = 32, THREADS = 128;
constexpr int P = BK + 8;  // halfs: 80-byte rows, ldmatrix conflict-free
struct Smem {
    half a[2][BM * P];
    half b[2][BN * P];
};
}  // namespace gf16

__device__ __forceinline__ void g_mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                           uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// grid (ceil(T/64), ceil(N/64)), block 128. Requires K % 32 == 0; grouped: rpg % 64 == 0.
__global__ void __launch_bounds__(gf16::THREADS)
k_gemm_f16(const float* __restrict__ x, int ldx, int rpg, int gstride, const half* __restrict__ w, int T, int N, int K,
           float* __restrict__ y, int ldy) {
    using namespace gf16;
    __shared__ __align__(16) Smem S;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const float* xg = x + (size_t)(n0 / rpg) * gstride;
    const int nkb = K / BK;
    const int wm = warp >> 1, wn = warp & 1;

    float4 ra[4];  // A prefetch: 64 rows x 32 floats = 512 float4 -> 4 per thread
    auto load_a = [&](int kb) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * THREADS, row = idx >> 3, c4 = idx & 7;
            const int t = m0 + row;
            ra[i] = t < T ? *(const float4*)&xg[(size_t)t * ldx + kb * BK + c4 * 4] : make_float4(0, 0, 0, 0);
        }
    };
    auto store_a = [&](int st) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * THREADS, row = idx >> 3, c4 = idx & 7;
            half2* d = (half2*)&S.a[st][row * P + c4 * 4];
            d[0] = __floats2half2_rn(ra[i].x, ra[i].y);
            d[1] = __floats2half2_rn(ra[i].z, ra[i].w);
        }
    };
    auto issue_b = [&](int kb, int st) {  // 64 rows x 32 halfs = 256 x 16B -> 2 per thread
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int idx = tid + i * THREADS, row = idx >> 2, c8 = idx & 3;
            const int n = n0 + row;
            g_cp16(&S.b[st][row * P + c8 * 8], w + (size_t)(n < N ? n : 0) * K + kb * BK + c8 * 8, n < N);
        }
        g_commit();
    };

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;

    load_a(0);
    issue_b(0, 0);
    const int g = lane >> 2, t4 = lane & 3;
    for (int kb = 0; kb < nkb; kb++) {
        const int st = kb & 1;
        store_a(st);
        if (kb + 1 < nkb) { load_a(kb + 1); issue_b(kb + 1, st ^ 1); } else { g_commit(); }
        g_wait<1>();
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t a[2][4], b[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++)
                g_ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3],
                          &S.a[st][(wm * 32 + i * 16 + (lane & 15)) * P + ks * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 4; j += 2)
                g_ldsm_x4(b[j][0], b[j][1], b[j + 1][0], b[j + 1][1],
                          &S.b[st][(wn * 32 + j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * P + ks * 16 + ((lane >> 3) & 1) * 8]);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) g_mma16816(acc[i][j], a[i][0], a[i][1], a[i][2], a[i][3], b[j][0], b[j][1]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int t = m0 + wm * 32 + i * 16 + g, n = n0 + wn * 32 + j * 8 + 2 * t4;
            if (n < N) {
                if (t < T) y[(size_t)t * ldy + n] = acc[i][j][0];
                if (t + 8 < T) y[(size_t)(t + 8) * ldy + n] = acc[i][j][2];
            }
            if (n + 1 < N) {
                if (t < T) y[(size_t)t * ldy + n + 1] = acc[i][j][1];
                if (t + 8 < T) y[(size_t)(t + 8) * ldy + n + 1] = acc[i][j][3];
            }
        }
}

}  // namespace raon
