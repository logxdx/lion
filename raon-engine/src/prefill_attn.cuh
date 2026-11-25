// Prefill MLA attention in expanded form (as llama.cpp's mla_expand path), f16 tensor cores, f32 accumulate.
//
// For a chunk of n queries of one slot, per block of cached keys:
//   Cf = dequant(latent cache rows)                      [KB][576] f16 (512 latent | 64 pe)
//   E  = Cf[:, :512] . Wexp^T                            [KB][8192] f16: per head K_nope (128) | then per head V (128)
//   flash attention per (128 queries, head): K = [K_nope_h | pe] (192), V = V_h (128), causal, scale folded into Q
// Results of key blocks are merged into global (O, m, l); k_fa_final normalises.
#pragma once
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "util.cuh"

namespace raon {

namespace fa {
constexpr int H = 32, DK = 192, DV = 128, NOPE = 128, PE = 64, RK = 512, L = 576;
constexpr int BQ = 128, BKEY = 64, THREADS = 256;  // 8 warps x 16 queries
constexpr int KP = DK + 8;                          // 200 halfs = 400 B rows: ldmatrix conflict-free
constexpr int VP = DV + 8;                          // 136 halfs = 272 B rows
constexpr int EW = 2 * H * NOPE;                    // 8192 expanded values per key
struct Smem {
    half k[2][BKEY * KP];
    half v[2][BKEY * VP];
};
}  // namespace fa

__device__ __forceinline__ void fa_cp16(void* smem, const void* gmem, bool pred) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void fa_ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void fa_ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void fa_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t*)&h;
}

// latent cache rows [k0, k0+n) of one slot/layer (q8) -> f16 [n][576]
__global__ void k_cache_dequant(const int8_t* __restrict__ cq, const half* __restrict__ cs, size_t row0, int n,
                                half* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per 8 values
    if (idx >= n * (fa::L / 8)) return;
    const int r = idx / (fa::L / 8), c = idx % (fa::L / 8);
    const size_t row = row0 + r;
    const int2 raw = *(const int2*)&cq[row * fa::L + c * 8];
    const float d = __half2float(cs[row * attn::KVS + (KV_ROWSCALE ? (c < fa::RK / 8 ? 0 : 1) : c / 4)]);
    const int8_t* b = (const int8_t*)&raw;
    uint4 o;
    o.x = pack_h2(d * b[0], d * b[1]); o.y = pack_h2(d * b[2], d * b[3]);
    o.z = pack_h2(d * b[4], d * b[5]); o.w = pack_h2(d * b[6], d * b[7]);
    *(uint4*)&out[(size_t)r * fa::L + c * 8] = o;
}

// E[M][N] (f16) = A[M][K] (f16, row stride lda) . B[N][K]^T (f16). Tile 64 x 64, 4 warps, K step 32, double-buffered.
__global__ void __launch_bounds__(128) k_gemm_hh(const half* __restrict__ A, int lda, const half* __restrict__ B, int M,
                                                 int N, int K, half* __restrict__ E, int lde) {
    constexpr int BM = 64, BN = 64, BK = 32, P = BK + 8;
    __shared__ __align__(16) half sa[2][BM * P], sb[2][BN * P];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, nkb = K / BK;
    const int wm = warp >> 1, wn = warp & 1;
    auto issue = [&](int kb, int st) {
        if (kb < nkb) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int idx = tid + i * 128, row = idx >> 2, c8 = idx & 3;
                const int m = m0 + row, n = n0 + row;
                fa_cp16(&sa[st][row * P + c8 * 8], A + (size_t)(m < M ? m : 0) * lda + kb * BK + c8 * 8, m < M);
                fa_cp16(&sb[st][row * P + c8 * 8], B + (size_t)(n < N ? n : 0) * K + kb * BK + c8 * 8, n < N);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };
    float acc[2][4][4] = {};
    issue(0, 0);
    const int g = lane >> 2, t4 = lane & 3;
    for (int kb = 0; kb < nkb; kb++) {
        const int st = kb & 1;
        issue(kb + 1, st ^ 1);
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t a[2][4], b[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++)
                fa_ldsm_x4(a[i][0], a[i][1], a[i][2], a[i][3],
                           &sa[st][(wm * 32 + i * 16 + (lane & 15)) * P + ks * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 4; j += 2)
                fa_ldsm_x4(b[j][0], b[j][1], b[j + 1][0], b[j + 1][1],
                           &sb[st][(wn * 32 + j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * P + ks * 16 + ((lane >> 3) & 1) * 8]);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) fa_mma(acc[i][j], a[i], b[j][0], b[j][1]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int m = m0 + wm * 32 + i * 16 + g, n = n0 + wn * 32 + j * 8 + 2 * t4;
            if (n + 1 < N + 1) {
                if (m < M) *(uint32_t*)&E[(size_t)m * lde + n] = pack_h2(acc[i][j][0], acc[i][j][1]);
                if (m + 8 < M) *(uint32_t*)&E[(size_t)(m + 8) * lde + n] = pack_h2(acc[i][j][2], acc[i][j][3]);
            }
        }
}

// queries: q [n][H][192] f32 -> Qf [H][n][192] f16 * scale
__global__ void k_fa_qprep(const float* __restrict__ q, int n, float scale, half* __restrict__ qf) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * fa::H * fa::DK) return;
    const int d = idx % fa::DK, h = (idx / fa::DK) % fa::H, r = idx / (fa::DK * fa::H);
    qf[((size_t)h * n + r) * fa::DK + d] = __float2half(q[idx] * scale);
}
__global__ void k_fa_init(float* __restrict__ O, float* __restrict__ ml, int count_rows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count_rows * fa::DV) O[idx] = 0.f;
    if (idx < count_rows) { ml[2 * idx] = -INFINITY; ml[2 * idx + 1] = 0.f; }
}
// o[r][h*128 + d] = O[h][r][d] / l[h][r]
__global__ void k_fa_final(const float* __restrict__ O, const float* __restrict__ ml, int n, float* __restrict__ o) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * fa::H * fa::DV) return;
    const int d = idx % fa::DV, h = (idx / fa::DV) % fa::H, r = idx / (fa::DV * fa::H);
    const size_t hr = (size_t)h * n + r;
    o[idx] = O[hr * fa::DV + d] / ml[2 * hr + 1];
}

// One block = (128 queries, head). Keys of this launch: expanded rows [0, nk) = cache positions [kpos0, kpos0 + nk).
// Query i of the chunk sits at position qpos0 + i; it sees keys with position <= its own.
__global__ void __launch_bounds__(fa::THREADS, 1)
k_fa(const half* __restrict__ qf, int n, int qpos0, const half* __restrict__ E, const half* __restrict__ Cf, int nk,
     int kpos0, float* __restrict__ O, float* __restrict__ ml) {
    using namespace fa;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int h = blockIdx.y, q0 = blockIdx.x * BQ, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t4 = lane & 3;
    // keys visible to the last query of the block
    const int qlast = min(n - 1, q0 + BQ - 1);
    const int kend = min(nk, qpos0 + qlast + 1 - kpos0);
    if (kend <= 0) return;  // ml/O untouched (-inf / 0 merge = no-op)

    // this warp's 16 queries: Q fragments in registers (12 k-steps x 4 regs)
    const int wq0 = q0 + warp * 16;
    uint32_t qa[DK / 16][4];
    {
        const half* qb = qf + ((size_t)h * n) * DK;
        const int r0 = wq0 + g, r1 = wq0 + g + 8;
#pragma unroll
        for (int ks = 0; ks < DK / 16; ks++) {
            const int c = ks * 16 + 2 * t4;
            qa[ks][0] = r0 < n ? *(const uint32_t*)&qb[(size_t)r0 * DK + c] : 0u;
            qa[ks][1] = r1 < n ? *(const uint32_t*)&qb[(size_t)r1 * DK + c] : 0u;
            qa[ks][2] = r0 < n ? *(const uint32_t*)&qb[(size_t)r0 * DK + c + 8] : 0u;
            qa[ks][3] = r1 < n ? *(const uint32_t*)&qb[(size_t)r1 * DK + c + 8] : 0u;
        }
    }
    const int qp0 = qpos0 + wq0 + g, qp1 = qp0 + 8;  // positions of the two rows this thread holds

    auto issue = [&](int t0, int st) {
        if (t0 < kend) {
            // K_nope: 64 keys x 16 chunks, pe: 64 x 8 chunks, V: 64 x 16 chunks = 2560 copies -> 10 per thread
            for (int c = tid; c < BKEY * 40; c += THREADS) {
                const int key = c / 40, part = c % 40, j = t0 + key;
                const bool ok = j < kend;
                const int jj = ok ? j : 0;
                if (part < 16) {
                    fa_cp16(&S.k[st][key * KP + part * 8], E + (size_t)jj * EW + h * NOPE + part * 8, ok);
                } else if (part < 24) {
                    fa_cp16(&S.k[st][key * KP + NOPE + (part - 16) * 8], Cf + (size_t)jj * L + RK + (part - 16) * 8, ok);
                } else {
                    fa_cp16(&S.v[st][key * VP + (part - 24) * 8], E + (size_t)jj * EW + H * NOPE + h * DV + (part - 24) * 8, ok);
                }
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    float o[DV / 8][4];
#pragma unroll
    for (int i = 0; i < DV / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    issue(0, 0);
    int it = 0;
    for (int t0 = 0; t0 < kend; t0 += BKEY, it++) {
        const int st = it & 1;
        issue(t0 + BKEY, st ^ 1);
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();
        // whole tile in the future of this warp's last query: skip
        if (kpos0 + t0 <= qpos0 + wq0 + 15) {
            float s[BKEY / 8][4];
#pragma unroll
            for (int j = 0; j < BKEY / 8; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < DK / 16; ks++) {
#pragma unroll
                for (int j = 0; j < BKEY / 8; j += 2) {
                    uint32_t b0, b1, b2, b3;
                    fa_ldsm_x4(b0, b1, b2, b3,
                               &S.k[st][(j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * KP + ks * 16 + ((lane >> 3) & 1) * 8]);
                    fa_mma(s[j], qa[ks], b0, b1);
                    fa_mma(s[j + 1], qa[ks], b2, b3);
                }
            }
            // causal + range mask, tile max
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int j = 0; j < BKEY / 8; j++) {
                const int kp = kpos0 + t0 + j * 8 + 2 * t4;  // positions of columns 2t, 2t+1
                const bool in0 = t0 + j * 8 + 2 * t4 < kend, in1 = t0 + j * 8 + 2 * t4 + 1 < kend;
                if (!(in0 && kp <= qp0)) s[j][0] = -INFINITY;
                if (!(in1 && kp + 1 <= qp0)) s[j][1] = -INFINITY;
                if (!(in0 && kp <= qp1)) s[j][2] = -INFINITY;
                if (!(in1 && kp + 1 <= qp1)) s[j][3] = -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(s[j][0], s[j][1]));
                tm1 = fmaxf(tm1, fmaxf(s[j][2], s[j][3]));
            }
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
            const float mn0 = fmaxf(m0, tm0), mn1 = fmaxf(m1, tm1);
            const float a0 = mn0 == -INFINITY ? 1.f : __expf(m0 - mn0);
            const float a1 = mn1 == -INFINITY ? 1.f : __expf(m1 - mn1);
            float ps0 = 0.f, ps1 = 0.f;
            uint32_t pa[BKEY / 16][4];  // P as A fragments (k16 = two n8 tiles)
#pragma unroll
            for (int j = 0; j < BKEY / 8; j++) {
                const float p0 = s[j][0] == -INFINITY ? 0.f : __expf(s[j][0] - mn0);
                const float p1 = s[j][1] == -INFINITY ? 0.f : __expf(s[j][1] - mn0);
                const float p2 = s[j][2] == -INFINITY ? 0.f : __expf(s[j][2] - mn1);
                const float p3 = s[j][3] == -INFINITY ? 0.f : __expf(s[j][3] - mn1);
                const uint32_t lo = pack_h2(p0, p1), hi = pack_h2(p2, p3);
                const float2 lf = __half22float2(*(const half2*)&lo), hf = __half22float2(*(const half2*)&hi);
                ps0 += lf.x + lf.y;
                ps1 += hf.x + hf.y;
                pa[j >> 1][(j & 1) * 2 + 0] = lo;
                pa[j >> 1][(j & 1) * 2 + 1] = hi;
            }
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
            l0 = l0 * a0 + ps0;
            l1 = l1 * a1 + ps1;
            m0 = mn0;
            m1 = mn1;
#pragma unroll
            for (int i = 0; i < DV / 8; i++) { o[i][0] *= a0; o[i][1] *= a0; o[i][2] *= a1; o[i][3] *= a1; }
            // O += P V: V tile [key][d]; B operand (k = keys, n = dims) via ldmatrix.trans
#pragma unroll
            for (int kk = 0; kk < BKEY / 16; kk++) {
#pragma unroll
                for (int i = 0; i < DV / 8; i += 2) {
                    uint32_t b0, b1, b2, b3;
                    const int mi = lane >> 3, rr = lane & 7;
                    // matrices: (keys 0-7, d i*8), (keys 8-15, d i*8), (keys 0-7, d (i+1)*8), (keys 8-15, d (i+1)*8)
                    fa_ldsm_x4_t(b0, b1, b2, b3, &S.v[st][(kk * 16 + rr + 8 * (mi & 1)) * VP + (i + (mi >> 1)) * 8]);
                    fa_mma(o[i], pa[kk], b0, b1);
                    fa_mma(o[i + 1], pa[kk], b2, b3);
                }
            }
        }
        __syncthreads();
    }
    // merge (o, m, l) of this launch into the global accumulators
    // the 4 lanes of a quad share a row: all of them read (m, l) before any of them writes
    const int rr[2] = {wq0 + g, wq0 + g + 8};
    const float mb[2] = {m0, m1}, lb[2] = {l0, l1};
    bool act[2];
    float mo[2] = {-INFINITY, -INFINITY}, lo[2] = {0.f, 0.f};
#pragma unroll
    for (int x = 0; x < 2; x++) {
        act[x] = rr[x] < n && mb[x] != -INFINITY;
        if (act[x]) {
            const size_t hr = (size_t)h * n + rr[x];
            mo[x] = ml[2 * hr];
            lo[x] = ml[2 * hr + 1];
        }
    }
    __syncwarp();
#pragma unroll
    for (int x = 0; x < 2; x++) {
        if (!act[x]) continue;
        const size_t hr = (size_t)h * n + rr[x];
        const float M = fmaxf(mo[x], mb[x]);
        const float ao = mo[x] == -INFINITY ? 0.f : __expf(mo[x] - M), ab = __expf(mb[x] - M);
        float* Or = O + hr * DV;
#pragma unroll
        for (int i = 0; i < DV / 8; i++) {
            const int d = i * 8 + 2 * t4;
            const float v0 = x ? o[i][2] : o[i][0], v1 = x ? o[i][3] : o[i][1];
            Or[d] = Or[d] * ao + v0 * ab;
            Or[d + 1] = Or[d + 1] * ao + v1 * ab;
        }
        if (t4 == 0) { ml[2 * hr] = M; ml[2 * hr + 1] = lo[x] * ao + lb[x] * ab; }
    }
}

// ---------------------------------------------------------------- int8 scores variant
// K and Q in int8 with one scale for the 128 nope dims and one for the 64 pe dims (per key/query and head);
// QK^T on int8 m16n8k32 (nope: 4 k-steps, pe: 2 k-steps, accumulated in int32), PV on f16 as k_fa.
namespace fa8 {
constexpr int KP = fa::DK + 16;  // 208-byte int8 rows: ldmatrix conflict-free
struct Smem {
    int8_t k[2][fa::BKEY * KP];
    float ks[2][2][fa::BKEY];      // [nope | pe] scale per key
    half v[2][fa::BKEY * fa::VP];
};
// FP8 P.V: V^T tiles in e4m3 [128 dims][64 key positions] and the per-dim scales of the tile
constexpr int VP8 = fa::BKEY + 16;   // 80-byte rows: ldmatrix conflict-free
constexpr float kP8 = 448.f;         // P' = 448 P (e4m3 max), P in [0, 1]
constexpr float kV8 = 2.f;           // V' = V * 2 / max|V| per (128-key tile, dim): |P' V'| <= 896, a 64-key tile sum
                                     // stays < 65504 in the f16 accumulator
struct Smem8 {
    int8_t k[2][fa::BKEY * KP];
    float ks[2][2][fa::BKEY];
    uint8_t v[2][fa::DV * VP8];
    float vs[2][fa::DV];
};
// Key order inside each 16-key group of the e4m3 V^T rows: position p = 4t + i holds key i < 2 ? 2t + i : 8 + 2t + i - 2,
// which is how the thread (t = lane % 4) holds P after QK^T (n8 tiles: keys 2t, 2t+1, 8+2t, 9+2t), so P needs no
// shuffles to become the A operand of m16n8k32.
__host__ __device__ constexpr int pos8(int k) {
    return (k & ~15) + ((k & 15) < 8 ? 4 * ((k & 15) >> 1) + (k & 1) : 4 * (((k & 15) - 8) >> 1) + 2 + (k & 1));
}
}  // namespace fa8

#ifndef FA_PV_F32ACC
#define FA_PV_F32ACC 0
#endif
__device__ __forceinline__ void fa_mma_h(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void fa_mma_e4m3(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {   // lo in the low byte
    return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float2 e4m3x2_f(uint32_t x) {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)x, __NV_E4M3);
    return __half22float2(*(const half2*)&r);
}
__device__ __forceinline__ void fa_mma_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// int8 rows: one warp per (row, part) with part 0 = 128 nope values (from E, per head), part 1 = 64 pe values (from Cf,
// shared by heads). K8 [nk][H][192] (pe copied into every head's row), ks [nk][H][2].
__global__ void k_fa_kquant(const half* __restrict__ E, const half* __restrict__ Cf, int nk, int8_t* __restrict__ K8,
                            float* __restrict__ ks) {
    using namespace fa;
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= nk * H) return;
    const int key = gw / H, h = gw % H;
    // nope: 128 values = 4 per lane
    const half2* src = (const half2*)&E[(size_t)key * EW + h * NOPE + lane * 4];
    const float2 a = __half22float2(src[0]), b = __half22float2(src[1]);
    float mx = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
    mx = warp_max(mx);
    const float d = mx / 127.f, id = d > 0.f ? 1.f / d : 0.f;
    int8_t* dst = K8 + ((size_t)key * H + h) * DK;
    char4 q;
    q.x = (int8_t)roundf(a.x * id); q.y = (int8_t)roundf(a.y * id); q.z = (int8_t)roundf(b.x * id); q.w = (int8_t)roundf(b.y * id);
    *(char4*)&dst[lane * 4] = q;
    // pe: 64 values = 2 per lane
    const float2 p = __half22float2(*(const half2*)&Cf[(size_t)key * L + RK + lane * 2]);
    float pm = warp_max(fmaxf(fabsf(p.x), fabsf(p.y)));
    const float dp = pm / 127.f, ip = dp > 0.f ? 1.f / dp : 0.f;
    char2 qp;
    qp.x = (int8_t)roundf(p.x * ip); qp.y = (int8_t)roundf(p.y * ip);
    *(char2*)&dst[NOPE + lane * 2] = qp;
    if (lane == 0) { ks[((size_t)key * H + h) * 2] = d; ks[((size_t)key * H + h) * 2 + 1] = dp; }
}
// ---------------------------------------------------------------- int8 expansion (replaces dequant + f16 GEMM + kquant)
// For a block of nk cached keys of one slot/layer, on int8 tensor cores (4x the rate of the f16 / f32-accumulate GEMM):
//   acc[key][c] = sum_r cache_q[key][r] * W8[c][r]      int32 over the 512 latent values, exact (|acc| < 2^24)
//   value       = acc * latent_scale[key] * wsc[c]      (W8: k_b^T / v_b quantized per output channel at load)
// Column tiles 0..31 are K_nope of head h: quantized to int8 with one scale per (key, head) in the epilogue (the tile is
// the head's 128 columns), and the pe part is copied from the cache (already int8 with its row scale):
//   K8 [nk][H][192], ks8 [nk][H][2] (nope, pe), as k_fa_kquant wrote them.
// Column tiles 32..63 are V of head h - 32: f16 V [nk][H * 128].
// Requires the per-row latent cache scales (KV_ROWSCALE).
namespace ex8 {
constexpr int BM = 128, BN = 128, BK = 64, P = BK + 16, THREADS = 256, KD = 512;
}
__global__ void __launch_bounds__(ex8::THREADS, 2)
k_expand8(const int8_t* __restrict__ cq, const half* __restrict__ cs, size_t row0, int nk, const int8_t* __restrict__ w8,
          const float* __restrict__ wsc, int8_t* __restrict__ K8, float* __restrict__ ks8, half* __restrict__ V,
          uint8_t* __restrict__ V8, float* __restrict__ vsc, int v8stride) {
    using namespace ex8;
    __shared__ __align__(16) int8_t sa[2][BM * P], sb[2][BN * P];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, ct = blockIdx.y, n0 = ct * BN;
    auto issue = [&](int kb, int st) {
        if (kb < KD / BK) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int c = tid + i * THREADS, row = c >> 2, part = c & 3;
                const int key = m0 + row;
                const bool ok = key < nk;
                fa_cp16(&sa[st][row * P + part * 16], cq + (row0 + (ok ? key : 0)) * fa::L + kb * BK + part * 16, ok);
                fa_cp16(&sb[st][row * P + part * 16], w8 + (size_t)(n0 + row) * KD + kb * BK + part * 16, true);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };
    int acc[BN / 8][4];
#pragma unroll
    for (int j = 0; j < BN / 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
    issue(0, 0);
    for (int kb = 0; kb < KD / BK; kb++) {
        const int st = kb & 1;
        issue(kb + 1, st ^ 1);
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();
#pragma unroll
        for (int u = 0; u < BK / 32; u++) {
            uint32_t a[4];
            fa_ldsm_x4(a[0], a[1], a[2], a[3], &sa[st][(warp * 16 + (lane & 15)) * P + u * 32 + (lane >> 4) * 16]);
#pragma unroll
            for (int j = 0; j < BN / 8; j += 2) {
                uint32_t b0, b1, b2, b3;
                fa_ldsm_x4(b0, b1, b2, b3, &sb[st][(j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * P + u * 32 + ((lane >> 3) & 1) * 16]);
                fa_mma_s8(acc[j], a, b0, b1);
                fa_mma_s8(acc[j + 1], a, b2, b3);
            }
        }
        __syncthreads();
    }
    const int r0 = m0 + warp * 16 + g, r1 = r0 + 8;
    const float sa0 = r0 < nk ? __half2float(cs[(row0 + r0) * attn::KVS]) : 0.f;
    const float sa1 = r1 < nk ? __half2float(cs[(row0 + r1) * attn::KVS]) : 0.f;
    float v[BN / 8][4];
#pragma unroll
    for (int j = 0; j < BN / 8; j++) {
        const int c = n0 + j * 8 + 2 * t4;
        const float w0 = wsc[c], w1 = wsc[c + 1];
        v[j][0] = __int2float_rn(acc[j][0]) * (sa0 * w0);
        v[j][1] = __int2float_rn(acc[j][1]) * (sa0 * w1);
        v[j][2] = __int2float_rn(acc[j][2]) * (sa1 * w0);
        v[j][3] = __int2float_rn(acc[j][3]) * (sa1 * w1);
    }
    if (ct < fa::H) {  // K_nope of head ct: int8 per (key, head)
        const int h = ct;
        float mx0 = 0.f, mx1 = 0.f;
#pragma unroll
        for (int j = 0; j < BN / 8; j++) {
            mx0 = fmaxf(mx0, fmaxf(fabsf(v[j][0]), fabsf(v[j][1])));
            mx1 = fmaxf(mx1, fmaxf(fabsf(v[j][2]), fabsf(v[j][3])));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float d0 = mx0 / 127.f, i0 = d0 > 0.f ? 1.f / d0 : 0.f;
        const float d1 = mx1 / 127.f, i1 = d1 > 0.f ? 1.f / d1 : 0.f;
#pragma unroll
        for (int j = 0; j < BN / 8; j++) {
            const int c = j * 8 + 2 * t4;
            if (r0 < nk) {
                char2 q;
                q.x = (int8_t)__float2int_rn(v[j][0] * i0); q.y = (int8_t)__float2int_rn(v[j][1] * i0);
                *(char2*)&K8[((size_t)r0 * fa::H + h) * fa::DK + c] = q;
            }
            if (r1 < nk) {
                char2 q;
                q.x = (int8_t)__float2int_rn(v[j][2] * i1); q.y = (int8_t)__float2int_rn(v[j][3] * i1);
                *(char2*)&K8[((size_t)r1 * fa::H + h) * fa::DK + c] = q;
            }
        }
        if (t4 == 0) {
            if (r0 < nk) ks8[((size_t)r0 * fa::H + h) * 2] = d0;
            if (r1 < nk) ks8[((size_t)r1 * fa::H + h) * 2] = d1;
        }
        // pe: 64 int8 per key from the cache, and its scale
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = tid + i * THREADS, row = c >> 2, part = c & 3, key = m0 + row;
            if (key < nk) {
                *(uint4*)&K8[((size_t)key * fa::H + h) * fa::DK + fa::NOPE + part * 16] =
                    *(const uint4*)&cq[(row0 + key) * fa::L + fa::RK + part * 16];
                if (part == 0) ks8[((size_t)key * fa::H + h) * 2 + 1] = __half2float(cs[(row0 + key) * attn::KVS + 1]);
            }
        }
    } else if (V8) {  // V of head ct - 32 for FP8 P.V: e4m3 V^T [h][d][key position], scale per (128-key tile, dim)
        const int h = ct - fa::H;
        float cm[BN / 8][2];
#pragma unroll
        for (int j = 0; j < BN / 8; j++) {
            cm[j][0] = fmaxf(fabsf(v[j][0]), fabsf(v[j][2]));
            cm[j][1] = fmaxf(fabsf(v[j][1]), fabsf(v[j][3]));
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                cm[j][0] = fmaxf(cm[j][0], __shfl_xor_sync(0xffffffffu, cm[j][0], o));
                cm[j][1] = fmaxf(cm[j][1], __shfl_xor_sync(0xffffffffu, cm[j][1], o));
            }
        }
        float* red = reinterpret_cast<float*>(&sb[0][0]);   // [8 warps][128 dims], then the column max [128]
        float* cmax = red + 8 * BN;
        if (g == 0)
#pragma unroll
            for (int j = 0; j < BN / 8; j++) {
                red[warp * BN + j * 8 + 2 * t4] = cm[j][0];
                red[warp * BN + j * 8 + 2 * t4 + 1] = cm[j][1];
            }
        __syncthreads();
        if (tid < BN) {
            float m = 0.f;
#pragma unroll
            for (int w = 0; w < 8; w++) m = fmaxf(m, red[w * BN + tid]);
            cmax[tid] = m;
            vsc[((size_t)blockIdx.x * fa::H + h) * fa::DV + tid] = m / (fa8::kV8 * fa8::kP8);
        }
        __syncthreads();
        constexpr int TP = BM + 16;
        uint8_t* T = reinterpret_cast<uint8_t*>(&sa[0][0]);  // [128 dims][128 key positions]
        const int p0 = fa8::pos8(warp * 16 + g), p1 = fa8::pos8(warp * 16 + g + 8);
#pragma unroll
        for (int j = 0; j < BN / 8; j++) {
            const int d = j * 8 + 2 * t4;
            const float i0 = cmax[d] > 0.f ? fa8::kV8 / cmax[d] : 0.f, i1 = cmax[d + 1] > 0.f ? fa8::kV8 / cmax[d + 1] : 0.f;
            T[d * TP + p0] = __nv_cvt_float_to_fp8(v[j][0] * i0, __NV_SATFINITE, __NV_E4M3);
            T[(d + 1) * TP + p0] = __nv_cvt_float_to_fp8(v[j][1] * i1, __NV_SATFINITE, __NV_E4M3);
            T[d * TP + p1] = __nv_cvt_float_to_fp8(v[j][2] * i0, __NV_SATFINITE, __NV_E4M3);
            T[(d + 1) * TP + p1] = __nv_cvt_float_to_fp8(v[j][3] * i1, __NV_SATFINITE, __NV_E4M3);
        }
        __syncthreads();
        // rows past nk hold zeros (their latent scale is 0): the buffer stays finite everywhere
        for (int c = tid; c < BN * (BM / 16); c += THREADS) {
            const int d = c / (BM / 16), q = c % (BM / 16);
            *(uint4*)&V8[((size_t)h * fa::DV + d) * v8stride + m0 + q * 16] = *(const uint4*)&T[d * TP + q * 16];
        }
    } else {  // V of head ct - 32: f16
        const int h = ct - fa::H;
#pragma unroll
        for (int j = 0; j < BN / 8; j++) {
            const int c = h * fa::DV + j * 8 + 2 * t4;
            if (r0 < nk) *(uint32_t*)&V[(size_t)r0 * (fa::H * fa::DV) + c] = pack_h2(v[j][0], v[j][1]);
            if (r1 < nk) *(uint32_t*)&V[(size_t)r1 * (fa::H * fa::DV) + c] = pack_h2(v[j][2], v[j][3]);
        }
    }
}

// queries: q [n][H][192] f32 -> Q8 [H][n][192] int8, qs [H][n][2] (scale for nope, pe; softmax scale folded in)
__global__ void k_fa_qprep8(const float* __restrict__ q, int n, float scale, int8_t* __restrict__ Q8,
                            float* __restrict__ qs) {
    using namespace fa;
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= n * H) return;
    const int r = gw / H, h = gw % H;
    const float* src = q + ((size_t)r * H + h) * DK;
    const float4 a = *(const float4*)&src[lane * 4];
    float mx = warp_max(fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
    const float d = mx / 127.f, id = d > 0.f ? 1.f / d : 0.f;
    int8_t* dst = Q8 + ((size_t)h * n + r) * DK;
    char4 qv;
    qv.x = (int8_t)roundf(a.x * id); qv.y = (int8_t)roundf(a.y * id); qv.z = (int8_t)roundf(a.z * id); qv.w = (int8_t)roundf(a.w * id);
    *(char4*)&dst[lane * 4] = qv;
    const float2 p = *(const float2*)&src[NOPE + lane * 2];
    const float pm = warp_max(fmaxf(fabsf(p.x), fabsf(p.y)));
    const float dp = pm / 127.f, ip = dp > 0.f ? 1.f / dp : 0.f;
    char2 qp;
    qp.x = (int8_t)roundf(p.x * ip); qp.y = (int8_t)roundf(p.y * ip);
    *(char2*)&dst[NOPE + lane * 2] = qp;
    if (lane == 0) { qs[((size_t)h * n + r) * 2] = d * scale; qs[((size_t)h * n + r) * 2 + 1] = dp * scale; }
}

// PV8: P.V on FP8 tensor cores (e4m3 P' and V^T from k_expand8, f16 accumulate per tile, 2x the f16 rate); V8 rows
// have stride vstride. Otherwise f16 V from Vb with row stride vstride.
template <bool PV8>
__global__ void __launch_bounds__(fa::THREADS, 1)
k_fa8(const int8_t* __restrict__ Q8, const float* __restrict__ qs, int n, int qpos0, const int8_t* __restrict__ K8,
      const float* __restrict__ ks, const half* __restrict__ Vb, const uint8_t* __restrict__ V8,
      const float* __restrict__ vsc, int vstride, int nk, int kpos0, float* __restrict__ O, float* __restrict__ ml) {
    using namespace fa;
    constexpr int KP8 = fa8::KP;
    using SM = typename std::conditional<PV8, fa8::Smem8, fa8::Smem>::type;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    SM& S = *reinterpret_cast<SM*>(smem_raw);
    const int h = blockIdx.y, q0 = blockIdx.x * BQ, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const int qlast = min(n - 1, q0 + BQ - 1);
    const int kend = min(nk, qpos0 + qlast + 1 - kpos0);
    if (kend <= 0) return;

    const int wq0 = q0 + warp * 16;
    // Q fragments: 6 k32-steps x 4 regs (int8 A: rows g / g+8, bytes 4t..4t+3 and 16+4t..)
    uint32_t qa[DK / 32][4];
    float qsn[2], qsp[2];
    {
        const int8_t* qb = Q8 + ((size_t)h * n) * DK;
        const int r0 = wq0 + g, r1 = wq0 + g + 8;
#pragma unroll
        for (int ks_ = 0; ks_ < DK / 32; ks_++) {
            const int c = ks_ * 32 + 4 * t4;
            qa[ks_][0] = r0 < n ? *(const uint32_t*)&qb[(size_t)r0 * DK + c] : 0u;
            qa[ks_][1] = r1 < n ? *(const uint32_t*)&qb[(size_t)r1 * DK + c] : 0u;
            qa[ks_][2] = r0 < n ? *(const uint32_t*)&qb[(size_t)r0 * DK + c + 16] : 0u;
            qa[ks_][3] = r1 < n ? *(const uint32_t*)&qb[(size_t)r1 * DK + c + 16] : 0u;
        }
        qsn[0] = r0 < n ? qs[((size_t)h * n + r0) * 2] : 0.f;
        qsp[0] = r0 < n ? qs[((size_t)h * n + r0) * 2 + 1] : 0.f;
        qsn[1] = r1 < n ? qs[((size_t)h * n + r1) * 2] : 0.f;
        qsp[1] = r1 < n ? qs[((size_t)h * n + r1) * 2 + 1] : 0.f;
    }
    const int qp0 = qpos0 + wq0 + g, qp1 = qp0 + 8;

    auto issue = [&](int t0, int st) {
        if (t0 < kend) {
            // K: 64 keys x 12 chunks (192 B), V: 64 x 16 chunks, scales: 64 keys x 2 floats (8 B -> one 16 B chunk per 2 keys)
            if constexpr (PV8) {
                // K: 64 keys x 12 chunks; V^T: 128 dims x 4 chunks (64 key positions); scales: 128 floats = 32 chunks
                for (int c = tid; c < BKEY * 12 + DV * 4 + DV / 4; c += THREADS) {
                    if (c < BKEY * 12) {
                        const int key = c / 12, part = c % 12, j = t0 + key;
                        const bool ok = j < kend;
                        fa_cp16(&S.k[st][key * KP8 + part * 16], K8 + ((size_t)(ok ? j : 0) * H + h) * DK + part * 16, ok);
                    } else if (c < BKEY * 12 + DV * 4) {
                        const int x = c - BKEY * 12, d = x >> 2, q = x & 3;
                        fa_cp16(&S.v[st][d * fa8::VP8 + q * 16], V8 + ((size_t)h * DV + d) * vstride + t0 + q * 16, true);
                    } else {
                        const int x = c - BKEY * 12 - DV * 4;
                        fa_cp16(&S.vs[st][x * 4], vsc + ((size_t)(t0 / 128) * H + h) * DV + x * 4, true);
                    }
                }
            } else {
                for (int c = tid; c < BKEY * 28; c += THREADS) {
                    const int key = c / 28, part = c % 28, j = t0 + key;
                    const bool ok = j < kend;
                    const int jj = ok ? j : 0;
                    if (part < 12) {
                        fa_cp16(&S.k[st][key * KP8 + part * 16], K8 + ((size_t)jj * H + h) * DK + part * 16, ok);
                    } else {
                        fa_cp16(&S.v[st][key * VP + (part - 12) * 8], Vb + (size_t)jj * vstride + h * DV + (part - 12) * 8, ok);
                    }
                }
            }
            if (tid < BKEY) {
                const int j = t0 + tid;
                const bool ok = j < kend;
                S.ks[st][0][tid] = ok ? ks[((size_t)j * H + h) * 2] : 0.f;
                S.ks[st][1][tid] = ok ? ks[((size_t)j * H + h) * 2 + 1] : 0.f;
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    float o[DV / 8][4];
#pragma unroll
    for (int i = 0; i < DV / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    issue(0, 0);
    int it = 0;
    for (int t0 = 0; t0 < kend; t0 += BKEY, it++) {
        const int st = it & 1;
        issue(t0 + BKEY, st ^ 1);
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();
        if (kpos0 + t0 <= qpos0 + wq0 + 15) {
            float s[BKEY / 8][4];
#pragma unroll
            for (int j = 0; j < BKEY / 8; j += 2) {
                int cn0[4] = {0, 0, 0, 0}, cn1[4] = {0, 0, 0, 0}, cp0[4] = {0, 0, 0, 0}, cp1[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks_ = 0; ks_ < DK / 32; ks_++) {
                    uint32_t b0, b1, b2, b3;
                    fa_ldsm_x4(b0, b1, b2, b3,
                               &S.k[st][(j * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * KP8 + ks_ * 32 + ((lane >> 3) & 1) * 16]);
                    if (ks_ < NOPE / 32) { fa_mma_s8(cn0, qa[ks_], b0, b1); fa_mma_s8(cn1, qa[ks_], b2, b3); }
                    else { fa_mma_s8(cp0, qa[ks_], b0, b1); fa_mma_s8(cp1, qa[ks_], b2, b3); }
                }
#pragma unroll
                for (int x = 0; x < 2; x++) {
                    const int* cn = x ? cn1 : cn0;
                    const int* cp = x ? cp1 : cp0;
                    const int kc = (j + x) * 8 + 2 * t4;  // tile-local key of columns 0/1
                    const float kn0 = S.ks[st][0][kc], kn1 = S.ks[st][0][kc + 1];
                    const float kp0 = S.ks[st][1][kc], kp1 = S.ks[st][1][kc + 1];
                    s[j + x][0] = i2f_exact(cn[0]) * (qsn[0] * kn0) + i2f_exact(cp[0]) * (qsp[0] * kp0);
                    s[j + x][1] = i2f_exact(cn[1]) * (qsn[0] * kn1) + i2f_exact(cp[1]) * (qsp[0] * kp1);
                    s[j + x][2] = i2f_exact(cn[2]) * (qsn[1] * kn0) + i2f_exact(cp[2]) * (qsp[1] * kp0);
                    s[j + x][3] = i2f_exact(cn[3]) * (qsn[1] * kn1) + i2f_exact(cp[3]) * (qsp[1] * kp1);
                }
            }
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int j = 0; j < BKEY / 8; j++) {
                const int kp = kpos0 + t0 + j * 8 + 2 * t4;
                const bool in0 = t0 + j * 8 + 2 * t4 < kend, in1 = t0 + j * 8 + 2 * t4 + 1 < kend;
                if (!(in0 && kp <= qp0)) s[j][0] = -INFINITY;
                if (!(in1 && kp + 1 <= qp0)) s[j][1] = -INFINITY;
                if (!(in0 && kp <= qp1)) s[j][2] = -INFINITY;
                if (!(in1 && kp + 1 <= qp1)) s[j][3] = -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(s[j][0], s[j][1]));
                tm1 = fmaxf(tm1, fmaxf(s[j][2], s[j][3]));
            }
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
            const float mn0 = fmaxf(m0, tm0), mn1 = fmaxf(m1, tm1);
            const float a0 = mn0 == -INFINITY ? 1.f : __expf(m0 - mn0);
            const float a1 = mn1 == -INFINITY ? 1.f : __expf(m1 - mn1);
            float ps0 = 0.f, ps1 = 0.f;
            uint32_t pa[BKEY / 16][4];
            uint32_t pa8[BKEY / 32][4] = {};
#pragma unroll
            for (int j = 0; j < BKEY / 8; j++) {
                const float p0 = s[j][0] == -INFINITY ? 0.f : __expf(s[j][0] - mn0);
                const float p1 = s[j][1] == -INFINITY ? 0.f : __expf(s[j][1] - mn0);
                const float p2 = s[j][2] == -INFINITY ? 0.f : __expf(s[j][2] - mn1);
                const float p3 = s[j][3] == -INFINITY ? 0.f : __expf(s[j][3] - mn1);
                if constexpr (PV8) {
                    // P' = 448 P in e4m3; l sums the quantized values, so numerator and denominator see the same P
                    const uint32_t x01 = e4m3x2(p0 * fa8::kP8, p1 * fa8::kP8), x23 = e4m3x2(p2 * fa8::kP8, p3 * fa8::kP8);
                    const float2 f01 = e4m3x2_f(x01), f23 = e4m3x2_f(x23);
                    ps0 += (f01.x + f01.y) * (1.f / fa8::kP8);
                    ps1 += (f23.x + f23.y) * (1.f / fa8::kP8);
                    const int kk = j >> 2, reg = (j & 2), sh = (j & 1) * 16;
                    pa8[kk][reg] |= x01 << sh;
                    pa8[kk][reg + 1] |= x23 << sh;
                } else {
                    ps0 += p0 + p1;
                    ps1 += p2 + p3;
                    pa[j >> 1][(j & 1) * 2 + 0] = pack_h2(p0, p1);
                    pa[j >> 1][(j & 1) * 2 + 1] = pack_h2(p2, p3);
                }
            }
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
            l0 = l0 * a0 + ps0;
            l1 = l1 * a1 + ps1;
            m0 = mn0;
            m1 = mn1;
            if constexpr (PV8) {
                // per 64 dims: P'(16 x 32 keys) . V'^T rows (e4m3, f16 accumulate), then scaled into the f32 output
#pragma unroll
                for (int hd = 0; hd < 2; hd++) {
                    uint32_t th[DV / 16][2];
#pragma unroll
                    for (int i = 0; i < DV / 16; i++) th[i][0] = th[i][1] = 0u;
#pragma unroll
                    for (int kk = 0; kk < BKEY / 32; kk++) {
#pragma unroll
                        for (int i = 0; i < DV / 16; i += 2) {
                            uint32_t b0, b1, b2, b3;
                            fa_ldsm_x4(b0, b1, b2, b3,
                                       &S.v[st][(hd * 64 + i * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * fa8::VP8 + kk * 32 +
                                                ((lane >> 3) & 1) * 16]);
                            fa_mma_e4m3(th[i], pa8[kk], b0, b1);
                            fa_mma_e4m3(th[i + 1], pa8[kk], b2, b3);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < DV / 16; i++) {
                        const int d = hd * 64 + i * 8 + 2 * t4;
                        const float sc0 = S.vs[st][d], sc1 = S.vs[st][d + 1];
                        float* oo = o[hd * 8 + i];
                        const float2 lo = __half22float2(*(const half2*)&th[i][0]);
                        const float2 hi = __half22float2(*(const half2*)&th[i][1]);
                        oo[0] = oo[0] * a0 + lo.x * sc0; oo[1] = oo[1] * a0 + lo.y * sc1;
                        oo[2] = oo[2] * a1 + hi.x * sc0; oo[3] = oo[3] * a1 + hi.y * sc1;
                    }
                }
            } else {
#if FA_PV_F32ACC
#pragma unroll
            for (int i = 0; i < DV / 8; i++) { o[i][0] *= a0; o[i][1] *= a0; o[i][2] *= a1; o[i][3] *= a1; }
#pragma unroll
            for (int kk = 0; kk < BKEY / 16; kk++) {
#pragma unroll
                for (int i = 0; i < DV / 8; i += 2) {
                    uint32_t b0, b1, b2, b3;
                    const int mi = lane >> 3, rr = lane & 7;
                    fa_ldsm_x4_t(b0, b1, b2, b3, &S.v[st][(kk * 16 + rr + 8 * (mi & 1)) * VP + (i + (mi >> 1)) * 8]);
                    fa_mma(o[i], pa[kk], b0, b1);
                    fa_mma(o[i + 1], pa[kk], b2, b3);
                }
            }
#else
            // the tile's P V (64 keys) accumulates in f16 (full-rate mma on GeForce), then folds into the f32 output;
            // done in two halves of the 128 dims to bound registers
#pragma unroll
            for (int hd = 0; hd < 2; hd++) {
                uint32_t th[DV / 16][2];
#pragma unroll
                for (int i = 0; i < DV / 16; i++) th[i][0] = th[i][1] = 0u;
#pragma unroll
                for (int kk = 0; kk < BKEY / 16; kk++) {
#pragma unroll
                    for (int i = 0; i < DV / 16; i += 2) {
                        uint32_t b0, b1, b2, b3;
                        const int mi = lane >> 3, rr = lane & 7;
                        fa_ldsm_x4_t(b0, b1, b2, b3,
                                     &S.v[st][(kk * 16 + rr + 8 * (mi & 1)) * VP + (hd * 8 + i + (mi >> 1)) * 8]);
                        fa_mma_h(th[i], pa[kk], b0, b1);
                        fa_mma_h(th[i + 1], pa[kk], b2, b3);
                    }
                }
#pragma unroll
                for (int i = 0; i < DV / 16; i++) {
                    float* oo = o[hd * 8 + i];
                    const float2 lo = __half22float2(*(const half2*)&th[i][0]);
                    const float2 hi = __half22float2(*(const half2*)&th[i][1]);
                    oo[0] = oo[0] * a0 + lo.x; oo[1] = oo[1] * a0 + lo.y;
                    oo[2] = oo[2] * a1 + hi.x; oo[3] = oo[3] * a1 + hi.y;
                }
            }
#endif
            }
        }
        __syncthreads();
    }
    // merge into the global accumulators (as k_fa)
    const int rr[2] = {wq0 + g, wq0 + g + 8};
    const float mb[2] = {m0, m1}, lb[2] = {l0, l1};
    bool act[2];
    float mo[2] = {-INFINITY, -INFINITY}, lo[2] = {0.f, 0.f};
#pragma unroll
    for (int x = 0; x < 2; x++) {
        act[x] = rr[x] < n && mb[x] != -INFINITY;
        if (act[x]) {
            const size_t hr = (size_t)h * n + rr[x];
            mo[x] = ml[2 * hr];
            lo[x] = ml[2 * hr + 1];
        }
    }
    __syncwarp();
#pragma unroll
    for (int x = 0; x < 2; x++) {
        if (!act[x]) continue;
        const size_t hr = (size_t)h * n + rr[x];
        const float M = fmaxf(mo[x], mb[x]);
        const float ao = mo[x] == -INFINITY ? 0.f : __expf(mo[x] - M), ab = __expf(mb[x] - M);
        float* Or = O + hr * DV;
#pragma unroll
        for (int i = 0; i < DV / 8; i++) {
            const int d = i * 8 + 2 * t4;
            const float v0 = x ? o[i][2] : o[i][0], v1 = x ? o[i][3] : o[i][1];
            Or[d] = Or[d] * ao + v0 * ab;
            Or[d + 1] = Or[d + 1] * ao + v1 * ab;
        }
        if (t4 == 0) { ml[2 * hr] = M; ml[2 * hr + 1] = lo[x] * ao + lb[x] * ab; }
    }
}

}  // namespace raon
