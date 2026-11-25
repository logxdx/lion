// Prefill MLA attention in expanded form (as llama.cpp's mla_expand path), f16 tensor cores, f32 accumulate.
//
// For a chunk of n queries of one slot, per block of cached keys:
//   Cf = dequant(latent cache rows)                      [KB][576] f16 (512 latent | 64 pe)
//   E  = Cf[:, :512] . Wexp^T                            [KB][8192] f16: per head K_nope (128) | then per head V (128)
//   flash attention per (128 queries, head): K = [K_nope_h | pe] (192), V = V_h (128), causal, scale folded into Q
// Results of key blocks are merged into global (O, m, l); k_fa_final normalises.
#pragma once
#include <cstdint>

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
    const float d = __half2float(cs[row * (fa::L / 32) + c / 4]);
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

__global__ void __launch_bounds__(fa::THREADS, 1)
k_fa8(const int8_t* __restrict__ Q8, const float* __restrict__ qs, int n, int qpos0, const int8_t* __restrict__ K8,
      const float* __restrict__ ks, const half* __restrict__ E, int nk, int kpos0, float* __restrict__ O,
      float* __restrict__ ml) {
    using namespace fa;
    constexpr int KP8 = fa8::KP;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    fa8::Smem& S = *reinterpret_cast<fa8::Smem*>(smem_raw);
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
            for (int c = tid; c < BKEY * 28; c += THREADS) {
                const int key = c / 28, part = c % 28, j = t0 + key;
                const bool ok = j < kend;
                const int jj = ok ? j : 0;
                if (part < 12) {
                    fa_cp16(&S.k[st][key * KP8 + part * 16], K8 + ((size_t)jj * H + h) * DK + part * 16, ok);
                } else {
                    fa_cp16(&S.v[st][key * VP + (part - 12) * 8], E + (size_t)jj * EW + H * NOPE + h * DV + (part - 12) * 8, ok);
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
#pragma unroll
            for (int j = 0; j < BKEY / 8; j++) {
                const float p0 = s[j][0] == -INFINITY ? 0.f : __expf(s[j][0] - mn0);
                const float p1 = s[j][1] == -INFINITY ? 0.f : __expf(s[j][1] - mn0);
                const float p2 = s[j][2] == -INFINITY ? 0.f : __expf(s[j][2] - mn1);
                const float p3 = s[j][3] == -INFINITY ? 0.f : __expf(s[j][3] - mn1);
                ps0 += p0 + p1;
                ps1 += p2 + p3;
                pa[j >> 1][(j & 1) * 2 + 0] = pack_h2(p0, p1);
                pa[j >> 1][(j & 1) * 2 + 1] = pack_h2(p2, p3);
            }
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
            l0 = l0 * a0 + ps0;
            l1 = l1 * a1 + ps1;
            m0 = mn0;
            m1 = mn1;
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
