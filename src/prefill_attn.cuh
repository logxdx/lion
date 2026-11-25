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

}  // namespace raon
