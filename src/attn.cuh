// MLA attention over the q8 latent cache, all 32 heads per pass, on tensor cores.
//
// Cache per (slot, MLA layer): int8 values [ctx][576] + half scales [ctx][18] (q8_0 per 32, as llama.cpp).
// A query row (one token) has Q = [q_lat (512) | q_pe (64)] per head, scaled by 1/sqrt(192), quantized to int8 per
// 32 values (like llama.cpp's q8_1 Q for quantized-K attention).
//
// Per tile of KT keys:
//   scores  S^T (keys x heads) = K Q^T on int8 mma m16n8k32: one k-step = one q8 block, rescaled by key x query scale
//   softmax online, per head
//   output  O^T (512 latent dims x heads) += V^T P^T on f16 mma (V = first 512 cache values, dequantized to f16)
// Raw int8 tiles stream through a 2-stage cp.async pipeline. Keys are split across blocks (flash-decoding);
// k_mla_combine merges the splits.
#pragma once
#include <cstdint>

#include "util.cuh"

namespace raon {

namespace attn {
constexpr int H = 32, L = 576, RK = 512, NB = L / 32;  // heads, row length, latent rank, q8 blocks per row
constexpr int KT = 32;                                  // keys per tile
constexpr int LQ = L + 16;                              // int8 row pitch (bytes): ldmatrix conflict-free
constexpr int VP = RK + 8;                              // f16 V row pitch (halfs)
constexpr int PP = KT + 8;                              // P row pitch (halfs)
constexpr int THREADS = 256;
struct Smem {
    int8_t q[H * LQ];
    float qs[H * NB];
    int8_t k[2][KT * LQ];
    half ks[2][KT * NB];
    half v[KT * VP];
    half p[H * PP];
    float pm[2][H], ps[2][H];
    float m[H], l[H], alpha[H];
};
}  // namespace attn

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_s8(int* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, bool pred) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// cache row = q8 of [RMSNorm_w(kv[0:512]) | kv[512:576]] at the row's position. grid R, block 256 (8 warps)
template <class RowT>
__global__ void k_mla_cache_write_q8(const float* __restrict__ kv, const float* __restrict__ w, float eps,
                                     const RowT* __restrict__ rows, int8_t* __restrict__ cq, half* __restrict__ cs,
                                     size_t slot_rows) {
    using namespace attn;
    __shared__ float red[32], v[L];
    const int r = blockIdx.x;
    const float* kr = kv + (size_t)r * L;
    float ss = 0.f;
    for (int i = threadIdx.x; i < RK; i += blockDim.x) ss += kr[i] * kr[i];
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / RK + eps);
    for (int i = threadIdx.x; i < L; i += blockDim.x) v[i] = i < RK ? kr[i] * inv * w[i] : kr[i];
    __syncthreads();
    const size_t row = (size_t)rows[r].slot * slot_rows + rows[r].pos;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < NB; b += blockDim.x >> 5) {
        const float x = v[b * 32 + lane];
        const float amax = warp_max(fabsf(x));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        cq[row * L + b * 32 + lane] = (int8_t)roundf(x * id);
        if (lane == 0) cs[row * NB + b] = __float2half(d);
    }
}

// Q = [q_lat | q_pe] * scale, int8 per 32 + float scale: qi [R][H][576], qsc [R][H][18]. one warp per 32-block
__global__ void k_mla_qprep(const float* __restrict__ qlat, const float* __restrict__ q, int qk, float scale, int R,
                            int8_t* __restrict__ qi, float* __restrict__ qsc) {
    using namespace attn;
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= R * H * NB) return;
    const int b = gw % NB, h = (gw / NB) % H, r = gw / (NB * H);
    const int i = b * 32 + lane;
    const float v = (i < RK ? qlat[((size_t)r * H + h) * RK + i] : q[((size_t)r * H + h) * qk + (qk - (L - RK)) + (i - RK)]) *
                    scale;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
    qi[((size_t)r * H + h) * L + i] = (int8_t)roundf(v * id);
    if (lane == 0) qsc[((size_t)r * H + h) * NB + b] = d;
}

// One block = (row, split). Partial results: opart [R][nsplit][H][512] (unnormalised), ml [R][nsplit][H][2] (max, sum)
template <class RowT>
__global__ void __launch_bounds__(attn::THREADS, 1)
k_mla_decode(const int8_t* __restrict__ qi, const float* __restrict__ qsc, const int8_t* __restrict__ cq,
             const half* __restrict__ cs, size_t slot_rows, const RowT* __restrict__ rows, int split, int nsplit,
             float* __restrict__ opart, float* __restrict__ ml) {
    using namespace attn;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int r = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nkeys = rows[r].pos + 1;
    const int k0 = sp * split, k1 = min(nkeys, k0 + split);
    float* op = opart + (((size_t)r * nsplit + sp) * H) * RK;
    float* mlp = ml + (((size_t)r * nsplit + sp) * H) * 2;
    if (k0 >= k1) {  // nothing in this split
        if (tid < H) { mlp[2 * tid] = -INFINITY; mlp[2 * tid + 1] = 0.f; }
        return;
    }
    const size_t base = (size_t)rows[r].slot * slot_rows;
    const int8_t* Kq = cq + base * L;
    const half* Ks = cs + base * NB;

    // issue the cp.async copies of the tile starting at key t0 into stage st (zero-filled beyond k1)
    auto issue = [&](int t0, int st) {
#ifdef ATTN_NO_LOAD
        if (t0 > k0 + KT) { cp_commit(); return; }  // ablation: only the first two tiles are ever loaded
#endif
        if (t0 < k1) {
            for (int c = tid; c < KT * (L / 16); c += THREADS) {
                const int key = c / (L / 16), col = c % (L / 16), gk = t0 + key;
                const bool ok = gk < k1;
                cp_async16(&S.k[st][key * LQ + col * 16], Kq + (size_t)(ok ? gk : t0) * L + col * 16, ok);
            }
            for (int wd = tid; wd < KT * NB / 2; wd += THREADS) {  // 9 words of scales per key
                const int key = wd / (NB / 2), wi = wd % (NB / 2), gk = t0 + key;
                const bool ok = gk < k1;
                cp_async4((char*)&S.ks[st][key * NB] + wi * 4, (const char*)(Ks + (size_t)(ok ? gk : t0) * NB) + wi * 4,
                          ok);
            }
        }
        cp_commit();  // always commit: keeps the group count uniform
    };
    issue(k0, 0);
    issue(k0 + KT, 1);

    // Q -> smem
    const int8_t* qr = qi + (size_t)r * H * L;
    for (int c = tid; c < H * L / 16; c += THREADS) {
        const int h = c / (L / 16), col = c % (L / 16);
        *(int4*)&S.q[h * LQ + col * 16] = *(const int4*)&qr[h * L + col * 16];
    }
    for (int i = tid; i < H * NB; i += THREADS) S.qs[i] = qsc[(size_t)r * H * NB + i];
    if (tid < H) { S.m[tid] = -INFINITY; S.l[tid] = 0.f; }

    // O^T accumulators: warp owns latent dims [warp*64, +64) = 4 m-tiles; heads = 4 n-tiles
    float acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

    const int g = lane >> 2, t4 = lane & 3;
    const int mt = warp >> 2, nt = warp & 3;  // score tile of this warp: keys mt*16.., heads nt*8..
    const int key0 = mt * 16 + g, h0 = nt * 8 + 2 * t4;
    float qreg0[NB], qreg1[NB];  // this thread's two heads' query scales, for every q8 block
#pragma unroll
    for (int b = 0; b < NB; b++) {
        qreg0[b] = qsc[((size_t)r * H + h0) * NB + b];
        qreg1[b] = qsc[((size_t)r * H + h0 + 1) * NB + b];
    }
    int it = 0;
    for (int t0 = k0; t0 < k1; t0 += KT, it++) {
        const int st = it & 1;
        cp_wait<1>();  // this tile's group has landed (the next one may still be in flight)
        __syncthreads();
#ifdef ATTN_NO_MATH
        __syncthreads();
        issue(t0 + 2 * KT, st);
        continue;
#endif

        // ---- scores on int8 tensor cores
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        {
            const int8_t* ab = &S.k[st][(mt * 16 + (lane & 15)) * LQ + (lane >> 4) * 16];
            const int8_t* bb = &S.q[(nt * 8 + (lane & 7)) * LQ + ((lane >> 3) & 1) * 16];
            const half* ksr0 = &S.ks[st][key0 * NB];
            const half* ksr1 = &S.ks[st][(key0 + 8) * NB];
            const float* qs0 = qreg0;
            const float* qs1 = qreg1;
#ifndef ATTN_SKIP_S
#pragma unroll
            for (int b = 0; b < NB; b++) {
#else
            for (int b = 0; b < 0; b++) {
#endif
                uint32_t a0, a1, a2, a3, b0, b1;
                ldsm_x4(a0, a1, a2, a3, ab + b * 32);
                ldsm_x2(b0, b1, bb + b * 32);
                int c[4];
                mma_s8(c, a0, a1, a2, a3, b0, b1);
                const float k0s = __half2float(ksr0[b]), k1s = __half2float(ksr1[b]);
                const float q0s = qs0[b], q1s = qs1[b];
                s[0] += (float)c[0] * (k0s * q0s);
                s[1] += (float)c[1] * (k0s * q1s);
                s[2] += (float)c[2] * (k1s * q0s);
                s[3] += (float)c[3] * (k1s * q1s);
            }
            if (t0 + key0 >= k1) { s[0] = -INFINITY; s[1] = -INFINITY; }
            if (t0 + key0 + 8 >= k1) { s[2] = -INFINITY; s[3] = -INFINITY; }
        }
        // ---- V tile (first 512 values of each key) to f16
#ifndef ATTN_SKIP_V
        for (int c = tid; c < KT * (RK / 16); c += THREADS) {
#else
        for (int c = tid; c < 0; c += THREADS) {
#endif
            const int key = c / (RK / 16), col = c % (RK / 16);
            const uint4 raw = *(const uint4*)&S.k[st][key * LQ + col * 16];
            const half2 d2 = __half2half2(S.ks[st][key * NB + col / 2]);
            // int8 -> f16 exactly: bytes b^0x80 = b+128 placed under exponent 0x64 give 1024+b+128; minus 1152 = b
            const half2 bias = __float2half2_rn(1152.f);
            const uint32_t wv[4] = {raw.x ^ 0x80808080u, raw.y ^ 0x80808080u, raw.z ^ 0x80808080u, raw.w ^ 0x80808080u};
            uint32_t o[8];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t lo = __byte_perm(wv[e], 0x64646464u, 0x5140), hi = __byte_perm(wv[e], 0x64646464u, 0x7362);
                const half2 hl = __hmul2(__hsub2(*(const half2*)&lo, bias), d2);
                const half2 hh = __hmul2(__hsub2(*(const half2*)&hi, bias), d2);
                o[2 * e] = *(const uint32_t*)&hl;
                o[2 * e + 1] = *(const uint32_t*)&hh;
            }
            *(uint4*)&S.v[key * VP + col * 16] = make_uint4(o[0], o[1], o[2], o[3]);
            *(uint4*)&S.v[key * VP + col * 16 + 8] = make_uint4(o[4], o[5], o[6], o[7]);
        }
        // ---- online softmax: per-head max over the tile's 32 keys (2 warps x 16 keys each)
        float mx0 = fmaxf(s[0], s[2]), mx1 = fmaxf(s[1], s[3]);
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
        }
        if (g == 0) { S.pm[mt][h0] = mx0; S.pm[mt][h0 + 1] = mx1; }
        __syncthreads();
        {
            const float mn0 = fmaxf(S.m[h0], fmaxf(S.pm[0][h0], S.pm[1][h0]));
            const float mn1 = fmaxf(S.m[h0 + 1], fmaxf(S.pm[0][h0 + 1], S.pm[1][h0 + 1]));
            const half p0 = __float2half(s[0] == -INFINITY ? 0.f : __expf(s[0] - mn0));
            const half p1 = __float2half(s[1] == -INFINITY ? 0.f : __expf(s[1] - mn1));
            const half p2 = __float2half(s[2] == -INFINITY ? 0.f : __expf(s[2] - mn0));
            const half p3 = __float2half(s[3] == -INFINITY ? 0.f : __expf(s[3] - mn1));
            S.p[h0 * PP + key0] = p0;
            S.p[(h0 + 1) * PP + key0] = p1;
            S.p[h0 * PP + key0 + 8] = p2;
            S.p[(h0 + 1) * PP + key0 + 8] = p3;
            float sm0 = __half2float(p0) + __half2float(p2), sm1 = __half2float(p1) + __half2float(p3);
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                sm0 += __shfl_xor_sync(0xffffffffu, sm0, o);
                sm1 += __shfl_xor_sync(0xffffffffu, sm1, o);
            }
            if (g == 0) { S.ps[mt][h0] = sm0; S.ps[mt][h0 + 1] = sm1; }
        }
        __syncthreads();
        if (tid < H) {
            const float mold = S.m[tid], mnew = fmaxf(mold, fmaxf(S.pm[0][tid], S.pm[1][tid]));
            const float al = mold == -INFINITY ? 0.f : __expf(mold - mnew);
            S.alpha[tid] = al;
            S.l[tid] = S.l[tid] * al + S.ps[0][tid] + S.ps[1][tid];
            S.m[tid] = mnew;
        }
        __syncthreads();
        issue(t0 + 2 * KT, st);  // this stage's raw data is consumed (scores + V tile)

        // ---- O^T = alpha * O^T + V^T P^T on f16 tensor cores
        {
            float al[4][2];
#pragma unroll
            for (int n = 0; n < 4; n++) { al[n][0] = S.alpha[n * 8 + 2 * t4]; al[n][1] = S.alpha[n * 8 + 2 * t4 + 1]; }
#pragma unroll
            for (int a = 0; a < 4; a++)
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    acc[a][n][0] *= al[n][0]; acc[a][n][1] *= al[n][1];
                    acc[a][n][2] *= al[n][0]; acc[a][n][3] *= al[n][1];
                }
#pragma unroll
#ifndef ATTN_SKIP_PV
            for (int ks = 0; ks < KT / 16; ks++) {
#else
            for (int ks = 0; ks < 0; ks++) {
#endif
                uint32_t b[4][2];
#pragma unroll
                for (int n = 0; n < 4; n++)
                    ldsm_x2(b[n][0], b[n][1], &S.p[(n * 8 + (lane & 7)) * PP + ks * 16 + ((lane >> 3) & 1) * 8]);
#pragma unroll
                for (int a = 0; a < 4; a++) {
                    const int d0 = warp * 64 + a * 16;
                    const int mi = lane >> 3, rr = lane & 7;
                    uint32_t a0, a1, a2, a3;
                    ldsm_x4_t(a0, a1, a2, a3, &S.v[(ks * 16 + rr + 8 * (mi >> 1)) * VP + d0 + 8 * (mi & 1)]);
#pragma unroll
                    for (int n = 0; n < 4; n++) mma16816(acc[a][n], a0, a1, a2, a3, b[n][0], b[n][1]);
                }
            }
        }
        // the next iteration's first __syncthreads protects S.v / S.p before they are rewritten
    }
    cp_wait<0>();
    __syncthreads();
    // write partial O (as [head][dim]) and (m, l)
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int n = 0; n < 4; n++) {
            const int d = warp * 64 + a * 16 + g, h = n * 8 + 2 * t4;
            op[h * RK + d] = acc[a][n][0];
            op[(h + 1) * RK + d] = acc[a][n][1];
            op[h * RK + d + 8] = acc[a][n][2];
            op[(h + 1) * RK + d + 8] = acc[a][n][3];
        }
    if (tid < H) { mlp[2 * tid] = S.m[tid]; mlp[2 * tid + 1] = S.l[tid]; }
}

// merge splits: grid (R, H), block 128 (4 dims each)
__global__ void k_mla_combine(const float* __restrict__ opart, const float* __restrict__ ml, int nsplit,
                              float* __restrict__ olat) {
    using namespace attn;
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; s++) M = fmaxf(M, ml[(((size_t)r * nsplit + s) * H + h) * 2]);
    float Lsum = 0.f, acc[4] = {0, 0, 0, 0};
    for (int s = 0; s < nsplit; s++) {
        const float* mp = &ml[(((size_t)r * nsplit + s) * H + h) * 2];
        if (mp[0] == -INFINITY) continue;
        const float wgt = __expf(mp[0] - M);
        Lsum += wgt * mp[1];
        const float4 o = *(const float4*)&opart[(((size_t)r * nsplit + s) * H + h) * RK + t * 4];
        acc[0] += wgt * o.x; acc[1] += wgt * o.y; acc[2] += wgt * o.z; acc[3] += wgt * o.w;
    }
    const float inv = 1.f / Lsum;
    *(float4*)&olat[((size_t)r * H + h) * RK + t * 4] = make_float4(acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv);
}

}  // namespace raon
