// Prefill flash attention on int8 / FP8 tensor cores over the q8_0 KV cache. Included by engine.cu after prefill.cuh.
// Same grid, warp roles and masks as k_flash_prefill<true, G> (prefill.cuh), which stays for the f16 KV cache and as
// the reference (LIQUID_FA16=1 selects it for q8 KV too).
//
//   S = Q K^T: int8 m16n8k32. Q is int8 with one scale per (token, head) (k_qk_rope_cache q8out; the softmax scale
//     and log2(e) are folded into it). K is the cache's int8 as stored: each k32 step is exactly one q8_0 block, and
//     its int32 sum is scaled by that block's scale (exact int -> float, |sum| <= 32 * 127^2 < 2^22).
//   O += P V: e4m3 m16n8k32 with f16 accumulation per 64-key tile (2x the f16 mma rate; f32-accumulate FP8 is not
//     faster on GeForce Ada). P' = 448 P; V' = V * 2 / (127 dmax) per (tile, 32-dim block), where dmax is the largest
//     q8 scale of that block in the tile, so |V'| <= 2 and a 64-key tile sum stays below 64 * 896 < 65504. l sums the
//     e4m3-rounded P, so numerator and denominator see the same weights (raon-engine's recipe, prefill_attn.cuh).
#pragma once
#include <cuda_fp8.h>

#include "prefill.cuh"

namespace le {

namespace f8 {
constexpr int KP = 144;   // int8 K rows: 128 + 16 pad (ldmatrix conflict-free)
constexpr int VP = 80;    // e4m3 V^T rows: 64 key positions + 16 pad (ldmatrix conflict-free)
constexpr float kP = 448.f, kV = 2.f;
struct Smem {
    int8_t k[2][FA_BK * KP];
    uint8_t v[2][128 * VP];
    float ks[2][4][FA_BK];   // K scale per (block, key)
    float vf[2][4];          // output factor per 32-dim block: 127 dmax / (kV kP)
};
constexpr size_t SMEM = sizeof(Smem);
// V^T key order inside each 16-key group: thread t of a quad holds P for keys 2t, 2t+1, 8+2t, 9+2t after QK^T (the
// m16n8 C layout); stored at positions 4t .. 4t+3 they are the A-operand bytes of m16n8k32 without shuffles.
__host__ __device__ constexpr int pos(int k) {
    return (k & ~15) + ((k & 15) < 8 ? 4 * ((k & 15) >> 1) + (k & 1) : 4 * (((k & 15) - 8) >> 1) + 2 + (k & 1));
}
__device__ __forceinline__ void mma_e4m3(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {  // lo in the low byte
    return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float2 e4m3x2_f(uint32_t x) {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)x, __NV_E4M3);
    return __half22float2(*(const half2*)&r);
}
// one word of 4 int8 values times c -> two e4m3 pairs (lo: bytes 0,1; hi: bytes 2,3). Exact int8 -> f16
// (bytes ^ 0x80 under exponent 0x64 give 1024 + u, minus 1152 gives x), one f16 multiply, one conversion.
__device__ __forceinline__ void q8x4_to_e4m3(uint32_t w, half2 c, uint32_t& lo, uint32_t& hi) {
    const half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
    const uint32_t x = w ^ 0x80808080u;
    uint32_t a = __byte_perm(x, 0x64646464u, 0x4140), b = __byte_perm(x, 0x64646464u, 0x4342);
    half2 ha = __hmul2(__hsub2(*(const half2*)&a, bias), c);
    half2 hb = __hmul2(__hsub2(*(const half2*)&b, bias), c);
    lo = __nv_cvt_halfraw2_to_fp8x2(*(const __half2_raw*)&ha, __NV_SATFINITE, __NV_E4M3);
    hi = __nv_cvt_halfraw2_to_fp8x2(*(const __half2_raw*)&hb, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half pick4(half2 a, half2 b, int i) {
    return i == 0 ? __low2half(a) : i == 1 ? __high2half(a) : i == 2 ? __low2half(b) : __high2half(b);
}
}  // namespace f8

template <bool G = false>
__global__ void __launch_bounds__(256) k_flash_prefill8(const int8_t* q8, const float* qs8, const float* qkvg,
                                                        const void* kc, const void* vc, const half* ks, const half* vs,
                                                        int cap, int window, int p0, int T, int8_t* oq, float* od,
                                                        const int* grp = nullptr, const int2* gspan = nullptr,
                                                        bool prefix_all = false) {
    extern __shared__ __align__(128) uint8_t fa_dyn[];
    f8::Smem& S = *reinterpret_cast<f8::Smem*>(fa_dyn);
    const int kvh = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int g = lane >> 2, tig = lane & 3;
    const int qhd = kvh * 4 + (warp & 3);                  // query head of this warp
    const int t0 = blockIdx.x * FA_BQ + (warp >> 2) * 16;  // first token (chunk-relative) of this warp
    const int qp_first = p0 + blockIdx.x * FA_BQ;
    const int qp_last = p0 + min(T, (blockIdx.x + 1) * FA_BQ) - 1;
    int k_lo, k_end, gap_lo = 0, gap_hi = 0;
    if constexpr (G) {
        const int gf = grp[qp_first], gl = grp[qp_last];
        k_lo = 0;
        k_end = gspan[0].y;
        if (prefix_all && gf == 0) {
            k_end = T;
        } else if (gl > 0) {
            gap_lo = k_end;
            gap_hi = gspan[max(gf, 1)].x;
            k_end = gspan[gl].y;
        }
    } else {
        k_lo = window ? max(0, qp_first - window + 1) : 0;
        k_end = qp_last + 1;
    }
    auto next_kb = [&](int kb_) {
        int nk = kb_ + FA_BK;
        if constexpr (G) {
            if (nk >= gap_lo && nk < gap_hi) nk = gap_hi;
        }
        return nk;
    };

    // Q fragments: 4 k32 steps (= the 4 q8_0 blocks of a K row)
    uint32_t qa[4][4];
    float qsa, qsb;
    {
        const int ta = min(t0 + g, T - 1), tb = min(t0 + g + 8, T - 1);
        const int8_t* ra = q8 + ((size_t)ta * 16 + qhd) * 128;
        const int8_t* rb = q8 + ((size_t)tb * 16 + qhd) * 128;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            qa[b][0] = *(const uint32_t*)(ra + b * 32 + tig * 4);
            qa[b][1] = *(const uint32_t*)(rb + b * 32 + tig * 4);
            qa[b][2] = *(const uint32_t*)(ra + b * 32 + 16 + tig * 4);
            qa[b][3] = *(const uint32_t*)(rb + b * 32 + 16 + tig * 4);
        }
        qsa = qs8[(size_t)ta * 16 + qhd];
        qsb = qs8[(size_t)tb * 16 + qhd];
    }
    const int qpa = p0 + t0 + g, qpb = qpa + 8;  // absolute positions of this thread's two rows
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;

    // ---- tile i+1 is fetched into registers while tile i is computed, then stored (V converted) ----
    const int8_t* kbase = (const int8_t*)kc;
    const int8_t* vbase = (const int8_t*)vc;
    auto row_of = [&](int kp, int kb_) { return (size_t)kvh * cap + (kp < k_end ? kp : kb_) % cap; };
    // V item of this thread: key group vg (V^T positions 4 vg .. 4 vg + 3), dims 8 vdc .. 8 vdc + 7. A warp reads 8 key
    // rows x 32 contiguous bytes per load (full sectors).
    const int vg = (lane & 7) + 8 * (warp & 1), vdc = (lane >> 3) + 4 * (warp >> 1);
    const int vkey0 = (vg >> 2) * 16 + 2 * (vg & 3);  // keys vkey0 + {0, 1, 8, 9}
    int4 pk[2];        // K: 2 x 16 B of raw int8
    float pks;         // K scale of (key tid / 4, block tid % 4)
    int2 pv[4];        // V: 8 int8 dims of each of the 4 keys
    half pvd[4];       // their q8 scales (block vdc / 4)
    uint2 pvm[2];      // the 4 block scales of keys 2 lane, 2 lane + 1 (for the tile max)
    auto fetch = [&](int kb_) {
#pragma unroll
        for (int it = 0; it < 2; it++) {
            const int i = tid + it * 256, r = i >> 3, c16 = i & 7;
            const int kp = kb_ + r;
            pk[it] = kp < k_end ? *(const int4*)(kbase + row_of(kp, kb_) * 128 + c16 * 16) : make_int4(0, 0, 0, 0);
        }
        {
            const int kp = kb_ + (tid >> 2);
            pks = kp < k_end ? __half2float(ks[row_of(kp, kb_) * 4 + (tid & 3)]) : 0.f;
        }
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int kp = kb_ + vkey0 + (e & 1) + (e >> 1) * 8;
            const size_t row = row_of(kp, kb_);
            const bool ok = kp < k_end;
            pv[e] = ok ? *(const int2*)(vbase + row * 128 + vdc * 8) : make_int2(0, 0);
            pvd[e] = ok ? vs[row * 4 + (vdc >> 2)] : __float2half(0.f);
        }
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int kp = kb_ + 2 * lane + e;
            pvm[e] = kp < k_end ? *(const uint2*)(vs + row_of(kp, kb_) * 4) : make_uint2(0u, 0u);
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int it = 0; it < 2; it++) {
            const int i = tid + it * 256, r = i >> 3, c16 = i & 7;
            *(int4*)(S.k[buf] + r * f8::KP + c16 * 16) = pk[it];
        }
        S.ks[buf][tid & 3][tid >> 2] = pks;
        // per-block max scale of the tile (every warp computes it; q8 scales are >= 0)
        half2 m01 = __hmax2(*(const half2*)&pvm[0].x, *(const half2*)&pvm[1].x);
        half2 m23 = __hmax2(*(const half2*)&pvm[0].y, *(const half2*)&pvm[1].y);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const uint32_t a = __shfl_xor_sync(0xffffffffu, *(const uint32_t*)&m01, off);
            const uint32_t b = __shfl_xor_sync(0xffffffffu, *(const uint32_t*)&m23, off);
            m01 = __hmax2(m01, *(const half2*)&a);
            m23 = __hmax2(m23, *(const half2*)&b);
        }
        if (warp == 0 && lane < 4) S.vf[buf][lane] = __half2float(f8::pick4(m01, m23, lane)) * (127.f / (f8::kV * f8::kP));
        const float dmax = __half2float(f8::pick4(m01, m23, vdc >> 2));
        const float cf = dmax > 0.f ? f8::kV / (127.f * dmax) : 0.f;
        // e4m3 pairs per key: p[e][i] = dims (2i, 2i+1) of key e
        uint32_t p[4][4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const half2 c = __float2half2_rn(__half2float(pvd[e]) * cf);
            f8::q8x4_to_e4m3((uint32_t)pv[e].x, c, p[e][0], p[e][1]);
            f8::q8x4_to_e4m3((uint32_t)pv[e].y, c, p[e][2], p[e][3]);
        }
        // the V^T word of dim j: keys (vkey0, +1, +8, +9) = bytes of positions 4 vg .. 4 vg + 3
        uint32_t w[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t A = p[0][i] | (p[1][i] << 16), B = p[2][i] | (p[3][i] << 16);
            w[2 * i] = __byte_perm(A, B, 0x6420);
            w[2 * i + 1] = __byte_perm(A, B, 0x7531);
        }
        // store j writes row 8 vdc + (j ^ 2a), a = vdc % 4: the 4 dim chunks of a warp land 8 banks apart
        const int a = vdc & 3;
        if (a & 1) {
#pragma unroll
            for (int j = 0; j < 8; j++)
                if (!(j & 2)) { const uint32_t t = w[j]; w[j] = w[j ^ 2]; w[j ^ 2] = t; }
        }
        if (a & 2) {
#pragma unroll
            for (int j = 0; j < 8; j++)
                if (!(j & 4)) { const uint32_t t = w[j]; w[j] = w[j ^ 4]; w[j ^ 4] = t; }
        }
        uint8_t* vt = S.v[buf] + (size_t)(vdc * 8) * f8::VP + vg * 4;
#pragma unroll
        for (int j = 0; j < 8; j++) *(uint32_t*)(vt + (j ^ (2 * a)) * f8::VP) = w[j];
    };
    fetch(k_lo);
    store(0);

    int ga = 0, gb = 0;  // G: groups of this thread's two query rows
    if constexpr (G) {
        ga = grp[min(qpa, qp_last)];
        gb = grp[min(qpb, qp_last)];
    }
    int iter = 0;
    for (int kb = k_lo; kb < k_end; kb = next_kb(kb), iter++) {
        const int cur = iter & 1;
        const int kb_next = next_kb(kb);
        const bool has_next = kb_next < k_end;
        __syncthreads();  // tile cur stored by everyone; buffer cur^1 no longer read
        if (has_next) fetch(kb_next);  // global loads overlap this tile's math; stored after it
        if (t0 >= T) {
            if (has_next) store(cur ^ 1);
            continue;
        }
        const int8_t* Ks = S.k[cur];
        // S = Q K^T (16 x 64): per q8_0 block an int32 m16n8k32 sum, scaled by the keys' block scales
        float s[8][4];
#pragma unroll
        for (int b = 0; b < 4; b++) {
#pragma unroll
            for (int np = 0; np < 4; np++) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3,
                        Ks + (np * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * f8::KP + b * 32 + ((lane >> 3) & 1) * 16);
                int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                mma_s8_k32(c0, qa[b], b0, b1);
                mma_s8_k32(c1, qa[b], b2, b3);
#pragma unroll
                for (int x = 0; x < 2; x++) {
                    const int nt = np * 2 + x;
                    const int* c = x ? c1 : c0;
                    const float2 kd = *(const float2*)&S.ks[cur][b][nt * 8 + 2 * tig];
                    if (b == 0) {
                        s[nt][0] = i2f_magic(c[0]) * kd.x; s[nt][1] = i2f_magic(c[1]) * kd.y;
                        s[nt][2] = i2f_magic(c[2]) * kd.x; s[nt][3] = i2f_magic(c[3]) * kd.y;
                    } else {
                        s[nt][0] += i2f_magic(c[0]) * kd.x; s[nt][1] += i2f_magic(c[1]) * kd.y;
                        s[nt][2] += i2f_magic(c[2]) * kd.x; s[nt][3] += i2f_magic(c[3]) * kd.y;
                    }
                }
            }
        }
        // mask + online softmax (log2 domain: qs8 carries log2(e))
        float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kp = kb + nt * 8 + tig * 2 + e;
                bool va, vb;
                if constexpr (G) {
                    const int gk = kp < k_end ? grp[kp] : -1;
                    va = gk == 0 || (gk > 0 && (gk == ga || (prefix_all && ga == 0)));
                    vb = gk == 0 || (gk > 0 && (gk == gb || (prefix_all && gb == 0)));
                } else {
                    va = kp <= qpa && (!window || qpa - kp < window);
                    vb = kp <= qpb && (!window || qpb - kp < window);
                }
                s[nt][e] = va ? s[nt][e] * qsa : -INFINITY;
                s[nt][2 + e] = vb ? s[nt][2 + e] * qsb : -INFINITY;
                mxa = fmaxf(mxa, s[nt][e]);
                mxb = fmaxf(mxb, s[nt][2 + e]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffff, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffff, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffff, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffff, mxb, 2));
        const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
        const float ca = mna == -INFINITY ? 1.f : exp2f(ma - mna);
        const float cb = mnb == -INFINITY ? 1.f : exp2f(mb - mnb);
        ma = mna; mb = mnb;
        // P' = 448 P in e4m3, packed as the A operand (32 keys per k step); l sums the rounded values
        uint32_t pa[2][4] = {};
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const float p0 = mna == -INFINITY ? 0.f : exp2f(s[nt][0] - mna);
            const float p1 = mna == -INFINITY ? 0.f : exp2f(s[nt][1] - mna);
            const float p2 = mnb == -INFINITY ? 0.f : exp2f(s[nt][2] - mnb);
            const float p3 = mnb == -INFINITY ? 0.f : exp2f(s[nt][3] - mnb);
            const uint32_t x01 = f8::e4m3x2(p0 * f8::kP, p1 * f8::kP), x23 = f8::e4m3x2(p2 * f8::kP, p3 * f8::kP);
            const float2 f01 = f8::e4m3x2_f(x01), f23 = f8::e4m3x2_f(x23);
            sa += f01.x + f01.y;
            sb += f23.x + f23.y;
            const int kk = nt >> 2, reg = nt & 2, sh = (nt & 1) * 16;
            pa[kk][reg] |= x01 << sh;
            pa[kk][reg + 1] |= x23 << sh;
        }
        la = la * ca + sa * (1.f / f8::kP);  // per-thread partial; reduced across tig at the end
        lb = lb * cb + sb * (1.f / f8::kP);
        // O = O * c + (P' V'^T) * vf, per 64 dims: e4m3 m16n8k32, f16 accumulation over the tile
        const uint8_t* Vs = S.v[cur];
#pragma unroll
        for (int hd = 0; hd < 2; hd++) {
            uint32_t th[8][2];
#pragma unroll
            for (int i = 0; i < 8; i++) th[i][0] = th[i][1] = 0u;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
#pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3,
                            Vs + (hd * 64 + i * 8 + (lane & 7) + ((lane >> 4) & 1) * 8) * f8::VP + kk * 32 + ((lane >> 3) & 1) * 16);
                    f8::mma_e4m3(th[i], pa[kk], b0, b1);
                    f8::mma_e4m3(th[i + 1], pa[kk], b2, b3);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float f = S.vf[cur][hd * 2 + (i >> 2)];
                float* oo = o[hd * 8 + i];
                const float2 lo = __half22float2(*(const half2*)&th[i][0]);
                const float2 hi = __half22float2(*(const half2*)&th[i][1]);
                oo[0] = oo[0] * ca + lo.x * f; oo[1] = oo[1] * ca + lo.y * f;
                oo[2] = oo[2] * cb + hi.x * f; oo[3] = oo[3] * cb + hi.y * f;
            }
        }
        if (has_next) store(cur ^ 1);  // next tile, converted while other warps still compute
    }
    if (t0 >= T) return;
    la += __shfl_xor_sync(0xffffffff, la, 1);
    la += __shfl_xor_sync(0xffffffff, la, 2);
    lb += __shfl_xor_sync(0xffffffff, lb, 1);
    lb += __shfl_xor_sync(0xffffffff, lb, 2);
    // gate, normalise, quantise per 32 dims (4 n-tiles x 4 tig lanes), as k_flash_prefill
#pragma unroll
    for (int half_ = 0; half_ < 2; half_++) {
        const int t = t0 + g + half_ * 8;
        const bool ok = t < T;
        const float gl = ok ? qkvg[(size_t)t * 3088 + 3072 + qhd] : 0.f;
        const float f = (1.f / (1.f + __expf(-gl))) / (half_ ? lb : la);
#pragma unroll
        for (int b = 0; b < 4; b++) {
            float v[8], amax = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                v[j * 2] = o[b * 4 + j][half_ * 2] * f;
                v[j * 2 + 1] = o[b * 4 + j][half_ * 2 + 1] * f;
                amax = fmaxf(amax, fmaxf(fabsf(v[j * 2]), fabsf(v[j * 2 + 1])));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
            const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
            if (ok) {
                int8_t* dst = oq + (size_t)t * 2048 + qhd * 128 + b * 32 + tig * 2;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    char2 q;
                    q.x = (int8_t)roundf(v[j * 2] * id);
                    q.y = (int8_t)roundf(v[j * 2 + 1] * id);
                    *(char2*)(dst + j * 8) = q;
                }
                if (tig == 0) od[(size_t)t * 64 + qhd * 4 + b] = d;
            }
        }
    }
}

}  // namespace le
