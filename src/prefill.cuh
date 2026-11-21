// Batched prefill: int8 tensor-core GEMMs over the repacked Q8_0/Q6_K weights, parallel short
// conv, f16 tensor-core flash attention (causal + sliding window). Included by engine.cu.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#include "engine.h"
#include "kernels.cuh"

namespace le {

constexpr int PF_TMAX = 1024;  // tokens per prefill chunk

struct PrefillBufs {
    float *X = nullptr, *O = nullptr, *F = nullptr, *big = nullptr;  // [T][E] x3, [T][6144]
    int8_t *xq = nullptr, *yq = nullptr, *aq = nullptr;                // [T][E], [T][E], [T][FF]
    float *xd = nullptr, *yd = nullptr, *ad = nullptr;
    half* qh = nullptr;                                                // [T][16][128]
    int* toks = nullptr;
    int* h_toks = nullptr;                                             // pinned
};

template <class Alloc>
inline void prefill_alloc(PrefillBufs& p, const Config& c, int, Alloc alloc) {
    const size_t T = PF_TMAX, E = c.n_embd, F = c.n_ff;
    p.X = (float*)alloc(T * E * 4); p.O = (float*)alloc(T * E * 4); p.F = (float*)alloc(T * E * 4);
    p.big = (float*)alloc(T * 6144 * 4);
    p.xq = (int8_t*)alloc(T * E); p.yq = (int8_t*)alloc(T * E); p.aq = (int8_t*)alloc(T * F);
    p.xd = (float*)alloc(T * E / 32 * 4); p.yd = (float*)alloc(T * E / 32 * 4); p.ad = (float*)alloc(T * F / 32 * 4);
    p.qh = (half*)alloc(T * 16 * 128 * 2);
    p.toks = (int*)alloc(T * 4);
    cudaMallocHost(&p.h_toks, T * 4);
}

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
// elementwise kernels
// ------------------------------------------------------------------------------------------------
__global__ void k_embed_batch(QWeight w, const int* toks, int p0, int* hist, float* X) {
    const int t = blockIdx.x, tok = toks[t];
    if (threadIdx.x == 0) hist[p0 + t] = tok;
    for (int e = threadIdx.x; e < w.K; e += blockDim.x) X[(size_t)t * w.K + e] = deq(w, tok, e);
}

// per token: x = X + (O ? rms(O)*w_post : 0) (written back to X); h = rms(x)*w_pre -> int8 + scales
__global__ void __launch_bounds__(256) k_prep(float* X, const float* O, const float* w_post, const float* w_pre,
                                              float eps, int8_t* xq, float* xd) {
    constexpr int K = 2048;
    __shared__ float red[32];
    const int t = blockIdx.x, e0 = threadIdx.x * 8;
    float* x = X + (size_t)t * K;
    float v[8];
    float4 a0 = *(const float4*)(x + e0), a1 = *(const float4*)(x + e0 + 4);
    v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w; v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
    if (O) {
        const float* o = O + (size_t)t * K;
        float4 b0 = *(const float4*)(o + e0), b1 = *(const float4*)(o + e0 + 4);
        float ov[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        float ss = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) ss += ov[i] * ov[i];
        ss = block_sum<256>(ss, red);
        const float r = rsqrtf(ss / K + eps);
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] += ov[i] * r * w_post[e0 + i];
        *(float4*)(x + e0) = make_float4(v[0], v[1], v[2], v[3]);
        *(float4*)(x + e0 + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
    float ss = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) ss += v[i] * v[i];
    ss = block_sum<256>(ss, red);
    const float r = rsqrtf(ss / K + eps);
    float amax = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) { v[i] = v[i] * r * w_pre[e0 + i]; amax = fmaxf(amax, fabsf(v[i])); }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) lo |= (uint32_t)(uint8_t)(int8_t)roundf(v[i] * id) << (8 * i);
#pragma unroll
    for (int i = 0; i < 4; i++) hi |= (uint32_t)(uint8_t)(int8_t)roundf(v[4 + i] * id) << (8 * i);
    *(uint2*)(xq + (size_t)t * K + e0) = make_uint2(lo, hi);
    if ((threadIdx.x & 3) == 0) xd[(size_t)t * (K / 32) + (e0 >> 5)] = d;
}

// short conv over a chunk. bcx: [T][6144] in repacked row order (tiles of [b32 c32 x32]).
__device__ __forceinline__ int bcx_col(int part, int ch) { return (ch >> 5) * 96 + part * 32 + (ch & 31); }

// short conv over a chunk: CTA = 32 tokens x 128 channels; b*x for tokens t0-6..t0+31 staged in smem
__global__ void __launch_bounds__(256) k_conv_prefill(const float* bcx, const float* conv_w, const float* state, int T,
                                                      int8_t* yq, float* yd) {
    constexpr int E = 2048, TB = 32, CB = 128;
    __shared__ float bx[TB + 6][CB];
    const int t0 = blockIdx.x * TB, ch0 = blockIdx.y * CB;
    for (int i = threadIdx.x; i < (TB + 6) * CB; i += 256) {
        const int r = i / CB, cl = i % CB, ch = ch0 + cl;
        const int tok = t0 - 6 + r;
        float v = 0.f;
        if (tok >= 0 && tok < T) {
            const float* row = bcx + (size_t)tok * 6144;
            v = row[bcx_col(0, ch)] * row[bcx_col(2, ch)];
        } else if (tok < 0) {
            v = state[ch * 6 + 6 + tok];
        }
        bx[r][cl] = v;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int item = warp; item < TB * (CB / 32); item += 8) {
        const int r = item / (CB / 32), cg = item % (CB / 32);
        const int t = t0 + r;
        if (t >= T) continue;  // uniform per warp
        const int cl = cg * 32 + lane, ch = ch0 + cl;
        const float* cw = conv_w + ch * 7;
        float acc = 0;
#pragma unroll
        for (int j = 0; j < 7; j++) acc += cw[j] * bx[r + j][cl];
        const float y = bcx[(size_t)t * 6144 + bcx_col(1, ch)] * acc;
        const float amax = warp_max(fabsf(y));
        const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
        yq[(size_t)t * E + ch] = (int8_t)roundf(y * id);
        if (lane == 0) yd[(size_t)t * (E / 32) + (ch >> 5)] = d;
    }
}

__global__ void k_conv_state_update(const float* bcx, int T, float* state) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= 2048) return;
    float old[6], nw[6];
#pragma unroll
    for (int j = 0; j < 6; j++) old[j] = state[ch * 6 + j];
#pragma unroll
    for (int j = 0; j < 6; j++) {
        const int tt = T - 6 + j;  // token index of new state slot j
        if (tt >= 0) {
            const float* row = bcx + (size_t)tt * 6144;
            nw[j] = row[bcx_col(0, ch)] * row[bcx_col(2, ch)];
        } else {
            nw[j] = old[j + T];
        }
    }
#pragma unroll
    for (int j = 0; j < 6; j++) state[ch * 6 + j] = nw[j];
}

// q/k RMS norm + NEOX rope; q -> qh (f16, pre-scaled by 1/sqrt(128)), k/v -> cache (f16)
__global__ void __launch_bounds__(768) k_qk_rope_cache(const float* qkvg, int p0, const float* qn, const float* kn,
                                                       const float* ff, float base, float eps, half* qh, half* kc,
                                                       half* vc, int cap) {
    const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = p0 + t;
    const float* row = qkvg + (size_t)t * 3088;
    float x[4];
    if (warp < 16) {
        float4 v = *(const float4*)(row + warp * 128 + lane * 4);
        x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
        rope_norm4(x, qn, ff, base, eps, pos, lane);
        const float s = rsqrtf(128.f);
        half* dst = qh + ((size_t)t * 16 + warp) * 128 + lane * 4;
        *(uint2*)dst = make_uint2(pack_h2(x[0] * s, x[1] * s), pack_h2(x[2] * s, x[3] * s));
    } else if (warp < 20) {
        const int h = warp - 16;
        float4 v = *(const float4*)(row + 2048 + h * 128 + lane * 4);
        x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
        rope_norm4(x, kn, ff, base, eps, pos, lane);
        half* dst = kc + ((size_t)h * cap + pos % cap) * 128 + lane * 4;
        *(uint2*)dst = make_uint2(pack_h2(x[0], x[1]), pack_h2(x[2], x[3]));
    } else {
        const int h = warp - 20;
        float4 v = *(const float4*)(row + 2560 + h * 128 + lane * 4);
        half* dst = vc + ((size_t)h * cap + pos % cap) * 128 + lane * 4;
        *(uint2*)dst = make_uint2(pack_h2(v.x, v.y), pack_h2(v.z, v.w));
    }
}

// ------------------------------------------------------------------------------------------------
// flash attention (prefill): grid (ceil(T/32), 4 kv heads); 8 warps = 4 q heads x 2 token tiles of 16
// ------------------------------------------------------------------------------------------------
constexpr int FA_BQ = 32, FA_BK = 64;

__global__ void __launch_bounds__(256) k_flash_prefill(const half* qh, const float* qkvg, const half* kc, const half* vc,
                                                       int cap, int window, int p0, int T, int8_t* oq, float* od) {
    __shared__ __align__(128) half Ks[FA_BK * 128];
    __shared__ __align__(128) half Vs[FA_BK * 128];
    const int kvh = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, tig = lane & 3;
    const int qhd = kvh * 4 + (warp & 3);          // query head of this warp
    const int t0 = blockIdx.x * FA_BQ + (warp >> 2) * 16;  // first token (chunk-relative) of this warp
    const int qp_first = p0 + blockIdx.x * FA_BQ;
    const int qp_last = p0 + min(T, (blockIdx.x + 1) * FA_BQ) - 1;
    const int k_lo = window ? max(0, qp_first - window + 1) : 0;

    // Q fragments (16 tokens x 128 dims) straight from global
    uint32_t qa[8][4];
    {
        const int ta = min(t0 + g, T - 1), tb = min(t0 + g + 8, T - 1);
        const half* ra = qh + ((size_t)ta * 16 + qhd) * 128;
        const half* rb = qh + ((size_t)tb * 16 + qhd) * 128;
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            qa[kk][0] = *(const uint32_t*)(ra + kk * 16 + tig * 2);
            qa[kk][1] = *(const uint32_t*)(rb + kk * 16 + tig * 2);
            qa[kk][2] = *(const uint32_t*)(ra + kk * 16 + 8 + tig * 2);
            qa[kk][3] = *(const uint32_t*)(rb + kk * 16 + 8 + tig * 2);
        }
    }
    const int qpa = p0 + t0 + g, qpb = qpa + 8;  // absolute positions of this thread's two rows
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
    const float LOG2E = 1.4426950408889634f;

    const half* kbase = kc + (size_t)kvh * cap * 128;
    const half* vbase = vc + (size_t)kvh * cap * 128;
    for (int kb = k_lo; kb <= qp_last; kb += FA_BK) {
        __syncthreads();
        // load K/V tile: 64 rows x 16 chunks of 16B each, swizzled
        for (int i = threadIdx.x; i < FA_BK * 16; i += 256) {
            const int r = i >> 4, c = i & 15;
            const int kp = kb + r;
            const bool ok = kp <= qp_last;
            const size_t off = (size_t)((ok ? kp : kb) % cap) * 128 + c * 8;
            const int sc = (c & 8) | ((c ^ r) & 7);
            cp_async16(Ks + r * 128 + sc * 8, kbase + off, ok);
            cp_async16(Vs + r * 128 + sc * 8, vbase + off, ok);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        if (t0 >= T) continue;
        // S = Q K^T  (16 x 64)
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
#pragma unroll
            for (int np = 0; np < 4; np++) {  // pairs of n-tiles (16 keys)
                const int mi = lane >> 3, r = (lane & 7) + (mi >> 1) * 8 + np * 16;
                const int c = kk * 2 + (mi & 1);
                const int sc = (c & 8) | ((c ^ r) & 7);
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, Ks + r * 128 + sc * 8);
                mma_f16(s[np * 2], qa[kk], b0, b1);
                mma_f16(s[np * 2 + 1], qa[kk], b2, b3);
            }
        }
        // mask + online softmax
        float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kp = kb + nt * 8 + tig * 2 + e;
                const bool va = kp <= qpa && (!window || qpa - kp < window);
                const bool vb = kp <= qpb && (!window || qpb - kp < window);
                s[nt][e] = va ? s[nt][e] * LOG2E : -INFINITY;
                s[nt][2 + e] = vb ? s[nt][2 + e] * LOG2E : -INFINITY;
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
        float sa = 0, sb = 0;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[nt][e] = mna == -INFINITY ? 0.f : exp2f(s[nt][e] - mna);
                s[nt][2 + e] = mnb == -INFINITY ? 0.f : exp2f(s[nt][2 + e] - mnb);
                sa += s[nt][e]; sb += s[nt][2 + e];
            }
        }
        la = la * ca + sa;  // per-thread partial; reduced across tig at the end
        lb = lb * cb + sb;
#pragma unroll
        for (int i = 0; i < 16; i++) { o[i][0] *= ca; o[i][1] *= ca; o[i][2] *= cb; o[i][3] *= cb; }
        // O += P V   (per 64-key tile: f16 partial sums, P in [0,1], then accumulated into f32 O)
#if FA_PV_F16
        uint32_t ot[16][2];
#pragma unroll
        for (int i = 0; i < 16; i++) ot[i][0] = ot[i][1] = 0u;
#endif
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {  // 16 keys per step
            uint32_t pa[4];
            pa[0] = pack_h2(s[2 * kk][0], s[2 * kk][1]);
            pa[1] = pack_h2(s[2 * kk][2], s[2 * kk][3]);
            pa[2] = pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[3] = pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int dp = 0; dp < 8; dp++) {  // pairs of 8-dim n-tiles
                const int mi = lane >> 3;
                const int r = kk * 16 + (lane & 7) + (mi & 1) * 8;
                const int c = dp * 2 + (mi >> 1);
                const int sc = (c & 8) | ((c ^ r) & 7);
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, Vs + r * 128 + sc * 8);
#if FA_PV_F16
                mma_f16_acc16(ot[dp * 2], pa, b0, b1);
                mma_f16_acc16(ot[dp * 2 + 1], pa, b2, b3);
#else
                mma_f16(o[dp * 2], pa, b0, b1);
                mma_f16(o[dp * 2 + 1], pa, b2, b3);
#endif
            }
        }
#if FA_PV_F16
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float2 x = __half22float2(*(const half2*)&ot[i][0]);
            const float2 y = __half22float2(*(const half2*)&ot[i][1]);
            o[i][0] += x.x; o[i][1] += x.y; o[i][2] += y.x; o[i][3] += y.y;
        }
#endif
    }
    if (t0 >= T) return;
    la += __shfl_xor_sync(0xffffffff, la, 1);
    la += __shfl_xor_sync(0xffffffff, la, 2);
    lb += __shfl_xor_sync(0xffffffff, lb, 1);
    lb += __shfl_xor_sync(0xffffffff, lb, 2);
    // gate, normalise, quantise per 32 dims (4 n-tiles x 4 tig lanes)
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
