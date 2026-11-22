// Batched prefill: int8 tensor-core GEMMs over the repacked Q8_0/Q6_K weights, parallel short
// conv, f16 tensor-core flash attention (causal + sliding window). Included by engine.cu.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#include "engine.h"
#include "kernels.cuh"
#include "gemm.cuh"

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
                                                       const float* ff, float base, float eps, half* qh, void* kc,
                                                       void* vc, half* ks, half* vs, int cap, bool q8) {
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
        store_kv_row4(x, kc, ks, (size_t)h * cap + pos % cap, lane, q8);
    } else {
        const int h = warp - 20;
        float4 v = *(const float4*)(row + 2560 + h * 128 + lane * 4);
        x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
        store_kv_row4(x, vc, vs, (size_t)h * cap + pos % cap, lane, q8);
    }
}

// ------------------------------------------------------------------------------------------------
// flash attention (prefill): grid (ceil(T/32), 4 kv heads); 8 warps = 4 q heads x 2 token tiles of 16
// ------------------------------------------------------------------------------------------------
constexpr int FA_BQ = 32, FA_BK = 64;
constexpr size_t FA_SMEM = (size_t)4 * FA_BK * 128 * 2;  // double-buffered f16 K and V tiles (64 KB, dynamic)

// dequantise one 16-value q8 chunk (+ its block scale) into two swizzled 8-half chunks of tile row r.
// int8 -> f16 without I2F: bytes^0x80 under exponent 0x64 give 1024+u exactly, minus 1152 gives x; then one
// f16 multiply by the scale (bit-identical to __float2half((float)x * d), see tests/dequant_q8_test.cu)
__device__ __forceinline__ void fa_store_q8_chunk(half* T_, int r, int c16, int4 q, float d) {
    const int w[4] = {q.x, q.y, q.z, q.w};
    uint32_t hv[8];
    const half2 d2 = __float2half2_rn(d);
    const half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint32_t x = (uint32_t)w[k] ^ 0x80808080u;
        uint32_t a = __byte_perm(x, 0x64646464u, 0x4140), b = __byte_perm(x, 0x64646464u, 0x4342);
        const half2 ha = __hmul2(__hsub2(*(const half2*)&a, bias), d2);
        const half2 hb = __hmul2(__hsub2(*(const half2*)&b, bias), d2);
        hv[k * 2] = *(const uint32_t*)&ha;
        hv[k * 2 + 1] = *(const uint32_t*)&hb;
    }
    const int c0 = c16 * 2, c1 = c16 * 2 + 1;
    *(uint4*)(T_ + r * 128 + ((c0 & 8) | ((c0 ^ r) & 7)) * 8) = make_uint4(hv[0], hv[1], hv[2], hv[3]);
    *(uint4*)(T_ + r * 128 + ((c1 & 8) | ((c1 ^ r) & 7)) * 8) = make_uint4(hv[4], hv[5], hv[6], hv[7]);
}

template <bool Q8>
__global__ void __launch_bounds__(256) k_flash_prefill(const half* qh, const float* qkvg, const void* kc, const void* vc,
                                                       const half* ks, const half* vs, int cap, int window, int p0,
                                                       int T, int8_t* oq, float* od) {
    extern __shared__ __align__(128) uint8_t fa_dyn[];
    half* const KsB = (half*)fa_dyn;          // [2][FA_BK*128]
    half* const VsB = KsB + 2 * FA_BK * 128;  // [2][FA_BK*128]
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

    // ---- K/V tile pipeline: tile i+1 is in flight while tile i is computed ----
    const half* kbase = (const half*)kc + (size_t)kvh * cap * 128;
    const half* vbase = (const half*)vc + (size_t)kvh * cap * 128;
    auto issue_f16 = [&](int kb_, int buf) {  // cp.async: 64 rows x 16 chunks of 16 B, swizzled
        half* Kd = KsB + buf * FA_BK * 128;
        half* Vd = VsB + buf * FA_BK * 128;
        for (int i = threadIdx.x; i < FA_BK * 16; i += 256) {
            const int r = i >> 4, c = i & 15;
            const int kp = kb_ + r;
            const bool ok = kp <= qp_last;
            const size_t off = (size_t)((ok ? kp : kb_) % cap) * 128 + c * 8;
            const int sc = (c & 8) | ((c ^ r) & 7);
            cp_async16(Kd + r * 128 + sc * 8, kbase + off, ok);
            cp_async16(Vd + r * 128 + sc * 8, vbase + off, ok);
        }
        cp_commit();
    };
    int4 pq[2][2];   // q8: prefetched raw chunks [item][k|v] (64 rows x 8 chunks = 2 items per thread)
    float pd[2][2];
    auto fetch_q8 = [&](int kb_) {
#pragma unroll
        for (int it = 0; it < 2; it++) {
            const int i = threadIdx.x + it * 256, r = i >> 3, c16 = i & 7;
            const int kp = kb_ + r;
            const bool ok = kp <= qp_last;
            const size_t row = (size_t)kvh * cap + (ok ? kp : kb_) % cap;
#pragma unroll
            for (int kv = 0; kv < 2; kv++) {
                const int8_t* src = (const int8_t*)(kv ? vc : kc) + row * 128 + c16 * 16;
                pq[it][kv] = ok ? *(const int4*)src : make_int4(0, 0, 0, 0);
                pd[it][kv] = ok ? __half2float((kv ? vs : ks)[row * 4 + (c16 >> 1)]) : 0.f;
            }
        }
    };
    auto store_q8 = [&](int buf) {
#pragma unroll
        for (int it = 0; it < 2; it++) {
            const int i = threadIdx.x + it * 256, r = i >> 3, c16 = i & 7;
            fa_store_q8_chunk(KsB + buf * FA_BK * 128, r, c16, pq[it][0], pd[it][0]);
            fa_store_q8_chunk(VsB + buf * FA_BK * 128, r, c16, pq[it][1], pd[it][1]);
        }
    };
    if constexpr (Q8) { fetch_q8(k_lo); store_q8(0); }
    else issue_f16(k_lo, 0);

    int iter = 0;
    for (int kb = k_lo; kb <= qp_last; kb += FA_BK, iter++) {
        const int cur = iter & 1;
        const bool has_next = kb + FA_BK <= qp_last;
        __syncthreads();  // all warps are done with buffer cur^1 (computed last iteration)
        if constexpr (Q8) {
            if (has_next) fetch_q8(kb + FA_BK);  // global loads overlap this tile's math; stored after it
        } else {
            if (has_next) issue_f16(kb + FA_BK, cur ^ 1);
            else cp_commit();  // keep the group count uniform
            cp_wait<1>();
            __syncthreads();  // tile `cur` landed for everyone
        }
        const half* Ks = KsB + cur * FA_BK * 128;
        const half* Vs = VsB + cur * FA_BK * 128;
        if (t0 >= T) {
            if constexpr (Q8) { if (has_next) store_q8(cur ^ 1); }
            continue;
        }
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
        if constexpr (Q8) { if (has_next) store_q8(cur ^ 1); }  // next tile, converted while others still compute
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

}  // namespace le
