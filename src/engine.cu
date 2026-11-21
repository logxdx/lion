// liquid-engine core: weight repacking, fused decode kernels, attention, sampling, host engine.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "engine.h"
#include "gguf.h"
#include "kernels.cuh"
#include "prefill.cuh"

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

namespace le {

// control block (device ints)
enum { CTRL_TOK = 0, CTRL_POS = 1, CTRL_DONE = 2, CTRL_NGEN = 3, CTRL_STAMP = 4, CTRL_N = 8 };
// debug reference mode: bit0 conv blocks, bit1 attention blocks, bit2 ffn, bit3 lm head (f32 activations)
static int g_exact_mask = 0;

size_t QWeight::bytes() const {
    if (type == W_Q8) return (size_t)N * K + (size_t)N * (K / 32) * 2;
    return (size_t)N * K / 2 + (size_t)N * K / 4 + (size_t)N * K / 16 + (size_t)N * (K / 256) * 2;
}

// ================================================================================================
// repacking (GGUF block formats -> engine layouts), done on the GPU at load time
// ================================================================================================
struct RowMap { int tile_in, tile_out, off; };
__device__ __forceinline__ int map_row(RowMap m, int r) { return (r / m.tile_in) * m.tile_out + m.off + r % m.tile_in; }

// Q8_0 block: half d; int8 qs[32]  (34 bytes)
__global__ void repack_q8(const uint8_t* __restrict__ src, int nrows, int K, QWeight dst, RowMap m) {
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
__global__ void repack_q6(const uint8_t* __restrict__ src, int nrows, int K, QWeight dst, RowMap m) {
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

// dequantize one element of a repacked row
__device__ __forceinline__ float deq(const QWeight& w, int row, int e) {
    const int K = w.K;
    if (w.type == W_Q8) {
        int8_t q = (int8_t)w.q[(size_t)row * K + perm_idx_rt(e, K)];
        return __half2float(w.d[(size_t)row * (K / 32) + e / 32]) * q;
    }
    const int g = e >> 5, j = e & 31, hf = j >> 4, jj = j & 15;
    uint8_t lo = w.q[(size_t)row * (K / 2) + g * 16 + jj];
    int nib = hf ? (lo >> 4) : (lo & 15);
    uint8_t hb = w.h[(size_t)row * (K / 4) + g * 8 + hf * 4 + (jj & 3)];
    int q = nib | (((hb >> (2 * (jj >> 2))) & 3) << 4);
    float d = __half2float(w.d[(size_t)row * (K / 256) + e / 256]);
    return d * (float)w.sc[(size_t)row * (K / 16) + g * 2 + hf] * (float)(q - 32);
}

// ================================================================================================
// decode step kernels
// ================================================================================================
__global__ void k_embed(QWeight w, int* ctrl, int* hist, float* out) {
    if (ctrl[CTRL_DONE]) return;
    const int tok = ctrl[CTRL_TOK];
    if (threadIdx.x == 0) hist[ctrl[CTRL_POS]] = tok;
    for (int e = threadIdx.x; e < w.K; e += blockDim.x) out[e] = deq(w, tok, e);
}

__global__ void k_advance(int* ctrl) {
    if (ctrl[CTRL_DONE]) return;
    ctrl[CTRL_POS] += 1;
}

enum { PRO_NORM = 0, PRO_Q = 1 };
enum { EPI_STORE = 0, EPI_SWIGLU = 1, EPI_CONV = 2 };

struct GemvArgs {
    QWeight w;
    int ntiles;
    const int* ctrl;
    // PRO_NORM: x = resid_in + (o_in ? rms(o_in)*w_post : 0); h = rms(x)*w_pre; quantize(h)
    const float* resid_in; const float* o_in; const float* w_post; const float* w_pre; float* resid_out; float eps;
    // PRO_Q: pre-quantized activations (plain element order)
    const int8_t* xq_in; const float* xd_in;
    // epilogues
    float* out;
    int8_t* yq; float* yd;
    const float* conv_w; float* conv_state;
    // exact (debug) mode: f32 activations end to end
    const float* xf_in; float* yf;
};

template <int K, int NT, bool EX>
__device__ __forceinline__ void prologue_norm(const GemvArgs& a, int8_t* xq, float* xd, float* red, float* xf) {
    static_assert(K == NT * 8, "norm prologue: 8 elements per thread");
    const int t = threadIdx.x, e0 = t * 8;
    float v[8];
    {
        float4 a0 = *(const float4*)(a.resid_in + e0), a1 = *(const float4*)(a.resid_in + e0 + 4);
        v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w; v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
    }
    if (a.o_in) {
        float o[8];
        float4 b0 = *(const float4*)(a.o_in + e0), b1 = *(const float4*)(a.o_in + e0 + 4);
        o[0] = b0.x; o[1] = b0.y; o[2] = b0.z; o[3] = b0.w; o[4] = b1.x; o[5] = b1.y; o[6] = b1.z; o[7] = b1.w;
        float ss = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) ss += o[i] * o[i];
        ss = block_sum<NT>(ss, red);
        const float r = rsqrtf(ss / K + a.eps);
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] += o[i] * r * a.w_post[e0 + i];
        if (a.resid_out && blockIdx.x == 0) {
            *(float4*)(a.resid_out + e0) = make_float4(v[0], v[1], v[2], v[3]);
            *(float4*)(a.resid_out + e0 + 4) = make_float4(v[4], v[5], v[6], v[7]);
        }
    }
    float ss = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) ss += v[i] * v[i];
    ss = block_sum<NT>(ss, red);
    const float r = rsqrtf(ss / K + a.eps);
    float amax = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) { v[i] = v[i] * r * a.w_pre[e0 + i]; amax = fmaxf(amax, fabsf(v[i])); }
    if constexpr (EX) {
#pragma unroll
        for (int i = 0; i < 8; i++) xf[e0 + i] = v[i];
        return;
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) lo |= (uint32_t)(uint8_t)(int8_t)roundf(v[i] * id) << (8 * i);
#pragma unroll
    for (int i = 0; i < 4; i++) hi |= (uint32_t)(uint8_t)(int8_t)roundf(v[4 + i] * id) << (8 * i);
    *(uint2*)(xq + perm_idx<K>(e0)) = make_uint2(lo, hi);
    if ((t & 3) == 0) xd[e0 >> 5] = d;
}

template <int K, int NT, bool EX>
__device__ __forceinline__ void prologue_q(const GemvArgs& a, int8_t* xq, float* xd, float* xf) {
    if constexpr (EX) {
        for (int i = threadIdx.x; i < K; i += NT) xf[i] = a.xf_in[i];
        return;
    }
    for (int i = threadIdx.x; i < K / 16; i += NT) {
        int4 v = *(const int4*)(a.xq_in + i * 16);
        *(int4*)(xq + perm_idx<K>(i * 16)) = v;
    }
    for (int i = threadIdx.x; i < K / 32; i += NT) xd[i] = a.xd_in[i];
}

// ---- per-warp dot products of RB consecutive rows with the shared-memory activation ----
template <int K, int RB>
__device__ __forceinline__ void dot_q8(const QWeight& w, int row, const int8_t* xq, const float* xd, int lane, float* acc) {
    constexpr int NG = K / 32, NC = (NG + 31) / 32;
#pragma unroll
    for (int i = 0; i < RB; i++) acc[i] = 0.f;
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int Gc = (NG - c * 32) < 32 ? (NG - c * 32) : 32;
        if (Gc == 32 || lane < Gc) {
            int4 wa[RB], wb[RB];
            float wd[RB];
#pragma unroll
            for (int i = 0; i < RB; i++) {
                const uint8_t* base = w.q + (size_t)(row + i) * K + c * 1024;
                wa[i] = ld_stream(base + lane * 16);
                wb[i] = ld_stream(base + Gc * 16 + lane * 16);
                wd[i] = __half2float(__ldg(w.d + (size_t)(row + i) * NG + c * 32 + lane));
            }
            const int4 xa = *(const int4*)(xq + c * 1024 + lane * 16);
            const int4 xb = *(const int4*)(xq + c * 1024 + Gc * 16 + lane * 16);
            const float xs = xd[c * 32 + lane];
#pragma unroll
            for (int i = 0; i < RB; i++) {
                int s = __dp4a(wa[i].x, xa.x, 0);
                s = __dp4a(wa[i].y, xa.y, s);
                s = __dp4a(wa[i].z, xa.z, s);
                s = __dp4a(wa[i].w, xa.w, s);
                s = __dp4a(wb[i].x, xb.x, s);
                s = __dp4a(wb[i].y, xb.y, s);
                s = __dp4a(wb[i].z, xb.z, s);
                s = __dp4a(wb[i].w, xb.w, s);
                acc[i] += wd[i] * xs * (float)s;
            }
        }
    }
}

__device__ __forceinline__ void q6_group(int4 lo, int2 hi, int4 xa, int4 xb, int& s0, int& s1) {
    const int l[4] = {lo.x, lo.y, lo.z, lo.w};
    const int a[4] = {xa.x, xa.y, xa.z, xa.w};
    const int b[4] = {xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int m = 0; m < 4; m++) {
        const int q0 = (l[m] & 0x0F0F0F0F) | (((hi.x >> (2 * m)) & 0x03030303) << 4);
        const int q1 = ((l[m] >> 4) & 0x0F0F0F0F) | (((hi.y >> (2 * m)) & 0x03030303) << 4);
        s0 = __dp4a((int)__vsub4(q0, 0x20202020), a[m], s0);
        s1 = __dp4a((int)__vsub4(q1, 0x20202020), b[m], s1);
    }
}

template <int K, int RB>
__device__ __forceinline__ void dot_q6(const QWeight& w, int row, const int8_t* xq, const float* xd, int lane, float* acc) {
    constexpr int NG = K / 32, NC = (NG + 31) / 32;
#pragma unroll
    for (int i = 0; i < RB; i++) acc[i] = 0.f;
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int Gc = (NG - c * 32) < 32 ? (NG - c * 32) : 32;
        if (Gc == 32 || lane < Gc) {
            const int g = c * 32 + lane;
            int4 lo[RB]; int2 hi[RB]; uint16_t sc[RB]; float dd[RB];
#pragma unroll
            for (int i = 0; i < RB; i++) {
                const size_t r = (size_t)(row + i);
                lo[i] = ld_stream(w.q + r * (K / 2) + g * 16);
                hi[i] = ld_stream2(w.h + r * (K / 4) + g * 8);
                sc[i] = ld_stream_u16(w.sc + r * (K / 16) + g * 2);
                dd[i] = __half2float(__ldg(w.d + r * (K / 256) + (g >> 3)));
            }
            const int4 xa = *(const int4*)(xq + c * 1024 + lane * 16);
            const int4 xb = *(const int4*)(xq + c * 1024 + Gc * 16 + lane * 16);
            const float xs = xd[c * 32 + lane];
#pragma unroll
            for (int i = 0; i < RB; i++) {
                int s0 = 0, s1 = 0;
                q6_group(lo[i], hi[i], xa, xb, s0, s1);
                const float f = (float)(int8_t)(sc[i] & 0xff) * s0 + (float)(int8_t)(sc[i] >> 8) * s1;
                acc[i] += dd[i] * xs * f;
            }
        }
    }
}

template <int K>
__device__ __forceinline__ float dot_exact(const QWeight& w, int row, const float* xf, int lane) {
    float acc = 0;
    for (int e = lane; e < K; e += 32) acc += deq(w, row, e) * xf[e];
    return acc;
}

template <int WT, int K, int PRO, int EPI, int TILE, int RB, bool EX = false>
__global__ void __launch_bounds__(256) k_gemv(GemvArgs a) {
    constexpr int NT = 256, NW = 8, RPW = TILE / NW;
    static_assert(TILE % NW == 0 && RPW % RB == 0, "tile config");
    if (a.ctrl && a.ctrl[CTRL_DONE]) return;
    __shared__ __align__(16) int8_t xq[K];
    __shared__ float xd[K / 32];
    __shared__ float red[32];
    __shared__ float res[EPI == EPI_STORE ? 1 : TILE];
    __shared__ float xf[EX ? K : 1];

    if constexpr (PRO == PRO_NORM) prologue_norm<K, NT, EX>(a, xq, xd, red, xf);
    else prologue_q<K, NT, EX>(a, xq, xd, xf);
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int tile = blockIdx.x; tile < a.ntiles; tile += gridDim.x) {
        const int row0 = tile * TILE + warp * RPW;
#pragma unroll
        for (int r = 0; r < RPW; r += RB) {
            float acc[RB];
            if constexpr (EX) {
#pragma unroll
                for (int i = 0; i < RB; i++) acc[i] = dot_exact<K>(a.w, row0 + r + i, xf, lane);
            } else if (WT == W_Q8) dot_q8<K, RB>(a.w, row0 + r, xq, xd, lane, acc);
            else dot_q6<K, RB>(a.w, row0 + r, xq, xd, lane, acc);
#pragma unroll
            for (int i = 0; i < RB; i++) {
                const float v = warp_sum(acc[i]);
                if (lane == 0) {
                    if (EPI == EPI_STORE) a.out[row0 + r + i] = v;
                    else res[warp * RPW + r + i] = v;
                }
            }
        }
        if (EPI != EPI_STORE) {
            __syncthreads();
            if (threadIdx.x < 32) {
                const int t = threadIdx.x;
                float y;
                if (EPI == EPI_SWIGLU) {
                    const float g = res[t], u = res[32 + t];
                    y = g / (1.f + __expf(-g)) * u;
                } else {  // EPI_CONV: rows [b32 c32 x32] of 32 channels
                    const int ch = tile * 32 + t;
                    const float bx = res[t] * res[64 + t];
                    float* st = a.conv_state + (size_t)ch * 6;
                    const float* cw = a.conv_w + (size_t)ch * 7;
                    float s[6];
#pragma unroll
                    for (int j = 0; j < 6; j++) s[j] = st[j];
                    float o = cw[6] * bx;
#pragma unroll
                    for (int j = 0; j < 6; j++) o += cw[j] * s[j];
#pragma unroll
                    for (int j = 0; j < 5; j++) st[j] = s[j + 1];
                    st[5] = bx;
                    y = res[32 + t] * o;
                }
                if constexpr (EX) {
                    a.yf[tile * 32 + t] = y;
                } else {
                    const float amax = warp_max(fabsf(y));
                    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
                    a.yq[tile * 32 + t] = (int8_t)roundf(y * id);
                    if (t == 0) a.yd[tile] = d;
                }
            }
            __syncthreads();
        }
    }
}

// ------------------------------------------------------------------------------------------------
// attention (decode): grid (n_kv=4, ATT_SMAX), 128 threads = 4 warps = the 4 query heads of a kv head
// ------------------------------------------------------------------------------------------------
constexpr int ATT_SMAX = 20, ATT_MIN_CHUNK = 128, PART_STRIDE = 132;

struct AttnArgs {
    const float* qkvg;  // q[2048] k[512] v[512] gate[16]
    const float *qn, *kn, *ff;
    float base, eps;
    half *kc, *vc;
    int cap, window;
    const int* ctrl;
    float* part;
    int* counter;
    int8_t* oq;
    float* od;
    float* of;  // exact mode: f32 output
};

__device__ __forceinline__ void rope_norm4(float* x, const float* nw, const float* ff, float base, float eps, int pos, int lane) {
    // x: dims lane*4..lane*4+3 of one 128-dim head. RMS norm * weight, then NEOX rope (pairs i, i+64)
    float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / 128.f + eps);
#pragma unroll
    for (int j = 0; j < 4; j++) x[j] *= r * nw[lane * 4 + j];
    const float theta_scale = powf(base, -2.0f / 128.f);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float p = __shfl_xor_sync(0xffffffff, x[j], 16);
        const int d = lane * 4 + j, i = d & 63;
        float theta = (float)pos * powf(theta_scale, (float)i);
        if (ff) theta /= ff[i];
        float sn, cs;
        sincosf(theta, &sn, &cs);
        x[j] = (lane < 16) ? (x[j] * cs - p * sn) : (p * sn + x[j] * cs);
    }
}

__device__ __forceinline__ void attn_finalize(float* acc, float l, float gate_logit, int qh, int lane, int8_t* oq, float* od, float* of) {
    const float sig = 1.f / (1.f + __expf(-gate_logit));
    const float f = sig / l;
    float o[4], amax = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) { o[j] = acc[j] * f; amax = fmaxf(amax, fabsf(o[j])); }
    if (of) {
        *(float4*)(of + qh * 128 + lane * 4) = make_float4(o[0], o[1], o[2], o[3]);
        return;
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
    uint32_t pk = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) pk |= (uint32_t)(uint8_t)(int8_t)roundf(o[j] * id) << (8 * j);
    *(uint32_t*)(oq + qh * 128 + lane * 4) = pk;
    if ((lane & 7) == 0) od[(qh * 128 + lane * 4) >> 5] = d;
}

__global__ void __launch_bounds__(128) k_attn_decode(AttnArgs a) {
    if (a.ctrl[CTRL_DONE]) return;
    const int h = blockIdx.x, s = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = a.ctrl[CTRL_POS];
    const int lo = a.window ? max(0, t - a.window + 1) : 0;
    const int L = t - lo + 1;
    const int S = min(ATT_SMAX, (L + ATT_MIN_CHUNK - 1) / ATT_MIN_CHUNK);
    if (s >= S) return;
    const int chunk = (L + S - 1) / S;
    const int p0 = lo + s * chunk, p1 = min(t + 1, p0 + chunk);

    __shared__ __align__(16) float qs[4][128];
    __shared__ float sc[4][128];
    __shared__ int is_last;

    const int qh = h * 4 + warp;
    {
        float x[4];
        float4 v = *(const float4*)(a.qkvg + qh * 128 + lane * 4);
        x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
        rope_norm4(x, a.qn, a.ff, a.base, a.eps, t, lane);
        const float scale = rsqrtf(128.f);
        *(float4*)&qs[warp][lane * 4] = make_float4(x[0] * scale, x[1] * scale, x[2] * scale, x[3] * scale);
    }
    if (s == S - 1 && warp < 2) {  // this split owns position t: write the new k (warp 0) / v (warp 1)
        const size_t slot = ((size_t)h * a.cap + (t % a.cap)) * 128 + lane * 4;
        float x[4];
        float4 v = *(const float4*)(a.qkvg + (warp == 0 ? 2048 : 2560) + h * 128 + lane * 4);
        x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
        if (warp == 0) rope_norm4(x, a.kn, a.ff, a.base, a.eps, t, lane);
        half* dst = (warp == 0 ? a.kc : a.vc) + slot;
        *(half2*)dst = __floats2half2_rn(x[0], x[1]);
        *(half2*)(dst + 2) = __floats2half2_rn(x[2], x[3]);
    }
    __syncthreads();

    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    const half* kbase = a.kc + (size_t)h * a.cap * 128;
    const half* vbase = a.vc + (size_t)h * a.cap * 128;
    for (int pb = p0; pb < p1; pb += 128) {
        const int p = pb + threadIdx.x;
        float s4[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        if (p < p1) {
            const int4* kr = (const int4*)(kbase + (size_t)(p % a.cap) * 128);
            float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll 4
            for (int i = 0; i < 16; i++) {
                int4 kv = kr[i];
                const half2* hp = (const half2*)&kv;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 kf = __half22float2(hp[j]);
                    const int d = i * 8 + j * 2;
                    d0 += kf.x * qs[0][d] + kf.y * qs[0][d + 1];
                    d1 += kf.x * qs[1][d] + kf.y * qs[1][d + 1];
                    d2 += kf.x * qs[2][d] + kf.y * qs[2][d + 1];
                    d3 += kf.x * qs[3][d] + kf.y * qs[3][d + 1];
                }
            }
            s4[0] = d0; s4[1] = d1; s4[2] = d2; s4[3] = d3;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) sc[j][threadIdx.x] = s4[j];
        __syncthreads();
        // online softmax for head `warp`
        const int n = min(128, p1 - pb);
        float mx = -INFINITY;
        for (int j = lane; j < n; j += 32) mx = fmaxf(mx, sc[warp][j]);
        mx = warp_max(mx);
        const float mn = fmaxf(m, mx);
        const float corr = __expf(m - mn);
        l *= corr;
#pragma unroll
        for (int j = 0; j < 4; j++) acc[j] *= corr;
        float ls = 0;
        for (int j = lane; j < n; j += 32) {
            const float pj = __expf(sc[warp][j] - mn);
            sc[warp][j] = pj;
            ls += pj;
        }
        l += warp_sum(ls);
        m = mn;
        __syncwarp();
        for (int j = 0; j < n; j++) {
            const float pj = sc[warp][j];
            const uint2 vv = *(const uint2*)(vbase + (size_t)((pb + j) % a.cap) * 128 + lane * 4);
            const float2 v01 = __half22float2(*(const half2*)&vv.x), v23 = __half22float2(*(const half2*)&vv.y);
            acc[0] += pj * v01.x; acc[1] += pj * v01.y; acc[2] += pj * v23.x; acc[3] += pj * v23.y;
        }
        __syncthreads();
    }

    const float gate_logit = a.qkvg[3072 + qh];
    if (S == 1) { attn_finalize(acc, l, gate_logit, qh, lane, a.oq, a.od, a.of); return; }

    float* pp = a.part + ((size_t)qh * ATT_SMAX + s) * PART_STRIDE;
    if (lane == 0) { pp[0] = m; pp[1] = l; }
    *(float4*)(pp + 4 + lane * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) is_last = (atomicAdd(&a.counter[h], 1) == S - 1);
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    float M = -INFINITY;
    for (int j = 0; j < S; j++) M = fmaxf(M, __ldcg(a.part + ((size_t)qh * ATT_SMAX + j) * PART_STRIDE));
    float o[4] = {0, 0, 0, 0}, Lt = 0;
    for (int j = 0; j < S; j++) {
        const float* q = a.part + ((size_t)qh * ATT_SMAX + j) * PART_STRIDE;
        const float w = __expf(__ldcg(q) - M);
        Lt += w * __ldcg(q + 1);
        const float4 v = __ldcg((const float4*)(q + 4 + lane * 4));
        o[0] += w * v.x; o[1] += w * v.y; o[2] += w * v.z; o[3] += w * v.w;
    }
    attn_finalize(o, Lt, gate_logit, qh, lane, a.oq, a.od, a.of);
    if (threadIdx.x == 0) a.counter[h] = 0;
}

// ------------------------------------------------------------------------------------------------
// sampling (all on device)
// ------------------------------------------------------------------------------------------------
struct SampDev {
    float temp, top_p, min_p, penalty;
    int top_k, last_n, n_stop, pad;
    unsigned long long seed;
    int stop[8];
};
constexpr int ARGMAX_BLOCKS = 128, CAND_MAX = 2048;

__global__ void k_penalty(float* logits, const int* hist, const int* ctrl, const SampDev* sp, int* stamp) {
    if (ctrl[CTRL_DONE] || sp->penalty == 1.0f) return;
    const int pos = ctrl[CTRL_POS];  // position of the token just consumed
    const int n = sp->last_n < 0 ? pos + 1 : min(sp->last_n, pos + 1);
    const int cur = ctrl[CTRL_STAMP] + 1;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int tok = hist[pos - i];
        if (atomicExch(&stamp[tok], cur) != cur) {
            const float v = logits[tok];
            logits[tok] = v > 0 ? v / sp->penalty : v * sp->penalty;
        }
    }
}

__global__ void __launch_bounds__(256) k_argmax1(const float* logits, int n, const int* ctrl, float* pm, int* pi) {
    if (ctrl[CTRL_DONE]) return;
    __shared__ float sv[8];
    __shared__ int si[8];
    float bv = -INFINITY; int bi = 0x7fffffff;
    const int per = (n + gridDim.x - 1) / gridDim.x;
    const int b0 = blockIdx.x * per, b1 = min(n, b0 + per);
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) {
        const float v = logits[i];
        if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        bv = sv[0]; bi = si[0];
        for (int w = 1; w < 8; w++) if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        pm[blockIdx.x] = bv; pi[blockIdx.x] = bi;
    }
}

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__global__ void __launch_bounds__(1024) k_select(const float* logits, int n, int* ctrl, const SampDev* sp,
                                                 const float* pm, const int* pi, int* out_tokens) {
    if (ctrl[CTRL_DONE]) return;
    __shared__ float cv[CAND_MAX];
    __shared__ int ci[CAND_MAX];
    __shared__ int ncand;
    __shared__ float M;
    __shared__ int best;
    const int t = threadIdx.x;
    if (t < 32) {
        float bv = -INFINITY; int bi = 0x7fffffff;
        for (int j = t; j < ARGMAX_BLOCKS; j += 32)
            if (pm[j] > bv || (pm[j] == bv && pi[j] < bi)) { bv = pm[j]; bi = pi[j]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, o);
            const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (t == 0) { M = bv; best = bi; ncand = 0; }
    }
    __syncthreads();
    int tok = best;
    const float temp = sp->temp;
    if (temp > 0.f) {
        const float thr = M - 20.f * temp;
        for (int i = t; i < n; i += blockDim.x) {
            const float v = logits[i];
            if (v >= thr) {
                const int k = atomicAdd(&ncand, 1);
                if (k < CAND_MAX) { cv[k] = v; ci[k] = i; }
            }
        }
        __syncthreads();
        const int nc = min(ncand, CAND_MAX);
        int np2 = 1;
        while (np2 < nc) np2 <<= 1;
        for (int i = nc + t; i < np2; i += blockDim.x) { cv[i] = -INFINITY; ci[i] = 0x7fffffff; }
        __syncthreads();
        // bitonic sort, descending by value (ties: ascending index)
        for (int k = 2; k <= np2; k <<= 1)
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = t; i < np2; i += blockDim.x) {
                    const int ixj = i ^ j;
                    if (ixj > i) {
                        const bool desc = (i & k) == 0;
                        const float a = cv[i], b = cv[ixj];
                        const bool a_before_b = a > b || (a == b && ci[i] < ci[ixj]);
                        if (desc ? !a_before_b : a_before_b) {
                            cv[i] = b; cv[ixj] = a;
                            const int x = ci[i]; ci[i] = ci[ixj]; ci[ixj] = x;
                        }
                    }
                }
                __syncthreads();
            }
        if (t == 0) {
            // llama.cpp default chain order: top_k -> top_p -> min_p -> temperature -> dist
            int k = nc;
            if (sp->top_k > 0) k = min(k, sp->top_k);
            const float v0 = cv[0];
            if (sp->top_p < 1.f) {
                float sum = 0;
                for (int i = 0; i < k; i++) sum += __expf(cv[i] - v0);
                float cum = 0;
                for (int i = 0; i < k; i++) {
                    cum += __expf(cv[i] - v0) / sum;
                    if (cum >= sp->top_p) { k = i + 1; break; }
                }
            }
            if (sp->min_p > 0.f) {
                const float lim = v0 + logf(sp->min_p);
                int kk = 1;
                while (kk < k && cv[kk] >= lim) kk++;
                k = kk;
            }
            float sum = 0;
            for (int i = 0; i < k; i++) sum += __expf((cv[i] - v0) / temp);
            const unsigned long long r = splitmix64(sp->seed ^ ((unsigned long long)ctrl[CTRL_NGEN] * 0x9E3779B97F4A7C15ull + ctrl[CTRL_POS]));
            const float u = (float)((r >> 40) * (1.0 / 16777216.0)) * sum;
            float cum = 0;
            tok = ci[k - 1];
            for (int i = 0; i < k; i++) {
                cum += __expf((cv[i] - v0) / temp);
                if (u < cum) { tok = ci[i]; break; }
            }
        }
    }
    if (t == 0) {
        ctrl[CTRL_TOK] = tok;
        out_tokens[ctrl[CTRL_NGEN]] = tok;
        ctrl[CTRL_NGEN] += 1;
        ctrl[CTRL_STAMP] += 1;
        for (int i = 0; i < sp->n_stop; i++)
            if (sp->stop[i] == tok) ctrl[CTRL_DONE] = 1;
    }
}

// ================================================================================================
// host side
// ================================================================================================
struct Engine::Bufs {
    float *resA, *resB, *o, *f, *qkvg, *logits, *part, *pm;
    int8_t *yq, *aq, *oq;
    float *yd, *ad, *od;
    float *yf, *af, *of;  // exact mode
    int *ctrl, *hist, *counter, *stamp, *pi, *out_tokens;
    SampDev* samp;
    int* h_ctrl;  // pinned
    PrefillBufs pf;
};



void* Engine::alloc(size_t bytes) {
    void* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(bytes, 16)));
    arena_size_ += bytes;
    return p;
}

float* Engine::upload_f32(const gguf::File& f, const std::string& name) {
    const auto& t = f.t(name);
    if (t.dtype != gguf::F32) throw std::runtime_error(name + ": expected F32");
    float* p = (float*)alloc(t.nbytes);
    CK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
    return p;
}

void Engine::upload_qweight(const gguf::File& f, QWeight& w, const std::vector<std::string>& parts, int tile_in, int tile_out) {
    // parts are interleaved tile-wise when tile_in > 0, else concatenated
    int N = 0, K = -1; uint32_t dt = 0;
    for (auto& p : parts) {
        const auto& t = f.t(p);
        if (K < 0) { K = (int)t.ne[0]; dt = t.dtype; }
        if ((int)t.ne[0] != K || t.dtype != dt) throw std::runtime_error("fused weight parts differ: " + p);
        N += (int)t.ne[1];
    }
    if (dt != gguf::Q8_0 && dt != gguf::Q6_K) throw std::runtime_error(parts[0] + ": only Q8_0/Q6_K supported");
    w.type = dt == gguf::Q8_0 ? W_Q8 : W_Q6;
    w.N = N; w.K = K;
    if (w.type == W_Q8) {
        w.q = (uint8_t*)alloc((size_t)N * K);
        w.d = (half*)alloc((size_t)N * (K / 32) * 2);
    } else {
        w.q = (uint8_t*)alloc((size_t)N * K / 2);
        w.h = (uint8_t*)alloc((size_t)N * K / 4);
        w.sc = (int8_t*)alloc((size_t)N * K / 16);
        w.d = (half*)alloc((size_t)N * (K / 256) * 2);
    }
    int off = 0;
    for (size_t pi = 0; pi < parts.size(); pi++) {
        const auto& t = f.t(parts[pi]);
        const int nr = (int)t.ne[1];
        void* staging = nullptr;
        CK(cudaMalloc(&staging, t.nbytes));
        CK(cudaMemcpy(staging, t.data, t.nbytes, cudaMemcpyHostToDevice));
        RowMap m;
        if (tile_in > 0) m = {tile_in, tile_out, (int)pi * tile_in};
        else { m = {nr, nr, off}; }
        if (w.type == W_Q8) {
            long n = (long)nr * (K / 32);
            repack_q8<<<(n + 255) / 256, 256>>>((const uint8_t*)staging, nr, K, w, m);
        } else {
            long n = (long)nr * (K / 256);
            repack_q6<<<(n + 127) / 128, 128>>>((const uint8_t*)staging, nr, K, w, m);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaFree(staging));
        off += nr;
    }
}

Engine::Engine(const std::string& path, int max_ctx, bool verbose, bool exact) : max_ctx_(max_ctx), verbose_(verbose) {
    g_exact_mask = exact ? 15 : 0;
    if (const char* m = getenv("LIQUID_EXACT_MASK")) g_exact_mask = atoi(m);
    auto t0 = std::chrono::steady_clock::now();
    gguf::File f(path);
    if (f.get("general.architecture").str != "lfm3") throw std::runtime_error("not an lfm3 model");
    cfg_.n_layer = (int)f.u("lfm3.block_count");
    cfg_.n_embd = (int)f.u("lfm3.embedding_length");
    cfg_.n_ff = (int)f.u("lfm3.feed_forward_length");
    cfg_.n_head = (int)f.u("lfm3.attention.head_count");
    cfg_.head_dim = (int)f.u("lfm3.attention.key_length");
    cfg_.eps = (float)f.u("lfm3.attention.layer_norm_rms_epsilon");
    cfg_.rope_base = (float)f.u("lfm3.rope.freq_base");
    cfg_.rope_base_swa = (float)f.u("lfm3.rope.freq_base_swa");
    cfg_.swa_window = (int)f.u("lfm3.attention.sliding_window");
    cfg_.conv_l = (int)f.u("lfm3.shortconv.l_cache");
    for (double v : f.get("lfm3.attention.head_count_kv").arr_num) cfg_.head_kv.push_back((int)v);
    for (double v : f.get("lfm3.attention.sliding_window_pattern").arr_num) cfg_.swa.push_back(v != 0);
    cfg_.n_vocab = (int)f.t("token_embd.weight").ne[1];
    if (cfg_.n_embd != 2048 || cfg_.n_ff != 7680 || cfg_.n_head != 16 || cfg_.head_dim != 128 || cfg_.conv_l != 7)
        throw std::runtime_error("unexpected LFM3 dimensions (engine is specialised for LFM3-1.7B)");
    for (int il = 0; il < cfg_.n_layer; il++)
        if (cfg_.head_kv[il] != 0 && cfg_.head_kv[il] != 4) throw std::runtime_error("unexpected n_head_kv");
    if (f.tp("output.weight")) throw std::runtime_error("untied output.weight not supported");
    if (max_ctx_ % 128) max_ctx_ = (max_ctx_ + 127) / 128 * 128;

    CK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    load(f);

    // buffers
    B_ = std::make_unique<Bufs>();
    auto& b = *B_;
    const int E = cfg_.n_embd, F = cfg_.n_ff;
    b.resA = (float*)alloc(E * 4); b.resB = (float*)alloc(E * 4);
    b.o = (float*)alloc(E * 4); b.f = (float*)alloc(E * 4);
    b.qkvg = (float*)alloc(3088 * 4);
    b.logits = (float*)alloc((size_t)cfg_.n_vocab * 4);
    b.part = (float*)alloc((size_t)16 * ATT_SMAX * PART_STRIDE * 4);
    b.pm = (float*)alloc(ARGMAX_BLOCKS * 4);
    b.pi = (int*)alloc(ARGMAX_BLOCKS * 4);
    b.yq = (int8_t*)alloc(E); b.yd = (float*)alloc(E / 32 * 4);
    b.aq = (int8_t*)alloc(F); b.ad = (float*)alloc(F / 32 * 4);
    b.oq = (int8_t*)alloc(E); b.od = (float*)alloc(E / 32 * 4);
    b.yf = (float*)alloc(E * 4); b.af = (float*)alloc(F * 4); b.of = (float*)alloc(E * 4);
    b.ctrl = (int*)alloc(CTRL_N * 4);
    b.hist = (int*)alloc((size_t)max_ctx_ * 4);
    b.counter = (int*)alloc(64 * 4);
    b.stamp = (int*)alloc((size_t)cfg_.n_vocab * 4);
    b.out_tokens = (int*)alloc((size_t)(max_ctx_ + 16) * 4);
    b.samp = (SampDev*)alloc(sizeof(SampDev));
    CK(cudaMemset(b.counter, 0, 64 * 4));
    CK(cudaMemset(b.stamp, 0, (size_t)cfg_.n_vocab * 4));
    CK(cudaMemset(b.ctrl, 0, CTRL_N * 4));
    CK(cudaMallocHost(&b.h_ctrl, 64 * CTRL_N * 4));
    prefill_alloc(b.pf, cfg_, max_ctx_, [this](size_t n) { return alloc(n); });
    reset();
    build_graphs();
    auto t1 = std::chrono::steady_clock::now();
    if (verbose_)
        fprintf(stderr, "[liquid] loaded %s: %s weights, %.0f MiB on GPU, ctx %d, %.2f s\n", path.c_str(),
                emb_.type == W_Q8 ? "Q8_0" : "Q6_K", arena_size_ / 1048576.0, max_ctx_,
                std::chrono::duration<double>(t1 - t0).count());
}

Engine::~Engine() {
    if (g_decode_) cudaGraphExecDestroy(g_decode_);
    if (g_step_nohead_) cudaGraphExecDestroy(g_step_nohead_);
    // device memory is released at process exit
}

void Engine::load(const gguf::File& f) {
    upload_qweight(f, emb_, {"token_embd.weight"}, 0, 0);
    out_norm_ = upload_f32(f, "output_norm.weight");
    float* rope_ff = upload_f32(f, "rope_freqs.weight");
    L_.resize(cfg_.n_layer);
    for (int il = 0; il < cfg_.n_layer; il++) {
        Layer& l = L_[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        l.conv = cfg_.head_kv[il] == 0;
        l.swa = cfg_.swa[il];
        l.attn_norm = upload_f32(f, p + "attn_norm.weight");
        l.ffn_norm = upload_f32(f, p + "ffn_norm.weight");
        l.post_attn_norm = upload_f32(f, p + "post_attention_norm.weight");
        l.post_ffn_norm = upload_f32(f, p + "post_ffw_norm.weight");
        upload_qweight(f, l.w_gu, {p + "ffn_gate.weight", p + "ffn_up.weight"}, 32, 64);
        upload_qweight(f, l.w_down, {p + "ffn_down.weight"}, 0, 0);
        if (l.conv) {
            l.conv_w = upload_f32(f, p + "shortconv.conv.weight");
            // in_proj rows [b(2048) c(2048) x(2048)] -> tiles of [b32 c32 x32]
            const auto& t = f.t(p + "shortconv.in_proj.weight");
            QWeight& w = l.w_in;
            w.type = t.dtype == gguf::Q8_0 ? W_Q8 : W_Q6; w.K = (int)t.ne[0]; w.N = (int)t.ne[1];
            // reuse upload_qweight on a virtual 3-part split
            {
                const int K = w.K, N = w.N;
                if (w.type == W_Q8) { w.q = (uint8_t*)alloc((size_t)N * K); w.d = (half*)alloc((size_t)N * (K / 32) * 2); }
                else {
                    w.q = (uint8_t*)alloc((size_t)N * K / 2); w.h = (uint8_t*)alloc((size_t)N * K / 4);
                    w.sc = (int8_t*)alloc((size_t)N * K / 16); w.d = (half*)alloc((size_t)N * (K / 256) * 2);
                }
                const size_t rb = gguf::row_bytes(t.dtype, K);
                for (int part = 0; part < 3; part++) {
                    void* staging = nullptr;
                    CK(cudaMalloc(&staging, rb * 2048));
                    CK(cudaMemcpy(staging, t.data + rb * 2048 * part, rb * 2048, cudaMemcpyHostToDevice));
                    RowMap m = {32, 96, part * 32};
                    if (w.type == W_Q8) {
                        long n = 2048L * (K / 32);
                        repack_q8<<<(n + 255) / 256, 256>>>((const uint8_t*)staging, 2048, K, w, m);
                    } else {
                        long n = 2048L * (K / 256);
                        repack_q6<<<(n + 127) / 128, 128>>>((const uint8_t*)staging, 2048, K, w, m);
                    }
                    CK(cudaGetLastError());
                    CK(cudaDeviceSynchronize());
                    CK(cudaFree(staging));
                }
            }
            upload_qweight(f, l.w_out, {p + "shortconv.out_proj.weight"}, 0, 0);
            l.conv_state = (float*)alloc((size_t)cfg_.n_embd * 6 * 4);
        } else {
            upload_qweight(f, l.w_in, {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "attn_gate.weight"}, 0, 0);
            upload_qweight(f, l.w_out, {p + "attn_output.weight"}, 0, 0);
            l.q_norm = upload_f32(f, p + "attn_q_norm.weight");
            l.k_norm = upload_f32(f, p + "attn_k_norm.weight");
            l.rope_ff = l.swa ? nullptr : rope_ff;
            l.cap = l.swa ? cfg_.swa_window : max_ctx_;
            l.kc = (half*)alloc((size_t)4 * l.cap * 128 * 2);
            l.vc = (half*)alloc((size_t)4 * l.cap * 128 * 2);
        }
    }
}

void Engine::reset() {
    for (auto& l : L_) {
        if (l.conv_state) CK(cudaMemset(l.conv_state, 0, (size_t)cfg_.n_embd * 6 * 4));
    }
    CK(cudaMemset(B_->ctrl, 0, CTRL_N * 4));
    n_past_ = 0;
    pending_ = -1;
}

// ---- launch helpers -------------------------------------------------------------------------

template <int K, int PRO, int EPI, int TILE, int RB>
static void launch_gemv(GemvArgs a, int max_grid, cudaStream_t s, bool ex = false) {
    const int grid = std::min(a.ntiles, max_grid);
    if (ex) {
        k_gemv<W_Q8, K, PRO, EPI, TILE, RB, true><<<grid, 256, 0, s>>>(a);  // WT unused in exact mode
        return;
    }
    if (a.w.type == W_Q8) k_gemv<W_Q8, K, PRO, EPI, TILE, RB><<<grid, 256, 0, s>>>(a);
    else k_gemv<W_Q6, K, PRO, EPI, TILE, RB><<<grid, 256, 0, s>>>(a);
}

void Engine::enqueue_step(bool with_head, cudaStream_t s) {
    auto& b = *B_;
    const int E = cfg_.n_embd;
    k_embed<<<1, 256, 0, s>>>(emb_, b.ctrl, b.hist, b.resA);
    float* cur = b.resA;                          // residual stream (ping-pong A/B, static per graph)
    auto other = [&](float* x) { return x == b.resA ? b.resB : b.resA; };
    const float* prev_f = nullptr;                // ffn output of previous layer (null for layer 0)
    const float* prev_post = nullptr;
    for (int il = 0; il < cfg_.n_layer; il++) {
        const Layer& l = L_[il];
        GemvArgs a{};
        a.ctrl = b.ctrl; a.eps = cfg_.eps;
        // operator input: x = cur + postnorm(prev ffn out) -> other(cur) ; h = attn_norm(x)
        a.resid_in = cur; a.o_in = prev_f; a.w_post = prev_post; a.w_pre = l.attn_norm;
        if (prev_f) { a.resid_out = other(cur); cur = other(cur); }
        a.w = l.w_in;
        if (l.conv) {
            a.ntiles = E / 32;
            a.conv_w = l.conv_w; a.conv_state = l.conv_state; a.yq = b.yq; a.yd = b.yd; a.yf = b.yf;
            launch_gemv<2048, PRO_NORM, EPI_CONV, 96, 2>(a, 160, s, g_exact_mask & 1);
            GemvArgs o{};
            o.ctrl = b.ctrl; o.w = l.w_out; o.ntiles = E / 16; o.xq_in = b.yq; o.xd_in = b.yd; o.xf_in = b.yf; o.out = b.o;
            launch_gemv<2048, PRO_Q, EPI_STORE, 16, 2>(o, 128, s, g_exact_mask & 1);
        } else {
            a.ntiles = l.w_in.N / 16; a.out = b.qkvg;
            launch_gemv<2048, PRO_NORM, EPI_STORE, 16, 2>(a, 193, s, g_exact_mask & 2);
            AttnArgs at{};
            at.qkvg = b.qkvg; at.qn = l.q_norm; at.kn = l.k_norm; at.ff = l.rope_ff;
            at.base = l.swa ? cfg_.rope_base_swa : cfg_.rope_base; at.eps = cfg_.eps;
            at.kc = l.kc; at.vc = l.vc; at.cap = l.cap; at.window = l.swa ? cfg_.swa_window : 0;
            at.ctrl = b.ctrl; at.part = b.part; at.counter = b.counter; at.oq = b.oq; at.od = b.od;
            at.of = (g_exact_mask & 2) ? b.of : nullptr;
            k_attn_decode<<<dim3(4, ATT_SMAX), 128, 0, s>>>(at);
            GemvArgs o{};
            o.ctrl = b.ctrl; o.w = l.w_out; o.ntiles = E / 16; o.xq_in = b.oq; o.xd_in = b.od; o.xf_in = b.of; o.out = b.o;
            launch_gemv<2048, PRO_Q, EPI_STORE, 16, 2>(o, 128, s, g_exact_mask & 2);
        }
        // ffn input: x = cur + postnorm(o) -> other(cur) ; h = ffn_norm(x)
        GemvArgs g{};
        g.ctrl = b.ctrl; g.eps = cfg_.eps; g.w = l.w_gu; g.ntiles = cfg_.n_ff / 32;
        g.resid_in = cur; g.o_in = b.o; g.w_post = l.post_attn_norm; g.w_pre = l.ffn_norm;
        g.resid_out = other(cur); cur = other(cur);
        g.yq = b.aq; g.yd = b.ad; g.yf = b.af;
        launch_gemv<2048, PRO_NORM, EPI_SWIGLU, 64, 2>(g, 240, s, g_exact_mask & 4);
        GemvArgs d{};
        d.ctrl = b.ctrl; d.w = l.w_down; d.ntiles = E / 16; d.xq_in = b.aq; d.xd_in = b.ad; d.xf_in = b.af; d.out = b.f;
        launch_gemv<7680, PRO_Q, EPI_STORE, 16, 1>(d, 128, s, g_exact_mask & 4);
        prev_f = b.f; prev_post = l.post_ffn_norm;
    }
    if (with_head) {
        GemvArgs h{};
        h.ctrl = b.ctrl; h.eps = cfg_.eps; h.w = emb_; h.ntiles = cfg_.n_vocab / 64;
        h.resid_in = cur; h.o_in = b.f; h.w_post = prev_post; h.w_pre = out_norm_; h.resid_out = nullptr;
        h.out = b.logits;
        launch_gemv<2048, PRO_NORM, EPI_STORE, 64, 2>(h, 400, s, g_exact_mask & 8);
        enqueue_sample(s);
    }
    k_advance<<<1, 1, 0, s>>>(b.ctrl);
}

void Engine::enqueue_sample(cudaStream_t s) {
    auto& b = *B_;
    k_penalty<<<64, 256, 0, s>>>(b.logits, b.hist, b.ctrl, b.samp, b.stamp);
    k_argmax1<<<ARGMAX_BLOCKS, 256, 0, s>>>(b.logits, cfg_.n_vocab, b.ctrl, b.pm, b.pi);
    k_select<<<1, 1024, 0, s>>>(b.logits, cfg_.n_vocab, b.ctrl, b.samp, b.pm, b.pi, b.out_tokens);
}

void Engine::build_graphs() {
    for (int with_head = 0; with_head < 2; with_head++) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        enqueue_step(with_head != 0, st_);
        CK(cudaStreamEndCapture(st_, &g));
        CK(cudaGraphInstantiate(with_head ? &g_decode_ : &g_step_nohead_, g, 0));
        CK(cudaGraphDestroy(g));
    }
}

void Engine::set_sampling(const SamplingParams& sp, const std::vector<int>& stop) {
    SampDev d{};
    d.temp = sp.temperature; d.top_p = sp.top_p; d.min_p = sp.min_p; d.penalty = sp.repeat_penalty;
    d.top_k = sp.top_k; d.last_n = sp.repeat_last_n; d.seed = sp.seed;
    d.n_stop = std::min<int>(8, (int)stop.size());
    for (int i = 0; i < d.n_stop; i++) d.stop[i] = stop[i];
    CK(cudaMemcpyAsync(B_->samp, &d, sizeof(d), cudaMemcpyHostToDevice, st_));
}

void Engine::prefill(const std::vector<int>& tokens) {
    if (tokens.empty()) return;
    std::vector<int> all;
    if (pending_ >= 0) all.push_back(pending_);
    all.insert(all.end(), tokens.begin(), tokens.end() - 1);
    pending_ = tokens.back();
    if (n_past_ + (int)all.size() + 1 > max_ctx_) throw std::runtime_error("context overflow");
    auto t0 = std::chrono::steady_clock::now();
    if (!all.empty()) {
        if (use_batched_prefill) prefill_batched(all.data(), (int)all.size());
        else {
            int* h = B_->h_ctrl;
            for (size_t i = 0; i < all.size(); i++) {
                h[(i % 64) * CTRL_N] = all[i];
                CK(cudaMemcpyAsync(B_->ctrl + CTRL_TOK, h + (i % 64) * CTRL_N, 4, cudaMemcpyHostToDevice, st_));
                CK(cudaGraphLaunch(g_step_nohead_, st_));
                if (i % 64 == 63) CK(cudaStreamSynchronize(st_));
            }
        }
        CK(cudaStreamSynchronize(st_));
        n_past_ += (int)all.size();
    }
    auto t1 = std::chrono::steady_clock::now();
    timing.prefill_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing.prefill_tokens += (int)all.size();
}

std::vector<int> Engine::generate(int n, const SamplingParams& sp, const std::vector<int>& stop,
                                  bool (*on_token)(int, void*), void* ud) {
    std::vector<int> out;
    if (pending_ < 0 || n <= 0) return out;
    n = std::min(n, max_ctx_ - n_past_ - 1);
    auto& b = *B_;
    set_sampling(sp, stop);
    int* hc = b.h_ctrl;
    hc[0] = pending_; hc[1] = n_past_; hc[2] = 0; hc[3] = 0;
    CK(cudaMemcpyAsync(b.ctrl, hc, 4 * 4, cudaMemcpyHostToDevice, st_));
    constexpr int RING = 32, AHEAD = 3;
    cudaEvent_t ev[RING];
    for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    auto t0 = std::chrono::steady_clock::now();
    int launched = 0;
    bool stop_now = false;
    auto launch = [&]() {
        const int slot = launched % RING;
        CK(cudaGraphLaunch(g_decode_, st_));
        CK(cudaMemcpyAsync(hc + CTRL_N * (1 + slot), b.ctrl, CTRL_N * 4, cudaMemcpyDeviceToHost, st_));
        CK(cudaEventRecord(ev[slot], st_));
        launched++;
    };
    for (int i = 0; i < std::min(n, AHEAD); i++) launch();
    for (int i = 0; i < n && !stop_now; i++) {
        const int slot = i % RING;
        CK(cudaEventSynchronize(ev[slot]));
        const int* c = hc + CTRL_N * (1 + slot);
        if (c[CTRL_NGEN] <= i) break;  // done flag was set earlier
        const int tok = c[CTRL_TOK];
        out.push_back(tok);
        bool keep = on_token ? on_token(tok, ud) : true;
        if (c[CTRL_DONE] || !keep) stop_now = true;
        else if (launched < n) launch();
    }
    CK(cudaStreamSynchronize(st_));
    auto t1 = std::chrono::steady_clock::now();
    timing.decode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing.decode_tokens += (int)out.size();
    // read back final state
    CK(cudaMemcpy(hc, b.ctrl, CTRL_N * 4, cudaMemcpyDeviceToHost));
    n_past_ = hc[CTRL_POS];
    pending_ = hc[CTRL_TOK];
    for (auto& e : ev) cudaEventDestroy(e);
    return out;
}

std::vector<float> Engine::logits() {
    std::vector<float> v(cfg_.n_vocab);
    CK(cudaMemcpy(v.data(), B_->logits, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
}

void Engine::prefill_batched(const int* tokens, int n) {
    // placeholder until the batched path lands: sequential steps
    int* h = B_->h_ctrl;
    for (int i = 0; i < n; i++) {
        h[(i % 64) * CTRL_N] = tokens[i];
        CK(cudaMemcpyAsync(B_->ctrl + CTRL_TOK, h + (i % 64) * CTRL_N, 4, cudaMemcpyHostToDevice, st_));
        CK(cudaGraphLaunch(g_step_nohead_, st_));
        if (i % 64 == 63) CK(cudaStreamSynchronize(st_));
    }
}

}  // namespace le
