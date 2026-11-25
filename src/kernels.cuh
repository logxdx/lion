// v1 kernels: straightforward, batched over rows. Each row = one token of one sequence slot.
#pragma once
#include <cstdint>

#include "util.cuh"

namespace raon {

// ---------------------------------------------------------------- activation quantization (int8 per 32, like q8_1)
// one warp per 32-element block
__global__ void k_quantize(const float* __restrict__ x, int ldx, int K, int R, int8_t* __restrict__ xq,
                           float* __restrict__ xs) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int nb = K / 32;
    if (gw >= R * nb) return;
    const int r = gw / nb, b = gw % nb;
    const float v = x[(size_t)r * ldx + b * 32 + lane];
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    xq[(size_t)r * K + b * 32 + lane] = (int8_t)roundf(v * id);
    if (lane == 0) xs[(size_t)r * nb + b] = d;
}

// ---------------------------------------------------------------- GEMV (one warp per output row, B batch rows)
// unit u = 16 consecutive input elements; x block of 32 = units 2v, 2v+1
template <int B>
__global__ void k_gemv_q8(const int8_t* __restrict__ wq, const half* __restrict__ wd, int N, int K,
                          const int8_t* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ y, int ldy) {
    const int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (n >= N) return;
    const int nu = K / 16, nb = K / 32;
    const int8_t* wr = wq + (size_t)n * K;
    const half* dr = wd + (size_t)n * nb;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; b++) acc[b] = 0.f;
    for (int u = lane; u < nu; u += 32) {
        const int4 w = *(const int4*)(wr + 16 * u);
        const float dw = __half2float(dr[u >> 1]);
#pragma unroll
        for (int b = 0; b < B; b++) {
            const int4 xv = *(const int4*)(xq + (size_t)b * K + 16 * u);
            int s = __dp4a(w.x, xv.x, 0);
            s = __dp4a(w.y, xv.y, s);
            s = __dp4a(w.z, xv.z, s);
            s = __dp4a(w.w, xv.w, s);
            acc[b] += (float)s * dw * xs[(size_t)b * nb + (u >> 1)];
        }
    }
#pragma unroll
    for (int b = 0; b < B; b++) {
        const float v = warp_sum(acc[b]);
        if (lane == 0) y[(size_t)b * ldy + n] = v;
    }
}

__device__ __forceinline__ int q6_unpack(int lo, int hi, int nib_shift, int h_shift) {
    const int l = (lo >> nib_shift) & 0x0F0F0F0F;
    const int h = ((hi >> h_shift) & 0x03030303) << 4;
    return __vsubss4(l | h, 0x20202020);
}

template <int B>
__global__ void k_gemv_q6(const int8_t* __restrict__ ql, const uint8_t* __restrict__ qh, const int8_t* __restrict__ sc,
                          const half* __restrict__ wd, int N, int K, const int8_t* __restrict__ xq,
                          const float* __restrict__ xs, float* __restrict__ y, int ldy) {
    const int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (n >= N) return;
    const int nu = K / 16, nb = K / 32;
    const uint8_t* lr = (const uint8_t*)ql + (size_t)n * (K / 2);
    const uint8_t* hr = qh + (size_t)n * (K / 4);
    const int8_t* sr = sc + (size_t)n * (K / 16);
    const half* dr = wd + (size_t)n * (K / 256);
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; b++) acc[b] = 0.f;
    for (int u = lane; u < nu; u += 32) {
        const int blk = u >> 4, w = u & 15, hf = w >> 3, r = (w >> 1) & 3, lh = w & 1;
        const int4 lo = *(const int4*)(lr + blk * 128 + hf * 64 + (r & 1) * 32 + lh * 16);
        const int4 hi = *(const int4*)(hr + blk * 64 + hf * 32 + lh * 16);
        const int ns = r >= 2 ? 4 : 0, hs = 2 * r;
        const int q0 = q6_unpack(lo.x, hi.x, ns, hs), q1 = q6_unpack(lo.y, hi.y, ns, hs);
        const int q2 = q6_unpack(lo.z, hi.z, ns, hs), q3 = q6_unpack(lo.w, hi.w, ns, hs);
        const float dw = __half2float(dr[blk]) * (float)sr[blk * 16 + hf * 8 + lh + 2 * r];
#pragma unroll
        for (int b = 0; b < B; b++) {
            const int4 xv = *(const int4*)(xq + (size_t)b * K + 16 * u);
            int s = __dp4a(q0, xv.x, 0);
            s = __dp4a(q1, xv.y, s);
            s = __dp4a(q2, xv.z, s);
            s = __dp4a(q3, xv.w, s);
            acc[b] += (float)s * dw * xs[(size_t)b * nb + (u >> 1)];
        }
    }
#pragma unroll
    for (int b = 0; b < B; b++) {
        const float v = warp_sum(acc[b]);
        if (lane == 0) y[(size_t)b * ldy + n] = v;
    }
}

// F16 weights, f32 activations. Grouped: row n reads x + b*ldx + (n / rpg) * gstride (per-head inputs).
template <int B>
__global__ void k_gemv_f16(const half* __restrict__ w, int N, int K, const float* __restrict__ x, int ldx, int rpg,
                           int gstride, float* __restrict__ y, int ldy) {
    const int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (n >= N) return;
    const half* wr = w + (size_t)n * K;
    const float* xg = x + (size_t)(n / rpg) * gstride;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; b++) acc[b] = 0.f;
    for (int u = lane; u < K / 8; u += 32) {
        const uint4 raw = *(const uint4*)(wr + 8 * u);
        const half2* h2 = (const half2*)&raw;
        float wf[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 f = __half22float2(h2[i]);
            wf[2 * i] = f.x; wf[2 * i + 1] = f.y;
        }
#pragma unroll
        for (int b = 0; b < B; b++) {
            const float4 a = *(const float4*)(xg + (size_t)b * ldx + 8 * u);
            const float4 c = *(const float4*)(xg + (size_t)b * ldx + 8 * u + 4);
            acc[b] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * c.x + wf[5] * c.y + wf[6] * c.z +
                      wf[7] * c.w;
        }
    }
#pragma unroll
    for (int b = 0; b < B; b++) {
        const float v = warp_sum(acc[b]);
        if (lane == 0) y[(size_t)b * ldy + n] = v;
    }
}

// ---------------------------------------------------------------- dequantize one row of a repacked matrix
__device__ __forceinline__ float mat_elem_q8(const int8_t* q, const half* d, int K, int row, int e) {
    return __half2float(d[(size_t)row * (K / 32) + e / 32]) * (float)q[(size_t)row * K + e];
}
__device__ __forceinline__ float mat_elem_q6(const int8_t* ql, const uint8_t* qh, const int8_t* sc, const half* d,
                                             int K, int row, int e) {
    const int blk = e >> 8, w = e & 255, hf = w >> 7, r = (w >> 5) & 3, l = w & 31;
    const uint8_t lo = ((const uint8_t*)ql)[(size_t)row * (K / 2) + blk * 128 + hf * 64 + (r & 1) * 32 + l];
    const uint8_t hi = qh[(size_t)row * (K / 4) + blk * 64 + hf * 32 + l];
    const int q = (int)(((r >= 2 ? lo >> 4 : lo & 15)) | (((hi >> (2 * r)) & 3) << 4)) - 32;
    const int s = sc[(size_t)row * (K / 16) + blk * 16 + hf * 8 + (l >> 4) + 2 * r];
    return __half2float(d[(size_t)row * (K / 256) + blk]) * (float)s * (float)q;
}

// token embedding: e[r] = tok_embd[tok]; X[r][s] = e for every mHC stream
__global__ void k_embed(int type, const int8_t* q, const uint8_t* qh, const int8_t* sc, const half* d, int K,
                        const int* __restrict__ toks, float* __restrict__ e, float* __restrict__ X, int hc) {
    const int r = blockIdx.x, tok = toks[r];
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const float v = type == 0 ? mat_elem_q8(q, d, K, tok, i) : mat_elem_q6(q, qh, sc, d, K, tok, i);
        e[(size_t)r * K + i] = v;
        for (int s = 0; s < hc; s++) X[((size_t)r * hc + s) * K + i] = v;
    }
}

// per-layer embedding rows from the host-mapped GGUF table, dequantized and scaled by sqrt(n_ple)
__global__ void k_ple_rows(const uint8_t* __restrict__ table, int gtype, size_t row_bytes, int W,
                           const int* __restrict__ toks, float scale, float* __restrict__ out) {
    const int r = blockIdx.x;
    const uint8_t* row = table + (size_t)toks[r] * row_bytes;
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        float v;
        if (gtype == 8) {  // Q8_0: half d, int8 qs[32]
            const uint8_t* b = row + (i / 32) * 34;
            const half dh = *(const half*)b;
            v = __half2float(dh) * (float)(int8_t)b[2 + (i & 31)];
        } else {  // Q6_K: ql[128] qh[64] scales[16] d
            const uint8_t* b = row + (i / 256) * 210;
            const int w = i & 255, hf = w >> 7, r2 = (w >> 5) & 3, l = w & 31;
            const uint8_t lo = b[hf * 64 + (r2 & 1) * 32 + l];
            const uint8_t hi = b[128 + hf * 32 + l];
            const int q = (int)(((r2 >= 2 ? lo >> 4 : lo & 15)) | (((hi >> (2 * r2)) & 3) << 4)) - 32;
            const int s = (int8_t)b[192 + hf * 8 + (l >> 4) + 2 * r2];
            const half dh = *(const half*)(b + 208);
            v = __half2float(dh) * (float)s * (float)q;
        }
        out[(size_t)r * W + i] = v * scale;
    }
}

// ple = (RMSNorm_w(proj) per 256-group + rows) / sqrt(2); grid (R, n_layer), block 256
__global__ void k_ple_combine(const float* __restrict__ proj, const float* __restrict__ rows, const float* __restrict__ w,
                              int G, float eps, float* __restrict__ ple) {
    __shared__ float red[32];
    const int r = blockIdx.x, l = blockIdx.y, NL = gridDim.y, i = threadIdx.x;
    const size_t off = ((size_t)r * NL + l) * G + i;
    const float p = proj[off];
    const float ss = block_sum(p * p, red);
    const float inv = rsqrtf(ss / G + eps);
    ple[off] = (p * inv * w[i] + rows[off]) * 0.70710678118654752f;
}

// ---------------------------------------------------------------- RMSNorm over groups: grid (R, ngroups)
// y = x / rms(x) * w (w may be null); x row stride ldx, group size G (<= blockDim * 8)
__global__ void k_rmsnorm(const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy, const float* __restrict__ w,
                          int G, float eps) {
    __shared__ float red[32];
    const int r = blockIdx.x, g = blockIdx.y;
    const float* xr = x + (size_t)r * ldx + (size_t)g * G;
    float ss = 0.f;
    for (int i = threadIdx.x; i < G; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / G + eps);
    float* yr = y + (size_t)r * ldy + (size_t)g * G;
    for (int i = threadIdx.x; i < G; i += blockDim.x) yr[i] = xr[i] * inv * (w ? w[i] : 1.f);
}

// ---------------------------------------------------------------- mHC
// pre step 2: mixes [R][17] -> pre/post/comb, h = RMSNorm_w(sum_i pre_i X_i). block 256, one row per block.
__global__ void k_hc_pre(const float* __restrict__ mixes, const float* __restrict__ base, const float* __restrict__ chart,
                         const float* __restrict__ X, const float* __restrict__ norm_w, int E, float eps, float mag,
                         float* __restrict__ h, float* __restrict__ post, float* __restrict__ comb) {
    constexpr int HC = 4, MIX = 17, HC2 = 16;
    __shared__ float m[MIX], pre[HC], cb[HC2], red[32];
    const int r = blockIdx.x, t = threadIdx.x;
    if (t < MIX) m[t] = mixes[(size_t)r * MIX + t] + base[t];
    __syncthreads();
    if (t < 2 * HC) {
        const float s = sigmoidf_(m[t]);
        if (t < HC) pre[t] = s;
        else post[(size_t)r * HC + t - HC] = s * mag;
    }
    if (t < HC2) {  // e'_j = sum_k K[j][k] m[k]; K row j = chart[j*MIX .. j*MIX+16]
        float e = 0.f;
        for (int k = 0; k < MIX; k++) e += chart[t * MIX + k] * m[k];
        cb[t] = e;
    }
    __syncthreads();
    if (t < HC2) {
        float mx = -INFINITY;  // rho = 1 + max_j(e_j * r_j), r = chart[MIX*16 ..]
        for (int j = 0; j < HC2; j++) mx = fmaxf(mx, cb[j] * chart[MIX * HC2 + j]);
        const float rho = 1.f + mx;
        comb[(size_t)r * HC2 + t] = cb[t] / rho + chart[MIX * HC2 + HC2 + t];
    }
    // h = sum_i pre_i X_i, then RMSNorm with norm_w
    const float* Xr = X + (size_t)r * HC * E;
    float* hr = h + (size_t)r * E;
    float ss = 0.f;
    for (int i = t; i < E; i += blockDim.x) {
        const float v = pre[0] * Xr[i] + pre[1] * Xr[E + i] + pre[2] * Xr[2 * E + i] + pre[3] * Xr[3 * E + i];
        hr[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / E + eps);
    for (int i = t; i < E; i += blockDim.x) hr[i] = hr[i] * inv * norm_w[i];
}

// post: X'_d = sum_s comb[s*4+d] X_s + post_d * y   (in place)
__global__ void k_hc_post(float* __restrict__ X, const float* __restrict__ y, const float* __restrict__ post,
                          const float* __restrict__ comb, int E, int R) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * E) return;
    const int r = idx / E, i = idx % E;
    float* Xr = X + (size_t)r * 4 * E;
    const float* c = comb + r * 16;
    const float* p = post + r * 4;
    const float x0 = Xr[i], x1 = Xr[E + i], x2 = Xr[2 * E + i], x3 = Xr[3 * E + i], yv = y[(size_t)r * E + i];
#pragma unroll
    for (int d = 0; d < 4; d++) Xr[d * E + i] = c[d] * x0 + c[4 + d] * x1 + c[8 + d] * x2 + c[12 + d] * x3 + p[d] * yv;
}

// ---------------------------------------------------------------- fused mHC step, one row per block (192 threads x 8)
// In order, all optional:
//   add:  X_s += RMSNorm_aw(A)          (per-layer-input output, added to every stream; aw == null: plain add)
//   post: X_d  = sum_s comb[s*4+d] X_s + post_d * Y    (with the post/comb of the previous pre step)
//   writes X back if it changed; sum: S = sscale * sum_s X_s
//   pre:  m = fn . (X / rms(X)) + base -> pre/post/comb (as k_hc_pre); H = RMSNorm_nw(sum_i pre_i X_i)
struct MhcArgs {
    float* X; int R;
    const float *A, *aw;                  // add
    const float* Y; float *post, *comb;   // post (read) and pre outputs (written; same buffers)
    float* S; float sscale;               // stream sum
    const half* fn; const float *base, *chart, *nw; float* H;  // pre
    float hc_eps, eps, mag;
};
__global__ void __launch_bounds__(192) k_mhc(MhcArgs a) {
    constexpr int E = 1536, HC = 4, MIX = 17, HC2 = 16, T = 192, V = 8;
    __shared__ float red[8][MIX + 1];
    __shared__ float sm_m[MIX], sm_pre[HC], sm_cb[HC2], sm_bc[24];
    const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5, e0 = t * V;
    float* Xr = a.X + (size_t)r * HC * E;
    float x[HC][V];
#pragma unroll
    for (int s = 0; s < HC; s++) {
        const float4 u = *(const float4*)&Xr[s * E + e0], w = *(const float4*)&Xr[s * E + e0 + 4];
        x[s][0] = u.x; x[s][1] = u.y; x[s][2] = u.z; x[s][3] = u.w; x[s][4] = w.x; x[s][5] = w.y; x[s][6] = w.z; x[s][7] = w.w;
    }
    // block reduction of n values (n <= 18) held per thread; result broadcast via red[0][i]
    auto reduce = [&](float* v, int n) {
#pragma unroll 1
        for (int i = 0; i < n; i++) {
            float s = v[i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            v[i] = s;
        }
        __syncthreads();
        if (lane == 0)
            for (int i = 0; i < n; i++) red[warp][i] = v[i];
        __syncthreads();
        if (t < n) {
            float s = 0.f;
            for (int w = 0; w < T / 32; w++) s += red[w][t];
            sm_bc[t] = s;
        }
        __syncthreads();
        for (int i = 0; i < n; i++) v[i] = sm_bc[i];
    };
    bool changed = false;
    if (a.A) {
        float av[V], ss = 0.f;
        const float4 u = *(const float4*)&a.A[(size_t)r * E + e0], w = *(const float4*)&a.A[(size_t)r * E + e0 + 4];
        av[0] = u.x; av[1] = u.y; av[2] = u.z; av[3] = u.w; av[4] = w.x; av[5] = w.y; av[6] = w.z; av[7] = w.w;
        float inv = 1.f;
        if (a.aw) {
#pragma unroll
            for (int i = 0; i < V; i++) ss += av[i] * av[i];
            reduce(&ss, 1);
            inv = rsqrtf(ss / E + a.eps);
        }
#pragma unroll
        for (int i = 0; i < V; i++) {
            const float v = av[i] * inv * (a.aw ? a.aw[e0 + i] : 1.f);
#pragma unroll
            for (int s = 0; s < HC; s++) x[s][i] += v;
        }
        changed = true;
    }
    if (a.Y) {
        float cm[HC2], pp[HC];
#pragma unroll
        for (int i = 0; i < HC2; i++) cm[i] = a.comb[r * HC2 + i];
#pragma unroll
        for (int i = 0; i < HC; i++) pp[i] = a.post[r * HC + i];
        const float4 u = *(const float4*)&a.Y[(size_t)r * E + e0], w = *(const float4*)&a.Y[(size_t)r * E + e0 + 4];
        const float yv[V] = {u.x, u.y, u.z, u.w, w.x, w.y, w.z, w.w};
#pragma unroll
        for (int i = 0; i < V; i++) {
            const float x0 = x[0][i], x1 = x[1][i], x2 = x[2][i], x3 = x[3][i];
#pragma unroll
            for (int d = 0; d < HC; d++) x[d][i] = cm[d] * x0 + cm[4 + d] * x1 + cm[8 + d] * x2 + cm[12 + d] * x3 + pp[d] * yv[i];
        }
        changed = true;
    }
    if (changed) {
#pragma unroll
        for (int s = 0; s < HC; s++) {
            *(float4*)&Xr[s * E + e0] = make_float4(x[s][0], x[s][1], x[s][2], x[s][3]);
            *(float4*)&Xr[s * E + e0 + 4] = make_float4(x[s][4], x[s][5], x[s][6], x[s][7]);
        }
    }
    if (a.S) {
        float sv[V];
#pragma unroll
        for (int i = 0; i < V; i++) sv[i] = (x[0][i] + x[1][i] + x[2][i] + x[3][i]) * a.sscale;
        *(float4*)&a.S[(size_t)r * E + e0] = make_float4(sv[0], sv[1], sv[2], sv[3]);
        *(float4*)&a.S[(size_t)r * E + e0 + 4] = make_float4(sv[4], sv[5], sv[6], sv[7]);
    }
    if (!a.fn) return;
    // ---- pre: rms over the 4 streams, 17 selector dots, then the mix
    float v[MIX + 1];
#pragma unroll
    for (int m = 0; m < MIX; m++) v[m] = 0.f;
    float ss = 0.f;
#pragma unroll
    for (int s = 0; s < HC; s++)
#pragma unroll
        for (int i = 0; i < V; i++) ss += x[s][i] * x[s][i];
#pragma unroll
    for (int m = 0; m < MIX; m++) {
#pragma unroll
        for (int s = 0; s < HC; s++) {
            const uint4 raw = *(const uint4*)&a.fn[(size_t)m * HC * E + s * E + e0];
            const half2* h2 = (const half2*)&raw;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float2 f = __half22float2(h2[q]);
                v[m] += f.x * x[s][2 * q] + f.y * x[s][2 * q + 1];
            }
        }
    }
    v[MIX] = ss;
    reduce(v, MIX + 1);
    const float inv = rsqrtf(v[MIX] / (HC * E) + a.hc_eps);
    if (t < MIX) sm_m[t] = v[t] * inv + a.base[t];
    __syncthreads();
    if (t < 2 * HC) {
        const float sg = sigmoidf_(sm_m[t]);
        if (t < HC) sm_pre[t] = sg;
        else a.post[r * HC + t - HC] = sg * a.mag;
    }
    if (t < HC2) {
        float e = 0.f;
        for (int k = 0; k < MIX; k++) e += a.chart[t * MIX + k] * sm_m[k];
        sm_cb[t] = e;
    }
    __syncthreads();
    if (t < HC2) {
        float mx = -INFINITY;
        for (int j = 0; j < HC2; j++) mx = fmaxf(mx, sm_cb[j] * a.chart[MIX * HC2 + j]);
        a.comb[r * HC2 + t] = sm_cb[t] / (1.f + mx) + a.chart[MIX * HC2 + HC2 + t];
    }
    float hv[V], hs = 0.f;
#pragma unroll
    for (int i = 0; i < V; i++) {
        hv[i] = sm_pre[0] * x[0][i] + sm_pre[1] * x[1][i] + sm_pre[2] * x[2][i] + sm_pre[3] * x[3][i];
        hs += hv[i] * hv[i];
    }
    reduce(&hs, 1);
    const float ih = rsqrtf(hs / E + a.eps);
    float* Hr = a.H + (size_t)r * E;
    float o[V];
#pragma unroll
    for (int i = 0; i < V; i++) o[i] = hv[i] * ih * a.nw[e0 + i];
    *(float4*)&Hr[e0] = make_float4(o[0], o[1], o[2], o[3]);
    *(float4*)&Hr[e0 + 4] = make_float4(o[4], o[5], o[6], o[7]);
}

// sum of streams (per-layer-input gate input) or mean (final)
__global__ void k_stream_sum(const float* __restrict__ X, float* __restrict__ s, int E, int R, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * E) return;
    const int r = idx / E, i = idx % E;
    const float* Xr = X + (size_t)r * 4 * E;
    s[idx] = (Xr[i] + Xr[E + i] + Xr[2 * E + i] + Xr[3 * E + i]) * scale;
}
__global__ void k_add_streams(float* __restrict__ X, const float* __restrict__ y, int E, int R) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * E) return;
    const int r = idx / E, i = idx % E;
    const float v = y[idx];
    float* Xr = X + (size_t)r * 4 * E;
    Xr[i] += v; Xr[E + i] += v; Xr[2 * E + i] += v; Xr[3 * E + i] += v;
}

// ---------------------------------------------------------------- elementwise
// SwiGLU fused with int8 quantization (per 32) for the next GEMM: gu row = [gate | up], one warp per 32-block
__global__ void k_swiglu_quantize(const float* __restrict__ gu, int F, int R, int8_t* __restrict__ xq,
                                  float* __restrict__ xs) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int nb = F / 32;
    if (gw >= R * nb) return;
    const int r = gw / nb, b = gw % nb, i = b * 32 + lane;
    const float v = siluf_(gu[(size_t)r * 2 * F + i]) * gu[(size_t)r * 2 * F + F + i];
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    xq[(size_t)r * F + i] = (int8_t)roundf(v * id);
    if (lane == 0) xs[(size_t)r * nb + b] = d;
}
__global__ void k_swiglu(const float* __restrict__ gu, float* __restrict__ out, int F, int R) {  // gu row = [gate|up]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * F) return;
    const int r = idx / F, i = idx % F;
    out[idx] = siluf_(gu[(size_t)r * 2 * F + i]) * gu[(size_t)r * 2 * F + F + i];
}
__global__ void k_gelu_mul(const float* __restrict__ a, const float* __restrict__ ple, int layer, int NL, int G,
                           float* __restrict__ out, int R) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * G) return;
    const int r = idx / G, i = idx % G;
    out[idx] = gelu_tanh(a[idx]) * ple[((size_t)r * NL + layer) * G + i];
}
__global__ void k_sigmul(float* __restrict__ o, const float* __restrict__ gate, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) o[idx] *= sigmoidf_(gate[idx]);
}

// ---------------------------------------------------------------- RDA
struct RowMeta { int slot, pos, tok, seg_row0; };
struct Seg { int slot, row0, n; };

// causal depthwise conv (4 taps) over q|k|v (C = 6144 channels) + SiLU. qkvg row stride 8192.
__global__ void k_rda_conv(const float* __restrict__ qkvg, int ldq, const RowMeta* __restrict__ rows,
                           const float* __restrict__ state, size_t state_slot_stride, const float* __restrict__ w, int C,
                           int R, float* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * C) return;
    const int r = idx / C, c = idx % C;
    const RowMeta m = rows[r];
    const float* st = state + (size_t)m.slot * state_slot_stride;  // [3][C], oldest first
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int src = r - 3 + k;  // batch row of the input at tap k
        const float v = src >= m.seg_row0 ? qkvg[(size_t)src * ldq + c] : st[(3 + (src - m.seg_row0)) * C + c];
        acc += v * w[c * 4 + k];
    }
    out[(size_t)r * C + c] = siluf_(acc);
}
// after the conv: the last 3 inputs of each segment become the slot's conv state
__global__ void k_rda_conv_state(const float* __restrict__ qkvg, int ldq, const Seg* __restrict__ segs, int nseg,
                                 float* __restrict__ state, size_t state_slot_stride, int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nseg * C) return;
    const int s = idx / C, c = idx % C;
    const Seg g = segs[s];
    float* st = state + (size_t)g.slot * state_slot_stride;
    float old[3] = {st[c], st[C + c], st[2 * C + c]};
    for (int i = 0; i < 3; i++) {
        const int m = g.n - 3 + i;  // relative to the segment start
        st[i * C + c] = m >= 0 ? qkvg[(size_t)(g.row0 + m) * ldq + c] : old[3 + m];
    }
}
// gates + normalisation: q,k L2-normalised per head; v scaled by sigmoid(b+d)/sigmoid(b); log-decay g; beta.
// grid (R, H), block 128 (head dim)
__global__ void k_rda_prep(const float* __restrict__ conv, const float* __restrict__ fab, int ldfab,
                           const float* __restrict__ fb, const float* __restrict__ dt_b, float lb, int H, int D, int rank,
                           float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ vs, float* __restrict__ gl,
                           float* __restrict__ beta) {
    __shared__ float red[32];
    const int r = blockIdx.x, h = blockIdx.y, j = threadIdx.x, C = H * D;
    const float* cr = conv + (size_t)r * 3 * C;
    const float q = cr[h * D + j], k = cr[C + h * D + j], v = cr[2 * C + h * D + j];
    const float qs = block_sum(q * q, red);
    const float ks = block_sum(k * k, red);
    const size_t o = (size_t)r * C + h * D + j;
    qn[o] = q / fmaxf(sqrtf(qs), 1e-6f);
    kn[o] = k / fmaxf(sqrtf(ks), 1e-6f);
    const float b = sigmoidf_(fab[(size_t)r * ldfab + rank + h]);
    const float wr = sigmoidf_(fab[(size_t)r * ldfab + rank + H + h]);
    vs[o] = v * (wr / b);
    gl[o] = lb * sigmoidf_(fb[o] + dt_b[h * D + j]);
    if (j == 0) beta[r * H + h] = b;
}
// gated delta rule, sequential over the rows of a segment. Value columns of S evolve independently (delta_j and the
// update of column j only involve column j), so a head is split over 4 blocks of 32 columns:
// grid (nseg, H, 4), block 128 = 4 warps; warp w holds rows [32w, 32w+32) of the block's 32 columns (lane = column).
// state layout per (slot, head): S[i][j] at i*D + j (i = key dim, j = value dim)
__global__ void __launch_bounds__(128) k_rda_scan(const Seg* __restrict__ segs, const float* __restrict__ qn,
                                                  const float* __restrict__ kn, const float* __restrict__ vs,
                                                  const float* __restrict__ gl, const float* __restrict__ beta,
                                                  float* __restrict__ state, size_t state_slot_stride, int H,
                                                  float* __restrict__ out) {
    constexpr int D = 128, CG = 32, RQ = 4, HR = D / RQ;
    __shared__ float sq[2][D], sk[2][D], sg[2][D], red[RQ][CG];
    const Seg g = segs[blockIdx.x];
    const int h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, rq = tid >> 5, C = H * D;
    const int j = blockIdx.z * CG + lane;  // this thread's column
    const int i0 = rq * HR;
    float* S = state + (size_t)g.slot * state_slot_stride + (size_t)h * D * D;
    float s[HR];
#pragma unroll
    for (int i = 0; i < HR; i++) s[i] = S[(i0 + i) * D + j];
    const float scale = rsqrtf((float)D);
    // the next token's inputs are loaded into registers one iteration ahead (the loop is latency-bound)
    float pq = 0.f, pk = 0.f, pg = 0.f, pv = 0.f, pb = 0.f;
    auto fetch = [&](int t) {
        const int r = g.row0 + t;
        const size_t o = (size_t)r * C + h * D;
        pq = qn[o + tid];
        pk = kn[o + tid];
        pg = gl[o + tid];
        pv = vs[o + j];
        pb = beta[r * H + h];
    };
    if (g.n > 0) fetch(0);
    for (int t = 0; t < g.n; t++) {
        const int r = g.row0 + t, bu = t & 1;
        const size_t o = (size_t)r * C + h * D;
        sq[bu][tid] = pq;
        sk[bu][tid] = pk;
        sg[bu][tid] = __expf(pg);
        const float v = pv, b = pb;
        if (t + 1 < g.n) fetch(t + 1);
        __syncthreads();
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < HR; i++) { s[i] *= sg[bu][i0 + i]; kv += s[i] * sk[bu][i0 + i]; }
        red[rq][lane] = kv;
        __syncthreads();
        const float delta = b * (v - (red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane]));
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < HR; i++) { s[i] += sk[bu][i0 + i] * delta; acc += s[i] * sq[bu][i0 + i]; }
        __syncthreads();  // every warp has read red before it is reused
        red[rq][lane] = acc;
        __syncthreads();
        if (rq == 0) out[o + j] = (red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane]) * scale;
    }
#pragma unroll
    for (int i = 0; i < HR; i++) S[(i0 + i) * D + j] = s[i];
}
// RMSNorm per head (w = ssm_norm) * sigmoid(output gate g2 = qkvg[3C + ...]). grid (R, H), block 128
__global__ void k_rda_out(const float* __restrict__ o, const float* __restrict__ qkvg, int ldq, const float* __restrict__ w,
                          int H, float eps, float* __restrict__ y) {
    __shared__ float red[32];
    const int r = blockIdx.x, h = blockIdx.y, j = threadIdx.x, D = blockDim.x, C = H * D;
    const float v = o[(size_t)r * C + h * D + j];
    const float ss = block_sum(v * v, red);
    const float g = qkvg[(size_t)r * ldq + 3 * C + h * D + j];
    y[(size_t)r * C + h * D + j] = v * rsqrtf(ss / D + eps) * w[j] * sigmoidf_(g);
}

// gather rows (for the LM head)
__global__ void k_gather(const float* __restrict__ x, const int* __restrict__ idx, int n, int E, float* __restrict__ y) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < E; i += blockDim.x) y[(size_t)r * E + i] = x[(size_t)idx[r] * E + i];
}

}  // namespace raon
