// LFM3-VL vision encoder on the GPU (see vision.h). Linear layers reuse the decoder's int8 tensor-core GEMM
// over repacked Q8_0 weights (W8A8, like llama.cpp's MMQ path); attention is a non-causal f16 flash kernel.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "gemm.cuh"
#include "gguf.h"
#include "kernels.cuh"
#include "repack.cuh"
#include "vision.h"

namespace le {

namespace {
#define VCK(x)                                                                                        \
    do {                                                                                              \
        cudaError_t e_ = (x);                                                                         \
        if (e_ != cudaSuccess)                                                                        \
            throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(e_) + " at " +   \
                                     __FILE__ + ":" + std::to_string(__LINE__));                      \
    } while (0)

constexpr int VD = 1024, VFF = 2560, VNH = 16, VNKV = 8, VHD = 64, VPATCH = 16, VMERGE = 2, VOUT = 2048;
constexpr int VPK = 3 * VPATCH * VPATCH;  // 768 inputs per patch

// ---------------------------------------------------------------------------------------------
// patch embedding: X[p][o] = sum_k patch_p[k] * W[o][k], k = kx + 16*ky + 256*c   (f32 tiled GEMM)
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) k_patch_embed(const float* __restrict__ img, int W, int H, int nx, int NP,
                                                     const float* __restrict__ Wp, float* __restrict__ X) {
    __shared__ float As[32][65];
    __shared__ float Bs[32][65];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int p0 = blockIdx.x * 64, o0 = blockIdx.y * 64;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < VPK; k0 += 32) {
        for (int i = threadIdx.x; i < 32 * 64; i += 256) {
            const int kk = i >> 6, pi = i & 63;
            const int p = p0 + pi, k = k0 + kk;
            float v = 0.f;
            if (p < NP) {
                const int c = k >> 8, ky = (k >> 4) & 15, kx = k & 15;
                const int py = p / nx, px = p % nx;
                v = img[(size_t)c * H * W + (size_t)(py * VPATCH + ky) * W + px * VPATCH + kx];
            }
            As[kk][pi] = v;
            const int oi = i & 63;
            Bs[kk][oi] = Wp[(size_t)(o0 + oi) * VPK + k0 + kk];
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; kk++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int p = p0 + ty * 4 + i;
        if (p >= NP) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) X[(size_t)p * VD + o0 + tx * 4 + j] = acc[i][j];
    }
}

// quantize 4 consecutive values per thread into int8 groups of 32 (8 threads per group)
__device__ __forceinline__ void quant4_store(const float y[4], int8_t* q, float* d, int row_k, int t, int e) {
    float amax = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    const float dd = amax / 127.f, id = dd > 0 ? 1.f / dd : 0.f;
    char4 v;
    v.x = (int8_t)roundf(y[0] * id);
    v.y = (int8_t)roundf(y[1] * id);
    v.z = (int8_t)roundf(y[2] * id);
    v.w = (int8_t)roundf(y[3] * id);
    *(char4*)(q + (size_t)t * row_k + e) = v;
    if ((e & 31) == 0) d[(size_t)t * (row_k / 32) + e / 32] = dd;
}

// RMSNorm in place (embedding_norm)
__global__ void __launch_bounds__(256) k_rmsnorm(float* X, const float* w, float eps) {
    __shared__ float red[32];
    float* x = X + (size_t)blockIdx.x * VD;
    const int e = threadIdx.x * 4;
    float4 v = *(float4*)(x + e);
    const float ss = block_sum<256>(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w, red);
    const float r = rsqrtf(ss / VD + eps);
    v.x *= r * w[e]; v.y *= r * w[e + 1]; v.z *= r * w[e + 2]; v.w *= r * w[e + 3];
    *(float4*)(x + e) = v;
}

// x = X (+ O); X = x; h = rms(x) * w -> int8 [t][1024] + scales
__global__ void __launch_bounds__(256) k_add_norm_quant(float* X, const float* O, const float* w, float eps,
                                                        int8_t* hq, float* hd) {
    __shared__ float red[32];
    const int t = blockIdx.x, e = threadIdx.x * 4;
    float* x = X + (size_t)t * VD;
    float4 v = *(float4*)(x + e);
    if (O) {
        const float4 o = *(const float4*)(O + (size_t)t * VD + e);
        v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
        *(float4*)(x + e) = v;
    }
    const float ss = block_sum<256>(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w, red);
    const float r = rsqrtf(ss / VD + eps);
    const float y[4] = {v.x * r * w[e], v.y * r * w[e + 1], v.z * r * w[e + 2], v.w * r * w[e + 3]};
    quant4_store(y, hq, hd, VD, t, e);
}

// conv mixer: P = [B | C | X] (3 x 1024); out = C * dwconv3x3(B * X) over the patch grid, quantized
__global__ void __launch_bounds__(256) k_conv_mix(const float* __restrict__ P, const float* __restrict__ cw, int nx,
                                                  int ny, int8_t* hq, float* hd) {
    const int t = blockIdx.x, py = t / nx, px = t % nx, e = threadIdx.x * 4;
    float y[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int c = e + i;
        float acc = 0.f;
#pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            const int yy = py + dy;
            if (yy < 0 || yy >= ny) continue;
#pragma unroll
            for (int dx = -1; dx <= 1; dx++) {
                const int xx = px + dx;
                if (xx < 0 || xx >= nx) continue;
                const float* q = P + ((size_t)yy * nx + xx) * (3 * VD);
                acc += cw[c * 9 + (dx + 1) + 3 * (dy + 1)] * (q[c] * q[2 * VD + c]);
            }
        }
        y[i] = P[(size_t)t * 3 * VD + VD + c] * acc;
    }
    quant4_store(y, hq, hd, VD, t, e);
}

// per patch: QK RMSNorm (over 64) + 2D RoPE on q (16 heads) and k (8 heads); q pre-scaled by 1/sqrt(64).
// RoPE pairs (l, l+32), l < 16 rotate by the row y, l >= 16 by the column x, theta = pos * base^(-(l%16)/16)
__global__ void __launch_bounds__(1024) k_qk_rope(const float* __restrict__ QKV, const float* qn, const float* kn,
                                                  int nx, float eps, float base, half* Q, half* K, half* V) {
    const int t = blockIdx.x, hr = threadIdx.x >> 5, l = threadIdx.x & 31;  // hr: 0-15 q, 16-23 k, 24-31 v
    const float* row = QKV + (size_t)t * 2048 + hr * VHD;
    float x0 = row[l], x1 = row[l + 32];
    if (hr >= 24) {
        half* dst = V + ((size_t)t * VNKV + (hr - 24)) * VHD;
        dst[l] = __float2half(x0);
        dst[l + 32] = __float2half(x1);
        return;
    }
    const bool isq = hr < 16;
    const float* nw = isq ? qn : kn;
    const float ss = warp_sum(x0 * x0 + x1 * x1);
    const float r = rsqrtf(ss / VHD + eps);
    x0 *= r * nw[l];
    x1 *= r * nw[l + 32];
    const int pos = l < 16 ? t / nx : t % nx;
    const float ang = (float)pos * powf(base, -(float)(l & 15) / 16.f);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs;
    if (isq) {
        y0 *= 0.125f;
        y1 *= 0.125f;
    }
    half* dst = isq ? Q + ((size_t)t * VNH + hr) * VHD : K + ((size_t)t * VNKV + (hr - 16)) * VHD;
    dst[l] = __float2half(y0);
    dst[l + 32] = __float2half(y1);
}

// non-causal GQA flash attention, head dim 64. grid (ceil(NP/64), 8 kv heads), 8 warps:
// warp w -> query head kvh*2 + (w&1), 16 query rows at blockIdx.x*64 + (w>>1)*16. Output int8 [t][1024] + scales.
constexpr int VA_BK = 64;
__global__ void __launch_bounds__(256) k_vit_attn(const half* __restrict__ Q, const half* __restrict__ K,
                                                  const half* __restrict__ V, int NP, int8_t* oq, float* od) {
    __shared__ __align__(128) half Ks[2][VA_BK * VHD];
    __shared__ __align__(128) half Vs[2][VA_BK * VHD];
    const int kvh = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, tig = lane & 3;
    const int qhd = kvh * 2 + (warp & 1);
    const int t0 = blockIdx.x * 64 + (warp >> 1) * 16;

    uint32_t qa[4][4];
    {
        const int ta = min(t0 + g, NP - 1), tb = min(t0 + g + 8, NP - 1);
        const half* ra = Q + ((size_t)ta * VNH + qhd) * VHD;
        const half* rb = Q + ((size_t)tb * VNH + qhd) * VHD;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            qa[kk][0] = *(const uint32_t*)(ra + kk * 16 + tig * 2);
            qa[kk][1] = *(const uint32_t*)(rb + kk * 16 + tig * 2);
            qa[kk][2] = *(const uint32_t*)(ra + kk * 16 + 8 + tig * 2);
            qa[kk][3] = *(const uint32_t*)(rb + kk * 16 + 8 + tig * 2);
        }
    }
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
    const float LOG2E = 1.4426950408889634f;

    auto issue = [&](int kb, int buf) {  // 64 rows x 8 chunks of 16 B, swizzled
        for (int i = threadIdx.x; i < VA_BK * 8; i += 256) {
            const int r = i >> 3, c = i & 7, kp = kb + r;
            const bool ok = kp < NP;
            const size_t off = ((size_t)(ok ? kp : 0) * VNKV + kvh) * VHD + c * 8;
            const int sc = c ^ (r & 7);
            cp_async16(&Ks[buf][r * VHD + sc * 8], K + off, ok);
            cp_async16(&Vs[buf][r * VHD + sc * 8], V + off, ok);
        }
        cp_commit();
    };
    issue(0, 0);
    int iter = 0;
    for (int kb = 0; kb < NP; kb += VA_BK, iter++) {
        const int cur = iter & 1;
        __syncthreads();
        if (kb + VA_BK < NP) issue(kb + VA_BK, cur ^ 1);
        else cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (t0 >= NP) continue;
        const half* Kt = Ks[cur];
        const half* Vt = Vs[cur];
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
#pragma unroll
            for (int np = 0; np < 4; np++) {
                const int mi = lane >> 3, r = (lane & 7) + (mi >> 1) * 8 + np * 16;
                const int c = kk * 2 + (mi & 1), sc = c ^ (r & 7);
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, Kt + r * VHD + sc * 8);
                mma_f16(s[np * 2], qa[kk], b0, b1);
                mma_f16(s[np * 2 + 1], qa[kk], b2, b3);
            }
        }
        float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const bool ok = kb + nt * 8 + tig * 2 + e < NP;
                s[nt][e] = ok ? s[nt][e] * LOG2E : -INFINITY;
                s[nt][2 + e] = ok ? s[nt][2 + e] * LOG2E : -INFINITY;
                mxa = fmaxf(mxa, s[nt][e]);
                mxb = fmaxf(mxb, s[nt][2 + e]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffff, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffff, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffff, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffff, mxb, 2));
        const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
        const float ca = exp2f(ma - mna), cb = exp2f(mb - mnb);  // ma = -inf on the first tile -> 0
        ma = mna;
        mb = mnb;
        float sa = 0, sb = 0;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[nt][e] = exp2f(s[nt][e] - mna);
                s[nt][2 + e] = exp2f(s[nt][2 + e] - mnb);
                sa += s[nt][e];
                sb += s[nt][2 + e];
            }
        }
        la = la * ca + sa;
        lb = lb * cb + sb;
#pragma unroll
        for (int i = 0; i < 8; i++) { o[i][0] *= ca; o[i][1] *= ca; o[i][2] *= cb; o[i][3] *= cb; }
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            uint32_t pa[4];
            pa[0] = pack_h2(s[2 * kk][0], s[2 * kk][1]);
            pa[1] = pack_h2(s[2 * kk][2], s[2 * kk][3]);
            pa[2] = pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[3] = pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int dp = 0; dp < 4; dp++) {
                const int mi = lane >> 3;
                const int r = kk * 16 + (lane & 7) + (mi & 1) * 8;
                const int c = dp * 2 + (mi >> 1), sc = c ^ (r & 7);
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, Vt + r * VHD + sc * 8);
                mma_f16(o[dp * 2], pa, b0, b1);
                mma_f16(o[dp * 2 + 1], pa, b2, b3);
            }
        }
    }
    if (t0 >= NP) return;
    la += __shfl_xor_sync(0xffffffff, la, 1);
    la += __shfl_xor_sync(0xffffffff, la, 2);
    lb += __shfl_xor_sync(0xffffffff, lb, 1);
    lb += __shfl_xor_sync(0xffffffff, lb, 2);
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        const int t = t0 + g + hf * 8;
        const bool ok = t < NP;
        const float f = 1.f / (hf ? lb : la);
#pragma unroll
        for (int b = 0; b < 2; b++) {
            float v[8], amax = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                v[j * 2] = o[b * 4 + j][hf * 2] * f;
                v[j * 2 + 1] = o[b * 4 + j][hf * 2 + 1] * f;
                amax = fmaxf(amax, fmaxf(fabsf(v[j * 2]), fabsf(v[j * 2 + 1])));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
            const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
            if (ok) {
                int8_t* dst = oq + (size_t)t * VD + qhd * VHD + b * 32 + tig * 2;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    char2 q;
                    q.x = (int8_t)roundf(v[j * 2] * id);
                    q.y = (int8_t)roundf(v[j * 2 + 1] * id);
                    *(char2*)(dst + j * 8) = q;
                }
                if (tig == 0) od[(size_t)t * (VD / 32) + qhd * 2 + b] = d;
            }
        }
    }
}

// projector input: Y = LayerNorm(X + O) per patch (eps 1e-6, weight + bias)
__global__ void __launch_bounds__(256) k_add_layernorm(const float* X, const float* O, const float* w, const float* b,
                                                       float* Y) {
    __shared__ float red[32];
    const int t = blockIdx.x, e = threadIdx.x * 4;
    float4 v = *(const float4*)(X + (size_t)t * VD + e);
    const float4 o = *(const float4*)(O + (size_t)t * VD + e);
    v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
    const float mean = block_sum<256>(v.x + v.y + v.z + v.w, red) / VD;
    const float a0 = v.x - mean, a1 = v.y - mean, a2 = v.z - mean, a3 = v.w - mean;
    const float var = block_sum<256>(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3, red) / VD;
    const float r = rsqrtf(var + 1e-6f);
    float4 y;
    y.x = a0 * r * w[e] + b[e];
    y.y = a1 * r * w[e + 1] + b[e + 1];
    y.z = a2 * r * w[e + 2] + b[e + 2];
    y.w = a3 * r * w[e + 3] + b[e + 3];
    *(float4*)(Y + (size_t)t * VD + e) = y;
}

// 2x2 merge: token (my, mx) = [p(2my,2mx), p(2my,2mx+1), p(2my+1,2mx), p(2my+1,2mx+1)] -> int8 [4096]
__global__ void __launch_bounds__(256) k_merge_quant(const float* Y, int nx, int8_t* hq, float* hd) {
    const int m = blockIdx.x, mxn = nx / 2, my = m / mxn, mx = m % mxn;
#pragma unroll
    for (int it = 0; it < 4; it++) {
        const int e = it * 1024 + threadIdx.x * 4;  // [0, 4096)
        const int sub = e >> 10, c = e & 1023;
        const int py = 2 * my + (sub >> 1), px = 2 * mx + (sub & 1);
        const float4 v = *(const float4*)(Y + ((size_t)py * nx + px) * VD + c);
        const float y[4] = {v.x, v.y, v.z, v.w};
        quant4_store(y, hq, hd, 4 * VD, m, e);
    }
}

// z = GELU_erf(Z + b) -> int8 [4096]
__global__ void __launch_bounds__(256) k_bias_gelu_quant(const float* Z, const float* b, int8_t* hq, float* hd) {
    const int t = blockIdx.x;
#pragma unroll
    for (int it = 0; it < 4; it++) {
        const int e = it * 1024 + threadIdx.x * 4;
        const float4 v = *(const float4*)(Z + (size_t)t * 4 * VD + e);
        float y[4] = {v.x + b[e], v.y + b[e + 1], v.z + b[e + 2], v.w + b[e + 3]};
#pragma unroll
        for (int i = 0; i < 4; i++) y[i] = 0.5f * y[i] * (1.f + erff(y[i] * 0.70710678118654752f));
        quant4_store(y, hq, hd, 4 * VD, t, e);
    }
}

__global__ void k_bias_add(float* E, const float* b, int n, int N) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long)n * N) E[i] += b[i % N];
}

template <int K>
void gemm_store(const int8_t* A, const float* Ad, int T, const QWeight& w, float* out, cudaStream_t s) {
    GemmArgs g{A, Ad, T, w, out, nullptr, nullptr};
    launch_gemm<K, GE_STORE>(g, s);
}
template <int K>
void gemm_swiglu(const int8_t* A, const float* Ad, int T, const QWeight& w, int8_t* yq, float* yd, cudaStream_t s) {
    GemmArgs g{A, Ad, T, w, nullptr, yq, yd};
    launch_gemm<K, GE_SWIGLU>(g, s);
}
}  // namespace

// ---------------------------------------------------------------------------------------------
struct Vision::Impl {
    struct Layer {
        bool conv = false;
        float *ln1 = nullptr, *ln2 = nullptr;
        QWeight w_in, w_out, w_gu, w_down;  // conv: in_proj (3072) | attn: q|k|v (2048); out; ffn
        float* conv_w = nullptr;            // [1024][9]
        float *qn = nullptr, *kn = nullptr;
    };
    std::vector<void*> allocs;
    size_t bytes = 0;
    float* patch_w = nullptr;  // [1024][768] f32, both temporal kernels summed
    float* emb_norm = nullptr;
    std::vector<Layer> layers;
    float *in_norm_w = nullptr, *in_norm_b = nullptr, *b0 = nullptr, *b2 = nullptr;
    QWeight mm0, mm2;
    float eps = 1e-5f, rope_base = 1e6f;
    int64_t min_px = 65536, max_px = 4096 * 1024;
    float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};

    // activation buffers, sized for cap_np patches
    int cap_np = 0;
    float *img = nullptr, *X = nullptr, *O = nullptr, *G = nullptr, *E = nullptr, *hd = nullptr, *yd = nullptr;
    int8_t *hq = nullptr, *yq = nullptr;
    half *Qh = nullptr, *Kh = nullptr, *Vh = nullptr;
    std::vector<void*> act;
    cudaStream_t st = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    void* alloc(size_t n) {
        void* p = nullptr;
        VCK(cudaMalloc(&p, n));
        allocs.push_back(p);
        bytes += n;
        return p;
    }
    float* upload_f32(const gguf::File& f, const std::string& name) {
        const auto& t = f.t(name);
        if (t.dtype != gguf::F32) throw std::runtime_error(name + ": expected F32");
        float* p = (float*)alloc(t.nbytes);
        VCK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
        return p;
    }
    void upload_q8(const gguf::File& f, QWeight& w, const std::vector<std::string>& parts, int tile_in, int tile_out) {
        int N = 0, K = -1;
        for (auto& p : parts) {
            const auto& t = f.t(p);
            if (t.dtype != gguf::Q8_0) throw std::runtime_error(p + ": the vision encoder needs a Q8_0 mmproj");
            if (K < 0) K = (int)t.ne[0];
            if ((int)t.ne[0] != K) throw std::runtime_error("fused weight parts differ: " + p);
            N += (int)t.ne[1];
        }
        w.type = W_Q8;
        w.N = N;
        w.K = K;
        const size_t NP = (size_t)(N + 127) / 128 * 128;
        w.q = (uint8_t*)alloc(NP * K);
        w.d = (half*)alloc(NP * (K / 32) * 2);
        VCK(cudaMemset(w.q, 0, NP * K));
        VCK(cudaMemset(w.d, 0, NP * (K / 32) * 2));
        int off = 0;
        for (size_t pi = 0; pi < parts.size(); pi++) {
            const auto& t = f.t(parts[pi]);
            const int nr = (int)t.ne[1];
            void* staging = nullptr;
            VCK(cudaMalloc(&staging, t.nbytes));
            VCK(cudaMemcpy(staging, t.data, t.nbytes, cudaMemcpyHostToDevice));
            RowMap m = tile_in > 0 ? RowMap{tile_in, tile_out, (int)pi * tile_in} : RowMap{nr, nr, off};
            const long n = (long)nr * (K / 32);
            repack_q8<<<(n + 255) / 256, 256>>>((const uint8_t*)staging, nr, K, w, m);
            VCK(cudaGetLastError());
            VCK(cudaDeviceSynchronize());
            VCK(cudaFree(staging));
            off += nr;
        }
    }
    void ensure(int np) {
        if (np <= cap_np) return;
        for (void* p : act) cudaFree(p);
        act.clear();
        const int n = (np + 127) / 128 * 128;
        auto a = [&](size_t sz) { void* p = nullptr; VCK(cudaMalloc(&p, sz)); act.push_back(p); return p; };
        img = (float*)a((size_t)n * VPK * 4);
        X = (float*)a((size_t)n * VD * 4);
        O = (float*)a((size_t)n * VD * 4);
        G = (float*)a((size_t)n * 3 * VD * 4);
        E = (float*)a((size_t)(n / 4) * VOUT * 4);
        hq = (int8_t*)a((size_t)n * VD);
        hd = (float*)a((size_t)n * (VD / 32) * 4);
        yq = (int8_t*)a((size_t)n * VFF);
        yd = (float*)a((size_t)n * (VFF / 32) * 4);
        Qh = (half*)a((size_t)n * VD * 2);
        Kh = (half*)a((size_t)n * VNKV * VHD * 2);
        Vh = (half*)a((size_t)n * VNKV * VHD * 2);
        cap_np = n;
    }
    ~Impl() {
        for (void* p : act) cudaFree(p);
        for (void* p : allocs) cudaFree(p);
        if (st) cudaStreamDestroy(st);
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
    }
};

Vision::Vision(const std::string& path, bool verbose) : p_(new Impl) {
    auto t0 = std::chrono::steady_clock::now();
    gguf::File f(path);
    auto& P = *p_;
    auto s = [&](const std::string& k) { return f.has(k) ? f.get(k).str : std::string(); };
    if (s("general.architecture") != "clip" || s("clip.vision.projector_type") != "lfm3vl")
        throw std::runtime_error(path + ": not an lfm3vl mmproj");
    auto chk = [&](const char* k, double want) {
        if (!f.has(k) || f.u(k) != want) throw std::runtime_error(std::string("unexpected mmproj ") + k);
    };
    chk("clip.vision.embedding_length", VD);
    chk("clip.vision.feed_forward_length", VFF);
    chk("clip.vision.attention.head_count", VNH);
    chk("clip.vision.attention.head_count_kv", VNKV);
    chk("clip.vision.patch_size", VPATCH);
    chk("clip.vision.spatial_merge_size", VMERGE);
    chk("clip.vision.projection_dim", VOUT);
    const int n_layer = (int)f.u("clip.vision.block_count");
    if (f.has("clip.vision.attention.layer_norm_epsilon")) P.eps = (float)f.u("clip.vision.attention.layer_norm_epsilon");
    // llama.cpp: max tokens capped at 4096 (OOM guard), min from the GGUF
    const int64_t patch_area = VPATCH * VPATCH * VMERGE * VMERGE;
    const int64_t gmin = (int64_t)f.u("clip.vision.image_min_pixels"), gmax = (int64_t)f.u("clip.vision.image_max_pixels");
    P.min_px = gmin / patch_area * patch_area;
    P.max_px = std::min<int64_t>(4096, gmax / patch_area) * patch_area;
    if (f.has("clip.vision.image_mean")) {
        const auto& m = f.get("clip.vision.image_mean").arr_num;
        const auto& d = f.get("clip.vision.image_std").arr_num;
        for (int c = 0; c < 3; c++) { P.mean[c] = (float)m[c]; P.stdv[c] = (float)d[c]; }
    }

    // patch embedding: two F16 temporal kernels; a still image goes through both -> sum them once
    {
        const auto& t0_ = f.t("v.patch_embd.weight");
        const auto& t1_ = f.t("v.patch_embd.weight.1");
        if (t0_.dtype != gguf::F16 || t1_.dtype != gguf::F16) throw std::runtime_error("patch_embd: expected F16");
        std::vector<float> w((size_t)VD * VPK);
        const uint16_t* a = (const uint16_t*)t0_.data;
        const uint16_t* b = (const uint16_t*)t1_.data;
        auto h2f = [](uint16_t h) {
            const uint32_t s_ = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
            uint32_t r;
            if (e == 0) {
                if (m == 0) r = s_;
                else { float v = std::ldexp((float)m, -24); return (h & 0x8000u) ? -v : v; }
            } else if (e == 31) r = s_ | 0x7f800000u | (m << 13);
            else r = s_ | ((e + 112) << 23) | (m << 13);
            float out;
            memcpy(&out, &r, 4);
            return out;
        };
        for (size_t i = 0; i < w.size(); i++) w[i] = h2f(a[i]) + h2f(b[i]);  // ggml order kx,ky,c,o == [o][768]
        P.patch_w = (float*)P.alloc(w.size() * 4);
        VCK(cudaMemcpy(P.patch_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    }
    P.emb_norm = P.upload_f32(f, "v.embedding_norm.weight");
    P.layers.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        auto& L = P.layers[il];
        const std::string p = "v.blk." + std::to_string(il) + ".";
        L.ln1 = P.upload_f32(f, p + "ln1.weight");
        L.ln2 = P.upload_f32(f, p + "ln2.weight");
        L.conv = f.tensors.count(p + "conv2d_in_proj.weight") != 0;
        if (L.conv) {
            P.upload_q8(f, L.w_in, {p + "conv2d_in_proj.weight"}, 0, 0);
            P.upload_q8(f, L.w_out, {p + "conv2d_out_proj.weight"}, 0, 0);
            L.conv_w = P.upload_f32(f, p + "conv2d_spatial.weight");
        } else {
            P.upload_q8(f, L.w_in, {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, 0, 0);
            P.upload_q8(f, L.w_out, {p + "attn_out.weight"}, 0, 0);
            L.qn = P.upload_f32(f, p + "attn_q_norm.weight");
            L.kn = P.upload_f32(f, p + "attn_k_norm.weight");
        }
        P.upload_q8(f, L.w_gu, {p + "ffn_gate.weight", p + "ffn_up.weight"}, 32, 64);
        P.upload_q8(f, L.w_down, {p + "ffn_down.weight"}, 0, 0);
    }
    P.in_norm_w = P.upload_f32(f, "mm.input_norm.weight");
    P.in_norm_b = P.upload_f32(f, "mm.input_norm.bias");
    P.upload_q8(f, P.mm0, {"mm.0.weight"}, 0, 0);
    P.b0 = P.upload_f32(f, "mm.0.bias");
    P.upload_q8(f, P.mm2, {"mm.2.weight"}, 0, 0);
    P.b2 = P.upload_f32(f, "mm.2.bias");
    VCK(cudaStreamCreateWithFlags(&P.st, cudaStreamNonBlocking));
    VCK(cudaEventCreate(&P.e0));
    VCK(cudaEventCreate(&P.e1));
    gpu_bytes_ = P.bytes;
    if (verbose) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[vision] loaded %s: %d blocks, %zu MiB on GPU, %d..%d tokens per image, %.0f ms\n",
                path.c_str(), n_layer, P.bytes >> 20, (int)(P.min_px / patch_area), (int)(P.max_px / patch_area), ms);
    }
}

Vision::~Vision() = default;

void Vision::target_size(int w, int h, int& tw, int& th) const {
    img::smart_resize(w, h, VPATCH * VMERGE, p_->min_px, p_->max_px, tw, th);
}
int Vision::n_tokens(int w, int h) const {
    int tw, th;
    target_size(w, h, tw, th);
    return (tw / 32) * (th / 32);
}

const float* Vision::encode(const Image& im, int& n_tok) {
    auto& P = *p_;
    int tw, th;
    target_size(im.w, im.h, tw, th);
    const Image r = img::resize_bicubic(im, tw, th);
    const int nx = tw / VPATCH, ny = th / VPATCH, NP = nx * ny;
    n_tok = NP / 4;
    P.ensure(NP);
    // normalize to planar f32 [3][H][W]
    std::vector<float> host((size_t)3 * tw * th);
    for (int c = 0; c < 3; c++) {
        const float m = P.mean[c], is = 1.f / P.stdv[c];
        float* dst = host.data() + (size_t)c * tw * th;
        for (size_t i = 0; i < (size_t)tw * th; i++) dst[i] = (r.rgb[i * 3 + c] / 255.f - m) * is;
    }
    cudaStream_t s = P.st;
    VCK(cudaEventRecord(P.e0, s));
    VCK(cudaMemcpyAsync(P.img, host.data(), host.size() * 4, cudaMemcpyHostToDevice, s));
    // LIQUID_VDUMP=dir: intermediate tensors under llama.cpp's names (stage-by-stage comparison with vl_ref_embed)
    const char* dump_dir = getenv("LIQUID_VDUMP");
    auto dump = [&](const std::string& name, const float* a, const float* b, int rows, int cols) {
        if (!dump_dir) return;
        VCK(cudaStreamSynchronize(s));
        std::vector<float> h((size_t)rows * cols), h2;
        VCK(cudaMemcpy(h.data(), a, h.size() * 4, cudaMemcpyDeviceToHost));
        if (b) {
            h2.resize(h.size());
            VCK(cudaMemcpy(h2.data(), b, h2.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < h.size(); i++) h[i] += h2[i];
        }
        FILE* fp = fopen((std::string(dump_dir) + "/" + name + ".bin").c_str(), "wb");
        if (!fp) return;
        int hdr[2] = {cols, rows};
        fwrite(hdr, 4, 2, fp);
        fwrite(h.data(), 4, h.size(), fp);
        fclose(fp);
    };
    k_patch_embed<<<dim3((NP + 63) / 64, VD / 64), 256, 0, s>>>(P.img, tw, th, nx, NP, P.patch_w, P.X);
    dump("patchifier_out", P.X, nullptr, NP, VD);
    k_rmsnorm<<<NP, 256, 0, s>>>(P.X, P.emb_norm, P.eps);
    dump("embedding_norm", P.X, nullptr, NP, VD);
    for (size_t il = 0; il < P.layers.size(); il++) {
        const auto& L = P.layers[il];
        k_add_norm_quant<<<NP, 256, 0, s>>>(P.X, il ? P.O : nullptr, L.ln1, P.eps, P.hq, P.hd);
        gemm_store<1024>(P.hq, P.hd, NP, L.w_in, P.G, s);
        if (L.conv) {
            k_conv_mix<<<NP, 256, 0, s>>>(P.G, L.conv_w, nx, ny, P.hq, P.hd);
        } else {
            k_qk_rope<<<NP, 1024, 0, s>>>(P.G, L.qn, L.kn, nx, P.eps, P.rope_base, P.Qh, P.Kh, P.Vh);
            k_vit_attn<<<dim3((NP + 63) / 64, VNKV), 256, 0, s>>>(P.Qh, P.Kh, P.Vh, NP, P.hq, P.hd);
        }
        gemm_store<1024>(P.hq, P.hd, NP, L.w_out, P.O, s);
        k_add_norm_quant<<<NP, 256, 0, s>>>(P.X, P.O, L.ln2, P.eps, P.hq, P.hd);
        dump("ffn_inp-" + std::to_string(il), P.X, nullptr, NP, VD);
        gemm_swiglu<1024>(P.hq, P.hd, NP, L.w_gu, P.yq, P.yd, s);
        gemm_store<2560>(P.yq, P.yd, NP, L.w_down, P.O, s);
        dump("layer_out-" + std::to_string(il), P.X, P.O, NP, VD);
    }
    k_add_layernorm<<<NP, 256, 0, s>>>(P.X, P.O, P.in_norm_w, P.in_norm_b, P.G);
    dump("proj_input_norm", P.G, nullptr, NP, VD);
    k_merge_quant<<<n_tok, 256, 0, s>>>(P.G, nx, P.hq, P.hd);
    gemm_store<4096>(P.hq, P.hd, n_tok, P.mm0, P.O, s);
    k_bias_gelu_quant<<<n_tok, 256, 0, s>>>(P.O, P.b0, P.hq, P.hd);
    gemm_store<4096>(P.hq, P.hd, n_tok, P.mm2, P.E, s);
    k_bias_add<<<(n_tok * VOUT + 255) / 256, 256, 0, s>>>(P.E, P.b2, n_tok, VOUT);
    VCK(cudaGetLastError());
    VCK(cudaEventRecord(P.e1, s));
    VCK(cudaEventSynchronize(P.e1));
    float ms = 0;
    cudaEventElapsedTime(&ms, P.e0, P.e1);
    last_ms_ = ms;
    return P.E;
}

std::vector<float> Vision::encode_host(const Image& im, int& n_tok) {
    const float* d = encode(im, n_tok);
    std::vector<float> h((size_t)n_tok * VOUT);
    VCK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace le
