// Sampling support on the GPU: penalties applied to the logits in place, then per row the top SAMPLE_K candidates
// and the log-normalisers at temperature 1 and at the row's temperature. The host picks the token from these
// (sampler in engine.cu), and copies the whole row only when the choice reaches past the candidates.
//
// Selection is a radix select (exact K-th largest value by 4 rounds of 8-bit histograms over order-preserving keys),
// not a sort: pass 1 keeps the top SAMPLE_K of each of CHUNKS slices of a row (unsorted), pass 2 the top SAMPLE_K of
// those, which it sorts (value descending, then index ascending).
#pragma once
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace raon::smp {

constexpr int SAMPLE_K = 128;        // candidates returned per row
constexpr int CHUNKS = 64;           // blocks per row in the first pass
constexpr int K1_THREADS = 256, K1_PER = 16;   // pass 1: up to 4096 values per slice in registers
constexpr int MERGE_N = CHUNKS * SAMPLE_K;     // 8192 candidates per row into pass 2
constexpr int K2_THREADS = 1024, K2_PER = MERGE_N / K2_THREADS;
constexpr size_t CK_PER_ROW = (size_t)CHUNKS * SAMPLE_K;
constexpr int K2_SMEM = 0;           // pass 2 uses static shared memory

struct Adjust { int row, token; float rep, add; };   // logit = (logit > 0 ? logit / rep : logit * rep) + add

__global__ void k_adjust(float* logits, int V, const Adjust* a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* p = logits + (size_t)a[i].row * V + a[i].token;
    float l = *p;
    if (a[i].rep != 1.f) l = l > 0.f ? l / a[i].rep : l * a[i].rep;
    *p = l + a[i].add;
}

__device__ __forceinline__ uint32_t okey(float f) {   // order-preserving: larger float -> larger key
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exact key of the K-th largest of the block's valid values (ky[i] with ok[i]), and how many values equal to it belong
// to the top K. hist: 256 ints of shared memory, sh: at least 2 ints.
template <int PER, int T>
__device__ void radix_kth(const uint32_t (&ky)[PER], const bool (&ok)[PER], int K, int* hist, int* sh, uint32_t& thr,
                          int& need_eq) {
    uint32_t prefix = 0, mask = 0;
    int need = K;
    const int tid = threadIdx.x, lane = tid & 31;
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += T) hist[i] = 0;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < PER; i++)
            if (ok[i] && (ky[i] & mask) == prefix) atomicAdd(&hist[(ky[i] >> shift) & 255], 1);
        __syncthreads();
        if (tid < 32) {
            // lane covers bins 255 - 8 lane .. 248 - 8 lane (descending); counts accumulate from the top bin
            int c[8], tot = 0;
#pragma unroll
            for (int q = 0; q < 8; q++) { c[q] = hist[255 - 8 * lane - q]; tot += c[q]; }
            int incl = tot;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const int excl = incl - tot;
            if (excl < need && need <= incl) {   // the boundary bin is in this lane's range
                int above = excl;
#pragma unroll
                for (int q = 0; q < 8; q++) {
                    if (above + c[q] >= need) { sh[0] = 255 - 8 * lane - q; sh[1] = need - above; break; }
                    above += c[q];
                }
            }
        }
        __syncthreads();
        prefix |= (uint32_t)sh[0] << shift;
        mask |= 255u << shift;
        need = sh[1];
        __syncthreads();
    }
    thr = prefix;
    need_eq = need;
}

// Pass 1: grid (CHUNKS, rows), block 256. Each block takes a slice of a row: its max, sum exp(l - max) at T = 1 and
// sum exp((l - max) / T) (stats), and its top SAMPLE_K values with their vocabulary indices (unsorted).
__global__ void __launch_bounds__(K1_THREADS) k_topk1(const float* logits, int V, const float* temp, float* ck, int* ci,
                                                      float* stats) {
    __shared__ int hist[256], sh[4];
    __shared__ float red[3][K1_THREADS / 32];
    const int row = blockIdx.y, b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int per = (V + CHUNKS - 1) / CHUNKS;
    const int lo = b * per, hi = min(V, lo + per);
    const float* L = logits + (size_t)row * V;
    float x[K1_PER];
    uint32_t ky[K1_PER];
    bool ok[K1_PER];
    float mx = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < K1_PER; i++) {
        const int v = lo + tid + i * K1_THREADS;
        ok[i] = v < hi;
        x[i] = ok[i] ? L[v] : -FLT_MAX;
        ky[i] = okey(x[i]);
        mx = fmaxf(mx, x[i]);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) red[0][warp] = mx;
    __syncthreads();
    mx = -FLT_MAX;
    for (int w = 0; w < K1_THREADS / 32; w++) mx = fmaxf(mx, red[0][w]);
    const float invT = temp[row] > 0.f ? 1.f / temp[row] : 1.f;
    float s1 = 0.f, sT = 0.f;
#pragma unroll
    for (int i = 0; i < K1_PER; i++)
        if (ok[i]) { const float d = x[i] - mx; s1 += __expf(d); sT += __expf(d * invT); }
    for (int o = 16; o; o >>= 1) {
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
        sT += __shfl_xor_sync(0xffffffffu, sT, o);
    }
    if (lane == 0) { red[1][warp] = s1; red[2][warp] = sT; }
    __syncthreads();
    if (tid == 0) {
        float a = 0.f, c = 0.f;
        for (int w = 0; w < K1_THREADS / 32; w++) { a += red[1][w]; c += red[2][w]; }
        float* st = stats + ((size_t)row * CHUNKS + b) * 3;
        st[0] = mx; st[1] = a; st[2] = c;
    }
    const int nvalid = max(0, hi - lo);
    const int K = min(SAMPLE_K, nvalid);
    float* outk = ck + ((size_t)row * CHUNKS + b) * SAMPLE_K;
    int* outi = ci + ((size_t)row * CHUNKS + b) * SAMPLE_K;
    for (int i = K + tid; i < SAMPLE_K; i += K1_THREADS) { outk[i] = -FLT_MAX; outi[i] = -1; }
    if (K == 0) return;
    uint32_t thr;
    int need_eq;
    radix_kth<K1_PER, K1_THREADS>(ky, ok, K, hist, sh, thr, need_eq);
    if (tid == 0) { sh[2] = 0; sh[3] = 0; }
    __syncthreads();
    // K - need_eq values lie strictly above the threshold; need_eq of the values equal to it fill the last slots
#pragma unroll
    for (int i = 0; i < K1_PER; i++) {
        if (!ok[i]) continue;
        const int v = lo + tid + i * K1_THREADS;
        if (ky[i] > thr) {
            const int s = atomicAdd(&sh[2], 1);
            outk[s] = x[i]; outi[s] = v;
        } else if (ky[i] == thr) {
            const int e = atomicAdd(&sh[3], 1);
            if (e < need_eq) { outk[K - need_eq + e] = x[i]; outi[K - need_eq + e] = v; }
        }
    }
}

// Pass 2: one block per row merges the CHUNKS * SAMPLE_K candidates and the per-slice sums.
// out_k/out_i: [rows][SAMPLE_K] sorted (value desc, index asc); out_lse: [rows][2] = log-normaliser at T = 1 and of
// logits / T.
__global__ void __launch_bounds__(K2_THREADS) k_topk2(const float* ck, const int* ci, const float* stats, const float* temp,
                                                      float* out_k, int* out_i, float* out_lse) {
    __shared__ int hist[256], sh[4];
    __shared__ float sk[SAMPLE_K];
    __shared__ int si[SAMPLE_K];
    const int row = blockIdx.x, tid = threadIdx.x;
    if (tid == 0) {
        const float* st = stats + (size_t)row * CHUNKS * 3;
        const float invT = temp[row] > 0.f ? 1.f / temp[row] : 1.f;
        float m = -FLT_MAX;
        for (int b = 0; b < CHUNKS; b++) m = fmaxf(m, st[b * 3]);
        double s1 = 0, sT = 0;
        for (int b = 0; b < CHUNKS; b++) {
            s1 += (double)st[b * 3 + 1] * exp((double)(st[b * 3] - m));
            sT += (double)st[b * 3 + 2] * exp((double)(st[b * 3] - m) * invT);
        }
        out_lse[row * 2 + 0] = m + (float)log(s1);
        out_lse[row * 2 + 1] = m * invT + (float)log(sT);
    }
    float x[K2_PER];
    int ix[K2_PER];
    uint32_t ky[K2_PER];
    bool ok[K2_PER];
    int nv = 0;
#pragma unroll
    for (int i = 0; i < K2_PER; i++) {
        const size_t at = (size_t)row * MERGE_N + tid + i * K2_THREADS;
        x[i] = ck[at];
        ix[i] = ci[at];
        ok[i] = ix[i] >= 0;
        ky[i] = okey(x[i]);
        nv += ok[i];
    }
    // number of valid candidates (>= SAMPLE_K whenever the vocabulary has that many entries)
    if (tid == 0) sh[2] = 0;
    __syncthreads();
    atomicAdd(&sh[2], nv);
    for (int i = tid; i < SAMPLE_K; i += K2_THREADS) { sk[i] = -FLT_MAX; si[i] = 0x7fffffff; }
    __syncthreads();
    const int K = min(SAMPLE_K, sh[2]);
    __syncthreads();
    uint32_t thr = 0;
    int need_eq = 0;
    if (K > 0) radix_kth<K2_PER, K2_THREADS>(ky, ok, K, hist, sh, thr, need_eq);
    if (tid == 0) { sh[2] = 0; sh[3] = 0; }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < K2_PER; i++) {
        if (!ok[i] || K == 0) continue;
        if (ky[i] > thr) {
            const int s = atomicAdd(&sh[2], 1);
            sk[s] = x[i]; si[s] = ix[i];
        } else if (ky[i] == thr) {
            const int e = atomicAdd(&sh[3], 1);
            if (e < need_eq) { sk[K - need_eq + e] = x[i]; si[K - need_eq + e] = ix[i]; }
        }
    }
    __syncthreads();
    // bitonic sort of the SAMPLE_K candidates: value descending, index ascending
    for (int k = 2; k <= SAMPLE_K; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (tid < SAMPLE_K) {
                const int i = tid, l = i ^ j;
                if (l > i) {
                    const bool desc = (i & k) == 0;
                    const float a = sk[i], c = sk[l];
                    const int ia = si[i], ic = si[l];
                    const bool a_first = a > c || (a == c && ia < ic);
                    if (desc ? !a_first : a_first) { sk[i] = c; sk[l] = a; si[i] = ic; si[l] = ia; }
                }
            }
            __syncthreads();
        }
    if (tid < SAMPLE_K) {
        out_k[row * SAMPLE_K + tid] = sk[tid];
        out_i[row * SAMPLE_K + tid] = si[tid] == 0x7fffffff ? -1 : si[tid];
    }
}

}  // namespace raon::smp
