// Sampling support on the GPU: penalties applied to the logits in place, then per row the top SAMPLE_K candidates
// and the log-normalisers at temperature 1 and at the row's temperature. The host picks the token from these
// (sampler in engine.cu), and copies a whole row only when the choice reaches past the candidates.
#pragma once
#include <cuda_runtime.h>

#include <cfloat>

namespace raon::smp {

constexpr int SAMPLE_K = 128;        // candidates returned per row
constexpr int CHUNKS = 64;           // blocks per row in the first pass
constexpr int CHUNK_N = 4096;        // values per block (vocab / CHUNKS rounded up to a power of two)

struct Adjust { int row, token; float rep, add; };   // logit = (logit > 0 ? logit / rep : logit * rep) + add

__global__ void k_adjust(float* logits, int V, const Adjust* a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* p = logits + (size_t)a[i].row * V + a[i].token;
    float l = *p;
    if (a[i].rep != 1.f) l = l > 0.f ? l / a[i].rep : l * a[i].rep;
    *p = l + a[i].add;
}

// Bitonic sort of N (key, index) pairs in shared memory, descending by key.
template <int N, int T>
__device__ void bitonic_desc(float* key, int* idx) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += T) {
                const int l = i ^ j;
                if (l > i) {
                    const bool desc = (i & k) == 0;
                    const float a = key[i], b = key[l];
                    if (desc ? a < b : a > b) {
                        key[i] = b; key[l] = a;
                        const int t = idx[i]; idx[i] = idx[l]; idx[l] = t;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Pass 1: grid (CHUNKS, rows). Each block takes a slice of a row: its max, sum exp(l - max) at T = 1 and sum
// exp((l - max) / T), and the slice sorted (descending; padding entries have index -1 and sort last).
__global__ void __launch_bounds__(512) k_topk1(const float* logits, int V, const float* temp, float* ck, int* ci,
                                               float* stats) {
    __shared__ float key[CHUNK_N];
    __shared__ int idx[CHUNK_N];
    __shared__ float red[3][16];
    const int row = blockIdx.y, b = blockIdx.x;
    const int per = (V + CHUNKS - 1) / CHUNKS;
    const int lo = b * per, hi = min(V, lo + per);
    const float* L = logits + (size_t)row * V;
    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < CHUNK_N; i += blockDim.x) {
        const int v = lo + i;
        const float x = v < hi ? L[v] : -FLT_MAX;
        key[i] = x;
        idx[i] = v < hi ? v : -1;
        mx = fmaxf(mx, x);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[0][warp] = mx;
    __syncthreads();
    mx = -FLT_MAX;
    for (int w = 0; w < nw; w++) mx = fmaxf(mx, red[0][w]);
    const float invT = temp[row] > 0.f ? 1.f / temp[row] : 1.f;
    float s1 = 0.f, sT = 0.f;
    for (int i = threadIdx.x; i < CHUNK_N; i += blockDim.x) {
        if (idx[i] < 0) continue;
        const float d = key[i] - mx;
        s1 += __expf(d);
        sT += __expf(d * invT);
    }
    for (int o = 16; o; o >>= 1) {
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
        sT += __shfl_xor_sync(0xffffffffu, sT, o);
    }
    __syncthreads();
    if (lane == 0) { red[1][warp] = s1; red[2][warp] = sT; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f, c = 0.f;
        for (int w = 0; w < nw; w++) { a += red[1][w]; c += red[2][w]; }
        float* st = stats + ((size_t)row * CHUNKS + b) * 3;
        st[0] = mx; st[1] = a; st[2] = c;
    }
    bitonic_desc<CHUNK_N, 512>(key, idx);
    // the whole sorted slice is kept: pass 2 merges the heads, the host merges further when a pick needs it
    for (int i = threadIdx.x; i < CHUNK_N; i += blockDim.x) {
        ck[((size_t)row * CHUNKS + b) * CHUNK_N + i] = key[i];
        ci[((size_t)row * CHUNKS + b) * CHUNK_N + i] = idx[i];
    }
}

// Pass 2: one block per row merges the CHUNKS * SAMPLE_K candidates and the per-slice sums.
// out_k/out_i: [rows][SAMPLE_K]; out_lse: [rows][2] = log-normaliser at T = 1 and of logits / T.
constexpr int MERGE_N = CHUNKS * SAMPLE_K;   // 8192
__global__ void __launch_bounds__(1024) k_topk2(const float* ck, const int* ci, const float* stats, const float* temp,
                                                float* out_k, int* out_i, float* out_lse) {
    extern __shared__ unsigned char sm[];
    float* key = (float*)sm;
    int* idx = (int*)(key + MERGE_N);
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < MERGE_N; i += blockDim.x) {
        const size_t at = ((size_t)row * CHUNKS + i / SAMPLE_K) * CHUNK_N + i % SAMPLE_K;
        key[i] = ck[at];
        idx[i] = ci[at];
    }
    if (threadIdx.x == 0) {
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
    __syncthreads();
    bitonic_desc<MERGE_N, 1024>(key, idx);
    for (int i = threadIdx.x; i < SAMPLE_K; i += blockDim.x) {
        out_k[row * SAMPLE_K + i] = key[i];
        out_i[row * SAMPLE_K + i] = idx[i];
    }
}

}  // namespace raon::smp
