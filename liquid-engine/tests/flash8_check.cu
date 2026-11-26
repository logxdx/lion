// Kernel check: k_flash_prefill8 (int8 QK^T + FP8 PV) vs k_flash_prefill<q8> (f16) vs an f64 reference, on a random
// q8_0 KV cache. Cases: causal from position 0, causal behind a cached prefix, sliding window over the ring, ragged T.
// Also times both kernels on a long-context chunk (warm the GPU first; no other GPU process).
// build: make build/flash8_check      run: build/flash8_check [ctx for the timing case, default 32768]
#include "../src/engine.cu"

#include <random>

using namespace le;

namespace {

struct Case {
    const char* name;
    int p0, T, window;  // window 0: global (cap = p0 + T)
};

struct Data {
    int cap = 0;
    std::vector<int8_t> kq, vq;       // [4][cap][128]
    std::vector<half> ksc, vsc;       // [4][cap][4]
    std::vector<float> q;             // [T][16][128] (unscaled)
    std::vector<float> qkvg;          // [T][3088] (only the gate columns are read)
};

void quant_row(const float* x, int8_t* q, half* s) {
    for (int b = 0; b < 4; b++) {
        float amax = 0;
        for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
        for (int i = 0; i < 32; i++) q[b * 32 + i] = (int8_t)std::lround(x[b * 32 + i] * id);
        s[b] = __float2half(d);
    }
}

Data make_data(const Case& c, std::mt19937& rng) {
    Data d;
    const int npos = c.p0 + c.T;
    d.cap = c.window ? c.window + 1024 : npos;
    d.kq.assign((size_t)4 * d.cap * 128, 0); d.vq.assign((size_t)4 * d.cap * 128, 0);
    d.ksc.assign((size_t)4 * d.cap * 4, __float2half(0.f)); d.vsc.assign((size_t)4 * d.cap * 4, __float2half(0.f));
    std::normal_distribution<float> N(0.f, 1.f);
    // per-dim magnitudes with a few outlier dims (stress the per-block scales); per-position V magnitude drift
    std::vector<float> kmag(4 * 128), vmag(4 * 128);
    for (auto& m : kmag) m = std::exp(0.5f * N(rng)) * (rng() % 17 == 0 ? 6.f : 1.f);
    for (auto& m : vmag) m = std::exp(0.5f * N(rng)) * (rng() % 23 == 0 ? 8.f : 1.f);
    std::vector<float> row(128);
    const int first = c.window ? std::max(0, npos - d.cap) : 0;
    for (int pos = first; pos < npos; pos++) {
        const float vdrift = pos == first ? 0.02f : std::exp(0.7f * N(rng));
        for (int h = 0; h < 4; h++) {
            const size_t r = (size_t)h * d.cap + pos % d.cap;
            for (int i = 0; i < 128; i++) row[i] = N(rng) * kmag[h * 128 + i];
            quant_row(row.data(), &d.kq[r * 128], &d.ksc[r * 4]);
            for (int i = 0; i < 128; i++) row[i] = N(rng) * vmag[h * 128 + i] * vdrift;
            quant_row(row.data(), &d.vq[r * 128], &d.vsc[r * 4]);
        }
    }
    // queries: correlated with some cached keys so attention is peaked in places
    d.q.resize((size_t)c.T * 16 * 128);
    for (int t = 0; t < c.T; t++)
        for (int qh = 0; qh < 16; qh++) {
            const int h = qh / 4;
            const int lo_vis = c.window ? std::max(0, c.p0 + t - c.window + 1) : 0;
            const bool sink = lo_vis == first && rng() % 3 == 0;   // the sink key is visible to this query
            const int tgt = sink ? first : std::max(first, (int)(rng() % (c.p0 + t + 1)));
            const size_t r = (size_t)h * d.cap + tgt % d.cap;
            const float a = sink ? 1.5f : 0.25f * (rng() % 4);
            for (int i = 0; i < 128; i++) {
                const float kd = d.kq[r * 128 + i] * __half2float(d.ksc[r * 4 + i / 32]);
                d.q[((size_t)t * 16 + qh) * 128 + i] = 0.6f * N(rng) + a * kd / (kmag[h * 128 + i] * kmag[h * 128 + i]);
            }
        }
    d.qkvg.assign((size_t)c.T * 3088, 0.f);
    for (int t = 0; t < c.T; t++)
        for (int qh = 0; qh < 16; qh++) d.qkvg[(size_t)t * 3088 + 3072 + qh] = N(rng);
    return d;
}

// f64 reference over the dequantized cache, unquantized queries
void reference(const Case& c, const Data& d, std::vector<double>& out) {
    out.assign((size_t)c.T * 2048, 0.0);
    std::vector<double> s, kd(128);
    for (int t = 0; t < c.T; t++) {
        const int qp = c.p0 + t;
        const int lo = c.window ? std::max(0, qp - c.window + 1) : 0;
        for (int qh = 0; qh < 16; qh++) {
            const int h = qh / 4;
            const float* q = &d.q[((size_t)t * 16 + qh) * 128];
            s.assign(qp - lo + 1, 0.0);
            double mx = -1e300;
            for (int k = lo; k <= qp; k++) {
                const size_t r = (size_t)h * d.cap + k % d.cap;
                double acc = 0;
                for (int i = 0; i < 128; i++) acc += (double)q[i] * d.kq[r * 128 + i] * __half2float(d.ksc[r * 4 + i / 32]);
                s[k - lo] = acc / std::sqrt(128.0);
                mx = std::max(mx, s[k - lo]);
            }
            double l = 0;
            for (auto& x : s) { x = std::exp(x - mx); l += x; }
            double* o = &out[(size_t)t * 2048 + qh * 128];
            for (int k = lo; k <= qp; k++) {
                const size_t r = (size_t)h * d.cap + k % d.cap;
                const double p = s[k - lo] / l;
                for (int i = 0; i < 128; i++) o[i] += p * d.vq[r * 128 + i] * __half2float(d.vsc[r * 4 + i / 32]);
            }
            const double gate = 1.0 / (1.0 + std::exp(-(double)d.qkvg[(size_t)t * 3088 + 3072 + qh]));
            for (int i = 0; i < 128; i++) o[i] *= gate;
        }
    }
}

template <class T>
T* dev(const std::vector<T>& v) {
    T* p;
    CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

struct Dev {
    int8_t *kq, *vq, *q8, *oq;
    half *ksc, *vsc, *qh;
    float *qs8, *qkvg, *od;
    int T;
    explicit Dev(const Data& d, int T_) : T(T_) {
        kq = dev(d.kq); vq = dev(d.vq); ksc = dev(d.ksc); vsc = dev(d.vsc); qkvg = dev(d.qkvg);
        // queries as k_qk_rope_cache writes them: f16 x / sqrt(128) | int8 per (token, head) + d / sqrt(128) * log2(e)
        std::vector<half> qhv(d.q.size());
        std::vector<int8_t> q8v(d.q.size());
        std::vector<float> qsv((size_t)T * 16);
        const float s = 1.f / std::sqrt(128.f);
        for (size_t r = 0; r < (size_t)T * 16; r++) {
            float amax = 0;
            for (int i = 0; i < 128; i++) amax = std::max(amax, std::fabs(d.q[r * 128 + i]));
            const float dd = amax / 127.f, id = dd > 0 ? 1.f / dd : 0.f;
            for (int i = 0; i < 128; i++) {
                qhv[r * 128 + i] = __float2half(d.q[r * 128 + i] * s);
                q8v[r * 128 + i] = (int8_t)std::lround(d.q[r * 128 + i] * id);
            }
            qsv[r] = dd * s * 1.4426950408889634f;
        }
        qh = dev(qhv); q8 = dev(q8v); qs8 = dev(qsv);
        CK(cudaMalloc(&oq, (size_t)T * 2048));
        CK(cudaMalloc(&od, (size_t)T * 64 * 4));
    }
    ~Dev() {
        for (void* p : {(void*)kq, (void*)vq, (void*)q8, (void*)oq, (void*)ksc, (void*)vsc, (void*)qh, (void*)qs8,
                        (void*)qkvg, (void*)od})
            cudaFree(p);
    }
    void run(bool fp8, const Case& c, int cap) {
        const dim3 fg((T + FA_BQ - 1) / FA_BQ, 4);
        CK(cudaMemset(oq, 0x80, (size_t)T * 2048));     // -128 everywhere
        CK(cudaMemset(od, 0xFF, (size_t)T * 64 * 4));   // NaN scales: an unwritten block fails the comparison
        if (fp8)
            k_flash_prefill8<false><<<fg, 256, f8::SMEM>>>(q8, qs8, qkvg, kq, vq, ksc, vsc, cap, c.window, c.p0, T, oq, od);
        else
            k_flash_prefill<true><<<fg, 256, FA_SMEM>>>(qh, qkvg, kq, vq, ksc, vsc, cap, c.window, c.p0, T, oq, od);
    }
    std::vector<double> out() {
        std::vector<int8_t> q((size_t)T * 2048);
        std::vector<float> s((size_t)T * 64);
        CK(cudaMemcpy(q.data(), oq, q.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s.data(), od, s.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<double> o(q.size());
        for (size_t i = 0; i < q.size(); i++) o[i] = q[i] * (double)s[i / 32];
        return o;
    }
};

struct Err { double rel_rms, rel_max; };
Err compare(const std::vector<double>& got, const std::vector<double>& ref) {
    double se = 0, sr = 0, mx = 0, rmax = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (!std::isfinite(got[i])) return {INFINITY, INFINITY};
        se += (got[i] - ref[i]) * (got[i] - ref[i]);
        sr += ref[i] * ref[i];
        mx = std::max(mx, std::fabs(got[i] - ref[i]));
        rmax = std::max(rmax, std::fabs(ref[i]));
    }
    return {std::sqrt(se / sr), mx / rmax};
}

}  // namespace

int main(int argc, char** argv) {
    const int tctx = argc > 1 ? atoi(argv[1]) : 32768;
    CK(cudaFuncSetAttribute(k_flash_prefill<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FA_SMEM));
    CK(cudaFuncSetAttribute(k_flash_prefill8<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8::SMEM));
    std::mt19937 rng(1234);
    const Case cases[] = {
        {"causal p0=0 T=1024", 0, 1024, 0},
        {"causal p0=4100 T=1000", 4100, 1000, 0},
        {"causal p0=63 T=37", 63, 37, 0},
        {"swa p0=3000 T=777 w=256", 3000, 777, 256},
        {"swa p0=0 T=1024 w=256", 0, 1024, 256},
    };
    int fails = 0;
    printf("%-28s %22s %22s\n", "case", "f16 kernel rms / max", "int8+fp8 rms / max");
    for (const Case& c : cases) {
        Data d = make_data(c, rng);
        std::vector<double> ref;
        reference(c, d, ref);
        Dev g(d, c.T);
        g.run(false, c, d.cap);
        CK(cudaDeviceSynchronize());
        const Err e16 = compare(g.out(), ref);
        g.run(true, c, d.cap);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const Err e8 = compare(g.out(), ref);
        // the f16 kernel's error is dominated by the int8 output quantization (~0.5 %); FP8 P and V add ~1-2 %
        const bool ok = e8.rel_rms < std::max(4 * e16.rel_rms, 0.02) && e8.rel_max < 0.1;
        fails += !ok;
        printf("%-28s %10.4f / %-9.4f %10.4f / %-9.4f %s\n", c.name, e16.rel_rms, e16.rel_max, e8.rel_rms, e8.rel_max,
               ok ? "ok" : "FAIL");
    }

    // timing: one 1024-token chunk at depth tctx (global layer)
    {
        const Case c{"timing", tctx - 1024, 1024, 0};
        Data d = make_data(c, rng);
        Dev g(d, c.T);
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        for (int w = 0; w < 30; w++) g.run(w & 1, c, d.cap);  // warm-up (clocks ramp)
        float ms[2] = {0, 0};
        for (int rep = 0; rep < 10; rep++)
            for (int k = 0; k < 2; k++) {  // interleaved
                cudaEventRecord(a);
                for (int i = 0; i < 5; i++) g.run(k, c, d.cap);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float t;
                cudaEventElapsedTime(&t, a, b);
                ms[k] += t / 5;
            }
        const double pairs = (double)c.T * c.p0 + (double)c.T * (c.T + 1) / 2;
        const double fl = pairs * 16 * 128 * 4;
        printf("timing, ctx %d, T=1024: f16 %.3f ms (%.1f TFLOPS), int8+fp8 %.3f ms (%.1f TFLOPS), %.2fx\n", tctx,
               ms[0] / 10, fl / (ms[0] / 10) / 1e9, ms[1] / 10, fl / (ms[1] / 10) / 1e9, ms[0] / ms[1]);
    }
    printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}
