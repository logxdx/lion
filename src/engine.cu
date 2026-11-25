// Engine: slots, caches, the forward pass over a batch of rows.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

#include "engine.h"
#include "kernels.cuh"
#include "attn.cuh"
#include "gemm.cuh"
#include "prefill_attn.cuh"
#include "sampling.cuh"
#include "prof.h"
#include "util.cuh"

namespace raon {

static Prof g_prof;

struct Engine::Ws {
    int R = 0;
    int *toks = nullptr, *sel = nullptr;
    RowMeta* rows = nullptr;
    Seg* segs = nullptr;
    float *X, *e, *flat, *mixes, *post, *comb, *h, *y, *y2, *big, *act, *s;
    float *ple_rows, *ple_proj, *ple;
    float *conv, *fab, *fb, *qn, *kn, *vs, *gl, *beta, *ro;
    float *qa, *qan, *q, *kv, *qlat, *olat, *o, *gate;
    int8_t* qi = nullptr;        // attention queries, int8 [R][H][576]
    float* qsc = nullptr;        // their scales [R][H][18]
    float *opart = nullptr, *ml = nullptr;
    size_t opart_n = 0;          // capacity in (row, split) pairs
    int8_t* xq = nullptr;
    float* xs = nullptr;
    float *hn, *hsel, *logits;
    std::map<int, cudaGraphExec_t> graphs;  // decode step graphs by batch size
    int lmax = 0;
    // prefill attention (expanded form)
    half *qf = nullptr, *cf = nullptr, *E = nullptr;
    int8_t *q8 = nullptr, *k8 = nullptr, *hq = nullptr;
    float* hs = nullptr;
    float *qs8 = nullptr, *ks8 = nullptr;
    float *ofa = nullptr, *mlfa = nullptr;
    // sampling
    float *stemp = nullptr, *ck = nullptr, *stats = nullptr, *ok = nullptr, *olse = nullptr;
    int *ci = nullptr, *oi = nullptr;
    smp::Adjust* adj = nullptr;
    int adj_cap = 0;
};
static constexpr int FA_KB = 4096;  // keys per expansion block

static float* falloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

Engine::Engine(const std::string& gguf, int n_slots, int max_ctx, bool verbose, int ckpt_per_slot)
    : n_slots_(n_slots), max_ctx_(max_ctx), rmax_(256) {
    m_ = load_model(gguf, verbose);
    for (const Layer& Ly : m_.layers)  // the fused path feeds these from int8 producers: they must be Q8_0 / Q6_K
        for (const Mat* M : {&Ly.wqkv, &Ly.rda_out, &Ly.q_a, &Ly.q_b, &Ly.attn_gate, &Ly.mla_out, &Ly.ffn_up, &Ly.ffn_down,
                             &Ly.ple_gate, &Ly.ple_proj})
            if (M->valid() && M->type == W_F16) throw std::runtime_error("unsupported: F16 weights on an int8-activation path");
    const Config& c = m_.cfg;
    const size_t conv_n = (size_t)n_slots * m_.n_rda * 3 * 3 * c.rda_heads * c.rda_dim;
    const size_t rda_n = (size_t)n_slots * m_.n_rda * c.rda_heads * c.rda_dim * c.rda_dim;
    const size_t kv_rows = (size_t)n_slots * m_.n_mla * max_ctx;
    CK(cudaMalloc(&conv_state_, conv_n * 4));
    CK(cudaMalloc(&rda_state_, rda_n * 4));
    CK(cudaMalloc(&kv_q_, kv_rows * c.latent()));
    CK(cudaMalloc(&kv_s_, kv_rows * attn::KVS * 2));
    CK(cudaMemset(conv_state_, 0, conv_n * 4));
    CK(cudaMemset(rda_state_, 0, rda_n * 4));
    state_bytes_ = conv_n * 4 + rda_n * 4 + kv_rows * (c.latent() + attn::KVS * 2);
    if (ckpt_per_slot > 0) {   // before the chunk is sized from the free memory
        ckpt_per_slot_ = ckpt_per_slot;
        ckpt_bytes_ = (size_t)n_slots * ckpt_per_slot * (conv_n + rda_n) / n_slots * 4;
        CK(cudaMalloc(&ckpt_, ckpt_bytes_));
    }
    size_t free0 = 0, tot0 = 0;
    CK(cudaMemGetInfo(&free0, &tot0));

    ws_ = new Ws;
    Ws& w = *ws_;
    const int E = c.n_embd, H = c.mla_heads, C = c.rda_heads * c.rda_dim;
    // decode-path buffers (absorbed attention) serve at most rdec_ rows per chunk; prefill chunks of one sequence use
    // the expanded attention and do not need them
    rdec_ = std::max(n_slots, 64);
    const int RD = rdec_;
    CK(cudaMalloc(&w.qi, (size_t)RD * H * c.latent()));
    w.qsc = falloc((size_t)RD * H * (c.latent() / 32));
    w.qlat = falloc((size_t)RD * H * c.kv_lora);
    w.olat = falloc((size_t)RD * H * c.kv_lora);
    w.opart_n = 256;
    w.opart = falloc(w.opart_n * H * c.kv_lora);
    w.ml = falloc(w.opart_n * H * 2);
    w.lmax = std::max(n_slots, 16);  // rows that can request logits per chunk
    w.logits = falloc((size_t)w.lmax * c.n_vocab);
    w.stemp = falloc(w.lmax);
    w.ck = falloc((size_t)w.lmax * smp::CHUNKS * smp::CHUNK_N);
    CK(cudaMalloc(&w.ci, (size_t)w.lmax * smp::CHUNKS * smp::CHUNK_N * 4));
    w.stats = falloc((size_t)w.lmax * smp::CHUNKS * 3);
    w.ok = falloc((size_t)w.lmax * smp::SAMPLE_K);
    CK(cudaMalloc(&w.oi, (size_t)w.lmax * smp::SAMPLE_K * 4));
    w.olse = falloc((size_t)w.lmax * 2);
    CK(cudaFuncSetAttribute(smp::k_topk2, cudaFuncAttributeMaxDynamicSharedMemorySize, smp::MERGE_N * 8));
    if (c.n_vocab > smp::CHUNKS * smp::CHUNK_N) throw std::runtime_error("vocab too large for the sampler");
    CK(cudaMalloc(&w.cf, (size_t)FA_KB * c.latent() * 2));
    CK(cudaMalloc(&w.E, (size_t)FA_KB * fa::EW * 2));
    CK(cudaMalloc(&w.k8, (size_t)FA_KB * H * c.k_mla));
    w.ks8 = falloc((size_t)FA_KB * H * 2);

    // per-row workspace: bytes for one row (floats unless noted)
    const size_t per_row = 4 * ((size_t)c.hc * E + 6 * (size_t)E + 4 + 16 + 2 * (size_t)c.n_ff + std::max(E, C) +
                                (size_t)c.n_ple * c.n_layer + (c.dt_rank + 2 * c.rda_heads) + 6 * (size_t)C +
                                c.rda_heads + c.q_lora + (size_t)H * c.k_mla + c.latent() + 3 * (size_t)H * c.v_mla +
                                2 * (size_t)c.n_ff / 32 + E / 32 + H * 2 + H * 2) +
                           2 * (size_t)c.n_ff + E + (size_t)H * c.k_mla /*int8*/ + 2 * (size_t)H * c.k_mla /*qf*/ + 64;
    if (getenv("RAON_CHUNK")) {
        rmax_ = atoi(getenv("RAON_CHUNK"));
    } else {  // largest chunk <= 4096 rows (multiple of 128) that fits in free VRAM with a 256 MB margin
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t margin = (size_t)256 << 20;
        const size_t fit = fr > margin ? (fr - margin) / per_row : 0;
        rmax_ = (int)std::min<size_t>(4096, fit) / 128 * 128;
        if (rmax_ < 64) rmax_ = 64;
    }
    const int R = rmax_;
    if (verbose) fprintf(stderr, "raon: prefill chunk %d rows (%.0f MiB workspace)\n", R, R * per_row / 1048576.0);
    w.R = R;
    CK(cudaMalloc(&w.toks, R * 4));
    CK(cudaMalloc(&w.sel, R * 4));
    CK(cudaMalloc(&w.rows, R * sizeof(RowMeta)));
    CK(cudaMalloc(&w.segs, R * sizeof(Seg)));
    w.X = falloc((size_t)R * c.hc * E);
    w.e = falloc((size_t)R * E);
    w.post = falloc((size_t)R * 4);
    w.comb = falloc((size_t)R * 16);
    w.h = falloc((size_t)R * E);
    w.y = falloc((size_t)R * E);
    w.y2 = falloc((size_t)R * E);
    w.big = falloc((size_t)R * 2 * c.n_ff);
    w.s = falloc((size_t)R * std::max(E, C));  // stream sum (E wide) and the RDA gated output (C = 2048 wide)
    w.ple = falloc((size_t)R * c.n_ple * c.n_layer);
    w.fab = falloc((size_t)R * (c.dt_rank + 2 * c.rda_heads));
    w.fb = falloc((size_t)R * C);
    w.qn = falloc((size_t)R * C);
    w.kn = falloc((size_t)R * C);
    w.vs = falloc((size_t)R * C);
    w.gl = falloc((size_t)R * C);
    w.beta = falloc((size_t)R * c.rda_heads);
    w.ro = falloc((size_t)R * C);
    w.qa = falloc((size_t)R * c.q_lora);
    w.q = falloc((size_t)R * H * c.k_mla);
    w.kv = falloc((size_t)R * c.latent());
    w.o = falloc((size_t)R * H * c.v_mla);
    w.gate = falloc((size_t)R * H * c.v_mla);
    CK(cudaFuncSetAttribute(k_mla_decode<RowMeta>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(attn::Smem)));
    CK(cudaFuncSetAttribute(k_gemm_q8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(gq8::Smem)));
    CK(cudaFuncSetAttribute(k_gemm_q6, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(gq6::Smem)));
    CK(cudaFuncSetAttribute(k_gemm_q8v<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(GQ8V<2>)));
    CK(cudaMalloc(&w.xq, (size_t)R * 2 * c.n_ff));
    w.xs = falloc((size_t)R * 2 * c.n_ff / 32);
    w.hn = falloc((size_t)R * E);
    w.hsel = falloc((size_t)R * E);
    w.ofa = falloc((size_t)H * R * c.v_mla);
    w.mlfa = falloc((size_t)H * R * 2);
    CK(cudaMalloc(&w.qf, (size_t)R * H * c.k_mla * 2));
    CK(cudaFuncSetAttribute(k_fa, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(fa::Smem)));
    CK(cudaFuncSetAttribute(k_fa8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(fa8::Smem)));
    CK(cudaMalloc(&w.q8, (size_t)R * H * c.k_mla));
    CK(cudaMalloc(&w.hq, (size_t)R * E));
    w.hs = falloc((size_t)R * E / 32);
    w.qs8 = falloc((size_t)R * H * 2);
    size_t free1 = 0, tot1 = 0;
    CK(cudaMemGetInfo(&free1, &tot1));
    ws_bytes_ = free0 > free1 ? free0 - free1 : 0;
    if (verbose) fprintf(stderr, "raon: VRAM %s\n", vram_report().c_str());
}

Engine::~Engine() {
    g_prof.report();
    free_model(m_);
    cudaFree(conv_state_); cudaFree(rda_state_); cudaFree(kv_q_); cudaFree(kv_s_); cudaFree(ckpt_);
    // workspace is freed with the process (v1)
    delete ws_;
}

void Engine::reset(int slot) {
    const Config& c = m_.cfg;
    const size_t conv1 = (size_t)m_.n_rda * 3 * 3 * c.rda_heads * c.rda_dim;
    const size_t rda1 = (size_t)m_.n_rda * c.rda_heads * c.rda_dim * c.rda_dim;
    CK(cudaMemset(conv_state_ + slot * conv1, 0, conv1 * 4));
    CK(cudaMemset(rda_state_ + slot * rda1, 0, rda1 * 4));
}

void Engine::save_state(int slot, int k) {
    const Config& c = m_.cfg;
    const size_t conv1 = (size_t)m_.n_rda * 3 * 3 * c.rda_heads * c.rda_dim;
    const size_t rda1 = (size_t)m_.n_rda * c.rda_heads * c.rda_dim * c.rda_dim;
    if (k < 0 || k >= ckpt_per_slot_) throw std::runtime_error("save_state: bad checkpoint index");
    if (!ckpt_) {
        ckpt_bytes_ = (size_t)n_slots_ * ckpt_per_slot_ * (conv1 + rda1) * 4;
        CK(cudaMalloc(&ckpt_, ckpt_bytes_));
    }
    float* dst = ckpt_ + ((size_t)slot * ckpt_per_slot_ + k) * (conv1 + rda1);
    CK(cudaMemcpy(dst, conv_state_ + slot * conv1, conv1 * 4, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(dst + conv1, rda_state_ + slot * rda1, rda1 * 4, cudaMemcpyDeviceToDevice));
}

void Engine::load_state(int slot, int k) {
    if (!ckpt_ || k < 0 || k >= ckpt_per_slot_) throw std::runtime_error("load_state: no such checkpoint");
    const Config& c = m_.cfg;
    const size_t conv1 = (size_t)m_.n_rda * 3 * 3 * c.rda_heads * c.rda_dim;
    const size_t rda1 = (size_t)m_.n_rda * c.rda_heads * c.rda_dim * c.rda_dim;
    const float* src = ckpt_ + ((size_t)slot * ckpt_per_slot_ + k) * (conv1 + rda1);
    CK(cudaMemcpy(conv_state_ + slot * conv1, src, conv1 * 4, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(rda_state_ + slot * rda1, src + conv1, rda1 * 4, cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------- matmul dispatch
__global__ void k_perturb(float* y, int ldy, int N, int R) {  // y *= 1 + 1e-6 * (pseudo-random sign)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * N) return;
    const int r = i / N, n = i % N;
    const float s = ((r * 7919 + n * 104729) & 1) ? 1e-6f : -1e-6f;
    y[(size_t)r * ldy + n] *= 1.f + s;
}

template <int B>
static void launch_gemv(const Mat& W, const float* x, int ldx, const int8_t* xq, const float* xs, float* y, int ldy,
                        int rpg, int gstride) {
    const int threads = 128, blocks = (W.N + 3) / 4;
    if (W.type == W_Q8) k_gemv_q8<B><<<blocks, threads>>>(W.q, W.d, W.N, W.K, xq, xs, y, ldy);
    else if (W.type == W_Q6) k_gemv_q6<B><<<blocks, threads>>>(W.q, W.qh, W.sc, W.d, W.N, W.K, xq, xs, y, ldy);
    else k_gemv_f16<B><<<blocks, threads>>>(W.d, W.N, W.K, x, ldx, rpg ? rpg : W.N, gstride, y, ldy);
}

// prequant: xq/xs already hold the int8 activations (x is not read for Q8/Q6 weights)
static void mm(const Mat& W, const float* x, int ldx, int R, float* y, int ldy, int8_t* xq, float* xs, int rpg = 0,
               int gstride = 0, bool prequant = false) {
    if (W.type != W_F16 && !prequant) {
        const int nb = W.K / 32;
        const int warps = R * nb;
        k_quantize<<<(warps * 32 + 255) / 256, 256>>>(x, ldx, W.K, R, xq, xs);
    }
    // prefill-sized batches: tensor-core GEMM (Q8_0 weights, N a multiple of 128)
    static const bool no_gemm = getenv("RAON_NOGEMM") != nullptr;
    if (!no_gemm && R > 8 && W.type == W_F16 && W.K % gf16::BK == 0 && (rpg == 0 || rpg % gf16::BN == 0)) {
        dim3 grid((R + gf16::BM - 1) / gf16::BM, (W.N + gf16::BN - 1) / gf16::BN);
        k_gemm_f16<<<grid, gf16::THREADS>>>(x, ldx, rpg ? rpg : W.N, gstride, W.d, R, W.N, W.K, y, ldy);
        return;
    }
    if (!no_gemm && R > 8 && W.type == W_Q6 && W.N % gq6::BN == 0 && W.K % 256 == 0) {
        dim3 grid((R + gq6::BM - 1) / gq6::BM, W.N / gq6::BN);
        k_gemm_q6<<<grid, gq6::THREADS, sizeof(gq6::Smem)>>>(xq, xs, (const uint8_t*)W.q, W.qh, W.sc, W.d, R, W.N, W.K, y,
                                                             ldy);
        return;
    }
    if (!no_gemm && R > 8 && W.type == W_Q8 && W.N % gq8::BN == 0) {
        dim3 grid((R + gq8::BM - 1) / gq8::BM, W.N / gq8::BN);
        if (W.K % 64 == 0) k_gemm_q8v<2><<<grid, 256, sizeof(GQ8V<2>)>>>(xq, xs, W.q, W.d, R, W.N, W.K, y, ldy);
        else k_gemm_q8<<<grid, gq8::THREADS, sizeof(gq8::Smem)>>>(xq, xs, W.q, W.d, R, W.N, W.K, y, ldy);
        return;
    }
    for (int b0 = 0; b0 < R; b0 += 8) {
        const int nb = std::min(8, R - b0);
        const float* xb = x + (size_t)b0 * ldx;
        const int8_t* qb = xq + (size_t)b0 * W.K;
        const float* sb = xs + (size_t)b0 * (W.K / 32);
        float* yb = y + (size_t)b0 * ldy;
        switch (nb) {
            case 1: launch_gemv<1>(W, xb, ldx, qb, sb, yb, ldy, rpg, gstride); break;
            case 2: launch_gemv<2>(W, xb, ldx, qb, sb, yb, ldy, rpg, gstride); break;
            case 3: launch_gemv<3>(W, xb, ldx, qb, sb, yb, ldy, rpg, gstride); break;
            case 4: launch_gemv<4>(W, xb, ldx, qb, sb, yb, ldy, rpg, gstride); break;
            case 5: launch_gemv<5>(W, xb, ldx, qb, sb, yb, ldy, rpg, gstride); break;
            case 6: launch_gemv<6>(W, xb, ldx, qb, sb, yb, ldy, rpg, gstride); break;
            case 7: launch_gemv<7>(W, xb, ldx, qb, sb, yb, ldy, rpg, gstride); break;
            default: launch_gemv<8>(W, xb, ldx, qb, sb, yb, ldy, rpg, gstride); break;
        }
    }
    static const bool perturb = getenv("RAON_PERTURB") != nullptr;  // debugging: noise sensitivity
    if (perturb && R > 8) k_perturb<<<(R * W.N + 255) / 256, 256>>>(y, ldy, W.N, R);
}

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

#define PROF(name) g_prof.mark(name)

// ---------------------------------------------------------------- forward over one chunk of rows
void Engine::run_chunk(const Row* rows, int R) {
    Ws& w = *ws_;
    std::vector<RowMeta> meta(R);
    std::vector<Seg> segs;
    std::vector<int> toks(R), sel;
    for (int i = 0; i < R; i++) {
        if (rows[i].slot < 0 || rows[i].slot >= n_slots_) throw std::runtime_error("bad slot");
        if (rows[i].pos >= max_ctx_) throw std::runtime_error("position beyond the slot context");
        if (i == 0 || rows[i].slot != rows[i - 1].slot) segs.push_back({rows[i].slot, i, 0});
        segs.back().n++;
        meta[i] = {rows[i].slot, rows[i].pos, rows[i].tok, segs.back().row0};
        toks[i] = rows[i].tok;
        if (rows[i].logits) sel.push_back(i);
    }
    const int nseg = (int)segs.size();
    CK(cudaMemcpy(w.rows, meta.data(), R * sizeof(RowMeta), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w.segs, segs.data(), nseg * sizeof(Seg), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w.toks, toks.data(), R * 4, cudaMemcpyHostToDevice));
    const int nout = (int)sel.size();
    if (nout > w.lmax) throw std::runtime_error("too many rows with logits in one chunk");
    if (nout) CK(cudaMemcpy(w.sel, sel.data(), nout * 4, cudaMemcpyHostToDevice));
    // attention splits per row: ~60 blocks in total (3 waves at 1 block/SM), within the partials buffer
    static const int nsplit_env = getenv("RAON_NSPLIT") ? atoi(getenv("RAON_NSPLIT")) : 0;  // experiments
    const int nsplit = std::max(1, std::min(nsplit_env ? nsplit_env : 60 / R, (int)w.opart_n / R));
    // a single segment of more than 16 rows is a prefill chunk: expanded attention (RAON_NOFA: decode kernel per row)
    static const bool no_fa = getenv("RAON_NOFA") != nullptr;
    const bool pf = !no_fa && nseg == 1 && R > 16;
    if (!pf && R > rdec_) throw std::runtime_error("decode-path chunk larger than its buffers");
    const int pf_slot = pf ? segs[0].slot : -1, pf_pos0 = pf ? rows[0].pos : 0;

    // decode batches (one row per slot, all with logits) replay a CUDA graph captured once per batch size; the
    // kernels read positions/tokens from device memory, so the graph is valid for every step and context length
    const bool graph = nseg == R && nout == R && !g_prof.on && !getenv("RAON_NOGRAPH");
    if (!graph) {
        launch_chunk(R, nseg, nout, nsplit, pf_slot, pf_pos0);
        return;
    }
    auto it = w.graphs.find(R);
    if (it == w.graphs.end()) {
        cudaGraph_t gr;
        CK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        launch_chunk(R, nseg, nout, nsplit, pf_slot, pf_pos0);
        CK(cudaStreamEndCapture(cudaStreamPerThread, &gr));
        cudaGraphExec_t ex;
        CK(cudaGraphInstantiate(&ex, gr, 0));
        CK(cudaGraphDestroy(gr));
        it = w.graphs.emplace(R, ex).first;
    }
    CK(cudaGraphLaunch(it->second, cudaStreamPerThread));
}

void Engine::launch_chunk(int R, int nseg, int nout, int nsplit, int pf_slot, int pf_pos0) {
    const Config& c = m_.cfg;
    Ws& w = *ws_;
    const int E = c.n_embd, HC = c.hc, H = c.mla_heads, RH = c.rda_heads, D = c.rda_dim, C = RH * D;
    const int NL = c.n_layer, G = c.n_ple, L = c.latent();

    // embeddings
    g_prof.start();
    const Mat& te = m_.tok_embd;
    k_embed<<<R, 256>>>(te.type, te.q, te.qh, te.sc, te.d, E, w.toks, w.e, w.X, HC);
    float* ple_rows = w.big;                          // the per-layer-input temporaries live in big (12288 per row)
    float* ple_proj = w.big + (size_t)R * G * NL;
    k_ple_rows<<<R, 256>>>(m_.ple_table, m_.ple_type, m_.ple_row_bytes, G * NL, w.toks, sqrtf((float)G), ple_rows);
    mm(m_.ple_model_proj, w.e, E, R, ple_proj, G * NL, w.xq, w.xs);
    k_ple_combine<<<dim3(R, NL), G>>>(ple_proj, ple_rows, m_.ple_proj_norm, G, c.eps, w.ple);
    PROF("embed+ple");

    const size_t conv_slot = (size_t)m_.n_rda * 3 * 3 * C, rda_slot = (size_t)m_.n_rda * RH * D * D;
    const size_t kv_slot_rows = (size_t)m_.n_mla * max_ctx_;

    // fused mHC step (k_mhc): optional PLE add, post-mix, stream sum (also int8), pre -> h (also int8 in hq/hs)
    auto mhc = [&](const float* addv, const float* aw, bool post, float* S, float sscale, bool squant, const Layer* pre,
                   bool ffn) {
        MhcArgs a{};
        a.X = w.X; a.R = R; a.A = addv; a.aw = aw;
        a.Y = post ? w.y : nullptr; a.post = w.post; a.comb = w.comb;
        a.S = S; a.sscale = sscale;
        if (squant) { a.Sq = w.xq; a.Ss = w.xs; }
        if (pre) {
            a.fn = (ffn ? pre->hc_ffn_fn : pre->hc_attn_fn).d;
            a.base = ffn ? pre->hc_ffn_base : pre->hc_attn_base;
            a.chart = ffn ? pre->hc_ffn_chart : pre->hc_attn_chart;
            a.nw = ffn ? pre->ffn_norm : pre->attn_norm;
            a.H = w.h;
            a.Hq = w.hq; a.Hs = w.hs;
        }
        a.hc_eps = c.hc_eps; a.eps = c.eps; a.mag = c.hc_mag;
        k_mhc<<<R, 192>>>(a);
        PROF("mhc");
    };
    // int8-activation matmuls whose activations were already quantized by the producing kernel
    auto mmq = [&](const Mat& W, int8_t* q, float* s, float* y, int ldy) { mm(W, nullptr, W.K, R, y, ldy, q, s, 0, 0, true); };

    for (int il = 0; il < NL; il++) {
        const Layer& Ly = m_.layers[il];
        mhc(il ? w.y2 : nullptr, il ? m_.layers[il - 1].ple_post_norm : nullptr, false, nullptr, 0.f, false, &Ly, false);
        if (Ly.rda) {
            float* cst = conv_state_ + (size_t)Ly.rda_idx * 3 * 3 * C;
            float* rst = rda_state_ + (size_t)Ly.rda_idx * RH * D * D;
            mmq(Ly.wqkv, w.hq, w.hs, w.big, 4 * C);
            PROF("rda_qkv");
            const int nfab = c.dt_rank + 2 * RH;
            mm(Ly.ssm_f_a, w.h, E, R, w.fab, nfab, w.xq, w.xs);
            mm(Ly.ssm_f_b, w.fab, nfab, R, w.fb, C, w.xq, w.xs);
            k_rda_conv_prep<<<dim3(R, RH), D>>>(w.big, 4 * C, w.rows, cst, conv_slot, Ly.conv_w, w.fab, nfab, w.fb, Ly.dt_b,
                                                c.gate_lb, RH, D, c.dt_rank, w.qn, w.kn, w.vs, w.gl, w.beta);
            k_rda_conv_state<<<cdiv(nseg * 3 * C, 256), 256>>>(w.big, 4 * C, w.segs, nseg, cst, conv_slot, 3 * C);
            PROF("rda_conv+gates");
            k_rda_scan<<<dim3(nseg, RH, 4), 128>>>(w.segs, w.qn, w.kn, w.vs, w.gl, w.beta, rst, rda_slot, RH, w.ro);
            PROF("rda_scan");
            k_rda_out_quant<<<dim3(R, RH), D>>>(w.ro, w.big, 4 * C, Ly.ssm_norm, RH, c.eps, w.xq, w.xs);
            mmq(Ly.rda_out, w.xq, w.xs, w.y, E);
            PROF("rda_out");
        } else {
            int8_t* kq = kv_q_ + (size_t)Ly.mla_idx * max_ctx_ * L;
            half* ks = kv_s_ + (size_t)Ly.mla_idx * max_ctx_ * attn::KVS;
            mmq(Ly.q_a, w.hq, w.hs, w.qa, c.q_lora);
            k_rmsnorm_quant<<<R, c.q_lora>>>(w.qa, c.q_lora, Ly.q_a_norm, c.eps_latent, w.xq, w.xs);
            mmq(Ly.q_b, w.xq, w.xs, w.q, H * c.k_mla);
            mm(Ly.kv_a, w.h, E, R, w.kv, L, w.xq, w.xs);
            k_mla_cache_write_q8<<<R, 256>>>(w.kv, Ly.kv_a_norm, c.eps_latent, w.rows, kq, ks, kv_slot_rows);
            PROF("mla_proj");
            if (pf_slot >= 0) {
                // prefill segment: expanded attention, key blocks of up to FA_KB keys
                const float scale = 1.f / sqrtf((float)c.k_mla);
                static const bool fa16 = getenv("RAON_FA16") != nullptr;  // f16 QK^T (A/B)
                if (fa16) k_fa_qprep<<<cdiv(R * H * c.k_mla, 256), 256>>>(w.q, R, scale, w.qf);
                else k_fa_qprep8<<<cdiv(R * H * 32, 256), 256>>>(w.q, R, scale, w.q8, w.qs8);
                k_fa_init<<<cdiv(H * R * c.v_mla, 256), 256>>>(w.ofa, w.mlfa, H * R);
                const size_t slot_row0 = (size_t)pf_slot * kv_slot_rows;
                const int kmax = pf_pos0 + R;
                for (int kb0 = 0; kb0 < kmax; kb0 += FA_KB) {
                    const int nk = std::min(FA_KB, kmax - kb0);
                    k_cache_dequant<<<cdiv(nk * (L / 8), 256), 256>>>(kq, ks, slot_row0 + kb0, nk, w.cf);
                    k_gemm_hh<<<dim3(cdiv(nk, 64), fa::EW / 64), 128>>>(w.cf, L, Ly.wexp, nk, fa::EW, c.kv_lora, w.E,
                                                                         fa::EW);
                    if (!fa16) k_fa_kquant<<<cdiv(nk * H * 32, 256), 256>>>(w.E, w.cf, nk, w.k8, w.ks8);
                    PROF("pf_expand");
                    if (fa16)
                        k_fa<<<dim3(cdiv(R, fa::BQ), H), fa::THREADS, sizeof(fa::Smem)>>>(w.qf, R, pf_pos0, w.E, w.cf, nk,
                                                                                           kb0, w.ofa, w.mlfa);
                    else
                        k_fa8<<<dim3(cdiv(R, fa::BQ), H), fa::THREADS, sizeof(fa8::Smem)>>>(
                            w.q8, w.qs8, R, pf_pos0, w.k8, w.ks8, w.E, nk, kb0, w.ofa, w.mlfa);
                    PROF("pf_flashattn");
                }
                k_fa_final<<<cdiv(R * H * c.v_mla, 256), 256>>>(w.ofa, w.mlfa, R, w.o);
                PROF("mla_prefill_attn");
            } else {
                mm(Ly.k_b, w.q, H * c.k_mla, R, w.qlat, H * c.kv_lora, w.xq, w.xs, c.kv_lora, c.k_mla);
                k_mla_qprep<<<cdiv(mla_qprep_threads(R), 256), 256>>>(w.qlat, w.q, c.k_mla, 1.f / sqrtf((float)c.k_mla), R, w.qi,
                                                            w.qsc);
                PROF("mla_kb+qprep");
                k_mla_decode<RowMeta><<<dim3(R, nsplit), attn::THREADS, sizeof(attn::Smem)>>>(
                    w.qi, w.qsc, kq, ks, kv_slot_rows, w.rows, nsplit, w.opart, w.ml);
                PROF("mla_attn");
                k_mla_combine<<<dim3(R, H), 128>>>(w.opart, w.ml, nsplit, w.olat);
                PROF("mla_combine");
                mm(Ly.v_b, w.olat, H * c.kv_lora, R, w.o, H * c.v_mla, w.xq, w.xs, c.v_mla, c.kv_lora);
            }
            mmq(Ly.attn_gate, w.hq, w.hs, w.gate, H * c.v_mla);
            k_sigmul_quant<<<cdiv(R * H * c.v_mla, 256), 256>>>(w.o, w.gate, H * c.v_mla, R, w.xq, w.xs);
            mmq(Ly.mla_out, w.xq, w.xs, w.y, E);
            PROF("mla_vb+gate+out");
        }
        // FFN
        mhc(nullptr, nullptr, true, nullptr, 0.f, false, &Ly, true);
        mmq(Ly.ffn_up, w.hq, w.hs, w.big, 2 * c.n_ff);
        PROF("ffn_up");
        k_swiglu_quantize<<<cdiv(R * c.n_ff, 256), 256>>>(w.big, c.n_ff, R, w.xq, w.xs);
        mmq(Ly.ffn_down, w.xq, w.xs, w.y, E);
        PROF("ffn_down");
        // per-layer input: gate(stream sum) -> gelu x ple -> proj; RMSNorm + add to the streams at the next mHC step
        mhc(nullptr, nullptr, true, w.s, 1.f, true, nullptr, false);
        mmq(Ly.ple_gate, w.xq, w.xs, w.qa, G);
        k_gelu_mul_quant<<<cdiv(R * G, 256), 256>>>(w.qa, w.ple, il, NL, G, R, w.xq, w.xs);
        mmq(Ly.ple_proj, w.xq, w.xs, w.y2, E);
        PROF("ple_layer");
        static const bool dump = getenv("RAON_DUMP") != nullptr;  // debugging: per-layer stream checksums
        if (dump) {
            std::vector<float> hx((size_t)R * HC * E);
            CK(cudaMemcpy(hx.data(), w.X, hx.size() * 4, cudaMemcpyDeviceToHost));
            double s0 = 0, sl = 0;
            for (int i = 0; i < HC * E; i++) { s0 += hx[i]; sl += hx[(size_t)(R - 1) * HC * E + i]; }
            fprintf(stderr, "dump layer %2d (%s) row0 sum %.6f  row%d sum %.6f\n", il, Ly.rda ? "rda" : "mla", s0, R - 1, sl);
        }
    }
    if (nout == 0) return;
    mhc(w.y2, m_.layers[NL - 1].ple_post_norm, false, w.s, 1.f / HC, false, nullptr, false);
    k_rmsnorm<<<dim3(R, 1), 256>>>(w.s, E, w.hn, E, m_.output_norm, E, c.eps);
    k_gather<<<nout, 256>>>(w.hn, w.sel, nout, E, w.hsel);
    mm(m_.tok_embd, w.hsel, E, nout, w.logits, c.n_vocab, w.xq, w.xs);
    PROF("lm_head");
}

std::vector<float> Engine::forward(const std::vector<Row>& rows) {
    const int V = m_.cfg.n_vocab;
    std::vector<float> out;
    for (size_t b0 = 0; b0 < rows.size();) {
        int n = (int)std::min<size_t>(rmax_, rows.size() - b0);
        // a chunk mixing sequences runs the decode-path attention: at most rdec_ rows
        for (int i = 1; i < n; i++)
            if (rows[b0 + i].slot != rows[b0].slot) { n = std::min(n, rdec_); break; }
        run_chunk(rows.data() + b0, n);
        g_prof.calls++;
        int nout = 0;
        for (int i = 0; i < n; i++) nout += rows[b0 + i].logits;
        if (nout) {
            const size_t old = out.size();
            out.resize(old + (size_t)nout * V);
            CK(cudaMemcpy(out.data() + old, ws_->logits, (size_t)nout * V * 4, cudaMemcpyDeviceToHost));
        }
        b0 += n;
    }
    CK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------- sampling
// Picks an entry of a list of logits sorted in descending order, following llama.cpp's chain (top_k -> top_p ->
// min_p on temperature-1 probabilities, then the draw at the temperature). `complete`: the list is the whole
// vocabulary; otherwise lse1/lseT (log-normalisers over the whole vocabulary at T = 1 and of logits / T) are used for
// the draw. Returns -1 when the answer depends on entries past the list.
static int pick(const float* v, int n, bool complete, double lse1, double lseT, const SamplingParams& p, double u) {
    if (p.temperature <= 0.f || p.top_k == 1 || n == 1) return 0;
    int keep = n;
    bool open = !complete;   // the kept set may continue past the list
    bool k_open = false;     // top_k reaches past the list: its normaliser is unknown
    if (p.top_k > 0) {
        if (p.top_k <= n) { keep = p.top_k; open = false; }
        else if (open) k_open = true;
    }
    if (p.top_p < 1.f) {
        if (k_open) return -1;
        double z = lse1;
        if (p.top_k > 0 && !open) {   // probabilities renormalised over the top-k set
            double s = 0;
            for (int i = 0; i < keep; i++) s += std::exp((double)v[i] - v[0]);
            z = v[0] + std::log(s);
        }
        double cum = 0;
        int i = 0;
        for (; i < keep; i++) {
            cum += std::exp((double)v[i] - z);
            if (cum >= p.top_p) break;
        }
        if (i < keep) { keep = i + 1; open = false; }
        else if (open) return -1;
    }
    if (p.min_p > 0.f) {
        const double thr = v[0] + std::log((double)p.min_p);
        int i = 0;
        while (i < keep && v[i] >= thr) i++;
        if (i < keep) { keep = std::max(i, 1); open = false; }
        else if (open) return -1;
    }
    const double invT = 1.0 / p.temperature;
    if (!open) {
        const double m = v[0] * invT;
        double s = 0;
        for (int i = 0; i < keep; i++) s += std::exp(v[i] * invT - m);
        const double target = u * s;
        double cum = 0;
        for (int i = 0; i < keep; i++) {
            cum += std::exp(v[i] * invT - m);
            if (cum > target) return i;
        }
        return keep - 1;
    }
    if (k_open) return -1;
    double cum = 0;   // whole vocabulary at temperature T: walk the list, the rest is the tail
    for (int i = 0; i < n; i++) {
        cum += std::exp(v[i] * invT - lseT);
        if (cum > u) return i;
    }
    return -1;
}

void Engine::sample_chunk(const SampleIn* in, int nout, SampleOut* out) {
    Ws& w = *ws_;
    const int V = m_.cfg.n_vocab, K = smp::SAMPLE_K;
    std::vector<smp::Adjust> adj;
    std::vector<float> temp(nout);
    for (int r = 0; r < nout; r++) {
        const SamplingParams& p = in[r].p;
        temp[r] = p.temperature > 0.f ? p.temperature : 1.f;
        std::map<int, std::pair<float, float>> a;   // token -> (rep, add)
        if (in[r].counts && (p.presence_penalty != 0.f || p.frequency_penalty != 0.f || p.repetition_penalty != 1.f))
            for (auto [tok, cnt] : *in[r].counts) {
                if (tok < 0 || tok >= V || cnt <= 0) continue;
                auto& e = a.try_emplace(tok, 1.f, 0.f).first->second;
                e.first = p.repetition_penalty;
                e.second -= p.presence_penalty + p.frequency_penalty * cnt;
            }
        if (in[r].bias)
            for (auto [tok, b] : *in[r].bias)
                if (tok >= 0 && tok < V) a.try_emplace(tok, 1.f, 0.f).first->second.second += b;
        for (auto& [tok, e] : a) adj.push_back({r, tok, e.first, e.second});
    }
    if (!adj.empty()) {
        if ((int)adj.size() > w.adj_cap) {
            cudaFree(w.adj);
            w.adj_cap = std::max<int>(1024, adj.size() * 2);
            CK(cudaMalloc(&w.adj, w.adj_cap * sizeof(smp::Adjust)));
        }
        CK(cudaMemcpy(w.adj, adj.data(), adj.size() * sizeof(smp::Adjust), cudaMemcpyHostToDevice));
        smp::k_adjust<<<cdiv((int)adj.size(), 256), 256>>>(w.logits, V, w.adj, (int)adj.size());
    }
    CK(cudaMemcpy(w.stemp, temp.data(), nout * 4, cudaMemcpyHostToDevice));
    smp::k_topk1<<<dim3(smp::CHUNKS, nout), 512>>>(w.logits, V, w.stemp, w.ck, w.ci, w.stats);
    smp::k_topk2<<<nout, 1024, smp::MERGE_N * 8>>>(w.ck, w.ci, w.stats, w.stemp, w.ok, w.oi, w.olse);
    std::vector<float> hv((size_t)nout * K), hl((size_t)nout * 2);
    std::vector<int> hi((size_t)nout * K);
    CK(cudaMemcpy(hv.data(), w.ok, hv.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hi.data(), w.oi, hi.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hl.data(), w.olse, hl.size() * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < nout; r++) {
        const SamplingParams& p = in[r].p;
        const float* v = hv.data() + (size_t)r * K;
        const int* id = hi.data() + (size_t)r * K;
        double lse1 = hl[r * 2], lseT = hl[r * 2 + 1];
        const int nl = std::min(std::max(p.n_logprobs, 0), 20);
        SampleOut& o = out[r];
        o = SampleOut();
        int k = pick(v, K, false, lse1, lseT, p, in[r].u);
        if (k >= 0) {
            o.token = id[k];
            o.logprob = (float)(v[k] - lse1);
            for (int j = 0; j < nl; j++) o.top.push_back({id[j], (float)(v[j] - lse1)});
            continue;
        }
        // the pick reaches past the candidates: merge the row's sorted slices (kept by pass 1) on the host, as far as
        // the decision needs; the list is extended 4x at a time
        const size_t RN = (size_t)smp::CHUNKS * smp::CHUNK_N;
        std::vector<float> rk(RN);
        std::vector<int> ri(RN);
        CK(cudaMemcpy(rk.data(), w.ck + r * RN, RN * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ri.data(), w.ci + r * RN, RN * 4, cudaMemcpyDeviceToHost));
        std::vector<int> head(smp::CHUNKS, 0);
        auto cmp = [&](int a, int b) { return rk[(size_t)a * smp::CHUNK_N + head[a]] < rk[(size_t)b * smp::CHUNK_N + head[b]]; };
        std::vector<int> heap;
        for (int b = 0; b < smp::CHUNKS; b++)
            if (ri[(size_t)b * smp::CHUNK_N] >= 0) heap.push_back(b);
        std::make_heap(heap.begin(), heap.end(), cmp);
        std::vector<float> sv;
        std::vector<int> sid;
        size_t want = 4 * K;
        while (true) {
            while (sv.size() < want && !heap.empty()) {
                std::pop_heap(heap.begin(), heap.end(), cmp);
                const int b = heap.back();
                const size_t at = (size_t)b * smp::CHUNK_N + head[b];
                sv.push_back(rk[at]);
                sid.push_back(ri[at]);
                if (++head[b] < smp::CHUNK_N && ri[(size_t)b * smp::CHUNK_N + head[b]] >= 0) std::push_heap(heap.begin(), heap.end(), cmp);
                else heap.pop_back();
            }
            const bool complete = heap.empty();
            k = pick(sv.data(), (int)sv.size(), complete, lse1, lseT, p, in[r].u);
            if (k >= 0 || complete) break;
            want *= 4;
        }
        if (k < 0) k = (int)sv.size() - 1;
        o.token = sid[k];
        o.logprob = (float)(sv[k] - lse1);
        for (int j = 0; j < nl; j++) o.top.push_back({sid[j], (float)(sv[j] - lse1)});
        o.full_copy = true;
    }
}

std::vector<SampleOut> Engine::step(const std::vector<Row>& rows, const std::vector<SampleIn>& in) {
    std::vector<SampleOut> out(in.size());
    size_t used = 0;
    for (size_t b0 = 0; b0 < rows.size();) {
        int n = (int)std::min<size_t>(rmax_, rows.size() - b0);
        for (int i = 1; i < n; i++)
            if (rows[b0 + i].slot != rows[b0].slot) { n = std::min(n, rdec_); break; }
        run_chunk(rows.data() + b0, n);
        g_prof.calls++;
        int nout = 0;
        for (int i = 0; i < n; i++) nout += rows[b0 + i].logits;
        if (nout) {
            if (used + nout > in.size()) throw std::runtime_error("step: fewer SampleIn than rows with logits");
            sample_chunk(in.data() + used, nout, out.data() + used);
            used += nout;
        }
        b0 += n;
    }
    CK(cudaStreamSynchronize(cudaStreamPerThread));   // the work is done when step returns (schedulers time it)
    CK(cudaGetLastError());
    return out;
}

std::vector<float> Engine::prefill(int slot, const std::vector<int>& toks, int pos0) {
    std::vector<Row> rows(toks.size());
    for (size_t i = 0; i < toks.size(); i++) rows[i] = {slot, pos0 + (int)i, toks[i], i + 1 == toks.size()};
    return forward(rows);
}

__global__ void k_fill_rand(int8_t* q, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        q[i] = (int8_t)((int)(x & 0xff) - 128);
    }
}
__global__ void k_fill_half(half* p, size_t n, float v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2half(v);
}

void Engine::bench_fill(int slot, int depth) {
    const Config& c = m_.cfg;
    const int L = c.latent();
    reset(slot);
    for (int l = 0; l < m_.n_mla; l++) {
        const size_t row0 = ((size_t)slot * m_.n_mla + l) * max_ctx_;
        k_fill_rand<<<1024, 256>>>(kv_q_ + row0 * L, (size_t)depth * L, 1234u + l);
        k_fill_half<<<1024, 256>>>(kv_s_ + row0 * attn::KVS, (size_t)depth * attn::KVS, 0.02f);
    }
    CK(cudaDeviceSynchronize());
}

size_t Engine::gpu_bytes() const { return m_.gpu_bytes + state_bytes_ + ckpt_bytes_; }

std::string Engine::vram_report() const {
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    const double M = 1048576.0;
    const Config& c = m_.cfg;
    const size_t kv = (size_t)n_slots_ * m_.n_mla * max_ctx_ * (c.latent() + attn::KVS * 2);
    char b[320];
    snprintf(b, sizeof b,
             "weights %.0f MiB, latent KV cache %.0f MiB (%d x %d tokens), recurrent state %.0f MiB, checkpoints %.0f MiB, "
             "workspace %.0f MiB (chunk %d rows); device used %.0f of %.0f MiB",
             m_.gpu_bytes / M, kv / M, n_slots_, max_ctx_, (state_bytes_ - kv) / M, ckpt_bytes_ / M, ws_bytes_ / M, rmax_,
             (tot - fr) / M, tot / M);
    return b;
}

}  // namespace raon
