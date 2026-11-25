// Engine: slots, caches, the forward pass over a batch of rows.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "engine.h"
#include "kernels.cuh"
#include "attn.cuh"
#include "gemm.cuh"
#include "prefill_attn.cuh"
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
    float *ofa = nullptr, *mlfa = nullptr;
};
static constexpr int FA_KB = 4096;  // keys per expansion block

static float* falloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

Engine::Engine(const std::string& gguf, int n_slots, int max_ctx, bool verbose)
    : n_slots_(n_slots), max_ctx_(max_ctx), rmax_(getenv("RAON_CHUNK") ? atoi(getenv("RAON_CHUNK")) : 256) {
    m_ = load_model(gguf, verbose);
    const Config& c = m_.cfg;
    const size_t conv_n = (size_t)n_slots * m_.n_rda * 3 * 3 * c.rda_heads * c.rda_dim;
    const size_t rda_n = (size_t)n_slots * m_.n_rda * c.rda_heads * c.rda_dim * c.rda_dim;
    const size_t kv_rows = (size_t)n_slots * m_.n_mla * max_ctx;
    CK(cudaMalloc(&conv_state_, conv_n * 4));
    CK(cudaMalloc(&rda_state_, rda_n * 4));
    CK(cudaMalloc(&kv_q_, kv_rows * c.latent()));
    CK(cudaMalloc(&kv_s_, kv_rows * (c.latent() / 32) * 2));
    CK(cudaMemset(conv_state_, 0, conv_n * 4));
    CK(cudaMemset(rda_state_, 0, rda_n * 4));
    state_bytes_ = conv_n * 4 + rda_n * 4 + kv_rows * (c.latent() + c.latent() / 32 * 2);

    ws_ = new Ws;
    Ws& w = *ws_;
    const int R = rmax_, E = c.n_embd, H = c.mla_heads, C = c.rda_heads * c.rda_dim;
    w.R = R;
    CK(cudaMalloc(&w.toks, R * 4));
    CK(cudaMalloc(&w.sel, R * 4));
    CK(cudaMalloc(&w.rows, R * sizeof(RowMeta)));
    CK(cudaMalloc(&w.segs, R * sizeof(Seg)));
    w.X = falloc((size_t)R * c.hc * E);
    w.e = falloc((size_t)R * E);
    w.flat = falloc((size_t)R * c.hc * E);
    w.mixes = falloc((size_t)R * 32);
    w.post = falloc((size_t)R * 4);
    w.comb = falloc((size_t)R * 16);
    w.h = falloc((size_t)R * E);
    w.y = falloc((size_t)R * E);
    w.y2 = falloc((size_t)R * E);
    w.big = falloc((size_t)R * 2 * c.n_ff);
    w.act = falloc((size_t)R * c.n_ff);
    w.s = falloc((size_t)R * std::max(E, C));  // stream sum (E wide) and the RDA gated output (C = 2048 wide)
    w.ple_rows = falloc((size_t)R * c.n_ple * c.n_layer);
    w.ple_proj = falloc((size_t)R * c.n_ple * c.n_layer);
    w.ple = falloc((size_t)R * c.n_ple * c.n_layer);
    w.conv = falloc((size_t)R * 3 * C);
    w.fab = falloc((size_t)R * (c.dt_rank + 2 * c.rda_heads));
    w.fb = falloc((size_t)R * C);
    w.qn = falloc((size_t)R * C);
    w.kn = falloc((size_t)R * C);
    w.vs = falloc((size_t)R * C);
    w.gl = falloc((size_t)R * C);
    w.beta = falloc((size_t)R * c.rda_heads);
    w.ro = falloc((size_t)R * C);
    w.qa = falloc((size_t)R * c.q_lora);
    w.qan = falloc((size_t)R * c.q_lora);
    w.q = falloc((size_t)R * H * c.k_mla);
    w.kv = falloc((size_t)R * c.latent());
    w.qlat = falloc((size_t)R * H * c.kv_lora);
    w.olat = falloc((size_t)R * H * c.kv_lora);
    w.o = falloc((size_t)R * H * c.v_mla);
    w.gate = falloc((size_t)R * H * c.v_mla);
    CK(cudaMalloc(&w.qi, (size_t)R * H * c.latent()));
    w.qsc = falloc((size_t)R * H * (c.latent() / 32));
    w.opart_n = 256;
    w.opart = falloc(w.opart_n * H * c.kv_lora);
    w.ml = falloc(w.opart_n * H * 2);
    CK(cudaFuncSetAttribute(k_mla_decode<RowMeta>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(attn::Smem)));
    CK(cudaFuncSetAttribute(k_gemm_q8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(gq8::Smem)));
    CK(cudaFuncSetAttribute(k_gemm_q6, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(gq6::Smem)));
    CK(cudaMalloc(&w.xq, (size_t)R * 2 * c.n_ff));
    w.xs = falloc((size_t)R * 2 * c.n_ff / 32);
    w.hn = falloc((size_t)R * E);
    w.hsel = falloc((size_t)R * E);
    w.lmax = std::max(n_slots, 16);  // rows that can request logits per chunk
    w.logits = falloc((size_t)w.lmax * c.n_vocab);
    CK(cudaMalloc(&w.qf, (size_t)R * H * c.k_mla * 2));
    CK(cudaMalloc(&w.cf, (size_t)FA_KB * c.latent() * 2));
    CK(cudaMalloc(&w.E, (size_t)FA_KB * fa::EW * 2));
    w.ofa = falloc((size_t)H * R * c.v_mla);
    w.mlfa = falloc((size_t)H * R * 2);
    CK(cudaFuncSetAttribute(k_fa, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(fa::Smem)));
}

Engine::~Engine() {
    g_prof.report();
    free_model(m_);
    cudaFree(conv_state_); cudaFree(rda_state_); cudaFree(kv_q_); cudaFree(kv_s_);
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

static void mm(const Mat& W, const float* x, int ldx, int R, float* y, int ldy, int8_t* xq, float* xs, int rpg = 0,
               int gstride = 0) {
    if (W.type != W_F16) {
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
        k_gemm_q8<<<grid, gq8::THREADS, sizeof(gq8::Smem)>>>(xq, xs, W.q, W.d, R, W.N, W.K, y, ldy);
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
    k_ple_rows<<<R, 256>>>(m_.ple_table, m_.ple_type, m_.ple_row_bytes, G * NL, w.toks, sqrtf((float)G), w.ple_rows);
    mm(m_.ple_model_proj, w.e, E, R, w.ple_proj, G * NL, w.xq, w.xs);
    k_ple_combine<<<dim3(R, NL), G>>>(w.ple_proj, w.ple_rows, m_.ple_proj_norm, G, c.eps, w.ple);
    PROF("embed+ple");

    const size_t conv_slot = (size_t)m_.n_rda * 3 * 3 * C, rda_slot = (size_t)m_.n_rda * RH * D * D;
    const size_t kv_slot_rows = (size_t)m_.n_mla * max_ctx_;

    auto hc_pre = [&](const Mat& fn, const float* base, const float* chart, const float* norm_w) {
        k_rmsnorm<<<dim3(R, 1), 256>>>(w.X, HC * E, w.flat, HC * E, nullptr, HC * E, c.hc_eps);
        mm(fn, w.flat, HC * E, R, w.mixes, c.hc_mix(), w.xq, w.xs);
        k_hc_pre<<<R, 256>>>(w.mixes, base, chart, w.X, norm_w, E, c.eps, c.hc_mag, w.h, w.post, w.comb);
        PROF("mhc_pre");
    };
    auto hc_post = [&]() {
        k_hc_post<<<cdiv(R * E, 256), 256>>>(w.X, w.y, w.post, w.comb, E, R);
        PROF("mhc_post");
    };

    for (int il = 0; il < NL; il++) {
        const Layer& Ly = m_.layers[il];
        hc_pre(Ly.hc_attn_fn, Ly.hc_attn_base, Ly.hc_attn_chart, Ly.attn_norm);
        if (Ly.rda) {
            float* cst = conv_state_ + (size_t)Ly.rda_idx * 3 * 3 * C;
            float* rst = rda_state_ + (size_t)Ly.rda_idx * RH * D * D;
            mm(Ly.wqkv, w.h, E, R, w.big, 4 * C, w.xq, w.xs);
            PROF("rda_qkv");
            k_rda_conv<<<cdiv(R * 3 * C, 256), 256>>>(w.big, 4 * C, w.rows, cst, conv_slot, Ly.conv_w, 3 * C, R, w.conv);
            k_rda_conv_state<<<cdiv(nseg * 3 * C, 256), 256>>>(w.big, 4 * C, w.segs, nseg, cst, conv_slot, 3 * C);
            const int nfab = c.dt_rank + 2 * RH;
            mm(Ly.ssm_f_a, w.h, E, R, w.fab, nfab, w.xq, w.xs);
            mm(Ly.ssm_f_b, w.fab, nfab, R, w.fb, C, w.xq, w.xs);
            k_rda_prep<<<dim3(R, RH), D>>>(w.conv, w.fab, nfab, w.fb, Ly.dt_b, c.gate_lb, RH, D, c.dt_rank, w.qn, w.kn,
                                           w.vs, w.gl, w.beta);
            PROF("rda_conv+gates");
            k_rda_scan<<<dim3(nseg, RH), 2 * D>>>(w.segs, w.qn, w.kn, w.vs, w.gl, w.beta, rst, rda_slot, RH, w.ro);
            PROF("rda_scan");
            k_rda_out<<<dim3(R, RH), D>>>(w.ro, w.big, 4 * C, Ly.ssm_norm, RH, c.eps, w.s);
            mm(Ly.rda_out, w.s, C, R, w.y, E, w.xq, w.xs);
            PROF("rda_out");
        } else {
            int8_t* kq = kv_q_ + (size_t)Ly.mla_idx * max_ctx_ * L;
            half* ks = kv_s_ + (size_t)Ly.mla_idx * max_ctx_ * (L / 32);
            mm(Ly.q_a, w.h, E, R, w.qa, c.q_lora, w.xq, w.xs);
            k_rmsnorm<<<dim3(R, 1), 256>>>(w.qa, c.q_lora, w.qan, c.q_lora, Ly.q_a_norm, c.q_lora, c.eps_latent);
            mm(Ly.q_b, w.qan, c.q_lora, R, w.q, H * c.k_mla, w.xq, w.xs);
            mm(Ly.kv_a, w.h, E, R, w.kv, L, w.xq, w.xs);
            k_mla_cache_write_q8<<<R, 256>>>(w.kv, Ly.kv_a_norm, c.eps_latent, w.rows, kq, ks, kv_slot_rows);
            PROF("mla_proj");
            if (pf_slot >= 0) {
                // prefill segment: expanded attention, key blocks of up to FA_KB keys
                const float scale = 1.f / sqrtf((float)c.k_mla);
                k_fa_qprep<<<cdiv(R * H * c.k_mla, 256), 256>>>(w.q, R, scale, w.qf);
                k_fa_init<<<cdiv(H * R * c.v_mla, 256), 256>>>(w.ofa, w.mlfa, H * R);
                const size_t slot_row0 = (size_t)pf_slot * kv_slot_rows;
                const int kmax = pf_pos0 + R;
                for (int kb0 = 0; kb0 < kmax; kb0 += FA_KB) {
                    const int nk = std::min(FA_KB, kmax - kb0);
                    k_cache_dequant<<<cdiv(nk * (L / 8), 256), 256>>>(kq, ks, slot_row0 + kb0, nk, w.cf);
                    k_gemm_hh<<<dim3(cdiv(nk, 64), fa::EW / 64), 128>>>(w.cf, L, Ly.wexp, nk, fa::EW, c.kv_lora, w.E,
                                                                         fa::EW);
                    PROF("pf_expand");
                    k_fa<<<dim3(cdiv(R, fa::BQ), H), fa::THREADS, sizeof(fa::Smem)>>>(w.qf, R, pf_pos0, w.E, w.cf, nk,
                                                                                       kb0, w.ofa, w.mlfa);
                    PROF("pf_flashattn");
                }
                k_fa_final<<<cdiv(R * H * c.v_mla, 256), 256>>>(w.ofa, w.mlfa, R, w.o);
                PROF("mla_prefill_attn");
            } else {
            mm(Ly.k_b, w.q, H * c.k_mla, R, w.qlat, H * c.kv_lora, w.xq, w.xs, c.kv_lora, c.k_mla);
            k_mla_qprep<<<cdiv(R * H * L, 256), 256>>>(w.qlat, w.q, c.k_mla, 1.f / sqrtf((float)c.k_mla), R, w.qi,
                                                        w.qsc);
            PROF("mla_kb+qprep");
            k_mla_decode<RowMeta><<<dim3(R, nsplit), attn::THREADS, sizeof(attn::Smem)>>>(
                w.qi, w.qsc, kq, ks, kv_slot_rows, w.rows, nsplit, w.opart, w.ml);
            PROF("mla_attn");
            k_mla_combine<<<dim3(R, H), 128>>>(w.opart, w.ml, nsplit, w.olat);
            PROF("mla_combine");
            mm(Ly.v_b, w.olat, H * c.kv_lora, R, w.o, H * c.v_mla, w.xq, w.xs, c.v_mla, c.kv_lora);
            }
            mm(Ly.attn_gate, w.h, E, R, w.gate, H * c.v_mla, w.xq, w.xs);
            k_sigmul<<<cdiv(R * H * c.v_mla, 256), 256>>>(w.o, w.gate, R * H * c.v_mla);
            mm(Ly.mla_out, w.o, H * c.v_mla, R, w.y, E, w.xq, w.xs);
            PROF("mla_vb+gate+out");
        }
        hc_post();
        // FFN
        hc_pre(Ly.hc_ffn_fn, Ly.hc_ffn_base, Ly.hc_ffn_chart, Ly.ffn_norm);
        mm(Ly.ffn_up, w.h, E, R, w.big, 2 * c.n_ff, w.xq, w.xs);
        PROF("ffn_up");
        k_swiglu<<<cdiv(R * c.n_ff, 256), 256>>>(w.big, w.act, c.n_ff, R);
        mm(Ly.ffn_down, w.act, c.n_ff, R, w.y, E, w.xq, w.xs);
        PROF("ffn_down");
        hc_post();
        // per-layer input
        k_stream_sum<<<cdiv(R * E, 256), 256>>>(w.X, w.s, E, R, 1.f);
        mm(Ly.ple_gate, w.s, E, R, w.qa, G, w.xq, w.xs);
        k_gelu_mul<<<cdiv(R * G, 256), 256>>>(w.qa, w.ple, il, NL, G, w.qan, R);
        mm(Ly.ple_proj, w.qan, G, R, w.y, E, w.xq, w.xs);
        k_rmsnorm<<<dim3(R, 1), 256>>>(w.y, E, w.y2, E, Ly.ple_post_norm, E, c.eps);
        k_add_streams<<<cdiv(R * E, 256), 256>>>(w.X, w.y2, E, R);
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
    k_stream_sum<<<cdiv(R * E, 256), 256>>>(w.X, w.s, E, R, 1.f / HC);
    k_rmsnorm<<<dim3(R, 1), 256>>>(w.s, E, w.hn, E, m_.output_norm, E, c.eps);
    k_gather<<<nout, 256>>>(w.hn, w.sel, nout, E, w.hsel);
    mm(m_.tok_embd, w.hsel, E, nout, w.logits, c.n_vocab, w.xq, w.xs);
    PROF("lm_head");
}

std::vector<float> Engine::forward(const std::vector<Row>& rows) {
    const int V = m_.cfg.n_vocab;
    std::vector<float> out;
    for (size_t b0 = 0; b0 < rows.size(); b0 += rmax_) {
        const int n = (int)std::min<size_t>(rmax_, rows.size() - b0);
        run_chunk(rows.data() + b0, n);
        g_prof.calls++;
        int nout = 0;
        for (int i = 0; i < n; i++) nout += rows[b0 + i].logits;
        if (nout) {
            const size_t old = out.size();
            out.resize(old + (size_t)nout * V);
            CK(cudaMemcpy(out.data() + old, ws_->logits, (size_t)nout * V * 4, cudaMemcpyDeviceToHost));
        }
    }
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
        k_fill_half<<<1024, 256>>>(kv_s_ + row0 * (L / 32), (size_t)depth * (L / 32), 0.02f);
    }
    CK(cudaDeviceSynchronize());
}

size_t Engine::gpu_bytes() const { return m_.gpu_bytes + state_bytes_; }

}  // namespace raon
