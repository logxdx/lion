// Engine: slots, caches, the forward pass over a batch of rows.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "engine.h"
#include "kernels.cuh"
#include "util.cuh"

namespace raon {

struct Engine::Ws {
    int R = 0;
    int *toks = nullptr, *sel = nullptr;
    RowMeta* rows = nullptr;
    Seg* segs = nullptr;
    float *X, *e, *flat, *mixes, *post, *comb, *h, *y, *y2, *big, *act, *s;
    float *ple_rows, *ple_proj, *ple;
    float *conv, *fab, *fb, *qn, *kn, *vs, *gl, *beta, *ro;
    float *qa, *qan, *q, *kv, *qlat, *olat, *o, *gate;
    float* scratch = nullptr;  // attention scores [R][H][max_keys]
    int max_keys = 0;
    int8_t* xq = nullptr;
    float* xs = nullptr;
    float *hn, *hsel, *logits;
};

static float* falloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

Engine::Engine(const std::string& gguf, int n_slots, int max_ctx, bool verbose)
    : n_slots_(n_slots), max_ctx_(max_ctx), rmax_(64) {
    m_ = load_model(gguf, verbose);
    const Config& c = m_.cfg;
    const size_t conv_n = (size_t)n_slots * m_.n_rda * 3 * 3 * c.rda_heads * c.rda_dim;
    const size_t rda_n = (size_t)n_slots * m_.n_rda * c.rda_heads * c.rda_dim * c.rda_dim;
    const size_t kv_n = (size_t)n_slots * m_.n_mla * max_ctx * c.latent();
    CK(cudaMalloc(&conv_state_, conv_n * 4));
    CK(cudaMalloc(&rda_state_, rda_n * 4));
    CK(cudaMalloc(&kv_cache_, kv_n * 2));
    CK(cudaMemset(conv_state_, 0, conv_n * 4));
    CK(cudaMemset(rda_state_, 0, rda_n * 4));
    CK(cudaMemset(kv_cache_, 0, kv_n * 2));

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
    w.s = falloc((size_t)R * E);
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
    w.max_keys = std::min(max_ctx, 16384);
    w.scratch = falloc((size_t)R * H * w.max_keys);
    CK(cudaMalloc(&w.xq, (size_t)R * 2 * c.n_ff));
    w.xs = falloc((size_t)R * 2 * c.n_ff / 32);
    w.hn = falloc((size_t)R * E);
    w.hsel = falloc((size_t)R * E);
    w.logits = falloc((size_t)R * c.n_vocab);
}

Engine::~Engine() {
    free_model(m_);
    cudaFree(conv_state_); cudaFree(rda_state_); cudaFree(kv_cache_);
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
}

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------- forward over one chunk of rows
void Engine::run_chunk(const Row* rows, int R) {
    const Config& c = m_.cfg;
    Ws& w = *ws_;
    const int E = c.n_embd, HC = c.hc, H = c.mla_heads, RH = c.rda_heads, D = c.rda_dim, C = RH * D;
    const int NL = c.n_layer, G = c.n_ple, L = c.latent();

    std::vector<RowMeta> meta(R);
    std::vector<Seg> segs;
    std::vector<int> toks(R), sel;
    for (int i = 0; i < R; i++) {
        if (rows[i].slot < 0 || rows[i].slot >= n_slots_) throw std::runtime_error("bad slot");
        if (rows[i].pos >= max_ctx_) throw std::runtime_error("position beyond the slot context");
        if (rows[i].pos >= w.max_keys) throw std::runtime_error("v1 attention supports at most 16384 positions");
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

    // embeddings
    const Mat& te = m_.tok_embd;
    k_embed<<<R, 256>>>(te.type, te.q, te.qh, te.sc, te.d, E, w.toks, w.e, w.X, HC);
    k_ple_rows<<<R, 256>>>(m_.ple_table, m_.ple_type, m_.ple_row_bytes, G * NL, w.toks, sqrtf((float)G), w.ple_rows);
    mm(m_.ple_model_proj, w.e, E, R, w.ple_proj, G * NL, w.xq, w.xs);
    k_ple_combine<<<dim3(R, NL), G>>>(w.ple_proj, w.ple_rows, m_.ple_proj_norm, G, c.eps, w.ple);

    const size_t conv_slot = (size_t)m_.n_rda * 3 * 3 * C, rda_slot = (size_t)m_.n_rda * RH * D * D;
    const size_t kv_slot = (size_t)m_.n_mla * max_ctx_ * L;

    auto hc_pre = [&](const Mat& fn, const float* base, const float* chart, const float* norm_w) {
        k_rmsnorm<<<dim3(R, 1), 256>>>(w.X, HC * E, w.flat, HC * E, nullptr, HC * E, c.hc_eps);
        mm(fn, w.flat, HC * E, R, w.mixes, c.hc_mix(), w.xq, w.xs);
        k_hc_pre<<<R, 256>>>(w.mixes, base, chart, w.X, norm_w, E, c.eps, c.hc_mag, w.h, w.post, w.comb);
    };
    auto hc_post = [&]() { k_hc_post<<<cdiv(R * E, 256), 256>>>(w.X, w.y, w.post, w.comb, E, R); };

    for (int il = 0; il < NL; il++) {
        const Layer& Ly = m_.layers[il];
        hc_pre(Ly.hc_attn_fn, Ly.hc_attn_base, Ly.hc_attn_chart, Ly.attn_norm);
        if (Ly.rda) {
            float* cst = conv_state_ + (size_t)Ly.rda_idx * 3 * 3 * C;
            float* rst = rda_state_ + (size_t)Ly.rda_idx * RH * D * D;
            mm(Ly.wqkv, w.h, E, R, w.big, 4 * C, w.xq, w.xs);
            k_rda_conv<<<cdiv(R * 3 * C, 256), 256>>>(w.big, 4 * C, w.rows, cst, conv_slot, Ly.conv_w, 3 * C, R, w.conv);
            k_rda_conv_state<<<cdiv(nseg * 3 * C, 256), 256>>>(w.big, 4 * C, w.segs, nseg, cst, conv_slot, 3 * C);
            const int nfab = c.dt_rank + 2 * RH;
            mm(Ly.ssm_f_a, w.h, E, R, w.fab, nfab, w.xq, w.xs);
            mm(Ly.ssm_f_b, w.fab, nfab, R, w.fb, C, w.xq, w.xs);
            k_rda_prep<<<dim3(R, RH), D>>>(w.conv, w.fab, nfab, w.fb, Ly.dt_b, c.gate_lb, RH, D, c.dt_rank, w.qn, w.kn,
                                           w.vs, w.gl, w.beta);
            k_rda_scan<<<dim3(nseg, RH), D>>>(w.segs, w.qn, w.kn, w.vs, w.gl, w.beta, rst, rda_slot, RH, w.ro);
            k_rda_out<<<dim3(R, RH), D>>>(w.ro, w.big, 4 * C, Ly.ssm_norm, RH, c.eps, w.s);
            mm(Ly.rda_out, w.s, C, R, w.y, E, w.xq, w.xs);
        } else {
            half* kc = kv_cache_ + (size_t)Ly.mla_idx * max_ctx_ * L;
            mm(Ly.q_a, w.h, E, R, w.qa, c.q_lora, w.xq, w.xs);
            k_rmsnorm<<<dim3(R, 1), 256>>>(w.qa, c.q_lora, w.qan, c.q_lora, Ly.q_a_norm, c.q_lora, c.eps_latent);
            mm(Ly.q_b, w.qan, c.q_lora, R, w.q, H * c.k_mla, w.xq, w.xs);
            mm(Ly.kv_a, w.h, E, R, w.kv, L, w.xq, w.xs);
            k_mla_cache_write<<<R, 256>>>(w.kv, Ly.kv_a_norm, c.eps_latent, w.rows, kc, kv_slot, L, c.kv_lora);
            mm(Ly.k_b, w.q, H * c.k_mla, R, w.qlat, H * c.kv_lora, w.xq, w.xs, c.kv_lora, c.k_mla);
            k_mla_attn_simple<<<dim3(R, H), 128>>>(w.qlat, w.q, kc, kv_slot, w.rows, H, 1.f / sqrtf((float)c.k_mla),
                                                    w.scratch, w.max_keys, w.olat);
            mm(Ly.v_b, w.olat, H * c.kv_lora, R, w.o, H * c.v_mla, w.xq, w.xs, c.v_mla, c.kv_lora);
            mm(Ly.attn_gate, w.h, E, R, w.gate, H * c.v_mla, w.xq, w.xs);
            k_sigmul<<<cdiv(R * H * c.v_mla, 256), 256>>>(w.o, w.gate, R * H * c.v_mla);
            mm(Ly.mla_out, w.o, H * c.v_mla, R, w.y, E, w.xq, w.xs);
        }
        hc_post();
        // FFN
        hc_pre(Ly.hc_ffn_fn, Ly.hc_ffn_base, Ly.hc_ffn_chart, Ly.ffn_norm);
        mm(Ly.ffn_up, w.h, E, R, w.big, 2 * c.n_ff, w.xq, w.xs);
        k_swiglu<<<cdiv(R * c.n_ff, 256), 256>>>(w.big, w.act, c.n_ff, R);
        mm(Ly.ffn_down, w.act, c.n_ff, R, w.y, E, w.xq, w.xs);
        hc_post();
        // per-layer input
        k_stream_sum<<<cdiv(R * E, 256), 256>>>(w.X, w.s, E, R, 1.f);
        mm(Ly.ple_gate, w.s, E, R, w.qa, G, w.xq, w.xs);
        k_gelu_mul<<<cdiv(R * G, 256), 256>>>(w.qa, w.ple, il, NL, G, w.qan, R);
        mm(Ly.ple_proj, w.qan, G, R, w.y, E, w.xq, w.xs);
        k_rmsnorm<<<dim3(R, 1), 256>>>(w.y, E, w.y2, E, Ly.ple_post_norm, E, c.eps);
        k_add_streams<<<cdiv(R * E, 256), 256>>>(w.X, w.y2, E, R);
    }
    if (sel.empty()) return;
    const int nout = (int)sel.size();
    k_stream_sum<<<cdiv(R * E, 256), 256>>>(w.X, w.s, E, R, 1.f / HC);
    k_rmsnorm<<<dim3(R, 1), 256>>>(w.s, E, w.hn, E, m_.output_norm, E, c.eps);
    CK(cudaMemcpy(w.sel, sel.data(), nout * 4, cudaMemcpyHostToDevice));
    k_gather<<<nout, 256>>>(w.hn, w.sel, nout, E, w.hsel);
    mm(m_.tok_embd, w.hsel, E, nout, w.logits, c.n_vocab, w.xq, w.xs);
}

std::vector<float> Engine::forward(const std::vector<Row>& rows) {
    const int V = m_.cfg.n_vocab;
    std::vector<float> out;
    for (size_t b0 = 0; b0 < rows.size(); b0 += rmax_) {
        const int n = (int)std::min<size_t>(rmax_, rows.size() - b0);
        run_chunk(rows.data() + b0, n);
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

}  // namespace raon
