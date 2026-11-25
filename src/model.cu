// Load a raon1nano GGUF: config, weights repacked for the GEMV kernels, per-layer embedding table in host memory.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "engine.h"
#include "gguf.h"
#include "util.cuh"

namespace raon {

static void* dupload(const void* src, size_t n, size_t& total) {
    void* p = nullptr;
    CK(cudaMalloc(&p, n));
    CK(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice));
    total += n;
    return p;
}

static float* upload_f32(const GGUF& g, const std::string& name, size_t& total, int64_t expect = -1) {
    const GTensor& t = g.t(name);
    if (t.type != G_F32) throw std::runtime_error(name + ": expected F32");
    if (expect >= 0 && t.nelem() != expect) throw std::runtime_error(name + ": unexpected size");
    return (float*)dupload(t.data, t.nbytes, total);
}

static Mat upload_mat(const GGUF& g, const std::string& name, size_t& total, int64_t K = -1, int64_t N = -1) {
    const GTensor& t = g.t(name);
    Mat m;
    m.K = (int)t.ne[0];
    m.N = (int)(t.nelem() / t.ne[0]);
    if ((K >= 0 && m.K != K) || (N >= 0 && m.N != N))
        throw std::runtime_error(name + ": shape " + std::to_string(m.K) + "x" + std::to_string(m.N) + " unexpected");
    const size_t NK = (size_t)m.N * m.K;
    if (t.type == G_Q8_0) {
        m.type = W_Q8;
        std::vector<int8_t> q(NK);
        std::vector<half> d(NK / 32);
        const uint8_t* b = t.data;
        for (size_t i = 0; i < NK / 32; i++, b += 34) {
            memcpy(&d[i], b, 2);
            memcpy(&q[i * 32], b + 2, 32);
        }
        m.q = (int8_t*)dupload(q.data(), NK, total);
        m.d = (half*)dupload(d.data(), d.size() * 2, total);
    } else if (t.type == G_Q6_K) {
        if (m.K % 256) throw std::runtime_error(name + ": Q6_K row not a multiple of 256");
        m.type = W_Q6;
        std::vector<uint8_t> ql(NK / 2), qh(NK / 4);
        std::vector<int8_t> sc(NK / 16);
        std::vector<half> d(NK / 256);
        const uint8_t* b = t.data;
        for (size_t i = 0; i < NK / 256; i++, b += 210) {  // block_q6_K: ql[128] qh[64] scales[16] d
            memcpy(&ql[i * 128], b, 128);
            memcpy(&qh[i * 64], b + 128, 64);
            memcpy(&sc[i * 16], b + 192, 16);
            memcpy(&d[i], b + 208, 2);
        }
        m.q = (int8_t*)dupload(ql.data(), ql.size(), total);
        m.qh = (uint8_t*)dupload(qh.data(), qh.size(), total);
        m.sc = (int8_t*)dupload(sc.data(), sc.size(), total);
        m.d = (half*)dupload(d.data(), d.size() * 2, total);
    } else if (t.type == G_F16) {
        m.type = W_F16;
        m.d = (half*)dupload(t.data, NK * 2, total);
    } else {
        throw std::runtime_error(name + ": unsupported type " + std::to_string(t.type));
    }
    return m;
}

Model load_model(const std::string& path, bool verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    GGUF g(path);
    if (g.s("general.architecture") != "raon1nano") throw std::runtime_error("not a raon1nano model");
    Model m;
    Config& c = m.cfg;
    const std::string a = "raon1nano.";
    const int blocks = (int)g.u(a + "block_count");
    const int nextn = g.has(a + "nextn_predict_layers") ? (int)g.u(a + "nextn_predict_layers") : 0;
    c.n_layer = blocks - nextn;
    c.n_embd = (int)g.u(a + "embedding_length");
    c.n_ff = (int)g.u(a + "feed_forward_length");
    c.eps = (float)g.f(a + "attention.layer_norm_rms_epsilon");
    c.eps_latent = (float)g.f(a + "attention.latent_norm_rms_epsilon");
    c.conv_k = (int)g.u(a + "ssm.conv_kernel");
    c.dt_rank = (int)g.u(a + "ssm.time_step_rank");
    c.rda_dim = (int)g.u(a + "kda.head_dim");
    c.gate_lb = (float)g.f(a + "kda.gate_lower_bound");
    c.q_lora = (int)g.u(a + "attention.q_lora_rank");
    c.kv_lora = (int)g.u(a + "attention.kv_lora_rank");
    c.k_mla = (int)g.u(a + "attention.key_length_mla");
    c.v_mla = (int)g.u(a + "attention.value_length_mla");
    c.rope_dim = (int)g.u(a + "rope.dimension_count");
    c.hc = (int)g.u(a + "hyper_connection.count");
    c.hc_eps = (float)g.f(a + "hyper_connection.epsilon");
    c.hc_mag = (float)g.f(a + "hyper_connection.magnitude");
    c.n_ple = (int)g.u(a + "embedding_length_per_layer_input");
    const auto& hk = g.get(a + "attention.head_count_kv").arr;
    const auto& hh = g.get(a + "attention.head_count").arr;
    c.is_rda.resize(c.n_layer);
    for (int i = 0; i < c.n_layer; i++) {
        c.is_rda[i] = hk[i].u == 0;
        if (c.is_rda[i]) c.rda_heads = (int)hh[i].u; else c.mla_heads = (int)hh[i].u;
    }
    c.n_vocab = (int)g.t("token_embd.weight").ne[1];
    if (c.n_embd != 1536 || c.hc != 4 || c.conv_k != 4 || c.rda_dim != 128 || c.rda_heads != 16 || c.mla_heads != 32 ||
        c.kv_lora != 512 || c.rope_dim != 64 || c.k_mla != 192 || c.v_mla != 128 || c.n_ple != 256 || c.dt_rank != 128 ||
        c.n_ff != 6144 || c.q_lora != 512 || c.hc_mag != 1.f)
        throw std::runtime_error("unexpected Raon-1 nano hyperparameters (kernels are specialised for the released model)");

    size_t& tot = m.gpu_bytes;
    const int E = c.n_embd, HCE = c.hc * E, MIX = c.hc_mix();
    m.tok_embd = upload_mat(g, "token_embd.weight", tot, E, c.n_vocab);
    m.output_norm = upload_f32(g, "output_norm.weight", tot, E);
    m.ple_model_proj = upload_mat(g, "per_layer_model_proj.weight", tot, E, c.n_ple * c.n_layer);
    m.ple_proj_norm = upload_f32(g, "per_layer_proj_norm.weight", tot, c.n_ple);
    {
        const GTensor& t = g.t("per_layer_token_embd.weight");
        if (t.ne[0] != (int64_t)c.n_ple * c.n_layer) throw std::runtime_error("per_layer_token_embd: unexpected width");
        if (t.type != G_Q8_0 && t.type != G_Q6_K) throw std::runtime_error("per_layer_token_embd: unsupported type");
        m.ple_type = (int)t.type;
        m.ple_row_bytes = gtype_bytes(t.type, t.ne[0]);
        CK(cudaHostAlloc(&m.ple_table_host, t.nbytes, cudaHostAllocMapped));
        memcpy(m.ple_table_host, t.data, t.nbytes);
        void* dp = nullptr;
        CK(cudaHostGetDevicePointer(&dp, m.ple_table_host, 0));
        m.ple_table = (const uint8_t*)dp;
    }
    m.layers.resize(c.n_layer);
    for (int il = 0; il < c.n_layer; il++) {
        Layer& L = m.layers[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        L.rda = c.is_rda[il];
        L.attn_norm = upload_f32(g, p + "attn_norm.weight", tot, E);
        L.ffn_norm = upload_f32(g, p + "ffn_norm.weight", tot, E);
        L.hc_attn_fn = upload_mat(g, p + "hc_attn_fn.weight", tot, HCE, MIX);
        L.hc_ffn_fn = upload_mat(g, p + "hc_ffn_fn.weight", tot, HCE, MIX);
        L.hc_attn_base = upload_f32(g, p + "hc_attn_base.weight", tot, MIX);
        L.hc_ffn_base = upload_f32(g, p + "hc_ffn_base.weight", tot, MIX);
        L.hc_attn_chart = upload_f32(g, p + "hc_attn_chart.weight", tot, MIX * 16 + 32);
        L.hc_ffn_chart = upload_f32(g, p + "hc_ffn_chart.weight", tot, MIX * 16 + 32);
        const int di = c.rda_heads * c.rda_dim;
        if (L.rda) {
            L.rda_idx = m.n_rda++;
            L.wqkv = upload_mat(g, p + "attn_qkv.weight", tot, E, 4 * di);
            L.conv_w = upload_f32(g, p + "ssm_conv1d.weight", tot, (int64_t)c.conv_k * 3 * di);
            L.ssm_f_a = upload_mat(g, p + "ssm_f_a.weight", tot, E, c.dt_rank + 2 * c.rda_heads);
            L.ssm_f_b = upload_mat(g, p + "ssm_f_b.weight", tot, c.dt_rank, di);
            L.dt_b = upload_f32(g, p + "ssm_dt.bias", tot, di);
            L.ssm_norm = upload_f32(g, p + "ssm_norm.weight", tot, c.rda_dim);
            L.rda_out = upload_mat(g, p + "attn_output.weight", tot, di, E);
        } else {
            L.mla_idx = m.n_mla++;
            const int H = c.mla_heads;
            L.q_a = upload_mat(g, p + "attn_q_a.weight", tot, E, c.q_lora);
            L.q_a_norm = upload_f32(g, p + "attn_q_a_norm.weight", tot, c.q_lora);
            L.q_b = upload_mat(g, p + "attn_q_b.weight", tot, c.q_lora, H * c.k_mla);
            L.kv_a = upload_mat(g, p + "attn_kv_a_mqa.weight", tot, E, c.latent());
            L.kv_a_norm = upload_f32(g, p + "attn_kv_a_norm.weight", tot, c.kv_lora);
            L.k_b = upload_mat(g, p + "attn_k_b.weight", tot, c.nope(), (int64_t)c.kv_lora * H);
            L.v_b = upload_mat(g, p + "attn_v_b.weight", tot, c.kv_lora, (int64_t)c.v_mla * H);
            {  // expansion weights for prefill: rows h*128+i = k_b[h][:][i] (transposed), then rows of v_b as stored
                const GTensor& kb = g.t(p + "attn_k_b.weight");
                const GTensor& vb = g.t(p + "attn_v_b.weight");
                if (kb.type != G_F16 || vb.type != G_F16) throw std::runtime_error("attn_k_b/v_b: expected F16");
                const int R = c.kv_lora, NP = c.nope(), DV = c.v_mla;
                std::vector<half> wx((size_t)2 * H * NP * R);
                const half* kbd = (const half*)kb.data;  // [h][r][i]
                for (int h = 0; h < H; h++)
                    for (int i = 0; i < NP; i++)
                        for (int r = 0; r < R; r++) wx[((size_t)h * NP + i) * R + r] = kbd[((size_t)h * R + r) * NP + i];
                memcpy(&wx[(size_t)H * NP * R], vb.data, (size_t)H * DV * R * 2);  // [h][d][r]
                L.wexp = (half*)dupload(wx.data(), wx.size() * 2, tot);
            }
            L.attn_gate = upload_mat(g, p + "attn_gate.weight", tot, E, H * c.v_mla);
            L.mla_out = upload_mat(g, p + "attn_output.weight", tot, H * c.v_mla, E);
        }
        L.ffn_up = upload_mat(g, p + "ffn_up.weight", tot, E, 2 * c.n_ff);
        L.ffn_down = upload_mat(g, p + "ffn_down.weight", tot, c.n_ff, E);
        L.ple_gate = upload_mat(g, p + "inp_gate.weight", tot, E, c.n_ple);
        L.ple_proj = upload_mat(g, p + "proj.weight", tot, c.n_ple, E);
        L.ple_post_norm = upload_f32(g, p + "post_norm.weight", tot, E);
    }
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "raon: loaded %s: %d layers (%d RDA, %d MLA), vocab %d, %.0f MiB on GPU, PLE table %.0f MiB "
                        "pinned host, %.1f s\n", path.c_str(), c.n_layer, m.n_rda, m.n_mla, c.n_vocab,
                tot / 1048576.0, (double)m.ple_row_bytes * c.n_vocab / 1048576.0, s);
    }
    return m;
}

static void free_mat(Mat& m) {
    cudaFree(m.q); cudaFree(m.qh); cudaFree(m.sc); cudaFree(m.d);
    m = Mat();
}

void free_model(Model& m) {
    free_mat(m.tok_embd);
    free_mat(m.ple_model_proj);
    cudaFree(m.ple_proj_norm); cudaFree(m.output_norm);
    if (m.ple_table_host) cudaFreeHost(m.ple_table_host);
    for (auto& L : m.layers) {
        for (Mat* x : {&L.hc_attn_fn, &L.hc_ffn_fn, &L.wqkv, &L.ssm_f_a, &L.ssm_f_b, &L.rda_out, &L.q_a, &L.q_b, &L.kv_a,
                       &L.k_b, &L.v_b, &L.attn_gate, &L.mla_out, &L.ffn_up, &L.ffn_down, &L.ple_gate, &L.ple_proj})
            free_mat(*x);
        cudaFree(L.wexp);
        for (float* p : {L.attn_norm, L.ffn_norm, L.hc_attn_base, L.hc_attn_chart, L.hc_ffn_base, L.hc_ffn_chart,
                         L.conv_w, L.dt_b, L.ssm_norm, L.q_a_norm, L.kv_a_norm, L.ple_post_norm})
            cudaFree(p);
    }
    m.layers.clear();
}

}  // namespace raon
