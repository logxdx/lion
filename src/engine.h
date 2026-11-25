// raon-engine: CUDA inference for Raon-1 nano (arch raon1nano), GGUF Q8_0 / Q6_K.
#pragma once
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

namespace raon {

struct Config {
    int n_layer = 24, n_embd = 1536, n_ff = 6144, n_vocab = 259840;
    int hc = 4;                       // mHC streams
    float hc_eps = 1e-5f, hc_mag = 1.f;
    int n_ple = 256;                  // per-layer embedding width
    float eps = 1e-6f, eps_latent = 1e-6f;
    // RDA (gated delta rule)
    int rda_heads = 16, rda_dim = 128, conv_k = 4, dt_rank = 128;
    float gate_lb = -5.f;
    // MLA
    int mla_heads = 32, q_lora = 512, kv_lora = 512, k_mla = 192, v_mla = 128, rope_dim = 64;
    std::vector<char> is_rda;         // per layer
    int hc_mix() const { return 2 * hc + (hc - 1) * (hc - 1); }   // 17
    int latent() const { return kv_lora + rope_dim; }             // 576 values per cached token
    int nope() const { return k_mla - rope_dim; }                 // 128
};

// A weight matrix of N output rows x K inputs on the device.
enum WType { W_Q8 = 0, W_Q6 = 1, W_F16 = 2 };
struct Mat {
    int type = -1, N = 0, K = 0;
    int8_t* q = nullptr;      // Q8: int8 [N][K]            | Q6: low nibbles [N][K/2] (GGUF block order)
    uint8_t* qh = nullptr;    // Q6: high bits [N][K/4]
    int8_t* sc = nullptr;     // Q6: sub-block scales [N][K/16]
    half* d = nullptr;        // Q8: [N][K/32]              | Q6: [N][K/256] | F16: the weights [N][K]
    bool valid() const { return type >= 0; }
};

struct Layer {
    bool rda = false;
    float *attn_norm = nullptr, *ffn_norm = nullptr;
    Mat hc_attn_fn, hc_ffn_fn;                      // F16 [17 x 6144]
    float *hc_attn_base, *hc_attn_chart, *hc_ffn_base, *hc_ffn_chart;
    // RDA
    Mat wqkv, ssm_f_a, ssm_f_b, rda_out;
    float *conv_w = nullptr, *dt_b = nullptr, *ssm_norm = nullptr;
    // MLA
    Mat q_a, q_b, kv_a, k_b, v_b, attn_gate, mla_out;
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    // FFN + per-layer input
    Mat ffn_up, ffn_down, ple_gate, ple_proj;
    float* ple_post_norm = nullptr;
    int rda_idx = -1, mla_idx = -1;                 // index among RDA / MLA layers
};

struct Model {
    Config cfg;
    Mat tok_embd;                                   // also the tied LM head
    Mat ple_model_proj;                             // F16 [6144 x 1536]
    float *ple_proj_norm = nullptr, *output_norm = nullptr;
    // per_layer_token_embd lives in pinned, device-mapped host memory, in GGUF block format
    const uint8_t* ple_table = nullptr;             // device-visible pointer
    void* ple_table_host = nullptr;
    int ple_type = 0;                               // G_Q8_0 or G_Q6_K
    size_t ple_row_bytes = 0;
    std::vector<Layer> layers;
    int n_rda = 0, n_mla = 0;
    size_t gpu_bytes = 0;
};

Model load_model(const std::string& path, bool verbose = true);
void free_model(Model& m);

// One row of a forward batch: a token at a position of a sequence slot.
struct Row { int slot, pos, tok; bool logits; };

class Engine {
public:
    Engine(const std::string& gguf, int n_slots, int max_ctx, bool verbose = true);
    ~Engine();
    const Config& cfg() const { return m_.cfg; }
    int n_slots() const { return n_slots_; }
    int max_ctx() const { return max_ctx_; }

    void reset(int slot);
    // Run a batch: rows of the same slot must be consecutive and in position order. Returns logits for rows with
    // logits=true, in row order, as [n_out][n_vocab] on the host.
    std::vector<float> forward(const std::vector<Row>& rows);
    // Feed a prompt into a slot (chunked); returns logits of its last token.
    std::vector<float> prefill(int slot, const std::vector<int>& toks, int pos0 = 0);
    // Benchmarking: fill a slot's latent cache [0, depth) and recurrent state with synthetic values.
    void bench_fill(int slot, int depth);
    size_t gpu_bytes() const;

private:
    void run_chunk(const Row* rows, int n);
    void launch_chunk(int R, int nseg, int nout, int nsplit);
    Model m_;
    int n_slots_, max_ctx_, rmax_;
    // per-slot state
    float* conv_state_ = nullptr;   // [slot][n_rda][3][6144]
    float* rda_state_ = nullptr;    // [slot][n_rda][16][128][128]
    int8_t* kv_q_ = nullptr;        // latent cache values [slot][n_mla][max_ctx][576]
    half* kv_s_ = nullptr;          // q8 scales            [slot][n_mla][max_ctx][18]
    size_t state_bytes_ = 0;
    // workspace
    struct Ws;
    Ws* ws_ = nullptr;
};

}  // namespace raon
