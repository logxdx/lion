// liquid-engine: a single-model CUDA inference engine for LFM3-1.7B (Q8_0 / Q6_K GGUF) on sm_89.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gguf { class File; }

namespace le {

enum WType { W_Q8 = 0, W_Q6 = 1 };

// Repacked quantized matrix: N output rows x K inputs (see kernels for the exact layouts).
struct QWeight {
    int type = W_Q8, N = 0, K = 0;
    uint8_t* q = nullptr;   // Q8: int8 [N][K] (per-1024 chunk half-block interleave) | Q6: low nibbles [N][K/2]
    uint8_t* h = nullptr;   // Q6: high 2-bit planes [N][K/4]
    int8_t* sc = nullptr;   // Q6: 16-element sub-scales [N][K/16]
    half* d = nullptr;      // Q8: [N][K/32] | Q6: [N][K/256]
    size_t bytes() const;
};

struct Config {
    int n_layer = 24, n_embd = 2048, n_ff = 7680, n_head = 16, n_head_kv = 4, head_dim = 128;
    int n_vocab = 128000, swa_window = 256, conv_l = 7;
    float eps = 1e-5f, rope_base = 1e6f, rope_base_swa = 1e4f;
    std::vector<int> head_kv;   // per layer (0 = conv)
    std::vector<bool> swa;      // per layer
};

struct SamplingParams {
    float temperature = 0.1f;   // <= 0: greedy
    int top_k = 50;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;     // -1: whole context
    uint64_t seed = 0;
};

struct Layer {
    bool conv = false, swa = false;
    QWeight w_in;       // conv: in_proj (tiles of [b32 c32 x32]) | attn: q,k,v,gate rows
    QWeight w_out;      // conv out_proj | attn o_proj
    QWeight w_gu;       // ffn gate/up (tiles of [gate32 up32])
    QWeight w_down;
    float *attn_norm = nullptr, *ffn_norm = nullptr, *post_attn_norm = nullptr, *post_ffn_norm = nullptr;
    float *conv_w = nullptr;            // [n_embd][7]
    float *conv_state = nullptr;        // [n_embd][6]
    float *q_norm = nullptr, *k_norm = nullptr, *rope_ff = nullptr;
    half *kc = nullptr, *vc = nullptr;  // [n_kv][cap][128]
    int cap = 0;
};

struct Timing { double prefill_ms = 0, decode_ms = 0; int prefill_tokens = 0, decode_tokens = 0; };

class Engine {
public:
    // exact: debug reference mode (f32 activations, exactly dequantized weights; slow)
    Engine(const std::string& gguf_path, int max_ctx, bool verbose = true, bool exact = false);
    ~Engine();

    const Config& config() const { return cfg_; }
    int n_past() const { return n_past_; }
    int max_ctx() const { return max_ctx_; }

    void reset();  // clear KV / conv state
    // Feed tokens (prompt). Returns nothing; logits of the last token are ready for sampling.
    void prefill(const std::vector<int>& tokens);
    // Generate up to n tokens; calls on_token for each (return false to stop). Stops at stop tokens
    // (the stop token is returned to the callback but NOT consumed into the state).
    std::vector<int> generate(int n, const SamplingParams& sp, const std::vector<int>& stop,
                              bool (*on_token)(int, void*) = nullptr, void* ud = nullptr);
    // logits of the last processed token (host copy), for testing
    std::vector<float> logits();
    // per-kernel timing of eager decode steps (stderr)
    void profile_decode(int steps);
    Timing timing;
    bool use_batched_prefill = true;

private:
    Config cfg_;
    int max_ctx_ = 0, n_past_ = 0, pending_ = -1;
    std::vector<Layer> L_;
    QWeight emb_;                 // token_embd == output (tied)
    float* out_norm_ = nullptr;
    void* arena_ = nullptr;
    size_t arena_off_ = 0, arena_size_ = 0;

    struct Bufs;
    std::unique_ptr<Bufs> B_;
    cudaStream_t st_ = nullptr;
    cudaGraphExec_t g_decode_ = nullptr, g_step_nohead_ = nullptr;
    bool verbose_ = true;

    void* alloc(size_t bytes);
    void load(const gguf::File& f);
    void upload_qweight(const gguf::File& f, QWeight& w, const std::vector<std::string>& parts, int tile_in, int tile_out);
    float* upload_f32(const gguf::File& f, const std::string& name);
    void build_graphs();
    void enqueue_step(bool with_head, cudaStream_t s);
    void enqueue_sample(cudaStream_t s);
    void prefill_batched(const int* tokens, int n);
    void set_sampling(const SamplingParams& sp, const std::vector<int>& stop);
};

}  // namespace le
