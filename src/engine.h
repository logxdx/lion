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
constexpr int MAX_SLOTS = 8;  // concurrent sequences (batched decode)

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
    int repeat_last_n = -1;     // -1: whole context (transformers semantics; 64 = llama.cpp default)
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
    float *conv_state = nullptr;        // [slot][n_embd][6]
    float *q_norm = nullptr, *k_norm = nullptr, *rope_ff = nullptr;
    void *kc = nullptr, *vc = nullptr;  // [slot][n_kv][cap][128] f16, or int8 with --kv q8
    half *ks = nullptr, *vs = nullptr;  // q8 KV: per-32 scales [slot][n_kv][cap][4]
    int cap = 0;
    size_t kv_slot_bytes = 0, kvs_slot_elems = 0;  // per-slot strides of kc/vc and ks/vs
};

struct Timing { double prefill_ms = 0, decode_ms = 0; int prefill_tokens = 0, decode_tokens = 0; };

class Engine {
public:
    // max_ctx: context per sequence; n_slots: concurrent sequences (1..MAX_SLOTS)
    // exact: debug reference mode (f32 activations, exactly dequantized weights; slow; 1 slot)
    // kv_q8: store the KV cache as q8_0 (int8 + f16 scale per 32) instead of f16
    Engine(const std::string& gguf_path, int max_ctx, bool verbose = true, bool exact = false, bool kv_q8 = false,
           int n_slots = 1);
    bool kv_q8() const { return kv_q8_; }
    ~Engine();

    const Config& config() const { return cfg_; }
    int n_past(int slot = 0) const { return n_past_[slot]; }
    int max_ctx() const { return max_ctx_; }
    int n_slots() const { return n_slots_; }
    size_t gpu_bytes() const { return arena_size_; }

    void reset(int slot = -1);  // clear KV / conv state of one slot (-1: all)
    // Feed prompt tokens into a slot (the last token stays pending until generation).
    void prefill(const std::vector<int>& tokens, int slot = 0);
    // Generate up to n tokens in slot 0; on_token is called per token (return false to stop). Stops at stop
    // tokens (returned to the callback but NOT consumed into the state).
    std::vector<int> generate(int n, const SamplingParams& sp, const std::vector<int>& stop,
                              bool (*on_token)(int, void*) = nullptr, void* ud = nullptr);
    // Batched decode of several slots at once (weights are streamed once per step for all of them).
    // on_token(slot, token, ud): return false to stop that slot.
    std::vector<std::vector<int>> generate_multi(const std::vector<int>& slots, int n, const SamplingParams& sp,
                                                 const std::vector<int>& stop,
                                                 bool (*on_token)(int, int, void*) = nullptr, void* ud = nullptr);
    // logits of the last processed token of batch row 0 (host copy), for testing
    std::vector<float> logits();

    // ---- step-level API (server scheduler); all calls on one thread ----
    // Consume prompt tokens into a slot (chunked internally); n_past += n. The pending token is untouched.
    void feed(int slot, const int* tokens, int n);
    void set_pending(int slot, int tok) { pending_[slot] = tok; ctrl_dirty_[slot] = true; }
    int pending(int slot) const { return pending_[slot]; }
    void set_sampling(int slot, const SamplingParams& sp, const std::vector<int>& stop);
    // One batched decode step: every listed slot consumes its pending token and samples the next one
    // (which becomes pending). done[i] = a stop token was sampled.
    void decode_step(const std::vector<int>& slots, std::vector<int>& toks, std::vector<char>& done);
    // Exact state of a slot after n_past consumed tokens, minus the global-attention KV: short-conv states and
    // the sliding-window KV rows still in reach. Restoring it requires the global KV rows [0, n_past) to be valid.
    struct Checkpoint {
        int n_past = -1;
        std::vector<uint8_t> conv, swa;
        size_t bytes() const { return conv.size() + swa.size(); }
    };
    void save_checkpoint(int slot, Checkpoint& c);
    void load_checkpoint(int slot, const Checkpoint& c);  // sets n_past; pending must be set separately
    // Global-attention KV rows [p0, p1) of a slot to/from host memory (kv_row_bytes() per token).
    size_t kv_row_bytes() const;
    void save_kv(int slot, int p0, int p1, void* host);
    void load_kv(int slot, int p0, int p1, const void* host);
    // token history (repeat penalty window) for positions [p0, p0+n)
    void write_history(int slot, int p0, const int* tokens, int n);
    void sync();
    // per-kernel timing of eager decode steps (stderr)
    void profile_decode(int steps);
    void profile_prefill(int n_tokens);
    Timing timing;
    bool use_batched_prefill = true;

private:
    Config cfg_;
    int max_ctx_ = 0, n_slots_ = 1;
    int n_past_[MAX_SLOTS] = {}, pending_[MAX_SLOTS];
    bool ctrl_dirty_[MAX_SLOTS] = {};  // host n_past/pending changed: upload before the next decode step
    std::vector<int> cur_slots_;       // slot table currently on the device
    bool kv_q8_ = false;
    std::vector<Layer> L_;
    QWeight emb_;                 // token_embd == output (tied)
    float* out_norm_ = nullptr;
    size_t arena_size_ = 0;

    struct Bufs;
    std::unique_ptr<Bufs> B_;
    cudaStream_t st_ = nullptr;
    cudaGraphExec_t g_decode_[MAX_SLOTS + 1] = {};  // by batch size
    cudaGraphExec_t g_step_nohead_ = nullptr;
    bool verbose_ = true;

    void* alloc(size_t bytes);
    void load(const gguf::File& f);
    void upload_qweight(const gguf::File& f, QWeight& w, const std::vector<std::string>& parts, int tile_in, int tile_out);
    float* upload_f32(const gguf::File& f, const std::string& name);
    void build_graphs();
    void enqueue_step(bool with_head, int nb, cudaStream_t s);
    void enqueue_sample(int nb, cudaStream_t s);
    void prefill_batched(int slot, const int* tokens, int n);
    void set_slots(const int* slots, int nb);
};

}  // namespace le
