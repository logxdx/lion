# liquid-engine

A from-scratch CUDA inference engine specialised for **LiquidAI LFM3-1.7B** (`Q8_0` and `Q6_K` GGUF)
on this machine: **RTX 4050 Laptop (AD107, sm_89, 6 GB, 96-bit GDDR6)** + i5-13420H.
No ggml / llama.cpp / cuBLAS at runtime: GGUF loader, BPE tokenizer, kernels, sampler and CLI are all here.
The only dependency is the CUDA runtime. Up to 8 concurrent sequences (batched decode), q8_0 or f16 KV cache.

All numbers below: same GPU, same GGUF files, one engine at a time. Reference: llama.cpp b12569
(`~/llama.cpp/cuda/current/llama-server`, `-fa on -ngl 99`). Raw outputs are in `bench/results-*`.

## Single stream

`ppN` = prompt tokens/s for an N-token prompt, `tg128 @N` = decode tokens/s at context N (`bench/compare.sh`).

| model | engine | KV | pp512 | pp2048 | pp8192 | tg128 @512 | tg128 @2048 | tg128 @8192 |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | llama.cpp | f16 | 8809 | 10269 | 10365 | 110.6 | 109.2 | 105.7 |
| Q8_0 | **liquid** | f16 | **11068** | **11216** | **10476** | **120.8** | **119.8** | **115.1** |
| Q6_K | llama.cpp | f16 | 7930 | 8929 | **8958** | 138.0 | 136.2 | 130.2 |
| Q6_K | **liquid** | f16 | **9093** | **9306** | 8793 | **153.9** | **152.3** | **144.8** |

Long context (Q8_0 weights, `KV=f16|q8 SIZES="65536 130048" bench/compare.sh`, liquid with pipelined flash attention):

| ctx | KV | prefill liquid | prefill llama.cpp | decode liquid | decode llama.cpp |
|---|---|---|---|---|---|
| 64k | f16 | 6896 | 6939 | **85.1** | 69.2 |
| 64k | q8 | **6725** | 6616 | **98.7** | 89.2 |
| 128k | f16 | 4939 | 5015 | **65.8** | 48.0 |
| 128k | q8 | 4699 | 4718 | **83.4** | 75.7 |

Long-context prefill is at parity (±2 %); decode is 10-37 % faster.

## Concurrency (batched decode, q8_0 KV)

N sequences decode together; every step streams the weights once for all of them (`--bench --parallel N`).
llama.cpp: `llama-server -np N -ctk q8_0 -ctv q8_0`, same procedure (all slots prefilled, then N concurrent
decode requests on the cached prompts; `bench/llama_parallel.sh`). Aggregate decode tokens/s, 128 tokens/stream:

| model | streams x ctx | llama.cpp | **liquid** | liquid per stream | liquid GPU MiB |
|---|---|---|---|---|---|
| Q8_0 | 1 x 512 | 110.3 | **121.9** | 121.9 | |
| Q8_0 | 2 x 512 | 208.2 | **240.5** | 120.2 | 1847 |
| Q8_0 | 3 x 512 | 296.7 | **355.1** | 118.4 | 1860 |
| Q8_0 | 4 x 512 | 380.0 | **464.7** (+22 %) | 116.2 | 1873 |
| Q8_0 | 4 x 32k | 244.7 | **319.2** (+30 %) | 79.8 | 2678 |
| Q8_0 | 4 x 100k | 142.8 | **196.1** (+37 %) | 49.0 | 4356 |
| Q6_K | 4 x 512 | 447.2 | **563.3** (+26 %) | 140.8 | 1474 |
| Q6_K | 4 x 100k | – | **211.4** | 52.9 | 3956 |

Per-slot prefill at 4 x 100k: liquid 5471 t/s, llama.cpp 5444 t/s. (The 1-3 stream llama.cpp rows at 512 tokens come
from a first harness that overlapped prefill with decode; at 512-token prompts that is negligible: 384.0 vs 380.0
for 4 streams with the fixed harness.)

At long context every stream must read its own KV cache each step, so the step is bandwidth-bound by
weights + all KV: 4 x 100k reads 4.4 GB per step, roofline 211 t/s at 233 GB/s. Batched streams produce
bit-identical tokens to running each alone (`--test-multi N`). Memory: 4 x 100k with q8 KV = 4.4 GB on the 6 GB card
(f16 KV would not fit).

## KV cache: q8_0 vs f16 (default: q8)

Measured, not assumed (`bench/kv_e2e.sh`, `bench/kv_ab.sh`, `tests/kv_compare.py`), Q8_0 weights, single stream:

| ctx | prefill f16 | prefill q8 | decode f16 | decode q8 |
|---|---|---|---|---|
| 2k | 11209 | 11220 | 120.4 | 121.2 (+0.7 %) |
| 8k | 10477 | 10478 | 115.7 | 118.5 (+2.4 %) |
| 32k | 8596 | 8466 | 100.0 | 108.7 (+8.7 %) |
| 64k | 6896 | 6725 | 85.1 | 98.7 (+16 %) |
| 128k | 4939 | 4699 | 65.8 | 83.4 (+27 %) |

Decode attention is DRAM-bound from ~2k context on (per-phase `%globaltimer` traces: the K and V phases take
exactly the time to stream their bytes), so halving the KV bytes pays off more the longer the context. Prefill
attention is tensor-core-bound, so q8 costs the dequantisation (0-5 %). Accuracy vs the exact f32 reference
(23 long prompts, mean max |Δp| of the top-5): f16 KV 0.0064, q8 KV 0.0105, argmax agreement 23/23 for both.
`--kv f16` remains available.

## Accuracy

Activations are int8 per 32 elements (the same scheme as llama.cpp's MMVQ/MMQ); weights are used exactly as
stored. `--exact` runs a reference path with f32 activations, exactly dequantised weights and an f16 KV cache;
engines are scored against it (`tests/drift_stats.py`, 30 random text snippets, Q6_K, top-5 probabilities):

| vs exact f32 reference | mean max\|Δp\| | p90 | argmax agreement |
|---|---|---|---|
| liquid (f16 KV) | **0.0099** | **0.023** | **29/30** |
| llama.cpp (f16 KV) | 0.0138 | 0.030 | 28/30 |

Tokenization matches llama.cpp id-for-id (special tokens, CJK, whitespace runs, digits).

## Build & run

```sh
make                       # needs /opt/cuda (nvcc 13.x), builds build/liquid for sm_89
M=~/llama.cpp/models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf

build/liquid -m $M -p "Explain RoPE in two paragraphs."          # chat template, thinking on
build/liquid -m $M -p "..." --no-think -n 512                    # no <think> block
build/liquid -m $M -i -s "You are a terse assistant."            # interactive chat (/reset, /exit)
build/liquid -m $M --raw -p "def fib(n):" --greedy -n 64         # raw completion
build/liquid -m $M -c 131072 -f long_prompt.txt                  # long context (q8 KV by default)
build/liquid -m $M --bench --pp 2049 --tg 128                    # pp2048 / tg128 benchmark
build/liquid -m $M --bench --parallel 4 --pp 32769 -c 33024      # 4 concurrent streams x 32k
build/liquid -m $M --profile --pp 8192                           # per-kernel prefill + decode timing
```

Sampling defaults follow the model card / GGUF (`temp 0.1, top-k 50, top-p 1.0, min-p 0, repeat-penalty 1.1`
over the last 64 tokens; `--repeat-last-n -1` = whole context). Options: `--temp --top-k --top-p --min-p
--repeat-penalty --repeat-last-n --seed --greedy`, `-c` context per sequence (default 32768), `-f FILE`,
`--kv q8|f16`. The library API (`src/engine.h`) exposes per-slot `prefill(tokens, slot)` and
`generate_multi(slots, ...)` for concurrent sequences.

## OpenAI-compatible server (`build/liquid-server`)

```sh
./serve.sh                                   # 4 slots x 100k context, q8 KV, RAM + disk prompt cache, port 8080
build/liquid-server -m $M -np 4 -c 100000 --port 8080 --cache-ram-mib 4096 --cache-disk-mib 16384
curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Endpoints: `POST /v1/chat/completions` (stream + non-stream, `tools` -> `tool_calls`, `reasoning_content`,
`chat_template_kwargs.enable_thinking/preserve_thinking`, `stop`, `seed`, `max_tokens`/`max_completion_tokens`,
`temperature/top_p/top_k/min_p/repeat_penalty/repeat_last_n`, `stream_options.include_usage`, `ignore_eos`),
`POST /v1/completions` (text or token ids), `GET /v1/models`, `GET /health`, `GET /slots` (slot + cache state),
`POST /tokenize`. Responses carry `usage.prompt_tokens_details.cached_tokens` and llama.cpp-style `timings`
(queue, prefill, restore, decode, cache tier). Point an agent at `http://127.0.0.1:8080/v1`, model `lfm3`.

How it works:
- **HTTP**: cpp-httplib + nlohmann/json, vendored header-only, no runtime dependencies. Chosen by measurement on this
  box (`bench` in the task log): 381k small req/s, 6.8 M SSE events/s, 0.48 ms per 476 KB (100k-token) POST,
  1.8 ms JSON parse; the GPU needs ~1k token events/s, so the HTTP layer has >1000x headroom. Bounded worker pool,
  read/write timeouts, 256 MB body cap, client disconnect -> generation cancelled, SIGINT/SIGTERM -> graceful stop.
- **Chat template**: C++ port of the GGUF jinja template (tools, tool calls/responses, thinking keep/strip), byte-
  identical to llama-server's `/apply-template` on the test conversations (`tests/compare_template.py`); output is
  parsed into reasoning / content / tool calls (typed by each tool's JSON schema), streaming-safe.
- **Scheduler** (one GPU thread): FIFO queue (`--max-queue`, then 503) -> free slot with the longest reusable prefix
  -> prompt fed in chunks; while other slots decode, one decode step runs after every `--busy-chunk` prompt tokens,
  so a long new prompt no longer freezes running streams. Finished sequences leave the batch at once.
- **Prefix reuse**: LFM3's conv layers cannot be rewound, so every slot keeps state checkpoints (conv state + the
  256-token sliding-window KV, ~1.9 MB) at the end of each prompt and every 16k tokens; a new request resumes from
  the newest checkpoint inside its shared prefix. All restore paths are bit-identical (`tests/test_engine_api.cpp`).
- **RAM / disk tiers** (`--cache-ram-mib`, `--cache-disk-mib`, `--cache-dir`): a slot's conversation is moved down
  a tier before the slot is reused for something else (pinned RAM, LRU), RAM spills to disk on a background
  thread, files are written atomically, keyed by model + KV geometry, and reloaded at startup; on shutdown the
  RAM tier is flushed to disk.

Measured (Q8_0, q8 KV, same GPU; llama.cpp = llama-server with the `[lfm3]` preset settings, same cache budgets):

| scenario | llama-server | liquid-server |
|---|---|---|
| 4 concurrent streams (512-token prompts, 256 tokens each), tok/s overall | 338 | **401** |
| 8 requests on 4 slots, tok/s overall | 336 | **422** |
| new ~24k-token prompt while 3 streams decode: its TTFT | 3194 ms | **3127 ms** |
| ... the 3 running streams meanwhile (total tok/s, worst gap) | 16.9, 288 ms | **25.9, 146 ms** |
| agent trace: 6 agents x 8 turns on 4 slots (3k system + 1.5k tool result per turn): wall | 24.8 s | **22.0 s** |
| ... TTFT p50 / p90 | 1577 / 2140 ms | **1238 / 1603 ms** |

Cache tiers for a 48k-token conversation (`bench/cache_tiers.py`): cold prefill 6382 ms; next turn in the same
slot 27 ms; after eviction to RAM 26 ms restore; after a server restart 82 ms restore from disk (warm page cache;
a cold read of the 318 MB entry at the NVMe's measured 1.4 GB/s adds ~220 ms).

## Design

**Weights** are repacked on the GPU at load time (lossless, bit-identical values):
- `Q8_0` -> int8 rows with the two 16-byte halves of every 32-group interleaved per 1024-element chunk, so lane *i*
  of a warp reads group *i* with two fully coalesced 16-byte loads; f16 scales in a separate array.
- `Q6_K` -> 4-bit plane (byte *j* = w[j] | w[j+16]<<4), 2-bit plane laid out so `(hi >> 2m) & 0x03030303` yields
  4 consecutive weights, int8 sub-scales, f16 super-block scale. One AND/shift/OR turns 4 weights into a `dp4a`
  operand (decode) or an int8 MMA fragment (prefill).
- Fused matrices: q/k/v/gate concatenated; conv `in_proj` rows regrouped into tiles of [b32 c32 x32]; FFN gate/up
  interleaved in [gate32 up32] tiles, so one CTA owns everything its epilogue needs.

**Decode** (one CUDA graph per batch size, ~110 kernels per step, host stays up to 3 steps ahead):
- `dp4a` GEMVs batched over up to 8 sequences (each weight chunk is loaded once and dotted with every sequence's
  activations); the prologue fuses residual add + sandwich post-norm + pre-norm + int8 quantisation, epilogues fuse
  the short conv (with its rolling per-slot state), SwiGLU and int8 re-quantisation of the next input.
- Occupancy-sized persistent grids (balanced tiles, single wave); each CTA prefetches the next matrix into L2.
- Attention: QK-RMSNorm + NEOX RoPE (partial rotary on global layers via `rope_freqs`), KV write (f16 or q8_0),
  split-K over context with a last-CTA combine, per-head sigmoid output gate; the 4 query heads of a KV head share
  each K row, the V phase is split across warps (each V row read once for all heads); sliding-window layers use a
  per-slot ring buffer.
- Sampling on the GPU (penalty, top-k/top-p/min-p/temperature in llama.cpp's order); token, position and stop
  flag live in device memory per slot, so steps chain without host round trips; finished sequences leave the batch.

**Prefill** (chunks of 1024 tokens, per slot):
- int8 tensor-core GEMM (`mma.sync m16n8k32`, `k16` pairs for Q6_K sub-scales), 128x128x128 tiles, cp.async
  double buffering, swizzled smem + ldmatrix, CTA order that keeps each weight tile hot in L2, exact per-group
  scaling with an FP-pipe int->float trick; Q6_K is unpacked to int8 once per CTA. Measured ceiling on this GPU:
  96 TOPS int8 / 24 TFLOPS f16 with f32 accumulate (`bench/mma_peak.cu`).
- Short conv as a parallel 7-tap FIR over the chunk (state from the previous chunk).
- Tensor-core flash attention (causal + sliding window, 4 query heads share each K/V tile, f32 scores, f16 per-tile
  P·V partial sums), double-buffered K/V tiles (cp.async for f16; register prefetch + `prmt` dequantisation for
  q8, bit-identical to the f32 conversion, `tests/dequant_q8_test.cu`); ~28 TFLOPS useful at 32k, bound by the
  f32-accumulate QK^T MMAs.

## Tests

Without a server:
- `tests/multiturn.sh build/liquid <model>` — incremental multi-turn chat must equal a fresh prefill
- `build/liquid -m <model> --test-multi 4` — 4 prompts decoded in one batch == each decoded alone
- `tests/kv_compare.py build/liquid <model> [files]` — f16 vs q8 KV vs exact on long prompts
- `tests/split_invariance.sh` — decode attention invariant to how the context is split (exact mode)
- `nvcc -O3 -arch=sm_89 tests/dequant_q8_test.cu` — bit-exactness of the q8 dequantisation trick

With a reference server (`bench/ref_server.sh ~/llama.cpp/cuda/current <model> 18081`):
- `tests/compare_llama.py`, `tests/drift_stats.py`, `tests/greedy_match.py` (see their headers)

Profiling: `--profile` (per-kernel time + achieved GB/s or TFLOPS), `-DATT_TRACE` builds (`%globaltimer` per
attention phase), `bench/attn_sweep.sh`, `bench/kv_ab.sh`, `LIQUID_OCC=1` (attention occupancy).

## Limitations

- Specialised: LFM3-1.7B dimensions are checked at load and hard-wired (2048/7680/16 heads/128 dims, conv 7);
  only `Q8_0` and `Q6_K` weights. Other LFM3 sizes or quant types are rejected.
- Server: up to 8 slots (compile-time `MAX_SLOTS`); one model per process; `n > 1`, logprobs and
  presence/frequency penalties are not implemented; tool-call arguments are streamed as one delta per call.
  Prompt chunks are interleaved with decode steps rather than fused into one batch (the running streams slow
  to ~8 tok/s each during a long prefill instead of stopping).
- A disk-tier restore runs on the GPU thread (~0.1-0.3 s for a 48k-token conversation); other streams pause
  for that time. Reuse needs a checkpoint inside the shared prefix: when an agent's earlier reply re-tokenizes
  differently, the tokens after the last checkpoint are prefilled again (llama.cpp behaves the same).
- Memory: KV per token 6.5 KiB (q8) / 12 KiB (f16) per sequence. Measured: 4 x 100k with q8 KV = 4.4 GB total
  (Q8_0 weights). `-c 262144` (the model's full context) allocates and runs: 3.5 GB with q8 KV, 4.9 GB with f16;
  long prompts were benchmarked up to 130k tokens.
- Sampling considers candidates within `20·temperature` logits of the maximum (exact up to ~e^-20 tail mass).
- GPU only; the CPU path was not built (the 4050's 233 GB/s is ~4x what this laptop's RAM delivers).
