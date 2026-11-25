# raon-engine

A CUDA inference engine for **Raon-1 nano** (KRAFTON, GGUF arch `raon1nano`, Q8_0 / Q6_K), written for one
RTX 4050 Laptop GPU (sm_89, 6 GB, 233 GB/s).

**Goal:** long-context concurrent serving that llama.cpp cannot do on this card. At 150k-token contexts llama.cpp
decodes 46-49 tok/s in total, whatever the concurrency, and prefills at ~640 tok/s.

## Status (2026-11-25)

Built and working:

- the forward pass, multi-stream decode and long prefill;
- the tokenizer, chat template and GPU sampling;
- `raon-server`, an OpenAI-compatible server with continuous batching and prompt reuse across turns.

Not built: MTP speculative decoding, structured output (`response_format` JSON / grammars), images.

### Server (`raon-server`, Q6_K, 4 slots × 128k)

Measured with `tests/server_bench.py` and `tests/server_test.py`:

| | result |
|---|---|
| cold 117k-token prompt, first token | 52.9 s (2,217 tok/s) |
| 4 streams at ~117k context, all generating | 56.6-57.2 tok/s per stream, 227 tok/s total |
| a stream generating while another 117k prompt prefills (`--prefill-share 0.5`) | ~49 tok/s |
| next turn of a 9.8k-token conversation | 9,898 of 9,915 prompt tokens reused, prefill 31 ms |
| 4 streams, short context | 108-111 tok/s per stream, ~420 tok/s total |

### Decode, tok/s per stream

Synthetic caches (`--bench --slots N --depth D`):

| streams | Q8_0 128k | Q6_K 128k | Q8_0 150k | Q6_K 150k | llama.cpp Q8_0 150k |
|---|---|---|---|---|---|
| 1 × short context | 113.4 | 137.3 | | | 112 (Q6_K: 136) |
| 1 | 89.7 | 104.9 | 87.1 | 100.6 | – |
| 2 | 74.1 | 82.7 | 70.0 | 78.3 | 24.6 |
| 3 | 62.4 | 69.3 | 58.7 | 64.5 | 15.4 |
| 4 | 54.6 | 59.4 | 50.7 | 54.4 | 12.0 |
| 5 | 47.8 | 51.6 | 43.8 | 47.3 | – |
| 6 | **43.1** | **45.6** | 38.0 | **41.6** | does not fit |

6 × 150k fits in VRAM with both quantizations.

After the split mHC for small batches (`1d2664e`, `--steps 400`, Q6_K), measured against the previous build in
alternating runs:

- 4 × 128k: 16.53 ms per step, 60.5 tok/s per stream (was 16.80 ms);
- 4 × 1k: 8.45 ms (was 8.72 ms);
- 1 × 1k: 7.26 ms (was 7.55 ms).

Short `--steps` runs can include a slow phase of the GPU, so use ≥ 400 steps for A/B.

### Prefill, one stream

`build/raon -m M --bench --prefill N`, 1 slot. The chunk size is chosen automatically from free VRAM: 4,096 rows here,
1,312 MiB of workspace. Device memory used is 3,680 MiB (Q8_0) / 3,460 MiB (Q6_K) at 150k.

| | 2048 tokens | 8192 tokens | 150k prompt |
|---|---|---|---|
| Q8_0 | **8,501 tok/s** (llama.cpp 6,970) | **7,769** (5,830) | **56.9 s, 2,697 tok/s** (llama.cpp ~235 s, ~640) |
| Q6_K | 6,471 (llama.cpp 6,670) | **5,975** (5,620) | **63.2 s, 2,432 tok/s** |

With 4 × 128k slots resident (`raon-server --slots 4 --ctx 131072`, 4 checkpoints per slot), Q6_K still gets
4,096-row chunks: 5,056 of 5,822 MiB used.

### Correctness

Front end, each checked against KRAFTON llama.cpp:

- **Tokenizer:** 337/337 cases token-identical to `llama-tokenize` (`tests/tokenizer_check.py`): real documents,
  source code, NFD input, invalid UTF-8, specials, 300 random multi-script strings. A 137k-token prompt tokenizes
  in 68 ms.
- **Chat template:** 16/16 byte-identical to llama-server `/apply-template` (`tests/template_check.py`), incl. tools,
  tool calls, tool results, reasoning and `enable_thinking`.
- **Sampler:** 289/289 picks identical to a full-vocabulary reference of llama.cpp's chain (`tests/sampling_check.cu`).
- **Whole stack, greedy chat vs llama-server (CPU):** 4/12 identical (`tests/chat_parity.py`).
  - 5 differ only in formatting: llama.cpp keeps a trailing `\n` in `reasoning_content`; raon-server trims it, as the
    template does when the reasoning comes back.
  - 3 diverge at a near-tie (top-2 log-prob gap 0.04, 0.14, 0.52).

Engine:

`tests/parity.py` compares greedy decoding and top-k log-probs against KRAFTON llama.cpp (q8_0 KV), on short chat
prompts and a 9,773-token document.

- **Log-prob differences** stay within llama.cpp's own CUDA-vs-CPU spread (≤ 1.06 nats on strong candidates).
- **Greedy divergences** happen only at near-ties (top-2 gap 0.02-0.27). Q6_K matches 181/181 tokens.
- **The model amplifies tiny numeric differences.** A ±1e-6 relative perturbation (`RAON_PERTURB`) moves first-token
  log-probs by up to ~0.6 at 9.8k context. llama.cpp moves 0.15-0.30 just from changing `-ub`.

## Design

Source of truth for the model: KRAFTON's llama.cpp `src/models/raon1nano.cpp` and the ggml op definitions.

### Weights

- **Q8_0:** int8 + fp16 scale per 32.
- **Q6_K:** GGUF block order, split into low-nibble, high-bit, sub-scale and scale planes.
- **F16:** kept as F16.
- **Per-layer embedding table (1.3-1.7 GB):** stays in pinned host memory; one 5-6.5 KB row is read zero-copy per
  token.

### Latent KV cache

- int8 with one scale for the 512 latent values and one for the 64 pe values per row: 584 B/row.
- `-DKV_ROWSCALE=0` gives llama.cpp's q8_0 layout instead (612 B/row).

### Decode

One CUDA graph per batch size.

- **GEMVs:** dp4a, ~96 % of DRAM bandwidth.
- **MLA flash-decode:** all 32 heads per pass.
  - scores on int8 tensor cores (`m16n8k32`), int32 across the row;
  - PV on f16 tensor cores, accumulating in f16 per 32-key tile, then f32 across tiles;
  - a 2-stage `cp.async` pipeline;
  - keys split over blocks on the device.
- **RDA delta-rule scan:** a head is split over 4 blocks of independent value columns; the next token's inputs are
  prefetched.

### Prefill

**Chunks:** the largest multiple of 128 rows, up to 4,096, that fits the free VRAM with a 256 MB margin, like
llama.cpp's `-ub`. Bigger chunks re-expand the cached keys less often. 4,096 instead of 2,048 rows makes a 128k
prefill 8% faster.

**GEMMs on tensor cores:**

- **Q8_0:** int8 `m16n8k32`, 64-deep stages, ~37 TOPS.
- **Q6_K:** two int8 `m16n8k16` per 32, with sub-scales folded into float factors prefetched one k-step ahead; ~24 TOPS.
  - Each partial converts to float exactly with the magic-number trick. The mma adds the magic number through its
    accumulator input, which leaves 5 instructions per element.
  - The kernel is instruction-bound: each output element needs two exact per-16 sub-scale terms per 32-wide step.
- **F16:** `m16n8k16`, including grouped per-head.

**MLA in expanded form**, per 4096-key block:

1. **Expansion** (`k_expand8`), on int8 tensor cores:
   - reads the int8 latent cache directly;
   - `k_b`/`v_b` are quantized to int8 at load, one scale per output channel;
   - int32 accumulation over the 512 latent values.

   Its epilogue writes:
   - int8 K, one scale per key and head, with the pe part copied from the cache;
   - V transposed in e4m3, one scale per 128-key tile and dim.
2. **Q** in int8, with nope/pe scales.
3. **Flash attention**, causal, merged across key blocks:
   - QKᵀ on int8 `m16n8k32`;
   - P·V on FP8 `m16n8k32` with f16 accumulation per 64-key tile (2 × the f16 rate on this GPU), with P' = 448 P in
     e4m3;
   - V's keys are permuted within 16-key groups so that P needs no shuffles.

`RAON_PV16` switches P·V back to f16; `RAON_EXPAND16` switches the expansion to the earlier dequantize + f16 GEMM.

### Fusion

- **One mHC kernel per step:** PLE add with RMSNorm, post-mix, stream sum and the full pre step in one pass over the
  4 streams.
- **Producers emit int8 activations** for the next GEMM/GEMV: h, stream sum, RDA output norm and gate, MLA gate, PLE
  gate, q_a norm, SwiGLU.
- **RDA conv** is folded into the gate / L2-norm kernel.

### Tokenizer

Byte-level BPE read from the GGUF, ported from KRAFTON llama.cpp:

- longest-first special tokens;
- NFC normalization (tables generated from llama.cpp's `unicode-data.cpp` by `tools/gen_unicode.py`);
- the `raon` pre-tokenizer split (multi-word chunks, right-aligned digit groups);
- merge-rank BPE over token-id pairs.

### Sampling

On the GPU, per output row:

1. presence / frequency / repetition penalties and `logit_bias` are applied to the logits in place;
2. two passes produce the top 128 candidates and the log-normalizers at T = 1 and at the row's temperature.

The host then runs llama.cpp's chain on those ~1 KB: top-k → top-p → min-p on T = 1 probabilities, then the draw at
the temperature, with the caller's uniform number (seedable).

When the decision reaches past the 128 candidates (a flat distribution with top-p, or the draw falls in the tail),
the host copies the row and selects with `nth_element` only as far as needed, so the result is exact.

Both passes are radix selects: an exact 128th-largest value by 8-bit histograms, then a sort of only the final 128.
They cost 0.10-0.13 ms per step for 4 rows (`tests/sample_bench.cu`). At 4 × 128k, decode with sampling runs at
16.27 ms per step, 61.5 tok/s per stream.

### Server

`tools/raon_server.cpp`, built on cpp-httplib and nlohmann/json (MIT, in `third_party/`).

**Endpoints:**

- `/v1/chat/completions` and `/v1/completions`, streaming (SSE) or not, with:
  - `reasoning_content`;
  - Qwen3-coder XML tool calls parsed into OpenAI `tool_calls`, argument types taken from the tool schema;
  - `logprobs`, `stop`, `seed`, `logit_bias`, `max_tokens`;
  - `chat_template_kwargs.enable_thinking`;
  - `stream_options.include_usage`, plus `timings`.
- `/v1/models`, `/health`, `/slots`, `/props` (flags and VRAM breakdown), `/tokenize`, `/detokenize`,
  `/apply-template`.

**Scheduling:**

- One thread owns the GPU.
- Each iteration runs at most one prefill chunk (2,048 tokens), then one batched decode step over all generating
  requests.
- While others generate, prefill gets `--prefill-share` of the GPU time (default 0.5).
- A client disconnect cancels its request.

**Prompt reuse:**

- A slot keeps the tokens its state covers.
- It also keeps up to `--checkpoints` (default 4) snapshots of the recurrent state (~19 MB each). Each is taken
  where a chat prompt's generation prefix begins, which is where the next turn of the same conversation diverges.
- A new request takes the idle slot with the longest reusable prefix. Re-sent history, edited last turns and
  agent tool loops then prefill only the new tokens.
- The latent cache needs no snapshot: positions past the reuse point are rewritten.

**Defaults:** sampling defaults follow the model's `generation_config.json` (temperature 0.1, presence_penalty 1.5),
as vLLM applies them. Request fields and CLI flags override them.

### Web UI

`raon-server` serves llama.cpp's chat web UI at `/`, ported from liquid-engine:

- **The bundle:** a prebuilt release (`b12569`, MIT), pinned and SHA-256-checked.
- **Branding:** rebranded "raon" by `brand/rebrand.py`, which changes names, help texts, the logo, icons and splash
  screens, and renames every content-hashed file it changes.
- **Icons:** `brand/build_assets.py` renders them from `brand/raon-mark.svg` (it needs `rsvg-convert` and
  ImageMagick; the rendered files are committed in `brand/webui/`).

Server support for the UI, in llama-server's formats:

- `/props`, including `default_generation_settings`, `modalities` and `total_slots`;
- `prompt_progress` chunks while a prompt prefills (`return_progress`), and SSE pings (`sse_ping_interval`);
- `timings` in the last chunk (or every chunk with `timings_per_token`);
- `reasoning_format: none` (thinking inline);
- `n_predict: 0` (prefill only, used by the UI's "Pre-fill KV cache after response": the next turn then starts from
  the saved state);
- empty answers for what raon-server does not have: `/v1/catalog.json`, `/v1/streams/lookup`, and the `/models/*`
  router calls.

**Server tools** (ported from liquid-engine): `web_search`, `web_fetch`, and `pdf` (when `liteparse` is installed).

- They are served at `/tools` in llama-server's format, so the UI offers them to the model and runs them when it
  calls one (the UI asks for permission per call).
- Each call runs `node tools/webtools/tools.ts call <name>` in its own process group, with a timeout
  (`--tool-timeout`, default 300 s), and is cancelled when the client disconnects.
- `web_search` tries TinyFish, LangSearch and Exa when their API key is set (environment or keyring), then
  Wikipedia (no key).
- `--no-tools` turns them off (`/tools` answers 403, as llama-server without `--tools`).
- `--tools-dir` and `--node` point elsewhere.
- Their Node dependencies install with `make webtools`.

```
make webui                         # downloads the pinned bundle (once), verifies it, rebrands into build/webui
make webtools                      # node dependencies of the server tools (npm ci in tools/webtools)
build/raon-server -m model.gguf    # UI on http://127.0.0.1:8080/ ; --ui-dir DIR, --no-ui
```

## Build and run

```
make                                   # nvcc 13.x, sm_89: build/raon, build/raon-server, build/raon-tok
build/raon-server -m model.gguf --slots 4 --ctx 131072 --port 8080  # OpenAI API on http://127.0.0.1:8080/v1
build/raon -m model.gguf --tokens 259201,882,198 -n 32 --top 5     # greedy, JSON lines with top-k log-probs
build/raon -m model.gguf --bench --slots 6 --depth 153600          # decode benchmark
build/raon -m model.gguf --bench --prefill 153600                  # prefill benchmark
python3 tests/parity.py --server ~/llama.cpp/raon-src/build/bin/llama-server --model model.gguf \
    --raon build/raon -n 64 --ref ref.json --ref-kv q8_0 --long doc.txt
```

Environment knobs:

- `RAON_PROFILE=1`: per-kernel-group timing; it also names the group in which a CUDA error surfaces.
- `RAON_CHUNK`: overrides the prefill chunk size.
- A/B switches: `RAON_NOGRAPH`, `RAON_NOGEMM`, `RAON_NOFA`, `RAON_FA16`, `RAON_EXPAND16`, `RAON_PV16`, `RAON_MHC_SPLIT`, `RAON_NSPLIT`.
- Debugging: `RAON_DUMP`, `RAON_PERTURB`.

Compile switches: `ATTN_PV_F32ACC`, `FA_PV_F32ACC`, `KV_ROWSCALE`.

Checks of the front end and the server:

```
python3 tests/tokenizer_check.py --model model.gguf --ref ~/llama.cpp/raon-src/build/bin/llama-tokenize --fuzz 300
python3 tests/template_check.py --url http://127.0.0.1:8191 --render build/template_render   # vs llama-server
make tests && build/sampling_check model.gguf
python3 tests/server_test.py --url http://127.0.0.1:8080 --doc doc.txt
python3 tests/chat_parity.py --raon http://127.0.0.1:8080 --ref http://127.0.0.1:8191
python3 tests/server_bench.py --url http://127.0.0.1:8080 --doc doc.txt --clients 4 --tokens 120000 --gen 512
```

`build/raon --bench ... --sample [--temp T --top-p P]` times decode steps with GPU sampling instead of a logits copy.

`tests/ab.sh` runs interleaved A/B benchmarks and discards any round that overlaps another GPU process.
`tests/gemm_bench.cu`, `gemm_q6_check.cu`, `gemm_f16_check.cu` and `gemv_bench.cu` are kernel benches and checks.
