# raon-engine

A CUDA inference engine for **Raon-1 nano** (KRAFTON, GGUF arch `raon1nano`, Q8_0 / Q6_K), written for one
RTX 4050 Laptop GPU (sm_89, 6 GB, 233 GB/s).

**Goal:** long-context concurrent serving that llama.cpp cannot do on this card. At 150k-token contexts llama.cpp
decodes 46-49 tok/s in total, whatever the concurrency, and prefills at ~640 tok/s.

## Status (2026-11-25)

Built and working: the forward pass, multi-stream decode and long prefill.

Not built yet: the tokenizer, chat template, sampling options and the server. The CLI takes token ids.

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

### Prefill, one stream

The chunk size is chosen automatically from free VRAM (2048 rows here):

| | 2048 tokens | 8192 tokens | 150k prompt |
|---|---|---|---|
| Q8_0 | **8,315 tok/s** (llama.cpp 6,970) | **7,465** (5,830) | **75.4 s, 2,037 tok/s** (llama.cpp ~235 s, ~640) |
| Q6_K | 6,105 (llama.cpp 6,670) | 5,618 (5,620) | **82.3 s, 1,867 tok/s** |

### Correctness

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

**Chunks:** sized to free VRAM, like llama.cpp's `-ub`.

**GEMMs on tensor cores:**

- **Q8_0:** int8 `m16n8k32`, 64-deep stages, ~37 TOPS.
- **Q6_K:** two int8 `m16n8k16` per 32, sub-scales folded into float factors, ~22 TOPS.
- **F16:** `m16n8k16`, including grouped per-head.

**MLA in expanded form**, per 4096-key block:

1. `k_b`/`v_b` expansion GEMM;
2. int8 K and Q with nope/pe scales;
3. flash attention with int8 QKᵀ and f16 PV, causal, merged across key blocks.

### Fusion

- **One mHC kernel per step:** PLE add with RMSNorm, post-mix, stream sum and the full pre step in one pass over the
  4 streams.
- **Producers emit int8 activations** for the next GEMM/GEMV: h, stream sum, RDA output norm and gate, MLA gate, PLE
  gate, q_a norm, SwiGLU.
- **RDA conv** is folded into the gate / L2-norm kernel.

## Build and run

```
make                                   # nvcc 13.x, sm_89
build/raon -m model.gguf --tokens 259201,882,198 -n 32 --top 5     # greedy, JSON lines with top-k log-probs
build/raon -m model.gguf --bench --slots 6 --depth 153600          # decode benchmark
build/raon -m model.gguf --bench --prefill 153600                  # prefill benchmark
python3 tests/parity.py --server ~/llama.cpp/raon-src/build/bin/llama-server --model model.gguf \
    --raon build/raon -n 64 --ref ref.json --ref-kv q8_0 --long doc.txt
```

Environment knobs:

- `RAON_PROFILE=1`: per-kernel-group timing; it also names the group in which a CUDA error surfaces.
- `RAON_CHUNK`: overrides the prefill chunk size.
- A/B switches: `RAON_NOGRAPH`, `RAON_NOGEMM`, `RAON_NOFA`, `RAON_FA16`, `RAON_NSPLIT`.
- Debugging: `RAON_DUMP`, `RAON_PERTURB`.

Compile switches: `ATTN_PV_F32ACC`, `FA_PV_F32ACC`, `KV_ROWSCALE`.

`tests/ab.sh` runs interleaved A/B benchmarks and discards any round that overlaps another GPU process.
`tests/gemm_bench.cu`, `gemm_q6_check.cu`, `gemm_f16_check.cu` and `gemv_bench.cu` are kernel benches and checks.
