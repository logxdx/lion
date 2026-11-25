# raon-engine

A CUDA inference engine for **Raon-1 nano** (KRAFTON, GGUF arch `raon1nano`, Q8_0 / Q6_K), written for one
RTX 4050 Laptop GPU (sm_89, 6 GB, 233 GB/s).

Goal: long-context concurrent serving that llama.cpp cannot do on this card. At 150k-token contexts llama.cpp decodes
46-49 tok/s *in total* whatever the concurrency, and prefills at ~640 tok/s. Targets here: ≥ 40 tok/s per stream at
2-6 × 150k, and faster prefill.

## Status (2026-11-25)

The forward pass, multi-stream decode and long prefill all work. Not yet built: the tokenizer, chat template,
sampling options and the server; the CLI takes token ids.

### Decode

Synthetic 150k caches (`--bench --slots N --depth 153600`), tok/s per stream:

| streams × 150k | Q8_0 | Q6_K | llama.cpp (Q8_0) |
|---|---|---|---|
| 1 × short context | 109.8 | 132.3 | 112 / 136 (Q6_K) |
| 1 | 80.0 | 91.6 | – |
| 2 | **61.9** | **68.6** | 24.6 |
| 3 | **50.9** | **55.2** | 15.4 |
| 4 | **42.7** | **45.7** | 12.0 |
| 5 | 37.2 | 39.4 | – |
| 6 | 32.4 | 34.0 | does not fit |

6 × 150k fits in VRAM with both quantizations (q8 latent cache).

### Prefill

One stream, `--bench --prefill N`, 2048-row chunks (`RAON_CHUNK=2048`):

| | 0-16k | 48-64k | 64k average | 150k average |
|---|---|---|---|---|
| Q8_0 | 4,173 tok/s | 1,637 | 2,356 | – |
| Q6_K | 3,390 tok/s | 1,495 | 2,076 | **1,224** (125.5 s; llama.cpp ~640, ~235 s) |

Chunk size works like llama.cpp's `-ub`. Each chunk re-expands the whole prefix (as llama.cpp's MLA expand path does
per ubatch), so bigger chunks are faster. The workspace is ~440 KB per row: 2048-row chunks need ~0.9 GB of VRAM.

### Correctness

`tests/parity.py` compares greedy decoding and top-k log-probs against KRAFTON llama.cpp, on short chat prompts and a
9,773-token document (`--long`, reference with `--ref-kv q8_0`).

- **Short prompts:** log-prob differences stay within llama.cpp's own CUDA-vs-CPU spread (chosen ≤ 0.28,
  candidates ≤ 1.06).
- **Long document:** greedy decoding matches 64/64 tokens (Q8_0 and Q6_K).
- **Divergences:** all at near-ties, with a top-2 gap of 0.02-0.27 nats.
- **Sensitivity:** the model amplifies tiny numeric differences. A ±1e-6 relative perturbation of the matmul outputs
  (`RAON_PERTURB`) already moves first-token log-probs by up to ~0.6 at 9.8k context. llama.cpp moves 0.15-0.30 just
  from changing `-ub`. So parity is judged against this noise floor, not bitwise.

## Design

Source of truth for the model: KRAFTON's llama.cpp `src/models/raon1nano.cpp` and the ggml op definitions.

- **Weights:**
  - Q8_0 → int8 + fp16 scale per 32;
  - Q6_K → GGUF block order split into low-nibble, high-bit, sub-scale and scale planes;
  - F16 kept as F16;
  - the per-layer-embedding table (1.3-1.7 GB) stays in pinned host memory, one 5-6.5 KB row read zero-copy per
    token.
- **Decode** (one CUDA graph per batch size):
  - dp4a GEMVs (~96 % of DRAM bandwidth);
  - MLA flash-decode over a q8 latent cache, with all 32 heads per pass: scores on int8 tensor cores
    (`m16n8k32`, one q8 block per k-step), PV on f16 tensor cores, a 2-stage `cp.async` pipeline, and keys split over
    blocks on the device;
  - the RDA delta-rule state (128×128 fp32 per head) is held in registers by 2 threads per column.
- **Prefill:**
  - tensor-core GEMMs: Q8_0 int8 `m16n8k32`; Q6_K two int8 `m16n8k16` per 32 combined in int32 with the
    sub-scales; F16 `m16n8k16`, incl. grouped per-head;
  - MLA in expanded form: per 4096-key block, the latent cache is expanded by `k_b`/`v_b` with one f16 GEMM, then
    flash attention runs with head dims 192/128, causal, merged across key blocks.

## Build and run

```
make                                   # nvcc 13.x, sm_89
build/raon -m model.gguf --tokens 259201,882,198 -n 32 --top 5        # greedy, JSON lines with top-k log-probs
build/raon -m model.gguf --bench --slots 4 --depth 153600             # decode benchmark
RAON_CHUNK=2048 build/raon -m model.gguf --bench --prefill 153600     # prefill benchmark
python3 tests/parity.py --server ~/llama.cpp/raon-src/build/bin/llama-server --model model.gguf \
    --raon build/raon -n 64 --ref ref.json --ref-kv q8_0 --long doc.txt
```

Environment knobs:

- `RAON_PROFILE=1`: per-kernel-group timing; it also names the group in which a CUDA error surfaces.
- `RAON_CHUNK`: prefill chunk size (default 256).
- `RAON_NOGRAPH`, `RAON_NOGEMM`, `RAON_NOFA`, `RAON_NSPLIT`: A/B switches.
- `RAON_DUMP`, `RAON_PERTURB`: debugging.

`tests/ab.sh` runs interleaved A/B benchmarks and discards any round that overlaps another GPU process.
