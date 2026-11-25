# raon-engine

A CUDA inference engine for **Raon-1 nano** (KRAFTON, GGUF arch `raon1nano`, Q8_0 / Q6_K), written for one
RTX 4050 Laptop GPU (sm_89, 6 GB, 233 GB/s).

Goal: long-context concurrent serving that llama.cpp cannot do on this card. At 150k-token contexts llama.cpp decodes
46-49 tok/s *in total* whatever the concurrency (2 × 150k: 24.6 tok/s per stream, 4 × 150k: 12.0). The target here is
≥ 40 tok/s per stream at 2-6 × 150k, with faster prefill.

## Status

v1: correct forward pass with straightforward batched kernels.

- **Matches KRAFTON llama.cpp** (`raon-1-nano` @ b662635) on greedy decoding (`tests/parity.py`). Log-prob differences
  are within llama.cpp's own CUDA-vs-CPU spread, measured on the same prompts:

  | | chosen token | candidates (logprob > −5) |
  |---|---|---|
  | raon vs llama.cpp CUDA, Q8_0 | ≤ 0.28 | ≤ 0.78 |
  | llama.cpp CPU vs CUDA, Q8_0 | ≤ 0.28 | ≤ 1.06 |
  | raon vs llama.cpp CUDA, Q6_K | ≤ 0.20 | ≤ 0.66 |

  All divergences are at near-ties (top-2 gap 0.07-0.18 nats).
- **Speed:** single-stream decode is 97 tok/s (Q8_0) / 115 tok/s (Q6_K) with no fusion and no CUDA graphs.
  Prefill (~800 tok/s) and attention are naive.

## Model, as implemented

Source of truth: KRAFTON's llama.cpp `src/models/raon1nano.cpp` and the ggml op definitions.

- **Layers:** 24.
  - 18 RDA layers: gated delta rule with a per-channel decay, 16 heads × 128, a 128×128 fp32 state per head, and a
    4-tap conv.
  - 6 MLA layers: absorbed MLA with 32 heads, no rotary embedding (NoPE), a 576-value latent cache row, scale 1/√192
    and a sigmoid output gate.
- **Residual:** a 4-stream manifold hyper-connection (mHC) around every operator and FFN.
- **Per-layer embedding:** a 256-wide input per layer, from a 1.3-1.7 GB table kept in pinned host memory. Each token
  reads one 5-6.5 KB row zero-copy.
- **Head:** tied LM head, vocab 259,840.
- **Weights:**
  - Q8_0 → int8 + fp16 scale per 32;
  - Q6_K → GGUF block order split into low-nibble, high-bit, sub-scale and scale planes;
  - F16 kept as F16.

  Decode GEMVs use dp4a on int8-quantized activations, the same as llama.cpp's matrix-vector path.

## Build and run

```
make                                   # nvcc 13.x, sm_89
build/raon -m raon-1-nano-Q8_0_MTP.gguf --tokens 259201,882,198 -n 32 --top 5
python3 tests/parity.py --server ~/llama.cpp/raon-src/build/bin/llama-server \
    --model raon-1-nano-Q8_0_MTP.gguf --raon build/raon -n 96 --ref ref.json [--vs-cpu cpu.json]
```

## llama.cpp baseline on this GPU

KRAFTON fork b662635, guarded runs (no other GPU process).

| | Q8_0 | Q6_K |
|---|---|---|
| prefill pp2048 / pp8192 | 6,960 / 5,700 tok/s | 6,670 / 5,620 tok/s |
| decode tg128 at depth 0 / 8k / 32k | 112 / 108 / 96 tok/s | 136 / 130 / 113 tok/s |

**Concurrent, 150k-token prompts per stream** (llama-server, q8_0 KV):

| streams | prefill, total | decode, total | decode per stream |
|---|---|---|---|
| 2 | 640 tok/s | 49 tok/s | 24.6 tok/s |
| 3 | 651 tok/s | 46 tok/s | 15.4 tok/s |
| 4 | 650 tok/s | 48 tok/s | 12.0 tok/s |

6 × 150k does not fit in llama.cpp on this card.
