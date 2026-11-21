# liquid-engine

A from-scratch CUDA inference engine specialised for **LiquidAI LFM3-1.7B** (`Q8_0` and `Q6_K` GGUF)
on this machine: **RTX 4050 Laptop (AD107, sm_89, 6 GB, 96-bit GDDR6)** + i5-13420H.
No ggml / llama.cpp / cuBLAS at runtime: GGUF loader, BPE tokenizer, kernels, sampler and CLI are all here
(~3k lines). The only dependency is the CUDA runtime.

## Results (same GPU, same GGUF files, one engine at a time)

Reference: llama.cpp b12569 (`~/llama.cpp/cuda/current/llama-server`, `-fa on -ngl 99`, single slot, f16 KV).
`ppN` = prompt tokens/s for an N-token prompt, `tg128 @N` = decode tokens/s at context N.
Reproduce with `bench/compare.sh ~/llama.cpp/cuda/current <models...>` (raw output: `bench/results-2026-11-21.md`).

| model | engine | pp512 | pp2048 | pp8192 | tg128 @512 | tg128 @2048 | tg128 @8192 |
|---|---|---|---|---|---|---|---|
| Q8_0 | llama.cpp | 8809 | 10269 | 10365 | 110.6 | 109.2 | 105.7 |
| Q8_0 | **liquid** | **11068** | **11216** | **10476** | **120.8** | **119.8** | **115.1** |
| Q6_K | llama.cpp | 7930 | 8929 | **8958** | 138.0 | 136.2 | 130.2 |
| Q6_K | **liquid** | **9093** | **9306** | 8793 | **153.9** | **152.3** | **144.8** |

Decode runs at ~95 % (Q8_0) / ~93 % (Q6_K) of the measured DRAM roofline (233 GB/s; every token
streams all weights incl. the 128k-row tied LM head). At 32k context: prefill 7.8k t/s, decode ~100 t/s (Q8_0),
where the f16 KV read of the 6 global-attention layers is itself bandwidth-bound.

## Accuracy

Activations are int8 per 32 elements (the same scheme as llama.cpp's MMVQ/MMQ); weights are used exactly as
stored. `--exact` runs a reference path with f32 activations and exactly dequantised weights; both engines are
scored against it (`tests/drift_stats.py`, 30 random text snippets, Q6_K, next-token top-5 probabilities):

| vs exact f32 reference | mean max\|Δp\| | p90 | argmax agreement |
|---|---|---|---|
| liquid | **0.0099** | **0.023** | **29/30** |
| llama.cpp | 0.0138 | 0.030 | 28/30 |

Tokenization matches llama.cpp id-for-id (special tokens, CJK, whitespace runs, digits).

## Build & run

```sh
make                       # needs /opt/cuda (nvcc 13.x), builds build/liquid for sm_89
M=~/llama.cpp/models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf

build/liquid -m $M -p "Explain RoPE in two paragraphs."          # chat template, thinking on
build/liquid -m $M -p "..." --no-think -n 512                    # no <think> block
build/liquid -m $M -i -s "You are a terse assistant."            # interactive chat (/reset, /exit)
build/liquid -m $M --raw -p "def fib(n):" --greedy -n 64         # raw completion
build/liquid -m $M --bench --pp 2049 --tg 128                    # pp2048 / tg128 benchmark
build/liquid -m $M --profile --pp 2048                           # per-kernel prefill + decode timing
```

Sampling defaults follow the model card / GGUF (`temp 0.1, top-k 50, top-p 1.0, min-p 0, repeat-penalty 1.1`
over the last 64 tokens; `--repeat-last-n -1` = whole context). Options: `--temp --top-k --top-p --min-p
--repeat-penalty --repeat-last-n --seed --greedy`, `-c` context size (default 32768), `-f FILE` prompt from file.

## Design

**Weights** are repacked on the GPU at load time (lossless, bit-identical values):
- `Q8_0` -> int8 rows with the two 16-byte halves of every 32-group interleaved per 1024-element chunk, so lane *i*
  of a warp reads group *i* with two fully coalesced 16-byte loads; f16 scales in a separate array.
- `Q6_K` -> 4-bit plane (byte *j* = w[j] | w[j+16]<<4), 2-bit plane laid out so `(hi >> 2m) & 0x03030303` yields
  4 consecutive weights, int8 sub-scales, f16 super-block scale. One AND/shift/OR turns 4 weights into a `dp4a`
  operand (decode) or an int8 MMA fragment (prefill).
- Fused matrices: q/k/v/gate concatenated; conv `in_proj` rows regrouped into tiles of [b32 c32 x32]; FFN gate/up
  interleaved in [gate32 up32] tiles, so one CTA owns everything its epilogue needs.

**Decode** (one CUDA graph per token, ~110 kernels, host stays up to 3 tokens ahead):
- `dp4a` GEMVs; the prologue fuses residual add + sandwich post-norm + pre-norm + int8 quantisation, epilogues
  fuse the short conv (with its rolling state), SwiGLU and int8 re-quantisation of the next input.
- Occupancy-sized persistent grids (balanced tiles, single wave); each CTA prefetches the next matrix into L2.
- Attention: QK-RMSNorm + NEOX RoPE (partial rotary on global layers via `rope_freqs`), KV write, split-K over
  context, per-head sigmoid output gate, last-CTA combine; sliding-window layers use a ring buffer.
- Sampling on the GPU (penalty, top-k/top-p/min-p/temperature in llama.cpp's order); token, position and stop
  flag live in device memory, so steps chain without host round trips.

**Prefill** (chunks of 1024 tokens):
- int8 tensor-core GEMM (`mma.sync m16n8k32`, `k16` pairs for Q6_K sub-scales), 128x128x128 tiles, cp.async
  double buffering, swizzled smem + ldmatrix, CTA order that keeps each weight tile hot in L2, exact per-group
  scaling with an FP-pipe int->float trick; Q6_K is unpacked to int8 once per CTA. Measured ceiling on this GPU:
  96 TOPS int8 / 24 TFLOPS f16 (`bench/mma_peak.cu`).
- Short conv as a parallel 7-tap FIR over the chunk (state from the previous chunk), tensor-core flash attention
  (causal + sliding window, 4 query heads share each K/V tile, f32 scores, f16 per-tile P·V partial sums, gate +
  int8 output fused).

## Tests

With a reference server running (`bench/ref_server.sh ~/llama.cpp/cuda/current <model> 18081`):
- `tests/compare_llama.py build/liquid <model> http://127.0.0.1:18081` — 8 prompts incl. 729/2170 tokens
- `tests/drift_stats.py build/liquid <model> <text files...> --url ...` — statistical drift vs `--exact`
- `tests/greedy_match.py build/liquid <model> <url> 600` — long greedy generations side by side
- `tests/exact_mask_sweep.sh` — which block's int8 path drifts (`LIQUID_EXACT_MASK`)

Without a server: `tests/multiturn.sh build/liquid <model>` — incremental multi-turn chat must equal a fresh prefill.

## Limitations

- Specialised: LFM3-1.7B dimensions are checked at load and hard-wired (2048/7680/16 heads/128 dims, conv 7);
  only `Q8_0` and `Q6_K` weights. Other LFM3 sizes or quant types are rejected.
- Single sequence (no batching / server API). Multi-turn chat appends to the cache when the rendered conversation
  extends it, otherwise re-prefills (the recurrent conv state cannot be rewound).
- KV cache is f16 (llama.cpp presets use q8_0 KV; f16 needs ~12 KiB/token -> ~1.5 GB at 128k).
- Sampling considers candidates within `20·temperature` logits of the maximum (exact up to ~e^-20 tail mass).
- GPU only; the CPU path was not built (the 4050's 233 GB/s is ~4x what this laptop's RAM delivers).
