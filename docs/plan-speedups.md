# raon-engine: plan for further speed-ups

Status: 2026-11-25, after `0d8b5f6`.

Done:

- prefill chunks up to 4,096 rows;
- the Q6_K GEMM prefetch and mma-magic conversion;
- items 3 and 4 below. Together they make a 128k Q6_K prefill 56.5 → 48.85 s, and a 150k prefill 73.9 → 63.2 s (Q6_K)
  and 67.8 → 56.9 s (Q8_0).

Every item below says:

- the measurement it rests on;
- the expected gain (an estimate unless marked *measured*);
- the risk, and how it will be validated.

## Ground rules

These apply to every item:

- **A/B:** interleaved A/B against the previous binary on the same GPU, with no other GPU process. Use `tests/ab.sh`,
  or alternate the runs by hand.
- **Parity:** `tests/parity.py --long` against the stored llama.cpp references; Q6_K must stay at 181/181 greedy, and
  log-prob differences must stay within the known noise floor (≤ ~1.06).
- **Reporting:** every result states the command and flags, and the VRAM it used. Take the VRAM from the engine's
  `raon: VRAM ...` line or `/props`.
- **Benchmarks:** `build/raon --bench` warms up for ≥ 1.5 s, because the GPU idles at 210 MHz.
- **Target configuration:** Q6_K, 4 slots × 128k (`raon-server --slots 4 --ctx 131072`): 5,134 of 5,822 MiB used, so
  ~690 MiB is left for anything new.

## Where the time goes (measured, Q6_K)

| workload | time | breakdown |
|---|---|---|
| decode, 4 × 128k | 16.8 ms/step (59.4 tok/s/stream) | MLA attention 9.0 ms at ~88 % of DRAM bandwidth; weight GEMVs at 95-100 % of bandwidth (`tests/gemv_bench.cu`); ~1.5-2.5 ms launch/latency overhead over ~300 small kernels; sampling 0.36 ms |
| prefill, 2,048-token chunk | ~320 ms | GEMMs ~67 %, mHC 13 %, RDA scan + conv/gates 12 % |
| prefill, 128k prompt | 56.5 s (48.85 s after items 3 and 4) | at 56.5 s: flash attention ~53 %, per-chunk re-expansion ~10 %, GEMMs ~25 % |

## Plan, in order

### 3. FP8 for P·V in prefill flash attention: DONE (`0d8b5f6`)

**Measured result:**

- e4m3 `m16n8k32` with f16 accumulation runs at 97.6 TFLOPS against 48.8 for f16 (`tests/mma_rate.cu`). With f32
  accumulation FP8 would not be faster than f16 on this GPU.
- Q6_K 128k prefill: 53.4 → 48.85 s (−8.5 %). Removing P·V entirely saves 22 % of FA; the extra gain over the
  estimate comes from V moving at 1 byte per element.
- Parity: Q6_K 181/181, worst 0.710 (f16 0.635); Q8_0 181/181, worst 0.858.
- `RAON_PV16` keeps the f16 path.

The original plan follows.

**Basis:**
- FA is ~49 % of a 128k prefill.
- QKᵀ already runs on int8 tensor cores; P·V runs in f16 (with f16 accumulation per tile).
- sm_89 has FP8 (e4m3) `mma` at the int8 rate: 2 × f16.

**Change:**
- P (in [0, 1] after the softmax shift) goes to e4m3.
- V comes from the expansion, quantized per channel to e4m3 with its scale folded into the output.
- This is the SageAttention2 recipe.

**Estimate:** P·V is ~40 % of FA's FLOPs, so FA −15…20 % and a 128k prefill −7…10 %.

**Risk:** accuracy, since P is quantized to 3 mantissa bits.

**Validation:**
- parity, plus the long-document log-prob spread compared with the current f16 path;
- a compile switch (like `FA_PV_F32ACC`) to keep both paths.

**Effort:** medium.

### 4. int8 k_b/v_b expansion fused with the K/V quantization: DONE (`422fb1d`)

**Measured result:**

- The old expansion was an f16 GEMM with f32 accumulation, which runs at 1/4 of the int8 rate here.
- `pf_expand` 159 → 72 ms per chunk.
- Q6_K 128k prefill 56.5 → 53.4 s; −62 MiB VRAM.
- Parity: Q6_K 181/181, worst 0.635; Q8_0 as with the f16 path.
- `RAON_EXPAND16` keeps the f16 path.

The original plan follows.

**Basis:**
- The expansion (latent → per-head K_nope / V) is re-done for every chunk over all previous keys. It is ~15 % of a
  128k prefill.
- It runs as an f16 GEMM from a dequantized copy (`k_cache_dequant`).
- Its output is then quantized again for the int8 QKᵀ (`k_fa_kquant`).

**Change:**
- Read the int8 latent cache directly, with the per-row scales folded into the epilogue.
- Use int8 k_b/v_b weights (per-channel scales; built at load time, ~4 MB).
- In the epilogue, write int8 K (per-head scale) and FP8 or f16 V directly.

**Estimate:**
- the int8 mma rate is 2 × f16, and two passes and their memory traffic disappear;
- expansion −50 %, so a 128k prefill −6…8 %.

**Risk:** accuracy of int8 k_b/v_b (per-channel scales should keep it within the noise floor).

**Validation:** parity; kernel check against the f16 path.

**Effort:** medium.

### 5. Delta-rule scan on tensor cores (chunked form)

**Basis:** the RDA scan plus conv/gates are ~12 % of short prefill. The scan is token-sequential per head: a 128×128
state, updated with a rank-1 delta-rule step per token.

**Change:** the chunked (WY / UT-transform) formulation used by flash-linear-attention for KDA.
- Within a chunk of 64 tokens: a small triangular solve plus GEMMs on tensor cores.
- Across chunks: the state carries over.
- The per-channel decay (KDA) needs the gated variant.

**Estimate:** scan −60…70 %, so short prefill −6…8 % and long prefill −2 %.

**Risk:** numerics of the chunked decay (precision of cumulative-decay products). Keep the f32 state and compare the
state after N tokens with the sequential scan.

**Effort:** large; it is the hardest kernel here.

### 6. Fewer, larger decode kernels

**Basis:**
- At 1 stream the step is 7.3 ms against a 5.9 ms weight-read floor.
- At 4 × 128k it is 16.8 ms against ~13.8 ms (weights + KV).
- Profile mode shows mHC at ~1.1 ms per step, which is latency, not bytes.

**Change:**
- Fuse the mHC mix into the producing GEMV's epilogue and the consuming GEMV's prologue.
- Fuse the RDA conv / gates / scan / out-norm into one kernel per layer.
- Then evaluate a persistent kernel per layer group.

**Estimate:** −0.8…1.5 ms per step, so +5…10 % decode at 4 × 128k and more at 1 stream.

**Risk:** none numerically if the arithmetic is kept identical; the code is complex.

**Validation:** bit-identical outputs (greedy and log-probs) against the current build.

**Effort:** large; do it incrementally, fusing one boundary at a time with an A/B each.

### 7. Sampler first pass: radix select instead of full bitonic sort

**Basis:**
- GPU sampling costs +0.36 ms per step at 4 × 128k (*measured*, `--sample`).
- Pass 1 bitonic-sorts 64 slices of 4,096 values per row.

**Change:**
- Find the 128th-largest value per slice by radix select over the float bits (2-3 histogram passes in shared memory).
- Gather and sort only those 128.
- Pass 2 merges 64 × 128 the same way.

**Constraint:** the exact host fallback, which needs sorted slices, must then copy the full logits row and select on
the host. That fallback is rare: 2 % of picks at 4 × 128k with top_p 0.95 on synthetic data.

**Estimate:** −0.25 ms per step (+1.5 % decode).

**Validation:** `build/sampling_check` 289/289.

**Effort:** small.

### 8. 16-bit storage of the 4-stream residual (mHC)

**Basis:**
- mHC is 13 % of short prefill and 5 % of long prefill.
- It is bandwidth-bound: it reads and writes 4 × 1,536 fp32 values per token, 3 times per layer.

**Change:** store the streams as bf16 (or f16 with a per-row scale), and compute in f32.

**Estimate:** short prefill −5…6 %, decode ~−1 %.

**Risk:** accuracy: the residual carries the signal through 24 layers. This may well fail parity. Test it first as a
switch, against the noise floor.

**Effort:** small to medium; every mHC producer and consumer changes.

### Scheduling (server) items

- **Decode rows inside prefill chunks.**
  - *Problem:* while a long prompt prefills, the generating streams get their tokens only in the decode-only steps
    (`--prefill-share`).
  - *Change:* the chunk's GEMMs could carry the decode rows. The weights are then read once; attention for those rows
    uses the decode kernel.
  - *Estimate:* streams at ~49 tok/s during another prefill could reach ~70-80 tok/s at the same prefill rate.
  - *Effort:* engine change (multi-segment rows in the prefill path).
- **Chunk size beyond 4,096 when VRAM allows.**
  - *Basis:* 2,048 → 4,096 rows was −8 % on 128k (*measured*). 8,192 would need ~2.4 GB of workspace, which only
    fits with fewer or shorter slots.
  - *Change:* make the cap a flag.

## Previously identified, larger gains

These are not part of this plan's order.

- **MTP speculative decoding with a frequency-ranked draft vocabulary:**
  - ~+36 % decode at 4 × 128k (estimate from measured step costs and the 0.73 acceptance measured with llama.cpp);
  - output unchanged;
  - needs a 7th latent cache per slot: +435 MiB at 4 × 128k, against ~690 MiB free today. That leaves almost no
    margin, so the automatic chunk size would shrink; 2 checkpoints per slot instead of 4 would restore it.
- **4-bit latent KV cache:** ~+24 % decode at 4 × 128k (estimate); accuracy risk; it also frees ~700 MiB.

## Ruled out (measured)

| idea | result |
|---|---|
| tensor-core GEMV for batched decode | GEMVs already at 230 GB/s with 4 rows (`tests/gemv_bench.cu`) |
| absorbed-form MLA for prefill (no expansion) | ~3.4 × the FLOPs of the expanded form |
| 16-bit recurrent (delta-rule) state | saves only 0.33 ms/step at 4 streams; the state accumulates over 128k tokens |
| Q6_K GEMM: integer combination of the two sub-blocks + I2F | −30 % (I2F is quarter-rate) |
| Q6_K GEMM: folding the magic offset into a per-column constant | +2 %, 200 × the rounding error |
| Q6_K GEMM: 1 block/SM (more registers) | −7 % |
| Q6_K GEMM: 32 × 32 warp tiles | no gain |
| Q6_K GEMM: deeper stages / fewer barriers | an ablation without unpack and barrier was slower |
| magic-number conversion in the Q8_0 GEMM | no gain (the Q8_0 GEMM is not conversion-bound) |
| Q6_K weights as f16 for prefill (dequantize + f16 GEMM) | f16 mma with f32 accumulation runs at 1/4 of the int8 rate on GeForce Ada |
| Q6_K → per-32 requantization for prefill | lossy (a second rounding) and +30 % weight bytes |
| the plain (non-MTP) GGUF | the engine never loads the MTP block: 63 MB of disk only |
