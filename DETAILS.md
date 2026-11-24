# liquine: details (benchmarks, design, server internals)

A from-scratch CUDA inference engine specialised for **LiquidAI LFM3-1.7B** (`Q8_0` and `Q6_K` GGUF)
on this machine: **RTX 4050 Laptop (AD107, sm_89, 6 GB, 96-bit GDDR6)** + i5-13420H.
No ggml / llama.cpp / cuBLAS at runtime: GGUF loader, BPE tokenizer, kernels, sampler and CLI are all here.
The only dependency is the CUDA runtime. Up to 8 concurrent sequences (batched decode), q8_0 or f16 KV cache.

All numbers below: same GPU, same GGUF files, one engine at a time. Reference: llama.cpp b12569
(`llama-server -fa on -ngl 99`). Raw outputs are in `bench/results-*`.

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
M=~/models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf

build/liquid -m $M -p "Explain RoPE in two paragraphs."          # chat template, thinking on
build/liquid -m $M -p "..." --no-think -n 512                    # no <think> block
build/liquid -m $M -i -s "You are a terse assistant."            # interactive chat (/reset, /exit)
build/liquid -m $M --raw -p "def fib(n):" --greedy -n 64         # raw completion
build/liquid -m $M -c 131072 -f long_prompt.txt                  # long context (q8 KV by default)
build/liquid -m $M --bench --pp 2049 --tg 128                    # pp2048 / tg128 benchmark
build/liquid -m $M --bench --parallel 4 --pp 32769 -c 33024      # 4 concurrent streams x 32k
build/liquid -m $M --profile --pp 8192                           # per-kernel prefill + decode timing
```

Sampling defaults are read from the GGUF's `general.sampling.*` keys (LFM3: `temp 0.1, top-k 50,
repeat-penalty 1.1`; top-p / min-p unset = off). The repetition-penalty window defaults to the whole context
(`-1`), as in transformers' `generate()` that the model card's settings come from (llama.cpp's default is the
last 64 tokens; with 64 the pi agent looped in the earlier preset tests, and the whole-context window costs
nothing measurable: 106.3 vs 106.2 tok/s at 40k context, `bench/penalty_window.py`). Options: `--temp --top-k --top-p --min-p
--repeat-penalty --repeat-last-n --seed --greedy`, `-c` context per sequence (default 32768), `-f FILE`,
`--kv q8|f16`. The library API (`src/engine.h`) exposes per-slot `prefill(tokens, slot)` and
`generate_multi(slots, ...)` for concurrent sequences.

## OpenAI-compatible server (`build/liquid-server`)

```sh
make webui webtools                          # once: llama.cpp web UI bundle + the tools' node deps
./serve.sh                                   # 4 slots x 100k context, q8 KV, RAM + disk prompt cache, port 11434
build/liquid-server -m $M -np 4 -c 100000 --port 8080 --cache-ram-mib 4096 --cache-disk-mib 16384
curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

**Web UI with tools.** `http://127.0.0.1:11434/` (serve.sh; the binary defaults to 8080) serves llama.cpp's web UI (the prebuilt b12569 bundle that
llama.cpp's build embeds; `make webui` downloads it to `build/webui`, checks its sha256 and rebrands it; `--ui-dir`,
`--no-ui`). **Branding**: `brand/rebrand.py` renames it liquine (titles, manifest, logo, favicons, PWA icons, splash
screens from `brand/webui/`, help texts that pointed at llama-server, the model catalog URL now served empty by
liquid-server), asserting every replacement so a different bundle fails loudly; changed bundle files get new names
because `_app/immutable/` is cached for a year (everything else revalidates). The logo is `brand/liquine-mark.svg`;
`brand/build_assets.py` renders the icons from it. The UI is sent without cross-origin isolation headers
(llama-server's COEP `require-corp` blocked images from most sites in replies; the UI does not need isolation), so
markdown images and links in model replies render.
Next to the UI's built-in browser tools (`get_datetime`, `get_info`, `run_javascript`, ...) the server provides
three tools in llama-server's server-tool format (`GET /tools` lists them, `POST /tools {tool, params}` runs one,
the same contract as llama-server's `get_info`/`read_file`), so the UI lists them under "Server Tools", enables
them by default and asks permission per call:
- `web_search` and `web_fetch`: the pi web-search / web-fetch extensions' code (TinyFish -> LangSearch -> Exa ->
  Wikipedia (keyless); TinyFish fetch -> Readability + Turndown -> Jina; keys from `secret-tool` or env), vendored in
  `tools/webtools/` with the same names, parameters and output text.
- `pdf`: the pi pdf-reader skill's `pdf` command (liteparse underneath): `map`, `ocr`, `outline`, `find`, `page`
  on a local path (relative to the UI's working directory) or a URL, with per-file caching and bounded output.

Each call runs `node tools/webtools/tools.ts call <name>` (`--tools-dir`, `--node`, `--tool-timeout`, `--no-tools`);
a closed client connection kills it. In the UI, LFM3 chains them on its own: search -> fetch -> answer
(`tests/agent_loop.py` replays the UI's loop: ~10 s for a "latest Rust release" question), and pdf map -> find ->
page for questions about a paper. `tests/test_tools.py` checks the endpoints.

Endpoints: `POST /v1/chat/completions` (stream + non-stream, `tools` -> `tool_calls`, `reasoning_content`,
`chat_template_kwargs.enable_thinking/preserve_thinking`, `stop`, `seed`, `max_tokens`/`max_completion_tokens`,
`temperature/top_p/top_k/min_p/repeat_penalty/repeat_last_n`, `stream_options.include_usage`, `ignore_eos`),
`POST /v1/completions` (text or token ids), `GET /v1/models`, `GET /health`, `GET /props`, `GET /slots` (slot +
cache state), `POST /tokenize`, `GET`/`POST /tools`, and the web UI at `/`. Responses carry `usage.prompt_tokens_details.cached_tokens` and llama.cpp-style `timings`
(queue, prefill, restore, decode, cache tier). Point an agent at `http://127.0.0.1:11434/v1`, model `lfm3`.

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
  RAM tier is flushed to disk. Disk directory: `--cache-dir`, else `$LIQUID_CACHE_DIR` (`serve.sh` sets the SSD0
  drive), else `~/.cache/liquid-engine`. When the chosen directory is missing or dies (drive unplugged, I/O errors;
  probed with a test write at startup and after a failed spill) the tier moves to `~/.cache/liquid-engine` and
  entries that were on the dead drive are dropped (their next request prefills).

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

## Images: LFM3-VL-2B and model swapping

`serve.sh` runs two models: LFM3-1.7B (default, 4 x 100k) and LFM3-VL-2B (`--vl-model`, `--vl-mmproj`, `--vl-ctx`,
default 64k; Q8_0 decoder + Q8_0 mmproj). One is on the GPU at a time. A request goes to the VL model when its
messages contain images (OpenAI `image_url` parts with base64 `data:` URLs, what the web UI sends) or when it names
`lfm3-vl`; everything else goes to the default. When the queue head needs the other model, the scheduler lets running
requests finish, moves every slot to the prompt cache, frees the GPU and loads the other model (requests keep arrival
order). With two models the web UI runs in llama-server's router mode (`/props` `role: "router"`, every model
`status: loaded` in `/v1/models`, per-model `/props?model=`, no-op `/models/load|unload`, a keep-alive `/models/sse`):
it shows a model picker, sends `model` with each request, and offers image attachments on `lfm3-vl`.

| measured (RTX 4050, serve.sh config) | |
|---|---|
| swap, model load included (35 swaps each way) | median 383 ms to LFM3, 425 ms to LFM3-VL (incl. encoder) |
| VRAM | LFM3 4 x 100k: 4550 MiB; LFM3-VL 4 x 64k + encoder + buffers for a 4096-token image: 3946 MiB |
| 56 swaps in a row (`tests/swap_stress.py`) | every answer right, VRAM back to the same value each time, RSS flat |
| image encode, 640x480 / 320x512 / 2816x1472 (300 / 160 / 4048 tokens) | 23 / 13 / 528 ms (llama.cpp: 95 / 13 / 786 ms) |
| a second question about the same image | 0.27 s instead of 1.71 s (4048-token image: no re-encode, 14 ms prefill) |

**Vision encoder** (`src/vision.cu`, `src/image.cpp`): image -> Qwen2-VL smart_resize to multiples of 32 (64..4096
tokens) -> Pillow/torchvision antialiased bicubic (a port of llama.cpp's, same int16 weights, same pixels) ->
16x16 patches (the two temporal kernels pre-summed for still images) -> 14 blocks: 8 with a 3x3 depthwise conv over
the patch grid (`out_proj(C * dwconv(B * X))`), 6 with GQA attention (16/8 heads, d 64, QK RMSNorm, 2D RoPE base 1e6
on pairs (k, k+32): y for k < 16, x above), RMSNorm, SwiGLU -> LayerNorm, 2x2 merge, Linear-GELU(erf)-Linear -> one
2048-dim embedding per 32x32 pixels. Linear layers reuse the decoder's int8 tensor-core GEMM over repacked Q8_0
(`src/gemm.cuh`, `src/repack.cuh`, W8A8 like llama.cpp's MMQ); attention is a non-causal f16 flash kernel.
The embeddings replace `<|image|>` between `<|vision_start|>` and `<|vision_end|>` (`Engine::feed_embd`); in the
server's token arrays image positions are negative ids derived from the image bytes, so prefix reuse and the
prompt cache work for images too, and a checkpoint at the end of every image lets the next question about it resume
there.

**Accuracy** (`tests/vl_ref_embed.cpp` dumps llama.cpp's embeddings and every intermediate tensor through libmtmd;
`tests/test_vision.cpp` compares): on `test-1.jpeg` our embeddings vs llama.cpp CUDA: cosine 0.99851 mean, rel L2
5.25%; llama.cpp's own CPU backend vs its CUDA backend: 0.99852, 5.28%. The error grows evenly through the 14 blocks
(no stage jumps), the same way between llama.cpp's two backends: W8A8 rounding noise. End to end (same Q8_0 decoder,
greedy): 5 of 7 answers on 3 images are word-for-word identical to llama-server's, the other 2 drift after the same
content (`What is the main headline?` -> "MEN WALK ON MOON ... Monday, July 21, 1969" from both).

CLI: `build/liquid -m LFM3-VL-2B-Q8_0.gguf --mmproj mmproj-LFM3-VL-2B-Q8_0.gguf --image photo.jpg -p "Describe it."`.
Tests: `tests/test_vl_server.py` (routing, images, cache reuse, swap back, 400 for images to a text model),
`tests/swap_stress.py` (repeated swaps, leak check), `tests/test_vision.cpp`.

## Video (LFM3-VL)

The server takes the web UI's `input_video` parts (`{data: <base64>, format}`) and `video_url` parts with a base64
`data:` URL; `/props` reports `modalities.video` (the UI then offers video attachments on `lfm3-vl`) when the VL model
is loaded and `ffmpeg` + `ffprobe` are on the PATH. CLI: `build/liquid ... --mmproj mmproj.gguf --video clip.mp4 -p "..."`.
It follows llama.cpp b12674's LFM3 video path (`MTMD_VIDEO_TEMPLATE_LFM3`), checked against it step by step:

- **Frames** (`src/video.cpp`): ffprobe -> n = clamp(trunc(frames / fps * 2), 4, 768), at most the source's, evenly
  spaced (`round(i * (frames - 1) / (n - 1))`); the size from Qwen-VL's video smart_resize over the whole clip
  (multiples of 32, 4096 .. 25,165,824 pixels over the frames); ffmpeg scales with the same filter
  (`bicubic+accurate_rnd+full_chroma_int`, B=0 C=0.5), so the frames are byte-identical to llama.cpp's. Display
  geometry like llama.cpp: width x sample aspect ratio, width/height swapped for +-90 degree rotation (phone videos).
  Without a frame count in the container (some webm/mkv): ffmpeg's `fps=2` filter at the source size, frames snapped
  to multiples of 32 (ties to even, no token cap; measured: 597x336 -> 608x320, 3200x1800 -> 3200x1792).
- **Encoder**: frames go in pairs, one encoder pass per pair: frame 0 through the first temporal patch kernel, frame 1
  through the second (`Vision::encode_pair`); a lone last frame goes through both, like a still image.
- **Prompt**: the template's `<|vision_start|><|video|><|vision_end|>`, `<|video|>` replaced per pair by
  `<t seconds>` (the pair's mean time, one decimal) `<|vision_start|>` (pair embeddings) `<|vision_end|>`.
  In the server a video is one checkpoint (after its last pair), so a follow-up question about it starts there.

| measured (RTX 4050) | |
|---|---|
| 5 s 640x360 clip: 10 frames, 5 pairs x 220 tokens | decode (ffmpeg) 160 ms, encode ~16 ms per pair (llama.cpp CUDA ~29 ms) |
| 3200x1800 webm, one 5600-token pair | encode 871 ms (llama.cpp 1795 ms); VRAM peak 4749 MiB with serve.sh's 4 x 64k |
| prompt tokens vs llama-server, 4 clips (plain, rotated, SAR 2:1, webm) x 3 questions | 12 / 12 identical |
| greedy 48-token answers vs llama-server, same 12 requests | 8 identical, 4 drift after 11-24 identical words |

**Accuracy** (`tests/test_video.cpp`; `tests/vl_ref_video.cpp` dumps llama.cpp's pair embeddings through libmtmd):
decoded frames byte-identical to llama.cpp's; pair embeddings vs llama.cpp (CPU): cosine 0.9959 .. 0.9991 mean,
rel L2 4.1 .. 7.8%, the same band as the still-image path on single frames of the same clip (0.9967 .. 0.9992):
the int8-activation encoder's content-dependent noise. The pair math itself is checked exactly: two identical frames
equal the still image at the patch embedding (rel 5e-7, f32 rounding; `LIQUID_VDUMP` stage dumps), and swapping the
two frames moves the result 13x further from llama.cpp's (rel L2 0.60 vs 0.046), so the kernel order is right.
Tests: `tests/test_video.cpp`, `tests/test_video_server.py` (web UI and data-URL formats, follow-up reuse, 400 for
a non-video; also runs against llama-server for parity).

## Decisions: d1-2B (`/v1/systemone`)

`serve.sh` adds LiquidAI's d1-2B decision model when `~/models/d1-2b` has it (`--d1-model`, `--d1-mmproj`,
`--d1-alias` default `d1`, `--d1-ctx` default 8192 tokens per forward pass; Q8_0 by default, Q6_K works too). It is
swapped onto the GPU like the VL model and answers llama.cpp's `POST /v1/systemone` (it does not chat: chat requests
naming it get llama.cpp's 501, and the web UI does not list it):

```sh
curl localhost:11434/v1/systemone -d '{"state": "I bought a kettle last week, it broke, I want my money back.",
  "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": "Repairs"}},
                "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}}}'
# {"model":"d1","answers":{"refund":{"type":"noul","noul":0.996},"team":{"type":"choice","choice":"billing",
#  "probabilities":{...},"confidence":0.8},"urgency":{"type":"score","score":1.0,"legend":{...},...}},"usage":{...}}
```

`state` is text, any JSON, chat messages (their image / video parts become media) or null; `files` / `images` take
up to 8 `data:image/...` or `data:video/...` URLs. Request checks, error messages, the systemone template (the
model's own, from the GGUF), the escaping of special-token text in user input (U+200B after the first character),
Python-style `tojson`, label tokens (yes/Yes vs no/No/NO, option codes A.. or the keys, digits), the temperature
(`lfm3.decision.temperature.*`, read through `%f` as llama.cpp does), median scores and the response format follow
llama.cpp (`src/decision.cpp`).

**One forward pass for up to 16 questions** (`Engine::decide`): the shared prefix (media, state, `QUESTION:`) once,
then each question's text (later ones after another `QUESTION:\n` when there is a state). d1 is LFM3 with
`attention.causal = false`: the 6 global attention layers attend both ways, except that a question's tokens do not
see the other questions (prefix tokens see everything); the 5 sliding-window layers and the short convs stay causal
over the whole packed sequence. That rule was measured against llama.cpp b12674, not taken from source: a prefix
that sees only itself gives P(yes) 0.9835 instead of 0.9961, and packed choice probabilities .67/.31 instead of
.80/.20; the rule above matches both. The whole pass is one prefill chunk (the decision engine sizes its chunk to
`--d1-ctx`); global layers use the flash prefill kernel with a group mask (`k_flash_prefill<Q8, true>`, key tiles
outside the prefix and the tile's own questions are skipped), the logits of the questions' last tokens come from one
GEMM over the tied embeddings. The text model's kernels are untouched: the group mask is a template branch
(A/B against the previous binary: pp512 / pp4096 / pp16384 / decode within run-to-run noise).

| measured (RTX 4050) | |
|---|---|
| parity vs llama-server b12674, 15 requests (packed, single, null state, JSON + special tokens + unicode, chat state, 21 questions = 2 passes, image, video, 7 error cases) | identical token counts, answers, legends, key order, error messages; probabilities within 0.015 (Q8_0) / 0.010 (Q6_K) |
| request latency, Q6_K, ours vs llama-server: 3 questions / 16 questions / 1.3k-token state / image + 3 questions | 21.0 / 49.6 / 163 / 86 ms vs 22.7 / 46.8 / 169 / 84 ms |
| the forward pass itself | the engine's prefill speed (361 tokens: 52 ms on Q6_K, the same as a causal prefill); Q8_0 prefills ~20% faster than Q6_K here |
| swap to d1 | ~0.5 s; text / d1 / VL rounds: same VRAM every round (4599 / 2849 / 3955 MiB) |

Tests: `tests/test_d1_server.py URL --save ref.json` against llama-server, then `--ref ref.json` against
liquid-server; `tests/test_decide.cpp` runs `Engine::decide` on hand-made packed prompts (`LIQUID_DECIDE_PROF=1`
times its phases).

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

With a reference server (`bench/ref_server.sh ~/llama.cpp/current <model> 18081`):
- `tests/compare_llama.py`, `tests/drift_stats.py`, `tests/greedy_match.py` (see their headers)

Profiling: `--profile` (per-kernel time + achieved GB/s or TFLOPS), `-DATT_TRACE` builds (`%globaltimer` per
attention phase), `bench/attn_sweep.sh`, `bench/kv_ab.sh`, `LIQUID_OCC=1` (attention occupancy).

## Limitations

- Specialised: LFM3-1.7B dimensions are checked at load and hard-wired (2048/7680/16 heads/128 dims, conv 7);
  only `Q8_0` and `Q6_K` weights. Other LFM3 sizes or quant types are rejected.
- Server: up to 8 slots (compile-time `MAX_SLOTS`); one model per process; `n > 1`, logprobs and the OpenAI
  presence/frequency penalties are not implemented (LFM3's recommended sampling does not use them; the server
  reports its effective defaults at `GET /props`); tool-call arguments are streamed as one delta per call.
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
