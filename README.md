<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="brand/liquine-logo-dark.svg">
  <img src="brand/liquine-logo.svg" alt="liquine" height="72">
</picture></p>

Made for my gpu poor self :)
lfm3 with tool support in the webui

A from-scratch CUDA engine and OpenAI-compatible server for [LFM3-1.7B](https://huggingface.co/LiquidAI) (Q8_0 / Q6_K GGUF),
tuned for an RTX 4050 Laptop (6 GB). It serves a chat web UI with `web_search`, `web_fetch` and `pdf` tools built in,
and images and video through LFM3-VL-2B, swapped onto the GPU when a request has one.

- Faster than llama.cpp on this GPU: ~121 tok/s decode (Q8_0, vs ~110), 4 streams x 100k context at once
- q8 KV cache, prompt cache in RAM and on disk, tool calls and reasoning in the API
- Images: own CUDA vision encoder (matches llama.cpp, ~20 ms for a 640x480 photo); text and VL models swap in ~0.4 s
- Video: 2 fps frame pairs, same frames and prompt layout as llama.cpp (needs `ffmpeg` + `ffprobe` on the PATH)
- Web UI: pick `lfm3` or `lfm3-vl`, attach images or videos; replies can show images and links from the web

## Run

```sh
make                      # CUDA 13 (nvcc in /opt/cuda), sm_89
make webui webtools       # web UI bundle (+ liquine branding) + the tools' node deps
./serve.sh                # text: 4 slots x 100k context; images: LFM3-VL at 64k, swapped in on demand
```

Open `http://127.0.0.1:11434/` for the web UI, or point any OpenAI client at `http://127.0.0.1:11434/v1` (model `lfm3`;
messages with images or videos go to `lfm3-vl` automatically). `MODEL=...` picks the text model, `VL_MODEL=... VL_MMPROJ=...` the
vision model (default `LFM3-VL-2B-Q8_0.gguf` + `mmproj-LFM3-VL-2B-Q8_0.gguf`), `LIQUID_CACHE_DIR` the disk prompt cache.

Tools need: `node` (web tools), API keys for TinyFish / LangSearch / Exa (without any, search falls back to Wikipedia only),
and `liteparse` + `jq` for `pdf`.

Benchmarks, design notes and all the flags: [DETAILS.md](DETAILS.md).

The web UI is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s (MIT), rebranded at build time by `brand/rebrand.py`.
