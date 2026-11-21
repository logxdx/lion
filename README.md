# liquine

Made for my gpu poor self :)
lfm3 with tool support in the webui

A from-scratch CUDA engine and OpenAI-compatible server for [LFM3-1.7B](https://huggingface.co/LiquidAI) (Q8_0 / Q6_K GGUF),
tuned for an RTX 4050 Laptop (6 GB). It serves llama.cpp's web UI with `web_search`, `web_fetch` and `pdf` tools built in.

- Faster than llama.cpp on this GPU: ~121 tok/s decode (Q8_0, vs ~110), 4 streams x 100k context at once
- q8 KV cache, prompt cache in RAM and on disk, tool calls and reasoning in the API

## Run

```sh
make                      # CUDA 13 (nvcc in /opt/cuda), sm_89
make webui webtools       # web UI bundle + the tools' node deps
./serve.sh                # 4 slots x 100k context
```

Open `http://127.0.0.1:11434/` for the web UI, or point any OpenAI client at `http://127.0.0.1:11434/v1` (model `lfm3`).
`MODEL=path/to/LFM3-1.7B-Q8_0.gguf` picks the model; `LIQUID_CACHE_DIR` sets where the disk prompt cache goes.

Tools need: `node` (web tools), API keys for TinyFish / LangSearch / Exa (without any, search falls back to Wikipedia only),
and `liteparse` + `jq` for `pdf`.

Benchmarks, design notes and all the flags: [DETAILS.md](DETAILS.md).
