#!/bin/sh
# Start liquid-server for agents: 4 concurrent sequences x 100k context, q8_0 KV, RAM + disk prompt cache.
#   usage: [MODEL=...] [VL_MODEL=... VL_MMPROJ=... VL_CTX=65536] [D1_MODEL=... D1_MMPROJ=...] [PORT=11434] [SLOTS=4]
#          [CTX=100000] [LIQUID_CACHE_DIR=...] ./serve.sh [extra liquid-server args...]
# Web UI (llama.cpp's, with web_search / web_fetch / pdf tools): http://127.0.0.1:$PORT/
# Clients: base URL http://127.0.0.1:$PORT/v1, model "lfm3" (text, default) or "lfm3-vl" (images).
# Requests with images go to LFM3-VL: the server swaps it onto the GPU and back when needed (one model loaded at a time).
# Decision questions (POST /v1/systemone, llama.cpp's API) go to d1-2B, swapped in the same way.
# Measured on the RTX 4050 6 GB: 4 x 100k with Q8_0 weights uses ~4.4 GB of VRAM; pinned RAM tier 4 GiB.
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)   # resolves symlinks (e.g. ~/.local/bin/liquid-serve)
M=$HOME/models
MODEL=${MODEL:-$M/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf}
VL_MODEL=${VL_MODEL:-$M/lfm3-vl-2b/LFM3-VL-2B-Q8_0.gguf}
VL_MMPROJ=${VL_MMPROJ:-$M/lfm3-vl-2b/mmproj-LFM3-VL-2B-Q8_0.gguf}
VL_CTX=${VL_CTX:-65536}   # images: short contexts
VL=""
if [ -f "$VL_MODEL" ] && [ -f "$VL_MMPROJ" ]; then VL="--vl-model $VL_MODEL --vl-mmproj $VL_MMPROJ --vl-ctx $VL_CTX"; fi
D1_MODEL=${D1_MODEL:-$M/d1-2b/d1-2B-Q8_0.gguf}   # Q8_0: faster than Q6_K on this engine (prefill-bound)
D1_MMPROJ=${D1_MMPROJ:-$M/d1-2b/mmproj-d1-2B-Q8_0.gguf}
D1=""
if [ -f "$D1_MODEL" ]; then
    D1="--d1-model $D1_MODEL"
    if [ -f "$D1_MMPROJ" ]; then D1="$D1 --d1-mmproj $D1_MMPROJ"; fi
fi
# Disk prompt-cache tier on the SSD0 drive; when it is unplugged or fails, the server uses ~/.cache/liquid-engine.
export LIQUID_CACHE_DIR=${LIQUID_CACHE_DIR:-/run/media/$USER/SSD0/liquid-engine-cache}
# shellcheck disable=SC2086
exec "$HERE/build/liquid-server" -m "$MODEL" $VL $D1 \
    --host 127.0.0.1 --port "${PORT:-11434}" \
    -np "${SLOTS:-4}" -c "${CTX:-100000}" --kv q8 \
    --cache-ram-mib 4096 --cache-disk-mib 16384 \
    "$@"
