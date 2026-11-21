#!/bin/sh
# Start liquid-server for agents: 4 concurrent sequences x 100k context, q8_0 KV, RAM + disk prompt cache.
#   usage: [MODEL=...] [PORT=8080] [SLOTS=4] [CTX=100000] ./serve.sh [extra liquid-server args...]
# Clients: base URL http://127.0.0.1:$PORT/v1, model "lfm3".
# Measured on the RTX 4050 6 GB: 4 x 100k with Q8_0 weights uses ~4.4 GB of VRAM; pinned RAM tier 4 GiB.
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-$HOME/llama.cpp/models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf}
exec "$HERE/build/liquid-server" -m "$MODEL" \
    --host 127.0.0.1 --port "${PORT:-8080}" \
    -np "${SLOTS:-4}" -c "${CTX:-100000}" --kv q8 \
    --cache-ram-mib 4096 --cache-disk-mib 16384 \
    "$@"
