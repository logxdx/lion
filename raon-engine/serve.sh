#!/bin/sh
# Start raon-server: Raon-1 nano, 4 concurrent sequences x 128k context.
#   usage: [MODEL=...] [PORT=11435] [SLOTS=4] [CTX=131072] ./serve.sh [extra raon-server args...]
# Web UI (llama.cpp's, with web_search / web_fetch / pdf tools): http://127.0.0.1:$PORT/
# Clients: base URL http://127.0.0.1:$PORT/v1, model "raon-1-nano".
# Measured on the RTX 4050 6 GB (Q6_K, 4 x 128k, 4 state checkpoints per slot): 5027 MiB; decode ~60 tok/s per stream
# with all four streams at 128k. The default port differs from liquid-serve (11434) so both can be installed.
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)   # resolves symlinks (e.g. ~/.local/bin/raon-serve)
MODEL=${MODEL:-$HOME/models/raon-1-nano/raon-1-nano-Q6_K_MTP.gguf}
exec "$HERE/build/raon-server" -m "$MODEL" \
    --host 127.0.0.1 --port "${PORT:-11435}" \
    --slots "${SLOTS:-4}" --ctx "${CTX:-131072}" \
    "$@"
