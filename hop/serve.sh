#!/bin/sh
# Start hop: the Lion web UI and one OpenAI-compatible API on http://127.0.0.1:11434/ (base URL .../v1),
# in front of liquid-engine (lfm3, lfm3-vl, d1) and raon-engine (raon-1-nano), each started on demand (one at a time:
# they share the 6 GB GPU) and stopped after 10 minutes idle. Servers: backends.json (or HOP_CONFIG=...).
#   usage: [HOP_CONFIG=...] ./serve.sh [--listen 127.0.0.1:11434] [--idle 600] [--no-tools]
# Server logs: ~/.cache/hop/logs/<server>.log
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)   # resolves symlinks (e.g. ~/.local/bin/hop)
exec python3 "$HERE/hop.py" --config "${HOP_CONFIG:-$HERE/backends.json}" "$@"
