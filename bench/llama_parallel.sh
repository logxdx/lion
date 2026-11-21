#!/bin/sh
# llama.cpp concurrency sweep: llama-server -np N (q8_0 KV, flash attention, per-slot context = PP+512).
# usage: bench/llama_parallel.sh <llama bin dir> <model.gguf> <pp> [streams...]
BIN=$1; M=$2; PP=$3; shift 3
NS=${*:-"1 2 3 4"}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PORT=18091
for n in $NS; do
  LD_LIBRARY_PATH=$BIN "$BIN/llama-server" -m "$M" -ngl 99 -fa on -np "$n" -c $((n * (PP + 512))) -ctk q8_0 -ctv q8_0 \
      --host 127.0.0.1 --port $PORT > /tmp/liquid-llama-parallel.log 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do curl -s localhost:$PORT/health 2>/dev/null | grep -q ok && break; sleep 1; done
  python3 "$HERE/bench/llama_parallel.py" http://127.0.0.1:$PORT --n "$n" --pp "$PP" --tg 128 --reps 2
  kill $SPID; wait $SPID 2>/dev/null
done
