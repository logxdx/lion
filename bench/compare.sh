#!/bin/sh
# Head-to-head: liquid vs llama.cpp llama-server (same GGUF, same GPU, run one at a time).
# usage: bench/compare.sh <llama.cpp bin dir> <model.gguf>... ; prints a markdown table
LLAMA_BIN=$1; shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
PORT=18090
SIZES="512 2048 8192"
echo "| model | engine | pp512 | pp2048 | pp8192 | tg128 @512 | tg128 @2048 | tg128 @8192 |"
echo "|---|---|---|---|---|---|---|---|"
for M in "$@"; do
  name=$(basename "$M" .gguf)
  # --- llama.cpp ---
  LD_LIBRARY_PATH=$LLAMA_BIN "$LLAMA_BIN/llama-server" -m "$M" -ngl 99 -c 12288 -fa on -np 1 \
      --host 127.0.0.1 --port $PORT > /tmp/liquid-compare-server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 90); do curl -s localhost:$PORT/health 2>/dev/null | grep -q ok && break; sleep 1; done
  pp=""; tg=""
  for n in $SIZES; do
    r=$(python3 "$HERE/bench/llama_server_bench.py" http://127.0.0.1:$PORT --pp $n --tg 128 --reps 3)
    pp="$pp | $(echo "$r" | sed -E 's/.*pp[0-9]+: ([0-9.]+).*/\1/')"
    tg="$tg | $(echo "$r" | sed -E 's/.*tg[0-9]+: ([0-9.]+).*/\1/')"
  done
  kill $SPID; wait $SPID 2>/dev/null
  echo "| $name | llama.cpp$pp$tg |"
  # --- liquid ---
  pp=""; tg=""
  for n in $SIZES; do
    r=$("$HERE/build/liquid" -m "$M" --bench --pp $((n + 1)) --tg 128 --reps 3 -q -c 12288)
    pp="$pp | $(echo "$r" | grep "| pp" | awk -F'|' '{print $4}' | awk '{print $1}')"
    tg="$tg | $(echo "$r" | grep "| tg" | awk -F'|' '{print $4}' | awk '{print $1}')"
  done
  echo "| $name | liquid$pp$tg |"
done
