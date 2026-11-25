#!/bin/sh
# Head-to-head: liquid vs llama.cpp llama-server (same GGUF, same GPU, run one at a time).
# usage: [KV=f16|q8] [SIZES="512 2048 8192"] bench/compare.sh <llama.cpp bin dir> <model.gguf>...
# KV=q8 runs llama.cpp with -ctk q8_0 -ctv q8_0 and liquid with --kv q8.
LLAMA_BIN=$1; shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
PORT=18090
KV=${KV:-f16}
SIZES=${SIZES:-"512 2048 8192"}
MAXN=0; for n in $SIZES; do [ "$n" -gt "$MAXN" ] && MAXN=$n; done
CTX=$((MAXN + 512))
if [ "$KV" = q8 ]; then LKV="-ctk q8_0 -ctv q8_0"; else LKV=""; fi
hdr="| model | engine | KV"; sep="|---|---|---"
for n in $SIZES; do hdr="$hdr | pp$n"; sep="$sep|---"; done
for n in $SIZES; do hdr="$hdr | tg128 @$n"; sep="$sep|---"; done
echo "$hdr |"; echo "$sep|"
for M in "$@"; do
  name=$(basename "$M" .gguf)
  # --- llama.cpp ---
  # shellcheck disable=SC2086
  LD_LIBRARY_PATH=$LLAMA_BIN "$LLAMA_BIN/llama-server" -m "$M" -ngl 99 -c $CTX -fa on -np 1 $LKV \
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
  echo "| $name | llama.cpp | $KV$pp$tg |"
  # --- liquid ---
  pp=""; tg=""
  for n in $SIZES; do
    r=$("$HERE/build/liquid" -m "$M" --bench --pp $((n + 1)) --tg 128 --reps 3 -q -c $CTX --kv "$KV")
    pp="$pp | $(echo "$r" | grep "| pp" | awk -F'|' '{print $4}' | awk '{print $1}')"
    tg="$tg | $(echo "$r" | grep "| tg" | awk -F'|' '{print $4}' | awk '{print $1}')"
  done
  echo "| $name | liquid | $KV$pp$tg |"
done
