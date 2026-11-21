#!/bin/sh
# End-to-end (graph mode) prefill/decode for both KV cache types at several context lengths.
# usage: bench/kv_e2e.sh <liquid> <model.gguf> [contexts...]
BIN=$1; M=$2; shift 2
CTXS=${*:-"2048 8192 32768"}
echo "| ctx | KV | prefill t/s | decode t/s (128 tok) |"
echo "|---|---|---|---|"
for n in $CTXS; do
  for kv in f16 q8; do
    r=$("$BIN" -m "$M" --bench --pp $((n + 1)) --tg 128 --reps 2 -q -c $((n + 256)) --kv $kv)
    pp=$(echo "$r" | grep "| pp" | awk -F'|' '{print $4}' | awk '{print $1}')
    tg=$(echo "$r" | grep "| tg" | awk -F'|' '{print $4}' | awk '{print $1}')
    echo "| $n | $kv | $pp | $tg |"
  done
done
