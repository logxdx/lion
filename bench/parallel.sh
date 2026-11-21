#!/bin/sh
# Concurrency sweep for liquid: N streams x (pp prompt + tg generated) each, batched decode.
# usage: bench/parallel.sh <liquid> <model.gguf> <pp> <kv f16|q8> [streams...]
BIN=$1; M=$2; PP=$3; KV=$4; shift 4
NS=${*:-"1 2 3 4"}
echo "| model | KV | streams | ctx/stream | prefill t/s | decode aggregate t/s | per stream t/s | GPU MiB |"
echo "|---|---|---|---|---|---|---|---|"
for n in $NS; do
  if [ "$n" = 1 ]; then
    r=$("$BIN" -m "$M" --bench --pp $((PP + 1)) --tg 128 --reps 2 -q -c $((PP + 256)) --kv "$KV")
    pp=$(echo "$r" | grep "| pp" | awk -F'|' '{print $4}' | awk '{print $1}')
    tg=$(echo "$r" | grep "| tg" | awk -F'|' '{print $4}' | awk '{print $1}')
    echo "| $(basename "$M" .gguf) | $KV | 1 | $PP | $pp | $tg | $tg | |"
  else
    "$BIN" -m "$M" --bench --parallel "$n" --pp $((PP + 1)) --tg 128 --reps 2 -q -c $((PP + 256)) --kv "$KV" |
      tail -1 | awk -F'|' -v kv="$KV" '{printf "|%s| %s |%s|%s|%s|%s|%s|%s|\n", $2, kv, $3, $4, $5, $6, $7, $8}'
  fi
done
