#!/bin/sh
# A/B the KV cache type: per-kernel attention timings (profile) at several context lengths.
# usage: bench/kv_ab.sh <liquid> <model.gguf> [contexts...]
BIN=$1; M=$2; shift 2
CTXS=${*:-"2048 8192 32768"}
for n in $CTXS; do
  for kv in f16 q8; do
    out=$("$BIN" -m "$M" --profile --pp "$n" -q -c $((n + 256)) --kv "$kv" 2>&1)
    pf=$(echo "$out" | grep "^prefill" | sed -E 's/.*\(([0-9]+) t\/s\)/\1/')
    pa=$(echo "$out" | grep "^attn(global)" | awk '{print $2}')
    pw=$(echo "$out" | grep "^attn.rope+kv_write" | awk '{print $2}')
    dg=$(echo "$out" | grep "attn.core(global)" | awk '{printf "%.1f us/call %s GB/s", $3/$2, $5}')
    ds=$(echo "$out" | grep "attn.core(swa)" | awk '{printf "%.1f us/call", $3/$2}')
    tot=$(echo "$out" | grep "^total" | awk '{print $2}')
    printf "ctx %-6s kv %-4s | prefill %6s t/s, attn(global) %8s ms, kv_write %6s ms | decode attn global %s, swa %s | eager step %s us\n" \
      "$n" "$kv" "$pf" "$pa" "$pw" "$dg" "$ds" "$tot"
  done
done
