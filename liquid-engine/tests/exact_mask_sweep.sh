#!/bin/sh
# Which block's int8 activation path drifts most from the f32 reference? (LIQUID_EXACT_MASK bits:
# 1 conv, 2 attention, 4 ffn, 8 head; 15 = all exact)
# usage: tests/exact_mask_sweep.sh <liquid> <model.gguf> "<prompt>"
BIN=$1; MODEL=$2; PROMPT=$3
for m in 0 1 2 4 8 14 13 11 7 15; do
  printf "mask %2s: " "$m"
  LIQUID_EXACT_MASK=$m "$BIN" -m "$MODEL" -c 4096 --raw -q -p "$PROMPT" --dump-logits | sed -n 2,3p | awk -F'\t' '{printf "%s %.4f  ", $1, $3}'
  echo
done
