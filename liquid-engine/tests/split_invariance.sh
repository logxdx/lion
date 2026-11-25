#!/bin/sh
# Decode attention must be invariant (up to float reordering, ~1e-5 logits) to how the context is split
# across CTAs. Exact mode has no int8 activation rounding, so differences expose combine bugs.
# usage: tests/split_invariance.sh <model.gguf> <prompt-file>
M=$1; PF=$2
D=$(mktemp -d)
for mc in 32 128 4096; do
  /opt/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 --use_fast_math -DATT_MIN_CHUNK=$mc -c src/engine.cu -o "$D/e.o" 2>/dev/null
  /opt/cuda/bin/nvcc -arch=sm_89 "$D/e.o" build/tokenizer.o build/main.o -o "$D/liquid$mc" -lcudart
done
for kv in f16 q8; do
  for mc in 32 128 4096; do
    r=$("$D/liquid$mc" -m "$M" -c 8192 --raw -q -f "$PF" --dump-logits --exact --kv $kv | sed -n 2,4p | awk -F'\t' '{printf "%s:%s ", $1, $2}')
    echo "kv=$kv MIN_CHUNK=$mc (exact): $r"
  done
done
rm -rf "$D"
