#!/bin/sh
# Sweep decode-attention split parameters (compile-time) and report per-call times from --profile.
# usage: bench/attn_sweep.sh <model.gguf> "<SMAX values>" "<MIN_CHUNK values>" [contexts]
M=$1; SMAXS=$2; CHUNKS=$3; CTXS=${4:-"2048 32768"}
D=$(mktemp -d)
for sm in $SMAXS; do for mc in $CHUNKS; do
  /opt/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 --use_fast_math -DATT_SMAX=$sm -DATT_MIN_CHUNK=$mc -c src/engine.cu -o "$D/e.o" 2>/dev/null || { echo "build failed"; continue; }
  /opt/cuda/bin/nvcc -arch=sm_89 "$D/e.o" build/tokenizer.o build/main.o -o "$D/liquid" -lcudart
  for n in $CTXS; do for kv in f16 q8; do
    r=$("$D/liquid" -m "$M" --profile --pp "$n" -q -c $((n + 256)) --kv $kv 2>&1 | grep -E "attn.core" | awk '{printf "%s %.1f  ", $1, $3/$2}')
    echo "SMAX=$sm MIN_CHUNK=$mc ctx=$n kv=$kv: $r"
  done; done
done; done
rm -rf "$D"
