#!/bin/sh
# Build engine variants with extra nvcc defines and dump top logits for one prompt (q8 KV), for bisecting
# numeric changes.  usage: tests/bisect_build.sh <model.gguf> <prompt-file> "<-D flags>"...
M=$1; PF=$2; shift 2
D=$(mktemp -d)
for flags in "$@"; do
  /opt/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 --use_fast_math $flags -c src/engine.cu -o "$D/e.o" 2>/dev/null || { echo "build failed: $flags"; continue; }
  /opt/cuda/bin/nvcc -arch=sm_89 "$D/e.o" build/tokenizer.o build/main.o -o "$D/liquid" -lcudart
  r=$("$D/liquid" -m "$M" -c 8192 --raw -q -f "$PF" --dump-logits --kv q8 | sed -n 2,4p | awk -F'\t' '{printf "%s:%s ", $1, $2}')
  echo "[$flags] $r"
done
rm -rf "$D"
