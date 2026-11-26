#!/bin/sh
# Validation of mixed chunks (a prefill run + decode rows of other slots in one batch) against a baseline build of main.
# Needs an otherwise idle GPU (aborts if another process holds it). The A/B rows alternate baseline and new binary.
# usage: tests/validate_mixed.sh MODEL.gguf [BASE_BIN (default build/baseline/raon)] [PARITY_REF.json DOC.txt]
#   with PARITY_REF/DOC: also tests/parity.py --long against the stored llama.cpp reference (needs the KRAFTON build)
M=$1; BASE=${2:-build/baseline/raon}; NEW=build/raon; REF=$3; DOC=$4
[ -f "$M" ] || { echo "usage: $0 MODEL.gguf [BASE_BIN] [PARITY_REF.json DOC.txt]"; exit 1; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
[ -z "$busy" ] || { echo "GPU busy: $busy"; exit 1; }
make -s build/raon || exit 1
set -x

# 1. the fed slot must match a plain prefill exactly; decode rows are reported (chaotic model: judge vs noise floor)
$NEW -m "$M" --test-mixed --chunk 256 --prefill 3000 -n 48
$NEW -m "$M" --test-mixed --chunk 1024 --prefill 6000 -n 48
if [ -n "$REF" ]; then
  python3 tests/parity.py --server ~/llama.cpp/raon-src/build/bin/llama-server --model "$M" --raon $NEW -n 64 \
      --ref "$REF" --ref-kv q8_0 --long "$DOC"
fi

# 2. A/B decode and prefill (must be unchanged: pure decode / pure prefill chunks take the same code)
sh tests/ab.sh $BASE $NEW "$M" 4 131072 4
for r in 1 2; do
  $BASE -m "$M" --bench --prefill 32768 | tail -1
  $NEW -m "$M" --bench --prefill 32768 | tail -1
done

# 3. mixed steps vs alternating chunk + decode step: 3 streams decoding at 128k, a 4th prompt at depth 0 / 64k
$NEW -m "$M" --bench-mixed --slots 3 --depth 131072 --pf-depth 0
$NEW -m "$M" --bench-mixed --slots 3 --depth 131072 --pf-depth 65536
