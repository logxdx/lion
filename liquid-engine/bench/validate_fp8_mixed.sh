#!/bin/sh
# Validation of the int8/FP8 prefill flash attention (src/flash8.cuh) and of mixed steps (Engine::mixed_step) against a
# baseline build of main. Needs an otherwise idle GPU (aborts if another process holds it). Prints everything; the
# A/B rows alternate baseline and new binary (the laptop throttles, so only interleaved runs are comparable).
# usage: bench/validate_fp8_mixed.sh MODEL.gguf [BASE_BIN (default build/baseline/liquid)]
M=$1; BASE=${2:-build/baseline/liquid}; NEW=build/liquid
[ -f "$M" ] || { echo "usage: $0 MODEL.gguf [BASE_BIN]"; exit 1; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
[ -z "$busy" ] || { echo "GPU busy: $busy"; exit 1; }
make -s build/liquid build/flash8_check || exit 1
set -x

# 1. kernel: int8+FP8 vs f16 flash attention vs an f64 reference, and kernel time at 32k / 128k context
build/flash8_check 32768
build/flash8_check 131072

# 2. end-to-end accuracy: q8 KV with the new kernel vs the f16 kernel (LIQUID_FA16=1) vs exact
python3 tests/kv_compare.py $NEW "$M"
LIQUID_FA16=1 python3 tests/kv_compare.py $NEW "$M"

# 3. correctness of the batched / mixed paths
$NEW -m "$M" --test-multi 4
$NEW -m "$M" --test-mixed --tg 64
sh tests/multiturn.sh $NEW "$M"

# 4. A/B prefill (interleaved): baseline, new, new with the f16 kernel
for pp in 2048 8192 32768 131072; do
  for r in 1 2; do
    $BASE -m "$M" --bench --pp $pp --tg 16 --reps 2 -q | grep pp
    $NEW -m "$M" --bench --pp $pp --tg 16 --reps 2 -q | grep pp
    LIQUID_FA16=1 $NEW -m "$M" --bench --pp $pp --tg 16 --reps 2 -q | grep pp
  done
done

# 5. A/B decode (must be unchanged: the decode path was not touched)
for r in 1 2 3; do
  $BASE -m "$M" --bench --pp 512 --tg 256 -q | grep tg
  $NEW -m "$M" --bench --pp 512 --tg 256 -q | grep tg
done
for r in 1 2; do
  $BASE -m "$M" --bench --parallel 4 --pp 32768 --tg 128 -q | tail -1
  $NEW -m "$M" --bench --parallel 4 --pp 32768 --tg 128 -q | tail -1
done

# 6. mixed steps vs alternating chunk + decode step: 3 streams decoding, a 4th prompt at depth 0 / 64k
$NEW -m "$M" --bench-mixed --parallel 3 --pp 32768 --pf-depth 0 -q
$NEW -m "$M" --bench-mixed --parallel 3 --pp 32768 --pf-depth 65536 -q
