#!/bin/sh
# Regression: a 2-turn interactive chat (incremental KV/conv state) must produce the same greedy answer as the
# same conversation prefilled from scratch.  usage: tests/multiturn.sh <liquid> <model.gguf>
BIN=$1; M=$2
inc=$(printf 'What is 12*13? Reply with just the number.\nNow add 6 to it. Just the number.\n/exit\n' |
      "$BIN" -m "$M" -i --no-think -n 64 -q --greedy 2>/dev/null | grep -v '^$' | sed -n 2p | sed 's/^> //')
first=$(printf 'What is 12*13? Reply with just the number.\n/exit\n' |
      "$BIN" -m "$M" -i --no-think -n 64 -q --greedy 2>/dev/null | grep -v '^$' | sed -n 1p | sed 's/^> //')
P=$(printf '<|startoftext|><|im_start|>user\nWhat is 12*13? Reply with just the number.<|im_end|>\n<|im_start|>assistant\n<think></think>%s<|im_end|>\n<|im_start|>user\nNow add 6 to it. Just the number.<|im_end|>\n<|im_start|>assistant\n<think></think>' "$first")
fresh=$("$BIN" -m "$M" --raw -q --greedy -n 64 -p "$P" 2>/dev/null | head -1)
echo "turn1: $first | turn2 incremental: $inc | turn2 fresh: $fresh"
[ "$inc" = "$fresh" ] && echo PASS || { echo FAIL; exit 1; }
