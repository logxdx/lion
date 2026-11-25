#!/bin/sh
# Interleaved A/B decode benchmark of two raon binaries; rounds overlapped by another GPU process are discarded.
# usage: tests/ab.sh BIN_A BIN_B MODEL SLOTS DEPTH [ROUNDS]
A=$1; B=$2; M=$3; S=$4; D=$5; N=${6:-4}
foreign() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -e raon -e '^$' ; }
r=0
while [ $r -lt $N ]; do
  while [ -n "$(foreign)" ]; do sleep 5; done
  a=$("$A" -m "$M" --bench --slots "$S" --depth "$D" --steps 12 2>/dev/null | sed -n 's/.*step \([0-9.]*\) ms.*/\1/p')
  f1=$(foreign)
  b=$("$B" -m "$M" --bench --slots "$S" --depth "$D" --steps 12 2>/dev/null | sed -n 's/.*step \([0-9.]*\) ms.*/\1/p')
  f2=$(foreign)
  if [ -n "$f1$f2" ]; then echo "round discarded (other GPU process: $f1 $f2)"; continue; fi
  echo "round $r: A ${a} ms  B ${b} ms"
  r=$((r + 1))
done
