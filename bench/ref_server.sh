#!/bin/sh
# Start a reference llama-server for the parity tests (background) and wait until healthy.
# usage: bench/ref_server.sh <llama.cpp bin dir> <model.gguf> <port>
BIN=$1; M=$2; PORT=$3
LD_LIBRARY_PATH=$BIN nohup "$BIN/llama-server" -m "$M" -ngl 99 -c 8192 -fa on -np 1 --host 127.0.0.1 --port "$PORT" \
    > "/tmp/liquid-ref-$PORT.log" 2>&1 &
echo "pid $!"
for i in $(seq 1 90); do curl -s "localhost:$PORT/health" 2>/dev/null | grep -q ok && { echo ready; exit 0; }; sleep 1; done
echo "server did not come up"; exit 1
