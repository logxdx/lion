#!/bin/sh
# Start a server in the background, record its PID in a file, wait until healthy.
# usage: bench/run_server.sh <pidfile> <port> <logfile> <command...>
PIDFILE=$1; PORT=$2; LOG=$3; shift 3
"$@" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
for i in $(seq 1 120); do
  curl -s "localhost:$PORT/health" 2>/dev/null | grep -q ok && { echo "up (pid $(cat "$PIDFILE"))"; exit 0; }
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "server exited:"; tail -5 "$LOG"; exit 1; }
  sleep 1
done
echo "timeout"; exit 1
