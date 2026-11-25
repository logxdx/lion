#!/usr/bin/env python3
"""A stand-in for an inference server (for tests/gateway_test.py): /health after a start delay, /props, and
/v1/chat/completions answering "<name>:<model>" as a stream of 5 chunks 0.3 s apart (or at once). Logs cancellations."""
import argparse
import http.server
import json
import signal
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--start-delay", type=float, default=1.0)
args = ap.parse_args()
time.sleep(args.start_delay)   # "loading the model"


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, obj):
        d = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(d)))
        self.end_headers()
        self.wfile.write(d)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self.send_json({"status": "ok"})
        if self.path.startswith("/props"):
            return self.send_json({"model_alias": args.name, "default_generation_settings": {"n_ctx": 4096, "params": {}},
                                   "total_slots": 1, "backend": args.name})
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        text = f"{args.name}:{body.get('model')}"
        if not body.get("stream"):
            return self.send_json({"choices": [{"message": {"role": "assistant", "content": text}}], "backend": args.name})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for i in range(5):
                d = b"data: " + json.dumps({"choices": [{"delta": {"content": f"{text}#{i} "}}]}).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n" % len(d) + d + b"\r\n")
                self.wfile.flush()
                time.sleep(0.3)
            d = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n" % len(d) + d + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            sys.stderr.write(f"{args.name}: client gone, generation cancelled\n")
            sys.stderr.flush()


srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), H)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
srv.serve_forever()
