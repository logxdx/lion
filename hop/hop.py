#!/usr/bin/env python3
"""hop: one web UI and one OpenAI-compatible endpoint in front of several local inference servers.

The servers (backends.json: liquid-server for lfm3 / lfm3-vl / d1, raon-server for raon-1-nano, ...) are started on
demand, one at a time since they share the GPU: a request for a model whose server is not running waits until the
running server has no request in flight, stops it, starts the other and is then forwarded (streams pass through as
they arrive). A server with nothing to do for `idle_unload_s` is stopped.

The web UI is llama.cpp's (rebranded Lion, `make webui`) in llama-server's router mode: a model picker over every model of
every server. The server tools (web_search, web_fetch, pdf) run here, so using them never loads a model.

Standard library only.   usage: hop.py [--config backends.json] [--listen 127.0.0.1:11434] [--idle S] [--no-tools]
"""
import argparse
import http.client
import http.server
import json
import mimetypes
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    sys.stderr.write(time.strftime("%H:%M:%S ") + "hop: " + msg + "\n")
    sys.stderr.flush()


def path_of(base, p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


# ---------------------------------------------------------------- backends

class Backend:
    def __init__(self, cfg, base):
        self.name = cfg["name"]
        c0 = cfg["command"][0]   # a path (relative to the config file) or a program on the PATH
        self.command = [path_of(base, c0) if ("/" in c0 or c0.startswith("~")) else c0] + cfg["command"][1:]
        self.cwd = base
        self.env = {k: str(v) for k, v in cfg.get("env", {}).items()}
        self.port = int(cfg["port"])
        self.models = cfg["models"]
        self.routes = cfg.get("routes", [])
        self.proc = None
        self.log_path = None

    def chat_models(self):
        return [m for m in self.models if m.get("chat", True)]


class Fleet:
    """Which backend runs. One at a time; swaps wait for the running one to have no request in flight."""

    def __init__(self, backends, start_timeout, idle, log_dir):
        self.backends = backends
        self.start_timeout = start_timeout
        self.idle = idle
        self.log_dir = log_dir
        self.cond = threading.Condition()
        self.active = None       # Backend whose process runs (ready once its /health answered)
        self.ready = False
        self.starting = False    # a thread is stopping / starting a process
        self.want = None         # a backend some request is waiting to swap in
        self.inflight = 0
        self.last_used = time.time()
        self.on_ready = None     # callback(backend) after a start (props cache)
        threading.Thread(target=self._idle_loop, daemon=True).start()

    # -- state machine, under self.cond
    def _next(self, b):
        if self.active is not None and self.active.proc is not None and self.active.proc.poll() is not None:
            log(f"{self.active.name} exited (code {self.active.proc.returncode}); see {self.active.log_path}")
            self.active, self.ready = None, False
        if self.starting:
            return "wait"
        if self.active is b and self.ready and self.want in (None, b):
            self.inflight += 1
            return "ready"
        if self.active is None:
            self.starting, self.want = True, b
            return "start"
        if self.active is not b:
            if self.want is None:
                self.want = b
            if self.want is b and self.inflight == 0:
                self.starting = True
                return "start"
        return "wait"

    def acquire(self, b, on_wait=None):
        """Blocks until b serves requests (starting / swapping as needed); counts one request in flight."""
        while True:
            with self.cond:
                action = self._next(b)
                if action == "wait":
                    self.cond.wait(1.0)
            if action == "ready":
                return
            if action == "start":
                self._swap_to(b, on_wait)
                continue
            if on_wait:
                on_wait()

    def release(self):
        with self.cond:
            self.inflight -= 1
            self.last_used = time.time()
            self.cond.notify_all()

    def _swap_to(self, b, on_wait=None):
        try:
            if self.active is not None:
                self._stop(self.active)
            with self.cond:
                self.active, self.ready = None, False
            self._start(b, on_wait)
            with self.cond:
                self.active, self.ready, self.starting = b, True, False
                if self.want is b:
                    self.want = None
                self.last_used = time.time()
                self.cond.notify_all()
        except Exception:
            with self.cond:
                self.active, self.ready, self.starting, self.want = None, False, False, None
                self.cond.notify_all()
            raise
        if self.on_ready:
            threading.Thread(target=self.on_ready, args=(b,), daemon=True).start()

    def _start(self, b, on_wait=None):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", b.port)) == 0:
                raise RuntimeError(f"port {b.port} for {b.name} is already in use by another process")
        os.makedirs(self.log_dir, exist_ok=True)
        b.log_path = os.path.join(self.log_dir, b.name + ".log")
        logf = open(b.log_path, "ab")
        env = dict(os.environ, **b.env)
        t0 = time.time()
        log(f"starting {b.name}: {' '.join(b.command)}")
        b.proc = subprocess.Popen(b.command, env=env, cwd=b.cwd, stdout=logf, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
        logf.close()
        last_ping = time.time()
        while True:
            if on_wait and time.time() - last_ping >= 1.0:   # e.g. keep the requester's event stream alive
                last_ping = time.time()
                on_wait()
            if b.proc.poll() is not None:
                raise RuntimeError(f"{b.name} exited during start (code {b.proc.returncode}): {tail(b.log_path)}")
            if time.time() - t0 > self.start_timeout:
                self._kill(b)
                raise RuntimeError(f"{b.name} did not become ready in {self.start_timeout} s")
            try:
                c = http.client.HTTPConnection("127.0.0.1", b.port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    log(f"{b.name} ready in {time.time() - t0:.1f} s")
                    return
            except OSError:
                pass
            time.sleep(0.25)

    def _stop(self, b):
        if b.proc is None or b.proc.poll() is not None:
            return
        t0 = time.time()
        try:
            os.killpg(b.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            b.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._kill(b)
        log(f"stopped {b.name} in {time.time() - t0:.1f} s")

    def _kill(self, b):
        try:
            os.killpg(b.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        b.proc.wait()

    def _idle_loop(self):
        while True:
            time.sleep(10)
            with self.cond:
                b = self.active
                idle = (b is not None and self.ready and not self.starting and self.inflight == 0 and self.want is None
                        and time.time() - self.last_used > self.idle)
                if idle:
                    self.starting = True
            if idle:
                log(f"{b.name} idle for {self.idle} s: unloading")
                try:
                    self._stop(b)
                finally:
                    with self.cond:
                        self.active, self.ready, self.starting = None, False, False
                        self.cond.notify_all()

    def shutdown(self):
        with self.cond:
            b = self.active
        if b is not None:
            self._stop(b)

    def running(self, b):
        with self.cond:
            return self.active is b and self.ready


def tail(path, n=600):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


# ---------------------------------------------------------------- server tools (tools.ts: llama-server's /tools contract)

class Tools:
    def __init__(self, tools_dir, node, timeout):
        self.dir, self.node, self.timeout = tools_dir, node, timeout
        self.list, self.error = [], ""
        script = os.path.join(tools_dir, "tools.ts")
        if not os.path.exists(script):
            self.error = script + " not found"
        elif not os.path.exists(os.path.join(tools_dir, "node_modules")):
            self.error = os.path.join(tools_dir, "node_modules") + " missing (make webtools)"
        else:
            try:
                r = subprocess.run([node, script, "list"], capture_output=True, timeout=30)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr.decode(errors="replace")[-500:])
                self.list = json.loads(r.stdout)
            except Exception as e:
                self.error = f"tools.ts list failed: {e}"
        self.enabled = not self.error

    def names(self):
        return [t.get("tool") for t in self.list]

    def call(self, name, params):
        p = subprocess.Popen([self.node, os.path.join(self.dir, "tools.ts"), "call", name], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            out, err = p.communicate(json.dumps(params).encode(), timeout=self.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return {"error": f"{name} timed out after {self.timeout} s"}
        try:
            j = json.loads(out)
            if isinstance(j, dict) and ("plain_text_response" in j or "error" in j):
                return j
        except ValueError:
            pass
        msg = (err or out).decode(errors="replace")[-2000:]
        return {"error": f"{name} failed (exit {p.returncode}): {msg}"}


# ---------------------------------------------------------------- HTTP

API_POST = {"/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/v1/systemone",
            "/tokenize", "/detokenize", "/apply-template", "/infill", "/v1/embeddings"}
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "te", "trailer", "upgrade", "proxy-connection"}


class Gateway:
    def __init__(self, cfg, base, args):
        self.cfg = cfg
        self.backends = [Backend(b, base) for b in cfg["backends"]]
        self.by_model = {m["id"]: b for b in self.backends for m in b.models}
        self.default_model = cfg.get("default_model") or self.backends[0].chat_models()[0]["id"]
        state = path_of(base, cfg.get("state_dir", "~/.cache/hop"))
        self.props_path = os.path.join(state, "props.json")
        try:
            self.props_cache = json.load(open(self.props_path))
        except (OSError, ValueError):
            self.props_cache = {}
        self.fleet = Fleet(self.backends, cfg.get("start_timeout_s", 300),
                           args.idle if args.idle is not None else cfg.get("idle_unload_s", 600), os.path.join(state, "logs"))
        self.fleet.on_ready = self.refresh_props
        # ui_dir: a directory or a list of candidates; the first that exists is used (install: webui, repo: build/webui)
        cands = cfg.get("ui_dir", ["webui", "build/webui"])
        cands = [path_of(base, d) for d in ([cands] if isinstance(cands, str) else cands)]
        self.ui_dir = next((d for d in cands if os.path.isdir(d)), cands[0])
        self.tools = None
        if not args.no_tools and cfg.get("tools", True):
            # tools_dir: a directory or a list of candidates; the first with its node dependencies installed is used
            cands = cfg.get("tools_dir", ["tools/webtools", "servers/liquine/tools/webtools"])
            cands = [path_of(base, d) for d in ([cands] if isinstance(cands, str) else cands)]
            tdir = next((d for d in cands if os.path.isdir(os.path.join(d, "node_modules"))), cands[0])
            self.tools = Tools(tdir, cfg.get("node", "node"), cfg.get("tool_timeout_s", 300))

    def model_cfg(self, mid):
        b = self.by_model.get(mid)
        return next((m for m in b.models if m["id"] == mid), {}) if b else {}

    def refresh_props(self, b):
        """Caches each model's /props while its server runs (the UI asks for them without wanting a model load)."""
        changed = False
        for m in b.chat_models():
            try:
                c = http.client.HTTPConnection("127.0.0.1", b.port, timeout=5)
                c.request("GET", "/props?model=" + urllib.parse.quote(m["id"]))
                r = c.getresponse()
                if r.status == 200:
                    self.props_cache[m["id"]] = json.loads(r.read())
                    changed = True
            except (OSError, ValueError):
                pass
        if changed:
            os.makedirs(os.path.dirname(self.props_path), exist_ok=True)
            json.dump(self.props_cache, open(self.props_path, "w"))

    def props(self, mid):
        if mid in self.props_cache:
            p = dict(self.props_cache[mid])
        else:
            m = self.model_cfg(mid)
            n_ctx = m.get("n_ctx", 0)
            p = {"default_generation_settings": {"n_ctx": n_ctx, "params": {}}, "n_ctx": n_ctx, "total_slots": m.get("slots", 1),
                 "chat_template": "", "cors_proxy_enabled": False, "ui_settings": {}}
        m = self.model_cfg(mid)
        p.update({"model": mid, "model_alias": mid, "build_info": "hop",
                  "modalities": {"vision": bool(m.get("vision")), "audio": False, "video": bool(m.get("video"))}})
        p.pop("role", None)
        return p

    def models(self):
        data = []
        for b in self.backends:
            for m in b.chat_models():
                data.append({"id": m["id"], "object": "model", "created": 0, "owned_by": b.name,
                             # every model reports "loaded": the gateway starts servers itself, the UI never has to
                             "status": {"value": "loaded"},
                             "meta": {"n_ctx": m.get("n_ctx"), "vision": bool(m.get("vision")), "video": bool(m.get("video")),
                                      "server": b.name, "running": self.fleet.running(b)}})
        return {"object": "list", "data": data}

    def backend_for(self, path, body):
        mid = body.get("model") if isinstance(body, dict) else None
        if mid:
            if mid not in self.by_model:
                return None, f"unknown model {mid!r}; available: {', '.join(self.by_model)}"
            return self.by_model[mid], None
        for b in self.backends:
            if path in b.routes:
                return b, None
        with self.fleet.cond:
            if self.fleet.active is not None:
                return self.fleet.active, None
        return self.by_model[self.default_model], None


def make_handler(gw):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "hop"

        def log_message(self, fmt, *a):
            pass

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError):
                pass   # a client that went away

        def send_json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_error_json(self, status, msg, kind=None):
            self.send_json({"error": {"message": msg, "type": kind or ("invalid_request_error" if status < 500 else "server_error"),
                                      "code": status}}, status)

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        # -- GET
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            p = u.path
            if p == "/health":
                return self.send_json({"status": "ok"})
            if p in ("/v1/models", "/models"):
                return self.send_json(gw.models())
            if p == "/props":
                mid = (q.get("model") or [""])[0]
                if mid:
                    if mid not in gw.by_model:
                        return self.send_error_json(404, "unknown model " + mid)
                    return self.send_json(gw.props(mid))
                rp = gw.props(gw.default_model)
                rp["role"] = "router"   # llama-server's router mode: the UI shows a model picker over /models
                return self.send_json(rp)
            if p == "/tools":
                if gw.tools is None or not gw.tools.enabled:
                    return self.send_error_json(403, "server tools are disabled", "permission_error")
                return self.send_json(gw.tools.list)
            if p == "/slots":
                with gw.fleet.cond:
                    b = gw.fleet.active if gw.fleet.ready else None
                return self.proxy(b, b"") if b else self.send_json([])
            if p == "/v1/catalog.json":
                return self.send_json([])
            if p == "/models/sse":
                return self.keepalive_sse()
            return self.static(p)

        # -- POST
        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            p = u.path
            raw = self.body()
            if p == "/tools":
                return self.tool_call(raw)
            if p == "/v1/streams/lookup":
                return self.send_json([])
            if p in ("/models/load", "/models/unload"):
                return self.send_json({"success": True})
            if p not in API_POST:
                return self.send_error_json(404, "not found: " + p)
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                return self.send_error_json(400, "invalid JSON body")
            b, err = gw.backend_for(p, body)
            if b is None:
                return self.send_error_json(404, err)
            self.proxy(b, raw, stream=isinstance(body, dict) and bool(body.get("stream")))

        def tool_call(self, raw):
            if gw.tools is None or not gw.tools.enabled:
                return self.send_error_json(403, "server tools are disabled", "permission_error")
            try:
                b = json.loads(raw)
            except ValueError:
                b = None
            if not isinstance(b, dict) or not isinstance(b.get("tool"), str):
                return self.send_error_json(400, 'expected {"tool": name, "params": {...}}')
            name = b["tool"]
            if name not in gw.tools.names():
                return self.send_error_json(404, f'unknown tool "{name}"')
            if b.get("stream"):
                return self.send_error_json(404, f'tool "{name}" does not support stream = true')
            params = b.get("params") if isinstance(b.get("params"), dict) else {}
            params.pop("cwd", None)   # as llama-server: the working directory comes only from the x-tool-cwd header
            if self.headers.get("x-tool-cwd"):
                params["cwd"] = self.headers["x-tool-cwd"]
            t0 = time.time()
            out = gw.tools.call(name, params)
            log(f"tool {name}: {'error' if 'error' in out else 'ok'} in {(time.time() - t0) * 1000:.0f} ms")
            self.send_json(out)

        # -- forwarding
        def proxy(self, b, raw, stream=False):
            sent_head = [False]

            def ping():   # while a server starts: open the event stream early and keep it alive
                if not stream:
                    return
                if not sent_head[0]:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    sent_head[0] = True
                self.chunk(b": loading " + b.name.encode() + b"\n\n")

            try:
                gw.fleet.acquire(b, ping)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:
                log(f"cannot start {b.name}: {e}")
                return self.fail(sent_head[0], 503, f"cannot start {b.name}: {e}")
            try:
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
                headers["Content-Length"] = str(len(raw))
                c = http.client.HTTPConnection("127.0.0.1", b.port, timeout=3600)
                try:
                    c.request(self.command, self.path, body=raw if self.command == "POST" else None, headers=headers)
                    r = c.getresponse()
                except OSError as e:
                    return self.fail(sent_head[0], 502, f"{b.name}: {e}")
                if sent_head[0]:
                    if r.status != 200:   # the stream is already open: report the error as an event
                        msg = r.read().decode(errors="replace")
                        self.chunk(b"data: " + json.dumps({"error": {"message": msg, "code": r.status}}).encode() + b"\n\n")
                        return self.chunk(b"")
                    return self.pipe(r, c)
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                length = r.getheader("Content-Length")
                if length is not None and r.getheader("Transfer-Encoding") is None:
                    data = r.read()
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.pipe(r, c)
            except (BrokenPipeError, ConnectionResetError):
                pass   # the client left; closing the upstream connection cancels the request there
            finally:
                gw.fleet.release()

        def pipe(self, r, c):
            try:
                while True:
                    data = r.read1(65536)
                    if not data:
                        break
                    self.chunk(data)
                self.chunk(b"")
            finally:
                c.close()

        def chunk(self, data):
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        def fail(self, head_sent, status, msg):
            if head_sent:
                self.chunk(b"data: " + json.dumps({"error": {"message": msg, "code": status}}).encode() + b"\n\n")
                self.chunk(b"")
            else:
                self.send_error_json(status, msg)

        def keepalive_sse(self):   # router-mode status feed: nothing changes that the UI must know
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for _ in range(6):
                    self.chunk(b":\n\n")
                    time.sleep(10)
                self.chunk(b"")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def static(self, p):
            if not os.path.isdir(gw.ui_dir):
                return self.send_error_json(404, f"web UI not built ({gw.ui_dir}; run: make webui)")
            rel = urllib.parse.unquote(p).lstrip("/") or "index.html"
            full = os.path.realpath(os.path.join(gw.ui_dir, rel))
            root = os.path.realpath(gw.ui_dir)
            if not full.startswith(root + os.sep) or not os.path.isfile(full):
                return self.send_error_json(404, "not found: " + p)
            with open(full, "rb") as f:
                data = f.read()
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if full.endswith(".webmanifest"):
                ctype = "application/manifest+json"
            elif full.endswith((".js", ".mjs")):
                ctype = "text/javascript"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            # content-named bundle files are cached for good; index, icons, manifest and service worker revalidate
            self.send_header("Cache-Control", "public, max-age=31536000, immutable" if p.startswith("/_app/immutable/") else "no-cache")
            self.end_headers()
            self.wfile.write(data)

    return H


def main():
    ap = argparse.ArgumentParser(description="hop: web UI + OpenAI API gateway over local inference servers")
    ap.add_argument("--config", default=os.path.join(HERE, "backends.json"))
    ap.add_argument("--listen", help="host:port (default from the config, 127.0.0.1:11434)")
    ap.add_argument("--idle", type=float, help="seconds without requests before a server is stopped (default 600)")
    ap.add_argument("--no-tools", action="store_true", help="do not serve /tools")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    base = os.path.dirname(os.path.abspath(args.config))
    gw = Gateway(cfg, base, args)
    host, port = (args.listen or cfg.get("listen", "127.0.0.1:11434")).rsplit(":", 1)
    srv = http.server.ThreadingHTTPServer((host, int(port)), make_handler(gw))
    srv.daemon_threads = True

    def stop(*_):
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for b in gw.backends:
        log(f"server {b.name} (port {b.port}): models {', '.join(m['id'] for m in b.models)}")
    log(f"tools: {', '.join(gw.tools.names()) if gw.tools and gw.tools.enabled else (gw.tools.error if gw.tools else 'off')}")
    log(f"web UI: {gw.ui_dir if os.path.isdir(gw.ui_dir) else 'not built (make webui)'}")
    log(f"idle unload after {gw.fleet.idle:.0f} s; listening on http://{host}:{port}/ (default model {gw.default_model})")
    try:
        srv.serve_forever()
    finally:
        gw.fleet.shutdown()
        log("bye")


if __name__ == "__main__":
    main()
