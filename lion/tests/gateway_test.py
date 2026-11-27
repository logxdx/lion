#!/usr/bin/env python3
"""Gateway behaviour on fake backends (no GPU): routing by model, on-demand start, swap after in-flight requests,
streaming with pings while a server starts, client disconnect, idle unload, router-mode endpoints.

    python3 tests/gateway_test.py        (starts lion.py with tests/fake_backends.json on :11444)
"""
import http.client
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 11444
fails = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        fails.append(name)


def req(method, path, body=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data


def stream(model, close_after=None, out=None, close_on_ping=False):
    """Returns (seconds to first byte, text, events, pings)."""
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps({"model": model, "stream": True, "messages": []}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    first, text, events, pings = None, "", 0, 0
    while True:
        line = r.readline()
        if not line:
            break
        if first is None:
            first = time.time() - t0
        line = line.strip()
        if line.startswith(b":"):
            pings += 1
            if close_on_ping:
                c.close()
                break
        if line.startswith(b"data: ") and line != b"data: [DONE]":
            text += json.loads(line[6:])["choices"][0]["delta"]["content"]
            events += 1
            if close_after and events >= close_after:
                c.close()
                break
        if line == b"data: [DONE]":
            break
    res = (first, text, events, pings, time.time() - t0)
    if out is not None:
        out.append(res)
    return res


def models_running():
    _, m = req("GET", "/v1/models")
    return {x["id"]: x["meta"]["running"] for x in m["data"]}


if os.path.exists("/tmp/lion-test/props.json"):   # this test's own props cache, from an earlier run
    os.remove("/tmp/lion-test/props.json")
PROMPT = "/tmp/lion-test/system_prompt.txt"   # fake_backends.json: system_prompt_file
os.makedirs("/tmp/lion-test", exist_ok=True)
open(PROMPT, "w").write("Use LaTeX for math.\n")
gw = subprocess.Popen([sys.executable, os.path.join(HERE, "..", "lion.py"), "--config", os.path.join(HERE, "fake_backends.json")],
                      stderr=open("/tmp/lion-test-gw.log", "w"))
try:
    for _ in range(40):
        try:
            if req("GET", "/health")[0] == 200:
                break
        except OSError:
            time.sleep(0.25)
    # router mode
    s, m = req("GET", "/v1/models")
    check("models listed", [x["id"] for x in m["data"]] == ["a1", "a2", "b1", "c1"], [x["id"] for x in m["data"]])
    s, p = req("GET", "/props")
    check("router props", p.get("role") == "router")
    check("UI defaults from the config + system_prompt_file",
          p.get("ui_settings") == {"theme": "dark", "systemMessage": "Use LaTeX for math."}, p.get("ui_settings"))
    open(PROMPT, "w").write("Edited prompt.")
    s, p = req("GET", "/props?model=b1")
    check("system prompt edits apply without a restart", p.get("ui_settings", {}).get("systemMessage") == "Edited prompt.",
          p.get("ui_settings"))
    s, p = req("GET", "/props?model=b1")
    check("props of a model never loaded", s == 200 and p["model_alias"] == "b1" and p["n_ctx"] == 8192)
    check("nothing started by listing / props", not any(models_running().values()))
    s, e = req("POST", "/v1/chat/completions", {"model": "zz", "messages": []})
    check("unknown model -> 404", s == 404)
    # on-demand start
    t = time.time()
    s, r = req("POST", "/v1/chat/completions", {"model": "a1", "messages": []})
    check("a1 starts fakeA", s == 200 and r["choices"][0]["message"]["content"] == "fakeA:a1", f"{time.time() - t:.1f} s")
    s, r = req("POST", "/v1/chat/completions", {"model": "a2", "messages": []})
    check("a2 on the same server, no swap", r["choices"][0]["message"]["content"] == "fakeA:a2")
    s, p = req("GET", "/props?model=a1")
    check("props cached from the running server", p.get("backend") == "fakeA")
    # swap with a streamed request: pings while fakeB starts, then its stream
    first, text, ev, pings, dt = stream("b1")
    check("swap to fakeB, streamed", text.startswith("fakeB:b1#0") and ev == 5, f"{dt:.1f} s, {pings} pings, first byte {first:.2f} s")
    check("pings while starting", pings >= 1)
    check("only fakeB runs", models_running() == {"a1": False, "a2": False, "b1": True, "c1": False})
    # a swap waits for the request in flight on the running server
    res = []
    tb = threading.Thread(target=stream, args=("b1",), kwargs={"out": res})
    tb.start()
    time.sleep(0.4)
    t = time.time()
    s, r = req("POST", "/v1/chat/completions", {"model": "a1", "messages": []})
    tb.join()
    check("fakeB's stream completed before the swap", res and res[0][2] == 5, f"stream events {res[0][2] if res else None}")
    check("then a1 answered by fakeA", r["choices"][0]["message"]["content"] == "fakeA:a1", f"waited {time.time() - t:.1f} s")
    # a client that disconnects mid-stream does not block the next swap
    stream("a1", close_after=2)
    time.sleep(0.5)
    t = time.time()
    first, text, ev, pings, dt = stream("b1")
    check("swap after a disconnected client", text.startswith("fakeB:b1") and dt < 10, f"{dt:.1f} s")
    # a client that leaves while its server starts (fakeC takes 4 s): the start completes and the next request is served
    # (aborting the start used to leave the process running on its port, untracked: "port ... already in use")
    stream("c1", close_on_ping=True)
    t = time.time()
    s, r = req("POST", "/v1/chat/completions", {"model": "c1", "messages": []})
    check("client gone during a start: the next request is served", s == 200 and r["choices"][0]["message"]["content"] == "fakeC:c1",
          f"HTTP {s} after {time.time() - t:.1f} s")
    # idle unload (6 s in this config; the loop checks every 10 s)
    time.sleep(18)
    check("idle server unloaded", not any(models_running().values()), json.dumps(models_running()))
    # UI and stubs
    s, _ = req("GET", "/")
    check("web UI served", s == 200)
    check("router stubs", req("GET", "/v1/catalog.json")[1] == [] and req("POST", "/models/load", {})[1] == {"success": True})
    check("tools off in this config -> 403", req("GET", "/tools")[0] == 403)
finally:
    gw.terminate()
    gw.wait(timeout=30)

# a stderr that blocks (a terminal stopped with Ctrl+S, a frozen pane): the gateway must keep serving, starting servers
# and unloading them (it used to hang in its first log line while starting a server). Simulated with a pipe that is full
# and never read.
PORT = 11445
rfd, wfd = os.pipe()
os.set_blocking(wfd, False)
try:
    while True:
        os.write(wfd, b"x" * 4096)
except BlockingIOError:
    pass
os.set_blocking(wfd, True)
gw2 = subprocess.Popen([sys.executable, os.path.join(HERE, "..", "lion.py"), "--config", os.path.join(HERE, "fake_backends.json"),
                        "--listen", f"127.0.0.1:{PORT}", "--idle", "2"], stdout=wfd, stderr=wfd)
try:
    for _ in range(40):
        try:
            if req("GET", "/health", timeout=2)[0] == 200:
                break
        except OSError:
            time.sleep(0.25)
    t = time.time()
    try:
        s, r = req("POST", "/v1/chat/completions", {"model": "a1", "messages": []}, timeout=15)
        ok = s == 200 and r["choices"][0]["message"]["content"] == "fakeA:a1"
    except OSError as e:
        ok, s = False, e
    check("blocked stderr: a server starts and answers", ok, f"{s}, {time.time() - t:.1f} s")
    time.sleep(13)   # idle 2 s; the loop checks every 10 s
    check("blocked stderr: idle unload", not any(models_running().values()), json.dumps(models_running()))
    t = time.time()
    try:
        s, r = req("POST", "/v1/chat/completions", {"model": "a1", "messages": []}, timeout=15)
        ok = s == 200 and r["choices"][0]["message"]["content"] == "fakeA:a1"
    except OSError as e:
        ok, s = False, e
    check("blocked stderr: started again after the idle unload", ok, f"{s}, {time.time() - t:.1f} s")
    check("the log file keeps every line", "idle for" in open("/tmp/lion-test/lion.log").read())
finally:
    gw2.kill()
    gw2.wait(timeout=30)
    os.close(rfd)
    os.close(wfd)
    for b in ("fake_backend.py --port 11600", "fake_backend.py --port 11601", "fake_backend.py --port 11603"):
        subprocess.run(["pkill", "-f", b])   # gw2 was killed: its server may still run
print(open("/tmp/lion-test-gw.log").read())
print(f"{len(fails)} failed" if fails else "all passed")
sys.exit(1 if fails else 0)
