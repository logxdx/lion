#!/usr/bin/env python3
"""Checks liquid-server's built-in tools (/tools, llama-server's contract) and web UI serving.

usage: tests/test_tools.py [base_url]     (default http://127.0.0.1:8080; needs network for the web tools)
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
PDF_URL = "https://arxiv.org/pdf/1706.03762"
failed = 0


def req(path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def call(tool, params, **kw):
    t0 = time.time()
    status, _, body = req("/tools", {"tool": tool, "params": params, **kw})
    return status, json.loads(body), time.time() - t0


def check(name, cond, detail=""):
    global failed
    failed += not cond
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


# web UI
status, h, body = req("/")
check("GET / serves the web UI", status == 200 and b"<html" in body.lower() and h.get("Cache-Control") == "no-cache")
status, h, _ = req("/build.json")
check("build.json", status == 200)

# listing
status, _, body = req("/tools")
tools = {t["tool"]: t for t in json.loads(body)} if status == 200 else {}
check("GET /tools lists web_search, web_fetch, pdf", {"web_search", "web_fetch", "pdf"} <= set(tools), str(list(tools)))
ok_shape = all(set(t) == {"display_name", "tool", "type", "permissions", "uses_cwd", "definition"} and t["type"] == "server"
               and t["definition"]["function"]["name"] == n for n, t in tools.items())
check("tool entries have llama-server's server_tool shape", ok_shape)

# web_search
status, out, dt = call("web_search", {"query": "llama.cpp github", "count": 3})
text = out.get("plain_text_response", "")
check("web_search", status == 200 and text.startswith("Provider: ") and "\n1. " in text, f"{dt:.1f} s, {text[:40]!r}")

# web_fetch
status, out, dt = call("web_fetch", {"url": "https://example.com"})
text = out.get("plain_text_response", "")
check("web_fetch", status == 200 and "Example Domain" in text, f"{dt:.1f} s, {len(text)} chars")
status, out, _ = call("web_fetch", {"url": "https://nonexistent.invalid/"})
check("web_fetch error -> {error}", status == 200 and "error" in out, out.get("error", "")[:80])

# pdf: map -> find -> page (second call reuses the download + parse cache)
status, out, dt = call("pdf", {"command": "map", "file": PDF_URL})
text = out.get("plain_text_response", "")
check("pdf map", "pages: 15" in text, f"{dt:.1f} s, {text.splitlines()[0] if text else out}")
status, out, dt = call("pdf", {"command": "find", "file": PDF_URL, "regex": "label smoothing"})
text = out.get("plain_text_response", "")
check("pdf find", text.startswith("p"), f"{dt:.1f} s, {text.splitlines()[0] if text else out}")
status, out, dt = call("pdf", {"command": "page", "file": PDF_URL, "page": 3})
text = out.get("plain_text_response", "")
check("pdf page", "--- page 3 of 15" in text and "Encoder" in text, f"{dt:.1f} s, {len(text)} chars")
status, out, dt = call("pdf", {"command": "page", "file": PDF_URL, "page": 1})
text = out.get("plain_text_response", "")
check("pdf page needing OCR points to ocr", "pdf ocr" in text, text[:90])
status, out, dt = call("pdf", {"command": "ocr", "file": PDF_URL, "pages": "1"})
text = out.get("plain_text_response", "")
check("pdf ocr", "page 1:" in text or "already done" in text, f"{dt:.1f} s, {text.splitlines()[0] if text else out}")
status, out, dt = call("pdf", {"command": "page", "file": PDF_URL, "page": 1})
text = out.get("plain_text_response", "")
check("pdf page after ocr", "Attention" in text and "OCR text" in text, f"{dt:.1f} s, {len(text)} chars")
status, out, _ = call("pdf", {"command": "page", "file": PDF_URL})
check("pdf page without page -> {error}", "error" in out, out.get("error", "")[:60])
status, out, _ = call("pdf", {"command": "map", "file": "no-such.pdf"})
check("pdf missing file -> {error}", "error" in out and "no such file" in out["error"], out.get("error", "")[:60])

# errors, as llama-server answers them
status, out, _ = call("nope", {})
check("unknown tool -> 404", status == 404)
status, out, _ = call("web_fetch", {"url": "https://example.com"}, stream=True)
check("stream=true -> 404 (not supported)", status == 404)
status, _, _ = req("/tools", {"params": {}})
check("missing tool -> 400", status == 400)

print("ALL PASS" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
