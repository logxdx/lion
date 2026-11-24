#!/usr/bin/env python3
"""End-to-end video checks for liquid-server (LFM3-VL via --vl-model; ffmpeg + ffprobe on the server's PATH).

usage: tests/test_video_server.py [base_url] [video] [--expect-tokens N]
  Sends the video as the web UI does (input_video, raw base64) and as a video_url data: URL, greedy, and prints the
  prompt size and answers. Works against llama-server too (for parity: same request, compare the output).
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE = args[0] if args else "http://127.0.0.1:8080"
VIDEO = args[1] if len(args) > 1 else os.path.join(os.path.dirname(__file__), "assets", "clock.mp4")
EXPECT = None
if "--expect-tokens" in sys.argv:
    EXPECT = int(sys.argv[sys.argv.index("--expect-tokens") + 1])
failed = 0


def req(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=900) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def check(name, cond, detail=""):
    global failed
    failed += not cond
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


def chat(content, **kw):
    t0 = time.time()
    st, out = req("/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": 60,
                                           "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **kw})
    return st, out, time.time() - t0


raw = open(VIDEO, "rb").read()
b64 = base64.b64encode(raw).decode()
mime = "video/webm" if VIDEO.endswith(".webm") else "video/mp4"
q = "Describe this video in one sentence."
st, p = req("/props")
check("/props reports video (web UI enables video attachments)", p.get("modalities", {}).get("video") is True)

parts_ui = [{"type": "input_video", "input_video": {"data": b64, "format": "mp4"}}, {"type": "text", "text": q}]
st, out, dt = chat(parts_ui)
ok = st == 200 and "choices" in out
text = out["choices"][0]["message"]["content"] if ok else json.dumps(out)[:300]
n_prompt = out.get("usage", {}).get("prompt_tokens") if ok else None
check("input_video (web UI format) answers", ok, f"{dt:.1f} s, {n_prompt} prompt tokens")
print("   ", repr(text))
if EXPECT is not None:
    check("prompt tokens as llama.cpp", n_prompt == EXPECT, f"{n_prompt} vs {EXPECT}")

parts_url = [{"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}}, {"type": "text", "text": q}]
st, out2, dt2 = chat(parts_url)
ok2 = st == 200 and "choices" in out2
text2 = out2["choices"][0]["message"]["content"] if ok2 else json.dumps(out2)[:300]
check("video_url data: URL gives the same answer", ok2 and text2 == text, f"{dt2:.1f} s")

# same video, another question: the prefix up to the end of the video is reused (no re-encode / re-prefill)
st, out3, dt3 = chat([parts_ui[0], {"type": "text", "text": "What colors appear in the video?"}])
ok3 = st == 200 and "choices" in out3
check("follow-up question about the same video", ok3, f"{dt3:.1f} s (first request {dt:.1f} s)")
if ok3:
    print("   ", repr(out3["choices"][0]["message"]["content"]))

st, bad, _ = chat([{"type": "input_video", "input_video": {"data": base64.b64encode(b"not a video").decode()}},
                   {"type": "text", "text": q}])
check("a non-video upload is a 400 with a message", st == 400 and "error" in bad, json.dumps(bad)[:160])

print("FAILED" if failed else "ALL PASS")
sys.exit(1 if failed else 0)
