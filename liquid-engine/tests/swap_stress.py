#!/usr/bin/env python3
"""Repeated model swaps: alternate text (default model) and image (VL model) requests, check every answer, and track
swap count, GPU memory and the server's resident memory after each one (a leak shows up as steady growth).

usage: tests/swap_stress.py base_url image server_pid [rounds] [tag]
  tag makes the text prompts new (cold prefill right after each swap, nothing from the prompt cache).
  A wrong answer is re-asked warm: identical text = the model's own greedy answer, different = a swap bug.
"""
import base64
import json
import subprocess
import sys
import time
import urllib.request

B, IMG, PID = sys.argv[1], sys.argv[2], sys.argv[3]
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
TAG = sys.argv[5] if len(sys.argv) > 5 else ""
img = base64.b64encode(open(IMG, "rb").read()).decode()
MIME = "image/png" if IMG.lower().endswith(".png") else "image/jpeg"


def chat(msgs):
    body = {"messages": msgs, "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    out = json.load(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                                                   {"Content-Type": "application/json"}), timeout=600))
    return time.time() - t0, out["model"], out["choices"][0]["message"]["content"]


def vram():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])


def rss():
    for line in open(f"/proc/{PID}/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) // 1024
    return -1


fails = 0
rows = []
for i in range(ROUNDS):
    for kind in ("image", "text"):
        if kind == "image":
            msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{MIME};base64," + img}},
                                                 {"type": "text", "text": f"Round {i}: what is the main headline?"}]}]
            dt, model, ans = chat(msgs)
            ok = model == "lfm3-vl" and "moon" in ans.lower()
        else:
            q = [{"role": "user", "content": f"{TAG}What is {i} + {i}? Answer with the number only."}]
            dt, model, ans = chat(q)
            ok = model == "lfm3" and str(2 * i) in ans
            if not ok and model == "lfm3":  # warm re-ask, no swap in between: a swap bug would answer differently
                _, _, again = chat(q)
                ok = again == ans  # a consistent (if wrong) greedy answer is the model, not the swap
                print(f"   round {i} re-asked warm: " + ("identical, so the model's own answer" if ok else "DIFFERENT: " + repr(again[:60])))
        st = json.load(urllib.request.urlopen(B + "/slots"))
        fails += not ok
        rows.append((i, kind, round(dt, 2), st["model_swaps"], vram(), rss(), ok))
        print(f"round {i:2d} {kind:5s} {dt:5.2f}s swaps={st['model_swaps']:3d} vram={rows[-1][4]} MiB rss={rows[-1][5]} MiB "
              f"{'ok' if ok else 'WRONG: ' + repr(ans[:60])}")

img_v = [r[4] for r in rows if r[1] == "image"]
txt_v = [r[4] for r in rows if r[1] == "text"]
print(f"VRAM with VL loaded: first {img_v[0]} last {img_v[-1]} MiB | with LFM3 loaded: first {txt_v[0]} last {txt_v[-1]} MiB")
print(f"server RSS: first {rows[0][5]} last {rows[-1][5]} MiB")
leak = img_v[-1] - img_v[0] > 64 or txt_v[-1] - txt_v[0] > 64
print("PASS" if fails == 0 and not leak else f"FAIL ({fails} wrong answers, vram leak={leak})")
sys.exit(1 if fails or leak else 0)
