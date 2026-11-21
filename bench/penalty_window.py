#!/usr/bin/env python3
"""Decode speed vs repetition-penalty window: repeat_last_n 64 (llama.cpp default) vs -1 (whole context,
transformers semantics) on a long prompt. usage: penalty_window.py <url> --doc FILE [--tokens N]"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("--doc", required=True); ap.add_argument("--tokens", type=int, default=48000)
a = ap.parse_args()
doc = open(a.doc, errors="ignore").read()[: int(a.tokens * 3)]
msgs = [{"role": "system", "content": doc}, {"role": "user", "content": "Summarize."}]
for last_n in (64, -1, 64, -1):
    body = {"messages": msgs, "max_tokens": 256, "temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1,
            "repeat_last_n": last_n, "ignore_eos": True, "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                                                 {"Content-Type": "application/json"}), timeout=3600).read())
    t = r["timings"]
    print(f"repeat_last_n {last_n:>3}: context {r['usage']['prompt_tokens']} tok, decode {t['predicted_per_second']:.1f} tok/s")
