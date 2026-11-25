#!/usr/bin/env python3
"""Greedy long-generation agreement with llama-server: how many characters match before the
first divergence (int8 noise makes eventual divergence at near-ties normal).
usage: greedy_match.py <liquid> <model.gguf> [url] [n_tokens]"""
import json, subprocess, sys, urllib.request

BIN, MODEL = sys.argv[1], sys.argv[2]
URL = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:18080"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 400
PROMPTS = [
    "Explain how a CPU cache hierarchy works, with concrete numbers for latency at each level.",
    "Write a C function that parses an IPv4 address string into a uint32_t, with error handling.",
    "List the planets of the solar system and one interesting fact about each.",
]
tot = 0
for p in PROMPTS:
    chat = f"<|startoftext|><|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think></think>"
    ours = subprocess.run([BIN, "-m", MODEL, "-q", "--raw", "--greedy", "-n", str(N), "-p", chat],
                          capture_output=True, text=True).stdout.rstrip("\n")
    req = {"prompt": chat, "n_predict": N, "temperature": 0, "cache_prompt": False, "repeat_penalty": 1.0,
           "samplers": ["top_k"], "top_k": 1}
    ref = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/completion", json.dumps(req).encode(),
                                                                   {"Content-Type": "application/json"})).read())["content"]
    k = 0
    while k < min(len(ours), len(ref)) and ours[k] == ref[k]:
        k += 1
    tot += k
    print(f"match {k:5d} / {len(ref):5d} chars  | diverges at: {ref[k:k+40]!r} vs {ours[k:k+40]!r}")
print(f"total matching chars: {tot}")
