#!/usr/bin/env python3
"""llama.cpp vs itself: batched prompt eval vs token-by-token eval (prompt cache), to size the
numeric noise floor that any int8-activation engine should be judged against.
usage: llama_noise_floor.py [server-url]"""
import json, math, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:18080'
PROMPTS = [
    "日本の首都は",
    "Once upon a time, in a small village nestled between two mountains, there lived an old clockmaker who",
    "<|startoftext|><|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>",
    "<|startoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a haiku about GPUs.<|im_end|>\n<|im_start|>assistant\n<think></think>",
]


def post(path, body):
    r = urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"}))
    return json.loads(r.read())


def probs(d):
    return {t["id"]: math.exp(t["logprob"]) for t in d["completion_probabilities"][0]["top_logprobs"]}


for p in PROMPTS:
    toks = post("/tokenize", {"content": p, "parse_special": True})["tokens"]
    batched = probs(post("/completion", {"prompt": toks, "n_predict": 1, "n_probs": 10, "temperature": 0, "cache_prompt": False}))
    # token by token: grow the cached prompt one token per request
    for k in range(1, len(toks)):
        post("/completion", {"prompt": toks[:k], "n_predict": 0, "cache_prompt": True, "id_slot": 0})
    seq = probs(post("/completion", {"prompt": toks, "n_predict": 1, "n_probs": 10, "temperature": 0, "cache_prompt": True, "id_slot": 0}))
    common = [t for t in batched if t in seq]
    tb, ts = max(batched, key=batched.get), max(seq, key=seq.get)
    print(f"n={len(toks):<4} argmax {'OK  ' if tb == ts else 'DIFF'} p_batched={batched[tb]:.4f} p_seq={seq.get(tb, 0):.4f} "
          f"max|dp|={max(abs(batched[t] - seq[t]) for t in common):.4f}")
