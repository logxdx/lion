#!/usr/bin/env python3
"""Measure llama-server prompt/decode speed via its own timings (single stream).
usage: llama_server_bench.py [url] [--pp N] [--tg N] [--reps N]"""
import argparse, json, random, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:18080")
ap.add_argument("--pp", type=int, default=2048)
ap.add_argument("--tg", type=int, default=256)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()


def post(body):
    r = urllib.request.urlopen(urllib.request.Request(a.url + "/completion", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}))
    return json.loads(r.read())


pps, tgs = [], []
for rep in range(a.reps + 1):
    random.seed(rep)
    toks = [1000 + random.randrange(100000) for _ in range(a.pp)]  # same token style as liquid --bench
    d = post({"prompt": toks, "n_predict": a.tg, "temperature": 0, "cache_prompt": False, "ignore_eos": True,
              "samplers": ["top_k"], "top_k": 1})
    t = d["timings"]
    if rep == 0:
        continue
    pps.append(t["prompt_per_second"]); tgs.append(t["predicted_per_second"])
print(f"llama-server pp{a.pp}: {sum(pps)/len(pps):.1f} t/s   tg{a.tg}: {sum(tgs)/len(tgs):.1f} t/s")
