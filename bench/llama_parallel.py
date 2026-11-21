#!/usr/bin/env python3
"""Concurrent decode throughput of a llama-server started with -np N: N simultaneous requests, each a
P-token prompt + T generated tokens. Aggregate decode t/s = sum of per-request predicted_per_second
(all requests decode in lockstep). usage: llama_parallel.py <url> --n N --pp P --tg T [--reps R]"""
import argparse, json, random, threading, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--pp", type=int, default=512)
ap.add_argument("--tg", type=int, default=128)
ap.add_argument("--reps", type=int, default=2)
a = ap.parse_args()


def one(i, rep, res):
    random.seed(1000 * rep + i)
    toks = [1000 + random.randrange(100000) for _ in range(a.pp)]
    body = {"prompt": toks, "n_predict": a.tg, "temperature": 0, "cache_prompt": False, "ignore_eos": True,
            "samplers": ["top_k"], "top_k": 1}
    r = urllib.request.urlopen(urllib.request.Request(a.url + "/completion", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=3600)
    res[i] = json.loads(r.read())["timings"]


aggs, pps = [], []
for rep in range(a.reps + 1):
    res = [None] * a.n
    th = [threading.Thread(target=one, args=(i, rep, res)) for i in range(a.n)]
    for t in th: t.start()
    for t in th: t.join()
    if rep == 0:
        continue
    aggs.append(sum(t["predicted_per_second"] for t in res))
    pps.append(sum(t["prompt_n"] for t in res) / max(t["prompt_ms"] for t in res) * 1e3)
print(f"llama-server streams={a.n} pp={a.pp}: prefill {sum(pps)/len(pps):.1f} t/s, "
      f"decode aggregate {sum(aggs)/len(aggs):.1f} t/s, per stream {sum(aggs)/len(aggs)/a.n:.1f} t/s")
