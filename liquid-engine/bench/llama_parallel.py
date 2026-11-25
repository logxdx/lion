#!/usr/bin/env python3
"""Concurrent decode throughput of a llama-server started with -np N, measured like liquid --bench --parallel:
phase 1 prefills every slot (one request per slot, n_predict 0, prompt cached in that slot), phase 2 runs N
decode requests simultaneously on the cached prompts (only the last prompt token is re-evaluated), so decode is
not mixed with other slots' prefill. Aggregate decode t/s = sum of per-request predicted_per_second.
usage: llama_parallel.py <url> --n N --pp P --tg T [--reps R]"""
import argparse, json, random, threading, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--pp", type=int, default=512)
ap.add_argument("--tg", type=int, default=128)
ap.add_argument("--reps", type=int, default=2)
a = ap.parse_args()


def post(body):
    r = urllib.request.urlopen(urllib.request.Request(a.url + "/completion", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=7200)
    return json.loads(r.read())


aggs, pps = [], []
for rep in range(a.reps + 1):
    prompts = []
    for i in range(a.n):
        random.seed(1000 * rep + i)
        prompts.append([1000 + random.randrange(100000) for _ in range(a.pp)])
    pre = []
    for i in range(a.n):  # phase 1: prefill each slot
        d = post({"prompt": prompts[i], "n_predict": 0, "cache_prompt": True, "id_slot": i})
        pre.append(d["timings"]["prompt_per_second"])
    res = [None] * a.n

    def dec(i):  # phase 2: concurrent decode on the cached prompts
        res[i] = post({"prompt": prompts[i], "n_predict": a.tg, "temperature": 0, "cache_prompt": True, "id_slot": i,
                       "ignore_eos": True, "samplers": ["top_k"], "top_k": 1})["timings"]

    th = [threading.Thread(target=dec, args=(i,)) for i in range(a.n)]
    for t in th: t.start()
    for t in th: t.join()
    if rep == 0:
        continue
    aggs.append(sum(t["predicted_per_second"] for t in res))
    pps.append(sum(pre) / len(pre))
    reused = [t["prompt_n"] for t in res]
print(f"llama-server streams={a.n} pp={a.pp}: prefill {sum(pps)/len(pps):.1f} t/s (per slot), "
      f"decode aggregate {sum(aggs)/len(aggs):.1f} t/s, per stream {sum(aggs)/len(aggs)/a.n:.1f} t/s "
      f"(phase-2 prompt tokens re-evaluated: {reused})")
