#!/usr/bin/env python3
"""Load scenarios against an OpenAI-compatible server (liquid-server or llama-server), streaming /v1/completions.
  throughput: N concurrent streams (prompt P tokens, T generated each, ignore_eos), aggregate tokens/s
  stall:      3 long decoders; after 2 s a long-prompt request arrives -> decoders' inter-token gaps, its TTFT
usage: server_load.py <url> throughput --n N [--pp P] [--tg T]
       server_load.py <url> stall --doc FILE [--big-tokens N]"""
import argparse, json, random, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("scenario", choices=["throughput", "stall"])
ap.add_argument("--n", type=int, default=4); ap.add_argument("--pp", type=int, default=512); ap.add_argument("--tg", type=int, default=256)
ap.add_argument("--doc"); ap.add_argument("--big-tokens", type=int, default=30000)
a = ap.parse_args()

WORDS = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega river stone cloud amber copper "
         "orchid falcon ember harbor meadow lantern quartz velvet willow").split()


def rand_text(n_tokens, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n_tokens))  # ~1 token per word


def run(prompt, max_tokens, rec):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True, "ignore_eos": True,
            "cache_prompt": False, "repeat_penalty": 1.0}
    rec["t0"] = time.time()
    rec["times"] = []
    req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if j.get("choices") and j["choices"][0].get("text"):
                rec["times"].append(time.time())
    rec["t1"] = time.time()


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0


if a.scenario == "throughput":
    recs = [dict() for _ in range(a.n)]
    th = [threading.Thread(target=run, args=(rand_text(a.pp, i), a.tg, recs[i])) for i in range(a.n)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    toks = sum(len(r["times"]) for r in recs)
    ttft = sorted((r["times"][0] - r["t0"]) * 1e3 for r in recs if r["times"])
    # steady-state aggregate: tokens emitted while every admitted stream was decoding (window between the
    # latest first token and the earliest last token of the first min(N, slots) streams is server specific,
    # so report the overall rate and the median per-stream rate)
    per = sorted(len(r["times"]) / max(1e-9, r["times"][-1] - r["times"][0]) for r in recs if len(r["times"]) > 1)
    print(f"throughput n={a.n} pp={a.pp} tg={a.tg}: {toks / wall:.1f} tok/s overall ({toks} tokens in {wall:.2f} s), "
          f"median per-stream {pct(per, 0.5):.1f} tok/s, TTFT ms min/median/max {ttft[0]:.0f}/{pct(ttft, 0.5):.0f}/{ttft[-1]:.0f}")
else:
    doc = open(a.doc, errors="ignore").read()[: a.big_tokens * 3]
    dec = [dict() for _ in range(3)]
    th = [threading.Thread(target=run, args=(rand_text(256, 100 + i), 1500, dec[i])) for i in range(3)]
    for t in th: t.start()
    time.sleep(2.0)
    big = dict()
    tb = threading.Thread(target=run, args=("Document:\n" + doc + "\nSummary:", 32, big))
    tb.start()
    tb.join()
    for t in th: t.join()
    t_arr = big["t0"]
    ttft = (big["times"][0] - t_arr) if big["times"] else float("nan")
    gaps_during, gaps_all = [], []
    for r in dec:
        ts = r["times"]
        for x, y in zip(ts, ts[1:]):
            gaps_all.append((y - x) * 1e3)
            if t_arr <= y and x <= t_arr + ttft:
                gaps_during.append((y - x) * 1e3)
    toks_during = sum(sum(1 for x in r["times"] if t_arr <= x <= t_arr + ttft) for r in dec)
    print(f"stall: big prompt TTFT {ttft * 1e3:.0f} ms; while it was prefilled the 3 decoders produced {toks_during} tokens "
          f"({toks_during / max(ttft, 1e-9):.1f} tok/s total), worst gap {max(gaps_during or [0]):.0f} ms, "
          f"median gap {pct(gaps_during, 0.5):.0f} ms; overall decoder gap p50/p99 {pct(gaps_all, 0.5):.1f}/{pct(gaps_all, 0.99):.1f} ms")
