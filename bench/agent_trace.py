#!/usr/bin/env python3
"""Agent-like workload: A concurrent agents x T turns. Each conversation starts with a long system prompt and
grows by a tool result per turn, so prompt caching (slot reuse, RAM/disk tiers) dominates the cost.
More agents than server slots forces evictions. Works against liquid-server and llama-server.
usage: agent_trace.py <url> [--agents 6] [--turns 8] [--system-tokens 3000] [--tool-tokens 1500] [--gen 64]"""
import argparse, json, random, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--agents", type=int, default=6); ap.add_argument("--turns", type=int, default=8)
ap.add_argument("--system-tokens", type=int, default=3000); ap.add_argument("--tool-tokens", type=int, default=1500)
ap.add_argument("--gen", type=int, default=64)
a = ap.parse_args()
WORDS = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega river stone cloud amber copper "
         "orchid falcon ember harbor meadow lantern quartz velvet willow").split()


def text(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n))


results = []
lock = threading.Lock()


def agent(i):
    msgs = [{"role": "system", "content": f"You are agent {i}. Follow the instructions.\n" + text(a.system_tokens, 1000 + i)},
            {"role": "user", "content": "Start the task and report briefly."}]
    for t in range(a.turns):
        body = {"messages": msgs, "max_tokens": a.gen, "temperature": 0, "stream": True, "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True, "repeat_penalty": 1.0,
                "stream_options": {"include_usage": True}}
        t0 = time.time()
        first = None
        content = ""
        timings = {}
        req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                j = json.loads(line[6:])
                if j.get("timings"): timings = j["timings"]
                for ch in j.get("choices", []):
                    d = (ch.get("delta") or {}).get("content")
                    if d:
                        if first is None: first = time.time()
                        content += d
        t1 = time.time()
        with lock:
            results.append({"agent": i, "turn": t, "ttft": (first or t1) - t0, "total": t1 - t0,
                            "cached": timings.get("cache_n", 0), "prompt_n": timings.get("prompt_n", 0)})
        msgs = msgs + [{"role": "assistant", "content": content},
                       {"role": "user", "content": "Tool result:\n" + text(a.tool_tokens, 7000 + 100 * i + t) + "\nContinue."}]


t0 = time.time()
th = [threading.Thread(target=agent, args=(i,)) for i in range(a.agents)]
for t in th: t.start()
for t in th: t.join()
wall = time.time() - t0
tt = sorted(r["ttft"] for r in results)
pre = sum(r["prompt_n"] for r in results)
cached = sum(r["cached"] for r in results)


def pct(v, p): return v[min(len(v) - 1, int(p * len(v)))]


print(f"agents={a.agents} turns={a.turns}: wall {wall:.1f} s | TTFT p50 {pct(tt, .5) * 1e3:.0f} ms, p90 {pct(tt, .9) * 1e3:.0f} ms, "
      f"max {tt[-1] * 1e3:.0f} ms | prompt tokens prefilled {pre}, reused from cache {cached}")
