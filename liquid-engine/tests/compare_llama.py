#!/usr/bin/env python3
"""Compare next-token distributions: liquid (fast int8 path), liquid --exact (f32 reference) and a
running llama-server on the same GGUF. Reports max |dp| over the reference top-10 vs the exact path.
usage: compare_llama.py <liquid-binary> <model.gguf> [server-url] [extra liquid args...]"""
import json, math, subprocess, sys, urllib.request

BIN, MODEL = sys.argv[1], sys.argv[2]
URL = sys.argv[3] if len(sys.argv) > 3 else 'http://127.0.0.1:18080'
EXTRA = sys.argv[4:]
PROMPTS = [
    "The capital of France is Paris. The capital of Germany is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "<|startoftext|><|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>",
    "Once upon a time, in a small village nestled between two mountains, there lived an old clockmaker who",
    "日本の首都は",
    "<|startoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a haiku about GPUs.<|im_end|>\n<|im_start|>assistant\n<think></think>",
    # > 256 tokens: exercises the sliding-window ring buffer and split-K attention
    ("In the beginning of the computing era, engineers built machines from vacuum tubes and relays. " * 40)
    + "Summarize the paragraph above in one word:",
    ("Alice has 3 apples. Bob gives her 5 more. Carol takes 2. " * 120) + "How many apples does Alice have now? Answer:",
]


def server_probs(prompt):
    req = {"prompt": prompt, "n_predict": 1, "n_probs": 20, "temperature": 0, "cache_prompt": False}
    r = urllib.request.urlopen(urllib.request.Request(URL + "/completion", json.dumps(req).encode(),
                                                      {"Content-Type": "application/json"}))
    d = json.loads(r.read())
    return {t["id"]: math.exp(t["logprob"]) for t in d["completion_probabilities"][0]["top_logprobs"]}


def liquid_probs(prompt, extra):
    out = subprocess.run([BIN, "-m", MODEL, "-c", "8192", "--raw", "-q", "-p", prompt, "--dump-logits"] + extra,
                         capture_output=True, text=True).stdout.splitlines()
    res = {}
    for line in out[1:]:
        f = line.split("\t")
        if len(f) >= 3 and f[0].isdigit():  # pieces containing '\n' spill onto extra lines
            res[int(f[0])] = float(f[2])
    return res, out[0]


def maxdp(a, ref):
    return max(abs(ref[t] - a.get(t, 0.0)) for t in list(ref)[:5])


tot = {"liquid": 0.0, "llama": 0.0}
agree = {"liquid": 0, "llama": 0}
print(f"{'prompt':<26} {'top(exact)':>10} {'p_exact':>8} {'p_liquid':>8} {'p_llama':>8} {'dp_liquid':>9} {'dp_llama':>9}")
for p in PROMPTS:
    ex, hdr = liquid_probs(p, EXTRA + ["--exact"])
    fast, _ = liquid_probs(p, EXTRA)
    ll = server_probs(p)
    top = max(ex, key=ex.get)
    exs = dict(sorted(ex.items(), key=lambda kv: -kv[1]))
    d1, d2 = maxdp(fast, exs), maxdp(ll, exs)
    tot["liquid"] += d1; tot["llama"] += d2
    agree["liquid"] += max(fast, key=fast.get) == top
    agree["llama"] += max(ll, key=ll.get) == top
    print(f"{hdr:<26} {top:>10} {ex[top]:8.4f} {fast.get(top, 0):8.4f} {ll.get(top, 0):8.4f} {d1:9.4f} {d2:9.4f}")
n = len(PROMPTS)
print(f"mean max|dp| vs exact: liquid {tot['liquid'] / n:.4f}  llama.cpp {tot['llama'] / n:.4f}; "
      f"argmax agreement with exact: liquid {agree['liquid']}/{n}  llama.cpp {agree['llama']}/{n}")
