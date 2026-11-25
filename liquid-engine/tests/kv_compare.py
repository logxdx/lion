#!/usr/bin/env python3
"""KV cache type accuracy: next-token top-k probabilities with --kv f16, --kv q8 and the exact f32 reference
(which keeps an f16 cache) on long prompts. Reports max |dp| over the reference top-5 per prompt.
usage: kv_compare.py <liquid> <model.gguf> [text files...]"""
import subprocess, sys

BIN, MODEL = sys.argv[1], sys.argv[2]
prompts = [
    ("In the beginning of the computing era, engineers built machines from vacuum tubes and relays. " * 40)
    + "Summarize the paragraph above in one word:",
    ("Alice has 3 apples. Bob gives her 5 more. Carol takes 2. " * 120) + "How many apples does Alice have now? Answer:",
]
import os, random
random.seed(int(os.environ.get("KV_SEED", "7")))
nrand = int(os.environ.get("KV_RANDOM", "0"))  # extra random snippets (1k-16k chars) per file
for f in sys.argv[3:]:
    t = open(f, errors="ignore").read()
    prompts += [t[:4000], t[:16000], t[len(t) // 2: len(t) // 2 + 8000]]
    for _ in range(nrand):
        L = random.choice([1000, 3000, 8000, 16000])
        s0 = random.randrange(0, max(1, len(t) - L))
        prompts.append(t[s0:s0 + L])


def run(p, extra):
    out = subprocess.run([BIN, "-m", MODEL, "-c", "8192", "--raw", "-q", "-p", p, "--dump-logits"] + extra,
                         capture_output=True, text=True).stdout.splitlines()
    r = {}
    for line in out[1:]:
        f = line.split("\t")
        if len(f) >= 3 and f[0].isdigit():
            r[int(f[0])] = float(f[2])
    return r, out[0] if out else "?"


tot = {"f16": 0.0, "q8": 0.0}
agree = {"f16": 0, "q8": 0}
for p in prompts:
    ex, hdr = run(p, ["--exact"])
    top = sorted(ex, key=ex.get, reverse=True)[:5]
    line = f"{hdr:<26}"
    for kv in ("f16", "q8"):
        r, _ = run(p, ["--kv", kv])
        dp = max(abs(ex[t] - r.get(t, 0.0)) for t in top)
        tot[kv] += dp
        agree[kv] += max(r, key=r.get) == top[0]
        line += f"  {kv}: p_top={r.get(top[0], 0):.4f} max|dp|={dp:.4f}"
    print(line + f"  exact p_top={ex[top[0]]:.4f}")
n = len(prompts)
print(f"mean max|dp| vs exact: f16 KV {tot['f16'] / n:.4f}  q8 KV {tot['q8'] / n:.4f}; "
      f"argmax agreement: f16 {agree['f16']}/{n}  q8 {agree['q8']}/{n}")
