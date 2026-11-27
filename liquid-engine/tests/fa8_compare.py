#!/usr/bin/env python3
"""Prefill attention kernels with the q8 KV cache: int8/FP8 (default) vs f16 (LIQUID_FA16=1), both against the exact
f32 reference (which keeps an f16 cache). Next-token top-5 probabilities after a prompt; reports max |dp| over the
reference top-5 per prompt, the mean, and argmax agreement.
usage: fa8_compare.py <liquid> <model.gguf> [text files...]   (env FA8_RANDOM=n: extra random snippets per file)"""
import os, random, subprocess, sys

BIN, MODEL = sys.argv[1], sys.argv[2]
random.seed(7)
prompts = [
    ("In the beginning of the computing era, engineers built machines from vacuum tubes and relays. " * 40)
    + "Summarize the paragraph above in one word:",
    ("Alice has 3 apples. Bob gives her 5 more. Carol takes 2. " * 120) + "How many apples does Alice have now? Answer:",
]
nrand = int(os.environ.get("FA8_RANDOM", "3"))
for f in sys.argv[3:]:
    t = open(f, errors="ignore").read()
    prompts += [t[:4000], t[:16000], t[len(t) // 2: len(t) // 2 + 8000]]
    for _ in range(nrand):
        L = random.choice([2000, 6000, 12000, 20000])
        s0 = random.randrange(0, max(1, len(t) - L))
        prompts.append(t[s0:s0 + L])


def run(p, extra, env=None):
    e = dict(os.environ)
    e.update(env or {})
    out = subprocess.run([BIN, "-m", MODEL, "-c", "16384", "--raw", "-q", "-p", p, "--dump-logits"] + extra,
                         capture_output=True, text=True, env=e).stdout.splitlines()
    r = {}
    for line in out[1:]:
        f = line.split("\t")
        if len(f) >= 3 and f[0].isdigit():
            r[int(f[0])] = float(f[2])
    return r, out[0] if out else "?"


kinds = {"fp8": {}, "f16": {"LIQUID_FA16": "1"}}
tot = {k: 0.0 for k in kinds}
agree = {k: 0 for k in kinds}
for p in prompts:
    ex, hdr = run(p, ["--exact"])
    top = sorted(ex, key=ex.get, reverse=True)[:5]
    line = f"{hdr:<26}"
    for k, env in kinds.items():
        r, _ = run(p, ["--kv", "q8"], env)
        dp = max(abs(ex[t] - r.get(t, 0.0)) for t in top)
        tot[k] += dp
        agree[k] += max(r, key=r.get) == top[0]
        line += f"  {k}: max|dp|={dp:.4f}"
    print(line, flush=True)
n = len(prompts)
print(f"{n} prompts, q8 KV, mean max|dp| vs exact: int8+fp8 kernel {tot['fp8'] / n:.4f}, f16 kernel {tot['f16'] / n:.4f}; "
      f"argmax agreement: int8+fp8 {agree['fp8']}/{n}, f16 {agree['f16']}/{n}")
