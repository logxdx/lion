#!/usr/bin/env python3
"""Statistical drift test: for N prompts (snippets of local text files), compare the top-5
next-token probabilities of liquid's fast path and of llama-server against liquid --exact
(f32 activations, exactly dequantized weights). Reports mean/p90 of max|dp| and KL over top-20.
usage: drift_stats.py <liquid> <model.gguf> <text-file>... [--url URL] [--n N]"""
import argparse, json, math, random, subprocess, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("bin"); ap.add_argument("model"); ap.add_argument("files", nargs="+")
ap.add_argument("--url", default="http://127.0.0.1:18080"); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

random.seed(args.seed)
text = ""
for f in args.files:
    text += open(f, errors="ignore").read() + "\n"
prompts = []
for _ in range(args.n):
    L = random.choice([20, 60, 200, 800, 3000])
    s = random.randrange(0, max(1, len(text) - L))
    prompts.append(text[s:s + L])


def liquid(prompt, extra):
    out = subprocess.run([args.bin, "-m", args.model, "-c", "8192", "--raw", "-q", "-p", prompt, "--dump-logits"] + extra,
                         capture_output=True, text=True).stdout.splitlines()
    r = {}
    for line in out[1:]:
        f = line.split("\t")
        if len(f) >= 3 and f[0].isdigit():
            r[int(f[0])] = float(f[2])
    return r


def server(prompt):
    req = {"prompt": prompt, "n_predict": 1, "n_probs": 20, "temperature": 0, "cache_prompt": False}
    d = json.loads(urllib.request.urlopen(urllib.request.Request(args.url + "/completion", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"})).read())
    return {t["id"]: math.exp(t["logprob"]) for t in d["completion_probabilities"][0]["top_logprobs"]}


def metrics(p, ref):
    top = sorted(ref, key=ref.get, reverse=True)
    dp = max(abs(ref[t] - p.get(t, 0.0)) for t in top[:5])
    kl = sum(ref[t] * math.log(ref[t] / max(p.get(t, 1e-9), 1e-9)) for t in top[:10] if ref[t] > 1e-6)
    return dp, kl, max(p, key=p.get) == top[0]


res = {"liquid": [], "llama": []}
for i, pr in enumerate(prompts):
    ex = liquid(pr, ["--exact"])
    res["liquid"].append(metrics(liquid(pr, []), ex))
    res["llama"].append(metrics(server(pr), ex))
for k, v in res.items():
    dps = sorted(x[0] for x in v); kls = sorted(x[1] for x in v)
    print(f"{k:7s} vs exact: mean max|dp| {sum(dps)/len(dps):.4f}  p90 {dps[int(.9*len(dps))]:.4f}  max {dps[-1]:.4f} | "
          f"mean KL(top10) {sum(kls)/len(kls):.5f} | argmax agree {sum(x[2] for x in v)}/{len(v)}")
