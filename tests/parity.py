#!/usr/bin/env python3
"""Greedy parity vs llama.cpp (KRAFTON raon-1-nano build): same prompt token ids, compare generated tokens and top-k
log-probs step by step.

usage: parity.py --server LLAMA_SERVER --model GGUF --raon build/raon [-n 24] [--top 5] [--ref ref.json]
The reference (llama.cpp) run is saved to --ref and reused if it exists.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

PROMPTS = [
    [{"role": "user", "content": "What is 84 * 3 / 2? Answer briefly."}],
    [{"role": "user", "content": "Write a haiku about the sea."}],
    [{"role": "user", "content": "한국의 수도는 어디인가요?"}],
    [{"role": "system", "content": "You are a terse assistant."},
     {"role": "user", "content": "List three prime numbers greater than 100, then explain how you checked them."}],
]
PORT = 18095
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def reference(a, ngl=99):
    kv = ["-ctk", a.ref_kv, "-ctv", a.ref_kv] if a.ref_kv else []
    p = subprocess.Popen([a.server, "-m", a.model, "-ngl", str(ngl)] + kv + [ "-c", "65536", "-np", "1", "--port", str(PORT),
                          "--no-warmup"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                if urllib.request.urlopen(BASE + "/health", timeout=1).status == 200: break
            except Exception:
                time.sleep(0.3)
        out = []
        for msgs in PROMPTS:
            text = post("/apply-template", {"messages": msgs})["prompt"]
            ids = post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
            d = post("/completion", {"prompt": ids, "n_predict": a.n, "temperature": 0, "samplers": ["top_k"],
                                     "top_k": 1, "n_probs": a.top, "cache_prompt": False})
            steps = []
            for cp in d["completion_probabilities"]:
                steps.append({"token": cp["id"], "logprob": cp["logprob"],
                              "top": [[t["id"], t["logprob"]] for t in cp["top_logprobs"]]})
            out.append({"text": text, "ids": ids, "steps": steps})
        return out
    finally:
        p.terminate(); p.wait(30)


def run_raon(a, ids):
    r = subprocess.run([a.raon, "-m", a.model, "--tokens", ",".join(map(str, ids)), "-n", str(a.n), "--top",
                        str(a.top), "--ctx", "65536"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("raon failed:\n" + r.stderr[-3000:])
    timing = [l for l in r.stderr.splitlines() if l.startswith("prefill")]
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")], (timing[-1] if timing else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--raon", required=True); ap.add_argument("-n", type=int, default=24)
    ap.add_argument("--top", type=int, default=5); ap.add_argument("--ref")
    ap.add_argument("--ref-kv", help="KV cache type for the llama.cpp reference (e.g. q8_0)")
    ap.add_argument("--long", help="text file: one extra prompt = this document + a question (long-context check)")
    ap.add_argument("--vs-cpu", help="instead of raon, compare against llama.cpp on CPU (-ngl 0), cached at this path")
    a = ap.parse_args()
    if a.long:
        doc = open(a.long, encoding="utf-8", errors="replace").read()
        PROMPTS.append([{"role": "user", "content": doc + "\n\nSummarize the document above in three sentences."}])
    if a.ref and os.path.exists(a.ref):
        ref = json.load(open(a.ref))
    else:
        ref = reference(a)
        if a.ref: json.dump(ref, open(a.ref, "w"))
    total_match = total = 0
    worst = 0.0
    other = None
    if a.vs_cpu:
        if os.path.exists(a.vs_cpu): other = json.load(open(a.vs_cpu))
        else: other = reference(a, ngl=0); json.dump(other, open(a.vs_cpu, "w"))
    for k, case in enumerate(ref):
        if other is not None: mine, timing = other[k]["steps"], "llama.cpp CPU"
        else: mine, timing = run_raon(a, case["ids"])
        div = None
        diffs, cdiffs = [], []  # |dlogprob| of candidates with logprob > -5, and of the chosen token
        for s, (r, m) in enumerate(zip(case["steps"], mine)):
            if div is None and r["token"] != m["token"]:
                div = s
            if div is None or s == div:
                rt = {i: lp for i, lp in r["top"]}
                for i, lp in m["top"]:
                    if i in rt and max(lp, rt[i]) > -5: diffs.append(abs(lp - rt[i]))
                if r["token"] == m["token"]: cdiffs.append(abs(r["logprob"] - m["logprob"]))
        n_ok = div if div is not None else min(len(case["steps"]), len(mine))
        total_match += n_ok; total += min(len(case["steps"]), len(mine))
        md = max(diffs) if diffs else float("nan")
        worst = max(worst, md if diffs else 0)
        extra = ""
        if div is not None:
            r, m = case["steps"][div], mine[div]
            extra = f" | diverge@{div}: llama {r['top'][:3]} vs raon {m['top'][:3]}"
        cd = max(cdiffs) if cdiffs else float("nan")
        print(f"prompt {k}: {len(case['ids'])} prompt tokens, greedy match {n_ok}/{len(case['steps'])}, "
              f"max |dlogprob| chosen {cd:.4f}, candidates>-5 {md:.4f}{extra}  [{timing}]")
    print(f"TOTAL greedy match {total_match}/{total}, worst top-k logprob diff {worst:.4f}")


if __name__ == "__main__":
    main()
