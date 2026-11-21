#!/usr/bin/env python3
"""Byte-for-byte comparison of the C++ template port (build/test_chat) with llama-server's /apply-template
(the GGUF jinja template).  usage: compare_template.py <test_chat-binary> <cases.json> [server-url]"""
import json, subprocess, sys, urllib.request

BIN, CASES = sys.argv[1], sys.argv[2]
URL = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:18081"
cases = json.load(open(CASES))
ours = [json.loads(l) for l in subprocess.run([BIN, CASES], capture_output=True, text=True).stdout.splitlines()]
fails = 0
for i, (c, o) in enumerate(zip(cases, ours)):
    body = dict(c)
    r = urllib.request.urlopen(urllib.request.Request(URL + "/apply-template", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}))
    ref = json.loads(r.read())["prompt"]
    got = o.get("prompt", "ERROR: " + o.get("error", ""))
    if got == ref:
        print(f"case {i}: IDENTICAL ({len(ref)} chars)")
    else:
        fails += 1
        k = next((j for j in range(min(len(got), len(ref))) if got[j] != ref[j]), min(len(got), len(ref)))
        print(f"case {i}: DIFF at char {k}\n  ref: {ref[max(0, k - 60):k + 80]!r}\n  got: {got[max(0, k - 60):k + 80]!r}")
print("PASS" if fails == 0 else f"FAIL ({fails})")
