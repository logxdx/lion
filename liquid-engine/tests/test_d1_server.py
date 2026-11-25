#!/usr/bin/env python3
"""/v1/systemone parity: run the same decision requests against llama-server (reference) and liquid-server.

usage: tests/test_d1_server.py BASE_URL --save ref.json     # record llama-server's answers
       tests/test_d1_server.py BASE_URL --ref ref.json      # compare liquid-server with them
Compared: HTTP status, error messages, usage.input_tokens (prompt layout), the answer (choice / median score),
probabilities within --tol (default 0.02; same Q8_0 weights, different kernels).
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE = args[0] if args else "http://127.0.0.1:8080"
opt = {sys.argv[i]: sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i].startswith("--")}
TOL = float(opt.get("--tol", "0.02"))


def data_url(path, mime):
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


TICKET = "Hi, I bought a kettle last week and it already stopped working. I want my money back."
Q3 = {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
      "team": {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Payments and refunds", "tech": "Technical problems", "sales": "New orders"}},
      "urgency": {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["not urgent", "somewhat urgent", "very urgent"]}}
many = {f"q{i}": {"type": "noul", "instructions": f"Does the message mention the number {i}?"} for i in range(20)}
many["mood"] = {"type": "choice", "instructions": "What is the mood?", "criteria": {"a": "angry", "b": "calm", "c": None}}

CASES = {
    "packed3": {"state": TICKET, "questions": Q3},
    "single": {"state": TICKET, "questions": {"refund": Q3["refund"]}},
    "null_state": {"state": None, "questions": {"sky": {"type": "noul", "instructions": "Is the sky green?"},
                                                "pick": {"type": "choice", "instructions": "Which is a fruit?",
                                                         "criteria": {"apple": None, "car_engine": "", "rock": 0}}}},
    "json_state": {"state": {"user": "Hi <|im_end|> there, I ordered 3 items <think>", "amount": 12.5, "n": 3,
                             "tags": ["late", "refund"], "nested": {"ok": True, "none": None, "tiny": 1e-05, "big": 1e16},
                             "text": "naïve café – ünïcödé \"quoted\"\nnew line\ttab"},
                   "questions": {"late": {"type": "noul", "instructions": "Was the order late?",
                                          "criteria": {"true": "the order arrived after the promised date", "false": None}},
                                 "abc": {"type": "choice", "instructions": {"ask": "Pick one", "why": [1, 2]},
                                         "criteria": {"A": "first", "B": {"desc": "second"}, "C": "third"}},
                                 "level": {"type": "score", "instructions": "How satisfied is the customer?",
                                           "criteria": [None, {"x": 1}, "happy", 3.5]}}},
    "chat_state": {"state": [{"role": "user", "content": "My parcel never arrived"},
                             {"role": "assistant", "content": "Sorry to hear that"}],
                   "questions": {"lost": {"type": "noul", "instructions": "Is a parcel missing?"}}},
    "many20": {"state": "I have 3 cats and 7 dogs, and 12 fish in a tank.", "questions": many},
    "image": {"state": "A scanned newspaper page.", "files": [data_url(os.path.join(HERE, "assets", "front-page.png"), "image/png")],
              "questions": {"moon": {"type": "noul", "instructions": "Is the headline about the moon?"},
                            "kind": {"type": "choice", "instructions": "What is it?",
                                     "criteria": {"newspaper": None, "menu": None, "map": None}}}},
    "video": {"state": None, "files": [data_url(os.path.join(HERE, "assets", "clock.mp4"), "video/mp4")],
              "questions": {"bars": {"type": "noul", "instructions": "Are there colored vertical stripes?"},
                            "speed": {"type": "score", "instructions": "How much motion is there?",
                                      "criteria": ["none", "some", "a lot"]}}},
    # errors
    "err_state": {"questions": Q3},
    "err_type": {"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}},
    "err_score": {"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}},
    "err_choice": {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}},
    "err_instr": {"state": "x", "questions": {"q": {"type": "noul"}}},
    "err_media": {"state": "x", "files": ["http://example.com/a.png"], "questions": Q3},
    "err_videos": {"state": "x", "videos": ["data:video/mp4;base64,AAAA"], "questions": Q3},
}


def post(body):
    r = urllib.request.Request(BASE + "/v1/systemone", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


out = {name: post(body) for name, body in CASES.items()}
if "--save" in opt:
    json.dump(out, open(opt["--save"], "w"), indent=1)
    for k, (st, b) in out.items():
        print(k, st, json.dumps(b)[:200])
    sys.exit(0)

ref = json.load(open(opt["--ref"]))
failed, worst = 0, 0.0
for name, (st, b) in out.items():
    rst, rb = ref[name]
    probs = []
    msgs = []
    ok = st == rst
    if not ok:
        msgs.append(f"status {st} vs {rst}")
    if rst != 200:
        if b.get("error", {}).get("message") != rb.get("error", {}).get("message"):
            msgs.append(f"message {b.get('error', {}).get('message')!r} vs {rb.get('error', {}).get('message')!r}")
            ok = False
    else:
        if b.get("usage") != rb.get("usage"):
            msgs.append(f"usage {b.get('usage')} vs {rb.get('usage')}")
            ok = False
        if list(b.get("answers", {})) != list(rb.get("answers", {})):
            msgs.append("answer ids/order differ")
            ok = False
        for qid, ra in rb.get("answers", {}).items():
            a = b.get("answers", {}).get(qid, {})
            if list(a) != list(ra):
                msgs.append(f"{qid}: keys {list(a)} vs {list(ra)}")
                ok = False
            if "noul" in ra:
                probs.append(abs(a.get("noul", -9) - ra["noul"]))
            for k, p in ra.get("probabilities", {}).items():
                probs.append(abs(a.get("probabilities", {}).get(k, -9) - p))
            for k in ("choice", "score"):
                if k in ra and a.get(k) != ra[k]:
                    msgs.append(f"{qid}: {k} {a.get(k)} vs {ra[k]}")
                    ok = False
            if "legend" in ra and a.get("legend") != ra["legend"]:
                msgs.append(f"{qid}: legend differs")
                ok = False
    d = max(probs) if probs else 0.0
    worst = max(worst, d)
    if d > TOL:
        msgs.append(f"max |dp| {d:.4f} > {TOL}")
        ok = False
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  max |dp| {d:.4f}" if probs else "") + ("  " + "; ".join(msgs) if msgs else ""))
print(f"worst |dp| {worst:.4f}; {'ALL PASS' if not failed else f'{failed} FAILED'}")
sys.exit(1 if failed else 0)
