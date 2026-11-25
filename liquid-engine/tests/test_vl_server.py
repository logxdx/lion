#!/usr/bin/env python3
"""End-to-end checks for images + model swapping in liquid-server (LFM3 default + LFM3-VL via --vl-model).

usage: tests/test_vl_server.py [base_url] [image]
  image defaults to tests/assets/front-page.png (a mock newspaper front page: "MEN WALK ON MOON", July 21, 1969)
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
IMG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                                          "front-page.png")
MIME = "image/png" if IMG.lower().endswith(".png") else "image/jpeg"
failed = 0


def req(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=900) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def check(name, cond, detail=""):
    global failed
    failed += not cond
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


def chat(messages, **kw):
    t0 = time.time()
    st, out = req("/v1/chat/completions", {"messages": messages, "max_tokens": 200, "temperature": 0,
                                           "chat_template_kwargs": {"enable_thinking": False}, **kw})
    return st, out, time.time() - t0


img_url = f"data:{MIME};base64," + base64.b64encode(open(IMG, "rb").read()).decode()

st, m = req("/v1/models")
ids = [x["id"] for x in m.get("data", [])]
check("/v1/models lists the text and the VL model", len(ids) >= 2, str(ids))
vl = next((x["id"] for x in m["data"] if x["meta"].get("vision")), None)
st, p = req("/props")
check("/props reports vision (web UI enables image attachments)", p.get("modalities", {}).get("vision") is True)
# web UI model picker: llama-server router mode
check("/props role router (web UI shows a model picker)", p.get("role") == "router")
check("every model reports status loaded", all(x.get("status", {}).get("value") == "loaded" for x in m["data"]))
st, pv = req(f"/props?model={vl}&autoload=false")
st2, pt = req(f"/props?model={ids[0]}&autoload=false")
check("per-model /props: vision only on the VL model",
      pv.get("modalities", {}).get("vision") is True and pt.get("modalities", {}).get("vision") is False,
      f"{vl} n_ctx {pv.get('default_generation_settings', {}).get('n_ctx')}, {ids[0]} n_ctx {pt.get('default_generation_settings', {}).get('n_ctx')}")

st, out, dt = chat([{"role": "user", "content": "What is 2+2? Answer with the number only."}])
check("text request -> default model", st == 200 and out.get("model") == ids[0] and "4" in out["choices"][0]["message"]["content"],
      f"{out.get('model')} {dt:.1f}s {out.get('choices', [{}])[0].get('message', {}).get('content', out)!r}")

q1 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url}},
                                   {"type": "text", "text": "What is the main headline of this newspaper?"}]}]
st, out, dt = chat(q1)
ans = out.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else str(out)
check("image request -> VL model reads the headline", st == 200 and out.get("model") == vl and "moon" in ans.lower(),
      f"{out.get('model')} {dt:.1f}s prompt={out.get('usage', {}).get('prompt_tokens')} {ans[:120]!r}")
st, s = req("/slots")
check("VL model is on the GPU now", s.get("model") == vl, f"model={s.get('model')} swaps={s.get('model_swaps')}")

q2 = q1 + [{"role": "assistant", "content": ans},
           {"role": "user", "content": "What date is printed on it?"}]
st, out2, dt = chat(q2)
cached = out2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) if st == 200 else 0
ans2 = out2.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else str(out2)
check("follow-up turn reuses the image prefix", st == 200 and cached > 300,
      f"{dt:.1f}s cached={cached} of {out2.get('usage', {}).get('prompt_tokens')} {ans2[:100]!r}")

st, out, dt = chat([{"role": "user", "content": "Name a primary colour. One word."}])
st2, s = req("/slots")
check("text request swaps back to the default model", st == 200 and out.get("model") == ids[0] and s.get("model") == ids[0],
      f"{dt:.1f}s model={s.get('model')} swaps={s.get('model_swaps')}")

st, out, _ = chat(q1, model=ids[0])
check("image to a model without vision -> 400", st == 400, str(out)[:100])

print("ALL PASS" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
