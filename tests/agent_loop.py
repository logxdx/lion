#!/usr/bin/env python3
"""Replays the llama.cpp web UI's agentic loop against liquid-server: stream a chat completion with the server tools,
run each tool call through POST /tools, append the result, repeat until the model answers.

usage: tests/agent_loop.py "question" [base_url] [max_turns]
"""
import json
import sys
import time
import urllib.request

Q = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"
MAX_TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def post(path, body, stream=False):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=900)


tools = [t["definition"] for t in json.load(urllib.request.urlopen(BASE + "/tools"))]
messages = [{"role": "user", "content": Q}]
for turn in range(MAX_TURNS):
    body = {"messages": messages, "tools": tools, "stream": True, "return_progress": True, "timings_per_token": True,
            "reasoning_format": "auto", "max_tokens": -1}
    content, reasoning, calls, finish, timings = "", "", {}, None, {}
    t0 = time.time()
    with post("/v1/chat/completions", body) as resp:
        done = False
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            ch = json.loads(data)
            timings = ch.get("timings", timings)
            if not ch.get("choices"):
                continue
            c = ch["choices"][0]
            d = c.get("delta", {})
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                e = calls.setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
                e["id"] = tc.get("id") or e["id"]
                f = tc.get("function", {})
                e["name"] += f.get("name") or ""
                e["arguments"] += f.get("arguments") or ""
            finish = c.get("finish_reason") or finish
    print(f"--- turn {turn + 1}: finish={finish} [DONE]={done} {time.time() - t0:.1f}s "
          f"prompt_n={timings.get('prompt_n')} cache_n={timings.get('cache_n')} predicted_n={timings.get('predicted_n')}")
    if reasoning:
        print(f"reasoning: {reasoning[:300]!r}")
    if not calls:
        print(f"answer: {content}")
        break
    msg = {"role": "assistant", "content": content, "reasoning_content": reasoning,
           "tool_calls": [{"id": e["id"], "type": "function", "function": {"name": e["name"], "arguments": e["arguments"]}}
                          for e in calls.values()]}
    messages.append(msg)
    for e in calls.values():
        args = json.loads(e["arguments"] or "{}")
        t1 = time.time()
        out = json.load(post("/tools", {"tool": e["name"], "params": args}))
        text = out.get("plain_text_response", out.get("error", json.dumps(out)))
        print(f"tool {e['name']}({json.dumps(args)}) -> {len(text)} chars in {time.time() - t1:.1f}s: {text[:160]!r}")
        messages.append({"role": "tool", "tool_call_id": e["id"], "content": text})
