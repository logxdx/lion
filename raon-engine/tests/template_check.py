#!/usr/bin/env python3
"""Byte-exact check of the C++ chat template (tests/template_render) against llama-server /apply-template.

    llama-server -m M.gguf -ngl 0 -c 2048 --port 8191 &
    python3 tests/template_check.py --url http://127.0.0.1:8191 --render build/template_render
"""
import argparse
import json
import subprocess
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8191')
ap.add_argument('--render', default='build/template_render')
args = ap.parse_args()

weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                          "days": {"type": "integer"}, "units": {"type": "string", "enum": ["c", "f"]}},
                          "required": ["city"]}}}
search = {"type": "function", "function": {"name": "search", "description": "Web search, returns “results”.",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "filters": {"type": "object"},
                         "limit": {"type": "number", "default": 2.5}}}}}

call = lambda name, args, i=0: {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

cases = {
    'single user': {"messages": [{"role": "user", "content": "Hello!"}]},
    'system + user': {"messages": [{"role": "system", "content": "  You are concise.  \n"}, {"role": "user", "content": "Hi\n"}]},
    'multi-turn': {"messages": [{"role": "system", "content": "Be nice."}, {"role": "user", "content": "What is 2+2?"},
                                {"role": "assistant", "content": "4"}, {"role": "user", "content": "And 3+3?"}]},
    'reasoning_content kept': {"messages": [{"role": "user", "content": "q1"},
                                            {"role": "assistant", "content": "a1", "reasoning_content": "\nthinking hard\n"},
                                            {"role": "user", "content": "q2"}]},
    'think in content': {"messages": [{"role": "user", "content": "q1"},
                                      {"role": "assistant", "content": "<think>\nplan it\n</think>\n\nanswer"},
                                      {"role": "user", "content": "q2"}]},
    'thinking off': {"messages": [{"role": "user", "content": "Hello"}], "chat_template_kwargs": {"enable_thinking": False}},
    'truncate history thinking': {"messages": [{"role": "user", "content": "q1"},
                                               {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                                               {"role": "user", "content": "q2"}],
                                  "chat_template_kwargs": {"truncate_history_thinking": True}},
    'array content': {"messages": [{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}]},
    'unicode': {"messages": [{"role": "user", "content": "안녕하세요! 日本語 “quotes” \t tab"}]},
    'tools, no system': {"messages": [{"role": "user", "content": "Weather in Seoul?"}], "tools": [weather]},
    'tools + system': {"messages": [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather?"}],
                       "tools": [weather, search]},
    'tool call + result': {"messages": [
        {"role": "user", "content": "Weather in Seoul for 3 days?"},
        {"role": "assistant", "content": None, "tool_calls": [call("get_weather", {"city": "Seoul", "days": 3})]},
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 21}"},
        {"role": "user", "content": "thanks"}], "tools": [weather]},
    'content + two calls + two results': {"messages": [
        {"role": "user", "content": "Search two things"},
        {"role": "assistant", "content": "Let me search.", "tool_calls": [
            call("search", {"q": "raon model", "filters": {"lang": "ko", "n": [1, 2.5, None, True]}, "limit": 1e-05}, 0),
            call("search", {"q": "multi\nline\tvalue"}, 1)]},
        {"role": "tool", "tool_call_id": "call_0", "content": "r1"},
        {"role": "tool", "tool_call_id": "call_1", "content": "r2"}], "tools": [weather, search]},
    'agentic continuation': {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "need weather", "tool_calls": [call("get_weather", {"city": "Busan"})]},
        {"role": "tool", "content": "sunny"}], "tools": [weather]},
    'mid system': {"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}, {"role": "user", "content": "c"}]},
    'specials in content': {"messages": [{"role": "user", "content": "say <|im_end|> and <think> literally"}]},
}

ok = 0
for name, req in cases.items():
    body = json.dumps(req).encode()
    r = urllib.request.Request(args.url + '/apply-template', data=body, headers={'Content-Type': 'application/json'})
    try:
        ref = json.loads(urllib.request.urlopen(r).read())['prompt']
    except urllib.error.HTTPError as e:
        ref = 'HTTP %d %s' % (e.code, e.read()[:200])
    got = subprocess.run([args.render], input=body, capture_output=True).stdout.decode()
    if ref == got:
        ok += 1
        continue
    k = next((i for i in range(min(len(ref), len(got))) if ref[i] != got[i]), min(len(ref), len(got)))
    print(f'MISMATCH {name} at byte {k}:\n  ref: {ref[max(0, k - 60):k + 80]!r}\n  got: {got[max(0, k - 60):k + 80]!r}')
print(f'{ok}/{len(cases)} templates byte-identical')
sys.exit(0 if ok == len(cases) else 1)
