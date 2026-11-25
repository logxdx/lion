#!/usr/bin/env python3
"""Greedy chat output of raon-server vs KRAFTON llama-server on the same messages (full stack: template, tokenizer,
engine, detokenizer, output parsing). Reports where the texts first differ.

    python3 tests/chat_parity.py --raon http://127.0.0.1:8090 --ref http://127.0.0.1:8191 [-n 160]
"""
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--raon', default='http://127.0.0.1:8090')
ap.add_argument('--ref', default='http://127.0.0.1:8191')
ap.add_argument('-n', type=int, default=160)
args = ap.parse_args()

prompts = [
    [{"role": "user", "content": "What is the capital of France? Answer briefly."}],
    [{"role": "system", "content": "You are a terse assistant."}, {"role": "user", "content": "List three prime numbers."}],
    [{"role": "user", "content": "Explain what a GPU is to a child, in two sentences."}],
    [{"role": "user", "content": "한국의 수도는 어디인가요?"}],
    [{"role": "user", "content": "Write a Python function that reverses a string."}],
    [{"role": "user", "content": "Translate 'good morning' into Japanese."}],
]


def chat(url, msgs, think=True):
    body = {"messages": msgs, "temperature": 0, "max_tokens": args.n, "presence_penalty": 0, "frequency_penalty": 0,
            "top_k": 1, "chat_template_kwargs": {"enable_thinking": think}}
    r = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    m = json.loads(urllib.request.urlopen(r, timeout=900).read())['choices'][0]['message']
    return (m.get('reasoning_content') or '') + '\n--\n' + (m.get('content') or '')


same = 0
total = 0
for think in (False, True):
    for msgs in prompts:
        a, b = chat(args.raon, msgs, think), chat(args.ref, msgs, think)
        total += 1
        if a == b:
            same += 1
            print(f'same    ({len(a)} chars) thinking={think}: {msgs[-1]["content"][:50]!r}')
        else:
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            print(f'differ at char {k}/{max(len(a), len(b))} thinking={think}: {msgs[-1]["content"][:50]!r}\n'
                  f'    raon : {a[max(0, k - 40):k + 40]!r}\n    llama: {b[max(0, k - 40):k + 40]!r}')
print(f'{same}/{total} identical')
