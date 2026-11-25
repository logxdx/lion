#!/usr/bin/env python3
"""Functional checks of raon-server (OpenAI API). Standard library only.

    build/raon-server -m M.gguf --slots 4 --ctx 131072 --port 8090 &
    python3 tests/server_test.py --url http://127.0.0.1:8090 [--doc longdoc.txt]
"""
import argparse
import http.client
import json
import sys
import threading
import time
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8090')
ap.add_argument('--doc', help='a long text for the prompt-reuse check')
args = ap.parse_args()
U = urllib.parse.urlparse(args.url)

fails = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name + (f'  ({detail})' if detail else ''))
    if not cond:
        fails.append(name)


def post(path, body, timeout=600):
    r = urllib.request.Request(args.url + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': json.loads(e.read() or b'{}')}


def stream(path, body, abort_after=None):
    """Returns (chunks, raw lines). abort_after: close the connection after that many data chunks."""
    c = http.client.HTTPConnection(U.hostname, U.port, timeout=600)
    c.request('POST', path, json.dumps(dict(body, stream=True)), {'Content-Type': 'application/json'})
    resp = c.getresponse()
    chunks, done = [], False
    buf = b''
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b'data: '):
            continue
        data = line[6:]
        if data == b'[DONE]':
            done = True
            break
        chunks.append(json.loads(data))
        if abort_after is not None and len(chunks) >= abort_after:
            c.close()
            return chunks, False
    c.close()
    return chunks, done


weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                          "required": ["city"]}}}

# 1. basic chat, greedy
r = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}],
                                   "temperature": 0, "max_tokens": 400})
msg = r['choices'][0]['message']
check('chat: answer', '391' in (msg.get('content') or ''), repr(msg.get('content')))
check('chat: reasoning separated', 'reasoning_content' in msg and '<think>' not in (msg.get('content') or ''))
check('chat: usage', r['usage']['completion_tokens'] > 0 and r['usage']['prompt_tokens'] > 0)

# 2. greedy is deterministic; seeded sampling is reproducible
a = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "temperature": 0, "max_tokens": 80})
b = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "temperature": 0, "max_tokens": 80})
check('greedy deterministic', a['choices'][0]['message'] == b['choices'][0]['message'])
s1 = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "Name a random animal."}], "temperature": 1.0, "seed": 7, "max_tokens": 60})
s2 = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "Name a random animal."}], "temperature": 1.0, "seed": 7, "max_tokens": 60})
check('seeded sampling reproducible', s1['choices'][0]['message'] == s2['choices'][0]['message'])

# 3. streaming equals non-streaming (greedy)
chunks, done = stream('/v1/chat/completions', {"messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "temperature": 0,
                                               "max_tokens": 80, "stream_options": {"include_usage": True}})
content = ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks if c['choices'])
reasoning = ''.join(c['choices'][0]['delta'].get('reasoning_content') or '' for c in chunks if c['choices'])
check('stream: [DONE]', done)
check('stream: same content as non-stream', content == (a['choices'][0]['message'].get('content') or ''),
      f'{content[:60]!r} vs {(a["choices"][0]["message"].get("content") or "")[:60]!r}')
check('stream: same reasoning', reasoning == a['choices'][0]['message'].get('reasoning_content', ''))
check('stream: usage chunk', any(c.get('usage') and not c['choices'] for c in chunks))

# 4. max_tokens and stop strings
r = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "Count from 1 to 50, comma separated."}],
                                   "temperature": 0, "max_tokens": 5, "enable_thinking": False})
check('max_tokens -> length', r['choices'][0]['finish_reason'] == 'length' and r['usage']['completion_tokens'] == 5)
r = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "Count from 1 to 50, comma separated."}],
                                   "temperature": 0, "max_tokens": 300, "stop": ["10"], "enable_thinking": False})
c = r['choices'][0]['message']['content'] or ''
check('stop string', r['choices'][0]['finish_reason'] == 'stop' and '10' not in c and '9' in c, repr(c[-30:]))

# 5. thinking off
r = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "Say hello."}], "temperature": 0, "max_tokens": 50,
                                   "chat_template_kwargs": {"enable_thinking": False}})
check('thinking off: no reasoning', 'reasoning_content' not in r['choices'][0]['message'], json.dumps(r['choices'][0]['message'])[:120])

# 6. logprobs
r = post('/v1/chat/completions', {"messages": [{"role": "user", "content": "Say hi."}], "temperature": 0, "max_tokens": 8,
                                   "logprobs": True, "top_logprobs": 3, "enable_thinking": False})
lp = (r['choices'][0].get('logprobs') or {}).get('content') or []
check('logprobs', len(lp) == r['usage']['completion_tokens'] and all(len(x['top_logprobs']) == 3 for x in lp)
      and all(x['top_logprobs'][0]['logprob'] >= x['top_logprobs'][1]['logprob'] for x in lp)
      and all(abs(x['logprob'] - x['top_logprobs'][0]['logprob']) < 1e-6 for x in lp))

# 7. completions endpoint
r = post('/v1/completions', {"prompt": "The quick brown fox", "max_tokens": 12, "temperature": 0})
check('completions', r['object'] == 'text_completion' and len(r['choices'][0]['text']) > 0, repr(r['choices'][0]['text']))
chunks, done = stream('/v1/completions', {"prompt": "The quick brown fox", "max_tokens": 12, "temperature": 0})
check('completions stream', done and ''.join(c['choices'][0]['text'] for c in chunks) == r['choices'][0]['text'])

# 8. tool call, then the tool result
q = {"messages": [{"role": "user", "content": "What's the weather in Seoul for the next 3 days? Use the tool."}],
     "tools": [weather], "temperature": 0, "max_tokens": 800}
r = post('/v1/chat/completions', q)
ch = r['choices'][0]
calls = ch['message'].get('tool_calls') or []
ok = ch['finish_reason'] == 'tool_calls' and calls and calls[0]['function']['name'] == 'get_weather'
args_ok = ok and json.loads(calls[0]['function']['arguments']).get('city', '').lower().startswith('seoul')
check('tool call', ok and args_ok, json.dumps(ch['message'])[:300])
if ok:
    days = json.loads(calls[0]['function']['arguments']).get('days')
    check('tool call: integer argument typed', days is None or isinstance(days, int), repr(days))
    chunks, done = stream('/v1/chat/completions', q)
    sc = [tc for c in chunks if c['choices'] for tc in (c['choices'][0]['delta'].get('tool_calls') or [])]
    fin = [c['choices'][0]['finish_reason'] for c in chunks if c['choices'] and c['choices'][0]['finish_reason']]
    check('tool call (stream)', sc and sc[0]['function']['name'] == 'get_weather' and fin == ['tool_calls'])
    q2 = dict(q)
    q2['messages'] = q['messages'] + [ch['message'], {"role": "tool", "tool_call_id": calls[0]['id'],
                                                      "content": json.dumps({"forecast": ["sunny 21C", "rain 17C", "cloudy 19C"]})}]
    r2 = post('/v1/chat/completions', q2)
    c2 = r2['choices'][0]['message'].get('content') or ''
    check('tool result answered', r2['choices'][0]['finish_reason'] == 'stop' and ('21' in c2 or 'sunny' in c2.lower()), repr(c2[:160]))
    check('tool turn reused the prompt', r2['usage']['prompt_tokens_details']['cached_tokens'] > 0,
          f"cached {r2['usage']['prompt_tokens_details']['cached_tokens']} of {r2['usage']['prompt_tokens']}")

# 9. errors
r = post('/v1/chat/completions', {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
check('image content rejected (400)', r.get('http_error') == 400)
r = post('/v1/chat/completions', {"messages": []})
check('empty messages rejected (400)', r.get('http_error') == 400)

# 10. client disconnect cancels generation
chunks, _ = stream('/v1/chat/completions', {"messages": [{"role": "user", "content": "Write a very long story."}],
                                            "max_tokens": 4000, "temperature": 0.7}, abort_after=5)
time.sleep(1.0)
slots = json.loads(urllib.request.urlopen(args.url + '/slots').read())
check('disconnect cancels', all(s['state'] == 'idle' for s in slots), json.dumps(slots))

# 11. multi-turn prompt reuse on a long document
if args.doc:
    doc = open(args.doc).read()
    m1 = [{"role": "system", "content": "You answer questions about the document."},
          {"role": "user", "content": doc + "\n\nQuestion: what is this document about? One sentence."}]
    t = time.time()
    r1 = post('/v1/chat/completions', {"messages": m1, "temperature": 0, "max_tokens": 300})
    t1 = time.time() - t
    m2 = m1 + [r1['choices'][0]['message'], {"role": "user", "content": "Name one number mentioned in it."}]
    t = time.time()
    r2 = post('/v1/chat/completions', {"messages": m2, "temperature": 0, "max_tokens": 300})
    t2 = time.time() - t
    cached = r2['usage']['prompt_tokens_details']['cached_tokens']
    check('multi-turn reuse', cached >= r1['usage']['prompt_tokens'] - 8,
          f"turn 1: {r1['usage']['prompt_tokens']} prompt tokens, {t1:.2f} s; turn 2: {r2['usage']['prompt_tokens']} prompt, "
          f"{cached} reused, prefill {r2['timings']['prompt_ms']:.0f} ms, {t2:.2f} s total")
    # same question again without the reasoning in history: still reuses up to the checkpoint
    m2b = m1 + [{"role": "assistant", "content": r1['choices'][0]['message']['content']}, {"role": "user", "content": "Another question?"}]
    r3 = post('/v1/chat/completions', {"messages": m2b, "temperature": 0, "max_tokens": 50})
    check('reuse without reasoning in history', r3['usage']['prompt_tokens_details']['cached_tokens'] >= r1['usage']['prompt_tokens'] - 8,
          f"{r3['usage']['prompt_tokens_details']['cached_tokens']} of {r3['usage']['prompt_tokens']}")

# 12. concurrency: 4 streams at once
res = {}


def worker(i):
    t = time.time()
    r = post('/v1/chat/completions', {"messages": [{"role": "user", "content": f"Write a detailed essay about topic number {i}: the history of computing."}],
                                       "temperature": 0.7, "seed": i, "max_tokens": 400, "enable_thinking": False})
    res[i] = (r, time.time() - t)


ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
t0 = time.time()
for th in ths:
    th.start()
for th in ths:
    th.join()
wall = time.time() - t0
toks = sum(r['usage']['completion_tokens'] for r, _ in res.values())
slots_used = sorted(r['timings']['slot'] for r, _ in res.values())
per = [r['timings']['predicted_per_second'] for r, _ in res.values()]
check('4 concurrent requests', len(res) == 4 and slots_used == [0, 1, 2, 3],
      f"{toks} tokens in {wall:.1f} s = {toks / wall:.0f} tok/s total; per stream {', '.join(f'{p:.0f}' for p in per)} tok/s")

print(f'\n{len(fails)} failed' if fails else '\nall passed')
sys.exit(1 if fails else 0)
