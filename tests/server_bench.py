#!/usr/bin/env python3
"""Concurrent long-context benchmark of an OpenAI-compatible server (raon-server or llama-server).

N clients send, at the same time, a prompt of about --tokens tokens (a document repeated, with a distinct first line
per client) and stream --gen tokens each (EOS suppressed with logit_bias so every stream has the same length).
Reports per request: time to first token, decode rate after the first token; and the aggregate token rate over the
window in which all streams were generating.

    python3 tests/server_bench.py --url http://127.0.0.1:8090 --doc longdoc.txt --clients 4 --tokens 120000 --gen 512
"""
import argparse
import http.client
import json
import threading
import time
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8090')
ap.add_argument('--doc', required=True)
ap.add_argument('--clients', type=int, default=4)
ap.add_argument('--tokens', type=int, default=120000)
ap.add_argument('--gen', type=int, default=512)
ap.add_argument('--eos', type=int, default=259202)
args = ap.parse_args()
U = urllib.parse.urlparse(args.url)

doc = open(args.doc).read()
r = urllib.request.Request(args.url + '/tokenize', data=json.dumps({"content": doc}).encode(), headers={'Content-Type': 'application/json'})
doc_toks = len(json.loads(urllib.request.urlopen(r).read())['tokens'])
reps = max(1, round(args.tokens / doc_toks))
body_text = (doc + '\n\n') * reps

res = {}


def client(i):
    msgs = [{"role": "user", "content": f"Request {i}. Read the following material.\n\n{body_text}\n\nSummarise it in detail."}]
    body = {"messages": msgs, "max_tokens": args.gen, "temperature": 0.7, "seed": i, "stream": True,
            "logit_bias": {str(args.eos): -100}, "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}
    c = http.client.HTTPConnection(U.hostname, U.port, timeout=3600)
    t0 = time.time()
    c.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    resp = c.getresponse()
    times, usage, timings = [], None, None
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b'data: '):
            continue
        if line[6:] == b'[DONE]':
            break
        j = json.loads(line[6:])
        if j.get('usage'):
            usage, timings = j['usage'], j.get('timings')
        if j.get('choices') and (j['choices'][0].get('delta', {}).get('content') or j['choices'][0].get('delta', {}).get('reasoning_content')):
            times.append(time.time())
    res[i] = dict(t0=t0, times=times, usage=usage, timings=timings)


ths = [threading.Thread(target=client, args=(i,)) for i in range(args.clients)]
T0 = time.time()
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.time() - T0

print(f'{args.clients} clients x ~{doc_toks * reps} prompt tokens, {args.gen} generated each; wall {wall:.1f} s')
for i in sorted(res):
    r = res[i]
    u, tm = r['usage'] or {}, r['timings'] or {}
    n = u.get('completion_tokens', 0)
    ttft = (r['times'][0] - r['t0']) if r['times'] else float('nan')
    print(f"  client {i}: prompt {u.get('prompt_tokens')} tokens, TTFT {ttft:.1f} s (prefill {tm.get('prompt_ms', 0) / 1000:.1f} s at "
          f"{tm.get('prompt_per_second', 0):.0f} tok/s, queued {tm.get('queue_ms', 0) / 1000:.1f} s), "
          f"{n} tokens at {tm.get('predicted_per_second', 0):.1f} tok/s overall")
# window in which every stream was generating: from the last first-token to the first last-token
starts = [r['times'][0] for r in res.values() if r['times']]
ends = [r['times'][-1] for r in res.values() if r['times']]
if len(starts) == args.clients and max(starts) < min(ends):
    a, b = max(starts), min(ends)
    n = sum(sum(1 for t in r['times'] if a <= t <= b) for r in res.values())
    print(f'  all {args.clients} generating for {b - a:.1f} s: {n / (b - a):.1f} tok/s total, {n / (b - a) / args.clients:.1f} tok/s per stream')
else:
    print('  the streams never all generated at the same time')
