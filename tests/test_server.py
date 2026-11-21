#!/usr/bin/env python3
"""End-to-end tests for liquid-server (OpenAI API, scheduler, prefix reuse, cache tiers).
usage: test_server.py <url> <liquid-cli-binary> <model.gguf> --slots N"""
import argparse, json, socket, subprocess, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("cli"); ap.add_argument("model")
ap.add_argument("--slots", type=int, default=4)
a = ap.parse_args()
fails = 0


def post(path, body, timeout=600):
    r = urllib.request.urlopen(urllib.request.Request(a.url + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=timeout)
    return json.loads(r.read())


def get(path):
    return json.loads(urllib.request.urlopen(a.url + path, timeout=30).read())


def stream(body):
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(dict(body, stream=True)).encode(),
                                 {"Content-Type": "application/json"})
    out = {"reasoning": "", "content": "", "tool_calls": [], "finish": None, "done": False, "chunks": 0, "timings": None}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                out["done"] = True
                continue
            j = json.loads(data)
            out["chunks"] += 1
            if j.get("timings"): out["timings"] = j["timings"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                out["reasoning"] += d.get("reasoning_content") or ""
                out["content"] += d.get("content") or ""
                out["tool_calls"] += d.get("tool_calls") or []
                if ch.get("finish_reason"): out["finish"] = ch["finish_reason"]
    return out


def check(name, cond, detail=""):
    global fails
    fails += not cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# 1. health / models
check("health", get("/health")["status"] == "ok")
m = get("/v1/models")
check("models", m["data"][0]["id"] == "lfm3", json.dumps(m["data"][0]["meta"]))

# 2. non-stream chat
r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is the capital of France? One word."}],
                                  "max_tokens": 200, "temperature": 0})
msg = r["choices"][0]["message"]
check("chat non-stream", "Paris" in msg["content"], f"content={msg['content']!r} reasoning={len(msg.get('reasoning_content', ''))} chars "
      f"finish={r['choices'][0]['finish_reason']} usage={r['usage']}")

# 3. stream chat with thinking
s = stream({"messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 300, "temperature": 0})
check("chat stream", s["done"] and "1" in s["content"] and s["finish"] in ("stop", "length"),
      f"chunks={s['chunks']} reasoning={len(s['reasoning'])} content={s['content'][:60]!r} finish={s['finish']}")

# 4. tool call
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                         "required": ["city"]}}}]
r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What's the weather in Tokyo for the next 3 days? Use the tool."}],
                                  "tools": tools, "max_tokens": 400, "temperature": 0})
c = r["choices"][0]
tc = c["message"].get("tool_calls") or []
args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
check("tool call", c["finish_reason"] == "tool_calls" and tc and tc[0]["function"]["name"] == "get_weather" and args.get("city"),
      f"finish={c['finish_reason']} calls={[(t['function']['name'], t['function']['arguments']) for t in tc]}")

# 5. greedy server output == CLI for the same raw prompt
prompt = "<|startoftext|><|im_start|>user\nWrite two sentences about the ocean.<|im_end|>\n<|im_start|>assistant\n<think></think>"
run = subprocess.run([a.cli, "-m", a.model, "-q", "--raw", "--greedy", "-n", "64", "-c", "4096", "-p", prompt],
                     capture_output=True, text=True)
cli = run.stdout.rstrip("\n")
srv = post("/v1/completions", {"prompt": prompt, "max_tokens": 64, "temperature": 0, "repeat_penalty": 1.0})["choices"][0]["text"]
if "out of memory" in run.stderr:  # the CLI needs ~2.1 GB next to the server: start the server smaller to run this
    print("SKIP  server == CLI (greedy)  CLI out of GPU memory next to this server (try -np 2 -c 16384)")
else:
    check("server == CLI (greedy)", srv.strip() == cli.strip(), f"server={srv[:50]!r} cli={cli[:50]!r}")

# 6. multi-turn prefix reuse
conv = [{"role": "system", "content": "You are a helpful assistant. " * 40},
        {"role": "user", "content": "Give me three tips for writing clean code."}]
r1 = post("/v1/chat/completions", {"messages": conv, "max_tokens": 150, "temperature": 0})
conv += [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]}, {"role": "user", "content": "Now one more tip."}]
r2 = post("/v1/chat/completions", {"messages": conv, "max_tokens": 60, "temperature": 0})
t2 = r2["timings"]
check("multi-turn prefix reuse", t2["cache_n"] > 0, f"prompt {r2['usage']['prompt_tokens']} tok, reused {t2['cache_n']} "
      f"({t2['cache_tier']}), prefilled {t2['prompt_n']}")

# 7. more concurrent requests than slots
n = a.slots + 2
res = [None] * n


def worker(i):
    res[i] = post("/v1/chat/completions", {"messages": [{"role": "user", "content": f"Say the number {i} in words."}],
                                           "max_tokens": 120, "temperature": 0})


t0 = time.time()
th = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
for t in th: t.start()
for t in th: t.join()
queued = sorted(round(x["timings"]["queue_ms"]) for x in res)
check(f"{n} concurrent requests on {a.slots} slots", all(x["choices"][0]["message"]["content"] for x in res),
      f"wall {time.time() - t0:.1f}s, queue_ms {queued}")

# 8. client disconnect mid-stream frees the slot
host, port = a.url.split("//")[1].split(":")
sock = socket.create_connection((host, int(port)))
body = json.dumps({"messages": [{"role": "user", "content": "Write a very long story about a dragon."}], "max_tokens": 3000,
                   "stream": True, "temperature": 0.7})
sock.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode())
sock.recv(4096); time.sleep(1.0); sock.close()
time.sleep(1.5)
st = get("/slots")
check("disconnect cancels generation", st["busy_slots"] == 0, json.dumps([x["state"] for x in st["slots"]]))

# 9. eviction to the RAM tier and restore: fill every slot with other conversations, then revisit
base = [{"role": "system", "content": "You are a careful reviewer. " * 60}, {"role": "user", "content": "Review: x = 1"}]
post("/v1/chat/completions", {"messages": base, "max_tokens": 20, "temperature": 0})
for i in range(a.slots + 1):
    post("/v1/chat/completions", {"messages": [{"role": "system", "content": f"Filler conversation {i}. " * 50},
                                               {"role": "user", "content": "hi"}], "max_tokens": 5, "temperature": 0})
r = post("/v1/chat/completions", {"messages": base + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "and y = 2?"}],
                                  "max_tokens": 20, "temperature": 0})
t = r["timings"]
check("RAM tier restore after eviction", t["cache_tier"] in ("ram", "disk") and t["cache_n"] > 0,
      f"tier={t['cache_tier']} reused={t['cache_n']} cache={json.dumps(get('/slots').get('cache'))}")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
