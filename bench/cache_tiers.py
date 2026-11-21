#!/usr/bin/env python3
"""Prompt-cache tier timings: cold prefill of a long-document conversation vs resuming it from VRAM, RAM
(after eviction by other conversations) and disk (run again with --phase disk after a server restart).
usage: cache_tiers.py <url> --doc FILE --tokens N --slots S [--phase ram|disk]"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("--doc", required=True)
ap.add_argument("--tokens", type=int, default=30000); ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--phase", default="ram")
a = ap.parse_args()


def chat(msgs, max_tokens=16):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0}
    r = urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=3600)
    return json.loads(r.read())


text = open(a.doc, errors="ignore").read()
doc = text[: a.tokens * 3]  # ~3 chars/token for source code
conv = [{"role": "system", "content": "You answer questions about the following document.\n\n" + doc},
        {"role": "user", "content": "What is this document about? One sentence."}]
turn2 = conv + [{"role": "assistant", "content": "It is a source file."}, {"role": "user", "content": "Name one function in it."}]


def show(tag, r):
    t = r["timings"]
    print(f"{tag:<34} prompt {r['usage']['prompt_tokens']:>6} tok | reused {t['cache_n']:>6} from {t['cache_tier'] or '-':<4} "
          f"restore {t['restore_ms']:8.1f} ms | prefilled {t['prompt_n']:>6} tok in {t['prompt_ms']:8.1f} ms")
    return t


if a.phase == "ram":
    show("cold (full prefill)", chat(conv))
    show("turn 2, same slot (vram)", chat(turn2))
    for i in range(a.slots):  # push the document conversation out of every slot
        chat([{"role": "system", "content": f"Unrelated conversation {i}. " * 40}, {"role": "user", "content": "hi"}], 4)
    show("turn 2 after eviction (ram)", chat(turn2))
else:
    show("turn 2 after server restart (disk)", chat(turn2))
