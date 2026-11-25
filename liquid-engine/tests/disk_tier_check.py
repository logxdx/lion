#!/usr/bin/env python3
"""After a server restart, a conversation from before the restart must resume from the disk tier.
(Repeats the last conversation of test_server.py check 9.)  usage: disk_tier_check.py <url>"""
import json, sys, urllib.request

url = sys.argv[1]
base = [{"role": "system", "content": "You are a careful reviewer. " * 60}, {"role": "user", "content": "Review: x = 1"},
        {"role": "assistant", "content": "ok"}, {"role": "user", "content": "and y = 2?"}]
body = {"messages": base, "max_tokens": 20, "temperature": 0}
r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                             {"Content-Type": "application/json"})).read())
t = r["timings"]
print(json.dumps(t))
ok = t["cache_tier"] == "disk" and t["cache_n"] > 0
print("PASS" if ok else "FAIL", f"tier={t['cache_tier']} reused={t['cache_n']} of {r['usage']['prompt_tokens']}")
sys.exit(0 if ok else 1)
