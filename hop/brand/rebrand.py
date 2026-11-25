#!/usr/bin/env python3
"""Rebrand the prebuilt web UI bundle (llama.cpp's, MIT) as Lion: name, titles, logo, icons, splash screens,
and the help texts that point at llama-server. Run by `make webui` on the extracted bundle.

Each text replacement must match (a different bundle version fails loudly instead of half-patching). Files under
_app/immutable/ are served with year-long caching, so every file whose content changes gets a new name and all
references are rewritten (cascading up to index.html, which is revalidated).

usage: brand/rebrand.py <webui dir>
"""
import hashlib
import os
import re
import shutil
import sys

UI = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
TEXT_EXT = (".js", ".css", ".html", ".json", ".webmanifest")

# (file glob: "main" = the big shared chunk, "*" = any JS/CSS/HTML file), old, new, expected count (None = >= 1)
REPLACEMENTS = [
    ("*", "`llama-ui`", "`Lion`", None),                       # app name constant, chat page title
    ("*", " - llama-ui`", " - Lion`", None),                   # page titles
    ("*", "`Search · llama.cpp`", "`Search · Lion`", 1),
    ("*", "`llama-ui-mcp`", "`lion-mcp`", 1),                  # MCP client name
    ("*", "LlamaUI-MCP-Server", "Lion-MCP-Server", 2),         # default id of an added MCP server
    ("*", "llama_settings_", "lion_settings_", 1),             # settings export file name
    ("*", "HARNESS:`llama.app`", "HARNESS:`lion`", 1),
    ("*", "`https://llama.app/v1/catalog.json`", "`./v1/catalog.json`", 1),  # served (empty) by hop
    ("*", "TOOLS:`--tools`", "TOOLS:`--no-tools`", 1),
    ("*", "`To enable Server Tools you need to run llama-server with ${ay.TOOLS} all or ${ay.TOOLS} <name> flag.${e}`",
     "`Server Tools are off: start hop without ${ay.TOOLS}.${e}`", 2),
    ("*", "<span>Run llama-server with <code> </code> flag to enable <strong>Server Tools</strong>.</span>",
     "<span>Start hop without <code> </code> to enable <strong>Server Tools</strong>.</span>", 1),
    ("*", "(Run <pre>llama-server</pre> with <pre> </pre> flag)",
     "(not available in <pre>hop</pre><pre class=\"hidden\"> </pre>)", 1),
    ("*", "Use llama-server proxy", "Use server proxy", 1),
    ("*", "Start the llama-server:", "Start the server:", 1),
    ("*", "<p>llama-server -hf ggml-org/gemma-3-4b-it-GGUF</p>", "<p>hop</p>", 1),
    ("*", "<p class=\"mt-1\">llama-server -m locally-stored-model.gguf</p>",
     "<p class=\"mt-1\">hop/serve.sh (servers in hop/backends.json)</p>", 1),
    ("*", " To get local file access, tell user to launch llama-server with the ${ay.AGENT} argument.`", "`", 1),
    ("*", "ggml-org/GLM-4.7-Flash-GGUF:Q8_0", "raon-1-nano", 1),
    ("*", "Llama Sans", "Lion Sans", None),                    # font-family names (the files keep their names)
    ("*", "Llama Mono", "Lion Mono", None),
    ("service-worker.js", "`llama-ui-${a}`", "`lion-${a}`", 1),  # cache name
    # a new service worker takes over at once (upstream waits for the "Update available -> Reload" click, and until
    # then serves the old cached index.html, headers included): one plain reload picks up a new UI version
    ("service-worker.js", "e.addAll(u.map(e=>new Request(e,{cache:`reload`})))))})",
     "e.addAll(u.map(e=>new Request(e,{cache:`reload`})))).then(()=>c.skipWaiting()))})", 1),
    ("service-worker.js", "e.filter(e=>e!==l).map(e=>caches.delete(e)))))})",
     "e.filter(e=>e!==l).map(e=>caches.delete(e)))).then(()=>c.clients.claim()))})", 1),
]
LOGO_OLD = re.compile(r'`<svg width="600" height="600" viewBox="0 0 600 600" fill="none" '
                      r'xmlns="http://www\.w3\.org/2000/svg">\n<path d="M600 392L504\.249 558.*?</svg>\n`', re.S)


def files():
    for dp, _, fns in os.walk(UI):
        for fn in fns:
            if fn.endswith(TEXT_EXT):
                yield os.path.join(dp, fn)


def main():
    if os.path.exists(os.path.join(UI, ".lion")):
        print("already rebranded")
        return
    content = {p: open(p, encoding="utf-8").read() for p in files()}
    orig = dict(content)
    for scope, old, new, want in REPLACEMENTS:
        n = 0
        for p in content:
            if scope != "*" and os.path.basename(p) != scope:
                continue
            c = content[p].count(old)
            if c:
                content[p] = content[p].replace(old, new)
                n += c
        if (want is None and n == 0) or (want is not None and n != want):
            sys.exit(f"rebrand: expected {want or '>= 1'} of {old[:70]!r}, found {n} (UI bundle changed?)")
    # in-app logo (an SVG string; the UI recolours its fills to currentColor)
    mark = open(os.path.join(HERE, "lion-mark.svg")).read().strip()
    n = 0
    for p in content:
        content[p], k = LOGO_OLD.subn(lambda _: "`" + mark + "\n`", content[p])
        n += k
    if n != 1:
        sys.exit(f"rebrand: logo SVG found {n} times (expected 1)")
    # index.html: drop the upstream build comment
    ip = os.path.join(UI, "index.html")
    content[ip] = re.sub(r"\A<!--.*?-->\s*", "", content[ip], flags=re.S)
    # manifest
    mp = os.path.join(UI, "manifest.webmanifest")
    content[mp] = (content[mp].replace('"name":"llama-ui"', '"name":"Lion"')
                   .replace('"short_name":"llama-ui"', '"short_name":"Lion"')
                   .replace("Local AI chat interface powered by llama.cpp", "Local models on your own GPU, loaded on demand"))
    for p, c in content.items():
        if c != orig[p]:
            open(p, "w", encoding="utf-8").write(c)

    # rename changed immutable files (content-addressed names), cascade through references
    imm = os.path.join(UI, "_app", "immutable")
    changed = {p for p in content if p.startswith(imm) and content[p] != orig[p]}
    done = set()
    while changed - done:
        for p in sorted(changed - done):
            data = open(p, "rb").read()
            base, ext = os.path.splitext(os.path.basename(p))
            new = f"{base}.li{hashlib.sha256(data).hexdigest()[:8]}{ext}"
            newp = os.path.join(os.path.dirname(p), new)
            os.rename(p, newp)
            done.add(newp)
            old_name = os.path.basename(p)
            for q in list(files()):
                s = open(q, encoding="utf-8").read()
                if old_name in s:
                    open(q, "w", encoding="utf-8").write(s.replace(old_name, new))
                    if q.startswith(imm) and q not in done:
                        changed.add(q)
            changed.discard(p)

    # icons and splash screens
    assets = os.path.join(HERE, "webui")
    for fn in os.listdir(assets):
        if not os.path.exists(os.path.join(UI, fn)):
            sys.exit(f"rebrand: {fn} is not in the bundle")
        shutil.copyfile(os.path.join(assets, fn), os.path.join(UI, fn))
    open(os.path.join(UI, "NOTICE.txt"), "w").write(
        "This web UI is llama.cpp's (https://github.com/ggml-org/llama.cpp, ui/apps/chat), MIT License,\n"
        "Copyright (c) 2023-2026 The ggml authors. Rebranded for Lion by brand/rebrand.py.\n")
    open(os.path.join(UI, ".lion"), "w").write("rebranded\n")
    left = [m for p in files() for m in re.findall(r"llama-ui|llama\.cpp|llama-server|ggml-org", open(p).read())]
    print(f"rebranded: {len(done)} files renamed, {len(os.listdir(assets))} icons/splashes, "
          f"remaining mentions: {sorted(set(left))}")


main()
