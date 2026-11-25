# hop

One web UI and one OpenAI-compatible API in front of the local inference servers of this repo. Each server starts
when a request needs one of its models and stops after 10 minutes idle. Only one runs at a time: they share the GPU.

- **Port:** `http://127.0.0.1:11434/`. The web UI is at `/`; API clients use base URL `http://127.0.0.1:11434/v1`.
- **Models** (`backends.json`):

  | model | server | what for |
  |---|---|---|
  | `lfm3`, `lfm3-vl` | liquid-engine | text; images and video |
  | `d1` | liquid-engine | decisions, at `/v1/systemone` |
  | `raon-1-nano` | raon-engine | text |

- **Web UI:** llama.cpp's chat UI (MIT), branded **Lion**. It runs in llama-server's router mode, so it shows a
  model picker over all models.
- **Server tools** (`web_search`, `web_fetch`, `pdf`) run in hop itself, so using one never loads a model.

## How a request is served

1. **Pick the server.** The request's `model` decides it. Without a model, the running server is used, or the
   default model's.
2. **Swap if needed.** When another server runs, hop waits until it has no request in flight, then stops it
   (SIGTERM to its process group). Requests for it that arrive meanwhile wait too, so a swap is not starved.
3. **Start the server** and wait for its `/health`. A streamed request gets SSE comments (`: loading ...`) meanwhile,
   so the UI and clients do not time out.
4. **Forward the request.** Streams pass through as they arrive. A client that disconnects closes the upstream
   connection, which cancels the generation there.
5. **Unload when idle.** A server with nothing in flight for `idle_unload_s` (600 s) is stopped.

A swap costs the stop plus the start: a few seconds for these engines.

`/props?model=X` and `/v1/models` never start a server. Props are cached (`~/.cache/hop/props.json`) from the last
time the model's server ran, or built from `backends.json`. Server logs are in `~/.cache/hop/logs/`.

## Run

```sh
make webui                 # the Lion web UI bundle (pinned llama.cpp release, checksum-verified)
make webtools              # node dependencies of the tools (in servers/liquine/tools/webtools)
./serve.sh                 # or: python3 hop.py --config backends.json [--listen host:port] [--idle S] [--no-tools]
```

**The servers.** `backends.json` starts each server as `servers/<name>/serve.sh` (paths are relative to the config) on
an internal port (11500, 11501), with `--no-ui --no-tools`. In this repo, `servers/liquine` and `servers/raon` are
symlinks to `../liquid-engine` and `../raon-engine`. Add another server by editing that file. A server needs:

- `/health`, answering 200 once it is ready;
- the OpenAI routes;
- exit on SIGTERM.

`ui_dir` and `tools_dir` take a list of candidates; the first present wins (`webui`, then `build/webui`).

**Installed copy.** Everything lives under `~/hop`, and hop starts as `hop` (`~/.local/bin/hop` → `~/hop/serve.sh`):

```
~/hop/
  serve.sh  hop.py  backends.json  README.md    the gateway
  webui/                                        the Lion web UI
  tools/webtools/                               web_search, web_fetch, pdf (one copy, shared)
  servers/liquine/                              liquid-server, liquid, its web UI; serve.sh  (liquid-serve)
  servers/raon/                                 raon-server, raon, its web UI; serve.sh      (raon-serve)
```

Each server's `tools` is a symlink to `../../tools`. Its standalone script (`~/.local/bin/liquid-serve`, `raon-serve`)
still runs it alone, with its own UI and tools.

**Port 11434** is also liquid-serve's default: run either hop or a standalone liquid-serve, not both. A standalone
server you start yourself (for example raon-serve on :11435) holds GPU memory that hop then cannot use.

## Checks

`python3 tests/gateway_test.py` runs hop on two fake servers (no GPU) and checks:

- routing by model, start on demand, and no swap between models of one server;
- a swap waits for the request in flight;
- pings while a server starts;
- a disconnected client does not block the next swap;
- idle unload;
- the router-mode endpoints and the UI.

## Brand

`brand/` holds:

- `lion-mark.svg`;
- `build_assets.py` (icons and splash screens, `make assets`);
- `rebrand.py` (names, help texts, logo, icons; renames each content-hashed file it changes).
