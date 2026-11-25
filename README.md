<p align="center"><img src="hop/brand/lion-mark.svg" alt="Lion" height="96"></p>

# Lion

From-scratch CUDA inference engines for small models on one RTX 4050 Laptop (6 GB), and a gateway that serves them
behind one web UI, Lion.

This repository was [liquine](https://github.com/logxdx/lion) (renamed to lion). It now holds both engines and the
gateway; the local checkout is the `inference-engines` folder.

| directory | what |
|---|---|
| [`liquid-engine/`](liquid-engine/) | LFM3-1.7B (`lfm3`), LFM3-VL-2B (`lfm3-vl`, images and video), d1-2B (`d1`, decisions). `liquid-server`, OpenAI-compatible. This repository's original content. |
| [`raon-engine/`](raon-engine/) | KRAFTON Raon-1 nano (`raon-1-nano`): long-context, 4 × 128k at ~60 tok/s per stream. `raon-server`, OpenAI-compatible. |
| [`hop/`](hop/) | The gateway: the **Lion** web UI and one API on `:11434`. It starts whichever server the requested model needs, one at a time, and unloads it after 10 minutes idle. |

## Quick start

```sh
make -C liquid-engine && make -C liquid-engine webtools    # CUDA 13 (nvcc in /opt/cuda), sm_89
make -C raon-engine
make -C hop webui
hop/serve.sh                     # http://127.0.0.1:11434/  (API base URL http://127.0.0.1:11434/v1)
```

Each engine also runs alone, with its own web UI:

- `liquid-engine/serve.sh` on :11434;
- `raon-engine/serve.sh` on :11435.

The READMEs in each directory have the details. `liquid-engine/DETAILS.md` and `raon-engine/docs/plan-speedups.md`
hold the measurements.

## History

The two engines were imported with their full history (`git subtree`):

- `liquid-engine/` is liquine's `main` at `a226b21`, which stays an ancestor of `main` here;
- `raon-engine/` is raon-engine's `main` at `d44ebe5`.

To bring in later work from either engine's own repo:

```sh
git subtree pull --prefix=<dir> <repo> <branch>
```
