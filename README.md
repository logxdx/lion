<p align="center"><img src="lion/brand/lion-mark.svg" alt="Lion" height="96"></p>

# Lion

CUDA inference engines for small models on a 6 GB RTX 4050, behind one gateway and web UI.

| directory | what |
|---|---|
| [`liquid-engine/`](liquid-engine/) | LFM3-1.7B, LFM3-VL-2B, d1-2B |
| [`raon-engine/`](raon-engine/) | KRAFTON Raon-1 nano |
| [`lion/`](lion/) | gateway: starts the server a model needs, on demand, on `:11434` |

```sh
make -C liquid-engine && make -C raon-engine && make -C lion webui
lion/serve.sh    # http://127.0.0.1:11434/  (API: http://127.0.0.1:11434/v1)
```
