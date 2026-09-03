# K2-Horizon-7B GGUF on RTX 5090 (llama.cpp)

IFM's BF16 GGUF of the dense 7B on the IFM llama.cpp fork at a
131,072-token context. Two measured layouts:

| layout | decode | prefill @4K | needle 60K-in-100K |
|---|---|---|---|
| one card, q8_0 KV (default) | 86-91 tok/s | 7,393 tok/s | found, 48 s |
| two cards, bf16 KV (`GPUS=0,1 KV=f16`) | 91-94 tok/s | 10,638 tok/s | found, 37 s |

bf16 KV on one card does not fit at 131K (the 18 GiB cache next to 18 GB of
weights fails at `cudaMalloc`). Single request at a time (`--parallel 1`):
four concurrent streams queued to 89-94 tok/s aggregate.

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call, reasoning in `reasoning_content`.

The vLLM recipe for this model (`../k2-horizon-7b-fp8/`) runs IFM's FP8
export at 132-135 tok/s on one card: it reads half the bytes per token.
IFM publishes only the BF16 GGUF; a Q8_0 made with `llama-quantize` from
the shared image would close most of that gap and was not needed to fit.

## Contents

- `launch.sh`: parameterized launcher (`GPUS`/`PORT`/`CTX`/`KV`/`GGUF`)
- `NOTES.md`: the KV arithmetic and the two layouts

## Start the server

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
hf download IFM/K2-Horizon-7B-GGUF
cd k2-horizon-7b-gguf && ./launch.sh
```

Or the full command without the script:

```bash
HF=$HOME/.cache/huggingface
GGUF=$(ls $HF/hub/models--IFM--K2-Horizon-7B-GGUF/snapshots/*/*.gguf | head -1)
docker run -d --restart unless-stopped --name k2-horizon-7b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES=0 -v $HF:$HF:ro -p 8062:8000 \
  k2-blackwell/llamacpp:r1 \
  -m $GGUF --alias llamacpp/k2-horizon-7b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c 131072 --parallel 1 --metrics \
  -ctk q8_0 -ctv q8_0 \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
```

Serving on `http://localhost:8062/v1`, model name `llamacpp/k2-horizon-7b`,
about 50 seconds after `docker run` once the 18 GB GGUF is in.
