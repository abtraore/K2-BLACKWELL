# K2-Horizon-0.9B GGUF on one RTX 5090 (llama.cpp)

IFM's BF16 GGUF of the 0.9B on the IFM llama.cpp fork, 131,072 context:
**519-523 tok/s** single-stream decode (the vLLM recipe does 447-475),
prefill **32,368 tok/s** at 4.3K. Single request at a time (`--parallel
1`): four concurrent streams queued to 512 tok/s aggregate.

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call through the fork's `--jinja` template, reasoning in
`reasoning_content`. The 60K-in-100K needle was not found, the same result
as on vLLM: the 0.9B does not retrieve at that depth.

## Contents

- `launch.sh`: parameterized launcher (`GPU`/`PORT`/`CTX`/`GGUF`)
- `NOTES.md`: engine-vs-engine notes at this size

## Start the server

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
hf download IFM/K2-Horizon-0.9B-GGUF
cd k2-horizon-0.9b-gguf && ./launch.sh
```

Or the full command without the script:

```bash
HF=$HOME/.cache/huggingface
GGUF=$(ls $HF/hub/models--IFM--K2-Horizon-0.9B-GGUF/snapshots/*/*.gguf | head -1)
docker run -d --restart unless-stopped --name k2-horizon-0.9b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES=0 -v $HF:$HF:ro -p 8060:8000 \
  k2-blackwell/llamacpp:r1 \
  -m $GGUF --alias llamacpp/k2-horizon-0.9b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c 131072 --parallel 1 --metrics \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
```

Serving on `http://localhost:8060/v1`, model name `llamacpp/k2-horizon-0.9b`,
about 20 seconds after `docker run` once the 2.2 GB GGUF is in.
