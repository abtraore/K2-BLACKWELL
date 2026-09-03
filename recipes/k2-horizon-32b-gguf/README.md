# K2-Horizon-32B GGUF on two RTX 5090 (llama.cpp)

The dense 32B as a Q8_0 GGUF on the IFM llama.cpp fork, two cards, q8_0
KV, 131,072-token context: **39-42 tok/s** single-stream decode, prefill
**4,022 tok/s** at 4K. Single request at a time (`--parallel 1`).

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call, reasoning in `reasoning_content`. The 60K-in-100K needle was NOT
retrieved (79 s, confident wrong answer), the same result the vLLM FP8
recipe got for this model, so it looks like the model rather than either
engine.

The vLLM recipe (`../k2-horizon-32b-fp8/`) is the faster way to run the
32B on two cards (66 tok/s on IFM's FP8 export vs 39-42 here on Q8_0,
half the bytes per token); this lane exists for people who want a GGUF.

## Contents

- `launch.sh`: parameterized launcher (`GPUS`/`PORT`/`CTX`/`KV`/`GGUF`)
- `NOTES.md`: KV arithmetic and the quantize step

## Make the GGUF

IFM publishes a 69.6 GB BF16 GGUF. Quantize it with the shared image:

```bash
hf download IFM/K2-Horizon-32B-GGUF
HF=$HOME/.cache/huggingface; mkdir -p $HF/hub/gguf-k2
BF=$(ls $HF/hub/models--IFM--K2-Horizon-32B-GGUF/snapshots/*/*.gguf | head -1)
docker run --rm -v $HF:$HF --entrypoint /app/llama-quantize k2-blackwell/llamacpp:r1 $BF $HF/hub/gguf-k2/K2-Horizon-32B-Q8_0.gguf Q8_0
```

Result: 34.8 GB. On our single spinning /tank the quantize took about
25 minutes, disk-bound.

## Start the server

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
cd k2-horizon-32b-gguf && ./launch.sh
```

Or the full command without the script:

```bash
HF=$HOME/.cache/huggingface
docker run -d --restart unless-stopped --name k2-horizon-32b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES=0,1 -v $HF:$HF:ro -p 8064:8000 \
  k2-blackwell/llamacpp:r1 \
  -m $HF/hub/gguf-k2/K2-Horizon-32B-Q8_0.gguf --alias llamacpp/k2-horizon-32b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c 131072 --parallel 1 --metrics \
  -ctk q8_0 -ctv q8_0 \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
```

Serving on `http://localhost:8064/v1`, model name `llamacpp/k2-horizon-32b`,
about 90 seconds after `docker run` with the GGUF in page cache.
