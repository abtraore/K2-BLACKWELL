# K2-Horizon-3.7B GGUF on one RTX 5090 (llama.cpp)

IFM's BF16 GGUF of the 3.7B on the IFM llama.cpp fork at a 131,072-token
context with bf16 KV on one 32 GB card: **140-152 tok/s** single-stream
decode, prefill **12,366 tok/s** at 3.9K, and the **60K-in-100K needle
found** (100,035-token prompt, 36 s end to end). Single request at a time
(`--parallel 1`): four concurrent streams queued to 151 tok/s aggregate.

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call, reasoning in `reasoning_content`.

Note what this card could not do on vLLM: the same model at 131K in bf16 KV
was refused there (18 GiB of cache needed after the weights and vLLM's
reserve). llama.cpp allocates the full 131K KV (144 KB per token, 18 GiB)
next to the 7.5 GB of weights and runs.

## Contents

- `launch.sh`: parameterized launcher (`GPU`/`PORT`/`CTX`/`GGUF`)
- `NOTES.md`: KV arithmetic and the engine comparison at this size

## Start the server

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
hf download IFM/K2-Horizon-3.7B-GGUF
cd k2-horizon-3.7b-gguf && ./launch.sh
```

Or the full command without the script:

```bash
HF=$HOME/.cache/huggingface
GGUF=$(ls $HF/hub/models--IFM--K2-Horizon-3.7B-GGUF/snapshots/*/*.gguf | head -1)
docker run -d --restart unless-stopped --name k2-horizon-3.7b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES=0 -v $HF:$HF:ro -p 8061:8000 \
  k2-blackwell/llamacpp:r1 \
  -m $GGUF --alias llamacpp/k2-horizon-3.7b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c 131072 --parallel 1 --metrics \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
```

Serving on `http://localhost:8061/v1`, model name `llamacpp/k2-horizon-3.7b`,
about 40 seconds after `docker run` once the 7.5 GB GGUF is in.
