# K2-Horizon-0.9B on one RTX 5090

IFM's smallest K2 Horizon model on vLLM, bf16 weights, at its native
131,072-token context: **447-475 tok/s** single-stream decode, **1,383 tok/s
aggregate** at four streams (363 per stream), prefill **58,684 tok/s** at
4.3K, and a **475,728-token KV pool** on one 32 GB card (3.63 full-length
sequences).

Gates: greedy 19*23 exact, generated `is_prime` executes, a tool call comes
back structured through the `k2_horizon` parser, reasoning lands in
`reasoning_content`. The 60K-in-100K needle was NOT found by this model; at
0.9B that is a capability limit, not a serving bug (the same prompt is
retrieved by the larger siblings in this repo).

## Contents

- `launch.sh`: parameterized launcher (`GPU`/`PORT`/`CTX`)
- `NOTES.md`: what the measurements mean at this size

## Start the server

Build the shared image once (`../vllm-k2-image/`, a pinned vLLM nightly
plus the six upstream K2 files), then:

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
cd k2-horizon-0.9b && ./launch.sh
```

Or the full command without the script:

```bash
docker run -d --restart unless-stopped --name k2-horizon-0.9b \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8050:8000 \
  k2-blackwell/vllm:r1 \
  --model IFM/K2-Horizon-0.9B --served-model-name vllm/k2-horizon-0.9b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --max-model-len 131072 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
```

Serving on `http://localhost:8050/v1`, model name `vllm/k2-horizon-0.9b`,
about four minutes after `docker run` (the 2.2 GB download is the long
part; boot itself is under a minute). Send
`{"chat_template_kwargs": {"reasoning_effort": "high"}}` with every request:
IFM reports all their numbers at high effort and the model thinks before
it answers, so budget `max_tokens` for the reasoning.
