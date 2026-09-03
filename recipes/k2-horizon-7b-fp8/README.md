# K2-Horizon-7B-FP8 on one RTX 5090

IFM's own FP8 export of the dense 7B on vLLM with fp8 KV at a 131,072-token
context: **132-135 tok/s** single-stream decode, **535 tok/s aggregate** at
four streams (135 per stream, so batching is free at this size), prefill
**17,552 tok/s** at 4K, and a **211,424-token KV pool** on one 32 GB card
(1.61 full-length sequences).

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call through the `k2_horizon` parser, reasoning in `reasoning_content`,
**needle found at 60K inside a 100K prompt** (100,035 tokens, 25 s end to
end).

## Contents

- `launch.sh`: parameterized launcher (`GPU`/`PORT`/`CTX`/`KV`)
- `NOTES.md`: why fp8 KV is not optional at 131K on this card

## Start the server

Build the shared image once (`../vllm-k2-image/`), then:

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
cd k2-horizon-7b-fp8 && ./launch.sh
```

Or the full command without the script:

```bash
docker run -d --restart unless-stopped --name k2-horizon-7b-fp8 \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8051:8000 \
  k2-blackwell/vllm:r1 \
  --model IFM/K2-Horizon-7B-FP8 --served-model-name vllm/k2-horizon-7b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
```

Serving on `http://localhost:8051/v1`, model name `vllm/k2-horizon-7b`,
about two minutes after `docker run` once the 11.1 GB download is in. Send
`{"chat_template_kwargs": {"reasoning_effort": "high"}}` on every request
and budget `max_tokens` for the reasoning; IFM reports all their numbers at
high effort.
