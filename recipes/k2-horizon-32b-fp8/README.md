# K2-Horizon-32B-FP8 on two RTX 5090

IFM's FP8 export of the dense 32B on vLLM, TP=2 with fp8 KV at a
131,072-token context: **65.6-66.0 tok/s** single-stream decode, **261 tok/s
aggregate** at four streams (66 per stream), prefill **5,948 tok/s** at 4K,
and a **141,088-token KV pool** across the two cards (1.08 full-length
sequences, so one deep session at a time).

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call through the `k2_horizon` parser, reasoning in `reasoning_content`
(1,227 characters at high effort for the arithmetic question). The
60K-in-100K needle was NOT retrieved in our run (100,035-token prompt,
65 s end to end); the 7B sibling found the same needle, so treat this as
a model result, not a serving fault, and re-test on your own prompts.

## Contents

- `launch.sh`: parameterized launcher (`GPUS`/`PORT`/`CTX`/`KV`)
- `NOTES.md`: KV arithmetic on 32 GB cards, why the pool is 1.08x

## Start the server

Build the shared image once (`../vllm-k2-image/`), then:

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
cd k2-horizon-32b-fp8 && ./launch.sh
```

Or the full command without the script:

```bash
docker run -d --restart unless-stopped --name k2-horizon-32b-fp8 \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8052:8000 \
  k2-blackwell/vllm:r1 \
  --model IFM/K2-Horizon-32B-FP8 --served-model-name vllm/k2-horizon-32b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
```

Serving on `http://localhost:8052/v1`, model name `vllm/k2-horizon-32b`,
about three and a half minutes after `docker run` once the 37.4 GB
download is in. Send `{"chat_template_kwargs": {"reasoning_effort": "high"}}`
on every request and budget `max_tokens` for the reasoning.
