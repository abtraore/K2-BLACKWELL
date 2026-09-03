# K2-Horizon-3.7B on one RTX 5090 (vLLM)

The dense 3.7B on vLLM, bf16 weights with fp8 KV at a 131,072-token
context: **147-151 tok/s** single-stream decode, **563 tok/s aggregate** at
four streams (144 per stream), prefill **18,300 tok/s** at 3.9K, and a
**250,768-token KV pool** on one 32 GB card (1.91 full-length sequences).

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call through the `k2_horizon` parser, reasoning in `reasoning_content`.
The 60K-in-100K needle was NOT found in this run. The same model on
llama.cpp with bf16 KV (`../k2-horizon-3.7b-gguf/`) found its needle, so
fp8 KV is the suspect; one sample each, so treat it as a flag to re-test on
your own prompts rather than a verdict.

fp8 KV is not a tuning choice here: in bf16 the 131K context needs 18 GiB
of cache and the card has about 15 after the weights, and vLLM refuses to
start (see NOTES.md).

## Contents

- `launch.sh`: parameterized launcher (`GPU`/`PORT`/`CTX`/`KV`)
- `NOTES.md`: the bf16-KV refusal, the fp8-KV needle miss, engine comparison

## Start the server

Build the shared image once (`../vllm-k2-image/`), then:

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
cd k2-horizon-3.7b && ./launch.sh
```

Or the full command without the script:

```bash
docker run -d --restart unless-stopped --name k2-horizon-3.7b \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8054:8000 \
  k2-blackwell/vllm:r1 \
  --model IFM/K2-Horizon-3.7B --served-model-name vllm/k2-horizon-3.7b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
```

Serving on `http://localhost:8054/v1`, model name `vllm/k2-horizon-3.7b`,
about 90 seconds after `docker run` once the 10.1 GB download is in. Send
`{"chat_template_kwargs": {"reasoning_effort": "high"}}` on every request
and budget `max_tokens` for the reasoning.
