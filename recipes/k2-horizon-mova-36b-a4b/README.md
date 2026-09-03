# K2-Horizon-MoVA-36B-A4B on two RTX 5090

The sparse K2 Horizon (100 routed MLP experts plus 64 routed value experts
per attention layer) on vLLM at TP=2, weights quantized to fp8 at load, fp8
KV, 65,536-token context: **52-53 tok/s** single-stream decode, **196 tok/s
aggregate** at four streams (50 per stream), prefill **5,375 tok/s** at 4K,
and an **80,320-token KV pool** across the two cards (1.23 full-length
sequences). Expert parallel (`EP=1`) measured the same decode (51-52) and a
slower prefill (3,153), so it is off by default.

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call through the `k2_horizon` parser, reasoning in `reasoning_content`.
(The 100K needle gate does not apply at a 65K context.)

Two things you need to know before the numbers make sense:

- **It needs one patch to run quantized at all.** Upstream's MoVA path
  feeds the raw value-expert weights to a bf16-only fused kernel, so
  `--quantization fp8` crashes at compile time (`Hidden size mismatch
  2560 != 512`). The Dockerfile here keeps those 64 experts per layer in
  bf16 (they are 7.5B of the 36B parameters) and quantizes the rest.
  Without any quantization the bf16 checkpoint is 74.9 GB and does not fit
  two 32 GB cards.
- **KV is dense GQA at 98 KB per token in fp8**, four times Flash-Next's
  cost and two times the 7B sibling's. With the bf16 value experts on the
  cards, 131,072 does not fit (6.0 GiB per rank needed, 4.65 free); 65,536
  is the two-card ceiling for this layout. The 524,288 in the config is a
  four-card-or-more number.

## Contents

- `Dockerfile` + `patches/k2_horizon.py`: the shared image plus the
  one-line value-expert change, with its upstream thread and retirement
  condition in the Dockerfile header
- `launch.sh`: parameterized launcher (`GPUS`/`PORT`/`CTX`/`EP`)
- `NOTES.md`: the crash, the memory arithmetic, and what the 51 tok/s is
  made of

## Start the server

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
cd k2-horizon-mova-36b-a4b && docker build -t k2-blackwell/vllm-mova:r1 . && ./launch.sh
```

Or the full command without the script:

```bash
docker run -d --restart unless-stopped --name k2-horizon-mova-36b \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8053:8000 \
  k2-blackwell/vllm-mova:r1 \
  --model IFM/K2-Horizon-MoVA-36B-A4B --served-model-name vllm/k2-horizon-mova-36b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --quantization fp8 --kv-cache-dtype fp8 \
  --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
```

Serving on `http://localhost:8053/v1`, model name
`vllm/k2-horizon-mova-36b`, about eight minutes after `docker run` once
the 74.9 GB download is in (load plus fp8 quantization plus compile). Send
`{"chat_template_kwargs": {"reasoning_effort": "high"}}` on every request
and budget `max_tokens` for the reasoning.
