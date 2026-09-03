#!/bin/bash
# K2-Horizon-0.9B on one RTX 5090 (sm120), vLLM, bf16 weights, native 131,072 context.
# Measured 2026-09-03: decode 447-475 tok/s single stream, 1,383 tok/s aggregate
# at 4 streams (363 per stream), prefill 58,684 tok/s at 4.3K, KV pool 475,728
# tokens (3.63 full-length sequences). Build the image first:
#   (cd ../vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
set -e
GPU="${GPU:-0}"
PORT="${PORT:-8050}"
CTX="${CTX:-131072}"
docker run -d --restart unless-stopped --name k2-horizon-0.9b \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES="$GPU" \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p "$PORT":8000 \
  k2-blackwell/vllm:r1 \
  --model IFM/K2-Horizon-0.9B --served-model-name vllm/k2-horizon-0.9b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --max-model-len "$CTX" --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
echo "serving on http://localhost:$PORT/v1 as vllm/k2-horizon-0.9b"
