#!/bin/bash
# K2-Horizon-32B-FP8 (IFM's own FP8 export) on two RTX 5090 (sm120), vLLM TP=2, fp8 KV, 131,072 context.
# Measured 2026-09-03: decode 65.6-66.0 tok/s single stream, 261 tok/s aggregate at 4 streams
# (66 per stream), prefill 5,948 tok/s at 4K, KV pool 141,088 tokens (1.08 full-length sequences).
# The 60K-in-100K needle was NOT retrieved by this model in our run. Build the image first:
#   (cd ../vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
set -e
GPUS="${GPUS:-0,1}"; PORT="${PORT:-8052}"; CTX="${CTX:-131072}"; KV="${KV:-fp8}"
KV_ARGS=(); [ "$KV" != auto ] && KV_ARGS=(--kv-cache-dtype "$KV")
docker run -d --restart unless-stopped --name k2-horizon-32b-fp8 \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p "$PORT":8000 \
  k2-blackwell/vllm:r1 \
  --model IFM/K2-Horizon-32B-FP8 --served-model-name vllm/k2-horizon-32b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len "$CTX" --gpu-memory-utilization 0.90 "${KV_ARGS[@]}" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
echo "serving on http://localhost:$PORT/v1 as vllm/k2-horizon-32b"
