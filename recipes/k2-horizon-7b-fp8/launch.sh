#!/bin/bash
# K2-Horizon-7B-FP8 (IFM's own FP8 export) on one RTX 5090 (sm120), vLLM, fp8 KV, 131,072 context.
# Measured 2026-09-03: decode 132-135 tok/s single stream, 535 tok/s aggregate at 4 streams
# (135 per stream), prefill 17,552 tok/s at 4K, KV pool 211,424 tokens (1.61 full-length
# sequences), needle found at 60K inside a 100K prompt. Build the image first:
#   (cd ../vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
# KV=auto reverts to bf16 KV; then the 131K context no longer fits one card (18 GiB needed
# vs ~15 free), drop CTX to 98304 or lower.
set -e
GPU="${GPU:-0}"; PORT="${PORT:-8051}"; CTX="${CTX:-131072}"; KV="${KV:-fp8}"
KV_ARGS=(); [ "$KV" != auto ] && KV_ARGS=(--kv-cache-dtype "$KV")
docker run -d --restart unless-stopped --name k2-horizon-7b-fp8 \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES="$GPU" \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p "$PORT":8000 \
  k2-blackwell/vllm:r1 \
  --model IFM/K2-Horizon-7B-FP8 --served-model-name vllm/k2-horizon-7b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --max-model-len "$CTX" --gpu-memory-utilization 0.90 "${KV_ARGS[@]}" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
echo "serving on http://localhost:$PORT/v1 as vllm/k2-horizon-7b"
