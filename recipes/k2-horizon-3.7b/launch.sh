#!/bin/bash
# K2-Horizon-3.7B (bf16 weights) on one RTX 5090 (sm120), vLLM, fp8 KV, 131,072 context.
# Measured 2026-09-03: decode 147-151 tok/s single stream, 563 tok/s aggregate at 4 streams
# (144 per stream), prefill 18,300 tok/s at 3.9K, KV pool 250,768 tokens (1.91 full-length
# sequences); the 60K-in-100K needle was NOT found on this engine+fp8-KV run (the GGUF/bf16 run found it). Build the image first:
#   (cd ../vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .)
# KV=auto reverts to bf16 KV; then the 131K context no longer fits one card (36 GiB at 262K, 18 GiB at 131K needed
# vs ~15 free), drop CTX to 98304 or lower.
set -e
GPU="${GPU:-0}"; PORT="${PORT:-8054}"; CTX="${CTX:-131072}"; KV="${KV:-fp8}"
KV_ARGS=(); [ "$KV" != auto ] && KV_ARGS=(--kv-cache-dtype "$KV")
docker run -d --restart unless-stopped --name k2-horizon-3.7b \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES="$GPU" \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p "$PORT":8000 \
  k2-blackwell/vllm:r1 \
  --model IFM/K2-Horizon-3.7B --served-model-name vllm/k2-horizon-3.7b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --max-model-len "$CTX" --gpu-memory-utilization 0.90 "${KV_ARGS[@]}" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
echo "serving on http://localhost:$PORT/v1 as vllm/k2-horizon-3.7b"
