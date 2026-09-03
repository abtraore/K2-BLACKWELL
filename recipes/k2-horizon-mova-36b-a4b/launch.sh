#!/bin/bash
# K2-Horizon-MoVA-36B-A4B on two RTX 5090 (sm120), vLLM TP=2 (no expert parallel), weights
# quantized to fp8 at load, MoVA value experts kept bf16 (see Dockerfile), fp8 KV, 65,536 context.
# Measured 2026-09-03: decode 52-53 tok/s single stream, 196 tok/s aggregate at 4 streams
# (50 per stream), prefill 5,375 tok/s at 4K, KV pool 80,320 tokens (1.23 full-length sequences).
# With --enable-expert-parallel (EP=2) decode is the same (51-52) and prefill drops to 3,153: EP=${EP:-0} adds it back.
# 131,072 does NOT fit on two cards with this layout (6.0 GiB of KV per rank needed, 4.65 free
# at 0.93 utilization). Build the images first:
#   (cd ../vllm-k2-image && docker build -t k2-blackwell/vllm:r1 .) && docker build -t k2-blackwell/vllm-mova:r1 .
set -e
GPUS="${GPUS:-0,1}"; PORT="${PORT:-8053}"; CTX="${CTX:-65536}"; EP="${EP:-0}"
EP_ARGS=(); [ "$EP" = 1 ] && EP_ARGS=(--enable-expert-parallel)
docker run -d --restart unless-stopped --name k2-horizon-mova-36b \
  --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p "$PORT":8000 \
  k2-blackwell/vllm-mova:r1 \
  --model IFM/K2-Horizon-MoVA-36B-A4B --served-model-name vllm/k2-horizon-mova-36b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 2 "${EP_ARGS[@]}" --disable-custom-all-reduce \
  --quantization fp8 --kv-cache-dtype fp8 \
  --max-model-len "$CTX" --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser k2_horizon --tool-call-parser k2_horizon --enable-auto-tool-choice
echo "serving on http://localhost:$PORT/v1 as vllm/k2-horizon-mova-36b"
