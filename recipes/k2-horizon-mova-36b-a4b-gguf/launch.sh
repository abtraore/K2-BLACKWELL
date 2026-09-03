#!/bin/bash
# K2-Horizon-MoVA-36B-A4B as a Q8_0 GGUF on two RTX 5090, llama.cpp (IFM fork), bf16 KV, 131,072 context.
# Measured 2026-09-03: decode 131-142 tok/s single stream, prefill 6,737 tok/s at 4K, needle found at
# 60K inside 100K (33 s). IFM publishes only a 74.9 GB BF16 GGUF; convert the safetensors yourself
# to Q8_0 (39.8 GB, see README) or quantize their BF16 GGUF with llama-quantize from the shared image.
# Build the image first:
#   (cd ../llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
set -e
GPUS="${GPUS:-0,1}"; PORT="${PORT:-8063}"; CTX="${CTX:-131072}"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
GGUF="${GGUF:-$HF/hub/gguf-k2/K2-Horizon-MoVA-36B-A4B-Q8_0.gguf}"
docker run -d --restart unless-stopped --name k2-horizon-mova-36b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -v "$HF":"$HF":ro -p "$PORT":8000 \
  k2-blackwell/llamacpp:r1 \
  -m "$GGUF" --alias llamacpp/k2-horizon-mova-36b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c "$CTX" --parallel 1 --metrics \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
echo "serving on http://localhost:$PORT/v1 as llamacpp/k2-horizon-mova-36b"
