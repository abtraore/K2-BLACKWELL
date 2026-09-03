#!/bin/bash
# K2-Horizon-7B, IFM's BF16 GGUF, llama.cpp (IFM fork). Default: one RTX 5090 with q8_0 KV at 131,072.
# Measured 2026-09-03: decode 86-91 tok/s single stream, prefill 7,393 tok/s at 4K, needle found at
# 60K inside 100K. Two cards with bf16 KV (GPUS=0,1 KV=f16): 91-94 tok/s, prefill 10,638, needle found.
# bf16 KV on ONE card does not fit at 131K (18 GiB of KV next to 18 GB of weights: cudaMalloc OOM).
# The BF16 GGUF is the only one IFM publishes; it reads twice the bytes of the FP8 vLLM recipe per
# token, which is where the 132-135 vs 86-94 gap comes from. Build the image first:
#   (cd ../llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
set -e
GPUS="${GPUS:-0}"; PORT="${PORT:-8062}"; CTX="${CTX:-131072}"; KV="${KV:-q8_0}"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
GGUF="${GGUF:-$(ls "$HF"/hub/models--IFM--K2-Horizon-7B-GGUF/snapshots/*/*.gguf | head -1)}"
docker run -d --restart unless-stopped --name k2-horizon-7b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -v "$HF":"$HF":ro -p "$PORT":8000 \
  k2-blackwell/llamacpp:r1 \
  -m "$GGUF" --alias llamacpp/k2-horizon-7b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c "$CTX" --parallel 1 --metrics \
  -ctk "$KV" -ctv "$KV" \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
echo "serving on http://localhost:$PORT/v1 as llamacpp/k2-horizon-7b"
