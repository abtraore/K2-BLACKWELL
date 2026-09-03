#!/bin/bash
# K2-Horizon-3.7B, IFM's BF16 GGUF, llama.cpp (IFM fork) on one RTX 5090, 131,072 context.
# Measured 2026-09-03: decode 140-152 tok/s single stream, prefill 12,366 tok/s at 3.9K, needle found at 60K in 100K.
# llama.cpp serves one request at a time here (--parallel 1); four concurrent streams
# measured 151 tok/s aggregate, i.e. they queue. Build the image first:
#   (cd ../llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
set -e
GPU="${GPU:-0}"; PORT="${PORT:-8061}"; CTX="${CTX:-131072}"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
GGUF="${GGUF:-$(ls "$HF"/hub/models--IFM--K2-Horizon-3.7B-GGUF/snapshots/*/*.gguf | head -1)}"
docker run -d --restart unless-stopped --name k2-horizon-3.7b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" \
  -v "$HF":"$HF":ro -p "$PORT":8000 \
  k2-blackwell/llamacpp:r1 \
  -m "$GGUF" --alias llamacpp/k2-horizon-3.7b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c "$CTX" --parallel 1 --metrics \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
echo "serving on http://localhost:$PORT/v1 as llamacpp/k2-horizon-3.7b"
