#!/bin/bash
# K2-Horizon-32B as a Q8_0 GGUF on two RTX 5090, llama.cpp (IFM fork), q8_0 KV, 131,072 context.
# Measured 2026-09-03: decode 39-42 tok/s single stream, prefill 4,022 tok/s at 4K. The 60K-in-100K
# needle was NOT retrieved (same result as the vLLM FP8 recipe for this model). bf16 KV at 131K does
# not fit next to the 35 GB of weights on two cards (cudaMalloc OOM on the 16.9 GB per-card KV).
# Quantize IFM's BF16 GGUF first (see README). Build the image first:
#   (cd ../llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
set -e
GPUS="${GPUS:-0,1}"; PORT="${PORT:-8064}"; CTX="${CTX:-131072}"; KV="${KV:-q8_0}"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
GGUF="${GGUF:-$HF/hub/gguf-k2/K2-Horizon-32B-Q8_0.gguf}"
docker run -d --restart unless-stopped --name k2-horizon-32b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -v "$HF":"$HF":ro -p "$PORT":8000 \
  k2-blackwell/llamacpp:r1 \
  -m "$GGUF" --alias llamacpp/k2-horizon-32b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c "$CTX" --parallel 1 --metrics \
  -ctk "$KV" -ctv "$KV" \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
echo "serving on http://localhost:$PORT/v1 as llamacpp/k2-horizon-32b"
