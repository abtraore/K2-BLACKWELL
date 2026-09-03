# K2-Horizon-0.9B GGUF on one RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

- The fork's `--jinja` chat template takes `reasoning_effort` through
  `--chat-template-kwargs`, and `--reasoning-format deepseek` splits the
  thinking into `reasoning_content` like vLLM's parser does. Tool calls
  came back structured without a dedicated parser flag.
- At 0.9B llama.cpp beats vLLM single-stream (523 vs 475 tok/s): the whole
  step is launch overhead, and llama.cpp's CUDA graphs are cheaper than
  vLLM's per-step scheduling at this size. vLLM wins the moment there is
  more than one stream (1,383 aggregate vs 512 here, where requests queue
  behind `--parallel 1`).
- Prefill is the other way round: 32,368 tok/s here versus 58,684 on vLLM.
- BF16 is the only GGUF IFM publishes for this size; at 2.2 GB there is
  no reason to quantize it.
