# K2-Horizon-0.9B on one RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

## What the numbers mean at this size

- 28 layers, 8 KV heads, head_dim 64: KV is 56 KB per token in bf16, which
  is why one 32 GB card holds 475,728 tokens of cache next to 2.2 GB of
  weights. The context is capped by the model (131,072 native), not by
  memory.
- Decode at 447-475 tok/s is one card's bandwidth divided by a 1.8 GB
  weight read per token; four streams at 363 each show the batch is still
  cheap. Prefill at 58,684 tok/s means a 100K prompt lands in under two
  seconds.
- The 60K-in-100K needle came back wrong. The prompt was accepted
  (100,036 tokens, 13 s end to end) and the answer was confident and
  unrelated. Treat 0.9B as a short-context model in practice, whatever
  the config says.
- Reasoning is on by default and short at this size (246 characters for a
  one-line arithmetic question at high effort).

## Operational

- The image is the shared one in `../vllm-k2-image/`: a pinned nightly
  plus the six upstream K2 files from vllm-project/vllm#55063. Nothing
  model-specific is patched.
- `--trust-remote-code` is required: the checkpoint ships its own
  `configuration_k2_horizon.py`, and vLLM reads the config through it.
- Zero error lines in the server log across the whole session (boot,
  probes, two suites, the four-stream stress).
