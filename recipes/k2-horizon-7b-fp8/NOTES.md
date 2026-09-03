# K2-Horizon-7B-FP8 on one RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

## fp8 KV is what makes 131K fit

- 36 layers, 8 KV heads, head_dim 128: 144 KB per token in bf16. One
  131,072-token request needs 18.0 GiB of cache, and after the 11 GB of
  weights the card has about 14.9 GiB for KV, so vLLM refuses to start
  (`To serve at least one request with the model's max seq len (131072),
  18.0 GiB KV cache is needed`). With `--kv-cache-dtype fp8` the same
  context costs 9 GiB and the pool comes out at 211,424 tokens.
- The model's config says 524,288 native. On a 32 GB card that is a
  number on paper: 72 GB of KV per sequence in bf16, 36 GB in fp8.

## What the numbers mean

- Decode 132-135 tok/s single stream is a 7 GB weight read per token on
  one 5090; the four-stream run held 135 per stream and 535 aggregate,
  which says the decode is bandwidth-bound with headroom in compute.
- Prefill 17,552 tok/s at 4K; the 100K needle prompt took 25 s end to end
  including the answer.
- Reasoning at high effort is short on simple prompts (81 characters for
  the arithmetic question) and the parser separates it cleanly.

## Operational

- Shared image from `../vllm-k2-image/`, nothing model-specific patched.
- `--trust-remote-code` is required for the checkpoint's config class.
- Zero error lines in the server log across boot, probes, both suites and
  the four-stream stress.
