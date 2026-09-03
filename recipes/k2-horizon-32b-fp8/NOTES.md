# K2-Horizon-32B-FP8 on two RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

## KV arithmetic

- 64 layers, 8 KV heads, head_dim 128: 256 KB per token in bf16, 128 KB in
  fp8. After 18.7 GB of fp8 weights per card, the pair has room for
  141,088 tokens of fp8 KV: one 131K session, or a handful of short ones.
  bf16 KV would halve that and refuse the 131K max length.
- The config advertises 524,288 native. One such sequence is 64 GB of fp8
  KV, more than both cards hold, so on 32 GB parts the usable context is
  what fits next to the weights, not what the config says.

## What the numbers mean

- 66 tok/s single stream is two cards moving ~19 GB of weights per token
  each over PCIe with PyNccl all-reduce (`--disable-custom-all-reduce` is
  set for the same reason as on every profile in this fleet: the custom
  kernel needs `expandable_segments` off, and that setting was left on for
  the KV pin). Four streams ran at 66 each, 261 aggregate, so the batch is
  free up to there.
- Prefill 5,948 tok/s at 4K. The 100K needle prompt took 65 s end to end.
- The needle miss: the answer came back confident and wrong. The 7B
  found the same needle in 25 s. We report it as measured; a second
  prompt family would be needed to call it a pattern.

## Operational

- Shared image from `../vllm-k2-image/`, nothing model-specific patched.
- `--trust-remote-code` required for the checkpoint's config class.
- Zero error lines in the server log across boot, probes, both suites and
  the four-stream stress.
