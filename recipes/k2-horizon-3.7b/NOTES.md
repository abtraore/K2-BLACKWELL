# K2-Horizon-3.7B on one RTX 5090 (vLLM): field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

## Why fp8 KV

- 36 layers, 8 KV heads, head_dim 128: 144 KB per token bf16. At the
  model's 524,288 native length one sequence is 72 GB of KV; at 262,144
  vLLM asked for 36 GiB against 16.5 available and refused; at 131,072 in
  bf16 it asked for 18 GiB against about 15 and refused again. fp8 KV
  halves it and yields a 250,768-token pool.
- The refusal message is explicit (`To serve at least one request with the
  model's max seq len ... KV cache is needed, which is larger than the
  available KV cache memory`), so a wrong context shows up at boot, not
  at runtime.

## The needle miss

- 100,035-token prompt, needle at 60K, 33 s end to end, answer confident
  and wrong. The llama.cpp recipe for the same weights (bf16 KV, same
  prompt family, different random filler) found its needle in 36 s. fp8
  KV on a 3.7B is the obvious suspect; it is one sample per engine, so we
  report both and do not generalize.

## Engine comparison at this size

- Single-stream decode is a tie (147-151 here, 140-152 on llama.cpp).
- vLLM wins concurrency (563 aggregate at four streams vs 151 where
  llama.cpp queues at `--parallel 1`) and prefill (18,300 vs 12,366).
- llama.cpp wins context per card: it runs the full 131K in bf16 KV on
  one 5090, which vLLM's reserve does not allow.

## Operational

- Shared image from `../vllm-k2-image/`, nothing model-specific patched.
- `--trust-remote-code` required for the checkpoint's config class.
- Zero error lines in the server log across boot, probes, both suites and
  the four-stream stress.
