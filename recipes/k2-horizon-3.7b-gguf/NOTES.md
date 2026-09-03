# K2-Horizon-3.7B GGUF on one RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

- 36 layers, 8 KV heads, head_dim 128: 144 KB per token in bf16. At
  131,072 that is 18 GiB of KV, allocated up front by llama.cpp; with
  7.5 GB of weights the card is at about 26 GB. vLLM keeps a larger
  reserve and refused this exact configuration; its recipe for the 3.7B
  runs fp8 KV instead.
- Decode 140-152 tok/s on the bf16 GGUF (the 4K-prompt run was the slower
  one, 140, as attention over the longer prefix starts to count).
- Prefill 12,366 tok/s at 3.9K; the 100K needle prompt took 36 s end to
  end and came back correct.
- `--parallel 1` serializes requests. Raise it if you need concurrency and
  have the KV to split, at the cost of a smaller per-slot context.
- BF16 is the only GGUF IFM publishes for this size (7.5 GB); a Q8_0
  would halve the weight read but was not needed to fit.
