# K2-Horizon-7B GGUF on RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

- 36 layers, 8 KV heads, head_dim 128: 144 KB per token bf16, 18 GiB at
  131,072. llama.cpp allocates it up front; with the 18 GB BF16 weights on
  one 32 GB card the allocation fails. Two ways out, both measured:
  `-ctk q8_0 -ctv q8_0` (9 GiB, one card) or a second card with bf16 KV.
- q8_0 KV cost nothing visible here: same needle result, decode within 5%
  of the two-card bf16 layout (the two-card layout is faster because each
  card reads half the weights per token, not because of the KV type).
- Prefill: 7,393 tok/s on one card, 10,638 on two. The 100K needle prompt
  took 48 s and 37 s respectively.
- Weight bytes decide decode at this size: 18 GB BF16 reads at ~90 tok/s
  on a 5090; IFM's 11 GB FP8 export on vLLM reads at 132-135. A Q8_0 GGUF
  (`llama-quantize` is in the shared image) would sit between them.
- `--parallel 1` serializes requests; raise it only with KV to spare.
