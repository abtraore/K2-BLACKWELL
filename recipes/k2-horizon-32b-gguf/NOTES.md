# K2-Horizon-32B GGUF on two RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

- 64 layers, 8 KV heads, head_dim 128: 256 KB per token bf16, 33.5 GiB at
  131,072. Split over two cards next to 17.5 GB of Q8_0 weights each, the
  bf16 cache fails to allocate (`cudaMalloc failed: out of memory` on a
  16.9 GB buffer). `-ctk q8_0 -ctv q8_0` halves it and fits.
- Decode 39-42 tok/s: two cards reading about 17.5 GB each per token
  through a layer split. IFM's FP8 export on vLLM at TP=2 reads half that
  and lands at 66 tok/s; a Q4_K_M GGUF would move this lane in the same
  direction at some quality cost we did not measure.
- Prefill 4,022 tok/s at 4K; the 100K needle prompt took 79 s end to end.
- Needle miss on both engines for this model; we report it and do not
  explain it.
- `--parallel 1` serializes requests.
