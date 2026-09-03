# K2-Horizon-MoVA-36B-A4B GGUF on two RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

## Engine comparison, same weights, same two cards

| | llama.cpp (this recipe) | vLLM (`../k2-horizon-mova-36b-a4b/`) |
|---|---|---|
| weights | Q8_0 GGUF, 39.8 GB | fp8 at load + bf16 value experts |
| context | 131,072, bf16 KV | 65,536 ceiling, fp8 KV |
| decode, single stream | 131-142 tok/s | 52-53 tok/s |
| prefill at 4K | 6,737 tok/s | 5,375 tok/s |
| four streams | 139 aggregate (queued) | 196 aggregate |
| needle 60K-in-100K | found | not testable at 65K |

The fork's MoVA path treats the 64 value experts like any other expert
tensor (Q8_0, fused expert GEMV), so nothing stays in bf16 and nothing is
transposed underneath a kernel that did not expect it. vLLM's merged path
has the bf16-only fused kernel described in the vLLM recipe's NOTES.

The 4B-active model reads roughly 5 GB of Q8_0 weights per token here,
which is consistent with 131-142 tok/s split over two 5090s with one
layer-split hop.

## The converter trap

`convert_hf_to_gguf.py` on this branch opens
`models/templates/k2-horizon.jinja` relative to the script's directory.
The `full` Docker target copies `*.py`, `conversion/`, `gguf-py/` and
`requirements/` but not `models/`, so the in-image copy fails with
`FileNotFoundError`. Bind-mount the clone and run the script from there.

## Memory

- Q8_0 weights split across two cards (about 20 GB each) plus the full
  131K bf16 KV (196 KB per token, 25 GiB, also split) fits with room; the
  fitter is bypassed by `-ngl 999`, so pass `-c` yourself if you change
  the layout.
- `--parallel 1` serializes requests; raise it if you need concurrency and
  can give each slot less context.
