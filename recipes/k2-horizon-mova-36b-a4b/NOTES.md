# K2-Horizon-MoVA-36B-A4B on two RTX 5090: field notes

Companion to README.md. Measured 2026-09-03 with `tools/llm-bench`, fresh
salt, warmup discarded, temperature 0.

## The crash, and the one-line fix

- Symptom under `--quantization fp8`: the engine dies in torch.compile with
  `AssertionError('Hidden size mismatch 2560 != 512')`, raised by the
  fused-MoE constraint check inside `fused_mova_impl`.
- Cause: `K2HorizonMoVAAttention.compute_mova_v_sparse()` does
  `torch.stack([expert.weight for expert in self.v_experts])` and calls
  the bf16 fused kernel with `use_fp8_w8a8=False` hardcoded. With a quant
  config, each value expert is a `ColumnParallelLinear` under
  `Fp8LinearMethod`, whose `process_weights_after_loading` replaces
  `weight` with its transpose in fp8. The stack becomes
  `[64, 2560, 512]` (hidden first, kv/tp second) instead of
  `[64, 512, 2560]`, and the first assertion fires.
- Fix: build the value experts with `quant_config=None`. They stay bf16
  and the rest of the model (attention projections, the 100 MLP experts,
  the shared expert, lm_head) quantizes to fp8 as requested. Cost: 7.5B
  parameters at 2 bytes instead of 1, about 3.75 GB per rank at TP=2. A
  real fix would teach `fused_mova_impl` fp8 weights plus scales; that is
  upstream work on vllm-project/vllm#55063's follow-ups.

## Memory arithmetic on 32 GB cards

- Weights per rank after the patch: roughly 14 GB fp8 for the non-MoVA
  parameters plus 3.75 GB bf16 value experts.
- KV: 48 layers, 8 KV heads, head_dim 128, 196 KB per token bf16, 98 KB
  fp8, split across two ranks. 65,536 tokens fit (pool 80,928);
  131,072 needs 6.0 GiB per rank and vLLM finds 4.65 at 0.93 utilization.
  Three or four cards, or a real fp8 MoVA kernel, are what it takes.
- bf16 KV would halve the pool again; `--kv-cache-dtype fp8` is not
  optional here.

## What 52 tok/s is made of

- 4B active parameters should read faster than this on a 5090. What the
  decode step pays for: the fp8 MoE path without DeepGEMM (disabled on
  sm120, where its E8M0 scale format is wrong), the bf16 Triton MoVA
  kernel for the value experts on top of the attention, and a PyNccl
  all-reduce per layer at TP=2 over PCIe. Four streams held 50 each and
  196 aggregate, so batching is nearly free; single-stream latency is the
  weak spot.
- Expert parallel vs tensor parallel for the MLP experts, same everything
  else: decode 51-52 (EP) vs 52-53 (TP), prefill 3,153 (EP) vs 5,375 (TP),
  pool 80,928 vs 80,320. At two ranks the EP dispatch costs more than it
  saves, so the recipe runs plain TP=2 and keeps EP behind `EP=1`.

## Operational

- `--trust-remote-code` required for the checkpoint's config class.
- Boot is about eight minutes: safetensors load, fp8 quantization at load,
  torch.compile, CUDA graph capture.
- Zero error lines in the server log across boot, probes, both suites and
  the four-stream stress.
