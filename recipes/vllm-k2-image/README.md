# Shared vLLM image for the K2 Horizon recipes

One image serves every vLLM recipe in this repo. Build it once:

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes/vllm-k2-image
docker build -t k2-blackwell/vllm:r1 .
```

It is the vLLM nightly of 2026-09-03 06:16Z (pinned by digest) plus the six
Python files from the K2 Horizon merge commit
([vllm-project/vllm#55063](https://github.com/vllm-project/vllm/pull/55063),
merged 2026-09-03 08:17Z, two hours after that nightly was cut). The files
in `patches/` are the upstream files unmodified. Any later nightly has them
built in, at which point this directory is just a pinned base.

One file differs from upstream by a single branch: `k2_horizon_reasoning_parser.py`
returns an unclosed think block as reasoning instead of content (upstream shows the
model's scratchpad as the answer when `max_tokens` cuts the thinking off; the chat
template opens the think block inside the generation prompt, so the output never
carries the start token and the parser cannot key on it). Verified live on the 3.7B:
truncated, full, low-effort, tool-call and streaming requests all parse. Retire it
when vllm-project/vllm fixes `_split_model_output`.
