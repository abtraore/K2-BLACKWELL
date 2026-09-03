# Shared llama.cpp image for the K2 Horizon GGUF recipes

Mainline llama.cpp has no K2 Horizon architecture yet. IFM maintains a fork
branch (`model/K2Horizon` on MBZUAI-IFM/llama.cpp) and says a PR is in
progress. This Dockerfile builds that branch at a pinned commit for sm120,
mirroring the fork's own server Dockerfile, plus `llama-quantize` for the
recipes that quantize IFM's BF16 GGUFs.

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes/llamacpp-k2-image
docker build -t k2-blackwell/llamacpp:r1 .
```

Retire it when ggml-org/llama.cpp master carries `LLM_ARCH_K2_HORIZON`.
