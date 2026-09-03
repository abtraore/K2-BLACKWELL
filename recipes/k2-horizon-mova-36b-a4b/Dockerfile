# K2-Horizon-MoVA-36B-A4B on vLLM with on-the-fly fp8: needs one patch on top of
# the shared K2 image (../vllm-k2-image). patches/k2_horizon.py is upstream's
# model file (vllm-project/vllm#55063 merge commit) with ONE change: the 64 MoVA
# value experts per attention layer are built with quant_config=None.
# Why: compute_mova_v_sparse() stacks the raw `expert.weight` tensors into a
# bf16-only fused kernel; under --quantization fp8 the Fp8LinearMethod
# transposes and casts those weights in place, the stack arrives as
# [E, 2560, 512] in fp8 and the kernel asserts "Hidden size mismatch 2560 != 512".
# Retirement condition: fused_mova_impl accepting fp8 weights + scales upstream.
FROM k2-blackwell/vllm:r1
COPY patches/k2_horizon.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/k2_horizon.py
