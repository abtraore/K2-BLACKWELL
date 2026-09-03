# Shared vLLM image for every K2 Horizon recipe in this repo.
# Base: the vLLM nightly built 2026-09-03 06:16Z, pinned by digest. K2 Horizon
# support merged upstream two hours later (vllm-project/vllm#55063, merge
# commit 1f76efaa), so the six Python files from that merge commit are laid
# over the image: the model, the reasoning parser, the tool-call parser and
# the three registry entries. They are the upstream files, byte for byte.
# Retirement condition: any vLLM image built after 2026-09-03 08:17Z carries
# them natively; switch the base and delete patches/.
FROM vllm/vllm-openai@sha256:7d5128a92db05e6d70661a18f10e40c17cd2d8fcaad9226e170a0d69f822b795
ARG P=/usr/local/lib/python3.12/dist-packages/vllm
COPY patches/k2_horizon.py $P/model_executor/models/k2_horizon.py
COPY patches/registry.py $P/model_executor/models/registry.py
COPY patches/reasoning_init.py $P/reasoning/__init__.py
COPY patches/k2_horizon_reasoning_parser.py $P/reasoning/k2_horizon_reasoning_parser.py
COPY patches/tool_init.py $P/tool_parsers/__init__.py
COPY patches/k2_horizon_tool_parser.py $P/tool_parsers/k2_horizon_tool_parser.py
