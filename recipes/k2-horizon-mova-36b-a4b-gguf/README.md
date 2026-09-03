# K2-Horizon-MoVA-36B-A4B GGUF on two RTX 5090 (llama.cpp)

The sparse K2 Horizon as a Q8_0 GGUF on the IFM llama.cpp fork, two 32 GB
cards, bf16 KV, 131,072-token context: **131-142 tok/s** single-stream
decode, prefill **6,737 tok/s** at 4K, and the **60K-in-100K needle found**
(100,035-token prompt, 33 s end to end). Single request at a time
(`--parallel 1`): four concurrent streams queued to 139 tok/s aggregate.

Gates: greedy 19*23 exact, generated `is_prime` executes, structured tool
call, reasoning in `reasoning_content`.

This is the fast way to run the 36B on two 5090s today. The vLLM recipe
(`../k2-horizon-mova-36b-a4b/`) needs a patch to quantize at all and then
does 52 tok/s at a 65K ceiling; here the fork's native MoVA path runs the
value experts in Q8_0 like everything else, reads about 40 GB per two
cards, and leaves room for the full 131K of KV.

## Contents

- `launch.sh`: parameterized launcher (`GPUS`/`PORT`/`CTX`/`GGUF`)
- `NOTES.md`: how the Q8_0 was made, the converter trap, engine comparison

## Make the GGUF

IFM publishes only a 74.9 GB BF16 GGUF for this model. Either quantize
that with `llama-quantize` (in the shared image) or convert the safetensors
directly, which is what we did:

```bash
hf download IFM/K2-Horizon-MoVA-36B-A4B
git clone https://github.com/MBZUAI-IFM/llama.cpp llama.cpp-k2 && git -C llama.cpp-k2 checkout 35999d101cf2233fc54f09c3c8d599da7303ce02
HF=$HOME/.cache/huggingface; SNAP=$(ls -d $HF/hub/models--IFM--K2-Horizon-MoVA-36B-A4B/snapshots/*/ | head -1); mkdir -p $HF/hub/gguf-k2
docker run --rm -v $HF:$HF -v $PWD/llama.cpp-k2:/src:ro -w /src --entrypoint python3 k2-blackwell/llamacpp:r1 \
  /src/convert_hf_to_gguf.py $SNAP --outtype q8_0 --outfile $HF/hub/gguf-k2/K2-Horizon-MoVA-36B-A4B-Q8_0.gguf
```

Run the converter from the clone, not from the image's `/app` copy: it
reads `models/templates/k2-horizon.jinja` relative to its own tree, and the
image does not ship that directory (our first attempt died on exactly
that). The result is 39.8 GB, 798 tensors, Q8_0 plus F32 norms, with the
MoVA metadata (`k2-horizon.attention.value_expert_count = 64`) intact.

## Start the server

```bash
git clone https://github.com/abtraore/K2-BLACKWELL && cd K2-BLACKWELL/recipes
(cd llamacpp-k2-image && docker build -t k2-blackwell/llamacpp:r1 .)
cd k2-horizon-mova-36b-a4b-gguf && ./launch.sh
```

Or the full command without the script:

```bash
HF=$HOME/.cache/huggingface
docker run -d --restart unless-stopped --name k2-horizon-mova-36b-gguf \
  --gpus all -e CUDA_VISIBLE_DEVICES=0,1 -v $HF:$HF:ro -p 8063:8000 \
  k2-blackwell/llamacpp:r1 \
  -m $HF/hub/gguf-k2/K2-Horizon-MoVA-36B-A4B-Q8_0.gguf --alias llamacpp/k2-horizon-mova-36b \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --jinja -c 131072 --parallel 1 --metrics \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95
```

Serving on `http://localhost:8063/v1`, model name
`llamacpp/k2-horizon-mova-36b`, about two minutes after `docker run` with
the GGUF in page cache (longer on a cold disk).
