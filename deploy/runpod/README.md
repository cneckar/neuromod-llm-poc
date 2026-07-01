# RunPod Serverless deployment (Deploy D0)

Serves neuromodulated LLM inference on a RunPod Serverless worker that **scales to zero** and
bills per GPU-second **only while a request is executing**. Part of Epic #5; this directory is
the D0 foundation ([#14](https://github.com/cneckar/neuromod-llm-poc/issues/14)).

## What's here

| File | Purpose |
|---|---|
| `handler.py` | RunPod handler wrapping the existing `api.model_manager` generate chain. Pure request/response logic is unit-tested (`tests/test_runpod_handler.py`); model load is lazy (cold start, module scope). |
| `Dockerfile` | CUDA+PyTorch image; installs the repo, caches HF weights on the network volume. |
| `requirements.txt` | Serving extras (runpod, transformers, accelerate). |
| `runpod.endpoint.example.json` | Reference endpoint settings — `workersMin: 0` is the scale-to-zero switch. |

## Why RunPod Serverless (not a spot Pod)

Neuromod packs need **HF Transformers-level access to model internals** (steering, attention/KV
surgery, logits processors), so the model must run in-process on a single cold-startable GPU.
RunPod **Serverless** bills per-second and scales to zero when idle — a persistent (spot) Pod
bills continuously and cannot meet "pay only during inference."

## Build & deploy

```bash
# 1. Build + push the image (from repo root).
docker build -f deploy/runpod/Dockerfile -t <registry>/neuromod-runpod:latest .
docker push <registry>/neuromod-runpod:latest

# 2. Create a RunPod network volume and pre-download weights into /hf so cold starts are fast.
#    (Run a one-off pod mounting the volume, then `huggingface-cli download openai/gpt-oss-20b`.)

# 3. Create a Serverless endpoint (UI or GraphQL) using runpod.endpoint.example.json:
#    workersMin=0, flashboot=true, attach the network volume, set MODEL_NAME + HUGGINGFACE_TOKEN.
```

## Test the endpoint

```bash
curl -s -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Describe a sunset."}],
                 "pack_name": "lsd", "intensity": 0.7, "max_tokens": 128}}'
```

Response mirrors the existing `ChatResponse` (`response`, `emotions`, `pack_applied`,
`intensity`, `tokens_generated`, `generation_time`, `model`), so the Cloudflare Worker,
Streamlit UI, and `demo/chat.py` work against it with only a URL change.

## Acceptance criteria (issue #14) — status

- [x] `handler.py` wraps `LocalModelInterface.generate_text(pack_name=...)`; model loaded once at module scope.
- [x] Dockerfile + requirements + endpoint config (`workersMin: 0`).
- [x] Request/response matches `ChatRequest`/`ChatResponse`; pure logic unit-tested (GPU-free).
- [ ] **Pending on a GPU box:** `/runsync` returns a real neuromodulated completion; measure cold-start; verify idle → $0.

## Local validation (no GPU)

```bash
python -m pytest tests/test_runpod_handler.py -q   # request/response contract + pack pass-through
```

## Next phases

- **D1** ([#15](https://github.com/cneckar/neuromod-llm-poc/issues/15)/[#16](https://github.com/cneckar/neuromod-llm-poc/issues/16)): streaming generator handler + Cloudflare Worker (SSE).
- **D2** ([#17](https://github.com/cneckar/neuromod-llm-poc/issues/17)): add gpt-oss-120b + Gemma-3-27B endpoints.
- **D4** ([#19](https://github.com/cneckar/neuromod-llm-poc/issues/19)): idle-timeout / max-workers caps + per-request GPU-seconds logging.
