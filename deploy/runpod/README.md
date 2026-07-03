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
| `runpod.endpoint.example.json` | Reference endpoint settings (gpt-oss-20b default) — `workersMin: 0` is the scale-to-zero switch. |
| `runpod.endpoint.gpt-oss-120b.json` | Hero text endpoint (80GB GPU), Deploy D2. |
| `runpod.endpoint.gemma-3-27b.json` | Google Gemma 3 27B endpoint (gated model), Deploy D2. |

## Switchable models (Deploy D2)

One image serves several models. The worker's default is `MODEL_NAME`; a request may also
select a model per-call via `"model"` in the input. Selection is resolved through a **vetted
allow-list** (`MODEL_REGISTRY` in `handler.py`) so an untrusted browser request can't make a
scale-to-zero GPU pull an arbitrary/giant checkpoint — an unknown id falls back to the default
(override with `ALLOW_ANY_MODEL=1` for dev).

| Alias (`"model"`) | HF id | GPU tier |
|---|---|---|
| `gpt-oss-20b` | `openai/gpt-oss-20b` | L40S / A100 (snappy default) |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | A100/H100 80GB (hero) |
| `gemma-3-27b` | `google/gemma-3-27b-it` | L40S / A100 (gated) |

Deploy one endpoint per model (each with its own `MODEL_NAME` and the configs above), and route
by model at the Worker/UI. The Worker already forwards `model` end-to-end (`buildRunpodInput`).

## Cost guardrails & observability (Deploy D4)

- **Idle → $0:** every endpoint sets `workersMin: 0`, so a warm worker scales down after
  `idleTimeout` seconds and idle time is never billed.
- **Spend cap:** `workersMax` caps concurrent GPUs (2–3), bounding worst-case spend.
- **Per-request GPU-seconds:** each response carries `gpu_seconds` (wall-clock the worker held
  the GPU) and, on the request that paged a model in, `cold_start_seconds`. The worker also logs
  a parseable `[billing] {json}` line per request (see `billing_record` / `log_billing`) for a
  log-based cost scrape. Multiply `gpu_seconds` by the endpoint's per-second GPU price for the
  request's cost; `cold_start_seconds` shows the amortizable cold-start overhead per model.

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
`intensity`, `tokens_generated`, `generation_time`, `model`) plus the D4 cost fields
(`gpu_seconds`, `cold_start_seconds`), so the Cloudflare Worker, Streamlit UI, and
`demo/chat.py` work against it with only a URL change. Select a non-default model by adding
`"model": "gpt-oss-120b"` (alias) to the `input`.

## Serving gpt-oss-120b (H200 / MXFP4)

- **Hardware:** gpt-oss-120b ships **MXFP4** weights (~63 GB). MXFP4 runs natively only on
  **Hopper (H100/H200)** — a 140 GB H200 fits it comfortably. On non-Hopper hardware Transformers
  dequantizes to bf16 (~240 GB) and will OOM, so keep it on an H200.
- **Deps:** the image already pins `transformers>=4.55`, `kernels`, `triton>=3.1` (required for the
  MXFP4 kernels). `model_support.py` detects gpt-oss and skips bitsandbytes automatically.
- **Weights on the network volume:** bake the MXFP4 checkpoint onto the volume once
  (`huggingface-cli download openai/gpt-oss-120b`) and set `HF_HOME=/runpod-volume/hf` so a
  cold start doesn't re-pull 63 GB. Config: `runpod.endpoint.gpt-oss-120b.json` (`MODEL_NAME=openai/gpt-oss-120b`).
- **Neuromod validity:** steering vectors are model-specific, so they live in per-model
  subdirectories (`outputs/steering_vectors/<model-slug>/`, e.g. `openai__gpt-oss-120b/`). If the
  repo ships vectors for this model they're used directly; otherwise the loader falls back to the
  flat legacy set (wrong-dim for gpt-oss → padded/truncated → steering is a no-op) until you run the
  `steering` job below **once** for this model.

## Server-side jobs — run the whole study serverless (no pod)

The handler routes on an `input.task` field, so heavy work runs **on the warm worker** (which has
the model in-process with full internals) and results land on the network volume — everything stays
scale-to-zero. Tasks: `generate` (default), `warmup`, `steering`, `endpoints`.

```bash
# Regenerate steering vectors for the served model (one-time; writes to the volume):
curl -s -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"task":"steering","model":"gpt-oss-120b"}}'

# Run the full internal-telemetry endpoint battery for a pack (returns the endpoints JSON inline):
curl -s -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"task":"endpoints","pack_name":"lsd","model":"gpt-oss-120b"}}'
```

Or drive it from a laptop (torch-free client — pay only for the worker's GPU-seconds):

```bash
export RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=...
python scripts/run_remote_study.py --mode steering  --model openai/gpt-oss-120b
python scripts/run_remote_study.py --mode endpoints --model openai/gpt-oss-120b --packs lsd,cocaine,morphine
python scripts/run_remote_study.py --mode behavioral --model openai/gpt-oss-120b --packs lsd,cocaine
```

`--mode endpoints` reproduces the **full** study (Table-1 battery, internal telemetry) via the
endpoint; `--mode behavioral` is a text-only dose sweep whose CSV feeds
`analysis/dose_response_stats.py`. Set `STEERING_DIR` / `ENDPOINTS_DIR` env on the endpoint to point
job outputs at the volume (defaults `/runpod-volume/{steering_vectors,endpoints}`).

`STEERING_DIR` is dual-purpose: the `steering` job **writes** vectors there, and the pack system
**loads** them from there at inference. Both sides are **model-aware** — the job writes to
`$STEERING_DIR/<model-slug>/` and `SteeringEffect` looks there first (keyed on the model id, which
it reads from the loaded model or the `MODEL_NAME` env), falling back to the flat `$STEERING_DIR/`
and finally the committed `outputs/steering_vectors/`. So one volume (or repo) can hold vectors for
several models at once, and after regenerating for the served model its generations pick them up
automatically — no code change, just the env var. Vectors are tiny (~hidden_size × 4 bytes), so a
model's set can also be **committed** under `outputs/steering_vectors/<model-slug>/` to skip
regeneration entirely.

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
- **D2** ([#17](https://github.com/cneckar/neuromod-llm-poc/issues/17)): gpt-oss-120b + Gemma-3-27B endpoints — model registry + configs landed; endpoints to be created on a GPU box.
- **D4** ([#19](https://github.com/cneckar/neuromod-llm-poc/issues/19)): idle-timeout / max-workers caps + per-request GPU-seconds logging — landed; cold-start numbers to be measured on a GPU box.
