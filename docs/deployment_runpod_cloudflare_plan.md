# Deployment Plan: Live Neuromod Chat + Image Demo on RunPod (scale-to-zero) behind Cloudflare

> Goal: run the chat demo (and the visual dose-response demo) against **current best-in-class
> models** on RunPod GPUs, fronted by a Cloudflare Worker, **paying for GPU-seconds only while
> an inference pass is actually executing** — zero cost when idle. Streaming is first-class.

## Governing constraint (why the model list is what it is)

Every neuromodulation pack works by reaching **inside** the model — activation steering,
attention/KV surgery, logits processors — which requires **HuggingFace Transformers-level
access** to weights and activations. That excludes vLLM/SGLang-only serving (fast but no hook
surface). Combined with scale-to-zero economics (a worker must fit and cold-start on **one**
GPU), this is the feasibility filter for "best-in-class."

RunPod **Serverless** satisfies the billing constraint natively: per-second billing, workers
scale to zero when idle (no idle charge), FlashBoot gives sub-2s cold starts, and weights are
cached on a network volume so they are not re-downloaded. A persistent **Pod** (even spot) does
**not** satisfy it — a Pod bills continuously while up; making it "scale to zero" means
programmatically stopping/starting it, an anti-pattern at minute-scale cold starts.

## Final model stack (per decisions)

| Role | Model | GPU | Neuromod hooks |
|---|---|---|---|
| Text default | **gpt-oss-20b** | ~16–24GB | full HF Transformers hooks |
| Text hero | **gpt-oss-120b** (MXFP4 ~66–80GB) | single 80GB H100 | full HF Transformers hooks |
| Text (Google) | **Gemma 3 27B** | 80GB (or 40GB quant) | full HF Transformers hooks |
| Image | **FLUX.2 [dev]** (32B DiT, Nov 2025) | single 80GB | **hooks to be ported** (see D3) |
| Fallback image | SDXL-Turbo (current) | small | already wired to metrics |

**Deferred:** DeepSeek — the real 671B V3/R1 can't run internal-hook neuromod under scale-to-zero
(multi-node, vLLM-served). Revisit later (via R1-Distill-32B, which *is* single-GPU + hookable).

**FLUX porting note:** FLUX.2 is a DiT with a **16-channel VAE**; the visual-neuromod layer and
the pharmacodynamics latent capture / `FrequencyAnalyzer` currently assume SDXL's UNet + 4-channel
latents. Porting (latent-capture callback, channel count, effect hooks) is scoped as its own
phase (D3). SDXL-Turbo keeps the dose-response study reproducible meanwhile.

## Architecture

```
Browser demo ──HTTPS──► Cloudflare Worker ──/run (RunPod API key = Worker secret)──► RunPod Serverless
   ▲   SSE stream          auth · rate-limit ·        per-second GPU billing,           min_workers=0
   └───────────────────────CORS · static site        only during inference             GPU worker on demand
                                                                                          │
                                                                                          ▼
                                                                              handler.py (generator)
                                                                              wraps existing
                                                                              LocalModelInterface + NeuromodTool
                                                                              (model loaded once at module scope)
```

## Reuse map (minimize new code — verified against the repo)

| Need | Reuse | File |
|---|---|---|
| Per-request pack application + generate | `LocalModelInterface.generate_text(pack_name=...)` chain | `api/model_manager.py` |
| Pack apply → logits processors → generate | `NeuromodTool.apply` → `PackManager.apply_pack` → `get_logits_processors` | `neuromod/neuromod_tool.py`, `neuromod/pack_system.py` |
| Request/response schema | `ChatRequest` / `ChatResponse` (keep compatible) | `api/server.py` |
| Model load w/ HF auth + quantization | Vertex prediction server loader | `vertex_container/prediction_server.py` |
| Web client already hitting the HTTP API | Streamlit `API_BASE_URL` client | `api/web_interface.py` |
| Pack catalog | `packs/config.json` (served/cached by Worker) | `packs/config.json` |

New code required (does not exist in repo — confirmed): RunPod `handler.py`, the Cloudflare
Worker (Wrangler project), SSE proxy, and the FLUX neuromod port.

## Streaming (first-class)

- Text: HF `TextIteratorStreamer` in a worker thread → **RunPod generator handler** (`yield`
  tokens, exposed at `/stream`) → **Cloudflare Worker re-emits as SSE** → browser renders live.
- Image: "stream" = push diffusion-step previews and the dose-response frames as they render —
  this is literally the "vitals monitor" demo from the dose-response plan.

## Phases

- **D0** — RunPod `handler.py` + Docker image + network-volume weights for **gpt-oss-20b**;
  validate `/runsync`; confirm idle→$0 and measure cold-start.
- **D1** — Generator handler + **Cloudflare Worker with SSE**; point the demo site at it; RunPod
  key as Worker secret; API-key auth + rate-limit + CORS.
- **D2** — Add **gpt-oss-120b** + **Gemma-3-27B** as switchable endpoints.
- **D3** — **FLUX.2 image endpoint** + port neuromod visual hooks + pharmacodynamics latent
  capture to the 16-channel DiT VAE; stream dose-response frames.
- **D4** — Cost guardrails: idle-timeout tuning, `max_workers` cap, per-request GPU-seconds
  logging, FlashBoot/volume cold-start optimization.

## Cost model

Pay only while a handler executes: a ~500-token text stream ≈ a few GPU-seconds; a FLUX image ≈
a few seconds on an 80GB card. Zero traffic ⇒ zero GPU cost. Cloudflare Worker + static site run
on the free/cheap tier.

## Risks

- **Cold start** on 80GB models (120b / FLUX.2): mitigate with FlashBoot + volume-baked weights;
  keep gpt-oss-20b as the always-snappy default.
- **Hook ↔ throughput tradeoff:** HF Transformers (required for hooks) is slower than vLLM;
  acceptable for a demo, not for high QPS.
- **FLUX port** is real work (16-ch VAE / DiT) — isolated to D3 so text demo isn't blocked.
- **Secrets:** RunPod API key lives only in the Worker; never shipped to the browser.

## References (verified 2026)

- RunPod Serverless pricing / scale-to-zero / FlashBoot — https://docs.runpod.io/serverless/pricing ,
  https://www.runpod.io/product/serverless
- FLUX on RunPod (ComfyUI serverless template) — https://www.runpod.io/articles/guides/comfy-ui-flux
- Best open image models 2026 (FLUX.2 [dev]) — https://www.bentoml.com/blog/a-guide-to-open-source-image-generation-models
- gpt-oss hardware requirements — https://arxiv.org/pdf/2508.10925 ,
  https://www.spheron.network/tools/gpu-recommender/openai/gpt-oss-120b/
