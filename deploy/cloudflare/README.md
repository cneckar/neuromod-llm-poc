# Cloudflare Worker — neuromod chat edge (Deploy D1)

Public edge for the neuromod chat demo. Fronts the RunPod Serverless endpoint, keeps the
RunPod API key server-side, authenticates + CORS-guards the browser, and re-emits the
streamed generator output as **Server-Sent Events**. Epic #5, issue #16 (pairs with the
streaming RunPod handler, #15).

```
Browser ──POST /api/chat──► Worker ──/run + poll /stream (RunPod key secret)──► RunPod Serverless
   ▲        SSE {chunk}…              re-emit as SSE                              stream_handler (D1)
   └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Routes

| Route | Purpose |
|---|---|
| `GET /` | Full drag-and-drop demo UI (`src/index.html`, ported from `docs/demo.html`): drag drug "pills" onto the avatar, set the dose (intensity slider, 0–5× — >1 overloads), streams tokens. Wired to the REAL backend (not the OpenAI-prompt simulation the GitHub Pages version used). |
| `GET /health` | `{ok:true}` |
| `GET /api/packs` | Pack catalog (from `PACKS` var or a default list) |
| `POST /api/chat` | SSE stream: `data: {chunk}` events, then the final `data: {done, response, …}` and `data: [DONE]` |

Request body (same shape as the RunPod handler / `ChatRequest`):
```json
{ "messages": [{"role":"user","content":"Describe a sunset."}],
  "pack_name": "lsd", "intensity": 0.7, "max_tokens": 128, "model": "openai/gpt-oss-20b" }
```

## Setup & deploy

```bash
cd deploy/cloudflare
npm install
# 1. Point at your RunPod Serverless endpoint:
#    edit wrangler.toml -> RUNPOD_ENDPOINT_ID
# 2. Secrets (never committed):
wrangler secret put RUNPOD_API_KEY      # required
wrangler secret put API_KEY             # optional client auth; omit = open (dev only)
# 3. Run / ship:
npm run dev                             # local (wrangler dev)
npm run deploy                          # publish
```

Requires the RunPod endpoint (#14) running the **streaming** handler (`STREAMING=1`, the
default) so `/stream/{id}` yields `{chunk}` / `{done}` events.

## Tests

```bash
npm test        # node --test on the pure helpers (buildRunpodInput, SSE, auth, stream parse)
```

The fetch/SSE orchestration in `worker.js` is exercised end-to-end against a live RunPod
endpoint (see below); the pure translation/auth logic is unit-tested here with no runtime.

## Acceptance (issue #16) — status

- [x] `/api/chat` translates the browser request → RunPod and re-emits as SSE.
- [x] RunPod API key held as a Worker secret (never sent to the client); API-key auth + CORS.
- [x] Static demo UI + pack catalog; pure logic unit-tested (`npm test`).
- [ ] **Pending live endpoint:** browser → Worker → RunPod round-trip streams tokens (needs #14 deployed + #15's streaming handler live).

## Security notes

`RUNPOD_API_KEY` lives only in the Worker (a secret), never in client payloads. Client access
is gated by `API_KEY` (bearer or `X-API-Key`). Set `ALLOW_ORIGIN` to your site origin in
production instead of `*`.
