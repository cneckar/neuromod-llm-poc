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
| `GET /` | Full drag-and-drop demo UI (`src/index.html`, ported from `docs/chat.html`): drag drug "pills" onto the avatar, set the dose (intensity slider, 0–5× — >1 overloads), streams tokens. Wired to the REAL backend (not the OpenAI-prompt simulation the GitHub Pages version used). |
| `GET /health` | `{ok:true}` |
| `GET /api/packs` | Pack catalog (from `PACKS` var or a default list) |
| `POST /api/chat` | SSE stream: `data: {chunk}` events, then the final `data: {done, response, …}` and `data: [DONE]` |
| `GET /api/chats` | **Chat archive** (D1). Unlock-gated. `?limit=N` recent rows; `/api/chats/<id>` one full transcript. Empty if D1 isn't bound. |

Request body (same shape as the RunPod handler / `ChatRequest`):
```json
{ "messages": [{"role":"user","content":"Describe a sunset."}],
  "pack_name": "lsd", "intensity": 0.7, "max_tokens": 128, "model": "openai/gpt-oss-20b" }
```

## Setup & deploy

```bash
cd deploy/cloudflare
npm install
# 1. Config (gitignored — real values live here, both endpoint ids + the unlock key):
cp wrangler.toml.example wrangler.toml
#    edit wrangler.toml -> RUNPOD_ENDPOINT_ID, RUNPOD_ENDPOINT_ID_PRO, UNLOCK_KEY
# 2. Credentials as encrypted secrets (never in any file):
wrangler secret put RUNPOD_API_KEY      # required — used for BOTH endpoints (same key)
wrangler secret put API_KEY             # optional client auth; omit = open (dev only)
# 3. Run / ship:
npm run dev                             # local (wrangler dev)
npm run deploy                          # publish
```

`wrangler.toml` is **gitignored**; the committed template is `wrangler.toml.example`.

### Chat archive (D1) — optional

Persist every completed chat for posterity. **D1** (Cloudflare's serverless SQLite) is the right fit
here over KV/R2: it's durable, queryable, and browsable — the whole point of an archive. Enable it by
binding a D1 database as `DB`; leave the binding out and persistence is simply off (the Worker no-ops).

```bash
npx wrangler d1 create neuromod_chats     # prints database_id -> paste into wrangler.toml [[d1_databases]]
npx wrangler d1 execute neuromod_chats --file=migrations/0001_create_chats.sql --remote
```

The Worker writes each finished exchange off the response path (`ctx.waitUntil`), so archiving never
slows or breaks a chat. Image *bytes* aren't stored (only a `had_image` flag) to keep rows small.
Read them back at `GET /api/chats` — **gated behind the unlock cookie** so transcripts aren't public.

### Two-tier routing (default 8B, gated 120B) — backend only

Everyone hits the **default** endpoint (`RUNPOD_ENDPOINT_ID`, e.g. llama-8b). Visiting
**`/?k=<UNLOCK_KEY>`** validates the key server-side, sets an **httpOnly** cookie, and redirects to
a clean `/`; that browser's `/api/chat` requests then route to the gated **`RUNPOD_ENDPOINT_ID_PRO`**
(gpt-oss-120b). `/?k=` (empty/wrong) locks back to the default.

Both endpoint ids + the unlock key live in `wrangler.toml`, which is **gitignored** — so they're
not in the repo. They're also never sent to the browser (Worker env is server-side), the tier
cookie is **httpOnly** (page scripts can't read it), and the streamed response has its `model`/tier
fields stripped, so the client cannot tell which endpoint served it. The UI is byte-for-byte
identical in both tiers. (`RUNPOD_ENDPOINT_ID_PRO`/`UNLOCK_KEY` may instead be set as
`wrangler secret`s if you prefer; env resolves either way — don't set both.)

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
