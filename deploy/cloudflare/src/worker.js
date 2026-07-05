/**
 * Neuromod Cloudflare Worker (Deploy D1).
 *
 * Public edge for the neuromod chat demo. Holds the RunPod API key as a Worker secret,
 * authenticates + rate-limits the browser, translates a chat request into a RunPod
 * Serverless call, and re-emits the streamed generator output as Server-Sent Events.
 *
 * Routes:
 *   GET  /            -> static chat demo UI
 *   GET  /health      -> {ok:true}
 *   GET  /api/packs   -> pack catalog (served/cached here)
 *   POST /api/chat    -> SSE stream of {chunk}/{done} events from the RunPod handler
 *
 * Secrets / vars (wrangler): RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, API_KEY (client auth),
 * ALLOW_ORIGIN, POLL_MS.
 */

import {
  RUNPOD_BASE, buildRunpodInput, corsHeaders, sseEncode,
  parseRunpodStream, isTerminal, checkAuth,
  UNLOCK_PARAM, resolveEndpointId, isProRequest, effectiveProTier, maxTokensForTier,
  modelLabelForTier, tierCookie, stripTierInfo, buildChatRecord,
} from "./lib.js";
// The full drag-and-drop demo UI (ported from docs/demo.html, rewired to the real backend).
import INDEX_HTML from "./index.html";

// Vestigial: the UI (index.html) ships its own full catalog. Kept for /api/packs consumers.
const DEFAULT_PACKS = ["none", "lsd", "dmt", "psilocybin", "mdma", "cocaine",
  "amphetamine", "ketamine", "morphine", "caffeine", "cannabis_thc"];

function json(obj, env, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", ...corsHeaders(env && env.ALLOW_ORIGIN) },
  });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders(env && env.ALLOW_ORIGIN) });
    }
    if (url.pathname === "/health") return json({ ok: true }, env);
    if (url.pathname === "/") {
      // Backend-only tier unlock: `/?k=<UNLOCK_KEY>` validates the key server-side and sets an
      // httpOnly cookie, then redirects to a clean "/" (key never lingers in the URL or reaches
      // page JS). `/?k=` (empty/wrong) locks back to the default endpoint.
      if (url.searchParams.has(UNLOCK_PARAM)) {
        const provided = url.searchParams.get(UNLOCK_PARAM) || "";
        const ok = env && env.UNLOCK_KEY && provided === env.UNLOCK_KEY;
        return new Response(null, {
          status: 302,
          headers: {
            location: "/",
            "set-cookie": tierCookie(ok ? env.UNLOCK_KEY : null),
            ...corsHeaders(env && env.ALLOW_ORIGIN),
          },
        });
      }
      return new Response(INDEX_HTML, {
        headers: { "content-type": "text/html; charset=utf-8", ...corsHeaders(env && env.ALLOW_ORIGIN) },
      });
    }
    if (url.pathname === "/api/packs") {
      return json({ packs: (env && env.PACKS && env.PACKS.split(",")) || DEFAULT_PACKS }, env);
    }
    if (url.pathname === "/api/model") {
      // Report which model this browser is talking to (resolved server-side from the tier cookie
      // + any downgrade preference). Display labels only — never the endpoint id or unlock key.
      // `pro` = raw PRO access (drives whether the UI offers the "use fast model" toggle);
      // `model` = the label actually in effect now; `fast` = the default-tier label to swap to.
      const hasPro = isProRequest(request, env);
      return json({
        model: modelLabelForTier(effectiveProTier(request, env), env),
        pro: hasPro,
        fast: modelLabelForTier(false, env),
      }, env);
    }
    if (url.pathname === "/api/chat" && request.method === "POST") {
      return handleChat(request, env, ctx);
    }
    // Chat archive (D1). Gated behind the unlock cookie so transcripts aren't world-readable.
    if (url.pathname === "/api/chats" || url.pathname.startsWith("/api/chats/")) {
      return handleChatsRead(request, env, url);
    }
    return json({ error: "not found" }, env, 404);
  },
};

async function handleChatsRead(request, env, url) {
  if (!isProRequest(request, env)) return json({ error: "unauthorized" }, env, 401);
  if (!env || !env.DB) return json({ chats: [], note: "D1 not configured" }, env);
  const cols = "id, created, tier, task, pack_name, custom_pack, intensity, had_image, assistant, reasoning, error";
  try {
    const single = url.pathname.startsWith("/api/chats/") && url.pathname.slice("/api/chats/".length);
    if (single) {
      const row = await env.DB.prepare(`SELECT ${cols}, messages FROM chats WHERE id = ?`).bind(single).first();
      return row ? json({ chat: row }, env) : json({ error: "not found" }, env, 404);
    }
    const limit = Math.max(1, Math.min(200, Number(url.searchParams.get("limit")) || 50));
    const { results } = await env.DB.prepare(
      `SELECT ${cols} FROM chats ORDER BY created DESC LIMIT ?`).bind(limit).all();
    return json({ chats: results || [] }, env);
  } catch (e) {
    return json({ error: `d1 read failed: ${e}` }, env, 500);
  }
}

/** Persist one completed chat to D1. No-op when D1 isn't bound; never throws into the response. */
async function saveChat(env, record) {
  if (!env || !env.DB) return;
  try {
    await env.DB.prepare(
      `INSERT INTO chats
         (id, created, tier, task, pack_name, custom_pack, intensity, had_image, messages, assistant, reasoning, error)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      crypto.randomUUID(), new Date().toISOString(), record.tier, record.task, record.pack_name,
      record.custom_pack, record.intensity, record.had_image, record.messages, record.assistant,
      record.reasoning, record.error
    ).run();
  } catch (e) {
    console.log("saveChat failed:", String(e));  // archival must never break the chat response
  }
}

async function handleChat(request, env, ctx) {
  if (!checkAuth(request, env)) return json({ error: "unauthorized" }, env, 401);
  if (!env || !env.RUNPOD_API_KEY || !env.RUNPOD_ENDPOINT_ID) {
    return json({ error: "worker not configured (missing RunPod secrets)" }, env, 500);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "invalid JSON body" }, env, 400);
  }

  const payload = buildRunpodInput(body);
  // Route to the PRO endpoint if this request carries a valid tier cookie; else the default.
  // resolveEndpointId reads only server-side env + the httpOnly cookie — the client never sees
  // which endpoint it hit. Ignore any client-supplied `model` so the browser can't force a tier.
  delete payload.input.model;
  // effectiveProTier honors a PRO browser's opt-in downgrade (X-Prefer-Tier: default): a downgrade
  // is always allowed, an upgrade is impossible (a non-PRO cookie can't reach the PRO endpoint).
  const pro = effectiveProTier(request, env);
  const endpointId = pro ? env.RUNPOD_ENDPOINT_ID_PRO : env.RUNPOD_ENDPOINT_ID;
  // Token budget is set per tier server-side (the frontend can't know which model it hit):
  // generous for the small/default model, reasonable for the large PRO model.
  payload.input.max_tokens = maxTokensForTier(pro, env);
  const base = `${RUNPOD_BASE}/${endpointId}`;
  const authHeaders = {
    "content-type": "application/json",
    authorization: `Bearer ${env.RUNPOD_API_KEY}`,
  };

  // Kick off the async job.
  const runRes = await fetch(`${base}/run`, {
    method: "POST", headers: authHeaders, body: JSON.stringify(payload),
  });
  if (!runRes.ok) {
    return json({ error: `runpod run failed: ${runRes.status}` }, env, 502);
  }
  const { id } = await runRes.json();

  const pollMs = Number(env.POLL_MS) || 300;
  const encoder = new TextEncoder();

  const collected = [];  // relayed outputs, for the D1 archive
  const stream = new ReadableStream({
    async start(controller) {
      try {
        while (true) {
          const r = await fetch(`${base}/stream/${id}`, { headers: authHeaders });
          if (!r.ok) {
            controller.enqueue(encoder.encode(sseEncode({ error: `stream ${r.status}` })));
            break;
          }
          const data = await r.json();
          // RunPod's /stream drains: each poll returns ONLY the outputs generated since the last
          // poll, not a cumulative list. So emit every item in this batch — do NOT carry a running
          // index across polls (that skipped most tokens, producing sparse/garbled output).
          for (const out of parseRunpodStream(data)) {
            collected.push(out);
            // Strip model/tier hints so the client can't tell which endpoint served it.
            controller.enqueue(encoder.encode(sseEncode(stripTierInfo(out))));
          }
          if (isTerminal(data.status)) {
            // Surface a failed/cancelled job (and an image task that produced nothing) instead of
            // ending on a bare [DONE] — otherwise the UI just "times out" with no reason.
            if (data.status !== "COMPLETED" && !collected.length) {
              controller.enqueue(encoder.encode(sseEncode({
                error: `job ${data.status}` + (data.error ? `: ${String(data.error).slice(0, 300)}` : ""),
              })));
            } else if (!collected.length) {
              controller.enqueue(encoder.encode(sseEncode({ error: "no output from worker" })));
            }
            break;
          }
          await new Promise((res) => setTimeout(res, pollMs));
        }
      } catch (err) {
        controller.enqueue(encoder.encode(sseEncode({ error: String(err) })));
      } finally {
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        // Archive the completed exchange to D1 (best-effort, off the response path).
        if (ctx && env && env.DB) {
          const record = buildChatRecord(body, collected, { tier: pro ? "pro" : "default" });
          ctx.waitUntil(saveChat(env, record));
        }
      }
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive",
      ...corsHeaders(env && env.ALLOW_ORIGIN),
    },
  });
}
