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
} from "./lib.js";
import { INDEX_HTML } from "./ui.js";

const DEFAULT_PACKS = ["none", "lsd", "dmt", "psilocybin", "mdma", "cocaine",
  "amphetamine", "ketamine", "morphine", "caffeine", "nicotine"];

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
      return new Response(INDEX_HTML, {
        headers: { "content-type": "text/html; charset=utf-8", ...corsHeaders(env && env.ALLOW_ORIGIN) },
      });
    }
    if (url.pathname === "/api/packs") {
      return json({ packs: (env && env.PACKS && env.PACKS.split(",")) || DEFAULT_PACKS }, env);
    }
    if (url.pathname === "/api/chat" && request.method === "POST") {
      return handleChat(request, env);
    }
    return json({ error: "not found" }, env, 404);
  },
};

async function handleChat(request, env) {
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
  const base = `${RUNPOD_BASE}/${env.RUNPOD_ENDPOINT_ID}`;
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

  const stream = new ReadableStream({
    async start(controller) {
      let seen = 0;
      try {
        while (true) {
          const r = await fetch(`${base}/stream/${id}`, { headers: authHeaders });
          if (!r.ok) {
            controller.enqueue(encoder.encode(sseEncode({ error: `stream ${r.status}` })));
            break;
          }
          const data = await r.json();
          const outputs = parseRunpodStream(data);
          for (; seen < outputs.length; seen++) {
            controller.enqueue(encoder.encode(sseEncode(outputs[seen])));
          }
          if (isTerminal(data.status)) break;
          await new Promise((res) => setTimeout(res, pollMs));
        }
      } catch (err) {
        controller.enqueue(encoder.encode(sseEncode({ error: String(err) })));
      } finally {
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
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
