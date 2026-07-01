/**
 * Pure helpers for the neuromod Cloudflare Worker (Deploy D1).
 *
 * Kept free of any Worker/RunPod runtime so they can be unit-tested with `node --test`.
 * The Worker (worker.js) composes these to translate a browser chat request into a RunPod
 * call and re-emit the streamed result as Server-Sent Events.
 */

export const RUNPOD_BASE = "https://api.runpod.ai/v2";

/** Clamp an intensity to [0, 1] with a fallback (mirrors the Python handler). */
export function clampIntensity(value, fallback = 0.5) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(1, n));
}

function num(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Build the RunPod `{ input: {...} }` payload from a browser chat request body.
 * Accepts either a raw `prompt` or a `messages` array; applies the same defaults and
 * intensity clamp as the RunPod handler's parse_event.
 */
export function buildRunpodInput(body) {
  const b = body || {};
  const input = {};
  if (b.prompt) input.prompt = String(b.prompt);
  else input.messages = Array.isArray(b.messages) ? b.messages : [];
  input.pack_name = b.pack_name || null;
  input.intensity = clampIntensity(b.intensity, 0.5);
  input.max_tokens = num(b.max_tokens, 128);
  input.temperature = num(b.temperature, 1.0);
  input.top_p = num(b.top_p, 1.0);
  if (b.model) input.model = String(b.model);
  return { input };
}

/** CORS headers for browser access. */
export function corsHeaders(origin = "*") {
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
    "Access-Control-Max-Age": "86400",
  };
}

/** Encode one SSE `data:` frame. */
export function sseEncode(obj) {
  return `data: ${JSON.stringify(obj)}\n\n`;
}

/**
 * Extract the ordered list of generator outputs from a RunPod `/stream/{id}` poll body.
 * RunPod wraps each yielded item as `{ output: <item> }` inside a `stream` array.
 */
export function parseRunpodStream(json) {
  const out = [];
  const arr = (json && json.stream) || [];
  for (const item of arr) {
    if (item && item.output !== undefined) out.push(item.output);
  }
  return out;
}

/** True once a RunPod job has reached a terminal status. */
export function isTerminal(status) {
  return status === "COMPLETED" || status === "FAILED" || status === "CANCELLED";
}

/**
 * Client auth check. If no API_KEY is configured (dev), allow. Otherwise require a matching
 * `Authorization: Bearer <key>` or `X-API-Key` header.
 */
export function checkAuth(request, env) {
  if (!env || !env.API_KEY) return true;
  const auth = request.headers.get("authorization") || "";
  const bearer = auth.replace(/^Bearer\s+/i, "");
  const key = bearer || request.headers.get("x-api-key") || "";
  return key === env.API_KEY;
}
