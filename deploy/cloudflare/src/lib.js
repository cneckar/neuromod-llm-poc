/**
 * Pure helpers for the neuromod Cloudflare Worker (Deploy D1).
 *
 * Kept free of any Worker/RunPod runtime so they can be unit-tested with `node --test`.
 * The Worker (worker.js) composes these to translate a browser chat request into a RunPod
 * call and re-emit the streamed result as Server-Sent Events.
 */

export const RUNPOD_BASE = "https://api.runpod.ai/v2";

// ---- Backend-only tier switch --------------------------------------------------------------
// Default endpoint (RUNPOD_ENDPOINT_ID, e.g. llama-8b) serves everyone. Visiting `/?<UNLOCK_PARAM>=
// <UNLOCK_KEY>` sets an httpOnly cookie; requests carrying that cookie route to the PRO endpoint
// (RUNPOD_ENDPOINT_ID_PRO, e.g. gpt-oss-120b). The PRO endpoint id and the unlock key live ONLY in
// Worker env/secrets — never in the served HTML or JS, and the cookie is httpOnly so page scripts
// can't read it either.
export const UNLOCK_PARAM = "k";       // query param on GET / that carries the key
export const TIER_COOKIE = "nm_tier";  // httpOnly cookie holding the validated key

/** Parse a Cookie header into a plain object. */
export function parseCookies(cookieHeader) {
  const out = {};
  for (const part of (cookieHeader || "").split(";")) {
    const i = part.indexOf("=");
    if (i < 0) continue;
    const k = part.slice(0, i).trim();
    if (k) out[k] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

/** True if this request is unlocked to the PRO tier (valid tier cookie === UNLOCK_KEY secret). */
export function isProRequest(request, env) {
  if (!env || !env.UNLOCK_KEY || !env.RUNPOD_ENDPOINT_ID_PRO) return false;
  const cookies = parseCookies(request.headers.get("cookie"));
  return cookies[TIER_COOKIE] === env.UNLOCK_KEY;
}

/** Resolve which RunPod endpoint id to use for this request (PRO if unlocked, else default). */
export function resolveEndpointId(request, env) {
  return isProRequest(request, env) ? env.RUNPOD_ENDPOINT_ID_PRO : (env && env.RUNPOD_ENDPOINT_ID);
}

/**
 * Max new tokens per tier. The frontend can't know the tier (it's server-side), so the Worker
 * sets the limit: generous for the default/small model, reasonable for the PRO/large model.
 * Configurable via MAX_TOKENS_DEFAULT / MAX_TOKENS_PRO env vars.
 */
export function maxTokensForTier(pro, env) {
  const d = Number(env && env.MAX_TOKENS_DEFAULT);
  const p = Number(env && env.MAX_TOKENS_PRO);
  if (pro) return Number.isFinite(p) && p > 0 ? p : 512;   // 120b: reasonable (matches the CLI)
  return Number.isFinite(d) && d > 0 ? d : 1536;           // 8b: plenty long
}

/**
 * Human-friendly label for the model the request is served by. The client can't know its tier
 * (server-side), so the Worker reports the label. This is a display name only — NOT the endpoint
 * id or unlock key — so exposing it leaks nothing about how to reach the PRO tier.
 * Configurable via MODEL_LABEL_DEFAULT / MODEL_LABEL_PRO.
 */
export function modelLabelForTier(pro, env) {
  if (pro) return (env && env.MODEL_LABEL_PRO) || "gpt-oss-120b";
  return (env && env.MODEL_LABEL_DEFAULT) || "Llama-3.1-8B";
}

/** Set-Cookie value that stores the validated key (httpOnly) — or clears it when key is null. */
export function tierCookie(key) {
  const base = `${TIER_COOKIE}=`;
  const attrs = "HttpOnly; Secure; SameSite=Lax; Path=/";
  return key ? `${base}${encodeURIComponent(key)}; ${attrs}; Max-Age=86400`
             : `${base}; ${attrs}; Max-Age=0`;
}

/** Strip fields that would reveal which backend/tier served a response (model, endpoint hints). */
export function stripTierInfo(obj) {
  if (!obj || typeof obj !== "object") return obj;
  // image_model would reveal the tier too (SDXL-Turbo default vs SDXL pro), so drop it as well.
  const { model, model_type, image_model, ...rest } = obj;
  return rest;
}

/**
 * Max intensity the edge forwards. Intensity is a MULTIPLIER on pack weights (>1 overloads),
 * bounded server-side by NEUROMOD_MAX_INTENSITY (default 5). Keep in sync with that default.
 */
export const MAX_INTENSITY = 5;

/** Clamp an intensity to [0, MAX_INTENSITY] with a fallback (mirrors the Python handler). */
export function clampIntensity(value, fallback = 0.5) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(MAX_INTENSITY, n));
}

function num(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** Clamp an optional positive integer within [min,max]; undefined/invalid -> undefined (omit). */
function optInt(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return undefined;
  return Math.max(min, Math.min(max, Math.round(n)));
}

/**
 * Build the RunPod `{ input: {...} }` payload from a browser chat request body.
 * Accepts either a raw `prompt` or a `messages` array; applies the same defaults and
 * intensity clamp as the RunPod handler's parse_event.
 *
 * `task` is whitelisted to "image" (or the default chat "generate") so the browser can only
 * ask for a chat completion or an image — never the server-side job tasks (steering/endpoints/
 * diag), which run heavy work on the worker.
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
  if (b.task === "image") {
    input.task = "image";
    // Image params (all optional; the worker applies per-model defaults for anything omitted).
    // Bounds keep a browser from requesting an absurd canvas / step count on a shared GPU.
    const w = optInt(b.width, 256, 1024);
    const h = optInt(b.height, 256, 1024);
    const steps = optInt(b.steps, 1, 80);
    const seed = optInt(b.seed, 0, 2 ** 31 - 1);
    if (w !== undefined) input.width = w;
    if (h !== undefined) input.height = h;
    if (steps !== undefined) input.steps = steps;
    if (seed !== undefined) input.seed = seed;
    if (Number.isFinite(Number(b.guidance_scale))) {
      input.guidance_scale = Math.max(0, Math.min(20, Number(b.guidance_scale)));
    }
  }
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
