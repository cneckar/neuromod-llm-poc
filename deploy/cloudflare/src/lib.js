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
// Header a PRO-unlocked browser sends to voluntarily use the cheap/default model instead. A
// downgrade is always allowed (you can't upgrade this way — it only ever drops PRO -> default).
export const PREFER_TIER_HEADER = "x-prefer-tier";

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

/** True if this request opts to use the cheap/default model despite holding PRO access. */
export function prefersDefaultTier(request) {
  const h = request && request.headers && request.headers.get(PREFER_TIER_HEADER);
  return (h || "").toLowerCase() === "default";
}

/**
 * The tier actually served for this request: PRO only if the browser is unlocked AND hasn't asked
 * to downgrade. (`isProRequest` reports raw ACCESS — used to decide whether to offer the toggle.)
 */
export function effectiveProTier(request, env) {
  return isProRequest(request, env) && !prefersDefaultTier(request);
}

/** Resolve which RunPod endpoint id to use for this request (PRO if unlocked+not downgraded). */
export function resolveEndpointId(request, env) {
  return effectiveProTier(request, env) ? env.RUNPOD_ENDPOINT_ID_PRO : (env && env.RUNPOD_ENDPOINT_ID);
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

// ---- Chemistry lab: custom packs -----------------------------------------------------------
// The effect types the browser may compose into a custom "drug". Whitelisted (a custom pack from an
// untrusted browser reaches the GPU) — only these names, matching neuromod.effects' registry, are
// forwarded; anything else is dropped. `steering: true` marks effects that take a steering_type.
export const STEERING_TYPES = ["associative", "visionary", "synesthesia", "ego_thin", "playful",
  "salient", "goal_focused", "prosocial", "abstract", "affiliative"];
export const PROJECTION_TYPES = ["creative", "analytical", "emotional", "spatial", "linguistic"];

// `group` drives the UI's <optgroup>s. `typeParam`/`typeValues` mark effects that also take one
// enumerated sub-type parameter (e.g. steering's flavor, soft_projection's mode). Excluded on
// purpose: effects that need structured inputs a slider can't supply (contrastive_decoding needs an
// amateur model, persona_voice_constraints a persona spec, pulsed_sampler an interval, etc.), the
// two inert stubs (expert_mixing, activation_additions), and the MoE-routing effects (no-op on the
// default non-MoE model).
export const CUSTOM_EFFECTS = {
  // Sampling & logits — decoding-time; model-agnostic.
  temperature:               { label: "Temperature / entropy",       group: "Sampling & logits" },
  top_p:                     { label: "Top-p (nucleus)",             group: "Sampling & logits" },
  frequency_penalty:         { label: "Frequency penalty",           group: "Sampling & logits" },
  presence_penalty:          { label: "Presence penalty",            group: "Sampling & logits" },
  style_affect_logit_bias:   { label: "Tone / style bias",           group: "Sampling & logits" },
  risk_preference_steering:  { label: "Risk preference",             group: "Sampling & logits" },
  // Steering & activation — residual-stream / hidden-state surgery.
  steering:                  { label: "Steering (concept vector)",   group: "Steering & activation", typeParam: "steering_type",   typeValues: STEERING_TYPES },
  soft_projection:           { label: "Soft projection (conceptor)", group: "Steering & activation", typeParam: "projection_type", typeValues: PROJECTION_TYPES },
  layer_wise_gain:           { label: "Layer-wise gain",             group: "Steering & activation" },
  lexical_jitter:            { label: "Lexical jitter (embedding)",  group: "Steering & activation" },
  noise_injection:           { label: "Noise injection",             group: "Steering & activation" },
  random_direction:          { label: "Placebo (random direction)",  group: "Steering & activation" },
  random_orthogonal_steering:{ label: "Placebo (orthogonal)",        group: "Steering & activation" },
  // Attention surgery — general transformer hooks.
  qk_score_scaling:          { label: "QK score scaling",            group: "Attention" },
  head_reweighting:          { label: "Head reweighting",            group: "Attention" },
  head_masking_dropout:      { label: "Head masking / dropout",      group: "Attention" },
  positional_bias_tweak:     { label: "Positional bias",             group: "Attention" },
  attention_oscillation:     { label: "Attention oscillation",       group: "Attention" },
  attention_sinks_anchors:   { label: "Attention sinks / anchors",   group: "Attention" },
  attention_focus:           { label: "Attention focus (induction)", group: "Attention" },
  attention_masking:         { label: "Attention masking",           group: "Attention" },
  // Working memory — KV-cache manipulation.
  kv_decay:                  { label: "Memory decay (KV)",           group: "Working memory (KV)" },
  kv_compression:            { label: "KV compression",              group: "Working memory (KV)" },
  exponential_decay_kv:      { label: "Exponential KV decay",        group: "Working memory (KV)" },
  truncation_kv:             { label: "KV truncation (keep last N)", group: "Working memory (KV)" },
  stride_compression_kv:     { label: "KV stride compression",       group: "Working memory (KV)" },
  segment_gains_kv:          { label: "KV segment gains",            group: "Working memory (KV)" },
  // Visual — only affect Stable-Diffusion image generation; inert in a text chat.
  color_bias:                { label: "Color bias (image)",          group: "Visual — image only" },
  style_transfer:            { label: "Style transfer (image)",      group: "Visual — image only" },
  composition_bias:          { label: "Composition bias (image)",    group: "Visual — image only" },
  visual_entropy:            { label: "Visual entropy (image)",      group: "Visual — image only" },
  synesthetic_mapping:       { label: "Synesthetic mapping (image)", group: "Visual — image only" },
  motion_blur:               { label: "Motion blur (image)",         group: "Visual — image only" },
};
const MAX_CUSTOM_EFFECTS = 8;

/**
 * Sanitize a browser-built custom pack to a safe, minimal shape (or null if empty/invalid):
 * whitelist effect names, clamp weight to [0,1] (the Python Pack requires it), restrict direction
 * to up/down, cap the number of effects, and validate steering_type against the known list.
 */
export function validateCustomPack(cp) {
  if (!cp || typeof cp !== "object" || !Array.isArray(cp.effects)) return null;
  const effects = [];
  for (const e of cp.effects.slice(0, MAX_CUSTOM_EFFECTS)) {
    if (!e || !CUSTOM_EFFECTS[e.effect]) continue;                 // whitelist the effect type
    const weight = Math.max(0, Math.min(1, Number(e.weight)));     // Pack validation requires [0,1]
    if (!Number.isFinite(weight) || weight === 0) continue;
    const out = { effect: e.effect, weight, direction: e.direction === "down" ? "down" : "up" };
    const meta = CUSTOM_EFFECTS[e.effect];
    if (meta.typeParam && e.parameters && meta.typeValues.includes(e.parameters[meta.typeParam])) {
      out.parameters = { [meta.typeParam]: e.parameters[meta.typeParam] };
    }
    effects.push(out);
  }
  if (!effects.length) return null;
  return {
    name: String(cp.name || "Custom Compound").slice(0, 40),
    description: String(cp.description || "User-designed compound").slice(0, 200),
    effects,
  };
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
  // Chemistry-lab custom pack (validated/whitelisted). Takes precedence over a named pack.
  if (b.custom_pack) {
    const cp = validateCustomPack(b.custom_pack);
    if (cp) { input.custom_pack = cp; input.pack_name = null; }
  }
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
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key, X-Prefer-Tier",
    "Access-Control-Max-Age": "86400",
  };
}

/** Encode one SSE `data:` frame. */
export function sseEncode(obj) {
  return `data: ${JSON.stringify(obj)}\n\n`;
}

/**
 * Build a flat, D1-insertable archive record for a completed chat from the request body and the
 * ordered list of RunPod output objects the Worker relayed. Pure (no Worker/D1 runtime) so it is
 * unit-testable. Image bytes are NOT stored (only a had_image flag) to keep rows small.
 */
export function buildChatRecord(body, outputs, extra = {}) {
  const b = body || {};
  let text = "", reasoning = null, hadImage = false, error = null, gotChunk = false;
  for (const o of outputs || []) {
    if (!o || typeof o !== "object") continue;
    if (typeof o.chunk === "string") { text += o.chunk; gotChunk = true; }
    if (typeof o.image === "string") hadImage = true;
    if (o.reasoning) reasoning = String(o.reasoning);
    if (o.error) error = String(o.error);
    if (!gotChunk && typeof o.response === "string") text = o.response;
  }
  const messages = Array.isArray(b.messages) ? b.messages
    : (b.prompt ? [{ role: "user", content: String(b.prompt) }] : []);
  return {
    tier: extra.tier || null,                              // server-side archive only
    task: b.task === "image" ? "image" : "chat",
    pack_name: b.pack_name || null,
    custom_pack: b.custom_pack ? JSON.stringify(b.custom_pack) : null,
    intensity: Number.isFinite(Number(b.intensity)) ? Number(b.intensity) : null,
    had_image: hadImage ? 1 : 0,
    messages: JSON.stringify(messages),
    assistant: text || null,
    reasoning: reasoning,
    error: error,
  };
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
