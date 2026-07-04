/** node:test unit tests for the Worker's pure helpers (no CF/RunPod runtime needed). */
import test from "node:test";
import assert from "node:assert/strict";
import {
  clampIntensity, buildRunpodInput, corsHeaders, sseEncode,
  parseRunpodStream, isTerminal, checkAuth,
  parseCookies, isProRequest, resolveEndpointId, tierCookie, stripTierInfo, TIER_COOKIE,
  maxTokensForTier, modelLabelForTier,
} from "../src/lib.js";

// Fake request carrying a cookie header.
function reqWithCookie(cookie) {
  return { headers: { get: (h) => (h.toLowerCase() === "cookie" ? cookie : null) } };
}

test("parseCookies", () => {
  assert.deepEqual(parseCookies("a=1; b=hello%20world"), { a: "1", b: "hello world" });
  assert.deepEqual(parseCookies(""), {});
});

test("tier switch: default endpoint unless a valid unlock cookie is present", () => {
  const env = { RUNPOD_ENDPOINT_ID: "default8b", RUNPOD_ENDPOINT_ID_PRO: "pro120b", UNLOCK_KEY: "s3cr3t" };
  // no cookie -> default
  assert.equal(resolveEndpointId(reqWithCookie(null), env), "default8b");
  assert.equal(isProRequest(reqWithCookie(null), env), false);
  // wrong cookie -> default
  assert.equal(resolveEndpointId(reqWithCookie(`${TIER_COOKIE}=nope`), env), "default8b");
  // correct cookie -> pro
  assert.equal(resolveEndpointId(reqWithCookie(`${TIER_COOKIE}=s3cr3t`), env), "pro120b");
  assert.equal(isProRequest(reqWithCookie(`${TIER_COOKIE}=s3cr3t`), env), true);
});

test("tier switch: PRO impossible when unlock not configured", () => {
  const env = { RUNPOD_ENDPOINT_ID: "default8b" }; // no UNLOCK_KEY / PRO id
  assert.equal(resolveEndpointId(reqWithCookie(`${TIER_COOKIE}=whatever`), env), "default8b");
});

test("tierCookie sets httpOnly + clears", () => {
  const set = tierCookie("s3cr3t");
  assert.match(set, /^nm_tier=s3cr3t;/);
  assert.match(set, /HttpOnly/);
  assert.match(set, /Secure/);
  assert.match(tierCookie(null), /Max-Age=0/);
});

test("stripTierInfo removes model/tier hints (incl. image_model)", () => {
  assert.deepEqual(stripTierInfo({ chunk: "hi", model: "gpt-oss-120b", model_type: "local" }),
                   { chunk: "hi" });
  // image_model (SDXL-Turbo default vs SDXL pro) would leak the tier too — must be stripped.
  assert.deepEqual(stripTierInfo({ image: "data:...", image_model: "stabilityai/sdxl-turbo" }),
                   { image: "data:..." });
  assert.equal(stripTierInfo("x"), "x");
});

test("buildRunpodInput image task: whitelisted task + clamped params", () => {
  const p = buildRunpodInput({ task: "image", prompt: "a fox", pack_name: "lsd", intensity: 2,
                               width: 1024, height: 768, steps: 30, seed: 7, guidance_scale: 9 });
  assert.equal(p.input.task, "image");
  assert.equal(p.input.prompt, "a fox");
  assert.equal(p.input.width, 1024);
  assert.equal(p.input.height, 768);
  assert.equal(p.input.steps, 30);
  assert.equal(p.input.seed, 7);
  assert.equal(p.input.guidance_scale, 9);
  // Out-of-range params get clamped; the chat default omits task entirely.
  const clamped = buildRunpodInput({ task: "image", prompt: "x", width: 9999, steps: 999 });
  assert.equal(clamped.input.width, 1024);
  assert.equal(clamped.input.steps, 80);
  const chat = buildRunpodInput({ messages: [{ role: "user", content: "hi" }] });
  assert.ok(!("task" in chat.input));
});

test("buildRunpodInput ignores non-image task values (no server-job passthrough)", () => {
  // A browser must not be able to trigger steering/endpoints/diag jobs via /api/chat.
  const p = buildRunpodInput({ task: "steering", prompt: "x" });
  assert.ok(!("task" in p.input));
});

test("clampIntensity clamps and defaults", () => {
  assert.equal(clampIntensity(0.7), 0.7);
  assert.equal(clampIntensity(3), 3);    // >1 overload is allowed
  assert.equal(clampIntensity(9), 5);    // capped at MAX_INTENSITY
  assert.equal(clampIntensity(-2), 0);
  assert.equal(clampIntensity("nope", 0.5), 0.5);
});

test("maxTokensForTier: per-tier limits from env, with fallbacks", () => {
  const env = { MAX_TOKENS_DEFAULT: "1536", MAX_TOKENS_PRO: "512" };
  assert.equal(maxTokensForTier(false, env), 1536); // 8b / default: plenty long
  assert.equal(maxTokensForTier(true, env), 512);   // 120b / pro: reasonable
  // Missing/invalid env -> built-in defaults (1536 default, 512 pro).
  assert.equal(maxTokensForTier(false, {}), 1536);
  assert.equal(maxTokensForTier(true, {}), 512);
  assert.equal(maxTokensForTier(false, { MAX_TOKENS_DEFAULT: "0" }), 1536);   // non-positive ignored
  assert.equal(maxTokensForTier(true, { MAX_TOKENS_PRO: "nope" }), 512);      // non-numeric ignored
  assert.equal(maxTokensForTier(false, undefined), 1536);
});

test("modelLabelForTier: per-tier display label from env, with fallbacks", () => {
  const env = { MODEL_LABEL_DEFAULT: "Llama-3.1-8B", MODEL_LABEL_PRO: "gpt-oss-120b" };
  assert.equal(modelLabelForTier(false, env), "Llama-3.1-8B");
  assert.equal(modelLabelForTier(true, env), "gpt-oss-120b");
  // Missing env -> built-in defaults.
  assert.equal(modelLabelForTier(false, {}), "Llama-3.1-8B");
  assert.equal(modelLabelForTier(true, {}), "gpt-oss-120b");
  assert.equal(modelLabelForTier(false, undefined), "Llama-3.1-8B");
});

test("buildRunpodInput from messages with defaults", () => {
  const p = buildRunpodInput({ messages: [{ role: "user", content: "hi" }], pack_name: "lsd" });
  assert.deepEqual(p.input.messages, [{ role: "user", content: "hi" }]);
  assert.equal(p.input.pack_name, "lsd");
  assert.equal(p.input.intensity, 0.5);
  assert.equal(p.input.max_tokens, 128);
});

test("buildRunpodInput from raw prompt clamps intensity + passes model", () => {
  const p = buildRunpodInput({ prompt: "x", intensity: 9, model: "openai/gpt-oss-120b" });
  assert.equal(p.input.prompt, "x");
  assert.equal(p.input.intensity, 5);   // capped at MAX_INTENSITY (overload allowed up to 5)
  assert.equal(p.input.model, "openai/gpt-oss-120b");
  assert.ok(!("messages" in p.input));
});

test("sseEncode formats a data frame", () => {
  assert.equal(sseEncode({ chunk: "hi" }), 'data: {"chunk":"hi"}\n\n');
});

test("parseRunpodStream extracts ordered outputs", () => {
  const outs = parseRunpodStream({ stream: [{ output: { chunk: "a" } }, { output: { chunk: "b" } }] });
  assert.deepEqual(outs, [{ chunk: "a" }, { chunk: "b" }]);
  assert.deepEqual(parseRunpodStream({}), []);
});

test("isTerminal recognizes terminal statuses", () => {
  assert.equal(isTerminal("COMPLETED"), true);
  assert.equal(isTerminal("FAILED"), true);
  assert.equal(isTerminal("IN_PROGRESS"), false);
});

test("corsHeaders includes origin", () => {
  assert.equal(corsHeaders("https://x.dev")["Access-Control-Allow-Origin"], "https://x.dev");
  assert.equal(corsHeaders()["Access-Control-Allow-Origin"], "*");
});

function req(headers = {}) {
  return { headers: { get: (k) => headers[k.toLowerCase()] ?? null } };
}

test("checkAuth open when no key configured", () => {
  assert.equal(checkAuth(req(), {}), true);
});

test("checkAuth validates bearer and x-api-key", () => {
  const env = { API_KEY: "secret" };
  assert.equal(checkAuth(req({ authorization: "Bearer secret" }), env), true);
  assert.equal(checkAuth(req({ "x-api-key": "secret" }), env), true);
  assert.equal(checkAuth(req({ authorization: "Bearer wrong" }), env), false);
  assert.equal(checkAuth(req(), env), false);
});
