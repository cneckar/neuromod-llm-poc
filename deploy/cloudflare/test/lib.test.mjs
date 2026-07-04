/** node:test unit tests for the Worker's pure helpers (no CF/RunPod runtime needed). */
import test from "node:test";
import assert from "node:assert/strict";
import {
  clampIntensity, buildRunpodInput, corsHeaders, sseEncode,
  parseRunpodStream, isTerminal, checkAuth,
  parseCookies, isProRequest, resolveEndpointId, tierCookie, stripTierInfo, TIER_COOKIE,
  maxTokensForTier, modelLabelForTier, prefersDefaultTier, effectiveProTier, buildChatRecord,
  validateCustomPack,
} from "../src/lib.js";

// Fake request carrying a cookie header.
function reqWithCookie(cookie) {
  return { headers: { get: (h) => (h.toLowerCase() === "cookie" ? cookie : null) } };
}

// Fake request carrying both a cookie and the X-Prefer-Tier header.
function reqWith(cookie, preferTier) {
  return { headers: { get: (h) => {
    const k = h.toLowerCase();
    if (k === "cookie") return cookie;
    if (k === "x-prefer-tier") return preferTier || null;
    return null;
  } } };
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

test("tier downgrade: PRO browser can opt into the fast/default model", () => {
  const env = { RUNPOD_ENDPOINT_ID: "default8b", RUNPOD_ENDPOINT_ID_PRO: "pro120b", UNLOCK_KEY: "s3cr3t" };
  const proCookie = `${TIER_COOKIE}=s3cr3t`;
  // Unlocked, no preference -> PRO.
  assert.equal(effectiveProTier(reqWith(proCookie, null), env), true);
  assert.equal(resolveEndpointId(reqWith(proCookie, null), env), "pro120b");
  // Unlocked but asking for default -> served the default endpoint...
  assert.equal(effectiveProTier(reqWith(proCookie, "default"), env), false);
  assert.equal(resolveEndpointId(reqWith(proCookie, "default"), env), "default8b");
  // ...yet raw access is still PRO (so the UI keeps offering the toggle).
  assert.equal(isProRequest(reqWith(proCookie, "default"), env), true);
  assert.equal(prefersDefaultTier(reqWith(proCookie, "default")), true);
  // A non-PRO browser can't upgrade via the header (no cookie -> still default).
  assert.equal(effectiveProTier(reqWith(null, "pro"), env), false);
  assert.equal(resolveEndpointId(reqWith(null, null), env), "default8b");
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

test("validateCustomPack: whitelist, clamp, steering_type, cap", () => {
  const cp = validateCustomPack({
    name: "Neurobloom", description: "d",
    effects: [
      { effect: "steering", weight: 0.5, direction: "up", parameters: { steering_type: "visionary" } },
      { effect: "temperature", weight: 3, direction: "down" },     // weight clamped to 1
      { effect: "not_a_real_effect", weight: 0.5 },                 // dropped (not whitelisted)
      { effect: "steering", weight: 0.4, parameters: { steering_type: "evil" } }, // bad steering_type dropped
      { effect: "kv_decay", weight: 0 },                           // zero weight dropped
    ],
  });
  assert.equal(cp.name, "Neurobloom");
  assert.equal(cp.effects.length, 3);                              // 2 dropped
  assert.deepEqual(cp.effects[0], { effect: "steering", weight: 0.5, direction: "up",
                                    parameters: { steering_type: "visionary" } });
  assert.equal(cp.effects[1].weight, 1);                           // clamped
  assert.equal(cp.effects[1].direction, "down");
  assert.ok(!("parameters" in cp.effects[2]));                     // steering w/ bad type -> no params
});

test("validateCustomPack: empty/invalid -> null; effect cap", () => {
  assert.equal(validateCustomPack(null), null);
  assert.equal(validateCustomPack({ effects: [] }), null);
  assert.equal(validateCustomPack({ effects: [{ effect: "nope", weight: 1 }] }), null);
  const many = { effects: Array.from({ length: 20 }, () => ({ effect: "temperature", weight: 0.5 })) };
  assert.equal(validateCustomPack(many).effects.length, 8);       // capped at MAX_CUSTOM_EFFECTS
});

test("buildRunpodInput forwards a validated custom_pack (overrides pack_name)", () => {
  const p = buildRunpodInput({ pack_name: "lsd", intensity: 1,
    custom_pack: { name: "X", effects: [{ effect: "steering", weight: 0.6, direction: "up",
                                          parameters: { steering_type: "playful" } }] } });
  assert.equal(p.input.pack_name, null);                          // custom overrides named pack
  assert.equal(p.input.custom_pack.name, "X");
  assert.equal(p.input.custom_pack.effects[0].effect, "steering");
  // An invalid custom pack is dropped, leaving the named pack.
  const q = buildRunpodInput({ pack_name: "lsd", custom_pack: { effects: [{ effect: "nope", weight: 1 }] } });
  assert.equal(q.input.pack_name, "lsd");
  assert.ok(!("custom_pack" in q.input));
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

test("buildChatRecord assembles a text chat from streamed chunks", () => {
  const body = { messages: [{ role: "user", content: "hi" }], pack_name: "lsd", intensity: 0.8 };
  const outputs = [{ chunk: "Hel" }, { chunk: "lo." }, { done: true, reasoning: "thinking" }];
  const r = buildChatRecord(body, outputs, { tier: "pro" });
  assert.equal(r.task, "chat");
  assert.equal(r.assistant, "Hello.");
  assert.equal(r.reasoning, "thinking");
  assert.equal(r.pack_name, "lsd");
  assert.equal(r.intensity, 0.8);
  assert.equal(r.tier, "pro");
  assert.equal(r.had_image, 0);
  assert.deepEqual(JSON.parse(r.messages), body.messages);
});

test("buildChatRecord records image + custom pack, no image bytes", () => {
  const body = { prompt: "a fox", task: "image", custom_pack: { name: "mydrug", effects: [] } };
  const r = buildChatRecord(body, [{ image: "data:image/png;base64,AAAA", pack_applied: null }]);
  assert.equal(r.task, "image");
  assert.equal(r.had_image, 1);
  assert.ok(!/AAAA/.test(JSON.stringify(r)));         // image bytes not persisted
  assert.equal(JSON.parse(r.custom_pack).name, "mydrug");
  assert.deepEqual(JSON.parse(r.messages), [{ role: "user", content: "a fox" }]);
});

test("buildChatRecord falls back to response field + records error", () => {
  assert.equal(buildChatRecord({}, [{ response: "full text", done: true }]).assistant, "full text");
  assert.equal(buildChatRecord({}, [{ error: "boom" }]).error, "boom");
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
