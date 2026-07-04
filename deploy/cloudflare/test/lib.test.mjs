/** node:test unit tests for the Worker's pure helpers (no CF/RunPod runtime needed). */
import test from "node:test";
import assert from "node:assert/strict";
import {
  clampIntensity, buildRunpodInput, corsHeaders, sseEncode,
  parseRunpodStream, isTerminal, checkAuth,
  parseCookies, isProRequest, resolveEndpointId, tierCookie, stripTierInfo, TIER_COOKIE,
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

test("stripTierInfo removes model/tier hints", () => {
  assert.deepEqual(stripTierInfo({ chunk: "hi", model: "gpt-oss-120b", model_type: "local" }),
                   { chunk: "hi" });
  assert.equal(stripTierInfo("x"), "x");
});

test("clampIntensity clamps and defaults", () => {
  assert.equal(clampIntensity(0.7), 0.7);
  assert.equal(clampIntensity(3), 3);    // >1 overload is allowed
  assert.equal(clampIntensity(9), 5);    // capped at MAX_INTENSITY
  assert.equal(clampIntensity(-2), 0);
  assert.equal(clampIntensity("nope", 0.5), 0.5);
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
