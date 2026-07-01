/** node:test unit tests for the Worker's pure helpers (no CF/RunPod runtime needed). */
import test from "node:test";
import assert from "node:assert/strict";
import {
  clampIntensity, buildRunpodInput, corsHeaders, sseEncode,
  parseRunpodStream, isTerminal, checkAuth,
} from "../src/lib.js";

test("clampIntensity clamps and defaults", () => {
  assert.equal(clampIntensity(0.7), 0.7);
  assert.equal(clampIntensity(5), 1);
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
  assert.equal(p.input.intensity, 1);
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
