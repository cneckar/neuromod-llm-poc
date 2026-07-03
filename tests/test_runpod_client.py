"""
Unit tests for the torch-free RunPod HTTP client (api/runpod_client.py).

The client is exercised against a fake ``requests`` transport, so the request shaping,
output extraction, and error handling are verified with no network and no torch.
"""

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE = os.path.join(_HERE, "..", "api", "runpod_client.py")


def _load():
    # Load by path with a stubbed `requests` so the import is torch/network free.
    spec = importlib.util.spec_from_file_location("runpod_client", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load()


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeRequests:
    """Fake transport. `runsync` (and generate_text) get `payload`; the async /run + /status
    flow returns a job id then a single COMPLETED status carrying `payload` as `output`.
    """

    def __init__(self, payload):
        self.payload = payload
        self.last = None

    def post(self, url, headers=None, json=None, timeout=None):
        self.last = {"url": url, "headers": headers, "json": json}
        if url.endswith("/run"):
            return _FakeResp({"id": "job-1"})
        # /runsync
        return _FakeResp(self.payload)

    def get(self, url, headers=None, timeout=None):
        # /status/<id> -> immediately COMPLETED with the payload as output
        out = self.payload.get("output", self.payload) if isinstance(self.payload, dict) else self.payload
        return _FakeResp({"status": "COMPLETED", "output": out})


def _client(monkeypatch, payload):
    fake = _FakeRequests(payload)
    monkeypatch.setattr(rc, "requests", fake)
    return rc.RunPodModelInterface("ep123", "key", model="openai/gpt-oss-120b"), fake


def test_generate_text_async_by_default(monkeypatch):
    # Default: submit via /run + poll /status, so a >90s big-model generation can't time out.
    client, fake = _client(monkeypatch, {"output": {"response": "hello", "emotions": {"joy": 1},
                                                    "gpu_seconds": 2.1}})
    out = client.generate_text("hi", pack_name="lsd", intensity=0.7, max_tokens=64, poll_interval=0)
    assert out["text"] == "hello" and out["emotions"] == {"joy": 1} and out["gpu_seconds"] == 2.1
    body = fake.last["json"]["input"]  # last POST is the /run submission
    assert body["prompt"] == "hi" and body["pack_name"] == "lsd" and body["intensity"] == 0.7
    assert body["model"] == "openai/gpt-oss-120b"
    assert fake.last["url"].endswith("/ep123/run")
    assert fake.last["headers"]["Authorization"] == "Bearer key"


def test_generate_text_wait_false_uses_runsync(monkeypatch):
    client, fake = _client(monkeypatch, {"output": {"response": "hi", "emotions": {}}})
    out = client.generate_text("hi", wait=False)
    assert out["text"] == "hi"
    assert fake.last["url"].endswith("/ep123/runsync")


def test_extract_output_from_aggregate_stream_list(monkeypatch):
    # return_aggregate_stream=True -> output is a list of yielded items; take the terminal one.
    payload = {"output": [{"chunk": "he"}, {"chunk": "llo"},
                          {"done": True, "response": "hello", "emotions": {}}]}
    client, _ = _client(monkeypatch, payload)
    assert client.generate_text("hi")["text"] == "hello"


def test_generate_text_raises_on_error_output(monkeypatch):
    client, _ = _client(monkeypatch, {"output": {"error": "boom"}})
    with pytest.raises(RuntimeError, match="boom"):
        client.generate_text("hi")


def test_run_task_async_submits_and_polls_to_completion(monkeypatch):
    # Default: long jobs go through /run + /status (not /runsync) so they can't time out.
    client, fake = _client(monkeypatch, {"output": {"ok": True, "task": "steering",
                                                    "artifacts": ["a.pt"]}})
    statuses = []
    res = client.run_task("steering", layer=-1, poll_interval=0,
                          on_status=lambda s, j: statuses.append(s))
    assert res["ok"] and res["task"] == "steering"
    # submitted via /run
    assert fake.last["url"].endswith("/ep123/run")
    body = fake.last["json"]["input"]
    assert body["task"] == "steering" and body["layer"] == -1 and body["model"] == "openai/gpt-oss-120b"
    assert statuses == ["COMPLETED"]


def test_run_task_wait_false_uses_runsync(monkeypatch):
    client, fake = _client(monkeypatch, {"output": {"ok": True, "task": "warmup"}})
    res = client.run_task("warmup", wait=False)
    assert res["ok"] and res["task"] == "warmup"
    assert fake.last["url"].endswith("/ep123/runsync")


def test_run_task_raises_on_error(monkeypatch):
    client, _ = _client(monkeypatch, {"output": {"ok": False, "error": "no weights"}})
    with pytest.raises(RuntimeError, match="no weights"):
        client.run_task("endpoints", pack_name="lsd", poll_interval=0)


def test_run_async_raises_on_failed_job(monkeypatch):
    fake = _FakeRequests({"output": {}})
    # override /status to report FAILED
    fake.get = lambda url, headers=None, timeout=None: _FakeResp({"status": "FAILED", "error": "boom"})
    monkeypatch.setattr(rc, "requests", fake)
    client = rc.RunPodModelInterface("ep123", "key")
    with pytest.raises(RuntimeError, match="boom"):
        client.run_task("endpoints", pack_name="lsd", poll_interval=0)


def test_interface_from_env_requires_both_vars(monkeypatch):
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        rc.interface_from_env()
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep")
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    c = rc.interface_from_env(model="gpt2")
    assert c.endpoint_id == "ep" and c.model == "gpt2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
